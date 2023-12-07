import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import pixel_generator.mage.models_mage as models_mage

from engine_mage import train_one_epoch, gen_img


def get_args_parser():
    parser = argparse.ArgumentParser('MAGE training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mage_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')

    parser.add_argument('--vqgan_ckpt_path',  default='vqgan_jax_strongaug.ckpt', type=str)

    # Pre-trained enc parameters
    parser.add_argument('--use_rep', action='store_true', help='use representation as condition.')
    parser.add_argument('--use_class_label', action='store_true', help='use class label as condition.')
    parser.add_argument('--rep_dim', default=256, type=int)
    parser.add_argument('--pretrained_enc_arch',  default=None, type=str)
    parser.add_argument('--pretrained_enc_path',  default=None, type=str)
    parser.add_argument('--pretrained_enc_proj_dim',  default=256, type=int)
    parser.add_argument('--pretrained_enc_withproj', action='store_true')

    # RDM parameters
    parser.add_argument('--pretrained_rdm_ckpt',  default=None, type=str)
    parser.add_argument('--pretrained_rdm_cfg',  default=None, type=str)
    parser.add_argument('--rdm_steps', default=250, type=int)
    parser.add_argument('--eta', default=1.0, type=float)

    # Pixel generation parameters
    parser.add_argument('--evaluate', action='store_true', help="perform only evaluation")
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--temp', default=6.0, type=float,
                        help='sampling temperature')
    parser.add_argument('--num_iter', default=16, type=int,
                        help='number of iterations for generation')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=0.0, type=float)
    parser.add_argument('--rep_drop_prob', default=0.0, type=float)

    # MAGE params
    parser.add_argument('--mask_ratio_min', type=float, default=0.5,
                        help='Minimum mask ratio')
    parser.add_argument('--mask_ratio_max', type=float, default=1.0,
                        help='Maximum mask ratio')
    parser.add_argument('--mask_ratio_mu', type=float, default=0.55,
                        help='Mask ratio distribution peak')
    parser.add_argument('--mask_ratio_std', type=float, default=0.25,
                        help='Mask ratio distribution std')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--augmentation', default='noaug', type=str,
                        help='Augmentation type')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # init log writer
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # simple augmentation
    if args.augmentation == "noaug":
        transform_train = transforms.Compose([
                transforms.Resize(292, interpolation=3),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor()])
    elif args.augmentation == "randcrop":
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    elif args.augmentation == "randresizedcrop":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    else:
        raise NotImplementedError

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model

    model = models_mage.__dict__[args.model](mask_ratio_mu=args.mask_ratio_mu, mask_ratio_std=args.mask_ratio_std,
                                             mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
                                             vqgan_ckpt_path=args.vqgan_ckpt_path,
                                             use_rep=args.use_rep,
                                             rep_dim=args.rep_dim,
                                             rep_drop_prob=args.rep_drop_prob,
                                             use_class_label=args.use_class_label,
                                             pretrained_enc_arch=args.pretrained_enc_arch,
                                             pretrained_enc_path=args.pretrained_enc_path,
                                             pretrained_enc_proj_dim=args.pretrained_enc_proj_dim,
                                             pretrained_enc_withproj=args.pretrained_enc_withproj,
                                             pretrained_rdm_ckpt=args.pretrained_rdm_ckpt,
                                             pretrained_rdm_cfg=args.pretrained_rdm_cfg)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # Log parameters
    n_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))
    if global_rank == 0:
        log_writer.add_scalar('num_params', n_params / 1e6, 0)
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.evaluate:
        print("Start evaluating")
        gen_img(model, args, 0, batch_size=16, log_writer=log_writer, cfg=0)
        if args.cfg > 0:
            gen_img(model, args, 0, batch_size=16, log_writer=log_writer, cfg=args.cfg)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 40 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        if args.output_dir and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            if args.pretrained_rdm_ckpt is not None or args.use_class_label:
                gen_img(model, args, epoch, batch_size=16, log_writer=log_writer, cfg=0)
                if args.cfg > 0:
                    gen_img(model, args, epoch, batch_size=16, log_writer=log_writer, cfg=args.cfg)

        misc.save_model_last(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
