import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import copy

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_adm import train_one_epoch, gen_img

from pixel_generator.guided_diffusion.resample import create_named_schedule_sampler
from pixel_generator.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_gaussian_diffusion,
    args_to_dict,
)

import pretrained_enc.models_pretrained_enc as models_pretrained_enc
from rdm.util import load_model
from rdm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf


def get_args_parser():
    parser = argparse.ArgumentParser('ADM training', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # config
    parser.add_argument('--image_size', default=256, type=int,
                        help='images input size')

    parser.add_argument('--config', type=str, help='config file')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-6, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--cosine_lr', action='store_true',
                        help='Use cosine lr scheduling.')
    parser.add_argument('--warmup_epochs', default=0, type=int)

    # ADM parameters
    parser.add_argument('--schedule_sampler', default='uniform', type=str)
    parser.add_argument('--lr_anneal_steps', default=0, type=int)
    parser.add_argument('--microbatch', default=-1, type=int)
    parser.add_argument('--ema_rate', default=0.9999, type=float)
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--fp16_scale_growth', default=1e-3, type=float)

    # ADM model parameters
    parser.add_argument('--num_channels', default=128, type=int)
    parser.add_argument('--num_res_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--num_heads_upsample', default=-1, type=int)
    parser.add_argument('--num_head_channels', default=-1, type=int)
    parser.add_argument('--attention_resolutions', default="16,8", type=str)
    parser.add_argument('--channel_mult', default="", type=str)
    parser.add_argument('--dropout', default=0.0, type=float)

    parser.add_argument('--class_cond', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--use_scale_shift_norm', action='store_true')
    parser.add_argument('--resblock_updown', action='store_true')
    parser.add_argument('--use_new_attention_order', action='store_true')

    # ADM diffusion parameters
    parser.add_argument('--learn_sigma', action='store_true')
    parser.add_argument('--use_kl', action='store_true')
    parser.add_argument('--predict_xstart', action='store_true')
    parser.add_argument('--rescale_timesteps', action='store_true')
    parser.add_argument('--rescale_learned_sigmas', action='store_true')
    parser.add_argument('--diffusion_steps', default=1000, type=int)
    parser.add_argument('--noise_schedule', default="linear", type=str)
    parser.add_argument('--timestep_respacing', default="", type=str)

    # RDM parameters
    parser.add_argument('--rep_cond', action='store_true')
    parser.add_argument('--rep_dim', default=256, type=int)
    parser.add_argument('--pretrained_enc_arch',  default=None, type=str)
    parser.add_argument('--pretrained_enc_path',  default=None, type=str)

    parser.add_argument('--rdm_steps',  default=250, type=int)
    parser.add_argument('--rdm_eta',  default=1.0, type=float)
    parser.add_argument('--pretrained_rdm_cfg',  default=None, type=str)
    parser.add_argument('--pretrained_rdm_ckpt',  default=None, type=str)

    # ADM generation parameters
    parser.add_argument('--evaluate', action='store_true', help="perform only evaluation")
    parser.add_argument('--eval_freq', type=int, default=8, help='evaluation frequency')
    parser.add_argument('--num_images', default=50000, type=int)
    parser.add_argument('--use_ddim', action='store_true')
    parser.add_argument('--gen_timestep_respacing', default="", type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')

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
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    transform_train = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # load model
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # pre-trained encoder
    if args.rep_cond:
        assert args.pretrained_enc_path is not None
        pretrained_encoder = models_pretrained_enc.__dict__[args.pretrained_enc_arch](proj_dim=args.rep_dim)
        # load pre-trained encoder parameters
        if 'moco' in args.pretrained_enc_arch:
            pretrained_encoder = models_pretrained_enc.load_pretrained_moco(pretrained_encoder, args.pretrained_enc_path)
        else:
            raise NotImplementedError

        for param in pretrained_encoder.parameters():
            param.requires_grad = False

        pretrained_encoder.to(device)
        pretrained_encoder.eval()
    else:
        pretrained_encoder = None

    # pre-trained RDM
    if args.rep_cond:
        rdm_config = OmegaConf.load(args.pretrained_rdm_cfg)
        ldm_model = load_model(rdm_config, args.pretrained_rdm_ckpt)
        rdm_sampler = DDIMSampler(ldm_model)
    else:
        rdm_sampler = None

    # sampling diffusion
    gen_diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        timestep_respacing=args.gen_timestep_respacing,
    )

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size

    print("base lr: %.2e" % (args.lr / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # Log parameters
    params = list(model_without_ddp.parameters())
    n_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))
    if global_rank == 0:
        log_writer.add_scalar('num_params', n_params / 1e6, 0)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    # Resume training or from pre-trained unconditional ADM
    if os.path.exists(os.path.join(args.resume, "checkpoint-last.pth")):
        resume_path = os.path.join(args.resume, "checkpoint-last.pth")
    else:
        resume_path = args.resume
    if resume_path:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint['model_ema']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

    if args.evaluate:
        print("Start evaluating")
        gen_img(model, model_without_ddp, gen_diffusion, ema_params, rdm_sampler, args, 0, batch_size=16, log_writer=log_writer, use_ema=True)
        return

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, diffusion, schedule_sampler, pretrained_encoder,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params)
            gen_img(model, model_without_ddp, gen_diffusion, ema_params, rdm_sampler, args, epoch, batch_size=16, log_writer=log_writer, use_ema=False)
            gen_img(model, model_without_ddp, gen_diffusion, ema_params, rdm_sampler, args, epoch, batch_size=16, log_writer=log_writer, use_ema=True)

        misc.save_model_last(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

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
