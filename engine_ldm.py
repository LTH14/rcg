import math
import sys
from typing import Iterable
import os

import torch

import util.misc as misc
import util.lr_sched as lr_sched

import cv2
import torch_fidelity
import numpy as np
import shutil


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0 and args.cosine_lr:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images = images.to(device, non_blocking=True)
        images = images * 2 - 1  # image to [-1, 1] to be compatible with LDM
        targets = targets.to(device, non_blocking=True)

        if args.class_cond:
            batch = {'image': images, 'class_label': targets}
        else:
            batch = {'image': images, 'class_label': torch.zeros_like(targets)}

        loss, loss_dict = model(x=None, c=None, batch=batch)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def gen_img(model, args, epoch, batch_size=16, log_writer=None):
    model.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "steps{}-eta{}".format(args.ldm_steps, args.eta))
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        with torch.no_grad():
            gen_images_batch = model(x=None, c=None, gen_img=True)
        gen_images_batch = misc.concat_all_gather(gen_images_batch)
        gen_images_batch = gen_images_batch.detach().cpu()

        # save img
        if misc.get_rank() == 0:
            for b_id in range(gen_images_batch.size(0)):
                if i*gen_images_batch.size(0)+b_id >= args.num_images:
                    break
                gen_img = np.clip(gen_images_batch[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255)
                gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(
                    os.path.join(save_folder, '{}.png'.format(str(i * gen_images_batch.size(0) + b_id).zfill(5))),
                    gen_img)

    # compute FID and IS
    if log_writer is not None:
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2='imagenet-val',
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        log_writer.add_scalar('fid', fid, epoch)
        log_writer.add_scalar('is', inception_score, epoch)
        print("FID: {}, Inception Score: {}".format(fid, inception_score))
        # remove temporal saving folder
        shutil.rmtree(save_folder)
