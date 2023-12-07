import math
import sys
import functools
from typing import Iterable
import os
import copy

import torch

import util.misc as misc
import cv2
import torch_fidelity
import numpy as np
import shutil

from pixel_generator.guided_diffusion import dist_util
from pixel_generator.guided_diffusion.resample import LossAwareSampler
from pixel_generator.guided_diffusion.nn import update_ema


def train_one_epoch(model: torch.nn.Module,
                    diffusion,
                    schedule_sampler,
                    pretrained_encoder,
                    model_params, ema_params,
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

        images = images.to(device, non_blocking=True)
        images = images * 2 - 1  # image to [-1, 1] to be compatible with ADM

        model_kwargs = {}
        if args.class_cond:
            model_kwargs["y"] = targets.to(device, non_blocking=True)

        # get loss
        t, weights = schedule_sampler.sample(images.shape[0], dist_util.dev())

        compute_losses = functools.partial(
            diffusion.training_losses,
            model,
            images,
            t,
            pretrained_encoder,
            model_kwargs=model_kwargs,
        )

        loss = compute_losses()

        if isinstance(schedule_sampler, LossAwareSampler):
            schedule_sampler.update_with_local_losses(
                t, loss["loss"].detach()
            )

        loss = (loss["loss"] * weights).mean()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # update ema
        update_ema(ema_params, model_params, rate=args.ema_rate)

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


def gen_img(model, model_without_ddp, diffusion, ema_params, rdm_sampler, args, epoch, batch_size=16, log_writer=None, use_ema=False):
    model.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "steps{}".format(args.gen_timestep_respacing))
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        # sample representation
        if args.rep_cond:
            with rdm_sampler.model.ema_scope("Plotting"):
                shape = [rdm_sampler.model.model.diffusion_model.in_channels,
                         rdm_sampler.model.model.diffusion_model.image_size,
                         rdm_sampler.model.model.diffusion_model.image_size]
                cond = {"class_label": torch.zeros(batch_size).cuda().long()}
                cond = rdm_sampler.model.get_learned_conditioning(cond)

                sampled_rep, _ = rdm_sampler.sample(args.rdm_steps, conditioning=cond, batch_size=batch_size,
                                                    shape=shape, eta=args.rdm_eta, verbose=False)
                sampled_rep = sampled_rep.squeeze(-1).squeeze(-1)
            model_kwargs = {'rep': sampled_rep}
        elif args.class_cond:
            model_kwargs = {'y': torch.randint(0, 1000, (batch_size,)).cuda()}
        else:
            model_kwargs = None

        with torch.no_grad():
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            gen_images_batch = sample_fn(
                model,
                (batch_size, 3, args.image_size, args.image_size),
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )
            gen_images_batch = (gen_images_batch + 1) / 2

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

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

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
        if use_ema:
            log_writer.add_scalar('fid_ema', fid, epoch)
            log_writer.add_scalar('is_ema', inception_score, epoch)
            print("EMA FID: {}, EMA Inception Score: {}".format(fid, inception_score))
        else:
            log_writer.add_scalar('fid', fid, epoch)
            log_writer.add_scalar('is', inception_score, epoch)
            print("FID: {}, Inception Score: {}".format(fid, inception_score))
        # remove temporal saving folder
        shutil.rmtree(save_folder)
