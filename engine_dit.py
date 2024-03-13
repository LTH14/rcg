import math
import sys
import functools
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from pixel_generator.guided_diffusion.nn import update_ema
import os
import copy
import numpy as np
import cv2
import shutil
import torch_fidelity


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def train_one_epoch(model: torch.nn.Module,
                    vae,
                    diffusion,
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

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_const(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images = images.to(device, non_blocking=True)

        # get rep
        if pretrained_encoder is not None:
            pretrained_encoder.eval()
            with torch.no_grad():
                mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                std = torch.Tensor([0.229, 0.224, 0.225]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                x_normalized = (images - mean) / std
                x_normalized = torch.nn.functional.interpolate(x_normalized, 224, mode='bicubic')
                rep = pretrained_encoder(x_normalized)
                rep_std = torch.std(rep, dim=1, keepdim=True)
                rep_mean = torch.mean(rep, dim=1, keepdim=True)
                rep = (rep - rep_mean) / rep_std
        else:
            rep = None

        images = images * 2 - 1  # image to [-1, 1] to be compatible with DiT
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(images).latent_dist.sample().mul_(0.18215)

        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        model_kwargs = dict(y=targets.long().cuda() if args.class_cond else torch.zeros_like(targets).long().cuda(),
                            rep=rep)  # uncond generation
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()

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


def gen_img(model, model_without_ddp, vae, diffusion, ema_params, rdm_sampler, args, epoch, batch_size=16, log_writer=None, use_ema=True, cfg=1.0):
    model.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "step{}-cfg{}".format(args.num_sampling_steps, cfg))
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
        else:
            sampled_rep = None

        with torch.no_grad():
            # Setup classifier-free guidance:
            z = torch.randn(batch_size, model_without_ddp.in_channels, args.latent_size, args.latent_size).cuda()
            if args.class_cond:
                y = torch.randint(0, 1000, (batch_size,)).long().cuda()
                y_null = 1000*torch.ones(batch_size).long().cuda()
            else:
                y = torch.zeros(batch_size).long().cuda()
            if not cfg == 1.0:
                z = torch.cat([z, z], 0)
                if args.class_cond:
                    y = torch.cat([y, y_null], 0)
                else:
                    y = torch.cat([y, y], 0)
                if args.rep_cond:
                    sampled_rep = torch.cat([sampled_rep, model_without_ddp.fake_latent.repeat(batch_size, 1)], 0)
                model_kwargs = dict(y=y, cfg_scale=cfg, rep=sampled_rep)
                sample_fn = model_without_ddp.forward_with_cfg
            else:
                model_kwargs = dict(y=y, rep=sampled_rep)
                sample_fn = model_without_ddp.forward

            # Sample images:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=torch.device(args.device)
            )
            if not cfg == 1.0:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            gen_images_batch = vae.decode(samples / 0.18215).sample
            gen_images_batch = (gen_images_batch + 1) / 2

        gen_images_batch = concat_all_gather(gen_images_batch)
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
        if args.dataset == "imagenet":
            val_folder = 'imagenet-val'
        elif args.dataset == "cifar10":
            val_folder = 'cifar10-train'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=val_folder,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        if cfg == 1.0:
            log_writer.add_scalar('fid', fid, epoch)
            log_writer.add_scalar('is', inception_score, epoch)
        else:
            log_writer.add_scalar('fid_cfg{}'.format(cfg), fid, epoch)
            log_writer.add_scalar('is_cfg{}'.format(cfg), inception_score, epoch)
        print("FID: {}, Inception Score: {}".format(fid, inception_score))
        # remove temporal saving folder
        shutil.rmtree(save_folder)