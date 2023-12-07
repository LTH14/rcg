from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed

from pixel_generator.mage.taming.models.vqgan import VQModel
from omegaconf import OmegaConf
import numpy as np
import scipy.stats as stats
import math
import pretrained_enc.models_pretrained_enc as models_pretrained_enc
from rdm.models.diffusion.ddim import DDIMSampler
from rdm.util import load_model


def mask_by_random_topk(mask_len, probs, temperature=1.0):
    mask_len = mask_len.squeeze()
    confidence = torch.log(probs) + torch.Tensor(temperature * np.random.gumbel(size=probs.shape)).cuda()
    sorted_confidence, _ = torch.sort(confidence, axis=-1)
    # Obtains cut off threshold given the mask lengths.
    cut_off = sorted_confidence[:, mask_len.long()-1:mask_len.long()]
    # Masks tokens with lower confidence.
    masking = (confidence <= cut_off)
    return masking


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        with torch.cuda.amp.autocast(enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale

        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            _, attn = self.attn(self.norm1(x))
            return attn
        else:
            y, _ = self.attn(self.norm1(x))
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        torch.nn.init.normal_(self.word_embeddings.weight, std=.02)
        torch.nn.init.normal_(self.position_embeddings.weight, std=.02)

    def forward(
        self, input_ids
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(mlm_hidden, word_embeddings)
        logits = logits + self.bias
        return logits


class MaskedGenerativeEncoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 vqgan_ckpt_path='vqgan_jax_strongaug.ckpt', use_rep=True, rep_dim=256,
                 rep_drop_prob=0.0,
                 use_class_label=False,
                 pretrained_enc_arch='mocov3_vit_base',
                 pretrained_enc_path='pretrained_enc_ckpts/mocov3/vitb.pth.tar',
                 pretrained_enc_proj_dim=256,
                 pretrained_enc_withproj=False,
                 pretrained_rdm_ckpt=None,
                 pretrained_rdm_cfg=None):
        super().__init__()
        assert not (use_rep and use_class_label)

        # --------------------------------------------------------------------------
        # VQGAN specifics
        vqgan_config = OmegaConf.load('config/mage/vqgan.yaml').model
        self.vqgan_cfg = vqgan_config

        self.codebook_size = vqgan_config.params.n_embed
        vocab_size = self.codebook_size + 1000 + 1  # 1024 codebook size, 1000 classes, 1 for mask token.
        self.fake_class_label = self.codebook_size + 1100 - 1024
        self.mask_token_label = vocab_size - 1
        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=embed_dim,
                                        max_position_embeddings=256+1,
                                        dropout=0.1)
        self.use_rep = use_rep
        self.use_class_label = use_class_label
        if self.use_rep:
            print("Use representation as condition!")
            self.latent_prior_proj = nn.Linear(rep_dim, embed_dim, bias=True)
        if self.use_class_label:
            print("Use class label as condition!")
            self.class_emb = nn.Embedding(1000, embed_dim)

        # CFG config
        self.rep_drop_prob = rep_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, rep_dim))
        torch.nn.init.normal_(self.fake_latent, std=.02)

        # MAGE variant masking ratio
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

        # --------------------------------------------------------------------------
        # MAGE encoder specifics
        dropout_rate = 0.1
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAGE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pad_with_cls_token = True

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))  # learnable pos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MlmLayer
        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=embed_dim, vocab_size=vocab_size)

        self.norm_pix_loss = norm_pix_loss

        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Load pre-trained encoder

        assert pretrained_enc_path is not None
        self.pretrained_encoder = models_pretrained_enc.__dict__[pretrained_enc_arch](proj_dim=pretrained_enc_proj_dim)
        # load pre-trained encoder parameters
        if 'moco' in pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_moco(self.pretrained_encoder,
                                                                                 pretrained_enc_path)
        elif 'dino' in pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_dino(self.pretrained_encoder,
                                                                                 pretrained_enc_path)
        elif 'ibot' in pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_ibot(self.pretrained_encoder,
                                                                                 pretrained_enc_path)
        elif 'deit' in pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_deit(self.pretrained_encoder,
                                                                                 pretrained_enc_path)
        else:
            raise NotImplementedError

        for param in self.pretrained_encoder.parameters():
            param.requires_grad = False

        self.pretrained_enc_withproj = pretrained_enc_withproj

        # --------------------------------------------------------------------------
        # Load pre-trained RDM sampler

        if pretrained_rdm_ckpt is not None and pretrained_rdm_cfg is not None:
            rdm_config = OmegaConf.load(pretrained_rdm_cfg)
            self.rdm_fake_class_label = rdm_config.model.params.cond_stage_config.params.n_classes - 1
            rdm_model = load_model(rdm_config, pretrained_rdm_ckpt)
            self.rdm_sampler = DDIMSampler(rdm_model)
        else:
            self.rdm_fake_class_label = 0

        # --------------------------------------------------------------------------
        # Load pre-trained VQGAN

        self.vqgan = VQModel(ddconfig=vqgan_config.params.ddconfig,
                             n_embed=vqgan_config.params.n_embed,
                             embed_dim=vqgan_config.params.embed_dim,
                             ckpt_path=vqgan_ckpt_path)
        for param in self.vqgan.parameters():
            param.requires_grad = False

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, rep, class_label):
        # tokenization
        with torch.no_grad():
            z_q, _, token_tuple = self.vqgan.encode(x)

        _, _, token_indices = token_tuple
        token_indices = token_indices.reshape(z_q.size(0), -1)
        gt_indices = token_indices.clone().detach().long()

        # masking
        bsz, seq_len = token_indices.size()
        mask_ratio_min = self.mask_ratio_min
        mask_rate = self.mask_ratio_generator.rvs(1)[0]

        num_dropped_tokens = int(np.ceil(seq_len * mask_ratio_min))
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))

        # it is possible that two elements of the noise is the same, so do a while loop to avoid it
        while True:
            noise = torch.rand(bsz, seq_len, device=x.device)  # noise in [0, 1]
            sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
            cutoff_drop = sorted_noise[:, num_dropped_tokens-1:num_dropped_tokens]
            cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]
            token_drop_mask = (noise <= cutoff_drop).float()
            token_all_mask = (noise <= cutoff_mask).float()
            if token_drop_mask.sum() == bsz*num_dropped_tokens and token_all_mask.sum() == bsz*num_masked_tokens:
                break
            else:
                print("Rerandom the noise!")
        # print(mask_rate, num_dropped_tokens, num_masked_tokens, token_drop_mask.sum(dim=1), token_all_mask.sum(dim=1))
        token_indices[token_all_mask.nonzero(as_tuple=True)] = self.mask_token_label
        # print("Masekd num token:", torch.sum(token_indices == self.mask_token_label, dim=1))

        # concate class token
        token_indices = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = self.fake_class_label
        token_drop_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_drop_mask], dim=1)
        token_all_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_all_mask], dim=1)
        token_indices = token_indices.long()
        # bert embedding
        input_embeddings = self.token_emb(token_indices)
        # print("Input embedding shape:", input_embeddings.shape)
        bsz, seq_len, emb_dim = input_embeddings.shape

        # dropping
        token_keep_mask = 1 - token_drop_mask
        input_embeddings_after_drop = input_embeddings[token_keep_mask.nonzero(as_tuple=True)].reshape(bsz, -1, emb_dim)
        # print("Input embedding after drop shape:", input_embeddings_after_drop.shape)

        # replace fake class token with rep
        if self.use_rep:
            # cfg by masking representation
            drop_rep_mask = torch.rand(bsz) < self.rep_drop_prob
            drop_rep_mask = drop_rep_mask.unsqueeze(-1).cuda().float()
            rep = drop_rep_mask * self.fake_latent + (1 - drop_rep_mask) * rep

            rep = self.latent_prior_proj(rep)
            input_embeddings_after_drop[:, 0] = rep
        # class-conditional MAGE
        if self.use_class_label:
            class_emb = self.class_emb(class_label)
            input_embeddings_after_drop[:, 0] = class_emb

        # apply Transformer blocks
        x = input_embeddings_after_drop
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # print("Encoder representation shape:", x.shape)

        return x, gt_indices, token_drop_mask, token_all_mask

    def forward_decoder(self, x, token_drop_mask, token_all_mask):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        if self.pad_with_cls_token:
            mask_tokens = x[:, 0:1].repeat(1, token_all_mask.shape[1], 1)
        else:
            mask_tokens = self.mask_token.repeat(token_all_mask.shape[0], token_all_mask.shape[1], 1)

        # put undropped tokens into original sequence
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - token_drop_mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # set undropped but masked positions with mask
        x_after_pad = torch.where(token_all_mask.unsqueeze(-1).bool(), mask_tokens, x_after_pad)

        # add pos embed
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)
        # print("Logits shape:", x.shape)

        return x

    def normalize(self, x):
        return (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)

    def forward_loss(self, gt_indices, logits, mask):
        bsz, seq_len = gt_indices.size()
        # logits and mask are with seq_len+1 but gt_indices is with seq_len
        loss = self.criterion(logits[:, 1:, :self.codebook_size].reshape(bsz*seq_len, -1), gt_indices.reshape(bsz*seq_len))
        loss = loss.reshape(bsz, seq_len)
        loss = (loss * mask[:, 1:]).sum() / mask[:, 1:].sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, class_label,
                gen_image=False, bsz=None, num_iter=None, choice_temperature=None,
                sampled_rep=None, rdm_steps=250, eta=1.0, cfg=0.0, class_label_gen=None):
        if gen_image:
            return self.gen_image(bsz, num_iter, choice_temperature, sampled_rep, rdm_steps, eta, cfg, class_label_gen)

        self.pretrained_encoder.eval()
        with torch.no_grad():
            mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            std = torch.Tensor([0.229, 0.224, 0.225]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            x_normalized = (imgs - mean) / std
            x_normalized = torch.nn.functional.interpolate(x_normalized, 224, mode='bicubic')
            rep = self.pretrained_encoder.forward_features(x_normalized)
            if self.pretrained_enc_withproj:
                rep = self.pretrained_encoder.head(rep)
            rep_std = torch.std(rep, dim=1, keepdim=True)
            rep_mean = torch.mean(rep, dim=1, keepdim=True)
            rep = (rep - rep_mean) / rep_std

        latent, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(imgs, rep, class_label)
        logits = self.forward_decoder(latent, token_drop_mask, token_all_mask)
        loss = self.forward_loss(gt_indices, logits, token_all_mask)
        return loss, imgs, token_all_mask

    def gen_image(self, bsz, num_iter=12, choice_temperature=4.5, sampled_rep=None, rdm_steps=250, eta=1.0,
                  cfg=0.0, class_label=None):
        mask_token_id = self.mask_token_label
        unknown_number_in_the_beginning = 256
        _CONFIDENCE_OF_KNOWN_TOKENS = +np.inf

        initial_token_indices = mask_token_id * torch.ones(bsz, unknown_number_in_the_beginning)

        token_indices = initial_token_indices.cuda()

        # Sample representation from RDM
        if self.use_rep and sampled_rep is None:
            with self.rdm_sampler.model.ema_scope("Plotting"):
                shape = [self.rdm_sampler.model.model.diffusion_model.in_channels,
                         self.rdm_sampler.model.model.diffusion_model.image_size,
                         self.rdm_sampler.model.model.diffusion_model.image_size]
                if self.rdm_sampler.model.class_cond:
                    cond = {"class_label": class_label}
                else:
                    class_label = self.rdm_fake_class_label * torch.ones(bsz).cuda().long()
                    cond = {"class_label": class_label}
                cond = self.rdm_sampler.model.get_learned_conditioning(cond)

                sampled_rep, _ = self.rdm_sampler.sample(rdm_steps, conditioning=cond, batch_size=bsz,
                                                                  shape=shape,
                                                                  eta=eta, verbose=False)
                sampled_rep = sampled_rep.squeeze(-1).squeeze(-1)

                # add uncond for cfg
                if cfg > 0:
                    uncond_rep = self.fake_latent.repeat(bsz, 1)
                    sampled_rep = torch.cat([sampled_rep, uncond_rep], dim=0)
        else:
            class_label = None

        if self.use_class_label:
            assert cfg == 0
            class_label = torch.randint(0, 1000, (bsz,)).cuda()

        # Parallel decoding of MAGE
        for step in range(num_iter):
            cur_ids = token_indices.clone().long()

            # duplicate for cfg
            if cfg > 0:
                token_indices = torch.cat([token_indices, token_indices], dim=0)

            token_indices = torch.cat(
                [torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
            token_indices[:, 0] = self.fake_class_label
            token_indices = token_indices.long()
            token_all_mask = token_indices == mask_token_id

            token_drop_mask = torch.zeros_like(token_indices)

            # token embedding
            input_embeddings = self.token_emb(token_indices)
            if self.use_rep:
                input_embeddings[:, 0] = self.latent_prior_proj(sampled_rep)
            if self.use_class_label:
                class_prior = self.class_emb(class_label)
                input_embeddings[:, 0] = class_prior

            # encoder
            x = input_embeddings
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

            # decoder
            logits = self.forward_decoder(x, token_drop_mask, token_all_mask)
            logits = logits[:, 1:, :self.codebook_size]

            # cfg
            if cfg > 0:
                cond_logits = logits[:bsz]
                neg_logits = logits[bsz:]
                # linear increase cfg
                cfg_iter = cfg * (step + 1) / num_iter
                logits = cond_logits - cfg_iter * (neg_logits - cond_logits)

            # get token prediction
            sample_dist = torch.distributions.categorical.Categorical(logits=logits)
            sampled_ids = sample_dist.sample()

            # get ids for next step
            unknown_map = (cur_ids == mask_token_id)
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1. * (step + 1) / num_iter

            mask_ratio = np.cos(math.pi / 2. * ratio)

            # sample ids according to prediction confidence
            probs = torch.nn.functional.softmax(logits, dim=-1)
            selected_probs = torch.squeeze(
                torch.gather(probs, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)

            selected_probs = torch.where(unknown_map, selected_probs.double(), _CONFIDENCE_OF_KNOWN_TOKENS).float()

            mask_len = torch.Tensor([np.floor(unknown_number_in_the_beginning * mask_ratio)]).cuda()
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(unknown_map, dim=-1, keepdims=True) - 1, mask_len))

            # Sample masking tokens for next iteration
            masking = mask_by_random_topk(mask_len[0], selected_probs, choice_temperature * (1 - ratio))
            # Masks tokens with lower confidence.
            token_indices = torch.where(masking, mask_token_id, sampled_ids)

        # vqgan visualization
        z_q = self.vqgan.quantize.get_codebook_entry(sampled_ids, shape=(bsz, 16, 16, self.vqgan_cfg.params.embed_dim))
        gen_images = self.vqgan.decode(z_q)
        return gen_images, class_label


def mage_vit_base_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mage_vit_large_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mage_vit_huge_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=1280, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
