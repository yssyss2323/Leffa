import contextvars
import logging
import math
import os
import random
import re

import time
from io import BytesIO
from math import pi as PI
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
    UNet2DConditionModel,
)
from diffusers.training_utils import compute_snr
from manifold.clients.python import ManifoldClient
from safetensors.torch import load_model
from torcheval.meta.metrics.utils import tensor_to_pil
from torchmultimodal import _PATH_MANAGER
from torchtnt.utils.distributed import get_global_rank

logger: logging.Logger = logging.getLogger(__name__)

# Contextvars used for learning flow fields in attention
image_var = contextvars.ContextVar("image", default=None)
condition_image_var = contextvars.ContextVar("condition_image", default=None)
mask_var = contextvars.ContextVar("mask", default=None)
use_attn_loss_var = contextvars.ContextVar("use_attn_loss", default=False)
attn_loss_var = contextvars.ContextVar("attn_loss", default=[])
use_flow_loss_var = contextvars.ContextVar("use_flow_loss", default=False)
flow_loss_var = contextvars.ContextVar("flow_loss", default=[])
file_name_var = contextvars.ContextVar("file_name", default=[])


class CatVtonModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "",
        pretrained_vae_name_or_path: str = "",
        pretrained_model: str = "",
        new_in_channels=9,  # noisy_image: 4, mask: 1, masked_image: 4
        height: int = 1024,
        width: int = 768,
        snr_gamma: Optional[float] = None,
        garment_dropout_ratio: float = 0.0,
        use_dream: bool = False,
        dream_detail_preservation: float = 10.0,
        use_garment_mask: bool = False,
        only_optimize_unet_attn1: bool = True,
        use_learning_flow_in_attention: bool = False,
        learning_flow_in_attention_lambda: float = 0.1,
        learning_flow_in_attention_stop_timestep: int = -1,
        use_attention_flow_loss: bool = False,
        attention_flow_loss_lambda: float = 0.1,
        use_pixel_space_supervision: bool = False,
        pixel_space_supervision_lambda: float = 10.0,
        use_densepose: bool = False,
    ):
        super(CatVtonModel, self).__init__()

        self.height = height
        self.width = width
        self.snr_gamma = snr_gamma
        self.garment_dropout_ratio = garment_dropout_ratio
        self.use_dream = use_dream
        self.dream_detail_preservation = dream_detail_preservation
        self.use_garment_mask = use_garment_mask
        self.only_optimize_unet_attn1 = only_optimize_unet_attn1
        self.use_learning_flow_in_attention = use_learning_flow_in_attention
        self.learning_flow_in_attention_lambda = learning_flow_in_attention_lambda
        self.learning_flow_in_attention_stop_timestep = (
            learning_flow_in_attention_stop_timestep
        )
        self.use_attention_flow_loss = use_attention_flow_loss
        self.attention_flow_loss_lambda = attention_flow_loss_lambda
        self.use_pixel_space_supervision = use_pixel_space_supervision
        self.pixel_space_supervision_lambda = pixel_space_supervision_lambda
        self.use_densepose = use_densepose

        self.build_models(
            pretrained_model_name_or_path,
            pretrained_vae_name_or_path,
            pretrained_model,
            new_in_channels,
        )

    def get_local_path(self, manifold_path, files_to_cache):
        for _file in files_to_cache:
            local_path = _PATH_MANAGER.get_local_path(
                os.path.join(manifold_path, _file)
            )
        local_path = local_path.replace(f"/{_file}", "")
        return local_path

    def build_models(
        self,
        pretrained_model_name_or_path: str = "",
        pretrained_vae_name_or_path: str = "",
        pretrained_model: str = "",
        new_in_channels=9,  # noisy_image: 4, mask: 1, masked_image: 4
    ):

        diffusion_model_type = ""
        if pretrained_model_name_or_path.startswith("manifold://"):
            if "stable-diffusion-inpainting" in pretrained_model_name_or_path:
                diffusion_model_type = "sd15"
                files_to_cache = [
                    "scheduler/scheduler_config.json",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.bin",
                    "unet/config.json",
                    "unet/diffusion_pytorch_model.bin",
                ]
            elif (
                "stable-diffusion-xl-1.0-inpainting-0.1"
                in pretrained_model_name_or_path
            ):
                diffusion_model_type = "sdxl"
                files_to_cache = [
                    "scheduler/scheduler_config.json",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.safetensors",
                    "unet/config.json",
                    "unet/diffusion_pytorch_model.safetensors",
                ]
            elif "FLUX.1" in pretrained_model_name_or_path:
                diffusion_model_type = "flux"
                files_to_cache = [
                    "scheduler/scheduler_config.json",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.safetensors",
                    "transformer/config.json",
                    "transformer/diffusion_pytorch_model.safetensors",
                ]
            else:
                raise ValueError(
                    f"pretrained_model_name_or_path: {pretrained_model_name_or_path} is not supported. Please provide a valid manifold path."
                )
            pretrained_model_name_or_path = self.get_local_path(
                pretrained_model_name_or_path, files_to_cache
            )

        # Noise Scheduler
        if diffusion_model_type in ["sd15", "sdxl"]:
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="scheduler",
                # for SD1.5, set to False
                rescale_betas_zero_snr=(
                    False if diffusion_model_type == "sd15" else True
                ),
            )
        elif diffusion_model_type == "flux":
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler"
            )
        # VAE
        if (
            pretrained_vae_name_or_path != ""
            and pretrained_vae_name_or_path is not None
        ):
            files_to_cache = [
                "config.json",
                "diffusion_pytorch_model.safetensors",
            ]
            pretrained_vae_name_or_path = self.get_local_path(
                pretrained_vae_name_or_path, files_to_cache
            )
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_vae_name_or_path,
            )
            logger.info(
                "Load pretrained vae from {}".format(pretrained_vae_name_or_path)
            )
        else:
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="vae",
            )
        # unet / transformer
        if diffusion_model_type in ["sd15", "sdxl"]:
            self.unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="unet",
                low_cpu_mem_usage=False,
                device_map=None,
            )
        elif diffusion_model_type == "flux":
            self.unet = FluxTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                low_cpu_mem_usage=False,
                device_map=None,
            )

        if diffusion_model_type in ["sd15", "sdxl"]:
            unet_conv_in_channel_changed = (
                self.unet.config.in_channels != new_in_channels
            )
            if unet_conv_in_channel_changed:
                self.unet.conv_in = self.replace_conv_in_layer(
                    self.unet, new_in_channels
                )
                self.unet.config.in_channels = new_in_channels
            unet_conv_out_channel_changed = (
                self.unet.config.out_channels != self.vae.config.latent_channels
            )
            if unet_conv_out_channel_changed:
                self.unet.conv_out = self.replace_conv_out_layer(
                    self.unet, self.vae.config.latent_channels
                )
                self.unet.config.out_channels = self.vae.config.latent_channels

            # add skip cross attention
            add_skip_cross_attention(self.unet)
            if (
                "stable-diffusion-xl-1.0-inpainting-0.1"
                in pretrained_model_name_or_path
            ):
                self.unet.config.addition_embed_type = None
        elif diffusion_model_type == "flux":
            # TODO: flux not finish.
            pass

        # Load pretrained model
        if pretrained_model != "" and pretrained_model is not None:
            local_pretrained_model = _PATH_MANAGER.get_local_path(pretrained_model)
            self.load_state_dict(torch.load(local_pretrained_model, map_location="cpu"))
            logger.info("Load pretrained model from {}".format(pretrained_model))

        if diffusion_model_type in ["sd15", "sdxl"]:
            # # Load CatVton official model
            # self.trainable_module = get_trainable_module(self.unet, "attention")
            # attn_ckpt = "manifold://genads_models/tree/zijianzhou/model/CatVTON"
            # auto_attn_ckpt_load(self.trainable_module, attn_ckpt, "mix")

            # freeze VAE and part of UNet
            self.vae.requires_grad_(False)
            if self.only_optimize_unet_attn1:
                self.unet.requires_grad_(False)
                for n, p in self.unet.named_parameters():
                    if "attn1" in n:
                        p.requires_grad_(True)
            else:
                self.unet.requires_grad_(True)
            if unet_conv_in_channel_changed:
                self.unet.conv_in.requires_grad_(True)
            if unet_conv_out_channel_changed:
                self.unet.conv_out.requires_grad_(True)
        elif diffusion_model_type == "flux":
            self.vae.requires_grad_(False)
            self.unet.requires_grad_(True)

    def replace_conv_in_layer(self, unet_model, new_in_channels):
        original_conv_in = unet_model.conv_in

        if original_conv_in.in_channels == new_in_channels:
            return original_conv_in

        new_conv_in = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=original_conv_in.out_channels,
            kernel_size=original_conv_in.kernel_size,
            padding=1,
        )
        new_conv_in.weight.data.zero_()
        new_conv_in.bias.data = original_conv_in.bias.data.clone()
        if original_conv_in.in_channels < new_in_channels:
            new_conv_in.weight.data[:, : original_conv_in.in_channels] = (
                original_conv_in.weight.data
            )
        else:
            new_conv_in.weight.data[:, :new_in_channels] = original_conv_in.weight.data[
                :, :new_in_channels
            ]
        return new_conv_in

    def replace_conv_out_layer(self, unet_model, new_out_channels):
        original_conv_out = unet_model.conv_out

        if original_conv_out.out_channels == new_out_channels:
            return original_conv_out

        new_conv_out = torch.nn.Conv2d(
            in_channels=original_conv_out.in_channels,
            out_channels=new_out_channels,
            kernel_size=original_conv_out.kernel_size,
            padding=1,
        )
        new_conv_out.weight.data.zero_()
        new_conv_out.bias.data[: original_conv_out.out_channels] = (
            original_conv_out.bias.data.clone()
        )
        if original_conv_out.out_channels < new_out_channels:
            new_conv_out.weight.data[: original_conv_out.out_channels] = (
                original_conv_out.weight.data
            )
        else:
            new_conv_out.weight.data[:new_out_channels] = original_conv_out.weight.data[
                :new_out_channels
            ]
        return new_conv_out

    def compute_time_ids(self, original_size, crops_coords_top_left, dtype):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (self.height, self.width)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(dtype=dtype)
        return add_time_ids

    def forward(self, batch: Dict[str, Any]):
        concat_dim = -2  # y axis

        image = batch["image"].to(device=self.vae.device, dtype=self.vae.dtype)
        condition_image = batch["cloth_pure"].to(
            device=self.vae.device, dtype=self.vae.dtype
        )
        mask = batch["inpaint_mask"].to(device=self.vae.device, dtype=self.vae.dtype)
        masked_image = image * (mask < 0.5)
        if self.use_garment_mask:
            garment_mask = batch["cloth_mask"].to(
                device=self.vae.device, dtype=self.vae.dtype
            )
        if self.use_densepose:
            densepose = batch["densepose"].to(
                device=self.vae.device, dtype=self.vae.dtype
            )
            if "cloth_densepose" in batch:
                cloth_densepose = batch["cloth_densepose"].to(
                    device=self.vae.device, dtype=self.vae.dtype
                )

        if self.use_learning_flow_in_attention or self.use_attention_flow_loss:
            image_var.set(image)
            condition_image_var.set(condition_image)
            warped_mask = batch["warped_mask"].to(
                device=self.vae.device, dtype=self.vae.dtype
            )
            mask_var.set(warped_mask)
            use_attn_loss_var.set(self.use_learning_flow_in_attention)
            attn_loss_var.set([])
            use_flow_loss_var.set(self.use_attention_flow_loss)
            flow_loss_var.set([])
            file_name_var.set([])

        # 1. VAE encoding
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            masked_latent = self.vae.encode(masked_image).latent_dist.sample()
            condition_latent = self.vae.encode(condition_image).latent_dist.sample()
            if self.use_densepose:
                if "cloth_densepose" not in batch:
                    densepose_latent = self.vae.encode(densepose).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        masked_latent = masked_latent * self.vae.config.scaling_factor
        condition_latent = condition_latent * self.vae.config.scaling_factor
        if self.use_densepose:
            if "cloth_densepose" in batch:
                densepose_latent = F.interpolate(
                    densepose, size=masked_latent.shape[-2:], mode="nearest"
                )
                cloth_densepose_latent = F.interpolate(
                    cloth_densepose, size=masked_latent.shape[-2:], mode="nearest"
                )
            else:
                densepose_latent = densepose_latent * self.vae.config.scaling_factor
        mask_latent = F.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
        if self.use_garment_mask:
            garment_mask_latent = F.interpolate(
                garment_mask, size=condition_latent.shape[-2:], mode="nearest"
            )

        # 2. concat latents
        latent_concat = torch.cat([latent, condition_latent], dim=concat_dim)
        masked_latent_concat = torch.cat(
            [
                masked_latent,
                (
                    torch.zeros_like(condition_latent)
                    if random.random() < self.garment_dropout_ratio
                    else condition_latent
                ),
            ],
            dim=concat_dim,
        )
        mask_latent_concat = torch.cat(
            [
                mask_latent,
                (
                    garment_mask_latent
                    if self.use_garment_mask
                    else torch.zeros_like(mask_latent)
                ),
            ],
            dim=concat_dim,
        )
        if self.use_densepose:
            densepose_latent_concat = torch.cat(
                [
                    densepose_latent,
                    (
                        cloth_densepose_latent
                        if "cloth_densepose" in batch
                        else torch.zeros_like(densepose_latent)
                    ),
                ],
                dim=concat_dim,
            )

        # 3. prepare noise
        noise = torch.randn_like(latent_concat)
        batch_size = latent_concat.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=latent_concat.device,
        )
        noisy_latent_concat = self.noise_scheduler.add_noise(
            latent_concat, noise, timesteps
        )

        # 4. prepare model input
        input_list = [noisy_latent_concat, mask_latent_concat, masked_latent_concat]
        if self.use_densepose:
            input_list.append(densepose_latent_concat)
        latent_model_input = torch.cat(input_list, dim=1)

        outputs = {}
        outputs["timesteps"] = timesteps
        outputs["noise"] = noise
        outputs["latent_concat"] = latent_concat
        outputs["latent_model_input"] = latent_model_input

        if self.training:
            if self.use_dream:
                model_pred, loss = self.loss_with_dream(outputs)
            else:
                model_pred, loss = self.loss(outputs)
            outputs["model_pred"] = model_pred
            if self.use_learning_flow_in_attention:
                attn_loss = attn_loss_var.get()
                if len(attn_loss) > 0:
                    attn_loss = torch.stack(attn_loss).mean(dim=0)
                    if self.learning_flow_in_attention_stop_timestep > 0:
                        attn_loss = attn_loss * (
                            timesteps < self.learning_flow_in_attention_stop_timestep
                        )
                    attn_loss = attn_loss.mean()
                    loss += attn_loss * self.learning_flow_in_attention_lambda
                else:
                    attn_loss = torch.tensor(0.0).to(loss.device)
                outputs["attn_loss"] = attn_loss
            if self.use_attention_flow_loss:
                flow_loss = flow_loss_var.get()
                if len(flow_loss) > 0:
                    flow_loss = torch.stack(flow_loss).mean()
                    loss += flow_loss * self.attention_flow_loss_lambda
                else:
                    flow_loss = torch.tensor(0.0).to(loss.device)
                outputs["flow_loss"] = flow_loss
            if self.use_pixel_space_supervision:
                outputs["noisy_latent_concat"] = noisy_latent_concat
                pixel_space_loss = self.loss_pixel_space(outputs)
                outputs["pixel_space_loss"] = pixel_space_loss
                loss += pixel_space_loss * self.pixel_space_supervision_lambda
            outputs["loss"] = loss
        else:
            model_pred = self.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=None,
                return_dict=False,
            )[0]
            outputs["model_pred"] = model_pred
        return outputs

    def loss(self, inputs):
        timesteps = inputs["timesteps"]
        noise = inputs["noise"]
        model_input = inputs["latent_concat"]
        latent_model_input = inputs["latent_model_input"]

        model_pred = self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=None,
            return_dict=False,
        )[0]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = model_input
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack(
                    [snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        return model_pred, loss

    def loss_with_dream(self, inputs):
        noise = inputs["noise"]
        timesteps = inputs["timesteps"]
        model_input = inputs["latent_concat"]
        latent_model_input = inputs["latent_model_input"]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        latent_model_input, target = compute_dream_and_update_latents_for_inpaint(
            self.unet,
            self.noise_scheduler,
            timesteps,
            noise,
            latent_model_input,
            target,
            encoder_hidden_states=None,
            dream_detail_preservation=self.dream_detail_preservation,
        )

        model_pred = self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=None,
            return_dict=False,
        )[0]

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = torch.stack(
                [snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1
            ).min(dim=1)[0]
            if self.noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return model_pred, loss

    def loss_pixel_space(self, inputs):
        timesteps = inputs["timesteps"]
        latent_concat = inputs["latent_concat"]
        noisy_latent_concat = inputs["noisy_latent_concat"]
        model_pred = inputs["model_pred"]

        noisy_latent_concat_pred = self.noise_scheduler.add_noise(
            latent_concat, model_pred, timesteps
        )

        noisy_latent_concat = 1 / self.vae.config.scaling_factor * noisy_latent_concat
        noisy_image = self.vae.decode(noisy_latent_concat).sample

        noisy_latent_concat_pred = (
            1 / self.vae.config.scaling_factor * noisy_latent_concat_pred
        )
        noisy_image_pred = self.vae.decode(noisy_latent_concat_pred).sample
        loss = F.mse_loss(noisy_image, noisy_image_pred, reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        loss = loss.mean()
        return loss


class SkipAttnProcessor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self, hidden_size=None, cross_attention_dim=None, layer_name=None, **kwargs
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.layer_name = layer_name
        self.model_type = kwargs.get("model_type", "none")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        #####################################
        # Learning Flow Fields in Attention #
        #####################################

        do_attn_map_vis = False
        if (
            do_attn_map_vis
            and self.model_type == "none"
            and self.layer_name
            == "up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor"
        ):

            shape_map = {
                786432: (1024, 768),
                196608: (512, 384),
                49152: (256, 192),
                12288: (128, 96),
                3072: (64, 48),
                768: (32, 24),
                192: (16, 12),
                48: (8, 6),
                12: (4, 3),
            }
            temperature = 2.0
            attn_map_size = query.shape[2] // 2
            person_query = query[:, :, :attn_map_size]
            garment_key = key[:, :, attn_map_size:]
            scale_factor = 1 / math.sqrt(person_query.size(-1))
            attn_map = person_query @ garment_key.transpose(-2, -1) * scale_factor
            attn_map = torch.softmax(attn_map / temperature, dim=-1)
            # attn_map = torch.mean(attn_map, dim=1, keepdim=False)
            batch_size, head_num, token_num, _ = attn_map.shape
            h, w = shape_map[attn_map_size]
            attn_map = attn_map.reshape(batch_size, head_num, token_num, h, w)
            self.attn_map = attn_map

        training_layers = [
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor",  # 1/8
            "down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor",  # 1/8
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor",  # 1/16
            "down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor",  # 1/16
            "down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor",  # 1/32
            "down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor",  # 1/32
            # "mid_block.attentions.0.transformer_blocks.0.attn1.processor",  # 1/64
            "up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor",  # 1/32
            "up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor",  # 1/32
            "up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor",  # 1/32
            "up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor",  # 1/16
            "up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor",  # 1/16
            "up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor",  # 1/16
            "up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor",  # 1/8
            "up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor",  # 1/8
            "up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor",  # 1/8
        ]
        testing_layers = [
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor",  # 1/8
            "down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor",  # 1/8
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor",  # 1/16
            "down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor",  # 1/16
            # "down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor",  # 1/32
            # "down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor",  # 1/32
            # "mid_block.attentions.0.transformer_blocks.0.attn1.processor",  # 1/64
            # "up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor",  # 1/32
            # "up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor",  # 1/32
            # "up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor",  # 1/32
            "up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor",  # 1/16
            "up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor",  # 1/16
            "up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor",  # 1/16
            "up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor",  # 1/8
            "up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor",  # 1/8
            "up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor",  # 1/8
        ]
        sdxl_layers = [
            # SDXL
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor",
            "down_blocks.1.attentions.0.transformer_blocks.1.attn1.processor",
            "down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor",
            "down_blocks.1.attentions.1.transformer_blocks.1.attn1.processor",
            "down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor",
            "down_blocks.2.attentions.0.transformer_blocks.1.attn1.processor",
            "down_blocks.2.attentions.0.transformer_blocks.2.attn1.processor",
            "down_blocks.2.attentions.0.transformer_blocks.3.attn1.processor",
            "down_blocks.2.attentions.0.transformer_blocks.4.attn1.processor",
            "down_blocks.2.attentions.0.transformer_blocks.5.attn1.processor",
            "down_blocks.2.attentions.0.transformer_blocks.6.attn1.processor",
            "down_blocks.2.attentions.0.transformer_blocks.7.attn1.processor",
            "down_blocks.2.attentions.0.transformer_blocks.8.attn1.processor",
            "down_blocks.2.attentions.0.transformer_blocks.9.attn1.processor",
            "down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor",
            "down_blocks.2.attentions.1.transformer_blocks.1.attn1.processor",
            "down_blocks.2.attentions.1.transformer_blocks.2.attn1.processor",
            "down_blocks.2.attentions.1.transformer_blocks.3.attn1.processor",
            "down_blocks.2.attentions.1.transformer_blocks.4.attn1.processor",
            "down_blocks.2.attentions.1.transformer_blocks.5.attn1.processor",
            "down_blocks.2.attentions.1.transformer_blocks.6.attn1.processor",
            "down_blocks.2.attentions.1.transformer_blocks.7.attn1.processor",
            "down_blocks.2.attentions.1.transformer_blocks.8.attn1.processor",
            "down_blocks.2.attentions.1.transformer_blocks.9.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.0.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.1.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.2.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.3.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.4.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.5.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.6.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.7.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.8.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.9.attn1.processor",
            "up_blocks.0.attentions.0.transformer_blocks.0.attn1.processor",
            "up_blocks.0.attentions.0.transformer_blocks.1.attn1.processor",
            "up_blocks.0.attentions.0.transformer_blocks.2.attn1.processor",
            "up_blocks.0.attentions.0.transformer_blocks.3.attn1.processor",
            "up_blocks.0.attentions.0.transformer_blocks.4.attn1.processor",
            "up_blocks.0.attentions.0.transformer_blocks.5.attn1.processor",
            "up_blocks.0.attentions.0.transformer_blocks.6.attn1.processor",
            "up_blocks.0.attentions.0.transformer_blocks.7.attn1.processor",
            "up_blocks.0.attentions.0.transformer_blocks.8.attn1.processor",
            "up_blocks.0.attentions.0.transformer_blocks.9.attn1.processor",
            "up_blocks.0.attentions.1.transformer_blocks.0.attn1.processor",
            "up_blocks.0.attentions.1.transformer_blocks.1.attn1.processor",
            "up_blocks.0.attentions.1.transformer_blocks.2.attn1.processor",
            "up_blocks.0.attentions.1.transformer_blocks.3.attn1.processor",
            "up_blocks.0.attentions.1.transformer_blocks.4.attn1.processor",
            "up_blocks.0.attentions.1.transformer_blocks.5.attn1.processor",
            "up_blocks.0.attentions.1.transformer_blocks.6.attn1.processor",
            "up_blocks.0.attentions.1.transformer_blocks.7.attn1.processor",
            "up_blocks.0.attentions.1.transformer_blocks.8.attn1.processor",
            "up_blocks.0.attentions.1.transformer_blocks.9.attn1.processor",
            "up_blocks.0.attentions.2.transformer_blocks.0.attn1.processor",
            "up_blocks.0.attentions.2.transformer_blocks.1.attn1.processor",
            "up_blocks.0.attentions.2.transformer_blocks.2.attn1.processor",
            "up_blocks.0.attentions.2.transformer_blocks.3.attn1.processor",
            "up_blocks.0.attentions.2.transformer_blocks.4.attn1.processor",
            "up_blocks.0.attentions.2.transformer_blocks.5.attn1.processor",
            "up_blocks.0.attentions.2.transformer_blocks.6.attn1.processor",
            "up_blocks.0.attentions.2.transformer_blocks.7.attn1.processor",
            "up_blocks.0.attentions.2.transformer_blocks.8.attn1.processor",
            "up_blocks.0.attentions.2.transformer_blocks.9.attn1.processor",
            "up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor",
            "up_blocks.1.attentions.0.transformer_blocks.1.attn1.processor",
            "up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor",
            "up_blocks.1.attentions.1.transformer_blocks.1.attn1.processor",
            "up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor",
            "up_blocks.1.attentions.2.transformer_blocks.1.attn1.processor",
        ]
        # training_layers += sdxl_layers
        # testing_layers += sdxl_layers
        if self.training:
            do_attn_loss = self.layer_name in training_layers and (
                self.model_type != "unet_encoder"
            )
        else:
            do_attn_loss = self.layer_name in testing_layers and (
                self.model_type != "unet_encoder"
            )
            do_attn_loss = do_attn_loss and os.environ.get("INFER_STEP", "-1") in [
                # "10",
                # "20",
                "30",
                "40",
                "50",
            ]
        # os.environ["ZZJ_DEBUG"] = "true"
        if (
            (self.training and (use_attn_loss_var.get() or use_flow_loss_var.get()))
            or os.environ.get("ZZJ_DEBUG", "false") == "true"
        ) and do_attn_loss:
            # Config
            attn_type = "g_to_p"  # "both", "g_to_p", "p_to_g"
            temperature = 2.0
            attn_loss_type = "smooth_l1"
            upsampling_flow_map = True
            attn_source = "v2"
            norm_flow_map = False
            flow_loss_type = "v2"

            # 1. Get the attention map
            attn_map_size = query.shape[2] // 2
            if attn_source == "v1":
                scale_factor = 1 / math.sqrt(query.size(-1))
                attn_map = query @ key.transpose(-2, -1) * scale_factor
                # Note: can not use softmax, because it causes OOM.
                # attn_map = torch.softmax(attn_map, dim=-1)
                if attn_type in ["both", "g_to_p"]:
                    attn_map_g_to_p = attn_map[:, :, :attn_map_size, attn_map_size:]
                    attn_map_g_to_p = torch.mean(attn_map_g_to_p, dim=1, keepdim=False)
                if attn_type in ["both", "p_to_g"]:
                    attn_map_p_to_g = attn_map[:, :, attn_map_size:, :attn_map_size]
                    attn_map_p_to_g = torch.mean(attn_map_p_to_g, dim=1, keepdim=False)
                    attn_map_p_to_g = attn_map_p_to_g.transpose(-2, -1)
            elif attn_source == "v2":
                assert attn_type == "g_to_p"
                person_query = query[:, :, :attn_map_size]
                garment_key = key[:, :, attn_map_size:]
                scale_factor = 1 / math.sqrt(person_query.size(-1))
                attn_map_g_to_p = (
                    person_query @ garment_key.transpose(-2, -1) * scale_factor
                )
                # SoftMax
                attn_map_g_to_p = torch.softmax(attn_map_g_to_p / temperature, dim=-1)
                # HardMax
                # attn_map_g_to_p = torch.argmax(attn_map_g_to_p, dim=-1)
                # attn_map_g_to_p = F.one_hot(
                #     attn_map_g_to_p, num_classes=attn_map_size
                # ).to(query.dtype)
                attn_map_g_to_p = torch.mean(attn_map_g_to_p, dim=1, keepdim=False)

            # 2. Get the height and width of the attention map.
            attn_map_size_to_height_width_map = {
                # for virtual try-on
                3145728: (2048, 1536),
                786432: (1024, 768),
                196608: (512, 384),
                49152: (256, 192),
                12288: (128, 96),
                3072: (64, 48),
                768: (32, 24),
                192: (16, 12),
                48: (8, 6),
                12: (4, 3),
                # for pose transfer
                720896: (1024, 704),
                180224: (512, 352),
                45056: (256, 176),
                11264: (128, 88),
                2816: (64, 44),
                704: (32, 22),
                176: (16, 11),
            }

            height, width = attn_map_size_to_height_width_map[attn_map_size]

            # 3. Get the image, condition image and mask, then resize to the size of the attention map.
            _image = image_var.get()
            _condition_image = condition_image_var.get()
            _mask = mask_var.get()

            if not upsampling_flow_map:
                image = F.interpolate(_image, size=(height, width), mode="bilinear")
                condition_image = F.interpolate(
                    _condition_image, size=(height, width), mode="bilinear"
                )
                mask = F.interpolate(_mask, size=(height, width), mode="nearest")
            else:
                image = _image
                condition_image = _condition_image
                mask = _mask

            if image.shape[0] != query.shape[0]:
                # for classifier free guidance
                assert query.shape[0] == 2 * image.shape[0]
                image = torch.repeat_interleave(image, 2, dim=0)
                condition_image = torch.repeat_interleave(condition_image, 2, dim=0)
                mask = torch.repeat_interleave(mask, 2, dim=0)

            def normalize(this_flow_map):
                min_vals = this_flow_map.view(this_flow_map.size(0), -1).min(
                    dim=1, keepdim=True
                )[0]
                max_vals = this_flow_map.view(this_flow_map.size(0), -1).max(
                    dim=1, keepdim=True
                )[0]
                min_vals = min_vals.view(
                    this_flow_map.size(0), *([1] * (this_flow_map.dim() - 1))
                ).expand_as(this_flow_map)
                max_vals = max_vals.view(
                    this_flow_map.size(0), *([1] * (this_flow_map.dim() - 1))
                ).expand_as(this_flow_map)
                normalized_flow_map = (this_flow_map - min_vals) / (max_vals - min_vals)
                return normalized_flow_map

            def calcu_attn_loss(
                this_attn_map, coord_map, image, condition_image, mask, attn_type=""
            ):
                # 4. Calculate the flow map (flow field)
                flow_map = torch.matmul(this_attn_map, coord_map)
                flow_map = flow_map.reshape((-1, height, width, 2))
                if norm_flow_map:
                    flow_map = normalize(flow_map)
                    flow_map = flow_map * 2.0 - 1.0
                # debug: use coord_map as flow_map
                # flow_map = torch.cat(
                #     [
                #         coord_map.reshape((1, height, width, 2))
                #         for _ in range(batch_size)
                #     ],
                #     dim=0,
                # )

                if upsampling_flow_map:
                    image_height = image.shape[-2]
                    image_width = image.shape[-1]
                    flow_map = flow_map.permute(0, 3, 1, 2)
                    flow_map = F.interpolate(
                        flow_map, size=(image_height, image_width), mode="bilinear"
                    )
                    flow_map = flow_map.permute(0, 2, 3, 1)
                flow_map = torch.clamp_(flow_map, min=-1.0, max=1.0)

                # 5. Warp the condition_image (garment image) to image (person image)
                warped_image = F.grid_sample(condition_image, flow_map)

                if os.environ.get("ZZJ_DEBUG", "false") == "true":
                    # Warped image visualization
                    image_pil = tensor_to_pil((image + 1.0) / 2.0 * mask)
                    # image_pil = tensor_to_pil((image + 1.0) / 2.0)
                    condition_image_pil = tensor_to_pil((condition_image + 1.0) / 2.0)
                    warped_image_pil = tensor_to_pil((warped_image + 1.0) / 2.0 * mask)
                    # warped_image_pil = tensor_to_pil((warped_image + 1.0) / 2.0)
                    mask_pil = tensor_to_pil(torch.cat([mask for _ in range(3)], dim=1))
                    # [B, H, W, 2] -> [B, 3, H, W]
                    flow_map_color = flow_to_color(flow_map.cpu())[:, [2, 1, 0]]
                    flow_map_pil = tensor_to_pil(flow_map_color / 255.0)

                    file_name_list = file_name_var.get()
                    assert len(file_name_list) == 1
                    file_name = file_name_list[-1]
                    infer_step = os.environ.get("INFER_STEP", "-1")

                    #################
                    # save to local #
                    #################
                    """
                    save_dir = "/home/zijianzhou/learning_flow_in_attention"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    image_pil[1].save(
                        f"{save_dir}/{file_name}-{self.layer_name}-image.png"
                    )
                    condition_image_pil[1].save(
                        f"{save_dir}/{file_name}-{self.layer_name}-condition_image.png"
                    )
                    mask_pil[1].save(
                        f"{save_dir}/{file_name}-{self.layer_name}-mask.png"
                    )
                    warped_image_pil[1].save(
                        f"{save_dir}/{file_name}-{self.layer_name}-{attn_type}-{infer_step}-warped_image.png"
                    )
                    flow_map_pil[1].save(
                        f"{save_dir}/{file_name}-{self.layer_name}-{attn_type}-{infer_step}-flow_map.png"
                    )
                    # """

                    ####################
                    # save to manifold #
                    ####################
                    # """
                    manifold_path = "manifold://genads_models/tree/zijianzhou/output/cat_vton/learning_flow_in_attention/tmp"
                    match = re.match(r"manifold://([^/]+)/(.+)", manifold_path)
                    bucket: str = match.group(1)
                    save_root: str = match.group(2)

                    client = ManifoldClient.get_client(bucket)

                    def save_image_to_manifold(image, save_root, file_name):
                        image_stream = BytesIO()
                        image.save(image_stream, format="PNG")
                        image_stream.seek(0)
                        save_path = f"{save_root}/{file_name}"
                        client.sync_put(
                            save_path,
                            image_stream,
                            predicate=ManifoldClient.Predicates.AllowOverwrite,
                        )

                    if get_global_rank() == 0:
                        if not client.sync_exists(save_root):
                            client.sync_mkdir(path=save_root, recursive=True)
                    else:
                        while not client.sync_exists(save_root):
                            logging.info(
                                f"Waiting for manifold folder {save_root} to be created"
                            )
                            time.sleep(5)

                    # save_image_to_manifold(
                    #     image_pil[1],
                    #     save_root,
                    #     f"{file_name}-{self.layer_name}-image.png",
                    # )
                    # save_image_to_manifold(
                    #     condition_image_pil[1],
                    #     save_root,
                    #     f"{file_name}-{self.layer_name}-condition_image.png",
                    # )
                    # save_image_to_manifold(
                    #     mask_pil[1],
                    #     save_root,
                    #     f"{file_name}-{self.layer_name}-mask.png",
                    # )
                    save_image_to_manifold(
                        warped_image_pil[1],
                        save_root,
                        f"{file_name}-{self.layer_name}-{attn_type}-{infer_step}-warped_image.png",
                    )
                    save_image_to_manifold(
                        flow_map_pil[1],
                        save_root,
                        f"{file_name}-{self.layer_name}-{attn_type}-{infer_step}-flow_map.png",
                    )
                    # """

                # 6. Calculate the attn loss
                pred = warped_image.float() * mask.float()
                target = image.float() * mask.float()
                if attn_loss_type == "l1":
                    attn_loss = F.l1_loss(pred, target, reduction="none")
                elif attn_loss_type in ["mse", "l2"]:
                    attn_loss = F.mse_loss(pred, target, reduction="none")
                elif attn_loss_type in ["smooth_l1", "huber"]:
                    attn_loss = F.smooth_l1_loss(pred, target, reduction="none")
                else:
                    raise ValueError(f"Not known attn loss type: {attn_loss_type}")
                attn_loss = attn_loss.mean(dim=list(range(1, len(attn_loss.shape))))
                return attn_loss

            def calcu_flow_loss(this_attn_map, coord_map, mask):
                # 7. Calculate the flow loss
                if flow_loss_type == "v1":
                    flow_map = torch.matmul(this_attn_map, coord_map)
                    flow_map = flow_map.reshape((-1, height, width, 2))
                    flow_map = normalize(flow_map)
                    flow_map = flow_map * 2.0 - 1.0

                    flow_loss = flow_map.sum(dim=list(range(1, len(flow_map.shape))))
                    flow_loss = flow_loss / (
                        mask.sum(dim=list(range(1, len(mask.shape)))) + 1e-9
                    )
                elif flow_loss_type == "v2":
                    flow_map = torch.matmul(this_attn_map, coord_map)
                    flow_map = flow_map.reshape((-1, height, width, 2))
                    mask = F.interpolate(
                        mask, size=(height, width), mode="nearest"
                    ).permute((0, 2, 3, 1))

                    mask_y = mask[:, 1:] * mask[:, :-1]
                    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
                    flow_map_y = torch.abs(flow_map[:, 1:] - flow_map[:, :-1]) * mask_y
                    flow_map_x = (
                        torch.abs(flow_map[:, :, 1:] - flow_map[:, :, :-1]) * mask_x
                    )
                    flow_loss_y = flow_map_y.sum(
                        dim=list(range(1, len(flow_map_y.shape)))
                    ) / (mask_y.sum(dim=list(range(1, len(mask_y.shape)))) + 1e-9)
                    flow_loss_x = flow_map_x.sum(
                        dim=list(range(1, len(flow_map_x.shape)))
                    ) / (mask_x.sum(dim=list(range(1, len(mask_x.shape)))) + 1e-9)
                    flow_loss = flow_loss_y + flow_loss_x
                else:
                    raise ValueError(f"Unknown flow loss type: {flow_loss_type}")

                return flow_loss

            if use_attn_loss_var.get():
                # Build coordinate map.
                ys = torch.linspace(-1, 1, height)
                xs = torch.linspace(-1, 1, width)
                grid_y, grid_x = torch.meshgrid(ys, xs)
                coord_map_attn = torch.stack([grid_x, grid_y], dim=-1)
                coord_map_attn = coord_map_attn.reshape((-1, 2)).to(
                    query.device, dtype=query.dtype
                )
                if attn_type in ["both", "g_to_p"]:
                    attn_loss_g_to_p = calcu_attn_loss(
                        attn_map_g_to_p,
                        coord_map_attn,
                        image,
                        condition_image,
                        mask,
                        attn_type="g_to_p",
                    )
                if attn_type in ["both", "p_to_g"]:
                    attn_loss_p_to_g = calcu_attn_loss(
                        attn_map_p_to_g,
                        coord_map_attn,
                        image,
                        condition_image,
                        mask,
                        attn_type="p_to_g",
                    )
                if attn_type == "both":
                    attn_loss = (attn_loss_g_to_p + attn_loss_p_to_g) / 2.0
                else:
                    attn_loss = (
                        attn_loss_g_to_p if attn_type == "g_to_p" else attn_loss_p_to_g
                    )
                attn_loss_list = attn_loss_var.get()
                attn_loss_list.append(attn_loss)
                attn_loss_var.set(attn_loss_list)

            if use_flow_loss_var.get():
                if flow_loss_type == "v1":
                    coord_map_flow = coord_map_attn
                elif flow_loss_type == "v2":
                    # Build coordinate map.
                    ys = torch.linspace(0, height - 1, height)
                    xs = torch.linspace(0, width - 1, width)
                    grid_y, grid_x = torch.meshgrid(ys, xs)
                    coord_map_flow = torch.stack([grid_x, grid_y], dim=-1)
                    coord_map_flow = coord_map_flow.reshape((-1, 2)).to(
                        query.device, dtype=query.dtype
                    )
                else:
                    raise ValueError(f"Unknown flow loss type: {flow_loss_type}")
                if attn_type in ["both", "g_to_p"]:
                    flow_loss_g_to_p = calcu_flow_loss(
                        attn_map_g_to_p, coord_map_flow, mask
                    )
                if attn_type in ["both", "p_to_g"]:
                    flow_loss_p_to_g = calcu_flow_loss(
                        attn_map_p_to_g, coord_map_flow, mask
                    )

                if attn_type == "both":
                    flow_loss = (flow_loss_g_to_p + flow_loss_p_to_g) / 2.0
                else:
                    flow_loss = (
                        flow_loss_g_to_p if attn_type == "g_to_p" else flow_loss_p_to_g
                    )
                flow_loss_list = flow_loss_var.get()
                flow_loss_list.append(flow_loss)
                flow_loss_var.set(flow_loss_list)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def add_skip_cross_attention(
    unet,
    cross_attn_cls=SkipAttnProcessor,
    self_attn_cls=None,
    cross_attn_dim=None,
    **kwargs,
):
    if cross_attn_dim is None:
        cross_attn_dim = unet.config.cross_attention_dim
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else cross_attn_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if self_attn_cls is not None:
                attn_procs[name] = self_attn_cls(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    **kwargs,
                )
            else:
                # retain the original attn processor
                attn_procs[name] = AttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    layer_name=name,
                    **kwargs,
                )
        else:
            attn_procs[name] = cross_attn_cls(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                **kwargs,
            )

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules


def get_trainable_module(unet, trainable_module_name):
    if trainable_module_name == "unet":
        return unet
    elif trainable_module_name == "transformer":
        trainable_modules = torch.nn.ModuleList()
        for blocks in [unet.down_blocks, unet.mid_block, unet.up_blocks]:
            if hasattr(blocks, "attentions"):
                trainable_modules.append(blocks.attentions)
            else:
                for block in blocks:
                    if hasattr(block, "attentions"):
                        trainable_modules.append(block.attentions)
        return trainable_modules
    elif trainable_module_name == "attention":
        attn_blocks = torch.nn.ModuleList()
        for name, param in unet.named_modules():
            if "attn1" in name:
                attn_blocks.append(param)
        return attn_blocks
    else:
        raise ValueError(f"Unknown trainable_module_name: {trainable_module_name}")


def auto_attn_ckpt_load(trainable_module, attn_ckpt, version):
    sub_folder = {
        "mix": "mix-48k-1024",
        "vitonhd": "vitonhd-16k-512",
        "dresscode": "dresscode-16k-512",
    }[version]
    attn_ckpt = os.path.join(attn_ckpt, sub_folder, "attention/model.safetensors")
    local_attn_ckpt = _PATH_MANAGER.get_local_path(attn_ckpt)
    load_model(trainable_module, local_attn_ckpt)
    logger.info("Load pretrained model from {}".format(attn_ckpt))


def compute_dream_and_update_latents_for_inpaint(
    unet,
    noise_scheduler,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    target: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    dream_detail_preservation: float = 1.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Implements "DREAM (Diffusion Rectification and Estimation-Adaptive Models)" from http://arxiv.org/abs/2312.00210.
    DREAM helps align training with sampling to help training be more efficient and accurate at the cost of an extra
    forward step without gradients.

    Args:
        `unet`: The state unet to use to make a prediction.
        `noise_scheduler`: The noise scheduler used to add noise for the given timestep.
        `timesteps`: The timesteps for the noise_scheduler to user.
        `noise`: A tensor of noise in the shape of noisy_latents.
        `noisy_latents`: Previously noise latents from the training loop.
        `target`: The ground-truth tensor to predict after eps is removed.
        `encoder_hidden_states`: Text embeddings from the text model.
        `dream_detail_preservation`: A float value that indicates detail preservation level.
          See reference.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Adjusted noisy_latents and target.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[
        timesteps, None, None, None
    ]
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # The paper uses lambda = sqrt(1 - alpha) ** p, with p = 1 in their experiments.
    dream_lambda = sqrt_one_minus_alphas_cumprod**dream_detail_preservation

    pred = None  # b, 4, h, w
    unet.eval()
    with torch.no_grad():
        pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    unet.train()

    noisy_latents_no_condition = noisy_latents[:, :4]
    _noisy_latents, _target = (None, None)
    if noise_scheduler.config.prediction_type == "epsilon":
        predicted_noise = pred
        delta_noise = (noise - predicted_noise).detach()
        delta_noise.mul_(dream_lambda)
        _noisy_latents = noisy_latents_no_condition.add(
            sqrt_one_minus_alphas_cumprod * delta_noise
        )
        _target = target.add(delta_noise)
    elif noise_scheduler.config.prediction_type == "v_prediction":
        raise NotImplementedError("DREAM has not been implemented for v-prediction")
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
        )

    _noisy_latents = torch.cat([_noisy_latents, noisy_latents[:, 4:]], dim=1)
    return _noisy_latents, _target


def get_color_wheel(device: torch.device) -> torch.Tensor:
    """
    Generates the color wheel.
    :param device: (torch.device) Device to be used
    :return: (torch.Tensor) Color wheel tensor of the shape [55, 3]
    """
    # Set constants
    RY: int = 15
    YG: int = 6
    GC: int = 4
    CB: int = 11
    BM: int = 13
    MR: int = 6
    # Init color wheel
    color_wheel: torch.Tensor = torch.zeros(
        (RY + YG + GC + CB + BM + MR, 3), dtype=torch.float32
    )
    # Init counter
    counter: int = 0
    # RY
    color_wheel[0:RY, 0] = 255
    color_wheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    counter: int = counter + RY
    # YG
    color_wheel[counter : counter + YG, 0] = 255 - torch.floor(
        255 * torch.arange(0, YG) / YG
    )
    color_wheel[counter : counter + YG, 1] = 255
    counter: int = counter + YG
    # GC
    color_wheel[counter : counter + GC, 1] = 255
    color_wheel[counter : counter + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
    counter: int = counter + GC
    # CB
    color_wheel[counter : counter + CB, 1] = 255 - torch.floor(
        255 * torch.arange(CB) / CB
    )
    color_wheel[counter : counter + CB, 2] = 255
    counter: int = counter + CB
    # BM
    color_wheel[counter : counter + BM, 2] = 255
    color_wheel[counter : counter + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
    counter: int = counter + BM
    # MR
    color_wheel[counter : counter + MR, 2] = 255 - torch.floor(
        255 * torch.arange(MR) / MR
    )
    color_wheel[counter : counter + MR, 0] = 255
    # To device
    color_wheel: torch.Tensor = color_wheel.to(device)
    return color_wheel


def _flow_hw_to_color(
    flow_vertical: torch.Tensor,
    flow_horizontal: torch.Tensor,
    color_wheel: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Private function applies the flow color wheel to flow components (vertical and horizontal).
    :param flow_vertical: (torch.Tensor) Vertical flow of the shape [height, width]
    :param flow_horizontal: (torch.Tensor) Horizontal flow of the shape [height, width]
    :param color_wheel: (torch.Tensor) Color wheel tensor of the shape [55, 3]
    :param: device: (torch.device) Device to be used
    :return: (torch.Tensor) Visualized flow of the shape [3, height, width]
    """
    # Get shapes
    _, height, width = flow_vertical.shape
    # Init flow image
    flow_image: torch.Tensor = torch.zeros(
        3, height, width, dtype=torch.float32, device=device
    )
    # Get number of colors
    number_of_colors: int = color_wheel.shape[0]
    # Compute norm, angle and factors
    flow_norm: torch.Tensor = (
        torch.pow(flow_vertical, 2) + torch.pow(flow_horizontal, 2)
    ).sqrt()
    angle: torch.Tensor = torch.atan2(-flow_vertical, -flow_horizontal) / PI
    fk: torch.Tensor = (angle + 1.0) / 2.0 * (number_of_colors - 1.0)
    k0: torch.Tensor = torch.floor(fk).long()
    k1: torch.Tensor = k0 + 1
    k1[k1 == number_of_colors] = 0
    f: torch.Tensor = fk - k0
    # Iterate over color components
    for index in range(color_wheel.shape[1]):
        # Get component of all colors
        tmp: torch.Tensor = color_wheel[:, index]
        # Get colors
        color_0: torch.Tensor = tmp[k0] / 255.0
        color_1: torch.Tensor = tmp[k1] / 255.0
        # Compute color
        color: torch.Tensor = (1.0 - f) * color_0 + f * color_1
        # Get color index
        color_index: torch.Tensor = flow_norm <= 1
        # Set color saturation
        color[color_index] = 1 - flow_norm[color_index] * (1.0 - color[color_index])
        color[~color_index] = color[~color_index] * 0.75
        # Set color in image
        flow_image[index] = torch.floor(255 * color)
    return flow_image


def flow_to_color(
    flow: torch.Tensor,
    clip_flow: Optional[Union[float, torch.Tensor]] = None,
    normalize_over_video: bool = False,
) -> torch.Tensor:
    """
    Function converts a given optical flow map into the classical color schema.
    :param flow: (torch.Tensor) Optical flow tensor of the shape [batch size (optional), 2, height, width].
    :param clip_flow: (Optional[Union[float, torch.Tensor]]) Max value of flow values for clipping (default None).
    :param normalize_over_video: (bool) If true scale is normalized over the whole video (batch).
    :return: (torch.Tensor) Flow visualization (float tensor) with the shape [batch size (if used), 3, height, width].
    """
    # Check parameter types
    assert torch.is_tensor(
        flow
    ), "Given flow map must be a torch.Tensor, {} given".format(type(flow))
    assert (
        torch.is_tensor(clip_flow) or isinstance(clip_flow, float) or clip_flow is None
    ), "Given clip_flow parameter must be a float, a torch.Tensor, or None, {} given".format(
        type(clip_flow)
    )
    # Check shapes
    assert flow.ndimension() in [
        3,
        4,
    ], "Given flow must be a 3D or 4D tensor, given tensor shape {}.".format(flow.shape)
    if torch.is_tensor(clip_flow):
        assert (
            clip_flow.ndimension() == 0
        ), "Given clip_flow tensor must be a scalar, given tensor shape {}.".format(
            clip_flow.shape
        )
    # Manage batch dimension
    batch_dimension: bool = True
    if flow.ndimension() == 3:
        flow = flow[None]
        batch_dimension: bool = False
    if flow.shape[-1] == 2:
        flow = flow.permute(0, 3, 1, 2)
    # Save shape
    batch_size, _, height, width = flow.shape
    # Check flow dimension
    assert (
        flow.shape[1] == 2
    ), "Flow dimension must have the shape 2 but tensor with {} given".format(
        flow.shape[1]
    )
    # Save device
    device: torch.device = flow.device
    # Clip flow if utilized
    if clip_flow is not None:
        flow = flow.clamp(max=clip_flow)
    # Get horizontal and vertical flow
    flow_vertical: torch.Tensor = flow[:, 0:1]
    flow_horizontal: torch.Tensor = flow[:, 1:2]
    # Get max norm of flow
    flow_max_norm: torch.Tensor = (
        (torch.pow(flow_vertical, 2) + torch.pow(flow_horizontal, 2))
        .sqrt()
        .view(batch_size, -1)
        .max(dim=-1)[0]
    )
    flow_max_norm: torch.Tensor = flow_max_norm.view(batch_size, 1, 1, 1)
    if normalize_over_video:
        flow_max_norm: Tensor = flow_max_norm.max(dim=0, keepdim=True)[0]
    # Normalize flow
    flow_vertical: torch.Tensor = flow_vertical / (flow_max_norm + 1e-05)
    flow_horizontal: torch.Tensor = flow_horizontal / (flow_max_norm + 1e-05)
    # Get color wheel
    color_wheel: torch.Tensor = get_color_wheel(device=device)
    # Init flow image
    flow_image = torch.zeros(batch_size, 3, height, width, device=device)
    # Iterate over batch dimension
    for index in range(batch_size):
        flow_image[index] = _flow_hw_to_color(
            flow_vertical=flow_vertical[index],
            flow_horizontal=flow_horizontal[index],
            color_wheel=color_wheel,
            device=device,
        )
    return flow_image if batch_dimension else flow_image[0]
