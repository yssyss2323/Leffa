import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0

from leffa.diffusion_model.unet_ref import (
    UNet2DConditionModel as ReferenceUNet,
)
from leffa.diffusion_model.unet_gen import (
    UNet2DConditionModel as GenerativeUNet,
)

logger: logging.Logger = logging.getLogger(__name__)


class LeffaModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "",
        pretrained_vae_name_or_path: str = "",
        pretrained_garmentnet_path: str = "",
        pretrained_model: str = "",
        new_in_channels: int = 12,  # noisy_image: 4, mask: 1, masked_image: 4, densepose: 3
        height: int = 1024,
        width: int = 768,
    ):
        super().__init__()

        self.height = height
        self.width = width

        self.build_models(
            pretrained_model_name_or_path,
            pretrained_vae_name_or_path,
            pretrained_garmentnet_path,
            pretrained_model,
            new_in_channels,
        )

    def build_models(
        self,
        pretrained_model_name_or_path: str = "",
        pretrained_vae_name_or_path: str = "",
        pretrained_garmentnet_path: str = "",
        pretrained_model: str = "",
        new_in_channels: int = 12,
    ):
        diffusion_model_type = ""
        if "stable-diffusion-inpainting" in pretrained_model_name_or_path:
            diffusion_model_type = "sd15"
        elif "stable-diffusion-xl-1.0-inpainting-0.1" in pretrained_model_name_or_path:
            diffusion_model_type = "sdxl"

        # Noise Scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
            rescale_betas_zero_snr=False if diffusion_model_type == "sd15" else True,
        )
        # VAE
        if (
            pretrained_vae_name_or_path != ""
            and pretrained_vae_name_or_path is not None
        ):
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_vae_name_or_path,
            )
        else:
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="vae",
            )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # Reference UNet
        self.unet_encoder = ReferenceUNet.from_pretrained(
            pretrained_garmentnet_path,
            subfolder="unet",
        )
        self.unet_encoder.config.addition_embed_type = None
        # Generative UNet
        self.unet = GenerativeUNet.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            low_cpu_mem_usage=False,
            device_map=None,
        )
        self.unet.config.addition_embed_type = None
        # Change Generative UNet conv_in and conv_out
        unet_conv_in_channel_changed = self.unet.config.in_channels != new_in_channels
        if unet_conv_in_channel_changed:
            self.unet.conv_in = self.replace_conv_in_layer(self.unet, new_in_channels)
            self.unet.config.in_channels = new_in_channels
        unet_conv_out_channel_changed = (
            self.unet.config.out_channels != self.vae.config.latent_channels
        )
        if unet_conv_out_channel_changed:
            self.unet.conv_out = self.replace_conv_out_layer(
                self.unet, self.vae.config.latent_channels
            )
            self.unet.config.out_channels = self.vae.config.latent_channels

        unet_encoder_conv_in_channel_changed = (
            self.unet_encoder.config.in_channels != self.vae.config.latent_channels
        )
        if unet_encoder_conv_in_channel_changed:
            self.unet_encoder.conv_in = self.replace_conv_in_layer(
                self.unet_encoder, self.vae.config.latent_channels
            )
            self.unet_encoder.config.in_channels = self.vae.config.latent_channels
        unet_encoder_conv_out_channel_changed = (
            self.unet_encoder.config.out_channels != self.vae.config.latent_channels
        )
        if unet_encoder_conv_out_channel_changed:
            self.unet_encoder.conv_out = self.replace_conv_out_layer(
                self.unet_encoder, self.vae.config.latent_channels
            )
            self.unet_encoder.config.out_channels = self.vae.config.latent_channels

        # Remove Cross Attention
        remove_cross_attention(self.unet)
        remove_cross_attention(self.unet_encoder, model_type="unet_encoder")

        # Load pretrained model
        if pretrained_model != "" and pretrained_model is not None:
            self.load_state_dict(torch.load(pretrained_model, map_location="cpu"))
            logger.info("Load pretrained model from {}".format(pretrained_model))

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

    def vae_encode(self, pixel_values):
        pixel_values = pixel_values.to(device=self.vae.device, dtype=self.vae.dtype)
        with torch.no_grad():
            latent = self.vae.encode(pixel_values).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        return latent


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


def remove_cross_attention(
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