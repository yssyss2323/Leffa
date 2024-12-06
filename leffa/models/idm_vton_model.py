import logging
import os
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor

from diffusers.training_utils import compute_snr
from leffa.models.cat_vton_model import add_skip_cross_attention
from leffa.models.diffusion_model.unet_hacked_garment import (
    UNet2DConditionModel as UNet2DConditionModel_ref,
)
from leffa.models.diffusion_model.unet_hacked_tryon import (
    UNet2DConditionModel as UNet2DConditionModel_tryon,
)
from leffa.models.ip_adapter.ip_adapter import Resampler
from torchmultimodal import _PATH_MANAGER
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

logger: logging.Logger = logging.getLogger(__name__)


class IdmVtonModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "",
        pretrained_vae_name_or_path: str = "",
        pretrained_garmentnet_path: str = "",
        pretrained_image_encoder_path: str = "",
        pretrained_ip_adapter_path: str = "",
        pretrained_model: str = "",
        new_in_channels: int = 13,  # noisy_image: 4, mask: 1, masked_image: 4, densepose: 4
        height: int = 1024,
        width: int = 768,
        snr_gamma: Optional[float] = None,
        garment_dropout_ratio: float = 0.0,
        use_dream: bool = False,
        dream_detail_preservation: float = 10.0,
        skip_cross_attention: bool = False,
        skip_cross_attention_garmentnet: bool = False,
        copy_unet_to_unet_encoder: bool = False,
        only_optimize_unet_attn1: bool = False,
        optimize_unet: bool = True,
        optimize_unet_encoder: bool = False,
        # <DOING
        use_learning_flow_in_attention: bool = False,
        learning_flow_in_attention_lambda: float = 0.1,
        learning_flow_in_attention_stop_timestep: int = -1,
        use_attention_flow_loss: bool = False,
        attention_flow_loss_lambda: float = 0.1,
        use_pixel_space_supervision: bool = False,
        pixel_space_supervision_lambda: float = 10.0,
        # DOING>
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.snr_gamma = snr_gamma
        self.garment_dropout_ratio = garment_dropout_ratio
        self.use_dream = use_dream
        self.dream_detail_preservation = dream_detail_preservation
        self.skip_cross_attention = skip_cross_attention
        self.skip_cross_attention_garmentnet = skip_cross_attention_garmentnet
        self.copy_unet_to_unet_encoder = copy_unet_to_unet_encoder
        self.only_optimize_unet_attn1 = only_optimize_unet_attn1
        self.optimize_unet = optimize_unet
        self.optimize_unet_encoder = optimize_unet_encoder
        # <DOING
        self.use_learning_flow_in_attention = use_learning_flow_in_attention
        self.learning_flow_in_attention_lambda = learning_flow_in_attention_lambda
        self.learning_flow_in_attention_stop_timestep = (
            learning_flow_in_attention_stop_timestep
        )
        self.use_attention_flow_loss = use_attention_flow_loss
        self.attention_flow_loss_lambda = attention_flow_loss_lambda
        self.use_pixel_space_supervision = use_pixel_space_supervision
        self.pixel_space_supervision_lambda = pixel_space_supervision_lambda
        # DOING>

        self.build_models(
            pretrained_model_name_or_path,
            pretrained_vae_name_or_path,
            pretrained_garmentnet_path,
            pretrained_image_encoder_path,
            pretrained_ip_adapter_path,
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
        pretrained_garmentnet_path: str = "",
        pretrained_image_encoder_path: str = "",
        pretrained_ip_adapter_path: str = "",
        pretrained_model: str = "",
        new_in_channels: int = 13,
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
                    "tokenizer/merges.txt",
                    "tokenizer/special_tokens_map.json",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/vocab.json",
                    "text_encoder/config.json",
                    "text_encoder/model.safetensors",
                    "tokenizer_2/merges.txt",
                    "tokenizer_2/special_tokens_map.json",
                    "tokenizer_2/tokenizer_config.json",
                    "tokenizer_2/vocab.json",
                    "text_encoder_2/config.json",
                    "text_encoder_2/model.safetensors",
                    "unet/config.json",
                    "unet/diffusion_pytorch_model.safetensors",
                ]
            else:
                raise ValueError(
                    f"pretrained_model_name_or_path: {pretrained_model_name_or_path} is not supported. Please provide a valid manifold path."
                )
            pretrained_model_name_or_path = self.get_local_path(
                pretrained_model_name_or_path, files_to_cache
            )
        self.diffusion_model_type = diffusion_model_type

        if pretrained_garmentnet_path.startswith("manifold://"):
            files_to_cache = [
                "unet/config.json",
            ]
            if diffusion_model_type == "sd15":
                files_to_cache.append("unet/diffusion_pytorch_model.bin")
            elif diffusion_model_type == "sdxl":
                files_to_cache.append("unet/diffusion_pytorch_model.safetensors")
            pretrained_garmentnet_path = self.get_local_path(
                pretrained_garmentnet_path, files_to_cache
            )
        if pretrained_image_encoder_path.startswith("manifold://"):
            files_to_cache = [
                "config.json",
                "model.safetensors",
                "pytorch_model.bin",
            ]
            pretrained_image_encoder_path = self.get_local_path(
                pretrained_image_encoder_path, files_to_cache
            )
        if pretrained_ip_adapter_path.startswith("manifold://"):
            pretrained_ip_adapter_path = _PATH_MANAGER.get_local_path(
                pretrained_ip_adapter_path
            )

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
        if diffusion_model_type == "sdxl":
            # Condition
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer",
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="text_encoder",
            )
            self.tokenizer_2 = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer_2",
            )
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="text_encoder_2",
            )
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                pretrained_image_encoder_path
            )
        # GarmentNet
        self.unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            pretrained_garmentnet_path,
            subfolder="unet",
        )
        self.unet_encoder.config.addition_embed_type = None
        # TryonNet
        self.unet = UNet2DConditionModel_tryon.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            low_cpu_mem_usage=False,
            device_map=None,
        )
        self.unet.config.addition_embed_type = None
        if diffusion_model_type == "sdxl":
            self.unet.config.encoder_hid_dim = self.image_encoder.config.hidden_size
            self.unet.config.encoder_hid_dim_type = "ip_image_proj"
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

        if diffusion_model_type == "sdxl":
            # Ip-Adapter state dict
            ip_state_dict = torch.load(pretrained_ip_adapter_path, map_location="cpu")
            adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())
            adapter_modules.load_state_dict(ip_state_dict["ip_adapter"], strict=True)

            image_proj_model = Resampler(
                dim=self.image_encoder.config.hidden_size,
                depth=4,
                dim_head=64,
                heads=20,
                num_queries=16,
                embedding_dim=self.image_encoder.config.hidden_size,
                output_dim=self.unet.config.cross_attention_dim,
                ff_mult=4,
            )
            image_proj_model.load_state_dict(ip_state_dict["image_proj"], strict=True)
            image_proj_model.requires_grad_(True)
            self.unet.encoder_hid_proj = image_proj_model

        # # Load the official pretrained unet model.
        # unet_model_path = "manifold://genads_models/tree/zijianzhou/model/IDM-VTON/unet/diffusion_pytorch_model.bin"
        # local_unet_model_path = _PATH_MANAGER.get_local_path(unet_model_path)
        # self.unet.load_state_dict(
        #     torch.load(local_unet_model_path, map_location="cpu"), strict=True
        # )
        # logger.info("Load unet model from {}".format(unet_model_path))

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor  # noqa
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.feature_extractor = CLIPImageProcessor()

        if self.skip_cross_attention:
            add_skip_cross_attention(self.unet)
        if self.skip_cross_attention_garmentnet:
            add_skip_cross_attention(self.unet_encoder)

        # Load pretrained model
        if pretrained_model != "" and pretrained_model is not None:
            local_pretrained_model = _PATH_MANAGER.get_local_path(pretrained_model)
            self.load_state_dict(torch.load(local_pretrained_model, map_location="cpu"))
            logger.info("Load pretrained model from {}".format(pretrained_model))

        self.vae.requires_grad_(False)
        if diffusion_model_type == "sdxl":
            self.text_encoder.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
            self.image_encoder.requires_grad_(False)
        if self.optimize_unet_encoder:
            if self.only_optimize_unet_attn1:
                self.unet_encoder.requires_grad_(False)
                for n, p in self.unet_encoder.named_parameters():
                    if "attn1" in n:
                        p.requires_grad_(True)
            else:
                self.unet_encoder.requires_grad_(True)
        else:
            self.unet_encoder.requires_grad_(False)
        if self.optimize_unet:
            if self.only_optimize_unet_attn1:
                self.unet.requires_grad_(False)
                for n, p in self.unet.named_parameters():
                    if "attn1" in n:
                        p.requires_grad_(True)
            else:
                self.unet.requires_grad_(True)
        else:
            self.unet.requires_grad_(False)
        if unet_conv_in_channel_changed:
            self.unet.conv_in.requires_grad_(True)
        if unet_conv_out_channel_changed:
            self.unet.conv_out.requires_grad_(True)
        if unet_encoder_conv_in_channel_changed:
            self.unet_encoder.conv_in.requires_grad_(True)
        if unet_encoder_conv_out_channel_changed:
            self.unet_encoder.conv_out.requires_grad_(True)

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

    def vae_encode(self, pixel_values):
        pixel_values = pixel_values.to(device=self.vae.device, dtype=self.vae.dtype)
        with torch.no_grad():
            latent = self.vae.encode(pixel_values).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        return latent

    def text_encode(self, text):
        with torch.no_grad():
            text_input_ids = self.tokenizer(
                text,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids
            text_input_ids_2 = self.tokenizer_2(
                text,
                max_length=self.tokenizer_2.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids

            encoder_output = self.text_encoder(
                text_input_ids.to(self.text_encoder.device), output_hidden_states=True
            )
            text_embeds = encoder_output.hidden_states[-2]
            encoder_output_2 = self.text_encoder_2(
                text_input_ids_2.to(self.text_encoder_2.device),
                output_hidden_states=True,
            )
            pooled_text_embeds = encoder_output_2[0]
            text_embeds_2 = encoder_output_2.hidden_states[-2]
            # concat
            encoder_hidden_states = torch.concat([text_embeds, text_embeds_2], dim=-1)
        return encoder_hidden_states, pooled_text_embeds

    @torch.no_grad()
    def do_copy_unet_to_unet_encoder(self):
        assert (
            self.skip_cross_attention == self.skip_cross_attention_garmentnet
        ), "TryonNet and GarmentNet should have the same skip_cross_attention setting."
        parameters = {}
        for n, p in self.unet.named_parameters():
            parameters[n] = p
        for r_n, r_p in self.unet_encoder.named_parameters():
            if "conv_in.weight" in r_n:
                r_p.data.copy_(
                    parameters["conv_in.weight"].data[:, : r_p.shape[1], ...]
                )
            else:
                r_p.data.copy_(parameters[r_n].data)

    def forward(self, batch: Dict[str, Any]):
        if self.copy_unet_to_unet_encoder:
            self.do_copy_unet_to_unet_encoder()

        # 1. VAE encoding
        model_input = self.vae_encode(batch["image"])
        masked_image = batch["image"] * (batch["inpaint_mask"] < 0.5)
        masked_latents = self.vae_encode(masked_image)
        mask = batch["inpaint_mask"]
        mask = torch.stack([F.interpolate(mask, size=model_input.shape[-2:])])
        mask = mask.reshape(-1, 1, model_input.shape[-2], model_input.shape[-1])
        densepose = self.vae_encode(batch["densepose"])

        # 2. prepare noise
        noise = torch.randn_like(model_input)
        batch_size = model_input.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=model_input.device,
        )
        # add noise to the latents
        noisy_latents = self.noise_scheduler.add_noise(model_input, noise, timesteps)

        # 3. prepare model input
        latent_model_input = torch.cat(
            [noisy_latents, mask, masked_latents, densepose], dim=1
        )

        if self.diffusion_model_type == "sdxl":
            encoder_hidden_states, pooled_text_embeds = self.text_encode(
                batch["caption"]
            )

            crops_coords_top_left = (0, 0)
            dtype = pooled_text_embeds.dtype
            add_time_ids = torch.cat(
                [
                    self.compute_time_ids(
                        (self.height, self.height), crops_coords_top_left, dtype
                    )
                    for i in range(batch_size)
                ]
            )
            add_time_ids = add_time_ids.to(device=pooled_text_embeds.device)

            image_embeds = batch["cloth4clip"].to(
                device=self.unet.device, dtype=self.unet.dtype
            )
            image_embeds = self.image_encoder(
                image_embeds, output_hidden_states=True
            ).hidden_states[-2]
            ip_tokens = self.unet.encoder_hid_proj(image_embeds)

            # add cond
            unet_added_cond_kwargs = {}
            unet_added_cond_kwargs["text_embeds"] = pooled_text_embeds
            unet_added_cond_kwargs["time_ids"] = add_time_ids
            unet_added_cond_kwargs["image_embeds"] = ip_tokens
        else:
            encoder_hidden_states = None
            unet_added_cond_kwargs = None

        cloth_values = self.vae_encode(batch["cloth_pure"])
        cloth_values = (
            torch.zeros_like(cloth_values)
            if random.random() < self.garment_dropout_ratio
            else cloth_values
        )

        if self.diffusion_model_type == "sdxl":
            text_embeds_cloth, _ = self.text_encode(batch["caption_cloth"])
        else:
            text_embeds_cloth = None

        down, reference_features = self.unet_encoder(
            cloth_values,
            timesteps,
            encoder_hidden_states=text_embeds_cloth,
            return_dict=False,
        )
        reference_features = list(reference_features)

        outputs = {}
        outputs["timesteps"] = timesteps
        outputs["noise"] = noise
        outputs["model_input"] = model_input
        outputs["latent_model_input"] = latent_model_input
        outputs["encoder_hidden_states"] = encoder_hidden_states
        outputs["unet_added_cond_kwargs"] = unet_added_cond_kwargs
        outputs["reference_features"] = reference_features

        if self.training:
            if self.use_dream:
                model_pred, loss = self.loss_with_dream(outputs)
            else:
                model_pred, loss = self.loss(outputs)
            outputs["model_pred"] = model_pred
            outputs["loss"] = loss
        else:
            model_pred = self.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states,  # used for attn2 cross attention
                added_cond_kwargs=unet_added_cond_kwargs,  # concat with encoder_hidden_states for attn2 cross attention
                garment_features=reference_features,  # used for attn1 self attention
                return_dict=False,
            )[0]
            outputs["model_pred"] = model_pred
        return outputs

    def loss(self, inputs):
        timesteps = inputs["timesteps"]
        noise = inputs["noise"]
        model_input = inputs["model_input"]
        latent_model_input = inputs["latent_model_input"]
        encoder_hidden_states = inputs["encoder_hidden_states"]
        unet_added_cond_kwargs = inputs["unet_added_cond_kwargs"]
        reference_features = inputs["reference_features"]

        model_pred = self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_cond_kwargs,
            garment_features=reference_features,
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
        timesteps = inputs["timesteps"]
        noise = inputs["noise"]
        model_input = inputs["model_input"]
        latent_model_input = inputs["latent_model_input"]
        encoder_hidden_states = inputs["encoder_hidden_states"]
        unet_added_cond_kwargs = inputs["unet_added_cond_kwargs"]
        reference_features = inputs["reference_features"]

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
            encoder_hidden_states=encoder_hidden_states,
            unet_added_cond_kwargs=unet_added_cond_kwargs,
            reference_features=reference_features,
            dream_detail_preservation=self.dream_detail_preservation,
        )

        model_pred = self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_cond_kwargs,
            garment_features=reference_features,
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


def compute_dream_and_update_latents_for_inpaint(
    unet,
    noise_scheduler,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    target: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    unet_added_cond_kwargs: Dict[str, Any],
    reference_features: List[torch.Tensor],
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
    with torch.no_grad():
        pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_cond_kwargs,
            garment_features=reference_features,
        ).sample

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


from functools import partial

import numpy as np
from timm.models.vision_transformer import Block, PatchEmbed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    # qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    # qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


class Mae4BgGen(MaskedAutoencoderViT):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        pretrained_path=None,
        bg_masking_type="mean",
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            decoder_embed_dim,
            decoder_depth,
            decoder_num_heads,
            mlp_ratio,
            norm_layer,
            norm_pix_loss,
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.bg_masking_type = bg_masking_type

        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

    def _load_pretrained_weights(self, pretrained_path):
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        if pretrained_path.startswith("manifold://"):
            pretrained_path = _PATH_MANAGER.get_local_path(pretrained_path)
        state_dict = torch.load(pretrained_path, map_location="cpu")
        if self.img_size != 224:
            state_dict["model"].pop("pos_embed")
        self.load_state_dict(state_dict["model"], strict=False)
        logger.info(f"Loaded pretrained weights from {pretrained_path}")

    def bg_masking(self, x, bg_mask):
        N, L, D = x.shape

        bg_mask = (bg_mask - bg_mask.min()) / (bg_mask.max() - bg_mask.min())
        bg_mask[bg_mask > 0.5] = 1
        bg_mask[bg_mask <= 0.5] = 0
        bg_mask = F.interpolate(
            bg_mask, size=self.patch_embed.grid_size, mode="nearest"
        )
        h, w = self.patch_embed.grid_size
        bg_mask = bg_mask.reshape((bg_mask.shape[0], h * w))
        mask = (1 - bg_mask).abs()

        ids_shuffle = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        if self.bg_masking_type == "mean":
            len_keep = bg_mask.sum(dim=1).mean().int()
        elif self.bg_masking_type == "median":
            len_keep = bg_mask.sum(dim=1).median().int()
        elif self.bg_masking_type == "max":
            len_keep = bg_mask.sum(dim=1).max().int()
        elif self.bg_masking_type == "min":
            len_keep = bg_mask.sum(dim=1).min().int()
        else:
            raise ValueError(f"Unknown bg_masking_type {self.bg_masking_type}")
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, batch):
        x = batch["image"]  # (batch_size, 3, H, W)
        bg_mask = batch["bg_mask"]  # (batch_size, 1, H, W)
        assert isinstance(bg_mask, torch.Tensor)
        x = x * bg_mask
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.bg_masking(x, bg_mask)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward(self, batch):
        imgs = batch["image"].clone().detach()
        latent, mask, ids_restore = self.forward_encoder(batch)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)

        outputs = {}
        outputs["loss"] = loss
        outputs["pred"] = pred
        outputs["mask"] = mask
        return outputs
