import logging
import os
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler

from diffusers.training_utils import compute_snr

from leffa.models.cat_vton_model import add_skip_cross_attention
from leffa.models.diffusion_model.unet_hacked_garment import (
    UNet2DConditionModel as UNet2DConditionModel_ref,
)
from leffa.models.diffusion_model.unet_hacked_tryon import (
    UNet2DConditionModel as UNet2DConditionModel_tryon,
)
from torchmultimodal import _PATH_MANAGER

logger: logging.Logger = logging.getLogger(__name__)


class SimpleVtonModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "",
        pretrained_vae_name_or_path: str = "",
        pretrained_garmentnet_path: str = "",
        pretrained_model: str = "",
        new_in_channels: int = 12,  # noisy_image: 4, mask: 1, masked_image: 4, densepose: 3
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
        use_learning_flow_in_attention: bool = False,
        learning_flow_in_attention_lambda: float = 0.1,
        learning_flow_in_attention_stop_timestep: int = -1,
        use_attention_flow_loss: bool = False,
        attention_flow_loss_lambda: float = 0.1,
        use_pixel_space_supervision: bool = False,
        pixel_space_supervision_lambda: float = 10.0,
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
        self.use_learning_flow_in_attention = use_learning_flow_in_attention
        self.learning_flow_in_attention_lambda = learning_flow_in_attention_lambda
        self.learning_flow_in_attention_stop_timestep = (
            learning_flow_in_attention_stop_timestep
        )
        self.use_attention_flow_loss = use_attention_flow_loss
        self.attention_flow_loss_lambda = attention_flow_loss_lambda
        self.use_pixel_space_supervision = use_pixel_space_supervision
        self.pixel_space_supervision_lambda = pixel_space_supervision_lambda

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
            else:
                raise ValueError(
                    f"pretrained_model_name_or_path: {pretrained_model_name_or_path} is not supported. Please provide a valid manifold path."
                )
            pretrained_model_name_or_path = self.get_local_path(
                pretrained_model_name_or_path, files_to_cache
            )

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
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
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

        if self.skip_cross_attention:
            add_skip_cross_attention(self.unet)
        if self.skip_cross_attention_garmentnet:
            add_skip_cross_attention(self.unet_encoder, model_type="unet_encoder")

        # Load pretrained model
        if pretrained_model != "" and pretrained_model is not None:
            local_pretrained_model = _PATH_MANAGER.get_local_path(pretrained_model)
            self.load_state_dict(torch.load(local_pretrained_model, map_location="cpu"))
            logger.info("Load pretrained model from {}".format(pretrained_model))

        self.vae.requires_grad_(False)
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

    def vae_encode(self, pixel_values):
        pixel_values = pixel_values.to(device=self.vae.device, dtype=self.vae.dtype)
        with torch.no_grad():
            latent = self.vae.encode(pixel_values).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        return latent

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
        densepose = F.interpolate(
            batch["densepose"],
            size=(model_input.shape[-2], model_input.shape[-1]),
            mode="nearest",
        )

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

        encoder_hidden_states = None
        unet_added_cond_kwargs = None

        cloth_values = self.vae_encode(batch["cloth_pure"])
        cloth_values = (
            torch.zeros_like(cloth_values)
            if random.random() < self.garment_dropout_ratio
            else cloth_values
        )

        down, reference_features = self.unet_encoder(
            cloth_values,
            timesteps,
            encoder_hidden_states,
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
                outputs["noisy_latents"] = noisy_latents
                pixel_space_loss = self.loss_pixel_space(outputs)
                loss += pixel_space_loss * self.pixel_space_supervision_lambda
                outputs["pixel_space_loss"] = pixel_space_loss
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
        encoder_hidden_states = inputs["encoder_hidden_states"]  # None
        unet_added_cond_kwargs = inputs["unet_added_cond_kwargs"]  # None
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

    def loss_pixel_space(self, inputs):
        timesteps = inputs["timesteps"]
        noisy_latents = inputs["noisy_latents"]
        model_pred = inputs["model_pred"]
        model_input = inputs["model_input"]

        # gt
        noisy_latents = 1 / self.vae.config.scaling_factor * noisy_latents
        noisy_image = self.vae.decode(noisy_latents).sample

        # pred
        noisy_latents_pred = self.noise_scheduler.add_noise(
            model_input, model_pred, timesteps
        )
        noisy_latents_pred = 1 / self.vae.config.scaling_factor * noisy_latents_pred
        noisy_image_pred = self.vae.decode(noisy_latents_pred).sample

        loss = F.mse_loss(noisy_image, noisy_image_pred, reduction="mean")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        loss = loss.mean()
        return loss


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
