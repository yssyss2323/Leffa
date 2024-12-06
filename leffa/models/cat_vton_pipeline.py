import inspect
import os
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from leffa.models.cat_vton_model import (
    attn_loss_var,
    condition_image_var,
    file_name_var,
    flow_loss_var,
    image_var,
    mask_var,
    use_attn_loss_var,
    use_flow_loss_var,
)
from PIL import Image, ImageFilter


class CatVtonPipeline:
    def __init__(
        self,
        model,
        repaint=True,
        weight_dtype=torch.float32,
        device="cuda",
        use_tf32=True,
    ):
        self.vae = model.vae
        self.unet = model.unet
        self.noise_scheduler = model.noise_scheduler
        self.use_learning_flow_in_attention = model.use_learning_flow_in_attention
        self.use_attention_flow_loss = model.use_attention_flow_loss
        self.repaint = repaint
        self.device = device
        self.weight_dtype = weight_dtype

        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        image,
        condition_image,
        mask,
        garment_mask: torch.Tensor = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        generator=None,
        eta=1.0,
        **kwargs,
    ):
        concat_dim = -2  # y axis

        image = image.to(device=self.vae.device, dtype=self.vae.dtype)
        condition_image = condition_image.to(
            device=self.vae.device, dtype=self.vae.dtype
        )
        mask = mask.to(device=self.vae.device, dtype=self.vae.dtype)
        masked_image = image * (mask < 0.5)
        if garment_mask is not None:
            garment_mask = garment_mask.to(device=self.vae.device, dtype=self.vae.dtype)
        use_densepose = kwargs.get("densepose", None) is not None
        use_cloth_densepose = kwargs.get("cloth_densepose", None) is not None
        if use_densepose:
            densepose = kwargs["densepose"].to(
                device=self.vae.device, dtype=self.vae.dtype
            )
            if use_cloth_densepose:
                cloth_densepose = kwargs["cloth_densepose"].to(
                    device=self.vae.device, dtype=self.vae.dtype
                )

        # Used for learning flow fields in attention
        if self.use_learning_flow_in_attention or self.use_attention_flow_loss:
            image_var.set(image)
            condition_image_var.set(condition_image)
            warped_mask = kwargs["warped_mask"].to(
                device=self.vae.device, dtype=self.vae.dtype
            )
            mask_var.set(warped_mask)
            use_attn_loss_var.set(self.use_learning_flow_in_attention)
            attn_loss_var.set([])
            use_flow_loss_var.set(self.use_attention_flow_loss)
            flow_loss_var.set([])
            file_name_var.set(kwargs.get("file_name_list", []))

        # 1. VAE encoding
        with torch.no_grad():
            masked_latent = self.vae.encode(masked_image).latent_dist.sample()
            condition_latent = self.vae.encode(condition_image).latent_dist.sample()
            if use_densepose:
                if not use_cloth_densepose:
                    densepose_latent = self.vae.encode(densepose).latent_dist.sample()
        masked_latent = masked_latent * self.vae.config.scaling_factor
        condition_latent = condition_latent * self.vae.config.scaling_factor
        if use_densepose:
            if use_cloth_densepose:
                densepose_latent = F.interpolate(
                    densepose, size=masked_latent.shape[-2:], mode="nearest"
                )
                cloth_densepose_latent = F.interpolate(
                    cloth_densepose, size=masked_latent.shape[-2:], mode="nearest"
                )
            else:
                densepose_latent = densepose_latent * self.vae.config.scaling_factor
        mask_latent = F.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
        if garment_mask is not None:
            garment_mask_latent = F.interpolate(
                garment_mask, size=condition_latent.shape[-2:], mode="nearest"
            )

        # 2. concat latents
        masked_latent_concat = torch.cat(
            [masked_latent, condition_latent], dim=concat_dim
        )
        if garment_mask is not None:
            mask_latent_concat = torch.cat(
                [mask_latent, garment_mask_latent], dim=concat_dim
            )
        else:
            mask_latent_concat = torch.cat(
                [mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim
            )
        if use_densepose:
            densepose_latent_concat = torch.cat(
                [
                    densepose_latent,
                    (
                        cloth_densepose_latent
                        if use_cloth_densepose
                        else torch.zeros_like(densepose_latent)
                    ),
                ],
                dim=concat_dim,
            )

        # 3. prepare noise
        noise = torch.randn_like(masked_latent_concat)

        # 4. prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        noise = noise * self.noise_scheduler.init_noise_sigma
        latents = noise

        # 5. Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat(
                        [masked_latent, torch.zeros_like(condition_latent)],
                        dim=concat_dim,
                    ),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)
            if use_densepose:
                densepose_latent_concat = torch.cat([densepose_latent_concat] * 2)

        # 6. Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.noise_scheduler.order
        )
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                os.environ["INFER_STEP"] = str(i + 1)
                # expand the latents if we are doing classifier free guidance
                non_inpainting_latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                non_inpainting_latent_model_input = (
                    self.noise_scheduler.scale_model_input(
                        non_inpainting_latent_model_input, t
                    )
                )
                # prepare the input for the inpainting model
                input_list = [
                    non_inpainting_latent_model_input,
                    mask_latent_concat,
                    masked_latent_concat,
                ]
                if use_densepose:
                    input_list.append(densepose_latent_concat)
                inpainting_latent_model_input = torch.cat(input_list, dim=1)
                # predict the noise residual
                noise_pred = self.unet(
                    inpainting_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=None,  # FIXME
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode the final latents
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        gen_image = self.vae.decode(
            latents.to(self.device, dtype=self.weight_dtype)
        ).sample
        gen_image = (gen_image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        gen_image = gen_image.cpu().permute(0, 2, 3, 1).float().numpy()
        gen_image = numpy_to_pil(gen_image)

        if self.repaint:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = numpy_to_pil(image)
            mask = mask.cpu().permute(0, 2, 3, 1).float().numpy()
            mask = numpy_to_pil(mask)
            mask = [i.convert("RGB") for i in mask]
            gen_image = [
                repaint(_image, _mask, _gen_image)
                for _image, _mask, _gen_image in zip(image, mask, gen_image)
            ]

        return (gen_image,)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 100
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result
