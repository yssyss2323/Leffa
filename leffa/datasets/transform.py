import logging

from typing import Any, Dict

import cv2
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from torch import nn
from transformers import CLIPImageProcessor

logger: logging.Logger = logging.getLogger(__name__)


class VtonTransform(nn.Module):
    def __init__(
        self,
        height: int = 1024,
        width: int = 768,
        is_train: bool = False,
        dataset: str = "viton_hd",
        garment_dropout_ratio: float = 0.0,
        aug_garment_ratio: float = 0.0,
        get_garment_from_person_ratio: float = 0.0,
        aug_mask_ratio: float = 0.0,
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.is_train = is_train
        self.dataset = dataset
        self.garment_dropout_ratio = garment_dropout_ratio
        self.aug_garment_ratio = aug_garment_ratio
        self.get_garment_from_person_ratio = get_garment_from_person_ratio
        self.aug_mask_ratio = aug_mask_ratio

        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.clip_processor = CLIPImageProcessor()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = len(batch["image"])

        if self.dataset in ["pose_transfer", "deepfashion"]:
            # batch is pil image inputs
            mask_data_keys = ["agnostic_mask", "cloth_mask"]

            for key in mask_data_keys:
                batch[key] = [
                    Image.new("RGB", (self.width, self.height),
                              (255, 255, 255))
                    for _ in range(batch_size)
                ]

            batch["image_parse"] = [
                densepose_to_mask(batch["image_densepose"][i])
                for i in range(batch_size)
            ]

        image_list = []
        cloth_list = []
        mask_list = []
        densepose_list = []
        cloth4clip_list = []
        if self.dataset in ["pose_transfer", "deepfashion"]:
            cloth_densepose_list = []
        for i in range(batch_size):
            # 1. get original data
            image = batch["image"][i]
            cloth = batch["cloth"][i]
            mask = batch["agnostic_mask"][i]
            densepose = batch["image_densepose"][i]
            if self.dataset in ["pose_transfer", "deepfashion"]:
                cloth_densepose = batch["cloth_densepose"][i]

            # 3. process data
            image = self.vae_processor.preprocess(
                image, self.height, self.width)[0]
            cloth = self.vae_processor.preprocess(
                cloth, self.height, self.width)[0]
            mask = self.mask_processor.preprocess(
                mask, self.height, self.width)[0]
            if self.dataset in ["pose_transfer", "deepfashion"]:
                densepose = densepose.resize(
                    (self.width, self.height), Image.NEAREST)
                cloth_densepose = cloth_densepose.resize(
                    (self.width, self.height), Image.NEAREST
                )
            else:
                densepose = self.vae_processor.preprocess(
                    densepose, self.height, self.width
                )[0]

            image = self.prepare_image(image)
            cloth = self.prepare_image(cloth)
            mask = self.prepare_mask(mask)
            if self.dataset in ["pose_transfer", "deepfashion"]:
                densepose = self.prepare_densepose(densepose)
                cloth_densepose = self.prepare_densepose(cloth_densepose)
            else:
                densepose = self.prepare_image(densepose)

            image_list.append(image)
            cloth_list.append(cloth)
            mask_list.append(mask)
            densepose_list.append(densepose)
            if self.dataset in ["pose_transfer", "deepfashion"]:
                cloth_densepose_list.append(cloth_densepose)
            cloth4clip_list.append(cloth4clip)

        image = torch.cat(image_list, dim=0)
        cloth = torch.cat(cloth_list, dim=0)
        mask = torch.cat(mask_list, dim=0)
        densepose = torch.cat(densepose_list, dim=0)
        if self.dataset in ["pose_transfer", "deepfashion"]:
            cloth_densepose = torch.cat(cloth_densepose_list, dim=0)
        cloth4clip = torch.cat(cloth4clip_list, dim=0)

        batch["image"] = image
        batch["cloth_pure"] = cloth
        batch["inpaint_mask"] = mask
        batch["densepose"] = densepose
        if self.dataset in ["pose_transfer", "deepfashion"]:
            batch["cloth_densepose"] = cloth_densepose
        batch["cloth4clip"] = cloth4clip

        return batch

    @staticmethod
    def prepare_image(image):
        if isinstance(image, torch.Tensor):
            # Batch single image
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = image.to(dtype=torch.float32)
        else:
            # preprocess image
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]
            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(
                dtype=torch.float32) / 127.5 - 1.0
        return image

    @staticmethod
    def prepare_mask(mask):
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                # Batch and add channel dim for single mask
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[0] == 1:
                # Single mask, the 0'th dimension is considered to be
                # the existing batch size of 1
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[0] != 1:
                # Batch of mask, the 0'th dimension is considered to be
                # the batching dimension
                mask = mask.unsqueeze(1)

            # Binarize mask
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
        else:
            # preprocess mask
            if isinstance(mask, (PIL.Image.Image, np.ndarray)):
                mask = [mask]

            if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
                mask = np.concatenate(
                    [np.array(m.convert("L"))[None, None, :] for m in mask],
                    axis=0,
                )
                mask = mask.astype(np.float32) / 255.0
            elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
                mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = torch.from_numpy(mask)

        return mask

    @staticmethod
    def prepare_densepose(densepose):
        """
        For internal (meta) densepose, the first and second channel should be normalized to 0~1 by 255.0,
        and the third channel should be normalized to 0~1 by 24.0
        """
        if isinstance(densepose, torch.Tensor):
            # Batch single densepose
            if densepose.ndim == 3:
                densepose = densepose.unsqueeze(0)
            densepose = densepose.to(dtype=torch.float32)
        else:
            # preprocess densepose
            if isinstance(densepose, (PIL.Image.Image, np.ndarray)):
                densepose = [densepose]
            if isinstance(densepose, list) and isinstance(
                densepose[0], PIL.Image.Image
            ):
                densepose = [np.array(i.convert("RGB"))[None, :]
                             for i in densepose]
                densepose = np.concatenate(densepose, axis=0)
            elif isinstance(densepose, list) and isinstance(densepose[0], np.ndarray):
                densepose = np.concatenate(
                    [i[None, :] for i in densepose], axis=0)
            densepose = densepose.transpose(0, 3, 1, 2)
            densepose = densepose.astype(np.float32)
            densepose[:, 0:2, :, :] /= 255.0
            densepose[:, 2:3, :, :] /= 24.0
            densepose = torch.from_numpy(densepose).to(
                dtype=torch.float32) * 2.0 - 1.0
        return densepose


def densepose_to_mask(densepose_pil):
    mask = np.array(densepose_pil).copy()
    fg_mask = mask[:, :, 2] != 0

    kernel_size = 10
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    fg_mask = cv2.dilate(fg_mask.astype(np.uint8) * 255, kernel, iterations=5)

    return Image.fromarray(np.stack([fg_mask, fg_mask, fg_mask], axis=-1))
