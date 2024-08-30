#
# Copyright 2023 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
from collections import OrderedDict
from copy import copy
from typing import List, Optional, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import PIL
import tensorrt as trt
import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionInpaintPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    prepare_mask_and_masked_image,
)
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import logging
from huggingface_hub import snapshot_download
from onnx import shape_inference
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.onnx.loader import fold_constants
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.backend.trt import util as trt_util
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from ..webui_utils import apply_overlay, crop_image_mask
from .pipeline_tensorrt_stable_diffusion_inpaint import (
    TensorRTStableDiffusionInpaintPipeline,
)


class TensorRTStableDiffusionFillInpaintPipeline(
    TensorRTStableDiffusionInpaintPipeline
):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_blur: int = 4,
        strength: float = 0.75,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        inpaint_full_res_padding: int = 32,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            mask_image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, H, W, 1)`.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.

        """
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        
        self.generator = generator
        self.denoising_steps = num_inference_steps
        self._guidance_scale = guidance_scale

        # Pre-compute latent input scales and linear multistep coefficients
        self.scheduler.set_timesteps(self.denoising_steps, device=self.torch_device)

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"Expected prompt to be of type list or str but got {type(prompt)}"
            )

        if negative_prompt is None:
            negative_prompt = [""] * batch_size

        if negative_prompt is not None and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        assert len(prompt) == len(negative_prompt)

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(prompt)} is larger than allowed {self.max_batch_size}. If dynamic shape is used, then maximum batch size is 4"
            )

        # Validate image dimensions
        mask_width, mask_height = mask_image.size
        if mask_height != self.image_height or mask_width != self.image_width:
            raise ValueError(
                f"Input image height and width {self.image_height} and {self.image_width} are not equal to "
                f"the respective dimensions of the mask image {mask_height} and {mask_width}"
            )

        # load resources
        self.loadResources(self.image_height, self.image_width, batch_size)

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            # Spatial dimensions of latent tensor
            latent_height = self.image_height // 8
            latent_width = self.image_width // 8

            # 4.5 crop image
            image, mask_image, paste_to, overlay_images = crop_image_mask(
                image,
                mask_image,
                self.image_width,
                self.image_height,
                mask_blur,
                inpaint_full_res_padding=inpaint_full_res_padding,
            )

            # 5. Pre-process input images
            mask, masked_image, init_image = self.preprocess_images(
                batch_size,
                prepare_mask_and_masked_image(
                    image,
                    mask_image,
                    self.image_height,
                    self.image_width,
                    return_image=True,
                ),
            )

            mask = torch.nn.functional.interpolate(
                mask, size=(latent_height, latent_width)
            )
            mask = torch.cat([mask] * 2)

            # Initialize timesteps
            timesteps, t_start = self.initialize_timesteps(
                self.denoising_steps, strength
            )

            # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
            latent_timestep = timesteps[:1].repeat(batch_size)
            # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
            is_strength_max = strength == 1.0

            # Pre-initialize latents
            num_channels_latents = self.vae.config.latent_channels
            latents_outputs = self.prepare_latents(
                batch_size,
                num_channels_latents,
                self.image_height,
                self.image_width,
                torch.float32,
                self.torch_device,
                generator,
                image=init_image,
                timestep=latent_timestep,
                is_strength_max=is_strength_max,
            )

            latents = latents_outputs[0]

            # VAE encode masked image
            masked_latents = self.encode_image(masked_image)
            masked_latents = torch.cat([masked_latents] * 2)

            # CLIP text encoder
            text_embeddings = self.encode_prompt(prompt, negative_prompt)

            # UNet denoiser
            latents = self.denoise_latent(
                latents,
                text_embeddings,
                timesteps=timesteps,
                step_offset=t_start,
                mask=mask,
                masked_image_latents=masked_latents,
            )

            # VAE decode latent
            images = self.decode_latent(latents)

        images = self.numpy_to_pil(images)
        images = [apply_overlay(x, paste_to, overlay_images[0]) for x in images]
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)
