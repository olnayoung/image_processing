# 목차
- [Warning](#warning)
- [Usage](#usage)
  * [FreeU](#FreeU)
  * [StableDiffusionFillInpaintPipeline](#stablediffusionfillinpaintpipeline)
  * [TensorRTStableDiffusionInpaintPipeline](#tensorrtstablediffusioninpaintpipeline)
  * [TensorRTStableDiffusionFillInpaintPipeline](#tensorrtstablediffusionfillinpaintpipeline)
  * [StableDiffusionControlNetFillInpaintPipeline](#stablediffusioncontrolnetfillinpaintpipeline)

# Warning
1 아래와 같은 에러 발생 시 pytorch nightly를 사용해야 함
```bash
# 아래 에러 발생 시
torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::scaled_dot_product_attention' to ONNX opset version 17 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub: https://github.com/pytorch/pytorch/issues.
```
2 nvcr.io/nvidia/tensorrt:23.06-py3을 사용해야 함
```bash
# 아래 에러 발생 시
NameError: name 'TRT_LOGGER' is not defined
```


# Usage

## FreeU
```python
from est_models.pipelines.stable_diffusion import StableDiffusionFillInpaintPipeline
import torch

pipeline = StableDiffusionFillInpaintPipeline.from_pretrained("Uminosachi/realisticVisionV40_v40VAE-inpainting", torch_dtype=torch.float16).to('cuda')

#Enable
pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)

#Disable
pipeline.disable_freeu()

```

## StableDiffusionFillInpaintPipeline

```python
from est_models.pipelines.stable_diffusion import StableDiffusionFillInpaintPipeline
from est_models.pipelines.clip_text_custom_embedder import text_embeddings
import torch

pipe = StableDiffusionFillInpaintPipeline.from_pretrained("Uminosachi/realisticVisionV40_v40VAE-inpainting", torch_dtype=torch.float16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')
image_size = image.size
cond, uncond = text_embeddings(pipe, prompt, negative_prompt, clip_stop_at_last_layers=2)
generator = torch.Generator(device='cuda').manual_seed(1234)
new_image = pipe(
    prompt_embeds=cond,
    negative_prompt_embeds=uncond,
    num_inference_steps=22,
    image=image,
    mask_image=mask_img,
    mask_blur=4,
    width=512,
    height=512,
    generator=generator,
    strength=0.75,
    guidance_scale=7
).images
```

## TensorRTStableDiffusionInpaintPipeline

```python
import torch
from diffusers import EulerAncestralDiscreteScheduler
from est_models.pipelines.stable_diffusion import TensorRTStableDiffusionInpaintPipeline
from pathlib import Path

# Use the DDIMScheduler scheduler here instead
weight_path = "whooray/stable_diffusion_tensorrt_inpaint"
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(weight_path,
                                            subfolder="scheduler")

pipe = TensorRTStableDiffusionInpaintPipeline.from_pretrained(weight_path,
                                                torch_dtype=torch.float16,
                                                scheduler=scheduler,

                                                             )

# re-use cached folder to save ONNX models and TensorRT Engines
Path(weight_path).mkdir(exist_ok=True)
pipe.set_cached_folder(weight_path)

pipe = pipe.to("cuda")

output = pipe(prompt=["pink hair"]*8,
              image=img,
              mask_image=mask,
              strength=0.5,
              num_inference_steps=50

             )
```

## TensorRTStableDiffusionFillInpaintPipeline

```python
import torch
from diffusers import EulerAncestralDiscreteScheduler
from est_models.pipelines.stable_diffusion import TensorRTStableDiffusionFillInpaintPipeline
from pathlib import Path

# Use the DDIMScheduler scheduler here instead
weight_path = "whooray/stable_diffusion_tensorrt_inpaint"
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(weight_path,
                                            subfolder="scheduler")

pipe = TensorRTStableDiffusionFillInpaintPipeline.from_pretrained(weight_path,
                                                torch_dtype=torch.float16,
                                                scheduler=scheduler,

                                                             )

# re-use cached folder to save ONNX models and TensorRT Engines
Path(weight_path).mkdir(exist_ok=True)
pipe.set_cached_folder(weight_path)

pipe = pipe.to("cuda")

output = pipe(prompt=["pink hair"]*8,
              image=img,
              mask_image=mask,
              strength=0.5,
              num_inference_steps=50

             )
```


## StableDiffusionControlNetFillInpaintPipeline

```python
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
from est_models.pipelines.controlnet import StableDiffusionControlNetFillInpaintPipeline
from controlnet_aux.processor import Processor

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetFillInpaintPipeline.from_pretrained(
     "whooray/realistic-vision-1.3-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()

pipe.to('cuda')
processor = Processor('depth_midas')

# generator = torch.manual_seed(0)
new_image = pipe(
    text_prompt,
    num_inference_steps=20,
#     generator=generator,
    image=img,
    control_image=masked_controlnet_img,
    mask_image=hair_mask
).images[0]
```
