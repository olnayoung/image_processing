from PIL import Image

image = Image.open("./tests/images/OG2.jpg")
mask = Image.open("./tests/images/OG2_mask.jpg").convert("L")

prompt = "brilliant and smart human with vivid eyes"
prompt = f"RAW photo, {prompt}, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
negative_prompt = "(deformed iris, eyelash, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), \
                    text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, \
                    mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, \
                    extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, \
                    fused fingers, too many fingers, long neck"

def test_TensorRTStableDiffusionInpaintPipeline():
    try:
        from est_models.pipelines.stable_diffusion import TensorRTStableDiffusionInpaintPipeline
        from pathlib import Path
        import torch

        pipe = TensorRTStableDiffusionInpaintPipeline.from_pretrained("Uminosachi/realisticVisionV40_v40VAE-inpainting", torch_dtype=torch.float16)
        pipe.set_cached_folder("whooray/stable_diffusion_tensorrt_inpaint")
        pipe.to('cuda')

        new_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=22,
            image=image.resize((512, 512)),
            mask_image=mask.resize((512, 512)),
            strength=0.75,
            guidance_scale=7
        ).images
    except Exception as e:
        raise e

def test_TensorRTStableDiffusionFillInpaintPipeline():
    try:
        from est_models.pipelines.stable_diffusion import TensorRTStableDiffusionFillInpaintPipeline
        import torch
        from pathlib import Path

        pipe = TensorRTStableDiffusionFillInpaintPipeline.from_pretrained("Uminosachi/realisticVisionV40_v40VAE-inpainting", torch_dtype=torch.float16)
        pipe.set_cached_folder("whooray/stable_diffusion_tensorrt_inpaint")
        pipe.to('cuda')

        new_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=22,
            image=image.resize((512, 512)),
            mask_image=mask.resize((512, 512)),
            mask_blur=4,
            strength=0.75,
            guidance_scale=7
        ).images
    except Exception as e:
        raise e