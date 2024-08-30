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

def test_freeU():
    try:
        from est_models.pipelines.stable_diffusion import StableDiffusionFillInpaintPipeline
        import torch

        pipeline = StableDiffusionFillInpaintPipeline.from_pretrained("Uminosachi/realisticVisionV40_v40VAE-inpainting", torch_dtype=torch.float16).to('cuda')
        pipeline.to('cuda')
        
        pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        generator = torch.Generator(device='cuda').manual_seed(1234)
        new_image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=22,
            image=image,
            mask_image=mask,
            mask_blur=4,
            width=512,
            height=512,
            generator=generator,
            strength=0.75,
            guidance_scale=7
        ).images

        pipeline.disable_freeu()
    except Exception as e:
        raise e


def test_StableDiffusionFillInpaintPipeline_textEmbedding():
    try:
        from est_models.pipelines.stable_diffusion import StableDiffusionFillInpaintPipeline
        from est_models.pipelines.clip_text_custom_embedder import text_embeddings
        from diffusers import EulerAncestralDiscreteScheduler
        import torch

        pipe = StableDiffusionFillInpaintPipeline.from_pretrained("Uminosachi/realisticVisionV40_v40VAE-inpainting", torch_dtype=torch.float16)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to('cuda')

        cond, uncond = text_embeddings(pipe, prompt, negative_prompt, clip_stop_at_last_layers=2)
        generator = torch.Generator(device='cuda').manual_seed(1234)
        new_image = pipe(
            prompt_embeds=cond,
            negative_prompt_embeds=uncond,
            num_inference_steps=22,
            image=image,
            mask_image=mask,
            mask_blur=4,
            width=512,
            height=512,
            generator=generator,
            strength=0.75,
            guidance_scale=7
        ).images
    except Exception as e:
        raise e

def test_StableDiffusionWebuiInpaintPipeline():
    try:
        from est_models.pipelines.stable_diffusion import StableDiffusionWebuiInpaintPipeline
        import torch

        pipe = StableDiffusionWebuiInpaintPipeline.from_pretrained("Uminosachi/realisticVisionV40_v40VAE-inpainting", torch_dtype=torch.float16)
        pipe.to('cuda')

        new_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=22,
            image=image,
            mask_image=mask,
            mask_blur=4,
            width=512,
            height=512,
            strength=0.75,
            guidance_scale=7
        ).images
    except Exception as e:
        raise e

def test_StableDiffusionControlNetFillInpaintPipeline():
    try:
        from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
        from est_models.pipelines.controlnet import StableDiffusionControlNetFillInpaintPipeline
        from controlnet_aux.processor import Processor
        import torch

        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetFillInpaintPipeline.from_pretrained(
            "whooray/realistic-vision-1.3-inpainting", controlnet=controlnet, torch_dtype=torch.float16
        )

        pipe.to('cuda')
        processor = Processor('depth_midas')
        processed_image = processor(image, to_pil=True)

        new_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=22,
            image=image.resize((512, 512)),
            mask_image=mask.resize((512, 512)),
            height=512,
            width=512,
            control_image=processed_image
        ).images
    except Exception as e:
        raise e