def test_bisenetv2_import():
    try:
        from est_models.models.bisenetv2 import (
            BiSeNetV2Config,
            BiSeNetV2ForSemanticSegmentation,
            BiseNetV2ImageProcessor,
        )
    except Exception as e:
        raise ImportError(e)


def test_segformer_import():
    try:
        from est_models.models.segformer import (
            SegformerConfig,
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
    except Exception as e:
        raise ImportError(e)


def test_ppmatingv2_import():
    try:
        from est_models.models.ppmattingv2 import (
            ORTPPMattingV2ForHumanMatting,
            PPMattingV2ImageProcessor,
        )
    except Exception as e:
        raise ImportError(e)


def test_tensorrt_stable_diffusion_import():
    try:
        from est_models.pipelines.stable_diffusion import (
            TensorRTStableDiffusionInpaintPipeline,
        )
    except Exception as e:
        raise ImportError(e)


def test_fill_inpaint_stable_diffusion_import():
    try:
        from est_models.pipelines.stable_diffusion import (
            StableDiffusionFillInpaintPipeline,
        )
    except Exception as e:
        raise ImportError(e)


def test_fill_inpaint_controlnet_import():
    try:
        from est_models.pipelines.controlnet import (
            StableDiffusionControlNetFillInpaintPipeline,
        )
    except Exception as e:
        raise ImportError(e)
