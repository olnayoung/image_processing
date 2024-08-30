import importlib.metadata
__version__= importlib.metadata.version('est_models')

import warnings
from .utils import (
    OptionalDependencyNotAvailable,
    is_diffusers_available,
    is_onnx_available,
    is_onnx_graphsurgeon_available,
    is_onnxruntime_available,
    is_polygraphy_available,
    is_tensorrt_available,
    is_torch_available,
    is_torchvision_available,
    is_transformers_available,
)

try:
    if (
        not is_torch_available()
        or not is_torchvision_available()
        or not is_transformers_available()
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    warnings.warn(
        f"""
        To use `est.models.segformer` you need to install
        PyTorch: {is_torch_available()}
        Torchvision: {is_torchvision_available()}
        Transformers: {is_transformers_available()}
        """
    )
else:
    from .models import (
        BiSeNetV2Config,
        BiseNetV2ImageProcessor,
        BiSeNetV2,
        BiSeNetV2ForSemanticSegmentation,
        PPMattingV2ImageProcessor,
        SegformerImageProcessor,
    )


try:
    if not (
        is_torch_available()
        and is_torchvision_available()
        and is_transformers_available()
        and is_onnxruntime_available()
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    warnings.warn(
        f"""
        To use `est.models.ppmattingv2` you need to install
        PyTorch: {is_torch_available()}
        Torchvision: {is_torchvision_available()}
        Transformers: {is_transformers_available()}
        onnxruntime: {is_onnxruntime_available()}
        """
    )
else:
    from .models import (
        ORTPPMattingV2ForHumanMatting
    )

try:
    if not (
        is_transformers_available()
        and is_torch_available()
        and is_diffusers_available()
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    warnings.warn(
        f"""
        To use `est.pipelines.stable_diffusion` you need to install
        PyTorch: {is_torch_available()}
        Transformers: {is_transformers_available()}
        Diffusers: {is_diffusers_available()}
        """
    )
else:
    from .pipelines import (
        StableDiffusionFillInpaintPipeline,
        StableDiffusionWebuiInpaintPipeline,
        StableDiffusionControlNetFillInpaintPipeline
    )

try:
    if not (
        is_transformers_available()
        and is_torch_available()
        and is_diffusers_available()
        and is_tensorrt_available()
        and is_polygraphy_available()
        and is_onnx_graphsurgeon_available()
        and is_onnxruntime_available()
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    warnings.warn(
        f"""
        To use `est.pipelines.stable_diffusion.TensorRTStableDiffusionFillInpaintPipeline` you need to install
        PyTorch: {is_torch_available()}
        Transformers: {is_transformers_available()}
        Diffusers: {is_diffusers_available()}
        TensorRT: {is_tensorrt_available()}
        Polygraphy: {is_polygraphy_available()}
        onnx_graphsurgeon: {is_onnx_graphsurgeon_available()}
        Onnxruntime: {is_onnxruntime_available()}
        """
    )
else:
    from .pipelines import (
        TensorRTStableDiffusionFillInpaintPipeline,
        TensorRTStableDiffusionInpaintPipeline,
    )
