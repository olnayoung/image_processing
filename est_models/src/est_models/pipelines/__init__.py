import warnings

from ..utils import (
    OptionalDependencyNotAvailable,
    is_diffusers_available,
    is_onnx_graphsurgeon_available,
    is_onnxruntime_available,
    is_polygraphy_available,
    is_tensorrt_available,
    is_torch_available,
    is_transformers_available,
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
    from .stable_diffusion import (
        StableDiffusionFillInpaintPipeline,
        StableDiffusionWebuiInpaintPipeline,
    )
    from .controlnet import (
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
    from .stable_diffusion import (
        TensorRTStableDiffusionFillInpaintPipeline,
        TensorRTStableDiffusionInpaintPipeline,
    )
