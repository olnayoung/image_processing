import warnings

from ...utils import (
    OptionalDependencyNotAvailable,
    is_onnxruntime_available,
    is_torch_available,
    is_torchvision_available,
    is_transformers_available,
)

try:
    if not (
        is_torch_available()
        and is_torchvision_available()
        and is_transformers_available()
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    warnings.warn(
        f"""
        To use `est.models.ppmattingv2` you need to install
        PyTorch: {is_torch_available()}
        Torchvision: {is_torchvision_available()}
        Transformers: {is_transformers_available()}
        """
    )
else:
    from .image_processing_ppmattingv2 import PPMattingV2ImageProcessor

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
    from .modeling_onnx_ppmatingv2 import ORTPPMattingV2ForHumanMatting
