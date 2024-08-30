import warnings

from ...utils import (
    OptionalDependencyNotAvailable,
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
        To use `est.models.bisenetv2` you need to install
        PyTorch: {is_torch_available()}
        Torchvision: {is_torchvision_available()}
        Transformers: {is_transformers_available()}
        """
    )

else:
    from .configuration_bisenetv2 import BiSeNetV2Config
    from .image_processing_bisenetv2 import BiseNetV2ImageProcessor
    from .modeling_bisenetv2 import BiSeNetV2, BiSeNetV2ForSemanticSegmentation
