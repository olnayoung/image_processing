import warnings

from ..utils import (
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_torchvision_available,
    is_transformers_available,
    is_onnxruntime_available
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
    from .bisenetv2 import (
        BiSeNetV2Config,
        BiseNetV2ImageProcessor,
        BiSeNetV2,
        BiSeNetV2ForSemanticSegmentation
    )
    from .ppmattingv2 import (
        PPMattingV2ImageProcessor,
    )
    from .segformer import (
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
    from .ppmattingv2 import (
        ORTPPMattingV2ForHumanMatting
    )