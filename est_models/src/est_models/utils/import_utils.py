import importlib.metadata
import importlib.util
from typing import Any, Tuple, Union


def _is_package_available(
    pkg_name: str, return_version: bool = False
) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_onnx_available = _is_package_available("onnx")
_torchvision_available = _is_package_available("torchvision")
_torch_available = _is_package_available("torch")
_transformers_available = _is_package_available("transformers")
_tensorrt_available = _is_package_available("tensorrt")
_polygraphy_available = _is_package_available("polygraphy")
_onnx_graphsurgeon_available = _is_package_available("onnx_graphsurgeon")
_onnxruntime_available = _is_package_available("onnxruntime")
_diffusers_available = _is_package_available("diffusers")


def is_torch_available():
    return _torch_available


def is_torchvision_available():
    return _torchvision_available


def is_onnx_available():
    return _onnx_available


def is_transformers_available():
    return _transformers_available


def is_tensorrt_available():
    return _tensorrt_available


def is_polygraphy_available():
    return _polygraphy_available


def is_onnx_graphsurgeon_available():
    return _onnx_graphsurgeon_available


def is_onnxruntime_available():
    return _onnxruntime_available


def is_diffusers_available():
    return _diffusers_available


class OptionalDependencyNotAvailable(BaseException):
    """Internally used error class for signalling an optional dependency was not found."""
