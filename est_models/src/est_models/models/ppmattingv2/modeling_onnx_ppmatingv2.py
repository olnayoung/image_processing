import os
from typing import Optional, Union

# For deployment of GPU/TensorRT, please refer to examples/vision/detection/paddledetection/python
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download, try_to_load_from_cache


class ORTPPMattingV2ForHumanMatting:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        if device == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.ort_session = onnxruntime.InferenceSession(model_path, providers=providers)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """
        Input batch of images (batch, channel=3, width, height)
        Output batch of masks (batch, channel=1, width, height)
        """
        ort_inputs = {self.ort_session.get_inputs()[0].name: images}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return ort_outs[0]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        device: str = "cuda",
    ):
        if not os.path.isfile(pretrained_model_name_or_path):
            model_path = try_to_load_from_cache(
                pretrained_model_name_or_path, "model.onnx"
            )
            if not isinstance(model_path, str):
                model_path = hf_hub_download(
                    pretrained_model_name_or_path, "model.onnx"
                )
        else:
            model_path = pretrained_model_name_or_path

        return cls(model_path, device)
