import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import SegformerImageProcessor as TransformersSegformerImageProcessor


class SegformerImageProcessor(TransformersSegformerImageProcessor):
    def post_process_binary_segmentation(
        self, outputs, threshold: float = 0.5, target_sizes: List[Tuple] = None
    ):
        logits = outputs.logits
        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            binary_segmentation = []

            for idx in range(len(logits)):
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                binary_map = (
                    torch.sigmoid(resized_logits)
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .squeeze(dim=-1)
                    > threshold
                )
                binary_map = np.where(binary_map.cpu().numpy() > 0, 255, 0)
                binary_segmentation.append(binary_map)
        else:
            binary_segmentation = torch.sigmoid(logits) > threshold
            binary_segmentation = [
                np.where(
                    binary_segmentation[i].permute(1, 2, 0).squeeze(dim=-1).numpy() > 0,
                    255,
                    0,
                )
                for i in range(binary_segmentation.shape[0])
            ]
        return binary_segmentation
