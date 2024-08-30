import os
import warnings
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from .configuration_bisenetv2 import BiSeNetV2Config

backbone_url = (
    "https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth"
)


_CONFIG_FOR_DOC = "BiSeNetV2Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "ai-human-lab/BiSeNetV2"
_EXPECTED_OUTPUT_SHAPE = [1, 19, 224, 224]


@dataclass
class BiSeNetV2BaseModelOutput(ModelOutput):
    """
    Base class for models that have been trained with the BiSeNetV2 loss objective.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, hidden_size, W, H)`):
            Sequence of hidden-states at the output of the last layer of the model.
        logits_auxs (`tuple(torch.FloatTensor)`, *optional*, returned when `config.return_auxs=True`):
            Tuple of `torch.FloatTensor` (logits of aux layers) of
            shape `(batch_size, hidden_size, W, H)`.
    """

    logits: torch.FloatTensor = None
    logits_auxs: Optional[Tuple[torch.FloatTensor]] = None


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        ks=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class UpSample(nn.Module):
    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.0)


class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class CEBlock(nn.Module):
    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        # TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class GELayerS1(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan,
                mid_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_chan,
                bias=False,
            ),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan,
                mid_chan,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_chan,
                bias=False,
            ),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan,
                mid_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=mid_chan,
                bias=False,
            ),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_chan,
                in_chan,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_chan,
                bias=False,
            ),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Module):
    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):
    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        # TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # not shown in paper
        )

    def forward(self, x_d, x_s):
        # dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out


class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1),
            )
            if aux
            else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode="bilinear", align_corners=False),
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


class BiSeNetV2(nn.Module):
    def __init__(self, config):
        super(BiSeNetV2, self).__init__()
        self.return_auxs = config.return_auxs
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()

        self.head = SegmentHead(
            config.head_hidden_size, 1024, config.num_classes, up_factor=8, aux=False
        )
        if self.return_auxs is True:
            self.aux2 = SegmentHead(
                config.aux_hidden_size[0], 128, config.num_classes, up_factor=4
            )
            self.aux3 = SegmentHead(
                config.aux_hidden_size[1], 128, config.num_classes, up_factor=8
            )
            self.aux4 = SegmentHead(
                config.aux_hidden_size[2], 128, config.num_classes, up_factor=16
            )
            self.aux5_4 = SegmentHead(
                config.aux_hidden_size[3], 128, config.num_classes, up_factor=32
            )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        return_auxs: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BiSeNetV2BaseModelOutput]:
        feat_d = self.detail(pixel_values)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(pixel_values)
        feat_head = self.bga(feat_d, feat_s)

        logits = self.head(feat_head)
        logits_auxs = None
        if return_auxs is True:
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            logits_auxs = [logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4]

        if not return_dict:
            return tuple(v for v in [logits, logits_auxs] if v is not None)
        return BiSeNetV2BaseModelOutput(
            logits=logits,
            logits_auxs=logits_auxs,
        )


BiSeNetV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BiSeNetV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BiSeNetV2_INPUTS_DOCSTRING = r"""

    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`BiSeNetV2ImageProcessor.__call__`] for details.

        return_auxs (`bool`, *optional*):
            Whether or not to return the aux tensors of all aux layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The BiSeNetV2.",
    BiSeNetV2_START_DOCSTRING,
)
class BiSeNetV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BiSeNetV2Config
    base_model_prefix = "bisenetv2"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            if hasattr(module, "last_bn") and module.last_bn:
                nn.init.zeros_(module.weight)
            else:
                nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def load_pretrain(self):
        state = modelzoo.load_url(backbone_url)
        for name, child in self.bisenetv2.named_children():
            if name in state.keys():
                child.load_state_dict(state[name], strict=True)


class BiSeNetV2ForSemanticSegmentation(BiSeNetV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bisenetv2 = BiSeNetV2(config)

        # Initialize weights and apply final processing
        self.post_init()
        self.load_pretrain()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        return_auxs: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BiSeNetV2BaseModelOutput]:
        return_auxs = (
            return_auxs if return_auxs is not None else self.config.return_auxs
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bisenetv2(
            pixel_values,
            return_auxs=return_auxs,
            return_dict=return_dict,
        )
        return outputs
