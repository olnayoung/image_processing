# coding=utf-8
# Copyright 2021 Google AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BiSeNetV2 model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BiSeNetV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BiSeNetV2Model`]. It is used to instantiate an BiSeNetV2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BiSeNetV2

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        encoder_stride (`int`, *optional*, defaults to 16):
           Factor to increase the spatial resolution by in the decoder head for masked image modeling.

    Example:

    ```python
    >>> from transformers import BiSeNetV2Config, BiSeNetV2Model

    >>> # Initializing a BiSeNetV2 BiSeNetV2 style configuration
    >>> configuration = BiSeNetV2Config()

    >>> # Initializing a model (with random weights) from the bisenetv2 style configuration
    >>> model = BiSeNetV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bisenetv2"

    def __init__(
        self,
        head_hidden_size=128,
        return_auxs=True,
        aux_hidden_size=[16, 32, 64, 128],
        num_classes=19,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.head_hidden_size = head_hidden_size
        self.return_auxs = return_auxs
        self.aux_hidden_size = aux_hidden_size
        self.num_classes = num_classes
