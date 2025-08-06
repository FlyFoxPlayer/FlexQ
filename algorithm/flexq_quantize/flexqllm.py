# Copyright 2024 ByteDance and/or its affiliates
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

import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer, QuantLlamaAttention
from models.int_opt_layer import QuantOPTDecoderLayer, QuantOPTAttention
from flexq_quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
from torch.nn import functional as F
import gc
import logging
import numpy as np
from flexq_quantize.utils import let_parameters, lwc_parameters, get_abq_parameters,com_parameters, \
                            abq_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            weight_quant_inplace,clear_temp_variable,set_quant_state


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def flexqllm(
    lm,
    args,
    logger=None,
):
    logger.info("Starting ...")

    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False

    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1",
            "down_proj":"fc2"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1",
            "fc2":"fc2"
        }
        layer_name_prefix = "model.decoder.layers"
    else:
        raise ValueError("Only support for llama/Llama-2 now")
    
    
    layers[0] = layers[0].to(dev)

    dtype = torch.float16
    traincast = torch.cuda.amp.autocast

    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        set_quant_state(qlayer, weight_quant=True, act_quant=True)
            
        weight_quant_inplace(qlayer, args, is_llama)

        qlayer.half()
        # Register quantization flags into the model
        register_scales_and_zeros(qlayer)
        layers[i] = qlayer.to("cpu")

        
        del layer
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

