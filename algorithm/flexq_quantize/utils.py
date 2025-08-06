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

from collections import OrderedDict
from flexq_quantize.int_linear import QuantLinear
import torch
from flexq_quantize.int_matmul import QuantMatMul
from models.transformation import *


def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def com_parameters(model, use_shift=True):
    params = []
    for n, m in model.named_parameters():
        if n.find('compensation') > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)  

def get_abq_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1 or n.find('compensation') > -1:
            params.append(m)
    return iter(params)  

def abq_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

def smooth_and_quant_temporary(model, args, isllama):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.temp_weight = module.weight
    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            # if hasattr(module, "temp_weight"):
            #     module.temp_weight = module.weight_quantizer(module.temp_weight)
            # else:
            #     module.temp_weight = module.weight_quantizer(module.weight)
            if hasattr(module, "temp_weight"):
                temp_weight = module.temp_weight
            else:
                temp_weight = module.weight
            name_tmp = name.replace(".","_")
            if hasattr(model, f"{name_tmp}_compensation_left"):
                compensation_left = getattr(model, f"{name_tmp}_compensation_left")
                compensation_right = getattr(model, f"{name_tmp}_compensation_right")
                temp_weight = temp_weight + compensation_left @ compensation_right
            module.temp_weight = module.weight_quantizer(temp_weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()   
def weight_quant_inplace(model, args, isllama):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            weight = module.weight
            name_tmp = name.replace(".","_")
            module.weight = module.weight_quantizer(weight)
            module.use_temporary_parameter=False

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
