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
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math

CLIPMIN = 1e-5




def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x



class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_group",
        group_size=None,
        shape=None,
        disable_zero_point=False,
        flex_quant=False
    ):
        """
        dynamic_method support per_token, per_channel and per_group
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        self.flex_quant = flex_quant
        # assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        Conver_flag = False
        if self.group_size:
            if len(x.shape) == 3 and x.shape[0] == 1:
                Conver_flag = True
                x = x.squeeze(0)
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        x_int = round_ste(x / scale)

        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int

        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)

        x_dequant = x_dequant.mul(scale)

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        if Conver_flag:
            x_dequant = x_dequant.unsqueeze(0)
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1" and self.flex_quant is False:
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)
        elif self.metric == "fix0to1":
            x = x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel" or self.dynamic_method == "per_group":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()   

        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            if self.n_bits == 2 or self.n_bits == 1:
                scale = range / (2**self.n_bits)
            else:
                scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point
