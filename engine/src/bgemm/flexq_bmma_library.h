// Copyright (C) 2024 ByteDance and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//          http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "common/base.h"
#include "flexq_bmma_op.h"

// ------------------------FlexQ W6A6------------------------
// cta<1,16,256> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 256, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 256, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 256, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 256, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 256, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,16,256> warp<16,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 256, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 256, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 256, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 256, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 256, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,16,256> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 256, 24, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 256, 24, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 256, 24, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 256, 24, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 256, 24, 48, 128, 8, 8, 128, 6, 1);

// cta<8,16,256> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 256, 48, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 256, 48, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 256, 48, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 256, 48, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 256, 48, 48, 128, 8, 8, 128, 6, 1);

// cta<1,32,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,32,256> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 256, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 256, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 256, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 256, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 256, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,32,256> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 256, 24, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 256, 24, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 256, 24, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 256, 24, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 256, 24, 48, 128, 8, 8, 128, 6, 1);

// cta<8,32,256> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 6, 1);

// cta<1,64,256> warp<8,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,64,256> warp<16,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 64, 256, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 64, 256, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 64, 256, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 64, 256, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 64, 256, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,64,256> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 64, 256, 24, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 64, 256, 24, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 64, 256, 24, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 64, 256, 24, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 64, 256, 24, 48, 128, 8, 8, 128, 6, 1);

// cta<8,64,256> warp<48,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 64, 256, 48, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 64, 256, 48, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 64, 256, 48, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 64, 256, 48, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 64, 256, 48, 48, 128, 8, 8, 128, 6, 1);

// cta<1,16,384> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 384, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 384, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 384, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 384, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 384, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,16,384> warp<16,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 384, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 384, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 384, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 384, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 384, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,16,384> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 384, 24, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 384, 24, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 384, 24, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 384, 24, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 384, 24, 48, 128, 8, 8, 128, 6, 1);

// cta<8,16,384> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 384, 48, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 384, 48, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 384, 48, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 384, 48, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 384, 48, 48, 128, 8, 8, 128, 6, 1);

// cta<1,32,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,32,384> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 384, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 384, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 384, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 384, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 384, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,32,384> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 6, 1);

// cta<8,32,384> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 384, 48, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 384, 48, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 384, 48, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 384, 48, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 384, 48, 48, 128, 8, 8, 128, 6, 1);

// cta<1,64,384> warp<8,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 5, 1);

// cta<2,64,384> warp<16,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 64, 384, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 64, 384, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 64, 384, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 64, 384, 16, 48, 128, 8, 8, 128, 5, 1);

// cta<4,64,384> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 64, 384, 24, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 64, 384, 24, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 64, 384, 24, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 64, 384, 24, 48, 128, 8, 8, 128, 5, 1);

// cta<8,64,384> warp<48,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 64, 384, 48, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 64, 384, 48, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 64, 384, 48, 48, 128, 8, 8, 128, 4, 1);

// cta<1,16,512> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 512, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 512, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 512, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 512, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 16, 512, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,16,512> warp<16,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 512, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 512, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 512, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 512, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 16, 512, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,16,512> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 512, 24, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 512, 24, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 512, 24, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 512, 24, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 16, 512, 24, 48, 128, 8, 8, 128, 6, 1);

// cta<8,16,512> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 512, 48, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 512, 48, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 512, 48, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 512, 48, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 16, 512, 48, 48, 128, 8, 8, 128, 6, 1);

// cta<1,32,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,32,512> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 512, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 512, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 512, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 512, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 32, 512, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,32,512> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 6, 1);

// cta<8,32,512> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 512, 48, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 512, 48, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 512, 48, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 512, 48, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 32, 512, 48, 48, 128, 8, 8, 128, 6, 1);

// cta<1,64,512> warp<8,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 64, 512, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 1, 64, 512, 8, 48, 128, 8, 8, 128, 3, 1);

// cta<2,64,512> warp<16,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 64, 512, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 2, 64, 512, 16, 48, 128, 8, 8, 128, 3, 1);

// cta<4,64,512> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 64, 512, 24, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 4, 64, 512, 24, 48, 128, 8, 8, 128, 3, 1);

// cta<8,64,512> warp<48,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 64, 512, 48, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 6, 6, true, 8, 64, 512, 48, 48, 128, 8, 8, 128, 3, 1);





// ------------------------FlexQ W6A8------------------------
// cta<1,16,256> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 256, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 256, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 256, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 256, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 256, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,16,256> warp<16,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 256, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 256, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 256, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 256, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 256, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,16,256> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 256, 32, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 256, 32, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 256, 32, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 256, 32, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 256, 32, 48, 128, 8, 8, 128, 6, 1);

// cta<8,16,256> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 256, 64, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 256, 64, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 256, 64, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 256, 64, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 256, 64, 48, 128, 8, 8, 128, 6, 1);

// cta<1,32,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,32,256> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 256, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 256, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 256, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 256, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 256, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,32,256> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 256, 32, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 256, 32, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 256, 32, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 256, 32, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 256, 32, 48, 128, 8, 8, 128, 6, 1);

// cta<8,32,256> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 6, 1);

// cta<1,64,256> warp<8,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,64,256> warp<16,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 64, 256, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 64, 256, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 64, 256, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 64, 256, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 64, 256, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,64,256> warp<32,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 64, 256, 32, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 64, 256, 32, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 64, 256, 32, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 64, 256, 32, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 64, 256, 32, 48, 128, 8, 8, 128, 6, 1);

// cta<8,64,256> warp<64,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 6, 1);

// cta<1,16,384> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 384, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 384, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 384, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 384, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 384, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,16,384> warp<16,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 384, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 384, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 384, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 384, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 384, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,16,384> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 384, 32, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 384, 32, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 384, 32, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 384, 32, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 384, 32, 48, 128, 8, 8, 128, 6, 1);

// cta<8,16,384> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 384, 64, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 384, 64, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 384, 64, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 384, 64, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 384, 64, 48, 128, 8, 8, 128, 6, 1);

// cta<1,32,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,32,384> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 384, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 384, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 384, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 384, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 384, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,32,384> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 384, 32, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 384, 32, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 384, 32, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 384, 32, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 384, 32, 48, 128, 8, 8, 128, 6, 1);

// cta<8,32,384> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 384, 64, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 384, 64, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 384, 64, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 384, 64, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 384, 64, 48, 128, 8, 8, 128, 6, 1);

// cta<1,64,384> warp<8,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 5, 1);

// cta<2,64,384> warp<16,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 64, 384, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 64, 384, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 64, 384, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 64, 384, 16, 48, 128, 8, 8, 128, 5, 1);

// cta<4,64,384> warp<32,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 64, 384, 32, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 64, 384, 32, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 64, 384, 32, 48, 128, 8, 8, 128, 4, 1);

// cta<8,64,384> warp<64,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 64, 384, 64, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 64, 384, 64, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 64, 384, 64, 48, 128, 8, 8, 128, 4, 1);

// cta<1,16,512> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 512, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 512, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 512, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 512, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 16, 512, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,16,512> warp<16,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 512, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 512, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 512, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 512, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 16, 512, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,16,512> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 512, 32, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 512, 32, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 512, 32, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 512, 32, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 16, 512, 32, 48, 128, 8, 8, 128, 6, 1);

// cta<8,16,512> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 512, 64, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 512, 64, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 512, 64, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 512, 64, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 16, 512, 64, 48, 128, 8, 8, 128, 6, 1);

// cta<1,32,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 6, 1);

// cta<2,32,512> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 512, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 512, 16, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 512, 16, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 512, 16, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 32, 512, 16, 48, 128, 8, 8, 128, 6, 1);

// cta<4,32,512> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 512, 32, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 512, 32, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 512, 32, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 512, 32, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 32, 512, 32, 48, 128, 8, 8, 128, 6, 1);

// cta<8,32,512> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 512, 64, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 512, 64, 48, 128, 8, 8, 128, 3, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 512, 64, 48, 128, 8, 8, 128, 4, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 512, 64, 48, 128, 8, 8, 128, 5, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 32, 512, 64, 48, 128, 8, 8, 128, 6, 1);

// cta<1,64,512> warp<8,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 64, 512, 8, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 1, 64, 512, 8, 48, 128, 8, 8, 128, 3, 1);

// cta<2,64,512> warp<16,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 64, 512, 16, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 2, 64, 512, 16, 48, 128, 8, 8, 128, 3, 1);

// cta<4,64,512> warp<32,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 64, 512, 32, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 4, 64, 512, 32, 48, 128, 8, 8, 128, 3, 1);

// cta<8,64,512> warp<64,48,128> mma<8,8,128>   WARPS[1x8]
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 64, 512, 64, 48, 128, 8, 8, 128, 2, 1);
FQ_DECL_FUN(FQBMMA, 8, 6, true, 8, 64, 512, 64, 48, 128, 8, 8, 128, 3, 1);