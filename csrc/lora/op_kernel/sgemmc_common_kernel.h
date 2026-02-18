/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/csrc/kernels/sgmv_expand.cpp
 */

#ifndef SGL_KERNEL_NPU_KERNEL_SGEMMC_COMMON_H
#define SGL_KERNEL_NPU_KERNEL_SGEMMC_COMMON_H

#include "kernel_operator.h"
#include "sgemmc_utils_kernel.h"

namespace SGEMMCommon {

struct SGEMMTiling {
    int32_t dataType;
    TCubeTiling tiling;
};



template <typename scalar_t>
class SGEMMCommon
{
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    using X_MAT_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::VECTOR, X_T, false>;
    using W_MAT_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, W_T, true>;
    using Y_MAT_TYPE = AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, Y_T>;
    using BIAS_MAT_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T>;

    using MAT_TYPE = AscendC::Matmul<X_MAT_TYPE, W_MAT_TYPE, Y_MAT_TYPE, BIAS_MAT_TYPE, CFG_MDL>;

    constexpr static bool transposeX = MAT_TYPE::AT::isTrans;
    constexpr static bool transposeW = MAT_TYPE::BT::isTrans;

public:
    __aicore__ inline SGEMMCommon(AscendC::TPipe *pipe) : pipe_(pipe) {}
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetX, int32_t &offsetW,
                                      int32_t &offsetY)
    {}

private:
    AscendC::TPipe *pipe_;
};

}  // namespace SGEMMCommon

#endif  // SGL_KERNEL_NPU_KERNEL_SGEMMC_COMMON_H
