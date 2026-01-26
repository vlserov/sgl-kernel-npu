/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/csrc/kernels/sgmv_shrink.cpp
 */

#ifndef SGL_KERNEL_NPU_KERNEL_SGEMMC_SHRINK_H
#define SGL_KERNEL_NPU_KERNEL_SGEMMC_SHRINK_H

#include "kernel_operator.h"
#include "sgemmc_utils_kernel.h"

template <typename scalar_t>
class SGEMMCShrink
{
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    using X_MAT_TYPE = MatMul<AscendC::TPosition::GM, CubeFormat::ND, X_T, false>;
    using W_MAT_TYPE = MatMul<AscendC::TPosition::GM, CubeFormat::ND, W_T, true>;
    using Y_MAT_TYPE = MatMul<AscendC::TPosition::GM, CubeFormat::ND, Y_T>;
    using BIAS_MAT_TYPE = MatMul<AscendC::TPosition::GM, CubeFormat::ND, X_T>;

    using MAT_TYPE = MMImplType<X_MAT_TYPE, W_MAT_TYPE, Y_MAT_TYPE, BIAS_MAT_TYPE, cfg>

    constexpr static bool transposeX = MAT_TYPE::AT::isTrans;
    constexpr static bool transposeW = MAT_TYPE::BT::isTrans;

    static constexpr uint64_t BUFFER_NUM = 1;
    static constexpr uint64_t TILE_LENGTH = 11776;  // optimal performance tile length

public:
    __aicore__ inline SGEMMCShrink(AscendC::TPipe *pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, uint32_t loraIndicesSize,
                                GM_ADDR seqLen, uint32_t seqLenSize, GM_ADDR loraRanks, uint32_t loraRanksSize,
                                GM_ADDR y, uint32_t batchSize,uint32_t numTokensPerCore, uint32_t inputHiddenDim,
                                uint32_t maxLoRARank)
    {
        batchSize_ = batchSize;
        numTokensPerCore_ = numTokensPerCore;
        inputHiddenDim_ = inputHiddenDim;
        maxLoRARank_ = maxLoRARank;
        singleLoRAWeightLen_ = inputHiddenDim_ * maxLoRARank_;
        incremental_ = inputHiddenDim_ > TILE_LENGTH;

        xInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T *>(x));
        yOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_T *>(y));
        wInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ W_T *>(weight));
        loraIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraIndices), loraIndicesSize);
        seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seqLen), seqLenSize);
        loraRanksGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraRanks), loraRanksSize);

        pipe_->InitBuffer(inQueueX_, BUFFER_NUM, TILE_LENGTH * sizeof(X_T));
        pipe_->InitBuffer(inQueueW_, BUFFER_NUM, TILE_LENGTH * sizeof(W_T));
        pipe_->InitBuffer(tmpBufferX_, TILE_LENGTH * sizeof(float));
        pipe_->InitBuffer(tmpBufferW_, TILE_LENGTH * sizeof(float));

        pipe_->InitBuffer(outQueueY_, 1, maxLoRARank_ * sizeof(Y_T));
        pipe_->InitBuffer(outBufferY_, maxLoRARank_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t startIdx = blockIdx * numTokensPerCore_;
        int64_t endIdx = startIdx + numTokensPerCore_;

        MNConfig mnConfig;
        preOffset = 0;

        AscendC::WaitPreTaskEnd();
        for (uint32_t loraIdx = 0, count = 0; loraIdx < usedLoraCount; ++loraIdx)
        {
            uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
            uint32_t curBlock = coreIdx >= count ? coreIdx : coreIdx + tiling->coreNum;
            uint32_t thresholdM_dimN;

            while (curBlock < curCount) {
                ComputeCube()
                ComputeVector() // Do nothing
                curBlock += tiling->coreNum;
            }
            count = curCount % tiling->coreNum;
        }
        AscendC::SetNextTaskStart();
    }

private:
    __aicore__ inline void ComputeVector()
    {
        if ASCEND_IS_AIC {
            return;
        }
        // Make computations on the vector utit
    }

    __aicore__ inline void ComputeCube()
    {
        if ASCEND_IS_AIC {
            
        }
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void ProcessImpl(const int64_t idx)
    {
        AscendC::LocalTensor<float> yOutLocal = outBufferY_.Get<float>();
        if constexpr (!INCREMENTAL_MODE) {
            CopyInX(idx, 0, inputHiddenDim_);
            AscendC::LocalTensor<float> xTmpTensor = tmpBufferX_.Get<float>();
            AscendC::LocalTensor<X_T> xLocal = inQueueX_.DeQue<X_T>();
            Cast(xTmpTensor, xLocal, AscendC::RoundMode::CAST_NONE, inputHiddenDim_);
            pipe_barrier(PIPE_V);
            inQueueX_.FreeTensor(xLocal);
        }

        for (int i = 0; i < reqLoRARank_; i++) {
            float acc(0);
            for (int32_t j = 0; j < inputHiddenDim_ / TILE_LENGTH; j++) {
                if constexpr (INCREMENTAL_MODE) {
                    CopyInX(idx, j);
                }
                CopyInW(i, j);
                Compute<INCREMENTAL_MODE>(acc);
            }
            CopyAndComputeLastIteration<INCREMENTAL_MODE>(idx, i, acc);
            yOutLocal.SetValue(i, acc);
        }
    }

    __aicore__ inline void CopyInIndex(const int64_t idx)
    {
        // look up the LoRA index
        int64_t weightIdx = idx;
        uint64_t i = 0;
        for (; i < seqLenGm_.GetSize(); i++) {
            int64_t repeatValue = seqLenGm_.GetValue(i);
            if (weightIdx >= repeatValue) {
                weightIdx -= repeatValue;
                continue;
            }
            break;
        }
        reqLoRAIndex_ = (i < seqLenGm_.GetSize()) ? loraIndicesGm_.GetValue(i) : -1;
    }


private:
    AscendC::TPipe *pipe_;
    typename MAT_TYPE::MT& matmul_;

    AscendC::GlobalTensor<X_T> xInGm_;
    AscendC::GlobalTensor<W_T> wInGm_;
    AscendC::GlobalTensor<Y_T> yOutGm_;
    AscendC::GlobalTensor<int32_t> loraIndicesGm_;
    AscendC::GlobalTensor<int32_t> seqLenGm_;
    AscendC::GlobalTensor<int32_t> loraRanksGm_;

    uint32_t batchSize_;
    uint32_t numTokensPerCore_;
    uint32_t inputHiddenDim_;
    uint32_t maxLoRARank_;
    uint32_t singleLoRAWeightLen_;

    uint64_t reqLoRAWeightOffset_;
    int32_t reqLoRAIndex_;
    int32_t reqLoRARank_;
};

extern "C" __global__ __aicore__ void sgemmc_shrink(
    GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, uint32_t loraIndicesSize, GM_ADDR seqLen, uint32_t seqLenSize,
    GM_ADDR loraRanks, uint32_t loraRanksSize, GM_ADDR y, uint32_t batchSize, uint32_t numTokensPerCore,
    uint32_t inputHiddenDim, uint32_t maxLoRARank, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GM_ADDR userWorkSpace = GetUserWorkSpacePtr(workspace);

    SGEMMCShrink<half> op(&pipe);
    op.Init(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, loraRanks, loraRanksSize,
            y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank);
    op.Process();
}

#endif  // SGL_KERNEL_NPU_KERNEL_SGEMMC_SHRINK_H
