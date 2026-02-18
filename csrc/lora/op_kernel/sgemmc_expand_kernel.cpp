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

#ifndef SGL_KERNEL_NPU_KERNEL_SGEMMC_EXPAND_H
#define SGL_KERNEL_NPU_KERNEL_SGEMMC_EXPAND_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "common_tiling_kernel.h"
#include "lora_common_kernel.h"

template <typename scalar_t>
class SGEMMCExpand
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

public:
    __aicore__ explicit SGEMMCExpand(AscendC::TPipe *pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, uint32_t loraIndicesSize,
                                GM_ADDR seqLen, uint32_t seqLenSize, GM_ADDR loraRanks, uint32_t loraRanksSize,
                                GM_ADDR sliceOffsets, uint32_t sliceOffsetsSize, GM_ADDR yIn, GM_ADDR yOut,
                                uint32_t batchSize, uint32_t numBlocksPerCore, uint32_t maxLoRARank,
                                uint32_t outputFullDim, GM_ADDR workspace, GM_ADDR tiling)
    {
        batchSize_ = batchSize;
        numBlocksPerCore_ = numBlocksPerCore;
        maxLoRARank_ = maxLoRARank;
        sliceCount_ = sliceOffsetsSize - 1;
        outputFullDim_ = outputFullDim;
        singleLoRAWeightLen_ = maxLoRARank_ * outputFullDim_;

        xInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T *>(x));
        wInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ W_T *>(weight));
        yInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_T *>(yIn));
        yOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_T *>(yOut));
        loraIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraIndices), loraIndicesSize);
        seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seqLen), seqLenSize);
        loraRanksGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraRanks), loraRanksSize);
        sliceOffsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sliceOffsets), sliceOffsetsSize);

        pipe_->InitBuffer(inQueueX_, 1, 1024 * sizeof(X_T));
        pipe_->InitBuffer(outQueueY_, 1, 1024 * sizeof(Y_T));
    }

    __aicore__ inline void Process()
    {
        int64_t blocks = AscendC::GetBlockNum();
        int64_t blockIdx = AscendC::GetBlockIdx();

        int64_t startIdx = blockIdx * numBlocksPerCore_;
        int64_t endIdx = startIdx + numBlocksPerCore_;
        reqSlice_ = blockIdx_Slice % sliceCount_;

        sliceOffset_ = sliceOffsetsGm_.GetValue(reqSlice_);
        outputHiddenDim_ = sliceOffsetsGm_.GetValue(reqSlice_ + 1) - sliceOffset_;

        if (endIdx > batchSize_) {
            endIdx = batchSize_;
        }

        int64_t requestBlock = 0;
        lora_common::BlockIterator blockIterator(seqLenGm_);
        for (int64_t idx = startIdx; idx < endIdx; idx++) {
            yOffset_ = outputFullDim_ * idx + sliceOffset_;

            // Set up LoRA index
            requestBlock = blockIterator.GetBlockIdx(idx);
            if (requestBlock < 0) {
                continue;
            }

            reqLoRAIndex_ = loraIndicesGm_.GetValue(requestBlock);
            if (reqLoRAIndex_ < 0) {
                continue;
            }

            reqLoRARank_ = loraRanksGm_.GetValue(reqLoRAIndex_);
            if (reqLoRARank_ == 0) {
                continue;
            }

            reqLoRAWeightOffset_ = reqLoRAIndex_ * singleLoRAWeightLen_ + sliceOffset_ * maxLoRARank_;

        }
    }

private:

    __aicore__ inline void CopyOut(int32_t progress, int32_t numElements = 1024)
    {
        AscendC::LocalTensor<Y_T> yOutLocal = outQueueY_.DeQue<Y_T>();
        DataCopy(yOutGm_[yOffset_ + progress * 1024], yOutLocal, numElements);
        outQueueY_.FreeTensor(yOutLocal);
    }

private:
    AscendC::TPipe *pipe_;
    MAT_TYPE matmulObj;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY_;

    AscendC::GlobalTensor<X_T> xInGm_;
    AscendC::GlobalTensor<W_T> wInGm_;
    AscendC::GlobalTensor<Y_T> yInGm_;
    AscendC::GlobalTensor<Y_T> yOutGm_;

    AscendC::GlobalTensor<int32_t> seqLenGm_;
    AscendC::GlobalTensor<int32_t> loraIndicesGm_;

    AscendC::GlobalTensor<int32_t> loraRanksGm_;
    AscendC::GlobalTensor<int32_t> sliceOffsetsGm_;

    uint32_t batchSize_;
    uint32_t sliceCount_;
    uint32_t numBlocksPerCore_;
    uint32_t maxLoRARank_;
    uint32_t outputHiddenDim_;
    uint32_t sliceOffset_;
    uint32_t outputFullDim_;
    uint32_t singleLoRAWeightLen_;
    int64_t reqLoRAIndex_;
    int32_t reqLoRARank_;
    uint64_t reqLoRAWeightOffset_;
    int32_t reqSlice_;
    uint32_t numOutputElementsPerInputTile_;
    uint32_t numStreamInPerOutputTile_;
    uint64_t yOffset_;
};

extern "C" __global__ __aicore__ void sgemmc_expand(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices,
                                                    uint32_t loraIndicesSize, GM_ADDR seqLen, uint32_t seqLenSize,
                                                    GM_ADDR loraRanks, uint32_t loraRanksSize, GM_ADDR sliceOffsets,
                                                    uint32_t sliceOffsetsSize, GM_ADDR yIn, GM_ADDR yOut,
                                                    uint32_t batchSize, uint32_t numBlocksPerCore, uint32_t maxLoRARank,
                                                    uint32_t outputFullDim, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    AscendC::TPipe pipe;
    SGEMMCommon::SGEMMTiling tilingData;
    kernel_utils::CopyTiling(&tilingData, tiling);
    GM_ADDR userWorkSpace = GetUserWorkSpacePtr(workspace);

    if (tilingData.dataType == matmul_tiling::DataType::DT_BFLOAT16) {
        SGEMMCExpand<bfloat16_t> op(&pipe);
        op.Init(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, loraRanks, loraRanksSize, sliceOffsets,
                sliceOffsetsSize, yIn, yOut, batchSize, numBlocksPerCore, maxLoRARank, outputFullDim, workspace,
                tilingData.tiling);
        op.Process();
    } else {
        SGEMMCExpand<half> op(&pipe);
        op.Init(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, loraRanks, loraRanksSize, sliceOffsets,
                sliceOffsetsSize, yIn, yOut, batchSize, numBlocksPerCore, maxLoRARank, outputFullDim, workspace,
                tilingData.tiling);
        op.Process();
    }
}

#endif  // SGL_KERNEL_NPU_KERNEL_SGEMMC_EXPAND_H
