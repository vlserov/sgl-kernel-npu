#ifndef SGL_KERNEL_NPU_KERNEL_SGEMMC_UTILS_H
#define SGL_KERNEL_NPU_KERNEL_SGEMMC_UTILS_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

struct MNConfig {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t wBaseOffset = 0;
    uint32_t mAxisBaseOffset = 0;
    uint32_t nAxisBaseOffset = 0;
    uint32_t xBaseOffset = 0;
    uint32_t yBaseOffset = 0;
    uint32_t workSpaceOffset = 0;
};

#endif  // SGL_KERNEL_NPU_KERNEL_SGEMMC_UTILS_H
