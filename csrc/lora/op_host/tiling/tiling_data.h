// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGEMMC_TILING_DATA_H
#define SGEMMC_TILING_DATA_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SGEMMCTilingData)
TILING_DATA_FIELD_DEF(int32_t, dataType);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulCustom, SGEMMCTilingData)

/**
 * @brief Generate matmul tiling.
 * @param blockDim: Number of cores involved in the computation.
 * @param cubeTiling: TCubeTiling structure.
 * @param caseParams: Testcase parameters.
 * @param tilingBuf: Data buffer.
 */
void GenerateTiling(uint32_t blockDim, optiling::SGEMMCTilingData *sgemmcTilingData, const TestcaseParams &caseParams,
                    uint8_t *tilingBuffer);

}  // namespace optiling

#endif  // SGEMMC_TILING_DATA_H

namespace sglang {
namespace npu_kernel {

struct SgemmcTilingData {
    int32_t batch;
    int32_t batch;

    TCubeTiling tiling;
    int32_t hidden_size;

    int32_t max_lora_rank;
    int32_t used_loras;
};

TCubeTiling GenerateTiling(const SgemmcTilingData &caseParams);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGEMMC_TILING_H
