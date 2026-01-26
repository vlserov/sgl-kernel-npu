// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGEMMC_TILING_H
#define SGEMMC_TILING_H

#include <cstdint>
namespace sglang {
namespace npu_kernel {

struct SgemmcTilingData {
    int32_t batch_size;
    int32_t used_core_num;
};

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGEMMC_TILING_H