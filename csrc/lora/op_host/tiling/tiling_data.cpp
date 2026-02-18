#include <map>
#include "tiling_data.h"
#include "common.h"
#include "common_tiling.h"

namespace sglang {
namespace npu_kernel {

SGEMMCTilingData GenerateTiling(int32_t batch, int32_t n, int32_t k, const MatMul::DataType type)
{
    SGEMMCTilingData tilingData;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(*ascendcPlatform);

    uint32_t M = caseParams.m;
    uint32_t N = caseParams.n;
    uint32_t K = caseParams.k;
    uint32_t blockDim = caseParams.usedCoreNum;

    cubeTiling.SetDim(batch);
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::VECTOR, type, false);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, type, true);
    cubeTiling.SetCType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND, type);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, type);

    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.EnableBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    if (cubeTiling.GetTiling(tilingData) == -1) {
        std::cout << "Generate tiling failed." << std::endl;
        return {};
    }
    return tilingData;
}

}  // namespace npu_kernel
}  // namespace sglang
