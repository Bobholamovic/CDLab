#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using torch::Tensor;


// CUDA forward declaration
Tensor RelToAbsIndex_CUDA_Forward(Tensor, Tensor, int, int);


// CPU forward
Tensor RelToAbsIndex_CPU_Forward(Tensor rel_idx_map, Tensor init_idx_map, int nw_spixels, int nh_spixels)
{
    const auto nBatchSize = rel_idx_map.size(0);
    const auto nHeight = rel_idx_map.size(2);
    const auto nWidth = rel_idx_map.size(3);

    const auto nSpixels = nw_spixels * nh_spixels;

    // Check size of index_map
    TORCH_CHECK(
        (init_idx_map.size(0) == nBatchSize) & (init_idx_map.size(2) == nHeight) & (init_idx_map.size(3) == nWidth),
        "The size of rel_idx_map must match that of init_idx_map at every dimension except dimension 1"
    );

    Tensor tAbsIdxMap = torch::zeros({nBatchSize, 1, nHeight, nWidth}).type_as(rel_idx_map);

    // Accessors
    auto aAbsIdxMap = tAbsIdxMap.accessor<int, 4>();
    auto aRelIdxMap = rel_idx_map.accessor<int, 4>();
    auto aInitIdxMap = init_idx_map.accessor<int, 4>();

    for (auto k = 0; k < nBatchSize; k++)
    {
        auto aAbsIdxMap_ = aAbsIdxMap[k][0];
        auto aRelIdxMap_ = aRelIdxMap[k][0];
        auto aInitIdxMap_ = aInitIdxMap[k][0];
        for (auto i = 0; i < nHeight; i++)
        {
            for (auto j = 0; j < nWidth; j++)
            {
                auto nCenAbsIdx = aInitIdxMap_[i][j];
                auto nCurrRelIdx = aRelIdxMap_[i][j];
  
                if (nCenAbsIdx >= nSpixels || nCenAbsIdx < 0)
                    continue;

                // Convert abs_idx to rel_idx
                auto curr_rel_i = nCurrRelIdx / 3 - 1;
                auto curr_rel_j = nCurrRelIdx % 3 - 1;
                auto curr_abs_i = nCenAbsIdx / nw_spixels + curr_rel_i;
                auto curr_abs_j = nCenAbsIdx % nw_spixels + curr_rel_j;

                if ((curr_abs_i < 0) || (curr_abs_i >= nh_spixels) || (curr_abs_j < 0) || (curr_abs_j >= nw_spixels))
                    aAbsIdxMap_[i][j] = nCenAbsIdx;
                else
                    aAbsIdxMap_[i][j] = curr_abs_i * nw_spixels + curr_abs_j;
            }
        }
    }
    
    return tAbsIdxMap;
}


// C++ interface
Tensor RelToAbsIndex_Forward(Tensor rel_idx_map, Tensor init_idx_map, int nw_spixels, int nh_spixels)
{
    // Check size of index maps
    TORCH_CHECK(rel_idx_map.dim() == 4, "expected 4 dims but tensor has ", rel_idx_map.dim());
    TORCH_CHECK(rel_idx_map.size(1) == 1, "rel_idx_map must have size 1 at dimension 1");
    TORCH_CHECK(init_idx_map.dim() == 4, "expected 4 dims but tensor has ", init_idx_map.dim());
    TORCH_CHECK(init_idx_map.size(1) == 1, "init_idx_map must have size 1 at dimension 1");

    if (rel_idx_map.type().is_cuda() && init_idx_map.type().is_cuda())
    {
        CHECK_CONTIGUOUS(rel_idx_map); CHECK_CONTIGUOUS(init_idx_map);
        return RelToAbsIndex_CUDA_Forward(rel_idx_map, init_idx_map, nw_spixels, nh_spixels);
    }
    else
    {
        return RelToAbsIndex_CPU_Forward(rel_idx_map, init_idx_map, nw_spixels, nh_spixels);
    }
}


Tensor RelToAbsIndex_Backward()
{
    // Not implemented yet
    AT_ERROR("the backward function is not yet implemented");
    return {};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &RelToAbsIndex_Forward, "rel_to_abs_index forward");
    m.def("backward", &RelToAbsIndex_Backward, "rel_to_abs_index backward");
}