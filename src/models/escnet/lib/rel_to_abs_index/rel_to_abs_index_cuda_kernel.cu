#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

using torch::Tensor;


__global__ void RelToAbsIndex_CUDA_Forward_Kernel(
    const int* __restrict__ rel_idx_map, 
    const int* __restrict__ init_idx_map,
    int* __restrict__ abs_idx_map,
    const int nw_spixels,
    const int nh_spixels,
    const int n_spixels,
    const int n_spatial_dim
)
{
    const int nCurrInd = blockIdx.x * blockDim.x + threadIdx.x;  // ind=i*w+j
    
    if (nCurrInd >= n_spatial_dim)
        return;

    const int nOffset = blockIdx.y * n_spatial_dim + nCurrInd;
    const int nCenAbsIdx = static_cast<int>(init_idx_map[nOffset]);
    const int nCurrRelIdx = static_cast<int>(rel_idx_map[nOffset]);

    if (nCenAbsIdx >= n_spixels || nCenAbsIdx < 0)
        return;

    // Convert absolute index to relative index
    const int curr_rel_i = nCurrRelIdx / 3 - 1;
    const int curr_rel_j = nCurrRelIdx % 3 - 1;
    const int curr_abs_i = nCenAbsIdx / nw_spixels + curr_rel_i;
    const int curr_abs_j = nCenAbsIdx % nw_spixels + curr_rel_j;

    const int nCurrAbsIdx = curr_abs_i * nw_spixels + curr_abs_j;

    if ((curr_abs_i < 0) || (curr_abs_i >= nh_spixels) || (curr_abs_j < 0) || (curr_abs_j >= nw_spixels))
        abs_idx_map[nOffset] = nCenAbsIdx;
    else
        abs_idx_map[nOffset] = nCurrAbsIdx;
}


Tensor RelToAbsIndex_CUDA_Forward(Tensor rel_idx_map, Tensor init_idx_map, int nw_spixels, int nh_spixels)
{
    const auto nBatchSize = rel_idx_map.size(0);
    const auto nHeight = rel_idx_map.size(2);
    const auto nWidth = rel_idx_map.size(3);

    TORCH_CHECK(
        (init_idx_map.size(0) == nBatchSize) & (init_idx_map.size(2) == nHeight) & (init_idx_map.size(3) == nWidth),
        "The size of rel_idx_map must match that of init_idx_map at every dimension except dimension 1"
    );

    Tensor tAbsIdxMap = torch::zeros_like(rel_idx_map);

    const int nThreads = 1024;
    const int nSpatialDim = nHeight * nWidth;
    
    const dim3 nBlocks((nSpatialDim + nThreads - 1) / nThreads, nBatchSize);

    RelToAbsIndex_CUDA_Forward_Kernel<<<nBlocks, nThreads>>>(
        rel_idx_map.data<int>(),
        init_idx_map.data<int>(),
        tAbsIdxMap.data<int>(),
        nw_spixels,
        nh_spixels,
        nw_spixels*nh_spixels,
        nSpatialDim
    );

    return tAbsIdxMap;
}