#include <torch/extension.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../utilities.cuh"

using torch::Tensor;
using std::vector;


template <typename scalar_t>
__global__ void CalcAssoc_CUDA_Forward_Kernel(
    const scalar_t* __restrict__ pixel_feats, 
    const scalar_t* __restrict__ spixel_feats,
    const int* __restrict__ index_map,
    scalar_t* assoc_map,
    const int nw_spixels,
    const int nh_spixels,
    const int n_spixels,
    const int n_spatial_dim,
    const int n_channels
)
{
    const int nCurrInd= blockIdx.x * blockDim.x + threadIdx.x;  // ind=i*w+j

    const int nCurrBatch = blockIdx.y;

    const int nCurrRelIdx = blockIdx.z;

    const int nBaseInd = nCurrBatch*n_channels;
    const int nCenAbsIdx = static_cast<int>(index_map[nCurrBatch * n_spatial_dim + nCurrInd]);

    scalar_t* const pq = &assoc_map[(nCurrBatch*9+nCurrRelIdx) * n_spatial_dim + nCurrInd];
    
    if (nCurrInd >= n_spatial_dim)
        return;

    if (nCenAbsIdx >= n_spixels || nCenAbsIdx < 0)
        return;

    // Convert absolute index to relative index
    const int curr_rel_i = nCurrRelIdx / 3 - 1;
    const int curr_rel_j = nCurrRelIdx % 3 - 1;
    const int curr_abs_i = nCenAbsIdx / nw_spixels + curr_rel_i;
    const int curr_abs_j = nCenAbsIdx % nw_spixels + curr_rel_j;

    if ((curr_abs_i < 0) || (curr_abs_i >= nh_spixels) || (curr_abs_j < 0) || (curr_abs_j >= nw_spixels))
    {
        *pq = -10000.0;
        return;
    }
    

    const int nCurrAbsIdx = curr_abs_i * nw_spixels + curr_abs_j;

    scalar_t sum = 0.0, diff = 0.0;
    for (int c = 0; c < n_channels; c++)
    {
        diff = pixel_feats[(nBaseInd + c) * n_spatial_dim + nCurrInd] - \
                        spixel_feats[(nBaseInd + c) * n_spixels + nCurrAbsIdx];
        sum -= diff * diff;
    }
    *pq = sum;
}


template <typename scalar_t>
__global__ void CalcAssoc_CUDA_Backward_Kernel(
    const scalar_t* __restrict__ grad_output, 
    const scalar_t* __restrict__ pixel_feats,
    const scalar_t* __restrict__ spixel_feats,
    const int* __restrict__ index_map,
    scalar_t* __restrict__ grad_pixel_feats,
    scalar_t* __restrict__ grad_spixel_feats,
    const int nw_spixels,
    const int nh_spixels,
    const int n_spixels,
    const int n_spatial_dim,
    const int n_channels
)
{
    const int nCurrInd= blockIdx.x * blockDim.x + threadIdx.x;  // ind=i*w+j

    const int nCurrBatch = blockIdx.y;

    const int nCurrRelIdx = blockIdx.z;

    const int nBaseInd = nCurrBatch*n_channels;
    const int nCenAbsIdx = static_cast<int>(index_map[nCurrBatch * n_spatial_dim + nCurrInd]);
    
    if (nCurrInd >= n_spatial_dim)
        return;

    if (nCenAbsIdx >= n_spixels || nCenAbsIdx < 0)
        return;
        
    // Convert absolute index to relative index
    const int curr_rel_i = nCurrRelIdx / 3 - 1;
    const int curr_rel_j = nCurrRelIdx % 3 - 1;
    const int curr_abs_i = nCenAbsIdx / nw_spixels + curr_rel_i;
    const int curr_abs_j = nCenAbsIdx % nw_spixels + curr_rel_j;

    if ((curr_abs_i < 0) || (curr_abs_i >= nh_spixels) || (curr_abs_j < 0) || (curr_abs_j >= nw_spixels))
        return;

    const int nCurrAbsIdx = curr_abs_i * nw_spixels + curr_abs_j;
    
    const scalar_t dq = grad_output[(nCurrBatch*9+nCurrRelIdx) * n_spatial_dim + nCurrInd];

    for (int c = 0; c < n_channels; c++)
    {
        int pixel_offset = (nBaseInd + c) * n_spatial_dim + nCurrInd;
        int spixel_offset = (nBaseInd + c) * n_spixels + nCurrAbsIdx;
        scalar_t diff = pixel_feats[pixel_offset] - spixel_feats[spixel_offset];
        AtomicAdd(
            &grad_pixel_feats[pixel_offset],
            -2 * dq * diff
        );
        AtomicAdd(
            &grad_spixel_feats[spixel_offset],
            2 * dq * diff
        );
    }
}


Tensor CalcAssoc_CUDA_Forward(Tensor pixel_feats, Tensor spixel_feats, Tensor index_map, int nw_spixels, int nh_spixels)
{
    const auto nBatchSize = pixel_feats.size(0);
    const auto nChannels = pixel_feats.size(1);
    const auto nHeight = pixel_feats.size(2);
    const auto nWidth = pixel_feats.size(3);

    const int nSpixels = nw_spixels * nh_spixels;

    Tensor tAssocMap = torch::zeros({nBatchSize, 9, nHeight, nWidth}).type_as(pixel_feats);

    const int nThreads = 1024;
    const int nSpatialDim = nHeight * nWidth;
    
    const dim3 nBlocks((nSpatialDim + nThreads - 1) / nThreads, nBatchSize, 9);

    AT_DISPATCH_FLOATING_TYPES(pixel_feats.type(), "calc_assoc forward", 
    [&] {
    CalcAssoc_CUDA_Forward_Kernel<scalar_t><<<nBlocks, nThreads>>>(
        pixel_feats.data<scalar_t>(),
        spixel_feats.data<scalar_t>(),
        index_map.data<int>(),
        tAssocMap.data<scalar_t>(),
        nw_spixels,
        nh_spixels,
        nSpixels,
        nSpatialDim,
        nChannels);
    });

    return tAssocMap;
}


vector<Tensor> CalcAssoc_CUDA_Backward(
    Tensor grad_output, 
    Tensor pixel_feats, Tensor spixel_feats, 
    Tensor index_map, 
    int nw_spixels, int nh_spixels
)
{
    const auto nBatchSize = pixel_feats.size(0);
    const auto nChannels = pixel_feats.size(1);
    const auto nHeight = pixel_feats.size(2);
    const auto nWidth = pixel_feats.size(3);

    Tensor tGradPixelFeats = torch::zeros_like(pixel_feats);
    Tensor tGradSpixelFeats = torch::zeros_like(spixel_feats);

    const int nThreads = 1024;
    const int nSpatialDim = nHeight * nWidth;
    
    const dim3 nBlocks((nSpatialDim + nThreads - 1) / nThreads, nBatchSize, 9);

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "calc_assoc backward", 
    [&] {
    CalcAssoc_CUDA_Backward_Kernel<scalar_t><<<nBlocks, nThreads>>>(
        grad_output.data<scalar_t>(),
        pixel_feats.data<scalar_t>(),
        spixel_feats.data<scalar_t>(),
        index_map.data<int>(),
        tGradPixelFeats.data<scalar_t>(),
        tGradSpixelFeats.data<scalar_t>(),
        nw_spixels,
        nh_spixels,
        nw_spixels * nh_spixels,
        nSpatialDim,
        nChannels);
    });

    return {tGradPixelFeats, tGradSpixelFeats};
}