#include <torch/extension.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../utilities.cuh"
#include "../constants.h"

using torch::Tensor;
using std::vector;


template <typename scalar_t>
__global__ void CalcSpixelFeats_CUDA_Forward_Kernel(
    const scalar_t* __restrict__ pixel_feats, 
    const scalar_t* __restrict__ assoc_map,
    const int* __restrict__ index_map,
    scalar_t* __restrict__ spixel_feats,
    scalar_t* __restrict__ weights,
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

    const scalar_t q = assoc_map[(nCurrBatch*9+nCurrRelIdx) * n_spatial_dim + nCurrInd];
    for (int c = 0; c < n_channels; c++)
    {
        AtomicAdd(
            &spixel_feats[(nBaseInd + c) * n_spixels + nCurrAbsIdx], 
            pixel_feats[(nBaseInd + c) * n_spatial_dim + nCurrInd] * q
        );
    }
    AtomicAdd(&weights[nCurrBatch * n_spixels + nCurrAbsIdx], q);
}


template <typename scalar_t>
__global__ void CalcAvgSpixelFeats_Kernel(
    scalar_t* __restrict__ spixel_feats,
    const scalar_t* __restrict__ weights,
    const int n_spixels,
    const int n_channels
)
{
    const int nCurrInd = blockIdx.x * blockDim.x + threadIdx.x;
    if (nCurrInd >= n_spixels*n_channels)
        return;
    const int nCurrSpixel = nCurrInd%n_spixels;
    const int nCurrChn = nCurrInd/n_spixels;
    const int nCurrBatch = blockIdx.y;

    auto weight = weights[nCurrBatch*n_spixels+nCurrSpixel];
    if (weight < FLT_MIN)
    {
        spixel_feats[(nCurrBatch*n_channels+nCurrChn)*n_spixels+nCurrSpixel] = 0;
    }
    else
    {
        spixel_feats[(nCurrBatch*n_channels+nCurrChn)*n_spixels+nCurrSpixel] /= weight;
    }
}


template <typename scalar_t>
__global__ void CalcSpixelFeats_CUDA_Backward_Kernel(
    const scalar_t* __restrict__ grad_output, 
    const scalar_t* __restrict__ pixel_feats,
    const scalar_t* __restrict__ assoc_map,
    const int* __restrict__ index_map,
    const scalar_t* __restrict__ spixel_feats,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ grad_pixel_feats,
    scalar_t* __restrict__ grad_assoc,
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
    
    const int assoc_offset = (nCurrBatch*9+nCurrRelIdx) * n_spatial_dim + nCurrInd;
    const scalar_t q = assoc_map[assoc_offset];
    const scalar_t w = weights[nCurrBatch * n_spixels + nCurrAbsIdx];

    if (w < FLT_MIN)
        return;

    scalar_t sum = 0.0;
    for (int c = 0; c < n_channels; c++)
    {
        int pixel_offset = (nBaseInd + c) * n_spatial_dim + nCurrInd;
        int spixel_offset = (nBaseInd + c) * n_spixels + nCurrAbsIdx;
        scalar_t go = grad_output[spixel_offset];
        AtomicAdd(
            &grad_pixel_feats[pixel_offset],
            go * q / w
        );
        sum += go * (pixel_feats[pixel_offset] - spixel_feats[spixel_offset]) / w;
    }
    AtomicAdd(&grad_assoc[assoc_offset], sum);
}


vector<Tensor> CalcSpixelFeats_CUDA_Forward(Tensor pixel_feats, Tensor assoc_map, Tensor index_map, int nw_spixels, int nh_spixels)
{
    const auto nBatchSize = pixel_feats.size(0);
    const auto nChannels = pixel_feats.size(1);
    const auto nHeight = pixel_feats.size(2);
    const auto nWidth = pixel_feats.size(3);

    const int nSpixels = nw_spixels * nh_spixels;

    Tensor tSpixelFeats = torch::zeros({nBatchSize, nChannels, nSpixels}).type_as(pixel_feats);
    Tensor tWeights = torch::zeros({nBatchSize, 1, nSpixels}).type_as(tSpixelFeats);

    const int nThreads = 1024;
    const int nSpatialDim = nHeight * nWidth;
    
    const dim3 nBlocks((nSpatialDim + nThreads - 1) / nThreads, nBatchSize, 9);

    AT_DISPATCH_FLOATING_TYPES(pixel_feats.type(), "calc_spixel_feats forward", 
    [&] {
    CalcSpixelFeats_CUDA_Forward_Kernel<scalar_t><<<nBlocks, nThreads>>>(
        pixel_feats.data<scalar_t>(),
        assoc_map.data<scalar_t>(),
        index_map.data<int>(),
        tSpixelFeats.data<scalar_t>(),
        tWeights.data<scalar_t>(),
        nw_spixels,
        nh_spixels,
        nSpixels,
        nSpatialDim,
        nChannels);
    });

    const dim3 nBlocks2((nSpixels*nChannels + nThreads - 1) / nThreads, nBatchSize);
    AT_DISPATCH_FLOATING_TYPES(pixel_feats.type(), "calc_avg_spixel_feats", 
    [&] {
    CalcAvgSpixelFeats_Kernel<scalar_t><<<nBlocks2, nThreads>>>(
        tSpixelFeats.data<scalar_t>(),
        tWeights.data<scalar_t>(),
        nSpixels,
        nChannels);
    }); 

    return {tSpixelFeats, tWeights};
}


vector<Tensor> CalcSpixelFeats_CUDA_Backward(
    Tensor grad_output, Tensor pixel_feats, Tensor assoc_map, Tensor index_map, 
    Tensor spixel_feats, Tensor weights, 
    int nw_spixels, int nh_spixels
)
{
    const auto nBatchSize = pixel_feats.size(0);
    const auto nChannels = pixel_feats.size(1);
    const auto nHeight = pixel_feats.size(2);
    const auto nWidth = pixel_feats.size(3);

    Tensor tGradPixelFeats = torch::zeros_like(pixel_feats);
    Tensor tGradAssoc = torch::zeros_like(assoc_map);

    const int nThreads = 1024;
    const int nSpatialDim = nHeight * nWidth;
    
    const dim3 nBlocks((nSpatialDim + nThreads - 1) / nThreads, nBatchSize, 9);


    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "calc_spixel_feats backward", 
    [&] {
    CalcSpixelFeats_CUDA_Backward_Kernel<scalar_t><<<nBlocks, nThreads>>>(
        grad_output.data<scalar_t>(),
        pixel_feats.data<scalar_t>(),
        assoc_map.data<scalar_t>(),
        index_map.data<int>(),
        spixel_feats.data<scalar_t>(),
        weights.data<scalar_t>(),
        tGradPixelFeats.data<scalar_t>(),
        tGradAssoc.data<scalar_t>(),
        nw_spixels,
        nh_spixels,
        nw_spixels * nh_spixels,
        nSpatialDim,
        nChannels);
    });

    return {tGradPixelFeats, tGradAssoc};
}