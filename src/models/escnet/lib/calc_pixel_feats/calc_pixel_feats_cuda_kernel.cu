#include <torch/extension.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../utilities.cuh"
#include "../constants.h"

using torch::Tensor;
using std::vector;


template <typename scalar_t>
__global__ void CalcPixelFeats_CUDA_Forward_Kernel(
    const scalar_t* __restrict__ spixel_feats, 
    const scalar_t* __restrict__ assoc_map,
    const int* __restrict__ index_map,
    scalar_t* __restrict__ pixel_feats,
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
            &pixel_feats[(nBaseInd + c) * n_spatial_dim + nCurrInd], 
            spixel_feats[(nBaseInd + c) * n_spixels + nCurrAbsIdx] * q
        );
    }
    AtomicAdd(&weights[nCurrBatch * n_spatial_dim + nCurrInd], q);
}


template <typename scalar_t>
__global__ void CalcAvgPixelFeats_Kernel(
    scalar_t* __restrict__ pixel_feats,
    const scalar_t* __restrict__ weights,
    const int n_spatial_dim,
    const int n_channels
)
{
    const int nCurrInd = blockIdx.x * blockDim.x + threadIdx.x;
    if (nCurrInd >= n_spatial_dim)
        return;
    const int nCurrBatch = blockIdx.y;
    const int nBaseInd = nCurrBatch*n_channels;

    auto weight = weights[nCurrBatch*n_spatial_dim+nCurrInd];
    if (weight < FLT_MIN)
    {
        for (auto c = 0; c < n_channels; c++)
            pixel_feats[(nBaseInd+c)*n_spatial_dim+nCurrInd] = 0;
    }
    else
    {
        for (auto c = 0; c < n_channels; c++)
            pixel_feats[(nBaseInd+c)*n_spatial_dim+nCurrInd] /= weight;
    }
}


template <typename scalar_t>
__global__ void CalcPixelFeats_CUDA_Backward_Kernel(
    const scalar_t* __restrict__ grad_output, 
    const scalar_t* __restrict__ spixel_feats,
    const scalar_t* __restrict__ assoc_map,
    const int* __restrict__ index_map,
    const scalar_t* __restrict__ pixel_feats,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ grad_spixel_feats,
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
    const scalar_t w = weights[nCurrBatch * n_spatial_dim + nCurrInd];

    if (w < FLT_MIN)
        return;

    scalar_t sum = 0.0;
    for (int c = 0; c < n_channels; c++)
    {
        int pixel_offset = (nBaseInd + c) * n_spatial_dim + nCurrInd;
        int spixel_offset = (nBaseInd + c) * n_spixels + nCurrAbsIdx;
        scalar_t go = grad_output[pixel_offset];
        AtomicAdd(
            &grad_spixel_feats[spixel_offset],
            go * q / w
        );
        sum += go * (spixel_feats[spixel_offset] - pixel_feats[pixel_offset]) / w;
    }
    AtomicAdd(&grad_assoc[assoc_offset], sum);
}


vector<Tensor> CalcPixelFeats_CUDA_Forward(Tensor spixel_feats, Tensor assoc_map, Tensor index_map, int nw_spixels, int nh_spixels)
{
    const auto nBatchSize = spixel_feats.size(0);
    const auto nChannels = spixel_feats.size(1);
    const auto nHeight = index_map.size(2);
    const auto nWidth = index_map.size(3);

    const int nSpixels = nw_spixels * nh_spixels;

    Tensor tPixelFeats = torch::zeros({nBatchSize, nChannels, nHeight, nWidth}).type_as(spixel_feats);
    Tensor tWeights = torch::zeros({nBatchSize, 1, nHeight, nWidth}).type_as(tPixelFeats);

    const int nThreads = 1024;
    const int nSpatialDim = nHeight * nWidth;
    
    const dim3 nBlocks((nSpatialDim + nThreads - 1) / nThreads, nBatchSize, 9);

    AT_DISPATCH_FLOATING_TYPES(spixel_feats.type(), "calc_pixel_feats forward", 
    [&] {
    CalcPixelFeats_CUDA_Forward_Kernel<scalar_t><<<nBlocks, nThreads>>>(
        spixel_feats.data<scalar_t>(),
        assoc_map.data<scalar_t>(),
        index_map.data<int>(),
        tPixelFeats.data<scalar_t>(),
        tWeights.data<scalar_t>(),
        nw_spixels,
        nh_spixels,
        nSpixels,
        nSpatialDim,
        nChannels);
    });

    const dim3 nBlocks2((nSpatialDim + nThreads - 1) / nThreads, nBatchSize);
    AT_DISPATCH_FLOATING_TYPES(spixel_feats.type(), "calc_avg_pixel_feats", 
    [&] {
    CalcAvgPixelFeats_Kernel<scalar_t><<<nBlocks2, nThreads>>>(
        tPixelFeats.data<scalar_t>(),
        tWeights.data<scalar_t>(),
        nSpatialDim,
        nChannels);
    }); 
    return {tPixelFeats, tWeights};
}


vector<Tensor> CalcPixelFeats_CUDA_Backward(
    Tensor grad_output, Tensor spixel_feats, Tensor assoc_map, Tensor index_map, 
    Tensor pixel_feats, Tensor weights, 
    int nw_spixels, int nh_spixels
)
{
    const auto nBatchSize = pixel_feats.size(0);
    const auto nChannels = pixel_feats.size(1);
    const auto nHeight = pixel_feats.size(2);
    const auto nWidth = pixel_feats.size(3);

    Tensor tGradSpixelFeats = torch::zeros_like(spixel_feats);
    Tensor tGradAssoc = torch::zeros_like(assoc_map);

    const int nThreads = 1024;
    const int nSpatialDim = nHeight * nWidth;
    
    const dim3 nBlocks((nSpatialDim + nThreads - 1) / nThreads, nBatchSize, 9);


    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "calc_spixel_feats backward", 
    [&] {
    CalcPixelFeats_CUDA_Backward_Kernel<scalar_t><<<nBlocks, nThreads>>>(
        grad_output.data<scalar_t>(),
        spixel_feats.data<scalar_t>(),
        assoc_map.data<scalar_t>(),
        index_map.data<int>(),
        pixel_feats.data<scalar_t>(),
        weights.data<scalar_t>(),
        tGradSpixelFeats.data<scalar_t>(),
        tGradAssoc.data<scalar_t>(),
        nw_spixels,
        nh_spixels,
        nw_spixels * nh_spixels,
        nSpatialDim,
        nChannels);
    });

    return {tGradSpixelFeats, tGradAssoc};
}