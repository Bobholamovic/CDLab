#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../utilities.cuh"

using torch::Tensor;


template <typename scalar_t>
__global__ void Smear_CUDA_Forward_Kernel(
    const scalar_t* __restrict__ spixel_feats, 
    const int* __restrict__ index_map,
    scalar_t* __restrict__ pixel_feats,
    const int n_spixels,
    const int n_spatial_dim,
    const int n_channels
)
{
    const int nCurrInd= blockIdx.x * blockDim.x + threadIdx.x;  // ind=i*w+j

    const int nCurrBatch = blockIdx.y;

    const int nBaseInd = nCurrBatch*n_channels;
    const int nCopyInd = static_cast<int>(index_map[nCurrBatch * n_spatial_dim + nCurrInd]);
    
    if (nCurrInd >= n_spatial_dim)
        return;

    if (nCopyInd >= n_spixels || nCopyInd < 0)
        return; // Ignore all invalid cases, could be unsafe here

    for (int c = 0; c < n_channels; c++)
    {
        pixel_feats[(nBaseInd + c) * n_spatial_dim + nCurrInd] = spixel_feats[(nBaseInd + c) * n_spixels + nCopyInd];
    }
}


template <typename scalar_t>
__global__ void Smear_CUDA_Backward_Kernel(
    const scalar_t* __restrict__ grad_output, 
    const int* __restrict__ index_map,
    scalar_t* __restrict__ grad_spixel_feats,
    const int n_spixels,
    const int n_spatial_dim,
    const int n_channels
)
{
    const int nCurrInd= blockIdx.x * blockDim.x + threadIdx.x;  // ind=i*w+j

    const int nCurrBatch = blockIdx.y;

    const int nBaseInd = nCurrBatch*n_channels;
    const int nCopyInd = static_cast<int>(index_map[nCurrBatch * n_spatial_dim + nCurrInd]);
    
    if (nCurrInd >= n_spatial_dim)
        return;

    if (nCopyInd >= n_spixels || nCopyInd < 0)
        return; // Ignore all invalid cases, could be unsafe here

    for (int c = 0; c < n_channels; c++)
    {
        AtomicAdd(
            &grad_spixel_feats[(nBaseInd + c) * n_spixels + nCopyInd], 
            grad_output[(nBaseInd + c) * n_spatial_dim + nCurrInd]
        );
    }
}


Tensor Smear_CUDA_Forward(Tensor spixel_feats, Tensor index_map, int n_spixels)
{
    const auto nBatchSize = spixel_feats.size(0);
    const auto nChannels = spixel_feats.size(1);
    const auto nHeight = index_map.size(2);
    const auto nWidth = index_map.size(3);

    Tensor tPixelFeats = torch::zeros({nBatchSize, nChannels, nHeight, nWidth}).type_as(spixel_feats);

    const int nThreads = 1024;
    const int nSpatialDim = nHeight * nWidth;
    
    const dim3 nBlocks((nSpatialDim + nThreads - 1) / nThreads, nBatchSize);

    AT_DISPATCH_FLOATING_TYPES(spixel_feats.type(), "smear forward", 
    [&] {
    Smear_CUDA_Forward_Kernel<scalar_t><<<nBlocks, nThreads>>>(
        spixel_feats.data<scalar_t>(),
        index_map.data<int>(),
        tPixelFeats.data<scalar_t>(),
        n_spixels,
        nSpatialDim,
        nChannels);
    });

    return tPixelFeats;
}


Tensor Smear_CUDA_Backward(Tensor grad_output, Tensor index_map, int n_spixels)
{
    const auto nBatchSize = grad_output.size(0);
    const auto nChannels = grad_output.size(1);
    const auto nHeight = index_map.size(2);
    const auto nWidth = index_map.size(3);

    Tensor tGradSpixelFeats = torch::zeros({nBatchSize, nChannels, n_spixels}).type_as(grad_output);

    const int nThreads = 1024;
    const int nSpatialDim = nHeight * nWidth;
    
    const dim3 nBlocks((nSpatialDim + nThreads - 1) / nThreads, nBatchSize);

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "smear backward", 
    [&] {
    Smear_CUDA_Backward_Kernel<scalar_t><<<nBlocks, nThreads>>>(
        grad_output.data<scalar_t>(),
        index_map.data<int>(),
        tGradSpixelFeats.data<scalar_t>(),
        n_spixels,
        nSpatialDim,
        nChannels);
    });

    return tGradSpixelFeats;
}