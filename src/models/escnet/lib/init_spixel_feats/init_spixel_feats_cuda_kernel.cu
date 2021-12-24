#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../utilities.cuh"

using torch::Tensor;


template <typename scalar_t>
__global__ void InitSpixelsFeats_CUDA_Forward_Kernel(
    const scalar_t* __restrict__ pixel_feats, 
    const int* __restrict__ index_map,
    scalar_t* __restrict__ spixel_feats,
    int* __restrict__ counters,
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
            &spixel_feats[(nBaseInd + c) * n_spixels + nCopyInd], 
            pixel_feats[(nBaseInd + c) * n_spatial_dim + nCurrInd]
        );
    }
    
    AtomicAdd(&counters[nCurrBatch * n_spixels + nCopyInd], 1);
}


template <typename scalar_t>
__global__ void CalcAvgSpixelFeats_Kernel(
    scalar_t* __restrict__ spixel_feats,
    const int* __restrict__ counters,
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

    auto counter = counters[nCurrBatch*n_spixels+nCurrSpixel];
    if (counter == 0)
    {
        spixel_feats[(nCurrBatch*n_channels+nCurrChn)*n_spixels+nCurrSpixel] = 0;
    }
    else
    {
        spixel_feats[(nCurrBatch*n_channels+nCurrChn)*n_spixels+nCurrSpixel] /= counter;
    }
}


Tensor InitSpixelsFeats_CUDA_Forward(Tensor pixel_feats, Tensor index_map, int n_spixels)
{
    const auto nBatchSize = pixel_feats.size(0);
    const auto nChannels = pixel_feats.size(1);
    const auto nHeight = pixel_feats.size(2);
    const auto nWidth = pixel_feats.size(3);

    TORCH_CHECK(
        (index_map.size(0) == nBatchSize) & (index_map.size(2) == nHeight) & (index_map.size(3) == nWidth),
        "The size of pixel_feats must match that of index_map at every dimension except dimension 1"
    );

    Tensor tSpixelFeats = torch::zeros({nBatchSize, nChannels, n_spixels}).type_as(pixel_feats);
    Tensor tCounters = torch::zeros({nBatchSize, 1, n_spixels}, torch::dtype(torch::kInt32).device(tSpixelFeats.device()));

    const int nThreads = 1024;
    const int nSpatialDim = nHeight * nWidth;
    
    const dim3 nBlocks((nSpatialDim + nThreads - 1) / nThreads, nBatchSize);

    AT_DISPATCH_FLOATING_TYPES(pixel_feats.type(), "init_spixel_feats forward", 
    [&] {
    InitSpixelsFeats_CUDA_Forward_Kernel<scalar_t><<<nBlocks, nThreads>>>(
        pixel_feats.data<scalar_t>(),
        index_map.data<int>(),
        tSpixelFeats.data<scalar_t>(),
        tCounters.data<int>(),
        n_spixels,
        nSpatialDim,
        nChannels);
    });

    const dim3 nBlocks2((n_spixels*nChannels + nThreads - 1) / nThreads, nBatchSize);
    AT_DISPATCH_FLOATING_TYPES(pixel_feats.type(), "calc_avg_spixel_feats", 
    [&] {
    CalcAvgSpixelFeats_Kernel<scalar_t><<<nBlocks2, nThreads>>>(
        tSpixelFeats.data<scalar_t>(),
        tCounters.data<int>(),
        n_spixels,
        nChannels);
    });

    return tSpixelFeats;
}