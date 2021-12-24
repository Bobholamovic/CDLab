#include <torch/extension.h>
#include <vector>

#include "../constants.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using torch::Tensor;
using std::vector;


// CUDA declarations
vector<Tensor> CalcPixelFeats_CUDA_Forward(Tensor, Tensor, Tensor, int, int);
vector<Tensor> CalcPixelFeats_CUDA_Backward(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int);


// CPU declarations
template <typename scalar_t>
vector<Tensor> CalcPixelFeats_CPU_Forward(Tensor spixel_feats, Tensor assoc_map, Tensor index_map, int nw_spixels, int nh_spixels)
{
    const auto nBatchSize = spixel_feats.size(0);
    const auto nChannels = spixel_feats.size(1);
    const auto nHeight = index_map.size(2);
    const auto nWidth = index_map.size(3);

    const auto nSpixels = nw_spixels * nh_spixels;

    Tensor tPixelFeats = torch::zeros({nBatchSize, nChannels, nHeight, nWidth}).type_as(spixel_feats);
    Tensor tWeights = torch::zeros({nBatchSize, 1, nHeight, nWidth}).type_as(tPixelFeats);

    // Accessors
    auto aIndexMap = index_map.accessor<int, 4>();
    auto aPixelFeats = tPixelFeats.accessor<scalar_t, 4>();
    auto aAssocMap = assoc_map.accessor<scalar_t, 4>();
    auto aSpixelFeats = spixel_feats.accessor<scalar_t, 3>();
    auto aWeights = tWeights.accessor<scalar_t, 4>();

    for (auto k = 0; k < nBatchSize; k++)
    {
        auto aIndexMap_ = aIndexMap[k][0];
        auto aSpixelFeats_ = aSpixelFeats[k];
        auto aPixelFeats_ = aPixelFeats[k];
        auto aWeights_ = aWeights[k][0];
        auto aAssocMap_ = aAssocMap[k];
        for (auto i = 0; i < nHeight; i++)
        {
            for (auto j = 0; j < nWidth; j++)
            {
                auto nCenAbsIdx = aIndexMap_[i][j];
                if (nCenAbsIdx >= nSpixels || nCenAbsIdx < 0)
                    continue;
                for (auto nCurrRelIdx = 0; nCurrRelIdx < 9; nCurrRelIdx++)
                {
                    // Convert abs_idx to rel_idx
                    auto curr_rel_i = nCurrRelIdx / 3 - 1;
                    auto curr_rel_j = nCurrRelIdx % 3 - 1;
                    auto curr_abs_i = nCenAbsIdx / nw_spixels + curr_rel_i;
                    auto curr_abs_j = nCenAbsIdx % nw_spixels + curr_rel_j;

                    if ((curr_abs_i < 0) || (curr_abs_i >= nh_spixels) || (curr_abs_j < 0) || (curr_abs_j >= nw_spixels))
                        continue;
                    auto nCurrAbsIdx = curr_abs_i * nw_spixels + curr_abs_j;

                    auto q = aAssocMap_[nCurrRelIdx][i][j];
                    for (auto c = 0; c < nChannels; c++)
                    {
                        aPixelFeats_[c][i][j] += aSpixelFeats_[c][nCurrAbsIdx] * q;
                    }
                    aWeights_[i][j] += q;
                }
            }
        }
    }

    for (auto k = 0; k < nBatchSize; k++)
    {
        for (auto i = 0; i < nHeight; i++)
        {
            for (auto j = 0; j < nWidth; j++)
            {
                auto weight = aWeights[k][0][i][j];
                if (weight < FLT_MIN)
                {
                    for (auto c = 0; c < nChannels; c++)
                    {
                        aPixelFeats[k][c][i][j] = 0;
                    }
                }
                else
                {
                    for (auto c = 0; c < nChannels; c++)
                    {
                        aPixelFeats[k][c][i][j] /= weight;
                    }
                }
            }
        }
    }

    return {tPixelFeats, tWeights};
}


template <typename scalar_t>
vector<Tensor> CalcPixelFeats_CPU_Backward(
    Tensor grad_output, Tensor spixel_feats, Tensor assoc_map, Tensor index_map, 
    Tensor pixel_feats, Tensor weights, 
    int nw_spixels, int nh_spixels
)
{
    const auto nBatchSize = pixel_feats.size(0);
    const auto nChannels = pixel_feats.size(1);
    const auto nHeight = pixel_feats.size(2);
    const auto nWidth = pixel_feats.size(3);

    const auto nSpixels = nw_spixels * nh_spixels;

    Tensor tGradSpixelFeats = torch::zeros_like(spixel_feats);
    Tensor tGradAssoc = torch::zeros_like(assoc_map);

    // Accessors
    auto aIndexMap = index_map.accessor<int, 4>();
    auto aPixelFeats = pixel_feats.accessor<scalar_t, 4>();
    auto aAssocMap = assoc_map.accessor<scalar_t, 4>();
    auto aSpixelFeats = spixel_feats.accessor<scalar_t, 3>();
    auto aWeights = weights.accessor<scalar_t, 4>();
    auto aGradSpixelFeats = tGradSpixelFeats.accessor<scalar_t, 3>();
    auto aGradAssoc = tGradAssoc.accessor<scalar_t, 4>();
    auto aGradOutput = grad_output.accessor<scalar_t, 4>();

    for (auto k = 0; k < nBatchSize; k++)
    {
        auto aIndexMap_ = aIndexMap[k][0];
        auto aSpixelFeats_ = aSpixelFeats[k];
        auto aPixelFeats_ = aPixelFeats[k];
        auto aWeights_ = aWeights[k][0];
        auto aAssocMap_ = aAssocMap[k];
        auto aGradSpixelFeats_ = aGradSpixelFeats[k];
        auto aGradAssoc_ = aGradAssoc[k];
        auto aGradOutput_ = aGradOutput[k];
        for (auto i = 0; i < nHeight; i++)
        {
            for (auto j = 0; j < nWidth; j++)
            {
                auto nCenAbsIdx = aIndexMap_[i][j];
                if (nCenAbsIdx >= nSpixels || nCenAbsIdx < 0)
                    continue;
                for (auto nCurrRelIdx = 0; nCurrRelIdx < 9; nCurrRelIdx++)
                {
                    // Convert abs_idx to rel_idx
                    auto curr_rel_i = nCurrRelIdx / 3 - 1;
                    auto curr_rel_j = nCurrRelIdx % 3 - 1;
                    auto curr_abs_i = nCenAbsIdx / nw_spixels + curr_rel_i;
                    auto curr_abs_j = nCenAbsIdx % nw_spixels + curr_rel_j;

                    if ((curr_abs_i < 0) || (curr_abs_i >= nh_spixels) || (curr_abs_j < 0) || (curr_abs_j >= nw_spixels))
                        continue;

                    auto nCurrAbsIdx = curr_abs_i * nw_spixels + curr_abs_j;

                    auto w = aWeights_[i][j];

                    if (w < FLT_MIN)
                    {
                        continue;
                    }

                    for (auto c = 0; c < nChannels; c++)
                    {
                        aGradSpixelFeats_[c][nCurrAbsIdx] += aGradOutput_[c][i][j] * aAssocMap_[nCurrRelIdx][i][j] / w;
                        aGradAssoc_[nCurrRelIdx][i][j] += aGradOutput_[c][i][j] * \
                            (aSpixelFeats_[c][nCurrAbsIdx] - aPixelFeats_[c][i][j]) / w;
                    }
                }
            }
        }
    }

    return {tGradSpixelFeats, tGradAssoc};
}


// C++ interface
vector<Tensor> CalcPixelFeats_Forward(
    Tensor spixel_feats, Tensor assoc_map, Tensor index_map, int nw_spixels, int nh_spixels
)
{
    // Check size of index_map
    TORCH_CHECK(index_map.dim() == 4, "expected 4 dims but tensor has ", index_map.dim());
    TORCH_CHECK(index_map.size(1) == 1, "index_map must have size 1 at dimension 1");
    // Check assoc_map
    TORCH_CHECK(assoc_map.size(1) == 9, "assoc_map must have size 9 at dimension 1");

    if (spixel_feats.type().is_cuda() && assoc_map.type().is_cuda() && index_map.type().is_cuda())
    {
        CHECK_CONTIGUOUS(spixel_feats); CHECK_CONTIGUOUS(assoc_map); CHECK_CONTIGUOUS(index_map);
        return CalcPixelFeats_CUDA_Forward(spixel_feats, assoc_map, index_map, nw_spixels, nh_spixels);
    }
    else
    {
        return AT_DISPATCH_FLOATING_TYPES(spixel_feats.type(), "calc_pixel_feats forward", 
        [&] {
            return CalcPixelFeats_CPU_Forward<scalar_t>(spixel_feats, assoc_map, index_map, nw_spixels, nh_spixels);
        });
    }
}


vector<Tensor> CalcPixelFeats_Backward(
    Tensor grad_output, Tensor spixel_feats, Tensor assoc_map, Tensor index_map, 
    Tensor pixel_feats, Tensor weights, 
    int nw_spixels, int nh_spixels
)
{
    // Check size of index_map
    TORCH_CHECK(index_map.dim() == 4, "expected 4 dims but tensor has ", index_map.dim());
    TORCH_CHECK(index_map.size(1) == 1, "index_map must have size 1 at dimension 1");
    // Check assoc_map
    TORCH_CHECK(assoc_map.size(1) == 9, "assoc_map must have size 9 at dimension 1");
    
    if (
        grad_output.type().is_cuda() && spixel_feats.type().is_cuda() && 
        assoc_map.type().is_cuda() && index_map.type().is_cuda() &&
        pixel_feats.type().is_cuda() && weights.type().is_cuda()
    )
    {
        CHECK_CONTIGUOUS(grad_output); CHECK_CONTIGUOUS(spixel_feats);
        CHECK_CONTIGUOUS(assoc_map); CHECK_CONTIGUOUS(index_map);
        CHECK_CONTIGUOUS(pixel_feats); CHECK_CONTIGUOUS(weights);
        return CalcPixelFeats_CUDA_Backward(grad_output, spixel_feats, assoc_map, index_map, pixel_feats, weights, nw_spixels, nh_spixels);
    }
    else
    {
        return AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "calc_pixel_feats backward", 
        [&] {
            return CalcPixelFeats_CPU_Backward<scalar_t>(grad_output, spixel_feats, assoc_map, index_map, pixel_feats, weights, nw_spixels, nh_spixels);
        });
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &CalcPixelFeats_Forward, "calc_pixel_feats forward");
    m.def("backward", &CalcPixelFeats_Backward, "calc_pixel_feats backward");
}