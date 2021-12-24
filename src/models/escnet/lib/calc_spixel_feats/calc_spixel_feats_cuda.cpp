#include <torch/extension.h>
#include <vector>

#include "../constants.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using torch::Tensor;
using std::vector;


// CUDA declarations
vector<Tensor> CalcSpixelFeats_CUDA_Forward(Tensor, Tensor, Tensor, int, int);
vector<Tensor> CalcSpixelFeats_CUDA_Backward(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int);


// CPU declarations
template <typename scalar_t>
vector<Tensor> CalcSpixelFeats_CPU_Forward(Tensor pixel_feats, Tensor assoc_map, Tensor index_map, int nw_spixels, int nh_spixels)
{
    const auto nBatchSize = pixel_feats.size(0);
    const auto nChannels = pixel_feats.size(1);
    const auto nHeight = pixel_feats.size(2);
    const auto nWidth = pixel_feats.size(3);

    const int nSpixels = nw_spixels * nh_spixels;

    Tensor tSpixelFeats = torch::zeros({nBatchSize, nChannels, nSpixels}).type_as(pixel_feats);
    Tensor tWeights = torch::zeros({nBatchSize, 1, nSpixels}).type_as(tSpixelFeats);

    // Accessors
    auto aIndexMap = index_map.accessor<int, 4>();
    auto aPixelFeats = pixel_feats.accessor<scalar_t, 4>();
    auto aAssocMap = assoc_map.accessor<scalar_t, 4>();
    auto aSpixelFeats = tSpixelFeats.accessor<scalar_t, 3>();
    auto aWeights = tWeights.accessor<scalar_t, 3>();

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
                        aSpixelFeats_[c][nCurrAbsIdx] += aPixelFeats_[c][i][j] * q;
                    }
                    aWeights_[nCurrAbsIdx] += q;
                }
            }
        }
    }

    for (auto k = 0; k < nBatchSize; k++)
    {
        for (auto s = 0; s < nSpixels; s++)
        {
            auto weight = aWeights[k][0][s];
            if (weight < FLT_MIN)
            {
                for (auto c = 0; c < nChannels; c++)
                {
                    aSpixelFeats[k][c][s] = 0;
                }
            }
            else
            {
                for (auto c = 0; c < nChannels; c++)
                {
                    aSpixelFeats[k][c][s] /= weight;
                }     
            }
        }
    }

    return {tSpixelFeats, tWeights};
}


template <typename scalar_t>
vector<Tensor> CalcSpixelFeats_CPU_Backward(
    Tensor grad_output, Tensor pixel_feats, Tensor assoc_map, Tensor index_map, 
    Tensor spixel_feats, Tensor weights, 
    int nw_spixels, int nh_spixels
)
{
    const auto nBatchSize = pixel_feats.size(0);
    const auto nChannels = pixel_feats.size(1);
    const auto nHeight = pixel_feats.size(2);
    const auto nWidth = pixel_feats.size(3);

    const int nSpixels = nw_spixels * nh_spixels;

    Tensor tGradPixelFeats = torch::zeros_like(pixel_feats);
    Tensor tGradAssoc = torch::zeros_like(assoc_map);

    // Accessors
    auto aIndexMap = index_map.accessor<int, 4>();
    auto aPixelFeats = pixel_feats.accessor<scalar_t, 4>();
    auto aAssocMap = assoc_map.accessor<scalar_t, 4>();
    auto aSpixelFeats = spixel_feats.accessor<scalar_t, 3>();
    auto aWeights = weights.accessor<scalar_t, 3>();
    auto aGradPixelFeats = tGradPixelFeats.accessor<scalar_t, 4>();
    auto aGradAssoc = tGradAssoc.accessor<scalar_t, 4>();
    auto aGradOutput = grad_output.accessor<scalar_t, 3>();

    for (auto k = 0; k < nBatchSize; k++)
    {
        auto aIndexMap_ = aIndexMap[k][0];
        auto aSpixelFeats_ = aSpixelFeats[k];
        auto aPixelFeats_ = aPixelFeats[k];
        auto aWeights_ = aWeights[k][0];
        auto aAssocMap_ = aAssocMap[k];
        auto aGradPixelFeats_ = aGradPixelFeats[k];
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

                    auto w = aWeights_[nCurrAbsIdx];

                    if (w < FLT_MIN)
                        continue;

                    for (auto c = 0; c < nChannels; c++)
                    {
                        aGradPixelFeats_[c][i][j] += aGradOutput_[c][nCurrAbsIdx] * aAssocMap_[nCurrRelIdx][i][j] / w;
                        aGradAssoc_[nCurrRelIdx][i][j] += aGradOutput_[c][nCurrAbsIdx] * \
                            (aPixelFeats_[c][i][j] - aSpixelFeats_[c][nCurrAbsIdx]) / w;
                    }
                }
            }
        }
    }

    return {tGradPixelFeats, tGradAssoc};
}


// C++ interface
vector<Tensor> CalcSpixelFeats_Forward(
    Tensor pixel_feats, Tensor assoc_map, Tensor index_map, int nw_spixels, int nh_spixels
)
{
    // Check size of index_map
    TORCH_CHECK(index_map.dim() == 4, "expected 4 dims but tensor has ", index_map.dim());
    TORCH_CHECK(index_map.size(1) == 1, "index_map must have size 1 at dimension 1");
    // Check assoc_map
    TORCH_CHECK(assoc_map.size(1) == 9, "assoc_map must have size 9 at dimension 1");

    if (pixel_feats.type().is_cuda() && assoc_map.type().is_cuda() && index_map.type().is_cuda())
    {
        CHECK_CONTIGUOUS(pixel_feats); CHECK_CONTIGUOUS(assoc_map); CHECK_CONTIGUOUS(index_map);
        return CalcSpixelFeats_CUDA_Forward(pixel_feats, assoc_map, index_map, nw_spixels, nh_spixels);
    }
    else
    {
        return AT_DISPATCH_FLOATING_TYPES(pixel_feats.type(), "calc_spixel_feats forward", 
        [&] {
            return CalcSpixelFeats_CPU_Forward<scalar_t>(pixel_feats, assoc_map, index_map, nw_spixels, nh_spixels);
        });
    }
}


vector<Tensor> CalcSpixelFeats_Backward(
    Tensor grad_output, Tensor pixel_feats, Tensor assoc_map, Tensor index_map, 
    Tensor spixel_feats, Tensor weights, 
    int nw_spixels, int nh_spixels
)
{
    // Check size of index_map
    TORCH_CHECK(index_map.dim() == 4, "expected 4 dims but tensor has ", index_map.dim());
    TORCH_CHECK(index_map.size(1) == 1, "index_map must have size 1 at dimension 1");
    // Check assoc_map
    TORCH_CHECK(assoc_map.size(1) == 9, "assoc_map must have size 9 at dimension 1");
    
    if (
        grad_output.type().is_cuda() && pixel_feats.type().is_cuda() && 
        assoc_map.type().is_cuda() && index_map.type().is_cuda() &&
        spixel_feats.type().is_cuda() && weights.type().is_cuda()
    )
    {
        CHECK_CONTIGUOUS(grad_output); CHECK_CONTIGUOUS(pixel_feats);
        CHECK_CONTIGUOUS(assoc_map); CHECK_CONTIGUOUS(index_map);
        CHECK_CONTIGUOUS(spixel_feats); CHECK_CONTIGUOUS(weights);
        return CalcSpixelFeats_CUDA_Backward(grad_output, pixel_feats, assoc_map, index_map, spixel_feats, weights, nw_spixels, nh_spixels);
    }
    else
    {
        return AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "calc_spixel_feats backward", 
        [&] {
            return CalcSpixelFeats_CPU_Backward<scalar_t>(grad_output, pixel_feats, assoc_map, index_map, spixel_feats, weights, nw_spixels, nh_spixels);
        });
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &CalcSpixelFeats_Forward, "calc_spixel_feats forward");
    m.def("backward", &CalcSpixelFeats_Backward, "calc_spixel_feats backward");
}