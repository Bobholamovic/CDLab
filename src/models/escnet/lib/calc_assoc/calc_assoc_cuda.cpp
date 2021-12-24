#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using torch::Tensor;
using std::vector;


// CUDA declarations
Tensor CalcAssoc_CUDA_Forward(Tensor, Tensor, Tensor, int, int);
vector<Tensor> CalcAssoc_CUDA_Backward(Tensor, Tensor, Tensor, Tensor, int, int);


// CPU declarations
template <typename scalar_t>
Tensor CalcAssoc_CPU_Forward(Tensor pixel_feats, Tensor spixel_feats, Tensor index_map, int nw_spixels, int nh_spixels)
{
    const auto nBatchSize = pixel_feats.size(0);
    const auto nChannels = pixel_feats.size(1);
    const auto nHeight = pixel_feats.size(2);
    const auto nWidth = pixel_feats.size(3);

    const int nSpixels = nw_spixels * nh_spixels;

    Tensor tAssocMap = torch::zeros({nBatchSize, 9, nHeight, nWidth}).type_as(pixel_feats);

    // Accessors
    auto aIndexMap = index_map.accessor<int, 4>();
    auto aPixelFeats = pixel_feats.accessor<scalar_t, 4>();
    auto aAssocMap = tAssocMap.accessor<scalar_t, 4>();
    auto aSpixelFeats = spixel_feats.accessor<scalar_t, 3>();

    for (auto k = 0; k < nBatchSize; k++)
    {
        auto aIndexMap_ = aIndexMap[k][0];
        auto aSpixelFeats_ = aSpixelFeats[k];
        auto aPixelFeats_ = aPixelFeats[k];
        auto aAssocMap_ = aAssocMap[k];
        for (auto i = 0; i < nHeight; i++)
        {
            for (auto j = 0; j < nWidth; j++)
            {
                auto nCenAbsIdx = aIndexMap_[i][j];
                // Check validity
                if (nCenAbsIdx >= nSpixels || nCenAbsIdx < 0)
                    continue;
                for (auto nCurrRelIdx = 0; nCurrRelIdx < 9; nCurrRelIdx++)
                {
                    auto &pAssocMap = aAssocMap_[nCurrRelIdx][i][j];
                    // Convert abs_idx to rel_idx
                    auto curr_rel_i = nCurrRelIdx / 3 - 1;
                    auto curr_rel_j = nCurrRelIdx % 3 - 1;
                    auto curr_abs_i = nCenAbsIdx / nw_spixels + curr_rel_i;
                    auto curr_abs_j = nCenAbsIdx % nw_spixels + curr_rel_j;

                    if ((curr_abs_i < 0) || (curr_abs_i >= nh_spixels) || (curr_abs_j < 0) || (curr_abs_j >= nw_spixels))
                    {
                        // Set a very large distance for the out-of-boundary spixels
                        // So that Q=exp(neg_dist) will be close to 0
                        pAssocMap = -10000.0;
                        continue;
                    }
                    auto nCurrAbsIdx = curr_abs_i * nw_spixels + curr_abs_j;
                    scalar_t sum = 0.0;
                    for (auto c = 0; c < nChannels; c++)
                    {
                        auto diff = aPixelFeats_[c][i][j] - aSpixelFeats_[c][nCurrAbsIdx];
                        sum -= diff * diff;
                    }
                    pAssocMap = sum;
                }
            }
        }
    }

    return tAssocMap;
}


template <typename scalar_t>
vector<Tensor> CalcAssoc_CPU_Backward(
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

    const auto nSpixels = nw_spixels * nh_spixels;

    Tensor tGradPixelFeats = torch::zeros_like(pixel_feats);
    Tensor tGradSpixelFeats = torch::zeros_like(spixel_feats);

    // Accessors
    auto aIndexMap = index_map.accessor<int, 4>();
    auto aPixelFeats = pixel_feats.accessor<scalar_t, 4>();
    auto aSpixelFeats = spixel_feats.accessor<scalar_t, 3>();
    auto aGradPixelFeats = tGradPixelFeats.accessor<scalar_t, 4>();
    auto aGradSpixelFeats = tGradSpixelFeats.accessor<scalar_t, 3>();
    auto aGradOutput = grad_output.accessor<scalar_t, 4>();

    for (auto k = 0; k < nBatchSize; k++)
    {
        auto aIndexMap_ = aIndexMap[k][0];
        auto aSpixelFeats_ = aSpixelFeats[k];
        auto aPixelFeats_ = aPixelFeats[k];
        auto aGradPixelFeats_ = aGradPixelFeats[k];
        auto aGradSpixelFeats_ = aGradSpixelFeats[k];
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

                    auto dq = aGradOutput_[nCurrRelIdx][i][j];

                    for (auto c = 0; c < nChannels; c++)
                    {
                        aGradPixelFeats_[c][i][j] += -2* dq * (aPixelFeats_[c][i][j] - aSpixelFeats_[c][nCurrAbsIdx]);
                        aGradSpixelFeats_[c][nCurrAbsIdx] += -2* dq * (aSpixelFeats_[c][nCurrAbsIdx] - aPixelFeats_[c][i][j]);
                    }
                }
            }
        }
    }

    return {tGradPixelFeats, tGradSpixelFeats};
}


// C++ interface
Tensor CalcAssoc_Forward(
    Tensor pixel_feats, Tensor spixel_feats, Tensor index_map, int nw_spixels, int nh_spixels
)
{
    // Check size of index_map
    TORCH_CHECK(index_map.dim() == 4, "expected 4 dims but tensor has ", index_map.dim());
    TORCH_CHECK(index_map.size(1) == 1, "index_map must have size 1 at dimension 1");
    TORCH_CHECK(pixel_feats.size(1) == spixel_feats.size(1), "pixel_feats and spixel_feats should have the same number of channels")

    if (pixel_feats.type().is_cuda() && spixel_feats.type().is_cuda() && index_map.type().is_cuda())
    {
        CHECK_CONTIGUOUS(pixel_feats); CHECK_CONTIGUOUS(spixel_feats); CHECK_CONTIGUOUS(index_map);
        return CalcAssoc_CUDA_Forward(pixel_feats, spixel_feats, index_map, nw_spixels, nh_spixels);
    }
    else
    {
        return AT_DISPATCH_FLOATING_TYPES(pixel_feats.type(), "calc_assoc forward", 
        [&] {
            return CalcAssoc_CPU_Forward<scalar_t>(pixel_feats, spixel_feats, index_map, nw_spixels, nh_spixels);
        });
    }
}


vector<Tensor> CalcAssoc_Backward(
    Tensor grad_output, 
    Tensor pixel_feats, Tensor spixel_feats, 
    Tensor index_map, 
    int nw_spixels, int nh_spixels
)
{
    // Check size of index_map
    TORCH_CHECK(index_map.dim() == 4, "expected 4 dims but tensor has ", index_map.dim());
    TORCH_CHECK(index_map.size(1) == 1, "index_map must have size 1 at dimension 1");
    TORCH_CHECK(pixel_feats.size(1) == spixel_feats.size(1), "pixel_feats and spixel_feats should have the same number of channels")
    
    if (
        grad_output.type().is_cuda() && pixel_feats.type().is_cuda() && 
        spixel_feats.type().is_cuda() && index_map.type().is_cuda()
    )
    {
        CHECK_CONTIGUOUS(grad_output); CHECK_CONTIGUOUS(pixel_feats);
        CHECK_CONTIGUOUS(spixel_feats); CHECK_CONTIGUOUS(index_map);
        return CalcAssoc_CUDA_Backward(grad_output, pixel_feats, spixel_feats, index_map, nw_spixels, nh_spixels);
    }
    else
    {
        return AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "calc_assoc backward", 
        [&] {
            return CalcAssoc_CPU_Backward<scalar_t>(grad_output, pixel_feats, spixel_feats, index_map, nw_spixels, nh_spixels);
        });
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &CalcAssoc_Forward, "calc_assoc forward");
    m.def("backward", &CalcAssoc_Backward, "calc_assoc backward");
}