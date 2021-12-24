#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using torch::Tensor;


// CUDA declarations
Tensor Smear_CUDA_Forward(Tensor, Tensor, int);
Tensor Smear_CUDA_Backward(Tensor, Tensor, int);


// CPU declarations
template <typename scalar_t>
Tensor Smear_CPU_Forward(Tensor spixel_feats, Tensor index_map, int n_spixels)
{
    const auto nBatchSize = spixel_feats.size(0);
    const auto nChannels = spixel_feats.size(1);
    const auto nHeight = index_map.size(2);
    const auto nWidth = index_map.size(3);

    Tensor tPixelFeats = torch::zeros({nBatchSize, nChannels, nHeight, nWidth}).type_as(spixel_feats);

    // Accessors
    auto aIndexMap = index_map.accessor<int, 4>();
    auto aPixelFeats = tPixelFeats.accessor<scalar_t, 4>();
    auto aSpixelFeats = spixel_feats.accessor<scalar_t, 3>();

    for (auto k = 0; k < nBatchSize; k++)
    {
        auto aIndexMap_ = aIndexMap[k][0];
        auto aSpixelFeats_ = aSpixelFeats[k];
        auto aPixelFeats_ = aPixelFeats[k];
        for (auto i = 0; i < nHeight; i++)
        {
            for (auto j = 0; j < nWidth; j++)
            {
                auto idx = aIndexMap_[i][j];
                // if (idx >= n_spixels)
                //     continue;   // Ignore invalid indexes
                // else if (idx < 0)
                //     AT_INDEX_ERROR("the indexes should be nonnegative");   // Throw index error

                // In consistency with CUDA version
                if (idx >= n_spixels || idx < 0)
                    continue;

                for (auto c = 0; c < nChannels; c++)
                    aPixelFeats_[c][i][j] = aSpixelFeats_[c][idx];
            }
        }
    }
    
    return tPixelFeats;
}


template <typename scalar_t>
Tensor Smear_CPU_Backward(Tensor grad_output, Tensor index_map, int n_spixels)
{
    const auto nBatchSize = grad_output.size(0);
    const auto nChannels = grad_output.size(1);
    const auto nHeight = index_map.size(2);
    const auto nWidth = index_map.size(3);
    Tensor tGradSpixelFeats = torch::zeros({nBatchSize, nChannels, n_spixels}).type_as(grad_output);

    // Accessors
    auto aIndexMap = index_map.accessor<int, 4>();
    auto aGradSpixelFeats = tGradSpixelFeats.accessor<scalar_t, 3>();
    auto aGradOutput = grad_output.accessor<scalar_t, 4>();

    for (auto k = 0; k < nBatchSize; k++)
    {
        auto aIndexMap_ = aIndexMap[k][0];
        auto aGradOutput_ = aGradOutput[k];
        auto aGradSpixelFeats_ = aGradSpixelFeats[k];
        for (auto i = 0; i < nHeight; i++)
        {
            for (auto j = 0; j < nWidth; j++)
            {
                auto idx = aIndexMap_[i][j];

                if (idx >= n_spixels || idx < 0)
                    continue;

                for (auto c = 0; c < nChannels; c++)
                    aGradSpixelFeats_[c][idx] += aGradOutput_[c][i][j];
            }
        }
    }
    return tGradSpixelFeats;
}


// C++ interface
Tensor Smear_Forward(Tensor spixel_feats, Tensor index_map, int n_spixels)
{
    // Check size of index_map
    TORCH_CHECK(index_map.dim() == 4, "expected 4 dims but tensor has ", index_map.dim());
    TORCH_CHECK(index_map.size(1) == 1, "index_map must have size 1 at dimension 1");

    if (spixel_feats.type().is_cuda() && index_map.type().is_cuda())
    {
        CHECK_CONTIGUOUS(spixel_feats); CHECK_CONTIGUOUS(index_map);
        return Smear_CUDA_Forward(spixel_feats, index_map, n_spixels);
    }
    else
    {
        return AT_DISPATCH_FLOATING_TYPES(spixel_feats.type(), "smear forward", 
        [&] {
            return Smear_CPU_Forward<scalar_t>(spixel_feats, index_map, n_spixels);
        });
    }
}


Tensor Smear_Backward(Tensor grad_output, Tensor index_map, int n_spixels)
{
    // Check size of index_map
    TORCH_CHECK(index_map.dim() == 4, "expected 4 dims but tensor has ", index_map.dim());
    TORCH_CHECK(index_map.size(1) == 1, "index_map must have size 1 at dimension 1");
    
    if (grad_output.type().is_cuda() && index_map.type().is_cuda())
    {
        CHECK_CONTIGUOUS(grad_output); CHECK_CONTIGUOUS(index_map);
        return Smear_CUDA_Backward(grad_output, index_map, n_spixels);
    }
    else
    {
        return AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "smear backward", 
        [&] {
            return Smear_CPU_Backward<scalar_t>(grad_output, index_map, n_spixels);
        });
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &Smear_Forward, "smear forward");
    m.def("backward", &Smear_Backward, "smear backward");
}