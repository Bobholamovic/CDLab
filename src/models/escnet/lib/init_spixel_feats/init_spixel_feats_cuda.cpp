#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using torch::Tensor;


// CUDA forward declaration
Tensor InitSpixelsFeats_CUDA_Forward(Tensor pixel_feats, Tensor index_map, int n_spixels);


// CPU forward
template <typename scalar_t>
Tensor InitSpixelFeats_CPU_Forward(Tensor pixel_feats, Tensor index_map, int n_spixels)
{
    const auto nBatchSize = pixel_feats.size(0);
    const auto nChannels = pixel_feats.size(1);
    const auto nHeight = pixel_feats.size(2);
    const auto nWidth = pixel_feats.size(3);

    // Check size of index_map
    TORCH_CHECK(
        (index_map.size(0) == nBatchSize) & (index_map.size(2) == nHeight) & (index_map.size(3) == nWidth),
        "The size of pixel_feats must match that of index_map at every dimension except dimension 1"
    );

    Tensor tSpixelFeats = torch::zeros({nBatchSize, nChannels, n_spixels}).type_as(pixel_feats);
    Tensor tCounters = torch::zeros({nBatchSize, 1, n_spixels}, torch::dtype(torch::kInt32));

    // Accessors
    auto aIndexMap = index_map.accessor<int, 4>();
    auto aPixelFeats = pixel_feats.accessor<scalar_t, 4>();
    auto aSpixelFeats = tSpixelFeats.accessor<scalar_t, 3>();
    auto aCounters = tCounters.accessor<int, 3>();

    for (auto k = 0; k < nBatchSize; k++)
    {
        auto aIndexMap_ = aIndexMap[k][0];
        auto aSpixelFeats_ = aSpixelFeats[k];
        auto aCounters_ = aCounters[k][0];
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
                // Is there anything like numpy indexing or slice references
                for (auto c = 0; c < nChannels; c++)
                    aSpixelFeats_[c][idx] += aPixelFeats_[c][i][j];
                aCounters_[idx] += 1;
            }
        }
    }
    
    for (auto k = 0; k < nBatchSize; k++)
    {
        for (auto s = 0; s < n_spixels; s++)
        {
            auto counter = aCounters[k][0][s];
            if (counter == 0)
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
                    aSpixelFeats[k][c][s] /= counter;
                }     
            }
        }
    }

    return tSpixelFeats;
}


// C++ interface
Tensor InitSpixelFeats_Forward(Tensor pixel_feats, Tensor index_map, int n_spixels)
{
    // Check size of index_map
    TORCH_CHECK(index_map.dim() == 4, "expected 4 dims but tensor has ", index_map.dim());
    TORCH_CHECK(index_map.size(1) == 1, "index_map must have size 1 at dimension 1");

    if (pixel_feats.type().is_cuda() && index_map.type().is_cuda())
    {
        CHECK_CONTIGUOUS(pixel_feats); CHECK_CONTIGUOUS(index_map);
        return InitSpixelsFeats_CUDA_Forward(pixel_feats, index_map, n_spixels);
    }
    else
    {
        return AT_DISPATCH_FLOATING_TYPES(pixel_feats.type(), "init_spixel_feats forward", 
        [&] {
            return InitSpixelFeats_CPU_Forward<scalar_t>(pixel_feats, index_map, n_spixels);
        });
    }
}


Tensor InitSpixelFeats_Backward()
{
    // Not implemented yet
    AT_ERROR("the backward function is not yet implemented");
    return {};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &InitSpixelFeats_Forward, "init_spixel_feats forward");
    m.def("backward", &InitSpixelFeats_Backward, "init_spixel_feats backward");
}