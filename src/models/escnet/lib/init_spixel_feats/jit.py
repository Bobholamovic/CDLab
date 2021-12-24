from torch.utils.cpp_extension import load


init_spixel_feats_cuda = load(
    'init_spixel_feats_cuda', ['init_spixel_feats_cuda.cpp', 'init_spixel_feats_cuda_kernel.cu'], verbose=True)
help(init_spixel_feats_cuda)