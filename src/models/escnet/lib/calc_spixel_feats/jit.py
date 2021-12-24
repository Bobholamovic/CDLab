from torch.utils.cpp_extension import load


calc_spixel_feats_cuda = load(
    'calc_spixel_feats_cuda', ['calc_spixel_feats_cuda.cpp', 'calc_spixel_feats_cuda_kernel.cu'], verbose=True)
help(calc_spixel_feats_cuda)