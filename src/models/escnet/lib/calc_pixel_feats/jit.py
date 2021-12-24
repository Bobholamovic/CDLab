from torch.utils.cpp_extension import load


calc_pixel_feats_cuda = load(
    'calc_pixel_feats_cuda', ['calc_pixel_feats_cuda.cpp', 'calc_pixel_feats_cuda_kernel.cu'], verbose=True)
help(calc_pixel_feats_cuda)