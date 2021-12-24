from torch.utils.cpp_extension import load


calc_assoc_cuda = load(
    'calc_assoc_cuda', ['calc_assoc_cuda.cpp', 'calc_assoc_cuda_kernel.cu'], verbose=True)
help(calc_assoc_cuda)