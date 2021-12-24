from torch.utils.cpp_extension import load


smear_cuda = load(
    'smear_cuda', ['smear_cuda.cpp', 'smear_cuda_kernel.cu'], verbose=True)
help(smear_cuda)