from torch.utils.cpp_extension import load


rel_to_abs_index_cuda = load(
    'rel_to_abs_index_cuda', ['rel_to_abs_index_cuda.cpp', 'rel_to_abs_index_cuda_kernel.cu'], verbose=True)
help(rel_to_abs_index_cuda)