from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(name='rel_to_abs_index_cuda',
      ext_modules=[cpp_extension.CUDAExtension('rel_to_abs_index_cuda', ['rel_to_abs_index_cuda.cpp', 'rel_to_abs_index_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})