from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(name='calc_assoc_cuda',
      ext_modules=[cpp_extension.CUDAExtension('calc_assoc_cuda', ['calc_assoc_cuda.cpp', 'calc_assoc_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})