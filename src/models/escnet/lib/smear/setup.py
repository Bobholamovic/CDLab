from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(name='smear_cuda',
      ext_modules=[cpp_extension.CUDAExtension('smear_cuda', ['smear_cuda.cpp', 'smear_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})