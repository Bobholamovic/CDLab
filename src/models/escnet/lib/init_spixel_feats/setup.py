from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(name='init_spixel_feats_cuda',
      ext_modules=[cpp_extension.CUDAExtension('init_spixel_feats_cuda', ['init_spixel_feats_cuda.cpp', 'init_spixel_feats_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})