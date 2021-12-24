from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(name='calc_pixel_feats_cuda',
      ext_modules=[cpp_extension.CUDAExtension('calc_pixel_feats_cuda', ['calc_pixel_feats_cuda.cpp', 'calc_pixel_feats_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})