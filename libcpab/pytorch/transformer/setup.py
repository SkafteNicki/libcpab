from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(name='cpab_module_cpu',
      ext_modules=[CppExtension('cpab_module_cpu', ['CPAB_ops.cpp'])],
      cmdclass={'build_ext': BuildExtension})

setup(
    name='cpab_module_gpu', 
    ext_modules=[
	CUDAExtension(
		name='cpab_module__gpu', 
		sources=['CPAB_ops_cuda.cpp', 'CPAB_ops_cuda_kernel.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
    )
