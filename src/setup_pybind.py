from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuFMHA',
    ext_modules=[
        CUDAExtension(
            name='cuFMHA',
            sources=[
                'my_layers/cuFMHA.cu',
                'my_layers/cuFMHA_bind.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3', '--use_fast_math',
                    '--expt-extended-lambda', '--expt-relaxed-constexpr'
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)