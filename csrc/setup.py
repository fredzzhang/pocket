from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='masks_cpp',
    ext_modules=[
        cpp_extension.CppExtension(
            name='masks_cpp',
            sources=['masks.cpp']
        )],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
