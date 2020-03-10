# from distutils.core import setup
# from Cython.Build import cythonize
# import numpy

# setup(
#     ext_modules=cythonize(['tree.pyx', 'utils.pyx', 'splitter.pyx'],
#                           compiler_directives={'language_level': '3'},
#                           annotate=True),
#     include_dirs=numpy.get_include()
# )

# from distutils.core import setup, Extension
# from Cython.Build import cythonize

# extensions = [Extension(name="tree", sources=["src/tree.pyx"], include_dirs=['.', '', 'src']),
#               Extension(name="splitter", sources=["src/splitter.pyx"], include_dirs=['.', '', 'src']),
#               Extension(name="utils", sources=["src/util/utils.pyx"], include_dirs=['.', '', 'src'])]

# setup(
#     name='cedar',
#     ext_modules=cythonize(extensions, compiler_directives={'language_level': '3'},
#                           annotate=True),
#     include_dirs=[numpy.get_include(), '.', 'src']
# )

import os

import numpy
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    config = Configuration('cedar', parent_name=parent_package, top_path=top_path)

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config.add_extension("_tree",
                         sources=["_tree.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("_splitter",
                         sources=["_splitter.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("_utils",
                         sources=["_utils.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])

    config.ext_modules = cythonize(
        config.ext_modules,
        compiler_directives={'language_level': 3},
        annotate=True
    )

    return config


if __name__ == "__main__":
    setup(**configuration(top_path='').todict())
    # setup(configuration)
