import os
from setuptools import setup
from setuptools import find_packages

import numpy as np
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_modules = ['dare/_argsort.pyx', 'dare/_config.pyx',
               'dare/_manager.pyx', 'dare/_remover.pyx', 'dare/_simulator.pyx',
               'dare/_splitter.pyx', 'dare/_tree.pyx', 'dare/_utils.pyx']

libraries = []
if os.name == 'posix':
    libraries.append('m')

setup(name="dare-rf",
      version="0.10.1",
      description="Data Removal-Enabled Random Forests",
      author="Jonathan Brophy",
      author_email="jbrophy@cs.uoregon.edu",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/jjbrophy47/dare",
      packages=find_packages(),
      include_package_data=True,
      package_dir={"dare": "dare"},
      classifiers=["Programming Language :: Python :: 3.7",
                   "License :: OSI Approved :: Apache Software License",
                   "Operating System :: OS Independent"],
      python_requires='>=3.7',
      install_requires=["numpy>=1.21"],
      ext_modules=cythonize(ext_modules,
                            compiler_directives={'language_level': 3},
                            annotate=True),
      include_dirs=np.get_include(),
      zip_safe=False)
