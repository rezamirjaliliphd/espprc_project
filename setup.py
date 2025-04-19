# setup.py for ESPPRC Project
# This script builds and installs the ESPPRC module with Cython acceleration.

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension
extensions = [
    Extension(
        name="espprc.cython_funcs",  # Module path
        sources=["espprc/cython_funcs.pyx"],  # Cython source
        language="c++",  # Enable C++ support
        extra_compile_args=["-O3", "-fopenmp"],  # Optimization + OpenMP for parallelism
        extra_link_args=["-fopenmp"],
        include_dirs=[np.get_include()],  # Required for NumPy
    )
]

# Configure the setup
setup(
    name="espprc_project",
    version="0.1",
    author="Reza Mirjalili",
    description="ESPPRC solver with Cython acceleration for label-setting algorithms",
    packages=["espprc"],  # Your Python package folder
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,        # Disable bounds checking for speed
            "wraparound": False,         # Disable negative indexing
            "cdivision": True,           # Enable C-style division
            "initializedcheck": False,   # Disable variable initialization checks
            "nonecheck": False           # Disable None checks
        }
    ),
    zip_safe=False,
)

