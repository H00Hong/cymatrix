from setuptools import Extension, setup

# from distutils.core import setup
from Cython.Build import cythonize

extensions = [Extension("cymatrix", ["cymatrix.pyx"])]

setup(ext_modules=cythonize(extensions))
# python setup.py build_ext --inplace
