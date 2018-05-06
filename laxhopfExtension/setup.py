# python setup.py install
from distutils.core import setup, Extension

laxhopf_module = Extension('laxhopf', sources=['laxhopf.cpp'])

setup(name='lax_hopf',
      version='1.0',
      description='Module for solving Hopfs equation.',
      ext_modules=[laxhopf_module])
