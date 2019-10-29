from distutils.core import setup, Extension

drl4amr_module = Extension('drl4amr', sources=['drl4amr.cpp'])

setup(name='drl4amr',
    version='1.0',
    description='DRL4AMR module.',
    ext_modules=[drl4amr_module])
