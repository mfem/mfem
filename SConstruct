# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.googlecode.com.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

Help("""
       Type: 'scons' to build the production library,
             'scons -c' to clean the build,
             'scons debug=1' to build the debug version,
             'scons openmp=1' to enable OpenMP support,
             'scons parallel=1' to build the parallel version.
       """)

import os

# Export the shell environment variables
env = Environment(ENV=os.environ)

CC_OPTS    = '-O3'
DEBUG_OPTS = '-g -Wall'

# External libraries
HYPRE_DIR  = "../hypre-2.8.0b/src/hypre"

# Which version of the METIS library should be used, 4 (default) or 5?
# env.Append(CPPDEFINES = ['MFEM_USE_METIS_5'])

# MFEM-specific options
env.Append(CPPDEFINES = ['MFEM_USE_MEMALLOC'])

# Debug options
debug = ARGUMENTS.get('debug', 0)
if int(debug):
   env.Prepend(CPPDEFINES = ['MFEM_DEBUG'])
   env.Append(CCFLAGS = DEBUG_OPTS)
else:
   env.Append(CCFLAGS = CC_OPTS)

# OpenMP options
openmp = ARGUMENTS.get('openmp', 0)
if int(openmp):
   env.Prepend(CPPDEFINES = ['MFEM_USE_OPENMP'])
   env.Append(CCFLAGS = '-fopenmp')
   print 'Enabled OpenMP'

# Parallel version
parallel = ARGUMENTS.get('parallel', 0)
if int(parallel):
   env.Append(CPPDEFINES = ['MFEM_USE_MPI'])
   env.Replace(CXX = 'mpicxx')
   env.Append(CPPPATH = [HYPRE_DIR+"/include"])
   print 'Building parallel version'
else:
   print 'Building serial version'

conf = Configure(env)

# Check for LAPACK
if conf.CheckLib('lapack', 'dsyevr_'):
   env.Append(CPPDEFINES = ['MFEM_USE_LAPACK'])
   print 'Using LAPACK'
else:
   print 'Did not find LAPACK, continuing without it'

env = conf.Finish()

env.Append(CPPPATH = ['.', 'general', 'linalg', 'mesh', 'fem'])

# general, linalg, mesh and fem sources
general_src = Glob('general/*.cpp')
linalg_src = Glob('linalg/*.cpp')
mesh_src = Glob('mesh/*.cpp')
fem_src = Glob('fem/*.cpp')

# libmfem.a library
env.Library('mfem',[general_src,linalg_src,mesh_src,fem_src])

# Always generate mfem_defs.hpp
def mfem_defs_build(target, source, env):
   mfem_defs = file("mfem_defs.hpp", "w")
   mfem_defs.write("// Auto-generated file.\n")
   for definition in env.Dictionary()['CPPDEFINES']:
      mfem_defs.write("#define "+definition+"\n")
   mfem_defs.close()
env.AlwaysBuild(env.Command('mfem_defs.hpp', 'mfem.hpp', mfem_defs_build))

