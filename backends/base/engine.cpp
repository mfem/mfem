// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../config/config.hpp"
#ifdef MFEM_USE_BACKENDS

#include "engine.hpp"
#include "fespace.hpp"
#include "bilinearform.hpp"

namespace mfem
{

Engine::Engine(Backend *b, int n_mem, int n_workers)
   : backend(b),
#ifdef MFEM_USE_MPI
     comm(MPI_COMM_NULL),
#endif
     num_mem_res(n_mem),
     num_workers(n_workers),
     memory_resources(new MemoryResource*[num_mem_res]()),
     workers_weights(new double[num_workers]()),
     workers_mem_res(new int[num_workers]())
{
   // Note: all arrays are value-initialized with zeros.
}

Engine::~Engine()
{
   delete [] workers_mem_res;
   delete [] workers_weights;
   for (int i = 0; i < num_mem_res; i++)
   {
      delete memory_resources[i];
   }
   delete [] memory_resources;
}

} // namespace mfem

#endif // MFEM_USE_BACKENDS
