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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

// *****************************************************************************
void Engine::Init(const std::string &engine_spec)
{
   nvtx_push();
   memory_resources[0] = NULL;
   workers_weights[0] = 1.0;
   workers_mem_res[0] = 0;
   dev = new device();

   bool cuda = false;
   const bool uvm = false;

   if (engine_spec.find("cpu")!=std::string::npos)
   {
      dbg("CPU engine");
      cuda = false;
   }
   if (engine_spec.find("gpu")!=std::string::npos)
   {
      dbg("GPU engine");
      cuda = true;
   }

   kernels::config::Get().Setup(world_rank,
                                world_size,
                                cuda,
                                false, // CG on device
                                uvm,
                                false, // MPI CUDA aware
                                false, // share
                                false, // occa
                                false, // hcpo
                                false, // sync
                                false, // dot
                                0); // rp_levels

   nvtx_pop();
}

// *****************************************************************************
Engine::Engine(const std::string &engine_spec) : mfem::Engine(NULL, 1, 1)
{
   nvtx_push();
   Init(engine_spec);
   nvtx_pop();
}

// *****************************************************************************
#ifdef MFEM_USE_MPI
Engine::Engine(MPI_Comm _comm,
               const std::string &engine_spec)
   : mfem::Engine(NULL, 1, 1),
     comm(_comm)
{
   nvtx_push();
   Init(engine_spec);
   nvtx_pop();
}

Engine::Engine(const MPI_Session *_mpi,
               const std::string &engine_spec)
   : mfem::Engine(NULL, 1, 1),
     comm(MPI_COMM_WORLD),
     mpi(_mpi),
     world_rank(mpi->WorldRank()),
     world_size(mpi->WorldSize())
{
   nvtx_push();
   Init(engine_spec);
   nvtx_pop();
}
#endif

// *****************************************************************************
DLayout Engine::MakeLayout(std::size_t size) const
{
   nvtx_push();
   const DLayout layout = DLayout(new kernels::Layout(*this, size));
   nvtx_pop();
   return layout;
}

// *****************************************************************************
DLayout Engine::MakeLayout(const mfem::Array<std::size_t> &offsets) const
{
   nvtx_push();
   MFEM_ASSERT(offsets.Size() == 2,
               "multiple workers are not supported yet");
   const DLayout layout = DLayout(new kernels::Layout(*this, offsets.Last()));
   nvtx_pop();
   return layout;
}

// *****************************************************************************
DArray Engine::MakeArray(PLayout &layout, std::size_t item_size) const
{
   nvtx_push();
   const DArray array = DArray(new kernels::Array(layout.As<Layout>(), item_size));
   nvtx_pop();
   return array;
}

// *****************************************************************************
DVector Engine::MakeVector(PLayout &layout, int type_id) const
{
   nvtx_push();
   const DVector vector(new kernels::Vector(layout.As<Layout>()));
   nvtx_pop();
   return vector;
}

// *****************************************************************************
#ifdef MFEM_USE_MPI
DFiniteElementSpace Engine::MakeFESpace(mfem::ParFiniteElementSpace &pfes) const
{
   nvtx_push();
   const DFiniteElementSpace dfes(new kFiniteElementSpace(*this, pfes));
   nvtx_pop();
   return dfes;
}
#endif

// *****************************************************************************
DFiniteElementSpace Engine::MakeFESpace(mfem::FiniteElementSpace &fes) const
{
   nvtx_push();
   DFiniteElementSpace dfes(new kFiniteElementSpace(*this, fes));
   nvtx_pop();
   return dfes;
}

// *****************************************************************************
DBilinearForm Engine::MakeBilinearForm(mfem::BilinearForm &bf) const
{
   nvtx_push();
   const DBilinearForm dbf(new BilinearForm(*this, bf));
   nvtx_pop();
   return dbf;
}

// *****************************************************************************
void Engine::AssembleLinearForm(LinearForm &l_form) const
{
   /// FIXME - What will the actual parameters be?
   MFEM_ABORT("FIXME");
}

// *****************************************************************************
mfem::Operator *Engine::MakeOperator(const MixedBilinearForm &mbl_form) const
{
   /// FIXME - What will the actual parameters be?
   MFEM_ABORT("FIXME");
   return NULL;
}

// *****************************************************************************
mfem::Operator *Engine::MakeOperator(const NonlinearForm &nl_form) const
{
   /// FIXME - What will the actual parameters be?
   MFEM_ABORT("FIXME");
   return NULL;
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
