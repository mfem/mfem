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
   push();
   memory_resources[0] = NULL;
   workers_weights[0] = 1.0;
   workers_mem_res[0] = 0;
   dev = new device();
   
   bool cuda = false;
   bool uvm = false;
   
   //if (engine_spec.find("cpu")!=std::string::npos) cuda = false;
   if (engine_spec.find("gpu")!=std::string::npos){
      cuda = true;
      uvm = false;
   }
   
   kernels::config::Get().Setup(world_rank,
                                world_size,
                                cuda,
                                false, // CG on device
                                uvm,
                                false, // aware
                                false, // share
                                false, // occa
                                false, // hcpo
                                false, // sync
                                false, // dot
                                0); // rp_levels

   pop();
}

// *****************************************************************************
Engine::Engine(const std::string &engine_spec) : mfem::Engine(NULL, 1, 1)
{
   push();
   Init(engine_spec);
   pop();
}

// *****************************************************************************
#ifdef MFEM_USE_MPI
Engine::Engine(MPI_Comm _comm,
               const std::string &engine_spec)
   : mfem::Engine(NULL, 1, 1),
     comm(_comm)
{
   push();
   Init(engine_spec);
   pop();
}
   
Engine::Engine(const MPI_Session *_mpi,
               const std::string &engine_spec)
   : mfem::Engine(NULL, 1, 1),
     comm(MPI_COMM_WORLD),
     mpi(_mpi),
     world_rank(mpi->WorldRank()),
     world_size(mpi->WorldSize())
{
   push();
   Init(engine_spec);
   pop();
}
#endif

// *****************************************************************************
DLayout Engine::MakeLayout(std::size_t size) const
{
   push();
   const DLayout layout = DLayout(new kernels::Layout(*this, size));
   pop();
   return layout;
}

// *****************************************************************************
DLayout Engine::MakeLayout(const mfem::Array<std::size_t> &offsets) const
{
   push();
   MFEM_ASSERT(offsets.Size() == 2,
               "multiple workers are not supported yet");
   const DLayout layout = DLayout(new kernels::Layout(*this, offsets.Last()));
   pop();
   return layout;
}

// *****************************************************************************
DArray Engine::MakeArray(PLayout &layout, std::size_t item_size) const
{
   push();
   const DArray array = DArray(new kernels::Array(layout.As<Layout>(), item_size));
   pop();
   return array;
}

// *****************************************************************************
DVector Engine::MakeVector(PLayout &layout, int type_id) const
{
   push();
   const DVector vector(new kernels::Vector(layout.As<Layout>()));
   pop();
   return vector;
}

// *****************************************************************************
#ifdef MFEM_USE_MPI
DFiniteElementSpace Engine::MakeFESpace(mfem::ParFiniteElementSpace &pfes) const
{
   push();
   const DFiniteElementSpace dfes(new kFiniteElementSpace(*this, pfes));
   pop();
   return dfes;
}
#endif

// *****************************************************************************
DFiniteElementSpace Engine::MakeFESpace(mfem::FiniteElementSpace &fes) const
{
   push();
   DFiniteElementSpace dfes(new kFiniteElementSpace(*this, fes));
   pop();
   return dfes;
}

// *****************************************************************************
DBilinearForm Engine::MakeBilinearForm(mfem::BilinearForm &bf) const
{
   push();
   const DBilinearForm dbf(new BilinearForm(*this, bf));
   pop();
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
