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
   //
   // Initialize inherited fields
   //
   push();
   memory_resources[0] = NULL;
   workers_weights[0]= 1.0;
   workers_mem_res[0] = 0;
   dev=new device();
   pop();
}

// *****************************************************************************
Engine::Engine(const std::string &engine_spec)
   : mfem::Engine(NULL, 1, 1)
{
   push();
   Init(engine_spec);
   pop();
}

// *****************************************************************************
#ifdef MFEM_USE_MPI
Engine::Engine(MPI_Comm _comm,
               const std::string &engine_spec) :
   mfem::Engine(NULL, 1, 1)
{
   push();
   comm = _comm;
   Init(engine_spec);
   pop();
}
#endif

// *****************************************************************************
DLayout Engine::MakeLayout(std::size_t size) const
{
   push(); pop();
   return DLayout(new kernels::Layout(*this, size));
}

// *****************************************************************************
DLayout Engine::MakeLayout(const mfem::Array<std::size_t> &offsets) const
{
   push();
   MFEM_ASSERT(offsets.Size() == 2,
               "multiple workers are not supported yet");
   pop();
   return DLayout(new kernels::Layout(*this, offsets.Last()));
}

// *****************************************************************************
DArray Engine::MakeArray(PLayout &layout, std::size_t item_size) const
{
   push();
   MFEM_ASSERT(dynamic_cast<Layout *>(&layout) != NULL,
               "invalid input layout");
   Layout *lt = static_cast<Layout *>(&layout);
   pop();
   return DArray(new kernels::Array(*lt, item_size));
}

// *****************************************************************************
DVector Engine::MakeVector(PLayout &layout, int type_id) const
{
   push();
   MFEM_ASSERT(type_id == ScalarId<double>::value, "invalid type_id");
   MFEM_ASSERT(dynamic_cast<Layout *>(&layout) != NULL,
               "invalid input layout");
   Layout *lt = static_cast<Layout *>(&layout);
   pop();
   return DVector(new kernels::Vector(*lt));
}

// *****************************************************************************
DFiniteElementSpace Engine::MakeFESpace(mfem::FiniteElementSpace &fespace) const
{
   push();
   pop();
   return DFiniteElementSpace(new KernelsFiniteElementSpace(*this, fespace));
}

// *****************************************************************************
DBilinearForm Engine::MakeBilinearForm(mfem::BilinearForm &bf) const
{
   push();
   pop();
   return DBilinearForm(new BilinearForm(*this, bf));
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
