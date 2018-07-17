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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "backend.hpp"
#include "bilinearform.hpp"
#include "../../general/array.hpp"

namespace mfem
{

namespace pa
{

void Engine::Init(const std::string &engine_spec)
{
   //
   // Initialize inherited fields
   //
   memory_resources[0] = NULL;
   workers_weights[0] = 1.0;
   workers_mem_res[0] = 0;
}

Engine::Engine()
   : mfem::Engine(NULL, 1, 1)
{
   Init("");
}

Engine::Engine(const std::string &engine_spec)
   : mfem::Engine(NULL, 1, 1)
{
   Init(engine_spec);
}

#ifdef MFEM_USE_MPI
Engine::Engine(MPI_Comm _comm, const std::string &engine_spec)
   : mfem::Engine(NULL, 1, 1)
{
   comm = _comm;
   Init(engine_spec);
}
#endif

DLayout Engine::MakeLayout(std::size_t size) const
{
   return DLayout(new Layout(*this, size));
}

DLayout Engine::MakeLayout(const mfem::Array<std::size_t> &offsets) const
{
   MFEM_ASSERT(offsets.Size() == 2,
               "multiple workers are not supported yet");
   return DLayout(new Layout(*this, offsets.Last()));
}

DArray Engine::MakeArray(PLayout &layout, std::size_t item_size) const
{
   MFEM_ASSERT(dynamic_cast<Layout *>(&layout) != NULL,
               "invalid input layout");
   Layout *lt = static_cast<Layout *>(&layout);
   return DArray(new Array(*lt, item_size));
}

DVector Engine::MakeVector(PLayout &layout, int type_id) const
{
   MFEM_ASSERT(dynamic_cast<Layout *>(&layout) != NULL,
               "invalid input layout");
   Layout *lt = static_cast<Layout *>(&layout);
   switch (type_id)
   {
   case ScalarId<double>::value:
      return DVector(new Vector<double>(*lt));
   case ScalarId<std::complex<double>>::value:
      return DVector(new Vector<std::complex<double>>(*lt));
   // case ScalarId<int>::value:
   //    return DVector(new Vector<int>(*lt));
   default:
      mfem_error("Invalid type_id");
   }
}

DFiniteElementSpace Engine::MakeFESpace(mfem::FiniteElementSpace &fespace) const
{
   return DFiniteElementSpace(new FiniteElementSpace(*this, fespace));
}

DBilinearForm Engine::MakeBilinearForm(mfem::BilinearForm &bf) const
{
   return DBilinearForm(new BilinearForm(*this, bf));
}

void Engine::AssembleLinearForm(LinearForm &l_form) const
{
   /// FIXME - What will the actual parameters be?
   MFEM_ABORT("FIXME");
}

mfem::Operator *Engine::MakeOperator(const MixedBilinearForm &mbl_form) const
{
   /// FIXME - What will the actual parameters be?
   MFEM_ABORT("FIXME");
   return NULL;
}

mfem::Operator *Engine::MakeOperator(const NonlinearForm &nl_form) const
{
   /// FIXME - What will the actual parameters be?
   MFEM_ABORT("FIXME");
   return NULL;
}

} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)
