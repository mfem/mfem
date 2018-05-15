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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "backend.hpp"
#include "bilinearform.hpp"
#include "../../general/array.hpp"

namespace mfem
{

namespace raja
{

Engine::Engine(const std::string &engine_spec)
   : mfem::Engine(NULL, 1, 1)
{
   //
   // Initialize inherited fields
   //
   memory_resources[0] = NULL;
   workers_weights[0]= 1.0;
   workers_mem_res[0] = 0;

   //
   // Initialize the RAJA engine
   //
   //::raja::properties props(engine_spec);
   //device = new ::raja::device[1];
   //device[0].setup(props);

   // FIXME
   // okl_path = "raja[mfem]:";
   std::string mfem_prefix = mfem::GetSourcePath();
   okl_path = mfem_prefix + "/backends/raja";
   // okl_path = mfem::GetInstallPath() + "lib/mfem/raja" ???
   // okl_defines = "defines: { MFEM_OKL_PREFIX: '\"" + okl_path + "\"' }";
}

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
   MFEM_ASSERT(type_id == ScalarId<double>::value, "invalid type_id");
   MFEM_ASSERT(dynamic_cast<Layout *>(&layout) != NULL,
               "invalid input layout");
   Layout *lt = static_cast<Layout *>(&layout);
   return DVector(new Vector(*lt));
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

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
