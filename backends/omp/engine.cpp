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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "engine.hpp"
#include "array.hpp"
#include "layout.hpp"
#include "vector.hpp"
#include "fespace.hpp"
#include "bilinearform.hpp"
#include "memory_resource.hpp"

namespace mfem
{

namespace omp
{

template<typename T, typename P>
static T remove_if(T beg, T end, P pred)
{
   T dest = beg;
   for (T itr = beg;itr != end; ++itr)
      if (!pred(*itr))
         *(dest++) = *itr;
   return dest;
}

void Engine::Init(const std::string &engine_spec)
{
   std::string spec(engine_spec);
   spec.erase(
      mfem::omp::remove_if(spec.begin(), spec.end(), isspace),
      spec.end());

   if (spec.find("exec_target:") != std::string::npos)
   {
      if (spec.find("device") != std::string::npos)
      {
         exec_target = Device;
         device_number = 0;
      }
      else if (spec.find("host") != std::string::npos)
      {
         exec_target = Host;
         device_number = -1;
      }
      else
         mfem_error("Parse error");
   }
   else
   {
      // Default to host if not specified as a device:'CPU' or device:'GPU'
      exec_target = Host;
      device_number = -1;
   }

   if (spec.find("mem_type:") != std::string::npos) {
      if (spec.find("unified") != std::string::npos)
      {
#if defined(MFEM_USE_CUDAUM)
         memory_resources[0] = new UnifiedMemoryResource();
         unified_memory = true;
#else
         mfem_error("Have not compiled support for CUDA unified memory.");
#endif
      }
      else if (spec.find("host") != std::string::npos)
      {
         memory_resources[0] = new NewDeleteMemoryResource();
         unified_memory = false;
      }
   }
   else {
      if (exec_target == Device)
      {
#if defined(MFEM_USE_CUDAUM)
         mfem::out << "Did not specify mem_type in engine spec. Defaulting to unified memory..." << std::endl;
         // Default to unified memory
         memory_resources[0] = new UnifiedMemoryResource();
         unified_memory = true;
#else
         mfem::out << "Did not specify mem_type in engine spec. Defaulting to standard host memory..." << std::endl;
         memory_resources[0] = new NewDeleteMemoryResource();
         unified_memory = false;
#endif
      }
      else
      {
         mfem::out << "Did not specify mem_type in engine spec. Defaulting to standard host memory..." << std::endl;
         memory_resources[0] = new NewDeleteMemoryResource();
         unified_memory = false;
      }
   }

   if (spec.find("mult_engine:") != std::string::npos)
   {
      if (spec.find("acrotensor") != std::string::npos)
      {
         mult_type = Acrotensor;
      }
      else
      {
         mfem_error("Supported engines: 'acrotensor'");
      }
   }
   else
   {
      mfem::out << "Did not specify mult_engine in engine spec. Defaulting to Acrotensor..." << std::endl;
#ifndef MFEM_USE_ACROTENSOR
      mfem_error("Must compile with Acrotensor support");
#endif
      mult_type = Acrotensor;
   }
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

} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)
