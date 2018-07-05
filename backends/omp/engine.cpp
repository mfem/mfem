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

#include <map>

namespace mfem
{

namespace omp
{

typedef std::map<std::string, std::string> keyval_pair_t;

template<typename T, typename P>
static T remove_if(T beg, T end, P pred)
{
   T dest = beg;
   for (T itr = beg;itr != end; ++itr)
      if (!pred(*itr))
         *(dest++) = *itr;
   return dest;
}

void parse_token(const std::string &token, std::string &key, std::string &val)
{
   std::size_t sep = token.find_first_of(':');
   if (sep > token.size()) mfem_error("Parse error");

   key = token.substr(0, sep);
   key.erase(mfem::omp::remove_if(key.begin(), key.end(), isspace), key.end());
   key.erase(std::remove(key.begin(), key.end(), '\''), key.end());

   val = token.substr(sep+1);
   val.erase(mfem::omp::remove_if(val.begin(), val.end(), isspace), val.end());
   val.erase(std::remove(val.begin(), val.end(), '\''), val.end());
}

keyval_pair_t parse_engine_spec(const std::string &engine_spec)
{
   keyval_pair_t map;
   std::size_t token_extent = 0;
   std::string key, val;
   while (token_extent < engine_spec.size())
   {
      const std::string remaining(engine_spec, token_extent);

      std::size_t next_comma = remaining.find_first_of(',');
      if (next_comma == std::string::npos) next_comma = engine_spec.size() - 1;

      const std::string token(remaining, 0, next_comma);
      parse_token(token, key, val);

      map[key] = val;
      token_extent += next_comma+1;
   }
   return map;
}

void Engine::Init(const std::string &engine_spec)
{
   keyval_pair_t tokens(parse_engine_spec(engine_spec));
   keyval_pair_t::iterator it;

   it = tokens.find("exec_target");
   if (it != tokens.end())
   {
      if (!std::strncmp(it->second.data(), "device", 6))
      {
         exec_target = Device;
         device_number = 0;
      }
      else if (!std::strncmp(it->second.data(), "host", 4))
      {
         exec_target = Host;
         device_number = -1;
      }
      else
      {
         mfem_error("Parse error. Possible values for exec_target are: ['host', 'device']");
      }
   }
   else
   {
      // Default to host if not specified
      mfem::out << "Did not specify exec_target. Defaulting to host..." << std::endl;
      exec_target = Host;
      device_number = -1;
   }

   it = tokens.find("mem_type");
   if (it != tokens.end())
   {
      if (!std::strncmp(it->second.data(), "unified", 7))
      {
#if defined(MFEM_USE_CUDAUM)
         memory_resources[0] = new UnifiedMemoryResource();
         unified_memory = true;
#else
         mfem_error("Have not compiled support for CUDA unified memory.");
#endif
      }
      else if (!std::strncmp(it->second.data(), "separate", 4))
      {
         memory_resources[0] = new NewDeleteMemoryResource();
         unified_memory = false;
      }
      else
      {
         mfem_error("Parse error. Possible values for mem_type are: ['separate', 'unified']");
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

   it = tokens.find("mult_engine");
   if (it != tokens.end())
   {
      if (!std::strncmp(it->second.data(), "acrotensor", 10))
      {
         mult_type = Acrotensor;
      }
      else
      {
         mfem_error("Parse error. Possible values for mem_type are: ['acrotensor'].");
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
