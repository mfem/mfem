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

#ifndef MFEM_BACKENDS_BASE_PFE_SPACE_HPP
#define MFEM_BACKENDS_BASE_PFE_SPACE_HPP

#include "../../config/config.hpp"
#ifdef MFEM_USE_BACKENDS

#include "engine.hpp"
#include "utils.hpp"

namespace mfem
{

class ParFiniteElementSpace;

/// TODO: doxygen
class PParFiniteElementSpace : public RefCounted
{
protected:
   /// Engine with shared ownership
   SharedPtr<const Engine> engine;
   /// Not owned.
   ParFiniteElementSpace *pfes;

public:
   /// TODO: doxygen
   PParFiniteElementSpace(const Engine &e, mfem::ParFiniteElementSpace &fes)
      : engine(&e), pfes(&fes) { }

   /// Virtual destructor
   virtual ~PParFiniteElementSpace() { }

   /// Get the associated engine
   const Engine &GetEngine() const { return *engine; }

   /// TODO
   template <typename derived_t>
   derived_t &As() { return *util::As<derived_t>(this); }

   /// TODO
   template <typename derived_t>
   const derived_t &As() const { return *util::As<const derived_t>(this); }
};

} // namespace mfem

#endif // MFEM_USE_BACKENDS

#endif // MFEM_BACKENDS_BASE_FE_SPACE_HPP
