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

#ifndef MFEM_BACKENDS_BASE_FE_SPACE_HPP
#define MFEM_BACKENDS_BASE_FE_SPACE_HPP

#include "../../config/config.hpp"
#ifdef MFEM_USE_BACKENDS

#include "engine.hpp"
#include "utils.hpp"

namespace mfem
{

class FiniteElementSpace;
class QuadratureSpace;

/// TODO: doxygen
class PFiniteElementSpace : public RefCounted
{
protected:
   /// Engine with shared ownership
   SharedPtr<const Engine> engine;
   /// Not owned.
   mfem::FiniteElementSpace *fes;

public:
   /// TODO: doxygen
   PFiniteElementSpace(const Engine &e, FiniteElementSpace &fespace)
      : engine(&e), fes(&fespace) { }

   /// Virtual destructor
   virtual ~PFiniteElementSpace() { }

   /// Get the associated engine
   const Engine &GetEngine() const { return *engine; }

   /// Return the associated mfem::FiniteElementSpace
   mfem::FiniteElementSpace *GetFESpace() const { return fes; }

   /// TODO
   template <typename derived_t>
   derived_t &As() { return *util::As<derived_t>(this); }

   /// TODO
   template <typename derived_t>
   const derived_t &As() const { return *util::As<const derived_t>(this); }


   /**
       @name Virtual interface: finite element space functionality
    */
   ///@{

   /// TODO
   /** Return the operator mapping T-vectors to L-vectors. If a NULL pointer is
       returned then the mapping is the idenity. */
   virtual const mfem::Operator *GetProlongationOperator() const = 0;

   /// TODO
   /** Return the operator mapping L-vectors to T-vectors that extracts the
       subset of all true dofs, i.e. no assembly is performed. If a NULL pointer
       is returned then the mapping is the idenity. */
   virtual const mfem::Operator *GetRestrictionOperator() const = 0;

   /// TODO
   /** Return the operator mapping L-vectors to Q-vectors that evaluates the
       values of a GridFunction as a QuadratureFunction on the given
       QuadratureSpace. If the returned pointer is NULL, then the mapping is the
       identity. */
   virtual const mfem::Operator *GetInterpolationOperator(
      const mfem::QuadratureSpace &qspace) const = 0;

   /// TODO
   /** Return the operator mapping L-vectors to Q-vectors that evaluates the
       _reference element_ gradients of a GridFunction as a QuadratureFunction
       on the given QuadratureSpace. */
   virtual const mfem::Operator *GetGradientOperator(
      const mfem::QuadratureSpace &qspace) const = 0;

   ///@}
   // End: Virtual interface
};

} // namespace mfem

#endif // MFEM_USE_BACKENDS

#endif // MFEM_BACKENDS_BASE_FE_SPACE_HPP
