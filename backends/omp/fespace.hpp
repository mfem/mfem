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

#ifndef MFEM_BACKENDS_OMP_FESPACE_HPP
#define MFEM_BACKENDS_OMP_FESPACE_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "engine.hpp"
#include "array.hpp"
#include "vector.hpp"
#include "../../fem/fem.hpp"

namespace mfem
{

namespace omp
{

/*
  Wraps an mfem::Operator that does not contain layout information.
 */
class BackendOperator : public mfem::Operator
{
   const mfem::Operator *op;

public:
   BackendOperator(Layout &in_layout, Layout &out_layout,
                   const mfem::Operator *op_) : Operator(in_layout, out_layout), op(op_) { }

   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const { op->Mult(x, y); }
   virtual void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const { op->MultTranspose(x, y); }
};

/// TODO: doxygen
class FiniteElementSpace : public mfem::PFiniteElementSpace
{
protected:
   //
   // Inherited fields
   //
   // SharedPtr<const mfem::Engine> engine;
   // mfem::FiniteElementSpace *fes;

   Layout e_layout;

   mfem::Array<int> *tensor_offsets, *tensor_indices;

   mutable mfem::Operator *prolongation, *restriction;

   void BuildDofMaps();

public:
   /// Nearly-empty class that stores a pointer to a mfem::FiniteElementSpace instance and the engine
   FiniteElementSpace(const Engine &e, mfem::FiniteElementSpace &fespace);

   /// Virtual destructor
   virtual ~FiniteElementSpace()
   {
      delete tensor_offsets;
      delete tensor_indices;
      delete prolongation;
      delete restriction;
   }

   Layout &GetELayout() { return e_layout; }

   Layout &GetVLayout() const
   { return *fes->GetVLayout().As<Layout>(); }

   Layout &GetTrueVLayout() const
   { return *fes->GetTrueVLayout().As<Layout>(); }

   /// Return the engine as an OpenMP engine
   const Engine &OmpEngine() { return static_cast<const Engine&>(*engine); }

   /// Convert an E vector to L vector
   void ToLVector(const Vector &e_vector, Vector &l_vector);

   /// Covert an L vector to E vector
   void ToEVector(const Vector &l_vector, Vector &e_vector);

   /// Get the finite element space prolongation matrix
   const Operator *GetProlongation() const;

   /// Get the finite element space restriction matrix
   const Operator *GetRestriction() const;

};

} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#endif // MFEM_BACKENDS_OMP_FESPACE_HPP
