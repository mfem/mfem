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

#ifndef MFEM_BACKENDS_PA_FESPACE_HPP
#define MFEM_BACKENDS_PA_FESPACE_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "engine.hpp"
#include "array.hpp"
#include "vector.hpp"
#include "../../fem/fem.hpp"

namespace mfem
{

namespace pa
{

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

   void BuildDofMaps();

public:
   FiniteElementSpace() = delete;

   /// Nearly-empty class that stores a pointer to a mfem::FiniteElementSpace instance and the engine
   FiniteElementSpace(const Engine &e, mfem::FiniteElementSpace &fespace);

   /// Virtual destructor
   virtual ~FiniteElementSpace()
   {
      delete tensor_offsets;
      delete tensor_indices;
   }

   Layout &GetELayout() { return e_layout; }

   /// Return the engine as an OpenMP engine
   const Engine &GetEngine() { return static_cast<const Engine&>(*engine); }

   /// Convert an E vector to L vector
   void ToLVector(const Vector<double>& e_vector, Vector<double>& l_vector);

   /// Covert an L vector to E vector
   void ToEVector(const Vector<double>& l_vector, Vector<double>& e_vector);

   const FiniteElement *GetFE(int i) const { return fes->GetFE(i); }

   /// Returns number of degrees of freedom in each direction.
   inline const int GetNDofs1d() const { return GetFE(0)->GetOrder() + 1; }

   /// Returns number of quadrature points in each direction.
   inline const int GetNQuads1d(const int order) const
   {
      const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, order);
      return ir1d.GetNPoints();
   }

   virtual const mfem::Operator *GetProlongationOperator() const {return NULL;}

   virtual const mfem::Operator *GetRestrictionOperator() const {return NULL;}

   virtual const mfem::Operator *GetInterpolationOperator(const mfem::QuadratureSpace &qspace) const {return NULL;}

   virtual const mfem::Operator *GetGradientOperator(const mfem::QuadratureSpace &qspace) const {return NULL;}
};

} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#endif // MFEM_BACKENDS_PA_FESPACE_HPP