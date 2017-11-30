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
//
// Defines the general object for the abstraction of bilinear and
// nonlinear forms.

#ifndef MFEM_BILINEARFORMOPER
#define MFEM_BILINEARFORMOPER

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "fespace.hpp"
#include "nonlininteg.hpp"

namespace mfem
{

struct IntegratorMap
{
   virtual LinearFESpaceIntegrator *DomainIntegrator(BilinearFormIntegrator *integ) const
   { mfem_error("Not supported."); return NULL; }
   virtual LinearFESpaceIntegrator *InteriorFaceIntegrator(BilinearFormIntegrator *integ) const
   { mfem_error("Not supported."); return NULL; }
   virtual LinearFESpaceIntegrator *BdrFaceIntegrator(BilinearFormIntegrator *integ) const
   { mfem_error("Not supported."); return NULL; }

   virtual NonlinearFESpaceIntegrator *DomainIntegrator(NonlinearFormIntegrator *integ) const
   { mfem_error("Not supported."); return NULL; }

   virtual ~IntegratorMap() { }
};

class BilinearFormOperator : public Operator
{
protected:
   BilinearForm *bf;  // Do not own
   MixedBilinearForm *mbf;  // Do not own

   FiniteElementSpace *trial_fes;  // Do not own
   FiniteElementSpace *test_fes;  // Do not own
   bool trial_gs, test_gs;

   Array<int> *trial_offsets, *trial_indices;
   Array<int> *test_offsets, *test_indices;
   mutable Vector *X;
   mutable Vector *Y;

   Array<LinearFESpaceIntegrator*> lfesi;
   Array<NonlinearFESpaceIntegrator*> nlfesi;

   IntegratorMap *map;

   // Convert between vector types before calling Mult.
   void LToEVector(const Array<int> &offsets, const Array<int> &indices,
                   const Vector &v, Vector &V) const;
   void EToLVector(const Array<int> &offsets, const Array<int> &indices,
                   const Vector &V, Vector &v) const;

   void Clear();

   void Init(FiniteElementSpace *_trial_fes, FiniteElementSpace *_test_fes);

public:
   // Create an empty object or assemble what is needed by the
   // bilinear form integrators to later compute the action.
   BilinearFormOperator(IntegratorMap *_map = NULL);
   ~BilinearFormOperator();

   void Assemble();
   void Assemble(BilinearForm *bf);
   void Assemble(MixedBilinearForm *bf);

   /// Perform the action of the bilinear form on a vector and set y.
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Perform the action of the bilinear form on a vector and add to y.
   void AddMult(const Vector &x, Vector &y) const;

   /// Perform the (transposed) action of the bilinear form on a vector and add to y.
   void AddMultTranspose(const Vector &x, Vector &y) const;
};

}

#endif
