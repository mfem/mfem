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

// This file contains operator-based bilinear form integrators used
// with BilinearFormOperator.

#ifndef MFEM_OBILININTEG
#define MFEM_OBILININTEG

#include "../config/config.hpp"
#include "bilininteg.hpp"

namespace mfem
{

/** Class for computing the action of (grad(u), grad(v)) from a scalar
 * fespace using a partially assembled operator at quadrature
 * points. */
class PADiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   // Carry pointer in order to have access to coefficient
   DiffusionIntegrator *integ;     // Do not own
   const FiniteElementSpace *fes;  // TODO: support mixed spaces
   DenseTensor Dtensor;
   DenseMatrix shape1d, dshape1d;

   // Action methods
   void MultSeg(const Vector &V, Vector &U);
   void MultQuad(const Vector &V, Vector &U);
   void MultHex(const Vector &V, Vector &U);

public:
   PADiffusionIntegrator(DiffusionIntegrator *_integ) : integ(_integ) {}

   virtual void AssembleOperator(const FiniteElementSpace *_trial_fes,
                                 const FiniteElementSpace *_test_fes);

   virtual void AssembleMult(const Vector &fun, Vector &vect);
};

/** Class for computing the action of (u, v) from a scalar fespace
 * using a partially assembled operator at quadrature points. */
class PAMassIntegrator : public BilinearFormIntegrator
{
protected:
   MassIntegrator *integ;     // Do not own
   const FiniteElementSpace *fes;  // TODO: support mixed spaces
   DenseMatrix Dmat;
   DenseMatrix shape1d;

   // Action methods
   void MultSeg(const Vector &V, Vector &U);
   void MultQuad(const Vector &V, Vector &U);
   void MultHex(const Vector &V, Vector &U);

public:
   PAMassIntegrator(MassIntegrator *_integ) : integ(_integ) {}

   virtual void AssembleOperator(const FiniteElementSpace *_trial_fes,
                                 const FiniteElementSpace *_test_fes);

   virtual void AssembleMult(const Vector &fun, Vector &vect);
};

}

#endif
