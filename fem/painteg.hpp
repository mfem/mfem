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

// This file contains FESpaceIntegrators.


#ifndef MFEM_PAINTEG
#define MFEM_PAINTEG

#include "../config/config.hpp"
#include "nonlininteg.hpp"
#include "bilinearformoper.hpp"

namespace mfem
{

/// Return the 1D basis for the given fe evaluated at the integration
/// points for the standard internal rule with order ir_order.
void Get1DBasis(const FiniteElement *fe, int ir_order,
                DenseMatrix &shape1d);
void Get1DBasis(const FiniteElement *fe, int ir_order,
                DenseMatrix &shape1d, DenseMatrix &dshape1d);

// These integrators use constructors based on the non-PA versions so
// that the options are consistent. If that is not the case, the
// friendship can be revoked and those constructors can be removed.

/** Class for computing the action of (grad(u), grad(v)) from a scalar
 * fespace using a partially assembled operator at quadrature
 * points. */
class PADiffusionIntegrator : public LinearFESpaceIntegrator
{
protected:
   // Carry pointer in order to have access to coefficient
   DiffusionIntegrator *integ;     // Own this
   const FiniteElementSpace *fes;  // TODO: support mixed spaces
   DenseTensor Dtensor;
   DenseMatrix shape1d, dshape1d;

   // Action methods
   void MultSeg(const Vector &V, Vector &U);
   void MultQuad(const Vector &V, Vector &U);
   void MultHex(const Vector &V, Vector &U);

public:
   PADiffusionIntegrator(DiffusionIntegrator *_integ) : integ(_integ) {}
   ~PADiffusionIntegrator() { delete integ; }

   virtual void Assemble(FiniteElementSpace *trial_fes,
                         FiniteElementSpace *test_fes);

   virtual void AddMult(const Vector &x, Vector &y);
};

/** Class for computing the action of (u, v) from a scalar fespace
 * using a partially assembled operator at quadrature points. */
class PAMassIntegrator : public LinearFESpaceIntegrator
{
protected:
   MassIntegrator *integ;     // Own this
   VectorMassIntegrator *vinteg;     // Own this
   const FiniteElementSpace *fes;  // TODO: support mixed spaces
   DenseTensor Dtensor;
   DenseMatrix shape1d;

   // Action methods
   void MultSeg(const Vector &V, Vector &U);
   void MultQuad(const Vector &V, Vector &U);
   void MultHex(const Vector &V, Vector &U);

public:
   PAMassIntegrator(MassIntegrator *_integ) : integ(_integ), vinteg(NULL) {}
   PAMassIntegrator(VectorMassIntegrator *_vinteg) : integ(NULL), vinteg(_vinteg) {}
   ~PAMassIntegrator() { delete integ; delete vinteg; }

   virtual void Assemble(FiniteElementSpace *_trial_fes,
                         FiniteElementSpace *_test_fes);

   virtual void AddMult(const Vector &x, Vector &y);
};

}

#endif
