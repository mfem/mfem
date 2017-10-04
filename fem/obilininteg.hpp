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

//Should be in DenseTensor?
template <int Dim>
class PAOperator
{

protected:
  DenseMatrix shape1d, dshape1d;

public:
  PAOperator(FiniteElementSpace *_fes, const int ir_order);
  /**
  * Computes V = B^T D B U where B is a tensor product of shape1d. 
  */
  void MultBtDB(const DenseMatrix D, const Vector &U, Vector &V);
  void MultGtDG(const DenseTensor D, const Vector &U, Vector &V);
  //void MultBtDG(const DenseMatrix D, const Vector &U, Vector &V);
  //void MultGtDB(const DenseMatrix D, const Vector &U, Vector &V);
  
};

/** Class for computing the action of (grad(u), grad(v)) from a scalar
 * fespace using a partially assembled operator at quadrature
 * points. */
class PADiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   FiniteElementSpace *fes;
   const FiniteElement *fe;
   const TensorBasisElement *tfe;

   const int dim;

   DenseTensor Dtensor;
   DenseMatrix shape1d, dshape1d;
   Coefficient *coeff;
   MatrixCoefficient *mcoeff;

   // Action methods
   void MultSeg(const Vector &V, Vector &U);
   void MultQuad(const Vector &V, Vector &U);
   void MultHex(const Vector &V, Vector &U);

   void ComputePA(const int ir_order);

public:
   PADiffusionIntegrator(FiniteElementSpace *_fes, const int ir_order);
   PADiffusionIntegrator(FiniteElementSpace *_fes, const int ir_order,
                         Coefficient &_coeff);
   PADiffusionIntegrator(FiniteElementSpace *_fes, const int ir_order,
                         MatrixCoefficient &_mcoeff);

   /// Perform the action of the BilinearFormIntegrator
   virtual void AssembleVector(const FiniteElementSpace &fes,
                               const Vector &fun, Vector &vect);
};

/** Class for computing the action of (alpha (q u, grad v)) from a scalar fespace
* using a partially assembled operator at quadrature points.
* */
class PAConvectionIntegrator : public BilinearFormIntegrator
{
   FiniteElementSpace *fes;
   const FiniteElement *fe;
   const TensorBasisElement *tfe;

   const int dim;

   DenseTensor Dtensor;
   DenseMatrix shape1d, dshape1d;
   VectorCoefficient *q;
public:
  PAConvectionIntegrator(FiniteElementSpace *_fes, const int ir_order
                        VectorCoefficient &q, double a = 1.0);
  
  // Perform the action of the BilinearFormIntegrator
  virtual void AssembleVector(const FiniteElementSpace &fes,
                              const Vector &fun, Vector &vect);
};

/** Class for computing the action of 
    (alpha < rho_q (q.n) {u},[v] > + beta < rho_q |q.n| [u],[v] >),
    where u and v are the trial and test variables, respectively, and rho/u are
    given scalar/vector coefficients. The vector coefficient, q, is assumed to
    be continuous across the faces and when given the scalar coefficient, rho,
    is assumed to be discontinuous. The integrator uses the upwind value of rho,
    rho_q, which is value from the side into which the vector coefficient, q,
    points. This uses a partially assembled operator at quadrature points.
* */
class PADGConvectionFaceIntegrator : public BilinearFormIntegrator
{

public:
  PADGConvectionFaceIntegrator(FiniteElementSpace *_fes, const int ir_order
                        VectorCoefficient &q, double a = 1.0);

  // Perform the action of the BilinearFormIntegrator
  virtual void AssembleVector(const FiniteElementSpace &fes,
                              const Vector &fun, Vector &vect);
};

/** Class for computing the action of (u, v) from a scalar fespace
 * using a partially assembled operator at quadrature points. */
class PAMassIntegrator : public BilinearFormIntegrator
{
protected:
   FiniteElementSpace *fes;
   const FiniteElement *fe;
   const TensorBasisElement *tfe;

   const int dim;

   Vector Dvec;
   DenseMatrix Dmat;
   DenseMatrix shape1d;
   Coefficient *coeff;

   // Action methods
   void MultSeg(const Vector &V, Vector &U);
   void MultQuad(const Vector &V, Vector &U);
   void MultHex(const Vector &V, Vector &U);

   void ComputePA(const int ir_order);

public:
   PAMassIntegrator(FiniteElementSpace *_fes, const int ir_order);
   PAMassIntegrator(FiniteElementSpace *_fes, const int ir_order, Coefficient &_coeff);

   /// Perform the action of the BilinearFormIntegrator
   virtual void AssembleVector(const FiniteElementSpace &fes,
                               const Vector &fun, Vector &vect);
};

}

#endif
