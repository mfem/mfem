// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// Implementation of Bilinear Form Integrators
//
// Contributed by: T. Horvath: Oakland University
//                 S. Rhebergen, A. Sivas: University of Waterloo

#ifndef MFEM_HDGINTEG
#define MFEM_HDGINTEG

#include "../config/config.hpp"

namespace mfem
{

//---------------------------------------------------------------------
// Advection integrator: to compute all the domain based integrals
//
// The output is
//
// elemmat = mass_coeff (u,v) + (v, avec.grad(u))
//
// mass_coeff is the reaction coefficient
// avec is the advection coefficient
class HDGDomainIntegratorAdvection : public BilinearFormIntegrator
{
private:
   Coefficient *mass_coeff;
   VectorCoefficient *avec;

   Vector shape1, shape2;
   DenseMatrix shape1_n, shape2_n, partelmat;

   Vector shapeq;
   Vector shapeu;
   Vector divshape, divshape_no_diffusion, vec2, BdFidxT;
   DenseMatrix dshape;
   DenseMatrix gshape;
   DenseMatrix Jadj;
   DenseMatrix Adv_ir;

public:
   HDGDomainIntegratorAdvection(Coefficient &mass, VectorCoefficient &_avec)
      : mass_coeff(&mass), avec(&_avec)  {  }

   using BilinearFormIntegrator::AssembleElementMatrix;
   virtual void AssembleElementMatrix(const FiniteElement &fe_u,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat1);
};

// Advection integrator to compute all the face based integrals
//
// The output is
//
// elemmat1 = - < 1, [zeta avec.n u v] >
// elemmat2 = < ubar, [zeta avec.n v] >
// elemmat3 = < ubar, [zeta avec.n v] >
// elemmat4 = < 1, [zeta avec.n ubar vbar] > + < 1, [(1-zeta) avec.n ubar vbar >_{\Gamma_N}
//
// avec is the advection coefficient
class HDGFaceIntegratorAdvection : public BilinearFormIntegrator
{
private:
   VectorCoefficient *avec;

   Vector shape, normal, normalJ, dshape_normal, shape_face, adv;
   DenseMatrix shape1_n, invJ, dshape, shape1_n_nodiff;

public:
   HDGFaceIntegratorAdvection(VectorCoefficient &_avec)
      : avec(&_avec)  {  }

   using BilinearFormIntegrator::AssembleFaceMatrixOneElement1and1FES;
   virtual void AssembleFaceMatrixOneElement1and1FES(const FiniteElement &fe_u,
                                                     const FiniteElement &face_fe,
                                                     FaceElementTransformations &Trans,
                                                     const int elem1or2,
                                                     const bool onlyB,
                                                     DenseMatrix &elmat1,
                                                     DenseMatrix &elmat2,
                                                     DenseMatrix &elmat3,
                                                     DenseMatrix &elmat4);

};

//---------------------------------------------------------------------
/** Boundary linear integrator for imposing inflow boundary
    conditions. Given the inflow data u_in, the linear form assembles the
    following integral on the boundary:

    + < g, vbar >

    where g = - u_in * a.n * zeta and vbar is the test function. */
class HDGInflowLFIntegrator : public LinearFormIntegrator
{
protected:
   Coefficient *u_in;
   VectorCoefficient *avec;

   // these are not thread-safe!
   Vector shape_f, n_L;

public:
   HDGInflowLFIntegrator(Coefficient &_u, VectorCoefficient &_avec)
   { u_in = &_u; avec = &_avec; }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
};

//---------------------------------------------------------------------
//---------------------------------------------------------------------

// Diffusion integrator: to compute all the domain based integrals
//
// The output is
//
//         [local_A11 local_A12]
// elmat = [local_A21    0.0   ]
//
// local_A11 = -(\nu^{-1} q, v)
// local_A12 = (u, div(v))
// local_A21 = (div(q), w)
//
// local_A21 = local_A12^T
//
// \nu is the constant diffusion coefficient
class HDGDomainIntegratorDiffusion : public BilinearFormIntegrator
{
private:
   ConstantCoefficient *nu;

   Vector shape, divshape;
   DenseMatrix partelmat, dshape, gshape, Jadj;

public:
   HDGDomainIntegratorDiffusion(ConstantCoefficient &_nu)
      : nu(&_nu) { }

   using BilinearFormIntegrator::AssembleElementMatrix2FES;
   virtual void AssembleElementMatrix2FES(const FiniteElement &fe_q,
                                          const FiniteElement &fe_u,
                                          ElementTransformation &Trans,
                                          DenseMatrix &elmat);

};

// Diffusion integrator to compute all the face based integrals
//
// The output is
//
//          [ 0.0    0.0   ]
// elmat1 = [ 0.0 local_A22 ]  - the face based integral for matrix A
//
//          [ local_B1 ]
// elmat2 = [ local_B2 ]  - the face based integral for matrix B
//
// elmat3 = [ local_C1  local_C2 ]  - the face based integral for matrix C
//
// elmat4 = local_D  - the face based integral for matrix D
//
// where
// local_B1 = < \lambda,v\cdot n>
// local_A22 = < \tau u, w>
// local_B2 = -< tau \lambda, w>
// local_C1 = < \lambda, v\cdot n>
// local_C2 = -< \tau \lambda, w>
// local_D = < \tau \lambda, \mu>
//
// q_diff_coeff is the constant diffusion coefficient
// local_C1 = local_B1^T
// local_C2 = local_B2^T
class HDGFaceIntegratorDiffusion : public BilinearFormIntegrator
{
private:
   double tauD;

   Vector shapeu, shapeq, normal, shape_face;
   DenseMatrix shape_dot_n;

public:
   HDGFaceIntegratorDiffusion(double a)
   { tauD = a; }

   using BilinearFormIntegrator::AssembleFaceMatrixOneElement2and1FES;
   virtual void AssembleFaceMatrixOneElement2and1FES(const FiniteElement &fe_q,
                                                     const FiniteElement &fe_u,
                                                     const FiniteElement &face_fe,
                                                     FaceElementTransformations &Trans,
                                                     const int elem1or2,
                                                     const bool onlyB,
                                                     DenseMatrix &elmat1,
                                                     DenseMatrix &elmat2,
                                                     DenseMatrix &elmat3,
                                                     DenseMatrix &elmat4);

};


}

#endif
