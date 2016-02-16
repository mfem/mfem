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

#ifndef MFEM_LININTEG
#define MFEM_LININTEG

#include "../config/config.hpp"
#include "coefficient.hpp"

namespace mfem
{

/// Abstract base class LinearFormIntegrator
class LinearFormIntegrator
{
protected:
   const IntegrationRule *IntRule;

   LinearFormIntegrator(const IntegrationRule *ir = NULL)
   { IntRule = ir; }

public:
   /** Given a particular Finite Element and a transformation (Tr)
       computes the element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) = 0;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   void SetIntRule(const IntegrationRule *ir) { IntRule = ir; }

   virtual ~LinearFormIntegrator() { }
};


/// Class for domain integration L(v) := (f, v)
class DomainLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   Coefficient &Q;
   int oa, ob;
public:
   /// Constructs a domain integrator with a given Coefficient
   DomainLFIntegrator(Coefficient &QF, int a = 2, int b = 0)
   // the old default was a = 1, b = 1
   // for simple elliptic problems a = 2, b = -2 is ok
      : Q(QF), oa(a), ob(b) { }

   /// Constructs a domain integrator with a given Coefficient
   DomainLFIntegrator(Coefficient &QF, const IntegrationRule *ir)
      : LinearFormIntegrator(ir), Q(QF), oa(1), ob(1) { }

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// Class for boundary integration L(v) := (g, v)
class BoundaryLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   Coefficient &Q;
   int oa, ob;
public:
   /// Constructs a boundary integrator with a given Coefficient QG
   BoundaryLFIntegrator(Coefficient &QG, int a = 1, int b = 1)
      : Q(QG), oa(a), ob(b) {};

   /** Given a particular boundary Finite Element and a transformation (Tr)
       computes the element boundary vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// Class for boundary integration \f$ L(v) = (g \cdot n, v) \f$
class BoundaryNormalLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   VectorCoefficient &Q;
   int oa, ob;
public:
   /// Constructs a boundary integrator with a given Coefficient QG
   BoundaryNormalLFIntegrator(VectorCoefficient &QG, int a = 1, int b = 1)
      : Q(QG), oa(a), ob(b) {};

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// Class for boundary integration \f$ L(v) = (g \dot tau, v) \f$ in 2D
class BoundaryTangentialLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   VectorCoefficient &Q;
   int oa, ob;
public:
   /// Constructs a boundary integrator with a given Coefficient QG
   BoundaryTangentialLFIntegrator(VectorCoefficient &QG, int a = 1, int b = 1)
      : Q(QG), oa(a), ob(b) {};

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/** Class for domain integration of L(v) := (f, v), where
    f=(f1,...,fn) and v=(v1,...,vn). */
class VectorDomainLFIntegrator : public LinearFormIntegrator
{
private:
   Vector shape, Qvec;
   VectorCoefficient &Q;

public:
   /// Constructs a domain integrator with a given VectorCoefficient
   VectorDomainLFIntegrator(VectorCoefficient &QF) : Q(QF) {};

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/** Class for boundary integration of L(v) := (g, v), where
    f=(f1,...,fn) and v=(v1,...,vn). */
class VectorBoundaryLFIntegrator : public LinearFormIntegrator
{
private:
   Vector shape, vec;
   VectorCoefficient &Q;

public:
   /// Constructs a boundary integrator with a given VectorCoefficient QG
   VectorBoundaryLFIntegrator(VectorCoefficient &QG) : Q(QG) {};

   /** Given a particular boundary Finite Element and a transformation (Tr)
       computes the element boundary vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// \f$ (f, v)_{\Omega} \f$ for VectorFiniteElements (Nedelec, Raviart-Thomas)
class VectorFEDomainLFIntegrator : public LinearFormIntegrator
{
private:
   VectorCoefficient &QF;
   DenseMatrix vshape;
   Vector vec;

public:
   VectorFEDomainLFIntegrator (VectorCoefficient &F) : QF(F) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};


/** \f$ (f, v \cdot n)_{\partial\Omega} \f$ for vector test function
    v=(v1,...,vn) where all vi are in the same scalar FE space and f is a
    scalar function. */
class VectorBoundaryFluxLFIntegrator : public LinearFormIntegrator
{
private:
   double Sign;
   Coefficient *F;
   Vector shape, nor;

public:
   VectorBoundaryFluxLFIntegrator(Coefficient &f, double s = 1.0,
                                  const IntegrationRule *ir = NULL)
      : LinearFormIntegrator(ir), Sign(s), F(&f) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/** Class for boundary integration of (f, v.n) for scalar coefficient f and
    RT vector test function v. This integrator works with RT spaces defined
    using the RT_FECollection class. */
class VectorFEBoundaryFluxLFIntegrator : public LinearFormIntegrator
{
private:
   Coefficient &F;
   Vector shape;

public:
   VectorFEBoundaryFluxLFIntegrator(Coefficient &f) : F(f) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// Class for boundary integration \f$ L(v) = (n \times f, v) \f$
class VectorFEBoundaryTangentLFIntegrator : public LinearFormIntegrator
{
private:
   VectorCoefficient &f;

public:
   VectorFEBoundaryTangentLFIntegrator(VectorCoefficient &QG) : f(QG) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};


/** Class for boundary integration of the linear form:
    (alpha/2) < (u.n) f, w > - beta < |u.n| f, w >,
    where f and u are given scalar and vector coefficients, respectively,
    and w is the scalar test function. */
class BoundaryFlowIntegrator : public LinearFormIntegrator
{
private:
   Coefficient *f;
   VectorCoefficient *u;
   double alpha, beta;

   Vector shape;

public:
   BoundaryFlowIntegrator(Coefficient &_f, VectorCoefficient &_u,
                          double a, double b)
   { f = &_f; u = &_u; alpha = a; beta = b; }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
};

/** Boundary linear integrator for imposing non-zero Dirichlet boundary
    conditions, to be used in conjunction with DGDiffusionIntegrator.
    Specifically, given the Dirichlet data u_D, the linear form assembles the
    following integrals on the boundary:

    sigma < u_D, (Q grad(v)).n > + kappa < {h^{-1} Q} u_D, v >,

    where Q is a scalar or matrix diffusion coefficient and v is the test
    function. The parameters sigma and kappa should be the same as the ones
    used in the DGDiffusionIntegrator. */
class DGDirichletLFIntegrator : public LinearFormIntegrator
{
protected:
   Coefficient *uD, *Q;
   MatrixCoefficient *MQ;
   double sigma, kappa;

   // these are not thread-safe!
   Vector shape, dshape_dn, nor, nh, ni;
   DenseMatrix dshape, mq, adjJ;

public:
   DGDirichletLFIntegrator(Coefficient &u, const double s, const double k)
      : uD(&u), Q(NULL), MQ(NULL), sigma(s), kappa(k) { }
   DGDirichletLFIntegrator(Coefficient &u, Coefficient &q,
                           const double s, const double k)
      : uD(&u), Q(&q), MQ(NULL), sigma(s), kappa(k) { }
   DGDirichletLFIntegrator(Coefficient &u, MatrixCoefficient &q,
                           const double s, const double k)
      : uD(&u), Q(NULL), MQ(&q), sigma(s), kappa(k) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
};

}

#endif
