// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_LININTEG
#define MFEM_LININTEG

/// Abstract base class LinearFormIntegrator
class LinearFormIntegrator
{
public:
   /** Given a particular Finite Element and a transformation (Tr)
       computes the element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) = 0;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
   virtual ~LinearFormIntegrator() { };
};


/// Class for domain integration L(v) := (f, v)
class DomainLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   Coefficient &Q;
   const IntegrationRule *IntRule;
   int oa, ob;
public:
   /// Constructs a domain integrator with a given Coefficient
   DomainLFIntegrator(Coefficient &QF, int a = 2, int b = 0)
      // the old default was a = 1, b = 1
      // for simple elliptic problems a = 2, b = -2 is ok
      : Q(QF), oa(a), ob(b) { IntRule = NULL; }

   /// Constructs a domain integrator with a given Coefficient
   DomainLFIntegrator(Coefficient &QF, const IntegrationRule *ir)
      : Q(QF), oa(1), ob(1) { IntRule = ir; }

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
   const IntegrationRule *IntRule;

public:
   VectorBoundaryFluxLFIntegrator (Coefficient &f, double s = 1.0, const IntegrationRule *ir = NULL)
      : Sign(s), F(&f), IntRule(ir) { };

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

#endif
