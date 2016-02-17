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

#ifndef MFEM_NONLININTEG
#define MFEM_NONLININTEG

#include "../config/config.hpp"
#include "fe.hpp"
#include "coefficient.hpp"

namespace mfem
{

/** The abstract base class NonlinearFormIntegrator is used to express the
    local action of a general nonlinear finite element operator. In addition
    it may provide the capability to assemble the local gradient operator
    and to compute the local energy. */
class NonlinearFormIntegrator
{
public:
   /// Perform the local action of the NonlinearFormIntegrator
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect) = 0;

   /// Assemble the local gradient matrix
   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun, DenseMatrix &elmat);

   /// Compute the local energy
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Tr,
                                   const Vector &elfun);

   virtual ~NonlinearFormIntegrator() { }
};


/// Abstract class for hyperelastic models
class HyperelasticModel
{
protected:
   ElementTransformation *T;

public:
   HyperelasticModel() { T = NULL; }

   /// An element transformation that can be used to evaluate coefficients.
   void SetTransformation(ElementTransformation &_T) { T = &_T; }

   /// Evaluate the strain energy density function, W=W(J).
   virtual double EvalW(const DenseMatrix &J) const = 0;

   /// Evaluate the 1st Piola-Kirchhoff stress tensor, P=P(J).
   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const = 0;

   /** Evaluate the derivative of the 1st Piola-Kirchhoff stress tensor
       and assemble its contribution to the local gradient matrix 'A'.
       'DS' is the gradient of the basis matrix (dof x dim), and 'weight'
       is the quadrature weight. */
   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const = 0;

   virtual ~HyperelasticModel() { }
};


/** Inverse-harmonic hyperelastic model with a strain energy density function
    given by the formula: W(J) = (1/2) det(J) Tr((J J^t)^{-1}) where J is the
    deformation gradient. */
class InverseHarmonicModel : public HyperelasticModel
{
protected:
   mutable DenseMatrix Z, S; // dim x dim
   mutable DenseMatrix G, C; // dof x dim

public:
   virtual double EvalW(const DenseMatrix &J) const;

   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};


/** Neo-Hookean hyperelastic model with a strain energy density function given
    by the formula: \f$(\mu/2)(\bar{I}_1 - dim) + (K/2)(det(J)/g - 1)^2\f$ where
    J is the deformation gradient and \f$\bar{I}_1 = (det(J))^{-2/dim} Tr(J
    J^t)\f$. The parameters \f$\mu\f$ and K are the shear and bulk moduli,
    respectively, and g is a reference volumetric scaling. */
class NeoHookeanModel : public HyperelasticModel
{
protected:
   mutable double mu, K, g;
   Coefficient *c_mu, *c_K, *c_g;
   bool have_coeffs;

   mutable DenseMatrix Z;    // dim x dim
   mutable DenseMatrix G, C; // dof x dim

   inline void EvalCoeffs() const;

public:
   NeoHookeanModel(double _mu, double _K, double _g = 1.0)
      : mu(_mu), K(_K), g(_g), have_coeffs(false) { c_mu = c_K = c_g = NULL; }

   NeoHookeanModel(Coefficient &_mu, Coefficient &_K, Coefficient *_g = NULL)
      : mu(0.0), K(0.0), g(1.0), c_mu(&_mu), c_K(&_K), c_g(_g),
        have_coeffs(true) { }

   virtual double EvalW(const DenseMatrix &J) const;

   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};


/// Hyperelastic integrator for any given HyperelasticModel
class HyperelasticNLFIntegrator : public NonlinearFormIntegrator
{
private:
   HyperelasticModel *model;

   DenseMatrix DSh, DS, J0i, J1, J, P, PMatI, PMatO;

public:
   HyperelasticNLFIntegrator(HyperelasticModel *m) : model(m) { }

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Tr,
                                   const Vector &elfun);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun, DenseMatrix &elmat);

   virtual ~HyperelasticNLFIntegrator();
};

}

#endif
