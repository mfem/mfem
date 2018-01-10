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
#include "fespace.hpp"

namespace mfem
{

/** The abstract base class NonlinearFormIntegrator is used to express the
    local action of a general nonlinear finite element operator. In addition
    it may provide the capability to assemble the local gradient operator
    and to compute the local energy. */
class NonlinearFormIntegrator
{
protected:
   const IntegrationRule *IntRule;

   NonlinearFormIntegrator(const IntegrationRule *ir = NULL)
      : IntRule(NULL) { }

public:
   /** @brief Prescribe a fixed IntegrationRule to use (when @a ir != NULL) or
       let the integrator choose (when @a ir == NULL). */
   void SetIntRule(const IntegrationRule *ir) { IntRule = ir; }

   /// Prescribe a fixed IntegrationRule to use.
   void SetIntegrationRule(const IntegrationRule &irule) { IntRule = &irule; }

   /// Perform the local action of the NonlinearFormIntegrator
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);

   /// @brief Perform the local action of the NonlinearFormIntegrator resulting
   /// from a face integral term.
   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);

   /// Assemble the local gradient matrix
   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun, DenseMatrix &elmat);

   /// @brief Assemble the local action of the gradient of the
   /// NonlinearFormIntegrator resulting from a face integral term.
   virtual void AssembleFaceGrad(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Tr,
                                 const Vector &elfun, DenseMatrix &elmat);

   /// Compute the local energy
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Tr,
                                   const Vector &elfun);

   virtual ~NonlinearFormIntegrator() { }
};

class NonlinearFESpaceIntegrator
{
public:
   NonlinearFESpaceIntegrator() { }

   virtual ~NonlinearFESpaceIntegrator() { }

   /// Internally assemble the integrator for the specific trial and
   /// test spaces, and a vector u to calculate the term.
   virtual void Assemble(FiniteElementSpace *trial_fes,
                         FiniteElementSpace *test_fes,
                         const Vector &u) { }

   /// Apply the operator/form the vector (assemble is called before this).
   virtual void FormVector(Vector &y) = 0;
};

/// Abstract class for hyperelastic models
class HyperelasticModel
{
protected:
   ElementTransformation *Ttr; /**< Reference-element to target-element
                                    transformation. */

public:
   HyperelasticModel() : Ttr(NULL) { }
   virtual ~HyperelasticModel() { }

   /// A reference-element to target-element transformation that can be used to
   /// evaluate Coefficient%s.
   /** @note It is assumed that _Ttr.SetIntPoint() is already called for the
       point of interest. */
   void SetTransformation(ElementTransformation &_Ttr) { Ttr = &_Ttr; }

   /** @brief Evaluate the strain energy density function, W = W(Jpt).
       @param[in] Jpt  Represents the target->physical transformation
                       Jacobian matrix. */
   virtual double EvalW(const DenseMatrix &Jpt) const = 0;

   /** @brief Evaluate the 1st Piola-Kirchhoff stress tensor, P = P(Jpt).
       @param[in] Jpt  Represents the target->physical transformation
                       Jacobian matrix.
       @param[out]  P  The evaluated 1st Piola-Kirchhoff stress tensor. */
   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const = 0;

   /** @brief Evaluate the derivative of the 1st Piola-Kirchhoff stress tensor
       and assemble its contribution to the local gradient matrix 'A'.
       @param[in] Jpt     Represents the target->physical transformation
                          Jacobian matrix.
       @param[in] DS      Gradient of the basis matrix (dof x dim).
       @param[in] weight  Quadrature weight coefficient for the point.
       @param[in,out]  A  Local gradient matrix where the contribution from this
                          point will be added.

       Computes weight * d(dW_dxi)_d(xj) at the current point, for all i and j,
       where x1 ... xn are the FE dofs. This function is usually defined using
       the matrix invariants and their derivatives.
   */
   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const = 0;
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


/** Hyperelastic integrator for any given HyperelasticModel.

    Represents @f$ \int W(Jpt) dx @f$ over a target zone, where W is the
    @a model's strain energy density function, and Jpt is the Jacobian of the
    target->physical coordinates transformation. The target configuration is
    given by the current mesh at the time of the evaluation of the integrator.
*/
class HyperelasticNLFIntegrator : public NonlinearFormIntegrator
{
private:
   HyperelasticModel *model;

   //   Jrt: the Jacobian of the target-to-reference-element transformation.
   //   Jpr: the Jacobian of the reference-to-physical-element transformation.
   //   Jpt: the Jacobian of the target-to-physical-element transformation.
   //     P: represents dW_d(Jtp) (dim x dim).
   //   DSh: gradients of reference shape functions (dof x dim).
   //    DS: gradients of the shape functions in the target (stress-free)
   //        configuration (dof x dim).
   // PMatI: coordinates of the deformed configuration (dof x dim).
   // PMatO: reshaped view into the local element contribution to the operator
   //        output - the result of AssembleElementVector() (dof x dim).
   DenseMatrix DSh, DS, Jrt, Jpr, Jpt, P, PMatI, PMatO;

public:
   /** @param[in] m  HyperelasticModel that will be integrated. */
   HyperelasticNLFIntegrator(HyperelasticModel *m) : model(m) { }

   /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone
       @param[in] el     Type of FiniteElement.
       @param[in] Ttr    Represents ref->target coordinates transformation.
       @param[in] elfun  Physical coordinates of the zone. */
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Ttr,
                                   const Vector &elfun);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Ttr,
                                      const Vector &elfun, Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Ttr,
                                    const Vector &elfun, DenseMatrix &elmat);
};

}

#endif
