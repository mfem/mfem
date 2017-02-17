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
protected:
   const IntegrationRule *ir;

public:
   NonlinearFormIntegrator() : ir(NULL) { }

   void SetIntegrationRule(const IntegrationRule &irule) { ir = &irule; }

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
   const DenseMatrix *W;

   static double Invariant1(const DenseMatrix &M);
   static double Invariant2(const DenseMatrix &M);

   static void Invariant1_dM(const DenseMatrix &M, DenseMatrix &dM);
   static void Invariant2_dM(const DenseMatrix &M, DenseMatrix &dM);

   static void Invariant1_dMdM(const DenseMatrix &M, int i, int j,
                               DenseMatrix &dMdM);
   static void Invariant2_dMdM(const DenseMatrix &M, int i, int j,
                               DenseMatrix &dMdM);

public:
   HyperelasticModel() : T(NULL), W(NULL) { }
   virtual ~HyperelasticModel() { }

   /// A ref->target transformation that can be used to evaluate coefficients.
   /** @note It's assumed that T.SetIntPoint() is already called for
       the point of interest. */
   void SetTransformation(ElementTransformation &_T) { T = &_T; }

   /// Ref->target transformation's Jacobian matrix for the point of interest.
   /** Using @a #W is an alternative to using @a #T, when one cannot define
       the target Jacobians by a single ElementTransformation for the whole
       zone, e.g., in the TMOP paradigm. */
   void SetTargetJacobian(const DenseMatrix &_W) { W = &_W; }

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

class TMOPHyperelasticModel001 : public HyperelasticModel
{
public:
   virtual double EvalW(const DenseMatrix &J) const;

   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel002 : public HyperelasticModel
{
public:
   virtual double EvalW(const DenseMatrix &J) const;

   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel007 : public HyperelasticModel
{
public:
   virtual double EvalW(const DenseMatrix &J) const;

   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/** Used to compute ref->target transformation Jacobian for different target
    options, used in class HyperelasticNLFIntegrator. */
class TargetJacobian
{
private:
   // Current nodes, initial nodes, target nodes that are
   // used in ComputeElementTargets(int), depending on target_type.
   const GridFunction *nodes, *nodes0, *tnodes;

   static void ConstructIdealJ(int geom, DenseMatrix &J);

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif

public:
   enum target {CURRENT, IDEAL, IDEAL_EQ_SIZE, IDEAL_INIT_SIZE, TARGET_MESH};
   const target target_type;

   TargetJacobian(target ttype)
      : nodes(NULL), nodes0(NULL), tnodes(NULL), target_type(ttype) { }
#ifdef MFEM_USE_MPI
   TargetJacobian(target ttype, MPI_Comm mpicomm)
      : nodes(NULL), nodes0(NULL), tnodes(NULL),
        comm(mpicomm), target_type(ttype) { }
#endif
   ~TargetJacobian() { }

   void SetNodes(const GridFunction &n)         { nodes  = &n;  }
   void SetInitialNodes(const GridFunction &n0) { nodes0 = &n0; }
   void SetTargetNodes(const GridFunction &tn)  { tnodes = &tn; }

   /** @brief Given an element and quadrature rule, computes Jacobian
       matrices for each quadrature point in the element. */
   void ComputeElementTargets(int e_id, const FiniteElement &fe,
                              const IntegrationRule &ir, DenseTensor &W) const;
};

/** Hyperelastic integrator for any given HyperelasticModel.
    Represents @f$ \int W(T) dx @f$ over a target zone,
    where W is the @a model's strain energy density function,
    T is the Jacobian of the target->physical coordinates transformation. */
class HyperelasticNLFIntegrator : public NonlinearFormIntegrator
{
private:
   HyperelasticModel *model;
   const TargetJacobian *targetJ;

   // Data used for "limiting" the HyperelasticNLFIntegrator.
   bool limited;
   double eps;
   const GridFunction *nodes0;

   // Can be used to create "composite" integrators for the TMOP purposes.
   Coefficient *coeff;

   //   J0i: the inverse of the ref->target transformation Jacobian.
   //    J1: the ref->physical transformation Jacobian.
   //     J: the target->physical transformation Jacobians.
   //     P: represents DF_dJ (dim x dim).
   //   DSh: gradients of reference shape functions (dof x dim).
   //    DS: represents DJ_dx (dof x dim).
   // PMatI: current coordinates of the nodes (dof x dim).
   // PMat0: represents DF_dx (dof x dim).
   DenseMatrix DSh, DS, J0i, J1, J, P, PMatI, PMatO;

public:
   /** @param[in] m  HyperelasticModel that defines F(T).
       @param[in] tJ See HyperelasticModel::SetTargetJacobian(). */
   HyperelasticNLFIntegrator(HyperelasticModel *m, TargetJacobian *tJ = NULL)
      : model(m), targetJ(tJ),
        limited(false), eps(0.0), nodes0(NULL), coeff(NULL) { }

   const TargetJacobian *GetTargetJacobian() { return targetJ; } const

   /// Adds an extra term to the integral.
   /** The integral of interest becomes
       @f$ \int \epsilon F(T) + 0.5 (x - x_0)^2 dx@f$,
       where the second term measures the change with respect to the
       original physical positions.
       @param[in] eps_  Scaling of the @a model's contribution.
       @param[in] n0    Original mesh coordinates. */
   void SetLimited(double eps_, const GridFunction &n0)
   {
      limited = true;
      eps = eps_;
      nodes0 = &n0;
   }

   /// Sets a scaling Coefficient for the integral.
   void SetCoefficient(Coefficient &c) { coeff = &c; }

   /** @brief Computes the integral of F(T) over a target zone
       @param[in] el    Type of FiniteElement.
       @param[in] Tr    Represents reference->target coordinates transformation.
       @param[in] elfun Physical coordinates of the zone. */
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

/// Interpolates the @a model's values at the nodes of @a gf.
/** Assumes that @a gf's FiniteElementSpace is initialized. */
void InterpolateHyperElasticModel(HyperelasticModel &model,
                                  const TargetJacobian &tj,
                                  const Mesh &mesh, GridFunction &gf);
}

#endif
