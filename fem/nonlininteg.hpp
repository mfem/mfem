// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_NONLININTEG
#define MFEM_NONLININTEG

#include "../config/config.hpp"
#include "fe.hpp"
#include "coefficient.hpp"
#include "fespace.hpp"
#include "ceed/interface/operator.hpp"

namespace mfem
{

/** @brief This class is used to express the local action of a general nonlinear
    finite element operator. In addition it may provide the capability to
    assemble the local gradient operator and to compute the local energy. */
class NonlinearFormIntegrator
{
public:
   enum Mode
   {
      ELEMENTWISE = 0,       /**< Element-wise integration (default) */
      PATCHWISE = 1,         /**< Patch-wise integration (NURBS meshes) */
      PATCHWISE_REDUCED = 2, /**< Patch-wise integration (NURBS meshes) with
                                  reduced integration rules. */
   };

protected:
   const IntegrationRule *IntRule;

   Mode integrationMode = Mode::ELEMENTWISE;

   // Prescribed integration rules (not reduced approximate rules).
   NURBSMeshRules *patchRules = nullptr;

   // CEED extension
   ceed::Operator* ceedOp;

   MemoryType pa_mt = MemoryType::DEFAULT;

   NonlinearFormIntegrator(const IntegrationRule *ir = NULL)
      : IntRule(ir), ceedOp(NULL) { }

public:
   /** @brief Prescribe a fixed IntegrationRule to use (when @a ir != NULL) or
       let the integrator choose (when @a ir == NULL). */
   virtual void SetIntRule(const IntegrationRule *ir) { IntRule = ir; }

   void SetIntegrationMode(Mode m) { integrationMode = m; }

   /// For patchwise integration, SetNURBSPatchIntRule must be called.
   void SetNURBSPatchIntRule(NURBSMeshRules *pr) { patchRules = pr; }
   bool HasNURBSPatchIntRule() const { return patchRules != nullptr; }

   bool Patchwise() const { return integrationMode != Mode::ELEMENTWISE; }

   /// Prescribe a fixed IntegrationRule to use.
   void SetIntegrationRule(const IntegrationRule &ir) { SetIntRule(&ir); }

   /// Set the memory type used for GeometricFactors and other large allocations
   /// in PA extensions.
   void SetPAMemoryType(MemoryType mt) { pa_mt = mt; }

   /// Get the integration rule of the integrator (possibly NULL).
   const IntegrationRule *GetIntegrationRule() const { return IntRule; }

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
   virtual real_t GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Tr,
                                   const Vector &elfun);

   /// Method defining partial assembly.
   /** The result of the partial assembly is stored internally so that it can be
       used later in the methods AddMultPA(). */
   virtual void AssemblePA(const FiniteElementSpace &fes);

   /** The result of the partial assembly is stored internally so that it can be
       used later in the methods AddMultPA().
       Used with BilinearFormIntegrators that have different spaces. */
   virtual void AssemblePA(const FiniteElementSpace &trial_fes,
                           const FiniteElementSpace &test_fes);

   /** @brief Prepare the integrator for partial assembly (PA) gradient
       evaluations on the given FE space @a fes at the state @a x. */
   /** The result of the partial assembly is stored internally so that it can be
       used later in the methods AddMultGradPA() and AssembleGradDiagonalPA().
       The state Vector @a x is an E-vector. */
   virtual void AssembleGradPA(const Vector &x, const FiniteElementSpace &fes);

   /// Compute the local (to the MPI rank) energy with partial assembly.
   /** Here the state @a x is an E-vector. This method can be called only after
       the method AssemblePA() has been called. */
   virtual real_t GetLocalStateEnergyPA(const Vector &x) const;

   /// Method for partially assembled action.
   /** Perform the action of integrator on the input @a x and add the result to
       the output @a y. Both @a x and @a y are E-vectors, i.e. they represent
       the element-wise discontinuous version of the FE space.

       This method can be called only after the method AssemblePA() has been
       called. */
   virtual void AddMultPA(const Vector &x, Vector &y) const;

   /// Method for partially assembled gradient action.
   /** All arguments are E-vectors. This method can be called only after the
       method AssembleGradPA() has been called.

       @param[in]     x  The gradient Operator is applied to the Vector @a x.
       @param[in,out] y  The result Vector: $ y += G x $. */
   virtual void AddMultGradPA(const Vector &x, Vector &y) const;

   /// Method for computing the diagonal of the gradient with partial assembly.
   /** The result Vector @a diag is an E-Vector. This method can be called only
       after the method AssembleGradPA() has been called.

       @param[in,out] diag  The result Vector: $ diag += diag(G) $. */
   virtual void AssembleGradDiagonalPA(Vector &diag) const;

   /// Indicates whether this integrator can use a Ceed backend.
   virtual bool SupportsCeed() const { return false; }

   /// Method defining fully unassembled operator.
   virtual void AssembleMF(const FiniteElementSpace &fes);

   /** Perform the action of integrator on the input @a x and add the result to
       the output @a y. Both @a x and @a y are E-vectors, i.e. they represent
       the element-wise discontinuous version of the FE space.

       This method can be called only after the method AssembleMF() has been
       called. */
   virtual void AddMultMF(const Vector &x, Vector &y) const;

   ceed::Operator& GetCeedOp() { return *ceedOp; }

   virtual ~NonlinearFormIntegrator()
   {
      delete ceedOp;
   }
};

/** The abstract base class BlockNonlinearFormIntegrator is
    a generalization of the NonlinearFormIntegrator class suitable
    for block state vectors. */
class BlockNonlinearFormIntegrator
{
public:
   /// Compute the local energy
   virtual real_t GetElementEnergy(const Array<const FiniteElement *>&el,
                                   ElementTransformation &Tr,
                                   const Array<const Vector *>&elfun);

   /// Perform the local action of the BlockNonlinearFormIntegrator
   virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                      ElementTransformation &Tr,
                                      const Array<const Vector *> &elfun,
                                      const Array<Vector *> &elvec);

   virtual void AssembleFaceVector(const Array<const FiniteElement *> &el1,
                                   const Array<const FiniteElement *> &el2,
                                   FaceElementTransformations &Tr,
                                   const Array<const Vector *> &elfun,
                                   const Array<Vector *> &elvect);

   /// Assemble the local gradient matrix
   virtual void AssembleElementGrad(const Array<const FiniteElement*> &el,
                                    ElementTransformation &Tr,
                                    const Array<const Vector *> &elfun,
                                    const Array2D<DenseMatrix *> &elmats);

   virtual void AssembleFaceGrad(const Array<const FiniteElement *>&el1,
                                 const Array<const FiniteElement *>&el2,
                                 FaceElementTransformations &Tr,
                                 const Array<const Vector *> &elfun,
                                 const Array2D<DenseMatrix *> &elmats);

   virtual ~BlockNonlinearFormIntegrator() { }
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
   /** @note It is assumed that Ttr_.SetIntPoint() is already called for the
       point of interest. */
   void SetTransformation(ElementTransformation &Ttr_) { Ttr = &Ttr_; }

   /** @brief Evaluate the strain energy density function, W = W(Jpt).
       @param[in] Jpt  Represents the target->physical transformation
                       Jacobian matrix. */
   virtual real_t EvalW(const DenseMatrix &Jpt) const = 0;

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
                          const real_t weight, DenseMatrix &A) const = 0;
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
   virtual real_t EvalW(const DenseMatrix &J) const;

   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const real_t weight, DenseMatrix &A) const;
};


/** Neo-Hookean hyperelastic model with a strain energy density function given
    by the formula: $(\mu/2)(\bar{I}_1 - dim) + (K/2)(det(J)/g - 1)^2$ where
    J is the deformation gradient and $$\bar{I}_1 = (det(J))^{-2/dim} Tr(J
    J^t)$$. The parameters $\mu$ and K are the shear and bulk moduli,
    respectively, and g is a reference volumetric scaling. */
class NeoHookeanModel : public HyperelasticModel
{
protected:
   mutable real_t mu, K, g;
   Coefficient *c_mu, *c_K, *c_g;
   bool have_coeffs;

   mutable DenseMatrix Z;    // dim x dim
   mutable DenseMatrix G, C; // dof x dim

   inline void EvalCoeffs() const;

public:
   NeoHookeanModel(real_t mu_, real_t K_, real_t g_ = 1.0)
      : mu(mu_), K(K_), g(g_), have_coeffs(false) { c_mu = c_K = c_g = NULL; }

   NeoHookeanModel(Coefficient &mu_, Coefficient &K_, Coefficient *g_ = NULL)
      : mu(0.0), K(0.0), g(1.0), c_mu(&mu_), c_K(&K_), c_g(g_),
        have_coeffs(true) { }

   virtual real_t EvalW(const DenseMatrix &J) const;

   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const real_t weight, DenseMatrix &A) const;
};


/** Hyperelastic integrator for any given HyperelasticModel.

    Represents $ \int W(Jpt) dx $ over a target zone, where W is the
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
   virtual real_t GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Ttr,
                                   const Vector &elfun);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Ttr,
                                      const Vector &elfun, Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Ttr,
                                    const Vector &elfun, DenseMatrix &elmat);
};

/** Hyperelastic incompressible Neo-Hookean integrator with the PK1 stress
    $P = \mu F - p F^{-T}$ where $\mu$ is the shear modulus,
    $p$ is the pressure, and $F$ is the deformation gradient */
class IncompressibleNeoHookeanIntegrator : public BlockNonlinearFormIntegrator
{
private:
   Coefficient *c_mu;
   DenseMatrix DSh_u, DS_u, J0i, J, J1, Finv, P, F, FinvT;
   DenseMatrix PMatI_u, PMatO_u, PMatI_p, PMatO_p, Z, G, C;
   Vector Sh_p;

public:
   IncompressibleNeoHookeanIntegrator(Coefficient &mu_) : c_mu(&mu_) { }

   virtual real_t GetElementEnergy(const Array<const FiniteElement *>&el,
                                   ElementTransformation &Tr,
                                   const Array<const Vector *> &elfun);

   /// Perform the local action of the NonlinearFormIntegrator
   virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                      ElementTransformation &Tr,
                                      const Array<const Vector *> &elfun,
                                      const Array<Vector *> &elvec);

   /// Assemble the local gradient matrix
   virtual void AssembleElementGrad(const Array<const FiniteElement*> &el,
                                    ElementTransformation &Tr,
                                    const Array<const Vector *> &elfun,
                                    const Array2D<DenseMatrix *> &elmats);
};


class VectorConvectionNLFIntegrator : public NonlinearFormIntegrator
{
private:
   Coefficient *Q{};
   DenseMatrix dshape, dshapex, EF, gradEF, ELV, elmat_comp;
   Vector shape;
   // PA extension
   Vector pa_data;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, nq;

public:
   VectorConvectionNLFIntegrator(Coefficient &q): Q(&q) { }

   VectorConvectionNLFIntegrator() = default;

   static const IntegrationRule &GetRule(const FiniteElement &fe,
                                         ElementTransformation &T);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat);

   using NonlinearFormIntegrator::AssemblePA;

   virtual void AssemblePA(const FiniteElementSpace &fes);

   virtual void AssembleMF(const FiniteElementSpace &fes);

   virtual void AddMultPA(const Vector &x, Vector &y) const;

   virtual void AddMultMF(const Vector &x, Vector &y) const;
};


/** This class is used to assemble the convective form of the nonlinear term
    arising in the Navier-Stokes equations $(u \cdot \nabla v, w )$ */
class ConvectiveVectorConvectionNLFIntegrator :
   public VectorConvectionNLFIntegrator
{
private:
   Coefficient *Q{};
   DenseMatrix dshape, dshapex, EF, gradEF, ELV, elmat_comp;
   Vector shape;

public:
   ConvectiveVectorConvectionNLFIntegrator(Coefficient &q): Q(&q) { }

   ConvectiveVectorConvectionNLFIntegrator() = default;

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat);
};


/** This class is used to assemble the skew-symmetric form of the nonlinear term
    arising in the Navier-Stokes equations
    $.5*(u \cdot \nabla v, w ) - .5*(u \cdot \nabla w, v )$ */
class SkewSymmetricVectorConvectionNLFIntegrator :
   public VectorConvectionNLFIntegrator
{
private:
   Coefficient *Q{};
   DenseMatrix dshape, dshapex, EF, gradEF, ELV, elmat_comp;
   Vector shape;

public:
   SkewSymmetricVectorConvectionNLFIntegrator(Coefficient &q): Q(&q) { }

   SkewSymmetricVectorConvectionNLFIntegrator() = default;

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat);
};

}

#endif
