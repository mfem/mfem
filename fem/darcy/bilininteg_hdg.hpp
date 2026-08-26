// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BILININTEG_HDG
#define MFEM_BILININTEG_HDG

#include "../bilininteg.hpp"

namespace mfem
{

/** Integrator for the DG form:
    $$
      \alpha \langle (u \cdot n) \{v\},[w] \rangle,
    $$
    where $v$ and $w$ are the trial and test variables, respectively, and $\rho$/$u$ are
    given scalar/vector coefficients. $\{v\}$ represents the average value of $v$ on
    the face and $[v]$ is the jump such that $\{v\}=(v_++v_-)/2$ and $[v]=(v_+-v_-)$ for the
    face with $+$ and $-$ sides. For boundary elements, $v_-=0$. The vector coefficient,
    $u$, is assumed to be continuous across the faces and when given the scalar coefficient.

    The corresponding HDG stabilization is then
    $$\begin{align}
        \langle \{\tau\} v_\pm, w_\pm \rangle, & -\langle \tau_\mp \lambda,          w_\pm \rangle,\\
        \langle \{\tau\} v_\pm, \mu   \rangle, & -\langle (\tau_+ + \tau_-) \lambda, \mu   \rangle,
    \end{align}$$
    where $\tau_\pm = |\alpha (u \cdot n)| \pm \alpha (u \cdot n)$
    and $\lambda$, $\mu$ are the trial and test trace functions, respectively.
    */
class HDGConvectionCenteredIntegrator : public DGTraceIntegrator
{
   Vector tr_shape, shape1, shape2;

public:
   HDGConvectionCenteredIntegrator(VectorCoefficient &u_, real_t a = 1.)
      : DGTraceIntegrator(u_, a, 0.) { }

   void AssembleHDGFaceMatrix(const FiniteElement &trace_el,
                              const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceMatrix(int side, const FiniteElement &trace_el,
                              const FiniteElement &el,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceVector(int type,
                              const FiniteElement &trace_face_fe,
                              const FiniteElement &fe,
                              FaceElementTransformations &Tr,
                              const Vector &trfun, const Vector &elfun,
                              Vector &elvect) override;
};

/** Integrator for the DG form:
    $$
      \alpha \langle (u \cdot n) \{v\},[w] \rangle + \beta \langle |u \cdot n| [v],[w] \rangle,
    $$
    where $v$ and $w$ are the trial and test variables, respectively, and $\rho$/$u$ are
    given scalar/vector coefficients. $\{v\}$ represents the average value of $v$ on
    the face and $[v]$ is the jump such that $\{v\}=(v_++v_-)/2$ and $[v]=(v_+-v_-)$ for the
    face with $+$ and $-$ sides. For boundary elements, $v_-=0$. The vector coefficient,
    $u$, is assumed to be continuous across the faces and when given the scalar coefficient.

    The corresponding HDG stabilization is then
    $$\begin{align}
        \langle \tau_\pm v_\pm, w_\pm \rangle, & -\langle \tau_\mp \lambda,          w_\pm \rangle,\\
        \langle \tau_\pm v_\pm, \mu   \rangle, & -\langle (\tau_+ + \tau_-) \lambda, \mu   \rangle,
    \end{align}$$
    where $\tau_\pm = (\beta |u \cdot n| \pm 1/2 \alpha (u \cdot n))$
    and $\lambda$, $\mu$ are the trial and test trace functions, respectively.
    */
class HDGConvectionUpwindedIntegrator : public DGTraceIntegrator
{
   Vector tr_shape, shape1, shape2;

public:
   /// Construct integrator with $\beta = \alpha/2$.
   HDGConvectionUpwindedIntegrator(VectorCoefficient &u_, real_t a = 1.)
      : DGTraceIntegrator(u_, a) { }

   HDGConvectionUpwindedIntegrator(VectorCoefficient &u_, real_t a, real_t b)
      : DGTraceIntegrator(u_, a, b) { }

   void AssembleHDGFaceMatrix(const FiniteElement &trace_el,
                              const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceMatrix(int side, const FiniteElement &trace_el,
                              const FiniteElement &el,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceVector(int type,
                              const FiniteElement &trace_face_fe,
                              const FiniteElement &fe,
                              FaceElementTransformations &Tr,
                              const Vector &trfun, const Vector &elfun,
                              Vector &elvect) override;
};

/** @brief Stabilization function of the HDG numerical flux,
    $$
       \hat q_h + \hat F_h = q_h + F(\hat u_h)
                            + s(u_h, \hat u_h)(u_h - \hat u_h) n .
    $$

    This is Eq. (5) of Nguyen, Peraire and Cockburn, J. Comput. Phys. 228
    (2009) 8841-8855, in which $s$ is a function of the potential and of its
    own trace rather than a coefficient. Their section 2.4 splits it as
    $s = s_{diff} + s_{conv}(u_h, \hat u_h)$ with $s_{diff} = \kappa/\ell$
    constant, and for a linear flux the positivity bound of their Eq. (7)
    reduces $s_{conv}$ to a constant as well -- which is exactly the
    stabilization HDGDiffusionIntegrator and HDGConvectionUpwindedIntegrator
    already apply. The constant case is therefore not a special case bolted on
    here; it is the specialization those integrators implement, and it keeps
    its own assembly path.

    A derived class that leaves IsConstant() true costs nothing at run time:
    the integrators query it once per face, never per quadrature point.

    A class that returns false makes the face term nonlinear in the unknowns
    even for a linear equation, so it is only meaningful on the residual and
    gradient assembly, which are the only paths that see the state. The
    bilinear-form path refuses it rather than silently dropping the
    dependence. */
class HDGStabilization
{
public:
   virtual ~HDGStabilization() { }

   /** @brief True when $s$ depends on neither the potential nor its trace.
       The default is the constant case. */
   virtual bool IsConstant() const { return true; }

   /** @brief The stabilization at one quadrature point.
       @param s_diff the value the integrator forms on its own, that is the
                     constant part built from the diffusion coefficient and the
                     local element size, with any quadrature weight removed
       @param un     the normal component of the convective velocity, unscaled
       @param u      the potential at the point
       @param uhat   the trace of the potential at the point
       @param Tr     element transformation, with the integration point set */
   virtual real_t Eval(real_t s_diff, real_t un, real_t u, real_t uhat,
                       ElementTransformation &Tr) const
   { return s_diff; }

   /** @brief The derivatives of $s$ with respect to the potential and to its
       trace, written $\partial_1 s$ and $\partial_2 s$ in Eq. (15) of the
       reference. Called only when IsConstant() is false.

       These are not optional refinements. In a hybridized method the Jacobian
       is never assembled globally, so omitting them gives no wrong answer,
       only slow Newton convergence -- a failure that survives a passing
       regression suite. */
   virtual void EvalGrad(real_t s_diff, real_t un, real_t u, real_t uhat,
                         ElementTransformation &Tr,
                         real_t &d1s, real_t &d2s) const
   { d1s = 0.; d2s = 0.; }
};

/** @brief A stabilization that is not allowed below a floor.

    `HDGDiffusionIntegrator` builds its own value as
    $\tau = \beta\,(\hat n \cdot Q \hat n)/h$, which vanishes wherever the
    diffusion does *in the direction of the face normal*. Two separate findings
    on this branch run into that, and both are recorded in section 3 of
    `doc/HDG-ROADMAP.md`:

    * a coefficient that **degenerates** on part of the boundary, where
      $Q \to 0$ and the potential loses order -- 2.18 against a clean 2.99 at
      $k = 2$;
    * a coefficient that is **anisotropic**, where $Q$ does not vanish at all
      but $\hat n \cdot Q \hat n$ is $\kappa_\perp$ on a face whose normal
      lies across the field, and the *flux* loses order -- 1.49 against 2.00 at
      $k = 1$ and $\kappa_\perp/\kappa_\parallel = 10^{-2}$.

    In both the misbehaviour to fear is $\tau \to 0$ rather than
    $\tau \to \infty$, and in both the remedy is the same: refuse the small
    values and keep the large ones. That is what this does. It is a floor and
    not a replacement, so on faces where the built-in value is already big
    enough -- the ones aligned with the strong direction -- nothing changes.

    The floor is an absolute stabilization, so it is the $\eta_d = \kappa/\ell$
    of Nguyen, Peraire and Cockburn section 3.6.3 with $\ell$ a fixed problem
    length scale, which is the scaling that holds $\tau$ constant under
    refinement. It is therefore a number of the size of
    $\kappa_\parallel/\ell$, not of $\kappa_\perp$.

    Constant, so it costs nothing per quadrature point and the bilinear
    assembly path accepts it. */
class HDGFloorStabilization : public HDGStabilization
{
   real_t tau_min;

public:
   /// @param tau_min_ the smallest stabilization any face may be given.
   HDGFloorStabilization(real_t tau_min_) : tau_min(tau_min_) { }

   real_t Eval(real_t s_diff, real_t, real_t, real_t,
               ElementTransformation &) const override
   { return (s_diff > tau_min) ? s_diff : tau_min; }
};


/** Integrator for the H/LDG diffusion stabilization term
    The LDG stabilization takes the form
    $$
        1/2 \beta \langle \{h^{-1} Q\} [v], [w] \rangle
    $$
    where $Q$ is a scalar or matrix diffusion coefficient and $v$, $w$ are the trial
    and test functions, respectively.

    The corresponding HDG stabilization is then
    $$\begin{align}
        \langle \tau_\pm v_\pm, w_\pm \rangle, & -\langle \tau_\pm \lambda,          w_\pm \rangle,\\
        \langle \tau_\pm v_\pm, \mu   \rangle, & -\langle (\tau_+ + \tau_-) \lambda, \mu   \rangle,
    \end{align}$$
    where $\tau_\pm = (\beta \pm 1/2 \alpha (u \cdot n) / |u \cdot n|) \{h^{-1} Q\}$
    and $\lambda$, $\mu$ are the trial and test trace functions, respectively. The vector
    coefficient $u$ is assumed continuous across the faces. */
class HDGDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient *v;
   Coefficient *Q;
   MatrixCoefficient *MQ;
   real_t alpha, beta;
   const HDGStabilization *stab{};

   /** @brief The weighted stabilization at one quadrature point.

       With no user object this is the built-in expression, unchanged and with
       no call. With one, the quadrature weight is divided out so that the
       object sees s itself, and put back afterwards. @a u and @a uhat are only
       meaningful where the state is available; the bilinear paths pass zero,
       which is why they insist the object be constant. */
   inline real_t StabValue(real_t wq, real_t ba, real_t un, real_t face_w,
                           real_t u, real_t uhat,
                           ElementTransformation &Tr) const
   {
      if (!stab) { return wq * ba; }
      const real_t s_diff = (face_w != 0.) ? (wq * ba / face_w) : 0.;
      return face_w * stab->Eval(s_diff, un, u, uhat, Tr);
   }

   // these are not thread-safe!
   Vector tr_shape, shape1, shape2, vu, nor, nh, ni;
   Vector nor_Jt, nor_Ji, ni_Jt, ni_Ji;
   DenseMatrix mq;

public:
   /// Construct integrator with $\alpha = 0$ and $\beta = a$.
   HDGDiffusionIntegrator(const real_t a = 0.5)
      : v(NULL), Q(NULL), MQ(NULL), alpha(0.), beta(a) { }

   /// Construct integrator with $\alpha = 0$ and $\beta = a$.
   HDGDiffusionIntegrator(Coefficient &q, const real_t a = 0.5)
      : v(NULL), Q(&q), MQ(NULL), alpha(0.), beta(a) { }

   /// Construct integrator with $\alpha = 0$ and $\beta = a$.
   HDGDiffusionIntegrator(MatrixCoefficient &q, const real_t a = 0.5)
      : v(NULL), Q(NULL), MQ(&q), alpha(0.), beta(a) { }

   /// Construct integrator with $\alpha = a$ and $\beta = a/2$.
   HDGDiffusionIntegrator(VectorCoefficient &v_, const real_t a = 0.5)
      : v(&v_), Q(NULL), MQ(NULL), alpha(a), beta(0.5*a) { }

   /// Construct integrator with $\alpha = a$ and $\beta = a/2$.
   HDGDiffusionIntegrator(VectorCoefficient &v_, Coefficient &q,
                          const real_t a = 0.5)
      : v(&v_), Q(&q), MQ(NULL), alpha(a), beta(0.5*a) { }

   /// Construct integrator with $\alpha = a$ and $\beta = a/2$.
   HDGDiffusionIntegrator(VectorCoefficient &v_, MatrixCoefficient &q,
                          const real_t a = 0.5)
      : v(&v_), Q(NULL), MQ(&q), alpha(a), beta(0.5*a) { }

   /** @brief Replace the stabilization with a user supplied one.

       The object is referenced, not owned, and must outlive the integrator.
       Leaving it unset keeps the built-in constant stabilization and the
       assembly path that goes with it. */
   void SetStabilization(const HDGStabilization &s) { stab = &s; }

   const HDGStabilization *GetStabilization() const { return stab; }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

   void AssembleHDGFaceMatrix(const FiniteElement &trace_el,
                              const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceMatrix(int side, const FiniteElement &trace_el,
                              const FiniteElement &el,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceVector(int type,
                              const FiniteElement &trace_face_fe,
                              const FiniteElement &fe,
                              FaceElementTransformations &Tr,
                              const Vector &trfun, const Vector &elfun,
                              Vector &elvect) override;

   /** @brief The gradient of the face residual.

       Overridden rather than inherited because the base class builds it from
       AssembleHDGFaceMatrix(), which cannot see the state and so cannot carry
       the derivatives of a solution dependent stabilization. */
   void AssembleHDGFaceGrad(int type,
                            const FiniteElement &trace_face_fe,
                            const FiniteElement &fe,
                            FaceElementTransformations &Tr,
                            const Vector &trfun, const Vector &elfun,
                            DenseMatrix &elmat) override;

   real_t ComputeHDGFaceEnergy(int side,
                               const FiniteElement &trace_face_fe,
                               const FiniteElement &fe,
                               FaceElementTransformations &Tr,
                               const Vector &trfun, const Vector &elfun,
                               Vector *d_energy = NULL) override;
};

}

#endif //MFEM_BILININTEG_HDG
