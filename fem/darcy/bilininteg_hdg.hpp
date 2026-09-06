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

    **What this hook cannot express, which is worth knowing before writing a
    subclass.** Eval() returns ONE number per quadrature point and the
    integrator applies it to both sides of the face, so every stabilization
    reachable through here is *symmetric* in the sign of $u \cdot n$.
    Upwinding is not: the upwinded face flux carries $\tau_\pm =
    \beta|u\cdot n| \pm \tfrac12\alpha(u\cdot n)$, whose antisymmetric half
    is the whole point of it. So a convection-dominated face cannot be served
    by any $\tau$ from this interface, however it is scaled -- measured, and
    the numbers are on HDGConvectiveFloorStabilization. Convective upwinding
    belongs on the convection integrator; this hook is for shaping the
    symmetric part.

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
    * ~~a coefficient that is **anisotropic**~~ -- **this second claim is
      withdrawn; see below.** It read "the *flux* loses order -- 1.49 against
      2.00 at $k = 1$ and $\kappa_\perp/\kappa_\parallel = 10^{-2}$", and a
      floor does not buy that back.

    For the degeneracy the misbehaviour to fear is $\tau \to 0$ and the remedy
    is to refuse the small values and keep the large ones. That is what this
    does. It is a floor and not a replacement, so on faces where the built-in
    value is already big enough -- the ones aligned with the strong direction
    -- nothing changes.

    **What the anisotropic claim got wrong, measured over a longer sequence
    than it was.** The original ran $n = 8$ to $64$, and on this problem every
    mesh in that window is pre-asymptotic. Taken to $n = 256$ on
    `anisodiff -p 11` at $\kappa_\perp/\kappa_\parallel = 10^{-2}$, `-tf 1`
    against `-tf 0` differs by 27% at $n = 8$ and is **identical to every
    printed digit from $n = 128$ onwards**, at $k = 1$ and $k = 2$ alike; the
    flux rate converges to $1.00$ and $2.02$ -- that is $k$ -- with the floor
    and without it. At $k = 2$ the pre-asymptotic rates read 3.28 and 3.25
    before falling to 2.02, so a sweep stopping at $n = 32$ reports the design
    order and is wrong.

    The reason is structural rather than particular to that problem: this floor
    binds only where $\hat n \cdot Q \hat n < \tau_{min} h / \beta$, and the
    built-in value grows like $1/h$, so the binding set shrinks under
    refinement -- on a sheared field it contracts onto the isolated points
    where the field meets a mesh direction. **A floor cannot change the
    scaling of $\tau$; it can only lift faces where the coefficient
    collapses.** Where the collapse is on a fixed set, as in the degeneracy
    above, it lifts a fixed set and the order really is recovered -- which is
    why the degenerate half stands and is pinned to $n = 128$ by
    "HDG: a tau floor recovers the order a degeneracy costs".

    What the anisotropy actually costs, and what recovers what, is the rate
    table on HDGDiffusionIntegrator: an $O(1)$ $\tau$ buys the flux order the
    $O(1/h)$ default gives up, and half an order remains that belongs to the
    anisotropy itself and that no $\tau$ here repairs.

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


/** @brief A stabilization that is never below the CONVECTIVE scale either.

    `HDGFloorStabilization` above answers $\tau \to 0$ arriving from the
    diffusion. This answers the other way a face can be under-stabilized: the
    problem is convection-dominated there, and a $\tau$ built from the
    diffusion alone knows nothing about it.

    $$\tau = \max\left(s_{diff},\ \beta_c |u \cdot n|\right),$$
    optionally floored again by an absolute @a tau_min.

    **It does not do what it looks like it should, and the measurement is the
    reason to keep reading.** The obvious use -- replacing the upwinded face
    flux with a stabilization that carries the same convective magnitude --
    does not work, and not because the magnitude is wrong. On
    `anisodiff -p 11`, order 1, $\kappa_\perp/\kappa_\parallel = 10^{-2}$,
    $c = 100$, $64\times64$, relative $L^2$ flux error:

        beta_c        0.125    0.25     0.5      1        2        4        8       16
        this class    3.555e-3 3.558e-3 3.566e-3 3.580e-3 3.609e-3 3.665e-3 3.772e-3 3.977e-3
        -vs alone     3.551e-3     (the same tau without the convective floor)
        no velocity   3.993e-3     (the plain symmetric built-in)
        upwinded flux 1.467e-3     (HDGConvectionUpwindedIntegrator)

    A 128x sweep of $\beta_c$ moves the error by 12%, *monotonically the wrong
    way*, and interpolates between the two symmetric endpoints rather than
    approaching the upwinded one. At $\beta_c = 16$ the floor binds on every
    face with $|u\cdot n| > 0.02\,|c|$, so it is certainly firing.

    **The obstruction is symmetry, not size.** `Eval` returns one number per
    quadrature point, which the integrator applies to both sides of the face.
    The upwinded flux applies $\tau_\pm = \beta|u\cdot n| \pm
    \tfrac12\alpha(u\cdot n)$, which is *antisymmetric* in the sign of
    $u\cdot n$ -- that is what upwinding is. No scalar returned from this hook
    can express it, whatever it is scaled by.

    Worse, it can destroy an asymmetry that is already there. Constructed as
    `HDGDiffusionIntegrator(v, Q, a)` the built-in carries $\alpha = a$ and
    $\beta = a/2$, so the weight is $\beta \pm \alpha/2$ -- $a$ on the upwind
    side and **exactly zero on the downwind side**. A maximum lifts that
    deliberate zero and symmetrizes the face. Combining this class with
    `HDGConvectionUpwindedIntegrator` therefore diverges rather than
    reinforcing: the same case gives a relative flux error of 7.6e+04.

    So this is a floor for a $\tau$ that is *meant* to be symmetric -- the
    convective analogue of HDGFloorStabilization, useful where the diffusive
    scale collapses and some stabilization is wanted -- and it is **not** a
    route to an upwinded method. For that, put the upwinded integrator on the
    potential mass form.

    Constant, so the bilinear assembly path accepts it. */
class HDGConvectiveFloorStabilization : public HDGStabilization
{
   real_t beta_c, tau_min;

public:
   /** @param beta_c_ the multiple of $|u \cdot n|$ a face must carry; $1/2$
                      matches `HDGConvectionUpwindedIntegrator`'s default.
       @param tau_min_ an absolute floor applied after the maximum, as in
                       HDGFloorStabilization. Zero disables it. */
   HDGConvectiveFloorStabilization(real_t beta_c_ = 0.5, real_t tau_min_ = 0.)
      : beta_c(beta_c_), tau_min(tau_min_) { }

   real_t Eval(real_t s_diff, real_t un, real_t, real_t,
               ElementTransformation &) const override
   {
      const real_t s_conv = beta_c * std::abs(un);
      const real_t s = (s_diff > s_conv) ? s_diff : s_conv;
      return (s > tau_min) ? s : tau_min;
   }
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
    coefficient $u$ is assumed continuous across the faces.

    ### Which $\tau$ recovers which rate, and what this one costs

    **The value built here is $O(1/h)$, and that is a trade rather than a
    defect.** $\tau = \beta (\hat n \cdot Q \hat n)/h$ grows without bound
    under refinement, which is the LDG-H regime that buys a superconvergent
    potential at the price of a suboptimal flux. Measured on
    `anisodiff -p 11` with no flow, quadrilaterals, equal-order $L^2$ flux,
    potential and `DG_Interface` trace all of degree $k$, over $n = 8$ to
    $256$ -- and taken to $256$ because everything below $64$ is still
    pre-asymptotic here:

        tau            aniso        k=1 flux  k=1 pot   k=2 flux  k=2 pot
        beta Q/h       kperp/kpar=1     1.01     ~2.9       1.96     ~3.6
        (this, O(1/h)) kperp/kpar=1e-2  1.00      2.86      2.02      3.98
        O(1)           kperp/kpar=1     2.02      2.03      2.99      2.99
        (beta ~ h)     kperp/kpar=1e-2  1.43      1.92      2.59      2.89

    Read the two blocks against $k+1$, the best a degree-$k$ space can do:

    * **$O(1/h)$ gives the flux $k$ and the potential $k+2$.** One order below
      optimal in the flux, one *above* it in the potential -- the potential is
      superconverging without any postprocessing. If the potential is the
      quantity of interest this is the better default, and it is the default.
    * **$O(1)$ gives $k+1$ in both, and that is the theoretical best.** It is
      reached exactly: 2.02 and 2.99 in the flux, 2.03 and 2.99 in the
      potential, flat over the last three refinements. Get it by scaling
      $\beta \propto h$ -- `anisodiff -td` divided by the cell count is what
      produced the rows above -- not by HDGFloorStabilization, which cannot
      (see below).
    * **Anisotropy costs a further half order in the flux and nothing in the
      potential**, and it is the one loss no $\tau$ scaling here repairs:
      $1.43$ and $2.59$ against $2.02$ and $2.99$, settled at $n = 256$. That
      is $k+\tfrac12$, and it is a separate phenomenon from the scaling above.

    **A floor cannot deliver the $O(1)$ row.** `HDGFloorStabilization` raises
    $\tau$ where the built-in value is small, but the built-in value grows like
    $1/h$, so any fixed floor is overtaken everywhere as the mesh refines. On
    the anisotropic case the floor binds only where
    $\hat n \cdot Q \hat n < \tau_{min} h/\beta$, a set that shrinks with
    $h$; measured, `-tf 1` against `-tf 0` differs by 27% at $n=8$ and is
    **identical to every printed digit at $n = 128$ and $n = 256$**, at both
    orders. It is a repair for faces where the coefficient collapses, not a
    route to a different scaling. */
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
