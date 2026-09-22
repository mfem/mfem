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

#ifndef MFEM_HDG_NSFLUX
#define MFEM_HDG_NSFLUX

#include "mfem.hpp"
#include <cmath>

namespace mfem
{
namespace hdg
{

/** @brief The inviscid flux of the incompressible Navier-Stokes equations in
    Chorin's artificial-compressibility form.

    The state is the primitive vector
    \verbatim
        u = (p, v_1, ..., v_d),     num_equations = dim + 1,
    \endverbatim
    with the pressure first so that the layout mirrors EulerFlux's
    `(rho, rho v, rho E)`: index 0 is the continuity variable and indices
    `1 .. dim` the momentum ones. Swapping this class for EulerFlux is then the
    whole of the change needed to move the miniapp to the compressible
    equations, which is why the pressure is not last.

    The flux is
    \verbatim
        F_{0,d}   = beta v_d,
        F_{1+i,d} = v_i v_d + p delta_{id},      i = 0 .. dim-1,
    \endverbatim
    so that the steady system `div F(u) + div G(u,q) = s` is

    \verbatim
        beta div v                     = s_0,
        div (v_i v) + d_i p - nu lap v_i = s_{1+i}.
    \endverbatim

    **beta does not perturb the steady answer.** The continuity row is
    `beta div v = s_0`, so at a root of the steady residual `div v = s_0/beta`
    exactly, whatever beta is; the incompressibility is imposed, not penalised.
    What beta changes is the *character* of the system, through the eigenvalues
    of the normal flux Jacobian

    \verbatim
        A_n = [ 0    beta n^T          ]
              [ n    (v.n) I + v n^T   ],
    \endverbatim
    which are `v.n` with multiplicity `dim-1` and
    `(v.n) +- sqrt((v.n)^2 + beta |n|^2)`. Hence

    \verbatim
        lambda_max(u,n) = |v.n| + sqrt((v.n)^2 + beta |n|^2),
    \endverbatim
    On a face whose normal lies along the flow that is `|v| + sqrt(v^2 + beta)`
    and on one across the flow it is `sqrt(beta)`, so this one expression is
    both a convective and a diffusive stabilization according to where it is
    evaluated. That is the open question the navierstokes miniapp was built to
    ask, and `-tau <c>` there swaps in the library's constant-`Ctau` HDGFlux to
    ask it.

    **What the answer turned out to be, because an earlier version of this
    comment asserted the opposite and was wrong.** It claimed a single constant
    `tau` "cannot be both" and that HDGLaxFriedrichsFlux is therefore the right
    stabilization here. Measured over 937 runs, a constant near 0.5 is
    *2.0-3.6x more accurate* than `lambda_max` in the flux and the pressure and
    indistinguishable in the potential; `lambda_max` wins only on nonlinear
    solvability, where it converges on coarse meshes at high Reynolds number
    that every constant `tau <= 1` diverges on. The reason is that
    `lambda_max = sqrt(beta)` on every face where `v.n = 0`, and on the
    along-flow faces -- the only ones where it is larger -- the miniapp's exact
    solutions are already representable, so the extra weight is a pure penalty.
    A sweep of `beta`, which cannot change the steady answer, confirmed it:
    `lambda_max`'s error moves with `beta` and tracks the constant `sqrt(beta)`
    to 5-16%, while a fixed `tau` is `beta`-independent to 0.02%. The miniapp's
    header comment carries the tables. So the honest statement is that
    `lambda_max`'s value here is robustness, not accuracy, and that its
    accuracy level is set by `beta` -- a free parameter of the formulation --
    rather than by the flow.

    The Jacobians are analytic. Nothing in FluxFunction finite-differences a
    Jacobian you do not supply -- the default implementations MFEM_ABORT -- so
    a missing one is a crash, not a silent degradation. The reverse is the risk
    worth guarding: a *wrong* analytic Jacobian is silent under hybridization,
    because the reduced Jacobian is never assembled globally and the only
    symptom is slow Newton convergence. See the finite-difference check in
    tests/unit/fem/test_hdg_nsflux.cpp -- not tests/unit/miniapps/, whose
    sources the unit test makefile filters out of libtests.o. */
class ArtificialCompressibilityFlux : public FluxFunction
{
   const real_t beta;
   bool stokes;

public:
   /** @param dim_    spatial dimension
       @param beta_   artificial compressibility, `beta > 0`. It sets the
                      pseudo-acoustic speed `sqrt(beta)` and so the floor of
                      the stabilization; it does not change the steady
                      solution.
       @param stokes_ drop the `v (x) v` term, leaving the Stokes problem. The
                      flux is then linear in the state, so Newton converges in
                      one step and every remaining term -- the pressure
                      coupling, the viscous blocks, the trace treatment -- is
                      exercised without the convective one. It is the control
                      that says whether a discrepancy is convection's fault. */
   ArtificialCompressibilityFlux(int dim_, real_t beta_ = 1.,
                                 bool stokes_ = false)
      : FluxFunction(dim_ + 1, dim_), beta(beta_), stokes(stokes_)
   {
      MFEM_VERIFY(beta_ > 0., "The artificial compressibility must be positive.");
   }

   real_t GetBeta() const { return beta; }
   bool IsStokes() const { return stokes; }

   /** @brief Turn the `v (x) v` term on or off between solves.

       Nothing caches the flux -- HyperbolicFormIntegrator asks for it afresh
       at every quadrature point of every residual and Jacobian -- so flipping
       this between two solves of the same DarcyForm gives Stokes continuation
       for free: solve the linear problem, then continue onto the nonlinear one
       from its answer. On the Kovasznay problem at Re = 40 that is the
       difference between a cold Newton that diverges and one that does not. */
   void SetStokes(bool s) { stokes = s; }

   /// The pressure component of a state vector.
   static inline real_t Pressure(const Vector &state) { return state(0); }

   /// @brief `lambda_max` for the *scaled* normal @a nor.
   /** The scaling is linear in `|nor|`, so passing the unnormalised normal
       that CalcOrtho() produces carries the face weight through and the
       stabilization needs no separate weighting. */
   real_t MaxCharSpeedDotN(const Vector &state, const Vector &nor) const
   {
      real_t vn = 0., nn = 0.;
      for (int d = 0; d < dim; d++)
      {
         vn += state(1 + d) * nor(d);
         nn += nor(d) * nor(d);
      }
      // Dropping v (x) v drops the (v.n) I + v n^T block of A_n with it, so
      // the eigenvalues collapse to +- sqrt(beta) |n| and the stabilization
      // stops depending on the state at all.
      if (stokes) { return std::sqrt(beta * nn); }
      return std::abs(vn) + std::sqrt(vn * vn + beta * nn);
   }

   /** @brief The derivative of MaxCharSpeedDotN() with respect to the state.
       Zero in the pressure component; `n_j (sgn(v.n) + (v.n)/r)` in the
       velocity ones, with `r = sqrt((v.n)^2 + beta |n|^2)`. */
   void MaxCharSpeedDotNGrad(const Vector &state, const Vector &nor,
                             Vector &dlambda) const
   {
      dlambda.SetSize(num_equations);
      dlambda = 0.;
      if (stokes) { return; }   // S is constant, so its derivative vanishes
      real_t vn = 0., nn = 0.;
      for (int d = 0; d < dim; d++)
      {
         vn += state(1 + d) * nor(d);
         nn += nor(d) * nor(d);
      }
      const real_t r = std::sqrt(vn * vn + beta * nn);
      // r >= sqrt(beta*nn) > 0 on a face of nonzero measure, so the division
      // is safe; sgn(0) is taken as 0, which is the subgradient |.| has there.
      const real_t s = ((vn > 0.) ? 1. : ((vn < 0.) ? -1. : 0.)) + vn / r;
      for (int d = 0; d < dim; d++) { dlambda(1 + d) = s * nor(d); }
   }

   real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override
   {
      MFEM_ASSERT(state.Size() == num_equations, "");
      flux.SetSize(num_equations, dim);
      const real_t p = state(0);

      real_t vv = 0.;
      for (int d = 0; d < dim; d++)
      {
         flux(0, d) = beta * state(1 + d);
         vv += state(1 + d) * state(1 + d);
      }
      for (int i = 0; i < dim; i++)
         for (int d = 0; d < dim; d++)
         {
            flux(1 + i, d) = ((i == d) ? p : 0.)
                             + (stokes ? 0. : state(1 + i) * state(1 + d));
         }

      if (stokes) { return std::sqrt(beta); }
      const real_t v = std::sqrt(vv);
      return v + std::sqrt(vv + beta);
   }

   real_t ComputeFluxDotN(const Vector &state, const Vector &nor,
                          FaceElementTransformations &Tr,
                          Vector &fluxDotN) const override
   {
      MFEM_ASSERT(state.Size() == num_equations, "");
      fluxDotN.SetSize(num_equations);
      const real_t p = state(0);

      real_t vn = 0., nn = 0.;
      for (int d = 0; d < dim; d++)
      {
         vn += state(1 + d) * nor(d);
         nn += nor(d) * nor(d);
      }

      fluxDotN(0) = beta * vn;
      for (int i = 0; i < dim; i++)
      {
         fluxDotN(1 + i) = p * nor(i) + (stokes ? 0. : state(1 + i) * vn);
      }

      if (stokes) { return std::sqrt(beta * nn); }
      return std::abs(vn) + std::sqrt(vn * vn + beta * nn);
   }

   void ComputeFluxJacobian(const Vector &state, ElementTransformation &Tr,
                            DenseTensor &J) const override
   {
      MFEM_ASSERT(state.Size() == num_equations, "");
      J.SetSize(num_equations, num_equations, dim);
      J = 0.;

      for (int d = 0; d < dim; d++)
      {
         // continuity row: d(beta v_d)/d v_j = beta delta_{jd}
         J(0, 1 + d, d) = beta;

         for (int i = 0; i < dim; i++)
         {
            // d(v_i v_d + p delta_{id})/dp
            if (i == d) { J(1 + i, 0, d) = 1.; }
            if (stokes) { continue; }
            // d(v_i v_d)/d v_j = delta_{ij} v_d + v_i delta_{jd}
            J(1 + i, 1 + i, d) += state(1 + d);
            J(1 + i, 1 + d, d) += state(1 + i);
         }
      }
   }

   /** @brief `A_n = sum_d n_d A_d`, formed directly.

       Overridden rather than inherited because the base class default
       allocates the full `(m, m, dim)` tensor and contracts it at every
       quadrature point, and this is called on every face of every Newton
       step. */
   void ComputeFluxJacobianDotN(const Vector &state, const Vector &nor,
                                ElementTransformation &Tr,
                                DenseMatrix &JDotN) const override
   {
      MFEM_ASSERT(state.Size() == num_equations, "");
      JDotN.SetSize(num_equations);
      JDotN = 0.;

      real_t vn = 0.;
      for (int d = 0; d < dim; d++) { vn += state(1 + d) * nor(d); }

      for (int j = 0; j < dim; j++) { JDotN(0, 1 + j) = beta * nor(j); }

      for (int i = 0; i < dim; i++)
      {
         JDotN(1 + i, 0) = nor(i);
         if (stokes) { continue; }
         JDotN(1 + i, 1 + i) += vn;
         for (int j = 0; j < dim; j++)
         {
            JDotN(1 + i, 1 + j) += state(1 + i) * nor(j);
         }
      }
   }
};

/** @brief The HDG numerical flux with the local Lax-Friedrichs stabilization
    matrix, `S = lambda_max(uhat, n) I`.

    This is Eq. (3) with Eq. (6) of Peraire, Nguyen and Cockburn, *A
    hybridizable discontinuous Galerkin method for the compressible Euler and
    Navier-Stokes equations*, AIAA 2010-363:
    \verbatim
        Fhat(uhat, u) . n = F(uhat) . n + S(uhat) (u - uhat).
    \endverbatim

    It differs from the library's HDGFlux, which is the same expression with a
    **constant** `S = Ctau I`, in that `S` follows the state and the face
    normal. For a system whose characteristic speeds differ by orders of
    magnitude between the along-flow and across-flow directions -- which is the
    whole point of the Poiseuille problem, and section 5 of the roadmap -- a
    constant `Ctau` is either far too large across the flow or far too small
    along it. Which of those actually costs accuracy is a question for
    measurement, and running the same problem through HDGFlux and through this
    class is how the miniapp asks it (`-tau 0` against `-tau <c>`).

    **Argument convention.** The HDG path in HyperbolicFormIntegrator calls
    `Average(state_tr, state_el, ...)`, so throughout this class
    `state1 == uhat` (the trace) and `state2 == u` (the element). That is not a
    guess: fem/hyperbolic.cpp passes them in that order, and AverageGrad's
    `side == 1` therefore means the derivative with respect to the *trace*.

    Deriving from RusanovFlux supplies the two-state `Eval`/`Grad` used by the
    non-hybridized DG path, which this class does not otherwise touch. */
class HDGLaxFriedrichsFlux : public RusanovFlux
{
   const ArtificialCompressibilityFlux *acf;
   bool frozen_stab;

   /// `S` at the trace state, for the scaled normal @a nor.
   real_t Stab(const Vector &uhat, const Vector &nor,
               FaceElementTransformations &Tr, Vector *dS = NULL) const
   {
      if (acf)
      {
         if (dS) { acf->MaxCharSpeedDotNGrad(uhat, nor, *dS); }
         return acf->MaxCharSpeedDotN(uhat, nor);
      }
      // A general FluxFunction reports its maximum characteristic speed as the
      // return value of ComputeFluxDotN, which is all a generic S can use. The
      // derivative of that number is not available through the interface, so
      // the generic path is necessarily the frozen one.
      Vector fN(fluxFunction.num_equations);
      if (dS) { dS->SetSize(fluxFunction.num_equations); *dS = 0.; }
      return fluxFunction.ComputeFluxDotN(uhat, nor, Tr, fN);
   }

public:
   /** @param f  the flux function. When it is an
                 ArtificialCompressibilityFlux the exact `dS/duhat` is
                 available and is used; for any other FluxFunction the
                 stabilization is frozen (see SetFrozenStabilizationJacobian). */
   HDGLaxFriedrichsFlux(const FluxFunction &f)
      : RusanovFlux(f),
        acf(dynamic_cast<const ArtificialCompressibilityFlux*>(&f)),
        frozen_stab(false) { }

   /** @brief Drop the `(u - uhat) (dS/duhat)^T` term from the trace Jacobian.

       The term is exact and cheap, so the default is to keep it. The switch
       exists because dropping it is the usual thing to do and its cost has
       never been measured on this branch: it cannot change the answer, only
       the Newton history, and `-fstab` is how the miniapp measures that. */
   void SetFrozenStabilizationJacobian(bool f = true) { frozen_stab = f; }

   real_t Average(const Vector &state1, const Vector &state2,
                  const Vector &nor, FaceElementTransformations &Tr,
                  Vector &flux) const override
   {
      const int neq = fluxFunction.num_equations;
      flux.SetSize(neq);

      const real_t speed = fluxFunction.ComputeFluxDotN(state1, nor, Tr, flux);
      const real_t S = Stab(state1, nor, Tr);

      for (int i = 0; i < neq; i++)
      {
         flux(i) += S * (state2(i) - state1(i));
      }
      return std::max(speed, S);
   }

   void AverageGrad(int side, const Vector &state1, const Vector &state2,
                    const Vector &nor, FaceElementTransformations &Tr,
                    DenseMatrix &grad) const override
   {
      MFEM_ASSERT(side == 1 || side == 2, "Unknown side");
      const int neq = fluxFunction.num_equations;
      grad.SetSize(neq);

      if (side == 2)
      {
         // d/du of  F(uhat).n + S(uhat) (u - uhat)  =  S I.
         const real_t S = Stab(state1, nor, Tr);
         grad = 0.;
         for (int i = 0; i < neq; i++) { grad(i, i) = S; }
         return;
      }

      // side == 1: d/duhat = dF(uhat).n/duhat - S I + (u - uhat) (dS/duhat)^T
      Vector dS;
      const real_t S = Stab(state1, nor, Tr, frozen_stab ? NULL : &dS);

      fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, grad);
      for (int i = 0; i < neq; i++) { grad(i, i) -= S; }

      if (!frozen_stab && dS.Size() == neq)
      {
         for (int i = 0; i < neq; i++)
         {
            const real_t du = state2(i) - state1(i);
            if (du == 0.) { continue; }
            for (int j = 0; j < neq; j++) { grad(i, j) += du * dS(j); }
         }
      }
   }
};


/** @brief The HDG boundary numerical flux, LINEARISED about a prescribed
    exterior state -- the cure for the outflow's family of roots.

    **What it repairs.** On a boundary face whose trace is not essential, the
    trace row reads `<(F^(u,uhat) + q^).n, mu> = <(F + G).n, mu>_datum`, and
    `HDGLaxFriedrichsFlux` puts `F(uhat).n + S(uhat)(u - uhat)` on the left.
    Both halves are QUADRATIC in `uhat` for this flux -- `v_i (v.n)` in the
    first and `S(uhat) uhat` in the second -- so per quadrature point the
    momentum row fixes `vhat.n` only up to sign, every free outlet velocity dof
    enters its own row quadratically, and the number of roots is combinatorial.
    Measured before this class existed, order 2 on 8x8 plane Poiseuille from
    fifteen uniform-stream starts: FOUR distinct roots, each converged
    quadratically to 1e-16, at 1.31, 3.30 and 5.63 relative in `q` besides the
    true one, and six divergences. They are not solutions of the continuous
    problem -- refining does not converge them.

    **What it does.** Freeze the flux and the stabilization at the prescribed
    exterior state `w = u_inf(x)`, so the row becomes

        F(w).n + A_n(w) (uhat - w) + S(w) (u - uhat)

    which is exactly LINEAR in `uhat` and in `u`. Consistency is unchanged:
    at `uhat = w` this is `F(w).n + S(w)(u - w)`, which is what the
    unlinearised flux gives there, so the exact solution remains a root of the
    same datum. And `S` frozen at `w` costs nothing in the Jacobian -- the
    `(u - uhat) dS/duhat^T` term that HDGLaxFriedrichsFlux carries is
    identically absent here rather than dropped, so `-fstab` has no meaning on
    this flux.

    **Why not the reference's characteristic condition**, `B^ = A+_n(u - uhat)
    - A-_n(u_inf - uhat)`, which was the other candidate and which the
    boundary-condition step of navierstokes.cpp used to name as the only cure.
    That form REPLACES the trace row rather than supplying a term of it, and
    the row also carries `q^.n` from the constraint block `C`, which
    `DarcyForm` installs from `B`'s boundary marker and which is needed by the
    FLUX row on the same face. Suppressing `C` in one row and not the other is
    machinery `DarcyHybridization` does not have. **Linearising the flux
    reaches the same end -- a row linear in `uhat`, hence one root -- without
    it**, and keeps the viscous half, which a viscous outflow wants anyway.

    @note This is a boundary flux and nothing else. On an interior face `w`
    has no meaning; register it with `AddBdrFaceIntegrator()` on the
    attributes carrying a free trace component and leave the interior to the
    ordinary flux. */
class HDGLinearisedBdrFlux : public RusanovFlux
{
   const ArtificialCompressibilityFlux *acf;
   VectorCoefficient &uinf;   ///< the prescribed exterior state
   real_t tau_const;          ///< >= 0 uses it; < 0 uses S(w)

   mutable Vector w;
   mutable DenseMatrix An;

   /// The exterior state at the current point, and the stabilization there.
   real_t Exterior(const Vector &nor, FaceElementTransformations &Tr) const
   {
      // Elem1 at its own integration point is the physical point on the face,
      // and is an element transformation, so a coefficient reading element
      // attributes still works.
      uinf.Eval(w, *Tr.Elem1, Tr.GetElement1IntPoint());
      if (tau_const >= 0.) { return tau_const * nor.Norml2(); }
      return acf ? acf->MaxCharSpeedDotN(w, nor) : 0.;
   }

public:
   /** @param f    the same FluxFunction the rest of the residual uses
       @param u    the prescribed exterior state, `num_equations` wide
       @param tau  a constant stabilization, or negative for `S(w)` -- match
                   whichever the interior faces are using */
   HDGLinearisedBdrFlux(const FluxFunction &f, VectorCoefficient &u,
                        real_t tau = -1.)
      : RusanovFlux(f),
        acf(dynamic_cast<const ArtificialCompressibilityFlux*>(&f)),
        uinf(u), tau_const(tau) { }

   real_t Average(const Vector &state1, const Vector &state2,
                  const Vector &nor, FaceElementTransformations &Tr,
                  Vector &flux) const override
   {
      const int neq = fluxFunction.num_equations;
      flux.SetSize(neq);

      const real_t S = Exterior(nor, Tr);
      const real_t speed = fluxFunction.ComputeFluxDotN(w, nor, Tr, flux);
      fluxFunction.ComputeFluxJacobianDotN(w, nor, Tr, An);

      // state1 is the TRACE and state2 the element state; that is the order
      // HyperbolicFormIntegrator::AssembleHDGFaceVector() calls with.
      for (int i = 0; i < neq; i++)
      {
         real_t a = 0.;
         for (int j = 0; j < neq; j++) { a += An(i, j) * (state1(j) - w(j)); }
         flux(i) += a + S * (state2(i) - state1(i));
      }
      return std::max(speed, S);
   }

   void AverageGrad(int side, const Vector &state1, const Vector &state2,
                    const Vector &nor, FaceElementTransformations &Tr,
                    DenseMatrix &grad) const override
   {
      MFEM_ASSERT(side == 1 || side == 2, "Unknown side");
      const int neq = fluxFunction.num_equations;
      grad.SetSize(neq);

      const real_t S = Exterior(nor, Tr);
      if (side == 2)
      {
         grad = 0.;
         for (int i = 0; i < neq; i++) { grad(i, i) = S; }
         return;
      }

      // side == 1, the trace: d/duhat of the whole row is A_n(w) - S I, and
      // it does not depend on either state -- which is the point.
      fluxFunction.ComputeFluxJacobianDotN(w, nor, Tr, grad);
      for (int i = 0; i < neq; i++) { grad(i, i) -= S; }
   }
};

/** @brief The CHARACTERISTIC boundary condition of Peraire, Nguyen & Cockburn,
    \verbatim
        B^ = A+_n (u - u^) - A-_n (u_inf - u^) = 0,
    \endverbatim
    as an HDG boundary numerical flux.

    `A_n` is the inviscid flux Jacobian normal to the face and `A+_n`, `A-_n`
    its positive and negative parts, so each characteristic direction takes its
    datum from the side the information comes from: the outgoing ones from the
    interior state @a u, the incoming ones from the prescribed far field
    @a u_inf. **That is what makes it usable on a problem whose answer is not
    known**, and it is the whole reason to have it -- the alternative already
    here, HDGPrescribedFluxLFIntegrator, supplies `<(F + G).n, mu>` from the
    EXACT solution and is therefore a verification device rather than a
    boundary condition.

    **THIS ROW REPLACES THE CONSERVATIVITY CONDITION, it does not add to it**,
    and that is why it needs DarcyHybridization::SetTraceBCAttributes(). An
    ordinary boundary trace row reads `<(F^ + q^).n, mu> = 0`, whose `q^.n`
    half comes from the flux constraint C; this condition carries no viscous
    flux at all, so C has to leave the row. No integrator can do that -- the
    potential mass form is in (p, lambda) and cannot cancel a term in q --
    which is the asymmetry that method exists for. A caller that registers this
    condition and does NOT mark the attribute gets `B^ + q^.n = 0`, which is a
    different and wrong condition, and nothing will say so.

    **Register it through HDGCharacteristicBdrIntegrator, never as a plain
    HyperbolicFormIntegrator.** That integrator makes one `Average()` call per
    quadrature point and writes the result into every row group asked of it,
    so this class would replace the ELEMENT row's flux as well -- and the
    element equation's boundary term must remain the physical numerical trace
    of the flux whatever condition is imposed on the trace.

    **Written in the form that costs one matrix square.** With
    `A+ = (A + |A|)/2` and `A- = (A - |A|)/2` the residual collapses to
    \verbatim
        B^ = ( A_n (u - u_inf) + |A_n| (u + u_inf - 2 u^) ) / 2,
    \endverbatim
    whose derivatives are `dB^/du^ = -|A_n|` and `dB^/du = A+_n`, both
    independent of the state. `A_n` is frozen at @a u_inf, so the row is LINEAR
    in both arguments -- which is what the reference's condition is, and the
    measurement note below says what that does and does not buy.

    **|A_n| in closed form, because this build has no LAPACK.** For the
    artificial-compressibility system the normal Jacobian
    \verbatim
        A_n = [ 0    beta n^T              ]
              [ n    (v.n) I + v n^T       ]
    \endverbatim
    has eigenvalues `un +- c` and `un` (multiplicity dim-1), with
    `un = v.n` and `c = sqrt(un^2 + beta |n|^2)` -- the same `c` that
    ArtificialCompressibilityFlux::MaxCharSpeedDotN() returns `|un| + c` from.
    `c > |un|` for `beta > 0`, so `un + c > 0 > un - c` and the three
    eigenvalues are always distinct; summing the spectral projectors and
    collecting powers of `M = A_n - un I` gives

    \verbatim
        |A_n| = ((c - |un|)/c^2) M^2 + (un/c) M + |un| I.
    \endverbatim

    Checked on each eigenvector rather than taken on faith: `M` annihilates the
    transverse modes, leaving `|un| I` on them; and `M r_+- = +-c r_+-` gives
    `(c - |un|) +- un + |un| = c +- un = |lambda_+-|`. Under `-stokes` the
    `(v.n) I + v n^T` block is gone, `un` is 0 and the expression reduces to
    `A_n^2 / c` with `c = sqrt(beta) |n|`, which is the right answer for a
    matrix whose eigenvalues are `+- c` and 0.

    **AND THAT ZERO IS A DEGENERACY, not a curiosity: where the flow stagnates
    this condition supplies no equation at all.** The eigenvalue `un` carries
    multiplicity `dim-1`, so at `un = 0` the left eigenvector `l` of the zero
    eigenvalue satisfies `l A_n = 0`, and then

    \verbatim
        l.B^ = ( l.A_n (u - u_inf) + l.|A_n| (u + u_inf - 2 u^) ) / 2 = 0
    \endverbatim

    identically -- both terms vanish, the first because `l A_n = 0` and the
    second because `|A_n|` is built from powers of `M = A_n` there. The
    TANGENTIAL velocity trace is left with an empty row and the trace system is
    singular. It is the condition that degenerates, not the discretisation:
    a characteristic with zero speed carries no information, so there is
    nothing for it to say.

    The symptom names nothing useful -- a NaN out of the first trace solve,
    reported as `IsFinite(norm)` in `NewtonSolver::Mult`. `-stokes` and `-cont`
    make it total (the convective flux is dropped, so `un` is zero on every
    face) and navierstokes.cpp refuses both by name. A COLD start meets it for
    one step only, and whether that is survivable is a property of the mesh
    rather than of the method: plane Poiseuille at order 2 reaches the exact
    answer from rest at `-r 1` and diverges at `-r 2`. Starting from a moving
    state is the fix and costs one flag.

    **WHAT THIS IS FOR, and what ONE MESH would have had it do.** Before it
    was built this was written up as NOT a cure for the spurious root family a
    free outlet admits -- the family having been attributed, in
    navierstokes.cpp's boundary-condition step, to the INTERIOR convective
    nonlinearity rather than to the boundary row's degree, and `-bclin` (which
    makes the outflow row exactly linear in u^ about a frozen exterior state,
    as this does) having moved the family without removing it: 5.63, 1.31,
    3.30 became 1.18, 2.77.

    That prediction is too strong and so is its opposite, and the two meshes
    disagree about which. `-bclin` is the smaller change -- it freezes the flux
    and leaves the row otherwise intact, while this REPLACES the row: the
    viscous q^.n goes, the diffusive stabilization is not added, and the datum
    is masked off. Plane Poiseuille, order 2, eleven `-uinit` starts from 0 to
    1.5, on two meshes:

    \verbatim
                                  exact answer    spurious root    diverges
        -r 1  -bcchar                  10               0              1
              -bcflux (control)         0               5              6
        -r 2  -bcchar                   4               5              2
              -bcflux (control)         0               5              6
    \endverbatim

    **At `-r 1` the family is gone and at `-r 2` it is not**, five of eleven
    starts landing on 0.769 / 0.066 or 1.62 / 0.054 there. So what is a
    property of the condition is that it ENLARGES the basin of the true root,
    on both meshes and by a lot; that it REMOVES the family is a property of
    the coarse mesh alone, and the `-r 1` arm on its own would have been
    written up as the second. The control is 0 / 5 / 6 on both -- it reaches
    the exact answer from no start at either refinement, landing on 6.38 / 0.52
    or 1.64 / 0.18 at `-r 1` and 6.60 / 0.53 or 1.76 / 0.19 at `-r 2` -- so the
    comparison itself is mesh-independent and only the size of the win moves.
    None of this contradicts the Reynolds attribution, which is about where the
    nonlinearity LIVES; this changes which root the boundary admits.

    **What it does NOT do is remove the exact solution from the problem, only
    from the outflow.** It covers the attributes whose VELOCITY trace is free;
    under `-bcphys` the walls and the inlet leave the PRESSURE trace free and
    those rows still take HDGPrescribedFluxLFIntegrator's datum. Measured:
    `-bcchar -no-bcflux` gives 3.84 / 1.01, against `-no-bcflux` alone at
    3.84 / 1.00 -- indistinguishable, because the rows that were carrying the
    exact solution are not the rows this replaces. Extending it to a wall is
    not a matter of marking one more attribute: `un = 0` there and the
    degeneracy above is exactly what would be hit.

    **Argument convention** is HDGLinearisedBdrFlux's and for the same reason:
    HyperbolicFormIntegrator's HDG path calls `Average(state_tr, state_el, ..)`,
    so @a state1 is the trace and @a state2 the element state, and
    `AverageGrad`'s `side == 1` is the derivative with respect to the trace. */
class HDGCharacteristicBdrFlux : public RusanovFlux
{
   const ArtificialCompressibilityFlux *acf;
   VectorCoefficient &uinf;   ///< the prescribed far-field state

   mutable Vector w, du;
   mutable DenseMatrix An, M, M2, Aabs;

   /** @brief `A_n` and `|A_n|` at the far-field state, both left in @a An and
       @a Aabs. @a w comes back holding that state. */
   void Jacobians(const Vector &nor, FaceElementTransformations &Tr) const
   {
      const int neq = fluxFunction.num_equations;

      // Elem1 at its own integration point is the physical point on the face,
      // and is an element transformation, so a coefficient reading element
      // attributes still works. HDGLinearisedBdrFlux does the same.
      uinf.Eval(w, *Tr.Elem1, Tr.GetElement1IntPoint());
      fluxFunction.ComputeFluxJacobianDotN(w, nor, Tr, An);

      const int dim = fluxFunction.dim;
      real_t un = 0., nn = 0.;
      for (int d = 0; d < dim; d++)
      {
         un += w(1 + d) * nor(d);
         nn += nor(d) * nor(d);
      }
      // Dropping v (x) v drops the (v.n) I + v n^T block with it, so the
      // spectrum collapses to +- sqrt(beta)|n| and 0; un = 0 is what says so.
      if (!acf || acf->IsStokes()) { un = 0.; }
      const real_t beta = acf ? acf->GetBeta() : 1.;
      const real_t c = std::sqrt(un * un + beta * nn);
      MFEM_ASSERT(c > 0., "a face of zero measure, or beta <= 0");

      M = An;
      for (int i = 0; i < neq; i++) { M(i, i) -= un; }
      M2.SetSize(neq);
      mfem::Mult(M, M, M2);

      const real_t aun = std::abs(un);
      Aabs.SetSize(neq);
      for (int i = 0; i < neq; i++)
         for (int j = 0; j < neq; j++)
         {
            Aabs(i, j) = ((c - aun) / (c * c)) * M2(i, j) + (un / c) * M(i, j);
         }
      for (int i = 0; i < neq; i++) { Aabs(i, i) += aun; }
   }

public:
   /** @param f  the same FluxFunction the rest of the residual uses
       @param u  the prescribed FAR-FIELD state, `num_equations` wide.  It is
                 not a datum on the row in the sense the prescribed flux is:
                 only the incoming characteristics read it. */
   HDGCharacteristicBdrFlux(const FluxFunction &f, VectorCoefficient &u)
      : RusanovFlux(f),
        acf(dynamic_cast<const ArtificialCompressibilityFlux*>(&f)),
        uinf(u) { }

   real_t Average(const Vector &state1, const Vector &state2,
                  const Vector &nor, FaceElementTransformations &Tr,
                  Vector &flux) const override
   {
      const int neq = fluxFunction.num_equations;
      flux.SetSize(neq);
      Jacobians(nor, Tr);

      //  B^ = ( A (u - u_inf) + |A| (u + u_inf - 2 u^) ) / 2
      du.SetSize(neq);
      for (int i = 0; i < neq; i++) { du(i) = state2(i) - w(i); }
      An.Mult(du, flux);
      for (int i = 0; i < neq; i++)
      {
         du(i) = state2(i) + w(i) - 2. * state1(i);
      }
      Aabs.AddMult(du, flux);
      flux *= 0.5;

      // The reported speed is what the caller uses to size a time step and to
      // report lambda_max; the row carries no stabilization of its own, the
      // characteristic splitting being the upwinding.
      return acf ? acf->MaxCharSpeedDotN(w, nor) : 0.;
   }

   void AverageGrad(int side, const Vector &state1, const Vector &state2,
                    const Vector &nor, FaceElementTransformations &Tr,
                    DenseMatrix &grad) const override
   {
      MFEM_ASSERT(side == 1 || side == 2, "Unknown side");
      const int neq = fluxFunction.num_equations;
      grad.SetSize(neq);
      Jacobians(nor, Tr);

      if (side == 1)
      {
         // d/du^ = -|A_n|, and it depends on neither state.
         for (int i = 0; i < neq; i++)
            for (int j = 0; j < neq; j++) { grad(i, j) = -Aabs(i, j); }
         return;
      }
      // side == 2, the element state: d/du = A+_n = (A_n + |A_n|)/2.
      for (int i = 0; i < neq; i++)
         for (int j = 0; j < neq; j++)
         {
            grad(i, j) = 0.5 * (An(i, j) + Aabs(i, j));
         }
   }
};

/** @brief The element row keeps the physical numerical flux; the TRACE row is
    the characteristic condition. One integrator, because one integrator is
    what DarcyHybridization asks for both.

    **This exists because a NumericalFlux cannot express it.**
    HyperbolicFormIntegrator::AssembleHDGFaceVector() makes ONE
    `numFlux.Average()` call per quadrature point and writes its result into
    whichever row groups @a type asks for -- so a flux registered there lands
    in both. The element (potential) equation's boundary term is
    `<F^.n, v>_dK` and must stay the physical numerical trace of the flux
    whatever boundary condition is imposed; the condition is imposed through
    the equation FOR u^, which is the trace row. Putting `B^` in both would
    enforce a conservation statement that is not the conservation law.

    HDGLinearisedBdrFlux needs none of this and the difference is the point:
    it replaces `F^` with a linearisation OF `F^`, which is still a flux, so
    both rows want it. `B^` is the residual of a boundary condition and is not
    a flux at all.

    So this delegates on the row type, to two ordinary
    HyperbolicFormIntegrators built on the two fluxes. For the RESIDUAL that
    is a plain branch -- AssembleHDGFaceVector()'s own assertion says @a type
    is either the element pair or the trace pair, never both. For the
    GRADIENT both groups arrive in one call from
    DarcyHybridization::ConstructGrad(), and the blocks are laid out group
    first and equation second, so the element rows are the first
    `dof_el * neq` and the trace rows the rest: two calls and a row splice. */
class HDGCharacteristicBdrIntegrator : public NonlinearFormIntegrator
{
   HyperbolicFormIntegrator ord;   ///< the physical flux, for the element row
   HyperbolicFormIntegrator chr;   ///< B^, for the trace row
   const int neq;

   mutable DenseMatrix m_ord, m_chr;

   static bool WantsTraceRow(int type)
   {
      return (type & (NonlinearFormIntegrator::HDGFaceType::CONSTR
                      | NonlinearFormIntegrator::HDGFaceType::FACE)) != 0;
   }
   static bool WantsElemRow(int type)
   {
      return (type & (NonlinearFormIntegrator::HDGFaceType::ELEM
                      | NonlinearFormIntegrator::HDGFaceType::TRACE)) != 0;
   }

public:
   /** @param ordinary        the numerical flux every other face uses
       @param characteristic  HDGCharacteristicBdrFlux on the same system
       @param sign            the residual convention, as passed to
                              HyperbolicFormIntegrator elsewhere */
   HDGCharacteristicBdrIntegrator(const NumericalFlux &ordinary,
                                  const NumericalFlux &characteristic,
                                  real_t sign = 1.)
      : ord(ordinary, 0, sign), chr(characteristic, 0, sign),
        neq(ordinary.GetFluxFunction().num_equations) { }

   void AssembleHDGFaceVector(int type, const FiniteElement &trace_face_fe,
                              const FiniteElement &fe,
                              FaceElementTransformations &Tr,
                              const Vector &trfun, const Vector &elfun,
                              Vector &elvect) override
   {
      if (WantsTraceRow(type))
      {
         chr.AssembleHDGFaceVector(type, trace_face_fe, fe, Tr, trfun, elfun,
                                   elvect);
      }
      else
      {
         ord.AssembleHDGFaceVector(type, trace_face_fe, fe, Tr, trfun, elfun,
                                   elvect);
      }
   }

   void AssembleHDGFaceGrad(int type, const FiniteElement &trace_face_fe,
                            const FiniteElement &fe,
                            FaceElementTransformations &Tr,
                            const Vector &trfun, const Vector &elfun,
                            DenseMatrix &elmat) override
   {
      const bool tr = WantsTraceRow(type), el = WantsElemRow(type);
      if (tr && !el)
      {
         chr.AssembleHDGFaceGrad(type, trace_face_fe, fe, Tr, trfun, elfun,
                                 elmat);
         return;
      }
      if (el && !tr)
      {
         ord.AssembleHDGFaceGrad(type, trace_face_fe, fe, Tr, trfun, elfun,
                                 elmat);
         return;
      }

      // Both groups in one call: same shape from both, element rows from the
      // physical flux and trace rows from B^. The split is at
      // `dof_dual_el * num_equations`, which is where
      // HyperbolicFormIntegrator::AssembleHDGFaceGrad() bases its trace rows.
      ord.AssembleHDGFaceGrad(type, trace_face_fe, fe, Tr, trfun, elfun, m_ord);
      chr.AssembleHDGFaceGrad(type, trace_face_fe, fe, Tr, trfun, elfun, m_chr);
      MFEM_ASSERT(m_ord.Height() == m_chr.Height() &&
                  m_ord.Width() == m_chr.Width(), "the two delegates disagree "
                  "about the block shape");
      elmat = m_ord;
      const int roff = fe.GetDof() * neq;
      MFEM_ASSERT(roff < elmat.Height(), "no trace rows to splice");
      for (int i = roff; i < elmat.Height(); i++)
      {
         for (int j = 0; j < elmat.Width(); j++) { elmat(i, j) = m_chr(i, j); }
      }
   }
};

/** @brief The prescribed HDG numerical flux on a boundary face, assembled as a
    linear form on the TRACE space.

    This is the Neumann half of a mixed HDG boundary condition. Per boundary
    face and per equation the hybridized system offers exactly one of two
    conditions: either the trace component is essential, and its constraint row
    is replaced by the prescribed value, or the constraint row survives as

    \verbatim
        < (F^ + G^) . n , mu_e >  =  < b_e , mu_e >,
    \endverbatim

    and @a b_e is the datum this integrator supplies. **A row left with no
    datum is not a free boundary; it is `b_e = 0`, i.e. zero numerical flux**,
    and on a boundary face nothing cancels it because the row has only one
    side. That is what made `navierstokes -bcphys` wrong by more than 100% --
    see the boundary-condition step of navierstokes.cpp.

    The datum assembled is the exact numerical flux of the problem,

    \verbatim
        b = F(u) . n + G(u, q) . n,        G_e . n = + q_e . n,
    \endverbatim

    **and that plus sign was measured, against a comment that said minus.**
    navierstokes.cpp's header used to write the first-order form as
    `q - nu grad u = 0` and hence `G = -q`; the code's flux is
    `q = -nu grad u`, which its own ExactFlux() and the doxygen there both say,
    and which `-bcfull` reproduces to 4.4e-15 -- a relative error that would be
    2, not 4e-15, had the sign been the other one. So in this code `G = +q`,
    the header was internally consistent and disagreed with the code it
    describes, and it has been corrected in place.

    with `F` taken from the FluxFunction itself -- so it follows `-stokes` and
    every other setting of the flux without a second implementation -- and `u`
    and `q` from coefficients the caller supplies. At the exact solution
    `u = u^` on the boundary, the Lax-Friedrichs stabilization `S(u - u^)`
    vanishes identically, so this datum is exactly the residual's own boundary
    term and a problem whose exact solution lies in the discrete space comes
    back at round-off. That is the acceptance test, and it is sharp.

    **A consistent datum is not a unique one, and on a face that also carries
    an essential component it will not be.** Where `F` is nonlinear in the
    components the trace leaves free, this row determines them only up to the
    roots of that nonlinearity. On the Navier-Stokes outlet, with `p^`
    essential and the momentum rows natural, the row is
    `v^_x^2 + p^ + q.n + S(u - u^) = b` and fixes `v^.n` only up to SIGN --
    per quadrature point, so the count of discrete roots is combinatorial
    rather than two. Four were reached at order 2 on an 8x8 channel, all
    converged to `||r||/||r_0||` of 1e-16, and the same fifteen initial states
    under `-bcfull`, where no component is free, all reach one root. So this
    is a property of the CONDITION, not of the integrator: the exact solution
    is a root to 5.1e-16 either way. The account, the controls and what does
    and does not repair it are at the boundary-condition step of
    navierstokes.cpp.

    @a scale_F and @a scale_G exist because the sign convention of the trace
    row is not derivable from the documentation and was fixed here by
    measurement, exactly as `-hsign` was: see the `-bcsf`/`-bcsg` sweep
    recorded at the boundary conditions in navierstokes.cpp.

    **The datum must be integrated with the RESIDUAL's quadrature rule, not
    with an accurate one, and this is the whole of @a ioff.** The trace row's
    own boundary term comes from HyperbolicFormIntegrator, whose rule is
    `2*max(order_el, order_tr) + IntOrderOffset`; with the convective term
    present the integrand `(v.n) v_i mu` is of degree `3k` on the outlet and
    that rule does not integrate it exactly at `k = 2`. Two inexact integrals
    of the same function cancel only if they are the same integral. Measured,
    plane Poiseuille at `k = 2` with the state set exactly, `||r_tr||` against
    the offset:

    erbatim
        ioff      -2        -1         0         1       2,3,5
        ||r_tr||  2.00e-3   2.00e-3   2.4e-17   2.4e-17  4.03e-6
    \endverbatim

    The plateaus are where a Gauss rule on a segment changes point count, which
    is why `0` and `1` agree; `0` is the one that matches. **A more accurate
    datum is a wrong datum** -- `ioff = 3` was the first thing written here and
    it left an error ten orders above round-off that looked like a missing
    term. Two controls say it is the convective degree and nothing else:
    at `k = 3` the residual's rule is already exact for the integrand and every
    offset from 0 upward gives round-off, and under `-stokes`, which removes
    `v (x) v`, `ioff = 0` and `ioff = 3` agree at 1.6e-17 and 5.7e-17.

    @note The trace space must be `Ordering::byNODES` with `vdim = neq`, which
    is what the rest of the miniapp requires anyway. The element vector is
    written as a `(dof, neq)` column-major block, matching
    `GetBdrElementVDofs`.

    @note Registered with `LinearForm::AddBoundaryIntegrator()`, not
    `AddBdrFaceIntegrator()`: the latter looks the test function up in the
    *element* space, which for a trace space is the wrong space. The outward
    normal therefore comes from the boundary element transformation, and MFEM's
    `Mesh::CheckBdrElementOrientation()` is what makes `CalcOrtho()` outward
    there. The inlet and the outlet of a channel carry opposite normals, so a
    single global sign zeroing the residual at both is itself the check that
    the orientation is right. */
class HDGPrescribedFluxLFIntegrator : public LinearFormIntegrator
{
   const FluxFunction &fluxfn;
   VectorCoefficient &ucoeff;    ///< the state, size num_equations
   VectorCoefficient *qcoeff;    ///< the flux, size num_equations*dim, or NULL
   real_t scale_F, scale_G;
   int int_order_offset;

   Vector shape, nor, state, qvec, bdotn;
   DenseMatrix Fmat;

public:
   /** @param f     the same FluxFunction the residual uses
       @param u     the prescribed state
       @param q     the prescribed flux `q = -nu grad u`, or NULL for none
       @param sF    scaling of the inviscid half, a diagnostic; 1 is right
       @param sG    scaling of the viscous half, a diagnostic; 1 is right,
                    and the sign it carries is `G.n = +q.n` -- see above
       @param ioff  added to `2*order` when picking the integration rule. Zero
                    is not a default to be tuned: it is what makes the rule the
                    residual's own, and raising it breaks the cancellation. */
   HDGPrescribedFluxLFIntegrator(const FluxFunction &f, VectorCoefficient &u,
                                 VectorCoefficient *q = NULL,
                                 real_t sF = 1., real_t sG = 1., int ioff = 0)
      : fluxfn(f), ucoeff(u), qcoeff(q), scale_F(sF), scale_G(sG),
        int_order_offset(ioff) { }

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      const int neq = fluxfn.num_equations;
      const int dim = fluxfn.dim;
      const int dof = el.GetDof();

      elvect.SetSize(dof * neq);
      elvect = 0.;
      DenseMatrix elvect_mat(elvect.GetData(), dof, neq);

      shape.SetSize(dof);
      nor.SetSize(dim);
      bdotn.SetSize(neq);

      const IntegrationRule *ir = IntRule;
      if (!ir)
      {
         ir = &IntRules.Get(el.GetGeomType(),
                            2 * el.GetOrder() + int_order_offset);
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Tr.SetIntPoint(&ip);

         // Unnormalised outward normal: it carries the face weight, exactly as
         // BoundaryNormalLFIntegrator relies on, so only ip.weight is applied.
         if (dim > 1) { CalcOrtho(Tr.Jacobian(), nor); }
         else { nor(0) = 1.; }

         el.CalcShape(ip, shape);
         ucoeff.Eval(state, Tr, ip);

         fluxfn.ComputeFlux(state, Tr, Fmat);
         Fmat.Mult(nor, bdotn);
         bdotn *= scale_F;

         if (qcoeff)
         {
            qcoeff->Eval(qvec, Tr, ip);
            for (int e = 0; e < neq; e++)
            {
               real_t qn = 0.;
               for (int d = 0; d < dim; d++) { qn += qvec(e * dim + d) * nor(d); }
               bdotn(e) += scale_G * qn;
            }
         }

         AddMult_a_VWt(ip.weight, shape, bdotn, elvect_mat);
      }
   }

   using LinearFormIntegrator::AssembleRHSElementVect;
};

} // namespace hdg
} // namespace mfem

#endif // MFEM_HDG_NSFLUX
