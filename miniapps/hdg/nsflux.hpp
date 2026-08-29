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
    and that is the object section 5 of doc/HDG-ROADMAP.md is about: on a face
    whose normal lies along the flow it is `|v| + sqrt(v^2 + beta)`, and on one
    across the flow it is `sqrt(beta)`. One scalar `tau` cannot be both, which
    is exactly why the constant-`Ctau` HDGFlux is not the right stabilization
    here and HDGLaxFriedrichsFlux below is.

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

} // namespace hdg
} // namespace mfem

#endif // MFEM_HDG_NSFLUX
