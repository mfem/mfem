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

#ifndef MFEM_UNIT_FEM_DARCY_PLAIN_NPC_HPP
#define MFEM_UNIT_FEM_DARCY_PLAIN_NPC_HPP

#include "mfem.hpp"

/** @file The PLAIN NPC path -- every batched mode left at its default -- with
    an answer that is arithmetic rather than a second run of the same code.

    **Why this exists, and why it is a header shared by two binaries.** meq
    reports (their M-79) that a whole NPC solve does not survive an
    mfem::Device: Newton takes zero steps where it needs four, and the flux
    disagrees by 4.006e-02, identically across three different trace solvers --
    two of which never touch the device at all. Their configuration turns on
    NONE of this class's batched modes.

    Every existing [DebugDevice] case compares two device-configured arms of
    the same problem against each other, because the Device belongs to the
    binary and no no-Device arm is available inside it; and every one of them
    switches a batched mode on somewhere. Two arms agreeing says nothing when
    the fault is common to both, which is exactly what three trace solvers
    returning the same wrong answer looks like. So the check here is against
    an EXACT solution: the discrete answer is known before the code runs.

    Compiled into unit_tests (no Device) and into debug_device_tests (which
    configures Device("debug")) from this one header, so the two differ in the
    Device and in nothing else. That is the "two TEST_CASEs over one static
    helper" shape this tree already uses for the SuiteSparse BLAS check.

    **The problem, and why the answer is exact.** The mixed Darcy system

        u + grad p = 0        - div u = 0

    on the unit square, with p = x + 2y and therefore u = -(1, 2) constant.
    Both lie in the discrete spaces at any order >= 1 and p restricted to a
    face lies in the trace space, so by consistency of the HDG method the
    discrete solution IS the exact one, to round-off, on ANY mesh. There is no
    discretisation error to hide a defect behind and no mesh refinement to run.

    **The structure is meq's, deliberately.** A domain integrator on the
    potential mass NONLINEAR form with the HDG face stabilization beside it,
    which is what sends DarcyForm::EnableHybridization() down its
    `Mnl_p && FaceIntegratorsAreLinear(...)` branch and installs the face
    constraint as the LINEAR c_bfi_p -- the route meq takes. That is this
    branch's own "look for the inert knob" note used as a construction rather
    than as a probe.

    **The inert knob is a blind spot as well as a construction, which is why
    @a mass exists.** A contribution whose correct value is exactly zero is
    indistinguishable from a contribution that was never read back. With
    `mass = 0` the domain term above is arithmetically inert, so on its own
    this case cannot tell a device that evaluates it from one that silently
    drops it -- which is the failure mode it exists to catch.
    Their own fixture puts a constant non-zero F on that integrator and gets
    exactly zero back.

    So `mass = a != 0` adds `a (p, phi)` to the potential row and the matching
    load `a (p_exact, phi)` to g, and BOTH halves are then individually
    detectable:

      * drop the domain term and the discrete problem becomes `lap p = a
        p_exact`, whose solution is not `x + 2y`;
      * drop the load and it becomes `lap p + a p = 0`, whose solution is not
        `x + 2y` either.

    Either way `err_p` leaves round-off, where with `mass = 0` neither could
    move it. **Both halves were checked by dropping them**, which is the only
    thing that makes this arm worth its build: with the domain term dropped
    `err_u` comes back 0.835 to 0.858, with the load dropped 0.739 to 0.754,
    against a threshold of 1e-10 -- and with `mass = 0` dropping either is
    invisible, which is the whole point meq was making. The exact solution is unchanged because `u = -grad p` and
    `-div u + a p = a p_exact` hold pointwise for `p = x + 2y`, and both
    integrands are polynomials the default rules integrate exactly on an
    affine mesh.

    The problem is affine, so ONE NPC step from any starting point is the
    answer; nothing here iterates, and a step that fails to land is a defect
    and not slow convergence.

    **Which route this runs, measured rather than assumed.**
    `CanBatchLinearResidual()` comes back FALSE here at either `mass`, and
    that is the right answer rather than a disappointment: a non-null
    `Mnl_p` makes `IsNonlinear()` true with no flux or block nonlinearity
    beside it, so `Finalize()` picks `LocalOpType::PotNL`, and the predicate
    refuses PotNL outright -- it cannot freeze a gradient to match a frozen
    residual there. So the element loop is the per-element one, which is
    **the same local operator meq runs**, their source integrator being
    genuinely nonlinear and refused by the same predicate one line earlier.
    The case pins it with REQUIRE_FALSE so that a change which silently moved
    this onto the batched route would fail here rather than quietly stop
    being meq's configuration. */
namespace darcy_plain_npc
{
using namespace mfem;

inline real_t PExact(const Vector &x) { return x(0) + 2.0 * x(1); }

inline void UExact(const Vector &, Vector &u) { u(0) = -1.0; u(1) = -2.0; }

/// What one plain NPC step produced, and enough to tell HOW it failed.
struct Result
{
   real_t err_u{};      ///< L2 error of the flux against UExact
   real_t err_p{};      ///< L2 error of the potential against PExact
   real_t r0_fields{};  ///< ||initial residual|| on the flux+potential blocks
   real_t r0_trace{};   ///< ||initial residual|| on the trace block
   real_t step{};       ///< ||the Newton increment||, fields and trace
   real_t datum{};      ///< max |essential trace datum|, so it is known nonzero
   real_t load{};       ///< max |g|, so a non-zero source is known to be live
   bool batched_residual{};  ///< whether LinearResidualBatched() is taken
   real_t r0_operator{}; ///< ||F(x0)|| taken THROUGH DarcyNPCOperator::Mult
   int its{-1};          ///< Newton iterations, when driven through the operator
   bool converged{};     ///< ... and whether it said so
   bool r_dev_valid{};   ///< is F(x0) still DEVICE-valid straight out of Mult()
   int nonfinite{};     ///< CheckFinite() summed over everything returned
   int trace_size{};
};

/** One NPC step on the problem above at @a order on an @a n by @a n mesh.

    Nothing is set on the hybridization but NPC and the essential boundary:
    AssemblyMode, LocalFactorMode, TraceAssemblyMode and GradientMode are all
    left at their defaults, which is the whole point. @a geom is a parameter
    because meq runs TRIANGLES and every other Darcy device case in this tree
    runs quadrilaterals; the exact solution above is exact on both.

    @a mass is the coefficient on the potential-mass domain term and on the
    load that balances it; see the note above on why zero is not enough on its
    own. The routing does not depend on it -- MassIntegrator is a
    BilinearFormIntegrator at either value -- so the two arms differ in the
    arithmetic and in nothing else. */
/** @a via_operator drives the step through **DarcyNPCOperator and
    DarcyNPCSolver under a NewtonSolver**, which is the layer a caller
    actually runs: DarcyNPCOperator::Mult() is the residual a Newton
    evaluates. Calling NPCResidual() directly skips it, and a fault inside it
    is then invisible to every assertion here. */
inline Result Solve(int order, int n, Element::Type geom, real_t mass = 0.0,
                    bool via_operator = false)
{
   const int dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, geom);

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), mass_coeff(mass);
   FunctionCoefficient load_coeff([mass](const Vector &x)
   {
      return mass * PExact(x);
   });

   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;

   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));

   // The boundary face integrator is read for its MARKER and for nothing
   // else: DarcyForm::Assemble() installs the constraint on those attributes,
   // and without one the constraint is interior-only and every element
   // touching the boundary is wrong. Same trap as the Navier-Stokes miniapp's.
   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)), all);

   // Everything on the potential block goes on the NONLINEAR form, which is
   // meq's structural decision and is what selects the c_bfi_p route here.
   // The domain term is decisive in the routing at either coefficient, and
   // arithmetically live only at a non-zero one.
   NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
   Mnl_p->AddDomainIntegrator(new MassIntegrator(mass_coeff));
   Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   // The load that balances it. Asked for ONLY when it is wanted: the
   // non-const accessors CONSTRUCT the form they return, and merely calling
   // GetPotentialMassForm() is enough to silence a face constraint placed on
   // the nonlinear form.
   if (mass != 0.0)
   {
      darcy.GetPotentialRHS()->AddDomainIntegrator(
         new DomainLFIntegrator(load_coeff));
   }

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);

   DarcyHybridization *dh = darcy.GetHybridization();
   dh->EnableNPC();
   dh->SetEssentialBC(all);

   darcy.Assemble();
   darcy.Finalize();

   Result res;
   res.trace_size = Mh.GetVSize();

   // The Dirichlet datum, on the boundary trace dofs. ProjectBdrCoefficient
   // and not ProjectCoefficient: the latter segfaults on a face-based space.
   GridFunction tr(&Mh);
   tr = 0.0;
   FunctionCoefficient pcoeff(PExact);
   tr.ProjectBdrCoefficient(pcoeff, all);

   Vector x_tr(Mh.GetVSize());
   x_tr = 0.0;
   Array<int> bdr_dofs;
   Mh.GetEssentialTrueDofs(all, bdr_dofs);
   for (int i = 0; i < bdr_dofs.Size(); i++)
   {
      const int d = bdr_dofs[i];
      x_tr(d) = tr(d);
      res.datum = std::max(res.datum, std::abs(tr(d)));
   }

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;   // f = 0; with mass = 0, g = 0 too and the datum drives it alone

   if (mass != 0.0)
   {
      // **MINUS, and the sign is pinned by the residual's own convention
      // rather than by taste.** LinearResidualBatched() states the potential
      // row as `rp = B u + D p + E x - sgn bp` with `sgn = bsym ? -1 : 1`,
      // and DarcyForm's bsymmetrize defaults to TRUE -- so a term that
      // arrives in D with a plus has to arrive in g with a minus. Measured
      // both ways before it was believed: the other sign gives err_u 1.53 and
      // err_p 0.327 where this one gives round-off.
      //
      // SyncAliasMemory() because a block of a BlockVector is an alias with
      // its own validity flags, and a host write through it leaves the base
      // unaware.
      darcy.GetPotentialRHS()->Assemble();
      b.GetBlock(1) -= *darcy.GetPotentialRHS();
      b.GetBlock(1).SyncAliasMemory(b);
      res.load = b.GetBlock(1).Normlinf();
   }

   // A NON-ZERO start, and it is not arbitrary. From x = 0 with the datum on
   // the boundary trace alone, the trace block of the residual is EXACTLY
   // zero -- the fields are zero, so the interior numerical flux is zero, so
   // the conservativity rows have nothing in them -- and a check on it would
   // assert what the starting point guarantees rather than what the code
   // does. Measured: r0_trace came back 0 at every order and mesh. Starting
   // away from zero puts all three blocks in play. The problem is affine, so
   // one step lands on the answer from anywhere.
   x = 1.0;

   if (via_operator)
   {
      // The recipe documented on DarcyNPCOperator, followed exactly.
      Array<int> npc_offs(4);
      npc_offs[0] = 0;
      npc_offs[1] = npc_offs[0] + Vh.GetVSize();
      npc_offs[2] = npc_offs[1] + Wh.GetVSize();
      npc_offs[3] = npc_offs[2] + Mh.GetVSize();

      BlockVector X(npc_offs);
      X.GetBlock(0) = x.GetBlock(0);
      X.GetBlock(1) = x.GetBlock(1);
      X.GetBlock(2) = x_tr;
      X.SyncFromBlocks();

      DarcyNPCOperator npc(*dh, npc_offs, b);

      // **The initial residual taken THROUGH the operator**, which is the one
      // quantity meq reports as identically zero. Taken before the solve,
      // because a Newton handed a zero residual reports convergence at
      // iteration zero and leaves the fields at the initial iterate -- so
      // this and @a its below are the two halves of M-79 as assertions.
      Vector R(X.Size());
      // **UseDevice(true), because that is what the caller does.**
      // NewtonSolver::SetOperator() sets it unconditionally on its residual
      // and correction, so a bare Vector here would take the host path and
      // exercise none of what the operator has to get right -- the probe
      // would be measuring a configuration nobody runs.
      R.UseDevice(true);
      R = 0.0;
      npc.Mult(X, R);
      // **Sampled BEFORE any host read, and it asserts where the data IS
      // rather than what it says.** meq's point, and it is a good one: a
      // check that ends in a host read cannot tell "the flags were repaired"
      // from "the data was dragged to the host", because SyncAlias() will
      // migrate device-valid data down and unprotect it. So the one-hop
      // version of DarcyNPCOperator::Mult() is accidentally correct at the
      // price of a d2h round trip per residual, and no value-based assertion
      // anywhere can see the difference. This one can.
      res.r_dev_valid = R.GetMemory().DeviceIsValid();
      R.HostRead();
      res.r0_operator = R.Norml2();
      res.nonfinite += R.CheckFinite();

      UMFPackSolver umf;
      DarcyNPCSolver lin(umf);
      NewtonSolver newton;
      newton.SetOperator(npc);
      newton.SetSolver(lin);
      newton.SetRelTol(1e-13);
      newton.SetAbsTol(1e-14);
      newton.SetMaxIter(20);
      newton.SetPrintLevel(-1);
      Vector zero(X.Size());
      zero = 0.0;
      newton.Mult(zero, X);
      res.its = newton.GetNumIterations();
      res.converged = newton.GetConverged();

      X.HostRead();
      res.nonfinite += X.CheckFinite();
      res.batched_residual = dh->CanBatchLinearResidual();

      GridFunction qo(&Vh), po(&Wh);
      qo.MakeRef(&Vh, X.GetBlock(0), 0);
      po.MakeRef(&Wh, X.GetBlock(1), 0);
      VectorFunctionCoefficient uc(dim, UExact);
      res.err_u = qo.ComputeL2Error(uc);
      res.err_p = po.ComputeL2Error(pcoeff);
      return res;
   }

   BlockVector r(darcy.GetOffsets());
   Vector r_tr, b_tr, dtr;
   dh->NPCResidual(b, x, x_tr, r, r_tr);
   // AFTER the residual: the route builds its cache on first use, so asking
   // before reports the question rather than the answer. Recorded because
   // WHICH route this case exercises is the point of it -- meq's own
   // configuration cannot take this one, their source integrator being
   // genuinely nonlinear.
   res.batched_residual = dh->CanBatchLinearResidual();
   res.r0_fields = r.Norml2();
   res.r0_trace = r_tr.Norml2();

   Operator &S = dh->NPCGradient(x, x_tr);
   dh->NPCReduce(r, r_tr, b_tr);

   dtr.SetSize(b_tr.Size());
   dtr = 0.0;
   {
      SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
#ifdef MFEM_USE_SUITESPARSE
      if (Sm)
      {
         UMFPackSolver umf(*Sm);
         umf.Mult(b_tr, dtr);
      }
      else
#endif
      {
         std::unique_ptr<GSSmoother> prec;
         if (Sm) { prec.reset(new GSSmoother(*Sm)); }
         GMRESSolver gmres;
         gmres.SetOperator(S);
         if (prec) { gmres.SetPreconditioner(*prec); }
         gmres.SetKDim(200);
         gmres.SetMaxIter(5000);
         gmres.SetRelTol(1e-14);
         gmres.SetAbsTol(0.0);
         gmres.SetPrintLevel(-1);
         gmres.Mult(b_tr, dtr);
      }
   }

   BlockVector dx(darcy.GetOffsets());
   dx = 0.0;
   dh->NPCRecover(r, dtr, dx);

   // NPCReduce()/NPCRecover() give the increment to ADD; the sign flip lives
   // in DarcyNPCSolver, which is not on this path.
   x += dx;
   x_tr += dtr;
   res.step = std::sqrt(dx.Norml2() * dx.Norml2() + dtr.Norml2() * dtr.Norml2());

   // Norml2() CANNOT see a NaN -- it guards its reduction with fabs(v) > 0,
   // which is false for NaN, so it returns the norm of the remainder and an
   // all-NaN vector reads as zero. Every norm above is therefore worthless
   // until this says so.
   res.nonfinite = x.CheckFinite() + x_tr.CheckFinite() + r.CheckFinite()
                   + r_tr.CheckFinite() + dtr.CheckFinite();

   GridFunction q(&Vh), p(&Wh);
   q.MakeRef(&Vh, x.GetBlock(0), 0);
   p.MakeRef(&Wh, x.GetBlock(1), 0);
   VectorFunctionCoefficient ucoeff(dim, UExact);
   res.err_u = q.ComputeL2Error(ucoeff);
   res.err_p = p.ComputeL2Error(pcoeff);

   return res;
}

} // namespace darcy_plain_npc

#endif // MFEM_UNIT_FEM_DARCY_PLAIN_NPC_HPP
