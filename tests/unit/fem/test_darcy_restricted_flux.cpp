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

// A flux space that carries FEWER components than the mesh has dimensions.
//
// **What it is for.** DarcyForm's flux mass is a VectorMassIntegrator(1/kappa),
// so a direction in which the medium does not diffuse asks for an infinite
// coefficient. A kinetic equation whose collision operator diffuses in the
// velocity coordinates only is exactly that: along the field line the operator
// is pure advection, and the diffusion tensor carries a structural zero
// eigenvalue at every point and permanently rather than in some limit. The
// workaround is to invent a small parallel diffusion, and it is a workaround
// rather than a limit -- the answer is insensitive to kappa over eight decades
// and then falls off a cliff, and a floored stabilization moves the cliff and
// makes the failure look plausible instead of astronomical.
//
// **The problem here, and why it discriminates.** p = x^2 + y on the unit
// square, so -Lap p = -2 and d_yy p = 0. A flux carrying only x therefore
// solves the SAME potential problem as the full one -- the y flux is a
// constant that divergences to nothing -- while a flux carrying only y solves
// -d_yy p = -2, whose answer is x^2 + y^2. So "keep {x}" must reproduce the
// full arm and "keep {y}" must NOT, which is what makes this a check rather
// than a null test: without the second arm, "keep {x}" passing would be
// consistent with the restriction being ignored altogether. p, its flux and
// its trace are all in the discrete spaces at order >= 2, so the discrete
// answer is the exact one on ANY mesh and there is no refinement to run.
//
// The NPC path throughout -- NPCResidual, NPCGradient, NPCReduce, NPCRecover
// -- because that is the only path a restricted flux is supported on, and
// DarcyHybridization::CheckRestrictedFluxConfiguration() refuses the rest.

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace darcy_restricted_flux
{

real_t PExact(const Vector &x) { return x(0) * x(0) + x(1); }

/// The full 2-D flux of PExact, from which each arm takes its own components.
void UFull(const Vector &x, Vector &u) { u(0) = -2.0 * x(0); u(1) = -1.0; }

struct Result
{
   real_t err_p{};      ///< L2 error of the potential against PExact
   real_t err_u{};      ///< L2 error of the flux against the components carried
   int nonfinite{};     ///< CheckFinite() over everything the step touched
   int flux_vsize{};    ///< so an arm cannot silently be the full one
};

/** One NPC step on the problem above.

    @a comps NULL means the stock integrators at this @a m, which is what the
    tree did before the restricted pair existed and is NOT a supported
    configuration at m < dim -- it is here only in the refusal section.

    @a advection replaces the diffusion with div(c p), c = (1,0), which is
    well posed with NO flux at all and is how m = 0 is exercised. PExact is
    the solution of that too, with the source 2x, and the essential trace is
    imposed on the whole boundary -- the one-sided physical inflow/outflow
    condition is a separate open question on this branch and is deliberately
    not what is being tested. */
Result Solve(int order, int n, int m, const Array<int> *comps,
             bool advection = false)
{
   const int dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, m), Wh(&mesh, &p_coll), Mh(&mesh,
                                                                    &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), zero(0.0);
   FunctionCoefficient pcoeff(PExact);
   Vector cvec(dim); cvec(0) = 1.0; cvec(1) = 0.0;
   VectorConstantCoefficient ccoeff(cvec);

   // **A flux block IS the diffusion here**, so the source follows @a m and
   // not only @a advection: -Lap p contributes -2 whenever there is a flux to
   // carry it, and div(c p) contributes 2x whenever the convection is
   // installed. Getting this wrong is not subtle and is how the control arm
   // below was first written -- an advection source against an
   // advection-diffusion operator, which came back 8.2e-02 instead of
   // round-off. The sign is DarcyForm's residual convention and was settled by
   // running both: the other gives 8.5e-02 and 8.9e-01.
   const bool diffusive = (m > 0);
   FunctionCoefficient fcoeff([advection, diffusive](const Vector &x)
   {
      return (advection ? (2.0 * x(0)) : 0.0) + (diffusive ? -2.0 : 0.0);
   });

   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;

   if (m > 0)
   {
      // SetVDim() and not the default: VectorMassIntegrator takes its
      // component count from the caller and defaults to the SPACE dimension,
      // so this is the one block of the four that needed nothing new.
      VectorMassIntegrator *vm = new VectorMassIntegrator(one);
      vm->SetVDim(m);
      darcy.GetFluxMassForm()->AddDomainIntegrator(vm);
   }

   MixedBilinearForm *B = darcy.GetFluxDivForm();
   if (comps)
   {
      B->AddDomainIntegrator(new RestrictedVectorDivergenceIntegrator(*comps));
   }
   else
   {
      B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   }
   // Read for its MARKER and nothing else on the hybridized path: it is what
   // puts the flux constraint on the boundary attributes, and without one the
   // constraint is interior-only and every element touching the boundary is
   // wrong.
   B->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)), all);

   NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
   Mnl_p->AddDomainIntegrator(new MassIntegrator(zero));
   if (advection)
   {
      Mnl_p->AddDomainIntegrator(new ConservativeConvectionIntegrator(ccoeff));
      Mnl_p->AddInteriorFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
      Mnl_p->AddBdrFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
   }
   // The HDG diffusion stabilization belongs to the flux and goes in exactly
   // when there is one, which is the other half of the source above.
   if (diffusive)
   {
      Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
      Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   }

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

   Array<int> ess_flux;
   BilinearFormIntegrator *cbfi =
      comps ? (BilinearFormIntegrator*)
      new RestrictedNormalTraceJumpIntegrator(*comps)
      : (BilinearFormIntegrator*) new NormalTraceJumpIntegrator();
   darcy.EnableHybridization(&Mh, cbfi, ess_flux);

   DarcyHybridization *dh = darcy.GetHybridization();
   dh->EnableNPC();
   dh->SetEssentialBC(all);

   darcy.Assemble();
   darcy.Finalize();

   // ProjectBdrCoefficient and not ProjectCoefficient: the latter segfaults
   // on a face-based trace space.
   GridFunction tr(&Mh);
   tr = 0.0;
   tr.ProjectBdrCoefficient(pcoeff, all);
   Vector x_tr(Mh.GetVSize());
   x_tr = 0.0;
   Array<int> bdr_dofs;
   Mh.GetEssentialTrueDofs(all, bdr_dofs);
   for (int i = 0; i < bdr_dofs.Size(); i++)
   {
      x_tr(bdr_dofs[i]) = tr(bdr_dofs[i]);
   }

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(1) -= *darcy.GetPotentialRHS();
   b.GetBlock(1).SyncAliasMemory(b);

   // A non-zero start, so all three blocks are in play; the problem is affine
   // so one step lands on the answer from anywhere.
   x = 1.0;

   BlockVector r(darcy.GetOffsets());
   Vector r_tr, b_tr, dtr;
   dh->NPCResidual(b, x, x_tr, r, r_tr);
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
   x += dx;
   x_tr += dtr;

   Result res;
   res.flux_vsize = Vh.GetVSize();
   // Norml2() cannot see a NaN, so every norm below is worthless until this
   // says so.
   res.nonfinite = x.CheckFinite() + x_tr.CheckFinite() + r.CheckFinite()
                   + r_tr.CheckFinite() + dtr.CheckFinite();

   GridFunction q(&Vh), p(&Wh);
   q.MakeRef(&Vh, x.GetBlock(0), 0);
   p.MakeRef(&Wh, x.GetBlock(1), 0);
   res.err_p = p.ComputeL2Error(pcoeff);

   if (m > 0)
   {
      Array<int> cc;
      if (comps) { comps->Copy(cc); }
      else { cc.SetSize(m); for (int i = 0; i < m; i++) { cc[i] = i; } }
      VectorFunctionCoefficient ucoeff(m, [&cc](const Vector &X, Vector &u)
      {
         Vector full(2);
         UFull(X, full);
         for (int i = 0; i < cc.Size(); i++) { u(i) = full(cc[i]); }
      });
      res.err_u = q.ComputeL2Error(ucoeff);
   }
   return res;
}

} // namespace darcy_restricted_flux

TEST_CASE("A flux carrying one direction solves what the full one does",
          "[DarcyForm][NonlinearDarcy][HDG][RestrictedFlux]")
{
   using namespace darcy_restricted_flux;

   const int order = GENERATE(2, 3);
   const int n = GENERATE(2, 4);

   CAPTURE(order, n);

   Array<int> kx(1); kx[0] = 0;
   Array<int> ky(1); ky[0] = 1;
   Array<int> kxy(2); kxy[0] = 0; kxy[1] = 1;

   const Result full = Solve(order, n, 2, NULL);
   const Result keep_x = Solve(order, n, 1, &kx);
   const Result keep_y = Solve(order, n, 1, &ky);
   const Result keep_xy = Solve(order, n, 2, &kxy);

   REQUIRE(full.nonfinite == 0);
   REQUIRE(keep_x.nonfinite == 0);
   REQUIRE(keep_y.nonfinite == 0);
   REQUIRE(keep_xy.nonfinite == 0);

   // The restricted arms really do carry fewer unknowns; without this an arm
   // that quietly fell back to the full space would pass everything below.
   REQUIRE(keep_x.flux_vsize * 2 == full.flux_vsize);
   REQUIRE(keep_xy.flux_vsize == full.flux_vsize);

   SECTION("the full flux is exact, which is the control on the problem")
   {
      REQUIRE(full.err_p == MFEM_Approx(0.0));
      REQUIRE(full.err_u == MFEM_Approx(0.0));
   }

   SECTION("a flux carrying only x reproduces it")
   {
      // Measured at order 2 on 4x4: 5.93e-15 against the full arm's 1.53e-15.
      REQUIRE(keep_x.err_p == MFEM_Approx(0.0));
      REQUIRE(keep_x.err_u == MFEM_Approx(0.0));
   }

   SECTION("a flux carrying only y does NOT, which is what makes that a check")
   {
      // -d_yy p = -2 with p = x^2 + y on the boundary has the solution
      // x^2 + y^2, so this arm is solving a different problem and must say
      // so. Measured 1.77e-01 at order 2 on 4x4. Were this to pass, the
      // section above would be consistent with the component list being
      // ignored entirely.
      REQUIRE(keep_y.err_p > 1e-3);
   }

   SECTION("restricting to every direction is the identity")
   {
      // The wrapper introduces nothing of its own. Measured bit-identical to
      // the stock arm at every order and mesh tried; asserted to a margin
      // rather than exactly, the claim being "no arithmetic of its own" and
      // not "the compiler emitted the same instructions".
      REQUIRE(keep_xy.err_p == MFEM_Approx(full.err_p));
      REQUIRE(keep_xy.err_u == MFEM_Approx(full.err_u));
   }
}

TEST_CASE("A flux carrying no direction at all is pure HDG advection",
          "[DarcyForm][NonlinearDarcy][HDG][RestrictedFlux]")
{
   using namespace darcy_restricted_flux;

   const int order = GENERATE(2, 3);
   CAPTURE(order);

   Array<int> none(0);
   Array<int> kxy(2); kxy[0] = 0; kxy[1] = 1;

   // div(c p) = 2x with c = (1,0) is well posed with no flux unknown at all,
   // and PExact solves it. This is the request taken to its end: the flux
   // space has vdim 0, no flux mass form is created, and every flux-shaped
   // block is zero-wide.
   const Result novdim = Solve(order, 2, 0, &none, true);
   REQUIRE(novdim.nonfinite == 0);
   REQUIRE(novdim.flux_vsize == 0);
   // Measured 5.68e-16; the wrong source sign gives 8.9e-01, so the value is
   // driven by the equation and not by the boundary datum alone.
   REQUIRE(novdim.err_p == MFEM_Approx(0.0));

   SECTION("and the same advection with a full flux beside it agrees")
   {
      // The control: the driver, the datum and the source signs are right
      // independently of the flux being absent. It is advection PLUS
      // diffusion, the flux block being the diffusion, so it carries both
      // halves of the source.
      const Result withflux = Solve(order, 2, 2, &kxy, true);
      REQUIRE(withflux.nonfinite == 0);
      REQUIRE(withflux.err_p == MFEM_Approx(0.0));
   }
}

TEST_CASE("The restricted flux blocks are slices of the unrestricted ones",
          "[DarcyForm][HDG][RestrictedFlux]")
{
   using namespace darcy_restricted_flux;

   // No solve and no exact solution: the design rests on the claim that the
   // restriction IS a slice, and that is arithmetic. Checked against
   // DerivativeIntegrator, which reaches the same quantity by a different
   // route (dshapedxt against GradToDiv on the adjugate).
   const int order = 2, dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vfull(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   const FiniteElement *feu = Vfull.GetFE(0);
   const FiniteElement *fep = Wh.GetFE(0);
   ElementTransformation *Tr = mesh.GetElementTransformation(0);
   const int nd = feu->GetDof();

   SECTION("the divergence, against DerivativeIntegrator")
   {
      VectorDivergenceIntegrator base;
      DenseMatrix full;
      base.AssembleElementMatrix2(*feu, *fep, *Tr, full);
      const IntegrationRule &ir = base.GetElementIntRule(*feu, *fep, *Tr);
      ConstantCoefficient one(1.0);

      for (int d = 0; d < dim; d++)
      {
         Array<int> comps(1); comps[0] = d;
         RestrictedVectorDivergenceIntegrator rest(comps);
         DenseMatrix R;
         rest.AssembleElementMatrix2(*feu, *fep, *Tr, R);
         REQUIRE(R.Height() == full.Height());
         REQUIRE(R.Width() == nd);

         DerivativeIntegrator di(one, d);
         di.SetIntRule(&ir);
         DenseMatrix D;
         di.AssembleElementMatrix2(*feu, *fep, *Tr, D);

         DenseMatrix diff(R);
         diff -= D;
         REQUIRE(D.MaxMaxNorm() > 1e-3);          // the comparison is not vacuous
         REQUIRE(diff.MaxMaxNorm() == MFEM_Approx(0.0));
      }
   }

   SECTION("the normal trace, and the element-2 offset it has to move")
   {
      int iface = -1;
      for (int f = 0; f < mesh.GetNumFaces(); f++)
      {
         if (mesh.GetInteriorFaceTransformations(f)) { iface = f; break; }
      }
      REQUIRE(iface >= 0);
      FaceElementTransformations *FTr = mesh.GetInteriorFaceTransformations(iface);
      const FiniteElement *fe1 = Vfull.GetFE(FTr->Elem1No);
      const FiniteElement *fe2 = Vfull.GetFE(FTr->Elem2No);
      const FiniteElement *fef = Mh.GetFaceElement(iface);
      const int nd1 = fe1->GetDof(), nd2 = fe2->GetDof(), nf = fef->GetDof();

      NormalTraceJumpIntegrator base;
      DenseMatrix full;
      base.AssembleFaceMatrix(*fef, *fe1, *fe2, *FTr, full);
      REQUIRE(full.Height() == (nd1 + nd2) * dim);

      for (int d = 0; d < dim; d++)
      {
         Array<int> comps(1); comps[0] = d;
         RestrictedNormalTraceJumpIntegrator rest(comps);
         DenseMatrix R;
         rest.AssembleFaceMatrix(*fef, *fe1, *fe2, *FTr, R);
         REQUIRE(R.Height() == nd1 + nd2);
         REQUIRE(R.Width() == nf);

         // Element 1's rows come from d*nd1 and element 2's from
         // dim*nd1 + d*nd2 -- the second offset is the one that moves, and
         // reading it at d*nd2 instead is the defect the shape guards exist
         // to make loud.
         real_t worst = 0.0;
         for (int i = 0; i < nd1; i++)
            for (int j = 0; j < nf; j++)
            {
               worst = std::max(worst, std::abs(R(i, j) - full(d * nd1 + i, j)));
            }
         for (int i = 0; i < nd2; i++)
            for (int j = 0; j < nf; j++)
            {
               worst = std::max(worst,
                                std::abs(R(nd1 + i, j)
                                         - full(dim * nd1 + d * nd2 + i, j)));
            }
         REQUIRE(worst == MFEM_Approx(0.0));
      }
   }
}

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("The unsupported restricted flux configurations are refused",
          "[DarcyForm][NonlinearDarcy][HDG][RestrictedFlux]")
{
   using namespace darcy_restricted_flux;

   // **This section does not execute in a build without exceptions, which is
   // both HDG trees**, so every refusal was additionally tripped BY HAND, one
   // process per guard, and the routine that aborted was read off the
   // backtrace. What that established, and it is recorded here because the
   // assertions below cannot:
   //
   //   configuration                          aborts in
   //   both integrators stock, vdim 1         CheckConstraintBlockShape
   //   stock divergence, restricted trace     CheckRestrictedFluxAgreement
   //   divergence {x} against trace {y}       CheckRestrictedFluxAgreement
   //   component 5 in two dimensions          CheckFluxComponents
   //   components {1, 0}                      CheckFluxComponents
   //   a broken-RT (vector-range) flux        CheckRestrictedFluxConfiguration
   //   a block nonlinear flux law             CheckRestrictedFluxConfiguration
   //   a flux mass with 1/kappa infinite      AssembleFluxMassMatrix
   //
   // TWO ENTRIES HAVE BEEN REMOVED FROM THAT TABLE and are recorded here so
   // the list is not read as still complete: "LINEAR, restricted, no
   // EnableNPC()" in CheckRestrictedFluxConfiguration and "the reduced
   // operator's own Mult()" in DarcyHybridization::Mult. Both refused the
   // condensation route, both were POLICY rather than defects -- the grounds
   // given were that nothing had run it -- and running it reproduces the NPC
   // route to every printed digit. The measurement is on
   // CheckRestrictedFluxConfiguration(); the coverage is six serial and six
   // parallel p11 references driving the condensation route, four nonlinear
   // and two linear.
   //
   // and the arms that must NOT abort and do not: the supported restricted
   // configuration, a linear one with EnableNPC(), a full flux with a finite
   // coefficient, and -- since the two refusals above went -- a restricted
   // flux on the condensation route, linear and nonlinear alike. A guard
   // nobody has seen refuse is a guard nobody has tested.
   //
   // The one that is worth stating twice: before the shape guards existed,
   // "both integrators stock, vdim 1" ran to completion and returned
   // err_p = 6.8e-01 against a correct 1.5e-15, finite, with CheckFinite()
   // returning 0 and no flag of any kind.

   SECTION("the stock integrators at a short flux vdim")
   {
      REQUIRE_THROWS(Solve(2, 2, 1, NULL));
   }
}
#endif // MFEM_USE_EXCEPTIONS

#ifdef MFEM_USE_MPI
TEST_CASE("A restricted flux is exact on more than one rank",
          "[DarcyForm][NonlinearDarcy][HDG][NPC][RestrictedFlux][Parallel]")
{
   using namespace darcy_restricted_flux;

   // **The one thing a restricted flux does differently in parallel is the
   // constraint on a SHARED face.** ConstructC() assembles those through a
   // separate loop, with the neighbour's element read through
   // ParFiniteElementSpace::GetFaceNbrFE(), and the row offset the restricted
   // integrator has to get right is the same one it gets right on an interior
   // face -- but nothing serial exercises that loop. The flux and potential
   // are L2 and need no mapping; the trace is prolonged in and assembled out.
   //
   // A problem whose discrete answer is exact is the sharp instrument for
   // that: if the shared-face constraint is built at the wrong stride, the
   // partition boundary is wrong and this cannot come back at round-off.
   CAPTURE(Mpi::WorldSize());

   const int order = 2, n = 4, dim = 2;
   Mesh serial_mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);

   // Both arms, on one mesh: {x} is the operator this problem wants and {y}
   // is not, exactly as in the serial case above.
   const int keep = GENERATE(0, 1);
   CAPTURE(keep);
   Array<int> comps(1); comps[0] = keep;

   ParFiniteElementSpace Vh(&mesh, &u_coll, 1), Wh(&mesh, &p_coll),
                         Mh(&mesh, &t_coll);
   ParDarcyForm darcy(&Vh, &Wh);

   ConstantCoefficient one(1.0), zero(0.0), fcoeff(-2.0);
   FunctionCoefficient pcoeff(PExact);
   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;

   VectorMassIntegrator *vm = new VectorMassIntegrator(one);
   vm->SetVDim(1);
   darcy.GetFluxMassForm()->AddDomainIntegrator(vm);

   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new RestrictedVectorDivergenceIntegrator(comps));
   B->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)), all);

   NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
   Mnl_p->AddDomainIntegrator(new MassIntegrator(zero));
   Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new RestrictedNormalTraceJumpIntegrator(comps),
                             ess_flux);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->EnableNPC();
   dh->SetEssentialBC(all);
   darcy.Assemble();
   darcy.Finalize();

   // The trace block is sized on TRUE dofs, which is what NPC's interface
   // takes; the other two are L2 and are the same either way.
   Array<int> offs(4);
   offs[0] = 0;
   offs[1] = Vh.GetVSize();
   offs[2] = Wh.GetVSize();
   offs[3] = Mh.GetTrueVSize();
   offs.PartialSum();
   BlockVector sol(offs);
   sol = 1.0;

   ParGridFunction tgf(&Mh);
   tgf = 0.0;
   tgf.ProjectBdrCoefficient(pcoeff, all);
   Vector x_tr(Mh.GetTrueVSize());
   tgf.ParallelProject(x_tr);

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(1) -= *darcy.GetPotentialRHS();
   b.GetBlock(1).SyncAliasMemory(b);
   x = 1.0;

   BlockVector r(darcy.GetOffsets()), dx(darcy.GetOffsets());
   Vector r_tr, b_tr, dtr;
   dh->NPCResidual(b, x, x_tr, r, r_tr);
   Operator &S = dh->NPCGradient(x, x_tr);
   dh->NPCReduce(r, r_tr, b_tr);

   dtr.SetSize(b_tr.Size());
   dtr = 0.0;
   HypreParMatrix *Sp = dynamic_cast<HypreParMatrix*>(&S);
   REQUIRE(Sp != nullptr);
   {
      HypreBoomerAMG amg(*Sp);
      amg.SetPrintLevel(0);
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetOperator(*Sp);
      gmres.SetPreconditioner(amg);
      gmres.SetKDim(200);
      gmres.SetMaxIter(2000);
      gmres.SetRelTol(1e-14);
      gmres.SetAbsTol(0.0);
      gmres.SetPrintLevel(-1);
      gmres.Mult(b_tr, dtr);
   }

   dx = 0.0;
   dh->NPCRecover(r, dtr, dx);
   x += dx;

   REQUIRE(x.CheckFinite() == 0);

   ParGridFunction p(&Wh);
   p.MakeRef(&Wh, x.GetBlock(1), 0);
   const real_t err_p = p.ComputeL2Error(pcoeff);
   CAPTURE(err_p);

   if (keep == 0)
   {
      // The same statement as the serial arm, over a partitioned mesh: the
      // shared-face constraint has to be right or this is not round-off.
      REQUIRE(err_p < 1e-10);
   }
   else
   {
      // And the discriminator travels: -d_yy p = -2 is a different problem
      // on any number of ranks.
      REQUIRE(err_p > 1e-3);
   }
}
#endif // MFEM_USE_MPI

namespace darcy_restricted_weak_bc
{

// p = x y + y^2: d_xx p = 0, d_yy p = 2, d_x p = y, so the problem
// -d_yy p + d_x(c p) = -2 + y is solved by it and a flux carrying {y} alone
// is the right flux for it. Degree 2, so exact in the discrete spaces.
//
// **What this adds to the file, and it is the boundary and not the flux.**
// Every case above imposes the trace ESSENTIALLY on the whole boundary. The
// caller who asked for a restricted flux does not: the datum goes in weakly,
// through VectorBoundaryFluxLFIntegrator on the flux load and
// BoundaryFlowIntegrator on the potential load, which is what
// miniapps/hdg/convdiff.cpp does by default. That route sizes its element
// vector from the MESH dimension, so at vdim < dim it writes the datum into
// the wrong component and LinearForm::Assemble() truncates in silence.
real_t PExact(const Vector &x) { return x(0)*x(1) + x(1)*x(1); }
real_t FExact(const Vector &x) { return -2.0 + x(1); }

/** One NPC step with the datum imposed WEAKLY on every attribute.

    @a lf_comps is what the boundary flux load is told the space carries. It
    is normally @a comps; passing something else is the falsification, and it
    is a mis-specification a caller can actually make. */
real_t Solve(int order, int n, const Array<int> &comps,
             const Array<int> &lf_comps)
{
   const int dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, comps.Size()), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), zero(0.0);
   FunctionCoefficient pcoeff(PExact), fcoeff(FExact);
   ProductCoefficient mpcoeff(-1.0, pcoeff);
   Vector cvec(dim); cvec(0) = 1.0; cvec(1) = 0.0;
   VectorConstantCoefficient ccoeff(cvec);

   VectorMassIntegrator *vm = new VectorMassIntegrator(one);
   vm->SetVDim(comps.Size());
   darcy.GetFluxMassForm()->AddDomainIntegrator(vm);

   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new RestrictedVectorDivergenceIntegrator(comps));

   NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
   Mnl_p->AddDomainIntegrator(new MassIntegrator(zero));
   Mnl_p->AddDomainIntegrator(new ConservativeConvectionIntegrator(ccoeff));
   Mnl_p->AddInteriorFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
   Mnl_p->AddBdrFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
   Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   // The two halves of a weak Dirichlet datum, on two different forms and
   // with OPPOSITE signs in this convention -- see convdiff.cpp, whose
   // gcoeff is -tcoeff while its BoundaryFlowIntegrator takes tcoeff.
   darcy.GetFluxRHS()->AddBdrFaceIntegrator(
      new RestrictedVectorBoundaryFluxLFIntegrator(lf_comps, pcoeff), all);
   darcy.GetPotentialRHS()->AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(mpcoeff, ccoeff, +1.0), all);

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new RestrictedNormalTraceJumpIntegrator(comps),
                             ess_flux);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->EnableNPC();

   darcy.Assemble();
   darcy.Finalize();

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   darcy.GetFluxRHS()->Assemble();
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(0) -= *darcy.GetFluxRHS();
   b.GetBlock(1) -= *darcy.GetPotentialRHS();
   b.GetBlock(0).SyncAliasMemory(b);
   b.GetBlock(1).SyncAliasMemory(b);

   x = 1.0;
   Vector x_tr(Mh.GetVSize());
   x_tr = 0.0;

   BlockVector r(darcy.GetOffsets());
   Vector r_tr, b_tr, dtr;
   dh->NPCResidual(b, x, x_tr, r, r_tr);
   Operator &S = dh->NPCGradient(x, x_tr);
   dh->NPCReduce(r, r_tr, b_tr);

   dtr.SetSize(b_tr.Size());
   dtr = 0.0;
   SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
   REQUIRE(Sm != nullptr);
#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver umf(*Sm);
   umf.Mult(b_tr, dtr);
#else
   GSSmoother prec(*Sm);
   GMRESSolver gmres;
   gmres.SetOperator(S);
   gmres.SetPreconditioner(prec);
   gmres.SetKDim(200);
   gmres.SetMaxIter(5000);
   gmres.SetRelTol(1e-14);
   gmres.SetAbsTol(0.0);
   gmres.SetPrintLevel(-1);
   gmres.Mult(b_tr, dtr);
#endif

   BlockVector dx(darcy.GetOffsets());
   dx = 0.0;
   dh->NPCRecover(r, dtr, dx);
   x += dx;

   // Norml2() cannot see a NaN; assert finiteness before believing any norm.
   REQUIRE(x.CheckFinite() == 0);
   REQUIRE(dtr.CheckFinite() == 0);

   GridFunction p(&Wh);
   p.MakeRef(&Wh, x.GetBlock(1), 0);
   return p.ComputeL2Error(pcoeff);
}

} // namespace darcy_restricted_weak_bc

TEST_CASE("A restricted flux takes its Dirichlet datum weakly",
          "[DarcyForm][NonlinearDarcy][HDG][RestrictedFlux]")
{
   using namespace darcy_restricted_weak_bc;

   Array<int> keep_y(1); keep_y[0] = 1;
   Array<int> keep_x(1); keep_x[0] = 0;

   SECTION("the load names the direction the space carries")
   {
      // The whole boundary weak, the trace non-essential everywhere, and the
      // solution not zero on any of it: 1 + y at the outflow, y + 1 at the
      // top. Exact because p is degree 2 and in the space.
      const real_t err = Solve(2, 4, keep_y, keep_y);
      CAPTURE(err);
      REQUIRE(err < 1e-10);
   }

   SECTION("and naming the wrong one is not a no-op")
   {
      // The falsification, and it is a mis-specification a caller can make:
      // the space carries y and the load is told x, so the datum multiplies
      // n_x on faces where the flux has no x component. Without it the
      // section above would pass just as well with the load doing nothing.
      const real_t err = Solve(2, 4, keep_y, keep_x);
      CAPTURE(err);
      REQUIRE(err > 1e-3);
   }
}
