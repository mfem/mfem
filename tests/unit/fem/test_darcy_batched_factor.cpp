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

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <cstring>
#include <memory>

using namespace mfem;

namespace darcy_batched_factor
{

// A semilinear problem: (c p^2, w) on the potential mass form and nothing
// else nonlinear. That combination is what puts DarcyHybridization into
// LocalOpType::PotNL, and PotNL is one of the only two cases whose local
// factorisation goes through InvertA() -- see DarcyHybridization::Finalize().
// A block or a coupled flux nonlinearity takes FullNL, where ComputeElementH()
// factors A itself, and would never reach the code under test here.
class SquareSource : public NonlinearFormIntegrator
{
public:
   explicit SquareSource(real_t c_) : c(c_) { }

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                               2*el.GetOrder() + 2);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcPhysShape(Tr, shape);
         const real_t u = shape * elfun;
         elvect.Add(ip.weight * Tr.Weight() * c * u * u, shape);
      }
   }

   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elmat.SetSize(dof);
      elmat = 0.0;
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                               2*el.GetOrder() + 2);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcPhysShape(Tr, shape);
         const real_t u = shape * elfun;
         AddMult_a_VVt(ip.weight * Tr.Weight() * 2.0 * c * u, shape, elmat);
      }
   }

private:
   real_t c;
   Vector shape;
};

/** The same bits, and equality rather than a tolerance is the right test
    here -- but only because of a fact about this build, not because
    batching is inherently exact.

    BatchedLinAlg's native backend factors through kernels::LUFactor(), and
    LUFactors::Factor() *is* that same routine when MFEM_USE_LAPACK is
    undefined: the same partial pivoting, the same elimination order, the
    same 1-based ipiv. So the two modes are the same arithmetic on the same
    data and a difference of any size is a defect. With LAPACK,
    LUFactors::Factor() calls getrf_ instead, whose blocked update
    reassociates, and the honest criterion becomes a tolerance -- which is
    why the assertion below is compiled differently in that build. */
bool BitwiseEqual(const Vector &a, const Vector &b)
{
   if (a.Size() != b.Size()) { return false; }
   return std::memcmp(a.GetData(), b.GetData(),
                      a.Size()*sizeof(real_t)) == 0;
}

void RequireSame(const Vector &ref, const Vector &got)
{
#ifndef MFEM_USE_LAPACK
   REQUIRE(BitwiseEqual(ref, got));
#else
   REQUIRE(ref.Size() == got.Size());
   Vector d(ref);
   d -= got;
   REQUIRE(d.Normlinf() == MFEM_Approx(0.0, 1e-12, 1e-12));
#endif
}

struct Outcome
{
   Vector q, p;
   bool can_batch = false;
   bool converged = false;
};

/// Solve the semilinear hybridized problem once, in the given factor mode.
Outcome Solve(Mesh &mesh, int order, real_t c,
              DarcyHybridization::LocalFactorMode mode, int max_it = 30)
{
   const int dim = mesh.Dimension();

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim);
   FiniteElementSpace Wh(&mesh, &p_coll);
   FiniteElementSpace Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0);
   FunctionCoefficient src([](const Vector &X)
   {
      return std::sin(M_PI*X(0))*std::sin(M_PI*X(1));
   });

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
   Mnl_p->AddDomainIntegrator(new SquareSource(c));
   Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);

   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetLocalFactorMode(mode);
   dh->SetLocalNLSolver(DarcyHybridization::LSsolveType::Newton, 1000, 1e-14,
                        1e-30);
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   dh->SetEssentialBC(ess_bdr);

   darcy.Assemble();

   Outcome out;
   out.can_batch = dh->CanBatchLocalFactor();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorHandle R;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux, x, R, X, B, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(200);
   lin.SetMaxIter(2000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(0.0);
   lin.SetPreconditioner(prec);

   NewtonSolver newton;
   newton.SetSolver(lin);
   newton.SetOperator(*R.Ptr());
   newton.SetRelTol(1e-12);
   newton.SetAbsTol(1e-14);
   newton.SetMaxIter(max_it);
   newton.SetPrintLevel(-1);
   newton.Mult(B, X);
   out.converged = newton.GetConverged();

   darcy.RecoverFEMSolution(X, x);
   out.q = x.GetBlock(0);
   out.p = x.GetBlock(1);
   return out;
}


struct LinearOutcome
{
   Vector q, p, tr;
   /// The assembled trace operator, entry for entry; empty under MatrixFree.
   Array<int> HI, HJ;
   Vector Hdata;
   bool can_batch_factor = false;
   bool can_batch_solve = false;
};

/// The same problem with a LINEAR potential mass, solved once in each mode.
///
/// The nonlinear case above exercises the local FACTORISATION, because
/// LocalOpType::PotNL reaches InvertA() and then solves its local problems
/// with MultInvNL(). A linear problem is what reaches MultInv(), through
/// ReduceRHS() and ComputeSolution() -- so it is the linear case that says
/// anything about the batched local SOLVE, and the two cases are not
/// substitutes for one another.
LinearOutcome SolveLinear(Mesh &mesh, int order,
                          DarcyHybridization::LocalFactorMode mode,
                          DarcyHybridization::GradientMode gmode =
                             DarcyHybridization::GradientMode::Assembled)
{
   const int dim = mesh.Dimension();

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim);
   FiniteElementSpace Wh(&mesh, &p_coll);
   FiniteElementSpace Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0);
   FunctionCoefficient src([](const Vector &X)
   {
      return std::sin(M_PI*X(0))*std::sin(M_PI*X(1));
   });

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   BilinearForm *M_p = darcy.GetPotentialMassForm();
   M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);

   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetLocalFactorMode(mode);
   dh->SetGradientMode(gmode);
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   dh->SetEssentialBC(ess_bdr);

   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorHandle R;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux, x, R, X, B, true);

   LinearOutcome out;
   out.can_batch_factor = dh->CanBatchLocalFactor();
   out.can_batch_solve = dh->CanBatchLocalSolve();

   // The operator itself, when there is one. Comparing a SOLUTION lets a
   // trace solve at a finite tolerance absorb a small error in the operator
   // it is solving; comparing the assembled matrix asks the same question
   // without that slack, and it is the only thing here that reaches
   // ComputeH()'s own factorisation and Schur complement. InvertA(), which
   // the semilinear case exercises, runs for LocalOpType::PotNL and FluxNL
   // alone -- a LINEAR problem's A is factored inside ComputeElementH().
   if (SparseMatrix *H = dynamic_cast<SparseMatrix*>(R.Ptr()))
   {
      const int nrows = H->Height(), nnz = H->NumNonZeroElems();
      out.HI.SetSize(nrows+1);
      std::copy(H->GetI(), H->GetI()+nrows+1, out.HI.begin());
      out.HJ.SetSize(nnz);
      std::copy(H->GetJ(), H->GetJ()+nnz, out.HJ.begin());
      out.Hdata.SetSize(nnz);
      std::copy(H->GetData(), H->GetData()+nnz, out.Hdata.GetData());
   }

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(200);
   lin.SetMaxIter(2000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(0.0);
   // GSSmoother needs the matrix, and GradientMode::MatrixFree carries none;
   // that mode runs unpreconditioned, which is what SetGradientMode() says it
   // costs.
   if (out.Hdata.Size() > 0) { lin.SetPreconditioner(prec); }
   lin.SetOperator(*R.Ptr());
   lin.SetPrintLevel(-1);
   lin.Mult(B, X);

   darcy.RecoverFEMSolution(X, x);
   out.q = x.GetBlock(0);
   out.p = x.GetBlock(1);
   out.tr = X;
   return out;
}


struct NPCOutcome
{
   BlockVector dx;
   Vector dtr;
   real_t n0 = 0.0, n1 = 0.0;
   bool can_batch_solve = false;
   NPCOutcome() : dx() { }
};

/// One NPC Newton step on the semilinear problem, in the given factor mode.
///
/// The linear case above reaches MultInv() through ReduceRHS() and
/// ComputeSolution(). A NONLINEAR problem never reaches either --
/// NPCEnabled() is `bnpc || IsNonlinear()`, so ReduceRHS() returns after
/// stashing the load -- and its local solves go through NPCReduce() and
/// NPCRecover() instead. Those are the ones that run once per Newton step,
/// so they are where batching is worth the most, and nothing else here
/// exercises them.
///
/// What this does NOT cover: the Bnl term inside MultInvBatched(). It is
/// reached only when the flux law depends on the potential
/// (LocalOpType::FluxNL), and a potential-mass nonlinearity leaves Bnl empty.
void NPCStep(Mesh &mesh, int order, real_t c,
             DarcyHybridization::LocalFactorMode mode, NPCOutcome &out,
             DarcyHybridization::GradientMode gmode =
                DarcyHybridization::GradientMode::Assembled)
{
   const int dim = mesh.Dimension();

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim);
   FiniteElementSpace Wh(&mesh, &p_coll);
   FiniteElementSpace Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0);
   FunctionCoefficient src([](const Vector &X)
   {
      return std::sin(M_PI*X(0))*std::sin(M_PI*X(1));
   });

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
   Mnl_p->AddDomainIntegrator(new SquareSource(c));
   Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);

   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetLocalFactorMode(mode);
   dh->SetGradientMode(gmode);
   dh->EnableNPC();
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   dh->SetEssentialBC(ess_bdr);

   darcy.Assemble();
   darcy.Finalize();

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   x = 0.0;
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(1) += *darcy.GetPotentialRHS();

   Vector x_tr(Mh.GetVSize());
   x_tr = 0.0;

   BlockVector r(darcy.GetOffsets());
   Vector r_tr, b_tr, dtr;

   auto full_norm = [](const BlockVector &rl, const Vector &rt)
   {
      return std::sqrt(rl*rl + rt*rt);
   };

   dh->NPCResidual(b, x, x_tr, r, r_tr);
   out.n0 = full_norm(r, r_tr);

   Operator &S = dh->NPCGradient(x, x_tr);
   out.can_batch_solve = dh->CanBatchLocalSolve();
   dh->NPCReduce(r, r_tr, b_tr);

   dtr.SetSize(b_tr.Size());
   dtr = 0.0;
   {
      SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
      // GradientMode::MatrixFree carries no matrix, so it runs
      // unpreconditioned; see SetGradientMode().
      std::unique_ptr<GSSmoother> prec;
      if (Sm) { prec.reset(new GSSmoother(*Sm)); }
      GMRESSolver gmres;
      gmres.SetOperator(S);
      if (prec) { gmres.SetPreconditioner(*prec); }
      gmres.SetKDim(200);
      gmres.SetMaxIter(2000);
      gmres.SetRelTol(1e-14);
      gmres.SetAbsTol(0.0);
      gmres.SetPrintLevel(-1);
      gmres.Mult(b_tr, dtr);
   }

   BlockVector dx(darcy.GetOffsets());
   dh->NPCRecover(r, dtr, dx);
   x += dx;
   x_tr += dtr;

   dh->NPCResidual(b, x, x_tr, r, r_tr);
   out.n1 = full_norm(r, r_tr);

   out.dx.Update(darcy.GetOffsets());
   out.dx = dx;
   out.dtr = dtr;
}

} // namespace darcy_batched_factor

TEST_CASE("The batched local factorisation gives the serial one's answer",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_factor;
   using LFM = DarcyHybridization::LocalFactorMode;

   const int order = GENERATE(0, 1, 2);
   const real_t c = 5.0;

   SECTION("a uniform mesh batches, and agrees with the loop")
   {
      const int n = GENERATE(2, 4);
      Mesh mesh_a = Mesh::MakeCartesian2D(n, n, Element::TRIANGLE);
      Mesh mesh_b = Mesh::MakeCartesian2D(n, n, Element::TRIANGLE);

      const Outcome ref = Solve(mesh_a, order, c, LFM::Serial);
      const Outcome got = Solve(mesh_b, order, c, LFM::Batched);

      CAPTURE(order, n);
      REQUIRE(ref.converged);
      REQUIRE(got.converged);

      // The point of the section: the batched path was actually taken. A
      // fallback here would make the comparison below compare Serial with
      // Serial and pass while testing nothing.
      REQUIRE(got.can_batch);

      RequireSame(ref.p, got.p);
      RequireSame(ref.q, got.q);
   }

   SECTION("a mesh of two element types falls back, and asking costs nothing")
   {
      // 8 triangles and 12 squares. At order 0 both carry one potential dof
      // and the blocks are uniform after all, which is why the assertion
      // below is on can_batch rather than on the element types: it is dof
      // counts that decide, and only at order >= 1 do P_k and Q_k differ.
      Mesh mesh_a("../../data/square-mixed.mesh", 1, 1);
      Mesh mesh_b("../../data/square-mixed.mesh", 1, 1);

      // Five steps, and convergence deliberately not required. A mixed-element
      // mesh at order >= 1 does not converge here, and that is a defect in the
      // hybridization rather than anything to do with this setting: the
      // residual is right and the Jacobian is not. Measured on this mesh at
      // order 1, with a *direct* trace solve so the linear solver is not in
      // question -- Newton falls by a constant factor of about 1.7 per step
      // and stalls at 1.2e-08 after 30, while LBFGS, which never asks for a
      // gradient, reaches 5.5e-14 in 36 and lands on the same solution to six
      // digits. The same problem on all-quadrilateral and on all-triangle
      // meshes converges in three Newton steps to 2e-16. Nothing else in the
      // suite runs Darcy on a mixed mesh, which is why it had not been seen.
      //
      // FIXING IT IS NOT THIS BRANCH'S, AND THE TWO HALVES NEVER MEET HERE.
      // gf-hdg-p-adaptivity wants a mixed mesh -- variable order needs an NC
      // mesh, and its hp work reaches simplices and 3D -- so the repair
      // belongs there and arrives with that branch. But that branch is off
      // the trunk and this file is this branch's alone: it does not exist
      // there, so whoever fixes the Jacobian will not see this comment, and
      // the fix and this reproduction first coexist in the meq-integration
      // tree, which carries both.
      //
      // So this section is the thing to revisit at that merge, and it will
      // then be asserting the wrong property. `max_it` of five and a
      // deliberate silence about convergence are here only because the
      // Jacobian is wrong; once it is right, this section should converge and
      // be asserted to, the way every other section in this file is. Do not
      // read a passing run as evidence either way in the meantime -- what is
      // asserted below is only that requesting Batched changes nothing.
      //
      // What this section can still assert, and does, is that requesting
      // Batched on a problem that cannot batch changes nothing at all --
      // which is the whole content of the fallback, since the code executed
      // is then identical to Serial's.
      const int max_it = 5;
      const Outcome ref = Solve(mesh_a, order, c, LFM::Serial, max_it);
      const Outcome got = Solve(mesh_b, order, c, LFM::Batched, max_it);

      CAPTURE(order);
      REQUIRE(got.can_batch == (order == 0));

      RequireSame(ref.p, got.p);
      RequireSame(ref.q, got.q);
   }
}

TEST_CASE("Essential flux dofs alone break the uniform block size",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_factor;

   // The trap this pins. A uniform mesh at a uniform order is *not* enough
   // for CanBatchLocalFactor(): Af_f_offsets sizes each element's block by
   // counting that element's free hat dofs, and a hat dof is essential when
   // it depends only on ess_flux_tdof_list. So a non-empty essential flux
   // list gives every boundary element a smaller block than every interior
   // one, on a mesh that is uniform by any other measure. Only the offsets
   // themselves can answer the question, which is what the implementation
   // asks and what this asserts.
   //
   // Read-only on the RT path: nothing here changes its discretisation, and
   // the linear problem below never reaches InvertA() at all -- what is
   // under test is the predicate.
   const int order = GENERATE(0, 1);
   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
   const int dim = mesh.Dimension();

   RT_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll);
   FiniteElementSpace Wh(&mesh, &p_coll);
   FiniteElementSpace Mh(&mesh, &t_coll);

   ConstantCoefficient one(1.0);

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   const bool essential = GENERATE(false, true);
   Array<int> ess_flux;
   if (essential) { Vh.GetEssentialTrueDofs(ess_bdr, ess_flux); }

   DarcyForm darcy(&Vh, &Wh);
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorFEMassIntegrator(one));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   darcy.Assemble();

   CAPTURE(order, essential, ess_flux.Size());
   if (essential)
   {
      REQUIRE(ess_flux.Size() > 0);
      REQUIRE_FALSE(darcy.GetHybridization()->CanBatchLocalFactor());
   }
   else
   {
      REQUIRE(ess_flux.Size() == 0);
      REQUIRE(darcy.GetHybridization()->CanBatchLocalFactor());
   }
}

TEST_CASE("The batched local solve gives the serial one's answer",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_factor;
   using LFM = DarcyHybridization::LocalFactorMode;

   const int order = GENERATE(0, 1, 2);
   const int n = GENERATE(2, 4);
   const bool tri = GENERATE(false, true);
   CAPTURE(order, n, tri);

   const auto et = tri ? Element::TRIANGLE : Element::QUADRILATERAL;
   Mesh mesh_a = Mesh::MakeCartesian2D(n, n, et);
   Mesh mesh_b = Mesh::MakeCartesian2D(n, n, et);

   const LinearOutcome ref = SolveLinear(mesh_a, order, LFM::Serial);
   const LinearOutcome got = SolveLinear(mesh_b, order, LFM::Batched);

   // The discriminating half: Serial must NOT take the batched solve and
   // Batched must. Without this the comparison below is Serial against
   // Serial, which passes while testing nothing -- the same trap the
   // factorisation case above guards against.
   REQUIRE_FALSE(ref.can_batch_solve);
   REQUIRE(got.can_batch_solve);
   REQUIRE(got.can_batch_factor);

   // ReduceRHS() feeds the trace solve, so a difference there would show up
   // in the trace before it showed up in the fields; checking the trace as
   // well as the fields separates "the reduced right-hand side moved" from
   // "the recovery moved".
   RequireSame(ref.tr, got.tr);
   RequireSame(ref.p, got.p);
   RequireSame(ref.q, got.q);
}

TEST_CASE("The batched element factorisation assembles the same trace operator",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_factor;
   using LFM = DarcyHybridization::LocalFactorMode;

   const int order = GENERATE(0, 1, 2);
   const int n = GENERATE(2, 4);
   CAPTURE(order, n);

   Mesh mesh_a = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   Mesh mesh_b = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);

   const LinearOutcome ref = SolveLinear(mesh_a, order, LFM::Serial);
   const LinearOutcome got = SolveLinear(mesh_b, order, LFM::Batched);

   // can_batch_solve, not can_batch_factor: the latter asks only whether the
   // STORAGE would allow batching and is true in both modes, so asserting it
   // false of the reference says nothing. CanBatchLocalSolve() carries the
   // `lfac_mode == Batched` half, which is the same condition
   // FactorElementsBatched() tests.
   REQUIRE_FALSE(ref.can_batch_solve);
   REQUIRE(got.can_batch_solve);

   // There is an operator to compare, and it is not the zero one.
   REQUIRE(ref.Hdata.Size() > 0);
   REQUIRE(ref.Hdata.Normlinf() > 1e-3);

   // The sparsity first: a wrong Schur complement that happened to agree
   // entrywise where both are structurally nonzero would still show here.
   REQUIRE(got.HI.Size() == ref.HI.Size());
   REQUIRE(got.HJ.Size() == ref.HJ.Size());
   for (int i = 0; i < ref.HI.Size(); i++) { REQUIRE(got.HI[i] == ref.HI[i]); }
   for (int i = 0; i < ref.HJ.Size(); i++) { REQUIRE(got.HJ[i] == ref.HJ[i]); }

   // And the entries, bitwise for the reason RequireSame() gives: the two
   // routes are the same scalar code in a build without LAPACK.
   // BatchedLinAlg::AddMult() reaches kernels::AddMult() and the element loop
   // reaches mfem::AddMult(), and those two run the identical j-k-i loop over
   // the identical products -- so the Schur complement is not merely close.
   RequireSame(ref.Hdata, got.Hdata);
}

TEST_CASE("The batched element factorisation serves a matrix-free gradient",
          "[DarcyHybridization][BatchedLinAlg][NPC]")
{
   using namespace darcy_batched_factor;
   using LFM = DarcyHybridization::LocalFactorMode;
   using GM = DarcyHybridization::GradientMode;

   const int order = GENERATE(0, 1, 2);
   const int n = GENERATE(2, 4);
   CAPTURE(order, n);

   Mesh mesh_a = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   Mesh mesh_b = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);

   // GradientMode::MatrixFree takes ComputeHMode::GradientFactorOnly, whose
   // whole body is the factorisation and the Schur complement -- there is no
   // face loop after it, which makes it the one end of this chain with
   // nothing to assemble and nothing to scatter.
   //
   // It has to be driven from the NONLINEAR side: GradientFactorOnly is
   // reached from NPCGradient()/GetGradient() alone, and a linear problem's
   // FormSystemMatrix() assembles H whatever the gradient mode says. A first
   // version of this case asked SolveLinear() for it and got an assembled
   // matrix back, which is what said so.
   NPCOutcome ref, got;
   NPCStep(mesh_a, order, 5.0, LFM::Serial, ref, GM::MatrixFree);
   NPCStep(mesh_b, order, 5.0, LFM::Batched, got, GM::MatrixFree);

   REQUIRE_FALSE(ref.can_batch_solve);
   REQUIRE(got.can_batch_solve);

   // There has to be a step to compare.
   CAPTURE(ref.n0, ref.n1);
   REQUIRE(ref.n0 > 1e-3);
   REQUIRE(ref.n1 < 0.1 * ref.n0);

   RequireSame(ref.dtr, got.dtr);
   RequireSame(ref.dx.GetBlock(0), got.dx.GetBlock(0));
   RequireSame(ref.dx.GetBlock(1), got.dx.GetBlock(1));
}

TEST_CASE("The batched local solve gives the serial one's NPC step",
          "[DarcyHybridization][BatchedLinAlg][NPC]")
{
   using namespace darcy_batched_factor;
   using LFM = DarcyHybridization::LocalFactorMode;

   const int order = GENERATE(0, 1, 2);
   const int n = GENERATE(2, 4);
   CAPTURE(order, n);

   Mesh mesh_a = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   Mesh mesh_b = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);

   NPCOutcome ref, got;
   NPCStep(mesh_a, order, 5.0, LFM::Serial, ref);
   NPCStep(mesh_b, order, 5.0, LFM::Batched, got);

   REQUIRE_FALSE(ref.can_batch_solve);
   REQUIRE(got.can_batch_solve);

   // There has to be a step to compare: a residual that starts small, or a
   // Newton step that achieves nothing, would make the comparison below
   // vacuous.
   CAPTURE(ref.n0, ref.n1);
   REQUIRE(ref.n0 > 1e-3);
   REQUIRE(ref.n1 < 0.1 * ref.n0);

   RequireSame(ref.dtr, got.dtr);
   RequireSame(ref.dx.GetBlock(0), got.dx.GetBlock(0));
   RequireSame(ref.dx.GetBlock(1), got.dx.GetBlock(1));
}
