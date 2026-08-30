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
