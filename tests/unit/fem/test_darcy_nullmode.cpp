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

using namespace mfem;

namespace darcy_nullmode
{

// A pure-Neumann Darcy problem on the unit square:
//
//    u + grad p = 0
//      - div u  = g
//
// with u.n given on the whole boundary and no zeroth-order term, so p is
// determined only up to an additive constant and the hybridized trace system
// inherits that null space.
//
// p = cos(pi x) cos(pi y) is the convenient exact solution: its normal
// derivative vanishes on all four sides, so the essential flux datum is
// homogeneous and no projection is needed; its mean over the square is zero,
// so the mean-zero normalisation picks out the exact solution itself and the
// errors below need no shifting; and the compatibility condition int g = 0
// that pure Neumann demands of the data holds for the same reason.

real_t pExact(const Vector &x)
{
   return cos(M_PI * x(0)) * cos(M_PI * x(1));
}

void uExact(const Vector &x, Vector &u)
{
   u(0) =  M_PI * sin(M_PI * x(0)) * cos(M_PI * x(1));
   u(1) =  M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
}

// g = -div u = laplace p.
real_t gExact(const Vector &x) { return -2.0 * M_PI * M_PI * pExact(x); }

/// The constant function 1, as coefficients in a face-based trace space.
/// GridFunction::ProjectCoefficient() cannot be used: a trace space has no
/// element finite elements at all, only face ones, so it dereferences null.
/// The projection is therefore taken face by face, which also keeps this
/// independent of whether the face basis happens to be nodal.
void TraceConstant(FiniteElementSpace &fes_t, GridFunction &cst)
{
   Mesh *mesh = fes_t.GetMesh();
   ConstantCoefficient one(1.0);
   MassIntegrator mass;
   DomainLFIntegrator lf(one);
   IsoparametricTransformation Tr;
   DenseMatrix M;
   Vector b, c;
   Array<int> dofs;

   cst = 0.0;
   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      const FiniteElement *fe = fes_t.GetFaceElement(f);
      mesh->GetFaceTransformation(f, &Tr);
      mass.AssembleElementMatrix(*fe, Tr, M);
      lf.AssembleRHSElementVect(*fe, Tr, b);
      c.SetSize(b.Size());
      DenseMatrixInverse(M).Mult(b, c);
      fes_t.GetFaceDofs(f, dofs);
      cst.SetSubVector(dofs, c);
   }
}

/** @brief P * inner * P, where P removes the component along a prescribed
    vector.

    MFEM's OrthoSolver is the same construction hard-wired to the vector of
    ones. That is the right vector only when every trace dof carries the
    constant, and here it is not: Raviart-Thomas registers no boundary trace
    constraint, so the boundary trace dofs sit outside the physical system with
    a unit diagonal from ComputeH's DIAG_ONE policy, and the null vector is the
    constant on the interior faces alone. */
class ModeProjectedSolver : public Solver
{
   Solver *inner = nullptr;
   Vector nv;             ///< the null direction, normalised

   void Project(const Vector &b, Vector &b_ortho) const
   {
      b_ortho.SetSize(b.Size());
      b_ortho = b;
      b_ortho.Add(-(nv * b), nv);
   }

public:
   ModeProjectedSolver(const Vector &mode) : Solver(mode.Size()), nv(mode)
   { nv /= nv.Norml2(); }

   void SetSolver(Solver &s) { inner = &s; width = height = s.Width(); }
   void SetOperator(const Operator &op) override { inner->SetOperator(op); }

   void Mult(const Vector &b, Vector &x) const override
   {
      Vector b_ortho;
      Project(b, b_ortho);
      inner->Mult(b_ortho, x);
      Vector x_ortho;
      Project(x, x_ortho);
      x = x_ortho;
   }
};

/// How the constant is pinned.
enum class Pin
{
   None,      ///< not pinned at all; H is singular and the RHS compatible
   Mass,      ///< a small zeroth-order term on the potential, eps * (p, v)
   MeanZero   ///< the null direction projected out of the Krylov space
};

struct NullResult
{
   int    iters;
   bool   converged;
   real_t err_p, err_u;   ///< L2 errors, with the mean removed from p
   real_t resid;          ///< final residual norm reported by the solver
   real_t null_defect;    ///< |H n| / |n| for n the constant on the free faces
   real_t compat;         ///< |n.b| / (|n| |b|): how compatible the data is
   int    size;
};

NullResult SolveNeumann(int order, int n, Pin pin, real_t eps = 1e-6,
                        bool precond = false, real_t incompat = 0.0)
{
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false, 1., 1.);
   const int dim = 2;

   RT_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   ConstantCoefficient one(1.0), epsc(eps);
   FunctionCoefficient gcoeff(gExact), pcoeff(pExact);
   VectorFunctionCoefficient ucoeff(dim, uExact);

   DarcyForm darcy(&fes_u, &fes_p);
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorFEMassIntegrator(one));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   if (pin == Pin::Mass)
   {
      darcy.GetPotentialMassForm()->AddDomainIntegrator(new MassIntegrator(epsc));
   }
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   ConstantCoefficient bad(incompat);
   if (incompat != 0.0)
   {
      // int g must vanish for a pure-Neumann problem to have a solution at
      // all. Adding a constant to g breaks that, and the trace system becomes
      // singular *and inconsistent* rather than singular and consistent.
      darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(bad));
   }

   // Pure Neumann: every boundary flux dof is essential, and du/dn = 0 makes
   // the datum homogeneous, so x starts at zero and stays right there.
   Array<int> bdr_all(mesh.bdr_attributes.Max());
   bdr_all = 1;
   Array<int> ess_flux_tdofs;
   fes_u.GetEssentialTrueDofs(bdr_all, ess_flux_tdofs);

   DG_Interface_FECollection trace_coll(order, dim);
   FiniteElementSpace fes_t(&mesh, &trace_coll);
   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(),
                             ess_flux_tdofs);
   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   OperatorPtr A;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux_tdofs, x, A, X, B, true);

   NullResult res;
   res.size = X.Size();

   // The null direction: the constant, on the faces that carry a constraint.
   // RT registers no boundary trace constraint, so the boundary trace dofs sit
   // outside the physical system with a unit diagonal from ComputeH's DIAG_ONE
   // policy and must be left out of it.
   GridFunction cst(&fes_t);
   TraceConstant(fes_t, cst);
   Array<int> bdr_dofs;
   fes_t.GetEssentialTrueDofs(bdr_all, bdr_dofs);
   for (int i = 0; i < bdr_dofs.Size(); i++) { cst(bdr_dofs[i]) = 0.0; }

   {
      Vector y(X.Size());
      A.As<SparseMatrix>()->Mult(cst, y);
      res.null_defect = y.Norml2() / cst.Norml2();
      res.compat = std::abs(cst * B) / (cst.Norml2() * B.Norml2());
   }

   GMRESSolver gmres;
   gmres.SetKDim(1000);
   gmres.SetMaxIter(5000);
   gmres.SetRelTol(1e-12);
   gmres.SetAbsTol(0.0);
   gmres.SetPrintLevel(-1);
   std::unique_ptr<GSSmoother> prec;
   if (precond)
   {
      prec.reset(new GSSmoother(*A.As<SparseMatrix>()));
      gmres.SetPreconditioner(*prec);
   }
   gmres.SetOperator(*A);

   if (pin == Pin::MeanZero)
   {
      ModeProjectedSolver proj(cst);
      proj.SetSolver(gmres);
      proj.Mult(B, X);
   }
   else
   {
      gmres.Mult(B, X);
   }
   res.iters = gmres.GetNumIterations();
   res.converged = gmres.GetConverged();
   res.resid = gmres.GetFinalNorm();

   darcy.RecoverFEMSolution(X, x);

   GridFunction u_h(&fes_u, x.GetBlock(0));
   GridFunction p_h(&fes_p, x.GetBlock(1));

   // p is only determined up to a constant, so the error is only meaningful
   // once the mean is removed. The exact solution has mean zero already.
   LinearForm vol(&fes_p);
   vol.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol.Assemble();
   GridFunction ones(&fes_p);
   ones = 1.0;
   const real_t mean = (vol * p_h) / (vol * ones);
   p_h -= mean;

   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      irs[i] = &(IntRules.Get(i, 2 * order + 3));
   }
   res.err_p = p_h.ComputeL2Error(pcoeff, irs);
   res.err_u = u_h.ComputeL2Error(ucoeff, irs);
   return res;
}

} // namespace darcy_nullmode





TEST_CASE("A compatible pure-Neumann problem does not need its constant pinned",
          "[DarcyForm][DarcyHybridization]")
{
   using namespace darcy_nullmode;

   // The trace system really is singular, and the constant really is its null
   // vector -- so this is not a case that avoids the question by accident.
   // What it establishes is that the question does not have to be answered:
   // the data of a pure-Neumann problem is compatible by construction, the
   // singular system is therefore consistent, and GMRES converges on it and
   // returns a solution correct up to the constant.
   const int order = GENERATE(1, 2);
   CAPTURE(order);

   real_t prev = -1.0;
   for (int n : {8, 16, 32})
   {
      const NullResult none = SolveNeumann(order, n, Pin::None);
      CAPTURE(n, none.size, none.iters, none.null_defect, none.compat,
              none.err_p);

      REQUIRE(none.null_defect < 1e-12);  // singular, with that null vector
      REQUIRE(none.compat < 1e-12);       // and consistent
      REQUIRE(none.converged);

      if (prev > 0.0)
      {
         REQUIRE(std::log2(prev / none.err_p) > order + 0.7);
      }
      prev = none.err_p;

      // Neither pinning changes the answer, and neither changes the cost.
      // The mode contributes nothing to the conditioning of this system. Taken
      // below the finest mesh only: unpreconditioned GMRES needs about a
      // thousand iterations there, and three more solves of it would cost the
      // suite more than the extra resolution is worth. The finest mesh is
      // carried by the rate check above, and by the eps sweep in the next
      // case, which does run at n = 32.
      if (n < 32)
      {
         const NullResult mass = SolveNeumann(order, n, Pin::Mass, 1e-8);
         const NullResult mz   = SolveNeumann(order, n, Pin::MeanZero);
         CAPTURE(mass.iters, mass.err_p, mz.iters, mz.err_p);
         REQUIRE(mass.err_p == MFEM_Approx(none.err_p, 1e-10, 1e-4));
         REQUIRE(mz.err_p   == MFEM_Approx(none.err_p, 1e-10, 1e-4));
         REQUIRE(mass.iters <= 1.05 * none.iters + 5);
         REQUIRE(mz.iters   <= 1.05 * none.iters + 5);
      }
   }
}

TEST_CASE("Pinning the constant with a mass costs accuracy, not iterations",
          "[DarcyForm][DarcyHybridization]")
{
   using namespace darcy_nullmode;

   // What the small mass does is perturb the operator by eps, so the error
   // floors at O(eps) and the rate collapses once that floor rises above the
   // discretisation error. It does not cost iterations at any eps measured
   // here, which is the opposite of what the roadmap recorded before this
   // case was written -- the iteration count it attributed to the pinning
   // belongs to incompatible data, and the case below is where it lives.
   //
   // The trap is that eps has to sit below the smallest error the study
   // intends to reach, and that threshold moves with the mesh: eps = 1e-3 is
   // harmless at n = 8 and costs an order of magnitude by n = 32.
   const int order = 2;

   const NullResult ref8  = SolveNeumann(order, 8,  Pin::None);
   const NullResult ref32 = SolveNeumann(order, 32, Pin::None);

   const NullResult f8  = SolveNeumann(order, 8,  Pin::Mass, 1e-3);
   const NullResult f32 = SolveNeumann(order, 32, Pin::Mass, 1e-3);
   CAPTURE(ref8.err_p, f8.err_p, ref32.err_p, f32.err_p);

   // Harmless where the discretisation error is still large ...
   REQUIRE(f8.err_p < 1.1 * ref8.err_p);
   // ... and not where it is not.
   REQUIRE(f32.err_p > 5.0 * ref32.err_p);

   // The floor is eps, not the mesh: a coarser grid with the same eps lands in
   // the same place, which is what a collapsed rate looks like.
   const NullResult c16 = SolveNeumann(order, 16, Pin::Mass, 1e-1);
   const NullResult c32 = SolveNeumann(order, 32, Pin::Mass, 1e-1);
   CAPTURE(c16.err_p, c32.err_p);
   REQUIRE(std::log2(c16.err_p / c32.err_p) < 0.2);

   // And none of it shows up in the iteration count.
   CAPTURE(ref32.iters, f32.iters, c32.iters);
   REQUIRE(f32.iters < 1.3 * ref32.iters);
   REQUIRE(c32.iters < 1.3 * ref32.iters);
}

TEST_CASE("Incompatible data, not the null mode, is what breaks the solve",
          "[DarcyForm][DarcyHybridization]")
{
   using namespace darcy_nullmode;

   // int g must vanish for a pure-Neumann problem to have a solution. Break
   // that and the trace system goes from singular-and-consistent to
   // singular-and-inconsistent, and the difference is the whole of the
   // pathology: the iteration count explodes and then stops converging
   // altogether. An unexplained blow-up on a problem of this shape is a
   // compatibility bug in the data, and this case is the evidence for reading
   // it that way rather than as a property of the mode.
   const int order = 2;
   const int n = 16;

   const NullResult ok  = SolveNeumann(order, n, Pin::None, 1e-6, false, 0.0);
   const NullResult bad = SolveNeumann(order, n, Pin::None, 1e-6, false, 1.0);
   CAPTURE(ok.iters, ok.compat, bad.iters, bad.compat, bad.converged);

   REQUIRE(ok.compat < 1e-12);
   REQUIRE(bad.compat > 1e-3);
   REQUIRE(ok.converged);
   REQUIRE(bad.iters > 10 * ok.iters);

   // The small mass is what rescues it, and it rescues the answer too: the
   // incompatible part of the data goes into eps * mean(p) and the rest comes
   // out unchanged. That is why pinning with a mass is reported to work.
   const NullResult bad_m = SolveNeumann(order, n, Pin::Mass, 1e-6, false, 1.0);
   CAPTURE(bad_m.iters, bad_m.err_p);
   REQUIRE(bad_m.converged);
   REQUIRE(bad_m.err_p == MFEM_Approx(ok.err_p, 1e-10, 1e-4));

   // Projecting the mode out of the Krylov space is much the cheaper repair
   // and much the worse one: it converges in a small multiple of the
   // compatible cost and silently discards the incompatible component, which
   // is real information about the data being wrong.
   const NullResult bad_z = SolveNeumann(order, n, Pin::MeanZero, 1e-6, false,
                                         1.0);
   CAPTURE(bad_z.iters, bad_z.err_p);
   REQUIRE(bad_z.converged);
   REQUIRE(bad_z.iters < 3 * ok.iters);
   REQUIRE(bad_z.err_p > 10 * ok.err_p);
}
