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

namespace darcy_reconstruction
{

// What DarcyForm::Reconstruct() does to the local problem it builds, and the
// two ways it used to build the wrong one.
//
// The post-processing solves an element-local problem on spaces one order
// higher, driven by the reconstructed total flux and by the element average
// of the computed potential. Its blocks are lifted from the forms the user
// filled in, and the lift is where both defects live: it read the potential
// mass only off the *linear* form, and it read either mass form's *domain*
// integrators only.
//
// Neither failure aborts. The first returns a potential wrong by orders of
// magnitude -- there is no potential block and no face constraint, so the
// local matrix is singular and gets solved anyway. The second returns a
// potential that is merely missing a term, which on the case it was found on
// still converged at the design order. Both are invisible to a caller, which
// is why they are checked here against a twin that has to agree.

/// The manufactured solution: p vanishes on the boundary of the unit cube,
/// so the natural flux datum is zero and the two formulations below differ
/// only in which form carries the potential mass.
real_t pExact(const Vector &x)
{
   real_t p = 1.0;
   for (int i = 0; i < x.Size(); i++) { p *= std::sin(M_PI * x(i)); }
   return p;
}

/// u = -grad p, the flux of the unit-conductivity problem.
void uExact(const Vector &x, Vector &u)
{
   const int dim = x.Size();
   for (int i = 0; i < dim; i++)
   {
      real_t v = -M_PI * std::cos(M_PI * x(i));
      for (int j = 0; j < dim; j++)
      {
         if (j != i) { v *= std::sin(M_PI * x(j)); }
      }
      u(i) = v;
   }
}

/// -div u = g with u = -grad p, so g = -dim pi^2 p.
real_t gExact(const Vector &x)
{
   return -x.Size() * M_PI * M_PI * pExact(x);
}

/// Everything the reconstruction produces, so that two paths can be compared
/// dof by dof rather than through an error norm that could agree by accident.
struct Post
{
   Vector ut, u_s, p_s, tr_s;
   real_t err_p, err_ps;
};

/// Where the potential mass is put. The two are the same operator; only the
/// form carrying it differs, and with it the code path the lift takes.
enum class PotForm { Linear, Nonlinear };

/// A boundary-face term on the flux mass, of the shape an extension method
/// installs: element-local by construction, which is why it goes on the mass
/// rather than on the constraint. Here it is a plain boundary mass, scaled,
/// so that its effect on the local problem is unambiguous and its size is
/// under the test's control.
class BdrFluxMass : public BilinearFormIntegrator
{
   real_t c;
   Vector shape;
public:
   BdrFluxMass(real_t c_) : c(c_) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override
   {
      MFEM_VERIFY(Trans.Elem2No < 0, "a boundary face only");

      const int dof = el1.GetDof();
      const int dim = Trans.GetSpaceDim();
      elmat.SetSize(dof * dim);
      elmat = 0.0;
      shape.SetSize(dof);

      const IntegrationRule &ir =
         IntRules.Get(Trans.GetGeometryType(), 2 * el1.GetOrder() + 2);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Trans.SetAllIntPoints(&ip);
         el1.CalcShape(Trans.GetElement1IntPoint(), shape);

         const real_t w = c * ip.weight * Trans.Weight();
         for (int d = 0; d < dim; d++)
            for (int i = 0; i < dof; i++)
               for (int j = 0; j < dof; j++)
               {
                  elmat(d * dof + i, d * dof + j) += w * shape(i) * shape(j);
               }
      }
   }
};

/// Solve the hybridized HDG problem and post-process it. @a bdr_flux_mass, if
/// non-zero, installs the boundary-face term above on the flux mass.
Post Solve(Mesh &mesh, int order, PotForm pot, real_t td,
           real_t bdr_flux_mass = 0.0)
{
   const int dim = mesh.Dimension();

   L2_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll, dim);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   ConstantCoefficient one(1.0);
   FunctionCoefficient gcoeff(gExact), pcoeff(pExact);
   VectorFunctionCoefficient ucoeff(dim, uExact);

   DarcyForm darcy(&fes_u, &fes_p);

   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   if (bdr_flux_mass != 0.0)
   {
      darcy.GetFluxMassForm()->AddBdrFaceIntegrator(
         new BdrFluxMass(bdr_flux_mass));
   }

   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   if (pot == PotForm::Linear)
   {
      darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
         new HDGDiffusionIntegrator(one, td));
   }
   else
   {
      darcy.GetPotentialMassNonlinearForm()->AddInteriorFaceIntegrator(
         new HDGDiffusionIntegrator(one, td));
   }

   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new DomainLFIntegrator(gcoeff, 6, 12));

   Array<int> ess;
   DG_Interface_FECollection trace_coll(order, dim);
   FiniteElementSpace fes_t(&mesh, &trace_coll);
   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(), ess);
   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, RHS;
   darcy.FormLinearSystem(ess, x, A, X, RHS, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(5000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(1e-16);
   lin.SetPreconditioner(prec);

   if (pot == PotForm::Linear)
   {
      lin.SetOperator(*A);
      lin.Mult(RHS, X);
      REQUIRE(lin.GetConverged());
   }
   else
   {
      // The operator is nonlinear only in the sense that it goes through the
      // nonlinear machinery: every integrator on it is bilinear, so Newton
      // lands on the same answer the linear path does, and that is exactly
      // what makes the two comparable below.
      darcy.GetHybridization()->SetLocalNLSolver(
         DarcyHybridization::LSsolveType::Newton, 100, 1e-14, 1e-16, -1);

      NewtonSolver newton;
      newton.SetSolver(lin);
      newton.SetOperator(*A);
      newton.SetRelTol(1e-13);
      newton.SetAbsTol(1e-15);
      newton.SetMaxIter(20);
      newton.SetPrintLevel(-1);
      newton.Mult(RHS, X);
      REQUIRE(newton.GetConverged());
   }

   darcy.RecoverFEMSolution(X, x);

   Post res;
   GridFunction ut, u_s, p_s, tr_s;
   darcy.Reconstruct(x, X, ut, u_s, p_s, tr_s);

   GridFunction p_h(&fes_p, x.GetBlock(1));

   const int quad_order = 2 * order + 6;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      irs[i] = &(IntRules.Get(i, quad_order));
   }

   res.err_p  = p_h.ComputeL2Error(pcoeff, irs);
   res.err_ps = p_s.ComputeL2Error(pcoeff, irs);
   res.ut = ut;
   res.u_s = u_s;
   res.p_s = p_s;
   res.tr_s = tr_s;
   return res;
}

real_t MaxDiff(const Vector &a, const Vector &b)
{
   REQUIRE(a.Size() == b.Size());
   real_t d = 0.0;
   for (int i = 0; i < a.Size(); i++)
   {
      d = std::max(d, std::abs(a(i) - b(i)));
   }
   return d;
}


/** A non-linear potential term whose Jacobian vanishes identically.

    F(u) = 0: it adds nothing to the residual and nothing to the Jacobian, so
    it cannot change the discrete problem, and a reconstruction that changes
    when it is installed is reading something it should not. It is the sharpest
    form of the case reported from outside, where dF/du vanished on part of the
    domain -- any tabulated profile with a flat segment produces one. */
class NullPotentialNL : public NonlinearFormIntegrator
{
public:
   void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override
   {
      elvect.SetSize(el.GetDof());
      elvect = 0.0;
   }

   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override
   {
      elmat.SetSize(el.GetDof());
      elmat = 0.0;
   }
};

/// Reconstruct with and without @a null_term installed on the potential mass.
Post SolveWithNullTerm(Mesh &mesh, int order, bool null_term)
{
   const int dim = mesh.Dimension();

   L2_FECollection u_coll(order, dim), p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll, dim), fes_p(&mesh, &p_coll);

   ConstantCoefficient one(1.0);
   FunctionCoefficient gcoeff(gExact), pcoeff(pExact);
   VectorFunctionCoefficient fcoeff(dim, [](const Vector &, Vector &f)
   {
      f = 0.0;
   });
   RatioCoefficient ik(1.0, one);

   DarcyForm darcy(&fes_u, &fes_p);
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   // The constraint is on the non-linear form either way, so the only
   // difference between the two runs is the domain term below.
   darcy.GetPotentialMassNonlinearForm()->AddInteriorFaceIntegrator(
      new HDGDiffusionIntegrator(one, 0.5));
   if (null_term)
   {
      darcy.GetPotentialMassNonlinearForm()->AddDomainIntegrator(
         new NullPotentialNL());
   }
   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new DomainLFIntegrator(gcoeff, 6, 12));

   Array<int> ess;
   DG_Interface_FECollection trace_coll(order, dim);
   FiniteElementSpace fes_t(&mesh, &trace_coll);
   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(), ess);
   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   OperatorPtr A;
   Vector X, RHS;
   darcy.FormLinearSystem(ess, x, A, X, RHS, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(5000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(1e-16);
   lin.SetPreconditioner(prec);
   darcy.GetHybridization()->SetLocalNLSolver(
      DarcyHybridization::LSsolveType::Newton, 100, 1e-14, 1e-16, -1);

   NewtonSolver newton;
   newton.SetSolver(lin);
   newton.SetOperator(*A);
   newton.SetRelTol(1e-12);
   newton.SetAbsTol(1e-15);
   newton.SetMaxIter(40);
   newton.SetPrintLevel(-1);
   newton.Mult(RHS, X);
   REQUIRE(newton.GetConverged());

   darcy.RecoverFEMSolution(X, x);

   Post res;
   GridFunction ut, u_s, p_s, tr_s;
   darcy.Reconstruct(x, X, ut, u_s, p_s, tr_s);

   GridFunction p_h(&fes_p, x.GetBlock(1));
   const int quad_order = 2 * order + 6;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   { irs[i] = &(IntRules.Get(i, quad_order)); }

   res.err_p  = p_h.ComputeL2Error(pcoeff, irs);
   res.err_ps = p_s.ComputeL2Error(pcoeff, irs);
   res.ut = ut;
   res.u_s = u_s;
   res.p_s = p_s;
   res.tr_s = tr_s;
   return res;
}

} // namespace darcy_reconstruction

TEST_CASE("Reconstruction reads the potential mass off either form",
          "[DarcyForm][Reconstruction]")
{
   using namespace darcy_reconstruction;

   // The defect: the lift had an `if (M_p)` and no `else if (Mnl_p)`, so a
   // potential block living entirely on the nonlinear form gave an enriched
   // form with no integrators at all. The local problem then had neither a
   // potential mass nor -- because the hybridization keeps a nonlinear
   // constraint in a different member -- any face constraint, and the
   // singular system was factored and solved without a word. Measured on an
   // application it returned 1e15 where the linear twin returned 1e-6; here
   // it returns NaN. The two solves agree in p_h to solver tolerance, so
   // anything the post-processing produces has to agree too.
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1, 2);
   CAPTURE(dim, order);

   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, false, 1., 1.)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1., 1., 1.);

   const real_t td = 0.5;
   const Post lin = Solve(mesh, order, PotForm::Linear, td);
   const Post nlp = Solve(mesh, order, PotForm::Nonlinear, td);

   // First the premise: the two solves really do produce the same p_h. If
   // they did not, the comparison below would be measuring the solvers.
   INFO("||p_h - p||: linear " << lin.err_p << ", nonlinear " << nlp.err_p);
   REQUIRE(nlp.err_p == MFEM_Approx(lin.err_p, 1e-9, 1e-11));

   // Then the reconstruction itself, dof by dof.
   const real_t scale = std::max(lin.p_s.Normlinf(), real_t(1.0));
   INFO("max |p_s difference| = " << MaxDiff(lin.p_s, nlp.p_s)
        << " against |p_s| = " << lin.p_s.Normlinf());
   REQUIRE(std::isfinite(nlp.p_s.Normlinf()));
   REQUIRE(MaxDiff(lin.ut,   nlp.ut)   < 1e-8 * std::max(lin.ut.Normlinf(),
                                                         real_t(1.0)));
   REQUIRE(MaxDiff(lin.u_s,  nlp.u_s)  < 1e-8 * std::max(lin.u_s.Normlinf(),
                                                         real_t(1.0)));
   REQUIRE(MaxDiff(lin.p_s,  nlp.p_s)  < 1e-8 * scale);
   REQUIRE(MaxDiff(lin.tr_s, nlp.tr_s) < 1e-8 * std::max(lin.tr_s.Normlinf(),
                                                         real_t(1.0)));

   // And the reason anyone calls it: the post-processed potential is the more
   // accurate one. A reconstruction that came back as garbage would pass the
   // agreement test above if both paths were broken the same way.
   INFO("||p_s - p|| = " << nlp.err_ps << " against ||p_h - p|| = " << nlp.err_p);
   REQUIRE(nlp.err_ps < nlp.err_p);
}

TEST_CASE("Reconstruction does not lift a boundary-face term on the flux mass",
          "[DarcyForm][Reconstruction]")
{
   using namespace darcy_reconstruction;

   // Reported from outside as a defect -- the lift copies the flux mass's
   // domain integrators and nothing else, so a term installed with
   // AddBdrFaceIntegrator, which is how an extension method installs the
   // element-local half of a transferred Dirichlet datum, never reaches the
   // local problem. Carrying it was implemented, and then withdrawn, because
   // it is measurably wrong.
   //
   // The local problem is not the assembled problem restricted to an element.
   // Its trace unknown is free on every face, boundary faces included, and the
   // boundary condition reaches it through the reconstructed total flux and
   // the element average rather than through the forms. A boundary-face term
   // on the flux mass is one half of a boundary condition -- the half that
   // depends on the flux -- and putting it in without the datum half, which
   // the local problem has no way to know and could not use without
   // double-counting against its own free trace, imposes half a condition.
   // On miniapps/hdg/extension at k = 2, problem 1, over the 8x8 to 64x64
   // sequence:
   //
   //            ||p - p*|| at n = 64      rate      ||u - u*||      rate
   //   dropped        1.58e-9             3.80        3.22e-8       3.63
   //   lifted         8.57e-5             1.27        2.43e-4       1.25
   //
   // -- k+2 for the potential either kept or lost, and a factor of 5e4 in the
   // error. So the drop is required, not tolerated, and this case says so: the
   // same solve, reconstructed with and without such a term installed, must
   // give the same answer to the last bit.
   //
   // The term is added after the solve, so nothing about the assembled
   // problem differs between the two calls -- not the hybridization, not the
   // total flux, not the element averages. Whether the *assembly* carries such
   // a term is a separate question with a separate answer: only a tree with
   // AssembleFluxMassBdrFaces(), which the extension work adds, puts it into
   // the solve. Holding the solve fixed makes this case say the same thing
   // either way.
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1);
   CAPTURE(dim, order);

   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, false, 1., 1.)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1., 1., 1.);

   L2_FECollection u_coll(order, dim), p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll, dim), fes_p(&mesh, &p_coll);

   ConstantCoefficient one(1.0);
   FunctionCoefficient gcoeff(gExact);

   DarcyForm darcy(&fes_u, &fes_p);
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));

   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
      new HDGDiffusionIntegrator(one, 0.5));
   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new DomainLFIntegrator(gcoeff, 6, 12));

   Array<int> ess;
   DG_Interface_FECollection trace_coll(order, dim);
   FiniteElementSpace fes_t(&mesh, &trace_coll);
   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(), ess);
   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, RHS;
   darcy.FormLinearSystem(ess, x, A, X, RHS, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(5000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(1e-16);
   lin.SetPreconditioner(prec);
   lin.SetOperator(*A);
   lin.Mult(RHS, X);
   REQUIRE(lin.GetConverged());
   darcy.RecoverFEMSolution(X, x);

   GridFunction ut0, u0, p0, tr0;
   darcy.Reconstruct(x, X, ut0, u0, p0, tr0);

   // Now the term, and the same reconstruction again. The output functions
   // are fresh, which is what makes the enriched form be rebuilt rather than
   // reused -- the lift is cached against the spaces it was built for -- so a
   // lift that took boundary faces would show here.
   darcy.GetFluxMassForm()->AddBdrFaceIntegrator(new BdrFluxMass(3.0));

   GridFunction ut1, u1, p1, tr1;
   darcy.Reconstruct(x, X, ut1, u1, p1, tr1);

   INFO("adding a boundary-face term to the flux mass moved u_s by "
        << MaxDiff(u0, u1) << " and p_s by " << MaxDiff(p0, p1));
   REQUIRE(MaxDiff(ut0, ut1) == 0.0);
   REQUIRE(MaxDiff(u0, u1) == 0.0);
   REQUIRE(MaxDiff(p0, p1) == 0.0);
   REQUIRE(MaxDiff(tr0, tr1) == 0.0);
   REQUIRE(u0.Normlinf() > 0.0);
}

TEST_CASE("Reconstruction leaves the hybridization as it found it",
          "[DarcyForm][Reconstruction]")
{
   using namespace darcy_reconstruction;

   // DarcyHybridization::ReconstructTotalFlux() walked the faces with one
   // DenseMatrix for the constraint block. On an interior face
   // GetCtFaceMatrix() *resets* that matrix onto the stored Ct_data; on a
   // boundary face the constraint integrator assembled into it -- and a
   // DenseMatrix that already has the right shape keeps the pointer it was
   // reset to, so the assembly landed in Ct_data. On a uniform mesh every
   // face has that shape, so every boundary face overwrote the stored block
   // of whichever interior face came before it.
   //
   // The answer the call returns is right -- the corruption is behind it, and
   // the miniapps' numbers do not move. What it damages is the object it was
   // called on. So it is invisible to a driver that solves once, reconstructs
   // once and stops, which is every driver in the tree, and wrong for
   // anything that reconstructs inside a loop: a time step, a Newton
   // iteration, an adaptive pass that estimates and then solves again.
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1);
   CAPTURE(dim, order);

   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, false, 1., 1.)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1., 1., 1.);

   L2_FECollection u_coll(order, dim), p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll, dim), fes_p(&mesh, &p_coll);

   ConstantCoefficient one(1.0);
   FunctionCoefficient gcoeff(gExact);

   DarcyForm darcy(&fes_u, &fes_p);
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
   darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
      new HDGDiffusionIntegrator(one, 0.5));
   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new DomainLFIntegrator(gcoeff, 6, 12));

   Array<int> ess;
   DG_Interface_FECollection trace_coll(order, dim);
   FiniteElementSpace fes_t(&mesh, &trace_coll);
   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(), ess);
   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, RHS;
   darcy.FormLinearSystem(ess, x, A, X, RHS, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(5000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(1e-16);
   lin.SetPreconditioner(prec);
   lin.SetOperator(*A);
   lin.Mult(RHS, X);
   REQUIRE(lin.GetConverged());
   darcy.RecoverFEMSolution(X, x);

   // Recovering the flux and potential from the trace is what reads the
   // stored constraint blocks after the system has been formed -- the reduced
   // operator itself is a SparseMatrix by then and would not notice. So this
   // is the sharp statement: the same trace must give the same solution.
   BlockVector before(darcy.GetOffsets());
   before = x;

   GridFunction ut0, u0, p0, tr0;
   darcy.Reconstruct(x, X, ut0, u0, p0, tr0);

   BlockVector after(darcy.GetOffsets());
   after = 0.0;
   darcy.RecoverFEMSolution(X, after);

   INFO("recovering the solution again moved it by " << MaxDiff(before, after)
        << " on a solution of size " << before.Normlinf());
   REQUIRE(MaxDiff(before, after) == 0.0);

   // And the consequence a caller sees: reconstructing twice must give the
   // same answer. It differed by half the field.
   GridFunction ut1, u1, p1, tr1;
   darcy.Reconstruct(x, X, ut1, u1, p1, tr1);

   INFO("second reconstruction moved ut by " << MaxDiff(ut0, ut1)
        << " and u_s by " << MaxDiff(u0, u1));
   REQUIRE(MaxDiff(ut0, ut1) == 0.0);
   REQUIRE(MaxDiff(u0, u1) == 0.0);
   REQUIRE(MaxDiff(p0, p1) == 0.0);
   REQUIRE(MaxDiff(tr0, tr1) == 0.0);
}

TEST_CASE("The scalar total flux law agrees with the system one",
          "[DarcyHybridization]")
{
   using namespace darcy_reconstruction;

   // ReconstructTotalFlux() takes the flux law either as total_flux_fun, whose
   // potential is a real_t, or as total_flux_sys_fun, whose potential is a
   // Vector with one entry per field. The scalar form is the signature this
   // branch inherited; the system form is what the neq > 1 reconstruction
   // needs. Keeping both means a caller written against the old one goes on
   // compiling, and the scalar overload is an adapter, so the thing to pin is
   // that the adapter changes no answer.
   //
   // The two are also required to be unambiguous as overloads, which holds
   // because Vector(int) is explicit: a lambda taking a real_t is not callable
   // with a const Vector &, and vice versa. That is a compile-time property
   // and this file exercises it by having both call sites below.
   const int order = GENERATE(1, 2);
   const int nx = 3;
   CAPTURE(order);

   Mesh mesh = Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int dim = mesh.Dimension();

   L2_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll, dim);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   ConstantCoefficient one(1.0);
   FunctionCoefficient gcoeff(gExact);

   DarcyForm darcy(&fes_u, &fes_p);
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
   darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
      new HDGDiffusionIntegrator(one, 0.5));
   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new DomainLFIntegrator(gcoeff, 6, 12));

   Array<int> ess;
   DG_Interface_FECollection trace_coll(order, dim);
   FiniteElementSpace fes_t(&mesh, &trace_coll);
   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(), ess);
   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   OperatorPtr A;
   Vector X, RHS;
   darcy.FormLinearSystem(ess, x, A, X, RHS, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(5000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(1e-16);
   lin.SetPreconditioner(prec);
   lin.SetOperator(*A);
   lin.Mult(RHS, X);
   REQUIRE(lin.GetConverged());
   darcy.RecoverFEMSolution(X, x);

   // A law with a term the potential actually reaches, so that the two forms
   // are distinguishable: a constant velocity convects the potential.
   Vector vel(dim);
   vel(0) = 1.7;
   vel(1) = -0.9;

   RT_FECollection ut_coll(order, dim);
   FiniteElementSpace fes_ut(&mesh, &ut_coll);

   GridFunction ut_scalar(&fes_ut), ut_system(&fes_ut);
   ut_scalar = 0.0;
   ut_system = 0.0;

   DarcyHybridization *h = darcy.GetHybridization();

   h->ReconstructTotalFlux(x, X,
                           [&vel](ElementTransformation &, const Vector &u,
                                  real_t p, Vector &t)
   {
      t = u;
      t.Add(p, vel);
   }, ut_scalar);

   h->ReconstructTotalFlux(x, X,
                           [&vel](ElementTransformation &, const Vector &u,
                                  const Vector &p, Vector &t)
   {
      t = u;
      t.Add(p(0), vel);
   }, ut_system);

   // Not vacuous: the reconstruction has to have produced something.
   REQUIRE(ut_system.Normlinf() > 1e-6);

   Vector d(ut_scalar);
   d -= ut_system;
   INFO("max difference " << d.Normlinf() << " on " << ut_system.Normlinf());
   REQUIRE(d.Normlinf() < 1e-14 * std::max(ut_system.Normlinf(), real_t(1.0)));
}

TEST_CASE("A potential term that contributes nothing changes nothing",
          "[DarcyForm][Reconstruction]")
{
   // The local problem the reconstruction solves is a pure Neumann one: the
   // total flux driving it is normally continuous, so the potential is
   // determined only up to a constant and the element average closes it --
   // NPC eq (25). That closure is unconditional, and this is what says so.
   //
   // It used to be skipped whenever a non-convective non-linear potential
   // integrator was merely *present*, on the reasoning that such a term is a
   // non-singular source. A term whose Jacobian vanishes is not, and the
   // matrix was then factored singular in silence -- DenseMatrixInverse
   // ::Factor() returns void and its tolerance is an exact zero, so nothing
   // complained. Measured before the fix, at order 2 on 4x4 through 32x32:
   // the postprocessed potential went from 2.19e-04 to 1.07e+00, and by the
   // finest mesh to 4.5e+12, while the traces went with it.
   using namespace darcy_reconstruction;

   const int order = GENERATE(1, 2);
   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   CAPTURE(order);

   const Post without = SolveWithNullTerm(mesh, order, false);
   const Post with    = SolveWithNullTerm(mesh, order, true);

   // The forward solve first: if this moved, the term was not inert and the
   // rest of the test would be measuring the wrong thing.
   REQUIRE(with.err_p == Approx(without.err_p).epsilon(1e-12));

   // Every field the local solve produces, not just the potential: they come
   // out of one factorisation, and the traces move with the potential.
   REQUIRE(MaxDiff(with.p_s,  without.p_s)  < 1e-10);
   REQUIRE(MaxDiff(with.tr_s, without.tr_s) < 1e-10);
   REQUIRE(MaxDiff(with.u_s,  without.u_s)  < 1e-10);
   REQUIRE(MaxDiff(with.ut,   without.ut)   < 1e-10);

   // And the postprocessing is still doing its job in both.
   REQUIRE(with.err_ps < 0.5 * with.err_p);
}

// ---------------------------------------------------------------------------
// The rich reconstruction for a system.
//
// The classic postprocessing (HDGPotentialPostprocessor) has always been
// general in vdim, because its local problems do not couple and it can be run
// one field at a time. The rich reconstruction cannot: it solves one mixed
// local problem for the enriched flux, potential AND traces together, so the
// fields share an element matrix and the generalisation is about where each
// field's rows and columns are and about how many closure rows the problem
// needs. Two things carry it, and both are measured below.
//
//  * **The closure count is neq, one per field, each in its own block.** The
//    local problem is driven entirely by the total flux, which lives in an
//    H(div) space with vdim == neq, so field e's own element balance holds
//    identically for each e and each field's potential is fixed only up to a
//    constant. The null space is therefore neq dimensional and neq equations
//    may be dropped -- no fewer, or the matrix is singular, and no more, or
//    information is lost. "Each field's own average in each field's own block"
//    is what the last case here pins.
//  * **Nothing needs to know the space's Ordering.** GetElementVDofs() and
//    GetFaceVDofs() lay an element's or a face's vdofs out field-outermost
//    whatever the global Ordering is, and every block the integrators build --
//    VectorBlockDiagonalIntegrator's, and the base class's HDG face slicer --
//    uses that same layout. So byNODES and byVDIM give the same answer, which
//    the ordering case here checks rather than assumes.
namespace darcy_reconstruction_system
{

/// Field e gets its own frequency, so a block read from the wrong place shows
/// up as a wrong answer rather than as a plausible one.
real_t pEx(int e, const Vector &x)
{
   real_t r = 1.0;
   for (int i = 0; i < x.Size(); i++) { r *= std::sin((e + 1) * M_PI * x(i)); }
   return r;
}

real_t gEx(int e, const Vector &x)
{
   return -x.Size() * (e + 1) * (e + 1) * M_PI * M_PI * pEx(e, x);
}

/** @brief A stabilization of size O(1) in place of the integrator's built-in
    O(1/h), which is what makes superconvergence visible at all.

    HDGDiffusionIntegrator's own stabilization is `beta {h^-1 Q}` -- see the
    class comment -- and an LDG-H method stabilized at O(1/h) loses an order in
    the flux, after which no postprocessing can reach k+2. Measured on the
    problem below at k = 1 over 4x4 to 32x32, with the built-in stabilization:

        ||p - p_h||  4.21e-2  6.72e-3  1.19e-3  2.65e-4   rate -> 2.17
        ||p - p*||   1.12e-2  1.15e-3  1.62e-4  3.31e-5   rate -> 2.29
        ||q - q_h||  1.50e-1  3.31e-2  1.11e-2  5.00e-3   rate -> 1.15

    -- the flux at k and the postprocessed potential at k+1, with the classic
    postprocessing giving 2.21 on the same run, so it is the discretisation and
    not the reconstruction. With tau = 1 the same three columns come out at
    2.04, 3.04 and 2.04. That is why this hook is here, and it is worth knowing
    before reading a rate off this integrator's defaults. */
class ConstTau : public HDGStabilization
{
   real_t t;
public:
   ConstTau(real_t t_) : t(t_) { }
   real_t Eval(real_t, real_t, real_t, real_t,
               ElementTransformation &) const override { return t; }
};

/// Everything one solve produces, kept whole so that two runs can be compared
/// dof by dof and not only through an error norm.
struct Sys
{
   std::unique_ptr<Mesh> mesh;
   std::unique_ptr<L2_FECollection> u_coll, p_coll;
   std::unique_ptr<DG_Interface_FECollection> t_coll;
   std::unique_ptr<FiniteElementSpace> fes_u, fes_p, fes_t;
   std::unique_ptr<DarcyForm> darcy;
   std::unique_ptr<ConstTau> tau;
   BlockVector x;
   Vector X;
   GridFunction ut, u_s, p_s, tr_s;
   std::unique_ptr<GridFunction> q_h, p_h;
   int neq{0};
};

/// L2 error of field @a e of a scalar-range grid function of `vdim == neq`
/// under byNODES: field e is the contiguous dof range [e*nd, (e+1)*nd).
real_t PotFieldError(const GridFunction &gf, int e)
{
   const FiniteElementSpace *fes = gf.FESpace();
   const int nd = fes->GetNDofs();
   FiniteElementSpace scalar(fes->GetMesh(), fes->FEColl());
   GridFunction blk(&scalar);
   for (int i = 0; i < nd; i++) { blk(i) = gf(e * nd + i); }

   FunctionCoefficient c([e](const Vector &x) { return pEx(e, x); });
   const int qo = 2 * fes->GetMaxElementOrder() + 6;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   { irs[i] = &(IntRules.Get(i, qo)); }
   return blk.ComputeL2Error(c, irs);
}

/** @brief The integral of field @a e of @a gf over each element.

    This is the quantity the closure row of the local problem sets, so it is
    the direct read-out of whether each field got a closure of its own. Read
    through GetElementVDofs(), which is field-outermost locally under either
    Ordering, so it does not care which one the space has. */
void ElementIntegrals(const GridFunction &gf, int e, Vector &vals)
{
   const FiniteElementSpace *fes = gf.FESpace();
   Mesh *mesh = fes->GetMesh();
   vals.SetSize(mesh->GetNE());

   Array<int> vdofs;
   Vector loc, shape;
   for (int z = 0; z < mesh->GetNE(); z++)
   {
      const FiniteElement *fe = fes->GetFE(z);
      ElementTransformation *T = mesh->GetElementTransformation(z);
      const int nd = fe->GetDof();
      fes->GetElementVDofs(z, vdofs);
      gf.GetSubVector(vdofs, loc);
      const Vector blk(loc.GetData() + e * nd, nd);
      shape.SetSize(nd);

      const IntegrationRule &ir =
         IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + 4);
      real_t s = 0.0;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         fe->CalcShape(ip, shape);
         s += ip.weight * T->Weight() * (shape * blk);
      }
      vals(z) = s;
   }
}

/// @a neq copies of the scalar Darcy problem, block diagonal and with a source
/// per field, hybridized, solved together and reconstructed. Block diagonal on
/// purpose: field 0's discrete problem is then *the* scalar problem, so the
/// scalar answer is available as a reference inside the system run.
std::unique_ptr<Sys> Solve(int n, int order, int neq, real_t tau = 1.0,
                           Ordering::Type ord = Ordering::byNODES)
{
   auto S = std::unique_ptr<Sys>(new Sys);
   S->neq = neq;
   const int dim = 2;
   S->mesh.reset(new Mesh(Mesh::MakeCartesian2D(n, n,
                                                Element::QUADRILATERAL)));
   Mesh &mesh = *S->mesh;

   S->u_coll.reset(new L2_FECollection(order, dim));
   S->p_coll.reset(new L2_FECollection(order, dim));
   S->t_coll.reset(new DG_Interface_FECollection(order, dim));
   S->fes_u.reset(new FiniteElementSpace(&mesh, S->u_coll.get(), neq * dim,
                                         ord));
   S->fes_p.reset(new FiniteElementSpace(&mesh, S->p_coll.get(), neq, ord));
   S->fes_t.reset(new FiniteElementSpace(&mesh, S->t_coll.get(), neq, ord));

   ConstantCoefficient one(1.0);
   RatioCoefficient ik(1.0, one);
   VectorFunctionCoefficient gcoeff(neq, [neq](const Vector &x, Vector &v)
   {
      for (int e = 0; e < neq; e++) { v(e) = gEx(e, x); }
   });

   S->darcy.reset(new DarcyForm(S->fes_u.get(), S->fes_p.get()));
   S->darcy->GetFluxMassForm()->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(neq, new VectorMassIntegrator(one)));
   MixedBilinearForm *B = S->darcy->GetFluxDivForm();
   B->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(neq, new VectorDivergenceIntegrator()));
   B->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(
                                   neq, new TransposeIntegrator(
                                      new DGNormalTraceIntegrator(-1.0))));

   S->tau.reset(new ConstTau(tau));
   std::vector<BilinearFormIntegrator*> hdgs;
   for (int e = 0; e < neq; e++)
   {
      auto *hdg = new HDGDiffusionIntegrator(ik, 0.5);
      if (tau > 0.0) { hdg->SetStabilization(*S->tau); }
      hdgs.push_back(hdg);
   }
   S->darcy->GetPotentialMassForm()->AddInteriorFaceIntegrator(
      new VectorBlockDiagonalIntegrator(hdgs));

   auto *glf = new VectorDomainLFIntegrator(gcoeff);
   glf->SetIntRule(&IntRules.Get(mesh.GetElementGeometry(0), 6 * order + 12));
   S->darcy->GetPotentialRHS()->AddDomainIntegrator(glf);

   Array<int> ess;
   S->darcy->EnableHybridization(
      S->fes_t.get(),
      new VectorBlockDiagonalIntegrator(neq, new NormalTraceJumpIntegrator()),
      ess);
   S->darcy->Assemble();

   S->x.Update(S->darcy->GetOffsets());
   S->x = 0.0;
   OperatorPtr A;
   Vector RHS;
   S->darcy->FormLinearSystem(ess, S->x, A, S->X, RHS, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(20000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(1e-18);
   lin.SetPreconditioner(prec);
   lin.SetOperator(*A);
   lin.Mult(RHS, S->X);
   REQUIRE(lin.GetConverged());
   S->darcy->RecoverFEMSolution(S->X, S->x);

   S->q_h.reset(new GridFunction(S->fes_u.get(), S->x.GetBlock(0)));
   S->p_h.reset(new GridFunction(S->fes_p.get(), S->x.GetBlock(1)));

   S->darcy->Reconstruct(S->x, S->X, S->ut, S->u_s, S->p_s, S->tr_s);
   return S;
}

} // namespace darcy_reconstruction_system

TEST_CASE("The rich reconstruction superconverges field by field",
          "[DarcyForm][Reconstruction][System]")
{
   using namespace darcy_reconstruction_system;

   // The point of the reconstruction, for a system as for one field: the
   // postprocessed potential gains an order over the computed one. Measured
   // per field over 4x4 to 32x32 with tau = 1, and the whole reason the
   // per-field figures differ is that field e carries frequency (e+1)pi and so
   // is that much less resolved on the same mesh -- the rates are the same
   // asymptotically and the fields are converging towards them from different
   // distances.
   //
   //   k = 1        ||p-p_h||   rate    ||p-p*||   rate
   //   field 0      1.15e-03    2.04    3.38e-05   3.04
   //   field 1      7.63e-03    1.82    4.34e-04   2.82
   //
   //   k = 2        ||p-p_h||   rate    ||p-p*||   rate
   //   field 0      7.94e-06    2.98    7.48e-08   3.96
   //   field 1      1.13e-04    2.87    1.57e-06   3.92
   //
   // Two independent things make this a real check rather than a plausible
   // one. The gain is a full order in every field, which is what k+2 against
   // k+1 means. And the rich reconstruction's answer tracks the classic
   // postprocessing's -- 2.82 against 2.82 and 3.92 against 3.92 in the rows
   // above -- and that path is a different piece of code, general in vdim long
   // before this one was.
   const int neq = GENERATE(1, 2, 3);
   const int order = GENERATE(1, 2);

   // The mesh pair is chosen so that every field is in its asymptotic range,
   // and at k = 1 that costs a refinement: field 2 carries frequency 3pi and
   // over 8x8 to 16x16 gains only 0.59 of an order (1.72 against 2.31) where
   // over 16x16 to 32x32 it gains 0.95 (1.65 against 2.60). Loosening the bar
   // to admit the coarse pair would have made the case pass on a run that does
   // not show the property, so the mesh moved instead of the threshold.
   const int nc = (order == 1) ? 16 : 8;
   CAPTURE(neq, order, nc);

   auto c = Solve(nc, order, neq);
   auto f = Solve(2 * nc, order, neq);

   for (int e = 0; e < neq; e++)
   {
      const real_t p_c = PotFieldError(*c->p_h, e);
      const real_t p_f = PotFieldError(*f->p_h, e);
      const real_t s_c = PotFieldError(c->p_s, e);
      const real_t s_f = PotFieldError(f->p_s, e);

      const real_t rate_p = std::log2(p_c / p_f);
      const real_t rate_s = std::log2(s_c / s_f);
      CAPTURE(e, p_c, p_f, s_c, s_f, rate_p, rate_s);

      // More accurate, and gaining an order rather than merely a constant --
      // the second is the claim that matters and the first alone would not
      // distinguish a better-scaled answer from a better-converging one.
      REQUIRE(s_f < p_f);
      REQUIRE(rate_p > order + 0.5);
      REQUIRE(rate_s > rate_p + 0.7);
   }
}

TEST_CASE("A block-diagonal system's first field is the scalar problem",
          "[DarcyForm][Reconstruction][System]")
{
   using namespace darcy_reconstruction_system;

   // The sharpest statement available about the generalisation, and it needs
   // no rate: the neq fields here do not couple, so field 0's discrete problem
   // *is* the one-field problem, and every quantity the reconstruction
   // produces for field 0 has to come back the same whether it was solved
   // alone or alongside others. Anything that mislaid a block -- a size that
   // forgot vdim, a closure row in the wrong place, a face right-hand side
   // that contracted the wrong slice of the total flux -- moves this.
   //
   // The agreement is to the linear solver rather than to the bit, and that is
   // not slack in the reconstruction: the neq-field trace system is a
   // different matrix, so GMRES takes a different path to it and stops a
   // different distance from its root. Measured relative differences are
   // ~1e-11 at a solver tolerance of 1e-14.
   const int order = GENERATE(1, 2);
   const int neq = GENERATE(2, 3);
   CAPTURE(order, neq);

   auto one = Solve(8, order, 1);
   auto many = Solve(8, order, neq);

   const real_t e1 = PotFieldError(one->p_s, 0);
   const real_t en = PotFieldError(many->p_s, 0);
   CAPTURE(e1, en);
   REQUIRE(en == Approx(e1).epsilon(1e-8));

   // And the computed solution too, so that a failure above cannot be blamed
   // on the forward solve having landed somewhere else.
   REQUIRE(PotFieldError(*many->p_h, 0)
           == Approx(PotFieldError(*one->p_h, 0)).epsilon(1e-8));

   // Dof by dof, on the enriched potential: the spaces are built by the
   // reconstruction itself, one scalar and one of vdim == neq, so field 0 of
   // the second is the leading dof range of the first under byNODES.
   const int nd = one->p_s.FESpace()->GetNDofs();
   REQUIRE(many->p_s.FESpace()->GetNDofs() == nd);
   real_t dmax = 0.0, ref = 0.0;
   for (int i = 0; i < nd; i++)
   {
      dmax = std::max(dmax, std::abs(many->p_s(i) - one->p_s(i)));
      ref = std::max(ref, std::abs(one->p_s(i)));
   }
   CAPTURE(dmax, ref);
   REQUIRE(dmax < 1e-8 * ref);
}

TEST_CASE("Each field of the reconstruction is closed by its own average",
          "[DarcyForm][Reconstruction][System]")
{
   using namespace darcy_reconstruction_system;

   // The closure row, read back directly. The local problem is pure Neumann
   // per field, so one equation of each field's block is replaced by
   //
   //     (p*_e, 1)_K = (p_h,e, 1)_K
   //
   // and this walks every element and every field and checks that identity.
   // It is the one place where the structural decision of the generalisation
   // is visible as a number rather than as a rate.
   //
   // What it would catch: neq closure rows all placed in field 0's block --
   // the obvious way to write the loop -- leaves fields 1..neq-1 with their
   // constants free and field 0's block overdetermined. DenseMatrixInverse
   // ::Factor() does not complain about that (its tolerance is an exact zero),
   // so the failure is silent, and the *rates* of the other case here can
   // still look plausible on a coarse mesh. This one cannot: the averages of
   // the unclosed fields are then whatever the singular solve returned.
   const int order = GENERATE(1, 2);
   const int neq = GENERATE(1, 2, 3);
   CAPTURE(order, neq);

   auto S = Solve(4, order, neq);

   for (int e = 0; e < neq; e++)
   {
      Vector got, want;
      ElementIntegrals(S->p_s, e, got);
      ElementIntegrals(*S->p_h, e, want);
      REQUIRE(got.Size() == want.Size());

      real_t dmax = 0.0, ref = 0.0;
      for (int z = 0; z < got.Size(); z++)
      {
         dmax = std::max(dmax, std::abs(got(z) - want(z)));
         ref = std::max(ref, std::abs(want(z)));
      }
      CAPTURE(e, dmax, ref);
      REQUIRE(ref > 0.0);
      REQUIRE(dmax < 1e-12 * ref);
   }
}

TEST_CASE("The reconstruction does not depend on the dof ordering",
          "[DarcyForm][Reconstruction][System]")
{
   using namespace darcy_reconstruction_system;

   // byNODES puts all of field 0's dofs first and byVDIM interleaves them, so
   // "the fields are contiguous blocks" is a statement about the *global*
   // numbering that only one of the two satisfies. The reconstruction never
   // relies on it: it reads through GetElementVDofs() and GetFaceVDofs(),
   // which lay an element's or a face's vdofs out field-outermost under either
   // Ordering, and it writes back through the same arrays. So neither has to
   // be required and neither has to be special-cased.
   //
   // The comparison is by an error norm against a VectorCoefficient rather
   // than dof by dof, because the two runs number their dofs differently and
   // ComputeL2Error() reads the space instead of assuming a layout. As above,
   // the agreement is the linear solver's: the two trace systems are
   // permutations of each other and GMRES does not commute with a permutation.
   const int order = GENERATE(1, 2);
   const int neq = 2;
   CAPTURE(order, neq);

   auto a = Solve(8, order, neq, 1.0, Ordering::byNODES);
   auto b = Solve(8, order, neq, 1.0, Ordering::byVDIM);

   VectorFunctionCoefficient pall(neq, [neq](const Vector &x, Vector &v)
   {
      for (int e = 0; e < neq; e++) { v(e) = pEx(e, x); }
   });
   const int qo = 2 * order + 8;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   { irs[i] = &(IntRules.Get(i, qo)); }

   const real_t ea = a->p_s.ComputeL2Error(pall, irs);
   const real_t eb = b->p_s.ComputeL2Error(pall, irs);
   CAPTURE(ea, eb);
   REQUIRE(eb == Approx(ea).epsilon(1e-8));

   // The unpostprocessed solve too, so that agreement above cannot be an
   // accident of both runs having solved something else.
   REQUIRE(b->p_h->ComputeL2Error(pall, irs)
           == Approx(a->p_h->ComputeL2Error(pall, irs)).epsilon(1e-8));
}
