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
