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

namespace darcy_nl_mms
{

// A manufactured solution for a *coupled nonlinear* Darcy system, and the
// convergence study it exists for.
//
// Everything else about nonlinear systems on this branch is checked
// pointwise: the element and face Jacobians against differences of their own
// residuals, the face blocks against each other, one Newton step against an
// exact linear answer. None of that says the discretization converges to the
// right thing -- a consistently wrong sign in a face term is invisible to a
// Jacobian check, because the Jacobian of a wrong residual is still its own
// Jacobian. Only measuring rates against a known solution finds it.
//
// The system solved is, for e = 0 .. NEQ-1,
//
//     D(p) u + grad p = 0,      -div u_e = g_e,
//
// which is what DarcyForm assembles: MixedFluxFunction::ComputeDualFlux
// returns D(p) u, and D is a *resistivity*. Writing the manufactured solution
// against D directly would need u = -D(p)^{-1} grad p and the divergence of
// that, so instead the conductivity K is the thing chosen -- a nonlinear,
// symmetric, fully coupled 2x2 matrix -- and the flux function hands the
// solver D = K^{-1}, inverted analytically. Then
//
//     u_e = - sum_j K_ej(p) grad p_j
//
// is explicit, and its divergence is elementary. Nothing about the
// discretization is special-cased for it.

constexpr int NEQ = 2;

/// Which mixed space the flux lives in, as in test_darcy_system.cpp. RT is
/// the hybridized mixed method, with no face stabilization; DG is the fully
/// discontinuous one, where the faces carry an explicit tau.
enum class Form { RT, DG };

/// The stabilization, as a constant. MixedConductionNLFIntegrator's
/// per-variable tau is a true constant -- the face weight it multiplies is
/// the physical face measure, with no 1/h -- so unlike HDGDiffusionIntegrator
/// there is no h in the value to correct for. NPC-1 section 3.6.3 asks for
/// eta_d = kappa/l with l a fixed length scale, and K here sits between 1 and
/// about 2.5 on a unit square. Measured on this problem, rates over the
/// 16x16 to 32x32 pair:
///
///          k=0: p, u        k=1: p, u
///   0.5    1.00, 0.67       1.83, 1.76
///   1      1.00, 0.83       1.91, 1.84
///   2      1.01, 0.93       1.96, 1.90
///   4      1.05, 0.94       1.98, 1.96
///   8      0.91, 0.86       1.99, 1.98
///
/// So tau of the size of kappa, and the flux is what pays for getting it
/// wrong in either direction: too small and it never reaches its order, too
/// large and k=0 starts to lock, which is the same trade the linear study in
/// test_darcy_system.cpp found. A per-variable tau tracking each equation's
/// own conductivity, (1, 2) against the scalar 1, moved nothing measurably --
/// the magnitude is what matters here, not the ratio.
constexpr real_t TAU = 2.0;

// The exact potentials, p_e(x) = c_e + a_e prod_d sin(w_e x_d + f_e). The
// offsets and phases keep them away from zero on the boundary, so the
// Dirichlet data is nontrivial and the conductivity is genuinely varying
// there rather than frozen at K(0); the two frequencies are unrelated so the
// equations cannot be proportional to one another.
const real_t pC[NEQ] = { 0.3, -0.4 };
const real_t pA[NEQ] = { 0.7,  0.5 };
const real_t pW[NEQ] = { M_PI, 1.5 };
const real_t pF[NEQ] = { 0.2,  0.9 };

real_t Pex(int e, const Vector &x)
{
   real_t s = pA[e];
   for (int d = 0; d < x.Size(); d++) { s *= std::sin(pW[e] * x(d) + pF[e]); }
   return pC[e] + s;
}

real_t dPex(int e, const Vector &x, int j)
{
   real_t s = pA[e] * pW[e];
   for (int d = 0; d < x.Size(); d++)
   {
      s *= (d == j) ? std::cos(pW[e] * x(d) + pF[e])
           : std::sin(pW[e] * x(d) + pF[e]);
   }
   return s;
}

/// The second derivative in any one direction. Every factor is a sine of the
/// same frequency, so it is the same in every direction.
real_t ddPex(int e, const Vector &x)
{
   return -pW[e] * pW[e] * (Pex(e, x) - pC[e]);
}

/// The off-diagonal strength of the conductivity.
constexpr real_t KOFF = 0.5;

/// K(p), symmetric and nonlinear in both potentials. Its determinant is
/// 2 + p1^2 + 2 p0^2 + (1 - KOFF^2) p0^2 p1^2, positive for every p, so the
/// law is uniformly elliptic and Newton cannot walk out of its domain.
void Kmat(const Vector &p, DenseMatrix &K)
{
   K.SetSize(NEQ);
   K(0, 0) = 1.0 + p(0) * p(0);
   K(1, 1) = 2.0 + p(1) * p(1);
   K(0, 1) = K(1, 0) = KOFF * p(0) * p(1);
}

/// dK/dp_m.
void dKmat(const Vector &p, int m, DenseMatrix &dK)
{
   dK.SetSize(NEQ);
   dK = 0.0;
   if (m == 0)
   {
      dK(0, 0) = 2.0 * p(0);
      dK(0, 1) = dK(1, 0) = KOFF * p(1);
   }
   else
   {
      dK(1, 1) = 2.0 * p(1);
      dK(0, 1) = dK(1, 0) = KOFF * p(0);
   }
}

void pExact(const Vector &x, Vector &p)
{
   p.SetSize(NEQ);
   for (int e = 0; e < NEQ; e++) { p(e) = Pex(e, x); }
}

/// The natural boundary datum of the flux equation, -p.
void pNatural(const Vector &x, Vector &v)
{
   pExact(x, v);
   v.Neg();
}

/// u_e = -K(p) grad p_e, in the equation-outermost layout both flux spaces
/// use: component e*dim + d.
void uExact(const Vector &x, Vector &u)
{
   const int dim = x.Size();
   Vector p;
   pExact(x, p);
   DenseMatrix K;
   Kmat(p, K);

   u.SetSize(NEQ * dim);
   for (int e = 0; e < NEQ; e++)
      for (int d = 0; d < dim; d++)
      {
         real_t s = 0.0;
         for (int j = 0; j < NEQ; j++) { s += K(e, j) * dPex(j, x, d); }
         u(e * dim + d) = -s;
      }
}

/// Drop the conductivity's own variation from the source -- the classic
/// manufactured-solution mistake, and the control for the study below.
bool drop_dK = false;

/// g_e = -div u_e = sum_d d_d ( sum_j K_ej d_d p_j ), which is
///
///   sum_j sum_d [ (sum_m dK_ej/dp_m d_d p_m) d_d p_j + K_ej d_d d_d p_j ].
///
/// The first group is the part that exists only because K depends on p.
void gExact(const Vector &x, Vector &g)
{
   const int dim = x.Size();
   Vector p;
   pExact(x, p);
   DenseMatrix K, dK;
   Kmat(p, K);

   g.SetSize(NEQ);
   for (int e = 0; e < NEQ; e++)
   {
      real_t s = 0.0;
      for (int j = 0; j < NEQ; j++)
         for (int d = 0; d < dim; d++)
         {
            s += K(e, j) * ddPex(j, x);
         }

      if (!drop_dK)
      {
         for (int m = 0; m < NEQ; m++)
         {
            dKmat(p, m, dK);
            for (int j = 0; j < NEQ; j++)
               for (int d = 0; d < dim; d++)
               {
                  s += dK(e, j) * dPex(m, x, d) * dPex(j, x, d);
               }
         }
      }
      g(e) = s;
   }
}

/// The constitutive law as the solver sees it: D(p) = K(p)^{-1}.
class MMSFlux : public MixedFluxFunction
{
   void Dmat(const Vector &p, DenseMatrix &D) const
   {
      DenseMatrix K;
      Kmat(p, K);
      const real_t det = K(0, 0) * K(1, 1) - K(0, 1) * K(1, 0);
      D.SetSize(NEQ);
      D(0, 0) =  K(1, 1) / det;
      D(1, 1) =  K(0, 0) / det;
      D(0, 1) = -K(0, 1) / det;
      D(1, 0) = -K(1, 0) / det;
   }

public:
   MMSFlux(int dim_) : MixedFluxFunction(NEQ, dim_) { }

   real_t ComputeDualFlux(const Vector &p, const DenseMatrix &flux,
                          ElementTransformation &, DenseMatrix &df) const override
   {
      DenseMatrix D;
      Dmat(p, D);
      df.SetSize(NEQ, dim);
      for (int d = 0; d < dim; d++)
         for (int i = 0; i < NEQ; i++)
         {
            real_t s = 0.0;
            for (int j = 0; j < NEQ; j++) { s += D(i, j) * flux(j, d); }
            df(i, d) = s;
         }
      return std::max(D(0, 0), D(1, 1));
   }

   real_t ComputeFlux(const Vector &, ElementTransformation &,
                      DenseMatrix &flux) const override
   {
      flux = 0.0;
      return 0.0;
   }

   void ComputeDualFluxJacobian(const Vector &p, const DenseMatrix &flux,
                                ElementTransformation &,
                                DenseMatrix &J_u, DenseMatrix &J_F) const override
   {
      DenseMatrix D;
      Dmat(p, D);

      J_F.SetSize(NEQ * dim, NEQ * dim);
      J_F = 0.0;
      for (int d = 0; d < dim; d++)
         for (int i = 0; i < NEQ; i++)
            for (int j = 0; j < NEQ; j++)
            {
               J_F(i * dim + d, j * dim + d) = D(i, j);
            }

      // dD/dp_m = -D (dK/dp_m) D, the derivative of an inverse.
      J_u.SetSize(NEQ * dim, NEQ);
      J_u = 0.0;
      DenseMatrix dK(NEQ), t(NEQ), dD(NEQ);
      for (int m = 0; m < NEQ; m++)
      {
         dKmat(p, m, dK);
         Mult(dK, D, t);
         Mult(D, t, dD);
         dD.Neg();
         for (int d = 0; d < dim; d++)
            for (int i = 0; i < NEQ; i++)
            {
               real_t s = 0.0;
               for (int j = 0; j < NEQ; j++) { s += dD(i, j) * flux(j, d); }
               J_u(i * dim + d, m) = s;
            }
      }
   }
};

struct Result
{
   real_t err_p, err_u;
   int newton_its;
   int solved_size;
};

/// Everything one solve needs, kept alive together so that a test can reach
/// the reduced operator and not only the errors. The arrangement is the one
/// test_darcy_system.cpp uses for the linear problem, with the linear flux
/// mass and face stabilization replaced by MixedConductionNLFIntegrator; the
/// boundary treatment -- Dirichlet data entering weakly through the flux
/// equation's natural term, no boundary stabilization -- is unchanged from
/// there.
class System
{
   static FiniteElementCollection *FluxColl(Form form, int order, int dim)
   {
      if (form == Form::DG) { return new L2_FECollection(order, dim); }
      return new RT_FECollection(order, dim);
   }

   const int dim, neq, order;
   const bool dg;
   std::unique_ptr<FiniteElementCollection> u_coll;
   L2_FECollection p_coll;
   DG_Interface_FECollection t_coll;
   FiniteElementSpace fes_u, fes_p, fes_t;
   MMSFlux flux;
   VectorFunctionCoefficient natcoeff, gcoeff;
   DarcyForm darcy;
   Array<int> ess;
   BlockVector x;

public:
   OperatorPtr op;
   Vector X, RHS;

   System(Mesh &mesh, int order_, Form form)
      : dim(mesh.Dimension()), neq(NEQ), order(order_), dg(form == Form::DG),
        u_coll(FluxColl(form, order_, dim)),
        p_coll(order_, dim), t_coll(order_, dim),
        fes_u(&mesh, u_coll.get(), dg ? neq * dim : neq, Ordering::byNODES),
        fes_p(&mesh, &p_coll, neq, Ordering::byNODES),
        fes_t(&mesh, &t_coll, neq, Ordering::byNODES),
        flux(dim),
        natcoeff(neq, pNatural), gcoeff(neq, gExact),
        darcy(&fes_u, &fes_p)
   {
      BlockNonlinearForm *Mnl = darcy.GetBlockNonlinearForm();
      Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(flux));

      MixedBilinearForm *Bform = darcy.GetFluxDivForm();
      if (dg)
      {
         // tau enters as beta times TauVar(e): one scalar per variable, all
         // equal here, which is where the NPC papers say to start.
         Vector taus(neq);
         taus = 1.0;
         auto *face = new MixedConductionNLFIntegrator(flux, TAU);
         face->SetVariableStabilization(taus);
         Mnl->AddInteriorFaceIntegrator(face);

         Bform->AddDomainIntegrator(
            new VectorBlockDiagonalIntegrator(neq, new VectorDivergenceIntegrator));
         Bform->AddInteriorFaceIntegrator(
            new VectorBlockDiagonalIntegrator(
               neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))));
      }
      else
      {
         Bform->AddDomainIntegrator(
            new VectorBlockDiagonalIntegrator(neq, new VectorFEDivergenceIntegrator));
      }

      if (dg)
      {
         darcy.GetFluxRHS()->AddBdrFaceIntegrator(
            new VectorBoundaryFluxLFIntegrator(natcoeff));
      }
      else
      {
         darcy.GetFluxRHS()->AddBoundaryIntegrator(
            new VectorFEBoundaryFluxLFIntegrator(natcoeff));
      }
      darcy.GetPotentialRHS()->AddDomainIntegrator(
         new VectorDomainLFIntegrator(gcoeff));

      darcy.EnableHybridization(
         &fes_t,
         new VectorBlockDiagonalIntegrator(neq, new NormalTraceJumpIntegrator),
         ess);

      darcy.Assemble();

      // The element-local solves are nonlinear too, and get their own Newton.
      darcy.GetHybridization()->SetLocalNLSolver(
         DarcyHybridization::LSsolveType::Newton, 100, 1e-13, 1e-15, -1);

      x.Update(darcy.GetOffsets());
      x = 0.0;
      darcy.FormLinearSystem(ess, x, op, X, RHS, true);
   }

   /// Newton to convergence, then the L2 errors against the exact fields.
   Result Solve()
   {
      GSSmoother prec;
      GMRESSolver lin;
      lin.SetKDim(500);
      lin.SetMaxIter(5000);
      lin.SetRelTol(1e-13);
      lin.SetAbsTol(0.0);
      lin.SetPreconditioner(prec);

      NewtonSolver newton;
      newton.SetSolver(lin);
      newton.SetOperator(*op);
      newton.SetRelTol(1e-12);
      newton.SetAbsTol(1e-14);
      newton.SetMaxIter(50);
      newton.SetPrintLevel(-1);
      newton.Mult(RHS, X);
      REQUIRE(newton.GetConverged());

      darcy.RecoverFEMSolution(X, x);

      GridFunction u_h(&fes_u, x.GetBlock(0));
      GridFunction p_h(&fes_p, x.GetBlock(1));

      const int quad_order = 2 * order + 4;
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i = 0; i < Geometry::NumGeom; i++)
      {
         irs[i] = &(IntRules.Get(i, quad_order));
      }

      VectorFunctionCoefficient pcoeff(neq, pExact), ucoeff(neq * dim, uExact);

      Result res;
      res.err_p = p_h.ComputeL2Error(pcoeff, irs);
      res.err_u = u_h.ComputeL2Error(ucoeff, irs);
      res.newton_its = newton.GetNumIterations();
      res.solved_size = X.Size();
      return res;
   }
};

Result Solve(Mesh &mesh, int order, Form form)
{
   System sys(mesh, order, form);
   return sys.Solve();
}

} // namespace darcy_nl_mms

namespace darcy_nl_mms
{

// Superconvergent postprocessing.
//
// The branch reconstructs the normally continuous total flux and then flux
// and potential one order higher, which is much of the point of an HDG
// method. It is implemented for a *scalar* field only:
// DarcyForm::ReconstructFluxAndPot asserts fes_p->GetVDim() == 1, and the
// kernel under it indexes the potential, trace and total-flux spaces without
// vdim, builds those enriched spaces with no vdim argument, and reaches
// DarcyHybridization::ReconstructTotalFlux whose callback takes a scalar
// potential. So the system above cannot be postprocessed as the branch
// stands. What can be measured is the same manufactured problem with one
// field, which is what follows.
//
// One field means a scalar conductivity, and the existing
// FunctionDiffusionFlux takes 1/k and its derivative directly.

real_t Kone(real_t p) { return 1.0 + p * p; }
real_t dKone(real_t p) { return 2.0 * p; }

real_t pScalar(const Vector &x) { return Pex(0, x); }
real_t pNatural1(const Vector &x) { return -Pex(0, x); }

void uExact1(const Vector &x, Vector &u)
{
   const int dim = x.Size();
   const real_t k = Kone(Pex(0, x));
   u.SetSize(dim);
   for (int d = 0; d < dim; d++) { u(d) = -k * dPex(0, x, d); }
}

/// g = -div u = sum_d [ k'(p) (d_d p)^2 + k(p) d_d d_d p ].
real_t gExact1(const Vector &x)
{
   const real_t p = Pex(0, x);
   real_t s = 0.0;
   for (int d = 0; d < x.Size(); d++)
   {
      s += dKone(p) * dPex(0, x, d) * dPex(0, x, d) + Kone(p) * ddPex(0, x);
   }
   return s;
}

struct PostResult
{
   real_t err_p, err_u, err_ut, err_ps, err_us;
   int newton_its;
};

/// Solve the single-field nonlinear problem hybridized, then postprocess.
/** The stabilization goes on the potential mass form as HDGDiffusionIntegrator
    rather than on the block nonlinear form, because the reconstruction needs a
    linear potential constraint integrator to build its local system from. That
    is also the arrangement convdiff uses for a hybridized nonlinear DG
    problem, so this measures the branch's own configuration. */
PostResult SolvePost(Mesh &mesh, int order, Form form)
{
   const int dim = mesh.Dimension();
   const bool dg = (form == Form::DG);

   std::unique_ptr<FiniteElementCollection> u_coll;
   if (dg) { u_coll.reset(new L2_FECollection(order, dim)); }
   else    { u_coll.reset(new RT_FECollection(order, dim)); }
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, u_coll.get(), dg ? dim : 1);
   FiniteElementSpace fes_p(&mesh, &p_coll);
   FiniteElementSpace fes_t(&mesh, &t_coll);

   DarcyForm darcy(&fes_u, &fes_p);

   auto kinv  = [](const Vector &, real_t s) { return 1.0 / Kone(s); };
   auto dkinv = [](const Vector &, real_t s)
   {
      return -dKone(s) / (Kone(s) * Kone(s));
   };
   FunctionDiffusionFlux flux(dim, kinv, dkinv);

   darcy.GetBlockNonlinearForm()->AddDomainIntegrator(
      new MixedConductionNLFIntegrator(flux));

   MixedBilinearForm *Bform = darcy.GetFluxDivForm();
   ConstantCoefficient one(1.0);
   if (dg)
   {
      Bform->AddDomainIntegrator(new VectorDivergenceIntegrator);
      Bform->AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.)));
      // tau = td kappa / h with the coefficient one, so td = TAU h is a fixed
      // tau of TAU, as in the linear study.
      const real_t td = TAU * mesh.GetElementSize(0);
      darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
         new HDGDiffusionIntegrator(one, td));
   }
   else
   {
      Bform->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   }

   FunctionCoefficient natcoeff(pNatural1), gcoeff(gExact1);
   if (dg)
   {
      darcy.GetFluxRHS()->AddBdrFaceIntegrator(
         new VectorBoundaryFluxLFIntegrator(natcoeff));
   }
   else
   {
      darcy.GetFluxRHS()->AddBoundaryIntegrator(
         new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   }
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));

   Array<int> ess;
   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator, ess);
   darcy.Assemble();
   darcy.GetHybridization()->SetLocalNLSolver(
      DarcyHybridization::LSsolveType::Newton, 100, 1e-13, 1e-15, -1);

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr op;
   Vector X, RHS;
   darcy.FormLinearSystem(ess, x, op, X, RHS, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(500);
   lin.SetMaxIter(5000);
   lin.SetRelTol(1e-13);
   lin.SetAbsTol(0.0);
   lin.SetPreconditioner(prec);

   NewtonSolver newton;
   newton.SetSolver(lin);
   newton.SetOperator(*op);
   newton.SetRelTol(1e-12);
   newton.SetAbsTol(1e-14);
   newton.SetMaxIter(50);
   newton.SetPrintLevel(-1);
   newton.Mult(RHS, X);
   REQUIRE(newton.GetConverged());

   darcy.RecoverFEMSolution(X, x);

   // Postprocess. The spaces are built by the reconstruction itself.
   GridFunction ut, u_s, p_s, tr_s;
   darcy.Reconstruct(x, X, ut, u_s, p_s, tr_s);

   GridFunction u_h(&fes_u, x.GetBlock(0));
   GridFunction p_h(&fes_p, x.GetBlock(1));

   const int quad_order = 2 * order + 6;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      irs[i] = &(IntRules.Get(i, quad_order));
   }

   FunctionCoefficient pcoeff(pScalar);
   VectorFunctionCoefficient ucoeff(dim, uExact1);

   PostResult res;
   res.err_p  = p_h.ComputeL2Error(pcoeff, irs);
   res.err_u  = u_h.ComputeL2Error(ucoeff, irs);
   res.err_ut = ut.ComputeL2Error(ucoeff, irs);
   res.err_ps = p_s.ComputeL2Error(pcoeff, irs);
   res.err_us = u_s.ComputeL2Error(ucoeff, irs);
   res.newton_its = newton.GetNumIterations();
   return res;
}

} // namespace darcy_nl_mms

TEST_CASE("The manufactured coupled nonlinear solution is self-consistent",
          "[DarcyForm][NonlinearDarcy][System]")
{
   using namespace darcy_nl_mms;

   // A convergence study is only as good as the solution it measures
   // against, and a manufactured source is the easiest thing in it to get
   // wrong -- a wrong g shows up as rate zero with nothing to say why. So
   // check the three statements the study rests on, away from the mesh and
   // the assembly entirely:
   //
   //   the flux law the solver is handed inverts the K the solution uses,
   //   the source really is minus the divergence of the exact flux, and
   //   the analytic flux Jacobian differentiates the flux law.
   const int dim = GENERATE(2, 3);
   CAPTURE(dim);

   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, false, 1., 1.)
               : Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON, 1., 1., 1.);
   ElementTransformation *T = mesh.GetElementTransformation(0);
   IntegrationPoint ip;
   if (dim == 2) { ip.Set2(0.5, 0.5); }
   else { ip.Set3(0.5, 0.5, 0.5); }
   T->SetIntPoint(&ip);

   MMSFlux flux(dim);

   const real_t pts[4][3] =
   {
      {0.13, 0.71, 0.42}, {0.50, 0.50, 0.50},
      {0.87, 0.22, 0.94}, {0.31, 0.94, 0.08}
   };

   for (const auto &pt : pts)
   {
      Vector x(dim);
      for (int d = 0; d < dim; d++) { x(d) = pt[d]; }
      CAPTURE(x(0), x(1));

      Vector p, uv;
      pExact(x, p);
      uExact(x, uv);

      DenseMatrix u(NEQ, dim), du(NEQ, dim);
      for (int e = 0; e < NEQ; e++)
         for (int d = 0; d < dim; d++)
         {
            u(e, d) = uv(e * dim + d);
         }

      // D(p) u = -grad p: the law handed to the solver is the inverse of the
      // K the exact flux was built from, and with the sign the mixed form
      // wants.
      flux.ComputeDualFlux(p, u, *T, du);
      for (int e = 0; e < NEQ; e++)
         for (int d = 0; d < dim; d++)
         {
            CAPTURE(e, d);
            REQUIRE(du(e, d) == MFEM_Approx(-dPex(e, x, d), 1e-11, 1e-12));
         }

      // g = -div u, differenced. h = 1e-5 on a central difference of a
      // smooth field leaves about 1e-9, far below any discretization error
      // the study will see.
      const real_t h = 1e-5;
      Vector divu(NEQ);
      divu = 0.0;
      for (int d = 0; d < dim; d++)
      {
         Vector xp(x), xm(x), up, um;
         xp(d) += h;
         xm(d) -= h;
         uExact(xp, up);
         uExact(xm, um);
         for (int e = 0; e < NEQ; e++)
         {
            divu(e) += (up(e * dim + d) - um(e * dim + d)) / (2.0 * h);
         }
      }
      Vector g;
      gExact(x, g);
      for (int e = 0; e < NEQ; e++)
      {
         CAPTURE(e, g(e), divu(e));
         REQUIRE(g(e) == MFEM_Approx(-divu(e), 1e-6, 1e-7));
      }

      // And the flux Jacobian, against a difference of the flux law in both
      // its arguments. A wrong one costs Newton its rate but not its answer,
      // so nothing else here would notice.
      DenseMatrix J_u, J_F;
      flux.ComputeDualFluxJacobian(p, u, *T, J_u, J_F);

      const real_t hj = std::cbrt(std::numeric_limits<real_t>::epsilon());
      DenseMatrix fp(NEQ, dim), fm(NEQ, dim);
      for (int m = 0; m < NEQ; m++)
      {
         Vector pp(p), pm(p);
         pp(m) += hj;
         pm(m) -= hj;
         flux.ComputeDualFlux(pp, u, *T, fp);
         flux.ComputeDualFlux(pm, u, *T, fm);
         for (int e = 0; e < NEQ; e++)
            for (int d = 0; d < dim; d++)
            {
               CAPTURE(m, e, d);
               const real_t fd = (fp(e, d) - fm(e, d)) / (2.0 * hj);
               REQUIRE(J_u(e * dim + d, m) == MFEM_Approx(fd, 1e-6, 1e-7));
            }
      }
      for (int j = 0; j < NEQ; j++)
         for (int dd = 0; dd < dim; dd++)
         {
            DenseMatrix up(u), um(u);
            up(j, dd) += hj;
            um(j, dd) -= hj;
            flux.ComputeDualFlux(p, up, *T, fp);
            flux.ComputeDualFlux(p, um, *T, fm);
            for (int e = 0; e < NEQ; e++)
               for (int d = 0; d < dim; d++)
               {
                  CAPTURE(j, dd, e, d);
                  const real_t fd = (fp(e, d) - fm(e, d)) / (2.0 * hj);
                  REQUIRE(J_F(e * dim + d, j * dim + dd)
                          == MFEM_Approx(fd, 1e-6, 1e-7));
               }
         }
   }
}

TEST_CASE("The hybridized gradient does not depend on being asked twice",
          "[DarcyForm][DarcyHybridization][NonlinearDarcy][System][HDG]")
{
   using namespace darcy_nl_mms;

   // DarcyHybridization keeps E, G and H in arrays that outlive one gradient
   // evaluation. H is reset at the top of GetGradient because it takes a
   // contribution from each side of a face and has to be accumulated; E and G
   // are not, because every writer in that file overwrites its own block --
   // except the BlockNonlinearFormIntegrator overload of AssembleHDGGrad,
   // which accumulated. So the second GetGradient at the same point returned
   // a different matrix from the first.
   //
   // Newton calls GetGradient exactly once per step, which is why no
   // Jacobian check finds this: every one of them evaluates the gradient
   // once. What it produced was a first step from a correct Jacobian and
   // every step after it from a doubled E and G, and the iteration diverged
   // -- from a 5% nonlinearity, on a 2x2 mesh.
   //
   // DG only. The RT form has no nonlinear face integrator, so it never
   // reaches the overload at all.
   const int order = GENERATE(0, 1, 2);
   CAPTURE(order);

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   System sys(mesh, order, Form::DG);

   Vector X0(sys.X.Size()), dy(sys.X.Size());
   for (int i = 0; i < X0.Size(); i++)
   {
      X0(i) = 0.03 * std::sin(1.7 * i + 0.4);
      dy(i) = 0.01 * std::cos(0.9 * i + 1.1);
   }

   Vector r(X0.Size()), J1(X0.Size()), J2(X0.Size());

   // Twice over, in the order NewtonSolver uses.
   sys.op->Mult(X0, r);
   sys.op->GetGradient(X0).Mult(dy, J1);
   sys.op->Mult(X0, r);
   sys.op->GetGradient(X0).Mult(dy, J2);

   REQUIRE(J1.Normlinf() > 1e-6);      // the gradient is not trivial

   Vector d(J2);
   d -= J1;
   INFO("the second gradient differs from the first by "
        << d.Normlinf() / J1.Normlinf());
   REQUIRE(d.Normlinf() < 1e-12 * J1.Normlinf());
}

TEST_CASE("A coupled nonlinear Darcy system converges at the design order",
          "[DarcyForm][DarcyHybridization][NonlinearDarcy][System][HDG]")
{
   using namespace darcy_nl_mms;

   const int order = GENERATE(0, 1, 2);
   const Form form = GENERATE(Form::RT, Form::DG);
   CAPTURE(order, int(form));

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);

   // Four meshes, 2x2 to 16x16, with the rate taken between the two finest.
   // The linear study in test_darcy_system.cpp found the coarsest pair still
   // pre-asymptotic for the DG form, reading about 1.6 where 2 is wanted.
   std::vector<real_t> ep, eu;
   int its = 0;
   for (int ref = 0; ref < 4; ref++)
   {
      const Result r = Solve(mesh, order, form);
      ep.push_back(r.err_p);
      eu.push_back(r.err_u);
      its = r.newton_its;
      mesh.UniformRefinement();
   }

   const int n = ep.size();
   const real_t rate_p = std::log2(ep[n-2] / ep[n-1]);
   const real_t rate_u = std::log2(eu[n-2] / eu[n-1]);
   CAPTURE(rate_p, rate_u, ep[n-1], eu[n-1], its);
   REQUIRE(rate_p > order + 0.7);
   REQUIRE(rate_u > order + 0.7);
}

TEST_CASE("Postprocessing lifts the potential a further order",
          "[DarcyForm][DarcyHybridization][NonlinearDarcy][HDG]")
{
   using namespace darcy_nl_mms;

   // The postprocessed potential is the quantity HDG is advertised for: it
   // should converge at k+2 where the solved one converges at k+1. Measured
   // here on a nonlinear problem, which is new -- the reconstruction had never
   // been run on one, and dereferenced a null flux mass form when asked to
   // (convdiff -rec -nld segfaulted).
   //
   // One field, because the reconstruction is scalar-only; see the note above
   // SolvePost.
   const int order = GENERATE(0, 1, 2);
   const Form form = GENERATE(Form::RT, Form::DG);
   CAPTURE(order, int(form));

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);

   std::vector<real_t> ep, eu, et, eps, eus;
   int its = 0;
   for (int ref = 0; ref < 4; ref++)
   {
      const PostResult r = SolvePost(mesh, order, form);
      ep.push_back(r.err_p);
      eu.push_back(r.err_u);
      et.push_back(r.err_ut);
      eps.push_back(r.err_ps);
      eus.push_back(r.err_us);
      its = r.newton_its;
      mesh.UniformRefinement();
   }

   const int n = ep.size();
   auto rate = [&](const std::vector<real_t> &e)
   {
      return std::log2(e[n-2] / e[n-1]);
   };
   const real_t rate_p = rate(ep), rate_u = rate(eu), rate_ut = rate(et);
   const real_t rate_ps = rate(eps), rate_us = rate(eus);
   CAPTURE(rate_p, rate_u, rate_ut, rate_ps, rate_us, its);
   CAPTURE(ep[n-1], eu[n-1], et[n-1], eps[n-1], eus[n-1]);

   // The solved fields at the design order, ...
   REQUIRE(rate_p > order + 0.7);
   REQUIRE(rate_u > order + 0.7);

   // ... and the postprocessed potential a full order better. Measured, over
   // the 8x8 to 16x16 pair:
   //
   //     k  form   p     u     ut    p_s    u_s
   //     0  RT     0.99  1.00  1.00  2.00   1.00
   //     0  DG     1.00  0.88  0.86  1.01   0.86
   //     1  RT     2.00  2.01  2.01  3.09   2.06
   //     1  DG     1.95  1.87  1.86  2.92   1.85
   //     2  RT     3.00  3.01  3.01  4.12   3.07
   //     2  DG     2.99  2.91  2.90  3.94   2.91
   //
   // k+2 everywhere except the fully discontinuous form at k=0, which is the
   // known restriction rather than a defect: the local postprocessing needs
   // the solved potential to be superconvergent in its own element averages,
   // and for an L2 flux that holds only from k=1. Chen, Cockburn, Singler and
   // Zhang, J. Sci. Comput. 81 (2019) 2188, Table 1, report 0.97 at k=0 and
   // 3.01 at k=1 for the same HDG_k method, and their theorem carries the
   // hypothesis k >= 1 explicitly. The hybridized mixed form has it at k=0
   // too, and shows 2.00.
   if (order >= 1 || form == Form::RT)
   {
      REQUIRE(rate_ps > order + 1.5);
   }
   else
   {
      REQUIRE(rate_ps > order + 0.7);
   }

   // The postprocessed potential is also smaller in absolute terms, which a
   // rate alone would not catch if the reconstruction were merely a rescaling.
   REQUIRE(eps[n-1] < ep[n-1]);

   // The flux is not superconvergent and is not claimed to be -- u_s tracks
   // u_h to within a few hundredths of an order in every row above. What the
   // reconstruction must not do is make it worse. This is also the answer to
   // the question the system study left open: the DG form's flux lags its
   // potential, and postprocessing is not what was missing. The lag is a
   // property of the fully discontinuous form on this problem, not of the
   // system -- the single field here lags by the same amount as the two
   // equations do -- and it closes as tau grows.
   REQUIRE(rate_ut > order + 0.7);
   REQUIRE(rate_us > order + 0.7);
}

TEST_CASE("A source that ignores the conductivity's variation does not",
          "[DarcyForm][DarcyHybridization][NonlinearDarcy][System][HDG]")
{
   using namespace darcy_nl_mms;

   // The control. Drop the terms in g that exist only because K depends on
   // p -- differentiate as though the conductivity were locally frozen --
   // and the exact solution no longer solves the discrete problem. If the
   // study above still passed with this source it would be measuring a
   // linear problem wearing a nonlinear law.
   const Form form = GENERATE(Form::RT, Form::DG);
   const int order = 1;
   CAPTURE(int(form));

   drop_dK = true;

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   std::vector<real_t> ep;
   for (int ref = 0; ref < 3; ref++)
   {
      ep.push_back(Solve(mesh, order, form).err_p);
      mesh.UniformRefinement();
   }

   drop_dK = false;

   const int n = ep.size();
   const real_t rate_p = std::log2(ep[n-2] / ep[n-1]);
   CAPTURE(rate_p, ep[0], ep[n-1]);
   REQUIRE(rate_p < 0.5);
}
