//                     MFEM HDG Stokes miniapp
//
// Compile with: make stokes
//
// Sample runs:  stokes -nx 8 -ny 8 -o 1
//               stokes -nx 8 -ny 8 -o 2 -nu 0.1
//               stokes -nx 16 -ny 16 -o 1 -c 3
//
// Description:  A hybridizable discontinuous Galerkin solver for Stokes flow,
//               written to be grown into the general Darcy-like operator that
//               doc/HDG-ROADMAP.md sections 3 and 4 ask for. It follows the
//               velocity-pressure-gradient formulation of
//
//                 Nguyen, Peraire & Cockburn, A hybridizable discontinuous
//                 Galerkin method for Stokes flow, Comput. Methods Appl. Mech.
//                 Engrg. 199 (2010) 582-597,
//
//               but arranges it as a *system of Darcy problems*, one per
//               velocity component, which is the shape DarcyForm and
//               DarcyHybridization already support:
//
//                 nu^-1 q_i + grad u_i = 0          (q_i = -nu grad u_i)
//                 -div q_i + d_i p     = f_i        momentum
//                 div u                = 0          incompressibility
//
//               The exact solution is Kovasznay's, which solves the steady
//               incompressible Navier-Stokes equations, so used here it needs
//               a momentum source -- and the source is exactly the convective
//               term the Stokes equations drop. Every term added later comes
//               with its own source correction, so Kovasznay stays the
//               manufactured solution throughout.
//
//               What is built so far is recorded in Stage below. -diag reads
//               the stage 2 blocks against the exact solution, which is what
//               found the sign error that had the pressure pinned to zero.
//
//               Validated against NPC's own convergence study, section 4.1,
//               which sweeps the stabilization as nu*tau = h^ts. Rates over
//               the two finest of four meshes from 4x4:
//
//                 k  ts   u      q      NPC says
//                 0   0   0.91   0.88   both k+1, and k=0 works at ts=0
//                 1   0   1.98   1.83   both k+1
//                 2   0   2.97   2.87   both k+1
//                 1  -1   2.01   1.00   u at k+1, gradient only k
//                 2  -1   2.97   1.99   u at k+1, gradient only k
//                 1  +1   1.01   1.05   u only k, gradient k+1
//
//               Five of the six reproduce the paper. The exception is the
//               gradient at ts = +1, which settles at k rather than k+1 -- it
//               is not pre-asymptotic, reading 1.03 against u's 1.03 on a
//               64x64 mesh. The most likely reason is simply that stage 1 is
//               not Stokes: with the pressure a known source these are d
//               decoupled diffusion problems, and NPC's table describes the
//               coupled system, where the gradient's convergence is tied to
//               the pressure's. That should be revisited once stage 2 lands
//               rather than explained away now.
//
//               Two things measured while chasing it, both worth keeping:
//
//               The Dirichlet datum reaches the flux equation through C^T,
//               independently of tau. DarcyForm::EnableHybridization walks the
//               *divergence form's* boundary face markers and registers the
//               trace-jump integrator as a boundary flux constraint on each,
//               so adding a boundary face integrator to B is what gives C a
//               boundary block. Perturbing the essential trace values wrecks
//               the solution, which is the check that the datum is live.
//
//               The boundary stabilization below was inert when this was
//               first written -- removing it entirely, or scaling it over six
//               decades, left the answer bit for bit identical -- which turned
//               out to be a defect in VectorBlockDiagonalIntegrator rather
//               than anything here. It is live now, and the rates above are
//               measured with it; they moved by less than a tenth of an order,
//               so the comparison with the paper is unaffected either way.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/// How much of the operator is assembled. The stages are additive: each one
/// keeps Kovasznay as the exact solution by correcting the source.
enum class Stage
{
   /// The momentum equations alone, with the pressure gradient a known
   /// source. d decoupled Darcy problems sharing one manufactured solution;
   /// this is the harness, not yet Stokes.
   Momentum = 1,
   /// Stokes. The pressure is a potential field with no flux and no trace of
   /// its own -- potential d+1, flux d*dim, trace d -- and it enters as the
   /// isotropic part of the total stress, sigma_i = -nu grad u_i + p e_i,
   /// which turns both of its couplings into blocks of B. DarcyForm forms the
   /// transpose that pairs them, which is Stokes' own symmetry.
   ///
   /// Converges at k+1 in all three variables, which is NPC's nu*tau = 1 row.
   /// Rates over the two finest of four meshes from 4x4, or the pair before
   /// that at k=2:
   ///
   ///     k    u      sigma   p
   ///     0    1.01   0.83    0.80
   ///     1    2.05   1.92    2.01
   ///     2    2.76   2.77    2.84
   ///
   /// The pressure is determined only up to a constant by an all-Dirichlet
   /// velocity, so a small multiple of the pressure mass pins it and the
   /// errors are measured with the mean of the difference removed -- from the
   /// stress as well, which carries the pressure on its diagonal and inherits
   /// the same constant. Forgetting the second is what made the stress look
   /// like it had stalled at 0.117 when it had not.
   ///
   /// Two things are still wrong with it. At k=2 the finest mesh breaks down,
   /// 2449 GMRES iterations and the rate collapsing, which is the solver
   /// meeting a nearly singular system rather than the discretization; and it
   /// has to take the Dirichlet datum weakly, because with the pressure
   /// coupled in the reduced matrix comes out structurally asymmetric and the
   /// essential-trace elimination refuses it.
   Stokes = 2,
};

// ---------------------------------------------------------------------------
// Kovasznay's solution, and everything derived from it.
//
//   u1 = 1 - exp(L x) cos(2 pi y)
//   u2 = (L / 2 pi) exp(L x) sin(2 pi y)
//   p  = (1/2) exp(2 L x)
//
// with L = Re/2 - sqrt(Re^2/4 + 4 pi^2) and Re = 1/nu. It solves
// -nu lap u + (u.grad) u + grad p = 0 exactly, and is divergence free.
// ---------------------------------------------------------------------------
class Kovasznay
{
   real_t nu, lam;

public:
   Kovasznay(real_t nu_) : nu(nu_)
   {
      const real_t Re = 1.0 / nu;
      lam = 0.5 * Re - std::sqrt(0.25 * Re * Re + 4.0 * M_PI * M_PI);
   }

   real_t Viscosity() const { return nu; }
   real_t Lambda() const { return lam; }

   /// The velocity, which is the potential of each component's Darcy problem.
   void Velocity(const Vector &x, Vector &u) const
   {
      const real_t e = std::exp(lam * x(0));
      u.SetSize(2);
      u(0) = 1.0 - e * std::cos(2.0 * M_PI * x(1));
      u(1) = lam / (2.0 * M_PI) * e * std::sin(2.0 * M_PI * x(1));
   }

   real_t Pressure(const Vector &x) const
   {
      return 0.5 * std::exp(2.0 * lam * x(0));
   }

   /// grad u, as (i, d) -> i*2 + d.
   void VelocityGrad(const Vector &x, Vector &g) const
   {
      const real_t e = std::exp(lam * x(0));
      const real_t c = std::cos(2.0 * M_PI * x(1)), s = std::sin(2.0 * M_PI * x(1));
      g.SetSize(4);
      g(0) = -lam * e * c;                        // d_x u1
      g(1) = 2.0 * M_PI * e * s;                  // d_y u1
      g(2) = lam * lam / (2.0 * M_PI) * e * s;    // d_x u2
      g(3) = lam * e * c;                         // d_y u2
   }

   /// The flux of component i, q_i = -nu grad u_i, in the branch's sign
   /// convention -- note this is minus nu times NPC's gradient tensor L.
   void Flux(const Vector &x, Vector &q, Stage stage) const
   {
      VelocityGrad(x, q);
      q *= -nu;
      if (stage >= Stage::Stokes)
      {
         const real_t p = Pressure(x);
         for (int i = 0; i < 2; i++) { q(i * 2 + i) += p; }
      }
   }

   /// lap u, per component.
   void VelocityLaplacian(const Vector &x, Vector &l) const
   {
      const real_t e = std::exp(lam * x(0));
      const real_t c = std::cos(2.0 * M_PI * x(1)), s = std::sin(2.0 * M_PI * x(1));
      const real_t k2 = 4.0 * M_PI * M_PI;
      l.SetSize(2);
      l(0) = (k2 - lam * lam) * e * c;
      l(1) = lam / (2.0 * M_PI) * (lam * lam - k2) * e * s;
   }

   void PressureGrad(const Vector &x, Vector &gp) const
   {
      gp.SetSize(2);
      gp(0) = lam * std::exp(2.0 * lam * x(0));
      gp(1) = 0.0;
   }

   /// The momentum source, f = -nu lap u + grad p. For Kovasznay this equals
   /// -(u.grad) u, the term Stokes drops; adding convection later removes it.
   void Momentum(const Vector &x, Vector &f) const
   {
      Vector l, gp;
      VelocityLaplacian(x, l);
      PressureGrad(x, gp);
      f.SetSize(2);
      for (int i = 0; i < 2; i++) { f(i) = -nu * l(i) + gp(i); }
   }

   /// The right-hand side of the potential equation -div q_i = g_i at the
   /// given stage. With the pressure a known source, g_i = nu lap u_i.
   void PotentialRHS(const Vector &x, Vector &g, Stage stage) const
   {
      if (stage >= Stage::Stokes)
      {
         // div sigma_i = -nu lap u_i + d_i p = f_i, so g_i = -f_i, and the
         // pressure's row is the constraint and has no source.
         Vector f;
         Momentum(x, f);
         g.SetSize(3);
         for (int i = 0; i < 2; i++) { g(i) = -f(i); }
         g(2) = 0.0;
         return;
      }
      Vector l;
      VelocityLaplacian(x, l);
      g.SetSize(2);
      for (int i = 0; i < 2; i++) { g(i) = nu * l(i); }
   }

   /// The potential, carrying the pressure as its last component at stage 2.
   void Potential(const Vector &x, Vector &w, Stage stage) const
   {
      Velocity(x, w);
      if (stage >= Stage::Stokes)
      {
         w.SetSize(3);
         w(2) = Pressure(x);
      }
   }
};

// A single instance the coefficient lambdas close over.
static const Kovasznay *kov = NULL;
static Stage kov_stage = Stage::Momentum;

static void uFun(const Vector &x, Vector &u) { kov->Velocity(x, u); }
static void wFun(const Vector &x, Vector &w) { kov->Potential(x, w, kov_stage); }
static void qFun(const Vector &x, Vector &q) { kov->Flux(x, q, kov_stage); }
static void gFun(const Vector &x, Vector &g) { kov->PotentialRHS(x, g, kov_stage); }
static real_t pFun(const Vector &x) { return kov->Pressure(x); }

// ---------------------------------------------------------------------------
/// Check the manufactured data against itself by central differences: that the
/// flux really is -nu grad u, that the velocity is divergence free, and that
/// the momentum source really is -nu lap u + grad p. A wrong source is the
/// easiest thing here to get wrong and shows up as rate zero with nothing to
/// say why, so it is checked before anything depends on it.
real_t CheckData(const Kovasznay &k)
{
   const real_t h = 1e-5;
   const real_t pts[4][2] = {{0.31, 0.22}, {1.00, 0.75}, {1.71, -0.13}, {0.55, 1.24}};
   real_t worst = 0.0;

   for (const auto &pt : pts)
   {
      Vector x(2);
      x(0) = pt[0];
      x(1) = pt[1];

      // grad u and lap u, differenced.
      Vector g_num(4), l_num(2), u0;
      k.Velocity(x, u0);
      l_num = 0.0;
      for (int d = 0; d < 2; d++)
      {
         Vector xp(x), xm(x), up, um;
         xp(d) += h;
         xm(d) -= h;
         k.Velocity(xp, up);
         k.Velocity(xm, um);
         for (int i = 0; i < 2; i++)
         {
            g_num(i * 2 + d) = (up(i) - um(i)) / (2.0 * h);
            l_num(i) += (up(i) - 2.0 * u0(i) + um(i)) / (h * h);
         }
      }

      Vector g_ana;
      k.VelocityGrad(x, g_ana);
      for (int j = 0; j < 4; j++)
      {
         worst = std::max(worst, std::abs(g_ana(j) - g_num(j)));
      }

      Vector l_ana;
      k.VelocityLaplacian(x, l_ana);
      for (int i = 0; i < 2; i++)
      {
         worst = std::max(worst, std::abs(l_ana(i) - l_num(i)));
      }

      // Divergence free.
      worst = std::max(worst, std::abs(g_num(0) + g_num(3)));

      // grad p.
      Vector gp_ana;
      k.PressureGrad(x, gp_ana);
      for (int d = 0; d < 2; d++)
      {
         Vector xp(x), xm(x);
         xp(d) += h;
         xm(d) -= h;
         const real_t gp = (k.Pressure(xp) - k.Pressure(xm)) / (2.0 * h);
         worst = std::max(worst, std::abs(gp_ana(d) - gp));
      }

      // And the momentum source is what it claims to be.
      Vector f_ana;
      k.Momentum(x, f_ana);
      for (int i = 0; i < 2; i++)
      {
         const real_t f = -k.Viscosity() * l_ana(i) + gp_ana(i);
         worst = std::max(worst, std::abs(f_ana(i) - f));
      }
   }
   return worst;
}

// ---------------------------------------------------------------------------
/// One solve. The arrangement is the branch's fully discontinuous one, with
/// the boundary treatment the sweep in section 7 of the roadmap settled on:
/// the traces are essential and carry the projected velocity.
struct Result
{
   real_t err_u, err_q, err_p;
   int    trace_size, iters;
};

Result Solve(Mesh &mesh, int order, real_t td, int ts, Stage stage,
             bool verbose)
{
   const int dim = mesh.Dimension();
   const bool with_p = (stage >= Stage::Stokes);
   const int nv = dim;                     // velocity components
   const int np = with_p ? (nv + 1) : nv;  // potential fields
   // Rectangular by construction: the pressure is a potential with no flux and
   // no trace. Nothing here is padded to make the field counts agree.
   L2_FECollection q_coll(order, dim), u_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace fes_q(&mesh, &q_coll, nv * dim, Ordering::byNODES);
   FiniteElementSpace fes_u(&mesh, &u_coll, np, Ordering::byNODES);
   FiniteElementSpace fes_t(&mesh, &t_coll, nv, Ordering::byNODES);

   DarcyForm darcy(&fes_q, &fes_u);

   const real_t nu = kov->Viscosity();
   ConstantCoefficient inu(1.0 / nu), one(1.0), dnu(-real_t(dim) / nu),
                       eps(1e-8);

   // nu^-1 (sigma, v), replicated down the momentum components.
   std::vector<BilinearFormIntegrator *> mass(nv);
   for (int i = 0; i < nv; i++) { mass[i] = new VectorMassIntegrator(inu); }
   darcy.GetFluxMassForm()->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(mass));

   Array<int> bdr_ess(mesh.bdr_attributes.Max());
   bdr_ess = 1;

   MixedBilinearForm *B = darcy.GetFluxDivForm();
   if (!with_p)
   {
      B->AddDomainIntegrator(
         new VectorBlockDiagonalIntegrator(nv, new VectorDivergenceIntegrator));
      B->AddInteriorFaceIntegrator(
         new VectorBlockDiagonalIntegrator(
            nv, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))));
      B->AddBdrFaceIntegrator(
         new VectorBlockDiagonalIntegrator(
            nv, new TransposeIntegrator(new DGNormalTraceIntegrator(-2.))),
         bdr_ess);
   }
   else
   {
      // The same divergence, placed a block at a time because the potential
      // space now has one more field than the flux space has flux fields.
      for (int i = 0; i < nv; i++)
      {
         B->AddDomainIntegrator(
            new VectorBlockIntegrator(np, nv * dim, i, i * dim,
                                      new VectorDivergenceIntegrator));
         B->AddInteriorFaceIntegrator(
            new VectorBlockIntegrator(
               np, nv * dim, i, i * dim,
               new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))));

         // The coupling, one object read two ways. Forward it collects
         // nu^-1 sum_i (sigma_i)_i into the pressure's row; since
         // (sigma_i)_i = -nu d_i u_i + p that is -div u + (d/nu) p, and the
         // potential mass block below cancels the second term. Transposed --
         // and DarcyForm builds B^T itself -- the same block puts nu^-1 p into
         // component i of the i-th flux equation, the pressure gradient.
         B->AddDomainIntegrator(
            new VectorBlockIntegrator(np, nv * dim, nv, i * dim + i,
                                      new MassIntegrator(inu)));
      }
      darcy.GetPotentialMassForm()->AddDomainIntegrator(
         new VectorBlockIntegrator(np, np, nv, nv, new MassIntegrator(dnu)));
      // All-Dirichlet velocity leaves the pressure determined only up to a
      // constant, so the constraint row is singular. A small multiple of the
      // pressure mass pins it; the error is measured with the mean removed, so
      // what this perturbs is below anything being read.
      darcy.GetPotentialMassForm()->AddDomainIntegrator(
         new VectorBlockIntegrator(np, np, nv, nv, new MassIntegrator(eps)));
   }

   // The stabilization. HDGDiffusionIntegrator builds tau = td Q / h, so with
   // Q = 1 the penalty is td/h; passing td * h^(ts+1) makes it td * h^ts, and
   // ts is NPC's exponent in nu*tau = h^ts. Their section 4.1 sweeps ts and
   // finds the order of unity best:
   //
   //     ts = -1   velocity k+1, pressure and gradient only k
   //     ts =  0   everything k+1                     <- their recommendation
   //     ts = +1   velocity only k, pressure and gradient k+1
   //
   // with the postprocessed velocity at k+2 for k >= 1 in the last two.
   const real_t h = mesh.GetElementSize(0);
   const real_t td_h = td * std::pow(h, ts + 1);
   std::vector<BilinearFormIntegrator *> stab(nv), bstab(nv);
   for (int i = 0; i < nv; i++)
   {
      stab[i]  = new HDGDiffusionIntegrator(one, td_h);
      bstab[i] = new HDGDiffusionIntegrator(one, td_h);
   }
   if (!with_p)
   {
      darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
         new VectorBlockDiagonalIntegrator(stab));
      darcy.GetPotentialMassForm()->AddBdrFaceIntegrator(
         new VectorBlockDiagonalIntegrator(bstab), bdr_ess);
   }
   else
   {
      // Only the velocities are stabilized, and there are more potential
      // fields than trace fields, which is what the rectangular wrapper is
      // for.
      darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
         new VectorBlockDiagonalHDGIntegrator(np, nv, stab));
      for (auto *i : bstab) { delete i; }
   }

   VectorFunctionCoefficient gcoeff(np, gFun), wcoeff(np, wFun);
   VectorFunctionCoefficient natcoeff(nv, [](const Vector &x, Vector &v)
   {
      kov->Velocity(x, v);
      v.Neg();
   });
   VectorFunctionCoefficient ucoeff(nv, uFun);
   VectorFunctionCoefficient qcoeff(nv * dim, qFun);
   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new VectorDomainLFIntegrator(gcoeff));

   Array<int> ess_flux_tdofs;
   darcy.EnableHybridization(
      &fes_t,
      new VectorBlockDiagonalIntegrator(nv, new NormalTraceJumpIntegrator),
      ess_flux_tdofs);
   // Stage 1 puts the velocity datum on an essential trace, which section 7's
   // sweep settled as the default for the discontinuous spaces. Stage 2 cannot
   // yet: with the pressure coupled in, the reduced matrix comes out
   // structurally asymmetric and SparseMatrix::EliminateRowCol refuses it. The
   // weak route -- the datum on the flux equation's natural term -- is correct
   // for solving, which is what is being established here, so it is used until
   // that is chased down.
   if (!with_p) { darcy.GetHybridization()->SetEssentialBC(bdr_ess); }
   else
   {
      darcy.GetFluxRHS()->AddBdrFaceIntegrator(
         new VectorBoundaryFluxLFIntegrator(natcoeff));
   }

   // skip_zeros = 0: with the pressure coupled in, entries of the reduced
   // matrix cancel exactly, and dropping them leaves a sparsity that is no
   // longer symmetric, which the essential-trace elimination requires.
   darcy.Assemble(with_p ? 0 : 1);

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, RHS;
   if (!with_p)
   {
      GridFunction tr0(&fes_t);
      tr0 = 0.0;
      tr0.ProjectBdrCoefficient(ucoeff, bdr_ess);
      X = tr0;
   }
   darcy.FormLinearSystem(ess_flux_tdofs, x, A, X, RHS, true);

   GSSmoother prec;
   GMRESSolver solver;
   solver.SetKDim(500);
   solver.SetMaxIter(5000);
   solver.SetRelTol(1e-13);
   solver.SetAbsTol(0.0);
   solver.SetPreconditioner(prec);
   solver.SetOperator(*A);
   solver.SetPrintLevel(verbose ? 1 : -1);
   solver.Mult(RHS, X);
   MFEM_VERIFY(solver.GetConverged(), "the trace solve did not converge");

   darcy.RecoverFEMSolution(X, x);

   GridFunction q_h(&fes_q, x.GetBlock(0));
   GridFunction u_h(&fes_u, x.GetBlock(1));

   const int quad_order = 2 * order + 4;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      irs[i] = &(IntRules.Get(i, quad_order));
   }

   Result r;
   r.err_p = 0.0;
   if (with_p)
   {
      // The velocity alone, and the pressure alone. All-Dirichlet velocity
      // leaves the pressure determined only up to a constant, so its error is
      // measured after removing the mean of the difference -- which is the
      // usual convention, and the only one that means anything here.
      FiniteElementSpace fes_1(&mesh, &u_coll);
      const int n1 = fes_1.GetVSize();
      GridFunction p_h(&fes_1), p_ex(&fes_1);
      for (int i = 0; i < n1; i++) { p_h(i) = x.GetBlock(1)(nv * n1 + i); }
      FunctionCoefficient pc(pFun);
      p_ex.ProjectCoefficient(pc);
      GridFunction diff(p_h);
      diff -= p_ex;
      ConstantCoefficient zero(0.0);
      LinearForm ones(&fes_1);
      ConstantCoefficient one_c(1.0);
      ones.AddDomainIntegrator(new DomainLFIntegrator(one_c));
      ones.Assemble();
      const real_t area = ones.Sum();
      const real_t shift = (ones * diff) / area;
      for (int i = 0; i < n1; i++) { p_h(i) -= shift; }
      r.err_p = p_h.ComputeL2Error(pc, irs);

      // The stress carries the pressure on its diagonal, so it inherits the
      // same undetermined constant and has to have it removed too.
      const int nq = fes_q.GetVSize() / (nv * dim);
      for (int i = 0; i < nv; i++)
         for (int j = 0; j < nq; j++)
         {
            q_h((i * dim + i) * nq + j) -= shift;
         }

      FiniteElementSpace fes_v(&mesh, &u_coll, nv, Ordering::byNODES);
      GridFunction uh(&fes_v);
      for (int i = 0; i < nv * n1; i++) { uh(i) = x.GetBlock(1)(i); }
      r.err_u = uh.ComputeL2Error(ucoeff, irs);
   }
   else
   {
      r.err_u = u_h.ComputeL2Error(wcoeff, irs);
   }
   r.err_q = q_h.ComputeL2Error(qcoeff, irs);
   r.trace_size = X.Size();
   r.iters = solver.GetNumIterations();
   MFEM_CONTRACT_VAR(stage);
   return r;
}

/// Read the stage 2 blocks against the exact solution, row by row.
///
/// The claim being tested is arithmetic, not asymptotic. B's pressure row
/// collects nu^-1 sum_i (sigma_i)_i, and since (sigma_i)_i = -nu d_i u_i + p
/// that is (-div u + (d/nu) p, w); Kovasznay is divergence free, so applied to
/// the exact solution it must equal ((d/nu) p, w), which is what the
/// cancelling potential mass block also produces. If the two agree, the
/// cancellation is available and only the sign with which the potential
/// equation combines them is in question -- read off directly rather than
/// guessed.
///
/// The momentum rows are checked the same way: B's row i applied to the exact
/// stress must be (f_i, w_i), the momentum source.
void Diagnose(Mesh &mesh, int order)
{
   const int dim = mesh.Dimension();
   const int nv = dim, np = nv + 1;
   const real_t nu = kov->Viscosity();

   L2_FECollection q_coll(order, dim), u_coll(order, dim);
   FiniteElementSpace fes_q(&mesh, &q_coll, nv * dim, Ordering::byNODES);
   FiniteElementSpace fes_u(&mesh, &u_coll, np, Ordering::byNODES);
   FiniteElementSpace fes_1(&mesh, &u_coll);

   ConstantCoefficient inu(1.0 / nu), dnu(real_t(dim) / nu);

   // B's volume blocks only: its faces never touch the pressure row.
   MixedBilinearForm Bd(&fes_q, &fes_u);
   for (int i = 0; i < nv; i++)
   {
      Bd.AddDomainIntegrator(
         new VectorBlockIntegrator(np, nv * dim, i, i * dim,
                                   new VectorDivergenceIntegrator));
      Bd.AddDomainIntegrator(
         new VectorBlockIntegrator(np, nv * dim, nv, i * dim + i,
                                   new MassIntegrator(inu)));
   }
   Bd.Assemble();
   Bd.Finalize();

   BilinearForm Md(&fes_u);
   Md.AddDomainIntegrator(
      new VectorBlockIntegrator(np, np, nv, nv, new MassIntegrator(dnu)));
   Md.Assemble();
   Md.Finalize();

   VectorFunctionCoefficient qc(nv * dim, qFun), wc(np, wFun);
   GridFunction sig(&fes_q), w(&fes_u);
   sig.ProjectCoefficient(qc);
   w.ProjectCoefficient(wc);

   Vector bs(fes_u.GetVSize()), mw(fes_u.GetVSize());
   Bd.Mult(sig, bs);
   Md.Mult(w, mw);

   const int n1 = fes_1.GetVSize();
   Vector fref(fes_u.GetVSize());
   fref = 0.0;
   for (int i = 0; i < nv; i++)
   {
      const int comp = i;
      FunctionCoefficient fi([comp](const Vector &x)
      {
         Vector f;
         kov->Momentum(x, f);
         return f(comp);
      });
      LinearForm lf(&fes_1);
      lf.AddDomainIntegrator(new DomainLFIntegrator(fi));
      lf.Assemble();
      for (int j = 0; j < n1; j++) { fref(i * n1 + j) = lf.Elem(j); }
   }

   auto bnorm = [&](const Vector &v, int c)
   {
      real_t m = 0.0;
      for (int j = 0; j < n1; j++) { m = std::max(m, std::abs(v(c * n1 + j))); }
      return m;
   };
   auto bdiff = [&](const Vector &a, const Vector &b, int c, real_t sgn)
   {
      real_t m = 0.0;
      for (int j = 0; j < n1; j++)
      { m = std::max(m, std::abs(a(c * n1 + j) + sgn * b(c * n1 + j))); }
      return m;
   };

   cout << "\n  blocks at the exact solution, " << mesh.GetNE()
        << " elements, order " << order << "\n";
   for (int i = 0; i < nv; i++)
   {
      cout << "    momentum " << i << ":  |B sigma| = " << bnorm(bs, i)
           << "   |(f,w)| = " << bnorm(fref, i)
           << "   |B sigma - (f,w)| = " << bdiff(bs, fref, i, -1.0) << "\n";
   }
   cout << "    pressure  :  |B sigma| = " << bnorm(bs, nv)
        << "   |M_p w| = " << bnorm(mw, nv)
        << "   |B sigma - M_p w| = " << bdiff(bs, mw, nv, -1.0)
        << "   |B sigma + M_p w| = " << bdiff(bs, mw, nv, +1.0) << "\n";
}

int main(int argc, char *argv[])
{
   int nx = 8, ny = 8, order = 1, ref_levels = 0, stage_i = 1, ts = 0;
   real_t nu = 0.1, td = 1.0;
   bool verbose = false, visualization = false, diagnose = false;

   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--ncells-x", "Number of cells in x.");
   args.AddOption(&ny, "-ny", "--ncells-y", "Number of cells in y.");
   args.AddOption(&ref_levels, "-r", "--ref-levels",
                  "Uniform refinements, run as a convergence study.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&nu, "-nu", "--viscosity", "Viscosity; Re = 1/nu.");
   args.AddOption(&td, "-td", "--stab",
                  "Stabilization constant; the penalty is td * h^ts.");
   args.AddOption(&ts, "-ts", "--stab-scaling",
                  "Stabilization exponent: penalty = td * h^ts (0 = best).");
   args.AddOption(&stage_i, "-c", "--stage",
                  "How much of the operator to assemble (1 = momentum only).");
   args.AddOption(&verbose, "-v", "--verbose", "-no-v", "--no-verbose",
                  "Print the solver history.");
   args.AddOption(&diagnose, "-diag", "--diagnose", "-no-diag", "--no-diagnose",
                  "Read the stage 2 blocks against the exact solution.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Send the solution to GLVis (not implemented yet).");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   MFEM_VERIFY(!visualization, "visualization is not implemented yet");

   Kovasznay k(nu);
   kov = &k;
   kov_stage = static_cast<Stage>(stage_i);

   cout << "lambda = " << k.Lambda() << "\n";
   const real_t data_err = CheckData(k);
   cout << "manufactured data self-check: " << data_err << "\n";
   MFEM_VERIFY(data_err < 1e-4, "the manufactured data is inconsistent");

   // The domain of NPC section 4, so the numbers are comparable.
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, false,
                                     2.0, 2.0);
   for (int i = 0; i < mesh.GetNV(); i++) { mesh.GetVertex(i)[1] -= 0.5; }

   if (diagnose)
   {
      for (int l = 0; l <= ref_levels; l++)
      {
         Diagnose(mesh, order);
         if (l < ref_levels) { mesh.UniformRefinement(); }
      }
      return 0;
   }

   cout << "\n  n      dim(M)     ||u-u_h||    rate     ||q-q_h||    rate"
        << ((kov_stage >= Stage::Stokes) ? "     ||p-p_h||    rate" : "")
        << "\n";
   real_t prev_u = -1.0, prev_q = -1.0, prev_p = -1.0;
   for (int l = 0; l <= ref_levels; l++)
   {
      const Result r = Solve(mesh, order, td, ts, kov_stage, verbose);
      cout << setw(4) << (nx << l) << setw(6) << r.iters << setw(9) << r.trace_size
           << setw(14) << scientific << setprecision(4) << r.err_u;
      if (prev_u > 0.0) { cout << setw(9) << fixed << setprecision(2) << std::log2(prev_u / r.err_u); }
      else { cout << setw(9) << "-"; }
      cout << setw(14) << scientific << setprecision(4) << r.err_q;
      if (prev_q > 0.0) { cout << setw(9) << fixed << setprecision(2) << std::log2(prev_q / r.err_q); }
      else { cout << setw(9) << "-"; }
      if (kov_stage >= Stage::Stokes)
      {
         cout << setw(14) << scientific << setprecision(4) << r.err_p;
         if (prev_p > 0.0)
         { cout << setw(9) << fixed << setprecision(2) << std::log2(prev_p / r.err_p); }
         else { cout << setw(9) << "-"; }
      }
      cout << "\n";
      prev_u = r.err_u;
      prev_q = r.err_q;
      prev_p = r.err_p;
      if (l < ref_levels) { mesh.UniformRefinement(); }
   }
   cout << endl;
   return 0;
}
