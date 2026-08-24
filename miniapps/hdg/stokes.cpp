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
//               What is built so far is recorded in Stage below.
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
   void Flux(const Vector &x, Vector &q) const
   {
      VelocityGrad(x, q);
      q *= -nu;
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
      Vector l;
      VelocityLaplacian(x, l);
      g.SetSize(2);
      for (int i = 0; i < 2; i++) { g(i) = nu * l(i); }
      MFEM_CONTRACT_VAR(stage);
   }
};

// A single instance the coefficient lambdas close over.
static const Kovasznay *kov = NULL;
static Stage kov_stage = Stage::Momentum;

static void uFun(const Vector &x, Vector &u) { kov->Velocity(x, u); }
static void qFun(const Vector &x, Vector &q) { kov->Flux(x, q); }
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
   real_t err_u, err_q;
   int    trace_size, iters;
};

Result Solve(Mesh &mesh, int order, real_t td, int ts, Stage stage,
             bool verbose)
{
   const int dim = mesh.Dimension();
   const int neq = dim;             // one Darcy problem per velocity component

   L2_FECollection q_coll(order, dim), u_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace fes_q(&mesh, &q_coll, neq * dim, Ordering::byNODES);
   FiniteElementSpace fes_u(&mesh, &u_coll, neq, Ordering::byNODES);
   FiniteElementSpace fes_t(&mesh, &t_coll, neq, Ordering::byNODES);

   DarcyForm darcy(&fes_q, &fes_u);

   ConstantCoefficient inu(1.0 / kov->Viscosity()), one(1.0);

   // nu^-1 (q, v), replicated down the components.
   std::vector<BilinearFormIntegrator *> mass(neq);
   for (int i = 0; i < neq; i++) { mass[i] = new VectorMassIntegrator(inu); }
   darcy.GetFluxMassForm()->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(mass));

   Array<int> bdr_ess(mesh.bdr_attributes.Max());
   bdr_ess = 1;

   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(
      new VectorBlockDiagonalIntegrator(neq, new VectorDivergenceIntegrator));
   B->AddInteriorFaceIntegrator(
      new VectorBlockDiagonalIntegrator(
         neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))));
   B->AddBdrFaceIntegrator(
      new VectorBlockDiagonalIntegrator(
         neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-2.))),
      bdr_ess);

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
   std::vector<BilinearFormIntegrator *> stab(neq), bstab(neq);
   for (int i = 0; i < neq; i++)
   {
      stab[i]  = new HDGDiffusionIntegrator(one, td_h);
      bstab[i] = new HDGDiffusionIntegrator(one, td_h);
   }
   darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
      new VectorBlockDiagonalIntegrator(stab));
   darcy.GetPotentialMassForm()->AddBdrFaceIntegrator(
      new VectorBlockDiagonalIntegrator(bstab), bdr_ess);

   VectorFunctionCoefficient gcoeff(neq, gFun), ucoeff(neq, uFun);
   VectorFunctionCoefficient qcoeff(neq * dim, qFun);
   darcy.GetPotentialRHS()->AddDomainIntegrator(
      new VectorDomainLFIntegrator(gcoeff));

   Array<int> ess_flux_tdofs;
   darcy.EnableHybridization(
      &fes_t,
      new VectorBlockDiagonalIntegrator(neq, new NormalTraceJumpIntegrator),
      ess_flux_tdofs);
   darcy.GetHybridization()->SetEssentialBC(bdr_ess);

   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, RHS;
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
   r.err_u = u_h.ComputeL2Error(ucoeff, irs);
   r.err_q = q_h.ComputeL2Error(qcoeff, irs);
   r.trace_size = X.Size();
   r.iters = solver.GetNumIterations();
   MFEM_CONTRACT_VAR(stage);
   return r;
}

int main(int argc, char *argv[])
{
   int nx = 8, ny = 8, order = 1, ref_levels = 0, stage_i = 1, ts = 0;
   real_t nu = 0.1, td = 1.0;
   bool verbose = false, visualization = false;

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

   cout << "\n  n      dim(M)     ||u-u_h||    rate     ||q-q_h||    rate\n";
   real_t prev_u = -1.0, prev_q = -1.0;
   for (int l = 0; l <= ref_levels; l++)
   {
      const Result r = Solve(mesh, order, td, ts, kov_stage, verbose);
      cout << setw(4) << (nx << l) << setw(11) << r.trace_size
           << setw(14) << scientific << setprecision(4) << r.err_u;
      if (prev_u > 0.0) { cout << setw(9) << fixed << setprecision(2) << std::log2(prev_u / r.err_u); }
      else { cout << setw(9) << "-"; }
      cout << setw(14) << scientific << setprecision(4) << r.err_q;
      if (prev_q > 0.0) { cout << setw(9) << fixed << setprecision(2) << std::log2(prev_q / r.err_q); }
      else { cout << setw(9) << "-"; }
      cout << "\n";
      prev_u = r.err_u;
      prev_q = r.err_q;
      if (l < ref_levels) { mesh.UniformRefinement(); }
   }
   cout << endl;
   return 0;
}
