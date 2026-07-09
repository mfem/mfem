// Compare fractional diffusion+mass solvers on a periodic MMS problem.
//
// The manufactured problem is posed on the periodic unit square with
//
//     A = I - Delta,     u = periodic_fraclap::U2D.
//
// For the multilevel generator parameter s, the inverse fractional exponent is
// p = 1 - s.  The RHS is f = A^p u, assembled as a weak RHS vector.  The test
// compares:
//
//   1. BalakrishnanFractionalSolver with operator mass shift beta=1, so each
//      shifted quadrature system is (K + (1+t) M) u_t = b;
//   2. the rational SPDE solver used in the original miniapp;
//   3. additive multilevel generators, with and without the SPDE coarse-grid
//      solve.
//
// All methods are compared to the same exact periodic solution.  The
// Balakrishnan solver is also directly compared with the SPDE and additive
// multilevel results on the shared finest finite element space.

#include "diffusion_mass_solver.hpp"
#include "frac_noise.hpp"
#include "periodic_fraclap_coefficients.hpp"
#include "spde_solver.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
using namespace mfem;

static Mesh MakePeriodicUnitSquareMesh(int nx, int ny)
{
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                     false, 1.0, 1.0, false);
   std::vector<Vector> translations =
   {
      Vector({1.0, 0.0}),
      Vector({0.0, 1.0})
   };
   return Mesh::MakePeriodic(mesh,
                             mesh.CreatePeriodicVertexMapping(translations));
}

static real_t ShiftedFracLapU2D(const Vector &X, real_t exponent)
{
   using periodic_fraclap::pi;

   const real_t x = X(0);
   const real_t y = X(1);
   const real_t t = 2.0*pi;
   const real_t k1 = t;
   const real_t k2 = t*std::sqrt(2.0);

   return std::pow(1.0, exponent)*3.0
        + std::pow(1.0 + k1*k1, exponent)*std::sin(t*x)
        + 2.0*std::pow(1.0 + k1*k1, exponent)*std::cos(t*y)
        + 0.5*std::pow(1.0 + k2*k2, exponent)*std::cos(t*(x + y));
}

static real_t L2Norm(ParGridFunction &x)
{
   ConstantCoefficient zero(0.0);
   return x.ComputeL2Error(zero);
}

static real_t L2Error(ParGridFunction &x, ParGridFunction &exact)
{
   ParGridFunction error(x.ParFESpace());
   error = x;
   error -= exact;
   return L2Norm(error);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   int order = 2;
   int nx = 8;
   int ny = 8;
   int par_ref_levels = 2;
   int ser_ref_levels = 0;
   int smoother_applications = 1;
   int print_level = -1;
   bool paraview = true;
   real_t s = 0.1;
   real_t quadrature_spacing = 0.25;
   real_t quadrature_scaling = 1.0;
   int negative_points = 32;
   int positive_points = 32;
   bool adaptive_quadrature = false;
   real_t adaptive_rel_tol = 1.0e-8;
   real_t adaptive_abs_tol = 0.0;
   int adaptive_max_negative_points = 400;
   int adaptive_max_positive_points = 400;
   int adaptive_consecutive_terms = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order.");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in the x direction.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in the y direction.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of parallel uniform refinements in the hierarchy.");
   args.AddOption(&ser_ref_levels, "-srl", "--ser-ref-levels",
                  "Number of serial uniform refinements before partitioning.");
   args.AddOption(&smoother_applications, "-ns",
                  "--num-smoother-applications",
                  "Number of smoother applications on every MG level.");
   args.AddOption(&s, "-s", "--s",
                  "MG fractional parameter; compares exponent p=1-s.");
   args.AddOption(&quadrature_spacing, "-k", "--quadrature-spacing",
                  "Balakrishnan exponential/sinc quadrature spacing.");
   args.AddOption(&quadrature_scaling, "-qs", "--quadrature-scaling",
                  "Balakrishnan scaling sigma in t=sigma*exp(y).");
   args.AddOption(&negative_points, "-mquad", "--negative-points",
                  "Number of negative Balakrishnan quadrature indices.");
   args.AddOption(&positive_points, "-nquad", "--positive-points",
                  "Number of positive Balakrishnan quadrature indices.");
   args.AddOption(&adaptive_quadrature, "-aq", "--adaptive-quadrature",
                  "-no-aq", "--no-adaptive-quadrature",
                  "Enable adaptive Balakrishnan quadrature truncation.");
   args.AddOption(&adaptive_rel_tol, "-aqrtol",
                  "--adaptive-relative-tolerance",
                  "Relative tolerance for adaptive quadrature tails.");
   args.AddOption(&adaptive_abs_tol, "-aqatol",
                  "--adaptive-absolute-tolerance",
                  "Absolute tolerance for adaptive quadrature tails.");
   args.AddOption(&adaptive_max_negative_points, "-aqm",
                  "--adaptive-max-negative-points",
                  "Maximum negative tail points in adaptive quadrature.");
   args.AddOption(&adaptive_max_positive_points, "-aqn",
                  "--adaptive-max-positive-points",
                  "Maximum positive tail points in adaptive quadrature.");
   args.AddOption(&adaptive_consecutive_terms, "-aqc",
                  "--adaptive-consecutive-terms",
                  "Consecutive small terms needed to stop a tail.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "Print level for internal linear solvers.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.ParseCheck();

   MFEM_VERIFY(nx >= 3 && ny >= 3,
               "Fully periodic quadrilateral meshes require nx, ny >= 3.");
   MFEM_VERIFY(order >= 1, "Expected finite element order >= 1.");
   MFEM_VERIFY(s > 0.0 && s < 0.5,
               "Expected s in (0,0.5); exponent p=1-s must satisfy 0<p<1.");
   MFEM_VERIFY(smoother_applications >= 1,
               "Expected at least one smoother application.");
   MFEM_VERIFY(quadrature_spacing > 0.0,
               "Expected positive quadrature spacing.");
   MFEM_VERIFY(quadrature_scaling > 0.0,
               "Expected positive quadrature scaling.");
   MFEM_VERIFY(negative_points >= 0 && positive_points >= 0,
               "Expected nonnegative quadrature truncation counts.");

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh mesh = MakePeriodicUnitSquareMesh(nx, ny);
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   const int dim = pmesh->Dimension();
   MFEM_VERIFY(dim == 2, "This comparison test is 2D only.");

   const real_t exponent = 1.0 - s;
   const real_t nu = 2.0*exponent - dim/2.0;
   const real_t corr_len = std::sqrt(2.0*nu);

   FracRandomFieldGenerator multilevel(*pmesh, par_ref_levels, order, 1.0, s,
                                       smoother_applications);
   FracRandomFieldGeneratorSPDE multilevel_spde(*pmesh, par_ref_levels,
                                                order, 1.0, s,
                                                smoother_applications);

   ParFiniteElementSpace &fes = multilevel.GetFinestFESpace();

   FunctionCoefficient exact_coeff(periodic_fraclap::U2D);
   FunctionCoefficient rhs_coeff(
      [exponent](const Vector &X) { return ShiftedFracLapU2D(X, exponent); });

   ParGridFunction exact(&fes);
   exact.ProjectCoefficient(exact_coeff);
   const real_t exact_l2 = L2Norm(exact);

   ParLinearForm rhs_lf(&fes);
   rhs_lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   rhs_lf.Assemble();
   unique_ptr<HypreParVector> rhs(rhs_lf.ParallelAssemble());
   rhs->UseDevice(true);

   BalakrishnanFractionalSolver bal_solver(fes);
   bal_solver.SetFractionalPower(exponent);
   bal_solver.SetQuadrature(quadrature_spacing, negative_points,
                            positive_points);
   bal_solver.SetQuadratureScaling(quadrature_scaling);
   bal_solver.SetOperatorMassShift(1.0);
   bal_solver.UseAdaptiveQuadrature(adaptive_quadrature);
   bal_solver.SetAdaptiveQuadrature(adaptive_rel_tol, adaptive_abs_tol,
                                    adaptive_max_negative_points,
                                    adaptive_max_positive_points,
                                    adaptive_consecutive_terms);
   bal_solver.GetDiffusionMassSolver().SetMassCoefficient(1.0);
   bal_solver.GetDiffusionMassSolver().SetPrintLevel(print_level);

   Vector bal_true;
   bal_solver.Mult(*rhs, bal_true);
   ParGridFunction bal_sol(&fes);
   bal_sol.SetFromTrueDofs(bal_true);

   ParLinearForm rhs_spde(&fes);
   rhs_spde.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   rhs_spde.Assemble();
   ParGridFunction spde_sol(&fes);
   spde_sol = 0.0;
   {
      spde::Boundary periodic_bc;
      spde::SPDESolver spde_solver(nu, periodic_bc, &fes,
                                   corr_len, corr_len, 1.0);
      spde_solver.SetPrintLevel(print_level);
      spde_solver.Solve(rhs_spde, spde_sol);
   }

   Vector mg_true(fes.GetTrueVSize());
   multilevel.Mult(*rhs, mg_true);
   ParGridFunction mg_sol(&fes);
   mg_sol.SetFromTrueDofs(mg_true);

   ParFiniteElementSpace &fes_spde = multilevel_spde.GetFinestFESpace();
   ParLinearForm rhs_mg_spde_lf(&fes_spde);
   rhs_mg_spde_lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   rhs_mg_spde_lf.Assemble();
   unique_ptr<HypreParVector> rhs_mg_spde(rhs_mg_spde_lf.ParallelAssemble());
   rhs_mg_spde->UseDevice(true);
   Vector mg_spde_true(fes_spde.GetTrueVSize());
   multilevel_spde.Mult(*rhs_mg_spde, mg_spde_true);
   ParGridFunction mg_spde_sol(&fes_spde);
   mg_spde_sol.SetFromTrueDofs(mg_spde_true);
   ParGridFunction exact_spde(&fes_spde);
   exact_spde.ProjectCoefficient(exact_coeff);

   ParGridFunction bal_minus_spde(&fes);
   bal_minus_spde = bal_sol;
   bal_minus_spde -= spde_sol;
   ParGridFunction bal_minus_mg(&fes);
   bal_minus_mg = bal_sol;
   bal_minus_mg -= mg_sol;

   const real_t bal_l2 = L2Error(bal_sol, exact);
   const real_t spde_l2 = L2Error(spde_sol, exact);
   const real_t mg_l2 = L2Error(mg_sol, exact);
   const real_t mg_spde_l2 = L2Error(mg_spde_sol, exact_spde);
   const real_t bal_spde_l2 = L2Norm(bal_minus_spde);
   const real_t bal_mg_l2 = L2Norm(bal_minus_mg);

   if (Mpi::Root())
   {
      cout << "Fractional diffusion+mass comparison on periodic (0,1)^2\n"
           << "  exponent=" << exponent
           << " s=" << s
           << " nu=" << nu
           << " l=" << corr_len
           << " smoother_applications=" << smoother_applications
           << " k=" << quadrature_spacing
           << " sigma=" << quadrature_scaling
           << " m=" << (adaptive_quadrature ?
                        bal_solver.GetLastNegativeQuadraturePoints() :
                        negative_points)
           << " n=" << (adaptive_quadrature ?
                        bal_solver.GetLastPositiveQuadraturePoints() :
                        positive_points)
           << endl;
      cout << "Errors against exact solution:" << endl;
      cout << "  Balakrishnan diffusion+mass absolute L2 = "
           << bal_l2 << endl;
      cout << "  Balakrishnan diffusion+mass relative L2 = "
           << bal_l2/exact_l2 << endl;
      cout << "  SPDE absolute L2                       = "
           << spde_l2 << endl;
      cout << "  SPDE relative L2                       = "
           << spde_l2/exact_l2 << endl;
      cout << "  additive MG absolute L2                = "
           << mg_l2 << endl;
      cout << "  additive MG relative L2                = "
           << mg_l2/exact_l2 << endl;
      cout << "  additive MG+SPDE coarse absolute L2    = "
           << mg_spde_l2 << endl;
      cout << "  additive MG+SPDE coarse relative L2    = "
           << mg_spde_l2/exact_l2 << endl;
      cout << "Pairwise differences on shared finest space:" << endl;
      cout << "  |Balakrishnan - SPDE|_L2 = "
           << bal_spde_l2 << endl;
      cout << "  |Balakrishnan - additive MG|_L2 = "
           << bal_mg_l2 << endl;
   }

   if (paraview)
   {
      ParaViewDataCollection pvdc("fractional_diffusion_mass_compare",
                                  fes.GetParMesh());
      pvdc.SetPrefixPath("ParaView");
      pvdc.SetLevelsOfDetail(order);
      pvdc.SetDataFormat(VTKFormat::BINARY);
      pvdc.SetHighOrderOutput(true);
      pvdc.SetCycle(0);
      pvdc.SetTime(0.0);
      pvdc.RegisterField("exact", &exact);
      pvdc.RegisterField("balakrishnan", &bal_sol);
      pvdc.RegisterField("spde", &spde_sol);
      pvdc.RegisterField("additive_mg", &mg_sol);
      pvdc.RegisterField("balakrishnan_minus_spde", &bal_minus_spde);
      pvdc.RegisterField("balakrishnan_minus_additive_mg", &bal_minus_mg);
      pvdc.Save();
   }

   delete pmesh;
   return EXIT_SUCCESS;
}
