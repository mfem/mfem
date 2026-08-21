// Compare inverse fractional diffusion+mass solvers with homogeneous Neumann
// boundary conditions on a domain derived from one quadrilateral or triangle.
//
// The manufactured problem is
//
//    A = I - theta*Delta_N,    A^p u = f,    p = 1-s,
//
// where u is a sum of cosine modes with zero normal derivative on every side.

#include "diffusion_mass_solver.hpp"
#include "frac_noise.hpp"
#include "periodic_fraclap_coefficients.hpp"
#include "spde_solver.hpp"

#include <cmath>
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;

#ifndef MFEM_FRACTIONAL_MG_GENERATOR
#define MFEM_FRACTIONAL_MG_GENERATOR FracRandomFieldGenerator
#define MFEM_FRACTIONAL_MG_DESCRIPTION "additive MG"
#define MFEM_FRACTIONAL_MG_FIELD "additive_mg"
#define MFEM_FRACTIONAL_OUTPUT_NAME \
   "fractional_diffusion_mass_compare_neumann"
#endif

static Mesh MakeSingleTriangleMesh()
{
   Mesh mesh(2, 3, 1, 3, 2);
   const real_t vertices[3][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
   for (int i = 0; i < 3; i++) { mesh.AddVertex(vertices[i]); }
   mesh.AddTriangle(0, 1, 2, 1);
   mesh.AddBdrSegment(0, 1, 1);
   mesh.AddBdrSegment(1, 2, 2);
   mesh.AddBdrSegment(2, 0, 3);
   mesh.FinalizeTriMesh(1, 1, true);
   return mesh;
}

static real_t NeumannExact(const Vector &X, int mode_x, int mode_y,
                           bool triangular)
{
   using periodic_fraclap::pi;

   const real_t kx = mode_x*pi;
   const real_t ky = mode_y*pi;
   if (triangular)
   {
      return std::cos(kx*X(0))*std::cos(kx*X(1));
   }
   return std::cos(kx*X(0))
        + 2.0*std::cos(ky*X(1))
        + 0.5*std::cos(kx*X(0))*std::cos(ky*X(1));
}

static real_t ShiftedNeumannRHS(const Vector &X, int mode_x, int mode_y,
                                real_t exponent, real_t diffusion_scale,
                                bool triangular)
{
   using periodic_fraclap::pi;

   const real_t x = X(0);
   const real_t y = X(1);
   const real_t kx = mode_x*pi;
   const real_t ky = mode_y*pi;
   const real_t lambda_x = kx*kx;
   const real_t lambda_y = ky*ky;
   const real_t lambda_xy = lambda_x + lambda_y;

   if (triangular)
   {
      return std::pow(1.0 + diffusion_scale*2.0*lambda_x, exponent)*
             std::cos(kx*x)*std::cos(kx*y);
   }

   return std::pow(1.0 + diffusion_scale*lambda_x, exponent)*std::cos(kx*x)
        + 2.0*std::pow(1.0 + diffusion_scale*lambda_y, exponent)*
          std::cos(ky*y)
        + 0.5*std::pow(1.0 + diffusion_scale*lambda_xy, exponent)*
          std::cos(kx*x)*std::cos(ky*y);
}

static real_t L2Norm(ParGridFunction &x)
{
   ConstantCoefficient zero(0.0);
   return x.ComputeL2Error(zero);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   int order = 2;
   int mode_x = 1;
   int mode_y = 2;
   int par_ref_levels = 4;
   int ser_ref_levels = 0;
   int print_level = -1;
   bool paraview = true;
   bool triangular = false;
   bool use_spde = true;
   bool use_balakrishnan = true;
   bool white_noise_rhs = true;
   int white_noise_seed = 5651;
   real_t s = 0.1;
   real_t corr_len = -1.0;
   real_t quadrature_spacing = 0.25;
   real_t quadrature_scaling = 1.0;
   int negative_points = 32;
   int positive_points = 32;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&mode_x, "-mx", "--mode-x",
                  "Positive x-mode, also used in both directions on a triangle.");
   args.AddOption(&mode_y, "-my", "--mode-y",
                  "Positive y-mode number on the quadrilateral domain.");
   args.AddOption(&triangular, "-tri", "--triangular",
                  "-quad", "--quadrilateral",
                  "Use one triangle or one quadrilateral as the coarse mesh.");
   args.AddOption(&use_spde, "-spde", "--spde",
                  "-no-spde", "--no-spde",
                  "Enable or disable the standalone SPDE comparison solver.");
   args.AddOption(&use_balakrishnan, "-bal", "--balakrishnan",
                  "-no-bal", "--no-balakrishnan",
                  "Enable or disable the Balakrishnan/direct reference solver.");
   args.AddOption(&white_noise_rhs, "-wn", "--white-noise",
                  "-no-wn", "--no-white-noise",
                  "Use a white Gaussian noise RHS instead of the MMS RHS.");
   args.AddOption(&white_noise_seed, "-seed", "--white-noise-seed",
                  "Positive base seed for the white Gaussian noise RHS.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of parallel refinements from the single cell.");
   args.AddOption(&ser_ref_levels, "-srl", "--ser-ref-levels",
                  "Number of serial refinements before partitioning.");
   args.AddOption(&s, "-s", "--s",
                  "MG parameter; the inverse fractional exponent is p=1-s.");
   args.AddOption(&corr_len, "-l", "--correlation-length",
                  "Isotropic SPDE correlation length; the default selects "
                  "unit diffusion scaling.");
   args.AddOption(&quadrature_spacing, "-k", "--quadrature-spacing",
                  "Balakrishnan sinc quadrature spacing.");
   args.AddOption(&quadrature_scaling, "-qs", "--quadrature-scaling",
                  "Balakrishnan scaling in t=sigma*exp(y).");
   args.AddOption(&negative_points, "-mquad", "--negative-points",
                  "Number of negative quadrature indices.");
   args.AddOption(&positive_points, "-nquad", "--positive-points",
                  "Number of positive quadrature indices.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "Print level for internal linear solvers.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.ParseCheck();

   MFEM_VERIFY(order >= 1, "Expected finite element order >= 1.");
   MFEM_VERIFY(mode_x >= 1 && mode_y >= 1,
               "Expected positive cosine mode numbers.");
   MFEM_VERIFY(par_ref_levels >= 0 && ser_ref_levels >= 0,
               "Expected nonnegative refinement counts.");
   MFEM_VERIFY(s >= 0.0 && s < 0.5,
               "Expected s in [0,0.5), giving 0<p<=1 and nu>0.");
   MFEM_VERIFY(corr_len == -1.0 || corr_len > 0.0,
               "Expected a positive correlation length.");
   MFEM_VERIFY(quadrature_spacing > 0.0 && quadrature_scaling > 0.0,
               "Expected positive quadrature spacing and scaling.");
   MFEM_VERIFY(negative_points >= 0 && positive_points >= 0,
               "Expected nonnegative quadrature truncation counts.");
   MFEM_VERIFY(white_noise_seed > 0,
               "Expected a positive white-noise seed.");

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh mesh = triangular ?
      MakeSingleTriangleMesh() :
      Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL,
                            false, 1.0, 1.0, false);
   for (int l = 0; l < ser_ref_levels; l++) { mesh.UniformRefinement(); }

   // Keep the mesh alive until all finite element objects have been destroyed.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   const real_t exponent = 1.0 - s;
   const real_t nu = 2.0*exponent - 1.0;
   if (corr_len < 0.0) { corr_len = std::sqrt(2.0*nu); }
   const real_t diffusion_scale = corr_len*corr_len/(2.0*nu);
   const real_t mg_sigma = 1.0/std::sqrt(diffusion_scale);

   // The MG hierarchy has no essential true dofs, hence natural homogeneous
   // Neumann boundary conditions.
   MFEM_FRACTIONAL_MG_GENERATOR mg_solver(pmesh, par_ref_levels, order,
                                          mg_sigma, s);
   ParFiniteElementSpace &fes = mg_solver.GetFinestFESpace();

   FunctionCoefficient exact_coeff(
      [mode_x, mode_y, triangular](const Vector &X)
      {
         return NeumannExact(X, mode_x, mode_y, triangular);
      });
   FunctionCoefficient rhs_coeff(
      [mode_x, mode_y, exponent, diffusion_scale,
       triangular](const Vector &X)
      {
         return ShiftedNeumannRHS(X, mode_x, mode_y, exponent,
                                  diffusion_scale, triangular);
      });

   ParGridFunction exact(&fes);
   exact.ProjectCoefficient(exact_coeff);

   ParLinearForm rhs_lf(&fes);
   if (white_noise_rhs)
   {
      rhs_lf.AddDomainIntegrator(new WhiteGaussianNoiseDomainLFIntegrator(
                                    pmesh.GetComm(), white_noise_seed));
   }
   else
   {
      rhs_lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   }
   rhs_lf.Assemble();
   unique_ptr<HypreParVector> rhs(rhs_lf.ParallelAssemble());
   rhs->UseDevice(true);

   Vector bal_true;
   ParGridFunction bal_sol(&fes);
   bal_sol = 0.0;
   if (use_balakrishnan && s == 0.0)
   {
      // The Balakrishnan integral requires 0 < exponent < 1.  At the
      // endpoint exponent=1, apply (diffusion_scale*K + M)^{-1} directly.
      DiffusionMassSolver endpoint_solver(fes);
      endpoint_solver.SetDiffusionCoefficient(diffusion_scale);
      endpoint_solver.SetMassCoefficient(1.0);
      endpoint_solver.SetPrintLevel(print_level);
      endpoint_solver.Mult(*rhs, bal_true);
   }
   else if (use_balakrishnan)
   {
      BalakrishnanFractionalSolver bal_solver(fes);
      bal_solver.SetFractionalPower(exponent);
      bal_solver.SetQuadrature(quadrature_spacing, negative_points,
                               positive_points);
      bal_solver.SetQuadratureScaling(quadrature_scaling);
      bal_solver.SetOperatorMassShift(1.0);
      bal_solver.GetDiffusionMassSolver().SetDiffusionCoefficient(
         diffusion_scale);
      bal_solver.GetDiffusionMassSolver().SetMassCoefficient(1.0);
      bal_solver.GetDiffusionMassSolver().SetPrintLevel(print_level);
      bal_solver.Mult(*rhs, bal_true);
   }
   if (use_balakrishnan) { bal_sol.SetFromTrueDofs(bal_true); }

   ParGridFunction spde_sol(&fes);
   spde_sol = 0.0;
   if (use_spde)
   {
      // An empty Boundary map selects natural homogeneous Neumann conditions.
      spde::Boundary neumann_bc;
      spde::SPDESolver spde_solver(nu, neumann_bc, &fes,
                                   corr_len, corr_len, 1.0);
      spde_solver.SetPrintLevel(print_level);
      spde_solver.Solve(rhs_lf, spde_sol);
   }

   Vector mg_true;
   mg_solver.Mult(*rhs, mg_true);
   ParGridFunction mg_sol(&fes);
   mg_sol.SetFromTrueDofs(mg_true);
   // I + theta*K = theta*(K + theta^{-1}*I), whereas the MG generator
   // approximates the inverse power of K + mg_sigma^2*I.
   mg_sol *= std::pow(diffusion_scale, -exponent);

   ParGridFunction bal_minus_spde(&fes);
   bal_minus_spde = 0.0;
   if (use_balakrishnan && use_spde)
   {
      bal_minus_spde = bal_sol;
      bal_minus_spde -= spde_sol;
   }
   ParGridFunction bal_minus_mg(&fes);
   bal_minus_mg = 0.0;
   if (use_balakrishnan)
   {
      bal_minus_mg = bal_sol;
      bal_minus_mg -= mg_sol;
   }
   ParGridFunction spde_minus_mg(&fes);
   spde_minus_mg = 0.0;
   if (use_spde)
   {
      spde_minus_mg = spde_sol;
      spde_minus_mg -= mg_sol;
   }

   const real_t exact_l2 = triangular ?
      1.0/(2.0*std::sqrt(2.0)) : std::sqrt(41.0)/4.0;
   const real_t bal_error = use_balakrishnan && !white_noise_rhs ?
      bal_sol.ComputeL2Error(exact_coeff) : 0.0;
   const real_t spde_error = use_spde && !white_noise_rhs ?
      spde_sol.ComputeL2Error(exact_coeff) : 0.0;
   const real_t mg_error = white_noise_rhs ? 0.0 :
      mg_sol.ComputeL2Error(exact_coeff);
   const real_t spde_l2 = use_spde ? L2Norm(spde_sol) : 0.0;
   const real_t spde_mg_l2 = use_spde ? L2Norm(spde_minus_mg) : 0.0;

   // Minimize ||c*mg_sol-reference||_L2 over the scalar c. For an MMS the
   // reference is the exact coefficient; for white noise it is the standalone
   // SPDE solution. Recover the cross term from the three available norms:
   //   2 (mg,reference) = ||mg||^2 + ||reference||^2 - ||mg-reference||^2.
   const real_t mg_l2 = L2Norm(mg_sol);
   MFEM_VERIFY(mg_l2 > 0.0, "Cannot scale a zero MG solution.");
   real_t mg_best_scale = 1.0;
   if (white_noise_rhs && use_spde)
   {
      const real_t mg_spde_inner =
         0.5*(mg_l2*mg_l2 + spde_l2*spde_l2 - spde_mg_l2*spde_mg_l2);
      mg_best_scale = mg_spde_inner/(mg_l2*mg_l2);
   }
   else if (!white_noise_rhs)
   {
      const real_t mg_exact_inner =
         0.5*(mg_l2*mg_l2 + exact_l2*exact_l2 - mg_error*mg_error);
      mg_best_scale = mg_exact_inner/(mg_l2*mg_l2);
   }

   ParGridFunction scaled_mg_sol(&fes);
   scaled_mg_sol = mg_sol;
   scaled_mg_sol *= mg_best_scale;
   const real_t scaled_mg_error = white_noise_rhs ? 0.0 :
      scaled_mg_sol.ComputeL2Error(exact_coeff);
   ParGridFunction scaled_mg_minus_spde(&fes);
   scaled_mg_minus_spde = 0.0;
   if (white_noise_rhs && use_spde)
   {
      scaled_mg_minus_spde = scaled_mg_sol;
      scaled_mg_minus_spde -= spde_sol;
   }
   const real_t scaled_mg_spde_l2 = white_noise_rhs && use_spde ?
      L2Norm(scaled_mg_minus_spde) : 0.0;

   if (Mpi::Root())
   {
      const char *bal_method = s == 0.0 ?
         "direct diffusion+mass" : "Balakrishnan";
      cout << "Fractional diffusion+mass comparison on a single-cell-derived "
           << (triangular ? "reference-triangle" : "unit-square")
           << " mesh with homogeneous Neumann BC\n"
           << "  order=" << order;
      if (triangular)
      {
         cout << " triangle_mode=" << mode_x;
      }
      else
      {
         cout << " mode_x=" << mode_x << " mode_y=" << mode_y;
      }
      cout << " serial_refinements=" << ser_ref_levels
           << " parallel_refinements=" << par_ref_levels
           << " true_size=" << fes.GlobalTrueVSize() << '\n'
           << "  exponent=" << exponent << " s=" << s << " nu=" << nu
           << " correlation_length=" << corr_len
           << " diffusion_scale=" << diffusion_scale
           << " mg_sigma=" << mg_sigma << '\n'
           << "  standalone SPDE enabled=" << use_spde
           << " reference enabled=" << use_balakrishnan
           << " white_noise_rhs=" << white_noise_rhs;
      if (white_noise_rhs) { cout << " seed=" << white_noise_seed; }
      cout << '\n';
      if (use_balakrishnan)
      {
         cout << "  reference fractional method=" << bal_method << '\n';
      }
      if (white_noise_rhs)
      {
         cout << "White-noise solution norms:\n";
         if (use_balakrishnan)
         {
            cout << "  ||reference||_L2 = " << L2Norm(bal_sol) << '\n';
         }
         if (use_spde)
         {
            cout << "  ||SPDE||_L2 = " << spde_l2 << '\n';
         }
         cout << "  ||" << MFEM_FRACTIONAL_MG_DESCRIPTION
              << "||_L2 = " << mg_l2 << '\n'
              << "Pairwise differences on the shared finest space:\n";
         if (use_balakrishnan && use_spde)
         {
            cout << "  |reference - SPDE|_L2 = "
                 << L2Norm(bal_minus_spde) << '\n';
         }
         if (use_balakrishnan)
         {
            cout << "  |reference - MG|_L2   = "
                 << L2Norm(bal_minus_mg) << '\n';
         }
         if (use_spde)
         {
            cout << "  |SPDE - MG|_L2        = "
                 << spde_mg_l2 << '\n'
                 << "  optimal MG-to-SPDE scale = " << mg_best_scale << '\n'
                 << "  |SPDE - scaled MG|_L2 = "
                 << scaled_mg_spde_l2 << '\n';
         }
      }
      else
      {
         cout << "Errors against the analytical solution:\n";
         if (use_balakrishnan)
         {
            cout << "  reference absolute L2     = " << bal_error << '\n'
                 << "  reference relative L2     = "
                 << bal_error/exact_l2 << '\n';
         }
         if (use_spde)
         {
            cout << "  SPDE absolute L2          = " << spde_error << '\n'
                 << "  SPDE relative L2          = "
                 << spde_error/exact_l2 << '\n';
         }
         cout << "  " << MFEM_FRACTIONAL_MG_DESCRIPTION
              << " absolute L2 = " << mg_error << '\n'
              << "  " << MFEM_FRACTIONAL_MG_DESCRIPTION
              << " relative L2 = " << mg_error/exact_l2 << '\n'
              << "  optimal MG scaling factor = " << mg_best_scale << '\n'
              << "  scaled MG absolute L2     = " << scaled_mg_error << '\n'
              << "  scaled MG relative L2     = "
              << scaled_mg_error/exact_l2 << '\n';
         if (use_balakrishnan)
         {
            cout << "Pairwise differences on the shared finest space:\n";
            if (use_spde)
            {
               cout << "  |reference - SPDE|_L2 = "
                    << L2Norm(bal_minus_spde) << '\n';
            }
            cout << "  |reference - MG|_L2   = "
                 << L2Norm(bal_minus_mg) << '\n';
         }
      }
      cout << flush;
   }

   if (paraview)
   {
      ParaViewDataCollection pvdc(MFEM_FRACTIONAL_OUTPUT_NAME,
                                  fes.GetParMesh());
      pvdc.SetPrefixPath("ParaView");
      pvdc.SetLevelsOfDetail(order);
      pvdc.SetDataFormat(VTKFormat::BINARY);
      pvdc.SetHighOrderOutput(true);
      pvdc.SetCycle(0);
      pvdc.SetTime(0.0);
      if (!white_noise_rhs) { pvdc.RegisterField("exact", &exact); }
      if (use_balakrishnan)
      {
         pvdc.RegisterField("balakrishnan", &bal_sol);
      }
      if (use_spde) { pvdc.RegisterField("spde", &spde_sol); }
      pvdc.RegisterField(MFEM_FRACTIONAL_MG_FIELD, &mg_sol);
      if (!white_noise_rhs || use_spde)
      {
         pvdc.RegisterField("scaled_additive_mg", &scaled_mg_sol);
      }
      if (use_balakrishnan && use_spde)
      {
         pvdc.RegisterField("balakrishnan_minus_spde", &bal_minus_spde);
      }
      if (use_balakrishnan)
      {
         pvdc.RegisterField("balakrishnan_minus_mg", &bal_minus_mg);
      }
      if (use_spde)
      {
         pvdc.RegisterField("spde_minus_mg", &spde_minus_mg);
      }
      if (white_noise_rhs && use_spde)
      {
         pvdc.RegisterField("scaled_mg_minus_spde", &scaled_mg_minus_spde);
      }
      pvdc.Save();
   }

   return EXIT_SUCCESS;
}
