// Fractional diffusion solver spectral test.
//
// This miniapp verifies FractionalDiffusionSolver for the pure diffusion
// operator L = M_0^{-1}K, where K is the stiffness matrix and M_0 is the
// standard finite element mass/Riesz map.  The input to Mult() is an assembled
// RHS true vector b.  For a manufactured primal field
//
//     f(x) = sum_{q=1}^{Q} prod_i sin(q pi x_i),
//
// the test assembles b = M_0 f and compares the computed solution with
//
//     L^{-s} f = sum_{q=1}^{Q} (dim q^2 pi^2)^{-s}
//                prod_i sin(q pi x_i).
//
// This is the fractional diffusion case; the mass matrix is the identity/Riesz
// map for the generalized eigenproblem and must not be set to zero.

#include "diffusion_mass_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

using namespace std;
using namespace mfem;

class SineSpectrumCoefficient : public Coefficient
{
public:
   SineSpectrumCoefficient(int modes, real_t fractional_power,
                           bool fractional_exact)
      : modes_(modes),
        fractional_power_(fractional_power),
        fractional_exact_(fractional_exact)
   {
      MFEM_VERIFY(modes_ >= 1, "Expected at least one sine mode.");
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      const real_t pi = 4.0*std::atan(1.0);
      real_t value = 0.0;
      for (int q = 1; q <= modes_; q++)
      {
         real_t mode_value = 1.0;
         for (int d = 0; d < x.Size(); d++)
         {
            mode_value *= std::sin(q*pi*x(d));
         }
         if (fractional_exact_)
         {
            const real_t lambda = x.Size()*q*q*pi*pi;
            mode_value *= std::pow(lambda, -fractional_power_);
         }
         value += mode_value;
      }
      return value;
   }

private:
   int modes_;
   real_t fractional_power_;
   bool fractional_exact_;
};

static int GlobalMax(MPI_Comm comm, int value)
{
   int global = 0;
   MPI_Allreduce(&value, &global, 1, MPI_INT, MPI_MAX, comm);
   return global;
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   const char *mesh_file = "";
   int dim = 2;
   int order = 2;
   int nx = 16;
   int ny = 16;
   int nz = 8;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int print_level = -1;
   int spectrum_modes = 6;
   bool paraview = true;
   real_t fractional_power = 0.5;
   real_t quadrature_spacing = 0.25;
   real_t quadrature_scaling = 1.0;
   int negative_points = 24;
   int positive_points = 24;
   bool estimate_spectrum = false;
   bool use_estimated_scaling = false;
   bool adaptive_quadrature = false;
   real_t adaptive_rel_tol = 1.0e-8;
   real_t adaptive_abs_tol = 0.0;
   int adaptive_max_negative_points = 400;
   int adaptive_max_positive_points = 400;
   int adaptive_consecutive_terms = 3;
   int power_iterations = 20;
   int inverse_power_iterations = 20;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. Empty string creates a Cartesian mesh.");
   args.AddOption(&dim, "-dim", "--dimension",
                  "Problem dimension for generated Cartesian meshes.");
   args.AddOption(&order, "-o", "--order",
                  "H1 finite element order.");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in x.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in y.");
   args.AddOption(&nz, "-nz", "--num-elements-z",
                  "Number of elements in z.");
   args.AddOption(&ser_ref_levels, "-srl", "--ser-ref-levels",
                  "Number of serial uniform refinements.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of parallel uniform refinements.");
   args.AddOption(&fractional_power, "-s", "--fractional-power",
                  "Fractional exponent s in L^{-s}; requires 0 < s < 1.");
   args.AddOption(&quadrature_spacing, "-k", "--quadrature-spacing",
                  "Exponential/sinc quadrature spacing k.");
   args.AddOption(&quadrature_scaling, "-qs", "--quadrature-scaling",
                  "Positive exponential scaling sigma in t=sigma*exp(y).");
   args.AddOption(&negative_points, "-mquad", "--negative-points",
                  "Number of negative quadrature indices m.");
   args.AddOption(&positive_points, "-nquad", "--positive-points",
                  "Number of positive quadrature indices n.");
   args.AddOption(&estimate_spectrum, "-es", "--estimate-spectrum",
                  "-no-es", "--no-estimate-spectrum",
                  "Estimate min/max eigenvalues of L=M_0^{-1}K.");
   args.AddOption(&use_estimated_scaling, "-ues", "--use-estimated-scaling",
                  "-no-ues", "--no-use-estimated-scaling",
                  "Use sqrt(lambda_min*lambda_max) as quadrature scaling.");
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
   args.AddOption(&power_iterations, "-pi", "--power-iterations",
                  "Power iterations for lambda_max estimate.");
   args.AddOption(&inverse_power_iterations, "-ipi",
                  "--inverse-power-iterations",
                  "Inverse power iterations for lambda_min estimate.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "Print level for the internal shifted solves.");
   args.AddOption(&spectrum_modes, "-sm", "--spectrum-modes",
                  "Number of sine product modes in the test RHS.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.ParseCheck();

   MFEM_VERIFY(order >= 1, "Expected order >= 1.");
   MFEM_VERIFY(nx > 0 && ny > 0 && nz > 0,
               "Expected positive mesh dimensions.");
   MFEM_VERIFY(fractional_power > 0.0 && fractional_power < 1.0,
               "Expected 0 < fractional power < 1.");
   MFEM_VERIFY(quadrature_spacing > 0.0,
               "Expected positive quadrature spacing.");
   MFEM_VERIFY(quadrature_scaling > 0.0,
               "Expected positive quadrature scaling.");
   MFEM_VERIFY(negative_points >= 0 && positive_points >= 0,
               "Expected nonnegative quadrature truncation counts.");
   MFEM_VERIFY(power_iterations > 0 && inverse_power_iterations > 0,
               "Expected positive eigenvalue iteration counts.");
   MFEM_VERIFY(spectrum_modes >= 1, "Expected at least one sine mode.");
   MFEM_VERIFY(adaptive_rel_tol >= 0.0 && adaptive_abs_tol >= 0.0,
               "Expected nonnegative adaptive tolerances.");
   MFEM_VERIFY(adaptive_rel_tol > 0.0 || adaptive_abs_tol > 0.0,
               "Expected at least one positive adaptive tolerance.");
   MFEM_VERIFY(adaptive_max_negative_points >= 0 &&
               adaptive_max_positive_points >= 0,
               "Expected nonnegative adaptive quadrature caps.");
   MFEM_VERIFY(adaptive_consecutive_terms > 0,
               "Expected positive adaptive consecutive count.");

   Device device(device_config);
   device.Print();

   std::unique_ptr<Mesh> mesh;
   if (std::strlen(mesh_file) > 0)
   {
      mesh.reset(new Mesh(mesh_file, 1, 1));
      dim = mesh->Dimension();
   }
   else
   {
      MFEM_VERIFY(dim == 2 || dim == 3, "Expected dimension 2 or 3.");
      if (dim == 2)
      {
         mesh.reset(new Mesh(Mesh::MakeCartesian2D(
            nx, ny, Element::QUADRILATERAL, true, 1.0, 1.0)));
      }
      else
      {
         mesh.reset(new Mesh(Mesh::MakeCartesian3D(
            nx, ny, nz, Element::HEXAHEDRON, 1.0, 1.0, 1.0, true)));
      }
   }
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   auto pmesh = std::make_shared<ParMesh>(MPI_COMM_WORLD, *mesh);
   mesh->Clear();
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }

   auto fec = std::make_shared<H1_FECollection>(order, dim,
                                                BasisType::GaussLobatto);
   auto fespace = std::make_shared<ParFiniteElementSpace>(pmesh.get(),
                                                          fec.get());

   FractionalDiffusionSolver solver(fespace);
   solver.SetFractionalPower(fractional_power);
   solver.SetQuadrature(quadrature_spacing, negative_points, positive_points);
   solver.SetQuadratureScaling(quadrature_scaling);
   solver.UseAdaptiveQuadrature(adaptive_quadrature);
   solver.SetAdaptiveQuadrature(adaptive_rel_tol, adaptive_abs_tol,
                                adaptive_max_negative_points,
                                adaptive_max_positive_points,
                                adaptive_consecutive_terms);
   solver.GetDiffusionSolver().SetPrintLevel(print_level);

   const int max_bdr_attr = GlobalMax(pmesh->GetComm(),
                                      pmesh->bdr_attributes.Size()
                                      ? pmesh->bdr_attributes.Max() : 0);
   for (int attr = 1; attr <= max_bdr_attr; attr++)
   {
      solver.GetDiffusionSolver().AddBoundaryID(attr);
   }

   real_t lambda_min_est = 0.0;
   real_t lambda_max_est = 0.0;
   real_t suggested_scaling = 0.0;
   if (estimate_spectrum || use_estimated_scaling)
   {
      solver.EstimateEigenvalueBounds(power_iterations,
                                      inverse_power_iterations,
                                      lambda_min_est,
                                      lambda_max_est,
                                      suggested_scaling);
      if (use_estimated_scaling)
      {
         solver.SetQuadratureScaling(suggested_scaling);
         quadrature_scaling = suggested_scaling;
      }
   }

   SineSpectrumCoefficient input_coeff(spectrum_modes, fractional_power, false);
   SineSpectrumCoefficient exact_coeff(spectrum_modes, fractional_power, true);
   ParGridFunction input_gf(fespace.get());
   input_gf.ProjectCoefficient(input_coeff);
   Vector input_true;
   input_gf.GetTrueDofs(input_true);

   Vector rhs_true;
   solver.GetMassMap().SetWeightCoefficient(1.0);
   solver.GetMassMap().Mult(input_true, rhs_true);

   Vector output_true;
   solver.Mult(rhs_true, output_true);

   ParGridFunction exact_gf(fespace.get());
   exact_gf.ProjectCoefficient(exact_coeff);
   Vector exact_true;
   exact_gf.GetTrueDofs(exact_true);

   Vector error_true(exact_true.Size());
   add(output_true, -1.0, exact_true, error_true);

   const real_t error_norm =
      std::sqrt(InnerProduct(fespace->GetComm(), error_true, error_true));
   const real_t exact_norm =
      std::sqrt(InnerProduct(fespace->GetComm(), exact_true, exact_true));
   const real_t relative_error = error_norm/exact_norm;

   if (Mpi::Root())
   {
      cout << "Fractional diffusion solver spectral test\n"
           << "  dim=" << dim
           << " order=" << order
           << " true_size=" << fespace->GlobalTrueVSize()
           << " spectrum_modes=" << spectrum_modes
           << " s=" << fractional_power
           << " k=" << quadrature_spacing
           << " sigma=" << quadrature_scaling
           << " adaptive=" << (adaptive_quadrature ? "true" : "false")
           << " m=" << (adaptive_quadrature ?
                        solver.GetLastNegativeQuadraturePoints() :
                        negative_points)
           << " n=" << (adaptive_quadrature ?
                        solver.GetLastPositiveQuadraturePoints() :
                        positive_points)
           << '\n';
      if (estimate_spectrum || use_estimated_scaling)
      {
         cout << "  lambda_min_est=" << lambda_min_est
              << " lambda_max_est=" << lambda_max_est
              << " suggested_sigma=" << suggested_scaling << '\n';
      }
      cout
           << "  absolute L2(true-vector) error=" << error_norm << '\n'
           << "  relative L2(true-vector) error=" << relative_error << endl;
   }

   if (paraview)
   {
      ParGridFunction output_gf(fespace.get());
      ParGridFunction error_gf(fespace.get());
      output_gf.SetFromTrueDofs(output_true);
      error_gf.SetFromTrueDofs(error_true);

      ParaViewDataCollection pvdc("FractionalDiffusionSolver", pmesh.get());
      pvdc.SetPrefixPath("ParaView");
      pvdc.RegisterField("input", &input_gf);
      pvdc.RegisterField("fractional_output", &output_gf);
      pvdc.RegisterField("exact", &exact_gf);
      pvdc.RegisterField("error", &error_gf);
      pvdc.SetLevelsOfDetail(std::max(order, 1));
      pvdc.SetDataFormat(VTKFormat::BINARY);
      pvdc.SetHighOrderOutput(true);
      pvdc.SetCycle(0);
      pvdc.SetTime(0.0);
      pvdc.Save();
   }

   return 0;
}
