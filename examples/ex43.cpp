//                                MFEM Example 43
//
// Compile with: make ex43
//
// Sample runs:
//    ex43 -e 0 -amr 3
//    ex43 -e 1 -amr 3 -eps-i 0.1 -sigma 0.2
//
// Description: This example solves the complex time-harmonic Maxwell equation
//
//     curl(muinv curl E) - omega^2 epsilon_eff E = f = i omega J_src,
//
// where epsilon_eff = epsilon_r + i (epsilon_i + sigma / omega). The
// e^{-i omega t} convention is used, so epsilon_i and sigma both represent
// passive loss.
//
// with homogeneous tangential-electric boundary conditions. It demonstrates a
// simple adaptive loop using one of two element error indicators:
//
//   0: complex ZZ recovery of curl E,
//   1: ComplexMaxwellResidualEstimator, or
//   2: GeneralErrorEstimator populated with the complex Maxwell residual terms.
//
// The last two choices use the residual indicator of Chaumont-Frelet and Vega.

#include "mfem.hpp"
#include <iostream>
#include <memory>

using namespace mfem;
using namespace std;

namespace
{

real_t omega = 1.0, epsilon_r = 1.0, epsilon_i = 0.1, sigma = 0.1;

// This field has zero tangential trace on the boundary of the unit cube.
void EExactReal(const Vector &x, Vector &e)
{
   e.SetSize(3);
   e = 0.0;
   e(2) = sin(M_PI * x(0)) * sin(M_PI * x(1));
}

void EExactImag(const Vector &x, Vector &e)
{
   e.SetSize(3);
   e = 0.0;
   e(2) = 0.3 * sin(M_PI * x(0)) * sin(M_PI * x(1));
}

void FExactReal(const Vector &x, Vector &f)
{
   f.SetSize(3); f = 0.0;
   const real_t u = sin(M_PI * x(0)) * sin(M_PI * x(1));
   const real_t lambda = 2.0 * M_PI * M_PI;
   const real_t eta = epsilon_i + sigma / omega;
   f(2) = (lambda - omega * omega * epsilon_r +
           0.3 * omega * omega * eta) * u;
}

void FExactImag(const Vector &x, Vector &f)
{
   f.SetSize(3); f = 0.0;
   const real_t u = sin(M_PI * x(0)) * sin(M_PI * x(1));
   const real_t lambda = 2.0 * M_PI * M_PI;
   const real_t eta = epsilon_i + sigma / omega;
   f(2) = (0.3 * (lambda - omega * omega * epsilon_r) -
           omega * omega * eta) * u;
}

void CurlEExact(const Vector &x, Vector &curl_e)
{
   curl_e.SetSize(3);
   curl_e(0) = M_PI * sin(M_PI * x(0)) * cos(M_PI * x(1));
   curl_e(1) = -M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
   curl_e(2) = 0.0;
}

} // namespace

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/inline-hex.mesh";
   int order = 1;
   int estimator_type = 2;
   int amr_iterations = 3;
   real_t fraction = 0.5;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Nedelec polynomial degree.");
   args.AddOption(&omega, "-w", "--omega", "Angular frequency.");
   args.AddOption(&epsilon_r, "-eps-r", "--epsilon-real", "Real permittivity.");
   args.AddOption(&epsilon_i, "-eps-i", "--epsilon-imag", "Dielectric loss.");
   args.AddOption(&sigma, "-sigma", "--conductivity", "Electric conductivity.");
   args.AddOption(&estimator_type, "-e", "--estimator",
                  "Error estimator: 1 = complex Maxwell residual, "
                  "2 = general Maxwell residual.");
   args.AddOption(&amr_iterations, "-amr", "--amr-iterations",
                  "Number of adaptive refinement steps.");
   args.AddOption(&fraction, "-f", "--fraction",
                  "Fraction of the maximum indicator used for refinement.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis output.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   MFEM_VERIFY(order > 0 && omega > 0.0, "order and omega must be positive.");
   MFEM_VERIFY(fraction > 0.0 && fraction < 1.0,
               "refinement fraction must be in (0, 1).");

   Mesh mesh(mesh_file, 1, 1);
   MFEM_VERIFY(mesh.Dimension() == 3 && mesh.SpaceDimension() == 3,
               "Example 42 requires a three-dimensional volume mesh.");

   ND_FECollection nd_fec(order, 3);
   FiniteElementSpace fespace(&mesh, &nd_fec);
   L2_FECollection source_fec(order, 3);
   FiniteElementSpace source_fes(&mesh, &source_fec, 3, Ordering::byVDIM);
   ComplexGridFunction solution(&fespace), source(&source_fes);

   const real_t mu_inv_value = 1.0;
   ConstantCoefficient mu_inv(mu_inv_value), epsilon(epsilon_r),
                       epsilon_loss(epsilon_i + sigma / omega);
   ConstantCoefficient negative_mass(-omega * omega * epsilon_r),
                       loss_mass(-omega * omega * epsilon_i - omega * sigma);
   VectorFunctionCoefficient e_exact_r(3, EExactReal), e_exact_i(3, EExactImag),
                             f_exact_r(3, FExactReal), f_exact_i(3, FExactImag),
                             curl_e_exact(3, CurlEExact);

   ComplexLinearForm b(&fespace, ComplexOperator::HERMITIAN);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_exact_r),
                         new VectorFEDomainLFIntegrator(f_exact_i));
   SesquilinearForm a(&fespace, ComplexOperator::HERMITIAN);
   a.AddDomainIntegrator(new CurlCurlIntegrator(mu_inv), nullptr);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(negative_mass),
                         new VectorFEMassIntegrator(loss_mass));

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   CurlCurlIntegrator zz_integrator(mu_inv);
   ND_FECollection zz_fec(order, 3);
   FiniteElementSpace zz_fes(&mesh, &zz_fec);
   unique_ptr<ErrorEstimator> estimator;
   switch (estimator_type)
   {
      case 0:
         estimator.reset(new ComplexZienkiewiczZhuEstimator(
                            zz_integrator, solution, zz_fes));
         break;
      case 1:
         estimator.reset(new ComplexMaxwellResidualEstimator(
                            solution, source, epsilon, epsilon_loss, mu_inv,
                            omega, order));
         break;
      case 2:
      {
         auto *general = new GeneralErrorEstimator(mesh);
         AddComplexMaxwellResidualEstimators(*general, solution, source, epsilon,
                                             epsilon_loss, mu_inv, omega, order);
         estimator.reset(general);
         break;
      }
      default:
         MFEM_ABORT("unknown estimator type");
   }

   ThresholdRefiner refiner(*estimator);
   refiner.SetTotalErrorFraction(fraction);
   refiner.PreferConformingRefinement();

   socketstream sol_sock_r, sol_sock_i;
   if (visualization)
   {
      sol_sock_r.open("localhost", 19916);
      sol_sock_i.open("localhost", 19916);
   }

   for (int it = 0; it <= amr_iterations; it++)
   {
      cout << "\nAMR iteration " << it << ", unknowns: "
           << fespace.GetTrueVSize() << endl;

      source.ProjectCoefficient(f_exact_r, f_exact_i);
      b.Assemble();
      a.Assemble();

      Array<int> ess_tdof_list;
      // Projecting the complete field is the standard vector-FE way to set
      // its tangential trace; FormLinearSystem retains only essential values.
      solution.ProjectCoefficient(e_exact_r, e_exact_i);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, solution, b, A, X, B);

      // The negative mass term makes the system indefinite; GMRES is used in
      // place of the CG solve used by the definite curl-curl example.
      GMRESSolver solver;
      solver.SetPrintLevel(0);
      solver.SetMaxIter(500);
      solver.SetKDim(50);
      solver.SetRelTol(1e-10);
      solver.SetAbsTol(0.0);
      solver.SetOperator(*A);
      solver.Mult(B, X);
      a.RecoverFEMSolution(X, b, solution);

      estimator->Reset();
      estimator->GetLocalErrors();
      const real_t estimated_error = estimator->GetTotalError();
      const real_t l2_error = hypot(solution.real().ComputeL2Error(e_exact_r),
                                    solution.imag().ComputeL2Error(e_exact_i));
      const real_t curl_error = hypot(solution.real().ComputeCurlError(&curl_e_exact),
                                      0.3 * solution.imag().ComputeCurlError(&curl_e_exact));
      const real_t energy_error = sqrt(mu_inv_value * curl_error * curl_error +
                                       omega * omega * epsilon_r * l2_error *
                                       l2_error);
      cout << "Estimated error: " << estimated_error << endl;
      cout << "L2 error: " << l2_error << endl;
      cout << "Maxwell energy-norm error: " << energy_error << endl;

      if (visualization && sol_sock_r.good() && sol_sock_i.good())
      {
         sol_sock_r << "solution\n" << mesh << solution.real()
                    << "window_title 'Electric field: Real Part'" << flush;
         sol_sock_i << "solution\n" << mesh << solution.imag()
                    << "window_title 'Electric field: Imaginary Part'" << flush;
      }
      if (it == amr_iterations) { break; }

      refiner.Apply(mesh);
      if (refiner.Stop())
      {
         cout << "No elements selected for refinement. Stop." << endl;
         break;
      }

      fespace.Update();
      source_fes.Update();
      solution.Update();
      source.Update();
      a.Update();
      b.Update();
      refiner.Reset();
   }

   return 0;
}
