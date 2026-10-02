//                                MFEM Example 42
//
// Compile with: make ex42
//
// Sample runs:
//    ex42 -e 0 -amr 3
//    ex42 -e 1 -amr 3
//    ex42 -e 2 -amr 3
//
// Description: This example solves the real time-harmonic Maxwell equation
//
//     curl(muinv curl E) - omega^2 epsilon E = f,
//
// with homogeneous tangential-electric boundary conditions. It demonstrates a
// simple adaptive loop using one of three element error indicators:
//
//   0: Zienkiewicz-Zhu recovery of curl E,
//   1: MaxwellResidualEstimator, and
//   2: GeneralErrorEstimator populated with the Maxwell residual terms.
//
// The last two choices use the residual indicator of Chaumont-Frelet and Vega.

#include "mfem.hpp"
#include <iostream>
#include <memory>

using namespace mfem;
using namespace std;

namespace
{

real_t omega = 1.0;

// This field has zero tangential trace on the boundary of the unit cube.
void EExact(const Vector &x, Vector &e)
{
   e.SetSize(3);
   e = 0.0;
   e(2) = sin(M_PI * x(0)) * sin(M_PI * x(1));
}

void FExact(const Vector &x, Vector &f)
{
   f.SetSize(3);
   f = 0.0;
   const real_t ez = sin(M_PI * x(0)) * sin(M_PI * x(1));
   f(2) = (2.0 * M_PI * M_PI - omega * omega) * ez;
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
   args.AddOption(&estimator_type, "-e", "--estimator",
                  "Error estimator: 0 = ZZ, 1 = Maxwell residual, "
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
   GridFunction solution(&fespace), source(&source_fes);

   const real_t mu_inv_value = 1.0, epsilon_value = 1.0;
   ConstantCoefficient mu_inv(mu_inv_value), epsilon(epsilon_value);
   ConstantCoefficient negative_mass(-omega * omega * epsilon_value);
   VectorFunctionCoefficient e_exact(3, EExact), f_exact(3, FExact),
                             curl_e_exact(3, CurlEExact);

   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(mu_inv));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(negative_mass));
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_exact));

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   // The ZZ estimator uses CurlCurlIntegrator's flux-recovery interface.
   CurlCurlIntegrator zz_integrator(mu_inv);
   // Curl E is naturally H(div)-conforming. Recover it in H(curl) instead,
   // so its tangential discontinuities provide a non-trivial ZZ indicator.
   ND_FECollection zz_fec(order, 3);
   unique_ptr<ErrorEstimator> estimator;
   switch (estimator_type)
   {
      case 0:
      {
         estimator.reset(new ZienkiewiczZhuEstimator(
                            zz_integrator, solution,
                            new FiniteElementSpace(&mesh, &zz_fec)));
         break;
      }
      case 1:
         estimator.reset(new MaxwellResidualEstimator(solution, source, epsilon,
                                                      mu_inv, omega, order));
         break;
      case 2:
      {
         auto *general = new GeneralErrorEstimator(*mesh);
         AddMaxwellResidualEstimators(*general, solution, source, epsilon,
                                      mu_inv, omega, order);
         estimator.reset(general);
         break;
      }
      default:
         MFEM_ABORT("unknown estimator type");
   }

   ThresholdRefiner refiner(*estimator);
   refiner.SetTotalErrorFraction(fraction);
   refiner.PreferConformingRefinement();

   socketstream sol_sock;
   if (visualization) { sol_sock.open("localhost", 19916); }

   for (int it = 0; it <= amr_iterations; it++)
   {
      cout << "\nAMR iteration " << it << ", unknowns: "
           << fespace.GetTrueVSize() << endl;

      source.ProjectCoefficient(f_exact);
      b.Assemble();
      a.Assemble();

      Array<int> ess_tdof_list;
      // Projecting the complete field is the standard vector-FE way to set
      // its tangential trace; FormLinearSystem retains only essential values.
      solution.ProjectCoefficient(e_exact);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, solution, b, A, X, B);

      // The negative mass term makes the system indefinite; GMRES is used in
      // place of the CG solve used by the definite curl-curl example.
      GSSmoother prec(*A.As<SparseMatrix>());
      GMRES(*A, prec, B, X, 0, 500, 50, 1e-10, 0.0);
      a.RecoverFEMSolution(X, b, solution);

      estimator->Reset();
      estimator->GetLocalErrors();
      const real_t estimated_error = estimator->GetTotalError();
      const real_t l2_error = solution.ComputeL2Error(e_exact);
      const real_t curl_error = solution.ComputeCurlError(&curl_e_exact);
      const real_t energy_error = sqrt(mu_inv_value * curl_error * curl_error +
                                       omega * omega * epsilon_value * l2_error *
                                       l2_error);
      cout << "Estimated error: " << estimated_error << endl;
      cout << "L2 error: " << l2_error << endl;
      cout << "Maxwell energy-norm error: " << energy_error << endl;

      if (visualization && sol_sock.good())
      {
         sol_sock << "solution\n" << mesh << solution << flush;
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
