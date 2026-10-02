//                         MFEM Example 42 - Parallel Version
//
// Parallel adaptive real time-harmonic Maxwell example. See ex42.cpp for the
// equation, manufactured solution, and estimator definitions.

#include "mfem.hpp"
#include <iostream>
#include <memory>

using namespace mfem;
using namespace std;

namespace
{
real_t omega = 1.0;
void EExact(const Vector &x, Vector &e)
{ e.SetSize(3); e = 0.0; e(2) = sin(M_PI*x(0))*sin(M_PI*x(1)); }
void FExact(const Vector &x, Vector &f)
{
   f.SetSize(3); f = 0.0;
   f(2) = (2.0*M_PI*M_PI - omega*omega)*sin(M_PI*x(0))*sin(M_PI*x(1));
}
void CurlEExact(const Vector &x, Vector &c)
{
   c.SetSize(3);
   c(0) = M_PI*sin(M_PI*x(0))*cos(M_PI*x(1));
   c(1) = -M_PI*cos(M_PI*x(0))*sin(M_PI*x(1));
   c(2) = 0.0;
}
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int myid = Mpi::WorldRank();
   const char *mesh_file = "../data/inline-hex.mesh";
   int order = 1, estimator_type = 2, amr_iterations = 2;
   real_t fraction = 0.5;
   bool visualization = true;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Nedelec polynomial degree.");
   args.AddOption(&omega, "-w", "--omega", "Angular frequency.");
   args.AddOption(&estimator_type, "-e", "--estimator",
                  "Error estimator: 0 = ZZ, 1 = Maxwell residual, 2 = general residual.");
   args.AddOption(&amr_iterations, "-amr", "--amr-iterations",
                  "Number of adaptive refinement steps.");
   args.AddOption(&fraction, "-f", "--fraction", "Refinement fraction.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis output.");
   args.Parse();
   if (!args.Good()) { if (!myid) { args.PrintUsage(cout); } return 1; }
   if (!myid) { args.PrintOptions(cout); }
   MFEM_VERIFY(order > 0 && omega > 0.0 && fraction > 0.0 && fraction < 1.0,
               "invalid options");

   Mesh serial_mesh(mesh_file, 1, 1);
   MFEM_VERIFY(serial_mesh.Dimension() == 3 && serial_mesh.SpaceDimension() == 3,
               "Example 42p requires a three-dimensional volume mesh.");
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   ND_FECollection nd_fec(order, 3);
   L2_FECollection l2_fec(order, 3);
   ParFiniteElementSpace fes(&mesh, &nd_fec);
   ParFiniteElementSpace source_fes(&mesh, &l2_fec, 3, Ordering::byVDIM);
   ParGridFunction solution(&fes), source(&source_fes);
   ConstantCoefficient mu_inv(1.0), epsilon(1.0), negative_mass(-omega*omega);
   VectorFunctionCoefficient e_exact(3, EExact), f_exact(3, FExact),
                             curl_e_exact(3, CurlEExact);
   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new CurlCurlIntegrator(mu_inv));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(negative_mass));
   ParLinearForm b(&fes);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_exact));
   Array<int> ess_bdr(mesh.bdr_attributes.Max()); ess_bdr = 1;
   CurlCurlIntegrator zz_integrator(mu_inv);
   ND_FECollection zz_fec(order, 3);
   unique_ptr<ErrorEstimator> estimator;
   if (estimator_type == 0)
   {
      estimator.reset(new ZienkiewiczZhuEstimator(
                         zz_integrator, solution,
                         new ParFiniteElementSpace(&mesh, &zz_fec)));
   }
   else if (estimator_type == 1)
   {
      estimator.reset(new MaxwellResidualEstimator(solution, source, epsilon,
                                                   mu_inv, omega, order));
   }
   else if (estimator_type == 2)
   {
      auto *general = new GeneralErrorEstimator(*pmesh);
      AddMaxwellResidualEstimators(*general, solution, source, epsilon,
                                   mu_inv, omega, order);
      estimator.reset(general);
   }
   else { MFEM_ABORT("unknown estimator type"); }
   ThresholdRefiner refiner(*estimator);
   refiner.SetTotalErrorFraction(fraction);
   refiner.PreferConformingRefinement();

   for (int it = 0; it <= amr_iterations; it++)
   {
      if (!myid)
      {
         cout << "\nAMR iteration " << it << ", unknowns: "
              << fes.GlobalTrueVSize() << endl;
      }
      source.ProjectCoefficient(f_exact);
      b.Assemble(); a.Assemble();
      solution.ProjectCoefficient(e_exact);
      Array<int> ess_tdof_list;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      HypreParMatrix A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, solution, b, A, X, B);
      HypreAMS ams(A, &fes);
      HypreGMRES gmres(A);
      gmres.SetTol(1e-10); gmres.SetMaxIter(500); gmres.SetKDim(50);
      gmres.SetPrintLevel(0); gmres.SetPreconditioner(ams); gmres.Mult(B, X);
      a.RecoverFEMSolution(X, b, solution);
      estimator->Reset(); estimator->GetLocalErrors();
      const real_t l2_error = solution.ComputeL2Error(e_exact);
      const real_t curl_error = solution.ComputeCurlError(&curl_e_exact);
      const real_t energy_error = sqrt(curl_error*curl_error +
                                       omega*omega*l2_error*l2_error);
      // GetTotalError performs a global reduction, so every rank must enter it.
      const real_t estimated_error = estimator->GetTotalError();
      if (!myid)
      {
         cout << "Estimated error: " << estimated_error
              << ", Maxwell energy-norm error: " << energy_error << endl;
      }
      if (it == amr_iterations) { break; }
      refiner.Apply(mesh);
      if (refiner.Stop()) { break; }
      fes.Update(); source_fes.Update(); solution.Update(); source.Update();
      a.Update(); b.Update(); refiner.Reset();
   }
   return 0;
}
