//                         MFEM Example 43 - Parallel Version
//
// Compile with: make ex43p
//
// Sample run:
//    mpirun -np 4 ex43p -amr 3 -eps-i 0.1 -sigma 0.2
//
// Description: Parallel adaptive solution of the complex time-harmonic
// Maxwell equation
//
//     curl(muinv curl E) - omega^2 epsilon_eff E = f,
//
// where epsilon_eff = epsilon_r + i (epsilon_i + sigma / omega). The
// e^{-i omega t} convention is used, so epsilon_i and sigma both model
// passive loss. Homogeneous tangential-electric boundary conditions are
// imposed strongly. The real and imaginary parts are recovered separately
// with a selectable error estimator and visualized in separate windows.

#include "mfem.hpp"
#include <iostream>
#include <memory>

using namespace mfem;
using namespace std;

namespace
{

real_t omega = 1.0, epsilon_r = 1.0, epsilon_i = 0.1, sigma = 0.1;
const real_t imag_amplitude = real_t(0.5);

real_t FundamentalMode(const Vector &x)
{
   return sin(M_PI * x(0)) * sin(M_PI * x(1));
}

real_t HigherMode(const Vector &x)
{
   return sin(2.0 * M_PI * x(0)) * sin(M_PI * x(1));
}

void EExactReal(const Vector &x, Vector &e)
{
   e.SetSize(3);
   e = 0.0;
   e(2) = FundamentalMode(x);
}

void EExactImag(const Vector &x, Vector &e)
{
   e.SetSize(3);
   e = 0.0;
   e(2) = imag_amplitude * HigherMode(x);
}

void FExactReal(const Vector &x, Vector &f)
{
   f.SetSize(3); f = 0.0;
   const real_t u = FundamentalMode(x);
   const real_t v = HigherMode(x);
   const real_t lambda = 2.0 * M_PI * M_PI;
   const real_t eta = epsilon_i + sigma / omega;
   f(2) = (lambda - omega * omega * epsilon_r) * u +
          imag_amplitude * omega * omega * eta * v;
}

void FExactImag(const Vector &x, Vector &f)
{
   f.SetSize(3); f = 0.0;
   const real_t u = FundamentalMode(x);
   const real_t v = HigherMode(x);
   const real_t lambda = 5.0 * M_PI * M_PI;
   const real_t eta = epsilon_i + sigma / omega;
   f(2) = imag_amplitude * (lambda - omega * omega * epsilon_r) * v -
          omega * omega * eta * u;
}

void CurlEExactReal(const Vector &x, Vector &curl_e)
{
   curl_e.SetSize(3);
   curl_e(0) = M_PI * sin(M_PI * x(0)) * cos(M_PI * x(1));
   curl_e(1) = -M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
   curl_e(2) = 0.0;
}

void CurlEExactImag(const Vector &x, Vector &curl_e)
{
   curl_e.SetSize(3);
   curl_e(0) = imag_amplitude * M_PI * sin(2.0 * M_PI * x(0)) *
               cos(M_PI * x(1));
   curl_e(1) = -2.0 * imag_amplitude * M_PI * cos(2.0 * M_PI * x(0)) *
               sin(M_PI * x(1));
   curl_e(2) = 0.0;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int myid = Mpi::WorldRank();
   const int nranks = Mpi::WorldSize();
   const char *mesh_file = "../data/inline-hex.mesh";
   int order = 1, estimator_type = 2, amr_iterations = 3;
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
                  "Error estimator: 0 = complex ZZ, "
                  "1 = complex Maxwell residual, "
                  "2 = generalized complex Maxwell residual.");
   args.AddOption(&amr_iterations, "-amr", "--amr-iterations",
                  "Number of adaptive refinement steps.");
   args.AddOption(&fraction, "-f", "--fraction",
                  "Fraction of the total estimated error to refine.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis output.");
   args.Parse();
   if (!args.Good())
   {
      if (!myid) { args.PrintUsage(cout); }
      return 1;
   }
   if (!myid) { args.PrintOptions(cout); }
   MFEM_VERIFY(order > 0 && omega > 0.0, "order and omega must be positive.");
   MFEM_VERIFY(fraction > 0.0 && fraction < 1.0,
               "refinement fraction must be in (0, 1).");

   Mesh mesh(mesh_file, 1, 1);
   MFEM_VERIFY(mesh.Dimension() == 3 && mesh.SpaceDimension() == 3,
               "Example 43p requires a three-dimensional volume mesh.");
   mesh.EnsureNCMesh();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   ND_FECollection nd_fec(order, 3);
   L2_FECollection source_fec(order, 3);
   ParFiniteElementSpace fespace(&pmesh, &nd_fec);
   ParFiniteElementSpace source_fes(&pmesh, &source_fec, 3, Ordering::byVDIM);
   ParComplexGridFunction solution(&fespace), source(&source_fes);

   const real_t mu_inv_value = 1.0;
   ConstantCoefficient mu_inv(mu_inv_value), epsilon(epsilon_r),
                       epsilon_loss(epsilon_i + sigma / omega);
   ConstantCoefficient negative_mass(-omega * omega * epsilon_r),
                       loss_mass(-omega * omega * epsilon_i - omega * sigma),
                       positive_mass(omega * omega * epsilon_r);
   VectorFunctionCoefficient e_exact_r(3, EExactReal), e_exact_i(3, EExactImag),
                             f_exact_r(3, FExactReal), f_exact_i(3, FExactImag),
                             curl_e_exact_r(3, CurlEExactReal),
                             curl_e_exact_i(3, CurlEExactImag);

   ParComplexLinearForm b(&fespace, ComplexOperator::HERMITIAN);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_exact_r),
                         new VectorFEDomainLFIntegrator(f_exact_i));
   ParSesquilinearForm a(&fespace, ComplexOperator::HERMITIAN);
   a.AddDomainIntegrator(new CurlCurlIntegrator(mu_inv), nullptr);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(negative_mass),
                         new VectorFEMassIntegrator(loss_mass));
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;

   CurlCurlIntegrator zz_integrator(mu_inv);
   ND_FECollection zz_fec(order, 3);
   ParFiniteElementSpace zz_fes(&pmesh, &zz_fec);
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
         auto *general = new GeneralErrorEstimator(pmesh);
         AddComplexMaxwellResidualEstimators(*general, solution, source,
                                             epsilon, epsilon_loss, mu_inv,
                                             omega, order);
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
      sol_sock_r.precision(8);
      sol_sock_i.open("localhost", 19916);
      sol_sock_i.precision(8);
   }

   for (int it = 0; it <= amr_iterations; it++)
   {
      const HYPRE_BigInt global_dofs = fespace.GlobalTrueVSize();
      if (!myid)
      {
         cout << "\nAMR iteration " << it << ", unknowns: "
              << global_dofs << endl;
      }

      source.ProjectCoefficient(f_exact_r, f_exact_i);
      b.Assemble();
      a.Assemble();
      solution.ProjectCoefficient(e_exact_r, e_exact_i);
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, solution, b, A, X, B);

      // The positive shifted curl-curl proxy gives AMS an H(curl)-elliptic
      // operator for both real-imaginary diagonal blocks.
      ParBilinearForm proxy(&fespace);
      proxy.AddDomainIntegrator(new CurlCurlIntegrator(mu_inv));
      proxy.AddDomainIntegrator(new VectorFEMassIntegrator(positive_mass));
      proxy.Assemble();
      OperatorHandle proxy_op;
      proxy.FormSystemMatrix(ess_tdof_list, proxy_op);
      auto *ams = new HypreAMS(*proxy_op.As<HypreParMatrix>(), &fespace);
      const int true_size = A->Height() / 2;
      Array<int> offsets(3);
      offsets[0] = 0;
      offsets[1] = true_size;
      offsets[2] = true_size;
      offsets.PartialSum();
      BlockDiagonalPreconditioner prec(offsets);
      prec.SetDiagonalBlock(0, ams);
      prec.SetDiagonalBlock(1, new ScaledOperator(ams, -1.0));
      prec.owns_blocks = 1;

      FGMRESSolver solver(MPI_COMM_WORLD);
      solver.SetOperator(*A);
      solver.SetPreconditioner(prec);
      solver.SetPrintLevel(0);
      solver.SetMaxIter(500);
      solver.SetKDim(50);
      solver.SetRelTol(1e-10);
      solver.SetAbsTol(0.0);
      solver.Mult(B, X);
      a.RecoverFEMSolution(X, b, solution);

      estimator->Reset();
      estimator->GetLocalErrors();
      const real_t estimated_error = estimator->GetTotalError();
      const real_t l2_error = solution.ComputeL2Error(e_exact_r, e_exact_i);
      const real_t curl_error = hypot(
         solution.real().ComputeCurlError(&curl_e_exact_r),
         solution.imag().ComputeCurlError(&curl_e_exact_i));
      const real_t energy_error = sqrt(mu_inv_value * curl_error * curl_error +
                                       omega * omega * epsilon_r * l2_error *
                                       l2_error);
      if (!myid)
      {
         cout << "Estimated error: " << estimated_error << endl;
         cout << "L2 error: " << l2_error << endl;
         cout << "Maxwell energy-norm error: " << energy_error << endl;
      }

      if (visualization && sol_sock_r.good() && sol_sock_i.good())
      {
         sol_sock_r << "parallel " << nranks << " " << myid << "\n";
         sol_sock_r << "solution\n" << pmesh << solution.real()
                    << "window_title 'Electric field: Real Part'"
                    << "window_geometry 0 0 400 350" << flush;
         MPI_Barrier(MPI_COMM_WORLD);
         sol_sock_i << "parallel " << nranks << " " << myid << "\n";
         sol_sock_i << "solution\n" << pmesh << solution.imag()
                    << "window_title 'Electric field: Imaginary Part'"
                    << "window_geometry 400 0 400 350" << flush;
         MPI_Barrier(MPI_COMM_WORLD);
      }
      if (it == amr_iterations) { break; }

      refiner.Apply(pmesh);
      if (refiner.Stop())
      {
         if (!myid) { cout << "No elements selected for refinement. Stop." << endl; }
         break;
      }
      fespace.Update();
      source_fes.Update();
      solution.Update();
      source.Update();
      a.Update(&fespace);
      b.Update(&fespace);
      refiner.Reset();
   }

   return 0;
}
