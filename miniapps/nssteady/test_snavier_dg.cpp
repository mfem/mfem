#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "snavier_dg.hpp"

using namespace std;
using namespace mfem;

struct s_NavierContext
{
   int ser_ref_levels = 1;
   int order = 1;
   bool reduce_order_pressure = true;
   bool reduce_order_stress = true;
   double kinvis = 1.0 / 40.0;
   double reference_pressure = 0.0;
   double reynolds = 1.0 / kinvis;

   bool iterative=false;
   bool petsc =true;
   const char *petscrc_file = "rc_direct";
   bool use_ksp_solver = false;
   int max_lin_it = 1000;
   double lin_it_tol = 1e-8;
   int setPrintLevel = 2;
   int max_picard_it = 20;
   double picard_it_tol = 1e-7;
   bool ni = false;
   bool visualization = false;
   bool checkres = false;
} ctx;

void vel_kovasznay(const Vector &x, Vector &u);
double pres_kovasznay(const Vector &x);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   bool verbose = (myrank == 0);

   // Parse command-line options.
   mfem::OptionsParser args(argc, argv);
   args.AddOption(&ctx.ser_ref_levels,
                     "-rs",
                     "--refine-serial",
                     "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&ctx.order, "-o", "--order",
                     "Finite element order (polynomial degree)");
   args.AddOption(&ctx.reduce_order_pressure, "-reduce-p", "--reduce-order-p", "-full-p",
				  	 "--full-order-p",
					 "Using k-1 order in approximation of pressure.");
   args.AddOption(&ctx.reduce_order_stress, "-reduce-s", "--reduce-order-s", "-full-s",
				  	 "--full-order-s",
					 "Using k-1 order in approximation of gradient of velocity.");
   args.AddOption(&ctx.petsc, "-petsc", "--use-petsc",
                  	 "-no-petsc", "--no-use-petsc",
					 "Enable or disable SC solver.");
   args.AddOption(&ctx.petscrc_file, "-petscopts", "--petscopts",
                  	  "PetscOptions file to use.");
   args.AddOption(&ctx.use_ksp_solver, "-ksp", "--ksp_solver", "-lu",
                  	  "--lu_solver", "Iterative solver or direct solver setup.");
   args.AddOption(&ctx.lin_it_tol, "-lin_tol", "--lin_tolerance",
                  	  "Tolerance in the iteration of the linear solver.");
   args.AddOption(&ctx.max_lin_it, "-lin_maxit", "--lin_max_nonlin_it",
                  	  "Maximum number of iterations of the linear solver.");
   args.AddOption(&ctx.setPrintLevel, "-printl", "--print_level",
                  	  "Setting the printlevel.");
   args.AddOption(&ctx.picard_it_tol,
                  	  "-pic-tol",
					  "--pictard-tolerance",
					  "Absolute tolerance for the Newton solve.");
   args.AddOption(&ctx.max_picard_it,
                  	  "-pic-it",
					  "--picard-iterations",
					  "Maximum iterations for the linear solve.");
   args.AddOption(&ctx.ni,
                  	  "-ni",
					  "--enable-ni",
					  "-no-ni",
					  "--disable-ni",
					  "Enable numerical integration rules.");
   args.AddOption(&ctx.visualization,
                      "-vis",
					  "--visualization",
					  "-no-vis",
					  "--no-visualization",
					  "Enable or disable GLVis visualization.");
   args.AddOption(
      &ctx.checkres,
      "-cr",
      "--checkresult",
      "-no-cr",
      "--no-checkresult",
      "Enable or disable checking of the result. Returns -1 on failure.");

   args.Parse();
   if (!args.Good())
   {
      if (verbose)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (verbose)
   {
       args.PrintOptions(std::cout);
   }

   // Create the rectangle domain (0,1.5)x(2.0) that is composed
   // by 2x4x2 triangle elements.
   Mesh mesh = Mesh::MakeCartesian2D(2, 4, Element::TRIANGLE, false, 1.5,
                                     2.0);

   // Shift entire domain to left with 0.5 unit. Now the domain
   // is (-0.5,1)x(2.0)
   mesh.EnsureNodes();
   GridFunction *nodes = mesh.GetNodes();
   *nodes -= 0.5;

   int dim = mesh.Dimension();
   for (int l = 0; l < ctx.ser_ref_levels; l++)
   {
       mesh.UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define the coefficients (e.g. parameters, analytical solution/s).
   //TODO

   // Create solver
   //TODO

   // Set parameters of the Fixed Point Solver
   //TODO

   // Set parameters of the Linear Solvers
   //TODO


   MPI_Finalize();
   return 0;
}

void vel_kovasznay(const Vector &x, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double lam = 0.5 * ctx.reynolds
                - sqrt(0.25 * ctx.reynolds * ctx.reynolds + 4.0 * M_PI * M_PI);
   u(0) = 1.0 - exp(lam * xi) * cos(2.0 * M_PI * yi);
   u(1) = lam / (2.0 * M_PI) * exp(lam * xi) * sin(2.0 * M_PI * yi);
}

double pres_kovasznay(const Vector &x)
{
   double xi = x(0);
   double lam = 0.5 * ctx.reynolds
                - sqrt(0.25 * ctx.reynolds * ctx.reynolds + 4.0 * M_PI * M_PI);
   return 0.5 * (1.0 - exp(2.0 * lam * xi)) + ctx.reference_pressure;
}
