
#include "mfem.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   // 10. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(&fespace);

   auto func = [&](Vector coord) { return std::sin(coord(0)*coord(1)); };
   FunctionCoefficient x_coeff(func);

   x.ProjectCoefficient(x_coeff);
   ParGridFunction grad_x(&fespace);
   ConstantCoefficient zero(0.0);
   for (int i = 0; i < dim; i++)
   {
      x.GetDerivative(1, i, grad_x);
      //x.GridFunction::GetDerivative(1, i, grad_x); // old behavior.
      double error = grad_x.ComputeL2Error(zero);
      if (myid == 0) { cout << setprecision(14) << error << endl; }
   }

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << pmesh << x << flush;

   return 0;
}
