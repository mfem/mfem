#include "mfem.hpp"
#include "parvariationalform.hpp"
#include "qfuncintegrator.hpp"
#include "tensor.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 1;
   int refinements = 0;
   double p = 5.0;

   OptionsParser args(argc, argv);
   args.AddOption(&refinements, "-r", "--ref", "");
   args.AddOption(&order, "-o", "--order", "");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh mesh(mesh_file, 1, 1);

   int dim = mesh.Dimension();
   {
      for (int l = 0; l < refinements; l++)
      {
         mesh.UniformRefinement();
      }
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   auto fec = H1_FECollection(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;

   FunctionCoefficient u_excoeff([&](const Vector &coords) {
      double x = coords(0);
      double y = coords(1);

      // return 1.0 - pow(sqrt(x * x + y * y), p / (p - 1.0));
      return x * x + y * y;
   });

   ParGridFunction x(&fespace);
   x.Randomize();

   x.ProjectBdrCoefficient(u_excoeff, ess_bdr);

   ParVariationalForm form(&fespace);

   auto plaplacian = new QFunctionIntegrator([&](auto u, auto du, auto x) {
      // auto f0 = -1.0;
      // auto f1 = pow(norm(du), p - 2.0) * du;
      auto f0 = 4.0 * (1.0 + 2.0 * x[0] * x[0] + 2.0 * x[1] * x[1]);
      auto f1 = (1.0 + u) * du;
      return std::tuple{f0, f1};
   }, 0, pmesh);

   form.AddDomainIntegrator(plaplacian);

   form.SetEssentialBC(ess_bdr);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-6);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetOperator(form);
   newton.SetSolver(cg);
   newton.SetPrintLevel(1);
   newton.SetRelTol(1e-8);
   newton.SetMaxIter(100);

   Vector zero;
   Vector X;
   x.GetTrueDofs(X);
   newton.Mult(zero, X);

   x.Distribute(X);

   // x.ProjectCoefficient(u_excoeff);
   
   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << pmesh << x << flush;

   MPI_Finalize();

   return 0;
}