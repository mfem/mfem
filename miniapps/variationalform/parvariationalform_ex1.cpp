#include "mfem.hpp"
#include "parvariationalform.hpp"
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
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order,
                  "-o",
                  "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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
      int ref_levels = (int) floor(log(10. / mesh.GetNE()) / log(2.) / dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (pmesh.GetNodes())
   {
      fec = pmesh.GetNodes()->OwnFEC();
      delete_fec = false;
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_Int size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;

   FunctionCoefficient u_excoeff([&](const Vector &coords) {
      double x = coords(0);
      double y = coords(1);
      return x * x + y * y;
   });

   ParGridFunction x(&fespace);

   x.ProjectBdrCoefficient(u_excoeff, ess_bdr);

   ParVariationalForm form(&fespace);

   auto b_coeff = [&](auto u, auto du, auto x) {
      return 4.0 * (1.0 + 2.0 * x[0] * x[0] + 2.0 * x[1] * x[1]);
   };

   form.AddDomainIntegrator<DomainLFIntegrator>(b_coeff);

   auto a_coeff = [&](auto u, auto du, auto x) { return 1.0 + u; };

   form.AddDomainIntegrator<DiffusionIntegrator>(a_coeff);

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

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   if (delete_fec)
   {
      delete fec;
   }
   MPI_Finalize();

   return 0;
}
