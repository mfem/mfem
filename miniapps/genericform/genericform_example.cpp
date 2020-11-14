#include "mfem.hpp"
#include "genericform.hpp"
#include "qfuncintegrator.hpp"
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
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   {
      for (int l = 0; l < refinements; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   auto fec = H1_FECollection(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ParLinearForm f(&fespace);
   ConstantCoefficient fcoeff(-4.0);
   f.AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   f.Assemble();

   ParGridFunction x(&fespace);
   x = 0.0;

   FunctionCoefficient u_excoeff([](const Vector &coords) {
      double x = coords(0);
      double y = coords(1);

      return x * x + y * y;
   });

   x.ProjectCoefficient(u_excoeff);

   GenericForm a(&fespace);
   a.AssumeLinear();

   auto diffusion = new QFunctionIntegrator(
      [](auto u, auto du) {
         // R(u, du) = -\nabla^2 u - f
         double f0 = 4.0;
         Vector f1(2);
         f1 = du;
         return qfunc_output_type{f0, f1};
      },
      [](auto u, auto du) {
         // R'(u, du)
         double f00 = 0.0;
         Vector f01(2);
         f01 = 0.0;
         Vector f10(2);
         f10 = 0.0;
         DenseMatrix f11;
         f11.Diag(1.0, 2);
         return qfunc_grad_output_type{f00, f01, f10, f11};
      });

   a.AddDomainIntegrator(diffusion);

   Vector X, B;
   x.GetTrueDofs(X);

   auto &A = a.GetGradient(X);

   B.SetSize(X.Size());
   B = 0.0;

   A.Mult(X, B);

   printf("sum(B) = %.1E\n", B.Sum());

   Vector Btmp;
   f.ParallelAssemble(Btmp);

   B -= Btmp;

   ParGridFunction b(&fespace);
   b.SetFromTrueDofs(B);

   ConstantCoefficient zero_coeff(0.0);
   b.ProjectBdrCoefficient(zero_coeff, ess_bdr);

   printf("||B|| = %.5E\n", b.Norml2());

   // a.Assemble();

   // OperatorPtr A;
   // Vector B, X;
   // a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // CGSolver cg(MPI_COMM_WORLD);
   // cg.SetRelTol(1e-12);
   // cg.SetMaxIter(2000);
   // cg.SetPrintLevel(1);
   // cg.SetOperator(*A);
   // cg.Mult(B, X);

   // a.RecoverFEMSolution(X, b, x);

   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << pmesh << b << flush;

   MPI_Finalize();

   return 0;
}