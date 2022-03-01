//                      MFEM Example 31
//
// Compile with: make ex42
//
// Sample runs:  mpirun -np 4 ex31 -m ../data/square-disc.mesh -alpha 0.33 -o 2
//               mpirun -np 4 ex31 -m ../data/star.mesh -alpha 0.99 -o 3
//               mpirun -np 4 ex31 -m ../data/inline-quad.mesh -alpha 0.2 -o 3
//               mpirun -np 4 ex31 -m ../data/disc-nurbs.mesh -alpha 0.33 -o 3


// Description:
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ex31.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI.
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = true;
   double alpha = 0.2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Fractional exponent");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Array<double> coeffs, poles;

   ComputePartialFractionApproximation(alpha,coeffs,poles);

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   mesh.UniformRefinement();
   mesh.UniformRefinement();
   mesh.UniformRefinement();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace fespace(&pmesh, fec);
   if (myid == 0)
   {
        cout << "Number of finite element unknowns: "
             << fespace.GetTrueVSize() << endl;
   }

   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ConstantCoefficient f(1.0);
   ConstantCoefficient one(1.0);
   ParGridFunction u(&fespace);
   u = 0.;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream xout;
   socketstream uout;
   if (visualization)
   {
      xout.open(vishost, visport);
      xout.precision(8);
      uout.open(vishost, visport);
      uout.precision(8);
   }

   for (int i = 0; i<coeffs.Size(); i++)
   {
      ParLinearForm b(&fespace);
      ProductCoefficient cf(coeffs[i], f);
      b.AddDomainIntegrator(new DomainLFIntegrator(cf));
      b.Assemble();

      ParGridFunction x(&fespace);
      x = 0.0;

      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      ConstantCoefficient c2(-poles[i]);
      a.AddDomainIntegrator(new MassIntegrator(c2));
      a.Assemble();

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cout << "Size of linear system: " << A->Height() << endl;

      Solver *M = new OperatorJacobiSmoother(a, ess_tdof_list);;
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(3);
      cg.SetPreconditioner(*M);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete M;

      // 12. Recover the solution as a finite element grid function.
      a.RecoverFEMSolution(X, b, x);

      u+=x;

      // 14. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         xout << "parallel " << num_procs << " " << myid << "\n";
         xout << "solution\n" << pmesh << x << flush;
         uout << "parallel " << num_procs << " " << myid << "\n";
         uout << "solution\n" << pmesh << u << flush;
      }
   }

   delete fec;
   return 0;
}
