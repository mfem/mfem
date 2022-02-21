//                      MFEM Example 42
//
// Compile with: make ex42
//
// Sample runs:  ex42 -m ../data/square-disc.mesh -alpha 0.33 -o 2
//               ex42 -m ../data/star.mesh -alpha 0.99 -o 3
//               ex42 -m ../data/inline-quad.mesh -alpha 0.2 -o 3


// Description:  
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ex42.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = true;
   double alpha = 0.5;

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
   args.PrintOptions(cout);


   Array<double> coeffs, poles;
   ComputePartialFractionApproximation(alpha,coeffs,poles);

   mfem::out << "coeffs = "; coeffs.Print(cout, coeffs.Size());
   mfem::out << "poles  = "; poles.Print(cout, poles.Size());

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();


   mesh.UniformRefinement();
   mesh.UniformRefinement();

   FiniteElementCollection *fec = new H1_FECollection(order, dim);

   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }


   ConstantCoefficient f(1.0);
   ConstantCoefficient one(1.0);
   GridFunction u(&fespace);
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
      LinearForm b(&fespace);
      ProductCoefficient cf(coeffs[i], f);
      b.AddDomainIntegrator(new DomainLFIntegrator(cf));
      b.Assemble();

      GridFunction x(&fespace);
      x = 0.0;

      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      ConstantCoefficient c2(-poles[i]);
      a.AddDomainIntegrator(new MassIntegrator(c2));
      a.Assemble();

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cout << "Size of linear system: " << A->Height() << endl;

      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);

      // 12. Recover the solution as a finite element grid function.
      a.RecoverFEMSolution(X, b, x);

      u+=x;

      // 14. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         xout << "solution\n" << mesh << x << flush;
         uout << "solution\n" << mesh << u << flush;
      }
      cin.get();
   }
   return 0;
}
