//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options
   const char *mesh_file = "../data/star.mesh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   mesh.UniformRefinement();

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace H1fes(&mesh, &fec);
   cout << "Number of unknowns: " << H1fes.GetTrueVSize() << endl;

   RT_Trace_FECollection trace_fec(order-1, mesh.Dimension());
   FiniteElementSpace trace_fes(&mesh, &trace_fec);

   int dim = mesh.Dimension();
   int test_order = order;
   if (dim == 2 && (order%2 == 0 || (mesh.MeshGenerator() & 2 && order > 1)))
   {
      test_order++;
   }

   L2_FECollection test_fec(test_order,mesh.Dimension());

   NormalEquationsWeakFormulation a(&H1fes,&trace_fes,&test_fec);
   ConstantCoefficient one(1.0);
   a.SetDomainBFIntegrator(new DiffusionIntegrator(one));
   BilinearFormIntegrator * diffusion = new DiffusionIntegrator(one);
   BilinearFormIntegrator * mass = new MassIntegrator(one);
   SumIntegrator * suminteg = new SumIntegrator();
   suminteg->AddIntegrator(diffusion);
   suminteg->AddIntegrator(mass);
   InverseIntegrator * Ginv = new InverseIntegrator(suminteg);
   a.SetTestIntegrator(Ginv);
   a.SetTraceElementBFIntegrator(new TraceIntegrator);

   a.SetDomainLFIntegrator(new DomainLFIntegrator(one));
   // a.SetDiagonalPolicy(mfem::Operator::DIAG_ZERO);
   a.Assemble();


   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   Vector X,B;
   OperatorPtr A;

   int size = H1fes.GetTrueVSize() + trace_fes.GetTrueVSize();

   Vector x(size);
   x = 0.0;

   a.FormLinearSystem(ess_tdof_list,x,A,X,B);



   GSSmoother M((SparseMatrix&)(*A));
   CGSolver cg;


   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(3);
   cg.SetPreconditioner(M);
   cg.SetOperator(*A);
   cg.Mult(B, X);


   // SparseMatrix & As = (SparseMatrix&)(*A);

   a.RecoverFEMSolution(X,x);

   GridFunction u_gf;
   double *data = x.GetData();
   u_gf.MakeRef(&H1fes,data);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream solu_sock(vishost, visport);
   solu_sock.precision(8);
   solu_sock << "solution\n" << mesh << u_gf <<
             "window_title 'Numerical u' "
             << flush;



}
