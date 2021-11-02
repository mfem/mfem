//                                MFEM Example 1
//
// Compile with: make ex_blk_fosls
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
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
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec0 = new H1_FECollection(order, dim);
   FiniteElementCollection *fec1 = new RT_FECollection(order-1, dim);
   FiniteElementSpace fespace0(&mesh, fec0);
   FiniteElementSpace fespace1(&mesh, fec1);

   Array<FiniteElementSpace *> fespaces(2);
   fespaces[0] = &fespace0;
   fespaces[1] = &fespace1;

   Array<int> ess_bdr;
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespaces[0]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   BlockBilinearForm a(fespaces);
   a.SetDiagonalPolicy(mfem::Operator::DIAG_KEEP);

   cout << "H1 fespace = " << fespace0.GetVSize() << endl;
   cout << "RT fespace = " << fespace1.GetVSize() << endl;

   cout << "height = " << a.Height() << endl;
   cout << "width = " << a.Width() << endl;
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);

   Array2D<BilinearFormIntegrator * > blfi(2,2);
   blfi(0,0) = new DiffusionIntegrator(one);
   blfi(0,1) = new MixedVectorWeakDivergenceIntegrator(one);
   blfi(1,0) = new MixedVectorGradientIntegrator(negone);
   BilinearFormIntegrator * divdiv = new DivDivIntegrator(one);
   BilinearFormIntegrator * mass = new VectorFEMassIntegrator(one);
   SumIntegrator * suminteg = new SumIntegrator();
   suminteg->AddIntegrator(divdiv);
   suminteg->AddIntegrator(mass);
   blfi(1,1) = suminteg;

   TestBlockBilinearFormIntegrator * integ = new TestBlockBilinearFormIntegrator();
   integ->SetIntegrators(blfi);
   a.AddDomainIntegrator(integ);
   a.Assemble();
   a.Finalize();


   BlockLinearForm b(fespaces);

   // TestBlockLinearFormIntegrator * lfi = new TestBlockLinearFormIntegrator(one);
   TestBlockLinearFormIntegrator * lininteg = new TestBlockLinearFormIntegrator();
   Array<LinearFormIntegrator * > lfi(2);
   lfi[0] = nullptr;
   lfi[1] = new VectorFEDomainLFDivIntegrator(negone);
   lininteg->SetIntegrators(lfi);
   b.AddDomainIntegrator(lininteg);
   // cout << "b size = " << b.Size() << endl;
   b.Assemble();


   // b.Print();


   OperatorPtr A;


   // need to implement blkgridfunction later but for now Vector would do
   int size = 0;
   for (int i = 0; i<fespaces.Size(); i++)
   {
      size += fespaces[i]->GetVSize();
   }

   Vector x(size);
   x = 0.0;

   // Vector b(size);

   // b = 1.0;

   Vector X,B;
   // Array<int> empty;
   // a.FormLinearSystem(empty,x,b,A,X,B);
   ess_tdof_list.Print();
   a.FormLinearSystem(ess_tdof_list,x,b,A,X,B);


   SparseMatrix * As = &(SparseMatrix&)(*A);

   As->Threshold(0.0);
   As->SortColumnIndices();
   As->PrintMatlab();

   return 0;


   GSSmoother M((SparseMatrix&)(*A));
   CGSolver cg;
   cg.SetRelTol(1e-6);
   cg.SetMaxIter(200);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(M);
   cg.SetOperator(*A);
   cg.Mult(B, X);


   GridFunction u_gf, sigma_gf;
   double *data = x.GetData();
   u_gf.MakeRef(fespaces[0],&data[0]);
   sigma_gf.MakeRef(fespaces[1],&data[fespaces[0]->GetVSize()]);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream solu_sock(vishost, visport);
      solu_sock.precision(8);
      solu_sock << "solution\n" << mesh << u_gf <<
                "window_title 'Numerical u' "
                << flush;
      socketstream sols_sock(vishost, visport);
      sols_sock.precision(8);
      sols_sock << "solution\n" << mesh << sigma_gf <<
                "window_title 'Numerical sigma' "
                << flush;
   }

   delete fec0;

   return 0;
}
