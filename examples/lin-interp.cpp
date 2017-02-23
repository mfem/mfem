#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double f(const Vector & x);
double g(const Vector & x);
void   F(const Vector & x, Vector & v);
void   G(const Vector & x, Vector & v);

double fg(const Vector & x);
double FdotG(const Vector & x);

void   fG(const Vector & x, Vector & v);
void   Fg(const Vector & x, Vector & v);
void   FcrossG(const Vector & x, Vector & v);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = 1;

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

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec0 = NULL;
   FiniteElementCollection *fec1 = NULL;
   FiniteElementCollection *fec2 = NULL;
   FiniteElementCollection *fec3 = NULL;

   fec0 = new H1_FECollection(order, dim);
   fec1 = new ND_FECollection(order, dim);
   if ( dim >= 3 )
   {
      fec2 = new RT_FECollection(order-1, dim);
      fec3 = new L2_FECollection(order-1, dim);
   }

   FiniteElementSpace *fespace0 = new FiniteElementSpace(mesh, fec0);
   FiniteElementSpace *fespace1 = new FiniteElementSpace(mesh, fec1);
   FiniteElementSpace *fespace2 = NULL;
   FiniteElementSpace *fespace3 = NULL;

   if ( fec2 != NULL ) { fespace2 = new FiniteElementSpace(mesh, fec2); }
   if ( fec3 != NULL ) { fespace3 = new FiniteElementSpace(mesh, fec3); }

   cout << "Number of finite element unknowns: "
        << fespace0->GetTrueVSize() << endl;

   FunctionCoefficient       fCoef(f);
   VectorFunctionCoefficient FCoef(dim, F);

   FunctionCoefficient       gCoef(g);
   VectorFunctionCoefficient GCoef(dim, G);

   FunctionCoefficient        fgCoef(fg);
   FunctionCoefficient        FGCoef(FdotG);
   VectorFunctionCoefficient  fGCoef(dim, fG);
   VectorFunctionCoefficient  FgCoef(dim, Fg);
   VectorFunctionCoefficient FxGCoef(dim, FcrossG);

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction f0(fespace0);
   f0.ProjectCoefficient(fCoef);

   GridFunction g0(fespace0);
   g0.ProjectCoefficient(gCoef);

   cout << "Error in f: " <<  f0.ComputeL2Error(fCoef) << endl;
   cout << "Error in g: " <<  g0.ComputeL2Error(gCoef) << endl;

   DiscreteLinearOperator Opf0(fespace0,fespace0);
   Opf0.AddDomainInterpolator(new ScalarProductInterpolator(fCoef));
   Opf0.Assemble();

   GridFunction fg0(fespace0);
   Opf0.Mult(g0,fg0);
   cout << "Error in fg: " <<  fg0.ComputeL2Error(fgCoef) << endl;

   //
   GridFunction G1(fespace1);
   G1.ProjectCoefficient(GCoef);

   cout << "Error in f: " <<  f0.ComputeL2Error(fCoef) << endl;
   cout << "Error in G: " <<  G1.ComputeL2Error(GCoef) << endl;

   DiscreteLinearOperator Opf1(fespace1,fespace1);
   Opf1.AddDomainInterpolator(new ScalarVectorProductInterpolator(dim, fCoef));
   Opf1.Assemble();

   GridFunction fG1(fespace1);
   Opf1.Mult(G1,fG1);
   cout << "Error in fG: " <<  fG1.ComputeL2Error(fGCoef) << endl;

   //
   GridFunction F1(fespace1);
   F1.ProjectCoefficient(FCoef);

   cout << "Error in F: " <<  F1.ComputeL2Error(FCoef) << endl;
   cout << "Error in g: " <<  g0.ComputeL2Error(gCoef) << endl;

   DiscreteLinearOperator OpF1(fespace0,fespace1);
   OpF1.AddDomainInterpolator(new VectorScalarProductInterpolator(FCoef));
   OpF1.Assemble();

   GridFunction Fg1(fespace1);
   OpF1.Mult(g0,Fg1);
   cout << "Error in Fg: " <<  Fg1.ComputeL2Error(FgCoef) << endl;

   //
   GridFunction G2(fespace2);
   G2.ProjectCoefficient(GCoef);

   cout << "Error in F: " <<  F1.ComputeL2Error(FCoef) << endl;
   cout << "Error in G: " <<  G2.ComputeL2Error(GCoef) << endl;

   DiscreteLinearOperator OpF3(fespace2,fespace3);
   OpF3.AddDomainInterpolator(new VectorInnerProductInterpolator(FCoef));
   OpF3.Assemble();

   GridFunction FG3(fespace3);
   OpF3.Mult(G2,FG3);
   cout << "Error in F.G: " <<  FG3.ComputeL2Error(FGCoef) << endl;

   if ( dim == 3 )
   {
      //
      cout << "Error in F: " <<  F1.ComputeL2Error(FCoef) << endl;
      cout << "Error in G: " <<  G1.ComputeL2Error(GCoef) << endl;

      DiscreteLinearOperator OpF2(fespace1,fespace2);
      OpF2.AddDomainInterpolator(new VectorCrossProductInterpolator(FCoef));
      OpF2.Assemble();

      GridFunction FxG2(fespace2);
      OpF2.Mult(G1,FxG2);
      cout << "Error in FxG: " <<  FxG2.ComputeL2Error(FxGCoef) << endl;

      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream sol_sock_FxG(vishost, visport);
      sol_sock_FxG.precision(8);
      sol_sock_FxG << "solution\n" << *mesh << FxG2 << flush;
   }

   // 12. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   /*
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);
   */

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_fg(vishost, visport);
      sol_sock_fg.precision(8);
      sol_sock_fg << "solution\n" << *mesh << fg0 << flush;

      socketstream sol_sock_fG(vishost, visport);
      sol_sock_fG.precision(8);
      sol_sock_fG << "solution\n" << *mesh << fG1 << flush;

      socketstream sol_sock_Fg(vishost, visport);
      sol_sock_Fg.precision(8);
      sol_sock_Fg << "solution\n" << *mesh << Fg1 << flush;

      socketstream sol_sock_FG(vishost, visport);
      sol_sock_FG.precision(8);
      sol_sock_FG << "solution\n" << *mesh << FG3 << flush;
   }

   // 14. Free the used memory.
   delete fespace0;
   delete fespace1;
   delete fespace2;
   delete fespace3;
   delete fec0;
   delete fec1;
   delete fec2;
   delete fec3;
   delete mesh;

   return 0;
}

double f(const Vector & x)
{
   double val = exp(x[0]);

   if ( x.Size() >= 2 )
   {
      val *= exp(x[1]/2.0);
   }
   if ( x.Size() >= 3 )
   {
      val *= exp(x[2]/3.0);
   }

   return val;
}

void F(const Vector & x, Vector & v)
{
   v.SetSize(x.Size());

   v[0] = exp(x[0]);

   if ( x.Size() >= 2 )
   {
      v[1] = exp(x[1]/2.0);
   }
   if ( x.Size() >= 3 )
   {
      v[2] = exp(x[2]/3.0);
   }
}

double g(const Vector & x)
{
   double val = cos(M_PI*x[0]);

   if ( x.Size() >= 2 )
   {
      val *= sin(M_PI*x[1]/2.0);
   }
   if ( x.Size() >= 3 )
   {
      val *= cos(M_PI*x[2]/3.0);
   }

   return val;
}

void G(const Vector & x, Vector & v)
{
   v.SetSize(x.Size());

   v[0] = cos(M_PI*x[0]);

   if ( x.Size() >= 2 )
   {
      v[1] = sin(M_PI*x[1]/2.0);
   }
   if ( x.Size() >= 3 )
   {
      v[2] = cos(M_PI*x[2]/3.0);
   }
}

double fg(const Vector & x)
{
   return f(x) * g(x);
}

void fG(const Vector & x, Vector &v)
{
   G(x, v);
   v *= f(x);
}

void Fg(const Vector & x, Vector &v)
{
   F(x, v);
   v *= g(x);
}

double FdotG(const Vector & x)
{
   Vector v1(x.Size()), v2(x.Size());

   F(x, v1);
   G(x, v2);

   return v1 * v2;
}

void FcrossG(const Vector & x, Vector & v)
{
   Vector v1(x.Size()), v2(x.Size());

   F(x, v1);
   G(x, v2);

   v.SetSize(3);

   v[0] = v1[1] * v2[2] - v1[2] * v2[1];
   v[1] = v1[2] * v2[0] - v1[0] * v2[2];
   v[2] = v1[0] * v2[1] - v1[1] * v2[0];
}
