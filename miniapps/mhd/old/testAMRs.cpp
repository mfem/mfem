#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

double InitialPsi(const Vector &x)
{
   double beta=1e-3, lambda=.5/M_PI, ep=.2;
   return -lambda*log( cosh(x(1)/lambda) +ep*cos(x(0)/lambda) )
          +beta*cos(M_PI*.5*x(1))*cos(M_PI*x(0));
}

int main(int argc, char *argv[])
{
   //++++Parse command-line options.
   const char *mesh_file = "./amr-periodic.mesh";
   int ser_ref_levels = 2;
   int order = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   //+++++Read the mesh from the given mesh file.    
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   //++++++Refine the mesh to increase the resolution.    
   mesh->EnsureNCMesh();
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll); 

   GridFunction psi(&fespace);
   FunctionCoefficient psiInit(InitialPsi);
   psi.ProjectCoefficient(psiInit);
   psi.SetTrueVector();
   psi.SetFromTrueVector(); 

   socketstream vis_phi;
   char vishost[] = "localhost";
   int  visport   = 19916;
   vis_phi.open(vishost, visport);

   vis_phi.precision(8);
   vis_phi << "solution\n" << *mesh << psi;
   vis_phi << "window_size 800 800\n"<< "window_title '" << "psi'" << "keys cm\n";
   vis_phi << flush;

   delete mesh;
   return 0;
}



