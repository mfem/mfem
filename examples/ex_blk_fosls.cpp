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
   FiniteElementCollection *fec1 = new ND_FECollection(order, dim);
   FiniteElementCollection *fec2 = new RT_FECollection(order-1, dim);
   FiniteElementCollection *fec3 = new L2_FECollection(order-1, dim);
   FiniteElementSpace fespace0(&mesh, fec0);
   FiniteElementSpace fespace1(&mesh, fec1);
   FiniteElementSpace fespace2(&mesh, fec2);
   FiniteElementSpace fespace3(&mesh, fec3);

   Array<FiniteElementSpace *> fespaces(4);
   fespaces[0] = &fespace0;
   fespaces[1] = &fespace1;
   fespaces[2] = &fespace2;
   fespaces[3] = &fespace3;

   BlockBilinearForm a(fespaces);

   cout << "H1 fespace 0 = " << fespace0.GetVSize() << endl;
   cout << "ND fespace 1 = " << fespace1.GetVSize() << endl;
   cout << "RT fespace 2 = " << fespace2.GetVSize() << endl;
   cout << "L2 fespace 3 = " << fespace3.GetVSize() << endl;
   cout << "Total dofs   = " << fespace0.GetVSize() + fespace1.GetVSize()
        + fespace2.GetVSize() + fespace3.GetVSize()
        << endl;


   cout << "height = " << a.Height() << endl;
   cout << "width = " << a.Width() << endl;

   delete fec0;

   return 0;
}
