
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "patch_gen.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/star.mesh";
   // const char *mesh_file = "../../../data/beam-quad.mesh";
   int order = 1;
   bool static_cond = false;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels = 0;
         // (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   // FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace * fespace = new FiniteElementSpace(mesh, fec);
   patch_assembly * p = new patch_assembly(fespace); 
   
   cout<< "TrueVsize =" << fespace->GetTrueVSize() << endl;

   // Table elem_table = fespace->GetElementToDofTable();
   // elem_table.Print();
   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      string keys;
      keys = "keys nn\n";
      sol_sock << "mesh\n" << *mesh << keys << flush;
   }

   // 15. Free the used memory.
   delete fec;
   delete fespace;
   delete p;
   delete mesh;
   return 0;
}
