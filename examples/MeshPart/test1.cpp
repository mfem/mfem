
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options
   const char *mesh_file = "../../data/periodic-annulus-sector.msh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh orig_mesh(mesh_file);
   orig_mesh.CheckElementOrientation(true);
   orig_mesh.CheckBdrElementOrientation(true);

   // mesh.EnsureNodes();
   // mesh.UniformRefinement();
   // Array<int> elems({1,3,21,10,20,2,0});
   Array<int> elems({0,1,2,3});
   // int nel = orig_mesh.GetNE();
   int nel = elems.Size();
   // Array<int> elems(nel);
   // for (int i = 0; i<nel; i++)
   // {
      // elems[i] = i;
   // }
   // elems.Print();

   Mesh new_mesh = Mesh::ExtractMesh(orig_mesh,elems);
   new_mesh.CheckElementOrientation(true);
   new_mesh.CheckBdrElementOrientation(true);
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mesh0_sock(vishost, visport);
      mesh0_sock.precision(8);
      mesh0_sock << "mesh\n" << orig_mesh << "keys n \n" << flush;

      socketstream mesh1_sock(vishost, visport);
      mesh1_sock.precision(8);
      mesh1_sock << "mesh\n" << new_mesh << "keys n \n" << flush;
   }

   Array<int> faces;
   for (int i = 0; i<orig_mesh.GetNBE(); i++)
   {
      if (orig_mesh.GetBdrAttribute(i) >= 1)
         faces.Append(orig_mesh.GetBdrFace(i));
   }
      // faces.Append(orig_mesh.GetBdrFace(1));
      // faces.Append(orig_mesh.GetBdrFace(2));

   Mesh surface_mesh = Mesh::ExtractSurfaceMesh(orig_mesh,faces);
   surface_mesh.CheckElementOrientation(true);
   surface_mesh.CheckBdrElementOrientation(true);

   surface_mesh.Print();

   if (surface_mesh.Dimension() > 1)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream mesh2_sock(vishost, visport);
      mesh2_sock.precision(8);
      mesh2_sock << "mesh\n" << surface_mesh << "keys n \n" << flush;
   }
   else
   {
      ParaViewDataCollection paraview_dc("surf_mesh", &surface_mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(3);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      H1_FECollection fec(order,surface_mesh.Dimension());
      FiniteElementSpace fespace(&surface_mesh,&fec);
      GridFunction gf(&fespace);
      gf.Randomize();
      paraview_dc.RegisterField("solution",&gf);
      paraview_dc.Save();
   }

   return 0;
}

