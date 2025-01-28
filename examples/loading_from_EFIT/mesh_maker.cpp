//
// Put it in mfem/example and compile with: make extruder_tokamak
//
// Sample run:      extruder_tokamak -nz 4 -hz 3 -o 2
// This works well: extruder_tokamak -nz 10 -hz 5 -trans -o 2
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double hz = 3.0;
void trans3D(const Vector &, Vector &);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int order = 3;
   int nz = 10; // nz = 3 is the minimal value for a periodic mesh
   bool trans = true;
   bool visualization = false;

   // Parse command line
   OptionsParser args(argc, argv);

   // r: 3.0:10.0:256, z: -6.0:6.0:512
   // Use Cartesian coordinates for the extrusion
   Mesh *mesh = new Mesh(Mesh::MakeCartesian2D(256, 512, Element::QUADRILATERAL));

   // translate to 1.0 in x direction
   // Upper right corner of the mesh is at 10 - 7.0 / 514, 6.0 - 12.0 / 1026
   // Lower left corner of the mesh is at 3.0 + 7.0 / 514, -6.0 + 12.0 / 1026

   // mesh->Transform([](const Vector &x, Vector &p)
   //                 { p[0] += 1.0; p[1] += 2.0; });
   mesh->Transform([](const Vector &x, Vector &p)
                   { p[0] = x[0]* ((10.0 - 7.0 / 514) - (3.0 + 7.0 / 514)) + 3.0 + 7.0 / 514; p[1] = x[1]* ((6.0 - 12.0 / 1026) - (-6.0 + 12.0 / 1026)) - 6.0 + 12.0 / 1026; });

   ofstream mesh_ofs("2d_mesh.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   mesh_ofs.close();
   mesh->PrintVTU("2d_mesh");

   int dim = mesh->Dimension();

   switch (dim)
   {
   case 2:
      nz = (nz < 3) ? 3 : nz;
      break;
   default:
      cout << "Extruding " << dim << "D meshes is not (yet) supported."
           << endl;
      delete mesh;
      return 1;
   }

   // Determine the order to use for a transformed mesh
   int meshOrder = 1;
   if (mesh->GetNodalFESpace() != NULL)
   {
      meshOrder = mesh->GetNodalFESpace()->GetElementOrder(0);
   }

   cout << "Extruding 2D mesh to a height of " << hz
        << " using " << nz << " elements." << endl;

   Mesh *mesh3d = Extrude2D(mesh, nz, hz);
   delete mesh;
   mesh = mesh3d;
   if (order != meshOrder)
   {
      mesh->SetCurvature(order, true, 3, Ordering::byVDIM);
   }

   // Make the extruded mesh periodic in z
   std::vector<Vector> translations =
       {
           Vector({0.0, 0.0, hz})};
   Mesh newmesh = Mesh::MakePeriodic(
       *mesh,
       mesh->CreatePeriodicVertexMapping(translations));
   newmesh.RemoveInternalBoundaries();
   if (trans)
   {
      newmesh.Transform(trans3D);
   }

   if (visualization)
   {
      cout << "Visualize extruded mesh vs extruded periodic mesh" << endl;
      // GLVis server to visualize to
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n"
               << *mesh << flush;
      socketstream sol_sock1(vishost, visport);
      sol_sock1.precision(8);
      sol_sock1 << "mesh\n"
                << newmesh << flush;
   }

   // Save the final mesh
   mesh_ofs = ofstream("3d_mesh.mesh");
   mesh_ofs.precision(8);
   newmesh.Print(mesh_ofs);
   mesh_ofs.close();
   newmesh.PrintVTU("3d_mesh");

   return 0;
}

void trans3D(const Vector &x, Vector &p)
{
   // a right oriented transformation: (x cos(2pi z / z_max), y, -x sin(2pi z / z_max))
   p[0] = x[0] * cos(2.0 * M_PI * x[2] / hz);
   p[2] = x[1];
   p[1] = -x[0] * sin(2.0 * M_PI * x[2] / hz);
}
