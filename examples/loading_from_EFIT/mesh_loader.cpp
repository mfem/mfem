#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "B_field_loader.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "2d_mesh.mesh";
   bool visualization = true;

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   //    load from solution file
   ifstream temp_log("./EFIT_loading/psi.gf");
   GridFunction x(&mesh, temp_log);

   // GridFunction x(&fespace);
   // x = 0.0;
   cout << "Mesh loaded" << endl;

   // saving as vtk
   const int ref = 0;
   ofstream vtk_ofs("psi.vtk");
   mesh.PrintVTK(vtk_ofs, ref);
   x.SaveVTK(vtk_ofs, "x", ref);
   vtk_ofs.close();
   cout << "Mesh saved" << endl;
   
   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }

   return 0;
}
