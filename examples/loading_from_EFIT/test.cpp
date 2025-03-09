#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "B_field_vec_coeffs_v1.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{

   // 1. Parse command-line options.
   const char *new_mesh_file = "2d_mesh.mesh";

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(new_mesh_file, 1, 1);

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   //    load from solution file
   ifstream temp_log("./EFIT_loading/psi.gf");
   GridFunction gf(&mesh, temp_log);

   // // project the grid function onto the new space
   // // 1. make a coefficient from the grid function
   FindPointsGSLIB finder;
   gf.FESpace()->GetMesh()->EnsureNodes();
   finder.Setup(*gf.FESpace()->GetMesh());

   return 0;
}
