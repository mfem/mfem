#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "B_field_vec_coeffs_v1.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "2d_mesh.mesh";
   bool visualization = true;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ifstream temp_log("./EFIT_loading/gg.gf");
   GridFunction gg(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   // r: 3.0:10.0:256, z: -6.0:6.0:512
   // Use Cartesian coordinates for the extrusion
   Mesh *new_mesh = new Mesh(Mesh::MakeCartesian2D(256, 512, Element::QUADRILATERAL));

   // translate to 1.0 in x direction
   // Upper right corner of the mesh is at 10 - 7.0 / 514, 6.0 - 12.0 / 1026
   // Lower left corner of the mesh is at 3.0 + 7.0 / 514, -6.0 + 12.0 / 1026

   // mesh->Transform([](const Vector &x, Vector &p)
   //                 { p[0] += 1.0; p[1] += 2.0; });
   new_mesh->Transform([](const Vector &x, Vector &p)
                       { p[0] = x[0]* ((10.0 - 7.0 / 514) - (3.0 + 7.0 / 514)) + 3.0 + 7.0 / 514; p[1] = x[1]* ((6.0 - 12.0 / 1026) - (-6.0 + 12.0 / 1026)) - 6.0 + 12.0 / 1026; });

   // refine the mesh
   // new_mesh->UniformRefinement();

   // make a H1 space with the mesh
   H1_FECollection fec(1, dim);
   FiniteElementSpace fespace(new_mesh, &fec);

   // make a grid function with the H1 space
   GridFunction B_tor(&fespace);
   cout << B_tor.FESpace()->GetTrueVSize() << endl;

   // project the grid function onto the new space
   // 1. make a coefficient from the grid function
   BTorFOverRGridFunctionCoefficient gg_coef(&gg);
   B_tor.ProjectCoefficient(gg_coef);

   // ifstream temp_log2("./EFIT_loading/B_phi.gf");
   // GridFunction B_psi(&mesh, temp_log2);

   // GridFunction B_tor_diff(&fespace);
   // B_tor_diff = B_tor;
   // B_tor_diff -= B_psi;

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *new_mesh << B_tor << flush;
   }

   delete new_mesh;

   return 0;
}
