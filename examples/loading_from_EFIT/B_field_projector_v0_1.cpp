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
   Mesh *new_mesh = new Mesh(Mesh::MakeCartesian2D(500, 750, Element::QUADRILATERAL));

   // translate to 1.0 in x direction
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
   B_tor = 0.0;

   // project the grid function onto the new space
   // solving <B_tor, v> = <gg/R, v> for all v in L2

   // 1. make the linear form
   LinearForm b(&fespace);
   BTorFromFGridFunctionCoefficient gg_coef(&gg);
   b.AddDomainIntegrator(new DomainLFIntegrator(gg_coef));
   b.Assemble();

   // 2. make the bilinear form
   BilinearForm a(&fespace);
   ConstantCoefficient one(1.0);
   a.AddDomainIntegrator(new MassIntegrator(one));
   a.Assemble();
   a.Finalize();

   // 3. solve the system
   CGSolver M_solver;
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-24);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(1e5);
   M_solver.SetPrintLevel(1);
   M_solver.SetOperator(a.SpMat());

   Vector X(B_tor.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   B_tor.SetFromTrueDofs(X);

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

   // paraview
   {
      ParaViewDataCollection paraview_dc("Bperp", new_mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("Bperp", &B_tor);
      paraview_dc.Save();
   }

   ofstream sol_ofs("Bperp.gf");
   sol_ofs.precision(8);
   B_tor.Save(sol_ofs);

   delete new_mesh;

   return 0;
}
