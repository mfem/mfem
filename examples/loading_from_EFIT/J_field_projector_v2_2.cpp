#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "J_field_vec_coeffs_v2.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "2d_mesh.mesh";
   bool visualization = true;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ifstream temp_log("./B_tor.gf");
   GridFunction psi(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   // r: 3.0:10.0:256, z: -6.0:6.0:512
   // Use Cartesian coordinates for the extrusion
   Mesh *new_mesh = new Mesh(Mesh::MakeCartesian2D(256, 512, Element::QUADRILATERAL));

   // translate to 1.0 in x direction
   new_mesh->Transform([](const Vector &x, Vector &p)
                       { p[0] = x[0]* ((10.0 - 7.0 / 514) - (3.0 + 7.0 / 514)) + 3.0 + 7.0 / 514; p[1] = x[1]* ((6.0 - 12.0 / 1026) - (-6.0 + 12.0 / 1026)) - 6.0 + 12.0 / 1026; });

   // refine the mesh
   // new_mesh->UniformRefinement();

   // make a Hcurl space with the mesh
   // L2_FECollection fec(0, dim);
   ND_FECollection fec(1, dim);
   FiniteElementSpace fespace(new_mesh, &fec);

   // make a grid function with the H1 space
   GridFunction J_perp(&fespace);
   cout << J_perp.FESpace()->GetTrueVSize() << endl;
   J_perp = 0.0;

   // project the grid function onto the new space
   // solving (f, Jperp) = (curl f, psi/R e_φ) + <f, n x psi/R e_φ>

   // 1. make the linear form
   LinearForm b(&fespace);
   JPerpBRGridFunctionCoefficient B_tor_r_coef(dim, &psi, false);
   b.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(B_tor_r_coef));

   JPerpBRGridFunctionCoefficient neg_B_tor_r_coef(dim, &psi, true);
   b.AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(neg_B_tor_r_coef));
   b.Assemble();

   // 2. make the bilinear form
   BilinearForm a(&fespace);
   RGridFunctionCoefficient r_coef;
   a.AddDomainIntegrator(new VectorFEMassIntegrator(r_coef));
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

   Vector X(J_perp.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   J_perp.SetFromTrueDofs(X);

   // ifstream temp_log2("./EFIT_loading/B_phi.gf");
   // GridFunction B_psi(&mesh, temp_log2);

   // GridFunction J_perp_diff(&fespace);
   // J_perp_diff = J_perp;
   // J_perp_diff -= B_psi;

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *new_mesh << J_perp << flush;
   }

   // paraview
   {
      ParaViewDataCollection paraview_dc("Jperp", new_mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("Jperp", &J_perp);
      paraview_dc.Save();
   }

   ofstream sol_ofs("Jperp.gf");
   sol_ofs.precision(8);
   J_perp.Save(sol_ofs);

   delete new_mesh;

   return 0;
}
