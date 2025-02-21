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
   bool bilinear_form = true;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ifstream temp_log("./EFIT_loading/psi.gf");
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
   GridFunction B_perp(&fespace);
   cout << B_perp.FESpace()->GetTrueVSize() << endl;
   B_perp = 0.0;
   LinearForm b(&fespace);
   if (!bilinear_form)
   {
      cout << "Using linear form" << endl;
      // project the grid function onto the new space
      // solving (f, B_perp) = (curl f, psi/R e_φ) + <f, n x psi/R e_φ>

      // 1. make the linear form
      BPerpPsiGridFunctionCoefficient psi_coef(dim, &psi, false);
      b.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(psi_coef));

      BPerpPsiGridFunctionCoefficient neg_psi_coef(dim, &psi, true);
      b.AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(neg_psi_coef));
      b.Assemble();
   }
   else
   {
      cout << "Using bilinear form" << endl;
      // project the grid function onto the new space
      // solving (f, B_perp) = (curl f, psi/R e_φ) + <f, n x psi/R e_φ>

      // 1.a make the RHS bilinear form
      MixedBilinearForm b_bi(psi.FESpace(), &fespace);
      ConstantCoefficient one(1.0);
      b_bi.AddDomainIntegrator(new MixedScalarWeakCurlIntegrator(one));
      b_bi.Assemble();

      // 1.b form linear form from bilinear form
      LinearForm b_li(&fespace);
      b_bi.Mult(psi, b_li);
      BPerpPsiGridFunctionCoefficient neg_psi_coef(dim, &psi, true);
      b.AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(neg_psi_coef));
      b.Assemble();
      b += b_li;
   }
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

   Vector X(B_perp.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   B_perp.SetFromTrueDofs(X);

   // ifstream temp_log2("./EFIT_loading/B_phi.gf");
   // GridFunction B_psi(&mesh, temp_log2);

   // GridFunction B_perp_diff(&fespace);
   // B_perp_diff = B_perp;
   // B_perp_diff -= B_psi;

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *new_mesh << B_perp << flush;
   }

   // paraview
   {
      ParaViewDataCollection paraview_dc("B_perp", new_mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("B_perp", &B_perp);
      paraview_dc.Save();
   }

   ofstream sol_ofs("B_perp.gf");
   sol_ofs.precision(8);
   B_perp.Save(sol_ofs);

   delete new_mesh;

   return 0;
}
