#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "JxB_vec_coeffs.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *new_mesh_file = "mesh/2d_mesh.mesh";
   bool visualization = true;

   Mesh mesh(new_mesh_file, 1, 1);
   // mesh.UniformRefinement();
   int dim = mesh.Dimension();

   ifstream temp_log("output/B_pol_Hdiv.gf");
   GridFunction B_pol(&mesh, temp_log);

   temp_log.close();                    // Close previous file
   temp_log.open("output/B_tor_DG.gf"); // Open new file
   GridFunction B_tor(&mesh, temp_log);

   temp_log.close();                       // Close previous file
   temp_log.open("output/J_pol_Hcurl.gf"); // Open new file
   GridFunction J_pol(&mesh, temp_log);

   temp_log.close();                      // Close previous file
   temp_log.open("output/J_tor_Hdiv.gf"); // Open new file
   GridFunction J_tor(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   // make a RT space with the mesh
   RT_FECollection fec(0, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // make a grid function with the H1 space
   GridFunction JxB_pol(&fespace);
   cout << JxB_pol.FESpace()->GetTrueVSize() << endl;
   JxB_pol = 0.0;
   LinearForm b(&fespace);
   b.Assemble();
   // project the grid function onto the new space
   // solving (f, B_pol) = (curl f, psi/R e_φ) + <f, n x psi/R e_φ>

   // 1.1.a make the RHS bilinear form for J_pol
   {
      MixedBilinearForm b_bi(J_pol.FESpace(), &fespace);
      BTorPerpMatrixGridFunctionCoefficient B_tor_perp_coef(&B_tor, true);
      b_bi.AddDomainIntegrator(new MixedVectorMassIntegrator(B_tor_perp_coef));
      b_bi.Assemble();

      // 1.1.b form linear form from bilinear form
      LinearForm b_li(&fespace);
      b_bi.Mult(J_pol, b_li);
      b += b_li;
   }

   // 1.2.a make the RHS bilinear form for B_pol
   {
      MixedBilinearForm b_bi(B_pol.FESpace(), &fespace);
      JTorPerpMatrixGridFunctionCoefficient J_tor_perp_coef(&J_tor, false);
      b_bi.AddDomainIntegrator(new MixedVectorMassIntegrator(J_tor_perp_coef));
      b_bi.Assemble();

      // 1.1.b form linear form from bilinear form
      LinearForm b_li(&fespace);
      b_bi.Mult(B_pol, b_li);
      b += b_li;
   }

   // 2. make the bilinear form
   BilinearForm a(&fespace);
   ConstantCoefficient one(1.0);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
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

   Vector X(JxB_pol.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   JxB_pol.SetFromTrueDofs(X);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << mesh << JxB_pol << flush;
   }

   // // paraview
   {
      ParaViewDataCollection paraview_dc("JxB_pol", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("JxB_pol", &JxB_pol);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/JxB_pol.gf");
   sol_ofs.precision(8);
   JxB_pol.Save(sol_ofs);

   return 0;
}
