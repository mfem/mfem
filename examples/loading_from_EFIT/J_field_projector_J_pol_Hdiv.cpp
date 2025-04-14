#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "J_field_vec_coeffs.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *new_mesh_file = "mesh/2d_mesh.mesh";
   bool visualization = true;

   Mesh mesh(new_mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ifstream temp_log("output/B_tor_CG.gf");
   GridFunction B_tor(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   // make a Hcurl space with the mesh
   // L2_FECollection fec(0, dim);
   RT_FECollection fec(0, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // make a grid function with the H1 space
   GridFunction J_pol(&fespace);
   cout << J_pol.FESpace()->GetTrueVSize() << endl;
   J_pol = 0.0;
   LinearForm b(&fespace);

   // project the grid function onto the new space
   // solving (f, B_pol) = (curl f, psi/R e_φ) + <f, n x psi/R e_φ>

   // 1.a make the RHS bilinear form
   MixedBilinearForm b_bi(B_tor.FESpace(), &fespace);
   RPerpMatrixGridFunctionCoefficient r_perp_coef(true);
   b_bi.AddDomainIntegrator(new MixedVectorGradientIntegrator(r_perp_coef));
   Vector zero_one(2);
   zero_one(0) = 0.0;
   zero_one(1) = 1.0;
   VectorConstantCoefficient zero_one_coef(zero_one);
   b_bi.AddDomainIntegrator(new MixedVectorProductIntegrator(zero_one_coef));
   b_bi.Assemble();

   // 1.b form linear form from bilinear form
   LinearForm b_li(&fespace);
   b_bi.Mult(B_tor, b_li);
   b.Assemble();
   b += b_li;

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

   Vector X(J_pol.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   J_pol.SetFromTrueDofs(X);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << mesh << J_pol << flush;
   }

   // // paraview
   {
      ParaViewDataCollection paraview_dc("J_pol_Hdiv", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("J_pol_Hdiv", &J_pol);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/J_pol_Hdiv.gf");
   sol_ofs.precision(8);
   J_pol.Save(sol_ofs);

   return 0;
}
