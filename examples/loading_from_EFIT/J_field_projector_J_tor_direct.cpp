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
   // mesh.UniformRefinement();
   int dim = mesh.Dimension();

   ifstream temp_log("input/psi.gf");
   GridFunction psi(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   // make a L2 space with the mesh
   H1_FECollection fec(1, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // make a grid function with the H1 space
   GridFunction J_tor(&fespace);
   cout << J_tor.FESpace()->GetTrueVSize() << endl;
   J_tor = 0.0;

   // project the grid function onto the new space
   // solving (f, J_tor) = (curl f, psi/R e_φ) + <f, n x psi/R e_φ>

   // 1.a make the RHS bilinear form
   MixedBilinearForm b_bi(psi.FESpace(), &fespace);
   // ConstantCoefficient neg_one(-1.0);
   OneOverRGridFunctionCoefficient neg_one(true);
   b_bi.AddDomainIntegrator(new MixedGradGradIntegrator(neg_one));
   b_bi.Assemble();

   // 1.b form linear form from bilinear form
   LinearForm b(&fespace);
   b_bi.Mult(psi, b);

   // 2. make the bilinear form
   BilinearForm a(&fespace);
   // RGridFunctionCoefficient r_coef;
   ConstantCoefficient r_coef(1.0);
   a.AddDomainIntegrator(new MassIntegrator(r_coef));
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

   Vector X(J_tor.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   J_tor.SetFromTrueDofs(X);

   // temp_log.close();
   // temp_log.open("output/J_tor.gf");
   // GridFunction J_tor_ori(&mesh, temp_log);

   // J_tor -= J_tor_ori;

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << mesh << J_tor << flush;
   }

   // paraview
   {
      ParaViewDataCollection paraview_dc("J_tor_direct", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("J_tor_direct", &J_tor);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/J_tor_direct.gf");
   sol_ofs.precision(8);
   J_tor.Save(sol_ofs);

   return 0;
}
