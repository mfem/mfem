#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "vec_coeffs.hpp"

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

   // make a vector CG space with the mesh
   // L2_FECollection fec(0, dim);
   H1_FECollection fec(1, dim);
   FiniteElementSpace fespace(&mesh, &fec, 2);

   GridFunction J_pol(&fespace);
   cout << J_pol.FESpace()->GetTrueVSize() << endl;
   J_pol = 0.0;

   // 1. make the linear form
   LinearForm b(&fespace);
   CurlRBTorVectorGridFunctionCoefficient curl_r_B_tor_coef(&B_tor, true);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(curl_r_B_tor_coef));
   b.Assemble();

   // 2. make the bilinear form
   BilinearForm a(&fespace);
   RGridFunctionCoefficient r_coef;
   a.AddDomainIntegrator(new VectorMassIntegrator(r_coef));
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

   // paraview
   {
      ParaViewDataCollection paraview_dc("J_pol_vec_CG", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("J_pol_vec_CG", &J_pol);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/J_pol_vec_CG.gf");
   sol_ofs.precision(8);
   J_pol.Save(sol_ofs);

   return 0;
}