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
   // mesh.UniformRefinement();
   int dim = mesh.Dimension();

   ifstream temp_log("output/B_pol_vec_CG.gf");
   GridFunction B_pol(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   // make a H1 space with the mesh
   H1_FECollection fec(1, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   
   GridFunction J_tor(&fespace);
   cout << J_tor.FESpace()->GetTrueVSize() << endl;
   J_tor = 0.0;

   // 1. make the linear form
   LinearForm b(&fespace);
   CurlBPolRGridFunctionCoefficient neg_curl_B_pol_r_coef(&B_pol, true);
   b.AddDomainIntegrator(new DomainLFIntegrator(neg_curl_B_pol_r_coef));
   b.Assemble();

   // 2. make the bilinear form
   BilinearForm a(&fespace);
   RGridFunctionCoefficient r_coef;
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
      ParaViewDataCollection paraview_dc("J_tor_vec_CG", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("J_tor_vec_CG", &J_tor);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/J_tor_vec_CG.gf");
   sol_ofs.precision(8);
   J_tor.Save(sol_ofs);

   return 0;
}
