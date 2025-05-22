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

   ifstream temp_log("output/B_pol_vec_CG.gf");
   GridFunction B_pol(&mesh, temp_log);

   temp_log.close();                    // Close previous file
   temp_log.open("output/B_tor_CG.gf"); // Open new file
   GridFunction B_tor(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   // make a H1 space with the mesh
   H1_FECollection fec(1, dim);
   FiniteElementSpace fespace(&mesh, &fec, 2);

   
   GridFunction JxB_pol(&fespace);
   cout << JxB_pol.FESpace()->GetTrueVSize() << endl;
   JxB_pol = 0.0;

   // 1. make the linear form
   LinearForm b(&fespace);
   BTorGradRBTorVectorGridFunctionCoefficient B_tor_grad_r_B_tor_coef(&B_tor);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(B_tor_grad_r_B_tor_coef));
   CurlBPolBPolPerpRGridFunctionCoefficient neg_curl_B_pol_B_pol_perp_r_coef(&B_pol, true);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(neg_curl_B_pol_B_pol_perp_r_coef));
   b.Assemble();

   // 2. make the bilinear form
   BilinearForm a(&fespace);
   RGridFunctionCoefficient r_coeff;
   a.AddDomainIntegrator(new VectorMassIntegrator(r_coeff));
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

   // paraview
   {
      ParaViewDataCollection paraview_dc("JxB_pol_C", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("JxB_pol_C", &JxB_pol);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/JxB_pol_C.gf");
   sol_ofs.precision(8);
   JxB_pol.Save(sol_ofs);

   return 0;
}
