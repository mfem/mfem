#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "JxB_vec_coeffs.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "2d_mesh.mesh";
   bool visualization = true;

   Mesh mesh(mesh_file, 1, 1);
   // mesh.UniformRefinement();
   int dim = mesh.Dimension();

   ifstream temp_log("./B_perp.gf");
   GridFunction B_perp(&mesh, temp_log);

   temp_log.close();             // Close previous file
   temp_log.open("./J_perp.gf"); // Open new file
   GridFunction J_perp(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   // make a Hcurl space with the mesh
   // L2_FECollection fec(0, dim);
   H1_FECollection fec(1, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // make a grid function with the H1 space
   GridFunction JxB(&fespace);
   cout << JxB.FESpace()->GetTrueVSize() << endl;
   JxB = 0.0;

   // 1.a make the RHS bilinear form
   MixedBilinearForm b_bi(B_perp.FESpace(), &fespace);
   JPerpRVectorGridFunctionCoefficient J_perp_r_coeff(&J_perp);
   b_bi.AddDomainIntegrator(new MixedScalarCrossProductIntegrator(J_perp_r_coeff));
   b_bi.Assemble();

   // 1.b form linear form from bilinear form
   LinearForm b(&fespace);
   b_bi.Mult(B_perp, b);

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

   Vector X(JxB.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   JxB.SetFromTrueDofs(X);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << mesh << JxB << flush;
   }

   // paraview
   {
      ParaViewDataCollection paraview_dc("JxB", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("JxB", &JxB);
      paraview_dc.Save();
   }

   ofstream sol_ofs("JxB.gf");
   sol_ofs.precision(8);
   JxB.Save(sol_ofs);

   return 0;
}
