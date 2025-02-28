#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "J_field_vec_coeffs_v2.hpp"
#include "JxB_vec_coeffs_v0.hpp"

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

   temp_log.close();            // Close previous file
   temp_log.open("./B_tor.gf"); // Open new file
   GridFunction B_tor(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   // make a Hcurl space with the mesh
   // L2_FECollection fec(0, dim);
   H1_FECollection fec(1, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // make a grid function with the H1 space
   GridFunction B_tor_r(&fespace);
   cout << B_tor_r.FESpace()->GetTrueVSize() << endl;
   B_tor_r = 0.0;
   // make a grid function with the H1 space
   GridFunction JxB(&fespace);
   cout << JxB.FESpace()->GetTrueVSize() << endl;
   JxB = 0.0;

   // A. compute B_tor_r

   {
      BilinearForm b_bi(&fespace);
      RSquareGridFunctionCoefficient r_sq_coef;
      b_bi.AddDomainIntegrator(new MassIntegrator(r_sq_coef));
      b_bi.Assemble();

      // 1.b form linear form from bilinear form
      LinearForm b(&fespace);
      b_bi.Mult(B_tor, b);

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

      Vector X(B_tor_r.Size());
      X = 0.0;
      M_solver.Mult(b, X);

      B_tor_r.SetFromTrueDofs(X);
   }

   { // B. compute JxB
      // 1.a make the RHS bilinear form
      MixedBilinearForm b_bi(B_tor_r.FESpace(), &fespace);
      VectorGridFunctionCoefficient B_perp_coeff(&B_perp);
      b_bi.AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(B_perp_coeff));
      b_bi.Assemble();

      // 1.b form linear form from bilinear form
      LinearForm b(&fespace);
      b_bi.Mult(B_tor_r, b);

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
   }

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
