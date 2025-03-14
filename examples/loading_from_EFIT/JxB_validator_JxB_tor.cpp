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

   ifstream temp_log("output/B_perp.gf");
   GridFunction B_perp(&mesh, temp_log);

   temp_log.close();            // Close previous file
   temp_log.open("output/B_tor.gf"); // Open new file
   GridFunction B_tor(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   // make a Hcurl space with the mesh
   // L2_FECollection fec(0, dim);
   H1_FECollection fec(1, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // A. compute B_tor_r
   // make a grid function with the H1 space
   GridFunction B_tor_r(B_tor.FESpace());
   cout << B_tor_r.FESpace()->GetTrueVSize() << endl;
   B_tor_r = 0.0;

   {
      BilinearForm b_bi(B_tor.FESpace());
      RSquareGridFunctionCoefficient r_sq_coef;
      b_bi.AddDomainIntegrator(new MassIntegrator(r_sq_coef));
      b_bi.Assemble();

      // 1.b form linear form from bilinear form
      LinearForm b(B_tor.FESpace());
      b_bi.Mult(B_tor, b);

      // 2. make the bilinear form
      BilinearForm a(B_tor.FESpace());
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

   // B. compute JxB
   // make a grid function with the H1 space
   GridFunction JxB_tor(&fespace);
   cout << JxB_tor.FESpace()->GetTrueVSize() << endl;
   JxB_tor = 0.0;

   {
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

      Vector X(JxB_tor.Size());
      X = 0.0;
      M_solver.Mult(b, X);
      JxB_tor.SetFromTrueDofs(X);
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << mesh << JxB_tor << flush;
   }

   // paraview
   {
      ParaViewDataCollection paraview_dc("JxB_tor", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("JxB_tor", &JxB_tor);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/JxB_tor.gf");
   sol_ofs.precision(8);
   JxB_tor.Save(sol_ofs);

   return 0;
}
