#include "JxB_vec_coeffs.hpp" // Include for RGridFunctionCoefficient
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "2d_mesh.mesh";
   bool visualization = true;

   // Read the mesh
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Load the pressure solution
   ifstream temp_log("./EFIT_loading/p.gf");
   GridFunction p(&mesh, temp_log);

   temp_log.close();               // Close previous file
   temp_log.open("./JxB_perp.gf"); // Open new file
   GridFunction JxB_perp(&mesh, temp_log);

   // Define the vector finite element space for grad p (ND)
   ND_FECollection fec(1, dim);
   FiniteElementSpace fespace(&mesh, &fec);
   // Initialize gradient field
   GridFunction grad_p(&fespace);
   grad_p = 0.0;

   // compute grad p
   {
      // 1.a make the RHS bilinear form
      MixedBilinearForm b_bi(p.FESpace(), &fespace);
      RGridFunctionCoefficient r_coef;
      b_bi.AddDomainIntegrator(new MixedVectorGradientIntegrator(r_coef));
      b_bi.Assemble();

      // 1.b form linear form from bilinear form
      LinearForm b(&fespace);
      b_bi.Mult(p, b);

      // 2. make the bilinear form
      BilinearForm a(&fespace);
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

      Vector X(grad_p.Size());
      X = 0.0;
      M_solver.Mult(b, X);
      grad_p.SetFromTrueDofs(X);
   }

   // mu * grad_p
   GridFunction scaled_grad_p(&fespace);
   scaled_grad_p = 0.0;
   cout << scaled_grad_p.FESpace()->GetTrueVSize() << endl;

   {
      // 1.a make the RHS bilinear form
      BilinearForm b_bi(&fespace);
      real_t mu = 4.0 * M_PI * 1e-7;
      RGridFunctionCoefficient scaled_r_coef(mu);
      b_bi.AddDomainIntegrator(new VectorFEMassIntegrator(scaled_r_coef));
      b_bi.Assemble();

      // 1.b form linear form from bilinear form
      LinearForm b(&fespace);
      b_bi.Mult(grad_p, b);

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

      Vector X(scaled_grad_p.Size());
      X = 0.0;
      M_solver.Mult(b, X);
      scaled_grad_p.SetFromTrueDofs(X);
   }
   cout << scaled_grad_p.FESpace()->GetTrueVSize() << endl;
   cout << JxB_perp.FESpace()->GetTrueVSize() << endl;
   scaled_grad_p += JxB_perp;

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << mesh << scaled_grad_p << flush;
   }

   // paraview
   {
      ParaViewDataCollection paraview_dc("scaled_grad_p", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("scaled_grad_p", &scaled_grad_p);
      paraview_dc.Save();
   }

   ofstream sol_ofs("scaled_grad_p.gf");
   sol_ofs.precision(8);
   scaled_grad_p.Save(sol_ofs);

   return 0;
}