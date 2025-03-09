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
   bool mixed_bilinear_form = false;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ifstream temp_log("./EFIT_loading/p.gf");
   GridFunction p(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   const char *new_mesh_file = "2d_mesh.mesh";
   Mesh *new_mesh = new Mesh(new_mesh_file, 1, 1);

   // load JxB_perp on the new mesh
   temp_log.close(); // close the file
   temp_log.open("./JxB_perp.gf");
   GridFunction JxB_perp(new_mesh, temp_log);

   // Define the vector finite element space for grad p (ND)
   ND_FECollection fec(1, dim);
   FiniteElementSpace fespace(new_mesh, &fec);
   // Initialize gradient field
   GridFunction grad_p(&fespace);
   cout << grad_p.FESpace()->GetTrueVSize() << endl;
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
   // scaled_grad_p += JxB_perp;

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *new_mesh << scaled_grad_p << flush;
   }

   // paraview
   {
      ParaViewDataCollection paraview_dc("scaled_grad_p", new_mesh);
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

   delete new_mesh;
   
   return 0;
}