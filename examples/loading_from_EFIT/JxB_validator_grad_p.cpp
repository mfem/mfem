#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "JxB_vec_coeffs.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "mesh/2d_mesh.mesh";
   bool visualization = true;
   bool mixed_bilinear_form = false;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ifstream temp_log("input/p.gf");
   GridFunction p(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   const char *new_mesh_file = "mesh/2d_mesh.mesh";
   Mesh *new_mesh = new Mesh(new_mesh_file, 1, 1);

   H1_FECollection scaler_fec(1, dim);
   FiniteElementSpace scaler_fespace(new_mesh, &scaler_fec);
   ND_FECollection fec(1, dim);
   FiniteElementSpace fespace(new_mesh, &fec);
   // First scale p by mu
   GridFunction scaled_p(&scaler_fespace);
   cout << scaled_p.FESpace()->GetTrueVSize() << endl;
   scaled_p = 0.0;

   {
      LinearForm b(&scaler_fespace);
      real_t mu = 4.0 * M_PI * 1e-7;
      if (!mixed_bilinear_form)
      {
         cout << "Using linear form" << endl;
         PRGridFunctionCoefficient scaled_p_r_coef(&p, mu);
         b.AddDomainIntegrator(new DomainLFIntegrator(scaled_p_r_coef));
         b.Assemble();
      }
      else
      {
         cout << "Using bilinear form" << endl;
         MFEM_ASSERT(p.FESpace()->GetMesh()->GetNE() == fespace.GetMesh()->GetNE(), "The two spaces are not on the same mesh");
         // Make the RHS bilinear form for scaling
         BilinearForm b_bi(&scaler_fespace);
         RGridFunctionCoefficient scaled_r_coef(mu);
         b_bi.AddDomainIntegrator(new MassIntegrator(scaled_r_coef));
         b_bi.Assemble();

         // Form linear form from bilinear form
         b_bi.Mult(p, b);
      }

      // 2. make the bilinear form
      BilinearForm a(&scaler_fespace);
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

      Vector X(scaled_p.Size());
      X = 0.0;
      M_solver.Mult(b, X);

      scaled_p.SetFromTrueDofs(X);
   }
   // Now compute gradient of scaled p
   GridFunction grad_p(&fespace);
   grad_p = 0.0;

   {
      // Make the RHS bilinear form for gradient
      MixedBilinearForm b_bi(scaled_p.FESpace(), &fespace);
      RGridFunctionCoefficient r_coef;
      b_bi.AddDomainIntegrator(new MixedVectorGradientIntegrator(r_coef));
      b_bi.Assemble();

      // Form linear form from bilinear form
      LinearForm b(&fespace);
      b_bi.Mult(scaled_p, b);

      // Make the mass matrix
      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new VectorFEMassIntegrator(r_coef));
      a.Assemble();
      a.Finalize();

      // Solve the system
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

   // load JxB_pol on the new mesh
   temp_log.close(); // close the file
   temp_log.open("output/JxB_pol.gf");
   GridFunction JxB_pol(new_mesh, temp_log);

   cout << grad_p.FESpace()->GetTrueVSize() << endl;
   cout << JxB_pol.FESpace()->GetTrueVSize() << endl;
   grad_p += JxB_pol;

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *new_mesh << grad_p << flush;
   }

   // paraview
   {
      ParaViewDataCollection paraview_dc("grad_p", new_mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("grad_p", &grad_p);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/grad_p.gf");
   sol_ofs.precision(8);
   grad_p.Save(sol_ofs);

   delete new_mesh;

   return 0;
}