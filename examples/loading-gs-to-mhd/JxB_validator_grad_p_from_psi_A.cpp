#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "vec_coeffs.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "mesh/2d_mesh.mesh";
   bool visualization = true;
   bool mixed_bilinear_form = false;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ifstream temp_log("input/psi.gf");
   GridFunction psi(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   const char *new_mesh_file = "mesh/2d_mesh.mesh";
   Mesh *new_mesh = new Mesh(new_mesh_file, 1, 1);

   H1_FECollection scaler_fec(1, dim);
   FiniteElementSpace scaler_fespace(new_mesh, &scaler_fec);
   RT_FECollection fec(0, dim);
   FiniteElementSpace fespace(new_mesh, &fec);

   // Now compute gradient of scaled p
   GridFunction grad_p(&fespace);
   grad_p = 0.0;

   {
      LinearForm b(&fespace);
   
      GradPRVectorGridFunctionCoefficient grad_p_r_coef(&psi, false);
      b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(grad_p_r_coef));
      b.Assemble();

      // Make the mass matrix
      BilinearForm a(&fespace);
      RGridFunctionCoefficient r_coef;
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
   temp_log.open("output/JxB_pol_A.gf");
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
      ParaViewDataCollection paraview_dc("grad_p_A", new_mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("grad_p_A", &grad_p);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/grad_p_A.gf");
   sol_ofs.precision(8);
   grad_p.Save(sol_ofs);

   delete new_mesh;

   return 0;
}