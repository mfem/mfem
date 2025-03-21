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

   ifstream temp_log("output/B_pol_Hcurl.gf");
   GridFunction B_pol(&mesh, temp_log);

   temp_log.close();            // Close previous file
   temp_log.open("output/B_tor.gf"); // Open new file
   GridFunction B_tor(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   // make a ND space with the mesh
   ND_FECollection fec(1, dim);
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

   // B. compute B_tor \grad B_tor_r
   GridFunction B_tor_grad_B_tor_r(&fespace);
   B_tor_grad_B_tor_r = 0.0;
   cout << B_tor_grad_B_tor_r.FESpace()->GetTrueVSize() << endl;

   {
      // 1.a make the RHS bilinear form
      MixedBilinearForm b_bi(B_tor_r.FESpace(), &fespace);
      GridFunctionCoefficient B_tor_coeff(&B_tor);
      b_bi.AddDomainIntegrator(new MixedVectorGradientIntegrator(B_tor_coeff));
      b_bi.Assemble();

      // 1.b form linear form from bilinear form
      LinearForm b(&fespace);
      b_bi.Mult(B_tor_r, b);

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

      Vector X(B_tor_grad_B_tor_r.Size());
      X = 0.0;
      M_solver.Mult(b, X);
      B_tor_grad_B_tor_r.SetFromTrueDofs(X);
   }

   // C. compute r Curl B_pol
   // make a grid function with the H1 space
   GridFunction R_Curl_B_pol(B_tor.FESpace());
   cout << R_Curl_B_pol.FESpace()->GetTrueVSize() << endl;
   R_Curl_B_pol = 0.0;

   {
      // 1.a make the RHS bilinear form
      MixedBilinearForm b_bi(B_pol.FESpace(), B_tor.FESpace());
      RSquareGridFunctionCoefficient r_sq_coef;
      b_bi.AddDomainIntegrator(new MixedScalarCurlIntegrator(r_sq_coef));
      b_bi.Assemble();

      // 1.b form linear form from bilinear form
      LinearForm b(B_tor.FESpace());
      b_bi.Mult(B_pol, b);

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

      Vector X(R_Curl_B_pol.Size());
      X = 0.0;
      M_solver.Mult(b, X);
      R_Curl_B_pol.SetFromTrueDofs(X);
   }

   // D. (r Curl B_pol) B_pol^perp
   GridFunction R_Curl_B_pol_B_pol_perp(&fespace);
   cout << R_Curl_B_pol_B_pol_perp.FESpace()->GetTrueVSize() << endl;
   R_Curl_B_pol_B_pol_perp = 0.0;
   {
      // 1. make the RHS linear form
      LinearForm b(&fespace);
      RCurlBPerpBPerpPerpVectorGridFunctionCoefficient r_curl_B_pol_B_pol_perp_coef(&R_Curl_B_pol, &B_pol);
      b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(r_curl_B_pol_B_pol_perp_coef));
      b.Assemble();

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

      Vector X(R_Curl_B_pol_B_pol_perp.Size());
      X = 0.0;
      M_solver.Mult(b, X);
      R_Curl_B_pol_B_pol_perp.SetFromTrueDofs(X);
   }

   // E. JxB_pol = B_tor_grad_B_tor_r - R_Curl_B_pol_B_pol_perp
   GridFunction JxB_pol(&fespace);
   JxB_pol = B_tor_grad_B_tor_r;
   JxB_pol -= R_Curl_B_pol_B_pol_perp;

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
      ParaViewDataCollection paraview_dc("JxB_pol", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("JxB_pol", &JxB_pol);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/JxB_pol.gf");
   sol_ofs.precision(8);
   JxB_pol.Save(sol_ofs);

   return 0;
}
