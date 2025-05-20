#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "B_field_vec_coeffs.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *new_mesh_file = "mesh/2d_mesh.mesh";
   bool visualization = true;

   Mesh mesh(new_mesh_file, 1, 1);
   // mesh.UniformRefinement();
   int dim = mesh.Dimension();

   ifstream temp_log("output/B_pol_Hdiv.gf");
   GridFunction B_pol(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   // make a H1 space with the mesh
   H1_FECollection fec(1, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // make a grid function with the H1 space
   GridFunction div_B_pol(&fespace);
   cout << div_B_pol.FESpace()->GetTrueVSize() << endl;
   div_B_pol = 0.0;
   LinearForm b(&fespace);

   // 1.a make the RHS bilinear form
   MixedBilinearForm b_bi(B_pol.FESpace(), &fespace);
   RGridFunctionCoefficient r_coef(false);
   b_bi.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(r_coef));
   Vector one_zero(2);
   one_zero(0) = 1.0;
   one_zero(1) = 0.0;
   VectorConstantCoefficient one_zero_coef(one_zero);
   b_bi.AddDomainIntegrator(new MixedDotProductIntegrator(one_zero_coef));
   b_bi.Assemble();

   // 1.b form linear form from bilinear form
   LinearForm b_li(&fespace);
   b_bi.Mult(B_pol, b_li);
   BPolRVectorGridFunctionCoefficient B_pol_r(&B_pol);
   b.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(B_pol_r));
   b.Assemble();
   b += b_li;

   // 2. make the bilinear form
   BilinearForm a(&fespace);
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

   Vector X(div_B_pol.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   div_B_pol.SetFromTrueDofs(X);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << mesh << div_B_pol << flush;
   }

   // paraview
   {
      ParaViewDataCollection paraview_dc("div_B_pol_Hdiv", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("div_B_pol_Hdiv", &div_B_pol);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/div_B_pol_Hdiv.gf");
   sol_ofs.precision(8);
   div_B_pol.Save(sol_ofs);

   return 0;
}
