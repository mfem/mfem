#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "J_field_vec_coeffs.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "mesh/new_2d_mesh.mesh";
   bool visualization = true;
   bool project_mesh = true;
   bool from_psi = true;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ifstream temp_log("input/new_psi.gf");
   GridFunction temp_gg(&mesh, temp_log);

   cout << "Mesh loaded" << endl;

   const char *new_mesh_file = "mesh/new_2d_mesh.mesh";
   Mesh *new_mesh = new Mesh(new_mesh_file, 1, 1);

   // refine the mesh
   // new_mesh->UniformRefinement();

   // make a Hcurl space with the mesh
   // L2_FECollection fec(0, dim);
   RT_FECollection fec(0, dim);
   FiniteElementSpace fespace(new_mesh, &fec);
   H1_FECollection scalar_fec(1, dim);
   FiniteElementSpace scalar_fespace(new_mesh, &scalar_fec);
   GridFunction gg(&scalar_fespace);

   if (project_mesh)
   {
      GridFunction gg_projected(&scalar_fespace);

      // 1. make the linear form
      LinearForm b(&scalar_fespace);
      FGridFunctionCoefficient f_coef(&temp_gg, from_psi);
      b.AddDomainIntegrator(new DomainLFIntegrator(f_coef));
      b.Assemble();

      // 2. make the bilinear form
      BilinearForm a(&scalar_fespace);
      ConstantCoefficient one(1.0);
      a.AddDomainIntegrator(new MassIntegrator(one));
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

      Vector X(gg_projected.Size());
      X = 0.0;
      M_solver.Mult(b, X);

      gg_projected.SetFromTrueDofs(X);

      gg = gg_projected;
   }
   else
   {
      MFEM_ASSERT(temp_gg.FESpace()->GetMesh()->GetNE() == scalar_fespace.GetMesh()->GetNE(), "The two spaces are not on the same mesh");
      gg = temp_gg;
   }

   // make a grid function with the H1 space
   GridFunction J_pol(&fespace);
   cout << J_pol.FESpace()->GetTrueVSize() << endl;
   J_pol = 0.0;

   LinearForm b(&fespace);

   // project the grid function onto the new space
   // solving (f, J_pol) = (curl f, gg/R e_φ) + <f, n x gg/R e_φ>

   // 1.a make the RHS bilinear form
   MixedBilinearForm b_bi(gg.FESpace(), &fespace);
   DenseMatrix perp_rotation(dim);
   perp_rotation(0, 0) = 0.0;
   perp_rotation(0, 1) = -1.0;
   perp_rotation(1, 0) = 1.0;
   perp_rotation(1, 1) = 0.0;
   MatrixConstantCoefficient perp_rot_coef(perp_rotation);
   b_bi.AddDomainIntegrator(new MixedVectorGradientIntegrator(perp_rot_coef));
   b_bi.Assemble();

   // 1.b form linear form from bilinear form
   LinearForm b_li(&fespace);
   b_bi.Mult(gg, b_li);
   b.Assemble();
   b += b_li;

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

   Vector X(J_pol.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   J_pol.SetFromTrueDofs(X);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *new_mesh << J_pol << flush;
   }

   // paraview
   {
      ParaViewDataCollection paraview_dc("J_pol_direct", new_mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("J_pol_direct", &J_pol);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/J_pol_direct.gf");
   sol_ofs.precision(8);
   J_pol.Save(sol_ofs);

   delete new_mesh;

   return 0;
}