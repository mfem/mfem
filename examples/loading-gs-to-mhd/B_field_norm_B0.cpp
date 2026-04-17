#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "vec_coeffs.hpp"

using namespace std;
using namespace mfem;

GridFunction computeBPol(GridFunction &temp_psi,
                         FiniteElementSpace &scalar_fespace,
                         FiniteElementSpace &vector_fespace,
                         bool project_mesh)
{
   int dim = vector_fespace.GetMesh()->Dimension();
   GridFunction psi(&scalar_fespace);

   if (project_mesh)
   {
      GridFunction psi_projected(&scalar_fespace);

      // 1. make the linear form
      LinearForm b(&scalar_fespace);
      PsiRGridFunctionCoefficient psi_r_coef(&temp_psi, false);
      b.AddDomainIntegrator(new DomainLFIntegrator(psi_r_coef));
      b.Assemble();

      // 2. make the bilinear form
      BilinearForm a(&scalar_fespace);
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

      Vector X(psi_projected.Size());
      X = 0.0;
      M_solver.Mult(b, X);

      psi_projected.SetFromTrueDofs(X);

      psi = psi_projected;
   }
   else
   {
      MFEM_ASSERT(temp_psi.FESpace()->GetMesh()->GetNE() == scalar_fespace.GetMesh()->GetNE(), "The two spaces are not on the same mesh");
      psi = temp_psi;
   }

   
   GridFunction B_pol(&vector_fespace);
   cout << B_pol.FESpace()->GetTrueVSize() << endl;
   B_pol = 0.0;

   LinearForm b(&vector_fespace);

   // project the grid function onto the new space
   

   // 1.a make the RHS bilinear form
   MixedBilinearForm b_bi(psi.FESpace(), &vector_fespace);
   DenseMatrix perp_rotation(dim);
   perp_rotation(0, 0) = 0.0;
   perp_rotation(0, 1) = -1.0;
   perp_rotation(1, 0) = 1.0;
   perp_rotation(1, 1) = 0.0;
   MatrixConstantCoefficient perp_rot_coef(perp_rotation);
   b_bi.AddDomainIntegrator(new MixedVectorGradientIntegrator(perp_rot_coef));
   b_bi.Assemble();

   // 1.b form linear form from bilinear form
   LinearForm b_li(&vector_fespace);
   b_bi.Mult(psi, b_li);
   b.Assemble();
   b += b_li;

   // 2. make the bilinear form
   BilinearForm a(&vector_fespace);
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

   Vector X(B_pol.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   B_pol.SetFromTrueDofs(X);
   return B_pol;
}

GridFunction computeBTor(GridFunction &gg,
                         FiniteElementSpace &scalar_fespace,
                         bool mixed_bilinear_form,
                         bool from_psi)
{
   GridFunction B_tor(&scalar_fespace);
   cout << B_tor.FESpace()->GetTrueVSize() << endl;
   B_tor = 0.0;
   LinearForm b(&scalar_fespace);
   if (!mixed_bilinear_form)
   {
      cout << "Using linear form" << endl;
      // Solve <B_tor, v> = <gg/R, v> for all v in H1.
      FGridFunctionCoefficient f_coef(&gg, from_psi);
      b.AddDomainIntegrator(new DomainLFIntegrator(f_coef));
      b.Assemble();
   }
   else
   {
      cout << "Using bilinear form" << endl;
      MFEM_ASSERT(gg.FESpace()->GetMesh()->GetNE() == scalar_fespace.GetMesh()->GetNE(),
                  "The two spaces are not on the same mesh");
      MFEM_ASSERT(from_psi == false, "from_psi is not implemented for mixed bilinear form");
      MixedBilinearForm b_bi(gg.FESpace(), &scalar_fespace);
      ConstantCoefficient one(1.0);
      b_bi.AddDomainIntegrator(new MixedScalarMassIntegrator(one));
      b_bi.Assemble();
      b_bi.Mult(gg, b);
   }

   BilinearForm a(&scalar_fespace);
   RGridFunctionCoefficient r_coef;
   a.AddDomainIntegrator(new MassIntegrator(r_coef));
   a.Assemble();
   a.Finalize();

   CGSolver M_solver;
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-24);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(1e5);
   M_solver.SetPrintLevel(1);
   M_solver.SetOperator(a.SpMat());

   Vector X(B_tor.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   B_tor.SetFromTrueDofs(X);
   return B_tor;
}

int main(int argc, char *argv[])
{
   const char *mesh_file = "mesh/2d_mesh.mesh";
   bool visualization = true;
   bool project_mesh = true;
   bool mixed_bilinear_form = false;
   bool from_psi = false;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ifstream psi_log("input/psi.gf");
   GridFunction temp_psi(&mesh, psi_log);
   ifstream gg_log("input/gg.gf");
   GridFunction gg(&mesh, gg_log);

   cout << "Mesh loaded" << endl;

   const char *out_mesh_file = "mesh/2d_mesh.mesh";
   Mesh *new_mesh = new Mesh(out_mesh_file, 1, 1);

   RT_FECollection rt_fec(0, dim);
   FiniteElementSpace vector_fespace(new_mesh, &rt_fec);
   H1_FECollection scalar_fec(1, dim);
   FiniteElementSpace scalar_fespace(new_mesh, &scalar_fec);

   GridFunction B_pol = computeBPol(temp_psi, scalar_fespace, vector_fespace, project_mesh);
   GridFunction B_tor = computeBTor(gg, scalar_fespace, mixed_bilinear_form, from_psi);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *new_mesh << B_pol << flush;
      socketstream tor_sock(vishost, visport);
      tor_sock.precision(8);
      tor_sock << "solution\n"
               << *new_mesh << B_tor << flush;
   }

   delete new_mesh;

   return 0;
}

