#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <string>
#include "vec_coeffs.hpp"

using namespace std;
using namespace mfem;

GridFunction computeBPol(GridFunction &psi,
                         FiniteElementSpace &scalar_fespace,
                         FiniteElementSpace &vector_fespace)
{
   int dim = vector_fespace.GetMesh()->Dimension();


   GridFunction B_pol(&vector_fespace);
   cout << B_pol.FESpace()->GetTrueVSize() << endl;
   B_pol = 0.0;

   LinearForm b(&vector_fespace);

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
   bool mixed_bilinear_form = false;
   bool from_psi = false;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   const string psi_log_filename = "input/psi.gf";
   const string gg_log_filename = "input/gg.gf";
   ifstream psi_log(psi_log_filename);
   GridFunction psi(&mesh, psi_log);
   ifstream gg_log(gg_log_filename);
   GridFunction gg(&mesh, gg_log);

   MFEM_ASSERT(psi.Size() == mesh.GetNV(),
               "This vertex scan assumes first-order CG (one DOF per vertex).");
   double psi_max = -std::numeric_limits<double>::infinity();
   int psi_max_vertex = -1;
   Vector psi_max_x(dim);
   for (int v = 0; v < mesh.GetNV(); v++)
   {
      const double psi_val = psi(v);
      if (psi_val > psi_max)
      {
         psi_max = psi_val;
         psi_max_vertex = v;
         const double *xv = mesh.GetVertex(v);
         for (int d = 0; d < dim; d++) { psi_max_x(d) = xv[d]; }
      }
   }
   cout << "psi max = " << psi_max << ", vertex = " << psi_max_vertex
        << ", at x = ";
   psi_max_x.Print(cout, dim);
   cout << "psi.Max() check = " << psi.Max() << endl;

   cout << "Mesh loaded" << endl;

   RT_FECollection rt_fec(0, dim);
   FiniteElementSpace vector_fespace(&mesh, &rt_fec);
   H1_FECollection scalar_fec(1, dim);
   FiniteElementSpace scalar_fespace(&mesh, &scalar_fec);

   GridFunction B_pol = computeBPol(psi, scalar_fespace, vector_fespace);
   GridFunction B_tor = computeBTor(gg, scalar_fespace, mixed_bilinear_form, from_psi);

   FindPointsGSLIBOneByOne finder(&psi);
   Vector B_tor_at_psi_max(1);
   finder.InterpolateOneByOne(psi_max_x, B_tor, B_tor_at_psi_max, Ordering::byNODES);
   Vector B_pol_at_psi_max(dim);
   finder.InterpolateOneByOne(psi_max_x, B_pol, B_pol_at_psi_max, Ordering::byNODES);
   cout << "B_tor(psi_max point) = " << B_tor_at_psi_max(0) << endl;
   cout << "B_pol(psi_max point) = ";
   B_pol_at_psi_max.Print(cout, dim);
   const double B_mag =
      std::sqrt(B_tor_at_psi_max(0) * B_tor_at_psi_max(0) + B_pol_at_psi_max * B_pol_at_psi_max);
   // precision 8
   cout << "B magnitude = " << std::setprecision(8) << B_mag << endl;
   MFEM_VERIFY(B_mag > 0.0, "B magnitude is zero; cannot normalize psi and gg.");

   const double inv_B_mag = 1.0 / B_mag;
   psi *= inv_B_mag;
   gg *= inv_B_mag;

   auto scaled_filename = [](const string &filename)
   {
      const size_t dot = filename.rfind('.');
      if (dot == string::npos) { return filename + "_s"; }
      return filename.substr(0, dot) + "_s" + filename.substr(dot);
   };
   const string psi_scaled_filename = scaled_filename(psi_log_filename);
   const string gg_scaled_filename = scaled_filename(gg_log_filename);
   ofstream psi_scaled_ofs(psi_scaled_filename);
   psi_scaled_ofs.precision(8);
   psi.Save(psi_scaled_ofs);
   ofstream gg_scaled_ofs(gg_scaled_filename);
   gg_scaled_ofs.precision(8);
   gg.Save(gg_scaled_ofs);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << mesh << B_pol << flush;
      socketstream tor_sock(vishost, visport);
      tor_sock.precision(8);
      tor_sock << "solution\n"
               << mesh << B_tor << flush;
   }

   return 0;
}

