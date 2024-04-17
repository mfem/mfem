//                                MFEM Example 40
//
// Compile with: make ex40
//
// Sample runs: ex40 -step 10 -gr 2.0
//              ex40 -step 10 -gr 2.0
//              ex40 -step 100 -gr 2.0 -r 4 -m ../data/l-shape.mesh
//
// Description: This example code demonstrates how to use MFEM to solve the
//              eikonal equation,
//
//                      |∇u| = 1 in Ω,  u = g on ∂Ω.
//
//              This example constructs a fast converging sequence,
//
//                      uₖ → u  as k → \infty,
//
//              by using in Newton's method to solve the sequence of nonlinear
//              saddle-point problems
//
//               Find ψₖ ∈ H(div,Ω) and uₖ ∈ L²(Ω) such that
//               ( Zₖ(ψₖ) , τ ) + ( uₖ , ∇⋅τ ) = 0                    ∀ τ ∈ H(div,Ω)
//               ( ∇⋅ψₖ , v )                 = ( -1 + ∇⋅ψₖ₋₁ , v )   ∀ v ∈ L²(Ω)
//
//              where Zₖ(ψ) = ψ / ( 1/αₖ + |ψ|² )^{1/2} and αₖ > 0.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ZCoefficient : public VectorCoefficient
{
protected:
   GridFunction *psi;
   real_t alpha;

public:
   ZCoefficient(int vdim, GridFunction &psi_, real_t alpha_ = 1.0)
      : VectorCoefficient(vdim), psi(&psi_), alpha(alpha_) { }

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
   void SetAlpha(real_t alpha_) { alpha = alpha_; }
};

class DZCoefficient : public MatrixCoefficient
{
protected:
   GridFunction *psi;
   real_t alpha;

public:
   DZCoefficient(int height, GridFunction &psi_, real_t alpha_ = 1.0)
      : MatrixCoefficient(height),  psi(&psi_), alpha(alpha_) { }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);
   void SetAlpha(real_t alpha_) { alpha = alpha_; }
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int max_it = 5;
   int ref_levels = 3;
   real_t alpha = 1.0;
   real_t growth_rate = 1.0;
   real_t newton_scaling = 0.9;
   real_t tichonov = 1e-1;
   real_t tol = 1e-4;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&tol, "-tol", "--tol",
                  "Stopping criteria based on the difference between"
                  "successive solution updates");
   args.AddOption(&alpha, "-step", "--step",
                  "Initial size alpha");
   args.AddOption(&growth_rate, "-gr", "--growth-rate",
                  "Growth rate of the step size alpha");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();

   MFEM_ASSERT(mesh.bdr_attributes.Size(),
               "This example does not currently support meshes"
               " without boundary attributes."
              )

   // 3. Postprocess the mesh.
   // 3A. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 3B. Interpolate the geometry after refinement to control geometry error.
   // NOTE: Minimum second-order interpolation is used to improve the accuracy.
   int curvature_order = max(order,2);
   mesh.SetCurvature(curvature_order);

   // 4. Define the necessary finite element spaces on the mesh.
   RT_FECollection RTfec(order, dim);
   FiniteElementSpace RTfes(&mesh, &RTfec);

   L2_FECollection L2fec(order, dim);
   FiniteElementSpace L2fes(&mesh, &L2fec);

   cout << "Number of H(div) dofs: "
        << RTfes.GetTrueVSize() << endl;
   cout << "Number of L² dofs: "
        << L2fes.GetTrueVSize() << endl;

   // 5. Define the offsets for the block matrices
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = RTfes.GetVSize();
   offsets[2] = L2fes.GetVSize();
   offsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   // 6. Define the solution vectors as a finite element grid functions
   //    corresponding to the fespaces.
   GridFunction u_gf, delta_psi_gf;
   delta_psi_gf.MakeRef(&RTfes,x,offsets[0]);
   u_gf.MakeRef(&L2fes,x,offsets[1]);

   GridFunction psi_old_gf(&RTfes);
   GridFunction psi_gf(&RTfes);
   GridFunction u_old_gf(&L2fes);

   // 7. Define initial guesses for the solution variables.
   delta_psi_gf = 0.0;
   psi_gf = 0.0;
   u_gf = 0.0;
   psi_old_gf = psi_gf;
   u_old_gf = u_gf;

   // 8. Prepare for glvis output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost,visport);
      sol_sock.precision(8);
   }

   // 9. Coefficients to be used later.
   ConstantCoefficient neg_one(-1.0);
   ConstantCoefficient zero(0.0);
   ConstantCoefficient tichonov_cf(tichonov);
   ConstantCoefficient neg_tichonov_cf(-1.0*tichonov);
   ZCoefficient Z(sdim, psi_gf, alpha);
   DZCoefficient DZ(sdim, psi_gf, alpha);
   ScalarVectorProductCoefficient neg_Z(-1.0, Z);
   DivergenceGridFunctionCoefficient div_psi_cf(&psi_gf);
   DivergenceGridFunctionCoefficient div_psi_old_cf(&psi_old_gf);
   SumCoefficient psi_old_minus_psi(div_psi_old_cf, div_psi_cf, 1.0, -1.0);

   // 10. Assemble constant matrices and vectors to avoid
   //     reassembly in the loop.
   LinearForm b0, b1;
   b0.MakeRef(&RTfes,rhs.GetBlock(0),0);
   b1.MakeRef(&L2fes,rhs.GetBlock(1),0);

   b0.AddDomainIntegrator(new VectorFEDomainLFIntegrator(neg_Z));
   b1.AddDomainIntegrator(new DomainLFIntegrator(neg_one));
   b1.AddDomainIntegrator(new DomainLFIntegrator(psi_old_minus_psi));

   BilinearForm a00(&RTfes);
   a00.AddDomainIntegrator(new VectorFEMassIntegrator(DZ));
   a00.AddDomainIntegrator(new VectorFEMassIntegrator(tichonov_cf));

   MixedBilinearForm a10(&RTfes,&L2fes);
   a10.AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   a10.Assemble();
   a10.Finalize();
   SparseMatrix &A10 = a10.SpMat();
   SparseMatrix *A01 = Transpose(A10);

   BilinearForm a11(&L2fes);
   a11.AddDomainIntegrator(new MassIntegrator(neg_tichonov_cf));
   a11.Assemble();
   a11.Finalize();
   SparseMatrix &A11 = a11.SpMat();

   // 11. Iterate.
   int k;
   int total_iterations = 0;
   real_t increment_u = 0.1;
   GridFunction u_tmp(&L2fes);
   for (k = 0; k < max_it; k++)
   {
      u_tmp = u_old_gf;
      Z.SetAlpha(alpha);
      DZ.SetAlpha(alpha);

      mfem::out << "\nOUTER ITERATION " << k+1 << endl;

      int j;
      for ( j = 0; j < 5; j++)
      {
         total_iterations++;

         b0.Assemble();
         b1.Assemble();

         a00.Assemble(false);
         a00.Finalize(false);
         SparseMatrix &A00 = a00.SpMat();

         // Construct Schur-complement preconditioner
         Vector A00_diag(a00.Height());
         A00.GetDiag(A00_diag);
         A00_diag.Reciprocal();
         SparseMatrix *S = Mult_AtDA(*A01, A00_diag);

         BlockDiagonalPreconditioner prec(offsets);
         prec.SetDiagonalBlock(0,new DSmoother(A00));
#ifndef MFEM_USE_SUITESPARSE
         prec.SetDiagonalBlock(1,new GSSmoother(*S));
#else
         prec.SetDiagonalBlock(1,new UMFPackSolver(*S));
#endif
         prec.owns_blocks = 1;

         BlockOperator A(offsets);
         A.SetBlock(0,0,&A00);
         A.SetBlock(1,0,&A10);
         A.SetBlock(0,1,A01);
         A.SetBlock(1,1,&A11);

         GMRES(A,prec,rhs,x,0,2000,500,1e-12,0.0);
         delete S;

         u_tmp -= u_gf;
         real_t Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         // Damped Newton update
         psi_gf.Add(newton_scaling, delta_psi_gf);
         a00.Update();

         if (visualization)
         {
            sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
                     << flush;
         }

         mfem::out << "Newton_update_size = " << Newton_update_size << endl;

         if (Newton_update_size < increment_u)
         {
            break;
         }

      }

      u_tmp = u_gf;
      u_tmp -= u_old_gf;
      increment_u = u_tmp.ComputeL2Error(zero);

      mfem::out << "Number of Newton iterations = " << j+1 << endl;
      mfem::out << "Increment (|| uₕ - uₕ_prvs||) = " << increment_u << endl;

      u_old_gf = u_gf;
      psi_old_gf = psi_gf;

      if (increment_u < tol || k == max_it-1)
      {
         break;
      }

      alpha *= max(growth_rate, 1.0);

   }

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << "\n Total dofs:       " << RTfes.GetTrueVSize() + L2fes.GetTrueVSize()
             << endl;

   delete A01;
   return 0;
}

void ZCoefficient::Eval(Vector &V, ElementTransformation &T,
                        const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");
   MFEM_ASSERT(alpha > 0, "alpha is not positive");

   Vector psi_vals(vdim);
   psi->GetVectorValue(T, ip, psi_vals);
   real_t norm = psi_vals.Norml2();
   real_t phi = 1.0 / sqrt(1.0/alpha + norm*norm);

   V = psi_vals;
   V *= phi;
}

void DZCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                         const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");
   MFEM_ASSERT(alpha > 0, "alpha is not positive");

   Vector psi_vals(height);
   psi->GetVectorValue(T, ip, psi_vals);
   real_t norm = psi_vals.Norml2();
   real_t phi = 1.0 / sqrt(1.0/alpha + norm*norm);

   K = 0.0;
   for (int i = 0; i < height; i++)
   {
      K(i,i) = phi;
      for (int j = 0; j < height; j++)
      {
         K(i,j) -= psi_vals(i) * psi_vals(j) * pow(phi, 3);
      }
   }
}