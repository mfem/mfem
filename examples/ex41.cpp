//                                MFEM Example 41
//
// Compile with: make ex41
//
// Sample runs: ex41 -o 2
//              ex41 -o 2 -r 4
//
// Description: This example code demonstrates how to use MFEM to solve the
//              Eikonal equation,
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
//               ( ϕₖ(ψₖ)ψₖ , τ ) + ( uₖ , ∇⋅τ ) = < g, τ⋅n >           ∀ τ ∈ H(div,Ω)
//               ( ∇⋅ψₖ , v )                   = ( -1 + ∇⋅ψₖ₋₁ , v )   ∀ v ∈ L²(Ω)
//
//              where ϕₖ(ψ) = (1/αₖ + |ψ|^2)^{-1/2} and αₖ > 0.


//              where ϕₖ(ψ) = exp(|ψ|)/(1/αₖ + exp(|ψ|)) and αₖ > 0.
//

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

   virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip);
};

class DZCoefficient : public MatrixCoefficient
{
protected:
   GridFunction *psi;
   real_t alpha;

public:
   DZCoefficient(int height, GridFunction &psi_, real_t alpha_ = 1.0)
      : MatrixCoefficient(height, true),  psi(&psi_), alpha(alpha_) { }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int max_it = 5;
   int ref_levels = 3;
   real_t alpha = 10.0;
   real_t tol = 1e-5;
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
                  "Step size alpha");
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

   cout << "Number of Hdiv finite element unknowns: "
        << RTfes.GetTrueVSize() << endl;
   cout << "Number of L2 finite element unknowns: "
        << L2fes.GetTrueVSize() << endl;

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = RTfes.GetVSize();
   offsets[2] = L2fes.GetVSize();
   offsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_bdr;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
   }

   // 6. Define an initial guess for the solution.
   ConstantCoefficient neg_one(-1.0);
   ConstantCoefficient zero(0.0);

   // 7. Define the solution vectors as a finite element grid functions
   //    corresponding to the fespaces.
   GridFunction u_gf, delta_psi_gf;

   delta_psi_gf.MakeRef(&RTfes,x,offsets[0]);
   u_gf.MakeRef(&L2fes,x,offsets[1]);
   delta_psi_gf = 0.0;

   GridFunction psi_old_gf(&RTfes);
   GridFunction psi_gf(&RTfes);
   GridFunction u_old_gf(&L2fes);
   psi_old_gf = 0.0;
   u_old_gf = 0.0;

   // 8. Define the function coefficients for the solution and use them to
   //    initialize the initial guess
   psi_gf = 0.0;
   psi_old_gf = psi_gf;
   u_old_gf = u_gf;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost,visport);
      sol_sock.precision(8);
   }

   // 10. Iterate
   int k;
   int total_iterations = 0;
   real_t increment_u = 0.1;
   for (k = 0; k < max_it; k++)
   {
      GridFunction u_tmp(&L2fes);
      u_tmp = u_old_gf;

      alpha *= 2.0;

      mfem::out << "\nOUTER ITERATION " << k+1 << endl;

      int j;
      for ( j = 0; j < 10; j++)
      {
         total_iterations++;

         ConstantCoefficient alpha_cf(alpha);

         LinearForm b0,b1;
         b0.Update(&RTfes,rhs.GetBlock(0),0);
         b1.Update(&L2fes,rhs.GetBlock(1),0);

         ZCoefficient Z(sdim, psi_gf, alpha);
         DZCoefficient DZ(sdim, psi_gf, alpha);

         ScalarVectorProductCoefficient neg_Z(-1.0, Z);
         b0.AddDomainIntegrator(new VectorFEDomainLFIntegrator(neg_Z));
         b0.Assemble();

         DivergenceGridFunctionCoefficient div_psi_cf(&psi_gf);
         DivergenceGridFunctionCoefficient div_psi_old_cf(&psi_old_gf);
         SumCoefficient psi_old_minus_psi(div_psi_old_cf, div_psi_cf, 1.0, -1.0);
         b1.AddDomainIntegrator(new DomainLFIntegrator(neg_one));
         b1.AddDomainIntegrator(new DomainLFIntegrator(psi_old_minus_psi));
         b1.Assemble();

         BilinearForm a00(&RTfes);
         a00.AddDomainIntegrator(new VectorFEMassIntegrator(DZ));
        //  ConstantCoefficient eps(1e-2);
        //  a00.AddDomainIntegrator(new VectorFEMassIntegrator(eps));
         a00.Assemble();
         a00.Finalize();
         SparseMatrix &A00 = a00.SpMat();

         MixedBilinearForm a10(&RTfes,&L2fes);
         a10.AddDomainIntegrator(new VectorFEDivergenceIntegrator());
         a10.Assemble();
        //  Array<int> dof_elim_array(1);
        //  int dof_elim = 10;
        //  dof_elim_array[0] = dof_elim;
         a10.Finalize();
         SparseMatrix &A10 = a10.SpMat();
        //  A10.EliminateRow(dof_elim);

         SparseMatrix *A01 = Transpose(A10);

        //  BilinearForm a11(&L2fes);
        //  ConstantCoefficient neg_eps(-1e-2);
        //  a11.AddDomainIntegrator(new MassIntegrator(neg_eps));
        //  a11.Assemble(false);
        //  a11.Finalize();
        //  SparseMatrix &A11 = a11.SpMat();

        //  BilinearForm a11(&L2fes);
        //  a11.AddDomainIntegrator(new MassIntegrator(zero));
        //  a11.Assemble(false);
        //  SparseMatrix A11;
        //  a11.FormSystemMatrix(dof_elim_array, A11);
        //  A11.Set(dof_elim, dof_elim, 1.0);
        //  rhs[offsets[1] + dof_elim] = 0.0;

        //  BlockOperator A(offsets);
        //  A.SetBlock(0,0,&A00);
        //  A.SetBlock(1,0,&A10);
        //  A.SetBlock(0,1,A01);

        //  BlockDiagonalPreconditioner prec(offsets);
        //  prec.SetDiagonalBlock(0,new GSSmoother(A00));
        //  prec.SetDiagonalBlock(1,new GSSmoother(A11));
        //  prec.owns_blocks = 1;

        //  GMRES(A,prec,rhs,x,0,10000,500,1e-12,0.0);

         BlockMatrix A(offsets);
         A.SetBlock(0,0,&A00);
         A.SetBlock(1,0,&A10);
         A.SetBlock(0,1,A01);
        //  A.SetBlock(1,1,&A11);

         SparseMatrix * A_mono = A.CreateMonolithic();
         UMFPackSolver umf(*A_mono);
         umf.Mult(rhs,x);

         delta_psi_gf.MakeRef(&RTfes, x.GetBlock(0), 0);
         u_gf.MakeRef(&L2fes, x.GetBlock(1), 0);

         u_tmp -= u_gf;
         real_t Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         real_t gamma = 1.0;
         delta_psi_gf *= gamma;
         psi_gf += delta_psi_gf;

         if (visualization)
         {
            sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
                     << flush;
            mfem::out << "Newton_update_size = " << Newton_update_size << endl;
         }

         delete A01;

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

      if (k == max_it-1)
      // if (increment_u < tol || k == max_it-1)
      {
         break;
      }

   }

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << "\n Total dofs:       " << RTfes.GetTrueVSize() + L2fes.GetTrueVSize()
             << endl;

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

   for (int i = 0; i < height; i++)
   {
    K(i,i) = phi;
    for (int j = 0; j < height; j++)
      {
        K(i,j) -= psi_vals(i) * psi_vals(j) * pow(phi, 3);
      }
   }
}