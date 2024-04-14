//                                MFEM Example 41
//
// Compile with: make ex41
//
// Sample runs: ex41 -o 2
//              ex41 -o 1 -r 4
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
//               Find ψₖ ∈ L²(Ω)ⁿ and uₖ ∈ H¹₀(Ω) such that
//               ( Zₖ(ψₖ) , τ ) + ( ∇uₖ , τ ) = 0                    ∀ τ ∈ L²(Ω)ⁿ
//                ( ψₖ , ∇v )                = ( -1 , v) + ( ψₖ₋₁ , ∇v )   ∀ v ∈ H¹₀(Ω)
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
   real_t alpha = 1.0;
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
                  "Step size alpha.");
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
   L2_FECollection L2fec(order, dim);
   FiniteElementSpace L2fes(&mesh, &L2fec, sdim);

   H1_FECollection H1fec(order, dim);
   FiniteElementSpace H1fes(&mesh, &H1fec);

   cout << "Number of L2 finite element unknowns: "
        << L2fes.GetTrueVSize() << endl;
   cout << "Number of H1 finite element unknowns: "
        << H1fes.GetTrueVSize() << endl;

   // 5. Determine the list of true (i.e., conforming) essential boundary dofs.
   Array<int> ess_vdof_list;
   ess_vdof_list.SetSize(H1fes.GetTrueVSize());
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      H1fes.GetEssentialVDofs(ess_bdr, ess_vdof_list);
   }
   
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = L2fes.GetVSize();
   offsets[2] = H1fes.GetVSize();
   offsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   // 6. Define an initial guess for the solution.
   ConstantCoefficient one(-1.0);
   ConstantCoefficient neg_one(-1.0);
   ConstantCoefficient zero(0.0);

   // 7. Define the solution vectors as a finite element grid functions
   //    corresponding to the fespaces.
   GridFunction u_gf, delta_psi_gf;

   delta_psi_gf.MakeRef(&L2fes,x,offsets[0]);
   u_gf.MakeRef(&H1fes,x,offsets[1]);
   delta_psi_gf = 0.0;

   GridFunction psi_old_gf(&L2fes);
   GridFunction psi_gf(&L2fes);
   GridFunction u_old_gf(&H1fes);
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
      GridFunction u_tmp(&H1fes);
      u_tmp = u_old_gf;

      mfem::out << "\nOUTER ITERATION " << k+1 << endl;

      int j;
      for ( j = 0; j < 5; j++)
      {
         total_iterations++;

         ConstantCoefficient alpha_cf(alpha);

         LinearForm b0,b1;
         b0.Update(&L2fes,rhs.GetBlock(0),0);
         b1.Update(&H1fes,rhs.GetBlock(1),0);

         ZCoefficient Z(sdim, psi_gf, alpha);
         DZCoefficient DZ(sdim, psi_gf, alpha);

         ScalarVectorProductCoefficient neg_Z(-1.0, Z);
         b0.AddDomainIntegrator(new VectorDomainLFIntegrator(neg_Z));
         b0.Assemble();

         VectorGridFunctionCoefficient psi_cf(&psi_gf);
         VectorGridFunctionCoefficient psi_old_cf(&psi_old_gf);
         VectorSumCoefficient psi_old_minus_psi(psi_old_cf, psi_cf, 1.0, -1.0);

         b1.AddDomainIntegrator(new DomainLFIntegrator(neg_one));
         b1.AddDomainIntegrator(new DomainLFGradIntegrator(psi_old_minus_psi));
         b1.Assemble();

         BilinearForm a00(&L2fes);
         a00.AddDomainIntegrator(new VectorMassIntegrator(DZ));
         // ConstantCoefficient eps(1e-2);
         // a00.AddDomainIntegrator(new VectorMassIntegrator(eps));
         a00.Assemble();
         a00.Finalize();
         SparseMatrix &A00 = a00.SpMat();

         MixedBilinearForm a01(&H1fes,&L2fes);
         a01.AddDomainIntegrator(new GradientIntegrator());
         a01.Assemble();
         a01.EliminateEssentialBCFromTrialDofs(ess_vdof_list,x.GetBlock(1),rhs.GetBlock(0));
         a01.Finalize();
         SparseMatrix &A01 = a01.SpMat();

         SparseMatrix *A10 = Transpose(A01);

         BilinearForm a11(&H1fes);
         a11.AddDomainIntegrator(new MassIntegrator(zero));
         a11.Assemble(false);
         a11.EliminateEssentialBCFromDofs(ess_vdof_list,x.GetBlock(1),rhs.GetBlock(1));
         a11.Finalize();
         SparseMatrix &A11 = a11.SpMat();

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
         A.SetBlock(1,0,A10);
         A.SetBlock(0,1,&A01);
         A.SetBlock(1,1,&A11);

         SparseMatrix * A_mono = A.CreateMonolithic();
         UMFPackSolver umf(*A_mono);
         umf.Mult(rhs,x);

         delta_psi_gf.MakeRef(&L2fes, x.GetBlock(0), 0);
         u_gf.MakeRef(&H1fes, x.GetBlock(1), 0);

         u_tmp -= u_gf;
         real_t Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         real_t gamma = 0.1;
         delta_psi_gf *= gamma;
         psi_gf += delta_psi_gf;

         if (visualization)
         {
            // sol_sock << "solution\n" << mesh << psi_gf << "window_title 'Discrete solution'"
            sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
                     << flush;
            mfem::out << "Newton_update_size = " << Newton_update_size << endl;
         }

         delete A10;

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

      // alpha *= 2.0;

   }

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << "\n Total dofs:       " << L2fes.GetTrueVSize() + H1fes.GetTrueVSize()
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