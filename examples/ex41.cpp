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
//                      |∇u| = 1 in Ω,  u = 0 on ∂Ω.
//
//              This example constructs a fast converging sequence,
//
//                      uₖ → u  as k → \infty,
//
//              by using in Newton's method to solve the sequence of nonlinear
//              saddle-point problems
//
//               Find ψₖ ∈ H₀(div,Ω) and uₖ ∈ L²(Ω) such that
//               ( ϕₖ(ψₖ)ψₖ , τ ) + ( uₖ , ∇⋅τ ) = 0                   ∀ τ ∈ H₀(div,Ω)
//               ( ∇⋅ψₖ , v )                  = ( 1 + ∇⋅ψₖ₋₁ , v )   ∀ v ∈ L²(Ω)
//
//              where ϕₖ(ψ) = (1/αₖ + |ψ|^2)^{-1/2} and αₖ > 0.


//              where ϕₖ(ψ) = exp(|ψ|)/(1/αₖ + exp(|ψ|)) and αₖ > 0.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t exact_solution(const Vector &pt);

class ZCoefficient : public VectorCoefficient
{
protected:
   GridFunction *psi;
   real_t alpha;

public:
   ZCoefficient(GridFunction &psi_, real_t alpha_ = 1.0) : psi(&psi_), alpha(alpha_) { }

   virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip);
};

class DZCoefficient : public MatrixCoefficient
{
protected:
   GridFunction *psi;
   real_t alpha;

public:
   DZCoefficient(GridFunction &psi_) : psi(&psi_), alpha(alpha_) { }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int order = 1;
   int max_it = 10;
   int ref_levels = 3;
   real_t alpha = 1.0;
   real_t tol = 1e-5;
   bool visualization = true;

   OptionsParser args(argc, argv);
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
   const char *mesh_file = "../data/disc-nurbs.mesh";
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

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
        << RTfec.GetTrueVSize() << endl;
   cout << "Number of L2 finite element unknowns: "
        << L2fes.GetTrueVSize() << endl;

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = RTfec.GetVSize();
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
   ConstantCoefficient one(1.0);
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
   FunctionCoefficient exact_coef(exact_solution);
   VectorFunctionCoefficient exact_grad_coef(dim,exact_solution_gradient);
   ConstantCoefficient f(0.0);
   psi_gf.ProjectCoefficient(zero);
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

      mfem::out << "\nOUTER ITERATION " << k+1 << endl;

      int j;
      for ( j = 0; j < 10; j++)
      {
         total_iterations++;

         ConstantCoefficient alpha_cf(alpha);

         LinearForm b0,b1;
         b0.Update(&RTfes,rhs.GetBlock(0),0);
         b1.Update(&L2fes,rhs.GetBlock(1),0);

         ZCoefficient Z(psi_gf, alpha);
         DZCoefficient DZ(psi_gf, alpha);

         ProductCoefficient neg_Z(-1.0, Z);
         b0.AddDomainIntegrator(new DomainLFIntegrator(neg_Z));
         b0.Assemble();

         DivergenceGridFunctionCoefficient div_psi_cf(psi_cf);        
         DivergenceGridFunctionCoefficient div_psi_old_cf(psi_old_cf);        
         SumCoefficient psi_old_minus_psi(div_psi_old_cf, div_psi_cf, 1.0, -1.0);
         b1.AddDomainIntegrator(new DomainLFIntegrator(one));
         b1.Assemble();

         BilinearForm a00(&H1fes);
         a00.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
         a00.AddDomainIntegrator(new DiffusionIntegrator(alpha_cf));
         a00.Assemble();
         a00.EliminateEssentialBC(ess_bdr,x.GetBlock(0),rhs.GetBlock(0),
                                  mfem::Operator::DIAG_ONE);
         a00.Finalize();
         SparseMatrix &A00 = a00.SpMat();

         MixedBilinearForm a10(&H1fes,&L2fes);
         a10.AddDomainIntegrator(new MixedScalarMassIntegrator());
         a10.Assemble();
         a10.EliminateTrialDofs(ess_bdr, x.GetBlock(0), rhs.GetBlock(1));
         a10.Finalize();
         SparseMatrix &A10 = a10.SpMat();

         SparseMatrix *A01 = Transpose(A10);

         BilinearForm a11(&L2fes);
         a11.AddDomainIntegrator(new MassIntegrator(neg_exp_psi));
         // NOTE: Shift the spectrum of the Hessian matrix for additional
         //       stability (Quasi-Newton).
         ConstantCoefficient eps_cf(-1e-6);
         if (order == 1)
         {
            // NOTE: ∇ₕuₕ = 0 for constant functions.
            //       Therefore, we use the mass matrix to shift the spectrum
            a11.AddDomainIntegrator(new MassIntegrator(eps_cf));
         }
         else
         {
            a11.AddDomainIntegrator(new DiffusionIntegrator(eps_cf));
         }
         a11.Assemble();
         a11.Finalize();
         SparseMatrix &A11 = a11.SpMat();

         BlockOperator A(offsets);
         A.SetBlock(0,0,&A00);
         A.SetBlock(1,0,&A10);
         A.SetBlock(0,1,A01);

         BlockDiagonalPreconditioner prec(offsets);
         prec.SetDiagonalBlock(0,new GSSmoother(A00));
         prec.SetDiagonalBlock(1,new GSSmoother(A11));
         prec.owns_blocks = 1;

         GMRES(A,prec,rhs,x,0,10000,500,1e-12,0.0);

         u_gf.MakeRef(&H1fes, x.GetBlock(0), 0);
         delta_psi_gf.MakeRef(&L2fes, x.GetBlock(1), 0);

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

      if (increment_u < tol || k == max_it-1)
      {
         break;
      }

      real_t H1_error = u_gf.ComputeH1Error(&exact_coef,&exact_grad_coef);
      mfem::out << "H1-error  (|| u - uₕᵏ||)       = " << H1_error << endl;

   }

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << "\n Total dofs:       " << H1fes.GetTrueVSize() + L2fes.GetTrueVSize()
             << endl;

   // 11. Exact solution.
   if (visualization)
   {
      socketstream err_sock(vishost, visport);
      err_sock.precision(8);

      GridFunction error_gf(&H1fes);
      error_gf.ProjectCoefficient(exact_coef);
      error_gf -= u_gf;

      err_sock << "solution\n" << mesh << error_gf << "window_title 'Error'"  <<
               flush;
   }

   {
      real_t L2_error = u_gf.ComputeL2Error(exact_coef);
      real_t H1_error = u_gf.ComputeH1Error(&exact_coef,&exact_grad_coef);

      ExponentialGridFunctionCoefficient u_alt_cf(psi_gf,obstacle);
      GridFunction u_alt_gf(&L2fes);
      u_alt_gf.ProjectCoefficient(u_alt_cf);
      real_t L2_error_alt = u_alt_gf.ComputeL2Error(exact_coef);

      mfem::out << "\n Final L2-error (|| u - uₕ||)          = " << L2_error <<
                endl;
      mfem::out << " Final H1-error (|| u - uₕ||)          = " << H1_error << endl;
      mfem::out << " Final L2-error (|| u - ϕ - exp(ψₕ)||) = " << L2_error_alt <<
                endl;
   }

   return 0;
}

efficient::Eval(Vector &V, ElementTransformation &T,
                                              const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");
   MFEM_ASSERT(alpha > 0, "alpha is not postive");

   int dim  = psi->VectorDim();
   Vector psi_vals(dim);
   void = psi->GetVectorValue(T, ip, psi_vals);
   real_t norm = psi_vals.Norml2();
   real_t phi = 1.0 / sqrt(1.0/alpha + norm*norm);

   V = psi_vals;
   V *= phi;
}

void DZCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                                                const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");
   MFEM_ASSERT(alpha > 0, "alpha is not postive");

   int dim  = psi->VectorDim();
   Vector psi_vals(dim);
   void = psi->GetVectorValue(T, ip, psi_vals);
   real_t norm = psi_vals.Norml2();
   real_t phi = 1.0 / sqrt(1.0/alpha + norm*norm);

   for (int i = 0; i < dim; i++)
   {
    K(i,i) = phi;
    for (int j = 0; j < dim; j++)
      {
        K(i,j) -= psi_vals(i) * psi_vals(j) * pow(phi, 3);
      }
   }
}

// void ZCoefficient::Eval(Vector &V, ElementTransformation &T,
//                                               const IntegrationPoint &ip)
// {
//    MFEM_ASSERT(psi != NULL, "grid function is not set");
//    MFEM_ASSERT(alpha > 0, "alpha is not postive");

//    int dim  = psi->VectorDim();
//    Vector psi_vals(dim);
//    void = psi->GetVectorValue(T, ip, psi_vals);
//    real_t norm = psi_vals.Norml2;
//    real_t phi = exp(norm)/(1.0/alpha + exp(norm));

//    V = psi_vals;
//    V *= phi;
// }

// void DZCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
//                                                 const IntegrationPoint &ip)
// {
//    MFEM_ASSERT(psi != NULL, "grid function is not set");
//    MFEM_ASSERT(alpha > 0, "alpha is not postive");

//    int dim  = psi->VectorDim();
//    Vector psi_vals(dim);
//    void = psi->GetVectorValue(T, ip, psi_vals);
//    real_t norm = psi_vals.Norml2;
//    real_t phi = exp(norm)/(1.0/alpha + exp(norm));

   

// }

real_t exact_solution(const Vector &pt)
{
   real_t x = pt(0), y = pt(1);
   real_t r = sqrt(x*x + y*y);
   real_t r0 = 0.5;
   real_t a =  0.348982574111686;
   real_t A = -0.340129705945858;

   if (r > a)
   {
      return A * log(r);
   }
   else
   {
      return sqrt(r0*r0-r*r);
   }
}