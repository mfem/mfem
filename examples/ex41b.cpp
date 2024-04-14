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
//               Find qₖ ∈ H¹₀(Ω) and uₖ ∈ H¹₀(Ω) such that
//               ( ϕₖ(|∇qₖ|) ∇qₖ , ∇w ) + ( ∇uₖ , ∇w ) = 0           ∀ w ∈ H¹₀(Ω)
//               ( ∇qₖ , ∇v )        = ( -1 , v ) + ( ∇qₖ₋₁ , ∇v )   ∀ v ∈ H¹₀(Ω)
//
//              where ϕₖ(s) = 1 / ( 1/αₖ + s² )^{1/2} and αₖ > 0.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ZCoefficient : public VectorCoefficient
{
protected:
   GridFunction *q;
   real_t alpha;

public:
   ZCoefficient(int vdim, GridFunction &q_, real_t alpha_ = 1.0)
      : VectorCoefficient(vdim), q(&q_), alpha(alpha_) { }

   virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip);
};

class DZCoefficient : public MatrixCoefficient
{
protected:
   GridFunction *q;
   real_t alpha;

public:
   DZCoefficient(int height, GridFunction &q_, real_t alpha_ = 1.0)
      : MatrixCoefficient(height, true),  q(&q_), alpha(alpha_) { }

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
   H1_FECollection H1fec(order, dim);
   FiniteElementSpace H1fes(&mesh, &H1fec);

   cout << "Number of dofs: "
        << H1fes.GetTrueVSize() * 2 << endl;

   // 5. Determine the list of true (i.e., conforming) essential boundary dofs.
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   else
   {
    ess_tdof_list.Append(0);
   }
   
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = H1fes.GetVSize();
   offsets[2] = H1fes.GetVSize();
   offsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   // 6. Define an initial guess for the solution.
   ConstantCoefficient neg_one(-1.0);
   ConstantCoefficient zero(0.0);

   // 7. Define the solution vectors as a finite element grid functions
   //    corresponding to the fespaces.
   GridFunction u_gf, delta_q_gf;

   delta_q_gf.MakeRef(&H1fes,x,offsets[0]);
   u_gf.MakeRef(&H1fes,x,offsets[1]);
   delta_q_gf = 0.0;

   GridFunction q_old_gf(&H1fes);
   GridFunction q_gf(&H1fes);
   GridFunction u_old_gf(&H1fes);
   q_old_gf = 0.0;
   u_old_gf = 0.0;

   // 8. Define the function coefficients for the solution and use them to
   //    initialize the initial guess
   q_gf = 0.0;
   q_old_gf = q_gf;
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

      ConstantCoefficient alpha_cf(alpha);

      int j;
      for ( j = 0; j < 5; j++)
      {
         total_iterations++;

         LinearForm b0,b1;
         b0.Update(&H1fes,rhs.GetBlock(0),0);
         b1.Update(&H1fes,rhs.GetBlock(1),0);

         ZCoefficient Z(sdim, q_gf, alpha);
         DZCoefficient DZ(sdim, q_gf, alpha);

         ScalarVectorProductCoefficient neg_Z(-1.0, Z);
         b0.AddDomainIntegrator(new DomainLFGradIntegrator(neg_Z));
         b0.Assemble();

         GradientGridFunctionCoefficient grad_q_cf(&q_gf);
         GradientGridFunctionCoefficient grad_q_old_cf(&q_old_gf);
         VectorSumCoefficient grad_q_old_minus_q(grad_q_old_cf, grad_q_cf, 1.0, -1.0);
         b1.AddDomainIntegrator(new DomainLFIntegrator(neg_one));
        //  b1.AddDomainIntegrator(new DomainLFGradIntegrator(grad_q_old_minus_q));
         b1.Assemble();

         BilinearForm a00(&H1fes);
        //  a00.AddDomainIntegrator(new DiffusionIntegrator());
         a00.AddDomainIntegrator(new DiffusionIntegrator(DZ));
         a00.Assemble();
         a00.EliminateVDofs(ess_tdof_list, mfem::Operator::DIAG_ZERO);
        //  a00.EliminateVDofs(ess_tdof_list,x.GetBlock(0),rhs.GetBlock(0),
        //                           mfem::Operator::DIAG_ONE);
         a00.Finalize();
         SparseMatrix &A00 = a00.SpMat();

         BilinearForm a10(&H1fes);
         a10.AddDomainIntegrator(new DiffusionIntegrator());
         a10.Assemble();
         a10.EliminateVDofs(ess_tdof_list,x.GetBlock(0),rhs.GetBlock(1),
                                  mfem::Operator::DIAG_ONE);
         a10.Finalize();
         SparseMatrix &A10 = a10.SpMat();

         SparseMatrix *A01 = Transpose(A10);

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
         A.SetBlock(0,1,A01);
         A.SetBlock(1,0,&A10);

         SparseMatrix * A_mono = A.CreateMonolithic();
         UMFPackSolver umf(*A_mono);
         umf.Mult(rhs,x);

         delta_q_gf.MakeRef(&H1fes, x.GetBlock(0), 0);
         u_gf.MakeRef(&H1fes, x.GetBlock(1), 0);

         u_tmp -= u_gf;
         real_t Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         real_t gamma = 1.0;
         delta_q_gf *= gamma;
         q_gf += delta_q_gf;

         if (visualization)
         {
            sol_sock << "solution\n" << mesh << u_tmp << "window_title 'Discrete solution'"
            // sol_sock << "solution\n" << mesh << q_gf << "window_title 'Discrete solution'"
            // sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
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
      q_old_gf = q_gf;

      if (increment_u < tol || k == max_it-1)
      {
         break;
      }

      alpha *= 2.0;

   }

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << "\n Total dofs:       " << H1fes.GetTrueVSize() * 2
             << endl;

   return 0;
}

void ZCoefficient::Eval(Vector &V, ElementTransformation &T,
                                              const IntegrationPoint &ip)
{
   MFEM_ASSERT(q != NULL, "grid function is not set");
   MFEM_ASSERT(alpha > 0, "alpha is not positive");

   Vector gradq(vdim);
   q->GetGradient(T,gradq);
   real_t norm = gradq.Norml2();
   real_t phi = 1.0 / sqrt(1.0/alpha + norm*norm);

   V = gradq;
   V *= phi;
}

void DZCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                                                const IntegrationPoint &ip)
{
   MFEM_ASSERT(q != NULL, "grid function is not set");
   MFEM_ASSERT(alpha > 0, "alpha is not positive");

   Vector gradq(height);
   q->GetGradient(T,gradq);
   real_t norm = gradq.Norml2();
   real_t phi = 1.0 / sqrt(1.0/alpha + norm*norm);

   K = 0.0;
   for (int i = 0; i < height; i++)
   {
    K(i,i) = phi;
    for (int j = 0; j < height; j++)
      {
        K(i,j) -= gradq(i) * gradq(j) * pow(phi, 3);
      }
   }
}