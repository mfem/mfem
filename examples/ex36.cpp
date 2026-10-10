//                                MFEM Example 36
//
// Compile with: make ex36
//
// Sample runs: ex36 -o 2
//              ex36 -o 2 -r 4
//
// Description: This example code demonstrates the use of MFEM to solve the
//              bound-constrained energy minimization problem
//
//                      minimize ||∇u||² subject to u ≥ ϕ in H¹₀.
//
//              This is known as the obstacle problem, and it is a simple
//              mathematical model for contact mechanics.
//
//              In this example, the obstacle ϕ is a half-sphere centered
//              at the origin of a circular domain Ω. After solving to a
//              specified tolerance, the numerical solution is compared to
//              a closed-form exact solution to assess accuracy.
//
//              The problem is discretized and solved using the proximal
//              Galerkin finite element method, introduced by Keith and
//              Surowiec [1].
//
//              This example highlights the ability of MFEM to deliver high-
//              order solutions to variation inequality problems and
//              showcases how to set up and solve nonlinear mixed methods
//              using mfem::NewtonSolver.
//
// [1] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
//     preserving finite element method for pointwise bound constraints.
//     arXiv:2307.12444 [math.NA]

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t spherical_obstacle(const Vector &pt);
real_t exact_solution_obstacle(const Vector &pt);
void exact_solution_gradient_obstacle(const Vector &pt, Vector &grad);

class LogarithmGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u;
   Coefficient *obstacle;

public:
   LogarithmGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_)
      : u(&u_), obstacle(&obst_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};

class ExponentialGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u;
   Coefficient *obstacle;

public:
   ExponentialGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_)
      : u(&u_), obstacle(&obst_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};

/**
 * @brief Nonlinear residual/Jacobian operator for one proximal step of the
 *        proximal-Galerkin obstacle problem, for use with mfem::NewtonSolver.
 *
 * This uses the equivalent (u, lambda) formulation: instead of solving
 * directly for the latent variable psi, we solve for the pair
 * (u in H1, lambda in L2) and recover psi afterwards. Acting on the monolithic
 * true-dof vector X = [U ; Lambda], this operator evaluates the block residual
 *
 *   F(X)[v] = (grad u, grad v) - (lambda, v) - (f, v)
 *   F(X)[w] = (u - exp(psi_cur) - phi, w)
 *
 * (Mult) and assembles the block tangent (GetGradient)
 *
 *   J(X) = |  (grad du, grad v)             -(dlambda, v)          |
 *          |  (du, w)            alpha (exp(psi_cur) dlambda, w)   |
 *
 * where psi_cur = psi_old - alpha*lambda is the current latent variable.
 */
class ProximalGalerkinOperator : public Operator
{
private:
   FiniteElementSpace &H1fes;
   FiniteElementSpace &L2fes;
   real_t alpha;
   Coefficient &load;
   Coefficient &obstacle;
   GridFunction &psi_old_gf;
   const Array<int> &ess_tdof_list;
   const Array<int> &block_toffsets;

   mutable ConstantCoefficient zero;
   mutable GridFunction psi_cur_gf;
   mutable GridFunction u_gf, lambda_gf;

   // Constant Jacobian blocks
   SparseMatrix *A00 = nullptr;
   SparseMatrix *A10 = nullptr;
   SparseMatrix *A01 = nullptr;

   // psi-dependent block, rebuilt each GetGradient()
   mutable SparseMatrix *A11 = nullptr;
   mutable BlockOperator *J  = nullptr;

public:
   ProximalGalerkinOperator(FiniteElementSpace &H1fes_,
                            FiniteElementSpace &L2fes_,
                            real_t alpha_, Coefficient &load_,
                            Coefficient &obstacle_,
                            GridFunction &psi_old_gf_,
                            const Array<int> &ess_tdof_list_,
                            const Array<int> &block_toffsets_);

   /// Evaluate the block residual R = F(X).
   void Mult(const Vector &x, Vector &y) const override;

   /// Assemble and return the block Jacobian J(X).
   Operator &GetGradient(const Vector &x) const override;

   /// Update the proximal parameter for the next proximal step.
   void UpdateAlpha(real_t alpha_)
   {
      alpha = alpha_;
   }

   ~ProximalGalerkinOperator() override
   {
      delete J;
      delete A11;
      delete A01;
      delete A10;
      delete A00;
   }
};

/**
 * @brief Block-lower-triangular preconditioner for the proximal-Galerkin
 *        Jacobian, rebuilt for the current Jacobian on each NewtonSolver
 *        iteration.
 *
 * The (0,0), (0,1), and (1,0) blocks are constant for the whole run, so they
 * are preconditioned once; only the preconditioner for the psi-dependent (1,1)
 * block is rebuilt each call.
 */
class ProximalGalerkinPreconditioner : public Solver
{
private:
   BlockLowerTriangularPreconditioner prec;
   SparseMatrix *GDGt = nullptr;
   SparseMatrix *S = nullptr;
   GSSmoother *P00 = nullptr;
   GSSmoother *P11 = nullptr;

public:
   ProximalGalerkinPreconditioner(const Array<int> &toffsets)
      : Solver(toffsets.Last()), prec(toffsets) { };

   void SetOperator(const Operator &op) override;

   void Mult(const Vector &x, Vector &y) const override { prec.Mult(x, y); }

   ~ProximalGalerkinPreconditioner() override
   {
      delete GDGt;
      delete S;
      delete P00;
      delete P11;
   }
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
                  "Maximum number of proximal iterations");
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
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

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

   // 3C. Rescale the domain to a unit circle (radius = 1).
   GridFunction *nodes = mesh.GetNodes();
   {
      const real_t scale = 2*sqrt(2);
      *nodes /= scale;
   }

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection H1fec(order+1, dim);
   FiniteElementSpace H1fes(&mesh, &H1fec);

   L2_FECollection L2fec(order-1, dim);
   FiniteElementSpace L2fes(&mesh, &L2fec);

   int num_dofs_H1 = H1fes.GetTrueVSize();
   int num_dofs_L2 = L2fes.GetTrueVSize();
   mfem::out << "Number of H1 finite element unknowns: "
             << num_dofs_H1 << endl;
   mfem::out << "Number of L2 finite element unknowns: "
             << num_dofs_L2 << endl;

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = H1fes.GetVSize();
   offsets[2] = L2fes.GetVSize();
   offsets.PartialSum();

   BlockVector x(offsets);

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Define an initial guess for the solution.
   auto IC_func = [](const Vector &x)
   {
      const real_t r0 = 1.0;
      return r0 * r0 - x * x;
   };
   ConstantCoefficient zero(0.0);

   // 7. Define the solution vectors as finite element grid functions
   //    corresponding to the fespaces.
   GridFunction u_gf(&H1fes);
   GridFunction u_old_gf(&H1fes);
   GridFunction psi_gf(&L2fes);
   GridFunction psi_old_gf(&L2fes);
   GridFunction lambda_gf(&L2fes);
   u_gf = 0.0;
   u_old_gf = 0.0;
   psi_gf = 0.0;
   psi_old_gf = 0.0;
   lambda_gf = 0.0;

   // 8. Define the function coefficients for the solution and use them to
   //    initialize the initial guess.
   FunctionCoefficient exact_coef(exact_solution_obstacle);
   VectorFunctionCoefficient exact_grad_coef(dim,exact_solution_gradient_obstacle);
   FunctionCoefficient IC_coef(IC_func);
   ConstantCoefficient f(0.0);
   FunctionCoefficient obstacle(spherical_obstacle);
   u_gf.ProjectCoefficient(IC_coef);
   u_old_gf = u_gf;

   // 9. Initialize the slack variable ψₕ = ln(uₕ - ϕ).
   LogarithmGridFunctionCoefficient ln_u(u_gf, obstacle);
   psi_gf.ProjectCoefficient(ln_u);
   psi_old_gf = psi_gf;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost,visport);
      sol_sock.precision(8);
   }

   // 10. Set up the nonlinear operator, the block linear solver used to invert
   //     each Jacobian, and the Newton solver. The operator encapsulates the
   //     residual b(.) and Jacobian a(.,.).
   ProximalGalerkinOperator pg_op(H1fes, L2fes, alpha, f, obstacle,
                                  psi_old_gf, ess_tdof_list, offsets);

   ProximalGalerkinPreconditioner prec(offsets);
   GMRESSolver gmres;
   gmres.SetPrintLevel(-1);
   gmres.SetRelTol(1e-8);
   gmres.SetMaxIter(1000);
   gmres.SetPreconditioner(prec);

   const int newton_max_it = 10;
   const real_t newton_rel_tol = 1e-3;
   NewtonSolver newton;
   newton.SetOperator(pg_op);
   newton.SetSolver(gmres);
   newton.SetPrintLevel(-1);
   newton.SetRelTol(newton_rel_tol);
   newton.SetAbsTol(0.0);
   newton.SetMaxIter(newton_max_it);
   newton.iterative_mode = true;

   u_gf.GetTrueDofs(x.GetBlock(0));
   x.GetBlock(1) = 0.0;
   Vector zero_rhs;

   // 11. Outer proximal loop.
   int k;
   int total_iterations = 0;
   real_t increment_u = 0.1;
   for (k = 0; k < max_it; k++)
   {
      mfem::out << "\nOUTER ITERATION " << k+1 << endl;

      // Fresh multiplier for this proximal step.
      x.GetBlock(1) = 0.0;

      // Solve the nonlinear proximal subproblem F(X) = 0.
      newton.Mult(zero_rhs, x);
      const real_t newton_iter = newton.GetNumIterations();
      total_iterations += newton_iter;

      // Distribute the result back to the grid functions.
      u_gf.SetFromTrueDofs(x.GetBlock(0));
      lambda_gf.SetFromTrueDofs(x.GetBlock(1));

      // Recover the latent variable.
      psi_gf = psi_old_gf;
      psi_gf.Add(-alpha, lambda_gf);

      if (visualization)
      {
         sol_sock << "solution\n" << mesh << u_gf
                  << "window_title 'Discrete solution'" << flush;
      }

      // Increment || u_h - u_h_prev ||.
      GridFunction u_diff(&H1fes);
      u_diff = u_gf;
      u_diff -= u_old_gf;
      increment_u = u_diff.ComputeL2Error(zero);

      mfem::out << "Number of Newton iterations = " << newton_iter << endl;
      mfem::out << "Increment (|| uₕ - uₕ_prvs||) = " << increment_u << endl;

      // Advance the proximal iterates.
      u_old_gf = u_gf;
      psi_old_gf = psi_gf;

      if (increment_u < tol || k == max_it-1)
      {
         break;
      }

      const real_t H1_error = u_gf.ComputeH1Error(&exact_coef,&exact_grad_coef);
      mfem::out << "H1-error  (|| u - uₕᵏ||)       = " << H1_error << endl;

      // Sync the operator with the current proximal parameter.
      pg_op.UpdateAlpha(alpha);
   }

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << "\n Total dofs:       " << num_dofs_H1 + num_dofs_L2
             << endl;

   // 12. Exact solution.
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
      const real_t L2_error = u_gf.ComputeL2Error(exact_coef);
      const real_t H1_error = u_gf.ComputeH1Error(&exact_coef,&exact_grad_coef);

      ExponentialGridFunctionCoefficient u_alt_cf(psi_gf,obstacle);
      GridFunction u_alt_gf(&L2fes);
      u_alt_gf.ProjectCoefficient(u_alt_cf);
      const real_t L2_error_alt = u_alt_gf.ComputeL2Error(exact_coef);

      mfem::out << "\n Final L2-error (|| u - uₕ||)          = "
                << L2_error << endl;
      mfem::out << " Final H1-error (|| u - uₕ||)          = "
                << H1_error << endl;
      mfem::out << " Final L2-error (|| u - ϕ - exp(ψₕ)||) = "
                << L2_error_alt << endl;
   }

   return 0;
}

real_t LogarithmGridFunctionCoefficient::Eval(ElementTransformation &T,
                                              const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   return log(u->GetValue(T, ip) - obstacle->Eval(T, ip));
}

real_t ExponentialGridFunctionCoefficient::Eval(ElementTransformation &T,
                                                const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   return exp(u->GetValue(T, ip)) + obstacle->Eval(T, ip);
}

real_t spherical_obstacle(const Vector &pt)
{
   const real_t r = pt.Norml2();
   const real_t r0 = 0.5;
   const real_t beta = 0.9;
   const real_t b = r0*beta;

   if (r > b)
   {
      const real_t tmp = sqrt(r0*r0 - b*b);
      const real_t B = tmp + b*b/tmp;
      const real_t C = -b/tmp;
      return B + r * C;
   }
   else
   {
      return sqrt(r0*r0 - r*r);
   }
}

real_t exact_solution_obstacle(const Vector &pt)
{
   const real_t r = pt.Norml2();
   const real_t r0 = 0.5;
   const real_t a =  0.348982574111686;
   const real_t A = -0.340129705945858;

   if (r > a)
   {
      return A * log(r);
   }
   else
   {
      return sqrt(r0*r0 - r*r);
   }
}

void exact_solution_gradient_obstacle(const Vector &pt, Vector &grad)
{
   const real_t r = pt.Norml2();
   const real_t r0 = 0.5;
   const real_t a  = 0.348982574111686;
   const real_t A = -0.340129705945858;

   grad = pt;
   if (r > a)
   {
      grad *= A / (r*r);
   }
   else
   {
      grad *= -1.0 / sqrt(r0*r0 - r*r);
   }
}

ProximalGalerkinOperator::ProximalGalerkinOperator(
   FiniteElementSpace &H1fes_,
   FiniteElementSpace &L2fes_,
   real_t alpha_, Coefficient &load_,
   Coefficient &obstacle_,
   GridFunction &psi_old_gf_,
   const Array<int> &ess_tdof_list_,
   const Array<int> &block_toffsets_)
   : Operator(block_toffsets_.Last()),
     H1fes(H1fes_), L2fes(L2fes_), alpha(alpha_), load(load_),
     obstacle(obstacle_), psi_old_gf(psi_old_gf_),
     ess_tdof_list(ess_tdof_list_), block_toffsets(block_toffsets_),
     zero(0.0), u_gf(&H1fes_), lambda_gf(&L2fes_)
{
   BilinearForm a00(&H1fes);
   a00.AddDomainIntegrator(new DiffusionIntegrator());
   a00.Assemble();
   a00.Finalize();
   A00 = a00.LoseMat();
   // Eliminate the essential rows and columns.
   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      A00->EliminateRowCol(ess_tdof_list[i], Operator::DIAG_ONE);
   }

   MixedBilinearForm a10(&H1fes, &L2fes);
   a10.AddDomainIntegrator(new MixedScalarMassIntegrator());
   a10.Assemble();
   a10.Finalize();
   A10 = new SparseMatrix(a10.SpMat());
   {
      Array<int> col_marker;
      FiniteElementSpace::ListToMarker(ess_tdof_list, A10->Width(), col_marker);
      A10->EliminateCols(col_marker);
   }

   A01 = Transpose(*A10);

   J = new BlockOperator(block_toffsets);
   J->SetBlock(0, 0, A00);
   J->SetBlock(1, 0, A10);
   J->SetBlock(0, 1, A01, -1.0);

   psi_cur_gf.SetSpace(&L2fes);
}

void ProximalGalerkinOperator::Mult(const Vector &x, Vector &y) const
{
   const BlockVector xb(const_cast<Vector&>(x), block_toffsets);
   u_gf.SetFromTrueDofs(xb.GetBlock(0));
   lambda_gf.SetFromTrueDofs(xb.GetBlock(1));

   GradientGridFunctionCoefficient grad_u(&u_gf);
   GridFunctionCoefficient lambda_cf(&lambda_gf);
   SumCoefficient neg_lambda_load(lambda_cf, load, -1.0, -1.0);

   LinearForm b0(&H1fes);
   b0.AddDomainIntegrator(new DomainLFGradIntegrator(grad_u));
   b0.AddDomainIntegrator(new DomainLFIntegrator(neg_lambda_load));
   b0.Assemble();

   GridFunctionCoefficient u_cf(&u_gf);
   psi_cur_gf = psi_old_gf;
   psi_cur_gf.Add(-alpha, lambda_gf);
   ExponentialGridFunctionCoefficient exp_psi(psi_cur_gf, obstacle);
   SumCoefficient u_minus_exp_psi(u_cf, exp_psi, 1.0, -1.0);

   LinearForm b1(&L2fes);
   b1.AddDomainIntegrator(new DomainLFIntegrator(u_minus_exp_psi));
   b1.Assemble();

   BlockVector yb(y, block_toffsets);
   Vector &y0 = yb.GetBlock(0);
   y0 = b0;
   yb.GetBlock(1) = b1;

   // Hold the essential (H1) dofs fixed: zero residual there.
   y0.SetSubVector(ess_tdof_list, 0.0);
}

Operator &ProximalGalerkinOperator::GetGradient(const Vector &x) const
{
   const BlockVector xb(const_cast<Vector&>(x), block_toffsets);
   lambda_gf.SetFromTrueDofs(xb.GetBlock(1));

   psi_cur_gf = psi_old_gf;
   psi_cur_gf.Add(-alpha, lambda_gf);
   ExponentialGridFunctionCoefficient exp_psi(psi_cur_gf, zero);
   ProductCoefficient alpha_exp_psi(alpha, exp_psi);

   BilinearForm a11(&L2fes);
   a11.AddDomainIntegrator(new MassIntegrator(alpha_exp_psi));
   a11.Assemble();
   a11.Finalize();
   delete A11;
   A11 = a11.LoseMat();

   J->SetBlock(1, 1, A11);
   return *J;
}

void ProximalGalerkinPreconditioner::SetOperator(const Operator &op)
{
   BlockOperator &Jop =
      const_cast<BlockOperator&>(dynamic_cast<const BlockOperator&>(op));
   SparseMatrix &A11 = dynamic_cast<SparseMatrix&>(Jop.GetBlock(1, 1));

   if (P00 == nullptr)
   {
      SparseMatrix &A00 = dynamic_cast<SparseMatrix&>(Jop.GetBlock(0, 0));
      SparseMatrix &A10 = dynamic_cast<SparseMatrix&>(Jop.GetBlock(1, 0));

      P00 = new GSSmoother(A00);

      // GDGt = A10 diag(A00)^{-1} A10^T
      Vector d;
      A00.GetDiag(d);
      d.Reciprocal();
      SparseMatrix *A10T = Transpose(A10);
      A10T->ScaleRows(d);
      GDGt = mfem::Mult(A10, *A10T);
      delete A10T;

      prec.SetDiagonalBlock(0, P00);
      prec.SetBlock(1, 0, &A10);
   }

   // Build an approximate Schur complement of the A11 block:
   //   S = A11 - A10 diag(A00)^{-1} A01
   //     = A11 + A10 diag(A00)^{-1} A10^T.
   delete S;
   delete P11;
   S = mfem::Add(1.0, A11, 1.0, *GDGt);
   P11 = new GSSmoother(*S);

   prec.SetDiagonalBlock(1, P11);
}
