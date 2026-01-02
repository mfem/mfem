#include "mfem.hpp"
#include "legendre.hpp"
#include "./schur_prec.hpp"

using namespace std;
using namespace mfem;

real_t spherical_obstacle(const Vector &pt);
real_t exact_solution(const Vector &pt);
void exact_solution_gradient(const Vector &pt, Vector &grad);

int main(int argc, char *argv[])
{
   const int order = 1;
   const int ref_levels = 4;
   const int max_prox_it = 1000;
   const real_t prox_tol = 1e-06;
   const int max_newt_it = 10;
   const real_t newt_tol = 1e-08;

   Mesh mesh("../../data/disc-nurbs.mesh", 1, 1);
   const int dim = mesh.Dimension();

   FunctionCoefficient obstacle(spherical_obstacle);
   FunctionCoefficient u_exact(exact_solution);
   VectorFunctionCoefficient u_grad_exact(dim, exact_solution_gradient);
   CoefficientScaledLegendreFunction entropy(new Shannon);
   entropy.SetShift(obstacle);

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }
   {
      int curvature_order = max(order,2);
      mesh.SetCurvature(curvature_order);
      GridFunction *nodes = mesh.GetNodes();
      real_t scale = 2*sqrt(2);
      *nodes /= scale;
   }

   H1_FECollection primal_fec(order+1, dim);
   L2_FECollection latent_fec(order-1, dim);

   FiniteElementSpace primal_fes(&mesh, &primal_fec);
   FiniteElementSpace latent_fes(&mesh, &latent_fec);

   const int num_dofs_primal = primal_fes.GetTrueVSize();
   const int num_dofs_latent = latent_fes.GetTrueVSize();

   Array<int> offsets({0, num_dofs_primal, num_dofs_latent});
   offsets.PartialSum();

   Array<int> ess_bdr(mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0);
   ess_bdr = 1;
   Array<int> ess_tdof_list;
   primal_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   BlockVector x(offsets), rhs(offsets);
   Vector bdry(num_dofs_latent);
   x = 0.0; rhs = 0.0; bdry = 0.0;

   // unknowns
   GridFunction u_gf(&primal_fes, x.GetBlock(0), 0);
   u_gf.ProjectBdrCoefficient(u_exact, ess_bdr);
   GridFunction lambda_gf(&latent_fes, x.GetBlock(1), 0);
   GridFunction u_prox_prev_gf(&primal_fes);
   GridFunction u_newt_prev_gf(&primal_fes);
   GridFunctionCoefficient u_prox_prev(&u_prox_prev_gf);
   GridFunctionCoefficient u_newt_prev(&u_newt_prev_gf);

   // proximal iteration variables
   GridFunction psi_gf(&latent_fes); // sum alpha_k lambda_k
   psi_gf = 0.0; // initialize psi^0 = 0
   PrimalCoefficient u_mapped(psi_gf, entropy); // u_k = grad R^*(psi_k)
   PrimalJacobianCoefficient du_mapped(psi_gf, entropy);
   ProductCoefficient neg_alpha_du_mapped(-1.0, du_mapped);
   GridFunctionCoefficient lambda_cf(&lambda_gf);
   ProductCoefficient neg_alpha_du_mapped_lambda(neg_alpha_du_mapped, lambda_cf);
   GridFunction psi_k_gf(&latent_fes); // previous psi_k

   // Setup bilinear forms
   BilinearForm diffusion(&primal_fes);
   diffusion.AddDomainIntegrator(new DiffusionIntegrator);
   diffusion.Assemble();
   diffusion.EliminateEssentialBC(ess_bdr, u_gf, rhs.GetBlock(0));
   diffusion.Finalize();

   MixedBilinearForm coupling(&primal_fes, &latent_fes);
   coupling.AddDomainIntegrator(new MassIntegrator);
   coupling.Assemble();
   coupling.EliminateTrialEssentialBC(ess_bdr, u_gf, bdry);
   coupling.Finalize();

   std::unique_ptr<SparseMatrix> coupling_transpose(Transpose(coupling.SpMat()));

   BilinearForm hess_map(&latent_fes);
   hess_map.AddDomainIntegrator(new MassIntegrator(neg_alpha_du_mapped));
   // ConstantCoefficient eps_cf(-1e-06);
   // hess_map.AddDomainIntegrator(new DiffusionIntegrator(eps_cf));

   LinearForm previous_primal_form(&latent_fes, rhs.GetBlock(1).GetData());
   previous_primal_form.AddDomainIntegrator(new DomainLFIntegrator(u_mapped));
   previous_primal_form.AddDomainIntegrator(
      new DomainLFIntegrator(neg_alpha_du_mapped_lambda));

   BlockOperator pg_op(offsets);
   pg_op.SetBlock(0, 0, &diffusion.SpMat());
   pg_op.SetBlock(1, 0, &coupling.SpMat());
   pg_op.SetBlock(0, 1, coupling_transpose.get());
   pg_op.owns_blocks = 0;

   // BlockDiagonalPreconditioner pg_prec(offsets);
   // GSSmoother diffusion_prec(diffusion.SpMat());
   // pg_prec.SetDiagonalBlock(0, &diffusion_prec);
   // GSSmoother hess_map_prec; // placeholder
   // pg_prec.owns_blocks = 0;
   //
   SchurPrec pg_prec;

   // BiCGSTABSolver pg_solver;
   GMRESSolver pg_solver;
   pg_solver.SetPreconditioner(pg_prec);
   pg_solver.SetKDim(1000);
   pg_solver.SetOperator(pg_op);
   pg_solver.SetAbsTol(0.0);
   pg_solver.SetRelTol(1e-09);
   pg_solver.SetMaxIter(1e05);
   pg_solver.SetPrintLevel(0);
   pg_solver.iterative_mode = true;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   sol_sock.open(vishost,visport);
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
            << std::flush;
   socketstream err_sock;

   QuadratureSpace qspace(&mesh, order*2 + 2);
   QuadratureFunction err_qf(qspace);
   GridFunctionCoefficient u_cf(&u_gf);
   SumCoefficient err_cf(u_exact, u_cf, 1.0, -1.0);

   real_t alpha = 1;
   for (int prox_it = 0; prox_it < max_prox_it; prox_it++)
   {
      psi_k_gf = psi_gf;
      u_prox_prev_gf = u_gf;
      neg_alpha_du_mapped.SetAConst(-alpha);
      bool newt_converged = false;
      for (int newt_it = 0; newt_it < max_newt_it; newt_it++)
      {
         // Store previous, and update latent variable
         u_newt_prev_gf = u_gf;
         add(psi_k_gf, alpha, lambda_gf, psi_gf);

         // Update LHS
         delete hess_map.LoseMat();
         hess_map.Update();
         hess_map.Assemble(0);
         hess_map.Finalize(0);

         // Update RHS
         previous_primal_form.Assemble();
         rhs.GetBlock(1) += bdry; // account for essential BCs on primal

         // Update Solver
         pg_op.SetBlock(1, 1, &hess_map.SpMat());
         pg_prec.Update(1);
         // hess_map_prec.SetOperator(hess_map.SpMat());
         // pg_prec.SetDiagonalBlock(1, &hess_map_prec);

         // Solve
         pg_solver.Mult(rhs, x);

         // Check Newton Convergence using primal Cauchy error
         real_t newt_cauchy = u_gf.ComputeL2Error(u_newt_prev);
         out << "    Newton Iteration " << newt_it + 1
             << ", ||u^{m+1} - u^m|| = " << newt_cauchy
             << ", GMRES iters = " << pg_solver.GetNumIterations()
             << std::endl;
         if (newt_cauchy < newt_tol)
         {
            newt_converged = true;
            break;
         }
      }
      if (!newt_converged)
      {
         MFEM_WARNING("Newton solver did not converge in "
                      << max_newt_it << " iterations.");
      }
      sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
               << std::flush;
      real_t prox_cauchy = u_gf.ComputeL2Error(u_prox_prev);
      out << "Proximal iteration " << prox_it + 1
          << ", ||u^{k+1} - u^k|| = " << prox_cauchy << std::endl;
      real_t err = u_gf.ComputeL2Error(u_exact);
      // err_cf.Project(err_qf);
      // err_sock << "quadrature\n" << mesh << err_qf
      //          << "window_title 'Error at quadrature points'" << std::flush;
      // out << "L2 Error = " << err << std::endl;
      if (prox_cauchy < prox_tol)
      {
         break;
      }
   }

   return 0;
}

real_t spherical_obstacle(const Vector &pt)
{
   real_t x = pt(0), y = pt(1);
   real_t r = sqrt(x*x + y*y);
   real_t r0 = 0.5;
   real_t beta = 0.9;

   real_t b = r0*beta;
   real_t tmp = sqrt(r0*r0 - b*b);
   real_t B = tmp + b*b/tmp;
   real_t C = -b/tmp;

   if (r > b)
   {
      return B + r * C;
   }
   else
   {
      return sqrt(r0*r0 - r*r);
   }
}

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

void exact_solution_gradient(const Vector &pt, Vector &grad)
{
   real_t x = pt(0), y = pt(1);
   real_t r = sqrt(x*x + y*y);
   real_t r0 = 0.5;
   real_t a =  0.348982574111686;
   real_t A = -0.340129705945858;

   if (r > a)
   {
      grad(0) =  A * x / (r*r);
      grad(1) =  A * y / (r*r);
   }
   else
   {
      grad(0) = - x / sqrt( r0*r0 - r*r );
      grad(1) = - y / sqrt( r0*r0 - r*r );
   }
}
