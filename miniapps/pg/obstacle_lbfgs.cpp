#include "mfem.hpp"
#include "legendre.hpp"
#include "./schur_prec.hpp"
#include "./point_pg_solver.hpp"

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
   const int max_newt_it = 1000;
   const real_t newt_tol = 1e-08;
   bool visualize_each_step = false;
   bool visualize = true;

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
   x = 0.0; rhs = 0.0;

   // unknowns
   GridFunction u_gf(&primal_fes, x.GetBlock(0));
   u_gf.ProjectBdrCoefficient(u_exact, ess_bdr);
   GridFunction lambda_gf(&latent_fes, x.GetBlock(1));

   // proximal iteration variables
   GridFunction psi_gf(&latent_fes); // sum alpha_k lambda_k
   GridFunction u_prox_prev_gf(&primal_fes);
   GridFunctionCoefficient u_prox_prev(&u_prox_prev_gf);

   psi_gf = 0.0; // initialize psi^0 = 0


   BilinearForm A(&primal_fes);
   A.AddDomainIntegrator(new DiffusionIntegrator);
   A.Assemble();
   A.EliminateEssentialBC(ess_bdr, u_gf, rhs.GetBlock(0));
   A.Finalize();

   CGSolver invA;
   invA.SetAbsTol(1e-10);
   invA.SetRelTol(0.0);
   invA.SetMaxIter(1e05);
   GSSmoother invA_prec;
   invA.SetPreconditioner(invA_prec);
   invA.SetOperator(A.SpMat());
   invA.iterative_mode = true;

   MixedBilinearForm B(&primal_fes, &latent_fes);
   B.AddDomainIntegrator(new MassIntegrator);
   B.Assemble();
   B.EliminateTrialEssentialBC(ess_bdr, u_gf, rhs.GetBlock(1));
   B.Finalize();

   real_t alpha = 1.0;
   LegendreFEFunctional entropy_op(entropy, latent_fes, alpha, true);
   CondensedGlobalPGOperator pg_op(invA, B.SpMat(), entropy_op);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   sol_sock.open(vishost,visport);
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
            << std::endl;

   GridFunctionCoefficient u_cf(&u_gf);
   LBFGSSolver pg_solver;
   pg_solver.iterative_mode = true;
   pg_solver.SetAbsTol(newt_tol);
   pg_solver.SetRelTol(0.0);
   pg_solver.SetMaxIter(max_newt_it);
   pg_solver.SetHistorySize(100);
   pg_solver.SetOperator(pg_op);
   pg_op.CondensedRHS(rhs.GetBlock(0), rhs.GetBlock(1));

   for (int prox_it = 0; prox_it < max_prox_it; prox_it++)
   {
      psi_gf.Add(alpha, lambda_gf);
      psi_gf.SetTrueVector();
      u_prox_prev_gf = u_gf;
      entropy_op.SetLatentSolution(psi_gf.GetTrueVector());
      pg_solver.Mult(rhs.GetBlock(1), x.GetBlock(1));
      pg_op.RecoverPrimal(rhs.GetBlock(0), x.GetBlock(1), x.GetBlock(0));
      u_gf.SetFromTrueDofs(x.GetBlock(0));
      real_t prox_cauchy = u_gf.ComputeL2Error(u_prox_prev);
      cout << "Proximal iteration " << prox_it + 1
           << ", ||u^{k+1} - u^k|| = " << prox_cauchy << " after "
           << pg_solver.GetNumIterations() << " LBFGS iterations." << endl;
      if (visualize)
      {
         sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
                  << std::endl;
      }
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
