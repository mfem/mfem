#include "mfem.hpp"
#include "legendre.hpp"
#include "./schur_prec.hpp"
#include "./point_pg_solver.hpp"

using namespace std;
using namespace mfem;

real_t exact_solution_1D(const Vector &pt);
real_t exact_solution_2D(const Vector &pt);
real_t exact_solution_3D(const Vector &pt);

int main(int argc, char *argv[])
{
   const int order = 3;
   const int ref_levels = 3;
   const int max_prox_it = 1000;
   const real_t prox_tol = 1e-06;
   const int max_nonlin_it = 1000;
   const real_t newt_tol = 1e-08;
   bool visualize_each_step = false;
   bool visualize = true;

   // Mesh mesh = Mesh::MakeCartesian1D(10, 1.0);
   Mesh mesh = Mesh::MakeCartesian2D(10, 10, Element::QUADRILATERAL);
   const int dim = mesh.Dimension();

   FermiDirac entropy;
   FunctionCoefficient u_exact(
      dim == 1 ? exact_solution_1D :
      dim == 2 ? exact_solution_2D :
      exact_solution_3D);

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   L2_FECollection primal_fec(order, dim);
   FiniteElementSpace primal_fes(&mesh, &primal_fec);

   L2_FECollection latent_fec(order, dim);
   FiniteElementSpace latent_fes(&mesh, &latent_fec);

   const int num_dofs_primal = primal_fes.GetTrueVSize();
   const int num_dofs_latent = latent_fes.GetTrueVSize();

   Array<int> offsets({0, num_dofs_primal, num_dofs_latent});
   offsets.PartialSum();


   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   // unknowns
   GridFunction u_gf(&primal_fes, x.GetBlock(0), 0);
   GridFunction lambda_gf(&latent_fes, x.GetBlock(1).GetData());

   // proximal iteration variables
   GridFunction psi_gf(&latent_fes); // sum alpha_k lambda_k
   GridFunction u_prox_prev_gf(&primal_fes);
   GridFunctionCoefficient u_prox_prev(&u_prox_prev_gf);

   psi_gf = 0.0; // initialize psi^0 = 0

   LinearForm f(&primal_fes, rhs.GetBlock(0).GetData());

   f.AddDomainIntegrator(new DomainLFIntegrator(u_exact));
   f.Assemble();

   BilinearForm invA(&primal_fes);
   invA.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator));
   invA.Assemble();
   invA.Finalize();

   MixedBilinearForm B(&primal_fes, &latent_fes);
   B.AddDomainIntegrator(new MassIntegrator);
   B.Assemble();
   B.Finalize();

   real_t alpha = 1.0;
   LegendreFEFunctional entropy_op(entropy, latent_fes, alpha, true);
   CondensedGlobalPGOperator pg_op(invA.SpMat(), B.SpMat(), entropy_op);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   sol_sock.open(vishost,visport);
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
            << std::flush;

   GridFunctionCoefficient u_cf(&u_gf);
   LBFGSSolver pg_solver;
   pg_solver.SetHistorySize(100);

   pg_solver.iterative_mode = true;
   pg_solver.SetAbsTol(newt_tol);
   pg_solver.SetRelTol(0.0);
   pg_solver.SetMaxIter(max_nonlin_it);
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
      MFEM_VERIFY(pg_solver.GetConverged(),
                  "Nonlinear solver did not converge in "
                  << max_nonlin_it << " iterations at proximal iteration "
                  << prox_it+1);
      cout << "Proximal iteration " << prox_it + 1
           << ", ||u^{k+1} - u^k|| = " << prox_cauchy << " after "
           << pg_solver.GetNumIterations() << " Nonlinear iterations." << endl;
      if (visualize)
      {
         sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
                  << std::flush;
      }
      if (prox_cauchy < prox_tol)
      {
         break;
      }
      alpha = prox_it+1.0;
   }


   return 0;
}

real_t exact_solution_1D(const Vector &pt)
{
   return sin(2.0*M_PI*pt(0));
}
real_t exact_solution_2D(const Vector &pt)
{
   return sin(2.0*M_PI*pt(0))*sin(2.0*M_PI*pt(1));
}
real_t exact_solution_3D(const Vector &pt)
{
   return sin(2.0*M_PI*pt(0))*sin(2.0*M_PI*pt(1))*sin(2.0*M_PI*pt(2));
}
