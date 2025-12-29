#include "linalg/hypre.hpp"
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
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int order = 1;
   int ref_levels = 5;
   int max_prox_it = 1000;
   real_t prox_tol = 1e-06;
   int max_newt_it = 3000;
   real_t newt_tol = 1e-08;

   real_t alpha = 1;

   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_prox_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&prox_tol, "-tol", "--tol",
                  "Stopping criteria based on the difference between"
                  "successive solution updates");
   args.AddOption(&alpha, "-step", "--step",
                  "Step size alpha");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   Mesh ser_mesh("../../data/disc-nurbs.mesh", 1, 1);
   const int dim = ser_mesh.Dimension();

   FunctionCoefficient obstacle(spherical_obstacle);
   FunctionCoefficient u_exact(exact_solution);
   VectorFunctionCoefficient u_grad_exact(dim, exact_solution_gradient);
   CoefficientScaledLegendreFunction entropy(new Shannon);
   entropy.SetShift(obstacle);

   for (int l = 0; l < ref_levels; l++)
   {
      ser_mesh.UniformRefinement();
   }
   {
      int curvature_order = max(order,2);
      ser_mesh.SetCurvature(curvature_order);
      GridFunction *nodes = ser_mesh.GetNodes();
      real_t scale = 2*sqrt(2);
      *nodes /= scale;
   }
   ParMesh mesh(MPI_COMM_WORLD, ser_mesh);
   ser_mesh.Clear();

   H1_FECollection primal_fec(order+1, dim);
   L2_FECollection latent_fec(order-1, dim);

   ParFiniteElementSpace primal_fes(&mesh, &primal_fec);
   ParFiniteElementSpace latent_fes(&mesh, &latent_fec);

   const int num_dofs_primal = primal_fes.GetTrueVSize();
   const int num_dofs_latent = latent_fes.GetTrueVSize();
   const HYPRE_BigInt glb_num_dofs_primal = primal_fes.GlobalTrueVSize();
   const HYPRE_BigInt glb_num_dofs_latent = latent_fes.GlobalTrueVSize();

   Array<int> offsets({0, primal_fes.GetVSize(), latent_fes.GetVSize()});
   offsets.PartialSum();
   Array<int> toffsets({0, num_dofs_primal, num_dofs_latent});
   toffsets.PartialSum();

   Array<int> ess_bdr(mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0);
   ess_bdr = 1;
   Array<int> ess_tdof_list;
   primal_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   BlockVector x(offsets), rhs(offsets);
   BlockVector tx(toffsets), trhs(toffsets);
   x = 0.0; rhs = 0.0;
   tx = 0.0; trhs = 0.0;

   // unknowns
   ParGridFunction u_gf(&primal_fes, x.GetBlock(0), 0);
   u_gf.ProjectBdrCoefficient(u_exact, ess_bdr);
   u_gf.ParallelAssemble(tx.GetBlock(0));
   ParGridFunction lambda_gf(&latent_fes, x.GetBlock(1), 0);

   // proximal iteration variables
   ParGridFunction psi_gf(&latent_fes); // sum alpha_k lambda_k
   psi_gf = 0.0; // initialize psi^0 = 0
   ParGridFunction u_prox_prev_gf(&primal_fes);
   GridFunctionCoefficient u_prox_prev(&u_prox_prev_gf);

   // Setup bilinear forms
   ParBilinearForm a(&primal_fes);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();
   HypreParMatrix A;
   a.FormLinearSystem(ess_tdof_list, x.GetBlock(0), rhs.GetBlock(0),
                      A, tx.GetBlock(0), trhs.GetBlock(0));

   ParMixedBilinearForm b(&primal_fes, &latent_fes);
   b.AddDomainIntegrator(new MassIntegrator);
   b.Assemble();
   HypreParMatrix B;
   Array<int> dummy;
   b.FormRectangularLinearSystem(ess_tdof_list, dummy, x.GetBlock(0),
                                 rhs.GetBlock(1), B,
                                 tx.GetBlock(0), trhs.GetBlock(1));

   CGSolver invA(MPI_COMM_WORLD);
   invA.SetAbsTol(1e-10);
   invA.SetRelTol(0.0);
   invA.SetMaxIter(1e05);
   HypreBoomerAMG invA_prec;
   invA_prec.SetPrintLevel(0);
   invA.SetPreconditioner(invA_prec);
   invA.SetOperator(A);
   invA.SetPrintLevel(0);
   invA.iterative_mode = true;

   LegendreFEFunctional entropy_op(entropy, latent_fes, alpha, true);
   CondensedGlobalPGOperator pg_op(invA, B, entropy_op);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   sol_sock.open(vishost,visport);
   sol_sock.precision(8);
   sol_sock << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
            << std::flush;
   MPI_Barrier(MPI_COMM_WORLD);

   GridFunctionCoefficient u_cf(&u_gf);
   LBFGSSolver pg_solver(MPI_COMM_WORLD);
   pg_solver.iterative_mode = true;
   pg_solver.SetAbsTol(newt_tol);
   pg_solver.SetRelTol(0.0);
   pg_solver.SetMaxIter(max_newt_it);
   pg_solver.SetHistorySize(1000);
   pg_solver.SetOperator(pg_op);
   pg_op.CondensedRHS(trhs.GetBlock(0), trhs.GetBlock(1));

   for (int prox_it = 0; prox_it < max_prox_it; prox_it++)
   {
      psi_gf.Add(alpha, lambda_gf);
      psi_gf.SetTrueVector();
      u_prox_prev_gf = u_gf;
      entropy_op.SetLatentSolution(psi_gf.GetTrueVector());
      pg_solver.Mult(trhs.GetBlock(1), tx.GetBlock(1));
      lambda_gf.SetFromTrueDofs(tx.GetBlock(1));
      pg_op.RecoverPrimal(trhs.GetBlock(0), tx.GetBlock(1), tx.GetBlock(0));
      u_gf.SetFromTrueDofs(tx.GetBlock(0));
      real_t prox_cauchy = u_gf.ComputeL2Error(u_prox_prev);
      if (myid == 0)
      {
         cout << "Proximal iteration " << prox_it + 1
              << ", ||u^{k+1} - u^k|| = " << prox_cauchy << " after "
              << pg_solver.GetNumIterations() << " LBFGS iterations." << endl;
      }
      if (visualization)
      {
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
                  << std::flush;
         MPI_Barrier(MPI_COMM_WORLD);
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
