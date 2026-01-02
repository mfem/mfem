#include "mfem.hpp"
#include "legendre.hpp"

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
   int max_newt_it = 10;
   real_t newt_tol = 1e-08;

   real_t alpha = 0.1;

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
   Vector bdry(num_dofs_latent);
   x = 0.0; rhs = 0.0; bdry = 0.0;
   tx = 0.0; trhs = 0.0;

   // unknowns
   ParGridFunction u_gf(&primal_fes, x.GetBlock(0), 0);
   u_gf.ProjectBdrCoefficient(u_exact, ess_bdr);
   u_gf.ParallelAssemble(tx.GetBlock(0));
   ParGridFunction lambda_gf(&latent_fes, x.GetBlock(1), 0);
   ParGridFunction u_prox_prev_gf(&primal_fes);
   ParGridFunction u_newt_prev_gf(&primal_fes);
   GridFunctionCoefficient u_prox_prev(&u_prox_prev_gf);
   GridFunctionCoefficient u_newt_prev(&u_newt_prev_gf);

   // proximal iteration variables
   ParGridFunction psi_gf(&latent_fes); // sum alpha_k lambda_k
   psi_gf = 0.0; // initialize psi^0 = 0
   ParGridFunction psi_k_gf(&latent_fes); // previous psi_k

   PrimalCoefficient u_mapped(psi_gf, entropy); // u_k = grad R^*(psi_k)
   PrimalJacobianCoefficient du_mapped(psi_gf, entropy);
   ProductCoefficient neg_alpha_du_mapped(-1.0, du_mapped);
   GridFunctionCoefficient lambda_cf(&lambda_gf);
   ProductCoefficient neg_alpha_du_mapped_lambda(neg_alpha_du_mapped, lambda_cf);

   // Setup bilinear forms
   ParBilinearForm diffusion(&primal_fes);
   diffusion.AddDomainIntegrator(new DiffusionIntegrator);
   diffusion.Assemble();
   HypreParMatrix diffusion_mat;
   diffusion.FormLinearSystem(ess_tdof_list, x.GetBlock(0), rhs.GetBlock(0),
                              diffusion_mat, tx.GetBlock(0), trhs.GetBlock(0));

   ParMixedBilinearForm coupling(&primal_fes, &latent_fes);
   coupling.AddDomainIntegrator(new MassIntegrator);
   coupling.Assemble();
   HypreParMatrix coupling_mat;
   Array<int> dummy;
   coupling.FormRectangularLinearSystem(ess_tdof_list, dummy, x.GetBlock(0),
                                        rhs.GetBlock(1), coupling_mat,
                                        tx.GetBlock(0), bdry);

   std::unique_ptr<HypreParMatrix> coupling_transpose(coupling_mat.Transpose());

   ParBilinearForm hess_map(&latent_fes);
   hess_map.AddDomainIntegrator(new MassIntegrator(neg_alpha_du_mapped));
   ConstantCoefficient eps_cf(-1e-06);
   hess_map.AddDomainIntegrator(new DiffusionIntegrator(eps_cf));

   ParLinearForm previous_primal_form(&latent_fes, rhs.GetBlock(1).GetData());
   previous_primal_form.AddDomainIntegrator(new DomainLFIntegrator(u_mapped));
   previous_primal_form.AddDomainIntegrator(
      new DomainLFIntegrator(neg_alpha_du_mapped_lambda));

   BlockOperator pg_op(toffsets);
   pg_op.SetBlock(0, 0, &diffusion_mat);
   pg_op.SetBlock(1, 0, &coupling_mat);
   pg_op.SetBlock(0, 1, coupling_transpose.get());
   pg_op.owns_blocks = 0;

   BlockDiagonalPreconditioner pg_prec(toffsets);
   HypreBoomerAMG diffusion_prec(diffusion_mat);
   diffusion_prec.SetPrintLevel(0);
   pg_prec.SetDiagonalBlock(0, &diffusion_prec);
   HypreSmoother hess_map_prec; // placeholder
   pg_prec.owns_blocks = 0;

   GMRESSolver pg_solver(MPI_COMM_WORLD);
   pg_solver.SetPreconditioner(pg_prec);
   pg_solver.SetKDim(10000);
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
   sol_sock << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
            << std::flush;
   MPI_Barrier(MPI_COMM_WORLD);

   QuadratureSpace qspace(&mesh, order*2 + 2);
   QuadratureFunction err_qf(qspace);
   GridFunctionCoefficient u_cf(&u_gf);
   SumCoefficient err_cf(u_exact, u_cf, 1.0, -1.0);

   socketstream err_sock;

   for (int prox_it = 0; prox_it < max_prox_it; prox_it++)
   {
      psi_k_gf = psi_gf;
      u_prox_prev_gf = u_gf;
      neg_alpha_du_mapped.SetAConst(-alpha);
      // out << "HI from proc " << myid << std::endl;
      // MPI_Barrier(MPI_COMM_WORLD);

      bool newt_converged = false;
      for (int newt_it = 0; newt_it < max_newt_it; newt_it++)
      {
         // Store previous, and update latent variable
         u_newt_prev_gf = u_gf;
         add(psi_k_gf, alpha, lambda_gf, psi_gf);

         // Update LHS
         if (hess_map.HasSpMat())
         {
            delete hess_map.LoseMat();
            hess_map.Update();
         }
         hess_map.Assemble();
         hess_map.Finalize(0);
         std::unique_ptr<HypreParMatrix> hess_mat(hess_map.ParallelAssemble());
         // out << "Assembled hess from proc " << myid << std::endl;
         // MPI_Barrier(MPI_COMM_WORLD);

         // Update RHS
         previous_primal_form.Assemble();
         previous_primal_form.ParallelAssemble(trhs.GetBlock(1));
         trhs.GetBlock(1) += bdry; // account for essential BCs on primal
         // out << "Assembled rhs from proc " << myid << std::endl;
         // MPI_Barrier(MPI_COMM_WORLD);

         // Update Solver
         pg_op.SetBlock(1, 1, hess_mat.get());
         hess_map_prec.SetOperator(*hess_mat);
         pg_prec.SetDiagonalBlock(1, &hess_map_prec);
         // out << "Set Solver from proc " << myid << std::endl;
         // MPI_Barrier(MPI_COMM_WORLD);

         // Solve
         pg_solver.Mult(trhs, tx);
         // out << "Solved from proc " << myid << std::endl;
         // MPI_Barrier(MPI_COMM_WORLD);
         u_gf.SetFromTrueDofs(tx.GetBlock(0));
         lambda_gf.SetFromTrueDofs(tx.GetBlock(1));

         // Check Newton Convergence using primal Cauchy error
         real_t newt_cauchy = u_gf.ComputeL2Error(u_newt_prev);
         if (myid == 0)
         {
            out << "    Newton Iteration " << newt_it + 1
                << ", ||u^{m+1} - u^m|| = " << newt_cauchy
                << ", GMRES iters = " << pg_solver.GetNumIterations()
                << std::endl;
         }
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
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
               << std::flush;
      real_t prox_cauchy = u_gf.ComputeL2Error(u_prox_prev);
      real_t err = u_gf.ComputeL2Error(u_exact);
      err_cf.Project(err_qf);
      if (!err_sock.is_open())
      {
         err_sock.open(vishost,visport);
         err_sock.precision(8);
      }
      err_sock << "parallel " << num_procs << " " << myid << "\n";
      err_sock << "quadrature\n" << mesh << err_qf
               << "window_title 'Error at quadrature points'" << std::flush;
      if (myid == 0)
      {
         out << "Proximal iteration " << prox_it + 1
             << ", ||u^{k+1} - u^k|| = " << prox_cauchy << std::endl;
         out << "L2 Error = " << err << std::endl;
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
