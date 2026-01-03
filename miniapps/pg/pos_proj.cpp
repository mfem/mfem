#include "mfem.hpp"
#include "legendre.hpp"

using namespace std;
using namespace mfem;

real_t obstacle_func(const Vector &pt);
real_t exact_solution(const Vector &pt);
void exact_solution_gradient(const Vector &pt, Vector &grad);
real_t target_solution(const Vector &pt);

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int order = 1;
   int ref_levels = 3;
   int max_prox_it = 1000;
   real_t prox_tol = 1e-06;
   int max_newt_it = 50;
   real_t newt_tol = 1e-08;

   real_t alpha = 0.1;

   bool visualization = true;

   const char *device_config = "cpu";

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
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.ParseCheck();
   MFEMInitializePetsc(nullptr, nullptr, "rc_bddc", nullptr);

   Device device(device_config);
   if (myid == 0) { device.Print(); }
   MemoryType mt = device.GetMemoryType();

   Mesh ser_mesh = Mesh::MakeCartesian2D(16, 16, Element::QUADRILATERAL, 1.0, 1.0);
   const int dim = ser_mesh.Dimension();

   FunctionCoefficient obstacle(obstacle_func);
   FunctionCoefficient u_exact(exact_solution);
   VectorFunctionCoefficient u_grad_exact(dim, exact_solution_gradient);
   CoefficientScaledLegendreFunction entropy(new Shannon);
   entropy.SetShift(obstacle);

   for (int l = 0; l < ref_levels; l++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, ser_mesh);
   ser_mesh.Clear();

   L2_FECollection primal_fec(order, dim);
   L2_FECollection latent_fec(order, dim);

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

   BlockVector x(offsets, mt), rhs(offsets, mt);
   BlockVector tx(toffsets, mt), trhs(toffsets, mt);
   Vector bdry(num_dofs_latent, mt);
   x = 0.0; rhs = 0.0; bdry = 0.0;
   tx = 0.0; trhs = 0.0;

   // unknowns
   ParGridFunction u_gf;
   u_gf.MakeRef(&primal_fes, x.GetBlock(0), 0);
   u_gf.ParallelAssemble(tx.GetBlock(0));
   ParGridFunction lambda_gf;
   lambda_gf.MakeRef(&latent_fes, x.GetBlock(1), 0);
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
   GridFunctionCoefficient lambda_cf(&lambda_gf);

   const auto tid = Operator::Type::PETSC_MATIS;
   OperatorHandle Ah(tid), Bh(tid), Hh(tid);
   // Setup bilinear forms
   ParBilinearForm dE(&primal_fes);
   dE.AddDomainIntegrator(new MassIntegrator);
   dE.SetOperatorType(tid);
   dE.Assemble();
   dE.Finalize();
   dE.ParallelAssemble(Ah);
   PetscParMatrix &A = *Ah.As<PetscParMatrix>();

   ParMixedBilinearForm b(&primal_fes, &latent_fes);
   b.AddDomainIntegrator(new MassIntegrator);
   b.Assemble();
   b.Finalize();
   b.ParallelAssemble(Bh);
   PetscParMatrix &B = *Bh.As<PetscParMatrix>();

   std::unique_ptr<PetscParMatrix> Bt(B.Transpose());

   ParBilinearForm hessR(&latent_fes);
   hessR.AddDomainIntegrator(new MassIntegrator(du_mapped));

   ParLinearForm targ_form(&primal_fes, rhs.GetBlock(0).GetData());
   FunctionCoefficient targ_cf(target_solution);
   targ_form.AddDomainIntegrator(new DomainLFIntegrator(targ_cf));
   targ_form.Assemble();
   targ_form.ParallelAssemble(trhs.GetBlock(0));
   trhs.GetBlock(0).SyncAliasMemory(trhs);
   ParLinearForm res_form(&latent_fes, rhs.GetBlock(1).GetData());
   res_form.AddDomainIntegrator(new DomainLFIntegrator(u_mapped));

   BlockOperator pg_op(toffsets);
   pg_op.SetBlock(0, 0, &A);
   pg_op.SetBlock(1, 0, &B);
   pg_op.SetBlock(0, 1, Bt.get());
   pg_op.owns_blocks = 0;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost,visport);
      sol_sock.precision(8);
      sol_sock << "parallel " << num_procs << " " << myid << "\n"
               << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
               << std::flush;
      MPI_Barrier(MPI_COMM_WORLD);
   }

   for (int prox_it = 0; prox_it < max_prox_it; prox_it++)
   {
      psi_k_gf = psi_gf;
      psi_k_gf.SyncAliasMemory(psi_k_gf);
      u_prox_prev_gf = u_gf;
      u_prox_prev_gf.SyncAliasMemory(u_prox_prev_gf);

      bool newt_converged = false;
      for (int newt_it = 0; newt_it < max_newt_it; newt_it++)
      {
         // Store previous, and update latent variable
         u_newt_prev_gf = u_gf;
         add(psi_k_gf, alpha, lambda_gf, psi_gf);
         psi_gf.SyncAliasMemory(psi_gf);

         // Update LHS
         if (hessR.HasSpMat())
         {
            delete hessR.LoseMat();
            hessR.Update();
         }
         hessR.Assemble();
         hessR.Finalize(0);
         hessR.ParallelAssemble(Hh);
         PetscParMatrix &hess_mat = *Hh.As<PetscParMatrix>();
         hess_mat *= -alpha;
         // out << "Assembled hess from proc " << myid << std::endl;
         // MPI_Barrier(MPI_COMM_WORLD);

         // Update RHS
         res_form.Assemble();
         res_form.ParallelAssemble(trhs.GetBlock(1));
         trhs.GetBlock(1).SyncAliasMemory(trhs.GetBlock(1));
         hess_mat.AddMult(tx.GetBlock(1), trhs.GetBlock(1), 1.0);
         trhs.GetBlock(1).SyncAliasMemory(trhs.GetBlock(1));
         // out << "Assembled rhs from proc " << myid << std::endl;
         // MPI_Barrier(MPI_COMM_WORLD);

         // Update Solver
         pg_op.SetBlock(1, 1, &hess_mat);
         PetscParMatrix pg_mat(MPI_COMM_WORLD, &pg_op, tid);
         PetscBDDCSolverParams opts;
         PetscBDDCSolver pg_prec(MPI_COMM_WORLD, pg_mat, opts, "prec_");
         PetscLinearSolver pg_solver(pg_mat, "gmres_");
         pg_solver.SetPreconditioner(pg_prec);
         pg_solver.SetAbsTol(1e-10);
         pg_solver.SetRelTol(1e-08);
         pg_solver.SetMaxIter(5e03);
         pg_solver.SetPrintLevel(0);
         pg_solver.Mult(trhs, tx);
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
      alpha *= 2.0;
      if (visualization)
      {
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
                  << std::flush;
      }
      real_t prox_cauchy = u_gf.ComputeL2Error(u_prox_prev);
      real_t err = u_gf.ComputeL2Error(u_exact);
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

real_t obstacle_func(const Vector &pt)
{
   return 0.0;
}

real_t target_solution(const Vector &pt)
{
   real_t x(pt(0)), y(pt(1));
   return std::sin(2*M_PI*x) * std::sin(2*M_PI*y);
}

real_t exact_solution(const Vector &pt)
{
   real_t x(pt(0)), y(pt(1));
   real_t val = std::sin(2*M_PI*x) * std::sin(2*M_PI*y);
   return val > 0 ? val : 0.0;
}

void exact_solution_gradient(const Vector &pt, Vector &grad)
{
   real_t x(pt(0)), y(pt(1));
   real_t val = std::sin(2*M_PI*x) * std::sin(2*M_PI*y);
   grad(0) = val > 0 ? 2*M_PI * std::cos(2*M_PI*x) * std::sin(2*M_PI*y) : 0.0;
   grad(1) = val > 0 ? 2*M_PI * std::sin(2*M_PI*x) * std::cos(2*M_PI*y) : 0.0;
}
