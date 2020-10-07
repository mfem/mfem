// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_GINKGO

#include "ginkgo.hpp"
#include "sparsemat.hpp"
#include "../general/globals.hpp"
#include "../general/error.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace mfem
{

namespace GinkgoWrappers
{

GinkgoIterativeSolverBase::GinkgoIterativeSolverBase(
   const std::string &exec_type, int print_iter, int max_num_iter,
   double RTOLERANCE, double ATOLERANCE)
   : exec_type(exec_type),
     print_lvl(print_iter),
     max_iter(max_num_iter),
     rel_tol(sqrt(RTOLERANCE)),
     abs_tol(sqrt(ATOLERANCE))
{
   if (exec_type == "reference")
   {
      executor = gko::ReferenceExecutor::create();
   }
   else if (exec_type == "omp")
   {
      executor = gko::OmpExecutor::create();
   }
   else if (exec_type == "cuda" && gko::CudaExecutor::get_num_devices() > 0)
   {
      executor = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
   }
   else
   {
      mfem::err <<
                " exec_type needs to be one of the three strings: \"reference\", \"cuda\" or \"omp\" "
                << std::endl;
   }
   using ResidualCriterionFactory = gko::stop::ResidualNormReduction<>;
   residual_criterion             = ResidualCriterionFactory::build()
                                    .with_reduction_factor(rel_tol)
                                    .on(executor);

   combined_factory =
      gko::stop::Combined::build()
      .with_criteria(residual_criterion,
                     gko::stop::Iteration::build()
                     .with_max_iters(max_iter)
                     .on(executor))
      .on(executor);
}

void
GinkgoIterativeSolverBase::initialize_ginkgo_log(gko::matrix::Dense<double>* b)
{
   // Add the logger object. See the different masks available in Ginkgo's
   // documentation
   convergence_logger = gko::log::Convergence<>::create(
                           executor, gko::log::Logger::criterion_check_completed_mask);
   residual_logger = std::make_shared<ResidualLogger<>>(executor,
                                                        gko::lend(system_matrix),b);

}

void
GinkgoIterativeSolverBase::apply(Vector &solution,
                                 const Vector &rhs)
{
   // some shortcuts.
   using val_array = gko::Array<double>;
   using vec       = gko::matrix::Dense<double>;

   MFEM_VERIFY(system_matrix, "System matrix not initialized");
   MFEM_VERIFY(executor, "executor is not initialized");
   MFEM_VERIFY(rhs.Size() == solution.Size(),
               "Mismatching sizes for rhs and solution");
   // Create the rhs vector in Ginkgo's format.
   std::vector<double> f(rhs.Size());
   std::copy(rhs.GetData(), rhs.GetData() + rhs.Size(), f.begin());
   auto b =
      vec::create(executor,
                  gko::dim<2>(rhs.Size(), 1),
                  val_array::view(executor->get_master(), rhs.Size(), f.data()),
                  1);

   // Create the solution vector in Ginkgo's format.
   std::vector<double> u(solution.Size());
   std::copy(solution.GetData(), solution.GetData() + solution.Size(), u.begin());
   auto x = vec::create(executor,
                        gko::dim<2>(solution.Size(), 1),
                        val_array::view(executor->get_master(),
                                        solution.Size(),
                                        u.data()),
                        1);

   // Create the logger object to log some data from the solvers to confirm
   // convergence.
   initialize_ginkgo_log(gko::lend(b));

   MFEM_VERIFY(convergence_logger, "convergence logger not initialized" );
   if (print_lvl==1)
   {
      MFEM_VERIFY(residual_logger, "residual logger not initialized" );
      solver_gen->add_logger(residual_logger);
   }

   // Generate the solver from the solver using the system matrix.
   auto solver = solver_gen->generate(system_matrix);

   // Add the convergence logger object to the combined factory to retrieve the
   // solver and other data
   combined_factory->add_logger(convergence_logger);

   // Finally, apply the solver to b and get the solution in x.
   solver->apply(gko::lend(b), gko::lend(x));

   // The convergence_logger object contains the residual vector after the
   // solver has returned. use this vector to compute the residual norm of the
   // solution. Get the residual norm from the logger. As the convergence logger
   // returns a `linop`, it is necessary to convert it to a Dense matrix.
   // Additionally, if the logger is logging on the gpu, it is necessary to copy
   // the data to the host and hence the `residual_norm_d_master`
   auto residual_norm = convergence_logger->get_residual_norm();
   auto residual_norm_d =
      gko::as<gko::matrix::Dense<double>>(residual_norm);
   auto residual_norm_d_master =
      gko::matrix::Dense<double>::create(executor->get_master(),
                                         gko::dim<2> {1, 1});
   residual_norm_d_master->copy_from(residual_norm_d);

   // Get the number of iterations taken to converge to the solution.
   auto num_iteration = convergence_logger->get_num_iterations();

   // Ginkgo works with a relative residual norm through its
   // ResidualNormReduction criterion. Therefore, to get the normalized
   // residual, we divide by the norm of the rhs.
   auto b_norm = gko::matrix::Dense<double>::create(executor->get_master(),
                                                    gko::dim<2> {1, 1});
   if (executor != executor->get_master())
   {
      auto b_master = vec::create(executor->get_master(),
                                  gko::dim<2>(rhs.Size(), 1),
                                  val_array::view(executor->get_master(),
                                                  rhs.Size(),
                                                  f.data()),
                                  1);
      b_master->compute_norm2(b_norm.get());
   }
   else
   {
      b->compute_norm2(b_norm.get());
   }

   MFEM_VERIFY(b_norm.get()->at(0, 0) != 0.0, " rhs norm is zero");
   // Some residual norm and convergence print outs. As both
   // `residual_norm_d_master` and `b_norm` are seen as Dense matrices, we use
   // the `at` function to get the first value here. In case of multiple right
   // hand sides, this will need to be modified.
   auto fin_res_norm = std::pow(residual_norm_d_master->at(0,0) / b_norm->at(0,0),
                                2);
   if (num_iteration==max_iter &&
       fin_res_norm > rel_tol )
   {
      converged = 1;
   }
   if (fin_res_norm < rel_tol)
   {
      converged =0;
   }
   if (print_lvl ==1)
   {
      residual_logger->write();
   }
   if (converged!=0)
   {
      mfem::err << "No convergence!" << '\n';
      mfem::out << "(B r_N, r_N) = " << fin_res_norm << '\n'
                << "Number of iterations: " << num_iteration << '\n';
   }
   if (print_lvl >=2 && converged==0 )
   {
      mfem::out << "Converged in " << num_iteration <<
                " iterations with final residual norm "
                << fin_res_norm << '\n';
   }

   // Check if the solution is on a CUDA device, if so, copy it over to the
   // host.
   if (executor != executor->get_master())
   {
      auto x_master = vec::create(executor->get_master(),
                                  gko::dim<2>(solution.Size(), 1),
                                  val_array::view(executor,
                                                  solution.Size(),
                                                  x->get_values()),
                                  1);
      x.reset(x_master.release());
   }
   // Finally copy over the solution vector to mfem's solution vector.
   std::copy(x->get_values(),
             x->get_values() + solution.Size(),
             solution.GetData());
}

void
GinkgoIterativeSolverBase::initialize(
   const SparseMatrix *matrix)
{
   // Needs to be a square matrix
   MFEM_VERIFY(matrix->Height() == matrix->Width(), "System matrix is not square");

   const int N = matrix->Size();
   using mtx = gko::matrix::Csr<double, int>;
   std::shared_ptr<mtx> system_matrix_compute;
   system_matrix_compute   = mtx::create(executor->get_master(),
                                         gko::dim<2>(N),
                                         matrix->NumNonZeroElems());
   double *mat_values   = system_matrix_compute->get_values();
   int *mat_row_ptrs = system_matrix_compute->get_row_ptrs();
   int *mat_col_idxs = system_matrix_compute->get_col_idxs();
   mat_row_ptrs[0] =0;
   for (int r=0; r< N; ++r)
   {
      const int* col = matrix->GetRowColumns(r);
      const double * val = matrix->GetRowEntries(r);
      mat_row_ptrs[r+1] = mat_row_ptrs[r] + matrix->RowSize(r);
      for (int cj=0; cj < matrix->RowSize(r); cj++ )
      {
         mat_values[mat_row_ptrs[r]+cj] = val[cj];
         mat_col_idxs[mat_row_ptrs[r]+cj] = col[cj];
      }
   }
   system_matrix =
      mtx::create(executor, gko::dim<2>(N), matrix->NumNonZeroElems());
   system_matrix->copy_from(system_matrix_compute.get());
}

void
GinkgoIterativeSolverBase::solve(const SparseMatrix *matrix,
                                 Vector &solution,
                                 const Vector &rhs)
{
   initialize(matrix);
   apply(solution, rhs);
}


/* ---------------------- CGSolver ------------------------ */
CGSolver::CGSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using cg = gko::solver::Cg<double>;
   this->solver_gen =
      cg::build().with_criteria(this->combined_factory).on(this->executor);
}

CGSolver::CGSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE,
   const gko::LinOpFactory* preconditioner
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using cg         = gko::solver::Cg<double>;
   this->solver_gen = cg::build()
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner)
                      .on(this->executor);
}


/* ---------------------- BICGSTABSolver ------------------------ */
BICGSTABSolver::BICGSTABSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using bicgstab   = gko::solver::Bicgstab<double>;
   this->solver_gen = bicgstab::build()
                      .with_criteria(this->combined_factory)
                      .on(this->executor);
}

BICGSTABSolver::BICGSTABSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE,
   const gko::LinOpFactory* preconditioner
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using bicgstab   = gko::solver::Bicgstab<double>;
   this->solver_gen = bicgstab::build()
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner)
                      .on(this->executor);
}


/* ---------------------- CGSSolver ------------------------ */
CGSSolver::CGSSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using cgs = gko::solver::Cgs<double>;
   this->solver_gen =
      cgs::build().with_criteria(this->combined_factory).on(this->executor);
}

CGSSolver::CGSSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE,
   const gko::LinOpFactory* preconditioner
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using cgs        = gko::solver::Cgs<double>;
   this->solver_gen = cgs::build()
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner)
                      .on(this->executor);
}


/* ---------------------- FCGSolver ------------------------ */
FCGSolver::FCGSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using fcg = gko::solver::Fcg<double>;
   this->solver_gen =
      fcg::build().with_criteria(this->combined_factory).on(this->executor);
}

FCGSolver::FCGSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE,
   const gko::LinOpFactory* preconditioner
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using fcg        = gko::solver::Fcg<double>;
   this->solver_gen = fcg::build()
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner)
                      .on(this->executor);
}


/* ---------------------- GMRESSolver ------------------------ */
GMRESSolver::GMRESSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using gmres      = gko::solver::Gmres<double>;
   this->solver_gen = gmres::build()
                      .with_krylov_dim(m)
                      .with_criteria(this->combined_factory)
                      .on(this->executor);
}

GMRESSolver::GMRESSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE,
   const gko::LinOpFactory* preconditioner
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using gmres      = gko::solver::Gmres<double>;
   this->solver_gen = gmres::build()
                      .with_krylov_dim(m)
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner)
                      .on(this->executor);
}


/* ---------------------- IRSolver ------------------------ */
IRSolver::IRSolver(
   const std::string &   exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using ir = gko::solver::Ir<double>;
   this->solver_gen =
      ir::build().with_criteria(this->combined_factory).on(this->executor);
}

IRSolver::IRSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE,
   const gko::LinOpFactory* inner_solver
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using ir         = gko::solver::Ir<double>;
   this->solver_gen = ir::build()
                      .with_criteria(this->combined_factory)
                      .with_solver(inner_solver)
                      .on(this->executor);
}


} // namespace GinkgoWrappers

} // namespace mfem

#endif // MFEM_USE_GINKGO
