// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

GinkgoIterativeSolver::GinkgoIterativeSolver(
   const std::string &exec_type, int print_iter, int max_num_iter,
   double RTOLERANCE, double ATOLERANCE)
   : Solver(),
     exec_type(exec_type),
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
GinkgoIterativeSolver::initialize_ginkgo_log(gko::matrix::Dense<double>* b)
const
{
   // Add the logger object. See the different masks available in Ginkgo's
   // documentation
   convergence_logger = gko::log::Convergence<>::create(
                           executor, gko::log::Logger::criterion_check_completed_mask);
   std::shared_ptr<gko::LinOp> gko_oper;
   if (system_matrix)
   {
      gko_oper = system_matrix;
   }
   else
   {
      gko_oper = wrapped_oper;
   }
   //   residual_logger = std::make_shared<ResidualLogger<>>(executor,
   //                                                        gko::lend(system_matrix),b);
   residual_logger = std::make_shared<ResidualLogger<>>(executor,
                                                        gko::lend(gko_oper),b);

}

void OperatorWrapper::apply_impl(const gko::LinOp *b, gko::LinOp *x) const
{

   // Cast to VectorWrapper; only accept this type for this impl
   const VectorWrapper *mfem_b = gko::as<const VectorWrapper>(b);
   VectorWrapper *mfem_x = gko::as<VectorWrapper>(x);

   this->mfem_oper_->Mult(mfem_b->get_mfem_vec_const_ref(),
                          mfem_x->get_mfem_vec_ref());
}
void OperatorWrapper::apply_impl(const gko::LinOp *alpha,
                                 const gko::LinOp *b,
                                 const gko::LinOp *beta,
                                 gko::LinOp *x) const
{
   // x = alpha * op (b) + beta * x
   // Cast to VectorWrapper; only accept this type for this impl
   const VectorWrapper *mfem_b = gko::as<const VectorWrapper>(b);
   VectorWrapper *mfem_x = gko::as<VectorWrapper>(x);

   // Check that alpha and beta are Dense<double> of size (1,1):
   // TODO: replace with MFEM error checking...
   if (alpha->get_size()[0] > 1 || alpha->get_size()[1] > 1)
   {
      throw gko::BadDimension(
         __FILE__, __LINE__, __func__, "alpha", alpha->get_size()[0],
         alpha->get_size()[1],
         "Expected an object of size [1 x 1] for scaling "
         " in this operator's apply_impl");
   }
   if (beta->get_size()[0] > 1 || beta->get_size()[1] > 1)
   {
      throw gko::BadDimension(
         __FILE__, __LINE__, __func__, "beta", beta->get_size()[0],
         beta->get_size()[1],
         "Expected an object of size [1 x 1] for scaling "
         " in this operator's apply_impl");
   }
   double alpha_f;
   double beta_f;

   if (alpha->get_executor() == alpha->get_executor()->get_master())
   {
      // Access value directly
      alpha_f = gko::as<gko::matrix::Dense<double>>(alpha)->at(0, 0);
   }
   else
   {
      // Copy from device to host
      this->get_executor()->get_master().get()->copy_from(this->get_executor().get(),
                                                          1, gko::as<gko::matrix::Dense<double>>(alpha)->get_const_values(),
                                                          &alpha_f);
   }
   if (beta->get_executor() == beta->get_executor()->get_master())
   {
      // Access value directly
      beta_f = gko::as<gko::matrix::Dense<double>>(beta)->at(0, 0);
   }
   else
   {
      // Copy from device to host
      this->get_executor()->get_master().get()->copy_from(this->get_executor().get(),
                                                          1, gko::as<gko::matrix::Dense<double>>(beta)->get_const_values(),
                                                          &beta_f);
   }
   // Scale x by beta
   mfem_x->get_mfem_vec_ref() *= beta_f;
   // Multiply operator with b and store in tmp
   mfem::Vector mfem_tmp =
      mfem::Vector(mfem_x->get_size()[0],
                   mfem_x->get_mfem_vec_ref().GetMemory().GetMemoryType());
   // Set UseDevice flag to match mfem_x (not automatically done through
   //  MemoryType)
   mfem_tmp.UseDevice(mfem_x->get_mfem_vec_ref().UseDevice());

   // Apply the operator
   this->mfem_oper_->Mult(mfem_b->get_mfem_vec_const_ref(), mfem_tmp);
   // Scale tmp by alpha and add
   mfem_x->get_mfem_vec_ref().Add(alpha_f, mfem_tmp);

   mfem_tmp.Destroy();
}

void
GinkgoIterativeSolver::Mult(const Vector &x, Vector &y) const
{

   MFEM_VERIFY(system_matrix ||
               wrapped_oper, "System matrix or operator not initialized");
   MFEM_VERIFY(executor, "executor is not initialized");
   MFEM_VERIFY(y.Size() == x.Size(),
               "Mismatching sizes for rhs and solution");

   using vec       = gko::matrix::Dense<double>;
   if (!iterative_mode)
   {
      y = 0.0;
   }

   // Create x and y vectors in Ginkgo's format. Wrap MFEM's data directly,
   // on CPU or GPU.
   bool on_device = false;
   if (executor != executor->get_master())
   {
      on_device = true;
   }
   std::unique_ptr<vec> gko_x;
   std::unique_ptr<vec> gko_y;

   if (system_matrix)
   {
      gko_x = vec::create(executor, gko::dim<2> {x.Size(), 1},
                          gko::Array<double>::view(executor,
                                                   x.Size(), const_cast<double *>(
                                                      x.Read(on_device))), 1);
      gko_y = vec::create(executor, gko::dim<2> {y.Size(), 1},
                          gko::Array<double>::view(executor,
                                                   y.Size(), y.ReadWrite(on_device)), 1);
   }
   else
   {
      gko_x = std::unique_ptr<vec>(new VectorWrapper(executor, x.Size(),
                                                     const_cast<Vector *>(&x), on_device, false));
      gko_y = std::unique_ptr<vec>(new VectorWrapper(executor, y.Size(), &y,
                                                     on_device, false));
   }

   // Create the logger object to log some data from the solvers to confirm
   // convergence.
   initialize_ginkgo_log(gko::lend(gko_x));

   MFEM_VERIFY(convergence_logger, "convergence logger not initialized" );
   if (print_lvl==1)
   {
      MFEM_VERIFY(residual_logger, "residual logger not initialized" );
      solver_gen->add_logger(residual_logger);
   }

   // Generate the solver from the solver using the system matrix.
   std::shared_ptr<gko::LinOp> oper_to_generate;
   if (system_matrix)
   {
      oper_to_generate = gko::as<gko::LinOp>(system_matrix);
   }
   else
   {
      oper_to_generate = gko::as<gko::LinOp>(wrapped_oper);
   }
   auto solver = solver_gen->generate(oper_to_generate);

   // Add the convergence logger object to the combined factory to retrieve the
   // solver and other data
   combined_factory->add_logger(convergence_logger);

   // Finally, apply the solver to x and get the solution in y.
   solver->apply(gko::lend(gko_x), gko::lend(gko_y));

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
   auto x_norm = gko::matrix::Dense<double>::create(executor->get_master(),
                                                    gko::dim<2> {1, 1});
   if (executor != executor->get_master())
   {
      auto gko_x_cpu = clone(executor->get_master(), gko::lend(gko_x));
      gko_x_cpu->compute_norm2(x_norm.get());
   }
   else
   {
      gko_x->compute_norm2(x_norm.get());
   }

   MFEM_VERIFY(x_norm.get()->at(0, 0) != 0.0, " rhs norm is zero");
   // Some residual norm and convergence print outs. As both
   // `residual_norm_d_master` and `y_norm` are seen as Dense matrices, we use
   // the `at` function to get the first value here. In case of multiple right
   // hand sides, this will need to be modified.
   auto fin_res_norm = std::pow(residual_norm_d_master->at(0,0) / x_norm->at(0,0),
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
}

void GinkgoIterativeSolver::SetOperator(const Operator &op)
{

   // Check for SparseMatrix:
   SparseMatrix *op_mat = const_cast<SparseMatrix*>(
                             dynamic_cast<const SparseMatrix*>(&op));
   if (op_mat != NULL)
   {
      // Needs to be a square matrix
      MFEM_VERIFY(op_mat->Height() == op_mat->Width(),
                  "System matrix is not square");

      bool on_device = false;
      if (executor != executor->get_master())
      {
         on_device = true;
      }

      using mtx = gko::matrix::Csr<double, int>;
      const int nnz =  op_mat->GetMemoryData().Capacity();
      system_matrix = mtx::create(
                         executor, gko::dim<2>(op_mat->Height(), op_mat->Width()),
                         gko::Array<double>::view(executor,
                                                  nnz,
                                                  op_mat->ReadWriteData(on_device)),
                         gko::Array<int>::view(executor,
                                               nnz,
                                               op_mat->ReadWriteJ(on_device)),
                         gko::Array<int>::view(executor, op_mat->Height() + 1,
                                               op_mat->ReadWriteI(on_device)));

   }
   else
   {
      wrapped_oper = std::shared_ptr<OperatorWrapper>(new
                                                      OperatorWrapper(executor, op.Height(), &op));
   }
}

/* ---------------------- CGSolver ------------------------ */
CGSolver::CGSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolver(exec_type, print_iter, max_num_iter, RTOLERANCE,
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
   const GinkgoPreconditioner &preconditioner
)
   : GinkgoIterativeSolver(exec_type, print_iter, max_num_iter, RTOLERANCE,
                           ATOLERANCE)
{
   using cg         = gko::solver::Cg<double>;
   // Check for a previously-generated preconditioner (for a specific matrix)
   if (preconditioner.HasGeneratedPreconditioner())
   {
      this->solver_gen = cg::build()
                         .with_criteria(this->combined_factory)
                         .with_generated_preconditioner(
                            preconditioner.GetGeneratedPreconditioner())
                         .on(this->executor);
   }
   else // Pass a preconditioner factory (will use same matrix as the solver)
   {
      this->solver_gen = cg::build()
                         .with_criteria(this->combined_factory)
                         .with_preconditioner(preconditioner.GetFactory())
                         .on(this->executor);
   }
}


/* ---------------------- BICGSTABSolver ------------------------ */
BICGSTABSolver::BICGSTABSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolver(exec_type, print_iter, max_num_iter, RTOLERANCE,
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
   const GinkgoPreconditioner &preconditioner
)
   : GinkgoIterativeSolver(exec_type, print_iter, max_num_iter, RTOLERANCE,
                           ATOLERANCE)
{
   using bicgstab   = gko::solver::Bicgstab<double>;
   this->solver_gen = bicgstab::build()
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner.GetFactory())
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
   : GinkgoIterativeSolver(exec_type, print_iter, max_num_iter, RTOLERANCE,
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
   const GinkgoPreconditioner &preconditioner
)
   : GinkgoIterativeSolver(exec_type, print_iter, max_num_iter, RTOLERANCE,
                           ATOLERANCE)
{
   using cgs        = gko::solver::Cgs<double>;
   this->solver_gen = cgs::build()
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner.GetFactory())
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
   : GinkgoIterativeSolver(exec_type, print_iter, max_num_iter, RTOLERANCE,
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
   const GinkgoPreconditioner &preconditioner
)
   : GinkgoIterativeSolver(exec_type, print_iter, max_num_iter, RTOLERANCE,
                           ATOLERANCE)
{
   using fcg        = gko::solver::Fcg<double>;
   this->solver_gen = fcg::build()
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner.GetFactory())
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
   : GinkgoIterativeSolver(exec_type, print_iter, max_num_iter, RTOLERANCE,
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
   const GinkgoPreconditioner &preconditioner
)
   : GinkgoIterativeSolver(exec_type, print_iter, max_num_iter, RTOLERANCE,
                           ATOLERANCE)
{
   using gmres      = gko::solver::Gmres<double>;
   this->solver_gen = gmres::build()
                      .with_krylov_dim(m)
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner.GetFactory())
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
   : GinkgoIterativeSolver(exec_type, print_iter, max_num_iter, RTOLERANCE,
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
   const GinkgoIterativeSolver &inner_solver
)
   : GinkgoIterativeSolver(exec_type, print_iter, max_num_iter, RTOLERANCE,
                           ATOLERANCE)
{
   using ir         = gko::solver::Ir<double>;
   this->solver_gen = ir::build()
                      .with_criteria(this->combined_factory)
                      .with_solver(inner_solver.GetFactory())
                      .on(this->executor);
}

/* --------------------------------------------------------------- */
/* ---------------------- Preconditioners ------------------------ */
GinkgoPreconditioner::GinkgoPreconditioner(
   const std::string &exec_type)
   : Solver(),
     exec_type(exec_type)
{
   has_generated_precond = false;

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
}

void
GinkgoPreconditioner::Mult(const Vector &x, Vector &y) const
{

   MFEM_VERIFY(generated_precond, "Preconditioner not initialized");
   MFEM_VERIFY(executor, "executor is not initialized");

   using vec       = gko::matrix::Dense<double>;
   if (!iterative_mode)
   {
      y = 0.0;
   }

   // Create x and y vectors in Ginkgo's format. Wrap MFEM's data directly,
   // on CPU or GPU.
   bool on_device = false;
   if (executor != executor->get_master())
   {
      on_device = true;
   }
   auto gko_x = vec::create(executor, gko::dim<2> {x.Size(), 1},
                            gko::Array<double>::view(executor,
                                                     x.Size(), const_cast<double *>(
                                                        x.Read(on_device))), 1);
   auto gko_y = vec::create(executor, gko::dim<2> {y.Size(), 1},
                            gko::Array<double>::view(executor,
                                                     y.Size(), y.ReadWrite(on_device)), 1);
   generated_precond.get()->apply(gko::lend(gko_x), gko::lend(gko_y));
}

void GinkgoPreconditioner::SetOperator(const Operator &op)
{

   // Only accept SparseMatrix for this type.
   SparseMatrix *op_mat = const_cast<SparseMatrix*>(
                             dynamic_cast<const SparseMatrix*>(&op));
   MFEM_VERIFY(op_mat != NULL,
               "GinkgoIterativeSolver::SetOperator : not a SparseMatrix!");

   bool on_device = false;
   if (executor != executor->get_master())
   {
      on_device = true;
   }

   using mtx = gko::matrix::Csr<double, int>;
   const int nnz =  op_mat->GetMemoryData().Capacity();
   auto gko_matrix = mtx::create(
                        executor, gko::dim<2>(op_mat->Height(), op_mat->Width()),
                        gko::Array<double>::view(executor,
                                                 nnz,
                                                 op_mat->ReadWriteData(on_device)),
                        gko::Array<int>::view(executor,
                                              nnz,
                                              op_mat->ReadWriteJ(on_device)),
                        gko::Array<int>::view(executor, op_mat->Height() + 1,
                                              op_mat->ReadWriteI(on_device)));

   // TODO: will this work to "re-set" the operator, or can we only do it once?
   generated_precond = precond_gen->generate(gko::give(gko_matrix));
   has_generated_precond = true;
}


/* ---------------------- JacobiPreconditioner  ------------------------ */
JacobiPreconditioner::JacobiPreconditioner(
   const std::string &exec_type,
   const std::string &storage_opt,
   const double accuracy,
   const int max_block_size
)
   : GinkgoPreconditioner(exec_type)
{

   if (storage_opt == "auto")
   {
      precond_gen = gko::preconditioner::Jacobi<double, int>::build()
                    .with_storage_optimization(
                       gko::precision_reduction::autodetect())
                    .with_accuracy(accuracy)
                    .with_max_block_size(max_block_size)
                    .on(executor);
   }
   else
   {
      precond_gen = gko::preconditioner::Jacobi<double, int>::build()
                    .with_storage_optimization(
                       gko::precision_reduction(0, 0))
                    .with_accuracy(accuracy)
                    .with_max_block_size(max_block_size)
                    .on(executor);
   }

}

/* ---------------------- Ilu/IluIsaiPreconditioner  ------------------------ */
IluPreconditioner::IluPreconditioner(
   const std::string &exec_type,
   const std::string &factorization_type,
   const int sweeps,
   const bool skip_sort
)
   : GinkgoPreconditioner(exec_type)
{
   if (factorization_type == "exact")
   {
      using ilu_fact_type = gko::factorization::Ilu<double, int>;
      std::shared_ptr<ilu_fact_type::Factory> fact_factory = std::move(
                                                                ilu_fact_type::build()
                                                                .with_skip_sorting(skip_sort)
                                                                .on(executor));
      precond_gen = gko::preconditioner::Ilu<>::build()
                    .with_factorization_factory(fact_factory)
                    .on(executor);
   }
   else
   {
      using ilu_fact_type = gko::factorization::ParIlu<double, int>;
      std::shared_ptr<ilu_fact_type::Factory> fact_factory = std::move(
                                                                ilu_fact_type::build()
                                                                .with_iterations(sweeps)
                                                                .with_skip_sorting(skip_sort)
                                                                .on(executor));
      precond_gen = gko::preconditioner::Ilu<>::build()
                    .with_factorization_factory(fact_factory)
                    .on(executor);
   }

}

IluIsaiPreconditioner::IluIsaiPreconditioner(
   const std::string &exec_type,
   const std::string &factorization_type,
   const int sweeps,
   const int sparsity_power,
   const bool skip_sort
)
   : GinkgoPreconditioner(exec_type)
{
   using l_solver_type = gko::preconditioner::LowerIsai<>;
   using u_solver_type = gko::preconditioner::UpperIsai<>;

   std::shared_ptr<l_solver_type::Factory> l_solver_factory = std::move(
                                                                 l_solver_type::build()
                                                                 .with_sparsity_power(sparsity_power)
                                                                 .on(executor));
   std::shared_ptr<u_solver_type::Factory> u_solver_factory = std::move(
                                                                 u_solver_type::build()
                                                                 .with_sparsity_power(sparsity_power)
                                                                 .on(executor));



   if (factorization_type == "exact")
   {
      using ilu_fact_type = gko::factorization::Ilu<double, int>;
      std::shared_ptr<ilu_fact_type::Factory> fact_factory = std::move(
                                                                ilu_fact_type::build()
                                                                .with_skip_sorting(skip_sort)
                                                                .on(executor));
      precond_gen = gko::preconditioner::Ilu<l_solver_type,
      u_solver_type>::build()
      .with_factorization_factory(fact_factory)
      .with_l_solver_factory(l_solver_factory)
      .with_u_solver_factory(u_solver_factory)
      .on(executor);

   }
   else
   {
      using ilu_fact_type = gko::factorization::ParIlu<double, int>;
      std::shared_ptr<ilu_fact_type::Factory> fact_factory = std::move(
                                                                ilu_fact_type::build()
                                                                .with_iterations(sweeps)
                                                                .with_skip_sorting(skip_sort)
                                                                .on(executor));
      precond_gen = gko::preconditioner::Ilu<l_solver_type,
      u_solver_type>::build()
      .with_factorization_factory(fact_factory)
      .with_l_solver_factory(l_solver_factory)
      .with_u_solver_factory(u_solver_factory)
      .on(executor);
   }
}


/* ---------------------- Ic/IcIsaiPreconditioner  ------------------------ */
IcPreconditioner::IcPreconditioner(
   const std::string &exec_type,
   const std::string &factorization_type,
   const int sweeps,
   const bool skip_sort
)
   : GinkgoPreconditioner(exec_type)
{

   if (factorization_type == "exact")
   {
      using ic_fact_type = gko::factorization::Ic<double, int>;
      std::shared_ptr<ic_fact_type::Factory> fact_factory = std::move(
                                                               ic_fact_type::build()
                                                               .with_both_factors(false)
                                                               .with_skip_sorting(skip_sort)
                                                               .on(executor));
      precond_gen = gko::preconditioner::Ic<>::build()
                    .with_factorization_factory(fact_factory)
                    .on(executor);
   }
   else
   {
      using ic_fact_type = gko::factorization::ParIc<double, int>;
      std::shared_ptr<ic_fact_type::Factory> fact_factory = std::move(
                                                               ic_fact_type::build()
                                                               .with_both_factors(false)
                                                               .with_iterations(sweeps)
                                                               .with_skip_sorting(skip_sort)
                                                               .on(executor));
      precond_gen = gko::preconditioner::Ic<>::build()
                    .with_factorization_factory(fact_factory)
                    .on(executor);
   }
}

IcIsaiPreconditioner::IcIsaiPreconditioner(
   const std::string &exec_type,
   const std::string &factorization_type,
   const int sweeps,
   const int sparsity_power,
   const bool skip_sort
)
   : GinkgoPreconditioner(exec_type)
{

   using l_solver_type = gko::preconditioner::LowerIsai<>;
   std::shared_ptr<l_solver_type::Factory> l_solver_factory = std::move(
                                                                 l_solver_type::build()
                                                                 .with_sparsity_power(sparsity_power)
                                                                 .on(executor));
   if (factorization_type == "exact")
   {
      using ic_fact_type = gko::factorization::Ic<double, int>;
      std::shared_ptr<ic_fact_type::Factory> fact_factory = std::move(
                                                               ic_fact_type::build()
                                                               .with_both_factors(false)
                                                               .with_skip_sorting(skip_sort)
                                                               .on(executor));
      precond_gen = gko::preconditioner::Ic<l_solver_type>::build()
                    .with_factorization_factory(fact_factory)
                    .with_l_solver_factory(l_solver_factory)
                    .on(executor);
   }
   else
   {
      using ic_fact_type = gko::factorization::ParIc<double, int>;
      std::shared_ptr<ic_fact_type::Factory> fact_factory = std::move(
                                                               ic_fact_type::build()
                                                               .with_both_factors(false)
                                                               .with_iterations(sweeps)
                                                               .with_skip_sorting(skip_sort)
                                                               .on(executor));
      precond_gen = gko::preconditioner::Ic<l_solver_type>::build()
                    .with_factorization_factory(fact_factory)
                    .with_l_solver_factory(l_solver_factory)
                    .on(executor);
   }
}

} // namespace GinkgoWrappers

} // namespace mfem

#endif // MFEM_USE_GINKGO
