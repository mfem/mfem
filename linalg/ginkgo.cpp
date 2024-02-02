// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include <cstring>

namespace mfem
{

namespace Ginkgo
{

// Create a GinkgoExecutor of type exec_type.
GinkgoExecutor::GinkgoExecutor(ExecType exec_type)
{
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
   gko::version_info gko_version = gko::version_info::get();
   bool gko_with_omp_support = (strcmp(gko_version.omp_version.tag,
                                       "not compiled") != 0);
#endif
   switch (exec_type)
   {
      case GinkgoExecutor::REFERENCE:
      {
         executor = gko::ReferenceExecutor::create();
         break;
      }
      case GinkgoExecutor::OMP:
      {
         executor = gko::OmpExecutor::create();
         break;
      }
      case GinkgoExecutor::CUDA:
      {
         if (gko::CudaExecutor::get_num_devices() > 0)
         {
#ifdef MFEM_USE_CUDA
            int current_device = 0;
            MFEM_GPU_CHECK(cudaGetDevice(&current_device));
            if (gko_with_omp_support)
            {
               executor = gko::CudaExecutor::create(current_device,
                                                    gko::OmpExecutor::create());
            }
            else
            {
               executor = gko::CudaExecutor::create(current_device,
                                                    gko::ReferenceExecutor::create());
            }
#endif
         }
         else
         {
            MFEM_ABORT("gko::CudaExecutor::get_num_devices() did not report "
                       "any valid devices.");
         }
         break;
      }
      case GinkgoExecutor::HIP:
      {
         if (gko::HipExecutor::get_num_devices() > 0)
         {
#ifdef MFEM_USE_HIP
            int current_device = 0;
            MFEM_GPU_CHECK(hipGetDevice(&current_device));
            if (gko_with_omp_support)
            {
               executor = gko::HipExecutor::create(current_device,
                                                   gko::OmpExecutor::create());
            }
            else
            {
               executor = gko::HipExecutor::create(current_device,
                                                   gko::ReferenceExecutor::create());
            }
#endif
         }
         else
         {
            MFEM_ABORT("gko::HipExecutor::get_num_devices() did not report "
                       "any valid devices.");
         }
         break;
      }
      default:
         MFEM_ABORT("Invalid ExecType specified");
   }
}

// Create a GinkgoExecutor of type exec_type, with host_exec_type for the
// related CPU Executor (only applicable to GPU backends).
GinkgoExecutor::GinkgoExecutor(ExecType exec_type, ExecType host_exec_type)
{
   switch (exec_type)
   {
      case GinkgoExecutor::REFERENCE:
      {
         MFEM_WARNING("Parameter host_exec_type ignored for CPU GinkgoExecutor.");
         executor = gko::ReferenceExecutor::create();
         break;
      }
      case GinkgoExecutor::OMP:
      {
         MFEM_WARNING("Parameter host_exec_type ignored for CPU GinkgoExecutor.");
         executor = gko::OmpExecutor::create();
         break;
      }
      case GinkgoExecutor::CUDA:
      {
         if (gko::CudaExecutor::get_num_devices() > 0)
         {
#ifdef MFEM_USE_CUDA
            int current_device = 0;
            MFEM_GPU_CHECK(cudaGetDevice(&current_device));
            if (host_exec_type == GinkgoExecutor::OMP)
            {
               executor = gko::CudaExecutor::create(current_device,
                                                    gko::OmpExecutor::create());
            }
            else
            {
               executor = gko::CudaExecutor::create(current_device,
                                                    gko::ReferenceExecutor::create());
            }
#endif
         }
         else
         {
            MFEM_ABORT("gko::CudaExecutor::get_num_devices() did not report "
                       "any valid devices.");
         }
         break;
      }
      case GinkgoExecutor::HIP:
      {
         if (gko::HipExecutor::get_num_devices() > 0)
         {
#ifdef MFEM_USE_HIP
            int current_device = 0;
            MFEM_GPU_CHECK(hipGetDevice(&current_device));
            if (host_exec_type == GinkgoExecutor::OMP)
            {
               executor = gko::HipExecutor::create(current_device,
                                                   gko::OmpExecutor::create());
            }
            else
            {
               executor = gko::HipExecutor::create(current_device,
                                                   gko::ReferenceExecutor::create());
            }
#endif
         }
         else
         {
            MFEM_ABORT("gko::HipExecutor::get_num_devices() did not report "
                       "any valid devices.");
         }
         break;
      }
      default:
         MFEM_ABORT("Invalid ExecType specified");
   }
}

// Create a GinkgoExecutor to match MFEM's device configuration.
GinkgoExecutor::GinkgoExecutor(Device &mfem_device)
{
   gko::version_info gko_version = gko::version_info::get();
   bool gko_with_omp_support = (strcmp(gko_version.omp_version.tag,
                                       "not compiled") != 0);
   if (mfem_device.Allows(Backend::CUDA_MASK))
   {
      if (gko::CudaExecutor::get_num_devices() > 0)
      {
#ifdef MFEM_USE_CUDA
         int current_device = 0;
         MFEM_GPU_CHECK(cudaGetDevice(&current_device));
         if (gko_with_omp_support)
         {
            executor = gko::CudaExecutor::create(current_device,
                                                 gko::OmpExecutor::create());
         }
         else
         {
            executor = gko::CudaExecutor::create(current_device,
                                                 gko::ReferenceExecutor::create());
         }
#endif
      }
      else
      {
         MFEM_ABORT("gko::CudaExecutor::get_num_devices() did not report "
                    "any valid devices.");
      }
   }
   else if (mfem_device.Allows(Backend::HIP_MASK))
   {
      if (gko::HipExecutor::get_num_devices() > 0)
      {
#ifdef MFEM_USE_HIP
         int current_device = 0;
         MFEM_GPU_CHECK(hipGetDevice(&current_device));
         if (gko_with_omp_support)
         {
            executor = gko::HipExecutor::create(current_device,
                                                gko::OmpExecutor::create());
         }
         else
         {
            executor = gko::HipExecutor::create(current_device,
                                                gko::ReferenceExecutor::create());
         }
#endif
      }
      else
      {
         MFEM_ABORT("gko::HipExecutor::get_num_devices() did not report "
                    "any valid devices.");
      }
   }
   else
   {
      if (mfem_device.Allows(Backend::OMP_MASK))
      {
         // Also use OpenMP for Ginkgo, if Ginkgo supports it
         if (gko_with_omp_support)
         {
            executor = gko::OmpExecutor::create();
         }
         else
         {
            executor = gko::ReferenceExecutor::create();
         }
      }
      else
      {
         executor = gko::ReferenceExecutor::create();
      }
   }
}

// Create a GinkgoExecutor to match MFEM's device configuration, with
// a specific host_exec_type for the associated CPU Executor (only
// applicable to GPU backends).
GinkgoExecutor::GinkgoExecutor(Device &mfem_device, ExecType host_exec_type)
{

   if (mfem_device.Allows(Backend::CUDA_MASK))
   {
      if (gko::CudaExecutor::get_num_devices() > 0)
      {
#ifdef MFEM_USE_CUDA
         int current_device = 0;
         MFEM_GPU_CHECK(cudaGetDevice(&current_device));
         if (host_exec_type == GinkgoExecutor::OMP)
         {
            executor = gko::CudaExecutor::create(current_device,
                                                 gko::OmpExecutor::create());
         }
         else
         {
            executor = gko::CudaExecutor::create(current_device,
                                                 gko::ReferenceExecutor::create());
         }
#endif
      }
      else
      {
         MFEM_ABORT("gko::CudaExecutor::get_num_devices() did not report "
                    "any valid devices.");
      }
   }
   else if (mfem_device.Allows(Backend::HIP_MASK))
   {
      if (gko::HipExecutor::get_num_devices() > 0)
      {
#ifdef MFEM_USE_HIP
         int current_device = 0;
         MFEM_GPU_CHECK(hipGetDevice(&current_device));
         if (host_exec_type == GinkgoExecutor::OMP)
         {
            executor = gko::HipExecutor::create(current_device,
                                                gko::OmpExecutor::create());
         }
         else
         {
            executor = gko::HipExecutor::create(current_device,
                                                gko::ReferenceExecutor::create());
         }
#endif
      }
      else
      {
         MFEM_ABORT("gko::HipExecutor::get_num_devices() did not report "
                    "any valid devices.");
      }
   }
   else
   {
      MFEM_WARNING("Parameter host_exec_type ignored for CPU GinkgoExecutor.");
      if (mfem_device.Allows(Backend::OMP_MASK))
      {
         // Also use OpenMP for Ginkgo, if Ginkgo supports it
         gko::version_info gko_version = gko::version_info::get();
         bool gko_with_omp_support = (strcmp(gko_version.omp_version.tag,
                                             "not compiled") != 0);
         if (gko_with_omp_support)
         {
            executor = gko::OmpExecutor::create();
         }
         else
         {
            executor = gko::ReferenceExecutor::create();
         }
      }
      else
      {
         executor = gko::ReferenceExecutor::create();
      }
   }
}

GinkgoIterativeSolver::GinkgoIterativeSolver(GinkgoExecutor &exec,
                                             bool use_implicit_res_norm)
   : Solver(),
     use_implicit_res_norm(use_implicit_res_norm)
{
   executor = exec.GetExecutor();
   print_level = -1;

   // Build default stopping criterion factory
   max_iter = 10;
   rel_tol = 0.0;
   abs_tol = 0.0;
   this->update_stop_factory();

   needs_wrapped_vecs = false;
   sub_op_needs_wrapped_vecs = false;
}

void GinkgoIterativeSolver::update_stop_factory()
{
   using ResidualCriterionFactory = gko::stop::ResidualNorm<>;
   using ImplicitResidualCriterionFactory = gko::stop::ImplicitResidualNorm<>;

   if (use_implicit_res_norm)
   {
      imp_rel_criterion  = ImplicitResidualCriterionFactory::build()
                           .with_reduction_factor(sqrt(rel_tol))
                           .with_baseline(gko::stop::mode::initial_resnorm)
                           .on(executor);
      imp_abs_criterion  = ImplicitResidualCriterionFactory::build()
                           .with_reduction_factor(sqrt(abs_tol))
                           .with_baseline(gko::stop::mode::absolute)
                           .on(executor);
      combined_factory =
         gko::stop::Combined::build()
         .with_criteria(imp_rel_criterion,
                        imp_abs_criterion,
                        gko::stop::Iteration::build()
                        .with_max_iters(static_cast<unsigned long>(max_iter))
                        .on(executor))
         .on(executor);
   }
   else
   {
      rel_criterion  = ResidualCriterionFactory::build()
                       .with_reduction_factor(rel_tol)
                       .with_baseline(gko::stop::mode::initial_resnorm)
                       .on(executor);
      abs_criterion  = ResidualCriterionFactory::build()
                       .with_reduction_factor(abs_tol)
                       .with_baseline(gko::stop::mode::absolute)
                       .on(executor);
      combined_factory =
         gko::stop::Combined::build()
         .with_criteria(rel_criterion,
                        abs_criterion,
                        gko::stop::Iteration::build()
                        .with_max_iters(static_cast<unsigned long>(max_iter))
                        .on(executor))
         .on(executor);
   }
}

void
GinkgoIterativeSolver::initialize_ginkgo_log(gko::matrix::Dense<double>* b)
const
{
   // Add the logger object. See the different masks available in Ginkgo's
   // documentation
#if MFEM_GINKGO_VERSION < 10500
   convergence_logger = gko::log::Convergence<>::create(
                           executor, gko::log::Logger::criterion_check_completed_mask);
#else
   convergence_logger = gko::log::Convergence<>::create(
                           gko::log::Logger::criterion_check_completed_mask);
#endif
   residual_logger = std::make_shared<ResidualLogger<>>(executor,
                                                        system_oper.get(),b);

}

void OperatorWrapper::apply_impl(const gko::LinOp *b, gko::LinOp *x) const
{

   // Cast to VectorWrapper; only accept this type for this impl
   const VectorWrapper *mfem_b = gko::as<const VectorWrapper>(b);
   VectorWrapper *mfem_x = gko::as<VectorWrapper>(x);

   this->wrapped_oper->Mult(mfem_b->get_mfem_vec_const_ref(),
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
      this->get_executor()->get_master().get()->copy_from(
         this->get_executor().get(),
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
      this->get_executor()->get_master().get()->copy_from(
         this->get_executor().get(),
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
   this->wrapped_oper->Mult(mfem_b->get_mfem_vec_const_ref(), mfem_tmp);
   // Scale tmp by alpha and add
   mfem_x->get_mfem_vec_ref().Add(alpha_f, mfem_tmp);

   mfem_tmp.Destroy();
}

void
GinkgoIterativeSolver::Mult(const Vector &x, Vector &y) const
{

   MFEM_VERIFY(system_oper, "System matrix or operator not initialized");
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

   // If we do not have an OperatorWrapper for the system operator or
   // preconditioner, or have an inner solver using VectorWrappers (as
   // for IR), then directly create Ginkgo vectors from MFEM's data.
   if (!needs_wrapped_vecs)
   {
      gko_x = vec::create(executor, gko::dim<2>(x.Size(), 1),
                          gko_array<double>::view(executor,
                                                  x.Size(), const_cast<double *>(
                                                     x.Read(on_device))), 1);
      gko_y = vec::create(executor, gko::dim<2>(y.Size(), 1),
                          gko_array<double>::view(executor,
                                                  y.Size(),
                                                  y.ReadWrite(on_device)), 1);
   }
   else // We have at least one wrapped MFEM operator; need wrapped vectors
   {
      gko_x = std::unique_ptr<vec>(
                 new VectorWrapper(executor, x.Size(),
                                   const_cast<Vector *>(&x), false));
      gko_y = std::unique_ptr<vec>(
                 new VectorWrapper(executor, y.Size(), &y,
                                   false));
   }

   // Create the logger object to log some data from the solvers to confirm
   // convergence.
   initialize_ginkgo_log(gko_x.get());

   MFEM_VERIFY(convergence_logger, "convergence logger not initialized" );
   if (print_level==1)
   {
      MFEM_VERIFY(residual_logger, "residual logger not initialized" );
      solver->clear_loggers(); // Clear any loggers from previous Mult() calls
      solver->add_logger(residual_logger);
   }

   // Add the convergence logger object to the combined factory to retrieve the
   // solver and other data
   combined_factory->clear_loggers();
   combined_factory->add_logger(convergence_logger);

   // Finally, apply the solver to x and get the solution in y.
#if MFEM_GINKGO_VERSION < 10600
   solver->apply(gko::lend(gko_x), gko::lend(gko_y));
#else
   solver->apply(gko_x, gko_y);
#endif

   // Get the number of iterations taken to converge to the solution.
   final_iter = convergence_logger->get_num_iterations();

   // Some residual norm and convergence print outs.
   double final_res_norm = 0.0;

   // The convergence_logger object contains the residual vector after the
   // solver has returned. use this vector to compute the residual norm of the
   // solution. Get the residual norm from the logger. As the convergence logger
   // returns a `linop`, it is necessary to convert it to a Dense matrix.
   // Additionally, if the logger is logging on the gpu, it is necessary to copy
   // the data to the host and hence the `residual_norm_d_master`
   auto residual_norm = convergence_logger->get_residual_norm();
   auto imp_residual_norm = convergence_logger->get_implicit_sq_resnorm();

   if (use_implicit_res_norm)
   {
      auto imp_residual_norm_d =
         gko::as<gko::matrix::Dense<double>>(imp_residual_norm);
      auto imp_residual_norm_d_master =
         gko::matrix::Dense<double>::create(executor->get_master(),
                                            gko::dim<2> {1, 1});
      imp_residual_norm_d_master->copy_from(imp_residual_norm_d);

      final_res_norm = imp_residual_norm_d_master->at(0,0);
   }
   else
   {
      auto residual_norm_d =
         gko::as<gko::matrix::Dense<double>>(residual_norm);
      auto residual_norm_d_master =
         gko::matrix::Dense<double>::create(executor->get_master(),
                                            gko::dim<2> {1, 1});
      residual_norm_d_master->copy_from(residual_norm_d);

      final_res_norm = residual_norm_d_master->at(0,0);
   }

   converged = 0;
   if (convergence_logger->has_converged())
   {
      converged = 1;
   }

   if (print_level == 1)
   {
      residual_logger->write();
   }
   if (converged == 0)
   {
      mfem::err << "No convergence!" << '\n';
   }
   if (print_level >=2 && converged==1 )
   {
      mfem::out << "Converged in " << final_iter <<
                " iterations with final residual norm "
                << final_res_norm << '\n';
   }
}

void GinkgoIterativeSolver::SetOperator(const Operator &op)
{

   if (system_oper)
   {
      // If the solver currently needs VectorWrappers, but not due to a
      // "sub-operator" (preconditioner or inner solver), then it's
      // because the current system_oper needs them.  Reset the property
      // to false in case the new op is a SparseMatrix.
      if (needs_wrapped_vecs == true && sub_op_needs_wrapped_vecs == false)
      {
         needs_wrapped_vecs = false;
      }
      // Reset the pointer
      system_oper.reset();
      // Reset the solver generated for the previous operator
      solver.reset();
   }

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
      system_oper = mtx::create(
                       executor, gko::dim<2>(op_mat->Height(), op_mat->Width()),
                       gko_array<double>::view(executor,
                                               nnz,
                                               op_mat->ReadWriteData(on_device)),
                       gko_array<int>::view(executor,
                                            nnz,
                                            op_mat->ReadWriteJ(on_device)),
                       gko_array<int>::view(executor, op_mat->Height() + 1,
                                            op_mat->ReadWriteI(on_device)));

   }
   else
   {
      needs_wrapped_vecs = true;
      system_oper = std::shared_ptr<OperatorWrapper>(
                       new OperatorWrapper(executor, op.Height(), &op));
   }

   // Set MFEM Solver size values
   height = op.Height();
   width = op.Width();

   // Generate the solver from the solver using the system matrix or operator.
   solver = solver_gen->generate(system_oper);
}

/* ---------------------- CGSolver ------------------------ */
CGSolver::CGSolver(GinkgoExecutor &exec)
   : EnableGinkgoSolver(exec, true)
{
   using cg = gko::solver::Cg<double>;
   this->solver_gen =
      cg::build().with_criteria(this->combined_factory).on(this->executor);
}

CGSolver::CGSolver(GinkgoExecutor &exec,
                   const GinkgoPreconditioner &preconditioner)
   : EnableGinkgoSolver(exec, true)
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
      if (dynamic_cast<const OperatorWrapper*>(preconditioner.
                                               GetGeneratedPreconditioner().get()))
      {
         this->sub_op_needs_wrapped_vecs = true;
         this->needs_wrapped_vecs = true;
      }
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
BICGSTABSolver::BICGSTABSolver(GinkgoExecutor &exec)
   : EnableGinkgoSolver(exec, true)
{
   using bicgstab   = gko::solver::Bicgstab<double>;
   this->solver_gen = bicgstab::build()
                      .with_criteria(this->combined_factory)
                      .on(this->executor);
}

BICGSTABSolver::BICGSTABSolver(GinkgoExecutor &exec,
                               const GinkgoPreconditioner &preconditioner)
   : EnableGinkgoSolver(exec, true)
{
   using bicgstab   = gko::solver::Bicgstab<double>;
   if (preconditioner.HasGeneratedPreconditioner())
   {
      this->solver_gen = bicgstab::build()
                         .with_criteria(this->combined_factory)
                         .with_generated_preconditioner(
                            preconditioner.GetGeneratedPreconditioner())
                         .on(this->executor);
      if (dynamic_cast<const OperatorWrapper*>(preconditioner.
                                               GetGeneratedPreconditioner().get()))
      {
         this->sub_op_needs_wrapped_vecs = true;
         this->needs_wrapped_vecs = true;
      }
   }
   else
   {
      this->solver_gen = bicgstab::build()
                         .with_criteria(this->combined_factory)
                         .with_preconditioner(preconditioner.GetFactory())
                         .on(this->executor);
   }
}


/* ---------------------- CGSSolver ------------------------ */
CGSSolver::CGSSolver(GinkgoExecutor &exec)
   : EnableGinkgoSolver(exec, true)
{
   using cgs = gko::solver::Cgs<double>;
   this->solver_gen =
      cgs::build().with_criteria(this->combined_factory).on(this->executor);
}

CGSSolver::CGSSolver(GinkgoExecutor &exec,
                     const GinkgoPreconditioner &preconditioner)
   : EnableGinkgoSolver(exec, true)
{
   using cgs        = gko::solver::Cgs<double>;
   if (preconditioner.HasGeneratedPreconditioner())
   {
      this->solver_gen = cgs::build()
                         .with_criteria(this->combined_factory)
                         .with_generated_preconditioner(
                            preconditioner.GetGeneratedPreconditioner())
                         .on(this->executor);
      if (dynamic_cast<const OperatorWrapper*>(preconditioner.
                                               GetGeneratedPreconditioner().get()))
      {
         this->sub_op_needs_wrapped_vecs = true;
         this->needs_wrapped_vecs = true;
      }
   }
   else
   {
      this->solver_gen = cgs::build()
                         .with_criteria(this->combined_factory)
                         .with_preconditioner(preconditioner.GetFactory())
                         .on(this->executor);
   }
}


/* ---------------------- FCGSolver ------------------------ */
FCGSolver::FCGSolver(GinkgoExecutor &exec)
   : EnableGinkgoSolver(exec, true)
{
   using fcg = gko::solver::Fcg<double>;
   this->solver_gen =
      fcg::build().with_criteria(this->combined_factory).on(this->executor);
}

FCGSolver::FCGSolver(GinkgoExecutor &exec,
                     const GinkgoPreconditioner &preconditioner)
   : EnableGinkgoSolver(exec, true)
{
   using fcg        = gko::solver::Fcg<double>;
   if (preconditioner.HasGeneratedPreconditioner())
   {
      this->solver_gen = fcg::build()
                         .with_criteria(this->combined_factory)
                         .with_generated_preconditioner(
                            preconditioner.GetGeneratedPreconditioner())
                         .on(this->executor);
      if (dynamic_cast<const OperatorWrapper*>(preconditioner.
                                               GetGeneratedPreconditioner().get()))
      {
         this->sub_op_needs_wrapped_vecs = true;
         this->needs_wrapped_vecs = true;
      }
   }
   else
   {
      this->solver_gen = fcg::build()
                         .with_criteria(this->combined_factory)
                         .with_preconditioner(preconditioner.GetFactory())
                         .on(this->executor);
   }
}


/* ---------------------- GMRESSolver ------------------------ */
GMRESSolver::GMRESSolver(GinkgoExecutor &exec, int dim)
   : EnableGinkgoSolver(exec, false),
     m{dim}
{
   using gmres      = gko::solver::Gmres<double>;
   if (this->m == 0) // Don't set a dimension, but let Ginkgo use its default
   {
      this->solver_gen = gmres::build()
                         .with_criteria(this->combined_factory)
                         .on(this->executor);
   }
   else
   {
      this->solver_gen = gmres::build()
                         .with_krylov_dim(static_cast<unsigned long>(m))
                         .with_criteria(this->combined_factory)
                         .on(this->executor);
   }
}

GMRESSolver::GMRESSolver(GinkgoExecutor &exec,
                         const GinkgoPreconditioner &preconditioner, int dim)
   : EnableGinkgoSolver(exec, false),
     m{dim}
{
   using gmres      = gko::solver::Gmres<double>;
   // Check for a previously-generated preconditioner (for a specific matrix)
   if (this->m == 0) // Don't set a dimension, but let Ginkgo use its default
   {
      if (preconditioner.HasGeneratedPreconditioner())
      {
         this->solver_gen = gmres::build()
                            .with_criteria(this->combined_factory)
                            .with_generated_preconditioner(
                               preconditioner.GetGeneratedPreconditioner())
                            .on(this->executor);
         if (dynamic_cast<const OperatorWrapper*>(preconditioner.
                                                  GetGeneratedPreconditioner().get()))
         {
            this->sub_op_needs_wrapped_vecs = true;
            this->needs_wrapped_vecs = true;
         }
      }
      else
      {
         this->solver_gen = gmres::build()
                            .with_criteria(this->combined_factory)
                            .with_preconditioner(preconditioner.GetFactory())
                            .on(this->executor);
      }
   }
   else
   {
      if (preconditioner.HasGeneratedPreconditioner())
      {
         this->solver_gen = gmres::build()
                            .with_krylov_dim(static_cast<unsigned long>(m))
                            .with_criteria(this->combined_factory)
                            .with_generated_preconditioner(
                               preconditioner.GetGeneratedPreconditioner())
                            .on(this->executor);
         if (dynamic_cast<const OperatorWrapper*>(preconditioner.
                                                  GetGeneratedPreconditioner().get()))
         {
            this->sub_op_needs_wrapped_vecs = true;
            this->needs_wrapped_vecs = true;
         }
      }
      else
      {
         this->solver_gen = gmres::build()
                            .with_krylov_dim(static_cast<unsigned long>(m))
                            .with_criteria(this->combined_factory)
                            .with_preconditioner(preconditioner.GetFactory())
                            .on(this->executor);
      }
   }
}

void GMRESSolver::SetKDim(int dim)
{
   m = dim;
   using gmres = gko::solver::Gmres<double>;
   // Create new solver factory with other parameters the same, but new value for krylov_dim
   auto current_params = gko::as<gmres::Factory>(solver_gen)->get_parameters();
   this->solver_gen = current_params.with_krylov_dim(static_cast<unsigned long>(m))
                      .on(this->executor);
   if (solver)
   {
      gko::as<gmres>(solver)->set_krylov_dim(static_cast<unsigned long>(m));
   }
}

/* ---------------------- CBGMRESSolver ------------------------ */
CBGMRESSolver::CBGMRESSolver(GinkgoExecutor &exec, int dim,
                             storage_precision prec)
   : EnableGinkgoSolver(exec, false),
     m{dim}
{
   using gmres      = gko::solver::CbGmres<double>;
   if (this->m == 0) // Don't set a dimension, but let Ginkgo use its default
   {
      this->solver_gen = gmres::build()
                         .with_criteria(this->combined_factory)
                         .with_storage_precision(prec)
                         .on(this->executor);
   }
   else
   {
      this->solver_gen = gmres::build()
                         .with_krylov_dim(static_cast<unsigned long>(m))
                         .with_criteria(this->combined_factory)
                         .with_storage_precision(prec)
                         .on(this->executor);
   }
}

CBGMRESSolver::CBGMRESSolver(GinkgoExecutor &exec,
                             const GinkgoPreconditioner &preconditioner,
                             int dim, storage_precision prec)
   : EnableGinkgoSolver(exec, false),
     m{dim}
{
   using gmres      = gko::solver::CbGmres<double>;
   // Check for a previously-generated preconditioner (for a specific matrix)
   if (this->m == 0) // Don't set a dimension, but let Ginkgo use its default
   {
      if (preconditioner.HasGeneratedPreconditioner())
      {
         this->solver_gen = gmres::build()
                            .with_criteria(this->combined_factory)
                            .with_storage_precision(prec)
                            .with_generated_preconditioner(
                               preconditioner.GetGeneratedPreconditioner())
                            .on(this->executor);
         if (dynamic_cast<const OperatorWrapper*>(preconditioner.
                                                  GetGeneratedPreconditioner().get()))
         {
            this->sub_op_needs_wrapped_vecs = true;
            this->needs_wrapped_vecs = true;
         }
      }
      else
      {
         this->solver_gen = gmres::build()
                            .with_criteria(this->combined_factory)
                            .with_storage_precision(prec)
                            .with_preconditioner(preconditioner.GetFactory())
                            .on(this->executor);
      }
   }
   else
   {
      if (preconditioner.HasGeneratedPreconditioner())
      {
         this->solver_gen = gmres::build()
                            .with_krylov_dim(static_cast<unsigned long>(m))
                            .with_criteria(this->combined_factory)
                            .with_storage_precision(prec)
                            .with_generated_preconditioner(
                               preconditioner.GetGeneratedPreconditioner())
                            .on(this->executor);
         if (dynamic_cast<const OperatorWrapper*>(preconditioner.
                                                  GetGeneratedPreconditioner().get()))
         {
            this->sub_op_needs_wrapped_vecs = true;
            this->needs_wrapped_vecs = true;
         }
      }
      else
      {
         this->solver_gen = gmres::build()
                            .with_krylov_dim(static_cast<unsigned long>(m))
                            .with_criteria(this->combined_factory)
                            .with_storage_precision(prec)
                            .with_preconditioner(preconditioner.GetFactory())
                            .on(this->executor);
      }
   }
}

void CBGMRESSolver::SetKDim(int dim)
{
   m = dim;
   using gmres = gko::solver::CbGmres<double>;
   // Create new solver factory with other parameters the same, but new value for krylov_dim
   auto current_params = gko::as<gmres::Factory>(solver_gen)->get_parameters();
   this->solver_gen = current_params.with_krylov_dim(static_cast<unsigned long>(m))
                      .on(this->executor);
   if (solver)
   {
      gko::as<gmres>(solver)->set_krylov_dim(static_cast<unsigned long>(m));
   }
}

/* ---------------------- IRSolver ------------------------ */
IRSolver::IRSolver(GinkgoExecutor &exec)
   : EnableGinkgoSolver(exec, false)
{
   using ir = gko::solver::Ir<double>;
   this->solver_gen =
      ir::build().with_criteria(this->combined_factory).on(this->executor);
}

IRSolver::IRSolver(GinkgoExecutor &exec,
                   const GinkgoIterativeSolver &inner_solver)
   : EnableGinkgoSolver(exec, false)
{
   using ir         = gko::solver::Ir<double>;
   this->solver_gen = ir::build()
                      .with_criteria(this->combined_factory)
                      .with_solver(inner_solver.GetFactory())
                      .on(this->executor);
   if (inner_solver.UsesVectorWrappers())
   {
      this->sub_op_needs_wrapped_vecs = true;
      this->needs_wrapped_vecs = true;
   }
}

/* --------------------------------------------------------------- */
/* ---------------------- Preconditioners ------------------------ */
GinkgoPreconditioner::GinkgoPreconditioner(
   GinkgoExecutor &exec)
   : Solver()
{
   has_generated_precond = false;
   executor = exec.GetExecutor();
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
   auto gko_x = vec::create(executor, gko::dim<2>(x.Size(), 1),
                            gko_array<double>::view(executor,
                                                    x.Size(), const_cast<double *>(
                                                       x.Read(on_device))), 1);
   auto gko_y = vec::create(executor, gko::dim<2>(y.Size(), 1),
                            gko_array<double>::view(executor,
                                                    y.Size(),
                                                    y.ReadWrite(on_device)), 1);
#if MFEM_GINKGO_VERSION < 10600
   generated_precond.get()->apply(gko::lend(gko_x), gko::lend(gko_y));
#else
   generated_precond.get()->apply(gko_x, gko_y);
#endif
}

void GinkgoPreconditioner::SetOperator(const Operator &op)
{

   if (has_generated_precond)
   {
      generated_precond.reset();
      has_generated_precond = false;
   }

   // Only accept SparseMatrix for this type.
   SparseMatrix *op_mat = const_cast<SparseMatrix*>(
                             dynamic_cast<const SparseMatrix*>(&op));
   MFEM_VERIFY(op_mat != NULL,
               "GinkgoPreconditioner::SetOperator : not a SparseMatrix!");

   bool on_device = false;
   if (executor != executor->get_master())
   {
      on_device = true;
   }

   using mtx = gko::matrix::Csr<double, int>;
   const int nnz =  op_mat->GetMemoryData().Capacity();
   auto gko_matrix = mtx::create(
                        executor, gko::dim<2>(op_mat->Height(), op_mat->Width()),
                        gko_array<double>::view(executor,
                                                nnz,
                                                op_mat->ReadWriteData(on_device)),
                        gko_array<int>::view(executor,
                                             nnz,
                                             op_mat->ReadWriteJ(on_device)),
                        gko_array<int>::view(executor, op_mat->Height() + 1,
                                             op_mat->ReadWriteI(on_device)));

   generated_precond = precond_gen->generate(gko::give(gko_matrix));
   has_generated_precond = true;

   // Set MFEM Solver size values
   height = op.Height();
   width = op.Width();
}


/* ---------------------- JacobiPreconditioner  ------------------------ */
JacobiPreconditioner::JacobiPreconditioner(
   GinkgoExecutor &exec,
   const std::string &storage_opt,
   const double accuracy,
   const int max_block_size
)
   : GinkgoPreconditioner(exec)
{

   if (storage_opt == "auto")
   {
      precond_gen = gko::preconditioner::Jacobi<double, int>::build()
                    .with_storage_optimization(
                       gko::precision_reduction::autodetect())
                    .with_accuracy(accuracy)
                    .with_max_block_size(static_cast<unsigned int>(max_block_size))
                    .on(executor);
   }
   else
   {
      precond_gen = gko::preconditioner::Jacobi<double, int>::build()
                    .with_storage_optimization(
                       gko::precision_reduction(0, 0))
                    .with_accuracy(accuracy)
                    .with_max_block_size(static_cast<unsigned int>(max_block_size))
                    .on(executor);
   }

}

/* ---------------------- Ilu/IluIsaiPreconditioner  ------------------------ */
IluPreconditioner::IluPreconditioner(
   GinkgoExecutor &exec,
   const std::string &factorization_type,
   const int sweeps,
   const bool skip_sort
)
   : GinkgoPreconditioner(exec)
{
   if (factorization_type == "exact")
   {
      using ilu_fact_type = gko::factorization::Ilu<double, int>;
      std::shared_ptr<ilu_fact_type::Factory> fact_factory =
         ilu_fact_type::build()
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ilu<>::build()
#if MFEM_GINKGO_VERSION < 10700
                    .with_factorization_factory(fact_factory)
#else
                    .with_factorization(fact_factory)
#endif
                    .on(executor);
   }
   else
   {
      using ilu_fact_type = gko::factorization::ParIlu<double, int>;
      std::shared_ptr<ilu_fact_type::Factory> fact_factory =
         ilu_fact_type::build()
         .with_iterations(static_cast<unsigned long>(sweeps))
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ilu<>::build()
#if MFEM_GINKGO_VERSION < 10700
                    .with_factorization_factory(fact_factory)
#else
                    .with_factorization(fact_factory)
#endif
                    .on(executor);
   }

}

IluIsaiPreconditioner::IluIsaiPreconditioner(
   GinkgoExecutor &exec,
   const std::string &factorization_type,
   const int sweeps,
   const int sparsity_power,
   const bool skip_sort
)
   : GinkgoPreconditioner(exec)
{
   using l_solver_type = gko::preconditioner::LowerIsai<>;
   using u_solver_type = gko::preconditioner::UpperIsai<>;

   std::shared_ptr<l_solver_type::Factory> l_solver_factory =
      l_solver_type::build()
      .with_sparsity_power(sparsity_power)
      .on(executor);
   std::shared_ptr<u_solver_type::Factory> u_solver_factory =
      u_solver_type::build()
      .with_sparsity_power(sparsity_power)
      .on(executor);



   if (factorization_type == "exact")
   {
      using ilu_fact_type = gko::factorization::Ilu<double, int>;
      std::shared_ptr<ilu_fact_type::Factory> fact_factory =
         ilu_fact_type::build()
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ilu<l_solver_type,
      u_solver_type>::build()
#if MFEM_GINKGO_VERSION < 10700
      .with_factorization_factory(fact_factory)
      .with_l_solver_factory(l_solver_factory)
      .with_u_solver_factory(u_solver_factory)
#else
      .with_factorization(fact_factory)
      .with_l_solver(l_solver_factory)
      .with_u_solver(u_solver_factory)
#endif
      .on(executor);

   }
   else
   {
      using ilu_fact_type = gko::factorization::ParIlu<double, int>;
      std::shared_ptr<ilu_fact_type::Factory> fact_factory =
         ilu_fact_type::build()
         .with_iterations(static_cast<unsigned long>(sweeps))
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ilu<l_solver_type,
      u_solver_type>::build()
#if MFEM_GINKGO_VERSION < 10700
      .with_factorization_factory(fact_factory)
      .with_l_solver_factory(l_solver_factory)
      .with_u_solver_factory(u_solver_factory)
#else
      .with_factorization(fact_factory)
      .with_l_solver(l_solver_factory)
      .with_u_solver(u_solver_factory)
#endif
      .on(executor);
   }
}


/* ---------------------- Ic/IcIsaiPreconditioner  ------------------------ */
IcPreconditioner::IcPreconditioner(
   GinkgoExecutor &exec,
   const std::string &factorization_type,
   const int sweeps,
   const bool skip_sort
)
   : GinkgoPreconditioner(exec)
{

   if (factorization_type == "exact")
   {
      using ic_fact_type = gko::factorization::Ic<double, int>;
      std::shared_ptr<ic_fact_type::Factory> fact_factory =
         ic_fact_type::build()
         .with_both_factors(false)
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ic<>::build()
#if MFEM_GINKGO_VERSION < 10700
                    .with_factorization_factory(fact_factory)
#else
                    .with_factorization(fact_factory)
#endif
                    .on(executor);
   }
   else
   {
      using ic_fact_type = gko::factorization::ParIc<double, int>;
      std::shared_ptr<ic_fact_type::Factory> fact_factory =
         ic_fact_type::build()
         .with_both_factors(false)
         .with_iterations(static_cast<unsigned long>(sweeps))
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ic<>::build()
#if MFEM_GINKGO_VERSION < 10700
                    .with_factorization_factory(fact_factory)
#else
                    .with_factorization(fact_factory)
#endif
                    .on(executor);
   }
}

IcIsaiPreconditioner::IcIsaiPreconditioner(
   GinkgoExecutor &exec,
   const std::string &factorization_type,
   const int sweeps,
   const int sparsity_power,
   const bool skip_sort
)
   : GinkgoPreconditioner(exec)
{

   using l_solver_type = gko::preconditioner::LowerIsai<>;
   std::shared_ptr<l_solver_type::Factory> l_solver_factory =
      l_solver_type::build()
      .with_sparsity_power(sparsity_power)
      .on(executor);
   if (factorization_type == "exact")
   {
      using ic_fact_type = gko::factorization::Ic<double, int>;
      std::shared_ptr<ic_fact_type::Factory> fact_factory =
         ic_fact_type::build()
         .with_both_factors(false)
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ic<l_solver_type>::build()
#if MFEM_GINKGO_VERSION < 10700
                    .with_factorization_factory(fact_factory)
                    .with_l_solver_factory(l_solver_factory)
#else
                    .with_factorization(fact_factory)
                    .with_l_solver(l_solver_factory)
#endif
                    .on(executor);
   }
   else
   {
      using ic_fact_type = gko::factorization::ParIc<double, int>;
      std::shared_ptr<ic_fact_type::Factory> fact_factory =
         ic_fact_type::build()
         .with_both_factors(false)
         .with_iterations(static_cast<unsigned long>(sweeps))
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ic<l_solver_type>::build()
#if MFEM_GINKGO_VERSION < 10700
                    .with_factorization_factory(fact_factory)
                    .with_l_solver_factory(l_solver_factory)
#else
                    .with_factorization(fact_factory)
                    .with_l_solver(l_solver_factory)
#endif
                    .on(executor);
   }
}

/* ---------------------- MFEMPreconditioner  ------------------------ */
MFEMPreconditioner::MFEMPreconditioner(
   GinkgoExecutor &exec,
   const Solver &mfem_precond
)
   : GinkgoPreconditioner(exec)
{
   generated_precond = std::shared_ptr<OperatorWrapper>(
                          new OperatorWrapper(executor,
                                              mfem_precond.Height(), &mfem_precond));
   has_generated_precond = true;
}

} // namespace Ginkgo

} // namespace mfem

#endif // MFEM_USE_GINKGO
