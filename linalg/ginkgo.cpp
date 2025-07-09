// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
#ifndef MFEM_USE_MPI
            if (gko_with_omp_support)
            {
               executor = gko::CudaExecutor::create(current_device,
                                                    gko::OmpExecutor::create());
            }
            else
#endif // with MPI, always use Reference for host Executor
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
#ifndef MFEM_USE_MPI
            if (gko_with_omp_support)
            {
               executor = gko::HipExecutor::create(current_device,
                                                   gko::OmpExecutor::create());
            }
            else
#endif // with MPI, always use Reference for host Executor
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
#ifndef MFEM_USE_MPI
         if (gko_with_omp_support)
         {
            executor = gko::CudaExecutor::create(current_device,
                                                 gko::OmpExecutor::create());
         }
         else
#endif // with MPI, always use Reference for host Executor
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
#ifndef MFEM_USE_MPI
         if (gko_with_omp_support)
         {
            executor = gko::HipExecutor::create(current_device,
                                                gko::OmpExecutor::create());
         }
         else
#endif // with MPI, always use Reference for host Executor
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
#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
     gko_comm(NULL),
#endif
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

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
GinkgoIterativeSolver::GinkgoIterativeSolver(GinkgoExecutor &exec,
                                             MPI_Comm comm,
                                             bool use_implicit_res_norm)
   : Solver(),
     use_implicit_res_norm(use_implicit_res_norm),
     needs_sorted_diagonal_mat(false)
{
   executor = exec.GetExecutor();
   gko_comm = std::shared_ptr<gko::experimental::mpi::communicator>(
                 new gko::experimental::mpi::communicator(comm));
   print_level = -1;

   // Build default stopping criterion factory
   max_iter = 10;
   rel_tol = 0.0;
   abs_tol = 0.0;
   this->update_stop_factory();

   // Distributed solvers always need wrapped vectors
   needs_wrapped_vecs = true;
   sub_op_needs_wrapped_vecs = false;
}
#endif

void GinkgoIterativeSolver::update_stop_factory()
{
   using ResidualCriterionFactory = gko::stop::ResidualNorm<real_t>;
   using ImplicitResidualCriterionFactory =
      gko::stop::ImplicitResidualNorm<real_t>;

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
GinkgoIterativeSolver::initialize_ginkgo_log(gko::matrix::Dense<real_t>* b)
const
{
   // Add the logger object. See the different masks available in Ginkgo's
   // documentation
   convergence_logger =
      std::make_shared<EnableConvergenceLogger<gko::matrix::Dense<real_t>>>
      (executor,
       system_oper.get(),b);
   residual_logger =
      std::make_shared<EnableResidualLogger<gko::matrix::Dense<real_t>>>
      (executor,
       system_oper.get(),b);

}

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
void
GinkgoIterativeSolver::initialize_ginkgo_log(ParallelVectorWrapper* b)
const
{
   // Add the logger object. See the different masks available in Ginkgo's
   // documentation
   convergence_logger =
      std::make_shared<EnableConvergenceLogger<gko::experimental::distributed::Vector<real_t>>>
      (executor,
       system_oper.get(),b);
   residual_logger =
      std::make_shared<EnableResidualLogger<gko::experimental::distributed::Vector<real_t>>>
      (executor,
       system_oper.get(),b);
}
#endif

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

   // Check that alpha and beta are Dense<real_t> of size (1,1):
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
   real_t alpha_f;
   real_t beta_f;

   if (alpha->get_executor() == alpha->get_executor()->get_master())
   {
      // Access value directly
      alpha_f = gko::as<gko::matrix::Dense<real_t>>(alpha)->at(0, 0);
   }
   else
   {
      // Copy from device to host
      this->get_executor()->get_master().get()->copy_from(
         this->get_executor().get(),
         1, gko::as<gko::matrix::Dense<real_t>>(alpha)->get_const_values(),
         &alpha_f);
   }
   if (beta->get_executor() == beta->get_executor()->get_master())
   {
      // Access value directly
      beta_f = gko::as<gko::matrix::Dense<real_t>>(beta)->at(0, 0);
   }
   else
   {
      // Copy from device to host
      this->get_executor()->get_master().get()->copy_from(
         this->get_executor().get(),
         1, gko::as<gko::matrix::Dense<real_t>>(beta)->get_const_values(),
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

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
void ParallelOperatorWrapper::apply_impl(const gko::LinOp *b,
                                         gko::LinOp *x) const
{
   // Cast local vector to VectorWrapper
   const mfem::Ginkgo::VectorWrapper *mfem_b =
      gko::as<const mfem::Ginkgo::VectorWrapper>(
         (gko::as<const ParallelVectorWrapper>(b))->get_local_wrapped_vec_const());
   mfem::Ginkgo::VectorWrapper *mfem_x = gko::as< mfem::Ginkgo::VectorWrapper>(
                                            (gko::as<ParallelVectorWrapper>(x))->get_local_wrapped_vec());
   this->wrapped_oper->Mult(mfem_b->get_mfem_vec_const_ref(),
                            mfem_x->get_mfem_vec_ref());
}

void ParallelOperatorWrapper::apply_impl(const gko::LinOp *alpha,
                                         const gko::LinOp *b,
                                         const gko::LinOp *beta,
                                         gko::LinOp *x) const
{
   // x = alpha * op (b) + beta * x
   // Cast local vector to mfem::Ginkgo::VectorWrapper; only accept this type for this impl
   const mfem::Ginkgo::VectorWrapper *mfem_b =
      gko::as<const mfem::Ginkgo::VectorWrapper>(
         (gko::as<const ParallelVectorWrapper>(b))->get_local_wrapped_vec_const());
   mfem::Ginkgo::VectorWrapper *mfem_x = gko::as< mfem::Ginkgo::VectorWrapper>(
                                            (gko::as<ParallelVectorWrapper>(x))->get_local_wrapped_vec());

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
      mfem::Vector(mfem_x->get_mfem_vec_ref().Size(),
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
#endif

void
GinkgoIterativeSolver::Mult(const Vector &x, Vector &y) const
{

   MFEM_VERIFY(system_oper, "System matrix or operator not initialized");
   MFEM_VERIFY(executor, "executor is not initialized");
   MFEM_VERIFY(y.Size() == x.Size(),
               "Mismatching sizes for rhs and solution");

   using vec       = gko::matrix::Dense<real_t>;
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
   std::unique_ptr<gko::LinOp> gko_x;
   std::unique_ptr<gko::LinOp> gko_y;

   // If we do not have an OperatorWrapper for the system operator or
   // preconditioner, or have an inner solver using VectorWrappers (as
   // for IR), then directly create Ginkgo vectors from MFEM's data.
   if (!needs_wrapped_vecs)
   {
      gko_x = vec::create(executor, gko::dim<2>(x.Size(), 1),
                          gko_array<real_t>::view(executor,
                                                  x.Size(), const_cast<real_t *>(
                                                     x.Read(on_device))), 1);
      gko_y = vec::create(executor, gko::dim<2>(y.Size(), 1),
                          gko_array<real_t>::view(executor,
                                                  y.Size(),
                                                  y.ReadWrite(on_device)), 1);
      initialize_ginkgo_log(gko::as<vec>(gko_x.get()));
   }
   else // We have at least one wrapped MFEM operator or a distributed matrix; need wrapped vectors
   {
#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
      if (gko_comm)
      {
         using par_vec = gko::experimental::distributed::Vector<real_t>;
         auto local_gko_x = new VectorWrapper(executor, x.Size(),
                                              const_cast<Vector *>(&x), false);
         auto local_gko_y = new VectorWrapper(executor, y.Size(), &y,
                                              false);
         gko_x = std::unique_ptr<ParallelVectorWrapper>(
                    new ParallelVectorWrapper(executor, *(gko_comm.get()),
                                              local_gko_x,
                                              system_oper->get_size()[0], 1));
         gko_y = std::unique_ptr<ParallelVectorWrapper>(
                    new ParallelVectorWrapper(executor, *(gko_comm.get()),
                                              local_gko_y,
                                              system_oper->get_size()[0], 1));
         // Create the logger object to log some data from the solvers to confirm
         // convergence.
         initialize_ginkgo_log(gko::as<ParallelVectorWrapper>(gko_x.get()));
      }
      else
#endif
      {
         gko_x = std::unique_ptr<vec>(
                    new VectorWrapper(executor, x.Size(),
                                      const_cast<Vector *>(&x), false));
         gko_y = std::unique_ptr<vec>(
                    new VectorWrapper(executor, y.Size(), &y,
                                      false));
         initialize_ginkgo_log(gko::as<vec>(gko_x.get()));
      }
   }

   solver->clear_loggers(); // Clear any loggers from previous Mult() calls
   MFEM_VERIFY(convergence_logger, "convergence logger not initialized" );
   solver->add_logger(convergence_logger);

   if (print_level==1)
   {
      MFEM_VERIFY(residual_logger, "residual logger not initialized" );
      solver->add_logger(residual_logger);
   }

   // Finally, apply the solver to x and get the solution in y.
   solver->apply(gko_x, gko_y);

   // Get the final stats for the solver.
   real_t final_res_norm = 0.0;
   converged = 0;
   final_iter = convergence_logger->get_num_iterations();
   final_res_norm = convergence_logger->get_final_residual_norm();
   if (convergence_logger->has_converged())
   {
      converged = 1;
   }

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
   if ((gko_comm && (gko_comm->rank() == 0)) || !gko_comm)
   {
#endif
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
#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
   }
#endif
}

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
std::unique_ptr<gko::experimental::distributed::Matrix<real_t, HYPRE_Int, HYPRE_BigInt>>
      GinkgoWrapHypreParMatrix(HypreParMatrix *par_mat,
                               std::shared_ptr<gko::Executor> executor,
                               std::shared_ptr<gko::experimental::mpi::communicator> gko_comm, bool sort_diag)
{
   // Needs to be a square matrix
   MFEM_VERIFY(par_mat->Height() == par_mat->Width(),
               "System matrix is not square");

   MemoryClass mc = MemoryClass::HOST;
   if (executor != executor->get_master())
   {
      mc = MemoryClass::DEVICE;
   }

   using local_mtx = gko::matrix::Csr<real_t, HYPRE_Int>;
   using global_mtx =
      gko::experimental::distributed::Matrix<real_t, HYPRE_Int, HYPRE_BigInt>;
   using gko_partition =
      gko::experimental::distributed::Partition<HYPRE_Int, HYPRE_BigInt>;
   using idx_map =
      gko::experimental::distributed::index_map<HYPRE_Int, HYPRE_BigInt>;
   // Create Ginkgo column partition from Hypre col_starts information
   // Collect the col_starts on every rank, via host memory
   auto mpi_exec = executor->get_master();
   gko::array<HYPRE_BigInt> col_ranges(mpi_exec, gko_comm->size() + 1);
   col_ranges.fill(gko::zero<HYPRE_BigInt>());
   // Gather the "ends" of the ranges such that col_ranges[i] contains the starting index for the
   // ith part of the partition
   gko_comm->all_gather(mpi_exec, par_mat->GetColStarts() + 1, 1,
                        col_ranges.get_data() + 1, 1);
   // Move to device, if necessary
   col_ranges.set_executor(executor);
   auto col_part = gko::share(gko_partition::build_from_contiguous(executor,
                                                                   col_ranges, {}));

   // Create Ginkgo off-process index mapping from Hypre's col_map_offd
   HYPRE_Int num_offd_cols;
   HYPRE_BigInt *cmap;
   par_mat->GetOffdColMap(cmap, num_offd_cols);
   gko_array<HYPRE_BigInt> recv_indices = gko_array<HYPRE_BigInt>::view(mpi_exec,
                                                                        num_offd_cols, cmap);
   // Move to device, if necessary
   recv_indices.set_executor(executor);
   idx_map imap(executor, col_part, gko_comm->rank(), recv_indices);

   // Create local diag and off-diag matrices that share memory with the HypreParMat
   HYPRE_Int local_diag_nnz = par_mat->GetDiagMemoryData().Capacity();
   std::shared_ptr<local_mtx> diag_mat = local_mtx::create(
                                            executor, gko::dim<2>(par_mat->GetNumRows(), par_mat->GetNumCols()),
                                            gko_array<real_t>::view(executor, local_diag_nnz,
                                                                    par_mat->GetDiagMemoryData().ReadWrite(mc, local_diag_nnz)),
                                            gko_array<HYPRE_Int>::view(executor, local_diag_nnz,
                                                                       par_mat->GetDiagMemoryJ().ReadWrite(mc, local_diag_nnz)),
                                            gko_array<HYPRE_Int>::view(executor, par_mat->GetNumRows() + 1,
                                                                       par_mat->GetDiagMemoryI().ReadWrite(mc, par_mat->GetNumRows() + 1))
                                         );
   if (sort_diag == true)
   {
      diag_mat->sort_by_column_index();
   }
   HYPRE_Int local_offd_nnz = par_mat->GetOffdMemoryData().Capacity();
   std::shared_ptr<local_mtx> off_diag_mat = local_mtx::create(
                                                executor, gko::dim<2>(par_mat->GetNumRows(), num_offd_cols),
                                                gko_array<real_t>::view(executor, local_offd_nnz,
                                                                        par_mat->GetOffdMemoryData().ReadWrite(mc, local_offd_nnz)),
                                                gko_array<HYPRE_Int>::view(executor, local_offd_nnz,
                                                                           par_mat->GetOffdMemoryJ().ReadWrite(mc, local_offd_nnz)),
                                                gko_array<HYPRE_Int>::view(executor, par_mat->GetNumRows() + 1,
                                                                           par_mat->GetOffdMemoryI().ReadWrite(mc, par_mat->GetNumRows() + 1))
                                             );
   // Finally, create Ginkgo distributed matrix point to Hypre data
   return global_mtx::create(executor, *(gko_comm.get()), imap, diag_mat,
                             off_diag_mat);
}
#endif

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

   // Needs to be square
   MFEM_VERIFY(op.Height() == op.Width(),
               "System operator is not square");

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
   // Check for HypreParMatrix:
   HypreParMatrix *par_op_mat = const_cast<HypreParMatrix*>(
                                   dynamic_cast<const HypreParMatrix*>(&op));
   if (par_op_mat != NULL)
   {
      system_oper = GinkgoWrapHypreParMatrix(par_op_mat, executor, gko_comm,
                                             needs_sorted_diagonal_mat);
      // We always need wrapped vectors for the parallel case, even with a matrix,
      // because the MFEM Vector passed to Mult() will be local-only, and Ginkgo needs
      // a distributed vector.
      needs_wrapped_vecs = true;
   }
   else
#endif
   {
      // Check for SparseMatrix:
      SparseMatrix *op_mat = const_cast<SparseMatrix*>(
                                dynamic_cast<const SparseMatrix*>(&op));
      if (op_mat != NULL)
      {

         bool on_device = false;
         if (executor != executor->get_master())
         {
            on_device = true;
         }

         using mtx = gko::matrix::Csr<real_t, int>;
         const int nnz =  op_mat->GetMemoryData().Capacity();
         system_oper = mtx::create(
                          executor, gko::dim<2>(op_mat->Height(), op_mat->Width()),
                          gko_array<real_t>::view(executor,
                                                  nnz,
                                                  op_mat->ReadWriteData(on_device)),
                          gko_array<int>::view(executor,
                                               nnz,
                                               op_mat->ReadWriteJ(on_device)),
                          gko_array<int>::view(executor, op_mat->Height() + 1,
                                               op_mat->ReadWriteI(on_device)));

      }
      else  // We don't have a HypreParMatrix or a SparseMatrix; need an operator wrapper
      {
         needs_wrapped_vecs = true;
#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
         if (gko_comm)
         {
            HYPRE_BigInt global_size = op.Height();
            auto mpi_exec = executor->get_master();
            gko_comm->all_reduce(mpi_exec, &global_size, 1, MPI_SUM);
            system_oper = std::shared_ptr<ParallelOperatorWrapper>(
                             new ParallelOperatorWrapper(executor, *(gko_comm.get()), global_size, &op));
         }
         else
#endif
         {
            system_oper = std::shared_ptr<OperatorWrapper>(
                             new OperatorWrapper(executor, op.Height(), &op));
         }
      }
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
   using cg = gko::solver::Cg<real_t>;
   this->solver_gen =
      cg::build().with_criteria(this->combined_factory).on(this->executor);
}

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
CGSolver::CGSolver(GinkgoExecutor &exec, MPI_Comm comm)
   : EnableGinkgoSolver(exec, comm, true)
{
   using cg = gko::solver::Cg<real_t>;
   this->solver_gen =
      cg::build().with_criteria(this->combined_factory).on(this->executor);
}
#endif

CGSolver::CGSolver(GinkgoExecutor &exec,
                   const GinkgoPreconditioner &preconditioner)
   : EnableGinkgoSolver(exec, true)
{
   using cg         = gko::solver::Cg<real_t>;
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

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
CGSolver::CGSolver(GinkgoExecutor &exec,
                   MPI_Comm comm,
                   const GinkgoPreconditioner &preconditioner)
   : EnableGinkgoSolver(exec, comm, true)
{
   using cg         = gko::solver::Cg<real_t>;
   this->needs_wrapped_vecs = true;
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
      this->needs_sorted_diagonal_mat = this->check_needs_sorted_diagonal_mat();
   }
}
#endif

/* ---------------------- BICGSTABSolver ------------------------ */
BICGSTABSolver::BICGSTABSolver(GinkgoExecutor &exec)
   : EnableGinkgoSolver(exec, true)
{
   using bicgstab   = gko::solver::Bicgstab<real_t>;
   this->solver_gen = bicgstab::build()
                      .with_criteria(this->combined_factory)
                      .on(this->executor);
}

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
BICGSTABSolver::BICGSTABSolver(GinkgoExecutor &exec, MPI_Comm comm)
   : EnableGinkgoSolver(exec, comm, true)
{
   using bicgstab   = gko::solver::Bicgstab<real_t>;
   this->solver_gen = bicgstab::build()
                      .with_criteria(this->combined_factory)
                      .on(this->executor);
}
#endif

BICGSTABSolver::BICGSTABSolver(GinkgoExecutor &exec,
                               const GinkgoPreconditioner &preconditioner)
   : EnableGinkgoSolver(exec, true)
{
   using bicgstab   = gko::solver::Bicgstab<real_t>;
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

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
BICGSTABSolver::BICGSTABSolver(GinkgoExecutor &exec,
                               MPI_Comm comm,
                               const GinkgoPreconditioner &preconditioner)
   : EnableGinkgoSolver(exec, comm, true)
{
   using bicgstab   = gko::solver::Bicgstab<real_t>;
   if (preconditioner.HasGeneratedPreconditioner())
   {
      this->solver_gen = bicgstab::build()
                         .with_criteria(this->combined_factory)
                         .with_generated_preconditioner(
                            preconditioner.GetGeneratedPreconditioner())
                         .on(this->executor);
   }
   else
   {
      this->solver_gen = bicgstab::build()
                         .with_criteria(this->combined_factory)
                         .with_preconditioner(preconditioner.GetFactory())
                         .on(this->executor);
      this->needs_sorted_diagonal_mat = this->check_needs_sorted_diagonal_mat();
   }
}
#endif

/* ---------------------- CGSSolver ------------------------ */
CGSSolver::CGSSolver(GinkgoExecutor &exec)
   : EnableGinkgoSolver(exec, true)
{
   using cgs = gko::solver::Cgs<real_t>;
   this->solver_gen =
      cgs::build().with_criteria(this->combined_factory).on(this->executor);
}

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
CGSSolver::CGSSolver(GinkgoExecutor &exec, MPI_Comm comm)
   : EnableGinkgoSolver(exec, comm, true)
{
   using cgs = gko::solver::Cgs<real_t>;
   this->solver_gen =
      cgs::build().with_criteria(this->combined_factory).on(this->executor);
}
#endif

CGSSolver::CGSSolver(GinkgoExecutor &exec,
                     const GinkgoPreconditioner &preconditioner)
   : EnableGinkgoSolver(exec, true)
{
   using cgs        = gko::solver::Cgs<real_t>;
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

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
CGSSolver::CGSSolver(GinkgoExecutor &exec, MPI_Comm comm,
                     const GinkgoPreconditioner &preconditioner)
   : EnableGinkgoSolver(exec, comm, true)
{
   using cgs        = gko::solver::Cgs<real_t>;
   if (preconditioner.HasGeneratedPreconditioner())
   {
      this->solver_gen = cgs::build()
                         .with_criteria(this->combined_factory)
                         .with_generated_preconditioner(
                            preconditioner.GetGeneratedPreconditioner())
                         .on(this->executor);
   }
   else
   {
      this->solver_gen = cgs::build()
                         .with_criteria(this->combined_factory)
                         .with_preconditioner(preconditioner.GetFactory())
                         .on(this->executor);
      this->needs_sorted_diagonal_mat = this->check_needs_sorted_diagonal_mat();
   }
}
#endif

/* ---------------------- FCGSolver ------------------------ */
FCGSolver::FCGSolver(GinkgoExecutor &exec)
   : EnableGinkgoSolver(exec, true)
{
   using fcg = gko::solver::Fcg<real_t>;
   this->solver_gen =
      fcg::build().with_criteria(this->combined_factory).on(this->executor);
}

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
FCGSolver::FCGSolver(GinkgoExecutor &exec, MPI_Comm comm)
   : EnableGinkgoSolver(exec, comm, true)
{
   using fcg = gko::solver::Fcg<real_t>;
   this->solver_gen =
      fcg::build().with_criteria(this->combined_factory).on(this->executor);
}
#endif

FCGSolver::FCGSolver(GinkgoExecutor &exec,
                     const GinkgoPreconditioner &preconditioner)
   : EnableGinkgoSolver(exec, true)
{
   using fcg        = gko::solver::Fcg<real_t>;
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

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
FCGSolver::FCGSolver(GinkgoExecutor &exec, MPI_Comm comm,
                     const GinkgoPreconditioner &preconditioner)
   : EnableGinkgoSolver(exec, comm, true)
{
   using fcg        = gko::solver::Fcg<real_t>;
   if (preconditioner.HasGeneratedPreconditioner())
   {
      this->solver_gen = fcg::build()
                         .with_criteria(this->combined_factory)
                         .with_generated_preconditioner(
                            preconditioner.GetGeneratedPreconditioner())
                         .on(this->executor);
   }
   else
   {
      this->solver_gen = fcg::build()
                         .with_criteria(this->combined_factory)
                         .with_preconditioner(preconditioner.GetFactory())
                         .on(this->executor);
      this->needs_sorted_diagonal_mat = this->check_needs_sorted_diagonal_mat();
   }
}
#endif

/* ---------------------- GMRESSolver ------------------------ */
GMRESSolver::GMRESSolver(GinkgoExecutor &exec, int dim)
   : EnableGinkgoSolver(exec, false),
     m{dim}
{
   using gmres      = gko::solver::Gmres<real_t>;
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

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
GMRESSolver::GMRESSolver(GinkgoExecutor &exec, MPI_Comm comm, int dim)
   : EnableGinkgoSolver(exec, comm, false),
     m{dim}
{
   using gmres      = gko::solver::Gmres<real_t>;
   this->needs_wrapped_vecs = true;
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
#endif

GMRESSolver::GMRESSolver(GinkgoExecutor &exec,
                         const GinkgoPreconditioner &preconditioner, int dim)
   : EnableGinkgoSolver(exec, false),
     m{dim}
{
   using gmres      = gko::solver::Gmres<real_t>;
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

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
GMRESSolver::GMRESSolver(GinkgoExecutor &exec, MPI_Comm comm,
                         const GinkgoPreconditioner &preconditioner, int dim)
   : EnableGinkgoSolver(exec, comm, false),
     m{dim}
{
   using gmres      = gko::solver::Gmres<real_t>;
   this->needs_wrapped_vecs = true;
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
      }
      else
      {
         this->solver_gen = gmres::build()
                            .with_krylov_dim(static_cast<unsigned long>(m))
                            .with_criteria(this->combined_factory)
                            .with_preconditioner(preconditioner.GetFactory())
                            .on(this->executor);
         this->needs_sorted_diagonal_mat = this->check_needs_sorted_diagonal_mat();
      }
   }
}
#endif

void GMRESSolver::SetKDim(int dim)
{
   m = dim;
   using gmres = gko::solver::Gmres<real_t>;
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
   using gmres      = gko::solver::CbGmres<real_t>;
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
   using gmres      = gko::solver::CbGmres<real_t>;
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
   using gmres = gko::solver::CbGmres<real_t>;
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
   using ir = gko::solver::Ir<real_t>;
   this->solver_gen =
      ir::build().with_criteria(this->combined_factory).on(this->executor);
}

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
IRSolver::IRSolver(GinkgoExecutor &exec, MPI_Comm comm)
   : EnableGinkgoSolver(exec, comm, false)
{
   using ir = gko::solver::Ir<real_t>;
   this->solver_gen =
      ir::build().with_criteria(this->combined_factory).on(this->executor);
}
#endif

IRSolver::IRSolver(GinkgoExecutor &exec,
                   const GinkgoIterativeSolver &inner_solver)
   : EnableGinkgoSolver(exec, false)
{
   using ir         = gko::solver::Ir<real_t>;
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

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
IRSolver::IRSolver(GinkgoExecutor &exec, MPI_Comm comm,
                   const GinkgoIterativeSolver &inner_solver)
   : EnableGinkgoSolver(exec, comm, false)
{
   using ir         = gko::solver::Ir<real_t>;
   this->solver_gen = ir::build()
                      .with_criteria(this->combined_factory)
                      .with_solver(inner_solver.GetFactory())
                      .on(this->executor);
}
#endif

/* --------------------------------------------------------------- */
/* ---------------------- Preconditioners ------------------------ */
GinkgoPreconditioner::GinkgoPreconditioner(
   GinkgoExecutor &exec)
   : Solver(),
#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
     gko_comm(NULL),
#endif
     has_generated_precond(false)
{
   executor = exec.GetExecutor();
}

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
GinkgoPreconditioner::GinkgoPreconditioner(
   GinkgoExecutor &exec,
   MPI_Comm comm)
   : Solver(),
     has_generated_precond(false)
{
   executor = exec.GetExecutor();
   gko_comm = std::shared_ptr<gko::experimental::mpi::communicator>(
                 new gko::experimental::mpi::communicator(comm));
}
#endif

void
GinkgoPreconditioner::Mult(const Vector &x, Vector &y) const
{

   MFEM_VERIFY(generated_precond, "Preconditioner not initialized");
   MFEM_VERIFY(executor, "executor is not initialized");

   using vec       = gko::matrix::Dense<real_t>;
   if (!iterative_mode)
   {
      y = 0.0;
   }

   std::unique_ptr<gko::LinOp> gko_x;
   std::unique_ptr<gko::LinOp> gko_y;

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
   if (gko_comm)
   {
      using par_vec = gko::experimental::distributed::Vector<real_t>;
      auto local_gko_x = new VectorWrapper(executor, x.Size(),
                                           const_cast<Vector *>(&x), false);
      auto local_gko_y = new VectorWrapper(executor, y.Size(), &y,
                                           false);
      gko_x = std::unique_ptr<ParallelVectorWrapper>(
                 new ParallelVectorWrapper(executor, *(gko_comm.get()),
                                           local_gko_x,
                                           generated_precond->get_size()[0], 1));
      gko_y = std::unique_ptr<ParallelVectorWrapper>(
                 new ParallelVectorWrapper(executor, *(gko_comm.get()),
                                           local_gko_y,
                                           generated_precond->get_size()[0], 1));
   }
   else
#endif
   {
      // Create x and y vectors in Ginkgo's format. Wrap MFEM's data directly,
      // on CPU or GPU.
      bool on_device = false;
      if (executor != executor->get_master())
      {
         on_device = true;
      }
      gko_x = vec::create(executor, gko::dim<2>(x.Size(), 1),
                          gko_array<real_t>::view(executor,
                                                  x.Size(), const_cast<real_t *>(
                                                     x.Read(on_device))), 1);
      gko_y = vec::create(executor, gko::dim<2>(y.Size(), 1),
                          gko_array<real_t>::view(executor,
                                                  y.Size(),
                                                  y.ReadWrite(on_device)), 1);
   }
   generated_precond.get()->apply(gko_x, gko_y);
}

void GinkgoPreconditioner::SetOperator(const Operator &op)
{

   if (has_generated_precond)
   {
      generated_precond.reset();
      has_generated_precond = false;
   }

   // Needs to be square
   MFEM_VERIFY(op.Height() == op.Width(),
               "System operator is not square");
#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
   // Check for HypreParMatrix:
   HypreParMatrix *par_op_mat = const_cast<HypreParMatrix*>(
                                   dynamic_cast<const HypreParMatrix*>(&op));
   if (par_op_mat != NULL)
   {
      // Finally, create Ginkgo distributed matrix point to Hypre data
      auto gko_matrix = GinkgoWrapHypreParMatrix(par_op_mat, executor, gko_comm,
                                                 false);
      generated_precond = precond_gen->generate(gko::give(gko_matrix));
      has_generated_precond = true;

      // Set MFEM Solver size values
      height = op.Height();
      width = op.Width();
   }
   else
#endif
   {
      // Only accept SparseMatrix for this type.
      SparseMatrix *op_mat = const_cast<SparseMatrix*>(
                                dynamic_cast<const SparseMatrix*>(&op));
      MFEM_VERIFY(op_mat != NULL,
                  "GinkgoPreconditioner::SetOperator : not a SparseMatrix or HypreParMatrix!");

      bool on_device = false;
      if (executor != executor->get_master())
      {
         on_device = true;
      }

      using mtx = gko::matrix::Csr<real_t, int>;
      const int nnz =  op_mat->GetMemoryData().Capacity();
      auto gko_matrix = mtx::create(
                           executor, gko::dim<2>(op_mat->Height(), op_mat->Width()),
                           gko_array<real_t>::view(executor,
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
}


/* ---------------------- JacobiPreconditioner  ------------------------ */
JacobiPreconditioner::JacobiPreconditioner(
   GinkgoExecutor &exec,
   const std::string &storage_opt,
   const real_t accuracy,
   const int max_block_size
)
   : GinkgoPreconditioner(exec)
{

   if (storage_opt == "auto")
   {
      precond_gen = gko::preconditioner::Jacobi<real_t, int>::build()
                    .with_storage_optimization(
                       gko::precision_reduction::autodetect())
                    .with_accuracy(accuracy)
                    .with_max_block_size(static_cast<unsigned int>(max_block_size))
                    .on(executor);
   }
   else
   {
      precond_gen = gko::preconditioner::Jacobi<real_t, int>::build()
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
      using ilu_fact_type = gko::factorization::Ilu<real_t, int>;
      std::shared_ptr<ilu_fact_type::Factory> fact_factory =
         ilu_fact_type::build()
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ilu<>::build()
                    .with_factorization(fact_factory)
                    .on(executor);
   }
   else
   {
      using ilu_fact_type = gko::factorization::ParIlu<real_t, int>;
      std::shared_ptr<ilu_fact_type::Factory> fact_factory =
         ilu_fact_type::build()
         .with_iterations(static_cast<unsigned long>(sweeps))
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ilu<>::build()
                    .with_factorization(fact_factory)
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
      using ilu_fact_type = gko::factorization::Ilu<real_t, int>;
      std::shared_ptr<ilu_fact_type::Factory> fact_factory =
         ilu_fact_type::build()
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ilu<l_solver_type,
      u_solver_type>::build()
      .with_factorization(fact_factory)
      .with_l_solver(l_solver_factory)
      .with_u_solver(u_solver_factory)
      .on(executor);

   }
   else
   {
      using ilu_fact_type = gko::factorization::ParIlu<real_t, int>;
      std::shared_ptr<ilu_fact_type::Factory> fact_factory =
         ilu_fact_type::build()
         .with_iterations(static_cast<unsigned long>(sweeps))
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ilu<l_solver_type,
      u_solver_type>::build()
      .with_factorization(fact_factory)
      .with_l_solver(l_solver_factory)
      .with_u_solver(u_solver_factory)
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
      using ic_fact_type = gko::factorization::Ic<real_t, int>;
      std::shared_ptr<ic_fact_type::Factory> fact_factory =
         ic_fact_type::build()
         .with_both_factors(false)
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ic<>::build()
                    .with_factorization(fact_factory)
                    .on(executor);
   }
   else
   {
      using ic_fact_type = gko::factorization::ParIc<real_t, int>;
      std::shared_ptr<ic_fact_type::Factory> fact_factory =
         ic_fact_type::build()
         .with_both_factors(false)
         .with_iterations(static_cast<unsigned long>(sweeps))
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ic<>::build()
                    .with_factorization(fact_factory)
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
      using ic_fact_type = gko::factorization::Ic<real_t, int>;
      std::shared_ptr<ic_fact_type::Factory> fact_factory =
         ic_fact_type::build()
         .with_both_factors(false)
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ic<l_solver_type>::build()
                    .with_factorization(fact_factory)
                    .with_l_solver(l_solver_factory)
                    .on(executor);
   }
   else
   {
      using ic_fact_type = gko::factorization::ParIc<real_t, int>;
      std::shared_ptr<ic_fact_type::Factory> fact_factory =
         ic_fact_type::build()
         .with_both_factors(false)
         .with_iterations(static_cast<unsigned long>(sweeps))
         .with_skip_sorting(skip_sort)
         .on(executor);
      precond_gen = gko::preconditioner::Ic<l_solver_type>::build()
                    .with_factorization(fact_factory)
                    .with_l_solver(l_solver_factory)
                    .on(executor);
   }
}

/* ---------------------- SchwarzPreconditioner  ------------------------ */
#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
SchwarzPreconditioner::SchwarzPreconditioner(
   GinkgoExecutor &exec,
   MPI_Comm comm,
   Solver &local_solver,
   const bool l1_smoother
)
   : GinkgoPreconditioner(exec, comm)
{
   using schwarz =
      gko::experimental::distributed::preconditioner::Schwarz<real_t, int, HYPRE_BigInt>;
   GinkgoIterativeSolver *local_gko_solver = dynamic_cast<GinkgoIterativeSolver*>
                                             (&local_solver);
   if (local_gko_solver != NULL)
   {
      precond_gen = schwarz::build()
                    .with_local_solver(local_gko_solver->GetFactory())
                    .with_l1_smoother(l1_smoother)
                    .on(executor);
   }
   else
   {
      MFEMPreconditioner *local_mfem_precond = dynamic_cast<MFEMPreconditioner*>
                                               (&local_solver);
      MFEM_VERIFY(local_mfem_precond == NULL,
                  "Ginkgo::SchwarzPreconditioner cannot use an MFEMPreconditioner "
                  "for the local solver.");

      GinkgoPreconditioner *local_gko_precond = dynamic_cast<GinkgoPreconditioner*>
                                                (&local_solver);
      if (local_gko_precond != NULL)
      {
         if (local_gko_precond->HasGeneratedPreconditioner())
         {
            MFEM_VERIFY(l1_smoother == false,
                        "L1 smoother not available for pre-generated local solvers");
            precond_gen = schwarz::build()
                          .with_generated_local_solver(local_gko_precond->GetGeneratedPreconditioner())
                          .on(executor);
         }
         else
         {
            precond_gen = schwarz::build()
                          .with_local_solver(local_gko_precond->GetFactory())
                          .with_l1_smoother(l1_smoother)
                          .on(executor);
         }
      }
      else
      {
         MFEM_ABORT("Ginkgo::SchwarzPreconditioner must take a GinkgoIterativeSolver or GinkgoPreconditioner object "
                    "for the local solver.");
      }
   }
}

// Custom version of SetOperator that will sort the diagonal matrix if using
// L1 smoothing. This should be fixed in a future version of Ginkgo.
void SchwarzPreconditioner::SetOperator(const Operator &op)
{
   if (has_generated_precond)
   {
      generated_precond.reset();
      has_generated_precond = false;
   }

   // Needs to be square
   MFEM_VERIFY(op.Height() == op.Width(),
               "System operator is not square");

   // Check for HypreParMatrix:
   HypreParMatrix *par_op_mat = const_cast<HypreParMatrix*>(
                                   dynamic_cast<const HypreParMatrix*>(&op));
   MFEM_VERIFY(par_op_mat != NULL,
               "GinkgoPreconditioner::SetOperator : not a HypreParMatrix!");

   // Needs to be a square matrix
   MFEM_VERIFY(par_op_mat->Height() == par_op_mat->Width(),
               "System matrix is not square");

   using schwarz =
      gko::experimental::distributed::preconditioner::Schwarz<real_t, int, HYPRE_BigInt>;
   auto factory_params = gko::as<typename schwarz::Factory>
                         (precond_gen)->get_parameters();
   bool sort_diag = false;
   if (factory_params.l1_smoother == true)
   {
      sort_diag = true;
   }
   auto gko_matrix = GinkgoWrapHypreParMatrix(par_op_mat, executor, gko_comm,
                                              sort_diag);
   generated_precond = precond_gen->generate(gko::give(gko_matrix));
   has_generated_precond = true;

   // Set MFEM Solver size values
   height = op.Height();
   width = op.Width();
}
#endif

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

#if defined(MFEM_USE_MPI) && GINKGO_BUILD_MPI
MFEMPreconditioner::MFEMPreconditioner(
   GinkgoExecutor &exec,
   const Solver &mfem_precond,
   MPI_Comm comm
)
   : GinkgoPreconditioner(exec, comm)
{
   // Get global size (the MFEM preconditioner's Height() will be local)
   auto mpi_exec = executor->get_master();
   HYPRE_BigInt global_size = mfem_precond.Height();
   gko_comm->all_reduce(mpi_exec, &global_size, 1, MPI_SUM);
   generated_precond = std::shared_ptr<ParallelOperatorWrapper>(
                          new ParallelOperatorWrapper(executor, *(this->gko_comm.get()),
                                                      global_size, &mfem_precond));
   has_generated_precond = true;
}
#endif

} // namespace Ginkgo

} // namespace mfem

#endif // MFEM_USE_GINKGO
