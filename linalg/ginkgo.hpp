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

#ifndef MFEM_GINKGO
#define MFEM_GINKGO

#include "../config/config.hpp"

#ifdef MFEM_USE_GINKGO

#include "operator.hpp"
#include "sparsemat.hpp"
#include "solvers.hpp"

#include <ginkgo/ginkgo.hpp>

#include <iomanip>
#include <ios>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#define MFEM_GINKGO_VERSION \
   ((GKO_VERSION_MAJOR*100 + GKO_VERSION_MINOR)*100 + GKO_VERSION_PATCH)

namespace mfem
{
namespace Ginkgo
{

/// The alias 'gko_array' refers to 'gko::Array' (Ginkgo < 1.5.0) or
/// 'gko::array' (Ginkgo >= 1.5.0).
#if MFEM_GINKGO_VERSION < 10500
template <typename T> using gko_array = gko::Array<T>;
#else
template <typename T> using gko_array = gko::array<T>;
#endif

/**
* Helper class for a case where a wrapped MFEM Vector
* should be owned by Ginkgo, and deleted when the wrapper
* object goes out of scope.
*/
template <typename T>
class gko_mfem_destroy
{
public:
   using pointer = T *;

   // Destroys an MFEM object.  Requires object to have a Destroy() method.
   void operator()(pointer ptr) const noexcept { ptr->Destroy(); }
};

/**
* This class wraps an MFEM vector object for Ginkgo's use. It
* allows Ginkgo and MFEM to operate directly on the same
* data, and is necessary to use MFEM Operators with Ginkgo
* solvers.
*
* @ingroup Ginkgo
*/

class VectorWrapper : public gko::matrix::Dense<real_t>
{
public:
   VectorWrapper(std::shared_ptr<const gko::Executor> exec,
                 gko::size_type size, Vector *mfem_vec,
                 bool ownership = false)
      : gko::matrix::Dense<real_t>(
           exec,
           gko::dim<2> {size, 1},
   gko_array<real_t>::view(exec,
                           size,
                           mfem_vec->ReadWrite(
                              exec != exec->get_master() ? true : false)),
   1)
   {
      // This controls whether or not we want Ginkgo to own its MFEM Vector.
      // Normally, when we are wrapping an MFEM Vector created outside
      // Ginkgo, we do not want ownership to be true. However, Ginkgo
      // creates its own temporary vectors as part of its solvers, and
      // these will be owned (and deleted) by Ginkgo.
      if (ownership)
      {
         using deleter = gko_mfem_destroy<Vector>;
         wrapped_vec = std::unique_ptr<Vector,
         std::function<void(Vector *)>>(
            mfem_vec, deleter{});
      }
      else
      {
         using deleter = gko::null_deleter<Vector>;
         wrapped_vec = std::unique_ptr<Vector,
         std::function<void(Vector *)>>(
            mfem_vec, deleter{});
      }
   }

   static std::unique_ptr<VectorWrapper> create(
      std::shared_ptr<const gko::Executor> exec,
      gko::size_type size,
      Vector *mfem_vec,
      bool ownership = false)
   {
      return std::unique_ptr<VectorWrapper>(
                new VectorWrapper(exec, size, mfem_vec, ownership));
   }

   // Return reference to MFEM Vector object
   Vector &get_mfem_vec_ref() { return *(this->wrapped_vec.get()); }

   // Return const reference to MFEM Vector object
   const Vector &get_mfem_vec_const_ref() const { return *(this->wrapped_vec.get()); }

   // Override base Dense class implementation for creating new vectors
   // with same executor and size as self
   std::unique_ptr<gko::matrix::Dense<real_t>>
                                            create_with_same_config() const override
   {
      Vector *mfem_vec = new Vector(
         this->get_size()[0],
         this->wrapped_vec.get()->GetMemory().GetMemoryType());

      mfem_vec->UseDevice(this->wrapped_vec.get()->UseDevice());

      // If this function is called, Ginkgo is creating this
      // object and should control the memory, so ownership is
      // set to true
      return VectorWrapper::create(this->get_executor(),
                                   this->get_size()[0],
                                   mfem_vec,
                                   true);
   }

   // Override base Dense class implementation for creating new vectors
   // with same executor and type as self, but with a different size.
   // This function will create "one large VectorWrapper" of size
   // size[0] * size[1], since MFEM Vectors only have one dimension.
   std::unique_ptr<gko::matrix::Dense<real_t>> create_with_type_of_impl(
                                               std::shared_ptr<const gko::Executor> exec,
                                               const gko::dim<2> &size,
                                               gko::size_type stride) const override
   {
      // Only stride of 1 is allowed for VectorWrapper type
      if (stride > 1)
      {
         throw gko::Error(
            __FILE__, __LINE__,
            "VectorWrapper cannot be created with stride > 1");
      }
      // Compute total size of new Vector
      gko::size_type total_size = size[0]*size[1];
      Vector *mfem_vec = new Vector(
         total_size,
         this->wrapped_vec.get()->GetMemory().GetMemoryType());

      mfem_vec->UseDevice(this->wrapped_vec.get()->UseDevice());

      // If this function is called, Ginkgo is creating this
      // object and should control the memory, so ownership is
      // set to true
      return VectorWrapper::create(
                this->get_executor(), total_size, mfem_vec,
                true);
   }

   // Override base Dense class implementation for creating new sub-vectors
   // from a larger vector.
   std::unique_ptr<gko::matrix::Dense<real_t>> create_submatrix_impl(
                                               const gko::span &rows,
                                               const gko::span &columns,
                                               const gko::size_type stride) override
   {

      gko::size_type num_rows = rows.end - rows.begin;
      gko::size_type num_cols = columns.end - columns.begin;
      // Data in the Dense matrix will be stored in row-major format.
      // Check that we only have one column, and that the stride = 1
      // (only allowed value for VectorWrappers).
      if (num_cols > 1 || stride > 1)
      {
         throw gko::BadDimension(
            __FILE__, __LINE__, __func__, "new_submatrix", num_rows,
            num_cols,
            "VectorWrapper submatrix must have one column and stride = 1");
      }
      int data_size = static_cast<int>(num_rows * num_cols);
      int start = static_cast<int>(rows.begin);
      // Create a new MFEM Vector pointing to this starting point in the data
      Vector *mfem_vec = new Vector();
      mfem_vec->MakeRef(*(this->wrapped_vec.get()), start, data_size);
      mfem_vec->UseDevice(this->wrapped_vec.get()->UseDevice());

      // If this function is called, Ginkgo is creating this
      // object and should control the memory, so ownership is
      // set to true (but MFEM doesn't own and won't delete
      // the data, at it's only a reference to the parent Vector)
      return VectorWrapper::create(
                this->get_executor(), data_size, mfem_vec,
                true);
   }

private:
   std::unique_ptr<Vector, std::function<void(Vector *)>> wrapped_vec;
};

/**
* This class wraps an MFEM Operator for Ginkgo, to make its Mult()
* function available to Ginkgo, provided the input and output vectors
* are of the VectorWrapper type.
* Note that this class does NOT take ownership of the MFEM Operator.
*
* @ingroup Ginkgo
*/
class OperatorWrapper
   : public gko::EnableLinOp<OperatorWrapper>,
     public gko::EnableCreateMethod<OperatorWrapper>
{
public:
   OperatorWrapper(std::shared_ptr<const gko::Executor> exec,
                   gko::size_type size = 0,
                   const Operator *oper = NULL)
      : gko::EnableLinOp<OperatorWrapper>(exec, gko::dim<2> {size, size}),
   gko::EnableCreateMethod<OperatorWrapper>()
   {
      this->wrapped_oper = oper;
   }

protected:
   void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override;
   void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                   const gko::LinOp *beta, gko::LinOp *x) const override;

private:
   const Operator *wrapped_oper;
};

// Utility function which gets the scalar value of a Ginkgo gko::matrix::Dense
// matrix representing the norm of a vector.
template <typename ValueType=real_t>
real_t get_norm(const gko::matrix::Dense<ValueType> *norm)
{
   // Put the value on CPU thanks to the master executor
   auto cpu_norm = clone(norm->get_executor()->get_master(), norm);
   // Return the scalar value contained at position (0, 0)
   return cpu_norm->at(0, 0);
}

// Utility function which computes the norm of a Ginkgo gko::matrix::Dense
// vector.
template <typename ValueType=real_t>
real_t compute_norm(const gko::matrix::Dense<ValueType> *b)
{
   // Get the executor of the vector
   auto exec = b->get_executor();
   // Initialize a result scalar containing the value 0.0.
   auto b_norm = gko::initialize<gko::matrix::Dense<ValueType>>({0.0}, exec);
   // Use the dense `compute_norm2` function to compute the norm.
#if MFEM_GINKGO_VERSION < 10600
   b->compute_norm2(gko::lend(b_norm));
#else
   b->compute_norm2(b_norm);
#endif
   // Use the other utility function to return the norm contained in `b_norm``
   return std::pow(get_norm(b_norm.get()),2);
}

/**
 * Custom logger class which intercepts the residual norm scalar and solution
 * vector in order to print a table of real vs recurrent (internal to the
 * solvers) residual norms.
 *
 * This has been taken from the custom-logger example of Ginkgo. See the
 * custom-logger example to understand how to write and modify your own loggers
 * with Ginkgo.
 *
 * @ingroup Ginkgo
 */
template <typename ValueType=real_t>
struct ResidualLogger : gko::log::Logger
{
   // Output the logger's data in a table format
   void write() const
   {
      // Print a header for the table
      if (compute_real_residual)
      {
         mfem::out << "Iteration log with real residual norms:" << std::endl;
      }
      else
      {
         mfem::out << "Iteration log with residual norms:" << std::endl;
      }
      mfem::out << '|' << std::setw(10) << "Iteration" << '|' << std::setw(25)
                << "Residual Norm" << '|' << std::endl;
      // Print a separation line. Note that for creating `10` characters
      // `std::setw()` should be set to `11`.
      mfem::out << '|' << std::setfill('-') << std::setw(11) << '|' <<
                std::setw(26) << '|' << std::setfill(' ') << std::endl;
      // Print the data one by one in the form
      mfem::out << std::scientific;
      for (std::size_t i = 0; i < iterations.size(); i++)
      {
         mfem::out << '|' << std::setw(10) << iterations[i] << '|'
                   << std::setw(25) << residual_norms[i] << '|' << std::endl;
      }
      // std::defaultfloat could be used here but some compilers do not support
      // it properly, e.g. the Intel compiler
      mfem::out.unsetf(std::ios_base::floatfield);
      // Print a separation line
      mfem::out << '|' << std::setfill('-') << std::setw(11) << '|' <<
                std::setw(26) << '|' << std::setfill(' ') << std::endl;
   }

   using gko_dense = gko::matrix::Dense<ValueType>;

   // Ginkgo 1.5 and older: version for solver that doesn't log implicit res norm
   void on_iteration_complete(const gko::LinOp *op,
                              const gko::size_type &iteration,
                              const gko::LinOp *residual,
                              const gko::LinOp *solution,
                              const gko::LinOp *residual_norm) const override
   {
      iteration_complete_core(iteration, residual, solution, residual_norm,
                              nullptr);
   }
   // Ginkgo 1.5 and older: version with implicit residual norm
   void on_iteration_complete(const gko::LinOp *op,
                              const gko::size_type &iteration,
                              const gko::LinOp *residual,
                              const gko::LinOp *solution,
                              const gko::LinOp *residual_norm,
                              const gko::LinOp *implicit_sq_residual_norm) const override
   {
      iteration_complete_core(iteration, residual, solution, residual_norm,
                              implicit_sq_residual_norm);
   }
#if MFEM_GINKGO_VERSION > 10500
   // Ginkgo 1.6 and newer
   void on_iteration_complete(const gko::LinOp *op,
                              const gko::LinOp *rhs,
                              const gko::LinOp *solution,
                              const gko::size_type &iteration,
                              const gko::LinOp *residual,
                              const gko::LinOp *residual_norm,
                              const gko::LinOp *implicit_sq_residual_norm,
                              const gko::array<gko::stopping_status>* status,
                              bool stopped) const override
   {
      iteration_complete_core(iteration, residual, solution, residual_norm,
                              implicit_sq_residual_norm);
   }
#endif

   // Construct the logger and store the system matrix and b vectors
   ResidualLogger(std::shared_ptr<const gko::Executor> exec,
                  const gko::LinOp *matrix, const gko_dense *b,
                  bool compute_real_residual=false)
      :
#if MFEM_GINKGO_VERSION < 10500
      gko::log::Logger(exec,
                       gko::log::Logger::iteration_complete_mask),
#else
      gko::log::Logger(gko::log::Logger::iteration_complete_mask),
#endif
      matrix {matrix},
      b{b},
      compute_real_residual{compute_real_residual}
   {
      if (compute_real_residual == true)
      {
         if (dynamic_cast<const VectorWrapper*>(b))
         {
            const VectorWrapper *b_cast = gko::as<const VectorWrapper>(b);
            res = std::move(gko_dense::create_with_config_of(b_cast).release());
         }
         else
         {
            res = std::move(gko::clone(b).release());
         }
      }
   }

private:
   // Customize the logging hook which is called every time an iteration is
   // completed.
   void iteration_complete_core(const gko::size_type &iteration,
                                const gko::LinOp *residual,
                                const gko::LinOp *solution,
                                const gko::LinOp *residual_norm,
                                const gko::LinOp *implicit_sq_residual_norm) const
   {
      // If the solver shares the current solution vector and we want to
      // compute the residual from that
      if (solution && compute_real_residual)
      {
         // Store the matrix's executor
         auto exec = matrix->get_executor();
         // Compute the real residual vector by calling apply on the system
         // First, compute res = A * x
#if MFEM_GINKGO_VERSION < 10600
         matrix->apply(gko::lend(solution), gko::lend(res));
#else
         matrix->apply(solution, res);
#endif
         // Now do res = res - b, depending on which vector/oper type
         // Check if b is a Ginkgo vector or wrapped MFEM Vector
         if (dynamic_cast<const VectorWrapper*>(b))
         {
            const VectorWrapper *b_cast = gko::as<const VectorWrapper>(b);
            // Copy the MFEM Vector stored in b
            VectorWrapper *res_cast = gko::as<VectorWrapper>(res);
            res_cast->get_mfem_vec_ref() -= b_cast->get_mfem_vec_const_ref();
         }
         else
         {
            // Create a scalar containing the value -1.0
            auto neg_one = gko::initialize<gko_dense>({-1.0}, exec);
#if MFEM_GINKGO_VERSION < 10600
            res->add_scaled(gko::lend(neg_one), gko::lend(b));
#else
            res->add_scaled(neg_one, b);
#endif
         }

         // Compute the norm of the residual vector and add it to the
         // `residual_norms` vector
#if MFEM_GINKGO_VERSION < 10600
         residual_norms.push_back(compute_norm(gko::lend(res)));
#else
         residual_norms.push_back(compute_norm(res));
#endif
      }
      else
      {
         // If the solver shares an implicit or recurrent residual norm, log its value
         if (implicit_sq_residual_norm)
         {
            auto dense_norm = gko::as<gko_dense>(implicit_sq_residual_norm);
            // Add the norm to the `residual_norms` vector
            residual_norms.push_back(get_norm(dense_norm));
            // Otherwise, use the recurrent residual vector
         }
         else if (residual_norm)
         {
            auto dense_norm = gko::as<gko_dense>(residual_norm);
            // Add the norm to the `residual_norms` vector
            residual_norms.push_back(get_norm(dense_norm));
            // Otherwise, use the recurrent residual vector
         }
         else
         {
            auto dense_residual = gko::as<gko_dense>(residual);
            // Compute the residual vector's norm
#if MFEM_GINKGO_VERSION < 10600
            auto norm = compute_norm(gko::lend(dense_residual));
#else
            auto norm = compute_norm(dense_residual);
#endif
            // Add the computed norm to the `residual_norms` vector
            residual_norms.push_back(norm);
         }
      }
      // Add the current iteration number to the `iterations` vector
      iterations.push_back(iteration);
   }

   // Pointer to the system matrix
   const gko::LinOp *matrix;
   // Pointer to the right hand sides
   const gko_dense *b;
   // Pointer to the residual workspace vector
   gko_dense *res;
   // Vector which stores all the residual norms
   mutable std::vector<ValueType> residual_norms{};
   // Vector which stores all the iteration numbers
   mutable std::vector<std::size_t> iterations{};
   // Whether or not to compute the residual at every iteration,
   //  rather than using the recurrent norm
   const bool compute_real_residual;
};

/**
* This class wraps a Ginkgo Executor for use in MFEM.
* Note that objects in the Ginkgo namespace intended to work
* together, e.g. a Ginkgo solver and preconditioner, should use the same
* GinkgoExecutor object.  In general, most users will want to create
* one GinkgoExecutor object for use with all Ginkgo-related objects.
* The wrapper can be created to match MFEM's device configuration.
*/
class GinkgoExecutor
{
public:
   // Types of Ginkgo Executors.
   enum ExecType
   {
      /// Reference CPU Executor.
      REFERENCE = 0,
      /// OpenMP CPU Executor.
      OMP = 1,
      /// CUDA GPU Executor.
      CUDA = 2,
      /// HIP GPU Executor.
      HIP = 3
   };
   /**
    * Constructor.
    * Takes an @p GinkgoExecType argument and creates an Executor.
    * In Ginkgo, GPU Executors must have an associated host Executor.
    * This routine will select a CPU Executor based on the OpenMP support
    * for Ginkgo.
    */
   GinkgoExecutor(ExecType exec_type);

   /**
    * Constructor.
    * Takes an @p GinkgoExecType argument and creates an Executor.
    * In Ginkgo, GPU Executors must have an associated host Executor.
    * This routine allows for explicite setting of the CPU Executor
    * for GPU backends.
    */
   GinkgoExecutor(ExecType exec_type, ExecType host_exec_type);

   /**
    * Constructor.
    * Takes an MFEM @p Device object and creates an Executor
    * that "matches" (e.g., if MFEM is using the CPU, Ginkgo
    * will choose the Reference or OmpExecutor based on MFEM's
    * configuration and Ginkgo's capabilities; if MFEM is using
    * CUDA, Ginkgo will choose the CudaExecutor with a default
    * CPU Executor based on Ginkgo's OpenMP support).
    */
   GinkgoExecutor(Device &mfem_device);

   /**
    * Constructor.
    * Takes an MFEM @p Device object and creates an Executor
    * that "matches", but allows the user to specify the host
    * Executor for GPU backends.
    */
   GinkgoExecutor(Device &mfem_device, ExecType host_exec_type);

   /**
    * Destructor.
    */
   virtual ~GinkgoExecutor() = default;

   std::shared_ptr<gko::Executor> GetExecutor() const
   {
      return this->executor;
   };

private:
   std::shared_ptr<gko::Executor> executor;

};

/**
* This class forms the base class for all of Ginkgo's preconditioners.  The
* various derived classes only take the additional data that is specific to them.
* The entire collection of preconditioners that Ginkgo implements is available
* at the Ginkgo documentation and manual pages,
* https://ginkgo-project.github.io/ginkgo/doc/develop.
*
* @ingroup Ginkgo
*/
class GinkgoPreconditioner : public Solver
{
public:
   /**
    * Constructor.
    *
    * The @p exec defines the paradigm where the solution is computed.
    * Ginkgo currently supports four different executor types:
    *
    * +    OmpExecutor specifies that the data should be stored and the
    *      associated operations executed on an OpenMP-supporting device (e.g.
    *      host CPU);
    * +    CudaExecutor specifies that the data should be stored and the
    *      operations executed on the NVIDIA GPU accelerator;
    * +    HipExecutor specifies that the data should be stored and the
    *      operations executed on the GPU accelerator using HIP;
    * +    ReferenceExecutor executes a non-optimized reference implementation,
    *      which can be used to debug the library.
    */
   GinkgoPreconditioner(GinkgoExecutor &exec);

   /**
    * Destructor.
    */
   virtual ~GinkgoPreconditioner() = default;

   /**
    * Generate the preconditioner for the given matrix @p op,
    * which must be of MFEM SparseMatrix type.
    * Calling this function is only required when creating a
    * preconditioner for use with another MFEM solver; to use with
    * a Ginkgo solver, get the LinOpFactory  pointer through @p GetFactory()
    * and pass to the Ginkgo solver constructor.
    */
   void SetOperator(const Operator &op) override;

   /**
    * Apply the preconditioner to input vector @p x, with out @p y.
    */
   void Mult(const Vector &x, Vector &y) const override;

   /**
    * Return a pointer to the LinOpFactory that will generate the preconditioner
    * with the parameters set through the specific constructor.
    */
   const std::shared_ptr<gko::LinOpFactory> GetFactory() const
   {
      return this->precond_gen;
   };

   /**
    * Return a pointer to the generated preconditioner for a specific matrix
    * (that has previously been set with @p SetOperator).
    */
   const std::shared_ptr<gko::LinOp> GetGeneratedPreconditioner() const
   {
      return this->generated_precond;
   };

   /**
    * Return whether this GinkgoPreconditioner object has an explicitly-
    * generated preconditioner, built for a specific matrix.
    */
   bool HasGeneratedPreconditioner() const
   {
      return this->has_generated_precond;
   };

protected:
   /**
    * The Ginkgo generated solver factory object.
    */
   std::shared_ptr<gko::LinOpFactory> precond_gen;

   /**
    * Generated Ginkgo preconditioner for a specific matrix, created through
    * @p SetOperator(), or a wrapped MFEM preconditioner.
    * Must exist to use @p Mult().
    */
   std::shared_ptr<gko::LinOp> generated_precond;

   /**
    * The execution paradigm in Ginkgo. The choices are between
    * `gko::OmpExecutor`, `gko::CudaExecutor` and `gko::ReferenceExecutor`
    * and more details can be found in Ginkgo's documentation.
    */
   std::shared_ptr<gko::Executor> executor;

   /**
    * Whether or not we have generated a specific preconditioner for
    * a matrix.
    */
   bool has_generated_precond;

};

/**
* This class forms the base class for all of Ginkgo's iterative solvers.
* It is not intended to be used directly by MFEM applications. The various
* derived classes only take the additional data that is specific to them
* and solve the given linear system. The entire collection of solvers that
* Ginkgo implements is available at the Ginkgo documentation and manual pages,
* https://ginkgo-project.github.io/ginkgo/doc/develop.
*
* @ingroup Ginkgo
*/
class GinkgoIterativeSolver : public Solver
{
public:
   /**
    * Return a pointer to the LinOpFactory that will generate the solver
    * with the parameters set through the specific constructor.
    */
   const std::shared_ptr<gko::LinOpFactory> GetFactory() const
   {
      return this->solver_gen;
   };

   void SetPrintLevel(int print_lvl) { print_level = print_lvl; }

   int GetNumIterations() const { return final_iter; }
   int GetConverged() const { return converged; }
   real_t GetFinalNorm() const { return final_norm; }

   /**
    * If the Operator is a SparseMatrix, set up a Ginkgo Csr matrix
    * to use its data directly.  If the Operator is not a matrix,
    * create an OperatorWrapper for it and store.
    */
   void SetOperator(const Operator &op) override;

   /**
    * Solve the linear system <tt>Ax=y</tt>. Dependent on the information
    * provided by derived classes one of Ginkgo's linear solvers is chosen.
    */
   void Mult(const Vector &x, Vector &y) const override;

   /**
    * Return whether this GinkgoIterativeSolver object will use
    * VectorWrapper types for input and output vectors.
    * Note that Mult() will automatically create these wrappers if needed.
    */
   bool UsesVectorWrappers() const
   {
      return this->needs_wrapped_vecs;
   };

   /**
    * Destructor.
    */
   virtual ~GinkgoIterativeSolver() = default;

protected:
   /**
    * Constructor.
    *
    * The @p exec defines the paradigm where the solution is computed.
    * @p use_implicit_res_norm is for internal use by the derived classes
    * for specific Ginkgo solvers; it indicates whether the solver makes
    * an implicit residual norm estimate available for convergence checking.
    * Each derived class automatically sets the correct value when calling this
    * base class constructor.
    *
    */
   GinkgoIterativeSolver(GinkgoExecutor &exec,
                         bool use_implicit_res_norm);

   bool use_implicit_res_norm;
   int print_level;
   int max_iter;
   real_t rel_tol;
   real_t abs_tol;
   mutable real_t final_norm;
   mutable int final_iter;
   mutable int converged;

   /**
    * The Ginkgo solver factory object, to generate specific solvers.
    */
   std::shared_ptr<gko::LinOpFactory> solver_gen;

   /**
    * The Ginkgo solver object, generated for a specific operator.
    */
   std::shared_ptr<gko::LinOp> solver;

   /**
    * The residual criterion object that controls the reduction of the residual
    * relative to the initial residual.
    */
   std::shared_ptr<gko::stop::ResidualNorm<>::Factory>
   rel_criterion;

   /**
    * The residual criterion object that controls the reduction of the residual
    * based on an absolute tolerance.
    */
   std::shared_ptr<gko::stop::ResidualNorm<>::Factory>
   abs_criterion;

   /**
    * The implicit residual criterion object that controls the reduction of the residual
    * relative to the initial residual, based on an implicit residual norm value.
    */
   std::shared_ptr<gko::stop::ImplicitResidualNorm<>::Factory>
   imp_rel_criterion;

   /**
    * The implicit residual criterion object that controls the reduction of the residual
    * based on an absolute tolerance, based on an implicit residual norm value.
    */
   std::shared_ptr<gko::stop::ImplicitResidualNorm<>::Factory>
   imp_abs_criterion;

   /**
    * The Ginkgo convergence logger used to check for convergence and other
    * solver data if needed.
    */
   mutable std::shared_ptr<gko::log::Convergence<>> convergence_logger;

   /**
    * The residual logger object used to check for convergence and other solver
    * data if needed.
    */
   mutable std::shared_ptr<ResidualLogger<>> residual_logger;

   /**
    * The Ginkgo combined factory object is used to create a combined stopping
    * criterion to be passed to the solver.
    */
   std::shared_ptr<gko::stop::Combined::Factory> combined_factory;

   /**
    * The execution paradigm in Ginkgo. The choices are between
    * `gko::OmpExecutor`, `gko::CudaExecutor` and `gko::ReferenceExecutor`
    * and more details can be found in Ginkgo's documentation.
    */
   std::shared_ptr<gko::Executor> executor;

   /**
    * Whether or not we need to use VectorWrapper types with this solver.
    */
   bool needs_wrapped_vecs;

   /**
    * Whether or not we need to use VectorWrapper types with the preconditioner
    * or an inner solver.  This value is set upon creation of the
    * GinkgoIterativeSolver object and should never change.
    */
   bool sub_op_needs_wrapped_vecs;

   /** Rebuild the Ginkgo stopping criterion factory with the latest values
    * of rel_tol, abs_tol, and max_iter.
    */
   void update_stop_factory();

private:
   /**
    * Initialize the Ginkgo logger object with event masks. Refer to the logging
    * event masks in Ginkgo's .../include/ginkgo/core/log/logger.hpp.
    */
   void
   initialize_ginkgo_log(gko::matrix::Dense<real_t>* b) const;

   /**
    * Pointer to either a Ginkgo CSR matrix or an OperatorWrapper wrapping
    * an MFEM Operator (for matrix-free evaluation).
    */
   std::shared_ptr<gko::LinOp> system_oper;

};

/**
 * This class adds helper functions for updating Ginkgo factories
 * and solvers, when the full class type is needed.  The derived classes
 * should inherit from this class, rather than from GinkgoIterativeSolver
 * directly.
 */
template<typename SolverType>
class EnableGinkgoSolver : public GinkgoIterativeSolver
{
public:
   EnableGinkgoSolver(GinkgoExecutor &exec, bool use_implicit_res_norm) :
      GinkgoIterativeSolver(exec, use_implicit_res_norm) {}

   void SetRelTol(real_t rtol)
   {
      rel_tol = rtol;
      this->update_stop_factory();
      auto current_params = gko::as<typename SolverType::Factory>
                            (solver_gen)->get_parameters();
      this->solver_gen = current_params.with_criteria(this->combined_factory)
                         .on(this->executor);
      if (solver)
      {
         gko::as<SolverType>(solver)->set_stop_criterion_factory(combined_factory);
      }
   }

   void SetAbsTol(real_t atol)
   {
      abs_tol = atol;
      this->update_stop_factory();
      auto current_params = gko::as<typename SolverType::Factory>
                            (solver_gen)->get_parameters();
      this->solver_gen = current_params.with_criteria(this->combined_factory)
                         .on(this->executor);
      if (solver)
      {
         gko::as<SolverType>(solver)->set_stop_criterion_factory(combined_factory);
      }
   }

   void SetMaxIter(int max_it)
   {
      max_iter = max_it;
      this->update_stop_factory();
      auto current_params = gko::as<typename SolverType::Factory>
                            (solver_gen)->get_parameters();
      this->solver_gen = current_params.with_criteria(this->combined_factory)
                         .on(this->executor);
      if (solver)
      {
         gko::as<SolverType>(solver)->set_stop_criterion_factory(combined_factory);
      }
   }
};


/**
 * An implementation of the solver interface using the Ginkgo CG solver.
 *
 * @ingroup Ginkgo
 */
class CGSolver : public EnableGinkgoSolver<gko::solver::Cg<real_t>>
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    */
   CGSolver(GinkgoExecutor &exec);

   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    * @param[in] preconditioner The preconditioner for the solver.
    */
   CGSolver(GinkgoExecutor &exec,
            const GinkgoPreconditioner &preconditioner);
};


/**
 * An implementation of the solver interface using the Ginkgo BiCGStab solver.
 *
 * @ingroup Ginkgo
 */
class BICGSTABSolver : public EnableGinkgoSolver<gko::solver::Bicgstab<real_t>>
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    */
   BICGSTABSolver(GinkgoExecutor &exec);

   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    * @param[in] preconditioner The preconditioner for the solver.
    */
   BICGSTABSolver(GinkgoExecutor &exec,
                  const GinkgoPreconditioner &preconditioner);

};

/**
 * An implementation of the solver interface using the Ginkgo CGS solver.
 *
 * CGS or the conjugate gradient square method is an iterative type Krylov
 * subspace method which is suitable for general systems.
 *
 * @ingroup Ginkgo
 */
class CGSSolver : public EnableGinkgoSolver<gko::solver::Cgs<real_t>>
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    */
   CGSSolver(GinkgoExecutor &exec);

   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    * @param[in] preconditioner The preconditioner for the solver.
    */
   CGSSolver(GinkgoExecutor &exec,
             const GinkgoPreconditioner &preconditioner);
};

/**
 * An implementation of the solver interface using the Ginkgo FCG solver.
 *
 * FCG or the flexible conjugate gradient method is an iterative type Krylov
 * subspace method which is suitable for symmetric positive definite methods.
 *
 * Though this method performs very well for symmetric positive definite
 * matrices, it is in general not suitable for general matrices.
 *
 * In contrast to the standard CG based on the Polack-Ribiere formula, the
 * flexible CG uses the Fletcher-Reeves formula for creating the orthonormal
 * vectors spanning the Krylov subspace. This increases the computational cost
 * of every Krylov solver iteration but allows for non-constant preconditioners.
 *
 * @ingroup Ginkgo
 */
class FCGSolver : public EnableGinkgoSolver<gko::solver::Fcg<real_t>>
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    */
   FCGSolver(GinkgoExecutor &exec);

   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    * @param[in] preconditioner The preconditioner for the solver.
    */
   FCGSolver(GinkgoExecutor &exec,
             const GinkgoPreconditioner &preconditioner);
};

/**
 * An implementation of the solver interface using the Ginkgo GMRES solver.
 *
 * @ingroup Ginkgo
 */
class GMRESSolver : public EnableGinkgoSolver<gko::solver::Gmres<real_t>>
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    * @param[in] dim  The Krylov dimension of the solver. Value of 0 will
    *                  let Ginkgo use its own internal default value.
    */
   GMRESSolver(GinkgoExecutor &exec, int dim = 0);

   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    * @param[in] preconditioner The preconditioner for the solver.
    * @param[in] dim  The Krylov dimension of the solver. Value of 0 will
    *                  let Ginkgo use its own internal default value.
    */
   GMRESSolver(GinkgoExecutor &exec,
               const GinkgoPreconditioner &preconditioner,
               int dim = 0);

   /**
    * Change the Krylov dimension of the solver.
    */
   void SetKDim(int dim);

protected:
   int m; // Dimension of Krylov subspace
};

using gko::solver::cb_gmres::storage_precision;
/**
 * An implementation of the solver interface using the Ginkgo
 * Compressed Basis GMRES solver. With CB-GMRES, the Krylov basis
 * is "compressed" by storing in a lower precision. Currently, computations
 * are always performed in the MFEM-defined `real_t` precision, when using this
 * MFEM integration.
 * The Ginkgo storage precision options are accessed
 * through Ginkgo::storage_precision::*.  The default choice
 * is Ginkgo::storage_precision::reduce1, i.e., store in float
 * instead of double or half instead of float.
 *
 * @ingroup Ginkgo
 */
class CBGMRESSolver : public EnableGinkgoSolver<gko::solver::CbGmres<real_t>>
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    * @param[in] dim  The Krylov dimension of the solver. Value of 0 will
    *  let Ginkgo use its own internal default value.
    * @param[in] prec  The storage precision used in the CB-GMRES. Options
    *  are: keep (keep `real_t` precision), reduce1 (double -> float
    *  or float -> half), reduce2 (double -> half or float -> half),
    *  integer (`real_t` -> int64), ireduce1 (double -> int32 or
    *  float -> int16), ireduce2 (double -> int16 or float -> int16).
    *  See Ginkgo documentation for more about CB-GMRES.
    */
   CBGMRESSolver(GinkgoExecutor &exec, int dim = 0,
                 storage_precision prec = storage_precision::reduce1);

   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    * @param[in] preconditioner The preconditioner for the solver.
    * @param[in] dim  The Krylov dimension of the solver. Value of 0 will
    *  let Ginkgo use its own internal default value.
    * @param[in] prec  The storage precision used in the CB-GMRES. Options
    *  are: keep (keep `real_t` precision), reduce1 (double -> float
    *  or float -> half), reduce2 (double -> half or float -> half),
    *  integer (`real_t` -> int64), ireduce1 (double -> int32 or
    *  float -> int16), ireduce2 (double -> int16 or float -> int16).
    *  See Ginkgo documentation for more about CB-GMRES.
    */
   CBGMRESSolver(GinkgoExecutor &exec,
                 const GinkgoPreconditioner &preconditioner,
                 int dim = 0,
                 storage_precision prec = storage_precision::reduce1);

   /**
    * Change the Krylov dimension of the solver.
    */
   void SetKDim(int dim);

protected:
   int m; // Dimension of Krylov subspace
};

/**
 * An implementation of the solver interface using the Ginkgo IR solver.
 *
 * Iterative refinement (IR) is an iterative method that uses another coarse
 * method to approximate the error of the current solution via the current
 * residual.
 *
 * @ingroup Ginkgo
 */
class IRSolver : public EnableGinkgoSolver<gko::solver::Ir<real_t>>
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    */
   IRSolver(GinkgoExecutor &exec);

   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the solver.
    * @param[in] inner_solver  The inner solver for the main solver.
    */
   IRSolver(GinkgoExecutor &exec,
            const GinkgoIterativeSolver &inner_solver);

};

/**
 * An implementation of the preconditioner interface using the Ginkgo Jacobi
 * preconditioner.
 *
 * @ingroup Ginkgo
 */
class JacobiPreconditioner : public GinkgoPreconditioner
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the preconditioner.
    * @param[in] storage_opt  The storage optimization parameter.
    * @param[in] accuracy The relative accuracy for the adaptive version.
    * @param[in] max_block_size Maximum block size.
    * See the Ginkgo documentation for more information on these parameters.
    */
   JacobiPreconditioner(
      GinkgoExecutor &exec,
      const std::string &storage_opt = "none",
      const real_t accuracy = 1.e-1,
      const int max_block_size = 32
   );
};

/**
 * An implementation of the preconditioner interface using the Ginkgo
 * Incomplete LU preconditioner (ILU(0)).
 *
 * @ingroup Ginkgo
 */
class IluPreconditioner : public GinkgoPreconditioner
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the preconditioner.
    * @param[in] factorization_type The factorization type: "exact" or
    *  "parilu".
    * @param[in] sweeps The number of sweeps to do in the ParIlu
    *  factorization algorithm.  A value of 0 tells Ginkgo to use its
    *  internal default value.  This parameter is ignored in the case
    *  of an exact factorization.
    * @param[in] skip_sort Only set this to true if the input matrix
    * that will be used to generate this preconditioner is guaranteed
    * to be sorted by column.
    *
    * Note: The use of this preconditioner will sort any input matrix
    * given to it, potentially changing the order of the stored values.
    */
   IluPreconditioner(
      GinkgoExecutor &exec,
      const std::string &factorization_type = "exact",
      const int sweeps = 0,
      const bool skip_sort = false
   );
};

/**
 * An implementation of the preconditioner interface using the Ginkgo
 * Incomplete LU-Incomplete Sparse Approximate Inverse preconditioner.
 * The Ilu-ISAI preconditioner differs from the Ilu preconditioner in
 * that Incomplete Sparse Approximate Inverses (ISAIs) are formed
 * to approximate solving the triangular systems defined by L and U.
 * When the preconditioner is applied, these ISAI matrices are applied
 * through matrix-vector multiplication.
 *
 * @ingroup Ginkgo
 */
class IluIsaiPreconditioner : public GinkgoPreconditioner
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the preconditioner.
    * @param[in] factorization_type The factorization type: "exact" or
    *  "parilu".
    * @param[in] sweeps The number of sweeps to do in the ParIlu
    *  factorization algorithm.  A value of 0 tells Ginkgo to use its
    *  internal default value.  This parameter is ignored in the case
    *  of an exact factorization.
    * @param[in] sparsity_power Parameter determining the sparsity pattern of
    * the ISAI approximations.
    * @param[in] skip_sort Only set this to true if the input matrix
    * that will be used to generate this preconditioner is guaranteed
    * to be sorted by column.
    * See the Ginkgo documentation for more information on these parameters.
    *
    * Note: The use of this preconditioner will sort any input matrix
    * given to it, potentially changing the order of the stored values.
    */
   IluIsaiPreconditioner(
      GinkgoExecutor &exec,
      const std::string &factorization_type = "exact",
      const int sweeps = 0,
      const int sparsity_power = 1,
      const bool skip_sort = false
   );
};

/**
 * An implementation of the preconditioner interface using the Ginkgo
 * Incomplete Cholesky preconditioner (IC(0)).
 *
 * @ingroup Ginkgo
 */
class IcPreconditioner : public GinkgoPreconditioner
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the preconditioner.
    * @param[in] factorization_type The factorization type: "exact" or
    *  "paric".
    * @param[in] sweeps The number of sweeps to do in the ParIc
    *  factorization algorithm.  A value of 0 tells Ginkgo to use its
    *  internal default value.  This parameter is ignored in the case
    *  of an exact factorization.
    * @param[in] skip_sort Only set this to true if the input matrix
    * that will be used to generate this preconditioner is guaranteed
    * to be sorted by column.
    *
    * Note: The use of this preconditioner will sort any input matrix
    * given to it, potentially changing the order of the stored values.
    */
   IcPreconditioner(
      GinkgoExecutor &exec,
      const std::string &factorization_type = "exact",
      const int sweeps = 0,
      const bool skip_sort = false
   );
};

/**
 * An implementation of the preconditioner interface using the Ginkgo
 * Incomplete Cholesky-Incomplete Sparse Approximate Inverse  preconditioner.
 * The Ic-ISAI preconditioner differs from the Ic preconditioner in
 * that Incomplete Sparse Approximate Inverses (ISAIs) are formed
 * to approximate solving the triangular systems defined by L and L^T.
 * When the preconditioner is applied, these ISAI matrices are applied
 * through matrix-vector multiplication.
 *
 * @ingroup Ginkgo
 */
class IcIsaiPreconditioner : public GinkgoPreconditioner
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the preconditioner.
    * @param[in] factorization_type The factorization type: "exact" or
    *  "paric".
    * @param[in] sweeps The number of sweeps to do in the ParIc
    *  factorization algorithm.  A value of 0 tells Ginkgo to use its
    *  internal default value.  This parameter is ignored in the case
    *  of an exact factorization.
    * @param[in] sparsity_power Parameter determining the sparsity pattern of
    * the ISAI approximations.
    * @param[in] skip_sort Only set this to true if the input matrix
    * that will be used to generate this preconditioner is guaranteed
    * to be sorted by column.
    * See the Ginkgo documentation for more information on these parameters.
    *
    * Note: The use of this preconditioner will sort any input matrix
    * given to it, potentially changing the order of the stored values.
    */
   IcIsaiPreconditioner(
      GinkgoExecutor &exec,
      const std::string &factorization_type = "exact",
      const int sweeps = 0,
      const int sparsity_power = 1,
      const bool skip_sort = false
   );
};

/**
 * A wrapper that allows Ginkgo to use MFEM preconditioners.
 *
 * @ingroup Ginkgo
 */
class MFEMPreconditioner : public GinkgoPreconditioner
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec The execution paradigm for the preconditioner.
    * @param[in] mfem_precond The MFEM Preconditioner to wrap.
    */
   MFEMPreconditioner(
      GinkgoExecutor &exec,
      const Solver &mfem_precond
   );

   /**
    * SetOperator is not allowed for this type of preconditioner;
    * this function overrides the base class in order to give an
    * error if SetOperator() is called for this class.
    */
   void SetOperator(const Operator &op) override
   {
      MFEM_ABORT("Ginkgo::MFEMPreconditioner must be constructed "
                 "with the MFEM Operator that it will wrap as an argument;\n"
                 "calling SetOperator() is not allowed.");
   };
};
} // namespace Ginkgo

} // namespace mfem

#endif // MFEM_USE_GINKGO

#endif // MFEM_GINKGO
