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

#ifndef MFEM_GINKGO
#define MFEM_GINKGO

#include <iomanip>
#include <ios>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>


#include "../config/config.hpp"
#include "operator.hpp"
#include "sparsemat.hpp"
#include "solvers.hpp"

#ifdef MFEM_USE_GINKGO
#include <ginkgo/ginkgo.hpp>
#endif

namespace mfem
{
namespace GinkgoWrappers
{
// Utility function which gets the scalar value of a Ginkgo gko::matrix::Dense
// matrix representing the norm of a vector.
template <typename ValueType=double>
double get_norm(const gko::matrix::Dense<ValueType> *norm)
{
   // Put the value on CPU thanks to the master executor
   auto cpu_norm = clone(norm->get_executor()->get_master(), norm);
   // Return the scalar value contained at position (0, 0)
   return cpu_norm->at(0, 0);
}

// Utility function which computes the norm of a Ginkgo gko::matrix::Dense
// vector.
template <typename ValueType=double>
double compute_norm(const gko::matrix::Dense<ValueType> *b)
{
   // Get the executor of the vector
   auto exec = b->get_executor();
   // Initialize a result scalar containing the value 0.0.
   auto b_norm = gko::initialize<gko::matrix::Dense<ValueType>>({0.0}, exec);
   // Use the dense `compute_norm2` function to compute the norm.
   b->compute_norm2(lend(b_norm));
   // Use the other utility function to return the norm contained in `b_norm``
   return std::pow(get_norm(lend(b_norm)),2);
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
 * @ingroup GinkgoWrappers
 */
template <typename ValueType=double>
struct ResidualLogger : gko::log::Logger
{
   // Output the logger's data in a table format
   void write() const
   {
      // Print a header for the table
      mfem::out << "Iteration log with real residual norms:" << std::endl;
      mfem::out << '|' << std::setw(10) << "Iteration" << '|' << std::setw(25)
                << "Real Residual Norm" << '|' << std::endl;
      // Print a separation line. Note that for creating `10` characters
      // `std::setw()` should be set to `11`.
      mfem::out << '|' << std::setfill('-') << std::setw(11) << '|' <<
                std::setw(26) << '|' << std::setfill(' ') << std::endl;
      // Print the data one by one in the form
      mfem::out << std::scientific;
      for (std::size_t i = 0; i < iterations.size(); i++)
      {
         mfem::out << '|' << std::setw(10) << iterations[i] << '|'
                   << std::setw(25) << real_norms[i] << '|' << std::endl;
      }
      // std::defaultfloat could be used here but some compilers do not support
      // it properly, e.g. the Intel compiler
      mfem::out.unsetf(std::ios_base::floatfield);
      // Print a separation line
      mfem::out << '|' << std::setfill('-') << std::setw(11) << '|' <<
                std::setw(26) << '|' << std::setfill(' ') << std::endl;
   }

   using gko_dense = gko::matrix::Dense<ValueType>;

   // Customize the logging hook which is called every time an iteration is
   // completed
   void on_iteration_complete(const gko::LinOp *,
                              const gko::size_type &iteration,
                              const gko::LinOp *residual,
                              const gko::LinOp *solution,
                              const gko::LinOp *residual_norm) const override
   {
      // // If the solver shares a residual norm, log its value
      // if (residual_norm)
      // {
      //    auto dense_norm = gko::as<gko_dense>(residual_norm);
      //    // Add the norm to the `recurrent_norms` vector
      //    recurrent_norms.push_back(get_norm(dense_norm));
      //    // Otherwise, use the recurrent residual vector
      // }
      // else
      // {
      //    auto dense_residual = gko::as<gko_dense>(residual);
      //    // Compute the residual vector's norm
      //    auto norm = compute_norm(gko::lend(dense_residual));
      //    // Add the computed norm to the `recurrent_norms` vector
      //    recurrent_norms.push_back(norm);
      // }

      // If the solver shares the current solution vector
      if (solution)
      {
         // Store the matrix's executor
         auto exec = matrix->get_executor();
         // Create a scalar containing the value 1.0
         auto one = gko::initialize<gko_dense>({1.0}, exec);
         // Create a scalar containing the value -1.0
         auto neg_one = gko::initialize<gko_dense>({-1.0}, exec);
         // Instantiate a temporary result variable
         auto res = gko::clone(b);
         // Compute the real residual vector by calling apply on the system
         // matrix
         matrix->apply(gko::lend(one), gko::lend(solution),
                       gko::lend(neg_one), gko::lend(res));

         // Compute the norm of the residual vector and add it to the
         // `real_norms` vector
         real_norms.push_back(compute_norm(gko::lend(res)));
      }
      else
      {
         // Add to the `real_norms` vector the value -1.0 if it could not be
         // computed
         real_norms.push_back(-1.0);
      }

      // Add the current iteration number to the `iterations` vector
      iterations.push_back(iteration);
   }

   // Construct the logger and store the system matrix and b vectors
   ResidualLogger(std::shared_ptr<const gko::Executor> exec,
                  const gko::LinOp *matrix, const gko_dense *b)
      : gko::log::Logger(exec,
                         gko::log::Logger::iteration_complete_mask),
        matrix{matrix},
        b{b}
   {}

private:
   // Pointer to the system matrix
   const gko::LinOp *matrix;
   // Pointer to the right hand sides
   const gko_dense *b;
   // Vector which stores all the recurrent residual norms
   mutable std::vector<ValueType> recurrent_norms{};
   // Vector which stores all the real residual norms
   mutable std::vector<ValueType> real_norms{};
   // Vector which stores all the iteration numbers
   mutable std::vector<std::size_t> iterations{};
};


/**
* This class forms the base class for all of Ginkgo's iterative solvers.  The
* various derived classes only take the additional data that is specific to them
* and solve the given linear system. The entire collection of solvers that
* Ginkgo implements is available at the Ginkgo documentation and manual pages,
* https://ginkgo-project.github.io/ginkgo/doc/develop.
*
* @ingroup GinkgoWrappers
*/
class GinkgoIterativeSolverBase
{
public:
   /**
    * Constructor.
    *
    * The @p exec_type defines the paradigm where the solution is computed.
    * It is a string and the choices are "omp" , "reference" or "cuda".
    * The respective strings create the respective executors as given below.
    *
    * Ginkgo currently supports three different executor types:
    *
    * +    OmpExecutor specifies that the data should be stored and the
    *      associated operations executed on an OpenMP-supporting device (e.g.
    *      host CPU);
    * ```
    * auto omp = gko::create<gko::OmpExecutor>();
    * ```
    * +    CudaExecutor specifies that the data should be stored and the
    *      operations executed on the NVIDIA GPU accelerator;
    * ```
    * if(gko::CudaExecutor::get_num_devices() > 0 ) {
    *    auto cuda = gko::create<gko::CudaExecutor>();
    * }
    * ```
    * +    ReferenceExecutor executes a non-optimized reference implementation,
    *      which can be used to debug the library.
    * ```
    * auto ref = gko::create<gko::ReferenceExecutor>();
    * ```
    *
    * The following code snippet demonstrates the using of the OpenMP executor
    * to create a solver which would use the OpenMP paradigm to the solve the
    * system on the CPU.
    *
    * ```
    * auto omp = gko::create<gko::OmpExecutor>();
    * using cg = gko::solver::Cg<>;
    * auto solver_gen =
    *     cg::build()
    *          .with_criteria(
    *              gko::stop::Iteration::build().with_max_iters(20u).on(omp),
    *              gko::stop::ResidualNormReduction<>::build()
    *                  .with_reduction_factor(1e-6)
    *                  .on(omp))
    *          .on(omp);
    * auto solver = solver_gen->generate(system_matrix);
    *
    * solver->apply(lend(rhs), lend(solution));
    * ```
    */
   GinkgoIterativeSolverBase(const std::string &exec_type, int print_iter,
                             int max_num_iter, double RTOLERANCE, double ATOLERANCE);

   /**
    * Destructor.
    */
   virtual ~GinkgoIterativeSolverBase() = default;

   /**
    * Initialize the matrix and copy over its data to Ginkgo's data structures.
    */
   void
   initialize(const SparseMatrix *matrix);

   /**
    * Solve the linear system <tt>Ax=b</tt>. Dependent on the information
    * provided by derived classes one of Ginkgo's linear solvers is chosen.
    */
   void
   apply(Vector &solution, const Vector &rhs);

   /**
    * Solve the linear system <tt>Ax=b</tt>. Dependent on the information
    * provided by derived classes one of Ginkgo's linear solvers is chosen.
    */
   void
   solve(const SparseMatrix *matrix,
         Vector &            solution,
         const Vector &      rhs);


protected:
   int print_lvl;
   int max_iter;
   double rel_tol;
   double abs_tol;
   mutable double final_norm;
   mutable int final_iter;
   mutable int converged;

   /**
    * The Ginkgo generated solver factory object.
    */
   std::shared_ptr<gko::LinOpFactory> solver_gen;

   /**
    * The residual criterion object that controls the reduction of the residual
    * based on the tolerance set in the solver_control member.
    */
   std::shared_ptr<gko::stop::ResidualNormReduction<>::Factory>
   residual_criterion;

   /**
    * The Ginkgo convergence logger used to check for convergence and other
    * solver data if needed.
    */
   std::shared_ptr<gko::log::Convergence<>> convergence_logger;

   /**
    * The residual logger object used to check for convergence and other solver
    * data if needed.
    */
   std::shared_ptr<ResidualLogger<>> residual_logger;

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

private:
   /**
    * Initialize the Ginkgo logger object with event masks. Refer to the logging
    * event masks in Ginkgo's .../include/ginkgo/core/log/logger.hpp.
    */
   void
   initialize_ginkgo_log(gko::matrix::Dense<double>* b);

   /**
    * Ginkgo matrix data structure. First template parameter is for storing the
    * array of the non-zeros of the matrix. The second is for the row pointers
    * and the column indices.
    *
    * @todo Templatize based on Matrix type.
    */
   std::shared_ptr<gko::matrix::Csr<>> system_matrix;

   /**
    * The execution paradigm as a string to be set by the user. The choices are
    * between `omp`, `cuda` and `reference` and more details can be found in
    * Ginkgo's documentation.
    */
   const std::string exec_type;
};


/**
 * An implementation of the solver interface using the Ginkgo CG solver.
 *
 * @ingroup GinkgoWrappers
 */
class CGSolver : public GinkgoIterativeSolverBase
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec_type The execution paradigm for the solver.
    * @param[in] print_iter  A setting to control the printing to the screen.
    * @param[in] max_num_iter  The maximum number of iterations to be run.
    * @param[in] RTOLERANCE  The relative tolerance to be achieved.
    * @param[in] ATOLERANCE The absolute tolerance to be achieved.
    */
   CGSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE
   );

   /**
    * Constructor.
    *
    * @param[in] exec_type The execution paradigm for the solver.
    * @param[in] print_iter  A setting to control the printing to the screen.
    * @param[in] max_num_iter  The maximum number of iterations to be run.
    * @param[in] RTOLERANCE  The relative tolerance to be achieved.
    * @param[in] ATOLERANCE The absolute tolerance to be achieved.
    * @param[in] preconditioner The preconditioner for the solver.
    */
   CGSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE,
      const gko::LinOpFactory* preconditioner
   );
};


/**
 * An implementation of the solver interface using the Ginkgo BiCGStab solver.
 *
 * @ingroup GinkgoWrappers
 */
class BICGSTABSolver : public GinkgoIterativeSolverBase
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec_type The execution paradigm for the solver.
    * @param[in] print_iter  A setting to control the printing to the screen.
    * @param[in] max_num_iter  The maximum number of iterations to be run.
    * @param[in] RTOLERANCE  The relative tolerance to be achieved.
    * @param[in] ATOLERANCE The absolute tolerance to be achieved.
    */
   BICGSTABSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE
   );

   /**
    * Constructor.
    *
    * @param[in] exec_type The execution paradigm for the solver.
    * @param[in] print_iter  A setting to control the printing to the screen.
    * @param[in] max_num_iter  The maximum number of iterations to be run.
    * @param[in] RTOLERANCE  The relative tolerance to be achieved.
    * @param[in] ATOLERANCE The absolute tolerance to be achieved.
    * @param[in] preconditioner The preconditioner for the solver.
    */
   BICGSTABSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE,
      const gko::LinOpFactory* preconditioner
   );

};

/**
 * An implementation of the solver interface using the Ginkgo CGS solver.
 *
 * CGS or the conjugate gradient square method is an iterative type Krylov
 * subspace method which is suitable for general systems.
 *
 * @ingroup GinkgoWrappers
 */
class CGSSolver : public GinkgoIterativeSolverBase
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec_type The execution paradigm for the solver.
    * @param[in] print_iter  A setting to control the printing to the screen.
    * @param[in] max_num_iter  The maximum number of iterations to be run.
    * @param[in] RTOLERANCE  The relative tolerance to be achieved.
    * @param[in] ATOLERANCE The absolute tolerance to be achieved.
    */
   CGSSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE
   );

   /**
    * Constructor.
    *
    * @param[in] exec_type The execution paradigm for the solver.
    * @param[in] print_iter  A setting to control the printing to the screen.
    * @param[in] max_num_iter  The maximum number of iterations to be run.
    * @param[in] RTOLERANCE  The relative tolerance to be achieved.
    * @param[in] ATOLERANCE The absolute tolerance to be achieved.
    * @param[in] preconditioner The preconditioner for the solver.
    */
   CGSSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE,
      const gko::LinOpFactory* preconditioner
   );
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
 * @ingroup GinkgoWrappers
 */
class FCGSolver : public GinkgoIterativeSolverBase
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec_type The execution paradigm for the solver.
    * @param[in] print_iter  A setting to control the printing to the screen.
    * @param[in] max_num_iter  The maximum number of iterations to be run.
    * @param[in] RTOLERANCE  The relative tolerance to be achieved.
    * @param[in] ATOLERANCE The absolute tolerance to be achieved.
    */
   FCGSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE
   );

   /**
    * Constructor.
    *
    * @param[in] exec_type The execution paradigm for the solver.
    * @param[in] print_iter  A setting to control the printing to the screen.
    * @param[in] max_num_iter  The maximum number of iterations to be run.
    * @param[in] RTOLERANCE  The relative tolerance to be achieved.
    * @param[in] ATOLERANCE The absolute tolerance to be achieved.
    * @param[in] preconditioner The preconditioner for the solver.
    */
   FCGSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE,
      const gko::LinOpFactory* preconditioner
   );
};

/**
 * An implementation of the solver interface using the Ginkgo GMRES solver.
 *
 * @ingroup GinkgoWrappers
 */
class GMRESSolver : public GinkgoIterativeSolverBase
{
public:
   void SetKDim(int dim) { m = dim; }


   /**
    * Constructor.
    *
    * @param[in] exec_type The execution paradigm for the solver.
    * @param[in] print_iter  A setting to control the printing to the screen.
    * @param[in] max_num_iter  The maximum number of iterations to be run.
    * @param[in] RTOLERANCE  The relative tolerance to be achieved.
    * @param[in] ATOLERANCE The absolute tolerance to be achieved.
    */
   GMRESSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE
   );

   /**
    * Constructor.
    *
    * @param[in] exec_type The execution paradigm for the solver.
    * @param[in] print_iter  A setting to control the printing to the screen.
    * @param[in] max_num_iter  The maximum number of iterations to be run.
    * @param[in] RTOLERANCE  The relative tolerance to be achieved.
    * @param[in] ATOLERANCE The absolute tolerance to be achieved.
    * @param[in] preconditioner The preconditioner for the solver.
    */
   GMRESSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE,
      const gko::LinOpFactory* preconditioner
   );

protected:
   int m; // see SetKDim()
};

/**
 * An implementation of the solver interface using the Ginkgo IR solver.
 *
 * Iterative refinement (IR) is an iterative method that uses another coarse
 * method to approximate the error of the current solution via the current
 * residual.
 *
 * @ingroup GinkgoWrappers
 */
class IRSolver : public GinkgoIterativeSolverBase
{
public:
   /**
    * Constructor.
    *
    * @param[in] exec_type The execution paradigm for the solver.
    * @param[in] print_iter  A setting to control the printing to the screen.
    * @param[in] max_num_iter  The maximum number of iterations to be run.
    * @param[in] RTOLERANCE  The relative tolerance to be achieved.
    * @param[in] ATOLERANCE The absolute tolerance to be achieved.
    */
   IRSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE
   );

   /**
    * Constructor.
    *
    * @param[in] exec_type The execution paradigm for the solver.
    * @param[in] print_iter  A setting to control the printing to the screen.
    * @param[in] max_num_iter  The maximum number of iterations to be run.
    * @param[in] RTOLERANCE  The relative tolerance to be achieved.
    * @param[in] ATOLERANCE The absolute tolerance to be achieved.
    * @param[in] inner_solver  The inner solver for the main solver.
    */
   IRSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE,
      const gko::LinOpFactory *inner_solver
   );
};


} // namespace GinkgoWrappers

}

#endif // MFEM_GINKGO
