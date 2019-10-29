// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_GINKGO
#define MFEM_GINKGO

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
/**
 * This class forms the base class for all of Ginkgo's iterative solvers.
 * The various derived classes only take
 * the additional data that is specific to them and solve the given linear
 * system. The entire collection of solvers that Ginkgo implements is
 * available at <a Ginkgo
 * href="https://ginkgo-project.github.io/ginkgo/doc/develop/"> documentation
 * and manual pages</a>.
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
    * associated operations executed on an OpenMP-supporting device (e.g. host
    * CPU);
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
    * provided by derived classes one of Ginkgo's linear solvers is
    * chosen.
    */
   void
   apply(Vector &solution, const Vector &rhs);

   /**
    * Solve the linear system <tt>Ax=b</tt>. Dependent on the information
    * provided by derived classes one of Ginkgo's linear solvers is
    * chosen.
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
    * Initialize the Ginkgo logger object with event masks. Refer to
    * <a
    * href="https://github.com/ginkgo-project/ginkgo/blob/develop/include/ginkgo/core/log/logger.hpp">Ginkgo's
    * logging event masks.</a>
    */
   void
   initialize_ginkgo_log();

   /**
    * Ginkgo matrix data structure. First template parameter is for storing the
    * array of the non-zeros of the matrix. The second is for the row pointers
    * and the column indices.
    *
    * @todo Templatize based on Matrix type.
    */
   std::shared_ptr<gko::matrix::Csr<>> system_matrix;

   /**
    * The execution paradigm as a string to be set by the user. The choices
    * are between `omp`, `cuda` and `reference` and more details can be found
    * in Ginkgo's documentation.
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
    * @param[in,out] solver_control The solver control object is then used to
    * set the parameters and setup the CG solver from the CG factory which
    * solves the linear system.
    *
    * @param[in] exec_type The execution paradigm for the CG solver.
    *
    * @param[in] data The additional data required by the solver.
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
    * @param[in,out] solver_control The solver control object is then used to
    * set the parameters and setup the CG solver from the CG factory which
    * solves the linear system.
    *
    * @param[in] exec_type The execution paradigm for the CG solver.
    *
    * @param[in] preconditioner The preconditioner for the solver.
    *
    * @param[in] data The additional data required by the solver.
    */
   CGSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE,
      const std::shared_ptr<gko::LinOpFactory> &preconditioner
   );
};


/**
 * An implementation of the solver interface using the Ginkgo Bicgstab solver.
 *
 * @ingroup GinkgoWrappers
 */
class BICGSTABSolver : public GinkgoIterativeSolverBase
{
public:
   /**
    * Constructor.
    *
    * @param[in,out] solver_control The solver control object is then used to
    * set the parameters and setup the Bicgstab solver from the Bicgstab
    * factory which solves the linear system.
    *
    * @param[in] exec_type The execution paradigm for the Bicgstab solver.
    *
    * @param[in] data The additional data required by the solver.
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
    * @param[in,out] solver_control The solver control object is then used to
    * set the parameters and setup the Bicgstab solver from the Bicgstab
    * factory which solves the linear system.
    *
    * @param[in] exec_type The execution paradigm for the Bicgstab solver.
    *
    * @param[in] preconditioner The preconditioner for the solver.
    *
    * @param[in] data The additional data required by the solver.
    */
   BICGSTABSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE,
      const std::shared_ptr<gko::LinOpFactory> &preconditioner
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
    * @param[in,out] solver_control The solver control object is then used to
    * set the parameters and setup the CGS solver from the CGS factory which
    * solves the linear system.
    *
    * @param[in] exec_type The execution paradigm for the CGS solver.
    *
    * @param[in] data The additional data required by the solver.
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
    * @param[in,out] solver_control The solver control object is then used to
    * set the parameters and setup the CGS solver from the CGS factory which
    * solves the linear system.
    *
    * @param[in] exec_type The execution paradigm for the CGS solver.
    *
    * @param[in] preconditioner The preconditioner for the solver.
    *
    * @param[in] data The additional data required by the solver.
    */
   CGSSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE,
      const std::shared_ptr<gko::LinOpFactory> &preconditioner
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
 * of every Krylov solver iteration but allows for non-constant
 * preconditioners.
 *
 * @ingroup GinkgoWrappers
 */
class FCGSolver : public GinkgoIterativeSolverBase
{
public:
   /**
    * Constructor.
    *
    * @param[in,out] solver_control The solver control object is then used to
    * set the parameters and setup the FCG solver from the FCG factory which
    * solves the linear system.
    *
    * @param[in] exec_type The execution paradigm for the FCG solver.
    *
    * @param[in] data The additional data required by the solver.
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
    * @param[in,out] solver_control The solver control object is then used to
    * set the parameters and setup the FCG solver from the FCG factory which
    * solves the linear system.
    *
    * @param[in] exec_type The execution paradigm for the FCG solver.
    *
    * @param[in] preconditioner The preconditioner for the solver.
    *
    * @param[in] data The additional data required by the solver.
    */
   FCGSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE,
      const std::shared_ptr<gko::LinOpFactory> &preconditioner
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
    * @param[in,out] solver_control The solver control object is then used to
    * set the parameters and setup the GMRES solver from the GMRES factory
    * which solves the linear system.
    *
    * @param[in] exec_type The execution paradigm for the GMRES solver.
    *
    * @param[in] data The additional data required by the solver.
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
    * @param[in,out] solver_control The solver control object is then used to
    * set the parameters and setup the GMRES solver from the GMRES factory
    * which solves the linear system.
    *
    * @param[in] exec_type The execution paradigm for the GMRES solver.
    *
    * @param[in] preconditioner The preconditioner for the solver.
    *
    * @param[in] data The additional data required by the solver.
    */
   GMRESSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE,
      const std::shared_ptr<gko::LinOpFactory> &preconditioner
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
    * @param[in,out] solver_control The solver control object is then used to
    * set the parameters and setup the IR solver from the IR factory which
    * solves the linear system.
    *
    * @param[in] exec_type The execution paradigm for the IR solver.
    *
    * @param[in] data The additional data required by the solver.
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
    * @param[in,out] solver_control The solver control object is then used to
    * set the parameters and setup the IR solver from the IR factory which
    * solves the linear system.
    *
    * @param[in] exec_type The execution paradigm for the IR solver.
    *
    * @param[in] inner_solver The Inner solver for the IR solver.
    *
    * @param[in] data The additional data required by the solver.
    */
   IRSolver(
      const std::string &   exec_type,
      int print_iter,
      int max_num_iter,
      double RTOLERANCE,
      double ATOLERANCE,
      const std::shared_ptr<gko::LinOpFactory> &inner_solver
   );
};


} // namespace GinkgoWrappers

}

#endif // MFEM_GINKGO
