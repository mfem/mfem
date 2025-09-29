#include "mfem.hpp"
#include "problems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#ifndef PARIPSOLVER
#define PARIPSOLVER

/**
 * @class IPSolver
 * @brief Class for interior-point solvers of contact-problems
 *        described by a OptContactProblem
 *
 * IPSolver is an implementation of a primal-dual interior-point
 * algorithm as described in 
 * Wächter, Andreas, and Lorenz T. Biegler.
 * "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming."
 * Mathematical programming 106.1 (2006): 25-57.
 * 
 * With inertia-free regularization as described in 
 * Chiang, Nai-Yuan, and Victor M. Zavala.
 * "An inertia-free filter line-search algorithm for large-scale nonlinear programming."
 * Computational Optimization and Applications 64.2 (2016): 327-354.
 *
 * This solver contains less solver options than say ipopt
 * but the user has full control over the linear solver.
 *
 * ## Typical usage
 * 1. Initialize IPSolver object.
 * 2. Set solver options e.g.,  SetTol, SetMaxIter, SetLinearSolver
 * 3. Use Mult to apply the solver.
 */

class IPSolver
{
public:
   /// Construct interior-point solver
   IPSolver(OptContactProblem*);
   
   /// Apply the interior-point solver
   void Mult(const Vector&, Vector &);
   
   /// Apply the interior-point solver 
   void Mult(const BlockVector&, BlockVector&);
   
   /// Set absolute tolerance
   void SetTol(real_t);
   
   /// Set maximum number of interior-point steps
   void SetMaxIter(int);
   
   /// Set linear solver
   void SetLinearSolver(Solver * solver_) { solver = solver_; };
   
   /// Set print level
   void SetPrintLevel(int print_level_) { print_level = print_level_; };
   
   /// Get convergence status of most recent Mult call
   bool GetConverged() const;
   
   /// get number of interior-point iterations of most recent Mult call 
   int GetNumIterations() {return iter;};

   /// Get CG iteration counts
   Array<int> & GetNumKrylovIterations() {return num_krylov_iterations;};
   
   virtual ~IPSolver();
protected:
   /// OptContactProblem (not owned).
   OptContactProblem* problem = nullptr;
   
   /// Linear solver (not owned)
   Solver * solver = nullptr;

   real_t abs_tol;
   int  max_iter;
   int  iter=0;
   real_t mu_k;
   Vector lk, zlk;

   real_t kSig, tauMin, eta, thetaMin, delta, sTheta, sPhi, kMu, thetaMu;
   real_t thetaMax, gTheta, gPhi, kEps;

   // filter
   Array<real_t> F1, F2;

   // quantities computed in lineSearch
   real_t alpha, alphaz;
   real_t thx0;
   real_t phx0;
   bool descentDirection = false;
   bool switchCondition = false;
   bool sufficientDecrease = false;
   bool lineSearchSuccess = false;

   int dimU, dimM, dimC;
   Array<int> constraint_offsets;
   int gdimU, gdimM, gdimC;
   Array<int> block_offsetsumlz, block_offsetsuml, block_offsetsx;
   Vector ml; // can this be removed?

   HypreParMatrix * Huu = nullptr;
   HypreParMatrix * Hmm = nullptr;
   HypreParMatrix * Wuu = nullptr;
   HypreParMatrix * Wmm = nullptr;
   HypreParMatrix * Ju = nullptr;
   HypreParMatrix * Jm = nullptr;
   HypreParMatrix * JuT = nullptr;
   HypreParMatrix * JmT = nullptr;

   Vector Mcslump;
   Vector Mvlump;
   Vector Mlump;

   real_t alphaCurvatureTest;
   real_t deltaRegLast;
   real_t deltaRegMin;
   real_t deltaRegMax;
   real_t deltaReg0;

   real_t kRegMinus;
   real_t kRegBarPlus;
   real_t kRegPlus;

   Array<int> num_krylov_iterations;

   bool converged = false;

   int myid = -1;

   int print_level = 0;
   MPI_Comm comm;
private:
   real_t GetMaxStepSize(Vector&, Vector&, Vector&, real_t);
   real_t GetMaxStepSize(Vector&, Vector&, real_t);
   void FormIPNewtonMat(BlockVector&, Vector&, Vector&, BlockOperator &,
                        real_t delta = 0.0);
   void IPNewtonSolve(BlockVector&, Vector&, Vector&, Vector&, BlockVector&,
                      bool &, real_t, real_t delta = 0.0);
   void LineSearch(BlockVector&, BlockVector&, real_t);
   void ProjectZ(const Vector &, Vector &, real_t);
   bool FilterCheck(real_t, real_t);
   real_t OptimalityError(const BlockVector &, const Vector &, const Vector &,
                          real_t mu = 0.0);
   real_t GetTheta(const BlockVector &);
   real_t GetPhi(const BlockVector &, real_t, int eval_err = 0);
   void GetDxphi(const BlockVector &, real_t, BlockVector &);
   real_t EvalLangrangian(const BlockVector &, const Vector &, const Vector &);
   void EvalLagrangianGradient(const BlockVector &, const Vector &, const Vector &,
                               BlockVector &);
   bool CurvatureTest(const BlockOperator & A, const BlockVector & Xhat,
                      const Vector &l, const BlockVector & b, const real_t & delta);
   void UpdateBarrierSubProblem()
   {
      // reduced barrier parameter
      mu_k = max(abs_tol / 10., min(kMu * mu_k, pow(mu_k, thetaMu)));
      // clear subproblem filter
      F1.DeleteAll();
      F2.DeleteAll();
   };
};

#endif
