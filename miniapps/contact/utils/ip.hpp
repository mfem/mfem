#include "mfem.hpp"
#include "problems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


#ifndef PARIPSOLVER
#define PARIPSOLVER

class IPSolver
{
protected:
   OptContactProblem* problem = nullptr;
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
   real_t thx0, thxtrial;
   real_t phx0, phxtrial;
   bool descentDirection = false;
   bool switchCondition = false;
   bool sufficientDecrease = false;
   bool lineSearchSuccess = false;
   //bool inFilterRegion = false;

   int dimU, dimM, dimC;
   int dimG; // num of gap constraints
   Array<int> constraint_offsets;
   int gdimU, gdimM, gdimC;
   Array<int> block_offsetsumlz, block_offsetsuml, block_offsetsx;
   Vector ml;

   HypreParMatrix * Huu = nullptr;
   HypreParMatrix * Hum = nullptr;
   HypreParMatrix * Hmu = nullptr;
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

   Array<int> cgnum_iterations;
   Solver * solver = nullptr;
   ParFiniteElementSpace *pfes = nullptr;

   bool converged = false;

   int myid = -1;

   int print_level = 0;
   MPI_Comm comm;
public:
   IPSolver(OptContactProblem*);
   void Mult(const BlockVector&, BlockVector&);
   void Mult(const Vector&, Vector &);
   void SetTol(real_t);
   void SetMaxIter(int);
   void SetLinearSolver(Solver * solver_) { solver = solver_; };
   void SetPrintLevel(int print_level_) { print_level = print_level_; };
   bool GetConverged() const;
   Array<int> & GetCGNumIterations() {return cgnum_iterations;};
   int GetNumIterations() {return iter;};
   virtual ~IPSolver();
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
