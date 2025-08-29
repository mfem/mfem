#include "mfem.hpp"
#include "problems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


#ifndef PARIPSOLVER
#define PARIPSOLVER

class ParInteriorPointSolver
{
protected:
   OptContactProblem* problem = nullptr;
   int numActiveConstraints = -1;
   real_t OptTol;
   int  max_iter;
   int  iter=0;
   real_t mu_k; // \mu_k
   Vector lk, zlk;

   real_t sMax, kSig, tauMin, eta, thetaMin, delta, sTheta, sPhi, kMu, thetaMu;
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
   bool inFilterRegion = false;
   real_t Dxphi0_xhat;

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

   int jOpt;
   bool converged;

   int MyRank;
   bool iAmRoot;

   bool useMassWeights = false;
   MPI_Comm comm;
public:
   ParInteriorPointSolver(OptContactProblem*);
   void Mult(const BlockVector&, BlockVector&);
   void Mult(const Vector&, Vector &);
   void SetTol(real_t);
   void SetMaxIter(int);
   void SetBarrierParameter(real_t);
   void SetUsingMassWeights(bool);
   void SetLinearSolver(Solver * solver_) { solver = solver_ ;}
   bool GetConverged() const;
   Array<int> & GetCGIterNumbers() {return cgnum_iterations;};
   int GetNumIterations() {return iter;};
   real_t GetNumActiveConstraints() { return numActiveConstraints;};
   virtual ~ParInteriorPointSolver();
private: 
   real_t MaxStepSize(Vector&, Vector&, Vector&, real_t);
   real_t MaxStepSize(Vector&, Vector&, real_t);
   void FormIPNewtonMat(BlockVector&, Vector&, Vector&, BlockOperator &,
                        real_t delta = 0.0);
   void IPNewtonSolve(BlockVector&, Vector&, Vector&, Vector&, BlockVector&,
                      bool &, real_t, real_t delta = 0.0);
   void lineSearch(BlockVector&, BlockVector&, real_t);
   void projectZ(const Vector &, Vector &, real_t);
   void filterCheck(real_t, real_t);
   real_t OptimalityError(const BlockVector &, const Vector &, const Vector &, real_t, bool);
   real_t OptimalityError(const BlockVector &, const Vector &, const Vector &, bool);
   real_t theta(const BlockVector &);
   real_t phi(const BlockVector &, real_t);
   real_t phi(const BlockVector &, real_t, int &);
   void Dxphi(const BlockVector &, real_t, BlockVector &);
   real_t L(const BlockVector &, const Vector &, const Vector &);
   void DxL(const BlockVector &, const Vector &, const Vector &, BlockVector &);
   bool CurvatureTest(const BlockOperator & A, const BlockVector & Xhat,
                      const Vector &l, const BlockVector & b, const real_t & delta);
   void Clear()
   {
      F1.DeleteAll();
      F2.DeleteAll();
      mu_k = 1.0;
   };
};

#endif
