#include "mfem.hpp"
#include "../problems/parproblems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


#ifndef PARIPSOLVER 
#define PARIPSOLVER

class ParInteriorPointSolver
{
protected:
    // QPOptParContactProblem* problem = nullptr;
    // QPOptParContactProblemTribol* problem = nullptr;
    QPOptParContactProblemSingleMesh* problem = nullptr;
    double OptTol;
    int  max_iter;
    int  iter=0;
    double mu_k; // \mu_k
    Vector lk, zlk;

    double sMax, kSig, tauMin, eta, thetaMin, delta, sTheta, sPhi, kMu, thetaMu;
    double thetaMax, kSoc, gTheta, gPhi, kEps;
	
    // filter
    Array<double> F1, F2;
	
    // quantities computed in lineSearch
    double alpha, alphaz;
    double thx0, thxtrial;
    double phx0, phxtrial;
    bool descentDirection, switchCondition, sufficientDecrease, lineSearchSuccess, inFilterRegion;
    double Dxphi0_xhat;

    int dimU, dimM, dimC;
    int gdimU, gdimM, gdimC;
    Array<int> block_offsetsumlz, block_offsetsuml, block_offsetsx;
    Vector ml;

    Vector ckSoc;
    HypreParMatrix * Huu = nullptr;
    HypreParMatrix * Hum = nullptr;
    HypreParMatrix * Hmu = nullptr;
    HypreParMatrix * Hmm = nullptr;
    HypreParMatrix * Wmm = nullptr;
    HypreParMatrix * Ju = nullptr;
    HypreParMatrix * Jm = nullptr;
    HypreParMatrix * JuT = nullptr;
    HypreParMatrix * JmT = nullptr;

    Array<int> cgnum_iterations;
    ParFiniteElementSpace *pfes = nullptr;
    
    int jOpt;
    bool converged;
    
    int MyRank;
    bool iAmRoot;

    bool saveLogBarrierIterates = false;

    int linSolver;
    double linSolveTol;
    int relax_type = 8;
public:
    // ParInteriorPointSolver(QPOptParContactProblem*);
    // ParInteriorPointSolver(QPOptParContactProblemTribol*);
    ParInteriorPointSolver(QPOptParContactProblemSingleMesh*);
    double MaxStepSize(Vector& , Vector& , Vector& , double);
    double MaxStepSize(Vector& , Vector& , double);
    void Mult(const BlockVector& , BlockVector&);
    void Mult(const Vector&, Vector &); 
    void FormIPNewtonMat(BlockVector& , Vector& , Vector& , BlockOperator &);
    void IPNewtonSolve(BlockVector& , Vector& , Vector& , Vector&, BlockVector& , double, bool);
    void lineSearch(BlockVector& , BlockVector& , double);
    void projectZ(const Vector & , Vector &, double);
    void filterCheck(double, double);
    double E(const BlockVector &, const Vector &, const Vector &, double, bool);
    double E(const BlockVector &, const Vector &, const Vector &, bool);
    bool GetConverged() const;
    Array<int> & GetCGIterNumbers() {return cgnum_iterations;}
    int GetNumIterations() {return iter;}
    // TO DO: include Hessian of Lagrangian
    double theta(const BlockVector &);
    double phi(const BlockVector &, double);
    void Dxphi(const BlockVector &, double, BlockVector &);
    double L(const BlockVector &, const Vector &, const Vector &);
    void DxL(const BlockVector &, const Vector &, const Vector &, BlockVector &);
    void SetTol(double);
    void SetMaxIter(int);
    void SetBarrierParameter(double);    
    void SaveLogBarrierHessianIterates(bool);
    void SetLinearSolver(int);
    void SetLinearSolveTol(double);
    void SetLinearSolveRelaxType(int);
    void SetFiniteElementSpace(ParFiniteElementSpace * pfes_)
    {
        pfes = pfes_;
    }
    virtual ~ParInteriorPointSolver();
};

#endif
