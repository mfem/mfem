#include "mfem.hpp"
#include "Problems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


#ifndef IPSOLVER 
#define IPSOLVER

class InteriorPointSolver
{
protected:
    GeneralOptProblem* optProblem;
    double rel_tol;
    int  max_iter;
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
    Array<int> block_offsetsumlz, block_offsetsuml, block_offsetsx;
    Vector ml;

    Vector ckSoc;
    SparseMatrix * Huu, * Hum, * Hmu, * Hmm, * Wmm, *D, * Ju, * Jm, * JuT, * JmT;
    SparseMatrix * Huucl, *HLuu;    
    int jOpt;
    bool converged;
    
    int MyRank;
    bool iAmRoot;

    bool saveLogBarrierIterates;

    int linSolver;

public:
    InteriorPointSolver(GeneralOptProblem*);
    void Mult(const BlockVector& , BlockVector&); // used when the user wants to be aware of bound-constrained variable m >= ml
    void Mult(const Vector&, Vector &); // useful when the user doesn't need to know about bound-constrained variable   m >= ml, e.g., when m is a slack variable
    double MaxStepSize(Vector& , Vector& , Vector& , double);
    double MaxStepSize(Vector& , Vector& , double);
    void FormIPNewtonMat(BlockVector& , Vector& , Vector& , BlockOperator &);
    void IPNewtonSolve(BlockVector& , Vector& , Vector& , Vector&, BlockVector& , double, bool);
    void lineSearch(BlockVector& , BlockVector& , double);
    void projectZ(const Vector & , Vector &, double);
    void filterCheck(double, double);
    double E(const BlockVector &, const Vector &, const Vector &, double, bool);
    double E(const BlockVector &, const Vector &, const Vector &, bool);
    bool GetConverged() const;
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
    virtual ~InteriorPointSolver();
};

#endif
