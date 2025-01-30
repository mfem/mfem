#ifndef PARIPSOLVER 
#define PARIPSOLVER

#include "mfem.hpp"
#include "Problem.hpp"
#include <fstream>
#include <iostream>

// using namespace std;
// using namespace mfem;

namespace mfem {

class ParInteriorPointSolver
{
protected:
    ParGeneralOptProblem* problem;
    double OptTol;
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
    int dimUGlb, dimMGlb, dimCGlb;
    Array<int> block_offsetsumlz, block_offsetsuml, block_offsetsx;
    Vector ml;

    Vector ckSoc;
    HypreParMatrix * Huu, * Hum, * Hmu, * Hmm, * Wmm, *D, * Ju, * Jm, * JuT, * JmT;
    
    int jOpt;
    bool converged;
    
    int MyRank;
    bool iAmRoot;

    bool saveLogBarrierIterates;
    bool saveIterates;
    int linSolver;
    double linSolveTol;
public:
    ParInteriorPointSolver(ParGeneralOptProblem*);
    double MaxStepSize(Vector& , Vector& , Vector& , double);
    double MaxStepSize(Vector& , Vector& , double);
    void Mult(const BlockVector& , BlockVector&);
    void Mult(const Vector&, Vector &); 
    void GetLagrangeMultiplier(Vector &);
    void FormIPNewtonMat(BlockVector& , Vector& , Vector& , BlockOperator &);
    void IPNewtonSolve(BlockVector& , Vector& , Vector& , Vector&, BlockVector& , bool &, double, bool);
    void lineSearch(BlockVector& , BlockVector& , double);
    void projectZ(const Vector & , Vector &, double);
    void filterCheck(double, double);
    double E(const BlockVector &, const Vector &, const Vector &, double, bool);
    double E(const BlockVector &, const Vector &, const Vector &, bool);
    bool GetConverged() const;
    double theta(const BlockVector &);
    double phi(const BlockVector &, double);
    void Dxphi(const BlockVector &, double, BlockVector &);
    double L(const BlockVector &, const Vector &, const Vector &);
    void DxL(const BlockVector &, const Vector &, const Vector &, BlockVector &);
    void SetTol(double);
    void SetMaxIter(int);
    void SetBarrierParameter(double);    
    void SaveIterates(bool);
    void SetLinearSolver(int);
    void SetLinearSolveTol(double);
    void FeasibilityRestoration(const BlockVector &, const Vector &, const Vector &, BlockVector &, double); 
    virtual ~ParInteriorPointSolver();
};

}

#endif
