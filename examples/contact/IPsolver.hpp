#include "mfem.hpp"
#include "problems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


#ifndef IPSOLVER 
#define IPSOLVER

class InteriorPointSolver
{
protected:
    OptProblem* problem;
    double tol;
    int  max_iter;
    double mu_k; // \mu_k
    Vector lk, zlk, mf;

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
    SparseMatrix * Huu = nullptr; 
    SparseMatrix * Hum = nullptr; 
    SparseMatrix * Hmu = nullptr; 
    SparseMatrix * Hmm = nullptr; 
    SparseMatrix * Wmm = nullptr; 
    SparseMatrix * Ju  = nullptr; 
    SparseMatrix * Jm  = nullptr; 
    SparseMatrix * JuT = nullptr; 
    SparseMatrix * JmT  = nullptr;;
    
    int jOpt;
    bool converged;
    
    int MyRank;
    bool iAmRoot;

    bool saveLogBarrierIterates;

    int linSolver;
    std::ofstream IPNewtonKrylovIters;

    ParFiniteElementSpace *Vh;
    Array<int> cgnum_iterations;


    // not sure if this data is needed or if it can 
    // all be accounted for in the problem class
    // which variables have equality constraints
    //Array<int> eqConstrainedVariables;
    //Array<double> eqConstrainedValues;



public:
    InteriorPointSolver(OptProblem*, ParFiniteElementSpace *);
    void Mult(const BlockVector& , BlockVector&); // used when the user wants to be aware of bound-constrained variable m >= ml
    void Mult(const Vector&, Vector &); // useful when the user doesn't need to know about bound-constrained variable   m >= ml
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
    Vector GetBoundConstrainedVariable() {return mf;}
    Array<int> & GetCGIterNumbers() {return cgnum_iterations;}
    virtual ~InteriorPointSolver();
};

#endif
