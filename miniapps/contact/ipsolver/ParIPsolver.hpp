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
    OptContactProblem* problem = nullptr;
    int numActiveConstraints = -1;
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
    HypreParMatrix * Wuu = nullptr;
    HypreParMatrix * Wmm = nullptr;
    HypreParMatrix * Ju = nullptr;
    HypreParMatrix * Jm = nullptr;
    HypreParMatrix * JuT = nullptr;
    HypreParMatrix * JmT = nullptr;


    double alphaCurvatureTest;
    double deltaRegLast;
    double deltaRegMin;
    double deltaRegMax;
    double deltaReg0;

    double kRegMinus;
    double kRegBarPlus;
    double kRegPlus;


    Array<int> cgnum_iterations;
    Array<int> amg_num_iterations;
    Array<double> dmaxmin_ratio;
    Array<double> jtdj_ratio;
    Array<double> Adiag_ratio;
    bool no_contact_solve = false;
    bool amg_contact_solve = false;
    Array<int> cgnum_iterations_nocontact;
    ParFiniteElementSpace *pfes = nullptr;
    
    int jOpt;
    bool converged;
    
    int MyRank;
    bool iAmRoot;

    bool saveLogBarrierIterates = false;

    int linSolver=0;
    bool dynamicsolver=false;
    int dynamiclinSolver=0;
    double linSolveAbsTol = 1e-12;
    double linSolveRelTol = 1e-6;
    int relax_type = 88;
    bool monitor = false;
    bool save_matrix_data = false;
    int label = -1;
    MPI_Comm comm;
public:
    ParInteriorPointSolver(OptContactProblem*);
    double MaxStepSize(Vector& , Vector& , Vector& , double);
    double MaxStepSize(Vector& , Vector& , double);
    void Mult(const BlockVector& , BlockVector&);
    void Mult(const Vector&, Vector &); 
    void FormIPNewtonMat(BlockVector& , Vector& , Vector& , BlockOperator &, double delta = 0.0);
    void IPNewtonSolve(BlockVector& , Vector& , Vector& , Vector&, BlockVector& , bool &, double, bool, double delta = 0.0);
    void lineSearch(BlockVector& , BlockVector& , double);
    void projectZ(const Vector & , Vector &, double);
    void filterCheck(double, double);
    double E(const BlockVector &, const Vector &, const Vector &, double, bool);
    double E(const BlockVector &, const Vector &, const Vector &, bool);
    bool GetConverged() const;
    Array<int> & GetCGIterNumbers() {return cgnum_iterations;};
    Array<int> & GetAMGIterNumbers() {return amg_num_iterations;};
    Array<double> & GetDMaxMinRatios() {return dmaxmin_ratio;};
    Array<double> & GetJtDJMaxMinRatios() {return jtdj_ratio;};
    Array<double> & GetAdiagMaxMinRatios() {return Adiag_ratio;};
    Array<int> & GetCGNoContactIterNumbers() {return cgnum_iterations_nocontact;};
    int GetNumIterations() {return iter;};
    double theta(const BlockVector &);
    double phi(const BlockVector &, double);
    double phi(const BlockVector &, double, int &);
    void Dxphi(const BlockVector &, double, BlockVector &);
    double L(const BlockVector &, const Vector &, const Vector &);
    void DxL(const BlockVector &, const Vector &, const Vector &, BlockVector &);
    void SetTol(double);
    void SetMaxIter(int);
    void SetBarrierParameter(double);    
    void SaveLogBarrierHessianIterates(bool);
    void SetLinearSolver(int);
    void SetLinearSolveAbsTol(double);
    void SetLinearSolveRelTol(double);
    void SetLinearSolveRelaxType(int);
    void SetElasticityOptions(ParFiniteElementSpace * pfes_)
    {
        pfes = pfes_;
    };
    void EnableDynamicSolverChoice() { dynamicsolver = true;};
    void DisableDynamicSolverChoice() { dynamicsolver = false;};
    bool CurvatureTest(const BlockOperator & A, const BlockVector & Xhat, const Vector &l, const BlockVector & b, const double & delta);
    void EnableMonitor() { monitor = true;};
    void DisableMonitor() { monitor = false;};
    void EnableSaveMatrix() { save_matrix_data = true;};
    void DisableSaveMatrix() { save_matrix_data = false;};
    void EnableNoContactSolve() {no_contact_solve = true;};
    void EnableAMGContactSolve() {amg_contact_solve = true;};
    void SetProblemLabel(int label_) { label = label_;};
    double GetNumActiveConstraints() { return numActiveConstraints;};
    void Clear() 
    {
       F1.DeleteAll();
       F2.DeleteAll();
       mu_k = 1.0;
    };
    virtual ~ParInteriorPointSolver();
};

#endif
