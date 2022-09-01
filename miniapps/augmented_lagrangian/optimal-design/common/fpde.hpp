
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


//  Class for solving the following fractional PDE with MFEM:
//
//  ( - Δ^α ) u = f  in Ω,   0 < α <= 1
//            u = 0  on ∂Ω,      
class FPDESolver 
{
private:
    Mesh * mesh = nullptr;
    int order = 1;
    Coefficient * diffcf = nullptr;
    Coefficient * cf = nullptr;
    Coefficient * essbdr_cf = nullptr;
    Coefficient * neumann_cf = nullptr;
    VectorCoefficient * gradient_cf = nullptr;
    double alpha = 1.0; 
    double beta = 0.0; 

    Array<double> coeffs, poles;
    // FEM solver
    int dim;
    FiniteElementCollection * fec = nullptr;    
    FiniteElementSpace * fes = nullptr;    
    Array<int> ess_bdr;
    Array<int> neumann_bdr;
    GridFunction * u = nullptr;
    LinearForm * b = nullptr;
    bool parallel = false;
#ifdef MFEM_USE_MPI
    MPI_Comm comm;
    ParMesh * pmesh = nullptr;
    ParFiniteElementSpace * pfes = nullptr;
#endif

public:
    FPDESolver() { }
    FPDESolver(Mesh * mesh_, int order_, Coefficient * diffcf_, Coefficient * cf_, 
               double alpha_, double beta_);

    void SetMesh(Mesh * mesh_) 
    { 
        mesh = mesh_; 
#ifdef MFEM_USE_MPI
        pmesh = dynamic_cast<ParMesh *>(mesh);
        if (pmesh) { parallel = true; }
#endif    
    }
    void SetOrder(int order_) { order = order_ ; }
    void SetDiffusionCoefficient(Coefficient * diffcf_) { diffcf = diffcf_; }
    void SetRHSCoefficient(Coefficient * cf_) { cf = cf_; }
    void SetEssentialBoundary(const Array<int> & ess_bdr_){ ess_bdr = ess_bdr_;};
    void SetNeumannBoundary(const Array<int> & neumann_bdr_){ neumann_bdr = neumann_bdr_;};
    void SetNeumannData(Coefficient * neumann_cf_) {neumann_cf = neumann_cf_;}
    void SetEssBdrData(Coefficient * essbdr_cf_) {essbdr_cf = essbdr_cf_;}
    void SetGradientData(VectorCoefficient * gradient_cf_) {gradient_cf = gradient_cf_;}

    void SetAlpha(double alpha_) { alpha = alpha_; }
    void SetBeta(double beta_) { beta = beta_; }

    void Init();
    void ResetFEM();
    void Reset();
    void SetupFEM();

    void Solve();
    GridFunction * GetFEMSolution();
    LinearForm * GetLinearForm() {return b;}
#ifdef MFEM_USE_MPI
    ParGridFunction * GetParFEMSolution();
    ParLinearForm * GetParLinearForm() 
    {
        if (parallel)
        {
           return dynamic_cast<ParLinearForm *>(b);
        }
        else
        {
            MFEM_ABORT("Wrong code path. Call GetLinearForm");
            return nullptr;
        }
    }
#endif
    void VisualizeSolution();

    ~FPDESolver();

};
