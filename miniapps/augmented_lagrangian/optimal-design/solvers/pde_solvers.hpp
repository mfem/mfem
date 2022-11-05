
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


//  - ∇ ⋅(κ ∇ u) = f  in Ω,  
class DiffusionSolver 
{
private:
    Mesh * mesh = nullptr;
    int order = 1;
    Coefficient * diffcf = nullptr;
    Coefficient * masscf = nullptr;
    Coefficient * rhscf = nullptr;
    Coefficient * essbdr_cf = nullptr;
    Coefficient * neumann_cf = nullptr;
    VectorCoefficient * gradient_cf = nullptr;

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
    ParMesh * pmesh = nullptr;
    ParFiniteElementSpace * pfes = nullptr;
#endif

public:
    DiffusionSolver() { }
    DiffusionSolver(Mesh * mesh_, int order_, Coefficient * diffcf_, Coefficient * cf_);

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
    void SetMassCoefficient(Coefficient * masscf_) { masscf = masscf_; }
    void SetRHSCoefficient(Coefficient * rhscf_) { rhscf = rhscf_; }
    void SetEssentialBoundary(const Array<int> & ess_bdr_){ ess_bdr = ess_bdr_;};
    void SetNeumannBoundary(const Array<int> & neumann_bdr_){ neumann_bdr = neumann_bdr_;};
    void SetNeumannData(Coefficient * neumann_cf_) {neumann_cf = neumann_cf_;}
    void SetEssBdrData(Coefficient * essbdr_cf_) {essbdr_cf = essbdr_cf_;}
    void SetGradientData(VectorCoefficient * gradient_cf_) {gradient_cf = gradient_cf_;}

    void ResetFEM();
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

    ~DiffusionSolver();

};


//  Class for solving the following fractional PDE with MFEM:
//
//  -∇ ⋅ (σ(u)) u = f  in Ω,   0 < α <= 1
//          + BC ...
// σ(u) = λ ∇⋅u I + μ (∇ u + (∇u)^T)
class LinearElasticitySolver 
{
private:
    Mesh * mesh = nullptr;
    int order = 1;
    Coefficient * lambda_cf = nullptr;
    Coefficient * mu_cf = nullptr;
    VectorCoefficient * essbdr_cf = nullptr;
    VectorCoefficient * rhs_cf = nullptr;

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
    ParMesh * pmesh = nullptr;
    ParFiniteElementSpace * pfes = nullptr;
#endif

public:
    LinearElasticitySolver() { }
    LinearElasticitySolver(Mesh * mesh_, int order_, 
    Coefficient * lambda_cf_, Coefficient * mu_cf_);

    void SetMesh(Mesh * mesh_) 
    { 
        mesh = mesh_; 
#ifdef MFEM_USE_MPI
        pmesh = dynamic_cast<ParMesh *>(mesh);
        if (pmesh) { parallel = true; }
#endif    
    }
    void SetOrder(int order_) { order = order_ ; }
    void SetLameCoefficients(Coefficient * lambda_cf_, Coefficient * mu_cf_) { lambda_cf = lambda_cf_; mu_cf = mu_cf_;  }
    void SetRHSCoefficient(VectorCoefficient * rhs_cf_) { rhs_cf = rhs_cf_; }
    void SetEssentialBoundary(const Array<int> & ess_bdr_){ ess_bdr = ess_bdr_;};
    void SetNeumannBoundary(const Array<int> & neumann_bdr_){ neumann_bdr = neumann_bdr_;};
    void SetEssBdrData(VectorCoefficient * essbdr_cf_) {essbdr_cf = essbdr_cf_;}

    void ResetFEM();
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

    ~LinearElasticitySolver();

};