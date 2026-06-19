#pragma once

#include "mfem.hpp"
#include <cmath>

using namespace std;
using namespace mfem;

class QuantityOfInterest
{
protected:

public:
   QuantityOfInterest() { }
   ~QuantityOfInterest() { }
   
   virtual real_t Eval() { return 0; };
};

// Evaluating the compliance: c = int_Ω E_e : ɛ(u) : ɛ(u) dx
class Compliance : public QuantityOfInterest
{
protected:
    ParFiniteElementSpace *fes;
    MPI_Comm comm;
    ProductCoefficient uku_cf;        // E_e : ɛ(u) : ɛ(u) = r(rho~) * psi0(u)

public:
    Compliance(MPI_Comm comm_, ParFiniteElementSpace *fes_,
               Coefficient &simp_cf_, Coefficient &energy_cf_)
        : fes(fes_), comm(comm_), uku_cf(simp_cf_, energy_cf_) { }

    real_t Eval() override
    {
        ParLinearForm lf(fes);
        lf.AddDomainIntegrator(new DomainLFIntegrator(uku_cf));
        lf.Assemble();
        std::unique_ptr<HypreParVector> v(lf.ParallelAssemble());

        real_t loc, val;
        loc = v->Sum();
        MPI_Allreduce(&loc, &val, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
        return val;
    }
};

// Max-length constraint  G = 1/2 ∫_Ω (γ − α)² dx  
class MaxFilterResidual : public QuantityOfInterest
{
protected:
    ParFiniteElementSpace *fes;
    MPI_Comm comm;

    GridFunctionCoefficient gamma_cf, alpha_cf;
    SumCoefficient     diff_cf;          //  γ − α  
    ProductCoefficient diff2_cf;         // (γ − α)²

public:
    MaxFilterResidual(MPI_Comm comm_, ParGridFunction &gamma_, ParGridFunction &alpha_)
    : fes(alpha_.ParFESpace()), comm(comm_),
      gamma_cf(&gamma_), alpha_cf(&alpha_),
      diff_cf(gamma_cf, alpha_cf, 1.0, -1.0),
      diff2_cf(diff_cf, diff_cf) { }
    ~MaxFilterResidual() { }

    // return coefficient evaluating (γ − α) 
    Coefficient *GetResidualCoefficient() { return &diff_cf; }

    // G = 1/2 ∫_Ω (γ − α)² dx
    real_t Eval() override
    {
        ParLinearForm lf(fes);
        lf.AddDomainIntegrator(new DomainLFIntegrator(diff2_cf));
        lf.Assemble();
        std::unique_ptr<HypreParVector> v(lf.ParallelAssemble());   // ∫_Ω (γ−α)² 

        real_t loc, val;
        loc = v->Sum();
        MPI_Allreduce(&loc, &val, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
        return 0.5 * val;
    }

    // dG/dρ = (s, ·)_L2  (s = max-filter adjoint soln);  dG/dα = (α − γ, ·)_L2.
    void GetGrad(ParGridFunction &s_filter, Vector &dGdrho, Vector &dGdalpha)
    {
        GridFunctionCoefficient s_cf(&s_filter);
        ParLinearForm lr(fes);
        lr.AddDomainIntegrator(new DomainLFIntegrator(s_cf));
        lr.Assemble();
        HypreParVector *vr = lr.ParallelAssemble();

        dGdrho = *vr;
        delete vr;                 

        ParLinearForm la(fes);
        la.AddDomainIntegrator(new DomainLFIntegrator(diff_cf)); 
        la.Assemble();
        HypreParVector *va = la.ParallelAssemble();

        dGdalpha = *va;  
        dGdalpha.Neg();              // - (γ - α)
        delete va;      
    }
};