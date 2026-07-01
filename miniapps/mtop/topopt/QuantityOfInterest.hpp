#pragma once

#include "mfem.hpp"
#include <cmath>

using namespace std;
using namespace mfem;

class QuantityOfInterest
{
public:
   QuantityOfInterest() { }
   ~QuantityOfInterest() { }
   
    virtual real_t Eval() { return 0; };
    virtual void GetGrad(Vector &grad) { grad.SetSize(0);  grad = 0.0;  }
    virtual void GetGrad(Vector &grad1, Vector &grad2) {
        grad1.SetSize(0);
        grad2.SetSize(0);

        grad1 = 0.0; grad2 = 0.0;
    }
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
        ~Compliance() { }

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


