#ifndef MFEM_NAVIER_SOLVER_GCN_HPP
#define MFEM_NAVIER_SOLVER_GCN_HPP

#define NAVIER_VERSION 0.1

#include "mfem.hpp"

namespace mfem {


class NavierSolverGCN
{
public:
    NavierSolverGCN(ParMesh* mesh, int order, std::shared_ptr<Coefficient> visc);


    ~NavierSolverGCN();

    /// Initialize forms, solvers and preconditioners.
    void Setup(real_t dt);

    /// Compute the solution at next time step t+dt
    void Step(real_t &time, real_t dt, int cur_step, bool provisional = false);

    /// Return the provisional velocity ParGridFunction.
    ParGridFunction& GetProvisionalVelocity() { return pvel; }

    /// Return the current velocity ParGridFunction.
    ParGridFunction& GetCurrentVelocity() { return cvel; }

    /// Return the current pressure ParGridFunction.
    ParGridFunction& GetCurrentPressure() { return pres; }

    /// Set CN theta coefficients
    void SetTheta(real_t t1=real_t(0.5),real_t t2=real_t(0.5),
                  real_t t3=real_t(0.5),real_t t4=real_t(0.5))
    {
        thet1=t1;
        thet2=t2;
        thet3=t3;
        thet4=t4;
    }

private:

    real_t thet1,thet2,thet3,thet4;


    /// Enable/disable debug output.
    bool debug = false;

    /// Enable/disable verbose output.
    bool verbose = true;

    /// Enable/disable partial assembly of forms.
    bool partial_assembly = false;

    /// The parallel mesh.
    ParMesh *pmesh = nullptr;

    /// The order of the velocity and pressure space.
    int order;

    std::shared_ptr<Coefficient> visc;

    ParGridFunction nvel; //next velocity
    ParGridFunction pvel; //previous velocity
    ParGridFunction cvel; //current velocity
    ParGridFunction pres; //current pressure


    std::unique_ptr<H1_FECollection> vfec;
    std::unique_ptr<H1_FECollection> pfec;
    std::unique_ptr<ParFiniteElementSpace> vfes;
    std::unique_ptr<ParFiniteElementSpace> pfes;



};//end NavierSolverGCN



}//end namespace mfem




#endif
