// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

/*
  Approach 1:
    - Updated Init function to take initial contion as input
    - Setting options must occur after initialization
    - Addition of LinSysSetup functions to setup linear systems
    - Addition of SUNLinSolEmpty() and SUNMatEmpty() functions to make
      creating wrappers to linear solver and matrix easier. Also protects
      against the addition of new optional operations to the APIs.
    - Simplified user-supplied methods for custom linear solvers.
    - Need to add ReInit and ReSize methods.
*/

#ifndef MFEM_SUNDIALS
#define MFEM_SUNDIALS

#include "../config/config.hpp"

#ifdef MFEM_USE_SUNDIALS

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include "ode.hpp"
#include "solvers.hpp"

#include <cvode/cvode.h>

namespace mfem
{
  // ---------------------------------------------------------------------------
  // Base class for interfacing with SUNMatrix and SUNLinearSolver API
  // ---------------------------------------------------------------------------

  /** Abstract base class for providing custom linear solvers to SUNDIALS ODE
      packages, CVODE and ARKODE. For a given ODE system

      dx/dt = f(x,t) or M dx/dt = f(x,t)

      the purpose of this class is to facilitate the (approximate) solution of
      linear systems of the form

      (I - gamma J) y = b or (M - gamma J) y = b,   J = J(x,t) = df/dx

      for given b, x, t and gamma, where gamma is a scaled time step. */
  class SundialsODELinearSolver
  {
  private:
    TimeDependentOperator *oper;

  protected:
    SundialsODELinearSolver() : oper(NULL) { }
    virtual ~SundialsODELinearSolver();

  public:
    /** Linear system interface method */
    ///@{
    virtual int LinSysSetup(double t, Vector y, Vector fy, int jok, int *jcur,
                            double gamma) = 0;
    ///@}

    /** Linear solver interface methods */
    ///@{
    virtual int LSInit() {return(0)};
    virtual int LSSetup() {return(0)};
    virtual int LSSolve(Vector &x, Vector b) = 0;
    ///@}
  };

  // ---------------------------------------------------------------------------
  // Base class for interfacing with SUNDIALS packages
  // ---------------------------------------------------------------------------

  class SundialsODESolver
  {
  protected:
    void *sundials_mem; /// SUNDIALS mem structure
    mutable int flag;   /// Last flag returned from a call to SUNDIALS
    int step_mode;      /// SUNDIALS step mode (NORMAL or ONE_STEP)

    N_Vector           y;   /// State vector
    SUNMatrix          A;   /// Linear system (I - gamma J) or (M - gamma J)
    SUNLinearSolver    LSA; /// Linear solver for A
    SUNNonlinearSolver NLS; /// Nonlinear solver

    /// Wrapper to compute the ODE Rhs function
    static int ODERhs(realtype t, const N_Vector y, N_Vector ydot,
                      void *user_data);

    /// Default scalar tolerances
    const double default_rel_tol = 1e-4;
    const double default_abs_tol = 1e-9;

    /// Constructors
    SundialsSolver() : sundials_mem(NULL), flag(0), step_mode(CV_NORMAL),
                       y(NULL), A(NULL), LSA(NULL), NLS(NULL) { }

    SundialsSolver(void *mem) : sundials_mem(mem) { }

  public:
    /// Access the SUNDIALS memory structure
    void *GetMem() const { return sundials_mem; }

    /// Returns the last flag retured a call to a SUNDIALS function
    int GetFlag() const { return flag; }
  };

  // ---------------------------------------------------------------------------
  // Interface to SUNDIALS' CVODE library -- Multi-step time integration
  // ---------------------------------------------------------------------------

  class CVODESolver : public ODESolver, public SundialsODESolver
  {
  public:
    /** Construct a serial wrapper to SUNDIALS' CVODE integrator
        @param[in] lmm Specifies the linear multistep method, the options are:
                       CV_ADAMS - implicit methods for non-stiff systems
                       CV_BDF   - implicit methods for stiff systems */
    CVODESolver(int lmm);

#ifdef MFEM_USE_MPI
    /** Construct a parallel wrapper to SUNDIALS' CVODE integrator
        @param[in] comm The MPI communicator used to partition the ODE system
        @param[in] lmm  Specifies the linear multistep method, the options are:
                        CV_ADAMS - implicit methods for non-stiff systems
                        CV_BDF   - implicit methods for stiff systems */
    CVODESolver(MPI_Comm comm, int lmm);
#endif

    /// Base class Init -- DO NOT CALL, use the below initialization function
    /// that takes the initial t and x as inputs.
    virtual void Init(TimeDependentOperator &f_);

    /** Initialize CVODE: Calls CVodeInit() and sets some defaults.
        @param[in] f_ the TimeDependentOperator that defines the ODE system
        @param[in] t  the initial time
        @param[in] x  the initial condition

        @note All other methods must be called after Init(). */
    void Init(TimeDependentOperator &f_, double &t, Vector &x);

    /** Integrate the ODE with CVODE using the specified step mode.

        @param[out]    x  Solution vector at the requested output timem x=x(t).
        @param[in/out] t  On output, the output time reached.
        @param[in/out] dt On output, the last time step taken.

        @note On input, the values of t and dt are used to compute desired
        output time for the integration, tout = t + dt.
    */
    virtual void Step(Vector &x, double &t, double &dt);

    /** Attach a custom linear solver solver to CVODE
        @param[in] ls_spec A SundialsODELinearSolver object defining the custom
                           linear solver */
    void SetLinearSolver(SundialsODELinearSolver &ls_spec);

    /** Select the CVode step mode: CV_NORMAL (default) or CV_ONE_STEP
        @param[in] itask  The desired step mode */
    void SetStepMode(int itask);

    /// Destroy the associated CVODE memory and SUNDIALS objects
    virtual ~CVODESolver();
  };

}  // namespace mfem

#endif // MFEM_USE_SUNDIALS

#endif // MFEM_SUNDIALS
