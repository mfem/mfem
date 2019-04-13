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
#include <arkode/arkode_arkstep.h>

namespace mfem
{
  // ---------------------------------------------------------------------------
  // Base class for interfacing with SUNMatrix and SUNLinearSolver API
  // ---------------------------------------------------------------------------

  /** Abstract base class for providing custom linear solvers to SUNDIALS ODE
      packages, CVODE and ARKODE. For a given ODE system

      dy/dt = f(y,t) or M dy/dt = f(y,t)

      the purpose of this class is to facilitate the (approximate) solution of
      linear systems of the form

      (I - gamma J) y = b or (M - gamma J) y = b,   J = J(y,t) = df/dy

      and mass matrix systems of the form

      M y = b,   M = M(t)

      for given b, y, t and gamma, where gamma is a scaled time step. */
  class SundialsODELinearSolver
  {
  private:
    TimeDependentOperator *oper;

  protected:
    SundialsODELinearSolver() : oper(NULL) { }
    virtual ~SundialsODELinearSolver();

  public:
    /** Setup the ODE linear system A(y,t) = (I - gamma J) or A = (M - gamma J)
        @param[in]  t     The time at which A(y,t)  should be evaluated
        @param[in]  y     The state at which A(y,t) should be evaluated
        @param[in]  fy    The current value of the ODE Rhs function, f(y,t)
        @param[in]  jok   Flag indicating if the Jacobian should be updated
        @param[out] jcur  Flag to signal if the Jacobian was updated
        @param[in]  gamma The scaled time step value */
    virtual int ODELinSys(double t, Vector y, Vector fy, int jok, int *jcur,
                          double gamma)
    {
      mfem_error("SundialsODELinearSolver::ODELinSys() is not overridden!");
      return(1);
    }

    /** Setup the ODE Mass matrix system M
        @param[in] t The time at which M(t) should be evaluated*/
    virtual int ODEMassSys(double t)
    {
      mfem_error("SundialsODELinearSolver::ODEMassSys() is not overridden!");
      return(1);
    }

    /** Initialize the linear solver (optional) */
    virtual int LSInit() { return(0); };

    /** Setup the linear solver (optional) */
    virtual int LSSetup() { return(0); };

    /** Solve the linear system A x = b
        @param[in/out]  x  On input, the initial guess. On output, the solution
        @param[in]      b  The linear system right-hand side */
    virtual int LSSolve(Vector &x, Vector b) = 0;
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
    SUNMatrix          A;   /// Linear system A = (I - gamma J) or (M - gamma J)
    SUNMatrix          M;   /// Mass matrix M
    SUNLinearSolver    LSA; /// Linear solver for A
    SUNLinearSolver    LSM; /// Linear solver for M
    SUNNonlinearSolver NLS; /// Nonlinear solver

#ifdef MFEM_USE_MPI
    bool Parallel() const
    { return (N_VGetVectorID(y) != SUNDIALS_NVEC_SERIAL); }
#else
    bool Parallel() const { return false; }
#endif

    /// Wrapper to compute the ODE Rhs function
    static int ODERhs(realtype t, const N_Vector y, N_Vector ydot,
                      void *user_data);

    /// Default scalar tolerances
    static constexpr double default_rel_tol = 1e-4;
    static constexpr double default_abs_tol = 1e-9;

    /// Constructors
    SundialsODESolver() : sundials_mem(NULL), flag(0), step_mode(1),
                          y(NULL), A(NULL), M(NULL), LSA(NULL), LSM(NULL),
                          NLS(NULL) { }

    SundialsODESolver(void *mem) : sundials_mem(mem) { }

  public:
    /// Access the SUNDIALS memory structure
    void *GetMem() const { return sundials_mem; }

    /// Returns the last flag retured a call to a SUNDIALS function
    int GetFlag() const { return flag; }
  };

  // ---------------------------------------------------------------------------
  // Interface to the CVODE library -- linear multi-step methods
  // ---------------------------------------------------------------------------

  class CVODESolver : public ODESolver, public SundialsODESolver
  {
  public:
    /** Construct a serial wrapper to SUNDIALS' CVODE integrator
        @param[in] lmm Specifies the linear multistep method, the options are:
                       CV_ADAMS - implicit methods for non-stiff systems
                       CV_BDF   - implicit methods for stiff systems */
    CVODESolver(int lmm = CV_BDF);

#ifdef MFEM_USE_MPI
    /** Construct a parallel wrapper to SUNDIALS' CVODE integrator
        @param[in] comm The MPI communicator used to partition the ODE system
        @param[in] lmm  Specifies the linear multistep method, the options are:
                        CV_ADAMS - implicit methods for non-stiff systems
                        CV_BDF   - implicit methods for stiff systems */
    CVODESolver(MPI_Comm comm, int lmm = CV_BDF);
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

    /** Select the CVODE step mode: CV_NORMAL (default) or CV_ONE_STEP
        @param[in] itask  The desired step mode */
    void SetStepMode(int itask);

    /** Print various CVODE statistics. */
    void PrintInfo() const;

    /// Destroy the associated CVODE memory and SUNDIALS objects
    virtual ~CVODESolver();
  };

  // ---------------------------------------------------------------------------
  // Interface to ARKode's ARKStep module -- Additive Runge-Kutta methods
  // ---------------------------------------------------------------------------

  class ARKStepSolver : public ODESolver, public SundialsODESolver
  {
  protected:
    bool use_implicit;
    int  irk_table, erk_table;

  public:
    /// Types of ARKODE solvers.
    enum Type { EXPLICIT, IMPLICIT };

    /** Construct a serial wrapper to SUNDIALS' ARKode integrator
        @param[in] type Specifies the RK method type
                        EXPLICIT - explicit RK method
                        IMPLICIT - implicit RK method */
    ARKStepSolver(Type type = EXPLICIT);

#ifdef MFEM_USE_MPI
    /** Construct a parallel wrapper to SUNDIALS' ARKode integrator
        @param[in] comm The MPI communicator used to partition the ODE system
        @param[in] type Specifies the RK method type
                        EXPLICIT - explicit RK method
                        IMPLICIT - implicit RK method */
    ARKStepSolver(MPI_Comm comm, Type type = EXPLICIT);
#endif

    /// Base class Init -- DO NOT CALL, use the below initialization function
    /// that takes the initial t and x as inputs.
    virtual void Init(TimeDependentOperator &f_);

    /** Initialize ARKode: Calls ARKStepInit() and sets some defaults.
        @param[in] f_ the TimeDependentOperator that defines the ODE system
        @param[in] t  the initial time
        @param[in] x  the initial condition

        @note All other methods must be called after Init(). */
    void Init(TimeDependentOperator &f_, double &t, Vector &x);

    /** Integrate the ODE with ARKode using the specified step mode.

        @param[out]    x  Solution vector at the requested output timem x=x(t).
        @param[in/out] t  On output, the output time reached.
        @param[in/out] dt On output, the last time step taken.

        @note On input, the values of t and dt are used to compute desired
        output time for the integration, tout = t + dt.
    */
    virtual void Step(Vector &x, double &t, double &dt);

    /** Attach a custom linear solver solver to ARKode
        @param[in] ls_spec A SundialsODELinearSolver object defining the custom
                           linear solver */
    void SetLinearSolver(SundialsODELinearSolver &ls_spec);

    /** Attach a custom mass matrix linear solver solver to ARKode
        @param[in] ls_spec A SundialsODELinearSolver object defining the custom
                           linear solver
        @param[in] tdep    A integer flag indicating if the mass matrix is time
                           dependent (1) or time independent (0). */
    void SetMassLinearSolver(SundialsODELinearSolver &ls_spec, int tdep);

    /** Select the ARKode step mode: ARK_NORMAL (default) or ARK_ONE_STEP
        @param[in] itask  The desired step mode */
    void SetStepMode(int itask);

    /** Print various ARKStep statistics. */
    void PrintInfo() const;

    /// Destroy the associated ARKode memory and SUNDIALS objects
    virtual ~ARKStepSolver();
  };

}  // namespace mfem

#endif // MFEM_USE_SUNDIALS

#endif // MFEM_SUNDIALS
