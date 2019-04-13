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

#include "sundials.hpp"

#ifdef MFEM_USE_SUNDIALS

#include "solvers.hpp"
#ifdef MFEM_USE_MPI
#include "hypre.hpp"
#endif

#include <sundials/sundials_config.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_linearsolver.h>
#include <nvector/nvector_serial.h>
#ifdef MFEM_USE_MPI
#include <nvector/nvector_parallel.h>
#endif

#include <cvode/cvode.h>
#include <sunlinsol/sunlinsol_spgmr.h>

using namespace std;

namespace mfem
{
  // ---------------------------------------------------------------------------
  // SUNMatrix interface functions
  // ---------------------------------------------------------------------------

  // Access the wrapped object in the SUNMatrix
  static inline SundialsODELinearSolver *GetObj(SUNMatrix A)
  {
    return (SundialsODELinearSolver *)(A->content);
  }

  // Return the matrix ID
  SUNMatrix_ID SUNMatGetID(SUNMatrix A)
  {
    return(SUNMATRIX_CUSTOM);
  }

  // Create an empty SUNMatrix <<<<<<< NEED TO ADD
  static SUNMatrix SUNMatEmpty()
  {
    SUNMatrix     A;
    SUNMatrix_Ops ops;

    /* create linear solver */
    A = NULL;
    A = (SUNMatrix) malloc(sizeof *A);
    if (A == NULL) return(NULL);

    /* create ops structure */
    ops = NULL;
    ops = (SUNMatrix_Ops) malloc(sizeof *ops);
    if (ops == NULL) return(NULL);

    /* initialize operations to NULL */
    ops->getid       = NULL;
    ops->clone       = NULL;
    ops->destroy     = NULL;
    ops->zero        = NULL;
    ops->copy        = NULL;
    ops->scaleadd    = NULL;
    ops->scaleaddi   = NULL;
    ops->matvecsetup = NULL;
    ops->matvec      = NULL;
    ops->space       = NULL;

    /* attach ops and initialize content to NULL */
    A->ops     = ops;
    A->content = NULL;

    return(A);
  }

  // ---------------------------------------------------------------------------
  // SUNLinearSolver interface functions
  // ---------------------------------------------------------------------------

  // Access wrapped object in the SUNLinearSolver
  static inline SundialsODELinearSolver *GetObj(SUNLinearSolver LS)
  {
    return (SundialsODELinearSolver *)(LS->content);
  }

  // Return the linear solver type
  static SUNLinearSolver_Type SUNLSGetType(SUNLinearSolver LS)
  {
    return(SUNLINEARSOLVER_MATRIX_ITERATIVE);
  }

  // Initialize the linear solver
  static int SUNLSInit(SUNLinearSolver LS)
  {
    return(GetObj(LS)->LSInit());
  }

  // Setup the linear solver
  static int SUNLSSetup(SUNLinearSolver LS, SUNMatrix A)
  {
    return(GetObj(LS)->LSSetup());
  }

  // Solve the linear system A x = b
  static int SUNLSSolve(SUNLinearSolver LS, SUNMatrix A, N_Vector x, N_Vector b,
                        realtype tol)
  {
    Vector mfem_x(x);
    const Vector mfem_b(b);
    return(GetObj(LS)->LSSolve(mfem_x, mfem_b));
  }

  // Create and empty SUNLinearSolver <<<<<<< NEED TO ADD
  static SUNLinearSolver SUNLinSolEmpty()
  {
    SUNLinearSolver     LS;
    SUNLinearSolver_Ops ops;

    /* create linear solver */
    LS = NULL;
    LS = (SUNLinearSolver) malloc(sizeof *LS);
    if (LS == NULL) return(NULL);

    /* create ops structure */
    ops = NULL;
    ops = (SUNLinearSolver_Ops) malloc(sizeof *ops);
    if (ops == NULL) return(NULL);

    /* initialize operations to NULL */
    ops->gettype           = NULL;
    ops->setatimes         = NULL;
    ops->setpreconditioner = NULL;
    ops->setscalingvectors = NULL;
    ops->initialize        = NULL;
    ops->setup             = NULL;
    ops->solve             = NULL;
    ops->numiters          = NULL;
    ops->resnorm           = NULL;
    ops->resid             = NULL;
    ops->lastflag          = NULL;
    ops->space             = NULL;
    ops->free              = NULL;

    /* attach ops and initialize content to NULL */
    LS->ops     = ops;
    LS->content = NULL;

    return(LS);
  }

  // ---------------------------------------------------------------------------
  // Wrappers for evaluating the ODE linear system
  // ---------------------------------------------------------------------------

  static int cvLinSysSetup(realtype t, N_Vector y, N_Vector fy, SUNMatrix A,
                           booleantype jok, booleantype *jcur, realtype gamma,
                           void *user_data, N_Vector tmp1, N_Vector tmp2,
                           N_Vector tmp3)
  {
    // Get data from N_Vectors
    const Vector mfem_y(y);
    const Vector mfem_fy(fy);

    // Compute the linear system
    return(GetObj(A)->ODELinSys(t, mfem_y, mfem_fy, jok, jcur, gamma));
  }

  static int arkLinSysSetup(realtype t, N_Vector y, N_Vector fy, SUNMatrix A,
                            SUNMatrix M, booleantype jok, booleantype *jcur,
                            realtype gamma, void *user_data, N_Vector tmp1,
                            N_Vector tmp2, N_Vector tmp3)
  {
    // Get data from N_Vectors
    const Vector mfem_y(y);
    Vector mfem_fy(fy);

    // Compute the linear system
    return(GetObj(A)->ODELinSys(t, mfem_y, mfem_fy, jok, jcur, gamma));
  }

  static int arkMassSysSetup(realtype t, SUNMatrix M, void *user_data,
                             N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
  {
    // Compute the mass matrix linear system
    return(GetObj(M)->ODEMassSys(t));
  }

  // ---------------------------------------------------------------------------
  // Wrapper for evaluating the ODE right-hand side
  // ---------------------------------------------------------------------------

  int SundialsODESolver::ODERhs(realtype t, const N_Vector y, N_Vector ydot,
                                void *user_data)
  {
    // Get data from N_Vectors
    const Vector mfem_y(y);
    Vector mfem_ydot(ydot);

    // Get TimeDependentOperator
    TimeDependentOperator *f = (TimeDependentOperator *)user_data;

    // Compute y' = f1(t, y)
    f->SetTime(t);
    f->Mult(mfem_y, mfem_ydot);

    // Return success
    return(0);
  }

  // ---------------------------------------------------------------------------
  // CVODE interface
  // ---------------------------------------------------------------------------

  CVODESolver::CVODESolver(int lmm)
  {
    // Create the solver memory
    sundials_mem = CVodeCreate(lmm);
    MFEM_VERIFY(sundials_mem, "error in CVodeCreate()");

    // Allocate an empty serial N_Vector
    y = N_VNewEmpty_Serial(0);
    MFEM_VERIFY(y, "error in N_VNewEmpty_Serial()");
  }

#ifdef MFEM_USE_MPI
  CVODESolver::CVODESolver(MPI_Comm comm, int lmm)
  {
    // Create the solver memory
    sundials_mem = CVodeCreate(lmm);
    MFEM_VERIFY(sundials_mem, "error in CVodeCreate()");

    if (comm == MPI_COMM_NULL) {

      // Allocate an empty serial N_Vector
      y = N_VNewEmpty_Serial(0);
      MFEM_VERIFY(y, "error in N_VNewEmpty_Serial()");

    } else {

      // Allocate an empty parallel N_Vector
      y = N_VNewEmpty_Parallel(comm, 0, 0);  // calls MPI_Allreduce()
      MFEM_VERIFY(y, "error in N_VNewEmpty_Parallel()");

    }
  }
#endif

  void CVODESolver::Init(TimeDependentOperator &f_)
  {
    mfem_error("CVODE Initialization error: use the initialization method\n"
      "CVODESolver::Init(TimeDependentOperator &f_, double &t, Vector &x)\n");
  }

  void CVODESolver::Init(TimeDependentOperator &f_, double &t, Vector &x)
  {
    // Check intputs for consistency
    int loc_size = f_.Height();
    MFEM_VERIFY(loc_size == x.Size(),
                "error inconsistent operator and vector size");

    MFEM_VERIFY(f_.GetTime() == t,
                "error inconsistent initial times");

    // Initialize the base class
    ODESolver::Init(f_);

    // Fill N_Vector wrapper with initial condition data
    if (!Parallel()) {
      NV_LENGTH_S(y) = x.Size();
      NV_DATA_S(y)   = x.GetData();
    } else {
#ifdef MFEM_USE_MPI
      long local_size = loc_size, global_size;
      MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM,
                    NV_COMM_P(y));
      NV_LOCLENGTH_P(y)  = x.Size();
      NV_GLOBLENGTH_P(y) = global_size;
      NV_DATA_P(y)       = x.GetData();
#endif
    }

    // Initialize CVODE
    flag = CVodeInit(sundials_mem, ODERhs, t, y);
    MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeInit()");

    // Attached the TimeDependentOperator pointer, f, as user-defined data
    flag = CVodeSetUserData(sundials_mem, f);
    MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetUserData()");

    // Set default tolerances
    flag = CVodeSStolerances(sundials_mem, default_rel_tol, default_abs_tol);
    MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetSStolerances()");

    // Set default linear solver (Newton is the default Nonlinear Solver)
    LSA = SUNLinSol_SPGMR(y, PREC_NONE, 0);
    MFEM_VERIFY(LSA, "error in SUNLinSol_SPGMR()");

    flag = CVodeSetLinearSolver(sundials_mem, LSA, NULL);
    MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetLinearSolver()");
  }

  void CVODESolver::SetLinearSolver(SundialsODELinearSolver &ls_spec)
  {
    // Free any existing linear solver
    if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

    // Wrap linear solver as SUNLinearSolver and SUNMatrix
    LSA = SUNLinSolEmpty();
    MFEM_VERIFY(sundials_mem, "error in SUNLinSolEmpty()");

    LSA->content         = &ls_spec;
    LSA->ops->gettype    = SUNLSGetType;
    LSA->ops->initialize = SUNLSInit;
    LSA->ops->setup      = SUNLSSetup;
    LSA->ops->solve      = SUNLSSolve;

    A = SUNMatEmpty();
    MFEM_VERIFY(sundials_mem, "error in SUNMatEmpty()");

    A->content    = &ls_spec;
    A->ops->getid = SUNMatGetID;

    // Attach the linear solver and matrix
    flag = CVodeSetLinearSolver(sundials_mem, LSA, A);
    MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetLinearSolver()");

    // Set the linear system function
    flag = CVodeSetLinSysFn(sundials_mem, cvLinSysSetup);
    MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetLinSysFn()");
  }

  void CVODESolver::Step(Vector &x, double &t, double &dt)
  {
    if (!Parallel()) {
      NV_DATA_S(y) = x.GetData();
      MFEM_VERIFY(NV_LENGTH_S(y) == x.Size(), "");
    } else {
#ifdef MFEM_USE_MPI
      NV_DATA_P(y) = x.GetData();
      MFEM_VERIFY(NV_LOCLENGTH_P(y) == x.Size(), "");
#endif
    }

    // Integrate the system
    double tout = t + dt;
    flag = CVode(sundials_mem, tout, y, &t, step_mode);
    MFEM_VERIFY(flag >= 0, "error in CVode()");

    // Return the last incremental step size
    flag = CVodeGetLastStep(sundials_mem, &dt);
    MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeGetLastStep()");
  }

  void CVODESolver::SetStepMode(int itask)
  {
    step_mode = itask;
  }

  void CVODESolver::PrintInfo() const
  {
    long int nsteps, nfevals, nlinsetups, netfails;
    int      qlast, qcur;
    double   hinused, hlast, hcur, tcur;
    long int nniters, nncfails;
    int      flag = 0;

    // Get integrator stats
    flag = CVodeGetIntegratorStats(sundials_mem,
                                   &nsteps,
                                   &nfevals,
                                   &nlinsetups,
                                   &netfails,
                                   &qlast,
                                   &qcur,
                                   &hinused,
                                   &hlast,
                                   &hcur,
                                   &tcur);
    MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeGetIntegratorStats()");

    // Get nonlinear solver stats
    flag = CVodeGetNonlinSolvStats(sundials_mem,
                                   &nniters,
                                   &nncfails);
    MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeGetNonlinSolvStats()");

    mfem::out <<
      "CVODE:\n"
      "num steps:            " << nsteps << "\n"
      "num rhs evals:        " << nfevals << "\n"
      "num lin setups:       " << nlinsetups << "\n"
      "num nonlin sol iters: " << nniters << "\n"
      "num nonlin conv fail: " << nncfails << "\n"
      "num error test fails: " << netfails << "\n"
      "last order:           " << qlast << "\n"
      "current order:        " << qcur << "\n"
      "initial dt:           " << hinused << "\n"
      "last dt:              " << hlast << "\n"
      "current dt:           " << hcur << "\n"
      "current t:            " << tcur << "\n" << endl;

    return;
  }

  CVODESolver::~CVODESolver()
  {
    N_VDestroy(y);
    SUNMatDestroy(A);
    SUNLinSolFree(LSA);
    SUNNonlinSolFree(NLS);
    CVodeFree(&sundials_mem);
  }

  // ---------------------------------------------------------------------------
  // ARKStep interface
  // ---------------------------------------------------------------------------

  ARKStepSolver::ARKStepSolver(Type type)
    : use_implicit(type == IMPLICIT), irk_table(-1), erk_table(-1)
  {
    // Allocate an empty serial N_Vector
    y = N_VNewEmpty_Serial(0);
    MFEM_VERIFY(y, "error in N_VNewEmpty_Serial()");

    flag = ARK_SUCCESS;
  }

#ifdef MFEM_USE_MPI
  ARKStepSolver::ARKStepSolver(MPI_Comm comm, Type type)
    : use_implicit(type == IMPLICIT), irk_table(-1), erk_table(-1)
  {
    if (comm == MPI_COMM_NULL) {

      // Allocate an empty serial N_Vector
      y = N_VNewEmpty_Serial(0);
      MFEM_VERIFY(y, "error in N_VNewEmpty_Serial()");

    } else {

      // Allocate an empty parallel N_Vector
      y = N_VNewEmpty_Parallel(comm, 0, 0);  // calls MPI_Allreduce()
      MFEM_VERIFY(y, "error in N_VNewEmpty_Parallel()");

    }
  }
#endif

  void ARKStepSolver::Init(TimeDependentOperator &f_)
  {
    mfem_error("ARKStep Initialization error: use the initialization method\n"
      "ARKStepSolver::Init(TimeDependentOperator &f_, double &t, Vector &x)\n");
  }

  void ARKStepSolver::Init(TimeDependentOperator &f_, double &t, Vector &x)
  {
    // Check intputs for consistency
    int loc_size = f_.Height();
    MFEM_VERIFY(loc_size == x.Size(),
                "error inconsistent operator and vector size");

    MFEM_VERIFY(f_.GetTime() == t,
                "error inconsistent initial times");

    // Initialize the base class
    ODESolver::Init(f_);

    // Fill N_Vector wrapper with initial condition data
    if (!Parallel()) {
      NV_LENGTH_S(y) = x.Size();
      NV_DATA_S(y)   = x.GetData();
    } else {
#ifdef MFEM_USE_MPI
      long local_size = loc_size, global_size;
      MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM,
                    NV_COMM_P(y));
      NV_LOCLENGTH_P(y)  = x.Size();
      NV_GLOBLENGTH_P(y) = global_size;
      NV_DATA_P(y)       = x.GetData();
#endif
    }

    // Initialize ARKStep
    if (use_implicit)
      sundials_mem = ARKStepCreate(NULL, ODERhs, t, y);
    else
      sundials_mem = ARKStepCreate(ODERhs, NULL, t, y);
    MFEM_VERIFY(sundials_mem, "error in ARKStepCreate()");

    // Attached the TimeDependentOperator pointer, f, as user-defined data
    flag = ARKStepSetUserData(sundials_mem, f);
    MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetUserData()");

    // Set default tolerances
    flag = ARKStepSStolerances(sundials_mem, default_rel_tol, default_abs_tol);
    MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetSStolerances()");

    // If implicit, set default linear solver
    if (use_implicit) {
      LSA = SUNLinSol_SPGMR(y, PREC_NONE, 0);
      MFEM_VERIFY(LSA, "error in SUNLinSol_SPGMR()");

      flag = ARKStepSetLinearSolver(sundials_mem, LSA, NULL);
      MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetLinearSolver()");
    }
  }

  void ARKStepSolver::SetLinearSolver(SundialsODELinearSolver &ls_spec)
  {
    // Free any existing matrix and linear solver
    if (A != NULL)   { SUNMatDestroy(A); A = NULL; }
    if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

    // Check for implicit method before attaching
    MFEM_VERIFY(use_implicit,
                "The function is applicable only to implicit time integration.");

    // Wrap linear solver as SUNLinearSolver and SUNMatrix
    LSA = SUNLinSolEmpty();
    MFEM_VERIFY(sundials_mem, "error in SUNLinSolEmpty()");

    LSA->content         = &ls_spec;
    LSA->ops->gettype    = SUNLSGetType;
    LSA->ops->initialize = SUNLSInit;
    LSA->ops->setup      = SUNLSSetup;
    LSA->ops->solve      = SUNLSSolve;

    A = SUNMatEmpty();
    MFEM_VERIFY(sundials_mem, "error in SUNMatEmpty()");

    A->content    = &ls_spec;
    A->ops->getid = SUNMatGetID;

    // Attach the linear solver and matrix
    flag = ARKStepSetLinearSolver(sundials_mem, LSA, A);
    MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetLinearSolver()");

    // Set the linear system function
    flag = ARKStepSetLinSysFn(sundials_mem, arkLinSysSetup);
    MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetLinSysFn()");
  }

  void ARKStepSolver::SetMassLinearSolver(SundialsODELinearSolver &ls_spec,
                                          int tdep)
  {
    // Free any existing matrix and linear solver
    if (M != NULL)   { SUNMatDestroy(M); A = NULL; }
    if (LSM != NULL) { SUNLinSolFree(LSM); LSA = NULL; }

    // Check for implicit method before attaching
    MFEM_VERIFY(use_implicit,
                "The function is applicable only to implicit time integration.");

    // Wrap linear solver as SUNLinearSolver and SUNMatrix
    LSM = SUNLinSolEmpty();
    MFEM_VERIFY(sundials_mem, "error in SUNLinSolEmpty()");

    LSM->content         = &ls_spec;
    LSM->ops->gettype    = SUNLSGetType;
    LSM->ops->initialize = SUNLSInit;
    LSM->ops->setup      = SUNLSSetup;
    LSM->ops->solve      = SUNLSSolve;

    M = SUNMatEmpty();
    MFEM_VERIFY(sundials_mem, "error in SUNMatEmpty()");

    M->content    = &ls_spec;
    M->ops->getid = SUNMatGetID;

    // Attach the linear solver and matrix
    flag = ARKStepSetMassLinearSolver(sundials_mem, LSM, M, tdep);
    MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetLinearSolver()");

    // Set the linear system function
    flag = ARKStepSetMassFn(sundials_mem, arkMassSysSetup);
    MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetMassFn()");
  }

  void ARKStepSolver::Step(Vector &x, double &t, double &dt)
  {
    if (!Parallel()) {
      NV_DATA_S(y) = x.GetData();
      MFEM_VERIFY(NV_LENGTH_S(y) == x.Size(), "");
    } else {
#ifdef MFEM_USE_MPI
      NV_DATA_P(y) = x.GetData();
      MFEM_VERIFY(NV_LOCLENGTH_P(y) == x.Size(), "");
#endif
    }

    // Integrate the system
    double tout = t + dt;
    flag = ARKStepEvolve(sundials_mem, tout, y, &t, step_mode);
    MFEM_VERIFY(flag >= 0, "error in ARKStepEvolve()");

    // Return the last incremental step size
    flag = ARKStepGetLastStep(sundials_mem, &dt);
    MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepGetLastStep()");
  }

  void ARKStepSolver::SetStepMode(int itask)
  {
    step_mode = itask;
  }

  void ARKStepSolver::PrintInfo() const
  {
    long int nsteps, expsteps, accsteps, step_attempts;
    long int nfe_evals, nfi_evals;
    long int nlinsetups, netfails;
    double   hinused, hlast, hcur, tcur;
    long int nniters, nncfails;
    int      flag = 0;

    // Get integrator stats

    flag = ARKStepGetTimestepperStats(sundials_mem,
                                      &expsteps,
                                      &accsteps,
                                      &step_attempts,
                                      &nfe_evals,
                                      &nfi_evals,
                                      &nlinsetups,
                                      &netfails);
    MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepGetTimestepperStats()");

    flag = ARKStepGetStepStats(sundials_mem,
                               &nsteps,
                               &hinused,
                               &hlast,
                               &hcur,
                               &tcur);

    // Get nonlinear solver stats
    flag = ARKStepGetNonlinSolvStats(sundials_mem,
                                     &nniters,
                                     &nncfails);
    MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepGetNonlinSolvStats()");

    mfem::out <<
      "ARKStep:\n"
      "num steps:                 " << nsteps << "\n"
      "num exp rhs evals:         " << nfe_evals << "\n"
      "num imp rhs evals:         " << nfi_evals << "\n"
      "num lin setups:            " << nlinsetups << "\n"
      "num nonlin sol iters:      " << nniters << "\n"
      "num nonlin conv fail:      " << nncfails << "\n"
      "num steps attempted:       " << step_attempts << "\n"
      "num acc limited steps:     " << accsteps << "\n"
      "num exp limited stepfails: " << expsteps << "\n"
      "num error test fails:      " << netfails << "\n"
      "initial dt:                " << hinused << "\n"
      "last dt:                   " << hlast << "\n"
      "current dt:                " << hcur << "\n"
      "current t:                 " << tcur << "\n" << endl;

    return;
  }

  ARKStepSolver::~ARKStepSolver()
  {
    N_VDestroy(y);
    SUNMatDestroy(A);
    SUNLinSolFree(LSA);
    SUNNonlinSolFree(NLS);
    ARKStepFree(&sundials_mem);
  }

} // namespace mfem

#endif
