// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "sundials.hpp"

#ifdef MFEM_USE_SUNDIALS

#include "solvers.hpp"
#ifdef MFEM_USE_MPI
#include "hypre.hpp"
#endif

// SUNDIALS vectors
#include <nvector/nvector_serial.h>
#ifdef MFEM_USE_MPI
#include <nvector/nvector_parallel.h>
#endif

// SUNDIALS linear solvers
#include <sunlinsol/sunlinsol_spgmr.h>

// Access SUNDIALS object's content pointer
#define GET_CONTENT(X) ( X->content )

using namespace std;

namespace mfem
{

// ---------------------------------------------------------------------------
// SUNMatrix interface functions
// ---------------------------------------------------------------------------

// Return the matrix ID
static SUNMatrix_ID MatGetID(SUNMatrix A)
{
   return (SUNMATRIX_CUSTOM);
}

static void MatDestroy(SUNMatrix A)
{
   if (A->content) { A->content = NULL; }
   if (A->ops) { free(A->ops); A->ops = NULL; }
   free(A); A = NULL;
   return;
}

// ---------------------------------------------------------------------------
// SUNLinearSolver interface functions
// ---------------------------------------------------------------------------

// Return the linear solver type
static SUNLinearSolver_Type LSGetType(SUNLinearSolver LS)
{
   return (SUNLINEARSOLVER_MATRIX_ITERATIVE);
}

static int LSFree(SUNLinearSolver LS)
{
   if (LS->content) { LS->content = NULL; }
   if (LS->ops) { free(LS->ops); LS->ops = NULL; }
   free(LS); LS = NULL;
   return (0);
}

// ---------------------------------------------------------------------------
// CVODE interface
// ---------------------------------------------------------------------------

int CVODESolver::RHS(realtype t, const N_Vector y, N_Vector ydot,
                     void *user_data)
{
   // Get data from N_Vectors
   const Vector mfem_y(y);
   Vector mfem_ydot(ydot);
   CVODESolver *self = static_cast<CVODESolver*>(user_data);

   // Compute y' = f(t, y)
   self->f->SetTime(t);
   self->f->Mult(mfem_y, mfem_ydot);

   // Return success
   return (0);
}

int CVODESolver::LinSysSetup(realtype t, N_Vector y, N_Vector fy, SUNMatrix A,
                             booleantype jok, booleantype *jcur,
                             realtype gamma, void *user_data, N_Vector tmp1,
                             N_Vector tmp2, N_Vector tmp3)
{
   // Get data from N_Vectors
   const Vector mfem_y(y);
   const Vector mfem_fy(fy);
   CVODESolver *self = static_cast<CVODESolver*>(GET_CONTENT(A));

   // Compute the linear system
   self->f->SetTime(t);
   return (self->f->SUNImplicitSetup(mfem_y, mfem_fy, jok, jcur, gamma));
}

int CVODESolver::LinSysSolve(SUNLinearSolver LS, SUNMatrix A, N_Vector x,
                             N_Vector b, realtype tol)
{
   Vector mfem_x(x);
   const Vector mfem_b(b);
   CVODESolver *self = static_cast<CVODESolver*>(GET_CONTENT(LS));

   // Solve the linear system
   return (self->f->SUNImplicitSolve(mfem_b, mfem_x, tol));
}

CVODESolver::CVODESolver(int lmm)
   : lmm_type(lmm), step_mode(CV_NORMAL)
{
   // Allocate an empty serial N_Vector
   y = N_VNewEmpty_Serial(0);
   MFEM_VERIFY(y, "error in N_VNewEmpty_Serial()");
}

#ifdef MFEM_USE_MPI
CVODESolver::CVODESolver(MPI_Comm comm, int lmm)
   : lmm_type(lmm), step_mode(CV_NORMAL)
{
   if (comm == MPI_COMM_NULL)
   {

      // Allocate an empty serial N_Vector
      y = N_VNewEmpty_Serial(0);
      MFEM_VERIFY(y, "error in N_VNewEmpty_Serial()");

   }
   else
   {

      // Allocate an empty parallel N_Vector
      y = N_VNewEmpty_Parallel(comm, 0, 0);  // calls MPI_Allreduce()
      MFEM_VERIFY(y, "error in N_VNewEmpty_Parallel()");

   }
}
#endif

void CVODESolver::Init(TimeDependentOperator &f_)
{
   // Initialize the base class
   ODESolver::Init(f_);

   // Get the vector length
   long local_size = f_.Height();
#ifdef MFEM_USE_MPI
   long global_size;
#endif

   if (Parallel())
   {
#ifdef MFEM_USE_MPI
      MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM,
                    NV_COMM_P(y));
#endif
   }

   // Get current time
   double t = f_.GetTime();

   if (sundials_mem)
   {
      // Check if the problem size has changed since the last Init() call
      int resize = 0;
      if (!Parallel())
      {
         resize = (NV_LENGTH_S(y) != local_size);
      }
      else
      {
#ifdef MFEM_USE_MPI
         int l_resize = (NV_LOCLENGTH_P(y) != local_size) ||
                        (saved_global_size != global_size);
         MPI_Allreduce(&l_resize, &resize, 1, MPI_INT, MPI_LOR, NV_COMM_P(y));
#endif
      }

      // Free existing solver memory and re-create with new vector size
      if (resize)
      {
         CVodeFree(&sundials_mem);
         sundials_mem = NULL;
      }
   }

   if (!sundials_mem)
   {
      // Temporarly set N_Vector wrapper data to create CVODE. The correct
      // initial condition will be set using CVodeReInit() when Step() is
      // called.
      if (!Parallel())
      {
         NV_LENGTH_S(y) = local_size;
         NV_DATA_S(y)   = new double[local_size](); // value-initialize
      }
      else
      {
#ifdef MFEM_USE_MPI
         NV_LOCLENGTH_P(y)  = local_size;
         NV_GLOBLENGTH_P(y) = global_size;
         saved_global_size  = global_size;
         NV_DATA_P(y)       = new double[local_size](); // value-initialize
#endif
      }

      // Create CVODE
      sundials_mem = CVodeCreate(lmm_type);
      MFEM_VERIFY(sundials_mem, "error in CVodeCreate()");

      // Initialize CVODE
      flag = CVodeInit(sundials_mem, CVODESolver::RHS, t, y);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeInit()");

      // Attach the CVODESolver as user-defined data
      flag = CVodeSetUserData(sundials_mem, this);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetUserData()");

      // Set default tolerances
      flag = CVodeSStolerances(sundials_mem, default_rel_tol, default_abs_tol);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetSStolerances()");

      // Attach MFEM linear solver by default
      UseMFEMLinearSolver();

      // Delete the allocated data in y.
      if (!Parallel())
      {
         delete [] NV_DATA_S(y);
         NV_DATA_S(y) = NULL;
      }
      else
      {
#ifdef MFEM_USE_MPI
         delete [] NV_DATA_P(y);
         NV_DATA_P(y) = NULL;
#endif
      }
   }

   // Set the reinit flag to call CVodeReInit() in the next Step() call.
   reinit = true;
}

void CVODESolver::Step(Vector &x, double &t, double &dt)
{
   if (!Parallel())
   {
      NV_DATA_S(y) = x.GetData();
      MFEM_VERIFY(NV_LENGTH_S(y) == x.Size(), "");
   }
   else
   {
#ifdef MFEM_USE_MPI
      NV_DATA_P(y) = x.GetData();
      MFEM_VERIFY(NV_LOCLENGTH_P(y) == x.Size(), "");
#endif
   }

   // Reinitialize CVODE memory if needed
   if (reinit)
   {
      flag = CVodeReInit(sundials_mem, t, y);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeReInit()");

      // reset flag
      reinit = false;
   }

   // Integrate the system
   double tout = t + dt;
   flag = CVode(sundials_mem, tout, y, &t, step_mode);
   MFEM_VERIFY(flag >= 0, "error in CVode()");

   // Return the last incremental step size
   flag = CVodeGetLastStep(sundials_mem, &dt);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeGetLastStep()");
}

void CVODESolver::UseMFEMLinearSolver()
{
   // Free any existing matrix and linear solver
   if (A != NULL)   { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Wrap linear solver as SUNLinearSolver and SUNMatrix
   LSA = SUNLinSolNewEmpty();
   MFEM_VERIFY(LSA, "error in SUNLinSolNewEmpty()");

   LSA->content      = this;
   LSA->ops->gettype = LSGetType;
   LSA->ops->solve   = CVODESolver::LinSysSolve;
   LSA->ops->free    = LSFree;

   A = SUNMatNewEmpty();
   MFEM_VERIFY(A, "error in SUNMatNewEmpty()");

   A->content      = this;
   A->ops->getid   = MatGetID;
   A->ops->destroy = MatDestroy;

   // Attach the linear solver and matrix
   flag = CVodeSetLinearSolver(sundials_mem, LSA, A);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetLinearSolver()");

   // Set the linear system evaluation function
   flag = CVodeSetLinSysFn(sundials_mem, CVODESolver::LinSysSetup);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetLinSysFn()");
}

void CVODESolver::UseSundialsLinearSolver()
{
   // Free any existing matrix and linear solver
   if (A != NULL)   { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Create linear solver
   LSA = SUNLinSol_SPGMR(y, PREC_NONE, 0);
   MFEM_VERIFY(LSA, "error in SUNLinSol_SPGMR()");

   // Attach linear solver
   flag = CVodeSetLinearSolver(sundials_mem, LSA, NULL);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetLinearSolver()");
}

void CVODESolver::SetStepMode(int itask)
{
   step_mode = itask;
}

void CVODESolver::SetSStolerances(double reltol, double abstol)
{
   flag = CVodeSStolerances(sundials_mem, reltol, abstol);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSStolerances()");
}

void CVODESolver::SetMaxStep(double dt_max)
{
   flag = CVodeSetMaxStep(sundials_mem, dt_max);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetMaxStep()");
}

void CVODESolver::SetMaxOrder(int max_order)
{
   flag = CVodeSetMaxOrd(sundials_mem, max_order);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetMaxOrd()");
}

void CVODESolver::PrintInfo() const
{
   long int nsteps, nfevals, nlinsetups, netfails;
   int      qlast, qcur;
   double   hinused, hlast, hcur, tcur;
   long int nniters, nncfails;

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

int ARKStepSolver::RHS1(realtype t, const N_Vector y, N_Vector ydot,
                        void *user_data)
{
   // Get data from N_Vectors
   const Vector mfem_y(y);
   Vector mfem_ydot(ydot);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(user_data);

   // Compute f(t, y) in y' = f(t, y) or fe(t, y) in y' = fe(t, y) + fi(t, y)
   self->f->SetTime(t);
   if (self->rk_type == IMEX)
   {
      self->f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
   }
   self->f->Mult(mfem_y, mfem_ydot);

   // Return success
   return (0);
}

int ARKStepSolver::RHS2(realtype t, const N_Vector y, N_Vector ydot,
                        void *user_data)
{
   // Get data from N_Vectors
   const Vector mfem_y(y);
   Vector mfem_ydot(ydot);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(user_data);

   // Compute fi(t, y) in y' = fe(t, y) + fi(t, y)
   self->f->SetTime(t);
   self->f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   self->f->Mult(mfem_y, mfem_ydot);

   // Return success
   return (0);
}

int ARKStepSolver::LinSysSetup(realtype t, N_Vector y, N_Vector fy, SUNMatrix A,
                               SUNMatrix M, booleantype jok, booleantype *jcur,
                               realtype gamma, void *user_data, N_Vector tmp1,
                               N_Vector tmp2, N_Vector tmp3)
{
   // Get data from N_Vectors
   const Vector mfem_y(y);
   const Vector mfem_fy(fy);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(A));

   // Compute the linear system
   self->f->SetTime(t);
   if (self->rk_type == IMEX)
   {
      self->f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   }
   return (self->f->SUNImplicitSetup(mfem_y, mfem_fy, jok, jcur, gamma));
}

int ARKStepSolver::LinSysSolve(SUNLinearSolver LS, SUNMatrix A, N_Vector x,
                               N_Vector b, realtype tol)
{
   Vector mfem_x(x);
   const Vector mfem_b(b);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(LS));

   // Solve the linear system
   if (self->rk_type == IMEX)
   {
      self->f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   }
   return (self->f->SUNImplicitSolve(mfem_b, mfem_x, tol));
}

int ARKStepSolver::MassSysSetup(realtype t, SUNMatrix M, void *user_data,
                                N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(M));

   // Compute the mass matrix system
   self->f->SetTime(t);
   return (self->f->SUNMassSetup());
}

int ARKStepSolver::MassSysSolve(SUNLinearSolver LS, SUNMatrix M, N_Vector x,
                                N_Vector b, realtype tol)
{
   Vector mfem_x(x);
   const Vector mfem_b(b);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(LS));

   // Solve the mass matrix system
   return (self->f->SUNMassSolve(mfem_b, mfem_x, tol));
}

int ARKStepSolver::MassMult1(SUNMatrix M, N_Vector x, N_Vector v)
{
   const Vector mfem_x(x);
   Vector mfem_v(v);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(M));

   // Compute the mass matrix-vector product
   return (self->f->SUNMassMult(mfem_x, mfem_v));
}

int ARKStepSolver::MassMult2(N_Vector x, N_Vector v, realtype t,
                             void* mtimes_data)
{
   const Vector mfem_x(x);
   Vector mfem_v(v);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(mtimes_data);

   // Compute the mass matrix-vector product
   self->f->SetTime(t);
   return (self->f->SUNMassMult(mfem_x, mfem_v));
}

ARKStepSolver::ARKStepSolver(Type type)
   : rk_type(type), step_mode(ARK_NORMAL),
     use_implicit(type == IMPLICIT || type == IMEX)
{
   // Allocate an empty serial N_Vector
   y = N_VNewEmpty_Serial(0);
   MFEM_VERIFY(y, "error in N_VNewEmpty_Serial()");
}

#ifdef MFEM_USE_MPI
ARKStepSolver::ARKStepSolver(MPI_Comm comm, Type type)
   : rk_type(type), step_mode(ARK_NORMAL),
     use_implicit(type == IMPLICIT || type == IMEX)
{
   if (comm == MPI_COMM_NULL)
   {

      // Allocate an empty serial N_Vector
      y = N_VNewEmpty_Serial(0);
      MFEM_VERIFY(y, "error in N_VNewEmpty_Serial()");

   }
   else
   {

      // Allocate an empty parallel N_Vector
      y = N_VNewEmpty_Parallel(comm, 0, 0);  // calls MPI_Allreduce()
      MFEM_VERIFY(y, "error in N_VNewEmpty_Parallel()");

   }
}
#endif

void ARKStepSolver::Init(TimeDependentOperator &f_)
{
   // Initialize the base class
   ODESolver::Init(f_);

   // Get the vector length
   long local_size = f_.Height();
#ifdef MFEM_USE_MPI
   long global_size;
#endif

   if (Parallel())
   {
#ifdef MFEM_USE_MPI
      MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM,
                    NV_COMM_P(y));
#endif
   }

   // Get current time
   double t = f_.GetTime();

   if (sundials_mem)
   {
      // Check if the problem size has changed since the last Init() call
      int resize = 0;
      if (!Parallel())
      {
         resize = (NV_LENGTH_S(y) != local_size);
      }
      else
      {
#ifdef MFEM_USE_MPI
         int l_resize = (NV_LOCLENGTH_P(y) != local_size) ||
                        (saved_global_size != global_size);
         MPI_Allreduce(&l_resize, &resize, 1, MPI_INT, MPI_LOR, NV_COMM_P(y));
#endif
      }

      // Free existing solver memory and re-create with new vector size
      if (resize)
      {
         ARKStepFree(&sundials_mem);
         sundials_mem = NULL;
      }
   }

   if (!sundials_mem)
   {

      // Temporarly set N_Vector wrapper data to create ARKStep. The correct
      // initial condition will be set using ARKStepReInit() when Step() is
      // called.
      if (!Parallel())
      {
         NV_LENGTH_S(y) = local_size;
         NV_DATA_S(y)   = new double[local_size](); // value-initialize
      }
      else
      {
#ifdef MFEM_USE_MPI
         NV_LOCLENGTH_P(y)  = local_size;
         NV_GLOBLENGTH_P(y) = global_size;
         saved_global_size  = global_size;
         NV_DATA_P(y)       = new double[local_size](); // value-initialize
#endif
      }

      // Create ARKStep memory
      if (rk_type == IMPLICIT)
      {
         sundials_mem = ARKStepCreate(NULL, ARKStepSolver::RHS1, t, y);
      }
      else if (rk_type == EXPLICIT)
      {
         sundials_mem = ARKStepCreate(ARKStepSolver::RHS1, NULL, t, y);
      }
      else
      {
         sundials_mem = ARKStepCreate(ARKStepSolver::RHS1, ARKStepSolver::RHS2,
                                      t, y);
      }
      MFEM_VERIFY(sundials_mem, "error in ARKStepCreate()");

      // Attach the ARKStepSolver as user-defined data
      flag = ARKStepSetUserData(sundials_mem, this);
      MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetUserData()");

      // Set default tolerances
      flag = ARKStepSStolerances(sundials_mem, default_rel_tol, default_abs_tol);
      MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetSStolerances()");

      // If implicit, attach MFEM linear solver by default
      if (use_implicit) { UseMFEMLinearSolver(); }

      // Delete the allocated data in y.
      if (!Parallel())
      {
         delete [] NV_DATA_S(y);
         NV_DATA_S(y) = NULL;
      }
      else
      {
#ifdef MFEM_USE_MPI
         delete [] NV_DATA_P(y);
         NV_DATA_P(y) = NULL;
#endif
      }
   }

   // Set the reinit flag to call ARKStepReInit() in the next Step() call.
   reinit = true;
}

void ARKStepSolver::Step(Vector &x, double &t, double &dt)
{
   if (!Parallel())
   {
      NV_DATA_S(y) = x.GetData();
      MFEM_VERIFY(NV_LENGTH_S(y) == x.Size(), "");
   }
   else
   {
#ifdef MFEM_USE_MPI
      NV_DATA_P(y) = x.GetData();
      MFEM_VERIFY(NV_LOCLENGTH_P(y) == x.Size(), "");
#endif
   }

   // Reinitialize ARKStep memory if needed
   if (reinit)
   {
      if (rk_type == IMPLICIT)
      {
         flag = ARKStepReInit(sundials_mem, NULL, ARKStepSolver::RHS1, t, y);
      }
      else if (rk_type == EXPLICIT)
      {
         flag = ARKStepReInit(sundials_mem, ARKStepSolver::RHS1, NULL, t, y);
      }
      else
      {
         flag = ARKStepReInit(sundials_mem,
                              ARKStepSolver::RHS1, ARKStepSolver::RHS2, t, y);
      }
      MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepReInit()");

      // reset flag
      reinit = false;
   }

   // Integrate the system
   double tout = t + dt;
   flag = ARKStepEvolve(sundials_mem, tout, y, &t, step_mode);
   MFEM_VERIFY(flag >= 0, "error in ARKStepEvolve()");

   // Return the last incremental step size
   flag = ARKStepGetLastStep(sundials_mem, &dt);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepGetLastStep()");
}

void ARKStepSolver::UseMFEMLinearSolver()
{
   // Free any existing matrix and linear solver
   if (A != NULL)   { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Wrap linear solver as SUNLinearSolver and SUNMatrix
   LSA = SUNLinSolNewEmpty();
   MFEM_VERIFY(LSA, "error in SUNLinSolNewEmpty()");

   LSA->content      = this;
   LSA->ops->gettype = LSGetType;
   LSA->ops->solve   = ARKStepSolver::LinSysSolve;
   LSA->ops->free    = LSFree;

   A = SUNMatNewEmpty();
   MFEM_VERIFY(A, "error in SUNMatNewEmpty()");

   A->content      = this;
   A->ops->getid   = MatGetID;
   A->ops->destroy = MatDestroy;

   // Attach the linear solver and matrix
   flag = ARKStepSetLinearSolver(sundials_mem, LSA, A);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetLinearSolver()");

   // Set the linear system evaluation function
   flag = ARKStepSetLinSysFn(sundials_mem, ARKStepSolver::LinSysSetup);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetLinSysFn()");
}

void ARKStepSolver::UseSundialsLinearSolver()
{
   // Free any existing matrix and linear solver
   if (A != NULL)   { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Create linear solver
   LSA = SUNLinSol_SPGMR(y, PREC_NONE, 0);
   MFEM_VERIFY(LSA, "error in SUNLinSol_SPGMR()");

   // Attach linear solver
   flag = ARKStepSetLinearSolver(sundials_mem, LSA, NULL);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetLinearSolver()");
}

void ARKStepSolver::UseMFEMMassLinearSolver(int tdep)
{
   // Free any existing matrix and linear solver
   if (M != NULL)   { SUNMatDestroy(M); M = NULL; }
   if (LSM != NULL) { SUNLinSolFree(LSM); LSM = NULL; }

   // Wrap linear solver as SUNLinearSolver and SUNMatrix
   LSM = SUNLinSolNewEmpty();
   MFEM_VERIFY(LSM, "error in SUNLinSolNewEmpty()");

   LSM->content      = this;
   LSM->ops->gettype = LSGetType;
   LSM->ops->solve   = ARKStepSolver::MassSysSolve;
   LSA->ops->free    = LSFree;

   M = SUNMatNewEmpty();
   MFEM_VERIFY(M, "error in SUNMatNewEmpty()");

   M->content      = this;
   M->ops->getid   = SUNMatGetID;
   M->ops->matvec  = ARKStepSolver::MassMult1;
   M->ops->destroy = MatDestroy;

   // Attach the linear solver and matrix
   flag = ARKStepSetMassLinearSolver(sundials_mem, LSM, M, tdep);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetLinearSolver()");

   // Set the linear system function
   flag = ARKStepSetMassFn(sundials_mem, ARKStepSolver::MassSysSetup);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetMassFn()");
}

void ARKStepSolver::UseSundialsMassLinearSolver(int tdep)
{
   // Free any existing matrix and linear solver
   if (M != NULL)   { SUNMatDestroy(A); M = NULL; }
   if (LSM != NULL) { SUNLinSolFree(LSM); LSM = NULL; }

   // Create linear solver
   LSM = SUNLinSol_SPGMR(y, PREC_NONE, 0);
   MFEM_VERIFY(LSM, "error in SUNLinSol_SPGMR()");

   // Attach linear solver
   flag = ARKStepSetMassLinearSolver(sundials_mem, LSM, NULL, tdep);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetMassLinearSolver()");

   // Attach matrix multiplication function
   flag = ARKStepSetMassTimes(sundials_mem, NULL, ARKStepSolver::MassMult2,
                              this);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetMassTimes()");
}

void ARKStepSolver::SetStepMode(int itask)
{
   step_mode = itask;
}

void ARKStepSolver::SetSStolerances(double reltol, double abstol)
{
   flag = ARKStepSStolerances(sundials_mem, reltol, abstol);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSStolerances()");
}

void ARKStepSolver::SetMaxStep(double dt_max)
{
   flag = ARKStepSetMaxStep(sundials_mem, dt_max);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetMaxStep()");
}

void ARKStepSolver::SetOrder(int order)
{
   flag = ARKStepSetOrder(sundials_mem, order);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetOrder()");
}

void ARKStepSolver::SetERKTableNum(int table_num)
{
   flag = ARKStepSetTableNum(sundials_mem, -1, table_num);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetTableNum()");
}

void ARKStepSolver::SetIRKTableNum(int table_num)
{
   flag = ARKStepSetTableNum(sundials_mem, table_num, -1);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetTableNum()");
}

void ARKStepSolver::SetIMEXTableNum(int etable_num, int itable_num)
{
   flag = ARKStepSetTableNum(sundials_mem, itable_num, etable_num);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetTableNum()");
}

void ARKStepSolver::SetFixedStep(double dt)
{
   flag = ARKStepSetFixedStep(sundials_mem, dt);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetFixedStep()");
}

void ARKStepSolver::PrintInfo() const
{
   long int nsteps, expsteps, accsteps, step_attempts;
   long int nfe_evals, nfi_evals;
   long int nlinsetups, netfails;
   double   hinused, hlast, hcur, tcur;
   long int nniters, nncfails;

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

// ---------------------------------------------------------------------------
// KINSOL interface
// ---------------------------------------------------------------------------

// Wrapper for evaluating the nonlinear residual F(u) = 0
int KINSolver::Mult(const N_Vector u, N_Vector fu, void *user_data)
{
   const Vector mfem_u(u);
   Vector mfem_fu(fu);
   KINSolver *self = static_cast<KINSolver*>(user_data);

   // Compute the non-linear action F(u).
   self->oper->Mult(mfem_u, mfem_fu);

   // Return success
   return 0;
}

// Wrapper for computing Jacobian-vector products
int KINSolver::GradientMult(N_Vector v, N_Vector Jv, N_Vector u,
                            booleantype *new_u, void *user_data)
{
   const Vector mfem_v(v);
   Vector mfem_Jv(Jv);
   KINSolver *self = static_cast<KINSolver*>(user_data);

   // Update Jacobian information if needed
   if (*new_u)
   {
      const Vector mfem_u(u);
      self->jacobian = &self->oper->GetGradient(mfem_u);
      *new_u = SUNFALSE;
   }

   // Compute the Jacobian-vector product
   self->jacobian->Mult(mfem_v, mfem_Jv);

   // Return success
   return 0;
}

// Wrapper for evaluating linear systems J u = b
int KINSolver::LinSysSetup(N_Vector u, N_Vector fu, SUNMatrix J,
                           void *user_data, N_Vector tmp1, N_Vector tmp2)
{
   const Vector mfem_u(u);
   KINSolver *self = static_cast<KINSolver*>(GET_CONTENT(J));

   // Update the Jacobian
   self->jacobian = &self->oper->GetGradient(mfem_u);

   // Set the Jacobian solve operator
   self->prec->SetOperator(*self->jacobian);

   // Return success
   return (0);
}

// Wrapper for solving linear systems J u = b
int KINSolver::LinSysSolve(SUNLinearSolver LS, SUNMatrix J, N_Vector u,
                           N_Vector b, realtype tol)
{
   Vector mfem_u(u), mfem_b(b);
   KINSolver *self = static_cast<KINSolver*>(GET_CONTENT(LS));

   // Solve for u = [J(u)]^{-1} b, maybe approximately.
   self->prec->Mult(mfem_b, mfem_u);

   // Return success
   return (0);
}

KINSolver::KINSolver(int strategy, bool oper_grad)
   : global_strategy(strategy), use_oper_grad(oper_grad), y_scale(NULL),
     f_scale(NULL), jacobian(NULL), maa(0)
{
   // Allocate empty serial N_Vectors
   y = N_VNewEmpty_Serial(0);
   y_scale = N_VNewEmpty_Serial(0);
   f_scale = N_VNewEmpty_Serial(0);
   MFEM_VERIFY(y && y_scale && f_scale, "Error in N_VNewEmpty_Serial().");

   // Default abs_tol and print_level
   abs_tol     = pow(UNIT_ROUNDOFF, 1.0/3.0);
   print_level = 0;
}

#ifdef MFEM_USE_MPI
KINSolver::KINSolver(MPI_Comm comm, int strategy, bool oper_grad)
   : global_strategy(strategy), use_oper_grad(oper_grad), y_scale(NULL),
     f_scale(NULL), jacobian(NULL), maa(0)
{
   if (comm == MPI_COMM_NULL)
   {

      // Allocate empty serial N_Vectors
      y = N_VNewEmpty_Serial(0);
      y_scale = N_VNewEmpty_Serial(0);
      f_scale = N_VNewEmpty_Serial(0);
      MFEM_VERIFY(y && y_scale && f_scale, "error in N_VNewEmpty_Serial()");

   }
   else
   {

      // Allocate empty parallel N_Vectors
      y = N_VNewEmpty_Parallel(comm, 0, 0);
      y_scale = N_VNewEmpty_Parallel(comm, 0, 0);
      f_scale = N_VNewEmpty_Parallel(comm, 0, 0);
      MFEM_VERIFY(y && y_scale && f_scale, "error in N_VNewEmpty_Parallel()");

   }

   // Default abs_tol and print_level
   abs_tol     = pow(UNIT_ROUNDOFF, 1.0/3.0);
   print_level = 0;
}
#endif


void KINSolver::SetOperator(const Operator &op)
{
   // Initialize the base class
   NewtonSolver::SetOperator(op);
   jacobian = NULL;

   // Get the vector length
   long local_size = height;
#ifdef MFEM_USE_MPI
   long global_size;
#endif

   if (Parallel())
   {
#ifdef MFEM_USE_MPI
      MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM,
                    NV_COMM_P(y));
#endif
   }

   if (sundials_mem)
   {
      // Check if the problem size has changed since the last SetOperator call
      int resize = 0;
      if (!Parallel())
      {
         resize = (NV_LENGTH_S(y) != local_size);
      }
      else
      {
#ifdef MFEM_USE_MPI
         int l_resize = (NV_LOCLENGTH_P(y) != local_size) ||
                        (saved_global_size != global_size);
         MPI_Allreduce(&l_resize, &resize, 1, MPI_INT, MPI_LOR, NV_COMM_P(y));
#endif
      }

      // Free existing solver memory and re-create with new vector size
      if (resize)
      {
         KINFree(&sundials_mem);
         sundials_mem = NULL;
      }
   }

   if (!sundials_mem)
   {
      // Set actual size and data in the N_Vector y.
      if (!Parallel())
      {
         NV_LENGTH_S(y)       = local_size;
         NV_DATA_S(y)         = new double[local_size](); // value-initialize
         NV_LENGTH_S(y_scale) = local_size;
         NV_DATA_S(y_scale)   = NULL;
         NV_LENGTH_S(f_scale) = local_size;
         NV_DATA_S(f_scale)   = NULL;
      }
      else
      {
#ifdef MFEM_USE_MPI
         NV_LOCLENGTH_P(y)        = local_size;
         NV_GLOBLENGTH_P(y)       = global_size;
         NV_DATA_P(y)             = new double[local_size](); // value-initialize
         NV_LOCLENGTH_P(y_scale)  = local_size;
         NV_GLOBLENGTH_P(y_scale) = global_size;
         NV_DATA_P(y_scale)       = NULL;
         NV_LOCLENGTH_P(f_scale)  = local_size;
         NV_GLOBLENGTH_P(f_scale) = global_size;
         NV_DATA_P(f_scale)       = NULL;
         saved_global_size        = global_size;
#endif
      }

      // Create the solver memory
      sundials_mem = KINCreate();
      MFEM_VERIFY(sundials_mem, "Error in KINCreate().");

      // Set number of acceleration vectors
      if (maa > 0)
      {
         flag = KINSetMAA(sundials_mem, maa);
         MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetMAA()");
      }

      // Initialize KINSOL
      flag = KINInit(sundials_mem, KINSolver::Mult, y);
      MFEM_VERIFY(flag == KIN_SUCCESS, "error in KINInit()");

      // Attach the KINSolver as user-defined data
      flag = KINSetUserData(sundials_mem, this);
      MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetUserData()");

      // Set the linear solver
      if (prec)
      {
         KINSolver::SetSolver(*prec);
      }
      else
      {
         // Free any existing linear solver
         if (A != NULL) { SUNMatDestroy(A); A = NULL; }
         if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

         LSA = SUNLinSol_SPGMR(y, PREC_NONE, 0);
         MFEM_VERIFY(LSA, "error in SUNLinSol_SPGMR()");

         flag = KINSetLinearSolver(sundials_mem, LSA, NULL);
         MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetLinearSolver()");

         // Set Jacobian-vector product function
         if (use_oper_grad)
         {
            flag = KINSetJacTimesVecFn(sundials_mem, KINSolver::GradientMult);
            MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetJacTimesVecFn()");
         }
      }

      // Delete the allocated data in y.
      if (!Parallel())
      {
         delete [] NV_DATA_S(y);
         NV_DATA_S(y) = NULL;
      }
      else
      {
#ifdef MFEM_USE_MPI
         delete [] NV_DATA_P(y);
         NV_DATA_P(y) = NULL;
#endif
      }
   }
}

void KINSolver::SetSolver(Solver &solver)
{
   // Store the solver
   prec = &solver;

   // Free any existing linear solver
   if (A != NULL) { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Wrap KINSolver as SUNLinearSolver and SUNMatrix
   LSA = SUNLinSolNewEmpty();
   MFEM_VERIFY(LSA, "error in SUNLinSolNewEmpty()");

   LSA->content      = this;
   LSA->ops->gettype = LSGetType;
   LSA->ops->solve   = KINSolver::LinSysSolve;
   LSA->ops->free    = LSFree;

   A = SUNMatNewEmpty();
   MFEM_VERIFY(A, "error in SUNMatNewEmpty()");

   A->content      = this;
   A->ops->getid   = MatGetID;
   A->ops->destroy = MatDestroy;

   // Attach the linear solver and matrix
   flag = KINSetLinearSolver(sundials_mem, LSA, A);
   MFEM_VERIFY(flag == KIN_SUCCESS, "error in KINSetLinearSolver()");

   // Set the Jacobian evaluation function
   flag = KINSetJacFn(sundials_mem, KINSolver::LinSysSetup);
   MFEM_VERIFY(flag == KIN_SUCCESS, "error in KINSetJacFn()");
}

void KINSolver::SetScaledStepTol(double sstol)
{
   flag = KINSetScaledStepTol(sundials_mem, sstol);
   MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetScaledStepTol()");
}

void KINSolver::SetMaxSetupCalls(int max_calls)
{
   flag = KINSetMaxSetupCalls(sundials_mem, max_calls);
   MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetMaxSetupCalls()");
}

void KINSolver::SetMAA(int m_aa)
{
   // Store internally as maa must be set before calling KINInit() to
   // set the maximum acceleration space size.
   maa = m_aa;
   if (sundials_mem)
   {
      flag = KINSetMAA(sundials_mem, maa);
      MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetMAA()");
   }
}

// Compute the scaling vectors and solve nonlinear system
void KINSolver::Mult(const Vector &b, Vector &x) const
{
   // residual norm tolerance
   double tol;

   // Uses c = 1, corresponding to x_scale.
   c = 1.0;

   if (!iterative_mode) { x = 0.0; }

   // For relative tolerance, r = 1 / |residual(x)|, corresponding to fx_scale.
   if (rel_tol > 0.0)
   {

      oper->Mult(x, r);

      // Note that KINSOL uses infinity norms.
      double norm = r.Normlinf();
#ifdef MFEM_USE_MPI
      if (Parallel())
      {
         double lnorm = norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_MAX, NV_COMM_P(y));
      }
#endif
      if (abs_tol > rel_tol * norm)
      {
         r = 1.0;
         tol = abs_tol;
      }
      else
      {
         r =  1.0 / norm;
         tol = rel_tol;
      }
   }
   else
   {
      r = 1.0;
      tol = abs_tol;
   }

   // Set the residual norm tolerance
   flag = KINSetFuncNormTol(sundials_mem, tol);
   MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetFuncNormTol()");

   // Solve the nonlinear system by calling the other Mult method
   KINSolver::Mult(x, c, r);
}

// Solve the nonlinear system using the provided scaling vectors
void KINSolver::Mult(Vector &x,
                     const Vector &x_scale, const Vector &fx_scale) const
{
   flag = KINSetNumMaxIters(sundials_mem, max_iter);
   MFEM_ASSERT(flag == KIN_SUCCESS, "KINSetNumMaxIters() failed!");

   if (!Parallel())
   {

      NV_DATA_S(y) = x.GetData();
      MFEM_VERIFY(NV_LENGTH_S(y) == x.Size(), "");
      NV_DATA_S(y_scale) = x_scale.GetData();
      NV_DATA_S(f_scale) = fx_scale.GetData();

      flag = KINSetPrintLevel(sundials_mem, print_level);
      MFEM_VERIFY(flag == KIN_SUCCESS, "KINSetPrintLevel() failed!");
   }
   else
   {

#ifdef MFEM_USE_MPI
      NV_DATA_P(y) = x.GetData();
      MFEM_VERIFY(NV_LOCLENGTH_P(y) == x.Size(), "");
      NV_DATA_P(y_scale) = x_scale.GetData();
      NV_DATA_P(f_scale) = fx_scale.GetData();

      int rank;
      MPI_Comm_rank(NV_COMM_P(y), &rank);
      if (rank == 0)
      {
         flag = KINSetPrintLevel(sundials_mem, print_level);
         MFEM_VERIFY(flag == KIN_SUCCESS, "KINSetPrintLevel() failed!");
      }
#endif

   }

   if (!iterative_mode) { x = 0.0; }

   // Solve the nonlinear system
   flag = KINSol(sundials_mem, y, global_strategy, y_scale, f_scale);
   converged = (flag >= 0);

   // Get number of nonlinear iterations
   long int tmp_nni;
   flag = KINGetNumNonlinSolvIters(sundials_mem, &tmp_nni);
   MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINGetNumNonlinSolvIters()");
   final_iter = (int) tmp_nni;

   // Get the residual norm
   flag = KINGetFuncNorm(sundials_mem, &final_norm);
   MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINGetFuncNorm()");
}

KINSolver::~KINSolver()
{
   N_VDestroy(y);
   N_VDestroy(y_scale);
   N_VDestroy(f_scale);
   SUNMatDestroy(A);
   SUNLinSolFree(LSA);
   KINFree(&sundials_mem);
}

} // namespace mfem

#endif
