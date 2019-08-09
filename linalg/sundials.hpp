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

#ifndef MFEM_SUNDIALS
#define MFEM_SUNDIALS

#include "../config/config.hpp"

#ifdef MFEM_USE_SUNDIALS

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include "ode.hpp"
#include "solvers.hpp"

#include <sundials/sundials_config.h>
// Check for appropriate SUNDIALS version
#if !defined(SUNDIALS_VERSION_MAJOR) || (SUNDIALS_VERSION_MAJOR < 5)
#error MFEM requires SUNDIALS version 5.0.0 or newer!
#endif
#include <cvode/cvode.h>
#include <arkode/arkode_arkstep.h>
#include <kinsol/kinsol.h>

namespace mfem
{

// ---------------------------------------------------------------------------
// Base class for interfacing with SUNDIALS packages
// ---------------------------------------------------------------------------

/// Base class for interfacing with SUNDIALS packages.
class SundialsSolver
{
protected:
   void *sundials_mem;     ///< SUNDIALS mem structure.
   mutable int flag;       ///< Last flag returned from a call to SUNDIALS.
   bool reinit;            ///< Flag to signal memory reinitialization is need.
   bool resize;            ///< Flag to singal memory resizing is needed.
   long saved_global_size; ///< Global length on last call to Init().

   N_Vector           y;   ///< State vector.
   SUNMatrix          A;   ///< Linear system A = I - gamma J, M - gamma J, or J.
   SUNMatrix          M;   ///< Mass matrix M.
   SUNLinearSolver    LSA; ///< Linear solver for A.
   SUNLinearSolver    LSM; ///< Linear solver for M.
   SUNNonlinearSolver NLS; ///< Nonlinear solver.

#ifdef MFEM_USE_MPI
   bool Parallel() const
   { return (N_VGetVectorID(y) != SUNDIALS_NVEC_SERIAL); }
#else
   bool Parallel() const { return false; }
#endif

   /// Default scalar relative tolerance.
   static constexpr double default_rel_tol = 1e-4;
   /// Default scalar absolute tolerance.
   static constexpr double default_abs_tol = 1e-9;

   /** @brief Protected constructor: objects of this type should be constructed
       only as part of a derived class. */
   SundialsSolver() : sundials_mem(NULL), flag(0), reinit(false), resize(false),
                      saved_global_size(0), y(NULL), A(NULL), M(NULL),
                      LSA(NULL), LSM(NULL), NLS(NULL) { }

public:
   /// Access the SUNDIALS memory structure.
   void *GetMem() const { return sundials_mem; }

   /// Returns the last flag retured by a call to a SUNDIALS function.
   int GetFlag() const { return flag; }
};


// ---------------------------------------------------------------------------
// Interface to the CVODE library -- linear multi-step methods
// ---------------------------------------------------------------------------

/// Interface to the CVODE library -- linear multi-step methods.
class CVODESolver : public ODESolver, public SundialsSolver
{
protected:
   int lmm_type;  ///< Linear multistep method type.
   int step_mode; ///< CVODE step mode (CV_NORMAL or CV_ONE_STEP).

   /// Wrapper to compute the ODE rhs function.
   static int RHS(realtype t, const N_Vector y, N_Vector ydot, void *user_data);

   /// Setup the linear system A x = b
   static int LinSysSetup(realtype t, N_Vector y, N_Vector fy, SUNMatrix A,
                          booleantype jok, booleantype *jcur,
                          realtype gamma, void *user_data, N_Vector tmp1,
                          N_Vector tmp2, N_Vector tmp3);

   /// Solve the linear system A x = b
   static int LinSysSolve(SUNLinearSolver LS, SUNMatrix A, N_Vector x,
                          N_Vector b, realtype tol);

public:
   /// Construct a serial wrapper to SUNDIALS' CVODE integrator.
   /** @param[in] lmm Specifies the linear multistep method, the options are:
                      - CV_ADAMS - implicit methods for non-stiff systems,
                      - CV_BDF   - implicit methods for stiff systems. */
   CVODESolver(int lmm);

#ifdef MFEM_USE_MPI
   /// Construct a parallel wrapper to SUNDIALS' CVODE integrator.
   /** @param[in] comm The MPI communicator used to partition the ODE system
       @param[in] lmm  Specifies the linear multistep method, the options are:
                       - CV_ADAMS - implicit methods for non-stiff systems,
                       - CV_BDF   - implicit methods for stiff systems. */
   CVODESolver(MPI_Comm comm, int lmm);
#endif

   /** @brief Initialize CVODE: calls CVodeCreate() to create the CVODE
       memory and set some defaults. If the CVODE memory has already been
       created, it signals if a call to CVodeReInit() or CVodeResize() needs
       to be made on the next Step() call. */
   /** @param[in] f_ The TimeDependentOperator that defines the ODE system.

       @note All other methods must be called after Init(). */
   void Init(TimeDependentOperator &f_);

   /// Integrate the ODE with CVODE using the specified step mode.
   /** @param[in,out] x  On output, the solution vector at the requested output
                         time tout = @a t + @a dt..
       @param[in,out] t  On output, the output time reached.
       @param[in,out] dt On output, the last time step taken.

       @note On input, the values of @a t and @a dt are used to compute desired
       output time for the integration, tout = @a t + @a dt.
   */
   virtual void Step(Vector &x, double &t, double &dt);

   /// Attach an MFEM linear solver to CVODE.
   void UseMFEMLinearSolver();

   /// Attach SUNDIALS linear solver to CVODE.
   void UseSundialsLinearSolver();

   /// Select the CVODE step mode: CV_NORMAL (default) or CV_ONE_STEP.
   /** @param[in] itask  The desired step mode. */
   void SetStepMode(int itask);

   /// Set the scalar relative and scalar absolute tolerances.
   void SetSStolerances(double reltol, double abstol);

   /// Set the maximum time step.
   void SetMaxStep(double dt_max);

   /** @brief Set the maximum method order.

       CVODE uses adaptive-order integration, based on the local truncation
       error. The default values for @a max_order are 12 for CV_ADAMS and
       5 for CV_BDF. Use this if you know a priori that your system is such
       that higher order integration formulas are unstable.

       @note @a max_order can't be higher than the current maximum order. */
   void SetMaxOrder(int max_order);

   /// Print various CVODE statistics.
   void PrintInfo() const;

   /// Destroy the associated CVODE memory and SUNDIALS objects.
   virtual ~CVODESolver();
};


// ---------------------------------------------------------------------------
// Interface to ARKode's ARKStep module -- Additive Runge-Kutta methods
// ---------------------------------------------------------------------------

/// Interface to ARKode's ARKStep module -- additive Runge-Kutta methods.
class ARKStepSolver : public ODESolver, public SundialsSolver
{
public:
   /// Types of ARKODE solvers.
   enum Type
   {
      EXPLICIT, ///< Explicit RK method
      IMPLICIT, ///< Implicit RK method
      IMEX      ///< Implicit-explicit ARK method
   };

protected:
   Type rk_type;      ///< Runge-Kutta type.
   int step_mode;     ///< ARKStep step mode (ARK_NORMAL or ARK_ONE_STEP).
   bool use_implicit; ///< True for implicit or imex integration.

   /** @name Wrappers to compute the ODE RHS functions.
       RHS1 is explicit RHS and RHS2 the implicit RHS for IMEX integration. When
       purely implicit or explicit only RHS1 is used. */
   ///@{
   static int RHS1(realtype t, const N_Vector y, N_Vector ydot, void *user_data);
   static int RHS2(realtype t, const N_Vector y, N_Vector ydot, void *user_data);
   ///@}

   /// Setup the linear system A x = b
   static int LinSysSetup(realtype t, N_Vector y, N_Vector fy, SUNMatrix A,
                          SUNMatrix M, booleantype jok, booleantype *jcur,
                          realtype gamma, void *user_data, N_Vector tmp1,
                          N_Vector tmp2, N_Vector tmp3);

   /// Solve the linear system A x = b
   static int LinSysSolve(SUNLinearSolver LS, SUNMatrix A, N_Vector x,
                          N_Vector b, realtype tol);

   /// Setup the linear system M x = b
   static int MassSysSetup(realtype t, SUNMatrix M, void *user_data,
                           N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

   /// Solve the linear system M x = b
   static int MassSysSolve(SUNLinearSolver LS, SUNMatrix M, N_Vector x,
                           N_Vector b, realtype tol);

   /// Compute the matrix-vector product M x = v
   static int MassMult1(SUNMatrix M, N_Vector x, N_Vector v);

   /// Compute the matrix-vector product M x = v
   static int MassMult2(N_Vector x, N_Vector v, realtype t,
                        void* mtimes_data);

public:
   /// Construct a serial wrapper to SUNDIALS' ARKode integrator.
   /** @param[in] type Specifies the RK method type:
                       - EXPLICIT - explicit RK method,
                       - IMPLICIT - implicit RK method,
                       - IMEX     - implicit-explicit ARK method. */
   ARKStepSolver(Type type = EXPLICIT);

#ifdef MFEM_USE_MPI
   /// Construct a parallel wrapper to SUNDIALS' ARKode integrator.
   /** @param[in] comm The MPI communicator used to partition the ODE system.
       @param[in] type Specifies the RK method type:
                       - EXPLICIT - explicit RK method,
                       - IMPLICIT - implicit RK method,
                       - IMEX     - implicit-explicit ARK method. */
   ARKStepSolver(MPI_Comm comm, Type type = EXPLICIT);
#endif

   /** @brief Initialize ARKode: calls ARKStepCreate() to create the ARKStep
       memory and set some defaults. If the ARKStep memory has already been
       created, it signals if a call to ARKReInit() or ARKStepResize() needs
       to be made on the next Step() call. */
   /** @param[in] f_ The TimeDependentOperator that defines the ODE system.

       @note All other methods must be called after Init(). */
   void Init(TimeDependentOperator &f_);

   /// Integrate the ODE with ARKode using the specified step mode.
   /**
       @param[in,out] x  On output, the solution vector at the requested output
                         time, tout = @a t + @a dt.
       @param[in,out] t  On output, the output time reached.
       @param[in,out] dt On output, the last time step taken.

       @note On input, the values of @a t and @a dt are used to compute desired
       output time for the integration, tout = @a t + @a dt.
   */
   virtual void Step(Vector &x, double &t, double &dt);

   /// Attach an MFEM linear solver solver to ARKode.
   void UseMFEMLinearSolver();

   /// Attach a SUNDIALS linear solver to ARKode.
   void UseSundialsLinearSolver();

   /// Attach an MFEM mass matrix linear solver to ARKode.
   /** @param[in] tdep    An integer flag indicating if the mass matrix is time
                          dependent (1) or time independent (0). */
   void UseMFEMMassLinearSolver(int tdep);

   /// Attach a SUNDIALS mass matrix linear solver to ARKode.
   /** @param[in] tdep    An integer flag indicating if the mass matrix is time
                          dependent (1) or time independent (0). */
   void UseSundialsMassLinearSolver(int tdep);

   /// Select the ARKode step mode: ARK_NORMAL (default) or ARK_ONE_STEP.
   /** @param[in] itask  The desired step mode. */
   void SetStepMode(int itask);

   /// Set the scalar relative and scalar absolute tolerances.
   void SetSStolerances(double reltol, double abstol);

   /// Set the maximum time step.
   void SetMaxStep(double dt_max);

   /// Chooses integration order for all explicit / implicit / IMEX methods.
   /** The default is 4, and the allowed ranges are: [2, 8] for explicit;
       [2, 5] for implicit; [3, 5] for IMEX. */
   void SetOrder(int order);

   /// Choose a specific Butcher table for an explicit RK method.
   /** See the documentation for all possible options, stability regions, etc.
       For example, table_num = BOGACKI_SHAMPINE_4_2_3 is 4-stage 3rd order. */
   void SetERKTableNum(int table_num);

   /// Choose a specific Butcher table for a diagonally implicit RK method.
   /** See the documentation for all possible options, stability regions, etc.
       For example, table_num = CASH_5_3_4 is 5-stage 4th order. */
   void SetIRKTableNum(int table_num);

   /// Choose a specific Butcher table for an IMEX RK method.
   /** See the documentation for all possible options, stability regions, etc.
       For example, etable_num = ARK548L2SA_DIRK_8_4_5 and
       itable_num = ARK548L2SA_ERK_8_4_5 is 8-stage 5th order. */
   void SetIMEXTableNum(int etable_num, int itable_num);

   /// Use a fixed time step size (disable temporal adaptivity).
   /** Use of this function is not recommended, since there is no assurance of
       the validity of the computed solutions. It is primarily provided for
       code-to-code verification testing purposes. */
   void SetFixedStep(double dt);

   /// Print various ARKStep statistics.
   void PrintInfo() const;

   /// Destroy the associated ARKode memory and SUNDIALS objects.
   virtual ~ARKStepSolver();
};


// ---------------------------------------------------------------------------
// Interface to the KINSOL library -- nonlinear solver methods
// ---------------------------------------------------------------------------

/// Interface to the KINSOL library -- nonlinear solver methods.
class KINSolver : public NewtonSolver, public SundialsSolver
{
protected:
   int global_strategy;               // KINSOL solution strategy
   bool use_oper_grad;                // use the Jv prod function
   mutable N_Vector y_scale, f_scale; // scaling vectors
   const Operator *jacobian;          // stores oper->GetGradient()

   /// Wrapper to compute the nonlinear residual F(u) = 0
   static int Mult(const N_Vector u, N_Vector fu, void *user_data);

   /// Wrapper to compute the Jacobian-vector product J(u) v = Jv
   static int GradientMult(N_Vector v, N_Vector Jv, N_Vector u,
                           booleantype *new_u, void *user_data);

   /// Setup the linear system J u = b
   static int LinSysSetup(N_Vector u, N_Vector fu, SUNMatrix J,
                          void *user_data, N_Vector tmp1, N_Vector tmp2);

   /// Solve the linear system J u = b
   static int LinSysSolve(SUNLinearSolver LS, SUNMatrix J, N_Vector u,
                          N_Vector b, realtype tol);

public:

   /// Construct a serial wrapper to SUNDIALS' KINSOL nonlinear solver.
   /** @param[in] strategy   Specifies the nonlinear solver strategy:
                             KIN_NONE / KIN_LINESEARCH / KIN_PICARD / KIN_FP.
       @param[in] oper_grad  Specifies whether the solver should use its
                             Operator's GetGradient() method to compute the
                             Jacobian of the system. */
   KINSolver(int strategy, bool oper_grad = true);

#ifdef MFEM_USE_MPI
   /// Construct a parallel wrapper to SUNDIALS' KINSOL nonlinear solver.
   /** @param[in] comm       The MPI communicator used to partition the system.
       @param[in] strategy   Specifies the nonlinear solver strategy:
                             KIN_NONE / KIN_LINESEARCH / KIN_PICARD / KIN_FP.
      @param[in] oper_grad   Specifies whether the solver should use its
                             Operator's GetGradient() method to compute the
                             Jacobian of the system. */
   KINSolver(MPI_Comm comm, int strategy, bool oper_grad = true);
#endif

   /// Destroy the associated KINSOL memory.
   virtual ~KINSolver();

   /// Set the nonlinear Operator of the system and initialize KINSOL.
   virtual void SetOperator(const Operator &op);

   /// Set the linear solver for inverting the Jacobian.
   /** @note This function assumes that Operator::GetGradient(const Vector &)
             is implemented by the Operator specified by
             SetOperator(const Operator &). */
   virtual void SetSolver(Solver &solver);

   /// Equivalent to SetSolver(solver).
   virtual void SetPreconditioner(Solver &solver) { SetSolver(solver); }

   /// Set KINSOL's scaled step tolerance.
   /** The default tolerance is U^(2/3), where U = machine unit roundoff. */
   void SetScaledStepTol(double sstol);

   /// Set maximum number of nonlinear iterations without a Jacobian update.
   /** The default is 10. */
   void SetMaxSetupCalls(int max_calls);

   /// Solve the nonlinear system F(x) = 0.
   /** This method computes the x_scale and fx_scale vectors and calls the
       other Mult(Vector&, Vector&, Vector&) const method. The x_scale vector
       is a vector of ones and values of fx_scale are determined by comparing
       the chosen relative and functional norm (i.e. absolute) tolerances.
       @param[in]     b  Not used, KINSOL always assumes zero RHS.
       @param[in,out] x  On input, initial guess, if @a #iterative_mode = true,
                         otherwise the initial guess is zero; on output, the
                         solution. */
   virtual void Mult(const Vector &b, Vector &x) const;

   /// Solve the nonlinear system F(x) = 0.
   /** Calls KINSol() to solve the nonlinear system. Before calling KINSol(),
       this functions uses the data members inherited from class IterativeSolver
       to set corresponding KINSOL options.
       @param[in,out] x         On input, initial guess, if @a #iterative_mode =
                                true, otherwise the initial guess is zero; on
                                output, the solution.
       @param[in]     x_scale   Elements of a diagonal scaling matrix D, s.t.
                                D*x has all elements roughly the same when
                                x is close to a solution.
       @param[in]     fx_scale  Elements of a diagonal scaling matrix E, s.t.
                                D*F(x) has all elements roughly the same when
                                x is not too close to a solution. */
   void Mult(Vector &x, const Vector &x_scale, const Vector &fx_scale) const;
};

}  // namespace mfem

#endif // MFEM_USE_SUNDIALS

#endif // MFEM_SUNDIALS
