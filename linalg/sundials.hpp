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
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_linearsolver.h>
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
   long saved_global_size; ///< Global vector length on last initialization.

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
   SundialsSolver() : sundials_mem(NULL), flag(0), reinit(false),
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

   /// Setup the linear system \f$ A x = b \f$.
   static int LinSysSetup(realtype t, N_Vector y, N_Vector fy, SUNMatrix A,
                          booleantype jok, booleantype *jcur,
                          realtype gamma, void *user_data, N_Vector tmp1,
                          N_Vector tmp2, N_Vector tmp3);

   /// Solve the linear system \f$ A x = b \f$.
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
       memory and set some defaults.

       If the CVODE memory has already been created, it checks if the problem
       size has changed since the last call to Init(). If the problem is the
       same then CVodeReInit() will be called in the next call to Step(). If
       the problem size has changed, the CVODE memory is freed and realloced
       for the new problem size. */
   /** @param[in] f_ The TimeDependentOperator that defines the ODE system.

       @note All other methods must be called after Init().

       @note If this method is called a second time with a different problem
       size, then any non-default user-set options will be lost and will need
       to be set again. */
   void Init(TimeDependentOperator &f_);

   /// Integrate the ODE with CVODE using the specified step mode.
   /** @param[in,out] x  On output, the solution vector at the requested output
                         time tout = @a t + @a dt.
       @param[in,out] t  On output, the output time reached.
       @param[in,out] dt On output, the last time step taken.

       @note On input, the values of @a t and @a dt are used to compute desired
       output time for the integration, tout = @a t + @a dt.
   */
   virtual void Step(Vector &x, double &t, double &dt);

   /** @brief Attach the linear system setup and solve methods from the
       TimeDependentOperator i.e., SUNImplicitSetup() and SUNImplicitSolve() to
       CVODE.
   */
   void UseMFEMLinearSolver();

   /// Attach SUNDIALS GMRES linear solver to CVODE.
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

   /// Setup the linear system \f$ A x = b \f$.
   static int LinSysSetup(realtype t, N_Vector y, N_Vector fy, SUNMatrix A,
                          SUNMatrix M, booleantype jok, booleantype *jcur,
                          realtype gamma, void *user_data, N_Vector tmp1,
                          N_Vector tmp2, N_Vector tmp3);

   /// Solve the linear system \f$ A x = b \f$.
   static int LinSysSolve(SUNLinearSolver LS, SUNMatrix A, N_Vector x,
                          N_Vector b, realtype tol);

   /// Setup the linear system \f$ M x = b \f$.
   static int MassSysSetup(realtype t, SUNMatrix M, void *user_data,
                           N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

   /// Solve the linear system \f$ M x = b \f$.
   static int MassSysSolve(SUNLinearSolver LS, SUNMatrix M, N_Vector x,
                           N_Vector b, realtype tol);

   /// Compute the matrix-vector product \f$ v = M x \f$.
   static int MassMult1(SUNMatrix M, N_Vector x, N_Vector v);

   /// Compute the matrix-vector product \f$v = M_t x \f$ at time t.
   static int MassMult2(N_Vector x, N_Vector v, realtype t,
                        void* mtimes_data);

public:
   /// Construct a serial wrapper to SUNDIALS' ARKode integrator.
   /** @param[in] type Specifies the RK method type:
                       - EXPLICIT - explicit RK method (default)
                       - IMPLICIT - implicit RK method
                       - IMEX     - implicit-explicit ARK method */
   ARKStepSolver(Type type = EXPLICIT);

#ifdef MFEM_USE_MPI
   /// Construct a parallel wrapper to SUNDIALS' ARKode integrator.
   /** @param[in] comm The MPI communicator used to partition the ODE system.
       @param[in] type Specifies the RK method type:
                       - EXPLICIT - explicit RK method (default)
                       - IMPLICIT - implicit RK method
                       - IMEX     - implicit-explicit ARK method */
   ARKStepSolver(MPI_Comm comm, Type type = EXPLICIT);
#endif

   /** @brief Initialize ARKode: calls ARKStepCreate() to create the ARKStep
       memory and set some defaults.

       If the ARKStep has already been created, it checks if the problem size
       has changed since the last call to Init(). If the problem is the same
       then ARKStepReInit() will be called in the next call to Step(). If the
       problem size has changed, the ARKStep memory is freed and realloced
       for the new problem size. */
   /** @param[in] f_ The TimeDependentOperator that defines the ODE system

       @note All other methods must be called after Init().

       @note If this method is called a second time with a different problem
       size, then any non-default user-set options will be lost and will need
       to be set again. */
   void Init(TimeDependentOperator &f_);

   /// Integrate the ODE with ARKode using the specified step mode.
   /**
       @param[in,out] x  On output, the solution vector at the requested output
                         time, tout = @a t + @a dt
       @param[in,out] t  On output, the output time reached
       @param[in,out] dt On output, the last time step taken

       @note On input, the values of @a t and @a dt are used to compute desired
       output time for the integration, tout = @a t + @a dt.
   */
   virtual void Step(Vector &x, double &t, double &dt);

   /** @brief Attach the linear system setup and solve methods from the
       TimeDependentOperator i.e., SUNImplicitSetup() and SUNImplicitSolve() to
       ARKode.
   */
   void UseMFEMLinearSolver();

   /// Attach a SUNDIALS GMRES linear solver to ARKode.
   void UseSundialsLinearSolver();

   /** @brief Attach mass matrix linear system setup, solve, and matrix-vector
       product methods from the TimeDependentOperator i.e., SUNMassSetup(),
       SUNMassSolve(), and SUNMassMult() to ARKode.

       @param[in] tdep    An integer flag indicating if the mass matrix is time
                          dependent (1) or time independent (0)
   */
   void UseMFEMMassLinearSolver(int tdep);

   /** @brief Attach the SUNDIALS GMRES linear solver and the mass matrix
       matrix-vector product method from the TimeDependentOperator i.e.,
       SUNMassMult() to ARKode to solve mass matrix systems.

       @param[in] tdep    An integer flag indicating if the mass matrix is time
                          dependent (1) or time independent (0)
   */
   void UseSundialsMassLinearSolver(int tdep);

   /// Select the ARKode step mode: ARK_NORMAL (default) or ARK_ONE_STEP.
   /** @param[in] itask  The desired step mode */
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
   /** See ARKODE documentation for all possible options, stability regions, etc.
       For example, table_num = BOGACKI_SHAMPINE_4_2_3 is 4-stage 3rd order. */
   void SetERKTableNum(int table_num);

   /// Choose a specific Butcher table for a diagonally implicit RK method.
   /** See ARKODE documentation for all possible options, stability regions, etc.
       For example, table_num = CASH_5_3_4 is 5-stage 4th order. */
   void SetIRKTableNum(int table_num);

   /// Choose a specific Butcher table for an IMEX RK method.
   /** See ARKODE documentation for all possible options, stability regions, etc.
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
   int global_strategy;               ///< KINSOL solution strategy
   bool use_oper_grad;                ///< use the Jv prod function
   mutable N_Vector y_scale, f_scale; ///< scaling vectors
   const Operator *jacobian;          ///< stores oper->GetGradient()
   int maa;                           ///< number of acceleration vectors

   /// Wrapper to compute the nonlinear residual \f$ F(u) = 0 \f$.
   static int Mult(const N_Vector u, N_Vector fu, void *user_data);

   /// Wrapper to compute the Jacobian-vector product \f$ J(u) v = Jv \f$.
   static int GradientMult(N_Vector v, N_Vector Jv, N_Vector u,
                           booleantype *new_u, void *user_data);

   /// Setup the linear system \f$ J u = b \f$.
   static int LinSysSetup(N_Vector u, N_Vector fu, SUNMatrix J,
                          void *user_data, N_Vector tmp1, N_Vector tmp2);

   /// Solve the linear system \f$ J u = b \f$.
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
       @param[in] oper_grad  Specifies whether the solver should use its
                             Operator's GetGradient() method to compute the
                             Jacobian of the system. */
   KINSolver(MPI_Comm comm, int strategy, bool oper_grad = true);
#endif

   /// Destroy the associated KINSOL memory.
   virtual ~KINSolver();

   /// Set the nonlinear Operator of the system and initialize KINSOL.
   /** @note If this method is called a second time with a different problem
       size, then non-default KINSOL-specific options will be lost and will need
       to be set again. */
   virtual void SetOperator(const Operator &op);

   /// Set the linear solver for inverting the Jacobian.
   /** @note This function assumes that Operator::GetGradient(const Vector &)
             is implemented by the Operator specified by
             SetOperator(const Operator &).

             This method must be called after SetOperator(). */
   virtual void SetSolver(Solver &solver);

   /// Equivalent to SetSolver(solver).
   virtual void SetPreconditioner(Solver &solver) { SetSolver(solver); }

   /// Set KINSOL's scaled step tolerance.
   /** The default tolerance is \f$ U^\frac{2}{3} \f$ , where
       U = machine unit roundoff.
       @note This method must be called after SetOperator(). */
   void SetScaledStepTol(double sstol);

   /// Set maximum number of nonlinear iterations without a Jacobian update.
   /** The default is 10.
       @note This method must be called after SetOperator(). */
   void SetMaxSetupCalls(int max_calls);

   /// Set the number of acceleration vectors to use with KIN_FP or KIN_PICARD.
   /** The default is 0.
       @ note This method must be called before SetOperator() to set the
       maximum size of the acceleration space. The value of @a maa can be
       altered after SetOperator() is called but it can't be higher than initial
       maximum. */
   void SetMAA(int maa);

   /// Solve the nonlinear system \f$ F(x) = 0 \f$.
   /** This method computes the x_scale and fx_scale vectors and calls the
       other Mult(Vector&, Vector&, Vector&) const method. The x_scale vector
       is a vector of ones and values of fx_scale are determined by comparing
       the chosen relative and functional norm (i.e. absolute) tolerances.
       @param[in]     b  Not used, KINSOL always assumes zero RHS
       @param[in,out] x  On input, initial guess, if @a #iterative_mode = true,
                         otherwise the initial guess is zero; on output, the
                         solution */
   virtual void Mult(const Vector &b, Vector &x) const;

   /// Solve the nonlinear system \f$ F(x) = 0 \f$.
   /** Calls KINSol() to solve the nonlinear system. Before calling KINSol(),
       this functions uses the data members inherited from class IterativeSolver
       to set corresponding KINSOL options.
       @param[in,out] x         On input, initial guess, if @a #iterative_mode =
                                true, otherwise the initial guess is zero; on
                                output, the solution
       @param[in]     x_scale   Elements of a diagonal scaling matrix D, s.t.
                                D*x has all elements roughly the same when
                                x is close to a solution
       @param[in]     fx_scale  Elements of a diagonal scaling matrix E, s.t.
                                D*F(x) has all elements roughly the same when
                                x is not too close to a solution */
   void Mult(Vector &x, const Vector &x_scale, const Vector &fx_scale) const;
};

}  // namespace mfem

#endif // MFEM_USE_SUNDIALS

#endif // MFEM_SUNDIALS
