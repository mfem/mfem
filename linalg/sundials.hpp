// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "hypre.hpp"
#endif

#include "ode.hpp"
#include "solvers.hpp"

#include <sundials/sundials_config.h>
// Check for appropriate SUNDIALS version
#if !defined(SUNDIALS_VERSION_MAJOR) || (SUNDIALS_VERSION_MAJOR < 5)
#error MFEM requires SUNDIALS version 5.0.0 or newer!
#endif
#if defined(MFEM_USE_CUDA) && ((SUNDIALS_VERSION_MAJOR == 5) && (SUNDIALS_VERSION_MINOR < 4))
#error MFEM requires SUNDIALS version 5.4.0 or newer when MFEM_USE_CUDA=TRUE!
#endif
#if defined(MFEM_USE_HIP) && ((SUNDIALS_VERSION_MAJOR == 5) && (SUNDIALS_VERSION_MINOR < 7))
#error MFEM requires SUNDIALS version 5.7.0 or newer when MFEM_USE_HIP=TRUE!
#endif
#if defined(MFEM_USE_CUDA) && !defined(SUNDIALS_NVECTOR_CUDA)
#error MFEM_USE_CUDA=TRUE requires SUNDIALS to be built with CUDA support
#endif
#if defined(MFEM_USE_HIP) && !defined(SUNDIALS_NVECTOR_HIP)
#error MFEM_USE_HIP=TRUE requires SUNDIALS to be built with HIP support
#endif
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_linearsolver.h>
#include <arkode/arkode_arkstep.h>
#include <cvodes/cvodes.h>
#include <kinsol/kinsol.h>
#if defined(MFEM_USE_CUDA)
#include <sunmemory/sunmemory_cuda.h>
#elif defined(MFEM_USE_HIP)
#include <sunmemory/sunmemory_hip.h>
#endif

#include <functional>

#if (SUNDIALS_VERSION_MAJOR < 6)

/// (DEPRECATED) Map SUNDIALS version >= 6 datatypes and constants to
/// version < 6 for backwards compatibility
using ARKODE_ERKTableID = int;
using ARKODE_DIRKTableID = int;
constexpr ARKODE_ERKTableID ARKODE_ERK_NONE = -1;
constexpr ARKODE_DIRKTableID ARKODE_DIRK_NONE = -1;
constexpr ARKODE_ERKTableID ARKODE_FEHLBERG_13_7_8 = FEHLBERG_13_7_8;

/// (DEPRECATED) There is no SUNContext in SUNDIALS version < 6 so set it to
/// arbitrary type for more compact backwards compatibility
using SUNContext = void*;

// KIN_ORTH_MGS was introduced in SUNDIALS v6; here, we define it just so that
// it can be used as the default option in the second parameter of
// KINSolver::EnableAndersonAcc -- the actual value of the parameter will be
// ignored when using SUNDIALS < v6.
#define KIN_ORTH_MGS 0

#endif // SUNDIALS_VERSION_MAJOR < 6

namespace mfem
{

#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)

// ---------------------------------------------------------------------------
// SUNMemory interface class (used when CUDA or HIP is enabled)
// ---------------------------------------------------------------------------
class SundialsMemHelper
{
   /// The actual SUNDIALS object
   SUNMemoryHelper h;

public:

   /// Default constructor -- object must be moved to
   SundialsMemHelper() = default;

   /// Require a SUNContext as an argument (rather than calling Sundials::GetContext)
   /// to avoid undefined behavior during the construction of the Sundials singleton.
   SundialsMemHelper(SUNContext context);

   /// Implement move assignment
   SundialsMemHelper(SundialsMemHelper&& that_helper);

   /// Disable copy construction
   SundialsMemHelper(const SundialsMemHelper& that_helper) = delete;

   ~SundialsMemHelper() { if (h) { SUNMemoryHelper_Destroy(h); } }

   /// Disable copy assignment
   SundialsMemHelper& operator=(const SundialsMemHelper&) = delete;

   /// Implement move assignment
   SundialsMemHelper& operator=(SundialsMemHelper&& rhs);

   /// Typecasting to SUNDIALS' SUNMemoryHelper type
   operator SUNMemoryHelper() const { return h; }

   static int SundialsMemHelper_Alloc(SUNMemoryHelper helper, SUNMemory* memptr,
                                      size_t memsize, SUNMemoryType mem_type
#if (SUNDIALS_VERSION_MAJOR >= 6)
                                      , void* queue
#endif
                                     );

   static int SundialsMemHelper_Dealloc(SUNMemoryHelper helper, SUNMemory sunmem
#if (SUNDIALS_VERSION_MAJOR >= 6)
                                        , void* queue
#endif
                                       );

};

#else // MFEM_USE_CUDA || MFEM_USE_HIP

// ---------------------------------------------------------------------------
// Dummy SUNMemory interface class (used when CUDA or HIP is not enabled)
// ---------------------------------------------------------------------------
class SundialsMemHelper
{
public:

   SundialsMemHelper() = default;

   SundialsMemHelper(SUNContext context)
   {
      // Do nothing
   }
};

#endif // MFEM_USE_CUDA || MFEM_USE_HIP


/// Singleton class for SUNContext and SundialsMemHelper objects
class Sundials
{
public:

   /// Disable copy construction
   Sundials(Sundials &other) = delete;

   /// Disable copy assignment
   void operator=(const Sundials &other) = delete;

   /// Initializes SUNContext and SundialsMemHelper objects. Should be called at
   /// the beginning of the calling program (after Mpi::Init if applicable)
   static void Init();

   /// Provides access to the SUNContext object
   static SUNContext &GetContext();

   /// Provides access to the SundialsMemHelper object
   static SundialsMemHelper &GetMemHelper();

private:
   /// Returns a reference to the singleton instance of the class.
   static Sundials &Instance();

   /// Constructor called by Sundials::Instance (does nothing for version < 6)
   Sundials();

   /// Destructor called at end of calling program (does nothing for version < 6)
   ~Sundials();

   SUNContext context;
   SundialsMemHelper memHelper;
};


/// Vector interface for SUNDIALS N_Vectors.
class SundialsNVector : public Vector
{
protected:
   int own_NVector;

   /// The actual SUNDIALS object
   N_Vector x;

   friend class SundialsSolver;

   /// Set data and length of internal N_Vector x from 'this'.
   void _SetNvecDataAndSize_(long glob_size = 0);

   /// Set data and length from the internal N_Vector x.
   void _SetDataAndSize_();

public:
   /// Creates an empty SundialsNVector.
   SundialsNVector();

   /// Creates a SundialsNVector referencing an array of doubles, owned by someone else.
   /** The pointer @a data_ can be NULL. The data array can be replaced later
       with SetData(). */
   SundialsNVector(double *data_, int size_);

   /// Creates a SundialsNVector out of a SUNDIALS N_Vector object.
   /** The N_Vector @a nv must be destroyed outside. */
   SundialsNVector(N_Vector nv);

#ifdef MFEM_USE_MPI
   /// Creates an empty SundialsNVector.
   SundialsNVector(MPI_Comm comm);

   /// Creates a SundialsNVector with the given local and global sizes.
   SundialsNVector(MPI_Comm comm, int loc_size, long glob_size);

   /// Creates a SundialsNVector referencing an array of doubles, owned by someone else.
   /** The pointer @a data_ can be NULL. The data array can be replaced later
       with SetData(). */
   SundialsNVector(MPI_Comm comm, double *data_, int loc_size, long glob_size);

   /// Creates a SundialsNVector from a HypreParVector.
   /** Ownership of the data will not change. */
   SundialsNVector(HypreParVector& vec);
#endif

   /// Calls SUNDIALS N_VDestroy function if the N_Vector is owned by 'this'.
   ~SundialsNVector();

   /// Returns the N_Vector_ID for the internal N_Vector.
   inline N_Vector_ID GetNVectorID() const { return N_VGetVectorID(x); }

   /// Returns the N_Vector_ID for the N_Vector @a x_.
   inline N_Vector_ID GetNVectorID(N_Vector x_) const { return N_VGetVectorID(x_); }

#ifdef MFEM_USE_MPI
   /// Returns the MPI communicator for the internal N_Vector x.
   inline MPI_Comm GetComm() const { return *static_cast<MPI_Comm*>(N_VGetCommunicator(x)); }

   /// Returns the MPI global length for the internal N_Vector x.
   inline long GlobalSize() const { return N_VGetLength(x); }
#endif

   /// Resize the vector to size @a s.
   void SetSize(int s, long glob_size = 0);

   /// Set the vector data.
   /// @warning This method should be called only when OwnsData() is false.
   void SetData(double *d);

   /// Set the vector data and size.
   /** The Vector does not assume ownership of the new data. The new size is
       also used as the new Capacity().
       @warning This method should be called only when OwnsData() is false. */
   void SetDataAndSize(double *d, int s, long glob_size = 0);

   /// Reset the Vector to be a reference to a sub-vector of @a base.
   inline void MakeRef(Vector &base, int offset, int s)
   {
      // Ensure that the base is registered/initialized before making an alias
      base.Read();
      Vector::MakeRef(base, offset, s);
      _SetNvecDataAndSize_();
   }

   /** @brief Reset the Vector to be a reference to a sub-vector of @a base
       without changing its current size. */
   inline void MakeRef(Vector &base, int offset)
   {
      // Ensure that the base is registered/initialized before making an alias
      base.Read();
      Vector::MakeRef(base, offset);
      _SetNvecDataAndSize_();
   }

   /// Typecasting to SUNDIALS' N_Vector type
   operator N_Vector() const { return x; }

   /// Changes the ownership of the the vector
   N_Vector StealNVector() { own_NVector = 0; return x; }

   /// Sets ownership of the internal N_Vector
   void SetOwnership(int own) { own_NVector = own; }

   /// Gets ownership of the internal N_Vector
   int GetOwnership() const { return own_NVector; }

   /// Copy assignment.
   /** @note Defining this method overwrites the implicitly defined copy
       assignment operator. */
   using Vector::operator=;

#ifdef MFEM_USE_MPI
   bool MPIPlusX() const
   { return (GetNVectorID() == SUNDIALS_NVEC_MPIPLUSX); }
#else
   bool MPIPlusX() const { return false; }
#endif

   /// Create a N_Vector.
   /** @param[in] use_device  If true, use the SUNDIALS CUDA or HIP N_Vector. */
   static N_Vector MakeNVector(bool use_device);

#ifdef MFEM_USE_MPI
   /// Create a parallel N_Vector.
   /** @param[in] comm  The MPI communicator to use.
       @param[in] use_device  If true, use the SUNDIALS CUDA or HIP N_Vector. */
   static N_Vector MakeNVector(MPI_Comm comm, bool use_device);
#endif

#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
   static bool UseManagedMemory()
   {
      return Device::GetDeviceMemoryType() == MemoryType::MANAGED;
   }
#else
   static bool UseManagedMemory()
   {
      return false;
   }
#endif

};

/// Base class for interfacing with SUNDIALS packages.
class SundialsSolver
{
protected:
   void *sundials_mem;        ///< SUNDIALS mem structure.
   mutable int flag;          ///< Last flag returned from a call to SUNDIALS.
   bool reinit;               ///< Flag to signal memory reinitialization is need.
   long saved_global_size;    ///< Global vector length on last initialization.

   SundialsNVector*   Y;      ///< State vector.
   SUNMatrix          A;      /**< Linear system A = I - gamma J,
                                   M - gamma J, or J. */
   SUNMatrix          M;      ///< Mass matrix M.
   SUNLinearSolver    LSA;    ///< Linear solver for A.
   SUNLinearSolver    LSM;    ///< Linear solver for M.
   SUNNonlinearSolver NLS;    ///< Nonlinear solver.

#ifdef MFEM_USE_MPI
   bool Parallel() const
   { return (Y->MPIPlusX() || Y->GetNVectorID() == SUNDIALS_NVEC_PARALLEL); }
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
      saved_global_size(0), Y(NULL), A(NULL), M(NULL),
      LSA(NULL), LSM(NULL), NLS(NULL) { }

   // Helper functions
   // Serial version
   void AllocateEmptyNVector(N_Vector &y);

#ifdef MFEM_USE_MPI
   void AllocateEmptyNVector(N_Vector &y, MPI_Comm comm);
#endif

public:
   /// Access the SUNDIALS memory structure.
   void *GetMem() const { return sundials_mem; }

   /// Returns the last flag returned by a call to a SUNDIALS function.
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
   int root_components; /// Number of components in gout

   /// Wrapper to compute the ODE rhs function.
   static int RHS(realtype t, const N_Vector y, N_Vector ydot, void *user_data);

   /// Setup the linear system $ A x = b $.
   static int LinSysSetup(realtype t, N_Vector y, N_Vector fy, SUNMatrix A,
                          booleantype jok, booleantype *jcur,
                          realtype gamma, void *user_data, N_Vector tmp1,
                          N_Vector tmp2, N_Vector tmp3);

   /// Solve the linear system $ A x = b $.
   static int LinSysSolve(SUNLinearSolver LS, SUNMatrix A, N_Vector x,
                          N_Vector b, realtype tol);

   /// Prototype to define root finding for CVODE
   static int root(realtype t, N_Vector y, realtype *gout, void *user_data);

   /// Typedef for root finding functions
   typedef std::function<int(realtype t, Vector y, Vector gout, CVODESolver *)>
   RootFunction;

   /// A class member to facilitate pointing to a user-specified root function
   RootFunction root_func;

   /// Typedef declaration for error weight functions
   typedef std::function<int(Vector y, Vector w, CVODESolver*)> EWTFunction;

   /// A class member to facilitate pointing to a user-specified error weight function
   EWTFunction ewt_func;

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
   void Step(Vector &x, double &t, double &dt) override;

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

   /// Set the scalar relative and vector of absolute tolerances.
   void SetSVtolerances(double reltol, Vector abstol);

   /// Initialize Root Finder.
   void SetRootFinder(int components, RootFunction func);

   /// Set the maximum time step.
   void SetMaxStep(double dt_max);

   /// Set the maximum number of time steps.
   void SetMaxNSteps(int steps);

   /// Get the number of internal steps taken so far.
   long GetNumSteps();

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
// Interface to the CVODES library -- linear multi-step methods
// ---------------------------------------------------------------------------

class CVODESSolver : public CVODESolver
{
private:
   using CVODESolver::Init;

protected:
   int ncheck; ///< number of checkpoints used so far
   int indexB; ///< backward problem index

   /// Wrapper to compute the ODE RHS Quadrature function.
   static int RHSQ(realtype t, const N_Vector y, N_Vector qdot, void *user_data);

   /// Wrapper to compute the ODE RHS backward function.
   static int RHSB(realtype t, N_Vector y,
                   N_Vector yB, N_Vector yBdot, void *user_dataB);

   /// Wrapper to compute the ODE RHS Backwards Quadrature function.
   static int RHSQB(realtype t, N_Vector y, N_Vector yB,
                    N_Vector qBdot, void *user_dataB);

   /// Error control function
   static int ewt(N_Vector y, N_Vector w, void *user_data);

   SUNMatrix          AB;   ///< Linear system A = I - gamma J, M - gamma J, or J.
   SUNLinearSolver    LSB;  ///< Linear solver for A.
   SundialsNVector*   q;    ///< Quadrature vector.
   SundialsNVector*   yB;   ///< State vector.
   SundialsNVector*   yy;   ///< State vector.
   SundialsNVector*   qB;   ///< State vector.

   /// Default scalar backward relative tolerance
   static constexpr double default_rel_tolB = 1e-4;
   /// Default scalar backward absolute tolerance
   static constexpr double default_abs_tolB = 1e-9;
   /// Default scalar backward absolute quadrature tolerance
   static constexpr double default_abs_tolQB = 1e-9;

public:
   /** Construct a serial wrapper to SUNDIALS' CVODE integrator.
       @param[in] lmm Specifies the linear multistep method, the options are:
                      CV_ADAMS - implicit methods for non-stiff systems
                      CV_BDF   - implicit methods for stiff systems */
   CVODESSolver(int lmm);

#ifdef MFEM_USE_MPI
   /** Construct a parallel wrapper to SUNDIALS' CVODE integrator.
       @param[in] comm The MPI communicator used to partition the ODE system
       @param[in] lmm  Specifies the linear multistep method, the options are:
                       CV_ADAMS - implicit methods for non-stiff systems
                       CV_BDF   - implicit methods for stiff systems */
   CVODESSolver(MPI_Comm comm, int lmm);
#endif

   /** Initialize CVODE: Calls CVodeInit() and sets some defaults. We define this
       to force the time dependent operator to be a TimeDependenAdjointOperator.
       @param[in] f_ the TimeDependentAdjointOperator that defines the ODE system

       @note All other methods must be called after Init(). */
   void Init(TimeDependentAdjointOperator &f_);

   /// Initialize the adjoint problem
   void InitB(TimeDependentAdjointOperator &f_);

   /** Integrate the ODE with CVODE using the specified step mode.

       @param[out]    x  Solution vector at the requested output time x=x(t).
       @param[in,out] t  On output, the output time reached.
       @param[in,out] dt On output, the last time step taken.

       @note On input, the values of t and dt are used to compute desired
       output time for the integration, tout = t + dt. */
   void Step(Vector &x, double &t, double &dt) override;

   /// Solve one adjoint time step
   virtual void StepB(Vector &w, double &t, double &dt);

   /// Set multiplicative error weights
   void SetWFTolerances(EWTFunction func);

   // Initialize Quadrature Integration
   void InitQuadIntegration(mfem::Vector &q0,
                            double reltolQ = 1e-3,
                            double abstolQ = 1e-8);

   /// Initialize Quadrature Integration (Adjoint)
   void InitQuadIntegrationB(mfem::Vector &qB0, double reltolQB = 1e-3,
                             double abstolQB = 1e-8);

   /// Initialize Adjoint
   void InitAdjointSolve(int steps, int interpolation);

   /// Set the maximum number of backward steps
   void SetMaxNStepsB(int mxstepsB);

   /// Get Number of Steps for ForwardSolve
   long GetNumSteps();

   /// Evaluate Quadrature
   void EvalQuadIntegration(double t, Vector &q);

   /// Evaluate Quadrature solution
   void EvalQuadIntegrationB(double t, Vector &dG_dp);

   /// Get Interpolated Forward solution y at backward integration time tB
   void GetForwardSolution(double tB, mfem::Vector & yy);

   /// Set Linear Solver for the backward problem
   void UseMFEMLinearSolverB();

   /// Use built in SUNDIALS Newton solver
   void UseSundialsLinearSolverB();

   /**
      \brief Tolerance specification functions for the adjoint problem.

      It should be called after InitB() is called.

      \param[in] reltol the scalar relative error tolerance.
      \param[in] abstol the scalar absolute error tolerance.
   */
   void SetSStolerancesB(double reltol, double abstol);

   /**
      \brief Tolerance specification functions for the adjoint problem.

      It should be called after InitB() is called.

      \param[in] reltol the scalar relative error tolerance
      \param[in] abstol the vector of absolute error tolerances
   */
   void SetSVtolerancesB(double reltol, Vector abstol);

   /// Setup the linear system A x = b
   static int LinSysSetupB(realtype t, N_Vector y, N_Vector yB, N_Vector fyB,
                           SUNMatrix A,
                           booleantype jok, booleantype *jcur,
                           realtype gamma, void *user_data, N_Vector tmp1,
                           N_Vector tmp2, N_Vector tmp3);

   /// Solve the linear system A x = b
   static int LinSysSolveB(SUNLinearSolver LS, SUNMatrix A, N_Vector x,
                           N_Vector b, realtype tol);


   /// Destroy the associated CVODES memory and SUNDIALS objects.
   virtual ~CVODESSolver();
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

   /// Setup the linear system $ A x = b $.
   static int LinSysSetup(realtype t, N_Vector y, N_Vector fy, SUNMatrix A,
                          SUNMatrix M, booleantype jok, booleantype *jcur,
                          realtype gamma, void *user_data, N_Vector tmp1,
                          N_Vector tmp2, N_Vector tmp3);

   /// Solve the linear system $ A x = b $.
   static int LinSysSolve(SUNLinearSolver LS, SUNMatrix A, N_Vector x,
                          N_Vector b, realtype tol);

   /// Setup the linear system $ M x = b $.
   static int MassSysSetup(realtype t, SUNMatrix M, void *user_data,
                           N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

   /// Solve the linear system $ M x = b $.
   static int MassSysSolve(SUNLinearSolver LS, SUNMatrix M, N_Vector x,
                           N_Vector b, realtype tol);

   /// Compute the matrix-vector product $ v = M x $.
   static int MassMult1(SUNMatrix M, N_Vector x, N_Vector v);

   /// Compute the matrix-vector product $v = M_t x $ at time t.
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
   void Step(Vector &x, real_t &t, real_t &dt) override;

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
       For example, table_id = BOGACKI_SHAMPINE_4_2_3 is 4-stage 3rd order. */
   void SetERKTableNum(ARKODE_ERKTableID table_id);

   /// Choose a specific Butcher table for a diagonally implicit RK method.
   /** See ARKODE documentation for all possible options, stability regions, etc.
       For example, table_id = CASH_5_3_4 is 5-stage 4th order. */
   void SetIRKTableNum(ARKODE_DIRKTableID table_id);

   /// Choose a specific Butcher table for an IMEX RK method.
   /** See ARKODE documentation for all possible options, stability regions, etc.
       For example, etable_id = ARK548L2SA_DIRK_8_4_5 and
       itable_id = ARK548L2SA_ERK_8_4_5 is 8-stage 5th order. */
   void SetIMEXTableNum(ARKODE_ERKTableID etable_id, ARKODE_DIRKTableID itable_id);

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
   int global_strategy;                        ///< KINSOL solution strategy
   bool use_oper_grad;                         ///< use the Jv prod function
   mutable SundialsNVector *y_scale, *f_scale; ///< scaling vectors
   const Operator *jacobian;                   ///< stores oper->GetGradient()
   int aa_n = 0;            ///< number of acceleration vectors
   int aa_delay;            ///< Anderson Acceleration delay
   double aa_damping;       ///< Anderson Acceleration damping
   int aa_orth;             ///< Anderson Acceleration orthogonalization routine
   double fp_damping = 1.0; ///< Fixed Point or Picard damping parameter
   bool jfnk = false;       ///< enable JFNK
   Vector wrk;              ///< Work vector needed for the JFNK PC
   int maxli = 5;           ///< Maximum linear iterations
   int maxlrs = 0;          ///< Maximum linear solver restarts

   /// Wrapper to compute the nonlinear residual $ F(u) = 0 $.
   static int Mult(const N_Vector u, N_Vector fu, void *user_data);

   /// Wrapper to compute the Jacobian-vector product $ J(u) v = Jv $.
   static int GradientMult(N_Vector v, N_Vector Jv, N_Vector u,
                           booleantype *new_u, void *user_data);

   /// Setup the linear system $ J u = b $.
   static int LinSysSetup(N_Vector u, N_Vector fu, SUNMatrix J,
                          void *user_data, N_Vector tmp1, N_Vector tmp2);

   /// Solve the linear system $ J u = b $.
   static int LinSysSolve(SUNLinearSolver LS, SUNMatrix J, N_Vector u,
                          N_Vector b, realtype tol);

   /// Setup the preconditioner.
   static int PrecSetup(N_Vector uu,
                        N_Vector uscale,
                        N_Vector fval,
                        N_Vector fscale,
                        void *user_data);

   /// Solve the preconditioner equation $ Pz = v $.
   static int PrecSolve(N_Vector uu,
                        N_Vector uscale,
                        N_Vector fval,
                        N_Vector fscale,
                        N_Vector vv,
                        void *user_data);

   void SetJFNKSolver(Solver &solver);

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
   void SetOperator(const Operator &op) override;

   /// Set the linear solver for inverting the Jacobian.
   /** @note This function assumes that Operator::GetGradient(const Vector &)
             is implemented by the Operator specified by
             SetOperator(const Operator &).

             This method must be called after SetOperator(). */
   void SetSolver(Solver &solver) override;

   /// Equivalent to SetSolver(solver).
   void SetPreconditioner(Solver &solver) override { SetSolver(solver); }

   /// Set KINSOL's scaled step tolerance.
   /** The default tolerance is $ U^\frac{2}{3} $ , where
       U = machine unit round-off.
       @note This method must be called after SetOperator(). */
   void SetScaledStepTol(double sstol);

   /// Set maximum number of nonlinear iterations without a Jacobian update.
   /** The default is 10.
       @note This method must be called after SetOperator(). */
   void SetMaxSetupCalls(int max_calls);

   /// Enable Anderson Acceleration for KIN_FP or KIN_PICARD.
   /** @note Has to be called once before SetOperator() in order to set up the
       maximum subspace size. Subsequent calls need @a n less or equal to the
       initial subspace size.
       @param[in] n Anderson Acceleration subspace size
       @param[in] orth Anderson Acceleration orthogonalization routine
       @param[in] delay Anderson Acceleration delay
       @param[in] damping Anderson Acceleration damping parameter valid from 0 <
       d <= 1.0. Default is 1.0 (no damping) */
   void EnableAndersonAcc(int n, int orth = KIN_ORTH_MGS, int delay = 0,
                          double damping = 1.0);

   /// Specifies the value of the damping parameter in the fixed point or Picard
   /// iteration.
   /** param[in] damping fixed point iteration or Picard damping parameter */
   void SetDamping(double damping);

   /// Set the Jacobian Free Newton Krylov flag. The default is false.
   /** This flag indicates to use JFNK as the linear solver for KINSOL. This
       means that the Solver object set in SetSolver() or SetPreconditioner() is
       used as a preconditioner for an FGMRES algorithm provided by SpFGMR from
       SUNDIALS. Furthermore, all Jacobian-vector products in the outer Krylov
       method are approximated by a difference quotient and the relative
       tolerance for the outer Krylov method is adaptive. See the KINSOL User
       Manual for details. */
   void SetJFNK(bool use_jfnk) { jfnk = use_jfnk; }

   /// Set the maximum number of linear solver iterations
   /** @note Only valid in combination with JFNK */
   void SetLSMaxIter(int m) { maxli = m; }

   /// Set the maximum number of linear solver restarts
   /** @note Only valid in combination with JFNK */
   void SetLSMaxRestarts(int m) { maxlrs = m; }

   /// Set the print level for the KINSetPrintLevel function.
   void SetPrintLevel(int print_lvl) override { print_level = print_lvl; }

   /// This method is not supported and will throw an error.
   void SetPrintLevel(PrintLevel) override;

   /// Solve the nonlinear system $ F(x) = 0 $.
   /** This method computes the x_scale and fx_scale vectors and calls the
       other Mult(Vector&, Vector&, Vector&) const method. The x_scale vector
       is a vector of ones and values of fx_scale are determined by comparing
       the chosen relative and functional norm (i.e. absolute) tolerances.
       @param[in]     b  Not used, KINSOL always assumes zero RHS
       @param[in,out] x  On input, initial guess, if @a #iterative_mode = true,
                         otherwise the initial guess is zero; on output, the
                         solution */
   void Mult(const Vector &b, Vector &x) const override;

   /// Solve the nonlinear system $ F(x) = 0 $.
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
