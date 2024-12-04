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

#include "sundials.hpp"

#ifdef MFEM_USE_SUNDIALS

#include "solvers.hpp"
#ifdef MFEM_USE_MPI
#include "hypre.hpp"
#endif

// SUNDIALS vectors
#include <nvector/nvector_serial.h>
#if defined(MFEM_USE_CUDA)
#include <nvector/nvector_cuda.h>
#elif defined(MFEM_USE_HIP)
#include <nvector/nvector_hip.h>
#endif
#ifdef MFEM_USE_MPI
#include <nvector/nvector_mpiplusx.h>
#include <nvector/nvector_parallel.h>
#endif

// SUNDIALS linear solvers
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunlinsol/sunlinsol_spfgmr.h>

// Access SUNDIALS object's content pointer
#define GET_CONTENT(X) ( X->content )

#if defined(MFEM_USE_CUDA)
#define SUN_Hip_OR_Cuda(X) X##_Cuda
#define SUN_HIP_OR_CUDA(X) X##_CUDA
#elif defined(MFEM_USE_HIP)
#define SUN_Hip_OR_Cuda(X) X##_Hip
#define SUN_HIP_OR_CUDA(X) X##_HIP
#endif

using namespace std;

#if (SUNDIALS_VERSION_MAJOR < 6)

/// (DEPRECATED) Wrapper function for backwards compatibility with SUNDIALS
/// version < 6
MFEM_DEPRECATED N_Vector N_VNewEmpty_Serial(sunindextype vec_length, SUNContext)
{
   return N_VNewEmpty_Serial(vec_length);
}

/// (DEPRECATED) Wrapper function for backwards compatibility with SUNDIALS
/// version < 6
MFEM_DEPRECATED SUNMatrix SUNMatNewEmpty(SUNContext)
{
   return SUNMatNewEmpty();
}

/// (DEPRECATED) Wrapper function for backwards compatibility with SUNDIALS
/// version < 6
MFEM_DEPRECATED SUNLinearSolver SUNLinSolNewEmpty(SUNContext)
{
   return SUNLinSolNewEmpty();
}

/// (DEPRECATED) Wrapper function for backwards compatibility with SUNDIALS
/// version < 6
MFEM_DEPRECATED SUNLinearSolver SUNLinSol_SPGMR(N_Vector y, int pretype,
                                                int maxl, SUNContext)
{
   return SUNLinSol_SPGMR(y, pretype, maxl);
}

/// (DEPRECATED) Wrapper function for backwards compatibility with SUNDIALS
/// version < 6
MFEM_DEPRECATED SUNLinearSolver SUNLinSol_SPFGMR(N_Vector y, int pretype,
                                                 int maxl, SUNContext)
{
   return SUNLinSol_SPFGMR(y, pretype, maxl);
}

/// (DEPRECATED) Wrapper function for backwards compatibility with SUNDIALS
/// version < 6
MFEM_DEPRECATED void* CVodeCreate(int lmm, SUNContext)
{
   return CVodeCreate(lmm);
}

/// (DEPRECATED) Wrapper function for backwards compatibility with SUNDIALS
/// version < 6
MFEM_DEPRECATED void* ARKStepCreate(ARKRhsFn fe, ARKRhsFn fi, sunrealtype t0,
                                    N_Vector y0, SUNContext)
{
   return ARKStepCreate(fe, fi, t0, y0);
}

/// (DEPRECATED) Wrapper function for backwards compatibility with SUNDIALS
/// version < 6
MFEM_DEPRECATED void* KINCreate(SUNContext)
{
   return KINCreate();
}

#ifdef MFEM_USE_MPI

/// (DEPRECATED) Wrapper function for backwards compatibility with SUNDIALS
/// version < 6
MFEM_DEPRECATED N_Vector N_VNewEmpty_Parallel(MPI_Comm comm,
                                              sunindextype local_length,
                                              sunindextype global_length,
                                              SUNContext)
{
   return N_VNewEmpty_Parallel(comm, local_length, global_length);
}

#endif // MFEM_USE_MPI

#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)

/// (DEPRECATED) Wrapper function for backwards compatibility with SUNDIALS
/// version < 6
MFEM_DEPRECATED N_Vector SUN_Hip_OR_Cuda(N_VNewWithMemHelp)(sunindextype length,
                                                            sunbooleantype use_managed_mem,
                                                            SUNMemoryHelper helper,
                                                            SUNContext)
{
   return SUN_Hip_OR_Cuda(N_VNewWithMemHelp)(length, use_managed_mem, helper);
}

/// (DEPRECATED) Wrapper function for backwards compatibility with SUNDIALS
/// version < 6
MFEM_DEPRECATED SUNMemoryHelper SUNMemoryHelper_NewEmpty(SUNContext)
{
   return SUNMemoryHelper_NewEmpty();
}

#endif // MFEM_USE_CUDA || MFEM_USE_HIP

#if defined(MFEM_USE_MPI) && (defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))

/// (DEPRECATED) Wrapper function for backwards compatibility with SUNDIALS
/// version < 6
MFEM_DEPRECATED N_Vector N_VMake_MPIPlusX(MPI_Comm comm, N_Vector local_vector,
                                          SUNContext)
{
   return N_VMake_MPIPlusX(comm, local_vector);
}

#endif // MFEM_USE_MPI && (MFEM_USE_CUDA || MFEM_USE_HIP)

#endif // SUNDIALS_VERSION_MAJOR < 6

#if MFEM_SUNDIALS_VERSION < 70100
#define MFEM_ARKode(FUNC) ARKStep##FUNC
#else
#define MFEM_ARKode(FUNC) ARKode##FUNC
#endif

// Macro STR(): expand the argument and add double quotes
#define STR1(s) #s
#define STR(s)  STR1(s)


namespace mfem
{

void Sundials::Init()
{
   Sundials::Instance();
}

Sundials &Sundials::Instance()
{
   static Sundials sundials;
   return sundials;
}

SUNContext &Sundials::GetContext()
{
   return Sundials::Instance().context;
}

SundialsMemHelper &Sundials::GetMemHelper()
{
   return Sundials::Instance().memHelper;
}

#if (SUNDIALS_VERSION_MAJOR >= 6)

Sundials::Sundials()
{
#ifdef MFEM_USE_MPI
   int mpi_initialized = 0;
   MPI_Initialized(&mpi_initialized);
   MPI_Comm communicator = mpi_initialized ? MPI_COMM_WORLD : MPI_COMM_NULL;
#if SUNDIALS_VERSION_MAJOR < 7
   int return_val = SUNContext_Create((void*) &communicator, &context);
#else
   int return_val = SUNContext_Create(communicator, &context);
#endif
#else // #ifdef MFEM_USE_MPI
#if SUNDIALS_VERSION_MAJOR < 7
   int return_val = SUNContext_Create(nullptr, &context);
#else
   int return_val = SUNContext_Create((SUNComm)(0), &context);
#endif
#endif // #ifdef MFEM_USE_MPI
   MFEM_VERIFY(return_val == 0, "Call to SUNContext_Create failed");
   SundialsMemHelper actual_helper(context);
   memHelper = std::move(actual_helper);
}

Sundials::~Sundials()
{
   SUNContext_Free(&context);
}

#else // SUNDIALS_VERSION_MAJOR >= 6

Sundials::Sundials()
{
   // Do nothing
}

Sundials::~Sundials()
{
   // Do nothing
}

#endif // SUNDIALS_VERSION_MAJOR >= 6

#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
SundialsMemHelper::SundialsMemHelper(SUNContext context)
{
   /* Allocate helper */
   h = SUNMemoryHelper_NewEmpty(context);

   /* Set the ops */
   h->ops->alloc     = SundialsMemHelper_Alloc;
   h->ops->dealloc   = SundialsMemHelper_Dealloc;
   h->ops->copy      = SUN_Hip_OR_Cuda(SUNMemoryHelper_Copy);
   h->ops->copyasync = SUN_Hip_OR_Cuda(SUNMemoryHelper_CopyAsync);
}

SundialsMemHelper::SundialsMemHelper(SundialsMemHelper&& that_helper)
{
   this->h = that_helper.h;
   that_helper.h = nullptr;
}

SundialsMemHelper& SundialsMemHelper::operator=(SundialsMemHelper&& rhs)
{
   this->h = rhs.h;
   rhs.h = nullptr;
   return *this;
}

int SundialsMemHelper::SundialsMemHelper_Alloc(SUNMemoryHelper helper,
                                               SUNMemory* memptr, size_t memsize,
                                               SUNMemoryType mem_type
#if (SUNDIALS_VERSION_MAJOR >= 6)
                                               , void*
#endif
                                              )
{
#if (SUNDIALS_VERSION_MAJOR < 7)
   SUNMemory sunmem = SUNMemoryNewEmpty();
#else
   SUNMemory sunmem = SUNMemoryNewEmpty(helper->sunctx);
#endif

   sunmem->ptr = NULL;
   sunmem->own = SUNTRUE;

   // memsize is the number of bytes to allocate, so we use Memory<char>
   if (mem_type == SUNMEMTYPE_HOST)
   {
      Memory<char> mem(memsize, Device::GetHostMemoryType());
      mem.SetHostPtrOwner(false);
      sunmem->ptr  = mfem::HostReadWrite(mem, memsize);
      sunmem->type = SUNMEMTYPE_HOST;
      mem.Delete();
   }
   else if (mem_type == SUNMEMTYPE_DEVICE || mem_type == SUNMEMTYPE_UVM)
   {
      Memory<char> mem(memsize, Device::GetDeviceMemoryType());
      mem.SetDevicePtrOwner(false);
      sunmem->ptr  = mfem::ReadWrite(mem, memsize);
      sunmem->type = mem_type;
      mem.Delete();
   }
   else
   {
      free(sunmem);
      return -1;
   }

   *memptr = sunmem;
   return 0;
}

int SundialsMemHelper::SundialsMemHelper_Dealloc(SUNMemoryHelper helper,
                                                 SUNMemory sunmem
#if (SUNDIALS_VERSION_MAJOR >= 6)
                                                 , void*
#endif
                                                )
{
   if (sunmem->ptr && sunmem->own && !mm.IsKnown(sunmem->ptr))
   {
      if (sunmem->type == SUNMEMTYPE_HOST)
      {
         Memory<char> mem(static_cast<char*>(sunmem->ptr), 1,
                          Device::GetHostMemoryType(), true);
         mem.Delete();
      }
      else if (sunmem->type == SUNMEMTYPE_DEVICE || sunmem->type == SUNMEMTYPE_UVM)
      {
         Memory<char> mem(static_cast<char*>(sunmem->ptr), 1,
                          Device::GetDeviceMemoryType(), true);
         mem.Delete();
      }
      else
      {
         MFEM_ABORT("Invalid SUNMEMTYPE");
         return -1;
      }
   }
   free(sunmem);
   return 0;
}

#endif // MFEM_USE_CUDA || MFEM_USE_HIP


// ---------------------------------------------------------------------------
// SUNDIALS N_Vector interface functions
// ---------------------------------------------------------------------------

void SundialsNVector::_SetNvecDataAndSize_(long glob_size)
{
#ifdef MFEM_USE_MPI
   N_Vector local_x = MPIPlusX() ? N_VGetLocalVector_MPIPlusX(x) : x;
#else
   N_Vector local_x = x;
#endif
   N_Vector_ID id = N_VGetVectorID(local_x);

   // Set the N_Vector data and length from the Vector data and size.
   switch (id)
   {
      case SUNDIALS_NVEC_SERIAL:
      {
         MFEM_ASSERT(NV_OWN_DATA_S(local_x) == SUNFALSE, "invalid serial N_Vector");
         NV_DATA_S(local_x) = HostReadWrite();
         NV_LENGTH_S(local_x) = size;
         break;
      }
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
      case SUN_HIP_OR_CUDA(SUNDIALS_NVEC):
      {
         SUN_Hip_OR_Cuda(N_VSetHostArrayPointer)(HostReadWrite(), local_x);
         SUN_Hip_OR_Cuda(N_VSetDeviceArrayPointer)(ReadWrite(), local_x);
         static_cast<SUN_Hip_OR_Cuda(N_VectorContent)>(GET_CONTENT(
                                                          local_x))->length = size;
         break;
      }
#endif
#ifdef MFEM_USE_MPI
      case SUNDIALS_NVEC_PARALLEL:
      {
         MFEM_ASSERT(NV_OWN_DATA_P(x) == SUNFALSE, "invalid parallel N_Vector");
         NV_DATA_P(x) = HostReadWrite();
         NV_LOCLENGTH_P(x) = size;
         if (glob_size == 0)
         {
            glob_size = GlobalSize();

            if (glob_size == 0 && glob_size != size)
            {
               long local_size = size;
               MPI_Allreduce(&local_size, &glob_size, 1, MPI_LONG,
                             MPI_SUM, GetComm());
            }
         }
         NV_GLOBLENGTH_P(x) = glob_size;
         break;
      }
#endif
      default:
         MFEM_ABORT("N_Vector type " << id << " is not supported");
   }

#ifdef MFEM_USE_MPI
   if (MPIPlusX())
   {
      if (glob_size == 0)
      {
         glob_size = GlobalSize();

         if (glob_size == 0 && glob_size != size)
         {
            long local_size = size;
            MPI_Allreduce(&local_size, &glob_size, 1, MPI_LONG,
                          MPI_SUM, GetComm());
         }
      }
      static_cast<N_VectorContent_MPIManyVector>(GET_CONTENT(x))->global_length =
         glob_size;
   }
#endif
}

void SundialsNVector::_SetDataAndSize_()
{
#ifdef MFEM_USE_MPI
   N_Vector local_x = MPIPlusX() ? N_VGetLocalVector_MPIPlusX(x) : x;
#else
   N_Vector local_x = x;
#endif
   N_Vector_ID id = N_VGetVectorID(local_x);

   // The SUNDIALS NVector owns the data if it created it.
   switch (id)
   {
      case SUNDIALS_NVEC_SERIAL:
      {
         const bool known = mm.IsKnown(NV_DATA_S(local_x));
         size = NV_LENGTH_S(local_x);
         data.Wrap(NV_DATA_S(local_x), size, false);
         if (known) { data.ClearOwnerFlags(); }
         break;
      }
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
      case SUN_HIP_OR_CUDA(SUNDIALS_NVEC):
      {
         double *h_ptr = SUN_Hip_OR_Cuda(N_VGetHostArrayPointer)(local_x);
         double *d_ptr = SUN_Hip_OR_Cuda(N_VGetDeviceArrayPointer)(local_x);
         const bool known = mm.IsKnown(h_ptr);
         size = SUN_Hip_OR_Cuda(N_VGetLength)(local_x);
         data.Wrap(h_ptr, d_ptr, size, Device::GetHostMemoryType(), false, false, true);
         if (known) { data.ClearOwnerFlags(); }
         UseDevice(true);
         break;
      }
#endif
#ifdef MFEM_USE_MPI
      case SUNDIALS_NVEC_PARALLEL:
      {
         const bool known = mm.IsKnown(NV_DATA_P(x));
         size = NV_LENGTH_S(x);
         data.Wrap(NV_DATA_P(x), NV_LOCLENGTH_P(x), false);
         if (known) { data.ClearOwnerFlags(); }
         break;
      }
#endif
      default:
         MFEM_ABORT("N_Vector type " << id << " is not supported");
   }
}

SundialsNVector::SundialsNVector()
   : Vector()
{
   // MFEM creates and owns the data,
   // and provides it to the SUNDIALS NVector.
   UseDevice(Device::IsAvailable());
   x = MakeNVector(UseDevice());
   own_NVector = 1;
}

SundialsNVector::SundialsNVector(double *data_, int size_)
   : Vector(data_, size_)
{
   UseDevice(Device::IsAvailable());
   x = MakeNVector(UseDevice());
   own_NVector = 1;
   _SetNvecDataAndSize_();
}

SundialsNVector::SundialsNVector(N_Vector nv)
   : x(nv)
{
   _SetDataAndSize_();
   own_NVector = 0;
}

#ifdef MFEM_USE_MPI
SundialsNVector::SundialsNVector(MPI_Comm comm)
   : Vector()
{
   UseDevice(Device::IsAvailable());
   x = MakeNVector(comm, UseDevice());
   own_NVector = 1;
}

SundialsNVector::SundialsNVector(MPI_Comm comm, int loc_size, long glob_size)
   : Vector(loc_size)
{
   UseDevice(Device::IsAvailable());
   x = MakeNVector(comm, UseDevice());
   own_NVector = 1;
   _SetNvecDataAndSize_(glob_size);
}

SundialsNVector::SundialsNVector(MPI_Comm comm, double *data_, int loc_size,
                                 long glob_size)
   : Vector(data_, loc_size)
{
   UseDevice(Device::IsAvailable());
   x = MakeNVector(comm, UseDevice());
   own_NVector = 1;
   _SetNvecDataAndSize_(glob_size);
}

SundialsNVector::SundialsNVector(HypreParVector& vec)
   : SundialsNVector(vec.GetComm(), vec.GetData(), vec.Size(), vec.GlobalSize())
{}
#endif

SundialsNVector::~SundialsNVector()
{
   if (own_NVector)
   {
#ifdef MFEM_USE_MPI
      if (MPIPlusX())
      {
         N_VDestroy(N_VGetLocalVector_MPIPlusX(x));
      }
#endif
      N_VDestroy(x);
   }
}

void SundialsNVector::SetSize(int s, long glob_size)
{
   Vector::SetSize(s);
   _SetNvecDataAndSize_(glob_size);
}

void SundialsNVector::SetData(double *d)
{
   Vector::SetData(d);
   _SetNvecDataAndSize_();
}

void SundialsNVector::SetDataAndSize(double *d, int s, long glob_size)
{
   Vector::SetDataAndSize(d, s);
   _SetNvecDataAndSize_(glob_size);
}

N_Vector SundialsNVector::MakeNVector(bool use_device)
{
   N_Vector x;
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
   if (use_device)
   {
      x = SUN_Hip_OR_Cuda(N_VNewWithMemHelp)(0, UseManagedMemory(),
                                             Sundials::GetMemHelper(),
                                             Sundials::GetContext());
   }
   else
   {
      x = N_VNewEmpty_Serial(0, Sundials::GetContext());
   }
#else
   x = N_VNewEmpty_Serial(0, Sundials::GetContext());
#endif

   MFEM_VERIFY(x, "Error in SundialsNVector::MakeNVector.");

   return x;
}

#ifdef MFEM_USE_MPI
N_Vector SundialsNVector::MakeNVector(MPI_Comm comm, bool use_device)
{
   N_Vector x;

   if (comm == MPI_COMM_NULL)
   {
      x = MakeNVector(use_device);
   }
   else
   {
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
      if (use_device)
      {
         x = N_VMake_MPIPlusX(comm, SUN_Hip_OR_Cuda(N_VNewWithMemHelp)(0,
                                                                       UseManagedMemory(),
                                                                       Sundials::GetMemHelper(),
                                                                       Sundials::GetContext()),
                              Sundials::GetContext());
      }
      else
      {
         x = N_VNewEmpty_Parallel(comm, 0, 0, Sundials::GetContext());
      }
#else
      x = N_VNewEmpty_Parallel(comm, 0, 0, Sundials::GetContext());
#endif // MFEM_USE_CUDA || MFEM_USE_HIP
   }

   MFEM_VERIFY(x, "Error in SundialsNVector::MakeNVector.");

   return x;
}
#endif // MFEM_USE_MPI


// ---------------------------------------------------------------------------
// SUNMatrix interface functions
// ---------------------------------------------------------------------------

// Return the matrix ID
static SUNMatrix_ID MatGetID(SUNMatrix)
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
static SUNLinearSolver_Type LSGetType(SUNLinearSolver)
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
int CVODESolver::RHS(sunrealtype t, const N_Vector y, N_Vector ydot,
                     void *user_data)
{
   // At this point the up-to-date data for N_Vector y and ydot is on the device.
   const SundialsNVector mfem_y(y);
   SundialsNVector mfem_ydot(ydot);

   CVODESolver *self = static_cast<CVODESolver*>(user_data);

   // Compute y' = f(t, y)
   self->f->SetTime(t);
   self->f->Mult(mfem_y, mfem_ydot);

   // Return success
   return (0);
}

int CVODESolver::root(sunrealtype t, N_Vector y, sunrealtype *gout,
                      void *user_data)
{
   CVODESolver *self = static_cast<CVODESolver*>(user_data);

   if (!self->root_func) { return CV_RTFUNC_FAIL; }

   SundialsNVector mfem_y(y);
   SundialsNVector mfem_gout(gout, self->root_components);

   return self->root_func(t, mfem_y, mfem_gout, self);
}

void CVODESolver::SetRootFinder(int components, RootFunction func)
{
   root_func = func;

   flag = CVodeRootInit(sundials_mem, components, root);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in SetRootFinder()");
}

int CVODESolver::LinSysSetup(sunrealtype t, N_Vector y, N_Vector fy,
                             SUNMatrix A, sunbooleantype jok,
                             sunbooleantype *jcur, sunrealtype gamma,
                             void*, N_Vector, N_Vector, N_Vector)
{
   // Get data from N_Vectors
   const SundialsNVector mfem_y(y);
   const SundialsNVector mfem_fy(fy);
   CVODESolver *self = static_cast<CVODESolver*>(GET_CONTENT(A));

   // Compute the linear system
   self->f->SetTime(t);
   return (self->f->SUNImplicitSetup(mfem_y, mfem_fy, jok, jcur, gamma));
}

int CVODESolver::LinSysSolve(SUNLinearSolver LS, SUNMatrix, N_Vector x,
                             N_Vector b, sunrealtype tol)
{
   SundialsNVector mfem_x(x);
   const SundialsNVector mfem_b(b);
   CVODESolver *self = static_cast<CVODESolver*>(GET_CONTENT(LS));
   // Solve the linear system
   return (self->f->SUNImplicitSolve(mfem_b, mfem_x, tol));
}

CVODESolver::CVODESolver(int lmm)
   : lmm_type(lmm), step_mode(CV_NORMAL)
{
   Y = new SundialsNVector();
}

#ifdef MFEM_USE_MPI
CVODESolver::CVODESolver(MPI_Comm comm, int lmm)
   : lmm_type(lmm), step_mode(CV_NORMAL)
{
   Y = new SundialsNVector(comm);
}
#endif

void CVODESolver::Init(TimeDependentOperator &f_)
{
   // Initialize the base class
   ODESolver::Init(f_);

   // Get the vector length
   long local_size = f_.Height();

#ifdef MFEM_USE_MPI
   long global_size = 0;
   if (Parallel())
   {
      MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM,
                    Y->GetComm());
   }
#endif

   // Get current time
   double t = f_.GetTime();

   if (sundials_mem)
   {
      // Check if the problem size has changed since the last Init() call
      int resize = 0;
      if (!Parallel())
      {
         resize = (Y->Size() != local_size);
      }
      else
      {
#ifdef MFEM_USE_MPI
         int l_resize = (Y->Size() != local_size) ||
                        (saved_global_size != global_size);
         MPI_Allreduce(&l_resize, &resize, 1, MPI_INT, MPI_LOR,
                       Y->GetComm());
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
      // Temporarily set N_Vector wrapper data to create CVODE. The correct
      // initial condition will be set using CVodeReInit() when Step() is
      // called.

      if (!Parallel())
      {
         Y->SetSize(local_size);
      }
#ifdef MFEM_USE_MPI
      else
      {
         Y->SetSize(local_size, global_size);
         saved_global_size = global_size;
      }
#endif

      // Create CVODE
      sundials_mem = CVodeCreate(lmm_type, Sundials::GetContext());
      MFEM_VERIFY(sundials_mem, "error in CVodeCreate()");

      // Initialize CVODE
      flag = CVodeInit(sundials_mem, CVODESolver::RHS, t, *Y);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeInit()");

      // Attach the CVODESolver as user-defined data
      flag = CVodeSetUserData(sundials_mem, this);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetUserData()");

      // Set default tolerances
      flag = CVodeSStolerances(sundials_mem, default_rel_tol, default_abs_tol);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetSStolerances()");

      // Attach MFEM linear solver by default
      UseMFEMLinearSolver();
   }

   // Set the reinit flag to call CVodeReInit() in the next Step() call.
   reinit = true;
}

void CVODESolver::Step(Vector &x, double &t, double &dt)
{
   Y->MakeRef(x, 0, x.Size());
   MFEM_VERIFY(Y->Size() == x.Size(), "size mismatch");

   // Reinitialize CVODE memory if needed
   if (reinit)
   {
      flag = CVodeReInit(sundials_mem, t, *Y);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeReInit()");
      // reset flag
      reinit = false;
   }

   // Integrate the system
   double tout = t + dt;
   flag = CVode(sundials_mem, tout, *Y, &t, step_mode);
   MFEM_VERIFY(flag >= 0, "error in CVode()");

   // Make sure host is up to date
   Y->HostRead();

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
   LSA = SUNLinSolNewEmpty(Sundials::GetContext());
   MFEM_VERIFY(LSA, "error in SUNLinSolNewEmpty()");

   LSA->content      = this;
   LSA->ops->gettype = LSGetType;
   LSA->ops->solve   = CVODESolver::LinSysSolve;
   LSA->ops->free    = LSFree;

   A = SUNMatNewEmpty(Sundials::GetContext());
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
   LSA = SUNLinSol_SPGMR(*Y, SUN_PREC_NONE, 0, Sundials::GetContext());
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

void CVODESolver::SetSVtolerances(double reltol, Vector abstol)
{
   MFEM_VERIFY(abstol.Size() == f->Height(),
               "abs tolerance is not the same size.");

   SundialsNVector mfem_abstol;
   mfem_abstol.MakeRef(abstol, 0, abstol.Size());

   flag = CVodeSVtolerances(sundials_mem, reltol, mfem_abstol);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSVtolerances()");
}

void CVODESolver::SetMaxStep(double dt_max)
{
   flag = CVodeSetMaxStep(sundials_mem, dt_max);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetMaxStep()");
}

void CVODESolver::SetMaxNSteps(int mxsteps)
{
   flag = CVodeSetMaxNumSteps(sundials_mem, mxsteps);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetMaxNumSteps()");
}

long CVODESolver::GetNumSteps()
{
   long nsteps;
   flag = CVodeGetNumSteps(sundials_mem, &nsteps);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeGetNumSteps()");
   return nsteps;
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
   delete Y;
   SUNMatDestroy(A);
   SUNLinSolFree(LSA);
   SUNNonlinSolFree(NLS);
   CVodeFree(&sundials_mem);
}

// ---------------------------------------------------------------------------
// CVODESSolver interface
// ---------------------------------------------------------------------------

CVODESSolver::CVODESSolver(int lmm) :
   CVODESolver(lmm),
   ncheck(0),
   indexB(0),
   AB(nullptr),
   LSB(nullptr)
{
   q  = new SundialsNVector();
   qB = new SundialsNVector();
   yB = new SundialsNVector();
   yy = new SundialsNVector();
}

#ifdef MFEM_USE_MPI
CVODESSolver::CVODESSolver(MPI_Comm comm, int lmm) :
   CVODESolver(comm, lmm),
   ncheck(0),
   indexB(0),
   AB(nullptr),
   LSB(nullptr)
{
   q  = new SundialsNVector(comm);
   qB = new SundialsNVector(comm);
   yB = new SundialsNVector(comm);
   yy = new SundialsNVector(comm);
}
#endif

void CVODESSolver::EvalQuadIntegration(double t, Vector &Q)
{
   MFEM_VERIFY(t <= f->GetTime(), "t > current forward solver time");

   flag = CVodeGetQuad(sundials_mem, &t, *q);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeGetQuad()");

   Q.Set(1., *q);
}

void CVODESSolver::EvalQuadIntegrationB(double t, Vector &dG_dp)
{
   MFEM_VERIFY(t <= f->GetTime(), "t > current forward solver time");

   flag = CVodeGetQuadB(sundials_mem, indexB, &t, *qB);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeGetQuadB()");

   dG_dp.Set(-1., *qB);
}

void CVODESSolver::GetForwardSolution(double tB, mfem::Vector &yyy)
{
   yy->MakeRef(yyy, 0, yyy.Size());

   flag = CVodeGetAdjY(sundials_mem, tB, *yy);
   MFEM_VERIFY(flag >= 0, "error in CVodeGetAdjY()");
}

// Implemented to enforce type checking for TimeDependentAdjointOperator
void CVODESSolver::Init(TimeDependentAdjointOperator &f_)
{
   CVODESolver::Init(f_);
}

void CVODESSolver::InitB(TimeDependentAdjointOperator &f_)
{
   long local_size = f_.GetAdjointHeight();

   // Get current time
   double tB = f_.GetTime();

   yB->SetSize(local_size);

   // Create the solver memory
   flag = CVodeCreateB(sundials_mem, CV_BDF, &indexB);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeCreateB()");

   // Initialize
   flag = CVodeInitB(sundials_mem, indexB, RHSB, tB, *yB);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeInit()");

   // Attach the CVODESSolver as user-defined data
   flag = CVodeSetUserDataB(sundials_mem, indexB, this);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetUserDataB()");

   // Set default tolerances
   flag = CVodeSStolerancesB(sundials_mem, indexB, default_rel_tolB,
                             default_abs_tolB);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetSStolerancesB()");

   // Attach MFEM linear solver by default
   UseMFEMLinearSolverB();

   // Set the reinit flag to call CVodeReInit() in the next Step() call.
   reinit = true;
}

void CVODESSolver::InitAdjointSolve(int steps, int interpolation)
{
   flag = CVodeAdjInit(sundials_mem, steps, interpolation);
   MFEM_VERIFY(flag == CV_SUCCESS, "Error in CVodeAdjInit");
}

void CVODESSolver::SetMaxNStepsB(int mxstepsB)
{
   flag = CVodeSetMaxNumStepsB(sundials_mem, indexB, mxstepsB);
   MFEM_VERIFY(flag == CV_SUCCESS, "Error in CVodeSetMaxNumStepsB()");
}

void CVODESSolver::InitQuadIntegration(mfem::Vector &q0, double reltolQ,
                                       double abstolQ)
{
   q->MakeRef(q0, 0, q0.Size());

   flag = CVodeQuadInit(sundials_mem, RHSQ, *q);
   MFEM_VERIFY(flag == CV_SUCCESS, "Error in CVodeQuadInit()");

   flag = CVodeSetQuadErrCon(sundials_mem, SUNTRUE);
   MFEM_VERIFY(flag == CV_SUCCESS, "Error in CVodeSetQuadErrCon");

   flag = CVodeQuadSStolerances(sundials_mem, reltolQ, abstolQ);
   MFEM_VERIFY(flag == CV_SUCCESS, "Error in CVodeQuadSStolerances");
}

void CVODESSolver::InitQuadIntegrationB(mfem::Vector &qB0, double reltolQB,
                                        double abstolQB)
{
   qB->MakeRef(qB0, 0, qB0.Size());

   flag = CVodeQuadInitB(sundials_mem, indexB, RHSQB, *qB);
   MFEM_VERIFY(flag == CV_SUCCESS, "Error in CVodeQuadInitB()");

   flag = CVodeSetQuadErrConB(sundials_mem, indexB, SUNTRUE);
   MFEM_VERIFY(flag == CV_SUCCESS, "Error in CVodeSetQuadErrConB");

   flag = CVodeQuadSStolerancesB(sundials_mem, indexB, reltolQB, abstolQB);
   MFEM_VERIFY(flag == CV_SUCCESS, "Error in CVodeQuadSStolerancesB");
}

void CVODESSolver::UseMFEMLinearSolverB()
{
   // Free any existing linear solver
   if (AB != NULL)   { SUNMatDestroy(AB); AB = NULL; }
   if (LSB != NULL) { SUNLinSolFree(LSB); LSB = NULL; }

   // Wrap linear solver as SUNLinearSolver and SUNMatrix
   LSB = SUNLinSolNewEmpty(Sundials::GetContext());
   MFEM_VERIFY(LSB, "error in SUNLinSolNewEmpty()");

   LSB->content         = this;
   LSB->ops->gettype    = LSGetType;
   LSB->ops->solve      = CVODESSolver::LinSysSolveB; // JW change
   LSB->ops->free       = LSFree;

   AB = SUNMatNewEmpty(Sundials::GetContext());
   MFEM_VERIFY(AB, "error in SUNMatNewEmpty()");

   AB->content      = this;
   AB->ops->getid   = MatGetID;
   AB->ops->destroy = MatDestroy;

   // Attach the linear solver and matrix
   flag = CVodeSetLinearSolverB(sundials_mem, indexB, LSB, AB);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetLinearSolverB()");

   // Set the linear system evaluation function
   flag = CVodeSetLinSysFnB(sundials_mem, indexB,
                            CVODESSolver::LinSysSetupB); // JW change
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetLinSysFn()");
}

void CVODESSolver::UseSundialsLinearSolverB()
{
   // Free any existing matrix and linear solver
   if (AB != NULL)   { SUNMatDestroy(AB); AB = NULL; }
   if (LSB != NULL) { SUNLinSolFree(LSB); LSB = NULL; }

   // Set default linear solver (Newton is the default Nonlinear Solver)
   LSB = SUNLinSol_SPGMR(*yB, SUN_PREC_NONE, 0, Sundials::GetContext());
   MFEM_VERIFY(LSB, "error in SUNLinSol_SPGMR()");

   /* Attach the matrix and linear solver */
   flag = CVodeSetLinearSolverB(sundials_mem, indexB, LSB, NULL);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetLinearSolverB()");
}

int CVODESSolver::LinSysSetupB(sunrealtype t, N_Vector y, N_Vector yB,
                               N_Vector fyB, SUNMatrix AB,
                               sunbooleantype jokB, sunbooleantype *jcurB,
                               sunrealtype gammaB, void *user_data,
                               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
   // Get data from N_Vectors
   const SundialsNVector mfem_y(y);
   const SundialsNVector mfem_yB(yB);
   SundialsNVector mfem_fyB(fyB);
   CVODESSolver *self = static_cast<CVODESSolver*>(GET_CONTENT(AB));
   TimeDependentAdjointOperator * f = static_cast<TimeDependentAdjointOperator *>
                                      (self->f);
   f->SetTime(t);
   // Compute the linear system
   return (f->SUNImplicitSetupB(t, mfem_y, mfem_yB, mfem_fyB, jokB, jcurB,
                                gammaB));
}

int CVODESSolver::LinSysSolveB(SUNLinearSolver LS, SUNMatrix AB, N_Vector yB,
                               N_Vector Rb, sunrealtype tol)
{
   SundialsNVector mfem_yB(yB);
   const SundialsNVector mfem_Rb(Rb);
   CVODESSolver *self = static_cast<CVODESSolver*>(GET_CONTENT(LS));
   TimeDependentAdjointOperator * f = static_cast<TimeDependentAdjointOperator *>
                                      (self->f);
   // Solve the linear system
   int ret = f->SUNImplicitSolveB(mfem_yB, mfem_Rb, tol);
   return (ret);
}

void CVODESSolver::SetSStolerancesB(double reltol, double abstol)
{
   flag = CVodeSStolerancesB(sundials_mem, indexB, reltol, abstol);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSStolerancesB()");
}

void CVODESSolver::SetSVtolerancesB(double reltol, Vector abstol)
{
   MFEM_VERIFY(abstol.Size() == f->Height(),
               "abs tolerance is not the same size.");

   SundialsNVector mfem_abstol;
   mfem_abstol.MakeRef(abstol, 0, abstol.Size());

   flag = CVodeSVtolerancesB(sundials_mem, indexB, reltol, mfem_abstol);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSVtolerancesB()");
}

void CVODESSolver::SetWFTolerances(EWTFunction func)
{
   ewt_func = func;
   CVodeWFtolerances(sundials_mem, ewt);
}

// CVODESSolver static functions

int CVODESSolver::RHSQ(sunrealtype t, const N_Vector y, N_Vector qdot,
                       void *user_data)
{
   CVODESSolver *self = static_cast<CVODESSolver*>(user_data);
   const SundialsNVector mfem_y(y);
   SundialsNVector mfem_qdot(qdot);
   TimeDependentAdjointOperator * f = static_cast<TimeDependentAdjointOperator *>
                                      (self->f);
   f->SetTime(t);
   f->QuadratureIntegration(mfem_y, mfem_qdot);
   return 0;
}

int CVODESSolver::RHSQB(sunrealtype t, N_Vector y, N_Vector yB, N_Vector qBdot,
                        void *user_dataB)
{
   CVODESSolver *self = static_cast<CVODESSolver*>(user_dataB);
   SundialsNVector mfem_y(y);
   SundialsNVector mfem_yB(yB);
   SundialsNVector mfem_qBdot(qBdot);
   TimeDependentAdjointOperator * f = static_cast<TimeDependentAdjointOperator *>
                                      (self->f);
   f->SetTime(t);
   f->QuadratureSensitivityMult(mfem_y, mfem_yB, mfem_qBdot);
   return 0;
}

int CVODESSolver::RHSB(sunrealtype t, N_Vector y, N_Vector yB, N_Vector yBdot,
                       void *user_dataB)
{
   CVODESSolver *self = static_cast<CVODESSolver*>(user_dataB);
   SundialsNVector mfem_y(y);
   SundialsNVector mfem_yB(yB);
   SundialsNVector mfem_yBdot(yBdot);

   mfem_yBdot = 0.;
   TimeDependentAdjointOperator * f = static_cast<TimeDependentAdjointOperator *>
                                      (self->f);
   f->SetTime(t);
   f->AdjointRateMult(mfem_y, mfem_yB, mfem_yBdot);
   return 0;
}

int CVODESSolver::ewt(N_Vector y, N_Vector w, void *user_data)
{
   CVODESSolver *self = static_cast<CVODESSolver*>(user_data);

   SundialsNVector mfem_y(y);
   SundialsNVector mfem_w(w);

   return self->ewt_func(mfem_y, mfem_w, self);
}

// Pretty much a copy of CVODESolver::Step except we use CVodeF instead of CVode
void CVODESSolver::Step(Vector &x, double &t, double &dt)
{
   Y->MakeRef(x, 0, x.Size());
   MFEM_VERIFY(Y->Size() == x.Size(), "size mismatch");

   // Reinitialize CVODE memory if needed, initializes the N_Vector y with x
   if (reinit)
   {
      flag = CVodeReInit(sundials_mem, t, *Y);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeReInit()");

      // reset flag
      reinit = false;
   }

   // Integrate the system
   double tout = t + dt;
   flag = CVodeF(sundials_mem, tout, *Y, &t, step_mode, &ncheck);
   MFEM_VERIFY(flag >= 0, "error in CVodeF()");

   // Make sure host is up to date
   Y->HostRead();

   // Return the last incremental step size
   flag = CVodeGetLastStep(sundials_mem, &dt);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeGetLastStep()");
}

void CVODESSolver::StepB(Vector &xB, double &tB, double &dtB)
{
   yB->MakeRef(xB, 0, xB.Size());
   MFEM_VERIFY(yB->Size() == xB.Size(), "");

   // Reinitialize CVODE memory if needed
   if (reinit)
   {
      flag = CVodeReInitB(sundials_mem, indexB, tB, *yB);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeReInit()");

      // reset flag
      reinit = false;
   }

   // Integrate the system
   double tout = tB - dtB;
   flag = CVodeB(sundials_mem, tout, step_mode);
   MFEM_VERIFY(flag >= 0, "error in CVodeB()");

   // Call CVodeGetB to get yB of the backward ODE problem.
   flag = CVodeGetB(sundials_mem, indexB, &tB, *yB);
   MFEM_VERIFY(flag >= 0, "error in CVodeGetB()");

   // Make sure host is up to date
   yB->HostRead();
}

CVODESSolver::~CVODESSolver()
{
   delete yB;
   delete yy;
   delete qB;
   delete q;
   SUNMatDestroy(AB);
   SUNLinSolFree(LSB);
}


// ---------------------------------------------------------------------------
// ARKStep interface
// ---------------------------------------------------------------------------

int ARKStepSolver::RHS1(sunrealtype t, const N_Vector y, N_Vector result,
                        void *user_data)
{
   // Get data from N_Vectors
   const SundialsNVector mfem_y(y);
   SundialsNVector mfem_result(result);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(user_data);

   // Compute either f(t, y) in one of
   //   1. y' = f(t, y)
   //   2. M y' = f(t, y)
   // or fe(t, y) in one of
   //   1. y' = fe(t, y) + fi(t, y)
   //   2. M y' = fe(t, y) + fi(t, y)
   self->f->SetTime(t);
   if (self->rk_type == IMEX)
   {
      self->f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
   }
   if (self->f->isExplicit()) // ODE is in form 1
   {
      self->f->Mult(mfem_y, mfem_result);
   }
   else // ODE is in form 2
   {
      self->f->ExplicitMult(mfem_y, mfem_result);
   }

   // Return success
   return (0);
}

int ARKStepSolver::RHS2(sunrealtype t, const N_Vector y, N_Vector result,
                        void *user_data)
{
   // Get data from N_Vectors
   const SundialsNVector mfem_y(y);
   SundialsNVector mfem_result(result);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(user_data);

   // Compute fi(t, y) in one of
   //   1. y' = fe(t, y) + fi(t, y)       (ODE is expressed in EXPLICIT form)
   //   2. M y' = fe(t, y) + fi(y, t)     (ODE is expressed in IMPLICIT form)
   self->f->SetTime(t);
   self->f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   if (self->f->isExplicit())
   {
      self->f->Mult(mfem_y, mfem_result);
   }
   else
   {
      self->f->ExplicitMult(mfem_y, mfem_result);
   }

   // Return success
   return (0);
}

int ARKStepSolver::LinSysSetup(sunrealtype t, N_Vector y, N_Vector fy,
                               SUNMatrix A, SUNMatrix, sunbooleantype jok,
                               sunbooleantype *jcur, sunrealtype gamma,
                               void*, N_Vector, N_Vector, N_Vector)
{
   // Get data from N_Vectors
   const SundialsNVector mfem_y(y);
   const SundialsNVector mfem_fy(fy);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(A));

   // Compute the linear system
   self->f->SetTime(t);
   if (self->rk_type == IMEX)
   {
      self->f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   }
   return (self->f->SUNImplicitSetup(mfem_y, mfem_fy, jok, jcur, gamma));
}

int ARKStepSolver::LinSysSolve(SUNLinearSolver LS, SUNMatrix, N_Vector x,
                               N_Vector b, sunrealtype tol)
{
   SundialsNVector mfem_x(x);
   const SundialsNVector mfem_b(b);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(LS));

   // Solve the linear system
   if (self->rk_type == IMEX)
   {
      self->f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   }
   return (self->f->SUNImplicitSolve(mfem_b, mfem_x, tol));
}

int ARKStepSolver::MassSysSetup(sunrealtype t, SUNMatrix M,
                                void*, N_Vector, N_Vector, N_Vector)
{
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(M));

   // Compute the mass matrix system
   self->f->SetTime(t);
   return (self->f->SUNMassSetup());
}

int ARKStepSolver::MassSysSolve(SUNLinearSolver LS, SUNMatrix, N_Vector x,
                                N_Vector b, sunrealtype tol)
{
   SundialsNVector mfem_x(x);
   const SundialsNVector mfem_b(b);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(LS));

   // Solve the mass matrix system
   return (self->f->SUNMassSolve(mfem_b, mfem_x, tol));
}

int ARKStepSolver::MassMult1(SUNMatrix M, N_Vector x, N_Vector v)
{
   const SundialsNVector mfem_x(x);
   SundialsNVector mfem_v(v);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(M));

   // Compute the mass matrix-vector product
   return (self->f->SUNMassMult(mfem_x, mfem_v));
}

int ARKStepSolver::MassMult2(N_Vector x, N_Vector v, sunrealtype t,
                             void* mtimes_data)
{
   const SundialsNVector mfem_x(x);
   SundialsNVector mfem_v(v);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(mtimes_data);

   // Compute the mass matrix-vector product
   self->f->SetTime(t);
   return (self->f->SUNMassMult(mfem_x, mfem_v));
}

ARKStepSolver::ARKStepSolver(Type type)
   : rk_type(type), step_mode(ARK_NORMAL),
     use_implicit(type == IMPLICIT || type == IMEX)
{
   Y = new SundialsNVector();
}

#ifdef MFEM_USE_MPI
ARKStepSolver::ARKStepSolver(MPI_Comm comm, Type type)
   : rk_type(type), step_mode(ARK_NORMAL),
     use_implicit(type == IMPLICIT || type == IMEX)
{
   Y = new SundialsNVector(comm);
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
                    Y->GetComm());
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
         resize = (Y->Size() != local_size);
      }
      else
      {
#ifdef MFEM_USE_MPI
         int l_resize = (Y->Size() != local_size) ||
                        (saved_global_size != global_size);
         MPI_Allreduce(&l_resize, &resize, 1, MPI_INT, MPI_LOR,
                       Y->GetComm());
#endif
      }

      // Free existing solver memory and re-create with new vector size
      if (resize)
      {
         MFEM_ARKode(Free)(&sundials_mem);
         sundials_mem = NULL;
      }
   }

   if (!sundials_mem)
   {
      if (!Parallel())
      {
         Y->SetSize(local_size);
      }
#ifdef MFEM_USE_MPI
      else
      {
         Y->SetSize(local_size, global_size);
         saved_global_size = global_size;
      }
#endif

      // Create ARKStep memory
      if (rk_type == IMPLICIT)
      {
         sundials_mem = ARKStepCreate(NULL, ARKStepSolver::RHS1, t, *Y,
                                      Sundials::GetContext());
      }
      else if (rk_type == EXPLICIT)
      {
         sundials_mem = ARKStepCreate(ARKStepSolver::RHS1, NULL, t, *Y,
                                      Sundials::GetContext());
      }
      else
      {
         sundials_mem = ARKStepCreate(ARKStepSolver::RHS1, ARKStepSolver::RHS2,
                                      t, *Y, Sundials::GetContext());
      }
      MFEM_VERIFY(sundials_mem, "error in ARKStepCreate()");

      // Attach the ARKStepSolver as user-defined data
      flag = MFEM_ARKode(SetUserData)(sundials_mem, this);
      MFEM_VERIFY(flag == ARK_SUCCESS,
                  "error in " STR(MFEM_ARKode(SetUserData)) "()");

      // Set default tolerances
      flag = MFEM_ARKode(SStolerances)(sundials_mem, default_rel_tol,
                                       default_abs_tol);
      MFEM_VERIFY(flag == ARK_SUCCESS,
                  "error in " STR(MFEM_ARKode(SStolerances)) "()");

      // If implicit, attach MFEM linear solver by default
      if (use_implicit) { UseMFEMLinearSolver(); }
   }

   // Set the reinit flag to call ARKStepReInit() in the next Step() call.
   reinit = true;
}

void ARKStepSolver::Step(Vector &x, real_t &t, real_t &dt)
{
   Y->MakeRef(x, 0, x.Size());
   MFEM_VERIFY(Y->Size() == x.Size(), "size mismatch");

   // Reinitialize ARKStep memory if needed
   if (reinit)
   {
      if (rk_type == IMPLICIT)
      {
         flag = ARKStepReInit(sundials_mem, NULL, ARKStepSolver::RHS1, t, *Y);
      }
      else if (rk_type == EXPLICIT)
      {
         flag = ARKStepReInit(sundials_mem, ARKStepSolver::RHS1, NULL, t, *Y);
      }
      else
      {
         flag = ARKStepReInit(sundials_mem,
                              ARKStepSolver::RHS1, ARKStepSolver::RHS2, t, *Y);
      }
      MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepReInit()");

      // reset flag
      reinit = false;
   }

   // Integrate the system
   double tout = t + dt;
   flag = MFEM_ARKode(Evolve)(sundials_mem, tout, *Y, &t, step_mode);
   MFEM_VERIFY(flag >= 0, "error in " STR(MFEM_ARKode(Evolve)) "()");

   // Make sure host is up to date
   Y->HostRead();

   // Return the last incremental step size
   flag = MFEM_ARKode(GetLastStep)(sundials_mem, &dt);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(GetLastStep)) "()");
}

void ARKStepSolver::UseMFEMLinearSolver()
{
   // Free any existing matrix and linear solver
   if (A != NULL)   { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Wrap linear solver as SUNLinearSolver and SUNMatrix
   LSA = SUNLinSolNewEmpty(Sundials::GetContext());
   MFEM_VERIFY(LSA, "error in SUNLinSolNewEmpty()");

   LSA->content      = this;
   LSA->ops->gettype = LSGetType;
   LSA->ops->solve   = ARKStepSolver::LinSysSolve;
   LSA->ops->free    = LSFree;

   A = SUNMatNewEmpty(Sundials::GetContext());
   MFEM_VERIFY(A, "error in SUNMatNewEmpty()");

   A->content      = this;
   A->ops->getid   = MatGetID;
   A->ops->destroy = MatDestroy;

   // Attach the linear solver and matrix
   flag = MFEM_ARKode(SetLinearSolver)(sundials_mem, LSA, A);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(SetLinearSolver)) "()");

   // Set the linear system evaluation function
   flag = MFEM_ARKode(SetLinSysFn)(sundials_mem, ARKStepSolver::LinSysSetup);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(SetLinSysFn)) "()");
}

void ARKStepSolver::UseSundialsLinearSolver()
{
   // Free any existing matrix and linear solver
   if (A != NULL)   { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Create linear solver
   LSA = SUNLinSol_SPGMR(*Y, SUN_PREC_NONE, 0, Sundials::GetContext());
   MFEM_VERIFY(LSA, "error in SUNLinSol_SPGMR()");

   // Attach linear solver
   flag = MFEM_ARKode(SetLinearSolver)(sundials_mem, LSA, NULL);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(SetLinearSolver)) "()");
}

void ARKStepSolver::UseMFEMMassLinearSolver(int tdep)
{
   // Free any existing matrix and linear solver
   if (M != NULL)   { SUNMatDestroy(M); M = NULL; }
   if (LSM != NULL) { SUNLinSolFree(LSM); LSM = NULL; }

   // Wrap linear solver as SUNLinearSolver and SUNMatrix
   LSM = SUNLinSolNewEmpty(Sundials::GetContext());
   MFEM_VERIFY(LSM, "error in SUNLinSolNewEmpty()");

   LSM->content      = this;
   LSM->ops->gettype = LSGetType;
   LSM->ops->solve   = ARKStepSolver::MassSysSolve;
   LSM->ops->free    = LSFree;

   M = SUNMatNewEmpty(Sundials::GetContext());
   MFEM_VERIFY(M, "error in SUNMatNewEmpty()");

   M->content      = this;
   M->ops->getid   = MatGetID;
   M->ops->matvec  = ARKStepSolver::MassMult1;
   M->ops->destroy = MatDestroy;

   // Attach the linear solver and matrix
   flag = MFEM_ARKode(SetMassLinearSolver)(sundials_mem, LSM, M, tdep);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(SetMassLinearSolver)) "()");

   // Set the linear system function
   flag = MFEM_ARKode(SetMassFn)(sundials_mem, ARKStepSolver::MassSysSetup);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(SetMassFn)) "()");

   // Check that the ODE is not expressed in EXPLICIT form
   MFEM_VERIFY(!f->isExplicit(), "ODE operator is expressed in EXPLICIT form")
}

void ARKStepSolver::UseSundialsMassLinearSolver(int tdep)
{
   // Free any existing matrix and linear solver
   if (M != NULL)   { SUNMatDestroy(A); M = NULL; }
   if (LSM != NULL) { SUNLinSolFree(LSM); LSM = NULL; }

   // Create linear solver
   LSM = SUNLinSol_SPGMR(*Y, SUN_PREC_NONE, 0, Sundials::GetContext());
   MFEM_VERIFY(LSM, "error in SUNLinSol_SPGMR()");

   // Attach linear solver
   flag = MFEM_ARKode(SetMassLinearSolver)(sundials_mem, LSM, NULL, tdep);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(SetMassLinearSolver)) "()");

   // Attach matrix multiplication function
   flag = MFEM_ARKode(SetMassTimes)(sundials_mem, NULL,
                                    ARKStepSolver::MassMult2, this);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(SetMassTimes)) "()");

   // Check that the ODE is not expressed in EXPLICIT form
   MFEM_VERIFY(!f->isExplicit(), "ODE operator is expressed in EXPLICIT form")
}

void ARKStepSolver::SetStepMode(int itask)
{
   step_mode = itask;
}

void ARKStepSolver::SetSStolerances(double reltol, double abstol)
{
   flag = MFEM_ARKode(SStolerances)(sundials_mem, reltol, abstol);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(SStolerances)) "()");
}

void ARKStepSolver::SetMaxStep(double dt_max)
{
   flag = MFEM_ARKode(SetMaxStep)(sundials_mem, dt_max);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(SetMaxStep)) "()");
}

void ARKStepSolver::SetOrder(int order)
{
   flag = MFEM_ARKode(SetOrder)(sundials_mem, order);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(SetOrder)) "()");
}

void ARKStepSolver::SetERKTableNum(ARKODE_ERKTableID table_id)
{
   flag = ARKStepSetTableNum(sundials_mem, ARKODE_DIRK_NONE, table_id);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetTableNum()");
}

void ARKStepSolver::SetIRKTableNum(ARKODE_DIRKTableID table_id)
{
   flag = ARKStepSetTableNum(sundials_mem, table_id, ARKODE_ERK_NONE);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetTableNum()");
}

void ARKStepSolver::SetIMEXTableNum(ARKODE_ERKTableID etable_id,
                                    ARKODE_DIRKTableID itable_id)
{
   flag = ARKStepSetTableNum(sundials_mem, itable_id, etable_id);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetTableNum()");
}

void ARKStepSolver::SetFixedStep(double dt)
{
   flag = MFEM_ARKode(SetFixedStep)(sundials_mem, dt);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(SetFixedStep)) "()");
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

   flag = MFEM_ARKode(GetStepStats)(sundials_mem,
                                    &nsteps,
                                    &hinused,
                                    &hlast,
                                    &hcur,
                                    &tcur);

   // Get nonlinear solver stats
   flag = MFEM_ARKode(GetNonlinSolvStats)(sundials_mem,
                                          &nniters,
                                          &nncfails);
   MFEM_VERIFY(flag == ARK_SUCCESS,
               "error in " STR(MFEM_ARKode(GetNonlinSolvStats)) "()");

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
   delete Y;
   SUNMatDestroy(A);
   SUNLinSolFree(LSA);
   SUNNonlinSolFree(NLS);
   MFEM_ARKode(Free)(&sundials_mem);
}

// ---------------------------------------------------------------------------
// KINSOL interface
// ---------------------------------------------------------------------------

// Wrapper for evaluating the nonlinear residual F(u) = 0
int KINSolver::Mult(const N_Vector u, N_Vector fu, void *user_data)
{
   const SundialsNVector mfem_u(u);
   SundialsNVector mfem_fu(fu);
   KINSolver *self = static_cast<KINSolver*>(user_data);

   // Compute the non-linear action F(u).
   self->oper->Mult(mfem_u, mfem_fu);

   // Return success
   return 0;
}

// Wrapper for computing Jacobian-vector products
int KINSolver::GradientMult(N_Vector v, N_Vector Jv, N_Vector u,
                            sunbooleantype *new_u, void *user_data)
{
   const SundialsNVector mfem_v(v);
   SundialsNVector mfem_Jv(Jv);
   KINSolver *self = static_cast<KINSolver*>(user_data);

   // Update Jacobian information if needed
   if (*new_u)
   {
      const SundialsNVector mfem_u(u);
      self->jacobian = &self->oper->GetGradient(mfem_u);
      *new_u = SUNFALSE;
   }

   // Compute the Jacobian-vector product
   self->jacobian->Mult(mfem_v, mfem_Jv);

   // Return success
   return 0;
}

// Wrapper for evaluating linear systems J u = b
int KINSolver::LinSysSetup(N_Vector u, N_Vector, SUNMatrix J,
                           void *, N_Vector, N_Vector )
{
   const SundialsNVector mfem_u(u);
   KINSolver *self = static_cast<KINSolver*>(GET_CONTENT(J));

   // Update the Jacobian
   self->jacobian = &self->oper->GetGradient(mfem_u);

   // Set the Jacobian solve operator
   self->prec->SetOperator(*self->jacobian);

   // Return success
   return (0);
}

// Wrapper for solving linear systems J u = b
int KINSolver::LinSysSolve(SUNLinearSolver LS, SUNMatrix, N_Vector u,
                           N_Vector b, sunrealtype)
{
   SundialsNVector mfem_u(u), mfem_b(b);
   KINSolver *self = static_cast<KINSolver*>(GET_CONTENT(LS));

   // Solve for u = [J(u)]^{-1} b, maybe approximately.
   self->prec->Mult(mfem_b, mfem_u);

   // Return success
   return (0);
}

int KINSolver::PrecSetup(N_Vector uu,
                         N_Vector uscale,
                         N_Vector fval,
                         N_Vector fscale,
                         void *user_data)
{
   SundialsNVector mfem_u(uu);
   KINSolver *self = static_cast<KINSolver *>(user_data);

   // Update the Jacobian
   self->jacobian = &self->oper->GetGradient(mfem_u);

   // Set the Jacobian solve operator
   self->prec->SetOperator(*self->jacobian);

   return 0;
}

int KINSolver::PrecSolve(N_Vector uu,
                         N_Vector uscale,
                         N_Vector fval,
                         N_Vector fscale,
                         N_Vector vv,
                         void *user_data)
{
   KINSolver *self = static_cast<KINSolver *>(user_data);
   SundialsNVector mfem_v(vv);

   self->wrk = 0.0;

   // Solve for u = P^{-1} v
   self->prec->Mult(mfem_v, self->wrk);

   mfem_v = self->wrk;

   return 0;
}

KINSolver::KINSolver(int strategy, bool oper_grad)
   : global_strategy(strategy), use_oper_grad(oper_grad), y_scale(NULL),
     f_scale(NULL), jacobian(NULL)
{
   Y = new SundialsNVector();
   y_scale = new SundialsNVector();
   f_scale = new SundialsNVector();

   // Default abs_tol and print_level
#if MFEM_SUNDIALS_VERSION < 70000
   abs_tol     = pow(UNIT_ROUNDOFF, 1.0/3.0);
#else
   abs_tol     = pow(SUN_UNIT_ROUNDOFF, 1.0/3.0);
#endif
   print_level = 0;
}

#ifdef MFEM_USE_MPI
KINSolver::KINSolver(MPI_Comm comm, int strategy, bool oper_grad)
   : global_strategy(strategy), use_oper_grad(oper_grad), y_scale(NULL),
     f_scale(NULL), jacobian(NULL)
{
   Y = new SundialsNVector(comm);
   y_scale = new SundialsNVector(comm);
   f_scale = new SundialsNVector(comm);

   // Default abs_tol and print_level
#if MFEM_SUNDIALS_VERSION < 70000
   abs_tol     = pow(UNIT_ROUNDOFF, 1.0/3.0);
#else
   abs_tol     = pow(SUN_UNIT_ROUNDOFF, 1.0/3.0);
#endif
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
                    Y->GetComm());
#endif
   }

   if (sundials_mem)
   {
      // Check if the problem size has changed since the last SetOperator call
      int resize = 0;
      if (!Parallel())
      {
         resize = (Y->Size() != local_size);
      }
      else
      {
#ifdef MFEM_USE_MPI
         int l_resize = (Y->Size() != local_size) ||
                        (saved_global_size != global_size);
         MPI_Allreduce(&l_resize, &resize, 1, MPI_INT, MPI_LOR,
                       Y->GetComm());
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
      if (!Parallel())
      {
         Y->SetSize(local_size);
      }
#ifdef MFEM_USE_MPI
      else
      {
         Y->SetSize(local_size, global_size);
         y_scale->SetSize(local_size, global_size);
         f_scale->SetSize(local_size, global_size);
         saved_global_size = global_size;
      }
#endif

      // Create the solver memory
      sundials_mem = KINCreate(Sundials::GetContext());
      MFEM_VERIFY(sundials_mem, "Error in KINCreate().");

      // Enable Anderson Acceleration
      if (aa_n > 0)
      {
         flag = KINSetMAA(sundials_mem, aa_n);
         MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetMAA()");

         flag = KINSetDelayAA(sundials_mem, aa_delay);
         MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetDelayAA()");

         flag = KINSetDampingAA(sundials_mem, aa_damping);
         MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetDampingAA()");

#if SUNDIALS_VERSION_MAJOR >= 6
         flag = KINSetOrthAA(sundials_mem, aa_orth);
         MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetOrthAA()");
#endif
      }

      // Initialize KINSOL
      flag = KINInit(sundials_mem, KINSolver::Mult, *Y);
      MFEM_VERIFY(flag == KIN_SUCCESS, "error in KINInit()");

      // Attach the KINSolver as user-defined data
      flag = KINSetUserData(sundials_mem, this);
      MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetUserData()");

      flag = KINSetDamping(sundials_mem, fp_damping);
      MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetDamping()");

      // Set the linear solver
      if (prec || jfnk)
      {
         KINSolver::SetSolver(*prec);
      }
      else
      {
         // Free any existing linear solver
         if (A != NULL) { SUNMatDestroy(A); A = NULL; }
         if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

         LSA = SUNLinSol_SPGMR(*Y, SUN_PREC_NONE, 0, Sundials::GetContext());
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
   }
}

void KINSolver::SetSolver(Solver &solver)
{
   if (jfnk)
   {
      SetJFNKSolver(solver);
   }
   else
   {
      // Store the solver
      prec = &solver;

      // Free any existing linear solver
      if (A != NULL) { SUNMatDestroy(A); A = NULL; }
      if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

      // Wrap KINSolver as SUNLinearSolver and SUNMatrix
      LSA = SUNLinSolNewEmpty(Sundials::GetContext());
      MFEM_VERIFY(LSA, "error in SUNLinSolNewEmpty()");

      LSA->content      = this;
      LSA->ops->gettype = LSGetType;
      LSA->ops->solve   = KINSolver::LinSysSolve;
      LSA->ops->free    = LSFree;

      A = SUNMatNewEmpty(Sundials::GetContext());
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
}

void KINSolver::SetJFNKSolver(Solver &solver)
{
   // Store the solver
   prec = &solver;

   wrk.SetSize(height);

   // Free any existing linear solver
   if (A != NULL) { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Setup FGMRES
   LSA = SUNLinSol_SPFGMR(*Y, prec ? SUN_PREC_RIGHT : SUN_PREC_NONE, maxli,
                          Sundials::GetContext());
   MFEM_VERIFY(LSA, "error in SUNLinSol_SPFGMR()");

   flag = SUNLinSol_SPFGMRSetMaxRestarts(LSA, maxlrs);
   MFEM_VERIFY(flag == SUN_SUCCESS, "error in SUNLinSol_SPFGMR()");

   flag = KINSetLinearSolver(sundials_mem, LSA, NULL);
   MFEM_VERIFY(flag == KIN_SUCCESS, "error in KINSetLinearSolver()");

   if (prec)
   {
      flag = KINSetPreconditioner(sundials_mem,
                                  KINSolver::PrecSetup,
                                  KINSolver::PrecSolve);
      MFEM_VERIFY(flag == KIN_SUCCESS, "error in KINSetPreconditioner()");
   }
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

void KINSolver::EnableAndersonAcc(int n, int orth, int delay, double damping)
{
   if (sundials_mem != nullptr)
   {
      if (aa_n < n)
      {
         MFEM_ABORT("Subsequent calls to EnableAndersonAcc() must set"
                    " the subspace size to less or equal to the initially requested size."
                    " If SetOperator() has already been called, the subspace size can't be"
                    " increased.");
      }

      flag = KINSetMAA(sundials_mem, n);
      MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetMAA()");

      flag = KINSetDelayAA(sundials_mem, delay);
      MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetDelayAA()");

      flag = KINSetDampingAA(sundials_mem, damping);
      MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetDampingAA()");

#if SUNDIALS_VERSION_MAJOR >= 6
      flag = KINSetOrthAA(sundials_mem, orth);
      MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetOrthAA()");
#else
      if (orth != KIN_ORTH_MGS)
      {
         MFEM_WARNING("SUNDIALS < v6 does not support setting the Anderson"
                      " acceleration orthogonalization routine!");
      }
#endif
   }

   aa_n = n;
   aa_delay = delay;
   aa_damping = damping;
   aa_orth = orth;
}

void KINSolver::SetDamping(double damping)
{
   fp_damping = damping;
   if (sundials_mem)
   {
      flag = KINSetDamping(sundials_mem, fp_damping);
      MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetDamping()");
   }
}

void KINSolver::SetPrintLevel(PrintLevel)
{
   MFEM_ABORT("this method is not supported! Use SetPrintLevel(int) instead.");
}

// Compute the scaling vectors and solve nonlinear system
void KINSolver::Mult(const Vector&, Vector &x) const
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
         MPI_Allreduce(&lnorm, &norm, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                       Y->GetComm());
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

   Y->MakeRef(x, 0, x.Size());
   y_scale->MakeRef(const_cast<Vector&>(x_scale), 0, x_scale.Size());
   f_scale->MakeRef(const_cast<Vector&>(fx_scale), 0, fx_scale.Size());

   int rank = -1;
   if (!Parallel())
   {
      rank = 0;
   }
   else
   {
#ifdef MFEM_USE_MPI
      MPI_Comm_rank(Y->GetComm(), &rank);
#endif
   }

   if (rank == 0)
   {
#if MFEM_SUNDIALS_VERSION < 70000
      flag = KINSetPrintLevel(sundials_mem, print_level);
      MFEM_VERIFY(flag == KIN_SUCCESS, "KINSetPrintLevel() failed!");
#endif
      // NOTE: there is no KINSetPrintLevel in SUNDIALS v7!

#ifdef SUNDIALS_BUILD_WITH_MONITORING
      if (jfnk && print_level)
      {
         flag = SUNLinSolSetInfoFile_SPFGMR(LSA, stdout);
         MFEM_VERIFY(flag == SUN_SUCCESS,
                     "error in SUNLinSolSetInfoFile_SPFGMR()");

         flag = SUNLinSolSetPrintLevel_SPFGMR(LSA, 1);
         MFEM_VERIFY(flag == SUN_SUCCESS,
                     "error in SUNLinSolSetPrintLevel_SPFGMR()");
      }
#endif
   }

   if (!iterative_mode) { x = 0.0; }

   // Solve the nonlinear system
   flag = KINSol(sundials_mem, *Y, global_strategy, *y_scale, *f_scale);
   converged = (flag >= 0);

   // Make sure host is up to date
   Y->HostRead();

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
   delete Y;
   delete y_scale;
   delete f_scale;
   SUNMatDestroy(A);
   SUNLinSolFree(LSA);
   KINFree(&sundials_mem);
}

} // namespace mfem

#endif // MFEM_USE_SUNDIALS
