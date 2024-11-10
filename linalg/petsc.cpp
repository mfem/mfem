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

// Author: Stefano Zampini <stefano.zampini@gmail.com>

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_PETSC

#include "linalg.hpp"
#include "../fem/fem.hpp"

#include "petsc.h"
#if defined(PETSC_HAVE_HYPRE)
#include "petscmathypre.h"
#endif

// Backward compatibility
#if PETSC_VERSION_LT(3,11,0)
#define VecLockReadPush VecLockPush
#define VecLockReadPop VecLockPop
#endif
#if PETSC_VERSION_LT(3,12,0)
#define VecGetArrayWrite VecGetArray
#define VecRestoreArrayWrite VecRestoreArray
#define MatComputeOperator(A,B,C) MatComputeExplicitOperator(A,C)
#define MatComputeOperatorTranspose(A,B,C) MatComputeExplicitOperatorTranspose(A,C)
#endif
#if PETSC_VERSION_LT(3,19,0)
#define PETSC_SUCCESS 0
#endif

#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

// Note: there are additional #include statements below.

#include "petscinternals.hpp"

// Callback functions: these functions will be called by PETSc
static PetscErrorCode __mfem_ts_monitor(TS,PetscInt,PetscReal,Vec,void*);
static PetscErrorCode __mfem_ts_rhsfunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode __mfem_ts_rhsjacobian(TS,PetscReal,Vec,Mat,Mat,
                                            void*);
static PetscErrorCode __mfem_ts_ifunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode __mfem_ts_ijacobian(TS,PetscReal,Vec,Vec,
                                          PetscReal,Mat,
                                          Mat,void*);
static PetscErrorCode __mfem_ts_computesplits(TS,PetscReal,Vec,Vec,
                                              Mat,Mat,Mat,Mat);
static PetscErrorCode __mfem_snes_monitor(SNES,PetscInt,PetscReal,void*);
static PetscErrorCode __mfem_snes_jacobian(SNES,Vec,Mat,Mat,void*);
static PetscErrorCode __mfem_snes_function(SNES,Vec,Vec,void*);
static PetscErrorCode __mfem_snes_objective(SNES,Vec,PetscReal*,void*);
static PetscErrorCode __mfem_snes_update(SNES,PetscInt);
static PetscErrorCode __mfem_snes_postcheck(SNESLineSearch,Vec,Vec,Vec,
                                            PetscBool*,PetscBool*,void*);
static PetscErrorCode __mfem_ksp_monitor(KSP,PetscInt,PetscReal,void*);
static PetscErrorCode __mfem_pc_shell_apply(PC,Vec,Vec);
static PetscErrorCode __mfem_pc_shell_apply_transpose(PC,Vec,Vec);
static PetscErrorCode __mfem_pc_shell_setup(PC);
static PetscErrorCode __mfem_pc_shell_destroy(PC);
static PetscErrorCode __mfem_pc_shell_view(PC,PetscViewer);
static PetscErrorCode __mfem_mat_shell_apply(Mat,Vec,Vec);
static PetscErrorCode __mfem_mat_shell_apply_transpose(Mat,Vec,Vec);
static PetscErrorCode __mfem_mat_shell_destroy(Mat);
static PetscErrorCode __mfem_mat_shell_copy(Mat,Mat,MatStructure);
static PetscErrorCode __mfem_array_container_destroy(void*);
static PetscErrorCode __mfem_matarray_container_destroy(void*);
static PetscErrorCode __mfem_monitor_ctx_destroy(void**);

// auxiliary functions
static PetscErrorCode Convert_Array_IS(MPI_Comm,bool,const mfem::Array<int>*,
                                       PetscInt,IS*);
static PetscErrorCode Convert_Vmarks_IS(MPI_Comm,mfem::Array<Mat>&,
                                        const mfem::Array<int>*,PetscInt,IS*);
static PetscErrorCode MakeShellPC(PC,mfem::Solver&,bool);
static PetscErrorCode MakeShellPCWithFactory(PC,
                                             mfem::PetscPreconditionerFactory*);

// Equivalent functions are present in PETSc source code
// if PETSc has been compiled with hypre support
// We provide them here in case PETSC_HAVE_HYPRE is not defined
#if !defined(PETSC_HAVE_HYPRE)
static PetscErrorCode MatConvert_hypreParCSR_AIJ(hypre_ParCSRMatrix*,Mat*);
static PetscErrorCode MatConvert_hypreParCSR_IS(hypre_ParCSRMatrix*,Mat*);
#endif

#if PETSC_VERSION_GE(3,15,0) && defined(PETSC_HAVE_DEVICE)
#if defined(MFEM_USE_CUDA) && defined(PETSC_HAVE_CUDA)
#define _USE_DEVICE
#define PETSC_VECDEVICE VECCUDA
#define PETSC_MATAIJDEVICE MATAIJCUSPARSE
#define VecDeviceGetArrayRead      VecCUDAGetArrayRead
#define VecDeviceGetArrayWrite     VecCUDAGetArrayWrite
#define VecDeviceGetArray          VecCUDAGetArray
#define VecDeviceRestoreArrayRead  VecCUDARestoreArrayRead
#define VecDeviceRestoreArrayWrite VecCUDARestoreArrayWrite
#define VecDeviceRestoreArray      VecCUDARestoreArray
#define VecDevicePlaceArray        VecCUDAPlaceArray
#define VecDeviceResetArray        VecCUDAResetArray
#elif defined(MFEM_USE_HIP) && defined(PETSC_HAVE_HIP)
#define _USE_DEVICE
#define PETSC_VECDEVICE VECHIP
#define PETSC_MATAIJDEVICE MATAIJHIPSPARSE
#define VecDeviceGetArrayRead      VecHIPGetArrayRead
#define VecDeviceGetArrayWrite     VecHIPGetArrayWrite
#define VecDeviceGetArray          VecHIPGetArray
#define VecDeviceRestoreArrayRead  VecHIPRestoreArrayRead
#define VecDeviceRestoreArrayWrite VecHIPRestoreArrayWrite
#define VecDeviceRestoreArray      VecHIPRestoreArray
#define VecDevicePlaceArray        VecHIPPlaceArray
#define VecDeviceResetArray        VecHIPResetArray
#else
#define VecDeviceGetArrayRead      VecGetArrayRead
#define VecDeviceGetArrayWrite     VecGetArrayWrite
#define VecDeviceGetArray          VecGetArray
#define VecDeviceRestoreArrayRead  VecRestoreArrayRead
#define VecDeviceRestoreArrayWrite VecRestoreArrayWrite
#define VecDeviceRestoreArray      VecRestoreArray
#define VecDevicePlaceArray        VecPlaceArray
#define VecDeviceResetArray        VecResetArray
#endif
#endif

#if defined(PETSC_HAVE_DEVICE)
static PetscErrorCode __mfem_VecSetOffloadMask(Vec,PetscOffloadMask);
#endif
static PetscErrorCode __mfem_VecBoundToCPU(Vec,PetscBool*);
static PetscErrorCode __mfem_PetscObjectStateIncrease(PetscObject);
static PetscErrorCode __mfem_MatCreateDummy(MPI_Comm,PetscInt,PetscInt,Mat*);

// structs used by PETSc code
typedef struct
{
   mfem::Solver                     *op;
   mfem::PetscPreconditionerFactory *factory;
   bool                             ownsop;
   unsigned long int                numprec;
} __mfem_pc_shell_ctx;

typedef struct
{
   mfem::Operator        *op;        // The nonlinear operator
   mfem::PetscBCHandler  *bchandler; // Handling of essential bc
   mfem::Vector          *work;      // Work vector
   mfem::Operator::Type  jacType;    // OperatorType for the Jacobian
   // Objective for line search
   void (*objective)(mfem::Operator *op, const mfem::Vector&, mfem::real_t*);
   // PostCheck function (to be called after successful line search)
   void (*postcheck)(mfem::Operator *op, const mfem::Vector&, mfem::Vector&,
                     mfem::Vector&, bool&, bool&);
   // General purpose update function (to be called at the beginning of
   // each nonlinear step)
   void (*update)(mfem::Operator *op, int,
                  const mfem::Vector&, const mfem::Vector&,
                  const mfem::Vector&, const mfem::Vector&);
} __mfem_snes_ctx;

typedef struct
{
   mfem::TimeDependentOperator     *op;        // The time-dependent operator
   mfem::PetscBCHandler            *bchandler; // Handling of essential bc
   mfem::Vector                    *work;      // Work vector
   mfem::Vector                    *work2;     // Work vector
   mfem::Operator::Type            jacType;    // OperatorType for the Jacobian
   enum mfem::PetscODESolver::Type type;
   PetscReal                       cached_shift;
   PetscObjectState                cached_ijacstate;
   PetscObjectState                cached_rhsjacstate;
   PetscObjectState                cached_splits_xstate;
   PetscObjectState                cached_splits_xdotstate;
} __mfem_ts_ctx;

typedef struct
{
   mfem::PetscSolver        *solver;  // The solver object
   mfem::PetscSolverMonitor *monitor; // The user-defined monitor class
} __mfem_monitor_ctx;

// use global scope ierr to check PETSc errors inside mfem calls
static PetscErrorCode ierr;
static PetscMPIInt mpiierr;

using namespace std;

namespace mfem
{

void MFEMInitializePetsc()
{
   MFEMInitializePetsc(NULL,NULL,NULL,NULL);
}

void MFEMInitializePetsc(int *argc,char*** argv)
{
   MFEMInitializePetsc(argc,argv,NULL,NULL);
}

void MFEMInitializePetsc(int *argc,char ***argv,const char rc_file[],
                         const char help[])
{
   // Tell PETSc to use the same CUDA or HIP device as MFEM:
   if (mfem::Device::Allows(mfem::Backend::CUDA_MASK))
   {
#if PETSC_VERSION_LT(3,17,0)
      const char *opts = "-cuda_device";
#else
      const char *opts = "-device_select_cuda";
#endif
      ierr = PetscOptionsSetValue(NULL,opts,
                                  to_string(mfem::Device::GetId()).c_str());
      MFEM_VERIFY(!ierr,"Unable to set initial option value to PETSc");
   }
   if (mfem::Device::Allows(mfem::Backend::HIP_MASK))
   {
#if PETSC_VERSION_LT(3,17,0)
      const char *opts = "-hip_device";
#else
      const char *opts = "-device_select_hip";
#endif
      ierr = PetscOptionsSetValue(NULL,opts,
                                  to_string(mfem::Device::GetId()).c_str());
      MFEM_VERIFY(!ierr,"Unable to set initial option value to PETSc");
   }
   ierr = PetscInitialize(argc,argv,rc_file,help);
   MFEM_VERIFY(!ierr,"Unable to initialize PETSc");
}

void MFEMFinalizePetsc()
{
   ierr = PetscFinalize();
   MFEM_VERIFY(!ierr,"Unable to finalize PETSc");
}

const mfem::real_t* PetscMemory::GetHostPointer() const
{
   int oflags = flags;
   SetHostValid();
   const mfem::real_t *v = mfem::Read(*this,Capacity(),false);
   flags = oflags;
   return v;
}

const mfem::real_t* PetscMemory::GetDevicePointer() const
{
   int oflags = flags;
   SetDeviceValid();
   const mfem::real_t *v = mfem::Read(*this,Capacity(),true);
   flags = oflags;
   return v;
}

// PetscParVector methods

void PetscParVector::SetDataAndSize_()
{
   PetscScalar *array;
   PetscInt    n;
   PetscBool   isnest;

   MFEM_VERIFY(x,"Missing Vec");
   ierr = VecSetUp(x); PCHKERRQ(x,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)x,VECNEST,&isnest); PCHKERRQ(x,ierr);
   MFEM_VERIFY(!isnest,"Not for type nest");
   ierr = VecGetLocalSize(x,&n); PCHKERRQ(x,ierr);
   MFEM_VERIFY(n >= 0,"Invalid local size");
   size = n;
#if defined(PETSC_HAVE_DEVICE)
   PetscOffloadMask omask;
   PetscBool        isdevice;

   ierr = VecGetOffloadMask(x,&omask); PCHKERRQ(x,ierr);
   if (omask != PETSC_OFFLOAD_BOTH)
   {
      ierr = __mfem_VecSetOffloadMask(x,PETSC_OFFLOAD_CPU); PCHKERRQ(x,ierr);
   }
#endif
   ierr = VecGetArrayRead(x,(const PetscScalar**)&array); PCHKERRQ(x,ierr);
#if defined(PETSC_HAVE_DEVICE)
   ierr = PetscObjectTypeCompareAny((PetscObject)x,&isdevice,
                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
                                    ""); PCHKERRQ(x,ierr);
   if (isdevice)
   {
      if (omask != PETSC_OFFLOAD_BOTH)
      {
         ierr = __mfem_VecSetOffloadMask(x,PETSC_OFFLOAD_GPU); PCHKERRQ(x,ierr);
      }
      PetscScalar *darray;
      ierr = VecDeviceGetArrayRead(x,(const PetscScalar**)&darray);
      PCHKERRQ(x,ierr);
      pdata.Wrap(array,darray,size,MemoryType::HOST,false);
      ierr = VecDeviceRestoreArrayRead(x,(const PetscScalar**)&darray);
      PCHKERRQ(x,ierr);
   }
   else
#endif
   {
      pdata.Wrap(array,size,MemoryType::HOST,false);
   }
   ierr = VecRestoreArrayRead(x,(const PetscScalar**)&array); PCHKERRQ(x,ierr);

#if defined(PETSC_HAVE_DEVICE)
   if (omask == PETSC_OFFLOAD_UNALLOCATED && isdevice) { omask = PETSC_OFFLOAD_CPU; }
   ierr = __mfem_VecSetOffloadMask(x,omask); PCHKERRQ(x,ierr);
#endif
   data.MakeAlias(pdata,0,size);
   SetFlagsFromMask_();
}

void PetscParVector::SetFlagsFromMask_() const
{
   MFEM_VERIFY(x,"Missing Vec");
#if defined(_USE_DEVICE)
   PetscOffloadMask mask;
   PetscBool isdevice;
   ierr = PetscObjectTypeCompareAny((PetscObject)x,&isdevice,
                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
                                    ""); PCHKERRQ(x,ierr);
   ierr = VecGetOffloadMask(x,&mask); PCHKERRQ(x,ierr);
   if (isdevice)
   {
      switch (mask)
      {
         case PETSC_OFFLOAD_CPU:
            pdata.SetHostValid();
            pdata.SetDeviceInvalid();
            break;
         case PETSC_OFFLOAD_GPU:
            pdata.SetHostInvalid();
            pdata.SetDeviceValid();
            break;
         case PETSC_OFFLOAD_BOTH:
            pdata.SetHostValid();
            pdata.SetDeviceValid();
            break;
         default:
            MFEM_ABORT("Unhandled case " << mask);
      }
   }
#endif
   data.Sync(pdata);
}

void PetscParVector::UpdateVecFromFlags()
{
   MFEM_VERIFY(x,"Missing Vec");
   ierr = __mfem_PetscObjectStateIncrease((PetscObject)x); PCHKERRQ(x,ierr);
#if defined(_USE_DEVICE)
   PetscBool isdevice;
   ierr = PetscObjectTypeCompareAny((PetscObject)x,&isdevice,
                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
                                    ""); PCHKERRQ(x,ierr);
   if (isdevice)
   {
      bool dv = pdata.DeviceIsValid();
      bool hv = pdata.HostIsValid();
      PetscOffloadMask mask;
      if (dv && hv) { mask = PETSC_OFFLOAD_BOTH; }
      else if (dv) { mask = PETSC_OFFLOAD_GPU; }
      else { mask = PETSC_OFFLOAD_CPU; }
      ierr = __mfem_VecSetOffloadMask(x,mask); PCHKERRQ(x,ierr);
   }
   else
#endif
   {
      /* Just make sure we have an up-to-date copy on the CPU for PETSc */
      PetscScalar *v;
      ierr = VecGetArrayWrite(x,&v); PCHKERRQ(x,ierr);
      pdata.CopyToHost(v,size);
      ierr = VecRestoreArrayWrite(x,&v); PCHKERRQ(x,ierr);
   }
}

void PetscParVector::SetVecType_()
{
   VecType vectype;
   MFEM_VERIFY(x,"Missing Vec");
   ierr = VecGetType(x,&vectype); PCHKERRQ(x,ierr);
#if defined(_USE_DEVICE)
   switch (Device::GetDeviceMemoryType())
   {
      case MemoryType::DEVICE:
      case MemoryType::MANAGED:
         ierr = VecSetType(x,PETSC_VECDEVICE); PCHKERRQ(x,ierr);
         break;
      default:
         ierr = VecSetType(x,VECSTANDARD); PCHKERRQ(x,ierr);
         break;
   }
#else
   if (!vectype)
   {
      ierr = VecSetType(x,VECSTANDARD); PCHKERRQ(x,ierr);
   }
#endif
}

const mfem::real_t* PetscParVector::Read(bool on_dev) const
{
   const PetscScalar *dummy;
   MFEM_VERIFY(x,"Missing Vec");
#if defined(PETSC_HAVE_DEVICE)
   PetscBool isdevice;
   ierr = PetscObjectTypeCompareAny((PetscObject)x,&isdevice,
                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
                                    ""); PCHKERRQ(x,ierr);
   if (on_dev && isdevice)
   {
      ierr = VecDeviceGetArrayRead(x,&dummy); PCHKERRQ(x,ierr);
      ierr = VecDeviceRestoreArrayRead(x,&dummy); PCHKERRQ(x,ierr);
   }
   else
#endif
   {
      ierr = VecGetArrayRead(x,&dummy); PCHKERRQ(x,ierr);
      ierr = VecRestoreArrayRead(x,&dummy); PCHKERRQ(x,ierr);
   }
   SetFlagsFromMask_();
   return mfem::Read(pdata, size, on_dev);
}

const mfem::real_t* PetscParVector::HostRead() const
{
   return Read(false);
}

mfem::real_t* PetscParVector::Write(bool on_dev)
{
   PetscScalar *dummy;
   MFEM_VERIFY(x,"Missing Vec");
#if defined(PETSC_HAVE_DEVICE)
   PetscBool isdevice;
   ierr = PetscObjectTypeCompareAny((PetscObject)x,&isdevice,
                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
                                    ""); PCHKERRQ(x,ierr);
   if (on_dev && isdevice)
   {
      ierr = VecDeviceGetArrayWrite(x,&dummy); PCHKERRQ(x,ierr);
      ierr = VecDeviceRestoreArrayWrite(x,&dummy); PCHKERRQ(x,ierr);
   }
   else
#endif
   {
      ierr = VecGetArrayWrite(x,&dummy); PCHKERRQ(x,ierr);
      ierr = VecRestoreArrayWrite(x,&dummy); PCHKERRQ(x,ierr);
   }
   ierr = __mfem_PetscObjectStateIncrease((PetscObject)x); PCHKERRQ(x,ierr);
   SetFlagsFromMask_();
   return mfem::Write(pdata, size, on_dev);
}

mfem::real_t* PetscParVector::HostWrite()
{
   return Write(false);
}

mfem::real_t* PetscParVector::ReadWrite(bool on_dev)
{
   PetscScalar *dummy;
   MFEM_VERIFY(x,"Missing Vec");
#if defined(PETSC_HAVE_DEVICE)
   PetscBool isdevice;
   ierr = PetscObjectTypeCompareAny((PetscObject)x,&isdevice,
                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
                                    ""); PCHKERRQ(x,ierr);
   if (on_dev && isdevice)
   {
      ierr = VecDeviceGetArray(x,&dummy); PCHKERRQ(x,ierr);
      ierr = VecDeviceRestoreArray(x,&dummy); PCHKERRQ(x,ierr);
   }
   else
#endif
   {
      ierr = VecGetArray(x,&dummy); PCHKERRQ(x,ierr);
      ierr = VecRestoreArray(x,&dummy); PCHKERRQ(x,ierr);
   }
   ierr = __mfem_PetscObjectStateIncrease((PetscObject)x); PCHKERRQ(x,ierr);
   SetFlagsFromMask_();
   return mfem::ReadWrite(pdata, size, on_dev);
}

mfem::real_t* PetscParVector::HostReadWrite()
{
   return ReadWrite(false);
}

void PetscParVector::UseDevice(bool dev) const
{
   MFEM_VERIFY(x,"Missing Vec");
#if defined(PETSC_HAVE_DEVICE)
   ierr = VecBindToCPU(x,!dev ? PETSC_TRUE : PETSC_FALSE); PCHKERRQ(x,ierr);
   SetFlagsFromMask_();
#endif
}

bool PetscParVector::UseDevice() const
{
   PetscBool flg;
   MFEM_VERIFY(x,"Missing Vec");
   ierr = __mfem_VecBoundToCPU(x,&flg); PCHKERRQ(x,ierr);
   return flg ? false : true;
}

PetscInt PetscParVector::GlobalSize() const
{
   PetscInt N;
   ierr = VecGetSize(x,&N); PCHKERRQ(x,ierr);
   return N;
}

void PetscParVector::SetBlockSize(PetscInt bs)
{
   ierr = VecSetBlockSize(x,bs); PCHKERRQ(x,ierr);
}

PetscParVector::PetscParVector(MPI_Comm comm, const Vector &x_,
                               bool copy) : Vector()
{
   PetscBool isdevice;

   int n = x_.Size();
   ierr = VecCreate(comm,&x); CCHKERRQ(comm,ierr);
   ierr = VecSetSizes(x,n,PETSC_DECIDE); PCHKERRQ(x,ierr);
   SetVecType_();
   SetDataAndSize_();
   ierr = PetscObjectTypeCompareAny((PetscObject)x,&isdevice,
                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
                                    ""); PCHKERRQ(x,ierr);
   if (copy)
   {
      /* we use PETSc accessors to flag valid memory location to PETSc */
      PetscErrorCode (*rest)(Vec,PetscScalar**);
      PetscScalar *array;
#if defined(PETSC_HAVE_DEVICE)
      if (isdevice && x_.UseDevice())
      {
         UseDevice(true);
         ierr = VecDeviceGetArrayWrite(x,&array); PCHKERRQ(x,ierr);
         rest = VecDeviceRestoreArrayWrite;
      }
      else
#endif
      {
         UseDevice(false);
         ierr = VecGetArrayWrite(x,&array); PCHKERRQ(x,ierr);
         rest = VecRestoreArrayWrite;
      }
      pdata.CopyFrom(x_.GetMemory(), n);
      ierr = (*rest)(x,&array); PCHKERRQ(x,ierr);
      SetFlagsFromMask_();
   }
   else // don't copy, just set the device flag
   {
      if (isdevice && x_.UseDevice())
      {
         UseDevice(true);
      }
      else
      {
         UseDevice(false);
      }
   }
}

PetscParVector::PetscParVector(MPI_Comm comm, PetscInt glob_size,
                               PetscInt *col) : Vector()
{
   ierr = VecCreate(comm,&x); CCHKERRQ(comm,ierr);
   if (col)
   {
      PetscMPIInt myid;
      mpiierr = MPI_Comm_rank(comm, &myid); CCHKERRQ(comm, mpiierr);
      ierr = VecSetSizes(x,col[myid+1]-col[myid],PETSC_DECIDE); PCHKERRQ(x,ierr);
   }
   else
   {
      ierr = VecSetSizes(x,PETSC_DECIDE,glob_size); PCHKERRQ(x,ierr);
   }
   SetVecType_();
   SetDataAndSize_();
}

PetscParVector::~PetscParVector()
{
   MPI_Comm comm = PetscObjectComm((PetscObject)x);
   ierr = VecDestroy(&x); CCHKERRQ(comm,ierr);
   pdata.Delete();
}

PetscParVector::PetscParVector(MPI_Comm comm, PetscInt glob_size,
                               PetscScalar *data_, PetscInt *col) : Vector()
{
   MFEM_VERIFY(col,"Missing distribution");
   PetscMPIInt myid;
   mpiierr = MPI_Comm_rank(comm, &myid); CCHKERRQ(comm, mpiierr);
   ierr = VecCreateMPIWithArray(comm,1,col[myid+1]-col[myid],glob_size,data_,
                                &x); CCHKERRQ(comm,ierr)
   SetVecType_();
   SetDataAndSize_();
}

PetscParVector::PetscParVector(const PetscParVector &y) : Vector()
{
   ierr = VecDuplicate(y.x,&x); PCHKERRQ(x,ierr);
   SetDataAndSize_();
}

PetscParVector::PetscParVector(MPI_Comm comm, const Operator &op,
                               bool transpose, bool allocate) : Vector()
{
   PetscInt loc = transpose ? op.Height() : op.Width();

   ierr = VecCreate(comm,&x);
   CCHKERRQ(comm,ierr);
   ierr = VecSetSizes(x,loc,PETSC_DECIDE);
   PCHKERRQ(x,ierr);

   SetVecType_();
   if (allocate)
   {
      SetDataAndSize_();
   }
   else /* Vector intended to be used with Place/ResetMemory calls */
   {
      size = loc;
   }
}

PetscParVector::PetscParVector(const PetscParMatrix &A,
                               bool transpose, bool allocate) : Vector()
{
   Mat pA = const_cast<PetscParMatrix&>(A);
   if (!transpose)
   {
      ierr = MatCreateVecs(pA,&x,NULL); PCHKERRQ(pA,ierr);
   }
   else
   {
      ierr = MatCreateVecs(pA,NULL,&x); PCHKERRQ(pA,ierr);
   }
   SetVecType_();
   if (!allocate) /* Vector intended to be used with Place/ResetMemory calls */
   {
      PetscInt n;
      ierr = VecGetLocalSize(x,&n); PCHKERRQ(x,ierr);
      size = n;
   }
   else
   {
      SetDataAndSize_();
   }
}

PetscParVector::PetscParVector(petsc::Vec y, bool ref) : Vector()
{
   if (ref)
   {
      ierr = PetscObjectReference((PetscObject)y); PCHKERRQ(y,ierr);
   }
   x = y;
   SetDataAndSize_();
}

PetscParVector::PetscParVector(ParFiniteElementSpace *pfes) : Vector()
{
   HYPRE_BigInt* offsets = pfes->GetTrueDofOffsets();
   MPI_Comm  comm = pfes->GetComm();
   ierr = VecCreate(comm,&x); CCHKERRQ(comm,ierr);

   PetscMPIInt myid = 0;
   if (!HYPRE_AssumedPartitionCheck())
   {
      mpiierr = MPI_Comm_rank(comm, &myid); CCHKERRQ(comm, mpiierr);
   }
   ierr = VecSetSizes(x,offsets[myid+1]-offsets[myid],PETSC_DECIDE);
   PCHKERRQ(x,ierr);
   SetVecType_();
   SetDataAndSize_();
}

MPI_Comm PetscParVector::GetComm() const
{
   return x ? PetscObjectComm((PetscObject)x) : MPI_COMM_NULL;
}

Vector * PetscParVector::GlobalVector() const
{
   VecScatter        scctx;
   Vec               vout;
   const PetscScalar *array;
   PetscInt          size;

   ierr = VecScatterCreateToAll(x,&scctx,&vout); PCHKERRQ(x,ierr);
   ierr = VecScatterBegin(scctx,x,vout,INSERT_VALUES,SCATTER_FORWARD);
   PCHKERRQ(x,ierr);
   ierr = VecScatterEnd(scctx,x,vout,INSERT_VALUES,SCATTER_FORWARD);
   PCHKERRQ(x,ierr);
   ierr = VecScatterDestroy(&scctx); PCHKERRQ(x,ierr);
   ierr = VecGetArrayRead(vout,&array); PCHKERRQ(x,ierr);
   ierr = VecGetLocalSize(vout,&size); PCHKERRQ(x,ierr);
   Array<PetscScalar> data(size);
   data.Assign(array);
   ierr = VecRestoreArrayRead(vout,&array); PCHKERRQ(x,ierr);
   ierr = VecDestroy(&vout); PCHKERRQ(x,ierr);
   Vector *v = new Vector(data, internal::to_int(size));
   v->MakeDataOwner();
   data.LoseData();
   return v;
}

PetscParVector& PetscParVector::operator=(PetscScalar d)
{
   ierr = VecSet(x,d); PCHKERRQ(x,ierr);
   SetFlagsFromMask_();
   return *this;
}

PetscParVector& PetscParVector::SetValues(const Array<PetscInt>& idx,
                                          const Array<PetscScalar>& vals)
{
   MFEM_VERIFY(idx.Size() == vals.Size(),
               "Size mismatch between indices and values");
   PetscInt n = idx.Size();
   ierr = VecSetValues(x,n,idx.GetData(),vals.GetData(),INSERT_VALUES);
   PCHKERRQ(x,ierr);
   ierr = VecAssemblyBegin(x); PCHKERRQ(x,ierr);
   ierr = VecAssemblyEnd(x); PCHKERRQ(x,ierr);
   SetFlagsFromMask_();
   return *this;
}

PetscParVector& PetscParVector::AddValues(const Array<PetscInt>& idx,
                                          const Array<PetscScalar>& vals)
{
   MFEM_VERIFY(idx.Size() == vals.Size(),
               "Size mismatch between indices and values");
   PetscInt n = idx.Size();
   ierr = VecSetValues(x,n,idx.GetData(),vals.GetData(),ADD_VALUES);
   PCHKERRQ(x,ierr);
   ierr = VecAssemblyBegin(x); PCHKERRQ(x,ierr);
   ierr = VecAssemblyEnd(x); PCHKERRQ(x,ierr);
   SetFlagsFromMask_();
   return *this;
}

PetscParVector& PetscParVector::operator=(const PetscParVector &y)
{
   ierr = VecCopy(y.x,x); PCHKERRQ(x,ierr);
   SetFlagsFromMask_();
   return *this;
}

PetscParVector& PetscParVector::operator+=(const PetscParVector &y)
{
   ierr = VecAXPY(x,1.0,y.x); PCHKERRQ(x,ierr);
   SetFlagsFromMask_();
   return *this;
}

PetscParVector& PetscParVector::operator-=(const PetscParVector &y)
{
   ierr = VecAXPY(x,-1.0,y.x); PCHKERRQ(x,ierr);
   SetFlagsFromMask_();
   return *this;
}

PetscParVector& PetscParVector::operator*=(PetscScalar s)
{
   ierr = VecScale(x,s); PCHKERRQ(x,ierr);
   SetFlagsFromMask_();
   return *this;
}

PetscParVector& PetscParVector::operator+=(PetscScalar s)
{
   ierr = VecShift(x,s); PCHKERRQ(x,ierr);
   SetFlagsFromMask_();
   return *this;
}

void PetscParVector::PlaceArray(PetscScalar *temp_data)
{
   ierr = VecPlaceArray(x,temp_data); PCHKERRQ(x,ierr);
}

void PetscParVector::ResetArray()
{
   ierr = VecResetArray(x); PCHKERRQ(x,ierr);
}

void PetscParVector::PlaceMemory(Memory<mfem::real_t>& mem, bool rw)
{
   PetscInt n;

   ierr = VecGetLocalSize(x,&n); PCHKERRQ(x,ierr);
   MFEM_VERIFY(n <= mem.Capacity(),
               "Memory size " << mem.Capacity() << " < " << n << " vector size!");
   MFEM_VERIFY(pdata.Empty(),"Vector data is not empty");
   MFEM_VERIFY(data.Empty(),"Vector data is not empty");
#if defined(_USE_DEVICE)
   PetscBool isdevice;
   ierr = PetscObjectTypeCompareAny((PetscObject)x,&isdevice,
                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
                                    ""); PCHKERRQ(x,ierr);
   if (isdevice)
   {
      bool usedev = mem.DeviceIsValid() || (!rw && mem.UseDevice());
      pdata.MakeAliasForSync(mem,0,n,rw,true,usedev);
      if (usedev)
      {
         ierr = __mfem_VecSetOffloadMask(x,PETSC_OFFLOAD_GPU); PCHKERRQ(x,ierr);
         ierr = VecDevicePlaceArray(x,pdata.GetDevicePointer()); PCHKERRQ(x,ierr);
      }
      else
      {
         ierr = __mfem_VecSetOffloadMask(x,PETSC_OFFLOAD_CPU); PCHKERRQ(x,ierr);
         ierr = VecPlaceArray(x,pdata.GetHostPointer()); PCHKERRQ(x,ierr);
      }
   }
   else
#endif
   {
      mfem::real_t *w = rw ? mfem::HostReadWrite(mem,size) : mfem::HostWrite(mem,
                                                                             size);
      pdata.MakeAliasForSync(mem,0,n,rw,true,false);
#if defined(PETSC_HAVE_DEVICE)
      ierr = __mfem_VecSetOffloadMask(x,PETSC_OFFLOAD_CPU); PCHKERRQ(x,ierr);
#endif
      ierr = VecPlaceArray(x,w); PCHKERRQ(x,ierr);
   }
   ierr = __mfem_PetscObjectStateIncrease((PetscObject)x); PCHKERRQ(x,ierr);
   data.MakeAlias(pdata,0,size);
}

void PetscParVector::PlaceMemory(const Memory<mfem::real_t>& mem)
{
   PetscInt n;

   ierr = VecGetLocalSize(x,&n); PCHKERRQ(x,ierr);
   MFEM_VERIFY(n <= mem.Capacity(),
               "Memory size " << mem.Capacity() << " < " << n << " vector size!");
   MFEM_VERIFY(pdata.Empty(),"Vector data is not empty");
   MFEM_VERIFY(data.Empty(),"Vector data is not empty");
#if defined(_USE_DEVICE)
   PetscBool isdevice;
   ierr = PetscObjectTypeCompareAny((PetscObject)x,&isdevice,
                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
                                    ""); PCHKERRQ(x,ierr);
   if (isdevice)
   {
      pdata.MakeAliasForSync(mem,0,n,mem.DeviceIsValid());
      if (mem.DeviceIsValid())
      {
         ierr = __mfem_VecSetOffloadMask(x,PETSC_OFFLOAD_GPU); PCHKERRQ(x,ierr);
         ierr = VecDevicePlaceArray(x,pdata.GetDevicePointer()); PCHKERRQ(x,ierr);
      }
      else
      {
         ierr = __mfem_VecSetOffloadMask(x,PETSC_OFFLOAD_CPU); PCHKERRQ(x,ierr);
         ierr = VecPlaceArray(x,pdata.GetHostPointer()); PCHKERRQ(x,ierr);
      }
   }
   else
#endif
   {
      const mfem::real_t *w = mfem::HostRead(mem,size);
      pdata.MakeAliasForSync(mem,0,n,false);
#if defined(PETSC_HAVE_DEVICE)
      ierr = __mfem_VecSetOffloadMask(x,PETSC_OFFLOAD_CPU); PCHKERRQ(x,ierr);
#endif
      ierr = VecPlaceArray(x,w); PCHKERRQ(x,ierr);
   }
   data.MakeAlias(pdata,0,size);
   ierr = __mfem_PetscObjectStateIncrease((PetscObject)x); PCHKERRQ(x,ierr);
   ierr = VecLockReadPush(x); PCHKERRQ(x,ierr);
}

void PetscParVector::ResetMemory()
{
   MFEM_VERIFY(pdata.IsAliasForSync(),"Vector data is not an alias");
   MFEM_VERIFY(!pdata.Empty(),"Vector data is empty");
   bool read = pdata.ReadRequested();
   bool usedev = pdata.DeviceRequested();
   bool write = pdata.WriteRequested();
   /*
     check for strange corner cases
      - device memory used but somehow PETSc ended up putting up to date data on host
      - host memory used but somehow PETSc ended up putting up to date data on device
   */
   if (write)
   {
      const PetscScalar *v;
#if defined(PETSC_HAVE_DEVICE)
      PetscOffloadMask mask;
      ierr = VecGetOffloadMask(x,&mask); PCHKERRQ(x,ierr);
      if ((usedev && (mask != PETSC_OFFLOAD_GPU && mask != PETSC_OFFLOAD_BOTH)) ||
          (!usedev && (mask != PETSC_OFFLOAD_CPU && mask != PETSC_OFFLOAD_BOTH)))
#endif
      {
         ierr = VecGetArrayRead(x,&v); PCHKERRQ(x,ierr);
         pdata.CopyFromHost(v, size);
         ierr = VecRestoreArrayRead(x,&v); PCHKERRQ(x,ierr);
      }
   }
   pdata.SyncBaseAndReset();
   data.Reset();
   if (read && !write) { ierr = VecLockReadPop(x); PCHKERRQ(x,ierr); }
   if (usedev)
   {
#if defined(PETSC_HAVE_DEVICE)
      ierr = VecDeviceResetArray(x); PCHKERRQ(x,ierr);
#else
      MFEM_VERIFY(false,"This should not happen");
#endif
   }
   else
   {
      ierr = VecResetArray(x); PCHKERRQ(x,ierr);
   }
}

void PetscParVector::Randomize(PetscInt seed)
{
   PetscRandom rctx = NULL;

   if (seed)
   {
      ierr = PetscRandomCreate(PetscObjectComm((PetscObject)x),&rctx);
      PCHKERRQ(x,ierr);
      ierr = PetscRandomSetSeed(rctx,(unsigned long)seed); PCHKERRQ(x,ierr);
      ierr = PetscRandomSeed(rctx); PCHKERRQ(x,ierr);
   }
   ierr = VecSetRandom(x,rctx); PCHKERRQ(x,ierr);
   ierr = PetscRandomDestroy(&rctx); PCHKERRQ(x,ierr);
}

void PetscParVector::Print(const char *fname, bool binary) const
{
   if (fname)
   {
      PetscViewer view;

      if (binary)
      {
         ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)x),fname,
                                      FILE_MODE_WRITE,&view);
      }
      else
      {
         ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)x),fname,&view);
      }
      PCHKERRQ(x,ierr);
      ierr = VecView(x,view); PCHKERRQ(x,ierr);
      ierr = PetscViewerDestroy(&view); PCHKERRQ(x,ierr);
   }
   else
   {
      ierr = VecView(x,NULL); PCHKERRQ(x,ierr);
   }
}

// PetscParMatrix methods

PetscInt PetscParMatrix::GetRowStart() const
{
   PetscInt N;
   ierr = MatGetOwnershipRange(A,&N,NULL); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::GetColStart() const
{
   PetscInt N;
   ierr = MatGetOwnershipRangeColumn(A,&N,NULL); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::GetNumRows() const
{
   PetscInt N;
   ierr = MatGetLocalSize(A,&N,NULL); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::GetNumCols() const
{
   PetscInt N;
   ierr = MatGetLocalSize(A,NULL,&N); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::M() const
{
   PetscInt N;
   ierr = MatGetSize(A,&N,NULL); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::N() const
{
   PetscInt N;
   ierr = MatGetSize(A,NULL,&N); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::NNZ() const
{
   MatInfo info;
   ierr = MatGetInfo(A,MAT_GLOBAL_SUM,&info); PCHKERRQ(A,ierr);
   return (PetscInt)info.nz_used;
}

void PetscParMatrix::SetBlockSize(PetscInt rbs, PetscInt cbs)
{
   if (cbs < 0) { cbs = rbs; }
   ierr = MatSetBlockSizes(A,rbs,cbs); PCHKERRQ(A,ierr);
}

void PetscParMatrix::Init()
{
   A = NULL;
   X = Y = NULL;
   height = width = 0;
}

PetscParMatrix::PetscParMatrix()
{
   Init();
}

PetscParMatrix::PetscParMatrix(const PetscParMatrix& pB,
                               const mfem::Array<PetscInt>& rows, const mfem::Array<PetscInt>& cols)
{
   Init();

   Mat B = const_cast<PetscParMatrix&>(pB);

   IS isr,isc;
   ierr = ISCreateGeneral(PetscObjectComm((PetscObject)B),rows.Size(),
                          rows.GetData(),PETSC_USE_POINTER,&isr); PCHKERRQ(B,ierr);
   ierr = ISCreateGeneral(PetscObjectComm((PetscObject)B),cols.Size(),
                          cols.GetData(),PETSC_USE_POINTER,&isc); PCHKERRQ(B,ierr);
   ierr = MatCreateSubMatrix(B,isr,isc,MAT_INITIAL_MATRIX,&A); PCHKERRQ(B,ierr);
   ierr = ISDestroy(&isr); PCHKERRQ(B,ierr);
   ierr = ISDestroy(&isc); PCHKERRQ(B,ierr);

   height = GetNumRows();
   width  = GetNumCols();
}

PetscParMatrix::PetscParMatrix(const PetscParMatrix *pa, Operator::Type tid)
{
   Init();
   height = pa->Height();
   width  = pa->Width();
   ConvertOperator(pa->GetComm(),*pa,&A,tid);
}

PetscParMatrix::PetscParMatrix(const HypreParMatrix *ha, Operator::Type tid)
{
   Init();
   height = ha->Height();
   width  = ha->Width();
   ConvertOperator(ha->GetComm(),*ha,&A,tid);
}

PetscParMatrix::PetscParMatrix(const SparseMatrix *sa, Operator::Type tid)
{
   Init();
   height = sa->Height();
   width  = sa->Width();
   ConvertOperator(PETSC_COMM_SELF,*sa,&A,tid);
}

PetscParMatrix::PetscParMatrix(MPI_Comm comm, const Operator *op,
                               Operator::Type tid)
{
   Init();
   height = op->Height();
   width  = op->Width();
   ConvertOperator(comm,*op,&A,tid);
}

PetscParMatrix::PetscParMatrix(MPI_Comm comm, PetscInt glob_size,
                               PetscInt *row_starts, SparseMatrix *diag,
                               Operator::Type tid)
{
   Init();
   BlockDiagonalConstructor(comm,row_starts,row_starts,diag,
                            tid==PETSC_MATAIJ,&A);
   SetUpForDevice();
   // update base class
   height = GetNumRows();
   width  = GetNumCols();
}

PetscParMatrix::PetscParMatrix(MPI_Comm comm, PetscInt global_num_rows,
                               PetscInt global_num_cols, PetscInt *row_starts,
                               PetscInt *col_starts, SparseMatrix *diag,
                               Operator::Type tid)
{
   Init();
   BlockDiagonalConstructor(comm,row_starts,col_starts,diag,
                            tid==PETSC_MATAIJ,&A);
   SetUpForDevice();
   // update base class
   height = GetNumRows();
   width  = GetNumCols();
}

PetscParMatrix& PetscParMatrix::operator=(const HypreParMatrix& B)
{
   if (A)
   {
      MPI_Comm comm = PetscObjectComm((PetscObject)A);
      ierr = MatDestroy(&A); CCHKERRQ(comm,ierr);
      if (X) { delete X; }
      if (Y) { delete Y; }
      X = Y = NULL;
   }
   height = B.Height();
   width  = B.Width();
#if defined(PETSC_HAVE_HYPRE)
   ierr = MatCreateFromParCSR(B,MATAIJ,PETSC_USE_POINTER,&A);
#else
   ierr = MatConvert_hypreParCSR_AIJ(B,&A); CCHKERRQ(B.GetComm(),ierr);
#endif
   SetUpForDevice();
   return *this;
}

PetscParMatrix& PetscParMatrix::operator=(const PetscParMatrix& B)
{
   if (A)
   {
      MPI_Comm comm = PetscObjectComm((PetscObject)A);
      ierr = MatDestroy(&A); CCHKERRQ(comm,ierr);
      if (X) { delete X; }
      if (Y) { delete Y; }
      X = Y = NULL;
   }
   height = B.Height();
   width  = B.Width();
   ierr   = MatDuplicate(B,MAT_COPY_VALUES,&A); CCHKERRQ(B.GetComm(),ierr);
   return *this;
}

PetscParMatrix& PetscParMatrix::operator+=(const PetscParMatrix& B)
{
   if (!A)
   {
      ierr = MatDuplicate(B,MAT_COPY_VALUES,&A); CCHKERRQ(B.GetComm(),ierr);
   }
   else
   {
      MFEM_VERIFY(height == B.Height(),"Invalid number of local rows");
      MFEM_VERIFY(width  == B.Width(), "Invalid number of local columns");
      ierr = MatAXPY(A,1.0,B,DIFFERENT_NONZERO_PATTERN); CCHKERRQ(B.GetComm(),ierr);
   }
   return *this;
}

PetscParMatrix& PetscParMatrix::operator-=(const PetscParMatrix& B)
{
   if (!A)
   {
      ierr = MatDuplicate(B,MAT_COPY_VALUES,&A); CCHKERRQ(B.GetComm(),ierr);
      ierr = MatScale(A,-1.0); PCHKERRQ(A,ierr);
   }
   else
   {
      MFEM_VERIFY(height == B.Height(),"Invalid number of local rows");
      MFEM_VERIFY(width  == B.Width(), "Invalid number of local columns");
      ierr = MatAXPY(A,-1.0,B,DIFFERENT_NONZERO_PATTERN); CCHKERRQ(B.GetComm(),ierr);
   }
   return *this;
}

void PetscParMatrix::
BlockDiagonalConstructor(MPI_Comm comm,
                         PetscInt *row_starts, PetscInt *col_starts,
                         SparseMatrix *diag, bool assembled, Mat* Ad)
{
   Mat      A;
   PetscInt lrsize,lcsize,rstart,cstart;
   PetscMPIInt myid = 0,commsize;

   mpiierr = MPI_Comm_size(comm,&commsize); CCHKERRQ(comm,mpiierr);
   if (!HYPRE_AssumedPartitionCheck())
   {
      mpiierr = MPI_Comm_rank(comm,&myid); CCHKERRQ(comm,mpiierr);
   }
   lrsize = row_starts[myid+1]-row_starts[myid];
   rstart = row_starts[myid];
   lcsize = col_starts[myid+1]-col_starts[myid];
   cstart = col_starts[myid];

   if (!assembled)
   {
      IS is;
      ierr = ISCreateStride(comm,diag->Height(),rstart,1,&is); CCHKERRQ(comm,ierr);
      ISLocalToGlobalMapping rl2g,cl2g;
      ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g); PCHKERRQ(is,ierr);
      ierr = ISDestroy(&is); CCHKERRQ(comm,ierr);
      if (row_starts != col_starts)
      {
         ierr = ISCreateStride(comm,diag->Width(),cstart,1,&is);
         CCHKERRQ(comm,ierr);
         ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g); PCHKERRQ(is,ierr);
         ierr = ISDestroy(&is); CCHKERRQ(comm,ierr);
      }
      else
      {
         ierr = PetscObjectReference((PetscObject)rl2g); PCHKERRQ(rl2g,ierr);
         cl2g = rl2g;
      }

      // Create the PETSc object (MATIS format)
      ierr = MatCreate(comm,&A); CCHKERRQ(comm,ierr);
      ierr = MatSetSizes(A,lrsize,lcsize,PETSC_DECIDE,PETSC_DECIDE);
      PCHKERRQ(A,ierr);
      ierr = MatSetType(A,MATIS); PCHKERRQ(A,ierr);
      ierr = MatSetLocalToGlobalMapping(A,rl2g,cl2g); PCHKERRQ(A,ierr);
      ierr = ISLocalToGlobalMappingDestroy(&rl2g); PCHKERRQ(A,ierr)
      ierr = ISLocalToGlobalMappingDestroy(&cl2g); PCHKERRQ(A,ierr)

      // Copy SparseMatrix into PETSc SeqAIJ format
      // pass through host for now
      Mat lA;
      ierr = MatISGetLocalMat(A,&lA); PCHKERRQ(A,ierr);
      const int *II = diag->HostReadI();
      const int *JJ = diag->HostReadJ();
#if defined(PETSC_USE_64BIT_INDICES)
      PetscInt *pII,*pJJ;
      int m = diag->Height()+1, nnz = II[diag->Height()];
      ierr = PetscMalloc2(m,&pII,nnz,&pJJ); PCHKERRQ(lA,ierr);
      for (int i = 0; i < m; i++) { pII[i] = II[i]; }
      for (int i = 0; i < nnz; i++) { pJJ[i] = JJ[i]; }
      ierr = MatSeqAIJSetPreallocationCSR(lA,pII,pJJ,
                                          diag->HostReadData()); PCHKERRQ(lA,ierr);
      ierr = PetscFree2(pII,pJJ); PCHKERRQ(lA,ierr);
#else
      ierr = MatSeqAIJSetPreallocationCSR(lA,II,JJ,
                                          diag->HostReadData()); PCHKERRQ(lA,ierr);
#endif
   }
   else
   {
      PetscScalar *da;
      PetscInt    *dii,*djj,*oii,
                  m = diag->Height()+1, nnz = diag->NumNonZeroElems();

      diag->SortColumnIndices();
      // if we can take ownership of the SparseMatrix arrays, we can avoid this
      // step
      ierr = PetscMalloc1(m,&dii); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = PetscMalloc1(nnz,&djj); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = PetscMalloc1(nnz,&da); CCHKERRQ(PETSC_COMM_SELF,ierr);
      if (sizeof(PetscInt) == sizeof(int))
      {
         ierr = PetscMemcpy(dii,diag->HostReadI(),m*sizeof(PetscInt));
         CCHKERRQ(PETSC_COMM_SELF,ierr);
         ierr = PetscMemcpy(djj,diag->HostReadJ(),nnz*sizeof(PetscInt));
         CCHKERRQ(PETSC_COMM_SELF,ierr);
      }
      else
      {
         const int *iii = diag->HostReadI();
         const int *jjj = diag->HostReadJ();
         for (int i = 0; i < m; i++) { dii[i] = iii[i]; }
         for (int i = 0; i < nnz; i++) { djj[i] = jjj[i]; }
      }
      ierr = PetscMemcpy(da,diag->HostReadData(),nnz*sizeof(PetscScalar));
      CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = PetscCalloc1(m,&oii);
      CCHKERRQ(PETSC_COMM_SELF,ierr);
      if (commsize > 1)
      {
         ierr = MatCreateMPIAIJWithSplitArrays(comm,lrsize,lcsize,PETSC_DECIDE,
                                               PETSC_DECIDE,
                                               dii,djj,da,oii,NULL,NULL,&A);
         CCHKERRQ(comm,ierr);
      }
      else
      {
         ierr = MatCreateSeqAIJWithArrays(comm,lrsize,lcsize,dii,djj,da,&A);
         CCHKERRQ(comm,ierr);
      }

      void *ptrs[4] = {dii,djj,da,oii};
      const char *names[4] = {"_mfem_csr_dii",
                              "_mfem_csr_djj",
                              "_mfem_csr_da",
                              "_mfem_csr_oii",
                             };
      for (PetscInt i=0; i<4; i++)
      {
         PetscContainer c;

         ierr = PetscContainerCreate(comm,&c); CCHKERRQ(comm,ierr);
         ierr = PetscContainerSetPointer(c,ptrs[i]); CCHKERRQ(comm,ierr);
         ierr = PetscContainerSetUserDestroy(c,__mfem_array_container_destroy);
         CCHKERRQ(comm,ierr);
         ierr = PetscObjectCompose((PetscObject)A,names[i],(PetscObject)c);
         CCHKERRQ(comm,ierr);
         ierr = PetscContainerDestroy(&c); CCHKERRQ(comm,ierr);
      }
   }

   // Tell PETSc the matrix is ready to be used
   ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); PCHKERRQ(A,ierr);
   ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); PCHKERRQ(A,ierr);

   *Ad = A;
}

MPI_Comm PetscParMatrix::GetComm() const
{
   return A ? PetscObjectComm((PetscObject)A) : MPI_COMM_NULL;
}

// TODO ADD THIS CONSTRUCTOR
//PetscParMatrix::PetscParMatrix(MPI_Comm comm, int nrows, PetscInt glob_nrows,
//                  PetscInt glob_ncols, int *I, PetscInt *J,
//                  mfem::real_t *data, PetscInt *rows, PetscInt *cols)
//{
//}

// TODO This should take a reference on op but how?
void PetscParMatrix::MakeWrapper(MPI_Comm comm, const Operator* op, Mat *A)
{
   ierr = MatCreate(comm,A); CCHKERRQ(comm,ierr);
   ierr = MatSetSizes(*A,op->Height(),op->Width(),
                      PETSC_DECIDE,PETSC_DECIDE); PCHKERRQ(A,ierr);
   ierr = MatSetType(*A,MATSHELL); PCHKERRQ(A,ierr);
   ierr = MatShellSetContext(*A,(void *)op); PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_MULT,
                               (void (*)())__mfem_mat_shell_apply);
   PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_MULT_TRANSPOSE,
                               (void (*)())__mfem_mat_shell_apply_transpose);
   PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_COPY,
                               (void (*)())__mfem_mat_shell_copy);
   PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_DESTROY,
                               (void (*)())__mfem_mat_shell_destroy);
#if defined(_USE_DEVICE)
   MemoryType mt = GetMemoryType(op->GetMemoryClass());
   if (mt == MemoryType::DEVICE || mt == MemoryType::MANAGED)
   {
      ierr = MatShellSetVecType(*A,PETSC_VECDEVICE); PCHKERRQ(A,ierr);
      ierr = MatBindToCPU(*A,PETSC_FALSE); PCHKERRQ(A,ierr);
   }
   else
   {
      ierr = MatBindToCPU(*A,PETSC_TRUE); PCHKERRQ(A,ierr);
   }
#endif
   PCHKERRQ(A,ierr);
   ierr = MatSetUp(*A); PCHKERRQ(*A,ierr);
}

void PetscParMatrix::ConvertOperator(MPI_Comm comm, const Operator &op, Mat* A,
                                     Operator::Type tid)
{
   PetscParMatrix   *pA = const_cast<PetscParMatrix *>
                          (dynamic_cast<const PetscParMatrix *>(&op));
   HypreParMatrix   *pH = const_cast<HypreParMatrix *>
                          (dynamic_cast<const HypreParMatrix *>(&op));
   BlockOperator    *pB = const_cast<BlockOperator *>
                          (dynamic_cast<const BlockOperator *>(&op));
   IdentityOperator *pI = const_cast<IdentityOperator *>
                          (dynamic_cast<const IdentityOperator *>(&op));
   SparseMatrix     *pS = const_cast<SparseMatrix *>
                          (dynamic_cast<const SparseMatrix *>(&op));

   if (pA && tid == ANY_TYPE) // use same object and return
   {
      ierr = PetscObjectReference((PetscObject)(pA->A));
      CCHKERRQ(pA->GetComm(),ierr);
      *A = pA->A;
      return;
   }

   PetscBool avoidmatconvert = PETSC_FALSE;
   if (pA) // we test for these types since MatConvert will fail
   {
      ierr = PetscObjectTypeCompareAny((PetscObject)(pA->A),&avoidmatconvert,MATMFFD,
                                       MATSHELL,"");
      CCHKERRQ(comm,ierr);
   }
   if (pA && !avoidmatconvert)
   {
      Mat       At = NULL;
      PetscBool istrans;
#if PETSC_VERSION_LT(3,10,0)
      PetscBool ismatis;
#endif

#if PETSC_VERSION_LT(3,18,0)
      ierr = PetscObjectTypeCompare((PetscObject)(pA->A),MATTRANSPOSEMAT,&istrans);
#else
      ierr = PetscObjectTypeCompare((PetscObject)(pA->A),MATTRANSPOSEVIRTUAL,
                                    &istrans);
#endif
      CCHKERRQ(pA->GetComm(),ierr);
      if (!istrans)
      {
         if (tid == pA->GetType()) // use same object and return
         {
            ierr = PetscObjectReference((PetscObject)(pA->A));
            CCHKERRQ(pA->GetComm(),ierr);
            *A = pA->A;
            return;
         }
#if PETSC_VERSION_LT(3,10,0)
         ierr = PetscObjectTypeCompare((PetscObject)(pA->A),MATIS,&ismatis);
         CCHKERRQ(pA->GetComm(),ierr);
#endif
      }
      else
      {
         ierr = MatTransposeGetMat(pA->A,&At); CCHKERRQ(pA->GetComm(),ierr);
#if PETSC_VERSION_LT(3,10,0)
         ierr = PetscObjectTypeCompare((PetscObject)(At),MATIS,&ismatis);
#endif
         CCHKERRQ(pA->GetComm(),ierr);
      }

      // Try to convert
      if (tid == PETSC_MATAIJ)
      {
#if PETSC_VERSION_LT(3,10,0)
         if (ismatis)
         {
            if (istrans)
            {
               Mat B;

               ierr = MatISGetMPIXAIJ(At,MAT_INITIAL_MATRIX,&B); PCHKERRQ(pA->A,ierr);
               ierr = MatCreateTranspose(B,A); PCHKERRQ(pA->A,ierr);
               ierr = MatDestroy(&B); PCHKERRQ(pA->A,ierr);
            }
            else
            {
               ierr = MatISGetMPIXAIJ(pA->A,MAT_INITIAL_MATRIX,A);
               PCHKERRQ(pA->A,ierr);
            }
         }
         else
#endif
         {
            PetscMPIInt size;
            mpiierr = MPI_Comm_size(comm,&size); CCHKERRQ(comm,mpiierr);

            // call MatConvert and see if a converter is available
            if (istrans)
            {
               Mat B;
               ierr = MatConvert(At,size > 1 ? MATMPIAIJ : MATSEQAIJ,MAT_INITIAL_MATRIX,&B);
               PCHKERRQ(pA->A,ierr);
               ierr = MatCreateTranspose(B,A); PCHKERRQ(pA->A,ierr);
               ierr = MatDestroy(&B); PCHKERRQ(pA->A,ierr);
            }
            else
            {
               ierr = MatConvert(pA->A, size > 1 ? MATMPIAIJ : MATSEQAIJ,MAT_INITIAL_MATRIX,A);
               PCHKERRQ(pA->A,ierr);
            }
         }
      }
      else if (tid == PETSC_MATIS)
      {
         if (istrans)
         {
            Mat B;
            ierr = MatConvert(At,MATIS,MAT_INITIAL_MATRIX,&B); PCHKERRQ(pA->A,ierr);
            ierr = MatCreateTranspose(B,A); PCHKERRQ(pA->A,ierr);
            ierr = MatDestroy(&B); PCHKERRQ(pA->A,ierr);
         }
         else
         {
            ierr = MatConvert(pA->A,MATIS,MAT_INITIAL_MATRIX,A); PCHKERRQ(pA->A,ierr);
         }
      }
      else if (tid == PETSC_MATHYPRE)
      {
#if defined(PETSC_HAVE_HYPRE)
         if (istrans)
         {
            Mat B;
            ierr = MatConvert(At,MATHYPRE,MAT_INITIAL_MATRIX,&B); PCHKERRQ(pA->A,ierr);
            ierr = MatCreateTranspose(B,A); PCHKERRQ(pA->A,ierr);
            ierr = MatDestroy(&B); PCHKERRQ(pA->A,ierr);
         }
         else
         {
            ierr = MatConvert(pA->A,MATHYPRE,MAT_INITIAL_MATRIX,A); PCHKERRQ(pA->A,ierr);
         }
#else
         MFEM_ABORT("Reconfigure PETSc with --download-hypre or --with-hypre")
#endif
      }
      else if (tid == PETSC_MATSHELL)
      {
         MakeWrapper(comm,&op,A);
      }
      else
      {
         MFEM_ABORT("Unsupported operator type conversion " << tid)
      }
   }
   else if (pH)
   {
      if (tid == PETSC_MATAIJ)
      {
#if defined(PETSC_HAVE_HYPRE)
         ierr = MatCreateFromParCSR(const_cast<HypreParMatrix&>(*pH),MATAIJ,
                                    PETSC_USE_POINTER,A);
#else
         ierr = MatConvert_hypreParCSR_AIJ(const_cast<HypreParMatrix&>(*pH),A);
#endif
         CCHKERRQ(pH->GetComm(),ierr);
      }
      else if (tid == PETSC_MATIS)
      {
#if defined(PETSC_HAVE_HYPRE)
         ierr = MatCreateFromParCSR(const_cast<HypreParMatrix&>(*pH),MATIS,
                                    PETSC_USE_POINTER,A);
#else
         ierr = MatConvert_hypreParCSR_IS(const_cast<HypreParMatrix&>(*pH),A);
#endif
         CCHKERRQ(pH->GetComm(),ierr);
      }
      else if (tid == PETSC_MATHYPRE || tid == ANY_TYPE)
      {
#if defined(PETSC_HAVE_HYPRE)
         ierr = MatCreateFromParCSR(const_cast<HypreParMatrix&>(*pH),MATHYPRE,
                                    PETSC_USE_POINTER,A);
         CCHKERRQ(pH->GetComm(),ierr);
#else
         MFEM_ABORT("Reconfigure PETSc with --download-hypre or --with-hypre")
#endif
      }
      else if (tid == PETSC_MATSHELL)
      {
         MakeWrapper(comm,&op,A);
      }
      else
      {
         MFEM_ABORT("Conversion from HypreParCSR to operator type = " << tid <<
                    " is not implemented");
      }
   }
   else if (pB)
   {
      Mat      *mats,*matsl2l = NULL;
      PetscInt i,j,nr,nc;

      nr = pB->NumRowBlocks();
      nc = pB->NumColBlocks();
      ierr = PetscCalloc1(nr*nc,&mats); CCHKERRQ(PETSC_COMM_SELF,ierr);
      if (tid == PETSC_MATIS)
      {
         ierr = PetscCalloc1(nr,&matsl2l); CCHKERRQ(PETSC_COMM_SELF,ierr);
      }
      for (i=0; i<nr; i++)
      {
         PetscBool needl2l = PETSC_TRUE;

         for (j=0; j<nc; j++)
         {
            if (!pB->IsZeroBlock(i,j))
            {
               ConvertOperator(comm,pB->GetBlock(i,j),&mats[i*nc+j],tid);
               if (tid == PETSC_MATIS && needl2l)
               {
                  PetscContainer c;
                  ierr = PetscObjectQuery((PetscObject)mats[i*nc+j],"_MatIS_PtAP_l2l",
                                          (PetscObject*)&c);
                  PCHKERRQ(mats[i*nc+j],ierr);
                  // special case for block operators: the local Vdofs should be
                  // ordered as:
                  // [f1_1,...f1_N1,f2_1,...,f2_N2,...,fm_1,...,fm_Nm]
                  // with m fields, Ni the number of Vdofs for the i-th field
                  if (c)
                  {
                     Array<Mat> *l2l = NULL;
                     ierr = PetscContainerGetPointer(c,(void**)&l2l);
                     PCHKERRQ(c,ierr);
                     MFEM_VERIFY(l2l->Size() == 1,"Unexpected size "
                                 << l2l->Size() << " for block row " << i );
                     ierr = PetscObjectReference((PetscObject)(*l2l)[0]);
                     PCHKERRQ(c,ierr);
                     matsl2l[i] = (*l2l)[0];
                     needl2l = PETSC_FALSE;
                  }
               }
            }
         }
      }
      ierr = MatCreateNest(comm,nr,NULL,nc,NULL,mats,A); CCHKERRQ(comm,ierr);
      if (tid == PETSC_MATIS)
      {
         ierr = MatConvert(*A,MATIS,MAT_INPLACE_MATRIX,A); CCHKERRQ(comm,ierr);

         mfem::Array<Mat> *vmatsl2l = new mfem::Array<Mat>(nr);
         for (int i=0; i<(int)nr; i++) { (*vmatsl2l)[i] = matsl2l[i]; }
         ierr = PetscFree(matsl2l); CCHKERRQ(PETSC_COMM_SELF,ierr);

         PetscContainer c;
         ierr = PetscContainerCreate(comm,&c); CCHKERRQ(comm,ierr);
         ierr = PetscContainerSetPointer(c,vmatsl2l); PCHKERRQ(c,ierr);
         ierr = PetscContainerSetUserDestroy(c,__mfem_matarray_container_destroy);
         PCHKERRQ(c,ierr);
         ierr = PetscObjectCompose((PetscObject)(*A),"_MatIS_PtAP_l2l",(PetscObject)c);
         PCHKERRQ((*A),ierr);
         ierr = PetscContainerDestroy(&c); CCHKERRQ(comm,ierr);
      }
      for (i=0; i<nr*nc; i++) { ierr = MatDestroy(&mats[i]); CCHKERRQ(comm,ierr); }
      ierr = PetscFree(mats); CCHKERRQ(PETSC_COMM_SELF,ierr);
   }
   else if (pI && tid == PETSC_MATAIJ)
   {
      PetscInt rst;

      ierr = MatCreate(comm,A); CCHKERRQ(comm,ierr);
      ierr = MatSetSizes(*A,pI->Height(),pI->Width(),PETSC_DECIDE,PETSC_DECIDE);
      PCHKERRQ(A,ierr);
      ierr = MatSetType(*A,MATAIJ); PCHKERRQ(*A,ierr);
      ierr = MatMPIAIJSetPreallocation(*A,1,NULL,0,NULL); PCHKERRQ(*A,ierr);
      ierr = MatSeqAIJSetPreallocation(*A,1,NULL); PCHKERRQ(*A,ierr);
      ierr = MatSetOption(*A,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE); PCHKERRQ(*A,ierr);
      ierr = MatGetOwnershipRange(*A,&rst,NULL); PCHKERRQ(*A,ierr);
      for (PetscInt i = rst; i < rst+pI->Height(); i++)
      {
         ierr = MatSetValue(*A,i,i,1.,INSERT_VALUES); PCHKERRQ(*A,ierr);
      }
      ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY); PCHKERRQ(*A,ierr);
      ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY); PCHKERRQ(*A,ierr);
   }
   else if (pS)
   {
      if (tid == PETSC_MATSHELL)
      {
         MakeWrapper(comm,&op,A);
      }
      else
      {
         /* from SparseMatrix to SEQAIJ -> always pass through host for now */
         Mat B;
         PetscScalar *pdata;
         PetscInt *pii,*pjj,*oii;
         PetscMPIInt size;

         int m = pS->Height();
         int n = pS->Width();
         const int *ii = pS->HostReadI();
         const int *jj = pS->HostReadJ();
         const mfem::real_t *data = pS->HostReadData();

         ierr = PetscMalloc1(m+1,&pii); CCHKERRQ(PETSC_COMM_SELF,ierr);
         ierr = PetscMalloc1(ii[m],&pjj); CCHKERRQ(PETSC_COMM_SELF,ierr);
         ierr = PetscMalloc1(ii[m],&pdata); CCHKERRQ(PETSC_COMM_SELF,ierr);
         pii[0] = ii[0];
         for (int i = 0; i < m; i++)
         {
            bool issorted = true;
            pii[i+1] = ii[i+1];
            for (int j = ii[i]; j < ii[i+1]; j++)
            {
               pjj[j] = jj[j];
               if (issorted && j != ii[i]) { issorted = (pjj[j] > pjj[j-1]); }
               pdata[j] = data[j];
            }
            if (!issorted)
            {
               ierr = PetscSortIntWithScalarArray(pii[i+1]-pii[i],pjj + pii[i],pdata + pii[i]);
               CCHKERRQ(PETSC_COMM_SELF,ierr);
            }
         }

         mpiierr = MPI_Comm_size(comm,&size); CCHKERRQ(comm,mpiierr);
         if (size == 1)
         {
            ierr = MatCreateSeqAIJWithArrays(comm,m,n,pii,pjj,pdata,&B);
            CCHKERRQ(comm,ierr);
            oii = NULL;
         }
         else // block diagonal constructor
         {
            ierr = PetscCalloc1(m+1,&oii); CCHKERRQ(PETSC_COMM_SELF,ierr);
            ierr = MatCreateMPIAIJWithSplitArrays(comm,m,n,PETSC_DECIDE,
                                                  PETSC_DECIDE,
                                                  pii,pjj,pdata,oii,NULL,NULL,&B);
            CCHKERRQ(comm,ierr);
         }
         void *ptrs[4] = {pii,pjj,pdata,oii};
         const char *names[4] = {"_mfem_csr_pii",
                                 "_mfem_csr_pjj",
                                 "_mfem_csr_pdata",
                                 "_mfem_csr_oii"
                                };
         for (int i=0; i<4; i++)
         {
            PetscContainer c;

            ierr = PetscContainerCreate(PETSC_COMM_SELF,&c); PCHKERRQ(B,ierr);
            ierr = PetscContainerSetPointer(c,ptrs[i]); PCHKERRQ(B,ierr);
            ierr = PetscContainerSetUserDestroy(c,__mfem_array_container_destroy);
            PCHKERRQ(B,ierr);
            ierr = PetscObjectCompose((PetscObject)(B),names[i],(PetscObject)c);
            PCHKERRQ(B,ierr);
            ierr = PetscContainerDestroy(&c); PCHKERRQ(B,ierr);
         }
         if (tid == PETSC_MATAIJ)
         {
            *A = B;
         }
         else if (tid == PETSC_MATHYPRE)
         {
            ierr = MatConvert(B,MATHYPRE,MAT_INITIAL_MATRIX,A); PCHKERRQ(B,ierr);
            ierr = MatDestroy(&B); PCHKERRQ(*A,ierr);
         }
         else if (tid == PETSC_MATIS)
         {
            ierr = MatConvert(B,MATIS,MAT_INITIAL_MATRIX,A); PCHKERRQ(B,ierr);
            ierr = MatDestroy(&B); PCHKERRQ(*A,ierr);
         }
         else
         {
            MFEM_ABORT("Unsupported operator type conversion " << tid)
         }
      }
   }
   else // fallback to general operator
   {
      MFEM_VERIFY(tid == PETSC_MATSHELL || tid == PETSC_MATAIJ || tid == ANY_TYPE,
                  "Supported types are ANY_TYPE, PETSC_MATSHELL or PETSC_MATAIJ");
      MakeWrapper(comm,&op,A);
      if (tid == PETSC_MATAIJ)
      {
         Mat B;
         PetscBool isaij;

         ierr = MatComputeOperator(*A,MATMPIAIJ,&B); CCHKERRQ(comm,ierr);
         ierr = PetscObjectTypeCompare((PetscObject)B,MATMPIAIJ,&isaij);
         CCHKERRQ(comm,ierr);
         ierr = MatDestroy(A); CCHKERRQ(comm,ierr);
         if (!isaij)
         {
            ierr = MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,A); CCHKERRQ(comm,ierr);
            ierr = MatDestroy(&B); CCHKERRQ(comm,ierr);
         }
         else
         {
            *A = B;
         }
      }
   }
   SetUpForDevice();
}

void PetscParMatrix::Destroy()
{
   if (A != NULL)
   {
      MPI_Comm comm = MPI_COMM_NULL;
      ierr = PetscObjectGetComm((PetscObject)A,&comm); PCHKERRQ(A,ierr);
      ierr = MatDestroy(&A); CCHKERRQ(comm,ierr);
   }
   delete X;
   delete Y;
   X = Y = NULL;
}

PetscParMatrix::PetscParMatrix(petsc::Mat a, bool ref)
{
   if (ref)
   {
      ierr = PetscObjectReference((PetscObject)a); PCHKERRQ(a,ierr);
   }
   Init();
   A = a;
   height = GetNumRows();
   width = GetNumCols();
}

void PetscParMatrix::SetMat(Mat A_)
{
   if (A_ == A) { return; }
   Destroy();
   ierr = PetscObjectReference((PetscObject)A_); PCHKERRQ(A_,ierr);
   A = A_;
   height = GetNumRows();
   width = GetNumCols();
}

void PetscParMatrix::SetUpForDevice()
{
#if !defined(_USE_DEVICE)
   return;
#else
   if (!A || (!Device::Allows(Backend::CUDA_MASK) &&
              !Device::Allows(Backend::HIP_MASK)))
   {
      if (A) { ierr = MatBindToCPU(A, PETSC_TRUE); PCHKERRQ(A,ierr); }
      return;
   }
   PetscBool ismatis,isnest,isaij;
   ierr = PetscObjectTypeCompare((PetscObject)A,MATIS,&ismatis);
   PCHKERRQ(A,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)A,MATNEST,&isnest);
   PCHKERRQ(A,ierr);
   Mat tA = A;
   if (ismatis)
   {
      ierr = MatISGetLocalMat(A,&tA); PCHKERRQ(A,ierr);
      ierr = PetscObjectTypeCompare((PetscObject)tA,MATNEST,&isnest);
      PCHKERRQ(tA,ierr);
   }
   if (isnest)
   {
      PetscInt n,m;
      Mat **sub;
      ierr = MatNestGetSubMats(tA,&n,&m,&sub); PCHKERRQ(tA,ierr);
      bool dvec = false;
      for (PetscInt i = 0; i < n; i++)
      {
         for (PetscInt j = 0; j < m; j++)
         {
            if (sub[i][j])
            {
               bool expT = false;
               Mat sA = sub[i][j];
               ierr = PetscObjectTypeCompareAny((PetscObject)sA,&isaij,MATSEQAIJ,MATMPIAIJ,"");
               PCHKERRQ(sA,ierr);
               if (isaij)
               {
                  ierr = MatSetType(sA,PETSC_MATAIJDEVICE); PCHKERRQ(sA,ierr);
                  dvec = true;
                  expT = true;
               }
               if (expT)
               {
                  ierr = MatSetOption(sA,MAT_FORM_EXPLICIT_TRANSPOSE,
                                      PETSC_TRUE); PCHKERRQ(sA,ierr);
               }
            }
         }
      }
      if (dvec)
      {
         ierr = MatSetVecType(tA,PETSC_VECDEVICE); PCHKERRQ(tA,ierr);
      }
   }
   else
   {
      bool expT = false;
      ierr = PetscObjectTypeCompareAny((PetscObject)tA,&isaij,MATSEQAIJ,MATMPIAIJ,"");
      PCHKERRQ(tA,ierr);
      if (isaij)
      {
         ierr = MatSetType(tA,PETSC_MATAIJDEVICE); PCHKERRQ(tA,ierr);
         expT = true;
      }
      if (expT)
      {
         ierr = MatSetOption(tA,MAT_FORM_EXPLICIT_TRANSPOSE,
                             PETSC_TRUE); PCHKERRQ(tA,ierr);
      }
   }
#endif
}

// Computes y = alpha * A  * x + beta * y
//       or y = alpha * A^T* x + beta * y
static void MatMultKernel(Mat A,PetscScalar a,Vec X,PetscScalar b,Vec Y,
                          bool transpose)
{
   PetscErrorCode (*f)(Mat,Vec,Vec);
   PetscErrorCode (*fadd)(Mat,Vec,Vec,Vec);
   if (transpose)
   {
      f = MatMultTranspose;
      fadd = MatMultTransposeAdd;
   }
   else
   {
      f = MatMult;
      fadd = MatMultAdd;
   }
   if (a != 0.)
   {
      if (b != 0.)
      {
         ierr = VecScale(Y,b/a); PCHKERRQ(A,ierr);
         ierr = (*fadd)(A,X,Y,Y); PCHKERRQ(A,ierr);
         ierr = VecScale(Y,a); PCHKERRQ(A,ierr);
      }
      else
      {
         ierr = (*f)(A,X,Y); PCHKERRQ(A,ierr);
         ierr = VecScale(Y,a); PCHKERRQ(A,ierr);
      }
   }
   else
   {
      if (b == 1.)
      {
         // do nothing
      }
      else if (b != 0.)
      {
         ierr = VecScale(Y,b); PCHKERRQ(A,ierr);
      }
      else
      {
         ierr = VecSet(Y,0.); PCHKERRQ(A,ierr);
      }
   }
}

void PetscParMatrix::MakeRef(const PetscParMatrix &master)
{
   ierr = PetscObjectReference((PetscObject)master.A); PCHKERRQ(master.A,ierr);
   Destroy();
   Init();
   A = master.A;
   height = master.height;
   width = master.width;
}

PetscParVector * PetscParMatrix::GetX() const
{
   if (!X)
   {
      MFEM_VERIFY(A,"Mat not present");
      X = new PetscParVector(*this,false,false); PCHKERRQ(A,ierr);
   }
   return X;
}

PetscParVector * PetscParMatrix::GetY() const
{
   if (!Y)
   {
      MFEM_VERIFY(A,"Mat not present");
      Y = new PetscParVector(*this,true,false); PCHKERRQ(A,ierr);
   }
   return Y;
}

PetscParMatrix * PetscParMatrix::Transpose(bool action)
{
   Mat B;
   if (action)
   {
      ierr = MatCreateTranspose(A,&B); PCHKERRQ(A,ierr);
   }
   else
   {
      ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&B); PCHKERRQ(A,ierr);
   }
   return new PetscParMatrix(B,false);
}

void PetscParMatrix::operator*=(mfem::real_t s)
{
   ierr = MatScale(A,s); PCHKERRQ(A,ierr);
}

void PetscParMatrix::Mult(mfem::real_t a, const Vector &x, mfem::real_t b,
                          Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Width());
   MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Height());

   PetscParVector *XX = GetX();
   PetscParVector *YY = GetY();
   bool rw = (b != 0.0);
   XX->PlaceMemory(x.GetMemory());
   YY->PlaceMemory(y.GetMemory(),rw);
   MatMultKernel(A,a,XX->x,b,YY->x,false);
   XX->ResetMemory();
   YY->ResetMemory();
}

void PetscParMatrix::MultTranspose(mfem::real_t a, const Vector &x,
                                   mfem::real_t b,
                                   Vector &y) const
{
   MFEM_ASSERT(x.Size() == Height(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Height());
   MFEM_ASSERT(y.Size() == Width(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Width());

   PetscParVector *XX = GetX();
   PetscParVector *YY = GetY();
   bool rw = (b != 0.0);
   XX->PlaceMemory(y.GetMemory(),rw);
   YY->PlaceMemory(x.GetMemory());
   MatMultKernel(A,a,YY->x,b,XX->x,true);
   XX->ResetMemory();
   YY->ResetMemory();
}

void PetscParMatrix::Print(const char *fname, bool binary) const
{
   if (fname)
   {
      PetscViewer view;

      if (binary)
      {
         ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)A),fname,
                                      FILE_MODE_WRITE,&view);
      }
      else
      {
         ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)A),fname,&view);
      }
      PCHKERRQ(A,ierr);
      ierr = MatView(A,view); PCHKERRQ(A,ierr);
      ierr = PetscViewerDestroy(&view); PCHKERRQ(A,ierr);
   }
   else
   {
      ierr = MatView(A,NULL); PCHKERRQ(A,ierr);
   }
}

void PetscParMatrix::ScaleRows(const Vector & s)
{
   MFEM_ASSERT(s.Size() == Height(), "invalid s.Size() = " << s.Size()
               << ", expected size = " << Height());

   PetscParVector *YY = GetY();
   YY->PlaceMemory(s.GetMemory());
   ierr = MatDiagonalScale(A,*YY,NULL); PCHKERRQ(A,ierr);
   YY->ResetMemory();
}

void PetscParMatrix::ScaleCols(const Vector & s)
{
   MFEM_ASSERT(s.Size() == Width(), "invalid s.Size() = " << s.Size()
               << ", expected size = " << Width());

   PetscParVector *XX = GetX();
   XX->PlaceMemory(s.GetMemory());
   ierr = MatDiagonalScale(A,NULL,*XX); PCHKERRQ(A,ierr);
   XX->ResetMemory();
}

void PetscParMatrix::Shift(mfem::real_t s)
{
   ierr = MatShift(A,(PetscScalar)s); PCHKERRQ(A,ierr);
}

void PetscParMatrix::Shift(const Vector & s)
{
   // for matrices with square diagonal blocks only
   MFEM_ASSERT(s.Size() == Height(), "invalid s.Size() = " << s.Size()
               << ", expected size = " << Height());
   MFEM_ASSERT(s.Size() == Width(), "invalid s.Size() = " << s.Size()
               << ", expected size = " << Width());

   PetscParVector *XX = GetX();
   XX->PlaceMemory(s.GetMemory());
   ierr = MatDiagonalSet(A,*XX,ADD_VALUES); PCHKERRQ(A,ierr);
   XX->ResetMemory();
}

PetscParMatrix * TripleMatrixProduct(PetscParMatrix *R, PetscParMatrix *A,
                                     PetscParMatrix *P)
{
   MFEM_VERIFY(A->Width() == P->Height(),
               "Petsc TripleMatrixProduct: Number of local cols of A " << A->Width() <<
               " differs from number of local rows of P " << P->Height());
   MFEM_VERIFY(A->Height() == R->Width(),
               "Petsc TripleMatrixProduct: Number of local rows of A " << A->Height() <<
               " differs from number of local cols of R " << R->Width());
   Mat B;
   ierr = MatMatMatMult(*R,*A,*P,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B);
   PCHKERRQ(*R,ierr);
   return new PetscParMatrix(B);
}

PetscParMatrix * RAP(PetscParMatrix *Rt, PetscParMatrix *A, PetscParMatrix *P)
{
   Mat       pA = *A,pP = *P,pRt = *Rt;
   Mat       B;
   PetscBool Aismatis,Pismatis,Rtismatis;

   MFEM_VERIFY(A->Width() == P->Height(),
               "Petsc RAP: Number of local cols of A " << A->Width() <<
               " differs from number of local rows of P " << P->Height());
   MFEM_VERIFY(A->Height() == Rt->Height(),
               "Petsc RAP: Number of local rows of A " << A->Height() <<
               " differs from number of local rows of Rt " << Rt->Height());
   ierr = PetscObjectTypeCompare((PetscObject)pA,MATIS,&Aismatis);
   PCHKERRQ(pA,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pP,MATIS,&Pismatis);
   PCHKERRQ(pA,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pRt,MATIS,&Rtismatis);
   PCHKERRQ(pA,ierr);
   if (Aismatis &&
       Pismatis &&
       Rtismatis) // handle special case (this code will eventually go into PETSc)
   {
      Mat                    lA,lP,lB,lRt;
      ISLocalToGlobalMapping cl2gP,cl2gRt;
      PetscInt               rlsize,clsize,rsize,csize;

      ierr = MatGetLocalToGlobalMapping(pP,NULL,&cl2gP); PCHKERRQ(pA,ierr);
      ierr = MatGetLocalToGlobalMapping(pRt,NULL,&cl2gRt); PCHKERRQ(pA,ierr);
      ierr = MatGetLocalSize(pP,NULL,&clsize); PCHKERRQ(pP,ierr);
      ierr = MatGetLocalSize(pRt,NULL,&rlsize); PCHKERRQ(pRt,ierr);
      ierr = MatGetSize(pP,NULL,&csize); PCHKERRQ(pP,ierr);
      ierr = MatGetSize(pRt,NULL,&rsize); PCHKERRQ(pRt,ierr);
      ierr = MatCreate(A->GetComm(),&B); PCHKERRQ(pA,ierr);
      ierr = MatSetSizes(B,rlsize,clsize,rsize,csize); PCHKERRQ(B,ierr);
      ierr = MatSetType(B,MATIS); PCHKERRQ(B,ierr);
      ierr = MatSetLocalToGlobalMapping(B,cl2gRt,cl2gP); PCHKERRQ(B,ierr);
      ierr = MatISGetLocalMat(pA,&lA); PCHKERRQ(pA,ierr);
      ierr = MatISGetLocalMat(pP,&lP); PCHKERRQ(pA,ierr);
      ierr = MatISGetLocalMat(pRt,&lRt); PCHKERRQ(pA,ierr);
      if (lRt == lP)
      {
         ierr = MatPtAP(lA,lP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&lB);
         PCHKERRQ(lA,ierr);
      }
      else
      {
         Mat lR;
         ierr = MatTranspose(lRt,MAT_INITIAL_MATRIX,&lR); PCHKERRQ(lRt,ierr);
         ierr = MatMatMatMult(lR,lA,lP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&lB);
         PCHKERRQ(lRt,ierr);
         ierr = MatDestroy(&lR); PCHKERRQ(lRt,ierr);
      }

      // attach lRt matrix to the subdomain local matrix
      // it may be used if markers on vdofs have to be mapped on
      // subdomain true dofs
      {
         mfem::Array<Mat> *vmatsl2l = new mfem::Array<Mat>(1);
         ierr = PetscObjectReference((PetscObject)lRt); PCHKERRQ(lRt,ierr);
         (*vmatsl2l)[0] = lRt;

         PetscContainer c;
         ierr = PetscContainerCreate(PetscObjectComm((PetscObject)B),&c);
         PCHKERRQ(B,ierr);
         ierr = PetscContainerSetPointer(c,vmatsl2l); PCHKERRQ(c,ierr);
         ierr = PetscContainerSetUserDestroy(c,__mfem_matarray_container_destroy);
         PCHKERRQ(c,ierr);
         ierr = PetscObjectCompose((PetscObject)B,"_MatIS_PtAP_l2l",(PetscObject)c);
         PCHKERRQ(B,ierr);
         ierr = PetscContainerDestroy(&c); PCHKERRQ(B,ierr);
      }

      // Set local problem
      ierr = MatISSetLocalMat(B,lB); PCHKERRQ(lB,ierr);
      ierr = MatDestroy(&lB); PCHKERRQ(lA,ierr);
      ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); PCHKERRQ(B,ierr);
      ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); PCHKERRQ(B,ierr);
   }
   else // it raises an error if the PtAP is not supported in PETSc
   {
      if (pP == pRt)
      {
         ierr = MatPtAP(pA,pP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B);
         PCHKERRQ(pA,ierr);
      }
      else
      {
         Mat pR;
         ierr = MatTranspose(pRt,MAT_INITIAL_MATRIX,&pR); PCHKERRQ(Rt,ierr);
         ierr = MatMatMatMult(pR,pA,pP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B);
         PCHKERRQ(pRt,ierr);
         ierr = MatDestroy(&pR); PCHKERRQ(pRt,ierr);
      }
   }
   return new PetscParMatrix(B);
}

PetscParMatrix * RAP(PetscParMatrix *A, PetscParMatrix *P)
{
   PetscParMatrix *out = RAP(P,A,P);
   return out;
}

PetscParMatrix * RAP(HypreParMatrix *hA, PetscParMatrix *P)
{
   PetscParMatrix *out,*A;
#if defined(PETSC_HAVE_HYPRE)
   A = new PetscParMatrix(hA,Operator::PETSC_MATHYPRE);
#else
   A = new PetscParMatrix(hA);
#endif
   out = RAP(P,A,P);
   delete A;
   return out;
}


PetscParMatrix * ParMult(const PetscParMatrix *A, const PetscParMatrix *B)
{
   Mat AB;

   ierr = MatMatMult(*A,*B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AB);
   CCHKERRQ(A->GetComm(),ierr);
   return new PetscParMatrix(AB);
}

PetscParMatrix* PetscParMatrix::EliminateRowsCols(const Array<int> &rows_cols)
{
   Mat Ae;

   PetscParVector dummy(GetComm(),0);
   ierr = MatDuplicate(A,MAT_COPY_VALUES,&Ae); PCHKERRQ(A,ierr);
   EliminateRowsCols(rows_cols,dummy,dummy);
   ierr = MatAXPY(Ae,-1.,A,SAME_NONZERO_PATTERN); PCHKERRQ(A,ierr);
   return new PetscParMatrix(Ae);
}

void PetscParMatrix::EliminateRowsCols(const Array<int> &rows_cols,
                                       const HypreParVector &X,
                                       HypreParVector &B,
                                       mfem::real_t diag)
{
   MFEM_ABORT("Missing PetscParMatrix::EliminateRowsCols() with HypreParVectors");
}

void PetscParMatrix::EliminateRowsCols(const Array<int> &rows_cols,
                                       const PetscParVector &X,
                                       PetscParVector &B,
                                       mfem::real_t diag)
{
   PetscInt M,N;
   ierr = MatGetSize(A,&M,&N); PCHKERRQ(A,ierr);
   MFEM_VERIFY(M == N,"Rectangular case unsupported");

   // TODO: what if a diagonal term is not present?
   ierr = MatSetOption(A,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE); PCHKERRQ(A,ierr);

   // rows need to be in global numbering
   PetscInt rst;
   ierr = MatGetOwnershipRange(A,&rst,NULL); PCHKERRQ(A,ierr);

   IS dir;
   ierr = Convert_Array_IS(GetComm(),true,&rows_cols,rst,&dir); PCHKERRQ(A,ierr);
   if (!X.GlobalSize() && !B.GlobalSize())
   {
      ierr = MatZeroRowsColumnsIS(A,dir,diag,NULL,NULL); PCHKERRQ(A,ierr);
   }
   else
   {
      ierr = MatZeroRowsColumnsIS(A,dir,diag,X,B); PCHKERRQ(A,ierr);
   }
   ierr = ISDestroy(&dir); PCHKERRQ(A,ierr);
}

void PetscParMatrix::EliminateRows(const Array<int> &rows)
{
   ierr = MatSetOption(A,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE); PCHKERRQ(A,ierr);

   // rows need to be in global numbering
   PetscInt rst;
   ierr = MatGetOwnershipRange(A,&rst,NULL); PCHKERRQ(A,ierr);

   IS dir;
   ierr = Convert_Array_IS(GetComm(),true,&rows,rst,&dir); PCHKERRQ(A,ierr);
   ierr = MatZeroRowsIS(A,dir,0.0,NULL,NULL); PCHKERRQ(A,ierr);
   ierr = ISDestroy(&dir); PCHKERRQ(A,ierr);
}

Mat PetscParMatrix::ReleaseMat(bool dereference)
{

   Mat B = A;
   if (dereference)
   {
      MPI_Comm comm = GetComm();
      ierr = PetscObjectDereference((PetscObject)A); CCHKERRQ(comm,ierr);
   }
   A = NULL;
   return B;
}

Operator::Type PetscParMatrix::GetType() const
{
   PetscBool ok;
   MFEM_VERIFY(A, "no associated PETSc Mat object");
   PetscObject oA = (PetscObject)(this->A);
   // map all of MATAIJ, MATSEQAIJ, and MATMPIAIJ to -> PETSC_MATAIJ
   ierr = PetscObjectBaseTypeCompare(oA, MATSEQAIJ, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATAIJ; }
   ierr = PetscObjectBaseTypeCompare(oA, MATMPIAIJ, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATAIJ; }
   ierr = PetscObjectTypeCompare(oA, MATIS, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATIS; }
   ierr = PetscObjectTypeCompare(oA, MATSHELL, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATSHELL; }
   ierr = PetscObjectTypeCompare(oA, MATNEST, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATNEST; }
#if defined(PETSC_HAVE_HYPRE)
   ierr = PetscObjectTypeCompare(oA, MATHYPRE, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATHYPRE; }
#endif
   return PETSC_MATGENERIC;
}

void EliminateBC(PetscParMatrix &A, PetscParMatrix &Ae,
                 const Array<int> &ess_dof_list,
                 const Vector &X, Vector &B)
{
   const PetscScalar *array;
   Mat pA = const_cast<PetscParMatrix&>(A);

   // B -= Ae*X
   Ae.Mult(-1.0, X, 1.0, B);

   Vec diag = const_cast<PetscParVector&>((*A.GetX()));
   ierr = MatGetDiagonal(pA,diag); PCHKERRQ(pA,ierr);
   ierr = VecGetArrayRead(diag,&array); PCHKERRQ(diag,ierr);
   for (int i = 0; i < ess_dof_list.Size(); i++)
   {
      int r = ess_dof_list[i];
      B(r) = array[r] * X(r);
   }
   ierr = VecRestoreArrayRead(diag,&array); PCHKERRQ(diag,ierr);
}

// PetscSolver methods

PetscSolver::PetscSolver() : clcustom(false)
{
   obj = NULL;
   B = X = NULL;
   cid         = -1;
   operatorset = false;
   bchandler   = NULL;
   private_ctx = NULL;
}

PetscSolver::~PetscSolver()
{
   delete B;
   delete X;
   FreePrivateContext();
}

void PetscSolver::SetTol(mfem::real_t tol)
{
   SetRelTol(tol);
}

void PetscSolver::SetRelTol(mfem::real_t tol)
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      ierr = KSPSetTolerances(ksp,tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
   }
   else if (cid == SNES_CLASSID)
   {
      SNES snes = (SNES)obj;
      ierr = SNESSetTolerances(snes,PETSC_DEFAULT,tol,PETSC_DEFAULT,PETSC_DEFAULT,
                               PETSC_DEFAULT);
   }
   else if (cid == TS_CLASSID)
   {
      TS ts = (TS)obj;
      ierr = TSSetTolerances(ts,PETSC_DECIDE,NULL,tol,NULL);
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
   }
   PCHKERRQ(obj,ierr);
}

void PetscSolver::SetAbsTol(mfem::real_t tol)
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,tol,PETSC_DEFAULT,PETSC_DEFAULT);
   }
   else if (cid == SNES_CLASSID)
   {
      SNES snes = (SNES)obj;
      ierr = SNESSetTolerances(snes,tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,
                               PETSC_DEFAULT);
   }
   else if (cid == TS_CLASSID)
   {
      TS ts = (TS)obj;
      ierr = TSSetTolerances(ts,tol,NULL,PETSC_DECIDE,NULL);
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
   }
   PCHKERRQ(obj,ierr);
}

void PetscSolver::SetMaxIter(int max_iter)
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,
                              max_iter);
   }
   else if (cid == SNES_CLASSID)
   {
      SNES snes = (SNES)obj;
      ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,
                               max_iter,PETSC_DEFAULT);
   }
   else if (cid == TS_CLASSID)
   {
      TS ts = (TS)obj;
      ierr = TSSetMaxSteps(ts,max_iter);
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
   }
   PCHKERRQ(obj,ierr);
}


void PetscSolver::SetPrintLevel(int plev)
{
   typedef PetscErrorCode (*myPetscFunc)(void**);
   PetscViewerAndFormat *vf = NULL;
   PetscViewer viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm(obj));

   if (plev > 0)
   {
      ierr = PetscViewerAndFormatCreate(viewer,PETSC_VIEWER_DEFAULT,&vf);
      PCHKERRQ(obj,ierr);
   }
   if (cid == KSP_CLASSID)
   {
      // there are many other options, see the function KSPSetFromOptions() in
      // src/ksp/ksp/interface/itcl.c
      typedef PetscErrorCode (*myMonitor)(KSP,PetscInt,PetscReal,void*);
      KSP ksp = (KSP)obj;
      if (plev >= 0)
      {
         ierr = KSPMonitorCancel(ksp); PCHKERRQ(ksp,ierr);
      }
      if (plev == 1)
      {
#if PETSC_VERSION_LT(3,15,0)
         ierr = KSPMonitorSet(ksp,(myMonitor)KSPMonitorDefault,vf,
#else
         ierr = KSPMonitorSet(ksp,(myMonitor)KSPMonitorResidual,vf,
#endif
                              (myPetscFunc)PetscViewerAndFormatDestroy);
         PCHKERRQ(ksp,ierr);
      }
      else if (plev > 1)
      {
         ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE); PCHKERRQ(ksp,ierr);
         ierr = KSPMonitorSet(ksp,(myMonitor)KSPMonitorSingularValue,vf,
                              (myPetscFunc)PetscViewerAndFormatDestroy);
         PCHKERRQ(ksp,ierr);
         if (plev > 2)
         {
            ierr = PetscViewerAndFormatCreate(viewer,PETSC_VIEWER_DEFAULT,&vf);
            PCHKERRQ(viewer,ierr);
#if PETSC_VERSION_LT(3,15,0)
            ierr = KSPMonitorSet(ksp,(myMonitor)KSPMonitorTrueResidualNorm,vf,
#else
            ierr = KSPMonitorSet(ksp,(myMonitor)KSPMonitorTrueResidual,vf,
#endif
                                 (myPetscFunc)PetscViewerAndFormatDestroy);
            PCHKERRQ(ksp,ierr);
         }
      }
   }
   else if (cid == SNES_CLASSID)
   {
      typedef PetscErrorCode (*myMonitor)(SNES,PetscInt,PetscReal,void*);
      SNES snes = (SNES)obj;
      if (plev >= 0)
      {
         ierr = SNESMonitorCancel(snes); PCHKERRQ(snes,ierr);
      }
      if (plev > 0)
      {
         ierr = SNESMonitorSet(snes,(myMonitor)SNESMonitorDefault,vf,
                               (myPetscFunc)PetscViewerAndFormatDestroy);
         PCHKERRQ(snes,ierr);
      }
   }
   else if (cid == TS_CLASSID)
   {
      TS ts = (TS)obj;
      if (plev >= 0)
      {
         ierr = TSMonitorCancel(ts); PCHKERRQ(ts,ierr);
      }
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
   }
}

MPI_Comm PetscSolver::GetComm() const
{
   return obj ? PetscObjectComm(obj) : MPI_COMM_NULL;
}

void PetscSolver::SetMonitor(PetscSolverMonitor *ctx)
{
   __mfem_monitor_ctx *monctx;
   ierr = PetscNew(&monctx); CCHKERRQ(PETSC_COMM_SELF,ierr);
   monctx->solver = this;
   monctx->monitor = ctx;
   if (cid == KSP_CLASSID)
   {
      ierr = KSPMonitorSet((KSP)obj,__mfem_ksp_monitor,monctx,
                           __mfem_monitor_ctx_destroy);
      PCHKERRQ(obj,ierr);
   }
   else if (cid == SNES_CLASSID)
   {
      ierr = SNESMonitorSet((SNES)obj,__mfem_snes_monitor,monctx,
                            __mfem_monitor_ctx_destroy);
      PCHKERRQ(obj,ierr);
   }
   else if (cid == TS_CLASSID)
   {
      ierr = TSMonitorSet((TS)obj,__mfem_ts_monitor,monctx,
                          __mfem_monitor_ctx_destroy);
      PCHKERRQ(obj,ierr);
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
   }
}

void PetscSolver::SetBCHandler(PetscBCHandler *bch)
{
   bchandler = bch;
   if (cid == SNES_CLASSID)
   {
      __mfem_snes_ctx* snes_ctx = (__mfem_snes_ctx*)private_ctx;
      snes_ctx->bchandler = bchandler;
   }
   else if (cid == TS_CLASSID)
   {
      __mfem_ts_ctx* ts_ctx = (__mfem_ts_ctx*)private_ctx;
      ts_ctx->bchandler = bchandler;
   }
   else
   {
      MFEM_ABORT("Handling of essential bc only implemented for nonlinear and time-dependent solvers");
   }
}

void PetscSolver::SetPreconditionerFactory(PetscPreconditionerFactory *factory)
{
   PC pc = NULL;
   if (cid == TS_CLASSID)
   {
      SNES snes;
      KSP  ksp;

      ierr = TSGetSNES((TS)obj,&snes); PCHKERRQ(obj,ierr);
      ierr = SNESGetKSP(snes,&ksp); PCHKERRQ(obj,ierr);
      ierr = KSPGetPC(ksp,&pc); PCHKERRQ(obj,ierr);
   }
   else if (cid == SNES_CLASSID)
   {
      KSP ksp;

      ierr = SNESGetKSP((SNES)obj,&ksp); PCHKERRQ(obj,ierr);
      ierr = KSPGetPC(ksp,&pc); PCHKERRQ(obj,ierr);
   }
   else if (cid == KSP_CLASSID)
   {
      ierr = KSPGetPC((KSP)obj,&pc); PCHKERRQ(obj,ierr);
   }
   else if (cid == PC_CLASSID)
   {
      pc = (PC)obj;
   }
   else
   {
      MFEM_ABORT("No support for PetscPreconditionerFactory for this object");
   }
   if (factory)
   {
      ierr = MakeShellPCWithFactory(pc,factory); PCHKERRQ(pc,ierr);
   }
   else
   {
      ierr = PCSetType(pc, PCNONE); PCHKERRQ(pc,ierr);
   }
}

void PetscSolver::Customize(bool customize) const
{
   if (!customize) { clcustom = true; }
   if (!clcustom)
   {
      if (cid == PC_CLASSID)
      {
         PC pc = (PC)obj;
         ierr = PCSetFromOptions(pc); PCHKERRQ(pc, ierr);
      }
      else if (cid == KSP_CLASSID)
      {
         KSP ksp = (KSP)obj;
         ierr = KSPSetFromOptions(ksp); PCHKERRQ(ksp, ierr);
      }
      else if (cid == SNES_CLASSID)
      {
         SNES snes = (SNES)obj;
         ierr = SNESSetFromOptions(snes); PCHKERRQ(snes, ierr);
      }
      else if (cid == TS_CLASSID)
      {
         TS ts = (TS)obj;
         ierr = TSSetFromOptions(ts); PCHKERRQ(ts, ierr);
      }
      else
      {
         MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
      }
   }
   clcustom = true;
}

int PetscSolver::GetConverged()
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      KSPConvergedReason reason;
      ierr = KSPGetConvergedReason(ksp,&reason);
      PCHKERRQ(ksp,ierr);
      return reason > 0 ? 1 : 0;
   }
   else if (cid == SNES_CLASSID)
   {
      SNES snes = (SNES)obj;
      SNESConvergedReason reason;
      ierr = SNESGetConvergedReason(snes,&reason);
      PCHKERRQ(snes,ierr);
      return reason > 0 ? 1 : 0;
   }
   else if (cid == TS_CLASSID)
   {
      TS ts = (TS)obj;
      TSConvergedReason reason;
      ierr = TSGetConvergedReason(ts,&reason);
      PCHKERRQ(ts,ierr);
      return reason > 0 ? 1 : 0;
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
      return -1;
   }
}

int PetscSolver::GetNumIterations()
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      PetscInt its;
      ierr = KSPGetIterationNumber(ksp,&its);
      PCHKERRQ(ksp,ierr);
      return its;
   }
   else if (cid == SNES_CLASSID)
   {
      SNES snes = (SNES)obj;
      PetscInt its;
      ierr = SNESGetIterationNumber(snes,&its);
      PCHKERRQ(snes,ierr);
      return its;
   }
   else if (cid == TS_CLASSID)
   {
      TS ts = (TS)obj;
      PetscInt its;
      ierr = TSGetStepNumber(ts,&its);
      PCHKERRQ(ts,ierr);
      return its;
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
      return -1;
   }
}

mfem::real_t PetscSolver::GetFinalNorm()
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      PetscReal norm;
      ierr = KSPGetResidualNorm(ksp,&norm);
      PCHKERRQ(ksp,ierr);
      return norm;
   }
   if (cid == SNES_CLASSID)
   {
      SNES snes = (SNES)obj;
      PetscReal norm;
      ierr = SNESGetFunctionNorm(snes,&norm);
      PCHKERRQ(snes,ierr);
      return norm;
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
      return PETSC_MAX_REAL;
   }
}

void PetscSolver::CreatePrivateContext()
{
   FreePrivateContext();
   if (cid == SNES_CLASSID)
   {
      __mfem_snes_ctx *snes_ctx;
      ierr = PetscNew(&snes_ctx); CCHKERRQ(PETSC_COMM_SELF,ierr);
      snes_ctx->op = NULL;
      snes_ctx->bchandler = NULL;
      snes_ctx->work = NULL;
      snes_ctx->jacType = Operator::PETSC_MATAIJ;
      private_ctx = (void*) snes_ctx;
   }
   else if (cid == TS_CLASSID)
   {
      __mfem_ts_ctx *ts_ctx;
      ierr = PetscNew(&ts_ctx); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ts_ctx->op = NULL;
      ts_ctx->bchandler = NULL;
      ts_ctx->work = NULL;
      ts_ctx->work2 = NULL;
      ts_ctx->cached_shift = std::numeric_limits<PetscReal>::min();
      ts_ctx->cached_ijacstate = -1;
      ts_ctx->cached_rhsjacstate = -1;
      ts_ctx->cached_splits_xstate = -1;
      ts_ctx->cached_splits_xdotstate = -1;
      ts_ctx->type = PetscODESolver::ODE_SOLVER_GENERAL;
      ts_ctx->jacType = Operator::PETSC_MATAIJ;
      private_ctx = (void*) ts_ctx;
   }
}

void PetscSolver::FreePrivateContext()
{
   if (!private_ctx) { return; }
   // free private context's owned objects
   if (cid == SNES_CLASSID)
   {
      __mfem_snes_ctx *snes_ctx = (__mfem_snes_ctx *)private_ctx;
      delete snes_ctx->work;
   }
   else if (cid == TS_CLASSID)
   {
      __mfem_ts_ctx *ts_ctx = (__mfem_ts_ctx *)private_ctx;
      delete ts_ctx->work;
      delete ts_ctx->work2;
   }
   ierr = PetscFree(private_ctx); CCHKERRQ(PETSC_COMM_SELF,ierr);
}

// PetscBCHandler methods

PetscBCHandler::PetscBCHandler(Array<int>& ess_tdof_list,
                               enum PetscBCHandler::Type type_)
   : bctype(type_), setup(false), eval_t(0.0),
     eval_t_cached(std::numeric_limits<mfem::real_t>::min())
{
   SetTDofs(ess_tdof_list);
}

void PetscBCHandler::SetTDofs(Array<int>& list)
{
   ess_tdof_list.SetSize(list.Size());
   ess_tdof_list.Assign(list);
   setup = false;
}

void PetscBCHandler::SetUp(PetscInt n)
{
   if (setup) { return; }
   if (bctype == CONSTANT)
   {
      eval_g.SetSize(n);
      this->Eval(eval_t,eval_g);
      eval_t_cached = eval_t;
   }
   else if (bctype == TIME_DEPENDENT)
   {
      eval_g.SetSize(n);
   }
   setup = true;
}

void PetscBCHandler::ApplyBC(const Vector &x, Vector &y)
{
   (*this).SetUp(x.Size());
   y = x;
   if (bctype == ZERO)
   {
      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         y[ess_tdof_list[i]] = 0.0;
      }
   }
   else
   {
      if (bctype != CONSTANT && eval_t != eval_t_cached)
      {
         Eval(eval_t,eval_g);
         eval_t_cached = eval_t;
      }
      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         y[ess_tdof_list[i]] = eval_g[ess_tdof_list[i]];
      }
   }
}

void PetscBCHandler::ApplyBC(Vector &x)
{
   (*this).SetUp(x.Size());
   if (bctype == ZERO)
   {
      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         x[ess_tdof_list[i]] = 0.0;
      }
   }
   else
   {
      if (bctype != CONSTANT && eval_t != eval_t_cached)
      {
         Eval(eval_t,eval_g);
         eval_t_cached = eval_t;
      }
      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         x[ess_tdof_list[i]] = eval_g[ess_tdof_list[i]];
      }
   }
}

void PetscBCHandler::FixResidualBC(const Vector& x, Vector& y)
{
   (*this).SetUp(x.Size());
   if (bctype == ZERO)
   {
      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         y[ess_tdof_list[i]] = x[ess_tdof_list[i]];
      }
   }
   else
   {
      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         y[ess_tdof_list[i]] = x[ess_tdof_list[i]] - eval_g[ess_tdof_list[i]];
      }
   }
}

void PetscBCHandler::Zero(Vector &x)
{
   (*this).SetUp(x.Size());
   for (int i = 0; i < ess_tdof_list.Size(); ++i)
   {
      x[ess_tdof_list[i]] = 0.0;
   }
}

void PetscBCHandler::ZeroBC(const Vector &x, Vector &y)
{
   (*this).SetUp(x.Size());
   y = x;
   for (int i = 0; i < ess_tdof_list.Size(); ++i)
   {
      y[ess_tdof_list[i]] = 0.0;
   }
}

// PetscLinearSolver methods

PetscLinearSolver::PetscLinearSolver(MPI_Comm comm, const std::string &prefix,
                                     bool wrapin, bool iter_mode)
   : PetscSolver(), Solver(0,iter_mode), wrap(wrapin)
{
   KSP ksp;
   ierr = KSPCreate(comm,&ksp); CCHKERRQ(comm,ierr);
   obj  = (PetscObject)ksp;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   ierr = KSPSetOptionsPrefix(ksp, prefix.c_str()); PCHKERRQ(ksp, ierr);
   ierr = KSPSetInitialGuessNonzero(ksp, (PetscBool)iterative_mode);
   PCHKERRQ(ksp, ierr);
}

PetscLinearSolver::PetscLinearSolver(const PetscParMatrix &A,
                                     const std::string &prefix, bool iter_mode)
   : PetscSolver(), Solver(0,iter_mode), wrap(false)
{
   KSP ksp;
   ierr = KSPCreate(A.GetComm(),&ksp); CCHKERRQ(A.GetComm(),ierr);
   obj  = (PetscObject)ksp;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   ierr = KSPSetOptionsPrefix(ksp, prefix.c_str()); PCHKERRQ(ksp, ierr);
   ierr = KSPSetInitialGuessNonzero(ksp, (PetscBool)iterative_mode);
   PCHKERRQ(ksp, ierr);
   SetOperator(A);
}

PetscLinearSolver::PetscLinearSolver(const HypreParMatrix &A, bool wrapin,
                                     const std::string &prefix, bool iter_mode)
   : PetscSolver(), Solver(0,iter_mode), wrap(wrapin)
{
   KSP ksp;
   ierr = KSPCreate(A.GetComm(),&ksp); CCHKERRQ(A.GetComm(),ierr);
   obj  = (PetscObject)ksp;
   ierr = PetscObjectGetClassId(obj, &cid); PCHKERRQ(obj, ierr);
   ierr = KSPSetOptionsPrefix(ksp, prefix.c_str()); PCHKERRQ(ksp, ierr);
   ierr = KSPSetInitialGuessNonzero(ksp, (PetscBool)iterative_mode);
   PCHKERRQ(ksp, ierr);
   SetOperator(A);
}

void PetscLinearSolver::SetOperator(const Operator &op)
{
   const HypreParMatrix *hA = dynamic_cast<const HypreParMatrix *>(&op);
   PetscParMatrix       *pA = const_cast<PetscParMatrix *>
                              (dynamic_cast<const PetscParMatrix *>(&op));
   const Operator       *oA = dynamic_cast<const Operator *>(&op);

   // update base classes: Operator, Solver, PetscLinearSolver
   bool delete_pA = false;
   if (!pA)
   {
      if (hA)
      {
         // Create MATSHELL object or convert into a format suitable to construct preconditioners
         pA = new PetscParMatrix(hA, wrap ? PETSC_MATSHELL : PETSC_MATAIJ);
         delete_pA = true;
      }
      else if (oA) // fallback to general operator
      {
         // Create MATSHELL or MATNEST (if oA is a BlockOperator) object
         // If oA is a BlockOperator, Operator::Type is relevant to the subblocks
         pA = new PetscParMatrix(PetscObjectComm(obj),oA,
                                 wrap ? PETSC_MATSHELL : PETSC_MATAIJ);
         delete_pA = true;
      }
   }
   MFEM_VERIFY(pA, "Unsupported operation!");

   // Set operators into PETSc KSP
   KSP ksp = (KSP)obj;
   Mat A = pA->A;
   if (operatorset)
   {
      Mat C;
      PetscInt nheight,nwidth,oheight,owidth;

      ierr = KSPGetOperators(ksp,&C,NULL); PCHKERRQ(ksp,ierr);
      ierr = MatGetSize(A,&nheight,&nwidth); PCHKERRQ(A,ierr);
      ierr = MatGetSize(C,&oheight,&owidth); PCHKERRQ(A,ierr);
      if (nheight != oheight || nwidth != owidth)
      {
         // reinit without destroying the KSP
         // communicator remains the same
         ierr = KSPReset(ksp); PCHKERRQ(ksp,ierr);
         delete X;
         delete B;
         X = B = NULL;
      }
   }
   ierr = KSPSetOperators(ksp,A,A); PCHKERRQ(ksp,ierr);

   // Update PetscSolver
   operatorset = true;

   // Update the Operator fields.
   height = pA->Height();
   width  = pA->Width();

   if (delete_pA) { delete pA; }
}

void PetscLinearSolver::SetOperator(const Operator &op, const Operator &pop)
{
   const HypreParMatrix *hA = dynamic_cast<const HypreParMatrix *>(&op);
   PetscParMatrix       *pA = const_cast<PetscParMatrix *>
                              (dynamic_cast<const PetscParMatrix *>(&op));
   const Operator       *oA = dynamic_cast<const Operator *>(&op);

   PetscParMatrix       *ppA = const_cast<PetscParMatrix *>
                               (dynamic_cast<const PetscParMatrix *>(&pop));
   const Operator       *poA = dynamic_cast<const Operator *>(&pop);

   // Convert Operator for linear system
   bool delete_pA = false;
   if (!pA)
   {
      if (hA)
      {
         // Create MATSHELL object or convert into a format suitable to construct preconditioners
         pA = new PetscParMatrix(hA, wrap ? PETSC_MATSHELL : PETSC_MATAIJ);
         delete_pA = true;
      }
      else if (oA) // fallback to general operator
      {
         // Create MATSHELL or MATNEST (if oA is a BlockOperator) object
         // If oA is a BlockOperator, Operator::Type is relevant to the subblocks
         pA = new PetscParMatrix(PetscObjectComm(obj),oA,
                                 wrap ? PETSC_MATSHELL : PETSC_MATAIJ);
         delete_pA = true;
      }
   }
   MFEM_VERIFY(pA, "Unsupported operation!");

   // Convert Operator to be preconditioned
   bool delete_ppA = false;
   if (!ppA)
   {
      if (oA == poA && !wrap) // Same operator, already converted
      {
         ppA = pA;
      }
      else
      {
         ppA = new PetscParMatrix(PetscObjectComm(obj), poA, PETSC_MATAIJ);
         delete_ppA = true;
      }
   }
   MFEM_VERIFY(ppA, "Unsupported operation!");

   // Set operators into PETSc KSP
   KSP ksp = (KSP)obj;
   Mat A = pA->A;
   Mat P = ppA->A;
   if (operatorset)
   {
      Mat C;
      PetscInt nheight,nwidth,oheight,owidth;

      ierr = KSPGetOperators(ksp,&C,NULL); PCHKERRQ(ksp,ierr);
      ierr = MatGetSize(A,&nheight,&nwidth); PCHKERRQ(A,ierr);
      ierr = MatGetSize(C,&oheight,&owidth); PCHKERRQ(A,ierr);
      if (nheight != oheight || nwidth != owidth)
      {
         // reinit without destroying the KSP
         // communicator remains the same
         ierr = KSPReset(ksp); PCHKERRQ(ksp,ierr);
         delete X;
         delete B;
         X = B = NULL;
         wrap = false;
      }
   }
   ierr = KSPSetOperators(ksp,A,P); PCHKERRQ(ksp,ierr);

   // Update PetscSolver
   operatorset = true;

   // Update the Operator fields.
   height = pA->Height();
   width  = pA->Width();

   if (delete_pA) { delete pA; }
   if (delete_ppA) { delete ppA; }
}

void PetscLinearSolver::SetPreconditioner(Solver &precond)
{
   KSP ksp = (KSP)obj;

   // Preserve Amat if already set
   Mat A = NULL;
   PetscBool amat;
   ierr = KSPGetOperatorsSet(ksp,&amat,NULL); PCHKERRQ(ksp,ierr);
   if (amat)
   {
      ierr = KSPGetOperators(ksp,&A,NULL); PCHKERRQ(ksp,ierr);
      ierr = PetscObjectReference((PetscObject)A); PCHKERRQ(ksp,ierr);
   }
   PetscPreconditioner *ppc = dynamic_cast<PetscPreconditioner *>(&precond);
   if (ppc)
   {
      ierr = KSPSetPC(ksp,*ppc); PCHKERRQ(ksp,ierr);
   }
   else
   {
      // wrap the Solver action
      // Solver is assumed to be already setup
      // ownership of precond is not transferred,
      // consistently with other MFEM's linear solvers
      PC pc;
      ierr = KSPGetPC(ksp,&pc); PCHKERRQ(ksp,ierr);
      ierr = MakeShellPC(pc,precond,false); PCHKERRQ(ksp,ierr);
   }
   if (A)
   {
      Mat P;

      ierr = KSPGetOperators(ksp,NULL,&P); PCHKERRQ(ksp,ierr);
      ierr = PetscObjectReference((PetscObject)P); PCHKERRQ(ksp,ierr);
      ierr = KSPSetOperators(ksp,A,P); PCHKERRQ(ksp,ierr);
      ierr = MatDestroy(&A); PCHKERRQ(ksp,ierr);
      ierr = MatDestroy(&P); PCHKERRQ(ksp,ierr);
   }
}

void PetscLinearSolver::MultKernel(const Vector &b, Vector &x, bool trans) const
{
   KSP ksp = (KSP)obj;

   if (!B || !X)
   {
      Mat pA = NULL;
      ierr = KSPGetOperators(ksp, &pA, NULL); PCHKERRQ(obj, ierr);
      if (!B)
      {
         PetscParMatrix A = PetscParMatrix(pA, true);
         B = new PetscParVector(A, true, false);
      }
      if (!X)
      {
         PetscParMatrix A = PetscParMatrix(pA, true);
         X = new PetscParVector(A, false, false);
      }
   }
   B->PlaceMemory(b.GetMemory());

   Customize();

   PetscBool flg;
   ierr = KSPGetInitialGuessNonzero(ksp, &flg);
   X->PlaceMemory(x.GetMemory(),flg);

   // Solve the system.
   if (trans)
   {
      ierr = KSPSolveTranspose(ksp, B->x, X->x); PCHKERRQ(ksp,ierr);
   }
   else
   {
      ierr = KSPSolve(ksp, B->x, X->x); PCHKERRQ(ksp,ierr);
   }
   B->ResetMemory();
   X->ResetMemory();
}

void PetscLinearSolver::Mult(const Vector &b, Vector &x) const
{
   (*this).MultKernel(b,x,false);
}

void PetscLinearSolver::MultTranspose(const Vector &b, Vector &x) const
{
   (*this).MultKernel(b,x,true);
}

PetscLinearSolver::~PetscLinearSolver()
{
   MPI_Comm comm;
   KSP ksp = (KSP)obj;
   ierr = PetscObjectGetComm((PetscObject)ksp,&comm); PCHKERRQ(ksp,ierr);
   ierr = KSPDestroy(&ksp); CCHKERRQ(comm,ierr);
}

// PetscPCGSolver methods

PetscPCGSolver::PetscPCGSolver(MPI_Comm comm, const std::string &prefix,
                               bool iter_mode)
   : PetscLinearSolver(comm,prefix,true,iter_mode)
{
   KSP ksp = (KSP)obj;
   ierr = KSPSetType(ksp,KSPCG); PCHKERRQ(ksp,ierr);
   // this is to obtain a textbook PCG
   ierr = KSPSetNormType(ksp,KSP_NORM_NATURAL); PCHKERRQ(ksp,ierr);
}

PetscPCGSolver::PetscPCGSolver(PetscParMatrix& A, const std::string &prefix,
                               bool iter_mode)
   : PetscLinearSolver(A,prefix,iter_mode)
{
   KSP ksp = (KSP)obj;
   ierr = KSPSetType(ksp,KSPCG); PCHKERRQ(ksp,ierr);
   // this is to obtain a textbook PCG
   ierr = KSPSetNormType(ksp,KSP_NORM_NATURAL); PCHKERRQ(ksp,ierr);
}

PetscPCGSolver::PetscPCGSolver(HypreParMatrix& A, bool wrap,
                               const std::string &prefix, bool iter_mode)
   : PetscLinearSolver(A,wrap,prefix,iter_mode)
{
   KSP ksp = (KSP)obj;
   ierr = KSPSetType(ksp,KSPCG); PCHKERRQ(ksp,ierr);
   // this is to obtain a textbook PCG
   ierr = KSPSetNormType(ksp,KSP_NORM_NATURAL); PCHKERRQ(ksp,ierr);
}

// PetscPreconditioner methods

PetscPreconditioner::PetscPreconditioner(MPI_Comm comm,
                                         const std::string &prefix)
   : PetscSolver(), Solver()
{
   PC pc;
   ierr = PCCreate(comm,&pc); CCHKERRQ(comm,ierr);
   obj  = (PetscObject)pc;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   ierr = PCSetOptionsPrefix(pc, prefix.c_str()); PCHKERRQ(pc, ierr);
}

PetscPreconditioner::PetscPreconditioner(PetscParMatrix &A,
                                         const string &prefix)
   : PetscSolver(), Solver()
{
   PC pc;
   ierr = PCCreate(A.GetComm(),&pc); CCHKERRQ(A.GetComm(),ierr);
   obj  = (PetscObject)pc;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   ierr = PCSetOptionsPrefix(pc, prefix.c_str()); PCHKERRQ(pc, ierr);
   SetOperator(A);
}

PetscPreconditioner::PetscPreconditioner(MPI_Comm comm, Operator &op,
                                         const string &prefix)
   : PetscSolver(), Solver()
{
   PC pc;
   ierr = PCCreate(comm,&pc); CCHKERRQ(comm,ierr);
   obj  = (PetscObject)pc;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   ierr = PCSetOptionsPrefix(pc, prefix.c_str()); PCHKERRQ(pc, ierr);
   SetOperator(op);
}

void PetscPreconditioner::SetOperator(const Operator &op)
{
   bool delete_pA = false;
   PetscParMatrix *pA = const_cast<PetscParMatrix *>
                        (dynamic_cast<const PetscParMatrix *>(&op));

   if (!pA)
   {
      const Operator *cop = dynamic_cast<const Operator *>(&op);
      pA = new PetscParMatrix(PetscObjectComm(obj),cop,PETSC_MATAIJ);
      delete_pA = true;
   }

   // Set operators into PETSc PC
   PC pc = (PC)obj;
   Mat A = pA->A;
   if (operatorset)
   {
      Mat C;
      PetscInt nheight,nwidth,oheight,owidth;

      ierr = PCGetOperators(pc,&C,NULL); PCHKERRQ(pc,ierr);
      ierr = MatGetSize(A,&nheight,&nwidth); PCHKERRQ(A,ierr);
      ierr = MatGetSize(C,&oheight,&owidth); PCHKERRQ(A,ierr);
      if (nheight != oheight || nwidth != owidth)
      {
         // reinit without destroying the PC
         // communicator remains the same
         ierr = PCReset(pc); PCHKERRQ(pc,ierr);
         delete X;
         delete B;
         X = B = NULL;
      }
   }
   ierr = PCSetOperators(pc,pA->A,pA->A); PCHKERRQ(obj,ierr);

   // Update PetscSolver
   operatorset = true;

   // Update the Operator fields.
   height = pA->Height();
   width  = pA->Width();

   if (delete_pA) { delete pA; };
}

void PetscPreconditioner::MultKernel(const Vector &b, Vector &x,
                                     bool trans) const
{
   MFEM_VERIFY(!iterative_mode,
               "Iterative mode not supported for PetscPreconditioner");
   PC pc = (PC)obj;

   if (!B || !X)
   {
      Mat pA = NULL;
      ierr = PCGetOperators(pc, NULL, &pA); PCHKERRQ(obj, ierr);
      if (!B)
      {
         PetscParMatrix A(pA, true);
         B = new PetscParVector(A, true, false);
      }
      if (!X)
      {
         PetscParMatrix A(pA, true);
         X = new PetscParVector(A, false, false);
      }
   }
   B->PlaceMemory(b.GetMemory());
   X->PlaceMemory(x.GetMemory());

   Customize();

   // Apply the preconditioner.
   if (trans)
   {
      ierr = PCApplyTranspose(pc, B->x, X->x); PCHKERRQ(pc, ierr);
   }
   else
   {
      ierr = PCApply(pc, B->x, X->x); PCHKERRQ(pc, ierr);
   }
   B->ResetMemory();
   X->ResetMemory();
}

void PetscPreconditioner::Mult(const Vector &b, Vector &x) const
{
   (*this).MultKernel(b,x,false);
}

void PetscPreconditioner::MultTranspose(const Vector &b, Vector &x) const
{
   (*this).MultKernel(b,x,true);
}

PetscPreconditioner::~PetscPreconditioner()
{
   MPI_Comm comm;
   PC pc = (PC)obj;
   ierr = PetscObjectGetComm((PetscObject)pc,&comm); PCHKERRQ(pc,ierr);
   ierr = PCDestroy(&pc); CCHKERRQ(comm,ierr);
}

// PetscBDDCSolver methods

// Coordinates sampling function
static void func_coords(const Vector &x, Vector &y)
{
   y = x;
}

void PetscBDDCSolver::BDDCSolverConstructor(const PetscBDDCSolverParams &opts)
{
   MPI_Comm comm = PetscObjectComm(obj);

   // get PETSc object
   PC pc = (PC)obj;
   Mat pA;
   ierr = PCGetOperators(pc,NULL,&pA); PCHKERRQ(pc,ierr);

   // matrix type should be of type MATIS
   PetscBool ismatis;
   ierr = PetscObjectTypeCompare((PetscObject)pA,MATIS,&ismatis);
   PCHKERRQ(pA,ierr);
   MFEM_VERIFY(ismatis,"PetscBDDCSolver needs the matrix in unassembled format");

   // Check options
   ParFiniteElementSpace *fespace = opts.fespace;
   if (opts.netflux && !fespace)
   {
      MFEM_WARNING("Don't know how to compute an auxiliary quadrature form without a ParFiniteElementSpace");
   }

   // Attach default near-null space to local matrices
   {
      MatNullSpace nnsp;
      Mat lA;
      ierr = MatISGetLocalMat(pA,&lA); CCHKERRQ(comm,ierr);
      ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)lA),PETSC_TRUE,0,NULL,
                                &nnsp); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = MatSetNearNullSpace(lA,nnsp); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = MatNullSpaceDestroy(&nnsp); CCHKERRQ(PETSC_COMM_SELF,ierr);
   }

   // set PETSc PC type to PCBDDC
   ierr = PCSetType(pc,PCBDDC); PCHKERRQ(obj,ierr);

   PetscInt rst,nl;
   ierr = MatGetOwnershipRange(pA,&rst,NULL); PCHKERRQ(pA,ierr);
   ierr = MatGetLocalSize(pA,&nl,NULL); PCHKERRQ(pA,ierr);

   // index sets for fields splitting and coordinates for nodal spaces
   IS *fields = NULL;
   PetscInt nf = 0;
   PetscInt sdim = 0;
   PetscReal *coords = NULL;
   if (fespace)
   {
      int vdim = fespace->GetVDim();

      // Ideally, the block size should be set at matrix creation
      // but the MFEM assembly does not allow to do so
      if (fespace->GetOrdering() == Ordering::byVDIM)
      {
         Mat lA;
         ierr = MatSetBlockSize(pA,vdim); PCHKERRQ(pA,ierr);
         ierr = MatISGetLocalMat(pA,&lA); CCHKERRQ(PETSC_COMM_SELF,ierr);
         ierr = MatSetBlockSize(lA,vdim); PCHKERRQ(pA,ierr);
      }

      // fields
      if (vdim > 1)
      {
         PetscInt st = rst, bs, inc, nlf;
         nf = vdim;
         nlf = nl/nf;
         ierr = PetscMalloc1(nf,&fields); CCHKERRQ(PETSC_COMM_SELF,ierr);
         if (fespace->GetOrdering() == Ordering::byVDIM)
         {
            inc = 1;
            bs = vdim;
         }
         else
         {
            inc = nlf;
            bs = 1;
         }
         for (PetscInt i = 0; i < nf; i++)
         {
            ierr = ISCreateStride(comm,nlf,st,bs,&fields[i]); CCHKERRQ(comm,ierr);
            st += inc;
         }
      }

      // coordinates
      const FiniteElementCollection *fec = fespace->FEColl();
      bool h1space = dynamic_cast<const H1_FECollection*>(fec);
      if (h1space)
      {
         ParFiniteElementSpace *fespace_coords = fespace;

         sdim = fespace->GetParMesh()->SpaceDimension();
         if (vdim != sdim || fespace->GetOrdering() != Ordering::byVDIM)
         {
            fespace_coords = new ParFiniteElementSpace(fespace->GetParMesh(),fec,sdim,
                                                       Ordering::byVDIM);
         }
         VectorFunctionCoefficient coeff_coords(sdim, func_coords);
         ParGridFunction gf_coords(fespace_coords);
         gf_coords.ProjectCoefficient(coeff_coords);
         HypreParVector *hvec_coords = gf_coords.ParallelProject();
         PetscScalar *data_coords = (PetscScalar*)mfem::Read(hvec_coords->GetMemory(),
                                                             hvec_coords->Size(),false);

         // likely elasticity -> we attach rigid-body modes as near-null space information to the local matrices
         // and to the global matrix
         if (vdim == sdim)
         {
            MatNullSpace nnsp;
            Mat lA;
            Vec pvec_coords,lvec_coords;
            ISLocalToGlobalMapping l2g;
            PetscSF sf;
            PetscLayout rmap;
            const PetscInt *gidxs;
            PetscInt nleaves;

            ierr = VecCreateMPIWithArray(comm,sdim,hvec_coords->Size(),
                                         hvec_coords->GlobalSize(),data_coords,&pvec_coords);
            CCHKERRQ(comm,ierr);
            ierr = MatGetNearNullSpace(pA,&nnsp); CCHKERRQ(comm,ierr);
            if (!nnsp)
            {
               ierr = MatNullSpaceCreateRigidBody(pvec_coords,&nnsp);
               CCHKERRQ(comm,ierr);
               ierr = MatSetNearNullSpace(pA,nnsp); CCHKERRQ(comm,ierr);
               ierr = MatNullSpaceDestroy(&nnsp); CCHKERRQ(comm,ierr);
            }
            ierr = MatISGetLocalMat(pA,&lA); CCHKERRQ(comm,ierr);
            ierr = MatCreateVecs(lA,&lvec_coords,NULL); CCHKERRQ(PETSC_COMM_SELF,ierr);
            ierr = VecSetBlockSize(lvec_coords,sdim); CCHKERRQ(PETSC_COMM_SELF,ierr);
            ierr = MatGetLocalToGlobalMapping(pA,&l2g,NULL); CCHKERRQ(comm,ierr);
            ierr = MatGetLayouts(pA,&rmap,NULL); CCHKERRQ(comm,ierr);
            ierr = PetscSFCreate(comm,&sf); CCHKERRQ(comm,ierr);
            ierr = ISLocalToGlobalMappingGetIndices(l2g,&gidxs); CCHKERRQ(comm,ierr);
            ierr = ISLocalToGlobalMappingGetSize(l2g,&nleaves); CCHKERRQ(comm,ierr);
            ierr = PetscSFSetGraphLayout(sf,rmap,nleaves,NULL,PETSC_OWN_POINTER,gidxs);
            CCHKERRQ(comm,ierr);
            ierr = ISLocalToGlobalMappingRestoreIndices(l2g,&gidxs); CCHKERRQ(comm,ierr);
            {
               const PetscScalar *garray;
               PetscScalar *larray;

               ierr = VecGetArrayRead(pvec_coords,&garray); CCHKERRQ(PETSC_COMM_SELF,ierr);
               ierr = VecGetArray(lvec_coords,&larray); CCHKERRQ(PETSC_COMM_SELF,ierr);
#if PETSC_VERSION_LT(3,15,0)
               ierr = PetscSFBcastBegin(sf,MPIU_SCALAR,garray,larray); CCHKERRQ(comm,ierr);
               ierr = PetscSFBcastEnd(sf,MPIU_SCALAR,garray,larray); CCHKERRQ(comm,ierr);
#else
               ierr = PetscSFBcastBegin(sf,MPIU_SCALAR,garray,larray,MPI_REPLACE);
               CCHKERRQ(comm,ierr);
               ierr = PetscSFBcastEnd(sf,MPIU_SCALAR,garray,larray,MPI_REPLACE);
               CCHKERRQ(comm,ierr);
#endif
               ierr = VecRestoreArrayRead(pvec_coords,&garray); CCHKERRQ(PETSC_COMM_SELF,ierr);
               ierr = VecRestoreArray(lvec_coords,&larray); CCHKERRQ(PETSC_COMM_SELF,ierr);
            }
            ierr = VecDestroy(&pvec_coords); CCHKERRQ(comm,ierr);
            ierr = MatNullSpaceCreateRigidBody(lvec_coords,&nnsp);
            CCHKERRQ(PETSC_COMM_SELF,ierr);
            ierr = VecDestroy(&lvec_coords); CCHKERRQ(PETSC_COMM_SELF,ierr);
            ierr = MatSetNearNullSpace(lA,nnsp); CCHKERRQ(PETSC_COMM_SELF,ierr);
            ierr = MatNullSpaceDestroy(&nnsp); CCHKERRQ(PETSC_COMM_SELF,ierr);
            ierr = PetscSFDestroy(&sf); CCHKERRQ(PETSC_COMM_SELF,ierr);
         }

         // each single dof has associated a tuple of coordinates
         ierr = PetscMalloc1(nl*sdim,&coords); CCHKERRQ(PETSC_COMM_SELF,ierr);
         if (nf > 0)
         {
            for (PetscInt i = 0; i < nf; i++)
            {
               const PetscInt *idxs;
               PetscInt nn;

               // It also handles the case of fespace not ordered by VDIM
               ierr = ISGetLocalSize(fields[i],&nn); CCHKERRQ(comm,ierr);
               ierr = ISGetIndices(fields[i],&idxs); CCHKERRQ(comm,ierr);
               for (PetscInt j = 0; j < nn; j++)
               {
                  PetscInt idx = idxs[j]-rst;
                  for (PetscInt d = 0; d < sdim; d++)
                  {
                     coords[sdim*idx+d] = PetscRealPart(data_coords[sdim*j+d]);
                  }
               }
               ierr = ISRestoreIndices(fields[i],&idxs); CCHKERRQ(comm,ierr);
            }
         }
         else
         {
            for (PetscInt j = 0; j < nl*sdim; j++) { coords[j] = PetscRealPart(data_coords[j]); }
         }
         if (fespace_coords != fespace)
         {
            delete fespace_coords;
         }
         delete hvec_coords;
      }
   }

   // index sets for boundary dofs specification (Essential = dir, Natural = neu)
   IS dir = NULL, neu = NULL;

   // Extract l2l matrices
   Array<Mat> *l2l = NULL;
   if (opts.ess_dof_local || opts.nat_dof_local)
   {
      PetscContainer c;

      ierr = PetscObjectQuery((PetscObject)pA,"_MatIS_PtAP_l2l",(PetscObject*)&c);
      MFEM_VERIFY(c,"Local-to-local PETSc container not present");
      ierr = PetscContainerGetPointer(c,(void**)&l2l); PCHKERRQ(c,ierr);
   }

   // check information about index sets (essential dofs, fields, etc.)
#ifdef MFEM_DEBUG
   {
      // make sure ess/nat_dof have been collectively set
      PetscBool lpr = PETSC_FALSE,pr;
      if (opts.ess_dof) { lpr = PETSC_TRUE; }
      mpiierr = MPI_Allreduce(&lpr,&pr,1,MPIU_BOOL,MPI_LOR,comm);
      CCHKERRQ(comm,mpiierr);
      MFEM_VERIFY(lpr == pr,"ess_dof should be collectively set");
      lpr = PETSC_FALSE;
      if (opts.nat_dof) { lpr = PETSC_TRUE; }
      mpiierr = MPI_Allreduce(&lpr,&pr,1,MPIU_BOOL,MPI_LOR,comm);
      CCHKERRQ(comm,mpiierr);
      MFEM_VERIFY(lpr == pr,"nat_dof should be collectively set");
      // make sure fields have been collectively set
      PetscInt ms[2],Ms[2];
      ms[0] = -nf; ms[1] = nf;
      mpiierr = MPI_Allreduce(&ms,&Ms,2,MPIU_INT,MPI_MAX,comm);
      CCHKERRQ(comm,mpiierr);
      MFEM_VERIFY(-Ms[0] == Ms[1],
                  "number of fields should be the same across processes");
   }
#endif

   // boundary sets
   if (opts.ess_dof)
   {
      PetscInt st = opts.ess_dof_local ? 0 : rst;
      if (!opts.ess_dof_local)
      {
         // need to compute the boundary dofs in global ordering
         ierr = Convert_Array_IS(comm,true,opts.ess_dof,st,&dir);
         CCHKERRQ(comm,ierr);
         ierr = PCBDDCSetDirichletBoundaries(pc,dir); PCHKERRQ(pc,ierr);
      }
      else
      {
         // need to compute a list for the marked boundary dofs in local ordering
         ierr = Convert_Vmarks_IS(comm,*l2l,opts.ess_dof,st,&dir);
         CCHKERRQ(comm,ierr);
         ierr = PCBDDCSetDirichletBoundariesLocal(pc,dir); PCHKERRQ(pc,ierr);
      }
   }
   if (opts.nat_dof)
   {
      PetscInt st = opts.nat_dof_local ? 0 : rst;
      if (!opts.nat_dof_local)
      {
         // need to compute the boundary dofs in global ordering
         ierr = Convert_Array_IS(comm,true,opts.nat_dof,st,&neu);
         CCHKERRQ(comm,ierr);
         ierr = PCBDDCSetNeumannBoundaries(pc,neu); PCHKERRQ(pc,ierr);
      }
      else
      {
         // need to compute a list for the marked boundary dofs in local ordering
         ierr = Convert_Vmarks_IS(comm,*l2l,opts.nat_dof,st,&neu);
         CCHKERRQ(comm,ierr);
         ierr = PCBDDCSetNeumannBoundariesLocal(pc,neu); PCHKERRQ(pc,ierr);
      }
   }

   // field splitting
   if (fields)
   {
      ierr = PCBDDCSetDofsSplitting(pc,nf,fields); PCHKERRQ(pc,ierr);
   }
   for (PetscInt i = 0; i < nf; i++)
   {
      ierr = ISDestroy(&fields[i]); CCHKERRQ(comm,ierr);
   }
   ierr = PetscFree(fields); CCHKERRQ(PETSC_COMM_SELF,ierr);

   // coordinates
   if (coords)
   {
      ierr = PCSetCoordinates(pc,sdim,nl,coords); PCHKERRQ(pc,ierr);
   }
   ierr = PetscFree(coords); CCHKERRQ(PETSC_COMM_SELF,ierr);

   // code for block size is disabled since we cannot change the matrix
   // block size after it has been setup
   // int bs = 1;

   // Customize using the finite element space (if any)
   if (fespace)
   {
      const     FiniteElementCollection *fec = fespace->FEColl();
      bool      edgespace, rtspace, h1space;
      bool      needint = opts.netflux;
      bool      tracespace, rt_tracespace, edge_tracespace;
      int       vdim, dim, p;
      PetscBool B_is_Trans = PETSC_FALSE;

      ParMesh *pmesh = (ParMesh *) fespace->GetMesh();
      dim = pmesh->Dimension();
      vdim = fespace->GetVDim();
      h1space = dynamic_cast<const H1_FECollection*>(fec);
      rtspace = dynamic_cast<const RT_FECollection*>(fec);
      edgespace = dynamic_cast<const ND_FECollection*>(fec);
      edge_tracespace = dynamic_cast<const ND_Trace_FECollection*>(fec);
      rt_tracespace = dynamic_cast<const RT_Trace_FECollection*>(fec);
      tracespace = edge_tracespace || rt_tracespace;

      p = 1;
      if (fespace->GetNE() > 0)
      {
         if (!tracespace)
         {
            p = fespace->GetElementOrder(0);
         }
         else
         {
            p = fespace->GetFaceOrder(0);
            if (dim == 2) { p++; }
         }
      }

      if (edgespace) // H(curl)
      {
         if (dim == 2)
         {
            needint = true;
            if (tracespace)
            {
               MFEM_WARNING("Tracespace case doesn't work for H(curl) and p=2,"
                            " not using auxiliary quadrature");
               needint = false;
            }
         }
         else
         {
            FiniteElementCollection *vfec;
            if (tracespace)
            {
               vfec = new H1_Trace_FECollection(p,dim);
            }
            else
            {
               vfec = new H1_FECollection(p,dim);
            }
            ParFiniteElementSpace *vfespace = new ParFiniteElementSpace(pmesh,vfec);
            ParDiscreteLinearOperator *grad;
            grad = new ParDiscreteLinearOperator(vfespace,fespace);
            if (tracespace)
            {
               grad->AddTraceFaceInterpolator(new GradientInterpolator);
            }
            else
            {
               grad->AddDomainInterpolator(new GradientInterpolator);
            }
            grad->Assemble();
            grad->Finalize();
            HypreParMatrix *hG = grad->ParallelAssemble();
            PetscParMatrix *G = new PetscParMatrix(hG,PETSC_MATAIJ);
            delete hG;
            delete grad;

            PetscBool conforming = PETSC_TRUE;
            if (pmesh->Nonconforming()) { conforming = PETSC_FALSE; }
            ierr = PCBDDCSetDiscreteGradient(pc,*G,p,0,PETSC_TRUE,conforming);
            PCHKERRQ(pc,ierr);
            delete vfec;
            delete vfespace;
            delete G;
         }
      }
      else if (rtspace) // H(div)
      {
         needint = true;
         if (tracespace)
         {
            MFEM_WARNING("Tracespace case doesn't work for H(div), not using"
                         " auxiliary quadrature");
            needint = false;
         }
      }
      else if (h1space) // H(grad), only for the vector case
      {
         if (vdim != dim) { needint = false; }
      }

      PetscParMatrix *B = NULL;
      if (needint)
      {
         // Generate bilinear form in unassembled format which is used to
         // compute the net-flux across subdomain boundaries for H(div) and
         // Elasticity/Stokes, and the line integral \int u x n of 2D H(curl) fields
         FiniteElementCollection *auxcoll;
         if (tracespace) { auxcoll = new RT_Trace_FECollection(p,dim); }
         else
         {
            if (h1space)
            {
               auxcoll = new H1_FECollection(std::max(p-1,1),dim);
            }
            else
            {
               auxcoll = new L2_FECollection(p,dim);
            }
         }
         ParFiniteElementSpace *pspace = new ParFiniteElementSpace(pmesh,auxcoll);
         ParMixedBilinearForm *b = new ParMixedBilinearForm(fespace,pspace);

         if (edgespace)
         {
            if (tracespace)
            {
               b->AddTraceFaceIntegrator(new VectorFECurlIntegrator);
            }
            else
            {
               b->AddDomainIntegrator(new VectorFECurlIntegrator);
            }
         }
         else if (rtspace)
         {
            if (tracespace)
            {
               b->AddTraceFaceIntegrator(new VectorFEDivergenceIntegrator);
            }
            else
            {
               b->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
            }
         }
         else
         {
            b->AddDomainIntegrator(new VectorDivergenceIntegrator);
         }
         b->Assemble();
         b->Finalize();
         OperatorHandle Bh(Operator::PETSC_MATIS);
         b->ParallelAssemble(Bh);
         Bh.Get(B);
         Bh.SetOperatorOwner(false);

         if (dir) // if essential dofs are present, we need to zero the columns
         {
            Mat pB = *B;
            ierr = MatTranspose(pB,MAT_INPLACE_MATRIX,&pB); PCHKERRQ(pA,ierr);
            if (!opts.ess_dof_local)
            {
               ierr = MatZeroRowsIS(pB,dir,0.,NULL,NULL); PCHKERRQ(pA,ierr);
            }
            else
            {
               ierr = MatZeroRowsLocalIS(pB,dir,0.,NULL,NULL); PCHKERRQ(pA,ierr);
            }
            B_is_Trans = PETSC_TRUE;
         }
         delete b;
         delete pspace;
         delete auxcoll;
      }

      if (B)
      {
         ierr = PCBDDCSetDivergenceMat(pc,*B,B_is_Trans,NULL); PCHKERRQ(pc,ierr);
      }
      delete B;
   }
   ierr = ISDestroy(&dir); PCHKERRQ(pc,ierr);
   ierr = ISDestroy(&neu); PCHKERRQ(pc,ierr);
}

PetscBDDCSolver::PetscBDDCSolver(PetscParMatrix &A,
                                 const PetscBDDCSolverParams &opts,
                                 const std::string &prefix)
   : PetscPreconditioner(A,prefix)
{
   BDDCSolverConstructor(opts);
   Customize();
}

PetscBDDCSolver::PetscBDDCSolver(MPI_Comm comm, Operator &op,
                                 const PetscBDDCSolverParams &opts,
                                 const std::string &prefix)
   : PetscPreconditioner(comm,op,prefix)
{
   BDDCSolverConstructor(opts);
   Customize();
}

PetscFieldSplitSolver::PetscFieldSplitSolver(MPI_Comm comm, Operator &op,
                                             const string &prefix)
   : PetscPreconditioner(comm,op,prefix)
{
   PC pc = (PC)obj;
   ierr = PCSetType(pc,PCFIELDSPLIT); PCHKERRQ(pc,ierr);

   Mat pA;
   ierr = PCGetOperators(pc,&pA,NULL); PCHKERRQ(pc,ierr);

   // Check if pA is of type MATNEST
   PetscBool isnest;
   ierr = PetscObjectTypeCompare((PetscObject)pA,MATNEST,&isnest);

   PetscInt nr = 0;
   IS  *isrow = NULL;
   if (isnest) // we know the fields
   {
      ierr = MatNestGetSize(pA,&nr,NULL); PCHKERRQ(pc,ierr);
      ierr = PetscCalloc1(nr,&isrow); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = MatNestGetISs(pA,isrow,NULL); PCHKERRQ(pc,ierr);
   }

   // We need to customize here, before setting the index sets.
   // This is because PCFieldSplitSetType customizes the function
   // pointers. SubSolver options will be processed during PCApply
   Customize();

   for (PetscInt i=0; i<nr; i++)
   {
      ierr = PCFieldSplitSetIS(pc,NULL,isrow[i]); PCHKERRQ(pc,ierr);
   }
   ierr = PetscFree(isrow); CCHKERRQ(PETSC_COMM_SELF,ierr);
}

PetscH2Solver::PetscH2Solver(Operator &op,
                             ParFiniteElementSpace *fes,
                             const std::string &prefix)
   : PetscPreconditioner(fes->GetParMesh()->GetComm(),prefix)
{
   PetscParMatrix A(GetComm(),&op,Operator::ANY_TYPE);
   ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE); PCHKERRQ(A,ierr);
   ierr = MatSetOption(A,MAT_SYMMETRY_ETERNAL,PETSC_TRUE); PCHKERRQ(A,ierr);
   SetOperator(A);
   H2SolverConstructor(fes);
   Customize();
}

void PetscH2Solver::H2SolverConstructor(ParFiniteElementSpace *fes)
{
#if defined(PETSC_HAVE_H2OPUS)
   int sdim = fes->GetParMesh()->SpaceDimension();
   int vdim = fes->GetVDim();
   const FiniteElementCollection *fec = fes->FEColl();
   ParFiniteElementSpace *fes_coords = NULL;

   if (vdim != sdim || fes->GetOrdering() != Ordering::byVDIM)
   {
      fes_coords = new ParFiniteElementSpace(fes->GetParMesh(),fec,sdim,
                                             Ordering::byVDIM);
      fes = fes_coords;
   }
   VectorFunctionCoefficient ccoords(sdim, func_coords);

   ParGridFunction coords(fes);
   coords.ProjectCoefficient(ccoords);
   Vector c(fes->GetTrueVSize());
   coords.ParallelProject(c);
   delete fes_coords;

   PC pc = (PC)obj;
   ierr = PCSetType(pc,PCH2OPUS); PCHKERRQ(obj, ierr);
   ierr = PCSetCoordinates(pc,sdim,c.Size()/sdim,
                           (PetscReal*)mfem::Read(c.GetMemory(),
                                                  c.Size(),false));
   ierr = PCSetFromOptions(pc); PCHKERRQ(obj, ierr);
#else
   MFEM_ABORT("Need PETSc configured with --download-h2opus");
#endif
}

// PetscNonlinearSolver methods

PetscNonlinearSolver::PetscNonlinearSolver(MPI_Comm comm,
                                           const std::string &prefix)
   : PetscSolver(), Solver()
{
   // Create the actual solver object
   SNES snes;
   ierr = SNESCreate(comm, &snes); CCHKERRQ(comm, ierr);
   obj  = (PetscObject)snes;
   ierr = PetscObjectGetClassId(obj, &cid); PCHKERRQ(obj, ierr);
   ierr = SNESSetOptionsPrefix(snes, prefix.c_str()); PCHKERRQ(snes, ierr);

   // Allocate private solver context
   CreatePrivateContext();
}

PetscNonlinearSolver::PetscNonlinearSolver(MPI_Comm comm, Operator &op,
                                           const std::string &prefix)
   : PetscSolver(), Solver()
{
   // Create the actual solver object
   SNES snes;
   ierr = SNESCreate(comm, &snes); CCHKERRQ(comm, ierr);
   obj  = (PetscObject)snes;
   ierr = PetscObjectGetClassId(obj, &cid); PCHKERRQ(obj, ierr);
   ierr = SNESSetOptionsPrefix(snes, prefix.c_str()); PCHKERRQ(snes, ierr);

   // Allocate private solver context
   CreatePrivateContext();

   SetOperator(op);
}

PetscNonlinearSolver::~PetscNonlinearSolver()
{
   MPI_Comm comm;
   SNES snes = (SNES)obj;
   ierr = PetscObjectGetComm(obj,&comm); PCHKERRQ(obj, ierr);
   ierr = SNESDestroy(&snes); CCHKERRQ(comm, ierr);
}

void PetscNonlinearSolver::SetOperator(const Operator &op)
{
   SNES snes = (SNES)obj;

   if (operatorset)
   {
      PetscBool ls,gs;
      void     *fctx,*jctx;

      ierr = SNESGetFunction(snes, NULL, NULL, &fctx);
      PCHKERRQ(snes, ierr);
      ierr = SNESGetJacobian(snes, NULL, NULL, NULL, &jctx);
      PCHKERRQ(snes, ierr);

      ls = (PetscBool)(height == op.Height() && width  == op.Width() &&
                       (void*)&op == fctx &&
                       (void*)&op == jctx);
      mpiierr = MPI_Allreduce(&ls,&gs,1,MPIU_BOOL,MPI_LAND,
                              PetscObjectComm((PetscObject)snes));
      CCHKERRQ(PetscObjectComm((PetscObject)snes),mpiierr);
      if (!gs)
      {
         ierr = SNESReset(snes); PCHKERRQ(snes,ierr);
         delete X;
         delete B;
         X = B = NULL;
      }
   }
   else
   {
      /* PETSc sets the linesearch type to basic (i.e. no linesearch) if not
         yet set. We default to backtracking */
      SNESLineSearch ls;
      ierr = SNESGetLineSearch(snes, &ls); PCHKERRQ(snes,ierr);
      ierr = SNESLineSearchSetType(ls, SNESLINESEARCHBT); PCHKERRQ(snes,ierr);
   }

   // If we do not pass matrices in, the default matrix type for DMShell is MATDENSE
   // in 3.15, which may cause issues.
   Mat dummy;
   ierr = __mfem_MatCreateDummy(PetscObjectComm((PetscObject)snes),op.Height(),
                                op.Height(),&dummy);

   __mfem_snes_ctx *snes_ctx = (__mfem_snes_ctx*)private_ctx;
   snes_ctx->op = (Operator*)&op;
   ierr = SNESSetFunction(snes, NULL, __mfem_snes_function, (void *)snes_ctx);
   PCHKERRQ(snes, ierr);
   ierr = SNESSetJacobian(snes, dummy, dummy, __mfem_snes_jacobian,
                          (void *)snes_ctx);
   PCHKERRQ(snes, ierr);

   ierr = MatDestroy(&dummy);
   PCHKERRQ(snes, ierr);

   // Update PetscSolver
   operatorset = true;

   // Update the Operator fields.
   height = op.Height();
   width  = op.Width();
}

void PetscNonlinearSolver::SetJacobianType(Operator::Type jacType)
{
   __mfem_snes_ctx *snes_ctx = (__mfem_snes_ctx*)private_ctx;
   snes_ctx->jacType = jacType;
}

void PetscNonlinearSolver::SetObjective(void (*objfn)(Operator *,const Vector&,
                                                      mfem::real_t*))
{
   __mfem_snes_ctx *snes_ctx = (__mfem_snes_ctx*)private_ctx;
   snes_ctx->objective = objfn;

   SNES snes = (SNES)obj;
   ierr = SNESSetObjective(snes, __mfem_snes_objective, (void *)snes_ctx);
   PCHKERRQ(snes, ierr);
}

void PetscNonlinearSolver::SetPostCheck(void (*post)(Operator *,const Vector&,
                                                     Vector&, Vector&,
                                                     bool&, bool&))
{
   __mfem_snes_ctx *snes_ctx = (__mfem_snes_ctx*)private_ctx;
   snes_ctx->postcheck = post;

   SNES snes = (SNES)obj;
   SNESLineSearch ls;
   ierr = SNESGetLineSearch(snes, &ls); PCHKERRQ(snes,ierr);
   ierr = SNESLineSearchSetPostCheck(ls, __mfem_snes_postcheck, (void *)snes_ctx);
   PCHKERRQ(ls, ierr);
}

void PetscNonlinearSolver::SetUpdate(void (*update)(Operator *,int,
                                                    const Vector&,
                                                    const Vector&,
                                                    const Vector&,
                                                    const Vector&))
{
   __mfem_snes_ctx *snes_ctx = (__mfem_snes_ctx*)private_ctx;
   snes_ctx->update = update;

   SNES snes = (SNES)obj;
   ierr = SNESSetUpdate(snes, __mfem_snes_update); PCHKERRQ(snes, ierr);
}

void PetscNonlinearSolver::Mult(const Vector &b, Vector &x) const
{
   SNES snes = (SNES)obj;

   bool b_nonempty = b.Size();
   if (!B) { B = new PetscParVector(PetscObjectComm(obj), *this, true); }
   if (!X) { X = new PetscParVector(PetscObjectComm(obj), *this, false, false); }
   X->PlaceMemory(x.GetMemory(),iterative_mode);
   if (b_nonempty) { B->PlaceMemory(b.GetMemory()); }
   else { *B = 0.0; }

   Customize();

   if (!iterative_mode) { *X = 0.; }

   // Solve the system.
   ierr = SNESSolve(snes, B->x, X->x); PCHKERRQ(snes, ierr);
   X->ResetMemory();
   if (b_nonempty) { B->ResetMemory(); }
}

// PetscODESolver methods

PetscODESolver::PetscODESolver(MPI_Comm comm, const string &prefix)
   : PetscSolver(), ODESolver()
{
   // Create the actual solver object
   TS ts;
   ierr = TSCreate(comm,&ts); CCHKERRQ(comm,ierr);
   obj  = (PetscObject)ts;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   ierr = TSSetOptionsPrefix(ts, prefix.c_str()); PCHKERRQ(ts, ierr);

   // Allocate private solver context
   CreatePrivateContext();

   // Default options, to comply with the current interface to ODESolver.
   ierr = TSSetMaxSteps(ts,PETSC_MAX_INT-1);
   PCHKERRQ(ts,ierr);
   ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);
   PCHKERRQ(ts,ierr);
   TSAdapt tsad;
   ierr = TSGetAdapt(ts,&tsad);
   PCHKERRQ(ts,ierr);
   ierr = TSAdaptSetType(tsad,TSADAPTNONE);
   PCHKERRQ(ts,ierr);
}

PetscODESolver::~PetscODESolver()
{
   MPI_Comm comm;
   TS ts = (TS)obj;
   ierr = PetscObjectGetComm(obj,&comm); PCHKERRQ(obj,ierr);
   ierr = TSDestroy(&ts); CCHKERRQ(comm,ierr);
}

void PetscODESolver::Init(TimeDependentOperator &f_,
                          enum PetscODESolver::Type type)
{
   TS ts = (TS)obj;

   __mfem_ts_ctx *ts_ctx = (__mfem_ts_ctx*)private_ctx;
   if (operatorset)
   {
      ierr = TSReset(ts); PCHKERRQ(ts,ierr);
      delete X;
      X = NULL;
      ts_ctx->cached_shift = std::numeric_limits<PetscReal>::min();
      ts_ctx->cached_ijacstate = -1;
      ts_ctx->cached_rhsjacstate = -1;
      ts_ctx->cached_splits_xstate = -1;
      ts_ctx->cached_splits_xdotstate = -1;
   }
   f = &f_;

   // Set functions in TS
   ts_ctx->op = &f_;
   if (f_.isImplicit())
   {
      Mat dummy;
      ierr = __mfem_MatCreateDummy(PetscObjectComm((PetscObject)ts),f_.Height(),
                                   f_.Height(),&dummy);
      PCHKERRQ(ts, ierr);
      ierr = TSSetIFunction(ts, NULL, __mfem_ts_ifunction, (void *)ts_ctx);
      PCHKERRQ(ts, ierr);
      ierr = TSSetIJacobian(ts, dummy, dummy, __mfem_ts_ijacobian, (void *)ts_ctx);
      PCHKERRQ(ts, ierr);
      ierr = TSSetEquationType(ts, TS_EQ_IMPLICIT);
      PCHKERRQ(ts, ierr);
      ierr = MatDestroy(&dummy);
      PCHKERRQ(ts, ierr);
   }
   if (!f_.isHomogeneous())
   {
      Mat dummy = NULL;
      if (!f_.isImplicit())
      {
         ierr = TSSetEquationType(ts, TS_EQ_EXPLICIT);
         PCHKERRQ(ts, ierr);
      }
      else
      {
         ierr = __mfem_MatCreateDummy(PetscObjectComm((PetscObject)ts),f_.Height(),
                                      f_.Height(),&dummy);
         PCHKERRQ(ts, ierr);
      }
      ierr = TSSetRHSFunction(ts, NULL, __mfem_ts_rhsfunction, (void *)ts_ctx);
      PCHKERRQ(ts, ierr);
      ierr = TSSetRHSJacobian(ts, dummy, dummy, __mfem_ts_rhsjacobian,
                              (void *)ts_ctx);
      PCHKERRQ(ts, ierr);
      ierr = MatDestroy(&dummy);
      PCHKERRQ(ts, ierr);
   }
   operatorset = true;

   SetType(type);

   // Set solution vector
   PetscParVector X(PetscObjectComm(obj),*f,false,true);
   ierr = TSSetSolution(ts,X); PCHKERRQ(ts,ierr);

   // Compose special purpose function for PDE-constrained optimization
   PetscBool use = PETSC_TRUE;
   ierr = PetscOptionsGetBool(NULL,NULL,"-mfem_use_splitjac",&use,NULL);
   if (use && f_.isImplicit())
   {
      ierr = PetscObjectComposeFunction((PetscObject)ts,"TSComputeSplitJacobians_C",
                                        __mfem_ts_computesplits);
      PCHKERRQ(ts,ierr);
   }
   else
   {
      ierr = PetscObjectComposeFunction((PetscObject)ts,"TSComputeSplitJacobians_C",
                                        NULL);
      PCHKERRQ(ts,ierr);
   }
}

void PetscODESolver::SetJacobianType(Operator::Type jacType)
{
   __mfem_ts_ctx *ts_ctx = (__mfem_ts_ctx*)private_ctx;
   ts_ctx->jacType = jacType;
}

PetscODESolver::Type PetscODESolver::GetType() const
{
   __mfem_ts_ctx *ts_ctx = (__mfem_ts_ctx*)private_ctx;
   return ts_ctx->type;
}

void PetscODESolver::SetType(PetscODESolver::Type type)
{
   __mfem_ts_ctx *ts_ctx = (__mfem_ts_ctx*)private_ctx;

   TS ts = (TS)obj;
   ts_ctx->type = type;
   if (type == ODE_SOLVER_LINEAR)
   {
      ierr = TSSetProblemType(ts, TS_LINEAR);
      PCHKERRQ(ts, ierr);
   }
   else
   {
      ierr = TSSetProblemType(ts, TS_NONLINEAR);
      PCHKERRQ(ts, ierr);
   }
}

void PetscODESolver::Step(Vector &x, mfem::real_t &t, mfem::real_t &dt)
{
   // Pass the parameters to PETSc.
   TS ts = (TS)obj;
   ierr = TSSetTime(ts, t); PCHKERRQ(ts, ierr);
   ierr = TSSetTimeStep(ts, dt); PCHKERRQ(ts, ierr);

   PetscInt i;
   ierr = TSGetStepNumber(ts, &i); PCHKERRQ(ts,ierr);

   if (!X) { X = new PetscParVector(PetscObjectComm(obj), *f, false, false); }
   X->PlaceMemory(x.GetMemory(),true);

   Customize();

   // Monitor initial step
   if (!i)
   {
      ierr = TSMonitor(ts, i, t, *X); PCHKERRQ(ts,ierr);
   }

   // Take the step.
   ierr = TSSetSolution(ts, *X); PCHKERRQ(ts, ierr);
   ierr = TSStep(ts); PCHKERRQ(ts, ierr);

   // Get back current time and the time step used to caller.
   // We cannot use TSGetTimeStep() as it returns the next candidate step
   PetscReal pt;
   ierr = TSGetTime(ts, &pt); PCHKERRQ(ts,ierr);
   dt = pt - (PetscReal)t;
   t = pt;

   // Monitor current step
   ierr = TSMonitor(ts, i+1, pt, *X); PCHKERRQ(ts,ierr);

   X->ResetMemory();
}

void PetscODESolver::Run(Vector &x, mfem::real_t &t, mfem::real_t &dt,
                         mfem::real_t t_final)
{
   // Give the parameters to PETSc.
   TS ts = (TS)obj;
   ierr = TSSetTime(ts, t); PCHKERRQ(ts, ierr);
   ierr = TSSetTimeStep(ts, dt); PCHKERRQ(ts, ierr);
   ierr = TSSetMaxTime(ts, t_final); PCHKERRQ(ts, ierr);
   ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);
   PCHKERRQ(ts, ierr);

   if (!X) { X = new PetscParVector(PetscObjectComm(obj), *f, false, false); }
   X->PlaceMemory(x.GetMemory(),true);

   Customize();

   // Reset Jacobian caching since the user may have changed
   // the parameters of the solver
   // We don't do this in the Step method because two consecutive
   // Step() calls are done with the same operator
   __mfem_ts_ctx *ts_ctx = (__mfem_ts_ctx*)private_ctx;
   ts_ctx->cached_shift = std::numeric_limits<PetscReal>::min();
   ts_ctx->cached_ijacstate = -1;
   ts_ctx->cached_rhsjacstate = -1;
   ts_ctx->cached_splits_xstate = -1;
   ts_ctx->cached_splits_xdotstate = -1;

   // Take the steps.
   ierr = TSSolve(ts, X->x); PCHKERRQ(ts, ierr);
   X->ResetMemory();

   // Get back final time and time step to caller.
   PetscReal pt;
   ierr = TSGetTime(ts, &pt); PCHKERRQ(ts,ierr);
   t = pt;
   ierr = TSGetTimeStep(ts,&pt); PCHKERRQ(ts,ierr);
   dt = pt;
}

}  // namespace mfem

#include "petsc/private/petscimpl.h"
#include "petsc/private/matimpl.h"

// auxiliary functions
static PetscErrorCode __mfem_ts_monitor(TS ts, PetscInt it, PetscReal t, Vec x,
                                        void* ctx)
{
   __mfem_monitor_ctx *monctx = (__mfem_monitor_ctx*)ctx;

   PetscFunctionBeginUser;
   if (!monctx)
   {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Missing monitor context");
   }
   mfem::PetscSolver *solver = (mfem::PetscSolver*)(monctx->solver);
   mfem::PetscSolverMonitor *user_monitor = (mfem::PetscSolverMonitor *)(
                                               monctx->monitor);

   if (user_monitor->mon_sol)
   {
      mfem::PetscParVector V(x,true);
      user_monitor->MonitorSolution(it,t,V);
   }
   user_monitor->MonitorSolver(solver);
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_ts_ifunction(TS ts, PetscReal t, Vec x, Vec xp,
                                          Vec f,void *ctx)
{
   __mfem_ts_ctx* ts_ctx = (__mfem_ts_ctx*)ctx;

   PetscFunctionBeginUser;
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(xp,true);
   mfem::PetscParVector ff(f,true);

   mfem::TimeDependentOperator *op = ts_ctx->op;
   op->SetTime(t);

   if (ts_ctx->bchandler)
   {
      // we evaluate the ImplicitMult method with the correct bc
      // this means the correct time derivative for essential boundary
      // dofs is zero
      if (!ts_ctx->work) { ts_ctx->work = new mfem::Vector(xx.Size()); }
      if (!ts_ctx->work2) { ts_ctx->work2 = new mfem::Vector(xx.Size()); }
      mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
      mfem::Vector* txx = ts_ctx->work;
      mfem::Vector* txp = ts_ctx->work2;
      bchandler->SetTime(t);
      bchandler->ApplyBC(xx,*txx);
      bchandler->ZeroBC(yy,*txp);
      op->ImplicitMult(*txx,*txp,ff);
      // and fix the residual (i.e. f_\partial\Omega = u - g(t))
      bchandler->FixResidualBC(xx,ff);
   }
   else
   {
      // use the ImplicitMult method of the class
      op->ImplicitMult(xx,yy,ff);
   }
   ff.UpdateVecFromFlags();
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_ts_rhsfunction(TS ts, PetscReal t, Vec x, Vec f,
                                            void *ctx)
{
   __mfem_ts_ctx* ts_ctx = (__mfem_ts_ctx*)ctx;

   PetscFunctionBeginUser;
   if (ts_ctx->bchandler) { MFEM_ABORT("RHS evaluation with bc not implemented"); } // TODO
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector ff(f,true);
   mfem::TimeDependentOperator *top = ts_ctx->op;
   top->SetTime(t);

   // use the ExplicitMult method - compute the RHS function
   top->ExplicitMult(xx,ff);

   ff.UpdateVecFromFlags();
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_ts_ijacobian(TS ts, PetscReal t, Vec x,
                                          Vec xp, PetscReal shift, Mat A, Mat P,
                                          void *ctx)
{
   __mfem_ts_ctx*   ts_ctx = (__mfem_ts_ctx*)ctx;
   mfem::Vector     *xx;
   PetscScalar      *array;
   PetscReal        eps = 0.001; /* 0.1% difference */
   PetscInt         n;
   PetscObjectState state;
   PetscErrorCode   ierr;

   PetscFunctionBeginUser;
   // Matrix-free case
   if (A && A != P)
   {
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   }

   // prevent to recompute a Jacobian if we already did so
   // the relative tolerance comparison should be fine given the fact
   // that two consecutive shifts should have similar magnitude
   ierr = PetscObjectStateGet((PetscObject)P,&state); CHKERRQ(ierr);
   if (ts_ctx->type == mfem::PetscODESolver::ODE_SOLVER_LINEAR &&
       std::abs(ts_ctx->cached_shift/shift - 1.0) < eps &&
       state == ts_ctx->cached_ijacstate) { PetscFunctionReturn(PETSC_SUCCESS); }

   // update time
   mfem::TimeDependentOperator *op = ts_ctx->op;
   op->SetTime(t);

   // wrap Vecs with Vectors
   ierr = VecGetLocalSize(x,&n); CHKERRQ(ierr);
   ierr = VecGetArrayRead(xp,(const PetscScalar**)&array); CHKERRQ(ierr);
   mfem::Vector yy(array,n);
   ierr = VecRestoreArrayRead(xp,(const PetscScalar**)&array); CHKERRQ(ierr);
   ierr = VecGetArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);
   if (!ts_ctx->bchandler)
   {
      xx = new mfem::Vector(array,n);
   }
   else
   {
      // make sure we compute a Jacobian with the correct boundary values
      if (!ts_ctx->work) { ts_ctx->work = new mfem::Vector(n); }
      mfem::Vector txx(array,n);
      mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
      xx = ts_ctx->work;
      bchandler->SetTime(t);
      bchandler->ApplyBC(txx,*xx);
   }
   ierr = VecRestoreArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);

   // Use TimeDependentOperator::GetImplicitGradient(x,y,s)
   mfem::Operator& J = op->GetImplicitGradient(*xx,yy,shift);
   if (!ts_ctx->bchandler) { delete xx; }
   ts_ctx->cached_shift = shift;

   // Convert to the operator type requested if needed
   bool delete_pA = false;
   mfem::PetscParMatrix *pA = const_cast<mfem::PetscParMatrix *>
                              (dynamic_cast<const mfem::PetscParMatrix *>(&J));
   if (!pA || (ts_ctx->jacType != mfem::Operator::ANY_TYPE &&
               pA->GetType() != ts_ctx->jacType))
   {
      pA = new mfem::PetscParMatrix(PetscObjectComm((PetscObject)ts),&J,
                                    ts_ctx->jacType);
      delete_pA = true;
   }

   // Eliminate essential dofs
   if (ts_ctx->bchandler)
   {
      mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
      mfem::PetscParVector dummy(PetscObjectComm((PetscObject)ts),0);
      pA->EliminateRowsCols(bchandler->GetTDofs(),dummy,dummy);
   }

   // Get nonzerostate
   PetscObjectState nonzerostate;
   ierr = MatGetNonzeroState(P,&nonzerostate); CHKERRQ(ierr);

   // Avoid unneeded copy of the matrix by hacking
   Mat B;
   B = pA->ReleaseMat(false);
   ierr = MatHeaderReplace(P,&B); CHKERRQ(ierr);
   if (delete_pA) { delete pA; }

   // When using MATNEST and PCFIELDSPLIT, the second setup of the
   // preconditioner fails because MatCreateSubMatrix_Nest does not
   // actually return a matrix. Instead, for efficiency reasons,
   // it returns a reference to the submatrix. The second time it
   // is called, MAT_REUSE_MATRIX is used and MatCreateSubMatrix_Nest
   // aborts since the two submatrices are actually different.
   // We circumvent this issue by incrementing the nonzero state
   // (i.e. PETSc thinks the operator sparsity pattern has changed)
   // This does not impact performances in the case of MATNEST
   PetscBool isnest;
   ierr = PetscObjectTypeCompare((PetscObject)P,MATNEST,&isnest);
   CHKERRQ(ierr);
   if (isnest) { P->nonzerostate = nonzerostate + 1; }

   // Jacobian reusage
   ierr = PetscObjectStateGet((PetscObject)P,&ts_ctx->cached_ijacstate);
   CHKERRQ(ierr);

   // Fool DM
   DM dm;
   MatType mtype;
   ierr = MatGetType(P,&mtype); CHKERRQ(ierr);
   ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);
   ierr = DMSetMatType(dm,mtype); CHKERRQ(ierr);
   ierr = DMShellSetMatrix(dm,P); CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_ts_computesplits(TS ts,PetscReal t,Vec x,Vec xp,
                                              Mat Ax,Mat Jx,
                                              Mat Axp,Mat Jxp)
{
   __mfem_ts_ctx*   ts_ctx;
   mfem::Vector     *xx;
   PetscScalar      *array;
   PetscInt         n;
   PetscObjectState state;
   PetscBool        rx = PETSC_TRUE, rxp = PETSC_TRUE;
   PetscBool        assembled;
   PetscErrorCode   ierr;

   PetscFunctionBeginUser;
   // Matrix-free cases
   if (Ax && Ax != Jx)
   {
      ierr = MatAssemblyBegin(Ax,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Ax,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   }
   if (Axp && Axp != Jxp)
   {
      ierr = MatAssemblyBegin(Axp,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Axp,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   }

   ierr = TSGetIJacobian(ts,NULL,NULL,NULL,(void**)&ts_ctx); CHKERRQ(ierr);

   // prevent to recompute the Jacobians if we already did so
   ierr = PetscObjectStateGet((PetscObject)Jx,&state); CHKERRQ(ierr);
   if (ts_ctx->type == mfem::PetscODESolver::ODE_SOLVER_LINEAR &&
       state == ts_ctx->cached_splits_xstate) { rx = PETSC_FALSE; }
   ierr = PetscObjectStateGet((PetscObject)Jxp,&state); CHKERRQ(ierr);
   if (ts_ctx->type == mfem::PetscODESolver::ODE_SOLVER_LINEAR &&
       state == ts_ctx->cached_splits_xdotstate) { rxp = PETSC_FALSE; }
   if (!rx && !rxp) { PetscFunctionReturn(PETSC_SUCCESS); }

   // update time
   mfem::TimeDependentOperator *op = ts_ctx->op;
   op->SetTime(t);

   // wrap Vecs with Vectors
   ierr = VecGetLocalSize(x,&n); CHKERRQ(ierr);
   ierr = VecGetArrayRead(xp,(const PetscScalar**)&array); CHKERRQ(ierr);
   mfem::Vector yy(array,n);
   ierr = VecRestoreArrayRead(xp,(const PetscScalar**)&array); CHKERRQ(ierr);
   ierr = VecGetArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);
   if (!ts_ctx->bchandler)
   {
      xx = new mfem::Vector(array,n);
   }
   else
   {
      // make sure we compute a Jacobian with the correct boundary values
      if (!ts_ctx->work) { ts_ctx->work = new mfem::Vector(n); }
      mfem::Vector txx(array,n);
      mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
      xx = ts_ctx->work;
      bchandler->SetTime(t);
      bchandler->ApplyBC(txx,*xx);
   }
   ierr = VecRestoreArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);

   // We don't have a specialized interface, so we just compute the split jacobians
   // evaluating twice the implicit gradient method with the correct shifts

   // first we do the state jacobian
   mfem::Operator& oJx = op->GetImplicitGradient(*xx,yy,0.0);

   // Convert to the operator type requested if needed
   bool delete_mat = false;
   mfem::PetscParMatrix *pJx = const_cast<mfem::PetscParMatrix *>
                               (dynamic_cast<const mfem::PetscParMatrix *>(&oJx));
   if (!pJx || (ts_ctx->jacType != mfem::Operator::ANY_TYPE &&
                pJx->GetType() != ts_ctx->jacType))
   {
      if (pJx)
      {
         Mat B = *pJx;
         ierr = PetscObjectReference((PetscObject)B); CHKERRQ(ierr);
      }
      pJx = new mfem::PetscParMatrix(PetscObjectComm((PetscObject)ts),&oJx,
                                     ts_ctx->jacType);
      delete_mat = true;
   }
   if (rx)
   {
      ierr = MatAssembled(Jx,&assembled); CHKERRQ(ierr);
      if (assembled)
      {
         ierr = MatCopy(*pJx,Jx,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
      }
      else
      {
         Mat B;
         ierr = MatDuplicate(*pJx,MAT_COPY_VALUES,&B); CHKERRQ(ierr);
         ierr = MatHeaderReplace(Jx,&B); CHKERRQ(ierr);
      }
   }
   if (delete_mat) { delete pJx; }
   pJx = new mfem::PetscParMatrix(Jx,true);

   // Eliminate essential dofs
   if (ts_ctx->bchandler)
   {
      mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
      mfem::PetscParVector dummy(PetscObjectComm((PetscObject)ts),0);
      pJx->EliminateRowsCols(bchandler->GetTDofs(),dummy,dummy);
   }

   // Then we do the jacobian wrt the time derivative of the state
   // Note that this is usually the mass matrix
   mfem::PetscParMatrix *pJxp = NULL;
   if (rxp)
   {
      delete_mat = false;
      mfem::Operator& oJxp = op->GetImplicitGradient(*xx,yy,1.0);
      pJxp = const_cast<mfem::PetscParMatrix *>
             (dynamic_cast<const mfem::PetscParMatrix *>(&oJxp));
      if (!pJxp || (ts_ctx->jacType != mfem::Operator::ANY_TYPE &&
                    pJxp->GetType() != ts_ctx->jacType))
      {
         if (pJxp)
         {
            Mat B = *pJxp;
            ierr = PetscObjectReference((PetscObject)B); CHKERRQ(ierr);
         }
         pJxp = new mfem::PetscParMatrix(PetscObjectComm((PetscObject)ts),
                                         &oJxp,ts_ctx->jacType);
         delete_mat = true;
      }

      ierr = MatAssembled(Jxp,&assembled); CHKERRQ(ierr);
      if (assembled)
      {
         ierr = MatCopy(*pJxp,Jxp,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
      }
      else
      {
         Mat B;
         ierr = MatDuplicate(*pJxp,MAT_COPY_VALUES,&B); CHKERRQ(ierr);
         ierr = MatHeaderReplace(Jxp,&B); CHKERRQ(ierr);
      }
      if (delete_mat) { delete pJxp; }
      pJxp = new mfem::PetscParMatrix(Jxp,true);

      // Eliminate essential dofs
      if (ts_ctx->bchandler)
      {
         mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
         mfem::PetscParVector dummy(PetscObjectComm((PetscObject)ts),0);
         pJxp->EliminateRowsCols(bchandler->GetTDofs(),dummy,dummy,2.0);
      }

      // Obtain the time dependent part of the  jacobian by subtracting
      // the state jacobian
      // We don't do it with the class operator "-=" since we know that
      // the sparsity pattern of the two matrices is the same
      ierr = MatAXPY(*pJxp,-1.0,*pJx,SAME_NONZERO_PATTERN); PCHKERRQ(ts,ierr);
   }

   // Jacobian reusage
   ierr = PetscObjectStateGet((PetscObject)Jx,&ts_ctx->cached_splits_xstate);
   CHKERRQ(ierr);
   ierr = PetscObjectStateGet((PetscObject)Jxp,&ts_ctx->cached_splits_xdotstate);
   CHKERRQ(ierr);

   delete pJx;
   delete pJxp;
   if (!ts_ctx->bchandler) { delete xx; }
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_ts_rhsjacobian(TS ts, PetscReal t, Vec x,
                                            Mat A, Mat P, void *ctx)
{
   __mfem_ts_ctx*   ts_ctx = (__mfem_ts_ctx*)ctx;
   mfem::Vector     *xx;
   PetscScalar      *array;
   PetscInt         n;
   PetscObjectState state;
   PetscErrorCode   ierr;

   PetscFunctionBeginUser;
   // Matrix-free case
   if (A && A != P)
   {
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   }

   // prevent to recompute a Jacobian if we already did so
   ierr = PetscObjectStateGet((PetscObject)P,&state); CHKERRQ(ierr);
   if (ts_ctx->type == mfem::PetscODESolver::ODE_SOLVER_LINEAR &&
       state == ts_ctx->cached_rhsjacstate) { PetscFunctionReturn(PETSC_SUCCESS); }

   // update time
   mfem::TimeDependentOperator *op = ts_ctx->op;
   op->SetTime(t);

   // wrap Vec with Vector
   ierr = VecGetLocalSize(x,&n); CHKERRQ(ierr);
   ierr = VecGetArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);
   if (!ts_ctx->bchandler)
   {
      xx = new mfem::Vector(array,n);
   }
   else
   {
      // make sure we compute a Jacobian with the correct boundary values
      if (!ts_ctx->work) { ts_ctx->work = new mfem::Vector(n); }
      mfem::Vector txx(array,n);
      mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
      xx = ts_ctx->work;
      bchandler->SetTime(t);
      bchandler->ApplyBC(txx,*xx);
   }
   ierr = VecRestoreArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);

   // Use TimeDependentOperator::GetExplicitGradient(x)
   mfem::Operator& J = op->GetExplicitGradient(*xx);
   if (!ts_ctx->bchandler) { delete xx; }

   // Convert to the operator type requested if needed
   bool delete_pA = false;
   mfem::PetscParMatrix *pA = const_cast<mfem::PetscParMatrix *>
                              (dynamic_cast<const mfem::PetscParMatrix *>(&J));
   if (!pA || (ts_ctx->jacType != mfem::Operator::ANY_TYPE &&
               pA->GetType() != ts_ctx->jacType))
   {
      pA = new mfem::PetscParMatrix(PetscObjectComm((PetscObject)ts),&J,
                                    ts_ctx->jacType);
      delete_pA = true;
   }

   // Eliminate essential dofs
   if (ts_ctx->bchandler)
   {
      mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
      mfem::PetscParVector dummy(PetscObjectComm((PetscObject)ts),0);
      pA->EliminateRowsCols(bchandler->GetTDofs(),dummy,dummy);
   }

   // Get nonzerostate
   PetscObjectState nonzerostate;
   ierr = MatGetNonzeroState(P,&nonzerostate); CHKERRQ(ierr);

   // Avoid unneeded copy of the matrix by hacking
   Mat B;
   B = pA->ReleaseMat(false);
   ierr = MatHeaderReplace(P,&B); CHKERRQ(ierr);
   if (delete_pA) { delete pA; }

   // When using MATNEST and PCFIELDSPLIT, the second setup of the
   // preconditioner fails because MatCreateSubMatrix_Nest does not
   // actually return a matrix. Instead, for efficiency reasons,
   // it returns a reference to the submatrix. The second time it
   // is called, MAT_REUSE_MATRIX is used and MatCreateSubMatrix_Nest
   // aborts since the two submatrices are actually different.
   // We circumvent this issue by incrementing the nonzero state
   // (i.e. PETSc thinks the operator sparsity pattern has changed)
   // This does not impact performances in the case of MATNEST
   PetscBool isnest;
   ierr = PetscObjectTypeCompare((PetscObject)P,MATNEST,&isnest);
   CHKERRQ(ierr);
   if (isnest) { P->nonzerostate = nonzerostate + 1; }

   // Jacobian reusage
   if (ts_ctx->type == mfem::PetscODESolver::ODE_SOLVER_LINEAR)
   {
      ierr = TSRHSJacobianSetReuse(ts,PETSC_TRUE); PCHKERRQ(ts,ierr);
   }
   ierr = PetscObjectStateGet((PetscObject)P,&ts_ctx->cached_rhsjacstate);
   CHKERRQ(ierr);

   // Fool DM
   DM dm;
   MatType mtype;
   ierr = MatGetType(P,&mtype); CHKERRQ(ierr);
   ierr = TSGetDM(ts,&dm); CHKERRQ(ierr);
   ierr = DMSetMatType(dm,mtype); CHKERRQ(ierr);
   ierr = DMShellSetMatrix(dm,P); CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_snes_monitor(SNES snes, PetscInt it, PetscReal res,
                                          void* ctx)
{
   __mfem_monitor_ctx *monctx = (__mfem_monitor_ctx*)ctx;

   PetscFunctionBeginUser;
   if (!monctx)
   {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Missing monitor context");
   }

   mfem::PetscSolver *solver = (mfem::PetscSolver*)(monctx->solver);
   mfem::PetscSolverMonitor *user_monitor = (mfem::PetscSolverMonitor *)(
                                               monctx->monitor);
   if (user_monitor->mon_sol)
   {
      Vec x;
      PetscErrorCode ierr;

      ierr = SNESGetSolution(snes,&x); CHKERRQ(ierr);
      mfem::PetscParVector V(x,true);
      user_monitor->MonitorSolution(it,res,V);
   }
   if (user_monitor->mon_res)
   {
      Vec x;
      PetscErrorCode ierr;

      ierr = SNESGetFunction(snes,&x,NULL,NULL); CHKERRQ(ierr);
      mfem::PetscParVector V(x,true);
      user_monitor->MonitorResidual(it,res,V);
   }
   user_monitor->MonitorSolver(solver);
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_snes_jacobian(SNES snes, Vec x, Mat A, Mat P,
                                           void *ctx)
{
   PetscScalar     *array;
   PetscInt         n;
   PetscErrorCode   ierr;
   mfem::Vector    *xx;
   __mfem_snes_ctx *snes_ctx = (__mfem_snes_ctx*)ctx;

   PetscFunctionBeginUser;
   ierr = VecGetArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);
   ierr = VecGetLocalSize(x,&n); CHKERRQ(ierr);
   if (!snes_ctx->bchandler)
   {
      xx = new mfem::Vector(array,n);
   }
   else
   {
      // make sure we compute a Jacobian with the correct boundary values
      if (!snes_ctx->work) { snes_ctx->work = new mfem::Vector(n); }
      mfem::Vector txx(array,n);
      mfem::PetscBCHandler *bchandler = snes_ctx->bchandler;
      xx = snes_ctx->work;
      bchandler->ApplyBC(txx,*xx);
   }

   // Use Operator::GetGradient(x)
   mfem::Operator& J = snes_ctx->op->GetGradient(*xx);
   ierr = VecRestoreArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);
   if (!snes_ctx->bchandler) { delete xx; }

   // Convert to the operator type requested if needed
   bool delete_pA = false;
   mfem::PetscParMatrix *pA = const_cast<mfem::PetscParMatrix *>
                              (dynamic_cast<const mfem::PetscParMatrix *>(&J));
   if (!pA || (snes_ctx->jacType != mfem::Operator::ANY_TYPE &&
               pA->GetType() != snes_ctx->jacType))
   {
      pA = new mfem::PetscParMatrix(PetscObjectComm((PetscObject)snes),&J,
                                    snes_ctx->jacType);
      delete_pA = true;
   }

   // Eliminate essential dofs
   if (snes_ctx->bchandler)
   {
      mfem::PetscBCHandler *bchandler = snes_ctx->bchandler;
      mfem::PetscParVector dummy(PetscObjectComm((PetscObject)snes),0);
      pA->EliminateRowsCols(bchandler->GetTDofs(),dummy,dummy);
   }

   // Get nonzerostate
   PetscObjectState nonzerostate;
   ierr = MatGetNonzeroState(P,&nonzerostate); CHKERRQ(ierr);

   // Avoid unneeded copy of the matrix by hacking
   Mat B = pA->ReleaseMat(false);
   ierr = MatHeaderReplace(P,&B); CHKERRQ(ierr);
   if (delete_pA) { delete pA; }

   // When using MATNEST and PCFIELDSPLIT, the second setup of the
   // preconditioner fails because MatCreateSubMatrix_Nest does not
   // actually return a matrix. Instead, for efficiency reasons,
   // it returns a reference to the submatrix. The second time it
   // is called, MAT_REUSE_MATRIX is used and MatCreateSubMatrix_Nest
   // aborts since the two submatrices are actually different.
   // We circumvent this issue by incrementing the nonzero state
   // (i.e. PETSc thinks the operator sparsity pattern has changed)
   // This does not impact performances in the case of MATNEST
   PetscBool isnest;
   ierr = PetscObjectTypeCompare((PetscObject)P,MATNEST,&isnest);
   CHKERRQ(ierr);
   if (isnest) { P->nonzerostate = nonzerostate + 1; }

   // Matrix-free case
   if (A && A != P)
   {
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   }

   // Fool DM
   DM dm;
   MatType mtype;
   ierr = MatGetType(P,&mtype); CHKERRQ(ierr);
   ierr = SNESGetDM(snes,&dm); CHKERRQ(ierr);
   ierr = DMSetMatType(dm,mtype); CHKERRQ(ierr);
   ierr = DMShellSetMatrix(dm,P); CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_snes_function(SNES snes, Vec x, Vec f, void *ctx)
{
   __mfem_snes_ctx* snes_ctx = (__mfem_snes_ctx*)ctx;

   PetscFunctionBeginUser;
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector ff(f,true);
   if (snes_ctx->bchandler)
   {
      // we evaluate the Mult method with the correct bc
      if (!snes_ctx->work) { snes_ctx->work = new mfem::Vector(xx.Size()); }
      mfem::PetscBCHandler *bchandler = snes_ctx->bchandler;
      mfem::Vector* txx = snes_ctx->work;
      bchandler->ApplyBC(xx,*txx);
      snes_ctx->op->Mult(*txx,ff);
      // and fix the residual (i.e. f_\partial\Omega = u - g)
      bchandler->FixResidualBC(xx,ff);
   }
   else
   {
      // use the Mult method of the class
      snes_ctx->op->Mult(xx,ff);
   }
   ff.UpdateVecFromFlags();
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_snes_objective(SNES snes, Vec x, PetscReal *f,
                                            void *ctx)
{
   __mfem_snes_ctx* snes_ctx = (__mfem_snes_ctx*)ctx;

   PetscFunctionBeginUser;
   if (!snes_ctx->objective)
   {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Missing objective function");
   }
   mfem::PetscParVector xx(x,true);
   mfem::real_t lf;
   (*snes_ctx->objective)(snes_ctx->op,xx,&lf);
   *f = (PetscReal)lf;
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_snes_postcheck(SNESLineSearch ls,Vec X,Vec Y,Vec W,
                                            PetscBool *cy,PetscBool *cw, void* ctx)
{
   __mfem_snes_ctx* snes_ctx = (__mfem_snes_ctx*)ctx;
   bool lcy = false,lcw = false;

   PetscFunctionBeginUser;
   mfem::PetscParVector x(X,true);
   mfem::PetscParVector y(Y,true);
   mfem::PetscParVector w(W,true);
   (*snes_ctx->postcheck)(snes_ctx->op,x,y,w,lcy,lcw);
   if (lcy) { y.UpdateVecFromFlags(); *cy = PETSC_TRUE; }
   if (lcw) { w.UpdateVecFromFlags(); *cw = PETSC_TRUE; }
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_snes_update(SNES snes, PetscInt it)
{
   Vec F,X,dX,pX;
   __mfem_snes_ctx* snes_ctx;

   PetscFunctionBeginUser;
   /* Update callback does not use the context */
   ierr = SNESGetFunction(snes,&F,NULL,(void **)&snes_ctx); CHKERRQ(ierr);
   ierr = SNESGetSolution(snes,&X); CHKERRQ(ierr);
   if (!it)
   {
      ierr = VecDuplicate(X,&pX); CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)snes,"_mfem_snes_xp",(PetscObject)pX);
      CHKERRQ(ierr);
      ierr = VecDestroy(&pX); CHKERRQ(ierr);
   }
   ierr = PetscObjectQuery((PetscObject)snes,"_mfem_snes_xp",(PetscObject*)&pX);
   CHKERRQ(ierr);
   if (!pX) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_USER,
                       "Missing previous solution");
   ierr = SNESGetSolutionUpdate(snes,&dX); CHKERRQ(ierr);
   mfem::PetscParVector f(F,true);
   mfem::PetscParVector x(X,true);
   mfem::PetscParVector dx(dX,true);
   mfem::PetscParVector px(pX,true);
   (*snes_ctx->update)(snes_ctx->op,it,f,x,dx,px);
   /* Store previous solution */
   ierr = VecCopy(X,pX); CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_ksp_monitor(KSP ksp, PetscInt it, PetscReal res,
                                         void* ctx)
{
   __mfem_monitor_ctx *monctx = (__mfem_monitor_ctx*)ctx;

   PetscFunctionBeginUser;
   if (!monctx)
   {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Missing monitor context");
   }

   mfem::PetscSolver *solver = (mfem::PetscSolver*)(monctx->solver);
   mfem::PetscSolverMonitor *user_monitor = (mfem::PetscSolverMonitor *)(
                                               monctx->monitor);
   if (user_monitor->mon_sol)
   {
      Vec x;
      PetscErrorCode ierr;

      ierr = KSPBuildSolution(ksp,NULL,&x); CHKERRQ(ierr);
      mfem::PetscParVector V(x,true);
      user_monitor->MonitorSolution(it,res,V);
   }
   if (user_monitor->mon_res)
   {
      Vec x;
      PetscErrorCode ierr;

      ierr = KSPBuildResidual(ksp,NULL,NULL,&x); CHKERRQ(ierr);
      mfem::PetscParVector V(x,true);
      user_monitor->MonitorResidual(it,res,V);
   }
   user_monitor->MonitorSolver(solver);
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_mat_shell_apply(Mat A, Vec x, Vec y)
{
   mfem::Operator *op;
   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,(void **)&op); CHKERRQ(ierr);
   if (!op) { SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_LIB,"Missing operator"); }
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   op->Mult(xx,yy);
   yy.UpdateVecFromFlags();
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_mat_shell_apply_transpose(Mat A, Vec x, Vec y)
{
   mfem::Operator *op;
   PetscErrorCode ierr;
   PetscBool      flg,symm;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,(void **)&op); CHKERRQ(ierr);
   if (!op) { SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_LIB,"Missing operator"); }
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   ierr = MatIsSymmetricKnown(A,&flg,&symm); CHKERRQ(ierr);
   if (flg && symm)
   {
      op->Mult(xx,yy);
   }
   else
   {
      op->MultTranspose(xx,yy);
   }
   yy.UpdateVecFromFlags();
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_mat_shell_copy(Mat A, Mat B, MatStructure str)
{
   mfem::Operator *op;
   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,(void **)&op); CHKERRQ(ierr);
   if (!op) { SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_LIB,"Missing operator"); }
   ierr = MatShellSetContext(B,(void *)op); CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_mat_shell_destroy(Mat A)
{
   PetscFunctionBeginUser;
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_pc_shell_view(PC pc, PetscViewer viewer)
{
   __mfem_pc_shell_ctx *ctx;
   PetscErrorCode      ierr;

   PetscFunctionBeginUser;
   ierr = PCShellGetContext(pc,(void **)&ctx); CHKERRQ(ierr);
   if (ctx->op)
   {
      PetscBool isascii;
      ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);
      CHKERRQ(ierr);

      mfem::PetscPreconditioner *ppc = dynamic_cast<mfem::PetscPreconditioner *>
                                       (ctx->op);
      if (ppc)
      {
         ierr = PCView(*ppc,viewer); CHKERRQ(ierr);
      }
      else
      {
         if (isascii)
         {
            ierr = PetscViewerASCIIPrintf(viewer,
                                          "No information available on the mfem::Solver\n");
            CHKERRQ(ierr);
         }
      }
      if (isascii && ctx->factory)
      {
         ierr = PetscViewerASCIIPrintf(viewer,
                                       "Number of preconditioners created by the factory %lu\n",ctx->numprec);
         CHKERRQ(ierr);
      }
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_pc_shell_apply(PC pc, Vec x, Vec y)
{
   __mfem_pc_shell_ctx *ctx;
   PetscErrorCode      ierr;

   PetscFunctionBeginUser;
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   ierr = PCShellGetContext(pc,(void **)&ctx); CHKERRQ(ierr);
   if (ctx->op)
   {
      ctx->op->Mult(xx,yy);
      yy.UpdateVecFromFlags();
   }
   else // operator is not present, copy x
   {
      yy = xx;
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_pc_shell_apply_transpose(PC pc, Vec x, Vec y)
{
   __mfem_pc_shell_ctx *ctx;
   PetscErrorCode      ierr;

   PetscFunctionBeginUser;
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   ierr = PCShellGetContext(pc,(void **)&ctx); CHKERRQ(ierr);
   if (ctx->op)
   {
      ctx->op->MultTranspose(xx,yy);
      yy.UpdateVecFromFlags();
   }
   else // operator is not present, copy x
   {
      yy = xx;
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_pc_shell_setup(PC pc)
{
   __mfem_pc_shell_ctx *ctx;

   PetscFunctionBeginUser;
   ierr = PCShellGetContext(pc,(void **)&ctx); CHKERRQ(ierr);
   if (ctx->factory)
   {
      // Delete any owned operator
      if (ctx->ownsop)
      {
         delete ctx->op;
      }

      // Get current preconditioning Mat
      Mat B;
      ierr = PCGetOperators(pc,NULL,&B); CHKERRQ(ierr);

      // Call user-defined setup
      mfem::OperatorHandle hB(new mfem::PetscParMatrix(B,true),true);
      mfem::PetscPreconditionerFactory *factory = ctx->factory;
      ctx->op = factory->NewPreconditioner(hB);
      ctx->ownsop = true;
      ctx->numprec++;
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_pc_shell_destroy(PC pc)
{
   __mfem_pc_shell_ctx *ctx;
   PetscErrorCode      ierr;

   PetscFunctionBeginUser;
   ierr = PCShellGetContext(pc,(void **)&ctx); CHKERRQ(ierr);
   if (ctx->ownsop)
   {
      delete ctx->op;
   }
   delete ctx;
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_array_container_destroy(void *ptr)
{
   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   ierr = PetscFree(ptr); CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_matarray_container_destroy(void *ptr)
{
   mfem::Array<Mat> *a = (mfem::Array<Mat>*)ptr;
   PetscErrorCode   ierr;

   PetscFunctionBeginUser;
   for (int i=0; i<a->Size(); i++)
   {
      Mat M = (*a)[i];
      MPI_Comm comm = PetscObjectComm((PetscObject)M);
      ierr = MatDestroy(&M); CCHKERRQ(comm,ierr);
   }
   delete a;
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_monitor_ctx_destroy(void **ctx)
{
   PetscErrorCode  ierr;

   PetscFunctionBeginUser;
   ierr = PetscFree(*ctx); CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Sets the type of PC to PCSHELL and wraps the solver action
// if ownsop is true, ownership of precond is transferred to the PETSc object
PetscErrorCode MakeShellPC(PC pc, mfem::Solver &precond, bool ownsop)
{
   PetscFunctionBeginUser;
   __mfem_pc_shell_ctx *ctx = new __mfem_pc_shell_ctx;
   ctx->op      = &precond;
   ctx->ownsop  = ownsop;
   ctx->factory = NULL;
   ctx->numprec = 0;

   // In case the PC was already of type SHELL, this will destroy any
   // previous user-defined data structure
   // We cannot call PCReset as it will wipe out any operator already set
   ierr = PCSetType(pc,PCNONE); CHKERRQ(ierr);

   ierr = PCSetType(pc,PCSHELL); CHKERRQ(ierr);
   ierr = PCShellSetName(pc,"MFEM Solver (unknown Pmat)"); CHKERRQ(ierr);
   ierr = PCShellSetContext(pc,(void *)ctx); CHKERRQ(ierr);
   ierr = PCShellSetApply(pc,__mfem_pc_shell_apply); CHKERRQ(ierr);
   ierr = PCShellSetApplyTranspose(pc,__mfem_pc_shell_apply_transpose);
   CHKERRQ(ierr);
   ierr = PCShellSetSetUp(pc,__mfem_pc_shell_setup); CHKERRQ(ierr);
   ierr = PCShellSetView(pc,__mfem_pc_shell_view); CHKERRQ(ierr);
   ierr = PCShellSetDestroy(pc,__mfem_pc_shell_destroy); CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Sets the type of PC to PCSHELL. Uses a PetscPreconditionerFactory to construct the solver
// Takes ownership of the solver created by the factory
PetscErrorCode MakeShellPCWithFactory(PC pc,
                                      mfem::PetscPreconditionerFactory *factory)
{
   PetscFunctionBeginUser;
   __mfem_pc_shell_ctx *ctx = new __mfem_pc_shell_ctx;
   ctx->op      = NULL;
   ctx->ownsop  = true;
   ctx->factory = factory;
   ctx->numprec = 0;

   // In case the PC was already of type SHELL, this will destroy any
   // previous user-defined data structure
   // We cannot call PCReset as it will wipe out any operator already set
   ierr = PCSetType(pc,PCNONE); CHKERRQ(ierr);

   ierr = PCSetType(pc,PCSHELL); CHKERRQ(ierr);
   ierr = PCShellSetName(pc,factory->GetName()); CHKERRQ(ierr);
   ierr = PCShellSetContext(pc,(void *)ctx); CHKERRQ(ierr);
   ierr = PCShellSetApply(pc,__mfem_pc_shell_apply); CHKERRQ(ierr);
   ierr = PCShellSetApplyTranspose(pc,__mfem_pc_shell_apply_transpose);
   CHKERRQ(ierr);
   ierr = PCShellSetSetUp(pc,__mfem_pc_shell_setup); CHKERRQ(ierr);
   ierr = PCShellSetView(pc,__mfem_pc_shell_view); CHKERRQ(ierr);
   ierr = PCShellSetDestroy(pc,__mfem_pc_shell_destroy); CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Converts from a list (or a marked Array if islist is false) to an IS
// st indicates the offset where to start numbering
static PetscErrorCode Convert_Array_IS(MPI_Comm comm, bool islist,
                                       const mfem::Array<int> *list,
                                       PetscInt st, IS* is)
{
   PetscInt       n = list ? list->Size() : 0,*idxs;
   const int      *data = list ? list->GetData() : NULL;
   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   ierr = PetscMalloc1(n,&idxs); CHKERRQ(ierr);
   if (islist)
   {
      for (PetscInt i=0; i<n; i++) { idxs[i] = data[i] + st; }
   }
   else
   {
      PetscInt cum = 0;
      for (PetscInt i=0; i<n; i++)
      {
         if (data[i]) { idxs[cum++] = i+st; }
      }
      n = cum;
   }
   ierr = ISCreateGeneral(comm,n,idxs,PETSC_OWN_POINTER,is);
   CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Converts from a marked Array of Vdofs to an IS
// st indicates the offset where to start numbering
// l2l is a vector of matrices generated during RAP
static PetscErrorCode Convert_Vmarks_IS(MPI_Comm comm,
                                        mfem::Array<Mat> &pl2l,
                                        const mfem::Array<int> *mark,
                                        PetscInt st, IS* is)
{
   mfem::Array<int> sub_dof_marker;
   mfem::Array<mfem::SparseMatrix*> l2l(pl2l.Size());
   PetscInt         nl;
   PetscErrorCode   ierr;

   PetscFunctionBeginUser;
   for (int i = 0; i < pl2l.Size(); i++)
   {
      PetscInt  m,n,*ii,*jj;
      PetscBool done;
      ierr = MatGetRowIJ(pl2l[i],0,PETSC_FALSE,PETSC_FALSE,&m,(const PetscInt**)&ii,
                         (const PetscInt**)&jj,&done); CHKERRQ(ierr);
      MFEM_VERIFY(done,"Unable to perform MatGetRowIJ on " << i << " l2l matrix");
      ierr = MatGetSize(pl2l[i],NULL,&n); CHKERRQ(ierr);
#if defined(PETSC_USE_64BIT_INDICES)
      int  nnz = (int)ii[m];
      int *mii = new int[m+1];
      int *mjj = new int[nnz];
      for (int j = 0; j < m+1; j++) { mii[j] = (int)ii[j]; }
      for (int j = 0; j < nnz; j++) { mjj[j] = (int)jj[j]; }
      l2l[i] = new mfem::SparseMatrix(mii,mjj,NULL,m,n,true,true,true);
#else
      l2l[i] = new mfem::SparseMatrix(ii,jj,NULL,m,n,false,true,true);
#endif
      ierr = MatRestoreRowIJ(pl2l[i],0,PETSC_FALSE,PETSC_FALSE,&m,
                             (const PetscInt**)&ii,
                             (const PetscInt**)&jj,&done); CHKERRQ(ierr);
      MFEM_VERIFY(done,"Unable to perform MatRestoreRowIJ on "
                  << i << " l2l matrix");
   }
   nl = 0;
   for (int i = 0; i < l2l.Size(); i++) { nl += l2l[i]->Width(); }
   sub_dof_marker.SetSize(nl);
   const int* vdata = mark->GetData();
   int* sdata = sub_dof_marker.GetData();
   int cumh = 0, cumw = 0;
   for (int i = 0; i < l2l.Size(); i++)
   {
      const mfem::Array<int> vf_marker(const_cast<int*>(vdata)+cumh,
                                       l2l[i]->Height());
      mfem::Array<int> sf_marker(sdata+cumw,l2l[i]->Width());
      l2l[i]->BooleanMultTranspose(vf_marker,sf_marker);
      cumh += l2l[i]->Height();
      cumw += l2l[i]->Width();
   }
   ierr = Convert_Array_IS(comm,false,&sub_dof_marker,st,is); CCHKERRQ(comm,ierr);
   for (int i = 0; i < pl2l.Size(); i++)
   {
      delete l2l[i];
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

#if !defined(PETSC_HAVE_HYPRE)

#if defined(HYPRE_MIXEDINT)
#error "HYPRE_MIXEDINT not supported"
#endif

#include "_hypre_parcsr_mv.h"
static PetscErrorCode MatConvert_hypreParCSR_AIJ(hypre_ParCSRMatrix* hA,Mat* pA)
{
   MPI_Comm        comm = hypre_ParCSRMatrixComm(hA);
   hypre_CSRMatrix *hdiag,*hoffd;
   PetscScalar     *da,*oa,*aptr;
   PetscInt        *dii,*djj,*oii,*ojj,*iptr;
   PetscInt        i,dnnz,onnz,m,n;
   PetscMPIInt     size;
   PetscErrorCode  ierr;

   PetscFunctionBeginUser;
   hdiag = hypre_ParCSRMatrixDiag(hA);
   hoffd = hypre_ParCSRMatrixOffd(hA);
   m     = hypre_CSRMatrixNumRows(hdiag);
   n     = hypre_CSRMatrixNumCols(hdiag);
   dnnz  = hypre_CSRMatrixNumNonzeros(hdiag);
   onnz  = hypre_CSRMatrixNumNonzeros(hoffd);
   ierr  = PetscMalloc1(m+1,&dii); CHKERRQ(ierr);
   ierr  = PetscMalloc1(dnnz,&djj); CHKERRQ(ierr);
   ierr  = PetscMalloc1(dnnz,&da); CHKERRQ(ierr);
   ierr  = PetscMemcpy(dii,hypre_CSRMatrixI(hdiag),(m+1)*sizeof(PetscInt));
   CHKERRQ(ierr);
   ierr  = PetscMemcpy(djj,hypre_CSRMatrixJ(hdiag),dnnz*sizeof(PetscInt));
   CHKERRQ(ierr);
   ierr  = PetscMemcpy(da,hypre_CSRMatrixData(hdiag),dnnz*sizeof(PetscScalar));
   CHKERRQ(ierr);
   iptr  = djj;
   aptr  = da;
   for (i=0; i<m; i++)
   {
      PetscInt nc = dii[i+1]-dii[i];
      ierr = PetscSortIntWithScalarArray(nc,iptr,aptr); CHKERRQ(ierr);
      iptr += nc;
      aptr += nc;
   }
   ierr = MPI_Comm_size(comm,&size); CHKERRQ(ierr);
   if (size > 1)
   {
      PetscInt *offdj,*coffd;

      ierr  = PetscMalloc1(m+1,&oii); CHKERRQ(ierr);
      ierr  = PetscMalloc1(onnz,&ojj); CHKERRQ(ierr);
      ierr  = PetscMalloc1(onnz,&oa); CHKERRQ(ierr);
      ierr  = PetscMemcpy(oii,hypre_CSRMatrixI(hoffd),(m+1)*sizeof(PetscInt));
      CHKERRQ(ierr);
      offdj = hypre_CSRMatrixJ(hoffd);
      coffd = hypre_ParCSRMatrixColMapOffd(hA);
      for (i=0; i<onnz; i++) { ojj[i] = coffd[offdj[i]]; }
      ierr  = PetscMemcpy(oa,hypre_CSRMatrixData(hoffd),onnz*sizeof(PetscScalar));
      CHKERRQ(ierr);
      iptr  = ojj;
      aptr  = oa;
      for (i=0; i<m; i++)
      {
         PetscInt nc = oii[i+1]-oii[i];
         ierr = PetscSortIntWithScalarArray(nc,iptr,aptr); CHKERRQ(ierr);
         iptr += nc;
         aptr += nc;
      }
      ierr = MatCreateMPIAIJWithSplitArrays(comm,m,n,PETSC_DECIDE,PETSC_DECIDE,dii,
                                            djj,da,oii,ojj,oa,pA); CHKERRQ(ierr);
   }
   else
   {
      oii = ojj = NULL;
      oa = NULL;
      ierr = MatCreateSeqAIJWithArrays(comm,m,n,dii,djj,da,pA); CHKERRQ(ierr);
   }
   /* We are responsible to free the CSR arrays.  However, since we can take
      references of a PetscParMatrix but we cannot take reference of PETSc
      arrays, we need to create a PetscContainer object to take reference of
      these arrays in reference objects */
   void *ptrs[6] = {dii,djj,da,oii,ojj,oa};
   const char *names[6] = {"_mfem_csr_dii",
                           "_mfem_csr_djj",
                           "_mfem_csr_da",
                           "_mfem_csr_oii",
                           "_mfem_csr_ojj",
                           "_mfem_csr_oa"
                          };
   for (i=0; i<6; i++)
   {
      PetscContainer c;

      ierr = PetscContainerCreate(comm,&c); CHKERRQ(ierr);
      ierr = PetscContainerSetPointer(c,ptrs[i]); CHKERRQ(ierr);
      ierr = PetscContainerSetUserDestroy(c,__mfem_array_container_destroy);
      CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)(*pA),names[i],(PetscObject)c);
      CHKERRQ(ierr);
      ierr = PetscContainerDestroy(&c); CHKERRQ(ierr);
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvert_hypreParCSR_IS(hypre_ParCSRMatrix* hA,Mat* pA)
{
   Mat                    lA;
   ISLocalToGlobalMapping rl2g,cl2g;
   IS                     is;
   hypre_CSRMatrix        *hdiag,*hoffd;
   MPI_Comm               comm = hypre_ParCSRMatrixComm(hA);
   void                   *ptrs[2];
   const char             *names[2] = {"_mfem_csr_aux",
                                       "_mfem_csr_data"
                                      };
   PetscScalar            *hdd,*hod,*aa,*data;
   PetscInt               *col_map_offd,*hdi,*hdj,*hoi,*hoj;
   PetscInt               *aux,*ii,*jj;
   PetscInt               cum,dr,dc,oc,str,stc,nnz,i,jd,jo;
   PetscErrorCode         ierr;

   PetscFunctionBeginUser;
   /* access relevant information in ParCSR */
   str   = hypre_ParCSRMatrixFirstRowIndex(hA);
   stc   = hypre_ParCSRMatrixFirstColDiag(hA);
   hdiag = hypre_ParCSRMatrixDiag(hA);
   hoffd = hypre_ParCSRMatrixOffd(hA);
   dr    = hypre_CSRMatrixNumRows(hdiag);
   dc    = hypre_CSRMatrixNumCols(hdiag);
   nnz   = hypre_CSRMatrixNumNonzeros(hdiag);
   hdi   = hypre_CSRMatrixI(hdiag);
   hdj   = hypre_CSRMatrixJ(hdiag);
   hdd   = hypre_CSRMatrixData(hdiag);
   oc    = hypre_CSRMatrixNumCols(hoffd);
   nnz  += hypre_CSRMatrixNumNonzeros(hoffd);
   hoi   = hypre_CSRMatrixI(hoffd);
   hoj   = hypre_CSRMatrixJ(hoffd);
   hod   = hypre_CSRMatrixData(hoffd);

   /* generate l2g maps for rows and cols */
   ierr = ISCreateStride(comm,dr,str,1,&is); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g); CHKERRQ(ierr);
   ierr = ISDestroy(&is); CHKERRQ(ierr);
   col_map_offd = hypre_ParCSRMatrixColMapOffd(hA);
   ierr = PetscMalloc1(dc+oc,&aux); CHKERRQ(ierr);
   for (i=0; i<dc; i++) { aux[i]    = i+stc; }
   for (i=0; i<oc; i++) { aux[i+dc] = col_map_offd[i]; }
   ierr = ISCreateGeneral(comm,dc+oc,aux,PETSC_OWN_POINTER,&is); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g); CHKERRQ(ierr);
   ierr = ISDestroy(&is); CHKERRQ(ierr);

   /* create MATIS object */
   ierr = MatCreate(comm,pA); CHKERRQ(ierr);
   ierr = MatSetSizes(*pA,dr,dc,PETSC_DECIDE,PETSC_DECIDE); CHKERRQ(ierr);
   ierr = MatSetType(*pA,MATIS); CHKERRQ(ierr);
   ierr = MatSetLocalToGlobalMapping(*pA,rl2g,cl2g); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingDestroy(&rl2g); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingDestroy(&cl2g); CHKERRQ(ierr);

   /* merge local matrices */
   ierr = PetscMalloc1(nnz+dr+1,&aux); CHKERRQ(ierr);
   ierr = PetscMalloc1(nnz,&data); CHKERRQ(ierr);
   ii   = aux;
   jj   = aux+dr+1;
   aa   = data;
   *ii  = *(hdi++) + *(hoi++);
   for (jd=0,jo=0,cum=0; *ii<nnz; cum++)
   {
      PetscScalar *aold = aa;
      PetscInt    *jold = jj,nc = jd+jo;
      for (; jd<*hdi; jd++) { *jj++ = *hdj++;      *aa++ = *hdd++; }
      for (; jo<*hoi; jo++) { *jj++ = *hoj++ + dc; *aa++ = *hod++; }
      *(++ii) = *(hdi++) + *(hoi++);
      ierr = PetscSortIntWithScalarArray(jd+jo-nc,jold,aold); CHKERRQ(ierr);
   }
   for (; cum<dr; cum++) { *(++ii) = nnz; }
   ii   = aux;
   jj   = aux+dr+1;
   aa   = data;
   ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,dr,dc+oc,ii,jj,aa,&lA);
   CHKERRQ(ierr);
   ptrs[0] = aux;
   ptrs[1] = data;
   for (i=0; i<2; i++)
   {
      PetscContainer c;

      ierr = PetscContainerCreate(PETSC_COMM_SELF,&c); CHKERRQ(ierr);
      ierr = PetscContainerSetPointer(c,ptrs[i]); CHKERRQ(ierr);
      ierr = PetscContainerSetUserDestroy(c,__mfem_array_container_destroy);
      CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)lA,names[i],(PetscObject)c);
      CHKERRQ(ierr);
      ierr = PetscContainerDestroy(&c); CHKERRQ(ierr);
   }
   ierr = MatISSetLocalMat(*pA,lA); CHKERRQ(ierr);
   ierr = MatDestroy(&lA); CHKERRQ(ierr);
   ierr = MatAssemblyBegin(*pA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   ierr = MatAssemblyEnd(*pA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

#include <petsc/private/matimpl.h>

static PetscErrorCode __mfem_MatCreateDummy(MPI_Comm comm, PetscInt m,
                                            PetscInt n, Mat *A)
{
   PetscFunctionBegin;
   ierr = MatCreate(comm,A); CHKERRQ(ierr);
   ierr = MatSetSizes(*A,m,n,PETSC_DECIDE,PETSC_DECIDE); CHKERRQ(ierr);
   ierr = PetscObjectChangeTypeName((PetscObject)*A,"mfemdummy"); CHKERRQ(ierr);
   (*A)->preallocated = PETSC_TRUE;
   ierr = MatSetUp(*A); CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petsc/private/vecimpl.h>

#if defined(PETSC_HAVE_DEVICE)
static PetscErrorCode __mfem_VecSetOffloadMask(Vec v, PetscOffloadMask m)
{
   PetscFunctionBegin;
   v->offloadmask = m;
   PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode __mfem_VecBoundToCPU(Vec v, PetscBool *flg)
{
   PetscFunctionBegin;
#if defined(PETSC_HAVE_DEVICE)
   *flg = v->boundtocpu;
#else
   *flg = PETSC_TRUE;
#endif
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_PetscObjectStateIncrease(PetscObject o)
{
   PetscErrorCode ierr;

   PetscFunctionBegin;
   ierr = PetscObjectStateIncrease(o); CHKERRQ(ierr);
   PetscFunctionReturn(PETSC_SUCCESS);
}

#endif  // MFEM_USE_PETSC
#endif  // MFEM_USE_MPI
