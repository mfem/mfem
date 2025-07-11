// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ANNOTATION_HPP
#define MFEM_ANNOTATION_HPP

#include "../config/config.hpp"

#define MFEM_CONCAT_(X,Y)  X##Y
#define MFEM_CONCAT(X,Y)  MFEM_CONCAT_(X,Y)

#ifdef MFEM_USE_CALIPER
#include "device.hpp"
#include "backends.hpp"
#ifdef MFEM_USE_MPI
#include "communication.hpp"
#endif
#include <optional>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#endif

namespace mfem
{

namespace internal
{

extern int annotation_sync_stream; // defined in globals.cpp
extern int annotation_sync_mpi; // defined in globals.cpp
extern int annotation_enabled; // defined in globals.cpp

#ifdef MFEM_USE_CALIPER

inline void AnnotationSync()
{
   if (!annotation_enabled) { return; }
   if (annotation_sync_stream && Device::Allows(Backend::DEVICE_MASK))
   {
      MFEM_STREAM_SYNC;
   }
#ifdef MFEM_USE_MPI
   if (annotation_sync_mpi && Mpi::IsInitialized() && !Mpi::IsFinalized())
   {
      MPI_Barrier(GetGlobalMPI_Comm());
   }
#endif
}

struct FunctionAnnotation
{
   std::optional<cali::Function> cali_func;

   FunctionAnnotation(const char *fname)
   {
      AnnotationSync();
      if (annotation_enabled) { cali_func.emplace(fname); }
   }

   ~FunctionAnnotation() { AnnotationSync(); }
};

struct ScopeAnnotation
{
   std::optional<cali::ScopeAnnotation> cali_scope;

   ScopeAnnotation(const char *name)
   {
      AnnotationSync();
      if (annotation_enabled) { cali_scope.emplace(name); }
   }

   ~ScopeAnnotation() { AnnotationSync(); }
};

#endif // #ifdef MFEM_USE_CALIPER

} // namespace internal

} // namespace mfem


#ifdef MFEM_USE_CALIPER

#define MFEM_PERF_FUNCTION \
   mfem::internal::FunctionAnnotation mfem_func_annotation_(_MFEM_FUNC_NAME)
#define MFEM_PERF_BEGIN(s) \
   (mfem::internal::AnnotationSync(), CALI_MARK_BEGIN(s))
#define MFEM_PERF_END(s) \
   (mfem::internal::AnnotationSync(), CALI_MARK_END(s))
#define MFEM_PERF_SCOPE(name) \
   mfem::internal::ScopeAnnotation \
      MFEM_CONCAT(mfem_scope_annotation_,__LINE__)(name)

#define MFEM_PERF_SYNC_STREAM(b) (mfem::internal::annotation_sync_stream = (b))
#define MFEM_PERF_SYNC_MPI(b) (mfem::internal::annotation_sync_mpi = (b))
#define MFEM_PERF_SYNC(b) (MFEM_PERF_SYNC_STREAM(b), MFEM_PERF_SYNC_MPI(b))
#define MFEM_PERF_ENABLE (mfem::internal::annotation_enabled = 1)
#define MFEM_PERF_DISABLE (mfem::internal::annotation_enabled = 0)

#else // #ifdef MFEM_USE_CALIPER

#define MFEM_PERF_FUNCTION
#define MFEM_PERF_BEGIN(s)
#define MFEM_PERF_END(s)
#define MFEM_PERF_SCOPE(name)

#define MFEM_PERF_SYNC_STREAM(b)
#define MFEM_PERF_SYNC_MPI(b)
#define MFEM_PERF_SYNC(b)
#define MFEM_PERF_ENABLE
#define MFEM_PERF_DISABLE

#endif // #ifdef MFEM_USE_CALIPER

#endif // MFEM_ANNOTATION_HPP
