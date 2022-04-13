// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_JIT_TOOLS_HPP
#define MFEM_JIT_TOOLS_HPP

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define MFEM_DEBUG_COLOR 118
#include "../debug.hpp"

#include "../../config/config.hpp"

namespace mfem
{

namespace jit
{

/// \brief GetRuntimeVersion Returns the library version of the current run.
///        Initialized at '0', can be incremented by setting increment to true.
/// \param increment
/// \return the current runtime version
int GetRuntimeVersion(bool increment = false);

const char* strrnc(const char *s, const unsigned char c, int n = 1);

#ifdef MFEM_USE_MPI
int ProcessFork(int argc, char *argv[]);
#endif // MFEM_USE_MPI

// Different commands that can be broadcasted
enum Command
{
   READY,
   SYSTEM_CALL
};

// In serial, implementation of mfem::jit::System
// In parallel, entry point toward System_[Std | MPISpawn | MPI_JIT_Session]
int System(const char *argv[]);

/// Returns true if MPI world rank is zero.
bool Root();

void MPI_Sync();
int MPI_Size();

bool MPI_Inited();

template<typename T>
static inline int argn(T argv[], int argc = 0)
{
   while (argv[argc]) { argc += 1; }
   return argc;
}

/// Root MPI process file creation, outputing the source of the kernel.
bool CreateMappedSharedMemoryOutputFile(const char *out, int &fd, char *&pmap);

void CreateMapSMemInputFile(const char *input, int size, int &fd, char *&pmap);

/// Root MPI process file creation, outputing the source of the kernel.
template<typename... Args>
inline bool CreateMappedSharedMemoryInputFile(const char *input,
                                              const size_t h,
                                              const char *src,
                                              int &fd,
                                              char *&pmap, Args... args)
{
   if (!Root()) { return true; }
   const int size = 1 + std::snprintf(nullptr, 0, src, h, h, h, args...);
   CreateMapSMemInputFile(input, size, fd, pmap);
   if (std::snprintf(pmap, size, src, h, h, h, args...) < 0)
   {
      return perror(strerror(errno)), false;
   }
   if (::close(fd) < 0) { return perror(strerror(errno)), false; }
   return true;
}

} // namespace jit

} // namespace mfem

#endif // MFEM_MJIT_TOOLS_HPP
