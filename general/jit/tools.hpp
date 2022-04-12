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

int MPI_Size();

bool MPI_Inited();

template<typename T>
static inline int argn(T argv[], int argc = 0)
{
   while (argv[argc]) { argc += 1; }
   return argc;
}


/// Root MPI process file creation, outputing the source of the kernel.
template<typename... Args>
inline bool CreateMappedSharedMemoryInputFile(const char *input,
                                              const size_t h,
                                              const char *src,
                                              int &fd,
                                              char *&pmap, Args... args)
{
   if (!Root()) { return true; }

   dbg("input: (/dev/shm/) %s", input);

   // Remove shared memory segment if it already exists.
   ::shm_unlink(input);

   // Attempt to create shared memory segment
   const mode_t mode = S_IRUSR | S_IWUSR;
   const int oflag = O_CREAT | O_RDWR | O_EXCL;
   fd = ::shm_open(input, oflag, mode);
   if (fd < 0) { return perror(strerror(errno)), false; }

   // determine the necessary buffer size
   const int size = 1 + std::snprintf(nullptr, 0, src, h, h, h, args...);
   dbg("size:%d", size);

   // resize the shared memory segment to the right size
   if (::ftruncate(fd, size) < 0)
   {
      ::shm_unlink(input); // ipcs -m
      dbg("!ftruncate");
      return false;
   }

   // Map the shared memory segment into the process address space
   const int prot = PROT_READ | PROT_WRITE;
   const int flags = MAP_SHARED;
   pmap = (char*) mmap(nullptr, // Most of the time set to nullptr
                       size,    // Size of memory mapping
                       prot,    // Allows reading and writing operations
                       flags,   // Segment visible by other processes
                       fd,      // File descriptor
                       0x00);   // Offset from beggining of file
   if (pmap == MAP_FAILED) { return perror(strerror(errno)), false; }

   if (std::snprintf(pmap, size, src, h, h, h, args...) < 0)
   {
      return perror("snprintf error occured"), false;
   }

   if (::close(fd) < 0) { return perror(strerror(errno)), false; }

   return true;
}

/// Root MPI process file creation, outputing the source of the kernel.
bool CreateMappedSharedMemoryOutputFile(const char *out, int &fd, char *&pmap);

} // namespace jit

} // namespace mfem

#endif // MFEM_MJIT_TOOLS_HPP
