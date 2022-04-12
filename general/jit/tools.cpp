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

#include "tools.hpp"

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 206
#include "../debug.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

namespace jit
{

const char* strrnc(const char *s, const unsigned char c, int n)
{
   size_t len = strlen(s);
   char* p = const_cast<char*>(s)+len-1;
   for (; n; n--,p--,len--)
   {
      for (; len; p--,len--)
         if (*p == c) { break; }
      if (!len) { return nullptr; }
      if (n == 1) { return p; }
   }
   return nullptr;
}

#ifdef MFEM_USE_MPI
constexpr int THREAD_TIMEOUT = 4000;

static inline int nsleep(const long us)
{
   const long ns = us *1000L;
   const struct timespec rqtp = { 0, ns };
   return nanosleep(&rqtp, nullptr);
}

static int THREAD_Worker(char *argv[], int *status)
{
   string command(argv[2]);
   for (int k = 3; argv[k]; k++)
   {
      command.append(" ");
      command.append(argv[k]);
   }
   const char *command_c_str = command.c_str();
   dbg("command_c_str: %s", command_c_str);
   dbg("Waiting for the cookie from parent...");
   int timeout = THREAD_TIMEOUT;
   while (*status != mfem::jit::SYSTEM_CALL && timeout > 0) { nsleep(timeout--); }
   dbg("Got it, now do the system call");
   const int return_value = std::system(command_c_str);
   dbg("return_value: %d", return_value);
   *status = (return_value == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
   return return_value;
}

static int MPI_Spawned(int argc, char *argv[], int *status)
{
   MPI_Init(&argc, &argv);
   dbg();
   MPI_Comm intercomm = MPI_COMM_NULL;
   MPI_Comm_get_parent(&intercomm);
   // This case should not happen
   if (intercomm == MPI_COMM_NULL) { dbg("ERROR"); return EXIT_FAILURE; }
   // MPI child
   int ready = 0;
   // Waiting for the parent MPI to set 'READY'
   MPI_Bcast(&ready, 1, MPI_INT, 0, intercomm);
   if (ready != READY)
   {
      dbg("Spawned JIT process did not received READY!");
      return EXIT_FAILURE;
   }
   // Now inform the thread worker to launch the system call
   int timeout = THREAD_TIMEOUT;
   for (*status  = mfem::jit::SYSTEM_CALL;
        *status == mfem::jit::SYSTEM_CALL && timeout>0;
        nsleep(timeout--));
   if (timeout == 0)
   {
      dbg("Spawned JIT process timeout!");
      return EXIT_FAILURE;
   }
   // The worker should have updated the status in memory
   // Broadcast back the result to the MPI world
   MPI_Bcast(status, 1, MPI_INT, MPI_ROOT, intercomm);
   MPI_Finalize();
   return EXIT_SUCCESS;
}

// The MPI sub group can fork before MPI_Init
int ProcessFork(int argc, char *argv[])
{
   dbg("\033[33m[ProcessFork]");
   constexpr void *addr = 0;
   constexpr int len = sizeof(int);
   constexpr int prot = PROT_READ | PROT_WRITE;
   constexpr int flags = MAP_SHARED | MAP_ANONYMOUS;
   // One integer in memory to exchange the status of the command
   int *status = (int*) mmap(addr, len, prot, flags, -1, 0);
   if (!status) { return EXIT_FAILURE; }
   *status = 0;
   const pid_t child_pid = fork();
   if (child_pid != 0 )
   {
      // The child will keep on toward MPI_Init and connect through intercomm
      if (MPI_Spawned(argc, argv, status) != 0) { return EXIT_FAILURE; }
      if (munmap(addr, len) != 0) { dbg("munmap error"); return EXIT_FAILURE; }
      return EXIT_SUCCESS;
   }
   // Worker will now be able to launch the std::system
   // but will wait for the MPI child in the sub world to connect with the world
   return THREAD_Worker(argv, status);
}
#endif // MFEM_USE_MPI


#ifndef MFEM_USE_MPI
bool Root() { return true;}
#else // MFEM_USE_MPI
bool Root()
{
   int world_rank = 0;
   if (MPI_Inited()) { MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); }
   return world_rank == 0;
}

int MPI_Size()
{
   int size = 1;
   if (MPI_Inited()) { MPI_Comm_size(MPI_COMM_WORLD, &size); }
   return size;
}

bool MPI_Inited()
{
   int ini = false;
   MPI_Initialized(&ini);
   return ini ? true : false;
}
#endif // MFEM_USE_MPI

/// \brief GetRuntimeVersion
/// \param increment
/// \return the current runtime version
int GetRuntimeVersion(bool increment)
{
   static int version = 0;
   const int actual = version;
   if (increment) { version += 1; }
   return actual;
}

/// Root MPI process file creation, outputing the source of the kernel.
/// ipcrm -a
/// ipcs -m
bool CreateMappedSharedMemoryOutputFile(const char *out, int &fd, char *&pmap)
{
   if (!Root()) { return true; }

   dbg("output: (/dev/shm/) %s",out);

   constexpr int SHM_MAX_SIZE = 2*1024*1024;

   // Remove shared memory segment if it already exists.
   ::shm_unlink(out);

   // Attempt to create shared  memory segment
   const mode_t mode = S_IRUSR | S_IWUSR;
   const int oflag = O_CREAT | O_RDWR | O_TRUNC;
   fd = ::shm_open(out, oflag, mode);
   if (fd < 0)
   {
      exit(EXIT_FAILURE|
           printf("\033[31;1m[shmOpen] Shared memory failed: %s\033[m\n",
                  strerror(errno)));
      return false;
   }

   // resize shm to the right size
   if (::ftruncate(fd, SHM_MAX_SIZE) < 0)
   {
      ::shm_unlink(out);
      dbg("!ftruncate");
      return false;
   }

   // Map the shared memory segment into the process address space
   const int prot = PROT_READ | PROT_WRITE;
   const int flags = MAP_SHARED;
   pmap = (char*) mmap(nullptr,      // Most of the time set to nullptr
                       SHM_MAX_SIZE, // Size of memory mapping
                       prot,         // Allows reading and writing operations
                       flags,        // Segment visible by other processes
                       fd,           // File descriptor
                       0x0);         // Offset from beggining of file
   if (pmap == MAP_FAILED) { dbg("!pmap"); return false; }


   dbg("ofd:%d",fd);
   if (::close(fd) < 0) { dbg("!close"); return false; }

   return true;
}

} // namespace jit

} // namespace mfem

