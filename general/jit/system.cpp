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

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include <string>
using std::string;

#include "jit.hpp"
#include "tools.hpp"

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 226
#include "../debug.hpp"

namespace mfem
{

namespace jit
{

#ifndef MFEM_USE_MPI

// The serial implementation does nothing special but the =system= command.
int System(const char *argv[])
{
   const int argc = argn(argv);
   if (argc < 2) { return EXIT_FAILURE; }
   string command(argv[1]);
   for (int k = 2; k < argc && argv[k]; k++)
   {
      command.append(" ");
      command.append(argv[k]);
   }
   const char *command_c_str = command.c_str();
   dbg(command_c_str);
   return ::system(command_c_str);
}

#else // MFEM_USE_MPI
/**
 *  MFEM ranks which calls mfem::jit::System are { MPI Parents }.
 *  The root of { MPI Parents } calls the binary mjit through MPI_Comm_spawn,
 *  which calls fork to create one { THREAD Worker } and { MPI Spawned }.
 *  The { THREAD Worker } is created before MPI_Init of the { MPI Spawned }.
 *  The { MPI Spawned } waits for MAGIC cookie check through the intercomm.
 *  The { MPI Spawned } triggers the { THREAD Worker } through mapped memory.
 *  The { THREAD Worker } waits for the SYSTEM_CALL order and calls system,
 *  which passes the command to the shell.
 *  The return status goes back through mapped memory and back to the
 *  { MPI Parents } with a broadcast.
 */

/// launch the std::system through:
///   - MPI_Comm_spawn => mjit (executable) (using the -c command line option)
///   - mjit => ProcessFork: MPI_Spawned and THREAD_Worker
///   - MPI_Spawned drives the working thread, provides the arguments and wait
///   - THREAD_Worker waits for commands and does the std::system call
static int System_MPISpawn(char *argv[])
{
   const int argc = argn(argv);
   if (argc < 2) { return EXIT_FAILURE; }
   // Point to the 'mjit' binary
   char mjit[PATH_MAX];
   if (snprintf(mjit, PATH_MAX, "%s/bin/mjit", MFEM_INSTALL_DIR) < 0)
   { return EXIT_FAILURE; }
   dbg(mjit);
   // If we have not been launch with mpirun, just fold back to serial case,
   // which has a shift in the arguments
   if (!MPI_Inited() || MPI_Size()==1)
   {
      string command(argv[1]);
      for (int k = 2; k < argc && argv[k]; k++)
      {
         command.append(" ");
         command.append(argv[k]);
      }
      const char *command_c_str = command.c_str();
      dbg(command_c_str);
      return system(command_c_str);
   }

   // Debug our command
   string command(argv[0]);
   for (int k = 1; k < argc && argv[k]; k++)
   {
      command.append(" ");
      command.append(argv[k]);
   }
   const char *command_c_str = command.c_str();
   dbg(command_c_str);

   // Spawn the sub MPI group
   constexpr int root = 0;
   int errcode = EXIT_FAILURE;
   const MPI_Info info = MPI_INFO_NULL;
   MPI_Comm comm = MPI_COMM_WORLD, intercomm = MPI_COMM_NULL;
   MPI_Barrier(comm);
   const int spawned = // Now spawn one binary
      MPI_Comm_spawn(mjit, argv, 1, info, root, comm, &intercomm, &errcode);
   if (spawned != MPI_SUCCESS) { return EXIT_FAILURE; }
   if (errcode != EXIT_SUCCESS) { return EXIT_FAILURE; }
   // Broadcast READY through intercomm, and wait for return
   int status = mfem::jit::READY;
   // MPI_ROOT stands as a special value for intercomms
   MPI_Bcast(&status, 1, MPI_INT, MPI_ROOT, intercomm);
   MPI_Bcast(&status, 1, MPI_INT, root, intercomm);
   MPI_Barrier(comm);
   MPI_Comm_free(&intercomm);
   return status;
}


/// Just launch the std::system
static int System_Std(const char *argv[])
{
   dbg();
   MPI_Comm comm = MPI_COMM_WORLD;
   if (MPI_Inited()) { MPI_Barrier(comm); }

   const int argc = argn(argv);
   if (argc < 2) { return EXIT_FAILURE; }

   // https://www.open-mpi.org/faq/?category=openfabrics#ofa-fork
   if (Root())
   {
      string command(argv[1]);
      for (int k = 2; k < argc && argv[k]; k++)
      {
         command.append(" ");
         command.append(argv[k]);
      }
      const char *command_c_str = command.c_str();
      dbg(command_c_str);
      std::system(command_c_str);
   }

   if (MPI_Inited()) { MPI_Barrier(comm); }
   return EXIT_SUCCESS;
}


/// launch the std::system through:
///   - child (jit_compiler_pid) of MPI_JIT_Session in communication.hpp
int System_MPI_JIT_Session(char *argv[])
{
   dbg();
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Barrier(comm);

   const int argc = argn(argv);
   if (argc < 2) { return EXIT_FAILURE; }

   // https://www.open-mpi.org/faq/?category=openfabrics#ofa-fork
   if (Root())
   {
      string command(argv[1]);
      for (int k = 2; k < argc && argv[k]; k++)
      {
         command.append(" ");
         command.append(argv[k]);
      }
      const char *command_c_str = command.c_str();
      dbg(command_c_str);
      std::system(command_c_str);
   }

   MPI_Barrier(comm);
   return EXIT_SUCCESS;
}

/// Entry point toward System_MPISpawn or System_MPI_JIT_Session
int System(const char *argv[])
{
   const bool spawn = getenv("MFEM_JIT_MPI_SPAWN") != nullptr;
   if (spawn) { return System_MPISpawn(const_cast<char**>(argv)); }

   const bool thread = getenv("MFEM_JIT_MPI_FORK") != nullptr;
   if (thread) { return System_MPI_JIT_Session(const_cast<char**>(argv)); }

   return  System_Std(argv);
}

#endif // MFEM_USE_MPI

} // namespace jit

} // namespace mfem

