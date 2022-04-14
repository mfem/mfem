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
#include <fstream>

#include "jit.hpp"
#include "tools.hpp"
#include "../communication.hpp"

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 226
#include "../debug.hpp"

namespace mfem
{

namespace jit
{

// The serial implementation does nothing special but the =system= command.
// Just launch the std::system on the Root MPI process
// https://www.open-mpi.org/faq/?category=openfabrics#ofa-fork
static int System_Serial(const char *argv[])
{
   const int argc = argn(argv);
   if (argc < 2) { return EXIT_FAILURE; }
   std::string command(argv[1]);
   for (int k = 2; k < argc && argv[k]; k++)
   {
      command += " ";
      command += argv[k];
   }
   const char *command_c_str = command.c_str();
   dbg(command_c_str);
   return ::system(command_c_str);
}

#ifdef MFEM_USE_MPI

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
/// MPI_ERR_SPAWN: could not spawn processes if not enough ranks
/*static int System_MPISpawn(const char *argv[], MPI_Comm comm = MPI_COMM_WORLD)
{
   constexpr int PM = PATH_MAX;
   const int argc = argn(argv);
   if (argc < 2) { return EXIT_FAILURE; }
   // Point to the 'mjit' binary
   char mjit[PM];
   if (snprintf(mjit, PM, "%s/bin/mjit", MFEM_INSTALL_DIR) < 0) { return EXIT_FAILURE; }
   dbg(mjit);
   assert(std::fstream(mjit)); // make sure the mjit executable exists

   // If we have not been launch with mpirun, just fold back to serial case,
   // which has a shift in the arguments
   if (!Mpi::IsInitialized() || MpiSize()==1) { return System_Serial(argv); }

   // Debug our command
   std::string dbg_command(argv[0]);
   for (int k = 1; k < argc && argv[k]; k++)
   {
      dbg_command.append(" ");
      dbg_command.append(argv[k]);
   }
   const char *dbg_command_c_str = dbg_command.c_str();
   dbg(dbg_command_c_str);

   // Spawn the sub MPI group
   constexpr int root = 0;
   constexpr int maxprocs = 1;
   int errcode = EXIT_FAILURE;
   const MPI_Info info = MPI_INFO_NULL;
   MPI_Comm intercomm = MPI_COMM_NULL;
   MPI_Barrier(comm);
   const int spawned =
      MPI_Comm_spawn(mjit, const_cast<char**>(argv),
                     maxprocs, info, root, comm, &intercomm, &errcode);
   if (spawned != MPI_SUCCESS) { return dbg("spawned"), EXIT_FAILURE; }
   if (errcode != EXIT_SUCCESS) { return dbg("errcode"), EXIT_FAILURE; }
   int status = mfem::jit::READY;
   // MPI_ROOT stands as a special value for intercomms
   MPI_Bcast(&status, 1, MPI_INT, MPI_ROOT, intercomm); // Broadcast READY
   MPI_Bcast(&status, 1, MPI_INT, root, intercomm); // Wait for return
   MPI_Barrier(comm);
   MPI_Comm_free(&intercomm);
   return status;
}

/// launch the std::system through:
///   - child (jit_compiler_pid) of MPI_JIT_Session in communication.hpp
int System_MPI_JIT_Session(const char *argv[], MPI_Comm comm = MPI_COMM_WORLD)
{
   dbg();
   MPI_Barrier(comm);
   const int argc = argn(argv);
   if (argc < 2) { return EXIT_FAILURE; }
   if (Mpi::Root()) { System_Serial(argv); }
   MPI_Barrier(comm);
   return EXIT_SUCCESS;
}*/

#endif // MFEM_USE_MPI

/// Entry point toward System_[Serial|MPISpawn|MPI_JIT_Session]
int System(const char *argv[])
{
   //return System_MPISpawn(argv);
   //return System_MPI_JIT_Session(argv);
   if (MpiRoot()) { System_Serial(argv); }
   MpiSync();
   return EXIT_SUCCESS;
}

} // namespace jit

} // namespace mfem

