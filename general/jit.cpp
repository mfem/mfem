// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"
#include "error.hpp"

#ifdef MFEM_USE_MPI

#include <mpi.h>
#include <sys/types.h>
#include <unistd.h>
#include <cassert>
#include <sys/wait.h>

#include "communication.hpp"

using namespace std;

namespace mfem
{

namespace jit
{

// *****************************************************************************
#define dbg(...) { printf("\033[33m"); \
                   printf(__VA_ARGS__);  \
                   printf(" \n\033[m");fflush(0); }

// *****************************************************************************
int System(char *argv[] = MPI_ARGV_NULL)
{
   const bool debug = getenv("JIT_DBG");
   assert(!debug);
   constexpr size_t size = 4096;
   char command[size];
   if (snprintf(command, size,  "%s/../jit", MFEM_INSTALL_DIR)<0)
   { return EXIT_FAILURE; }
   dbg("command: %s", command);
   MPI_Info info = MPI_INFO_NULL;
   const int root = 0;
   MPI_Comm comm = MPI_COMM_WORLD, intercomm;
   int errcode;
   assert(unsetenv("JIT_DBG")==0);
   MPI_Comm_spawn(command, argv, 1, info, root, comm, &intercomm, &errcode);

   {
      int status = EXIT_FAILURE;
      dbg("MPI_Bcast");
      //dbg("\033[32m[System] status:%d,\n", status);

      constexpr size_t SZ = 4096;
      char command[SZ];
      command[0] = 0;
      for (int k=1; argv[k]; k++)
      {
         strcat(command, argv[k]);
         strcat(command, " ");
      }

      int length = strlen(command)-1;
      //dbg("\033[32m[System] length:%d,\n", length);
      MPI_Bcast(&length, 1, MPI_INT, MPI_ROOT, intercomm);

      //dbg("\033[32m[System] command:%s,\n", command);
      MPI_Bcast(&command, length, MPI_CHAR, MPI_ROOT, intercomm);

      MPI_Bcast(&status, 1, MPI_INT, 0, intercomm);
      assert(status == EXIT_SUCCESS);
      //dbg("\033[32m[System] status:%d,\n", status);
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Comm_free(&intercomm);
   }
   dbg("done");
   return EXIT_SUCCESS;
}

} // namespace jit

} // namespace mfem

// *****************************************************************************
int MPISpawn(int argc, char *argv[])
{
   const bool debug = getenv("JIT_DBG");
   dbg("[MPISpawn] %s", debug?"debug":"");
   MPI_Init(&argc, &argv);
   MPI_Comm parent;
   MPI_Comm_get_parent(&parent);
   // debug mode where 'jit' has been launched directly
   if (debug && parent == MPI_COMM_NULL)
   {
      const char *argv[3] = {"uname", "-a", nullptr};
      assert(unsetenv("JIT_DBG")==0);
      mfem::jit::System(const_cast<char**>(argv));
      MPI_Finalize();
      return EXIT_SUCCESS;
   }
   if (parent == MPI_COMM_NULL)
   {
      std::cerr << "Error: Should have been spawned by another MPI process!"
                << std::endl;
      return EXIT_FAILURE;
   }
   // MPI child
   dbg("\033[32m[MPI child] I have been spawned by MPI processes. argc:%d\n",
       argc-2);
   const bool dbg = (argc > 1) && (argv[1][0] == 0x31);
   if (dbg)
   {
      for (int k=2; argv[k]; k++) { dbg("\033[32m%s ", argv[k]); }
   }
   {
      int length;
      MPI_Bcast(&length, 1, MPI_INT, 0, parent);
      //dbg("\033[32m[MPI child] length:%d,\n", length);

      constexpr size_t SZ = 4096;
      char command[SZ];
      MPI_Bcast(&command, length, MPI_CHAR, 0, parent);

      dbg("\033[32m[MPI child] command: '%s'\n", command);

      // Launch here the command, even if it should be done in the other process
      system(command);

      int status = EXIT_SUCCESS;
      MPI_Bcast(&status, 1, MPI_INT, MPI_ROOT, parent);
   }
   //while (true);
   MPI_Finalize();
   return EXIT_SUCCESS;
}

// *****************************************************************************
int ProcessFork(int argc, char *argv[])
{
   const bool debug = getenv("JIT_DBG");
   dbg("[ProcessFork] %s", debug?"debug":"");
   const pid_t child_pid = fork();
   //dbg("child_pid: %d", child_pid);

   if (child_pid != 0) // MPI parent stuff
   {
      int status;
      MPISpawn(argc, argv);
      dbg("Waiting child %d",child_pid);
      waitpid(child_pid, &status, 0);
      return EXIT_SUCCESS;
   }
   if (child_pid != 0 && !debug) { return MPISpawn(argc, argv); }
   if (debug) { return EXIT_SUCCESS; }

   // Forked child, child_pid == 0
   const pid_t forked = getpid();
   dbg("Forked pid: %d", forked);
   dbg("Waiting for some compilation to do\n");
   while (true);
   return EXIT_SUCCESS;
}

#ifdef MFEM_INCLUDE_MAIN
// *****************************************************************************
int main(int argc, char *argv[]) { return ProcessFork(argc, argv); }
#endif // MFEM_INCLUDE_MAIN

#endif // MFEM_USE_MPI
