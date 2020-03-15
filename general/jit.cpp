// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"
#include "error.hpp"

#define dbg(...) { printf("\033[33m"); \
                   printf(__VA_ARGS__);  \
                   printf(" \n\033[m");fflush(0); }


#ifndef MFEM_USE_MPI

namespace mfem
{

namespace jit
{

int System(int argc =1, char *argv[] =nullptr)
{
   std::string command(argv[1]);
   for (int k=2; argv[k]; k++)
   {
      command.append(" ");
      command.append(argv[k]);
   }
   const char * command_c_str = command.c_str();
   const bool argdbg = (argc > 1) && (argv[0][0] == 0x31);
   if (argdbg) { dbg("%s", command_c_str); }
   const int return_value = system(command_c_str);
   return return_value;
}

} // namespace jit

} // namespace mfem

#else

#include <mpi.h>
#include <cassert>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>

#include "communication.hpp"

using namespace std;

namespace mfem
{

namespace jit
{

// *****************************************************************************
#define MFEM_JIT_RUN_COMPILATION 0x12345678

// *****************************************************************************
int System(int argc =1, char *argv[] = MPI_ARGV_NULL)
{
   const bool debug = argc ==1;
   assert(!debug);
   constexpr size_t size = 4096;
   char command[size];
   if (snprintf(command, size,  "%s/../jit", MFEM_INSTALL_DIR)<0)
   { return EXIT_FAILURE; }
   dbg("command: %s", command);
   MPI_Info info = MPI_INFO_NULL;
   const int root = 0;
   MPI_Comm comm = MPI_COMM_WORLD, intercomm = MPI_COMM_NULL;
   int errcode = 0;
   int spawned =
      MPI_Comm_spawn(command, argv, 1, info, root, comm, &intercomm, &errcode);
   if (spawned != 0 || errcode !=0) { return EXIT_FAILURE; }
   dbg("MPI_Bcast");
   int status = EXIT_FAILURE;
   MPI_Bcast(&status, 1, MPI_INT, MPI_ROOT, intercomm);
   // Wait for return
   MPI_Bcast(&status, 1, MPI_INT, 0, intercomm);
   assert(status == EXIT_SUCCESS);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Comm_free(&intercomm);
   dbg("done");
   return EXIT_SUCCESS;
}

} // namespace jit

} // namespace mfem

// *****************************************************************************
int MPISpawn(int argc, char *argv[], int *cookie)
{
   const bool debug = argc == 1;
   dbg("[MPISpawn] %s", debug?"debug":"");
   MPI_Init(&argc, &argv);
   MPI_Comm parent = MPI_COMM_NULL;
   MPI_Comm_get_parent(&parent);

   // debug mode where 'jit' has been launched directly, without any arguments
   if (debug && parent == MPI_COMM_NULL)
   {
      constexpr int argc = 4;
      const char *argv[argc] = {"1", "uname", "-a", nullptr};
      mfem::jit::System(argc, const_cast<char**>(argv));
      MPI_Finalize();
      return EXIT_SUCCESS;
   }

   // This case should not happen
   if (parent == MPI_COMM_NULL) { return EXIT_FAILURE; }

   // MPI child
   dbg("\033[32m[MPI child] Waiting initial Bcast...");

   int status = EXIT_SUCCESS;
   MPI_Bcast(&status, 1, MPI_INT, 0, parent);
   assert(status == EXIT_FAILURE);

   dbg("\033[32m[MPI child] Now telling compile process to work!");
   for (*cookie  = MFEM_JIT_RUN_COMPILATION;
        *cookie == MFEM_JIT_RUN_COMPILATION;
        usleep(100)) {}
   dbg("\033[32m[MPI child] done");
   status = *cookie;
   MPI_Bcast(&status, 1, MPI_INT, MPI_ROOT, parent);
   MPI_Finalize();
   return EXIT_SUCCESS;
}

// *****************************************************************************
int ProcessFork(int argc, char *argv[])
{
   int *shared_return_value = nullptr;
   const bool debug = argc == 1;
   dbg("[ProcessFork] %s", debug?"debug":"");

   if (!debug)
   {
      constexpr int size = sizeof(int);
      constexpr int prot = PROT_READ | PROT_WRITE;
      constexpr int flag = MAP_SHARED | MAP_ANONYMOUS;
      shared_return_value = static_cast<int*>(mmap(0, size, prot, flag, -1, 0));
      *shared_return_value = 0;
   }

   const pid_t child_pid = fork();

   // MPI parent stuff
   if (child_pid != 0 ) { return MPISpawn(argc, argv, shared_return_value); }

   // Thread child stuff
   if (debug) { return EXIT_SUCCESS; }

   std::string command(argv[2]);
   for (int k=3; argv[k]; k++)
   {
      command.append(" ");
      command.append(argv[k]);
   }
   const char * command_c_str = command.c_str();
   const bool argdbg = (argc > 1) && (argv[1][0] == 0x31);
   if (argdbg) { dbg("\033[31m%s", command_c_str); }
   while (*shared_return_value != MFEM_JIT_RUN_COMPILATION) { usleep(1000); }
   const int return_value = system(command_c_str);
   *shared_return_value = return_value == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
   return return_value;
}

#endif // MFEM_USE_MPI

#ifdef MFEM_INCLUDE_MAIN
// *****************************************************************************
int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   return ProcessFork(argc, argv);
#else
   return 0;
#endif
}
#endif // MFEM_INCLUDE_MAIN
