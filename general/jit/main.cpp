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


/** @file main.cpp
 *  This file produces the @c mjit executable, which can:
 *     - pre-process MFEM source files,
 *     - spawn & fork in parallel to be able to @c system calls.
 *
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
 *
 */

#include <list>
#include <cmath>
#include <regex>
#include <string>
#include <vector>
#include <ciso646>
#include <cassert>
#include <fstream>
#include <climits>
#include <iostream>
#include <algorithm>

#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

using std::list;
using std::regex;
using std::vector;
using std::string;
using std::istream;
using std::ostream;

#include "../../config/config.hpp"

#include "../globals.hpp"

#include "parser.hpp"
#include "jit.hpp"

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 159
#include "../debug.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

namespace jit
{

static int help(char* argv[])
{
   std::cout << "MFEM mjit: ";
   std::cout << argv[0] << " [-ch] [-o output] input" << std::endl;
   return ~0;
}

static const char* strrnc(const char *s, const unsigned char c, int n =1)
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
static int ProcessFork(int argc, char *argv[])
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

} // namespace jit

} // namespace mfem


// The parallel implementation will spawn the =mjit= binary on one mpi rank to
// be able to run on one core and use MPI to broadcast the compilation output.

int main(const int argc, char* argv[])
{
   string input, output, file;

   if (argc <= 1) { return mfem::jit::help(argv); }

   for (int i = 1; i < argc; i++)
   {
      // -h lauches help
      if (argv[i] == string("-h")) { return mfem::jit::help(argv); }

      // Will launch ProcessFork in parallel mode, nothing otherwise
      if (argv[i] == string(MFEM_JIT_COMMAND_LINE_OPTION))
      {
#ifdef MFEM_USE_MPI
         dbg("Compilation requested, forking...");
         return mfem::jit::ProcessFork(argc, argv);
#else
         return 0;
#endif
      }
      // -o fills output
      if (argv[i] == string("-o"))
      {
         output = argv[i+=1];
         continue;
      }
      // should give input file
      const char* last_dot = mfem::jit::strrnc(argv[i],'.');
      const size_t ext_size = last_dot?strlen(last_dot):0;
      if (last_dot && ext_size>0)
      {
         assert(file.size()==0);
         file = input = argv[i];
      }
   }
   assert(!input.empty());
   const bool output_file = !output.empty();
   std::ifstream in(input.c_str(), std::ios::in | std::ios::binary);
   std::ofstream out(output.c_str(),
                     std::ios::out | std::ios::binary | std::ios::trunc);
   assert(!in.fail());
   assert(in.is_open());
   if (output_file) {assert(out.is_open());}
   ostream &mfem_out(std::cout);
   mfem::jit::context_t pp(in, output_file ? out : mfem_out, file);
   try { mfem::jit::preprocess(pp); }
   catch (mfem::jit::error_t err)
   {
      std::cerr << std::endl << err.file << ":" << err.line << ":"
                << " mpp error" << (err.msg?": ":"") << (err.msg?err.msg:"")
                << std::endl;
      remove(output.c_str());
      return ~0;
   }
   in.close();
   out.close();
   return 0;
}
