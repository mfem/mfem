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

using std::list;
using std::regex;
using std::vector;
using std::string;
using std::istream;
using std::ostream;

#include "../../config/config.hpp"

#include "../error.hpp"
#include "../globals.hpp"

#include "jit.hpp"
#include "tools.hpp"
#include "compile.hpp"

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 226
#include "../debug.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

namespace jit
{

/// \brief CloseAndWait
/// \param fd descriptor to be closed.
static inline void CloseAndWait(int fd)
{
   if (::close(fd) < 0) { ::perror(strerror(errno)); }
   // block parent process until any of its children has finished
   ::wait(nullptr);
}

/// In-memory compilation
static int CompileInMemory(const char *argv[],
                           const char *input_mem,
                           char *&output_mem, size_t &size)
{
   const int argc = argn(argv);
   if (argc < 1) { return EXIT_FAILURE; }

   // debug command line
   {
      string command(argv[0]);
      for (int k = 1; k < argc && argv[k]; k++)
      {
         command.append(" ");
         command.append(argv[k]);
      }
      const char *command_c_str = command.c_str();
      dbg(command_c_str);
   }

   // input, output and error pipes
   int ip[2], op[2], ep[2];
   constexpr size_t PIPE_READ = 0;
   constexpr size_t PIPE_WRITE = 1;

   if (::pipe(ip)<0 || ::pipe(op)<0 || ::pipe(ep)<0)
   {
      return ::perror(strerror(errno)), EXIT_FAILURE;
   }

   if (fork() == 0) // Child process which calls the compiler
   {
      ::close(STDIN_FILENO);    // closing stdin
      ::dup2(ip[PIPE_READ],0);  // replacing stdin with pipe read
      ::close(ip[PIPE_READ]);   // close reading end
      ::close(ip[PIPE_WRITE]);  // no longer needed

      ::dup2(op[PIPE_WRITE],1); // send stdout to the output pipe
      ::close(op[PIPE_READ]);   // no longer needed
      ::close(op[PIPE_WRITE]);  // no longer needed

      ::dup2(ep[PIPE_WRITE],2); // send stderr to the error pipe
      ::close(ep[PIPE_READ]);   // no longer needed
      ::close(ep[PIPE_WRITE]);  // no longer needed

      ::execvp(argv[0], const_cast<char* const*>(argv));
      // only here if an error has occurred
      return ::perror(strerror(errno)), EXIT_FAILURE;
   }

   // Parent process

   // close unused sides of the pipes
   ::close(op[PIPE_WRITE]);
   ::close(ep[PIPE_WRITE]);

   // write all the source present in memory to the 'input' pipe
   const size_t nbytes = std::strlen(input_mem);
   ::write(ip[PIPE_WRITE], input_mem, nbytes);
   ::close(ip[PIPE_WRITE]);

   char buffer[1<<16];
   constexpr size_t SIZE = sizeof(buffer);
   size_t nr, ne = 0, osize = 0; // number of read & error bytes

   // Scan error pipe with timeout
   {
      fd_set set;
      struct timeval timeout {1, 0}; // one second
      FD_ZERO(&set); // clear the set
      FD_SET(ep[PIPE_READ], &set); // add the descriptor we need a timeout on
      const int rv = ::select(ep[PIPE_READ]+1, &set, NULL, NULL, &timeout);
      if (rv == -1) { return perror(strerror(errno)), EXIT_FAILURE; }
      else if (rv == 0) { dbg("No error found!"); }
      else
      {
         dbg("Error data is available now!");
         while ((nr = ::read(ep[PIPE_READ], buffer, SIZE)) > 0)
         {
            ::write(STDOUT_FILENO, buffer, nr);
            ne += nr;
         }
         ::close(ep[PIPE_READ]);
         if (ne > 0)
         {
            CloseAndWait(op[PIPE_READ]);
            return perror("Compilation error!"), EXIT_FAILURE;
         }
      }
      ::close(ep[PIPE_READ]);
   }

   // then get the object from output pipe
   size = 0;
   osize = SIZE;
   output_mem = (char*) ::realloc(nullptr, SIZE);
   while ((nr = ::read(op[PIPE_READ], buffer, SIZE)) > 0)
   {
      size += nr;
      if (size > osize)
      {
         dbg("omem.realloc(%d + %d);", osize, SIZE);
         output_mem = (char*) ::realloc(output_mem, osize + SIZE);
         osize += SIZE;
      }
      ::memcpy(output_mem + size - nr, buffer, nr);
   }
   dbg("object size: %d", size);
   CloseAndWait(op[PIPE_READ]);
   return EXIT_SUCCESS;
}

/// \brief Compile the source file with PIC flags, updating the cache library.
/// \param input_mem
/// \param cc
/// \param co
/// \param mfem_cxx
/// \param mfem_cxxflags
/// \param mfem_source_dir
/// \param mfem_install_dir
/// \param check_for_ar
/// \return EXIT_SUCCESS or EXIT_FAILURE
int Compile(const char *input_mem,
            char cc[MFEM_JIT_FILENAME_SIZE],
            char co[MFEM_JIT_FILENAME_SIZE],
            const char *mfem_cxx,
            const char *mfem_cxxflags,
            const char *mfem_source_dir,
            const char *mfem_install_dir,
            const bool check_for_ar)
{
#ifndef MFEM_USE_CUDA
#define MFEM_JIT_DEVICE_CODE ""
#define MFEM_JIT_COMPILER_OPTION
#define MFEM_JIT_LINKER_OPTION "-Wl,"
#else
   // Compile each input file into an relocatable device code object file
#define MFEM_JIT_DEVICE_CODE "--device-c"
#define MFEM_JIT_LINKER_OPTION "-Xlinker="
#define MFEM_JIT_COMPILER_OPTION "-Xcompiler="
#endif

#ifndef __APPLE__
#define MFEM_JIT_INSTALL_BACKUP  "--backup=none"
#define MFEM_JIT_AR_LOAD_PREFIX MFEM_JIT_LINKER_OPTION "--whole-archive"
#define MFEM_JIT_AR_LOAD_POSTFIX  MFEM_JIT_LINKER_OPTION "--no-whole-archive";
#else
#define MFEM_JIT_AR_LOAD_PREFIX  "-all_load"
#define MFEM_JIT_AR_LOAD_POSTFIX  ""
#define MFEM_JIT_INSTALL_BACKUP  ""
#endif

   constexpr const char *beg_load = MFEM_JIT_AR_LOAD_PREFIX;
   constexpr const char *end_load = MFEM_JIT_AR_LOAD_POSTFIX;

   constexpr int PM = PATH_MAX;
   constexpr const char *fPIC = MFEM_JIT_COMPILER_OPTION "-fPIC";
   constexpr const char *system = MFEM_JIT_COMMAND_LINE_OPTION;
   constexpr const char *libar = "lib" MFEM_JIT_LIB_NAME ".a";
   constexpr const char *libso = "lib" MFEM_JIT_LIB_NAME ".so";

   // If there is already a JIT archive, use it and create the lib_so
   if (check_for_ar && GetRuntimeVersion() == 0 && std::fstream(libar))
   {
      dbg();
      if (Root()) { dbg("Using JIT archive!"); }
      const char *argv_so[] =
      {
         system, mfem_cxx, "-shared", "-o", libso, beg_load, libar, end_load,
         MFEM_JIT_LINKER_OPTION "-rpath,.", nullptr
      };
      if (mfem::jit::System(argv_so)) { return EXIT_FAILURE; }
      delete input_mem;
      return EXIT_SUCCESS;
   }

   // MFEM source path, include path & lib_so_v
   constexpr bool increment = true;
   const int version = GetRuntimeVersion(increment);
   char libso_n[PM], Imsrc[PM], Iminc[PM];
   if (snprintf(Imsrc, PM, "-I%s ", mfem_source_dir) < 0) { return EXIT_FAILURE; }
   if (snprintf(Iminc, PM, "-I%s/include ", mfem_install_dir) < 0) { return EXIT_FAILURE; }
   if (snprintf(libso_n, PM, "lib%s.so.%d", MFEM_JIT_LIB_NAME, version) < 0)
   { return EXIT_FAILURE; }

   // If there is an input_mem, do the compilation in memory
   if (input_mem)
   {
      dbg("Tokenizing MFEM_CXXFLAGS");
      regex reg("\\s+");
      string cxxflags(mfem_cxxflags);
      auto cxxargv =
         vector<std::string>(
            std::sregex_token_iterator{begin(cxxflags), end(cxxflags), reg, -1},
            std::sregex_token_iterator{});
      //for (auto a : cxxargv) { mfem::out << a << std::endl; }

      vector<const char *> argv;
      argv.push_back(mfem_cxx);
      for (auto &a : cxxargv)
      {
         // instead of: argv.push_back(strdup(a.data()));
         using uptr = std::unique_ptr<char, decltype(&std::free)>;
         uptr *t_copy = new uptr(strdup(a.data()), &std::free);
         argv.push_back(t_copy->get());
      }

      argv.push_back(MFEM_JIT_COMPILER_OPTION "-Wno-unused-variable");
      argv.push_back(fPIC);
      argv.push_back("-c");
      argv.push_back(MFEM_JIT_COMPILER_OPTION "-pipe");
      // avoid redefinition, as with nvcc, the option -x cu is already present
#ifndef MFEM_USE_CUDA
      argv.push_back("-x");
      argv.push_back("c++");
#else
      argv.push_back(MFEM_JIT_DEVICE_CODE);
#endif // MFEM_USE_CUDA
      //argv.push_back("-I.");
      //argv.push_back(Imsrc);
      //argv.push_back(Iminc);
      argv.push_back("-o");
#ifdef __APPLE__
      argv.push_back("/dev/stdout"); // through output fork/pipes
#else
      argv.push_back(co);
#endif // __APPLE__
      argv.push_back("-");
      argv.push_back(nullptr);

      size_t size = 0;
      char *output_mem = nullptr;
      if (CompileInMemory(argv.data(), input_mem, output_mem, size))
      {
         dbg("InMemCompile error!");
         return EXIT_FAILURE;
      }
      delete input_mem;

#ifdef __APPLE__
      // Not possible to output directly in memory, saving on disk the object
      assert(output_mem != nullptr && size > 0);
      const mode_t mode = S_IRUSR | S_IWUSR;
      const int oflag = O_CREAT | O_RDWR | O_TRUNC;
      const int co_fd = ::open(co, oflag, mode);
      const size_t written = ::write(co_fd, output_mem, size);
      if (written != size) { return perror("!write object"), EXIT_FAILURE; }
      if (::close(co_fd) < 0) { return perror("!close object"), EXIT_FAILURE; }
      free(output_mem); // done with realloc
#endif // __APPLE__
   }
   else // !in_memory_compilation
   {
      const char *argv_co[] =
      {
         system, mfem_cxx, mfem_cxxflags, fPIC, "-c",
         MFEM_JIT_DEVICE_CODE, Imsrc, Iminc, "-o", co, cc,
         nullptr
      };
      if (mfem::jit::System(argv_co)) { return EXIT_FAILURE; }
      if (!getenv("MFEM_NUNLINK")) { ::unlink(cc); }
   }

   // Update archive
   const char *argv_ar[] = { system, "ar", "-rv", libar, co, nullptr };
   if (mfem::jit::System(argv_ar)) { return EXIT_FAILURE; }
   if (!getenv("MFEM_NUNLINK")) { ::unlink(co); }

   // Create shared library
   const char *argv_so[] = { system,
                             mfem_cxx, "-shared", "-o", libso,
                             beg_load, libar, end_load,
                             nullptr
                           };
   if (mfem::jit::System(argv_so)) { return EXIT_FAILURE; }

   // Install shared library
   const char *install[] = { system, "install",
                             MFEM_JIT_INSTALL_BACKUP, libso, libso_n,
                             nullptr
                           };
   if (mfem::jit::System(install)) { return EXIT_FAILURE; }
   return EXIT_SUCCESS;
}

} // namespace jit

} // namespace mfem

