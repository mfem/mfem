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

#ifdef MFEM_USE_JIT

#include <cassert> // for assert
#include <cstdlib> // for exit, system

#include <chrono>
#include <fstream>
#include <memory>
#include <regex>
#include <string>
#include <thread>


#include <sys/wait.h> // for waitpid
#include <sys/mman.h> // for memory map
#include <signal.h> // for signals
#include <dlfcn.h> // for dlopen/dlsym, not available on Windows

#include <unistd.h> // for fork

#include "../communication.hpp" // will pull mpi.h
#include "../globals.hpp"

#include "jit.hpp"

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 206
#include "../debug.hpp"

namespace mfem
{

namespace internal
{

static constexpr int SYMBOL_SIZE = 1+16+1;
static constexpr int DLOPEN_MODE = RTLD_NOW | RTLD_LOCAL;
static constexpr char ACK = ~0, SYSTEM = 0x55, EXIT = 0xAA;

/// 64 bits hash to char* function
static void Hash64(const size_t h, char *str, const char *ext = "")
{
   std::stringstream ss;
   ss << 'k' << std::setfill('0')
      << std::setw(16) << std::hex << (h|0) << std::dec << ext;
   memcpy(str, ss.str().c_str(), SYMBOL_SIZE + strlen(ext));
}

struct Cxx
{
   pid_t pid; // of child process
   char *ack, *cmd; // shared memory
   uintptr_t size; // of the memory mapping

   template <typename Op> static void Ack(const int xx)
   {
      while (Op()(*Cxx::Ack(), xx))
      { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
   }
   static void AckEQ(const int xx = ACK) { Ack<std::equal_to<int>>(xx); }
   static void AckNE(const int xx = ACK) { Ack<std::not_equal_to<int>>(xx); }

   // wait until code has been flushed
   static void Send(const int xx) { Cxx::AckNE(*Cxx::Ack() = xx); }

   static Cxx cxx_singleton;
   Cxx(Cxx const&) = delete;
   void operator=(Cxx const&) = delete;

   static Cxx& Get() { return cxx_singleton; }

   static pid_t& Pid() { return Get().pid; }
   static char* Ack() { return Get().ack; }
   static char* Cmd() { return Get().cmd; }
   static uintptr_t Size() { return Get().size; }
   static void Init()
   {
      const int prot = PROT_READ | PROT_WRITE;
      const int flags = MAP_SHARED | MAP_ANONYMOUS;
      Get().size = (uintptr_t) sysconf(_SC_PAGE_SIZE);
      Get().ack = (char*) ::mmap(nullptr, Get().size, prot, flags, -1, 0);
      Get().cmd = Get().ack + 1;
      *Get().ack = ACK; // initialize state
      ::signal(SIGINT,  Handler); // interrupt
      ::signal(SIGQUIT, Handler); // quit
      ::signal(SIGABRT, Handler); // abort
      ::signal(SIGKILL, Handler); // kill
      ::signal(SIGFPE,  Handler); // floating point exception
      ::signal(SIGTERM, Handler); // software termination from kill
   }

   static void Handler(int /*signum*/)
   {
      ::kill(Cxx::Pid(), SIGTERM);
      while ((Cxx::Pid() = ::wait(nullptr)) > 0);
      _exit(~0);
   }
};
Cxx Cxx::cxx_singleton {}; // Initialize the unique global Cxx context.

/// Return the MPI rank in MPI_COMM_WORLD.
static int Rank()
{
   int world_rank = 0;
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized()) { world_rank = Mpi::WorldRank(); }
#endif
   return world_rank;
}

/// Return true if JIT has been initialized.
static bool IsInitialized()
{
#ifndef MFEM_USE_MPI
   return false;
#else
   return Mpi::IsInitialized();
#endif
}

/// Return true if the rank in MPI_COMM_WORLD is zero.
static bool Root() { return Rank() == 0; }

/// Do a MPI barrier if MPI has been initialized.
static void Sync(bool status)
{
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized())
   {
      MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
      assert(status == 0);
   }
#endif
}

/// Ask the JIT process to launch a system call.
static int System(const char *argv[])
{
   assert(Root());
   auto CreateCommandLine = [](const char *argv[])
   {
      auto argn = [&](const char *v[], int c = 0)
      { while (v[c]) { c += 1; } return c; };
      const int argc = argn(argv);
      assert(argc > 0);
      std::string cmd(argv[0]);
      for (int k = 1; k < argc && argv[k]; k++) { cmd += " "; cmd += argv[k]; }
      return cmd;
   };
   const std::string cmd_str = CreateCommandLine(argv);
   const char *cmd = cmd_str.c_str();

   dbg(cmd);
   if (!IsInitialized()) { return std::system(cmd); }

   // write the command in shared mem
   assert(std::strlen(cmd) < (Cxx::Size() - 2)); // skip ack and avoid zero
   std::memcpy(Cxx::Cmd(), cmd, std::strlen(cmd) + 1);
   Cxx::Send(SYSTEM);
   Cxx::AckNE(); // wait for the child to acknowledge
   return EXIT_SUCCESS;
}

} // namespace internal

using namespace internal;

int Jit::Init(int *argc, char ***argv)
{
   Cxx::Init();

   if ((Cxx::Pid() = fork()) != 0) // parent
   {
#ifdef MFEM_USE_MPI
      ::MPI_Init(argc, argv);
#else
      MFEM_CONTRACT_VAR(argc);
      MFEM_CONTRACT_VAR(argv);
#endif
      *Cxx::Ack() = Rank(); // inform the child about the rank
      Cxx::AckNE(); // wait for the child to acknowledge
   }
   else // JIT compiler child process
   {
      assert(*Cxx::Ack() == ACK); // Child init error
      Cxx::AckEQ(); // wait for parent's rank

      const int rank = *Cxx::Ack();
      *Cxx::Ack() = ACK; // now acknowledge back
      //dbg("[thd:%d] entering",rank);

      auto work = [&]() // looping working thread
      {
         while (true)
         {
            Cxx::AckEQ(); // waiting for somebody to wake us...
            if (*Cxx::Ack() == SYSTEM)
            {
               const int status = std::system(Cxx::Get().cmd);
               if (status != EXIT_SUCCESS) { assert(false); return; }
            }
            if (*Cxx::Ack() == EXIT) { return;}
            *Cxx::Ack() = ACK; // send back ACK
         }
      };

      // only root is kept for work
      if (rank == 0) { std::thread (work).join(); }
      //dbg("[thd:%d] EXIT_SUCCESS",rank);
      std::exit(EXIT_SUCCESS);
   }
   return EXIT_SUCCESS;
}

void Jit::Finalize()
{
   dbg();
   assert(*Cxx::Ack() == ACK); // Parent finalize error!
   if (Root())
   {
      int status;
      dbg("[mpi:0] send thd:0 exit");
      Cxx::Send(EXIT);
      // waits for any child process in the process group of the caller
      ::waitpid(0, &status, WUNTRACED | WCONTINUED);
      assert(status == 0); // Error with JIT compiler thread!
   }
   // release shared memory
   if (::munmap(Cxx::Ack(), Cxx::Size()) != 0) { assert(false); }
}

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
#define MFEM_JIT_AR_LOAD_POSTFIX  MFEM_JIT_LINKER_OPTION "--no-whole-archive"
#else
#define MFEM_JIT_AR_LOAD_PREFIX  "-all_load"
#define MFEM_JIT_AR_LOAD_POSTFIX  ""
#define MFEM_JIT_INSTALL_BACKUP  ""
#endif

#define MFEM_JIT_BEG_LOAD MFEM_JIT_AR_LOAD_PREFIX
#define MFEM_JIT_END_LOAD MFEM_JIT_AR_LOAD_POSTFIX

static constexpr char LIB_AR[] = "libmjit.a";
static constexpr char LIB_SO[] = "./libmjit.so";

void Jit::AROpen(void* &handle)
{
   if (!std::fstream(LIB_AR)) { return; }

   int status = EXIT_SUCCESS;
   if (Root())
   {
      dbg("%s => %s", LIB_AR, LIB_SO);
      const char *argv_so[] =
      {
         MFEM_JIT_CXX,
#undef MFEM_USE_SANITIZER
#ifdef MFEM_USE_SANITIZER
         "-fsanitize=address",
#endif // MFEM_USE_SANITIZER
         "-shared", "-o", LIB_SO,
         MFEM_JIT_BEG_LOAD, LIB_AR, MFEM_JIT_END_LOAD,
         MFEM_JIT_LINKER_OPTION "-rpath,.",
         nullptr
      };
      status = System(argv_so);
   }
   Sync(status);
   handle = ::dlopen(LIB_SO, DLOPEN_MODE);
   assert(handle);
}

int Jit::Compile(const uint64_t hash, const char *src, const char *symbol,
                 void *&handle)
{
   int status = EXIT_SUCCESS;
   char cc[SYMBOL_SIZE+3], co[SYMBOL_SIZE+3], so[SYMBOL_SIZE+3];
   Hash64(hash, cc, ".cc");
   Hash64(hash, co, ".co");
   Hash64(hash, so, ".so");

   auto Compile = [&]()
   {
      std::regex reg("\\s+");
      std::vector<const char*> argv;
      argv.push_back(MFEM_JIT_CXX);

      // tokenize MFEM_JIT_BUILD_FLAGS
      std::string flags(MFEM_JIT_BUILD_FLAGS);
      for (auto &a :
           std::vector<std::string>(
              std::sregex_token_iterator{begin(flags), end(flags), reg, -1},
              std::sregex_token_iterator{}))
      {
         using uptr = std::unique_ptr<char, decltype(&std::free)>;
         uptr *t_copy = new uptr(strdup(a.data()), &std::free);
         argv.push_back(t_copy->get());
      }

      //argv.push_back(MFEM_JIT_COMPILER_OPTION "-Wno-unused-variable");
      argv.push_back("-fPIC");
      argv.push_back("-c");

      // avoid redefinition, as with nvcc, the option -x cu is already present
#ifdef MFEM_USE_CUDA
      argv.push_back(MFEM_JIT_DEVICE_CODE);
#else
      argv.push_back(MFEM_JIT_COMPILER_OPTION "-pipe");
#endif // MFEM_USE_CUDA

      argv.push_back("-o"); argv.push_back(co); // output
      argv.push_back(cc); // input
      argv.push_back(nullptr);

      // write kernel source into cc file
      std::ofstream input_src_file(cc);
      assert(input_src_file.good());
      input_src_file << src;
      input_src_file.close();

      // Compilation cc => co
      if (System(argv.data())) { return EXIT_FAILURE; }
      std::remove(cc);

      // Update archive
      const char *argv_ar[] = { "ar", "-rv", LIB_AR, co, nullptr };
      if (System(argv_ar)) { return EXIT_FAILURE; }
      std::remove(co);

      // Create shared library
      const char *argv_so[] =
      {
         MFEM_JIT_CXX,  "-shared", "-o", so,
         MFEM_JIT_BEG_LOAD, LIB_AR, MFEM_JIT_END_LOAD, nullptr
      };
      if (System(argv_so)) { return EXIT_FAILURE; }

      // Install shared library
      const char *install[] =
      { "install", "-v", MFEM_JIT_INSTALL_BACKUP, so, LIB_SO, nullptr };
      if (System(install)) { return EXIT_FAILURE; }
      return EXIT_SUCCESS;
   }; // Compile

   if (Root()) { status = Compile(); }
   Sync(status);

   if (!handle) { handle = ::dlopen(LIB_SO, DLOPEN_MODE); }
   else
   {
      // we had a handle, but no symbol, use the newest
      handle = ::dlopen(so, DLOPEN_MODE);
   }

   Sync(EXIT_SUCCESS);
   std::remove(so); // remove the temporary shared lib after use
   assert(handle);

   dbg(symbol);
   assert(::dlsym(handle, symbol));

   return EXIT_SUCCESS;
}

void* Jit::DLOpen(const char *path) { return ::dlopen(path, DLOPEN_MODE); }

void* Jit::DlSym(void *handle, const char *sym) { return ::dlsym(handle, sym); }

} // namespace mfem

#endif // MFEM_USE_JIT
