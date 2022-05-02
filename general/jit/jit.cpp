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

#include <cassert>
#include <cstdlib> // for exit, system

#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include <dlfcn.h> // for dlopen/dlsym, not available on Windows
#include <sys/mman.h> // for memory map
#include <signal.h> // for signals
#include <unistd.h> // for fork
#include <sys/wait.h> // for waitpid

#include "../communication.hpp" // will pull mpi.h
#include "../globals.hpp" // needed when !MFEM_USE_MPI
#include "../error.hpp" // for MFEM_CONTRACT_VAR

#include "jit.hpp"

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 206
#include "../debug.hpp"

namespace mfem
{

namespace internal
{

static constexpr int SYMBOL_SIZE = 1+16+1; // k + hash + 0

static constexpr char LIB_AR[] = "libmjit.a";
static constexpr char LIB_SO[] = "./libmjit.so";

static constexpr int DLOPEN_MODE = RTLD_NOW | RTLD_LOCAL;

/// 64 bits hash to char* function
static void Hash64(const size_t h, char *str, const char *ext = "")
{
   std::stringstream ss;
   ss << 'k' << std::setfill('0')
      << std::setw(16) << std::hex << (h|0) << std::dec << ext;
   memcpy(str, ss.str().c_str(), SYMBOL_SIZE + strlen(ext));
}

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
static void Sync(bool status = EXIT_SUCCESS)
{
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized())
   {
      MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
      assert(status == EXIT_SUCCESS); // synchronization error
   }
#endif
}

struct Cxx
{
   pid_t pid; // of child process
   int *s_int; // should be large enough to store an MPI rank
   char *s_chr; // shared memory to store the command for the system call
   uintptr_t size; // of the shared memory

   struct Command
   {
      Command() {}
      std::ostringstream cmd {};
      Command& operator<<(const char *c) { cmd << c << ' '; return *this; }
      operator const char *()
      {
         std::ostringstream cmd_mv = std::move(cmd);
         static thread_local std::string sl_cmd;
         sl_cmd = cmd_mv.str();
         cmd.clear(); // flush for next command
         return sl_cmd.c_str();
      }
   } command;

   template <typename T> static void Ack(const int xx)
   {
      while (T()(*Mem(), xx))
      { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
   }
   static constexpr int ACK = ~0, SYSTEM = 0x55555555, EXIT = 0xAAAAAAAA;
   static void AckEQ(const int xx = ACK) { Ack<std::equal_to<int>>(xx); }
   static void AckNE(const int xx = ACK) { Ack<std::not_equal_to<int>>(xx); }
   // will block until code has been acknowledged
   static void Send(const int xx) { AckNE(*Mem() = xx); }

   static int Read() { return *Mem(); }
   static void Acknowledge() { Write(ACK); }
   static void Write(const int mem) { *Mem() = mem; }
   static void Wait(const bool eq = true) { eq ? AckEQ() : AckNE(); }

   static bool IsSystem() { return Read() == SYSTEM; }
   static bool IsExit() { return Read() == EXIT; }
   static bool IsAck() { return Read() == ACK; }

   /// Ask the JIT process to launch a system call.
   static int System()
   {
      const char *command = Command();
      dbg(command);
      assert(Root());
      // In serial mode, just call std::system
      if (!IsInitialized()) { return std::system(command); }
      // Otherwise, write the command to the cxx process
      assert((1 + std::strlen(command)) < Size());
      std::memcpy(Cmd(), command, std::strlen(command) + 1);
      Send(SYSTEM); // call std::system through the cxx process
      AckNE(); // wait for the acknowledgment
      return EXIT_SUCCESS;
   }

   static Cxx cxx_singleton;
   static Cxx& Get() { return cxx_singleton; }

   static pid_t& Pid() { return Get().pid; }
   static int* Mem() { return Get().s_int; }
   static char* Cmd() { return Get().s_chr; }
   static uintptr_t Size() { return Get().size; }
   static Command& Command() { return Get().command; }

   static void Handler(int signum)
   {
      dbg(signum);
      MFEM_CONTRACT_VAR(signum);
      ::kill(Pid(), SIGKILL);
      while ((Pid() = ::wait(nullptr)) > 0);
      std::exit(EXIT_FAILURE);
   }

   static void Init()
   {
      const int prot = PROT_READ | PROT_WRITE;
      const int flags = MAP_SHARED | MAP_ANONYMOUS;
      Get().size = (uintptr_t) sysconf(_SC_PAGE_SIZE);
      Get().s_int = (int*) ::mmap(nullptr, sizeof(int), prot, flags, -1, 0);
      Get().s_chr = (char*) ::mmap(nullptr, Get().size, prot, flags, -1, 0);
      Write(ACK); // initialize state
      ::signal(SIGINT,  Handler); // interrupt
      ::signal(SIGQUIT, Handler); // quit
      ::signal(SIGABRT, Handler); // abort
      ::signal(SIGKILL, Handler); // kill
      ::signal(SIGFPE,  Handler); // floating point exception
   }

   static void Finalize()
   {
      assert(IsAck()); // Finalize error
      int status;
      Send(EXIT);
      // wait for any child process in the process group of the caller
      ::waitpid(0, &status, WUNTRACED | WCONTINUED);
      assert(status == 0); // Error with the compiler thread!
      if (::munmap(Cmd(), Size()) != 0 || // release shared memory
          ::munmap(Mem(), sizeof(int)) != 0)
      { assert(false); /* finalize memory error! */ }
   }
};
Cxx Cxx::cxx_singleton {}; // Initialize the unique Cxx context.

} // namespace internal

using namespace internal;

int Jit::Init(int *argc, char ***argv)
{
   if (Root()) { Cxx::Init(); }

   if ((Cxx::Pid() = fork()) != 0) // parent
   {
#ifdef MFEM_USE_MPI
      ::MPI_Init(argc, argv);
#else
      MFEM_CONTRACT_VAR(argc);
      MFEM_CONTRACT_VAR(argv);
#endif
      Cxx::Write(Rank()); // inform the child about the rank
      Cxx::Wait(false); // wait for the child to acknowledge
   }
   else // JIT compiler child process
   {
      assert(Cxx::IsAck()); // Child process initialization error
      Cxx::Wait(); // wait for parent's rank
      const int rank = Cxx::Read();
      Cxx::Acknowledge();
      if (rank == 0) // only root is kept for work
      {
         std::thread([&]()
         {
            while (true)
            {
               Cxx::Wait(); // waiting for the root to wake us
               if (Cxx::IsSystem()) { if (std::system(Cxx::Cmd())) { return; } }
               if (Cxx::IsExit()) { return;}
               Cxx::Acknowledge();
            }
         }).join();
      }
      std::exit(EXIT_SUCCESS);
   }
   return EXIT_SUCCESS;
}

void Jit::Finalize() { if (Root()) { Cxx::Finalize(); } }

#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
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
#define MFEM_JIT_INSTALL_BACKUP  ""
#define MFEM_JIT_AR_LOAD_PREFIX  "-all_load"
#define MFEM_JIT_AR_LOAD_POSTFIX  ""
#endif

#define MFEM_JIT_BEG_LOAD MFEM_JIT_AR_LOAD_PREFIX
#define MFEM_JIT_END_LOAD MFEM_JIT_AR_LOAD_POSTFIX

void Jit::AROpen(void* &handle)
{
   if (!std::fstream(LIB_AR)) { return; }

   int status = EXIT_SUCCESS;
   if (Root())
   {
      dbg("%s => %s", LIB_AR, LIB_SO);
      Cxx::Command()
            << MFEM_JIT_CXX << "-shared" << "-o" << LIB_SO
            << MFEM_JIT_BEG_LOAD << LIB_AR << MFEM_JIT_END_LOAD
            << MFEM_JIT_LINKER_OPTION "-rpath,.";
      status = Cxx::System();
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
      // write kernel source into cc file
      std::ofstream input_src_file(cc);
      assert(input_src_file.good());
      input_src_file << src;
      input_src_file.close();

      // Compilation cc => co
      Cxx::Command()
            << MFEM_JIT_CXX << MFEM_JIT_BUILD_FLAGS
            << MFEM_JIT_COMPILER_OPTION "-fPIC"
            << MFEM_JIT_COMPILER_OPTION "-pipe"
            << MFEM_JIT_COMPILER_OPTION "-Wno-unused-variable"
            << MFEM_JIT_DEVICE_CODE
            << "-c" << "-o" << co << cc;
      if (Cxx::System()) { return EXIT_FAILURE; }
      std::remove(cc);

      // Update archive
      Cxx::Command() << "ar -rv" << LIB_AR << co;
      if (Cxx::System()) { return EXIT_FAILURE; }
      std::remove(co);

      // Create shared library
      Cxx::Command()
            << MFEM_JIT_CXX << "-shared" << "-o" << so
            << MFEM_JIT_BEG_LOAD << LIB_AR << MFEM_JIT_END_LOAD;
      if (Cxx::System()) { return EXIT_FAILURE; }

      // Install shared library
      Cxx::Command() << "install" << "-v" << MFEM_JIT_INSTALL_BACKUP
                     << so << LIB_SO;
      if (Cxx::System()) { return EXIT_FAILURE; }
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
   Sync();
   if (!handle)
   {
      fprintf(stderr, "%s\n", dlerror());
      exit(EXIT_FAILURE);
   }

   Sync();
   std::remove(so); // remove the temporary shared lib after use

   assert(::dlsym(handle, symbol)); // no symbol found
   return EXIT_SUCCESS;
}

void* Jit::DLOpen() { return ::dlopen(LIB_SO, DLOPEN_MODE); }

void* Jit::DlSym(void *handle, const char *sym) { return ::dlsym(handle, sym); }

} // namespace mfem

#endif // MFEM_USE_JIT
