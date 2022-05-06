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
#include <signal.h> // for signals
#include <unistd.h> // for fork
#include <sys/wait.h> // for waitpid

#if !(defined(__linux__) || defined(__APPLE__))
#error mmap(2) implementation as defined in POSIX.1-2001 not supported.
#else
#include <sys/mman.h> // for memory map
#endif

#include "../communication.hpp" // pulls mpi.h
#include "../globals.hpp" // needed when !MFEM_USE_MPI
#include "../error.hpp" // for MFEM_CONTRACT_VAR

#include "jit.hpp"

namespace mfem
{

namespace jit_internal
{

static constexpr char LIB_AR[] = "libmjit.a";
static constexpr char LIB_SO[] = "./libmjit.so";
static constexpr int SYMBOL_SIZE = 1+16+1; // k + hash + 0
static constexpr int DLOPEN_MODE = RTLD_LAZY | RTLD_LOCAL;

// Return the MPI world rank
static int Rank()
{
   int world_rank = 0;
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized()) { world_rank = Mpi::WorldRank(); }
#endif
   return world_rank;
}

// Return true if MPI has been initialized
static bool IsInitialized()
{
#ifndef MFEM_USE_MPI
   return false;
#else
   return Mpi::IsInitialized();
#endif
}

// Return true if the rank in world rank is zero
static bool Root() { return Rank() == 0; }

// Do a MPI barrier and status reduction if MPI has been initialized
static void Sync(bool status = EXIT_SUCCESS)
{
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized())
   {
      MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
      assert(status == EXIT_SUCCESS); // synchronization error
   }
#else
   MFEM_CONTRACT_VAR(status);
#endif
}

struct Sys // System singleton object
{
   pid_t pid; // of child process
   int *s_ack; // shared status, should be large enough to store one MPI rank
   char *s_mem; // shared memory to store the command for the system call
   uintptr_t size; // of the shared memory

   struct Command
   {
      Command() {}
      std::ostringstream cmd {};
      Command& operator<<(const char *c) { cmd << c << ' '; return *this; }
      Command& operator<<(const std::string &s) { return *this << s.c_str(); }
      operator const char *()
      {
         std::ostringstream cmd_mv = std::move(cmd);
         static thread_local std::string sl_cmd;
         sl_cmd = cmd_mv.str();
         cmd.clear(); cmd.str(""); // flush for next command
         return sl_cmd.c_str();
      }
   } command;

   template <typename OP> static void Ack(const int xx)
   {
      while (OP()(*Ack(), xx))
      { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }
   }
   static constexpr int ACK = ~0, SYSTEM = 0x55555555, EXIT = 0xAAAAAAAA;
   static void AckEQ(const int xx = ACK) { Ack<std::equal_to<int>>(xx); }
   static void AckNE(const int xx = ACK) { Ack<std::not_equal_to<int>>(xx); }
   // will block until code has been acknowledged
   static void Send(const int xx) { AckNE(*Ack() = xx); }

   static int Read() { return *Ack(); }
   static void Acknowledge() { Write(ACK); }
   static void Write(const int xx) { *Ack() = xx; }
   static void Wait(const bool eq = true) { eq ? AckEQ() : AckNE(); }

   static bool IsSystem() { return Read() == SYSTEM; }
   static bool IsExit() { return Read() == EXIT; }
   static bool IsAck() { return Read() == ACK; }

   // Ask the parent to launch a system call
   static int System(const char *command = Command())
   {
      MFEM_WARNING("\033[33m" << command << "\033[m");
      assert(Root());
      // In serial mode, just call std::system
      if (!IsInitialized()) { return std::system(command); }
      // Otherwise, write the command to the child process
      assert((1 + std::strlen(command)) < Size());
      std::memcpy(Mem(), command, std::strlen(command) + 1);
      Send(SYSTEM); // call std::system through the child process
      AckNE();//Wait(false); // wait for the acknowledgment
      return EXIT_SUCCESS;
   }

   static Sys singleton;
   static Sys& Get() { return singleton; }

   static pid_t& Pid() { return Get().pid; }
   static int* Ack() { return Get().s_ack; }
   static char* Mem() { return Get().s_mem; }
   static uintptr_t Size() { return Get().size; }
   static Command& Command() { return Get().command; }

   static void Handler(int signum)
   {
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
      Get().s_ack = (int*) ::mmap(nullptr, sizeof(int), prot, flags, -1, 0);
      Get().s_mem = (char*) ::mmap(nullptr, Get().size, prot, flags, -1, 0);
      Write(ACK); // initialize state
      ::signal(SIGINT,  Handler); // interrupt
      ::signal(SIGQUIT, Handler); // quit
      ::signal(SIGABRT, Handler); // abort
      ::signal(SIGKILL, Handler); // kill
      ::signal(SIGFPE,  Handler); // floating point exception
   }

   static void Parent(int *argc, char ***argv)
   {
#ifdef MFEM_USE_MPI
      ::MPI_Init(argc, argv);
#else
      MFEM_CONTRACT_VAR(argc);
      MFEM_CONTRACT_VAR(argv);
#endif
      Sys::Write(Rank()); // inform the child about the rank
      Sys::Wait(false); // wait for the child to acknowledge
   }

   static void Child()
   {
      assert(IsAck()); // Child process initialization error
      Wait(); // wait for parent's rank
      const int rank = Read();
      Acknowledge();
      if (rank == 0) // only root is kept for work
      {
         std::thread([&]()
         {
            while (true)
            {
               Wait(); // waiting for the root to wake us
               if (IsSystem()) { if (std::system(Mem())) { return; } }
               if (IsExit()) { return;}
               Acknowledge();
            }
         }).join();
      }
      std::exit(EXIT_SUCCESS);
   }

   static void Finalize()
   {
      assert(IsAck()); // Finalize error
      int status;
      Send(EXIT);
      ::waitpid(0, &status, WUNTRACED | WCONTINUED); // wait for any child
      assert(status == 0); // Error with the compiler thread
      if (::munmap(Mem(), Size()) != 0 || // release shared memory
          ::munmap(Ack(), sizeof(int)) != 0)
      { assert(false); /* finalize memory error! */ }
   }

   struct Options
   {
      virtual std::string Device() { return ""; }
      virtual std::string Compiler() { return ""; }
      virtual std::string Linker() { return "-Wl,"; }
   };

   struct HostOptions: Options {};

   struct DeviceOptions: Options
   {
      std::string Device() override { return "--device-c"; }
      std::string Compiler() override { return "-Xcompiler="; }
      std::string Linker() override { return "-Xlinker="; }
   };

   struct Linux
   {
      virtual std::string Prefix() { return Xlinker() + "--whole-archive"; }
      virtual std::string Postfix() { return Xlinker() + "--no-whole-archive"; }
      virtual std::string Backup() { return "--backup=none"; }
   };

   struct Apple: public Linux
   {
      std::string Prefix() override { return "-all_load"; }
      std::string Postfix() override { return ""; }
      std::string Backup() override { return ""; }
   };

#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
   HostOptions opt;
#else
   DeviceOptions opt;
#endif

#ifdef __APPLE__
   Apple ar;
#else
   Linux ar;
#endif

   static std::string Xdevice() { return Get().opt.Device(); }
   static std::string Xlinker() { return Get().opt.Linker(); }
   static std::string Xcompiler() { return Get().opt.Compiler(); }
   static std::string ARprefix() { return Get().ar.Prefix();  }
   static std::string ARpostfix() { return Get().ar.Postfix(); }
   static std::string ARbackup() { return Get().ar.Backup(); }
};
Sys Sys::singleton {}; // Initialize the unique Sys context.

} // namespace internal

using namespace jit_internal;

void Jit::Init(int *argc, char ***argv)
{
   if (Root()) { Sys::Init(); }
   if ((Sys::Pid()=::fork()) != 0) { Sys::Parent(argc,argv); }
   else { Sys::Child(); }
}

void Jit::Finalize() { if (Root()) { Sys::Finalize(); } }

void* Jit::Lookup(const size_t hash, const char *src, const char *symbol)
{
   int status = EXIT_SUCCESS;
   void *handle = // try to open libmjit.so first
      std::fstream(LIB_SO) ? ::dlopen(LIB_SO, DLOPEN_MODE) : nullptr;

   // if libmjit.so was not found, try libmjit.a
   if (!handle && std::fstream(LIB_AR))
   {
      if (Root())
      {
         Sys::Command() << MFEM_JIT_CXX << "-shared" << "-o" << LIB_SO
                        << Sys::ARprefix() << LIB_AR << Sys::ARpostfix()
                        << Sys::Xlinker() + "-rpath,.";
         status = Sys::System();
      }
      Sync(status);
      handle = ::dlopen(LIB_SO, DLOPEN_MODE);
      assert(handle);
   }

   auto Compile = [&]()
   {
      char cc[SYMBOL_SIZE+3], co[SYMBOL_SIZE+3];

      // 64 bits hash to char* function
      auto Hash64 = [](const size_t h, char *str, const char *ext)
      {
         std::stringstream ss;
         ss << 'k' << std::setfill('0')
            << std::setw(16) << std::hex << (h|0) << std::dec << ext;
         memcpy(str, ss.str().c_str(), SYMBOL_SIZE + strlen(ext));
      };
      Hash64(hash, cc, ".cc");
      Hash64(hash, co, ".co");
      // write kernel source into cc file
      std::ofstream input_src_file(cc);
      assert(input_src_file.good());
      input_src_file << src;
      input_src_file.close();
      // Compilation: cc => co
      Sys::Command() << MFEM_JIT_CXX << MFEM_JIT_BUILD_FLAGS
                     << Sys::Xdevice()
                     << Sys::Xcompiler() + "-fPIC"
                     << Sys::Xcompiler() + "-pipe"
                     << Sys::Xcompiler() + "-Wno-unused-variable"
                     << "-c" << "-o" << co << cc;
      if (Sys::System()) { return EXIT_FAILURE; }
      std::remove(cc);
      // Update archive: ar += co
      Sys::Command() << "ar -rv" << LIB_AR << co;
      if (Sys::System()) { return EXIT_FAILURE; }
      std::remove(co);
      // Create shared library: new (ar + symbol), used afterward
      Sys::Command() << MFEM_JIT_CXX << "-shared" << "-o" << symbol
                     << Sys::ARprefix() << LIB_AR << Sys::ARpostfix();
      if (Sys::System()) { return EXIT_FAILURE; }
      // Update shared cache library: new (ar + symbol) => LIB_SO
      Sys::Command() << "install" << "-v" << Sys::ARbackup() << symbol << LIB_SO;
      if (Sys::System()) { return EXIT_FAILURE; }
      return EXIT_SUCCESS;
   }; // Compile

   auto Compilation = [&]()
   {
      if (Root()) { status = Compile(); } // only root rank does the compilation
      Sync(status); // all ranks verify the status
      handle = ::dlopen(symbol, DLOPEN_MODE); // opens (ar + symbol)
      Sync();
      if (!handle) { std::cerr << ::dlerror() << std::endl; }
      assert(handle);
   };

   // no caches => launch compilation
   if (!handle) { Compilation(); }
   assert(handle); // we have a handle

   void *kernel = ::dlsym(handle, symbol); // symbol lookup
   // no symbol => launch compilation & update kernel symbol
   if (!kernel) { Compilation(); kernel = ::dlsym(handle, symbol); }
   assert(kernel);

   // remove temporary shared (ar + symbol), the cache will be used afterward
   std::remove(symbol);
   return kernel;
}

} // namespace mfem

#endif // MFEM_USE_JIT
