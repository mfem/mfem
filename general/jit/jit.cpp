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
#include <regex>
#include <thread>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <cassert>
#include <fstream>
#include <algorithm>

#include <fcntl.h>
#include <sys/select.h> // not available with _MSC_VER
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>

#include "../communication.hpp"
#include "../globals.hpp"
#include "../error.hpp"
#include "jit.hpp"

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

static struct
{
   pid_t pid;
   uintptr_t pagesize;
   int *ack;
   size_t *isz, *osz; // status, source length, obj size
   char *cmd, *src, *obj; // command, source, *obj
} Smem;

static constexpr size_t SRC_FACTOR = 256;
static constexpr size_t OBJ_FACTOR = 1024;
static constexpr int ACK = ~0, COMPILE=0x1, EXIT=0x2;

template <typename Op> static bool Ack(const int check)
{
   const long ms = 100L;
   const struct timespec rqtp = {0, ms*1000000L};
   while (Op()(*Smem.ack, check)) { ::nanosleep(&rqtp, nullptr); }
   return true;
}

static bool AckEQ(const int check = ACK) { return Ack<std::equal_to<int>>(check); }

static bool AckNE(const int check = ACK) { return Ack<std::not_equal_to<int>>(check); }

template<typename T>
static inline int argn(T v[], int c = 0) { while (v[c]) { c += 1; } return c; }

/// \brief CloseAndWait
/// \param fd descriptor to be closed.
static void CloseAndWait(int fd)
{
   if (::close(fd) < 0) { ::perror(strerror(errno)); }
   // block parent process until any of its children has finished
   ::wait(nullptr);
}

/// In-memory compilation
static int CompileInMemory(const char *cmd, const size_t isz, const char *src,
                           char *obj, size_t *osz)
{
   dbg();
   // input, output and error pipes
   int ip[2], op[2], ep[2];
   constexpr size_t PIPE_READ = 0;
   constexpr size_t PIPE_WRITE = 1;

   if (::pipe(ip)<0 || ::pipe(op)<0 || ::pipe(ep)<0)
   {
      return ::perror(strerror(errno)), EXIT_FAILURE;
   }

   if (::fork() == 0) // Child process which calls the compiler
   {
      //dbg("child fork cmd:'%s', isz:%d",cmd,isz);
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

      const int status = std::system(cmd);
      return (assert(status == EXIT_SUCCESS), status);
   }

   // Parent process
   // WARNING: cannot use any debug output until the pipes are back to normal!

   // close unused sides of the pipes
   ::close(op[PIPE_WRITE]);
   ::close(ep[PIPE_WRITE]);

   // write all the source present in memory to the 'input' pipe
   size_t ni_w = ::write(ip[PIPE_WRITE], src, isz-1);
   if (ni_w != isz - 1) { return CloseAndWait(op[PIPE_READ]), EXIT_FAILURE; }
   ::close(ip[PIPE_WRITE]);

   char buffer[1<<16];
   constexpr int SIZE = sizeof(buffer);
   size_t nr, ne = 0; // number of read, error & output bytes

   // Scan error pipe with timeout
   {
      fd_set set;
      //#warning should be {1,0}
      struct timeval timeout {0, 0}; // one second
      FD_ZERO(&set); // clear the set
      FD_SET(ep[PIPE_READ], &set); // add the descriptor we need a timeout on
      const int rv = ::select(ep[PIPE_READ]+1, &set, NULL, NULL, &timeout);
      //assert(false);
      if (rv == -1) { assert(false); return CloseAndWait(op[PIPE_READ]), EXIT_FAILURE; }
      else if (rv == 0) { assert(true); /*dbg("No error found!");*/ }
      else
      {
         assert(false);
         dbg("Error data is available!");
         while ((nr = ::read(ep[PIPE_READ], buffer, SIZE)) > 0)
         {
            size_t nr_w = ::write(STDOUT_FILENO, buffer, nr);
            if (nr_w != nr) { return CloseAndWait(op[PIPE_READ]), EXIT_FAILURE; }
            ne += nr;
         }
         ::close(ep[PIPE_READ]);
         if (ne > 0)
         {
            CloseAndWait(op[PIPE_READ]);
            return ::perror("Compilation error!"), EXIT_FAILURE;
         }
      }
      ::close(ep[PIPE_READ]);
   }
   // then get the object from output pipe
   for (*osz = 0; (nr = ::read(op[PIPE_READ], buffer, SIZE)) > 0;)
   {
      *osz += nr;
      assert(*osz < OBJ_FACTOR*Smem.pagesize);
      ::memcpy(obj + *osz - nr, buffer, nr);
   }
   CloseAndWait(op[PIPE_READ]);
   return EXIT_SUCCESS;
}

} // namespace jit

using namespace jit;

/**
 * @brief Jit::MPI_Init
 * @param argc
 * @param argv
 * @return
 */
int Jit::Init(int *argc, char ***argv)
{
   Smem.pagesize = (uintptr_t) sysconf(_SC_PAGE_SIZE);

   struct
   {
      const int prot = PROT_READ | PROT_WRITE;
      const int flags = MAP_SHARED | MAP_ANONYMOUS;
      void *operator()(size_t len)
      {
         void *mem = ::mmap(nullptr, len, prot, flags, -1, 0);
         return assert(mem), mem;
      }
   } Mmap;

   Smem.ack = (int*) Mmap(sizeof(int));
   Smem.isz = (size_t*) Mmap(sizeof(size_t));
   Smem.osz = (size_t*) Mmap(sizeof(size_t));

   Smem.cmd = (char*) Mmap(Smem.pagesize);
   Smem.src = (char*) Mmap(SRC_FACTOR*Smem.pagesize);
   Smem.obj = (char*) Mmap(OBJ_FACTOR*Smem.pagesize);

   *Smem.ack = jit::ACK; // initialize state

   if ((Smem.pid = ::fork()) != 0) // parent
   {
#ifdef MFEM_USE_MPI
      ::MPI_Init(argc, argv);
#else
      MFEM_CONTRACT_VAR(argc);
      MFEM_CONTRACT_VAR(argv);
#endif
      *Smem.ack = Jit::Rank(); // inform the child about the rank
      AckNE(); // wait for the child to acknowledge
   }
   else // JIT compiler child process
   {
      assert(*Smem.ack == ACK); // Child init error
      AckEQ(); // wait for parent's rank

      const int rank = *Smem.ack;
      *Smem.ack = ACK; // now acknowledge back
      dbg("[thd:%d] entering",rank);

      auto work = [&]() // looping working thread
      {
         while (true)
         {
            AckEQ(); // waiting for somebody to wake us...
            if (*Smem.ack == COMPILE)
            {
               dbg("COMPILE, isz:%d", *Smem.isz);
               const int status =
                  CompileInMemory(Smem.cmd, *Smem.isz, Smem.src,
                                  Smem.obj, Smem.osz);
               if (status != EXIT_SUCCESS) { dbg("ERROR"); return; }
            }
            if (*Smem.ack == EXIT) { dbg("EXIT"); return;}
            *Smem.ack = ACK; // send back ACK
         }
      };

      // only root is kept for compilation
      if (rank == 0) { std::thread (work).join(); }
      dbg("[thd:%d] EXIT_SUCCESS",rank);
      exit(EXIT_SUCCESS);
   }
   return EXIT_SUCCESS;
}

void Jit::Finalize()
{
   dbg();
   assert(*Smem.ack == ACK); // Parent finalize error!
   if (Jit::Root())
   {
      int status;
      dbg("[mpi:0] send thd:0 exit");
      Jit::Exit();
      ::waitpid(Smem.pid, &status, WUNTRACED | WCONTINUED);
      assert(status == 0); // Error with JIT compiler thread!
   }
   // release shared memory
   if (::munmap(Smem.ack, sizeof(int)) != 0) { assert(false); }
   if (::munmap(Smem.isz, sizeof(int)) != 0) { assert(false); }
   if (::munmap(Smem.osz, sizeof(int)) != 0) { assert(false); }
   if (::munmap(Smem.cmd, Smem.pagesize) != 0) { assert(false); }
   if (::munmap(Smem.src, SRC_FACTOR*Smem.pagesize) != 0) { assert(false); }
   if (::munmap(Smem.obj, OBJ_FACTOR*Smem.pagesize) != 0) { assert(false); }
}

bool Jit::Root() { return Rank() == 0; }

int Jit::Rank()
{
   int world_rank = 0;
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized()) { world_rank = Mpi::WorldRank(); }
#endif
   return world_rank;
}

int Jit::Size()
{
   int world_size = 1;
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized()) { world_size = Mpi::WorldSize(); }
#endif
   return world_size;
}

void Jit::Sync()
{
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized()) { MPI_Barrier(MPI_COMM_WORLD); }
#endif
}

static int Send(const int code)
{
   *Smem.ack = code;
   AckNE(code); // wait until code has been flushed
   return EXIT_SUCCESS;
}

void Jit::Exit() { Send(jit::EXIT); }

static std::string CreateCommandLine(const char *argv[])
{
   const int argc = jit::argn(argv);
   assert(argc > 0);
   std::string cmd(argv[0]);
   for (int k = 1; k < argc && argv[k]; k++) { cmd += " "; cmd += argv[k]; }
   return cmd;
}

// #warning should use the JIT thread
int Jit::System(const char *argv[])
{
   if (Jit::Root())
   {
      dbg(CreateCommandLine(argv).c_str());
      int status = ::system(CreateCommandLine(argv).c_str());
      if (status != EXIT_SUCCESS) { return EXIT_FAILURE; }
   }
   Jit::Sync();
   return EXIT_SUCCESS;
}

int Jit::Compile(const char *argv[], const int n, const char *src,
                 char *&obj, size_t &size)
{
   dbg();
#ifndef MFEM_USE_MPI
   // In serial mode, JIT/MPI has not been initialized: do it once here
   // The Finalized is not done yet in serial!
   // Maybe foldback to standard ::system call here,
   // however we would loose the input source from stdin and memory compilation
   static bool initialized = false;
   if (!initialized) { Jit::Init(); initialized = true;}
#else
   if (!Mpi::IsInitialized())
   {
      dbg("Initializing...");
      static bool initialized = false;
      if (!initialized) { Jit::Init(); initialized = true;}
   }
#endif

   assert(Jit::Root()); // make sure we are the JIT root

   const std::string cmd = CreateCommandLine(argv);
   const char *cmd_c_str = cmd.c_str();

   // write the command in shared mem
   assert(std::strlen(cmd_c_str) < Smem.pagesize);
   ::memcpy(Smem.cmd, cmd_c_str, std::strlen(cmd_c_str));

   // write the src in shared mem
   assert(n == static_cast<int>(1 + std::strlen(src)));
   assert(n < static_cast<int>(SRC_FACTOR*Smem.pagesize));
   ::memcpy(Smem.src, src, n);

   // write the size of src in shared mem
   *Smem.isz = n;
   dbg("Smem.isz:%d",*Smem.isz);
   *Smem.osz = 4;

   Send(COMPILE);
   AckNE(); // wait for the child to acknowledge

   // read back the object's size
   size = *Smem.osz;
   dbg("size:%d",size);
   assert(*Smem.osz != 4);
   assert(size < OBJ_FACTOR*Smem.pagesize);
   // set back the object from child's memory
   obj = Smem.obj;
   return EXIT_SUCCESS;
}

inline int CreateKernelSourceFile(const char *cc, const char *src)
{
   if (!Jit::Root()) { return EXIT_SUCCESS; }
   const int fd = ::open(cc, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
   if (fd < 0) { return false; }
   if (::dprintf(fd, "%s", src) < 0) { return EXIT_FAILURE; }
   if (::close(fd) < 0) { return EXIT_FAILURE; }
   return EXIT_SUCCESS;
}

int Jit::Update(const size_t hash, const size_t n, char *src,
                const char *cxx, const char *cxxflags,
                const char *mfem_source_dir, const char *mfem_install_dir,
                void *&handle)
{
   assert(src);
   char co[MFEM_JIT_SYMBOL_SIZE+3];
   Jit::uint64str(hash, co, ".co");

   constexpr int ld_mode = RTLD_NOW | RTLD_LOCAL;
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
#define MFEM_JIT_INSTALL_BACKUP ""
   //#define MFEM_JIT_INSTALL_BACKUP  "--backup=none"
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
   constexpr const char *lib_ar = "lib" MFEM_JIT_LIB_NAME ".a";
   constexpr const char *lib_so = "./lib" MFEM_JIT_LIB_NAME ".so";
   constexpr const char *lib_to = "lib" MFEM_JIT_LIB_NAME ".to";

   // If there is already a JIT archive, use it and create the lib_so
   if (!handle && std::fstream(lib_ar))
   {
      dbg("%s => %s", lib_ar, lib_so);
      const char *argv_so[] =
      {
         cxx, "-shared", "-o", lib_so, beg_load, lib_ar, end_load,
         MFEM_JIT_LINKER_OPTION "-rpath,.", nullptr
      };
      if (Jit::System(argv_so)) { return EXIT_FAILURE; }
      Jit::Sync();
      delete[] src;
      handle = ::dlopen(lib_so, ld_mode);
      assert(handle);
      return EXIT_SUCCESS;
   }

   if (handle)
   {
      while (::dlclose(handle) != -1) { dbg("dlclose:%d",::dlclose(handle)); }
   }

   // MFEM source path, include path & lib_so_v
   char Imsrc[PM], Iminc[PM];
   if (snprintf(Imsrc, PM, "-I%s ", mfem_source_dir) < 0) { return EXIT_FAILURE; }
   if (snprintf(Iminc, PM, "-I%s/include ", mfem_install_dir) < 0) { return EXIT_FAILURE; }

   if (Jit::Root())
   {
      std::regex reg("\\s+");
      std::string flags(cxxflags);
      const auto cxxargv =
         std::vector<std::string>(
            std::sregex_token_iterator{begin(flags), end(flags), reg, -1},
            std::sregex_token_iterator{});
      std::vector<const char*> argv, argv_files;
      argv.push_back(cxx);
      for (auto &a : cxxargv)
      {
         using uptr = std::unique_ptr<char, decltype(&std::free)>;
         uptr *t_copy = new uptr(strdup(a.data()), &std::free);
         argv.push_back(t_copy->get());
      }
      //argv.push_back(MFEM_JIT_COMPILER_OPTION "-Wno-unused-variable");
      argv.push_back(fPIC);
      argv.push_back("-c");
      // avoid redefinition, as with nvcc, the option -x cu is already present
#ifdef MFEM_USE_CUDA
      argv.push_back(MFEM_JIT_DEVICE_CODE);
#else
      argv.push_back(MFEM_JIT_COMPILER_OPTION "-pipe");
#endif // MFEM_USE_CUDA
      argv_files = argv;

#if 0
      char cc[MFEM_JIT_SYMBOL_SIZE+3];
      Jit::uint64str(hash, cc, ".cc");
      argv_files.push_back("-o");
      argv_files.push_back(co); // output
      argv_files.push_back(cc); // input
      argv_files.push_back(nullptr);

      // dump src in file k****************.cc
      if (CreateKernelSourceFile(cc,src)) { return EXIT_FAILURE; }
      if (Jit::System(argv_files.data())) { return EXIT_FAILURE; }
#else
      // error: -E or -x required when input is from standard input
      argv.push_back("-x"); argv.push_back("c++");
      argv.push_back("-o"); argv.push_back("/dev/stdout"); // stdout output
      argv.push_back("-"); // stdin input
      argv.push_back(nullptr);

      dbg("%s", CreateCommandLine(argv.data()).c_str()); // debug output
      // compile in memory
      size_t size = 0;
      char *obj = nullptr;
      //dbg("\033[33m'%s', n:%d\033[m",src,n);
      if (Jit::Compile(argv.data(), n, src, obj, size))
      {
         dbg("EXIT_FAILURE"); return EXIT_FAILURE;
      }
      dbg("done");
      delete[] src;

      // security issues, => write back object into output file
      dbg("size:%d",size);
      assert(obj && size > 0);
      const mode_t mode = S_IRUSR | S_IWUSR;
      const int oflag = O_CREAT | O_RDWR | O_TRUNC;
      const int co_fd = ::open(co, oflag, mode);
      const size_t written = ::write(co_fd, obj, size);
      if (written != size) { return perror("!write object"), EXIT_FAILURE; }
      if (::close(co_fd) < 0) { return perror("!close object"), EXIT_FAILURE; }
#endif
   }
   Jit::Sync();

   // Update archive
   const char *argv_ar[] = { "ar", "-rv", lib_ar, co, nullptr };
   if (Jit::System(argv_ar)) { dbg("ar error"); return EXIT_FAILURE; }
   if (Jit::Root()) { ::unlink(co); }

   // Create shared library
   const char *argv_so[] = { cxx, "-shared", "-o", lib_to,
                             beg_load, lib_ar, end_load,
                             nullptr
                           };
   if (Jit::System(argv_so)) { dbg("shared error"); return EXIT_FAILURE; }

   // Install shared library
   const char *install[] =
   { "install", "-v", MFEM_JIT_INSTALL_BACKUP, lib_to, lib_so, nullptr };
   if (Jit::System(install)) { dbg("install error"); return EXIT_FAILURE; }
   if (Jit::Root()) { ::unlink(lib_to); } // remove the temporary shared lib after use

   Jit::Sync();

   handle = ::dlopen(lib_so, ld_mode);
   assert(handle);

   dbg("EXIT_SUCCESS");
   return EXIT_SUCCESS;
}

} // namespace mfem

#endif // MFEM_USE_JIT
