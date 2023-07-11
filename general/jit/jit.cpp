// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

#include <fstream>
#include <thread> // sleep_for
#include <chrono> // (milli) seconds

#include <cstring> // strlen
#include <cstdlib> // exit, system
#include <signal.h> // signals
#include <unistd.h> // fork
#include <sys/file.h> // flock
#include <sys/wait.h> // waitpid
#include <sys/stat.h>

#include "jit.hpp"
#include "../error.hpp"
#include "../globals.hpp"
#include "../communication.hpp"

#if !(defined(__linux__) || defined(__APPLE__))
#error MFEM JIT implementation is not supported on this platform!
#else
#include <sys/mman.h> // mmap
#include <dlfcn.h> // dlopen/dlsym, not available on Windows
#endif

// MFEM_AR, MFEM_XLINKER, MFEM_SO_EXT, MFEM_SO_PREFIX, MFEM_SO_POSTFIX and
// MFEM_INSTALL_BACKUP have to be defined at compile time.
// They are set by default in defaults.mk and MjitCmakeUtilities.cmake.

// The 'MFEM_JIT_DEBUG' environment variable can be set to:
//   - output dl errors,
//   - keep intermediate sources files.

// The 'MFEM_JIT_FORK' environment variable can be set to fork before
// MPI initialization and use this process to do the std::system calls.

namespace mfem
{

namespace jit
{

#if !(defined(MFEM_AR) && defined(MFEM_SO_EXT) && defined(MFEM_XLINKER) && \
      defined(MFEM_SO_PREFIX) && defined(MFEM_SO_POSTFIX) && defined(MFEM_INSTALL_BACKUP))
#define MFEM_AR ""
#define MFEM_SO_EXT ""
#define MFEM_XLINKER ""
#define MFEM_BUILD_DIR ""
#define MFEM_SO_PREFIX ""
#define MFEM_SO_POSTFIX ""
#define MFEM_INSTALL_BACKUP ""
#endif

namespace time
{

template <typename T, unsigned int TIMEOUT = 200> static void Sleep(T &&op)
{
   constexpr std::chrono::milliseconds tock(TIMEOUT);
   for (std::chrono::milliseconds tick(0); op(); tick += tock)
   {
      std::this_thread::sleep_for(tock);
   }
}

} // namespace time

namespace io
{

static inline bool Exists(const char *path)
{
   struct stat buf;
   // stat obtains information about the file pointed to by path
   return ::stat(path, &buf) == 0; // check successful completion
}

// fcntl wrapper to provide file locks
// Warning: 'FileLock' variables must be 'named' to live during its scope
class FileLock
{
   std::string s_name;
   const char *f_name;
   std::ofstream lock;
   int fd;

   int FCntl(int cmd, int type, bool check) const
   {
      struct ::flock data {};
      (data.l_type = type, data.l_whence = SEEK_SET);
      const int ret = ::fcntl(fd, cmd, &data);
      if (check) { MFEM_VERIFY(ret != -1, "[JIT] fcntl error");}
      return check ? ret : (ret != -1);
   }

public:
   // if 'now' is set, the lock will immediately happen,
   // otherwise the user must use the 'Wait' function below.
   FileLock(std::string name, const char *ext, bool now = true):
      s_name(name + "." + ext), f_name(s_name.c_str()), lock(f_name),
      fd(::open(f_name, O_RDWR))
   {
      MFEM_VERIFY(lock.good() && fd > 0, "[JIT] File lock " << f_name << " error!");
      if (now) { FCntl(F_SETLKW, F_WRLCK, true); } // wait now if locked
   }

   // bool cast operator, to be able to 'if' test a lock directly
   operator bool() const { return FCntl(F_SETLK, F_WRLCK, false); }

   ~FileLock()
   {
      FCntl(F_SETLK, F_UNLCK, true); // unlock
      ::close(fd); // close
      std::remove(f_name); // remove
   }

   void Wait() const // spinwait for the lock to be release with default timeout
   { time::Sleep([&]() { return static_cast<bool>(std::fstream(f_name)); }); }
};

} // namespace io

namespace mpi
{

// Return true if MPI has been initialized
static bool IsInitialized()
{
#ifndef MFEM_USE_MPI
   return false;
#else
   return Mpi::IsInitialized();
#endif
}

// Do the MPI_Init, which should be called from Mpi::Init when MFEM_USE_JIT
static int Init(int *argc, char ***argv)
{
#ifdef MFEM_USE_MPI
   return ::MPI_Init(argc, argv);
#else
   MFEM_CONTRACT_VAR(argc);
   MFEM_CONTRACT_VAR(argv);
   return EXIT_SUCCESS;
#endif
}

// Return the MPI world rank if it has been initialized, 0 otherwise
static int Rank()
{
   int world_rank = 0;
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized()) { world_rank = Mpi::WorldRank(); }
#endif
   return world_rank;
}

// Return the environment MPI world rank if set, -1 otherwise
static int EnvRank()
{
   const char *mv2   = std::getenv("MV2_COMM_WORLD_RANK"); // MVAPICH2
   const char *ompi  = std::getenv("OMPI_COMM_WORLD_RANK"); // OpenMPI
   const char *mpich = std::getenv("PMI_RANK"); // MPICH
   const char *rank  = mv2 ? mv2 : ompi ? ompi : mpich ? mpich : nullptr;
   return rank ? std::stoi(rank) : -1;
}

// Return true if the rank in world rank is zero
static bool Root() { return Rank() == 0; }

// Do a MPI barrier and status reduction if MPI has been initialized
static void Sync(int status = EXIT_SUCCESS)
{
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized())
   { MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD); }
#endif
   MFEM_VERIFY(status == EXIT_SUCCESS, "[JIT] mpi::Sync error!");
}

// Do a MPI broadcast from rank 0 if MPI has been initialized
static int Bcast(int value)
{
   int status = EXIT_SUCCESS;
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized())
   {
      const int ret = MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
      status = ret == MPI_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
   }
#endif
   MFEM_VERIFY(status == EXIT_SUCCESS, "[JIT] mpi::Bcast error!");
   return value;
}

} // namespace mpi

namespace dl
{

static const char* Error(const bool warning = false) noexcept
{
   const char *error = dlerror();
   if (warning && error) { MFEM_WARNING("[JIT] " << error); }
   MFEM_VERIFY(!::dlerror(), "[JIT] Should result in NULL being returned!");
   return error;
}

static void* Sym(void *handle, const char *name) noexcept
{
   void *sym = ::dlsym(handle, name);
   return (Error(), sym);
}

static void *Open(const char *path)
{
   void *handle = ::dlopen(path, RTLD_LAZY | RTLD_LOCAL);
   return (Error(), handle);
}

} // namespace dl

} // namespace jit

struct JIT::System // forked std::system implementation
{
   pid_t pid; // of the child process
   int *s_ack, rank; // shared status, must be able to store one MPI rank
   char *s_mem; // shared memory to store the command for the system call
   size_t size; // size of the s_mem shared memory

   // Acknowledgement status values
   static constexpr int ACK = ~0x0, CALL = 0x3243F6A8, EXIT = 0x9e3779b9;

   // Acknowledge functions (EQ and NE) with thread sleep
   template <typename OP>
   void AckOP(int xx) const { jit::time::Sleep([&]() { return OP()(*s_ack, xx); }); }
   void AckEQ(int xx = ACK) const { AckOP<std::equal_to<int>>(xx); }
   void AckNE(int xx = ACK) const { AckOP<std::not_equal_to<int>>(xx); }

   // Read, Write, Acknowledge, Send & Wait using the shared memory 's_ack'
   int Read() const { return *s_ack; }
   int Write(int xx) const { return *s_ack = xx; }

   void Acknowledge() const { Write(ACK); }
   void Send(int xx) const { AckNE(Write(xx)); } // blocks until != xx
   void Wait(bool EQ = true) const { EQ ? AckEQ() : AckNE(); }

   // Ack/Call/Exit 's_ack' decode
   bool IsAck() const { return Read() == ACK; }
   bool IsCall() const { return Read() == CALL; }
   bool IsExit() const { return Read() == EXIT; }

   // Ask the parent to launch a system call using this command
   int Run(const char *cmd)
   {
      const size_t cmd_length = 1 + std::strlen(cmd);
      MFEM_VERIFY(cmd_length < size, "[JIT] length error!");
      std::memcpy(s_mem, cmd, cmd_length); // copy command to shared memory
      Send(CALL); // call std::system through the child process
      Wait(false); // wait for the acknowledgment after compilation
      return EXIT_SUCCESS;
   }
};

struct JIT::Command // convenient command builder & const char* type cast
{
   Command& operator<<(const char *c) { Get().command << c << ' '; return *this; }
   Command& operator<<(const std::string &s) { return *this << s.c_str(); }
   operator const char *()
   {
      static thread_local std::string sl_cmd;
      sl_cmd = Get().command.str();
      (Get().command.clear(), Get().command.str("")); // real flush
      return sl_cmd.c_str();
   }
};

using namespace jit;

void JIT::Init(int *argc, char ***argv)
{
   MFEM_VERIFY(!mpi::IsInitialized(), "[JIT] MPI already initialized!");

   if (Get().std_system) // each rank does the MPI init and update their rank
   {
      mpi::Init(argc, argv);
      Get().rank = mpi::Rank(); // used in destructor
      return;
   }

   System &sys = *Get().sys;

   // Initialization of the shared memory between the MPI root
   // and the thread that will launch the std::system commands
   auto SysInit = [&]()
   {
      constexpr int prot = PROT_READ | PROT_WRITE;
      constexpr int flags = MAP_SHARED | MAP_ANONYMOUS;
      sys.size = (size_t) sysconf(_SC_PAGE_SIZE);
      sys.s_ack = (int*) ::mmap(nullptr, sizeof(int), prot, flags, -1, 0);
      sys.s_mem = (char*) ::mmap(nullptr, sys.size, prot, flags, -1, 0);
      sys.Write(sys.ACK); // initialize state
   };

   // MPI rank is looked for in the environment (-1 if not found)
   const int env_rank = mpi::EnvRank();
   if (env_rank >= 0) // MPI rank is known from environment
   {
      if (env_rank == 0) { SysInit(); } // if set, only root will use mmap
      if (env_rank > 0) // other ranks only MPI_Init
      {
         mpi::Init(argc, argv);
         sys.pid = getpid(); // set our pid for JIT finalize to be an no-op
         return;
      }
   }
   else { SysInit(); } // cannot know root before MPI::Init: everyone gets ready

   if ((sys.pid = ::fork()) != 0)
   {
      mpi::Init(argc, argv);
      sys.Write(Get().rank = mpi::Rank()); // inform the child about our rank
      sys.Wait(false); // wait for the child to acknowledge
   }
   else
   {
      MFEM_VERIFY(sys.pid == 0, "[JIT] Child pid error!");
      MFEM_VERIFY(sys.IsAck(), "[JIT] Child initialize state error!");
      sys.Wait(); // wait for parent's rank
      const int rank = sys.Read(); // Save the rank
      sys.Acknowledge();
      if (rank == 0) // only root is kept for system calls
      {
         while (true)
         {
            sys.Wait(); // waiting for the root to wake us
            if (sys.IsCall()) { if (std::system(sys.s_mem)) break; }
            if (sys.IsExit()) { break; }
            sys.Acknowledge();
         }
      }
      std::exit(EXIT_SUCCESS); // no children are coming back
   }
   MFEM_VERIFY(sys.pid != 0, "[JIT] Children shall not pass!");
}

void JIT::Finalize()
{
   if (!mpi::IsInitialized() || Get().std_system) { return; } // nothing to do
   System &sys = *Get().sys;
   // child and env-ranked have nothing to do
   if (sys.pid == 0 || sys.pid == getpid()) { return; }
   MFEM_VERIFY(!Get().std_system, "std::system should be used!");
   MFEM_VERIFY(sys.IsAck(), "[JIT] Finalize acknowledgment error!");
   int status;
   sys.Send(sys.EXIT);
   ::waitpid(sys.pid, &status, WUNTRACED | WCONTINUED); // wait for child
   MFEM_VERIFY(status == 0, "[JIT] Error with the compiler thread");
   if (::munmap(sys.s_mem, sys.size) != 0 || // release shared memory
       ::munmap(sys.s_ack, sizeof(int)) != 0)
   { MFEM_ABORT("[JIT] Finalize memory error!"); }
}

void JIT::Configure(const char *name, const char *path, bool keep)
{
   Get().path = path;
   Get().keep_cache = keep;
   Get().rank = mpi::Rank();

   auto CreateFullPath = [&](const char *ext)
   {
      std::string lib = Get().path;
      lib += std::string("/") + std::string("lib") + name;
      lib += std::string(".") + ext;
      return lib;
   };

   Get().lib_ar = CreateFullPath("a");
   const char *lib_ar = Get().lib_ar.c_str();
   if (mpi::Root() && !io::Exists(lib_ar)) // if lib_ar does not exist
   {
      MFEM_VERIFY(std::ofstream(lib_ar), "[JIT] Error creating " << lib_ar);
      std::remove(lib_ar); // try to touch and remove
   }
   mpi::Sync();
   Get().lib_so = CreateFullPath("" MFEM_SO_EXT);
}

void* JIT::Lookup(const size_t hash, const char *name, const char *cxx,
                  const char *flags, const char *link, const char *libs,
                  const char *dir, const char *source, const char *symbol)
{
   dl::Error(false); // flush dl errors
   mpi::Sync(); // make sure file testing is done at the same time
   const char *lib_ar = Get().lib_ar.c_str();
   const char *lib_so = Get().lib_so.c_str();
   void *handle = io::Exists(lib_so) ? dl::Open(lib_so) : nullptr;
   if (!handle && io::Exists(lib_ar)) // if .so not found, try archive
   {
      int status = EXIT_SUCCESS;
      if (mpi::Root())
      {
         io::FileLock ar_lock(lib_ar, "ak");
         Command() << cxx << link << "-shared" << "-o" << lib_so
                   << "-L" MFEM_BUILD_DIR
                   << "-L" MFEM_INSTALL_DIR "/lib -lmfem"
                   << MFEM_SO_PREFIX << lib_ar << MFEM_SO_POSTFIX
                   << MFEM_XLINKER + std::string("-rpath,") + Get().path
                   << libs;
         status = Run();
      }
      mpi::Sync(status);
      handle = dl::Open(lib_so);
      if (!handle) // happens when Lib_so is removed in the meantime
      { return Lookup(hash, name, cxx, flags, link, libs, dir, source, symbol); }
      MFEM_VERIFY(handle, "[JIT] Error " << lib_ar << " => " << lib_so);
   }

   auto WorldCompile = [&]() // but only MPI root does the compilation
   {
      // each compilation process adds their id to the hash,
      // this is used to handle parallel compilations of the same source
      const auto id = std::string("_") + std::to_string(mpi::Bcast(getpid()));
      const auto so = JIT::ToString(hash, id.c_str());

      std::function<int(const char *)> RootCompile = [&](const char *so)
      {
         auto install = [](const char *in, const char *out)
         {
            Command() << "install" << MFEM_INSTALL_BACKUP << in << out;
            MFEM_VERIFY(Run() == EXIT_SUCCESS,
                        "[JIT] install error: " << in << " => " << out);
         };
         io::FileLock cc_lock(JIT::ToString(hash), "ck", false);
         if (cc_lock)
         {
            // Create source file: source => cc
            const auto cc = JIT::ToString(hash, ".cc"); // input source
            {
               std::ofstream cc_file(cc); // open the source file
               MFEM_VERIFY(cc_file.good(), "[JIT] Source file error!");
               cc_file << source;
               cc_file.close();
            }
            // Compilation: cc => co
            const auto co = JIT::ToString(hash, ".co"); // output object
            {
               MFEM_VERIFY(io::Exists(MFEM_INSTALL_DIR "/include/mfem/mfem.hpp"),
                           "[JIT] Could not find any MFEM header!");
               std::string incs;
               for (auto &inc: Get().includes) { incs += "-include \"" + inc + "\" "; }
               std::string mfem_install_include_dir(MFEM_INSTALL_DIR "/include/mfem");
               Command() << cxx << flags
                         << "-I" << mfem_install_include_dir
                         << "-I" << mfem_install_include_dir + "/" + dir
                         << incs.c_str()
#ifdef MFEM_USE_CUDA
                         // nvcc option to embed relocatable device code
                         << "--relocatable-device-code=true"
#endif
                         << "-c" << "-o" << co << cc;
               if (Run(name)) { return EXIT_FAILURE; }
               if (!Get().debug) { std::remove(cc.c_str()); }
            }
            // Update archive: ar += co, (ar + co) => so, so => lib_so
            io::FileLock ar_lock(lib_ar, "ak");
            {
               Command() << ("" MFEM_AR) << "-r" << lib_ar << co;
               if (Run()) { return EXIT_FAILURE; }
               if (!Get().debug) { std::remove(co.c_str()); }
               // Create temporary shared library:
               // Warning: there are usually two mfem libraries after installation:
               //   - the one in MFEM_BUILD_DIR: used for all internal MFEM examples, miniapps & tests
               //   - the other ine MFEM_INSTALL_DIR/lib: should be the only one used otherwise
               Command() << cxx << link << "-o" << so
                         << "-shared"
                         << "-L" MFEM_BUILD_DIR // MFEM_SOURCE_DIR with make, CMAKE_CURRENT_BINARY_DIR with cmake
                         << "-L" MFEM_INSTALL_DIR "/lib -lmfem"
                         << MFEM_SO_PREFIX << lib_ar << MFEM_SO_POSTFIX
                         << MFEM_XLINKER + std::string("-rpath,") + Get().path
                         << libs;
               if (Run()) { return EXIT_FAILURE; }
               // Install temporary shared library: so => lib_so
               install(so, lib_so);
            }
         }
         else // avoid duplicate compilation
         {
            cc_lock.Wait();
            io::FileLock ar_lock(lib_ar, "ak");
            MFEM_VERIFY(io::Exists(lib_so), "[JIT] lib_so not found!");
            install(lib_so, so); // but still install temporary shared library
         }
         return EXIT_SUCCESS;
      };
      const int status = mpi::Root() ? RootCompile(so.c_str()) : EXIT_SUCCESS;
      MFEM_VERIFY(status == EXIT_SUCCESS, "[JIT] RootCompile error!");
      mpi::Sync(status); // all ranks verify the status
      std::string symbol_path(Get().path + "/");
      handle = dl::Open((symbol_path + so).c_str()); // opens symbol
      mpi::Sync();
      MFEM_VERIFY(handle, "[JIT] Error creating handle!");
      if (mpi::Root()) { std::remove(so.c_str()); }
   }; // WorldCompile

   // no cache => launch compilation
   if (!handle) { WorldCompile(); }
   MFEM_VERIFY(handle, "[JIT] No handle created!");
   void *kernel = dl::Sym(handle, symbol); // symbol lookup

   // no symbol => launch compilation & update kernel symbol
   if (!kernel) { WorldCompile(); kernel = dl::Sym(handle, symbol); }
   MFEM_VERIFY(kernel, "[JIT] No kernel found!");
   return kernel;
}

JIT::JIT():
   debug(std::getenv("MFEM_JIT_DEBUG") != nullptr),
   std_system(std::getenv("MFEM_JIT_FORK") == nullptr),
   keep_cache(true),
   path("."),
   lib_ar("libmjit.a"),
   lib_so("./libmjit." MFEM_SO_EXT),
   rank(0),
   sys(std_system ? nullptr : new System)
{
   includes.push_back("mfem.hpp");
   includes.push_back("general/forall.hpp"); // for mfem::forall
   includes.push_back("general/jit/jit.hpp"); // for Hash, Find
}

JIT::~JIT() // warning: can't use mpi::Root here
{
   const char *ar = Get().lib_ar.c_str();
   if (!keep_cache && Get().rank == 0 && io::Exists(ar)) { std::remove(ar); }
   delete sys;
}

JIT JIT::jit_singleton; // Initialize the unique global Jit singleton

int JIT::Run(const char *kernel_name)
{
   const char *cmd = Command();
   auto warn = [](const std::string &msg) { mfem::out << msg.c_str() << std::endl; };
   std::string msg(std::string("JIT: "));
#ifdef MFEM_DEBUG
   if (kernel_name) { msg += kernel_name; msg += " "; }
   if (kernel_name || Get().debug) { warn(msg + cmd); }
#else
   if (kernel_name) { warn(msg + "compiling " + kernel_name); }
#endif // MFEM_DEBUG
   // In serial mode or with the std_system option set, just call std::system
   if (!mpi::IsInitialized() || Get().std_system) { return std::system(cmd); }
   // Otherwise, write the command to the child process
   MFEM_VERIFY(Get().sys, "[JIT] Thread system error!");
   return Get().sys->Run(cmd);
}

} // namespace mfem

#endif // MFEM_USE_JIT
