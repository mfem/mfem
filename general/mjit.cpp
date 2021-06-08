// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

/** @file mjit.cpp
 *  This file has multiple purposes:
 *
 *      - Implements the serial and parallel @c mfem::System call.
 *
 *      - Produces the @c mjit executable, which can:
 *          - preprocess source files,
 *          - spawn & fork in parallel to be able to @c system calls.
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
#include <cstring>
#include <ciso646>
#include <cassert>
#include <fstream>
#include <climits>
#include <algorithm>
#include <iostream>

#include <dlfcn.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>

using std::list;
using std::regex;
using std::vector;
using std::string;
using std::istream;
using std::ostream;

#include "../config/config.hpp"

#include "error.hpp"
#include "mjit.hpp"
#include "globals.hpp"

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 198
#include "debug.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <sys/mman.h>
#endif

// *****************************************************************************
#define MFEM_JIT_STR(...) #__VA_ARGS__
#define MFEM_JIT_STRINGIFY(...) MFEM_JIT_STR(__VA_ARGS__)

namespace mfem
{

namespace jit
{

static inline int argn(char *argv[], int argc = 0)
{
   while (argv[argc]) { argc += 1; }
   return argc;
}

// Implementation of mfem::jit::System
#if !defined(MFEM_USE_MPI)
// The serial implementation does nothing special but launching
// the =system= command.
static int System(char *argv[])
{
   const int argc = argn(argv);
   if (argc < 2) { return EXIT_FAILURE; }
   string command(argv[1]);
   for (int k = 2; k < argc && argv[k]; k++)
   {
      command.append(" ");
      command.append(argv[k]);
   }
   const char *command_c_str = command.c_str();
   dbg(command_c_str);
   return ::system(command_c_str);
}

bool Root() { return true; }

static inline void FlushAndWait(const int fd)
{
   if (::close(fd) < 0) { perror(strerror(errno)); }
   wait(nullptr);
}

// In-memory compilation
static int InMemCompile(const char *imem,
                        char *&omem,
                        size_t &nw,
                        char *argv[])
{
   const size_t nbytes = std::strlen(imem);
   dbg("imem:%p, nbytes:%d", (void*)imem, nbytes);
   //dbg("kernel:\n%s", imem);

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

   if (pipe(ip) == -1) { perror("Input Pipe failed"); exit(1); }
   if (pipe(op) == -1) { perror("Output Pipe failed"); exit(1); }
   if (pipe(ep) == -1) { perror("Error Pipe failed"); exit(1); }

   if (fork() == 0)           // Child process which calls the compiler
   {
      close(STDIN_FILENO);    // closing stdin
      dup2(ip[PIPE_READ],0);  // replacing stdin with pipe read
      close(ip[PIPE_READ]);   // close reading end
      close(ip[PIPE_WRITE]);  // no longer needed

      dup2(op[PIPE_WRITE],1); // send stdout to the output pipe
      close(op[PIPE_READ]);   // no longer needed
      close(op[PIPE_WRITE]);  // no longer needed

      dup2(ep[PIPE_WRITE],2); // send stderr to the error pipe
      close(ep[PIPE_READ]);   // no longer needed
      close(ep[PIPE_WRITE]);  // no longer needed

      ::execvp(argv[0], const_cast<char* const*>(argv));
      exit((perror("execvp failed"),1));
   }

   // Parent process

   // close unused sides of the pipes
   close(op[PIPE_WRITE]);
   close(ep[PIPE_WRITE]);

   // write all source present in memory to the 'input' pipe
   write(ip[PIPE_WRITE], imem, nbytes);
   close(ip[PIPE_WRITE]);

   char buffer[8192];
   constexpr size_t SIZE = sizeof(buffer);
   size_t nr, ne = 0, osize = 0; // number of read & error bytes

   // Scan error pipe with timeout
   {
      fd_set set;
      struct timeval timeout {1, 0}; // one second
      FD_ZERO(&set); // clear the set
      FD_SET(ep[PIPE_READ], &set); // add the descriptor we need a timeout on
      const int rv = select(ep[PIPE_READ]+1, &set, NULL, NULL, &timeout);
      if (rv == -1) { return perror("select()"), EXIT_FAILURE; }
      else if (rv == 0) { /* printf("No data within 1 seconds.\n"); */ }
      else
      {
         dbg("Error data is available now.");
         while ((nr = ::read(ep[PIPE_READ], buffer, SIZE)) > 0)
         {
            const size_t nw = ::write(STDOUT_FILENO, buffer, nr);
            dbg("buffer: %s, nw:%d",buffer,nw);
            ne += nr;
         }
         dbg("ne: %d",ne);
         ::close(ep[PIPE_READ]);
         if (ne > 0)
         {
            FlushAndWait(op[PIPE_READ]);
            return perror("Compilation error!"), EXIT_FAILURE;
         }
      }
      ::close(ep[PIPE_READ]);
   }

   // then get the object from output pipe
   nw = 0;
   osize = SIZE;
   omem = (char*) ::realloc(nullptr, SIZE);
   while ((nr = ::read(op[PIPE_READ], buffer, SIZE)) > 0)
   {
      nw += nr;
      if (nw > osize)
      {
         dbg("omem.realloc(%d + %d);", osize, SIZE);
         omem = (char*) ::realloc(omem, osize + SIZE);
         osize += SIZE;
      }
      ::memcpy(omem + nw - nr, buffer, nr);
   }
   dbg("object size:%d", nw);
   FlushAndWait(op[PIPE_READ]);
   return EXIT_SUCCESS;
}

#else
// The parallel implementation will spawn the =mjit= binary on one mpi rank to
// be able to run on one core and use MPI to broadcast the compilation output.

// Different commands that can be broadcasted
enum Command
{
   READY,
   SYSTEM_CALL
};

static bool MPI_Inited()
{
   int ini = false;
   MPI_Initialized(&ini);
   return ini ? true : false;
}

static int MPI_Size()
{
   int size = 1;
   if (MPI_Inited()) { MPI_Comm_size(MPI_COMM_WORLD, &size); }
   return size;
}

bool Root()
{
   int world_rank = 0;
   if (MPI_Inited()) { MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); }
   return world_rank == 0;
}

int System(char *argv[])
{
   const int argc = argn(argv);
   if (argc < 2) { return EXIT_FAILURE; }
   // Point to the 'mjit' binary
   char mjit[PATH_MAX];
   if (snprintf(mjit, PATH_MAX, "%s/bin/mjit", MFEM_INSTALL_DIR) < 0)
   { return EXIT_FAILURE; }
   dbg(mjit);
   // If we have not been launch with mpirun, just fold back to serial case,
   // which has a shift in the arguments
   if (!MPI_Inited() || MPI_Size()==1)
   {
      string command(argv[1]);
      for (int k = 2; k < argc && argv[k]; k++)
      {
         command.append(" ");
         command.append(argv[k]);
      }
      const char *command_c_str = command.c_str();
      dbg(command_c_str);
      return system(command_c_str);
   }

   // Debug our command
   string command(argv[0]);
   for (int k = 1; k < argc && argv[k]; k++)
   {
      command.append(" ");
      command.append(argv[k]);
   }
   const char *command_c_str = command.c_str();
   dbg(command_c_str);

   // Spawn the sub MPI group
   constexpr int root = 0;
   int errcode = EXIT_FAILURE;
   const MPI_Info info = MPI_INFO_NULL;
   MPI_Comm comm = MPI_COMM_WORLD, intercomm = MPI_COMM_NULL;
   MPI_Barrier(comm);
   const int spawned = // Now spawn one binary
      MPI_Comm_spawn(mjit, argv, 1, info, root, comm, &intercomm, &errcode);
   if (spawned != MPI_SUCCESS) { return EXIT_FAILURE; }
   if (errcode != EXIT_SUCCESS) { return EXIT_FAILURE; }
   // Broadcast READY through intercomm, and wait for return
   int status = mfem::jit::READY;
   // MPI_ROOT stands as a special value for intercomms
   MPI_Bcast(&status, 1, MPI_INT, MPI_ROOT, intercomm);
   MPI_Bcast(&status, 1, MPI_INT, root, intercomm);
   MPI_Barrier(comm);
   MPI_Comm_free(&intercomm);
   return status;
}

#undef MFEM_JIT_SHOW_MAIN_CODE
#if defined(MFEM_JIT_MAIN) || defined(MFEM_JIT_SHOW_MAIN_CODE)

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
   while (*status != SYSTEM_CALL && timeout > 0) { nsleep(timeout--); }
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
#endif // MFEM_JIT_MAIN

#endif // MFEM_USE_MPI

// *****************************************************************************
int GetCurrentRuntimeVersion(bool increment)
{
   static int version = 0;
   const int actual = version;
   if (increment) { version += 1; }
   return actual;
}

// *****************************************************************************
/// Compile the source file with PIC flags, updating the cache library.
bool Compile(const char *imem,
             char *&omem,
             char cc[21],
             char co[21],
             const char *cxx,
             const char *flags,
             const char *msrc,
             const char *mins,
             const bool check_for_ar)
{
   dbg("imem: %p, omem: %p == nullptr", (void*)imem, (void*)omem);
   assert(omem == nullptr);
   //dbg("%s",imem);
#ifndef MFEM_USE_CUDA
#define MFEM_JIT_DEVICE_CODE
#define MFEM_JIT_COMPILER_OPTION
#define MFEM_JIT_LINKER_OPTION "-Wl,"
#else
   // Compile each input file into an object file
   // that contains relocatable device code
#define MFEM_JIT_DEVICE_CODE "--device-c"
#define MFEM_JIT_LINKER_OPTION "-Xlinker="
#define MFEM_JIT_COMPILER_OPTION "-Xcompiler="
#endif

#ifndef __APPLE__
   constexpr const char *beg_load = MFEM_JIT_LINKER_OPTION "--whole-archive";
   constexpr const char *end_load = MFEM_JIT_LINKER_OPTION "--no-whole-archive";
#else
   constexpr const char *beg_load = "-all_load";
   constexpr const char *end_load = "";
#endif

   constexpr int PM = PATH_MAX;
   constexpr const char *fPIC = MFEM_JIT_COMPILER_OPTION "-fPIC";
   constexpr const char *shell = MFEM_JIT_SHELL_COMMAND;
   constexpr const char *libar = MFEM_JIT_CACHE ".a";
   constexpr const char *libso = MFEM_JIT_CACHE ".so";

   // If there is already a JIT archive, use it and create the lib_so
   if (check_for_ar && GetCurrentRuntimeVersion() == 0 && std::fstream(libar))
   {
      dbg("Using JIT archive!");
      const char *argv_so[] =
      {
         shell, cxx, "-shared", "-o", libso, beg_load, libar, end_load,
         MFEM_JIT_LINKER_OPTION "-rpath,.", nullptr
      };
      if (mfem::jit::System(const_cast<char**>(argv_so)) != 0) { return false; }
      delete imem;
      return true;
   }

   // MFEM source path, include path & lib_so_v
   const int version = GetCurrentRuntimeVersion(true);
   char libso_v[PM], Imsrc[PM], Iminc[PM];
   if (snprintf(Imsrc, PM, "-I%s ", msrc) < 0) { return false; }
   if (snprintf(Iminc, PM, "-I%s/include ", mins) < 0) { return false; }
   if (snprintf(libso_v, PM, "%s.so.%d", MFEM_JIT_CACHE, version) < 0)
   { return false; }

#if !defined(MFEM_USE_MPI)

   dbg("Tokenizing MFEM_CXXFLAGS");
   regex reg("\\s+");
   string cxxflags(flags);
   auto cxxargv =
      vector<std::string>(
         std::sregex_token_iterator{begin(cxxflags), end(cxxflags), reg, -1},
         std::sregex_token_iterator{});
   //for (auto a : cxxargv) { std::cout << a << std::endl; }

   vector<const char *> argv;
   argv.push_back(cxx);
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
   argv.push_back("-I.");
   argv.push_back(Imsrc);
   argv.push_back(Iminc);
   argv.push_back("-o");
#ifdef __APPLE__
   argv.push_back("/dev/stdout");
#else
   argv.push_back(co);
#endif // __APPLE__
   argv.push_back("-");
   argv.push_back(nullptr);

   size_t nw = 0;
   if (mfem::jit::InMemCompile(imem, omem, nw,
                               const_cast<char**>(argv.data())) != 0)
   {
      return false;
   }
   delete imem;

#ifdef __APPLE__
   // saving on disk the object that is still in memory
   dbg("nw:%d",nw);
   assert(omem != nullptr && nw > 0);
   const mode_t mode = S_IRUSR | S_IWUSR;
   const int oflag = O_CREAT | O_RDWR | O_TRUNC;
   int co_fd = ::open(co, oflag, mode);
   const size_t written = ::write(co_fd, omem, nw);
   if (written != nw) { return perror("!write object"), false; }
   if (::close(co_fd) < 0) { return perror("!close object"), false; }
   free(omem); // done with realloc
#endif

#else
   // Compilation
   const char *argv_co[] =
   {
      shell, cxx, flags, fPIC, "-c",
#ifdef MFEM_USE_CUDA
      MFEM_JIT_DEVICE_CODE,
#endif
      Imsrc, Iminc, "-o", co, cc, nullptr
   };
   if (mfem::jit::System(const_cast<char**>(argv_co)) != 0) { return false; }
   if (!getenv("MFEM_NUNLINK")) { ::unlink(cc); }
#endif

   dbg("Update archive");
   const char *argv_ar[] = { shell, "ar", "-rv", libar, co, nullptr };
   if (mfem::jit::System(const_cast<char**>(argv_ar)) != 0) { return false; }
   if (!getenv("MFEM_NUNLINK")) { ::unlink(co); }

   dbg("Create shared library");
   const char *argv_so[] =
   {shell, cxx, "-shared", "-o", libso, beg_load, libar, end_load, nullptr};
   if (mfem::jit::System(const_cast<char**>(argv_so)) != 0) { return false; }

   dbg("Install shared library");
   const char *install[] =
#ifdef __APPLE__
   { shell, "install", libso, libso_v, nullptr };
#else
      { shell, "install", "--backup=none", libso, libso_v, nullptr };
#endif
   if (mfem::jit::System(const_cast<char**>(install)) != 0) { return false; }
   return true;
}

// *****************************************************************************
// * STRUCTS: argument_t, kernel_t, context_t and error_t
// *****************************************************************************
struct argument_t
{
   int default_value = 0;
   string type, name;
   bool is_ptr = false, is_amp = false, is_const = false,
        is_restrict = false, is_tpl = false, has_default_value = false;
   std::list<int> range;
   bool operator==(const argument_t &arg) { return name == arg.name; }
   argument_t() {}
   argument_t(string id): name(id) {}
};
typedef list<argument_t>::iterator argument_it;

// *****************************************************************************
struct forall_t { int d; string e, N, X, Y, Z, body; };

// *****************************************************************************
struct kernel_t
{
   bool is_jit;
   bool is_embed;
   bool is_forall;
   bool is_single_source;
   string mfem_cxx;           // holds MFEM_CXX
   string mfem_build_flags;   // holds MFEM_BUILD_FLAGS
   string mfem_source_dir;    // holds MFEM_SOURCE_DIR
   string mfem_install_dir;   // holds MFEM_INSTALL_DIR
   string name;               // kernel name
   string space;              // kernel namespace
   // Templates: format, arguments and parameters
   string Tformat;            // template format, as in printf
   string Targs;              // template arguments, for hash and call
   string Tparams;            // template parameters, for the declaration
   string Tparams_src;        // template parameters, from original source
   // Arguments and parameter for the standard calls
   // We need two kinds of arguments because of the '& <=> *' transformation
   // (This might be no more the case as we are getting rid of Array/Vector).
   string params;
   string args;
   string args_wo_amp;
   string d2u, u2d;           // double to unsigned place holders
   string embed;              // source of the embed function
   struct forall_t forall;    // source of the lambda forall
};

// *****************************************************************************
struct context_t
{
   kernel_t ker;
   istream& in;
   ostream& out;
   string& file;
   list<argument_t> args;
   int line, block, parenthesis;
public:
   context_t(istream& i, ostream& o, string &f)
      : in(i), out(o), file(f), line(1), block(-2), parenthesis(-2) {}
};

// *****************************************************************************
struct error_t
{
   int line;
   string file;
   const char *msg;
   error_t(int l, string f, const char *m): line(l), file(f), msg(m) {}
};

// *****************************************************************************
int help(char* argv[])
{
   std::cout << "MFEM mjit: ";
   std::cout << argv[0] << " -o output input" << std::endl;
   return ~0;
}

// *****************************************************************************
const char* strrnc(const char *s, const unsigned char c, int n =1)
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

// *****************************************************************************
void check(context_t &pp, const bool test, const char *msg = nullptr)
{ if (not test) { throw error_t(pp.line, pp.file,msg); } }

// *****************************************************************************
void addComa(string &arg) { if (not arg.empty()) { arg += ",";  } }

void addArg(string &list, const char *arg) { addComa(list); list += arg; }

// *****************************************************************************
bool is_newline(const int ch) { return static_cast<unsigned char>(ch) == '\n'; }

// *****************************************************************************
bool good(context_t &pp) { pp.in.peek(); return pp.in.good(); }

// *****************************************************************************
char get(context_t &pp) { return static_cast<char>(pp.in.get()); }

// *****************************************************************************
int put(const char c, context_t &pp)
{
   if (is_newline(c)) { pp.line++; }
   if (pp.ker.is_embed) { pp.ker.embed += c; }
   // if we are storing the lambda body, just save it w/o output
   if (pp.ker.is_forall) { pp.ker.forall.body += c; return c;}
   pp.out.put(c);
   return c;
}

// *****************************************************************************
int put(context_t &pp) { return put(get(pp),pp); }

// *****************************************************************************
void skip_space(context_t &pp, string &out)
{
   while (isspace(pp.in.peek())) { out += get(pp); }
}

// *****************************************************************************
void skip_space(context_t &pp, bool keep=true)
{
   while (isspace(pp.in.peek())) { keep?put(pp):get(pp); }
}

// *****************************************************************************
void drop_space(context_t &pp)
{
   while (isspace(pp.in.peek())) { get(pp); }
}

// *****************************************************************************
bool is_comments(context_t &pp)
{
   if (pp.in.peek() != '/') { return false; }
   pp.in.get();
   assert(!pp.in.eof());
   const int c = pp.in.peek();
   pp.in.unget();
   if (c == '/' || c == '*') { return true; }
   return false;
}

// *****************************************************************************
void singleLineComments(context_t &pp, bool keep=true)
{
   while (!is_newline(pp.in.peek())) { keep?put(pp):get(pp); }
}

// *****************************************************************************
void blockComments(context_t &pp, bool keep=true)
{
   for (char c; pp.in.get(c);)
   {
      if (keep) { put(c,pp); }
      if (c == '*' && pp.in.peek() == '/')
      {
         keep?put(pp):get(pp);
         skip_space(pp, keep);
         return;
      }
   }
}

// *****************************************************************************
void comments(context_t &pp, bool keep=true)
{
   if (not is_comments(pp)) { return; }
   keep?put(pp):get(pp);
   if (keep?put(pp):get(pp) == '/') { return singleLineComments(pp,keep); }
   return blockComments(pp,keep);
}

// *****************************************************************************
void next(context_t &pp, bool keep=true)
{
   keep?skip_space(pp):drop_space(pp);
   comments(pp,keep);
}

// *****************************************************************************
void drop(context_t &pp)
{
   next(pp, false);
}

// *****************************************************************************
bool is_id(context_t &pp)
{
   const int c = pp.in.peek();
   return isalnum(c) or c == '_';
}

// *****************************************************************************
bool is_semicolon(context_t &pp)
{
   skip_space(pp);
   const int c = pp.in.peek();
   return c == ';';
}

// *****************************************************************************
string get_id(context_t &pp)
{
   string id;
   check(pp,is_id(pp),"name w/o alnum 1st letter");
   while (is_id(pp)) { id += get(pp); }
   return id;
}

// *****************************************************************************
bool is_digit(context_t &pp)
{ return isdigit(static_cast<char>(pp.in.peek())); }

// *****************************************************************************
int get_digit(context_t &pp)
{
   string digit;
   check(pp,is_digit(pp),"unknown number");
   while (is_digit(pp)) { digit += get(pp); }
   return atoi(digit.c_str());
}

// *****************************************************************************
string peekn(context_t &pp, const int n)
{
   int k = 0;
   assert(n < 64);
   static char c[64];
   for (k = 0; k <= n; k++) { c[k] = 0; }
   for (k = 0; k < n and good(pp); k++) { c[k] = get(pp); }
   string rtn(c);
   assert(!pp.in.fail());
   for (int l = 0; l < k; l++) { pp.in.unget(); }
   return rtn;
}

// *****************************************************************************
string peekid(context_t &pp)
{
   int k = 0;
   const int n = 64;
   static char c[64];
   for (k = 0; k < n; k++) { c[k] = 0; }
   for (k = 0; k < n; k++)
   {
      if (not is_id(pp)) { break; }
      c[k] = get(pp);
      assert(not pp.in.eof());
   }
   string rtn(c);
   for (int l = 0; l < k; l++) { pp.in.unget(); }
   return rtn;
}

// *****************************************************************************
void drop_name(context_t &pp) { while (is_id(pp)) { get(pp); } }

// *****************************************************************************
bool is_void(context_t &pp)
{
   skip_space(pp);
   const string void_peek = peekn(pp,4);
   assert(not pp.in.eof());
   if (void_peek == "void") { return true; }
   return false;
}

// *****************************************************************************
bool is_namespace(context_t &pp)
{
   skip_space(pp);
   const string namespace_peek = peekn(pp,2);
   assert(not pp.in.eof());
   if (namespace_peek == "::") { return true; }
   return false;
}

// *****************************************************************************
bool is_static(context_t &pp)
{
   skip_space(pp);
   const string void_peek = peekn(pp,6);
   assert(not pp.in.eof());
   if (void_peek == "static") { return true; }
   return false;
}

// *****************************************************************************
bool is_template(context_t &pp)
{
   skip_space(pp);
   const string void_peek = peekn(pp,8);
   assert(not pp.in.eof());
   if (void_peek == "template") { return true; }
   return false;
}

// *****************************************************************************
bool is_star(context_t &pp)
{
   skip_space(pp);
   if (pp.in.peek() == '*') { return true; }
   return false;
}

// *****************************************************************************
bool is_amp(context_t &pp)
{
   skip_space(pp);
   if (pp.in.peek() == '&') { return true; }
   return false;
}

// *****************************************************************************
bool is_left_parenthesis(context_t &pp)
{
   skip_space(pp);
   if (pp.in.peek() == '(') { return true; }
   return false;
}

// *****************************************************************************
bool is_right_parenthesis(context_t &pp)
{
   skip_space(pp);
   if (pp.in.peek() == ')') { return true; }
   return false;
}

// *****************************************************************************
bool is_coma(context_t &pp)
{
   skip_space(pp);
   if (pp.in.peek() == ',') { return true; }
   return false;
}

// *****************************************************************************
bool is_eq(context_t &pp)
{
   skip_space(pp);
   if (pp.in.peek() == '=') { return true; }
   return false;
}

// *****************************************************************************
// * MFEM_JIT
// *****************************************************************************
void jitHeader(context_t &pp)
{
   pp.out << "#include \"general/mjit.hpp\"\n";
   pp.out << "#line 1 \"" << pp.file <<"\"\n";
}

// *****************************************************************************
void ppKerDbg(context_t &pp)
{
   pp.ker.Targs += "\033[33mTargs\033[m";
   pp.ker.Tparams += "\033[33mTparams\033[m";
   pp.ker.Tformat += "\033[33mTformat\033[m";
   pp.ker.args += "\033[33margs\033[m";
   pp.ker.params += "\033[33mparams\033[m";
   pp.ker.args_wo_amp += "\033[33margs_wo_amp\033[m";
}

// *****************************************************************************
void jitArgs(context_t &pp)
{
   if (! pp.ker.is_jit) { return; }
   pp.ker.mfem_cxx = MFEM_JIT_STRINGIFY(MFEM_CXX);
   pp.ker.mfem_build_flags = MFEM_JIT_STRINGIFY(MFEM_BUILD_FLAGS);
   pp.ker.mfem_source_dir = MFEM_SOURCE_DIR;
   pp.ker.mfem_install_dir = MFEM_INSTALL_DIR;
   pp.ker.Targs.clear();
   pp.ker.Tparams.clear();
   pp.ker.Tformat.clear();
   pp.ker.args.clear();
   pp.ker.params.clear();
   pp.ker.args_wo_amp.clear();
   pp.ker.d2u.clear();
   pp.ker.u2d.clear();
   const bool single_source = pp.ker.is_single_source;
   //ppKerDbg(pp);
   //DBG("%s",single_source?"single_source":"");
   for (argument_it ia = pp.args.begin(); ia != pp.args.end() ; ia++)
   {
      const argument_t &arg = *ia;
      const bool is_const = arg.is_const;
      const bool is_amp = arg.is_amp;
      const bool is_ptr = arg.is_ptr;
      const bool is_pointer = is_ptr or is_amp;
      const char *type = arg.type.c_str();
      const char *name = arg.name.c_str();
      const bool has_default_value = arg.has_default_value;
      //DBG("\narg: %s %s %s%s", is_const?"const":"", type, is_pointer?"*|& ":"",name)
      // const and not is_pointer => add it to the template args
      if (is_const and not is_pointer and (has_default_value or not single_source))
      {
         //DBG(" => 1")
         const bool is_double = strcmp(type,"double")==0;
         // Tformat
         addComa(pp.ker.Tformat);
         if (! has_default_value)
         {
            pp.ker.Tformat += is_double ? "0x%lx" : "%d";
         }
         else
         {
            pp.ker.Tformat += "%d";
         }
         // Targs
         addComa(pp.ker.Targs);
         pp.ker.Targs += is_double?"u":"";
         pp.ker.Targs += is_pointer?"_":"";
         pp.ker.Targs += name;
         // Tparams
         if (!has_default_value)
         {
            addComa(pp.ker.Tparams);
            pp.ker.Tparams += "const ";
            pp.ker.Tparams += is_double?"uint64_t":type;
            pp.ker.Tparams += " ";
            pp.ker.Tparams += is_double?"t":"";
            pp.ker.Tparams += is_pointer?"_":"";
            pp.ker.Tparams += name;
         }
         if (is_double)
         {
            {
               pp.ker.d2u += "\n\tconst union_du_t union_";
               pp.ker.d2u += name;
               pp.ker.d2u += " = (union_du_t){u:t";
               pp.ker.d2u += is_pointer?"_":"";
               pp.ker.d2u += name;
               pp.ker.d2u += "};";

               pp.ker.d2u += "\n\tconst double ";
               pp.ker.d2u += is_pointer?"_":"";
               pp.ker.d2u += name;
               pp.ker.d2u += " = union_";
               pp.ker.d2u += name;
               pp.ker.d2u += ".d;";
            }
            {
               pp.ker.u2d += "\n\tconst uint64_t u";
               pp.ker.u2d += is_pointer?"_":"";
               pp.ker.u2d += name;
               pp.ker.u2d += " = (union_du_t){";
               pp.ker.u2d += is_pointer?"_":"";
               pp.ker.u2d += name;
               pp.ker.u2d += "}.u;";
            }
         }
      }

      //
      if (is_const and not is_pointer and not has_default_value and single_source)
      {
         //DBG(" => 2")
         addArg(pp.ker.args, name);
         addArg(pp.ker.args_wo_amp, name);
         addArg(pp.ker.params, "const ");
         pp.ker.params += type;
         pp.ker.params += " ";
         pp.ker.params += name;
      }

      // !const && !pointer => std args
      if (not is_const and not is_pointer)
      {
         //DBG(" => 3")
         addArg(pp.ker.args, name);
         addArg(pp.ker.args_wo_amp, name);
         addArg(pp.ker.params, type);
         pp.ker.params += " ";
         pp.ker.params += name;
      }
      //
      if (is_const and not is_pointer and has_default_value)
      {
         //DBG(" => 4")
         // other_parameters
         addArg(pp.ker.params, " const ");
         pp.ker.params += type;
         pp.ker.params += " ";
         pp.ker.params += name;
         // other_arguments_wo_amp
         addArg(pp.ker.args_wo_amp, "0");
         // other_arguments
         addArg(pp.ker.args, "0");
      }

      // pointer
      if (is_pointer)
      {
         //DBG(" => 5")
         // other_arguments
         if (! pp.ker.args.empty()) { pp.ker.args += ","; }
         pp.ker.args += is_amp?"&":"";
         pp.ker.args += is_pointer?"_":"";
         pp.ker.args += name;
         // other_arguments_wo_amp
         if (! pp.ker.args_wo_amp.empty()) {  pp.ker.args_wo_amp += ","; }
         pp.ker.args_wo_amp += is_pointer?"_":"";
         pp.ker.args_wo_amp += name;
         // other_parameters
         if (not pp.ker.params.empty()) { pp.ker.params += ",";  }
         pp.ker.params += is_const?"const ":"";
         pp.ker.params += type;
         pp.ker.params += " *";
         pp.ker.params += is_pointer?"_":"";
         pp.ker.params += name;
      }
   }
   if (pp.ker.is_single_source)
   {
      //DBG(" => 6")
      addComa(pp.ker.Tparams);
      pp.ker.Tparams += pp.ker.Tparams_src;
   }
}

// *****************************************************************************
void jitPrefix(context_t &pp)
{
   if (not pp.ker.is_jit) { return; }
   pp.out << "\n\tconst char *src=R\"_(";
   pp.out << "#include <cstdint>\n";
   pp.out << "#include <limits>\n";
   pp.out << "#include <cstring>\n";
   pp.out << "#include <stdbool.h>\n";
   pp.out << "#define MJIT_FORALL\n";
   pp.out << "#include \"/Users/camier1/home/sawmill/okina-jit/mfem/include/mfem/general/mjit.hpp\"\n";
   if (not pp.ker.embed.empty())
   {
      // push to suppress 'declared but never referenced' warnings
      pp.out << "\n#pragma push";
      pp.out << "\n#pragma diag_suppress 177\n";
      pp.out << pp.ker.embed.c_str();
      pp.out << "\n#pragma pop";
   }
   pp.out << "\nusing namespace mfem;\n";
   pp.out << "\ntemplate<" << pp.ker.Tparams << ">";
   pp.out << "\nvoid " << pp.ker.name << "_%016lx(";
   pp.out << "const bool use_dev,";
   pp.out << pp.ker.params << "){";
   if (not pp.ker.d2u.empty()) { pp.out << "\n\t" << pp.ker.d2u; }
   // Starts counting the block depth
   pp.block = 0;
}

// *****************************************************************************
void jitPostfix(context_t &pp)
{
   if (not pp.ker.is_jit) { return; }
   if (pp.block >= 0 && pp.in.peek() == '{') { pp.block++; }
   if (pp.block >= 0 && pp.in.peek() == '}') { pp.block--; }
   if (pp.block != -1) { return; }
   pp.out << "}\nextern \"C\" void "
          << MFEM_JIT_SYMBOL_PREFIX << "%016lx("
          << "const bool use_dev, " << pp.ker.params << "){";
   pp.out << pp.ker.name << "_%016lx<" << pp.ker.Tformat << ">"
          << "(" << "use_dev, " << pp.ker.args_wo_amp << ");";
   pp.out << "})_\";";
   // typedef, hash map and launch
   pp.out << "\n\ttypedef void (*kernel_t)(const bool use_dev, "
          << pp.ker.params << ");";
   pp.out << "\n\tstatic std::unordered_map<size_t,jit::kernel<kernel_t>*> ks;";
   if (not pp.ker.u2d.empty()) { pp.out << "\n\t" << pp.ker.u2d; }
   pp.out << "\n\tconst char *cxx = \"" << pp.ker.mfem_cxx << "\";";
   pp.out << "\n\tconst char *mfem_build_flags = \""
          << pp.ker.mfem_build_flags <<  "\";";
   pp.out << "\n\tconst char *mfem_source_dir = \""
          << pp.ker.mfem_source_dir <<  "\";";
   pp.out << "\n\tconst char *mfem_install_dir = \""
          << pp.ker.mfem_install_dir <<  "\";";
   pp.out << "\n\tconst size_t args_seed = std::hash<size_t>()(0);";
   pp.out << "\n\tconst size_t args_hash = jit::hash_args(args_seed,"
          << pp.ker.Targs << ");";
   pp.out << "\n\tif (!ks[args_hash]){";
   pp.out << "\n\t\tks[args_hash] = new jit::kernel<kernel_t>"
          << "(\"" << pp.ker.name << "\", "
          << "cxx, src, mfem_build_flags, mfem_source_dir, mfem_install_dir, "
          << pp.ker.Targs << ");";
   pp.out << "\n\t}";
   pp.out << "\n\tks[args_hash]->operator_void("
          << "Device::Allows(Backend::CUDA_MASK), "
          << pp.ker.args << ");\n";
   // Stop counting the blocks and flush the kernel status
   pp.block--;
   pp.ker.is_jit = false;
}

// *****************************************************************************
string arg_get_array_type(context_t &pp)
{
   string type;
   skip_space(pp);
   check(pp,pp.in.peek()=='<',"no '<' while in get_array_type");
   put(pp);
   type += "<";
   skip_space(pp);
   check(pp,is_id(pp),"no type found while in get_array_type");
   string id = get_id(pp);
   pp.out << id.c_str();
   type += id;
   skip_space(pp);
   check(pp,pp.in.peek()=='>',"no '>' while in get_array_type");
   put(pp);
   type += ">";
   return type;
}

// *****************************************************************************
bool jitGetArgs(context_t &pp)
{
   bool empty = true;
   argument_t arg;
   pp.args.clear();
   // Go to first possible argument
   skip_space(pp);

   if (is_void(pp)) { drop_name(pp); return true; }

   for (int p=0; true; empty=false)
   {
      if (is_star(pp))
      {
         arg.is_ptr = true;
         put(pp);
         continue;
      }
      if (is_amp(pp))
      {
         arg.is_amp = true;
         put(pp);
         continue;
      }
      if (is_coma(pp))
      {
         put(pp);
         continue;
      }
      if (is_left_parenthesis(pp))
      {
         p+=1;
         put(pp);
         continue;
      }
      const string &id = peekid(pp);
      drop_name(pp);
      // Qualifiers
      if (id=="const") { pp.out << id; arg.is_const = true; continue; }
      if (id=="restrict") { pp.out << id; arg.is_restrict = true; continue; }
      if (id=="__restrict") { pp.out << id; arg.is_restrict = true; continue; }
      // Types
      if (id=="char") { pp.out << id; arg.type = id; continue; }
      if (id=="int") { pp.out << id; arg.type = id; continue; }
      if (id=="short") { pp.out << id; arg.type = id; continue; }
      if (id=="unsigned") { pp.out << id; arg.type = id; continue; }
      if (id=="long") { pp.out << id; arg.type = id; continue; }
      if (id=="bool") { pp.out << id; arg.type = id; continue; }
      if (id=="float") { pp.out << id; arg.type = id; continue; }
      if (id=="double") { pp.out << id; arg.type = id; continue; }
      if (id=="size_t") { pp.out << id; arg.type = id; continue; }
      if (id=="Array")
      {
         pp.out << id; arg.type = id;
         arg.type += arg_get_array_type(pp);
         continue;
      }
      if (id=="Vector") { pp.out << id; arg.type = id; continue; }
      if (id=="DofToQuad") { pp.out << id; arg.type = id; continue; }
      const bool is_pointer = arg.is_ptr || arg.is_amp;
      const bool underscore = is_pointer;
      pp.out << (underscore?"_":"") << id;
      // focus on the name, we should have qual & type
      arg.name = id;
      // now check for a possible default value
      next(pp);
      if (is_eq(pp)) // found a default value after a '='
      {
         put(pp);
         next(pp);
         arg.has_default_value = true;
         arg.default_value = get_digit(pp);
         pp.out << arg.default_value;
      }
      else
      {
         // check if id has a T_id in pp.ker.Tparams_src
         string t_id("t_");
         t_id += id;
         std::transform(t_id.begin(), t_id.end(), t_id.begin(), ::toupper);
         // if we have a hit, fake it has_default_value to trig the args to <>
         if (pp.ker.Tparams_src.find(t_id) != string::npos)
         {
            arg.has_default_value = true;
            arg.default_value = 0;
         }
      }
      pp.args.push_back(arg);
      arg = argument_t();
      int c = pp.in.peek();
      assert(not pp.in.eof());
      if (c == ')') { p-=1; if (p>=0) { put(pp); continue; } }
      // end of the arguments
      if (p<0) { break; }
      check(pp, pp.in.peek()==',', "no coma while in args");
      put(pp);
   }
   // Prepare the kernel strings from the arguments
   jitArgs(pp);
   return empty;
}

// *****************************************************************************
void jitAmpFromPtr(context_t &pp)
{
   for (argument_it ia = pp.args.begin(); ia != pp.args.end() ; ia++)
   {
      const argument_t a = *ia;
      const bool is_const = a.is_const;
      const bool is_ptr = a.is_ptr;
      const bool is_amp = a.is_amp;
      const bool is_pointer = is_ptr || is_amp;
      const char *type = a.type.c_str();
      const char *name = a.name.c_str();
      const bool underscore = is_pointer;
      if (is_const && underscore)
      {
         pp.out << "\n\tconst " << type << (is_amp?"&":"*") << name
                << " = " <<  (is_amp?"*":"")
                << " _" << name << ";";
      }
      if (!is_const && underscore)
      {
         pp.out << "\n\t" << type << (is_amp?"&":"*") << name
                << " = " << (is_amp?"*":"")
                << " _" << name << ";";
      }
   }
}

// *****************************************************************************
void jit(context_t &pp)
{
   pp.ker.is_jit = true;
   next(pp);
   // return type should be void for now, or we could hit a 'static'
   // or even a 'template' which triggers the '__single_source' case
   const bool check_next_id = is_void(pp) or is_static(pp) or is_template(pp);
   // first check for the template
   check(pp,  check_next_id, "kernel w/o void, static or template");
   if (is_template(pp))
   {
      // copy the 'template<...>' in Tparams_src
      pp.out << get_id(pp);
      // tag our kernel as a '__single_source' one
      pp.ker.is_single_source = true;
      next(pp);
      check(pp, pp.in.peek()=='<',"no '<' in single source kernel!");
      put(pp);
      pp.ker.Tparams_src.clear();
      while (pp.in.peek() != '>')
      {
         assert(not pp.in.eof());
         char c = get(pp);
         put(c,pp);
         pp.ker.Tparams_src += c;
      }
      put(pp);
   }
   // 'static' check
   if (is_static(pp)) { pp.out << get_id(pp); }
   next(pp);
   const string void_return_type = get_id(pp);
   pp.out << void_return_type;
   // Get kernel's name or namespace
   pp.ker.name.clear();
   pp.ker.space.clear();
   next(pp);
   const string name = get_id(pp);
   pp.out << name;
   pp.ker.name = name;
   if (is_namespace(pp))
   {
      check(pp,pp.in.peek()==':',"no 1st ':' in namespaced kernel");
      put(pp);
      check(pp,pp.in.peek()==':',"no 2st ':' in namespaced kernel");
      put(pp);
      const string real_name = get_id(pp);
      pp.out << real_name;
      pp.ker.name = real_name;
      pp.ker.space = name;
   }
   next(pp);
   // check we are at the left parenthesis
   check(pp,pp.in.peek()=='(',"no 1st '(' in kernel");
   put(pp);
   // Get the arguments
   jitGetArgs(pp);
   // Make sure we have hit the last ')' of the arguments
   check(pp,pp.in.peek()==')',"no last ')' in kernel");
   put(pp);
   next(pp);
   // Make sure we are about to start a compound statement
   check(pp,pp.in.peek()=='{',"no compound statement found");
   put(pp);
   // Generate the kernel prefix for this kernel
   jitPrefix(pp);
   // Generate the & <=> * transformations
   jitAmpFromPtr(pp);
   // Push the right #line directive
   pp.out << "\n#line " << pp.line
          << " \"" //<< pp.ker.mfem_source_dir << "/"
          << pp.file << "\"";
}

// *****************************************************************************
// * MFEM_EMBED
// *****************************************************************************
void embed(context_t &pp)
{
   pp.ker.is_embed = true;
   // Goto first '{'
   while ('{' != put(pp));
   // Starts counting the compound statements
   pp.block = 0;
}

// *****************************************************************************
void embedPostfix(context_t &pp)
{
   if (not pp.ker.is_embed) { return; }
   if (pp.block>=0 && pp.in.peek() == '{') { pp.block++; }
   if (pp.block>=0 && pp.in.peek() == '}') { pp.block--; }
   if (pp.block!=-1) { return; }
   check(pp,pp.in.peek()=='}',"no compound statements found");
   put(pp);
   pp.block--;
   pp.ker.is_embed = false;
   pp.ker.embed += "\n";
}

// *****************************************************************************
// * MFEM_UNROLL
// *****************************************************************************
void unroll(context_t &pp)
{
   //DBG("__unroll")
   while ('(' != get(pp)) {assert(not pp.in.eof());}
   drop(pp);
   string depth = get_id(pp);
   //DBG("(%s)",depth.c_str());
   drop(pp);
   check(pp,is_right_parenthesis(pp),"no last right parenthesis found");
   get(pp);
   drop(pp);
   check(pp,is_semicolon(pp),"no last semicolon found");
   get(pp);
   // only if we are in a forall, we push the unrolling
   if (pp.ker.is_forall)
   {
      pp.ker.forall.body += "#pragma unroll ";
      pp.ker.forall.body += depth.c_str();
   }
}

// *****************************************************************************
// * MFEM_FORALL_[2|3]D
// *****************************************************************************
void forall(const string &id, context_t &pp)
{
   const int d = pp.ker.forall.d = id.c_str()[12] - 0x30;
   if (not pp.ker.is_jit)
   {
      //DBG("id:%s, d:%d",id.c_str(),d)
      if (d == 2) { pp.out << "MFEM_FORALL_2D"; }
      if (d == 3) { pp.out << "MFEM_FORALL_3D"; }
      return;
   }
   //DBG("is_forall")
   pp.ker.is_forall = true;
   pp.ker.forall.body.clear();

   check(pp,is_left_parenthesis(pp),"no 1st '(' in MFEM_FORALL");
   get(pp); // drop '('
   pp.ker.forall.e = get_id(pp);
   //DBG("iterator:'%s'", pp.ker.forall.e.c_str());

   check(pp,is_coma(pp),"no 1st coma in MFEM_FORALL");
   get(pp); // drop ','

   drop(pp);
   check(pp,is_id(pp),"no 1st id(N) in MFEM_FORALL");
   pp.ker.forall.N = get_id(pp);
   //DBG("N:'%s'", pp.ker.forall.N.c_str());
   drop(pp);
   check(pp,is_coma(pp),"no 2nd coma in MFEM_FORALL");
   get(pp); // drop ','

   drop(pp);
   check(pp,is_id(pp),"no 2st id (X) in MFEM_FORALL");
   pp.ker.forall.X = get_id(pp);
   //DBG("X:'%s'", pp.ker.forall.X.c_str());
   drop(pp);
   //DBG(">%c<", put(pp));
   check(pp,is_coma(pp),"no 3rd coma in MFEM_FORALL");
   get(pp); // drop ','

   drop(pp);
   check(pp,is_id(pp),"no 3rd id (Y) in MFEM_FORALL");
   pp.ker.forall.Y = get_id(pp);
   //DBG("Y:'%s'", pp.ker.forall.Y.c_str());
   drop(pp);
   check(pp,is_coma(pp),"no 4th coma in MFEM_FORALL");
   get(pp); // drop ','

   drop(pp);
   check(pp,is_id(pp),"no 4th id (Y) in MFEM_FORALL");
   pp.ker.forall.Z = get_id(pp);
   //DBG("Z:'%s'", pp.ker.forall.Z.c_str());
   drop(pp);
   check(pp,is_coma(pp),"no last coma in MFEM_FORALL");
   get(pp); // drop ','

   // Starts counting the parentheses
   pp.parenthesis = 0;
}
// *****************************************************************************
void forallPostfix(context_t &pp)
{
   if (not pp.ker.is_forall) { return; }
   //DBG("forallPostfix 1")
   if (pp.parenthesis >= 0 && pp.in.peek() == '(') { pp.parenthesis++; }
   if (pp.parenthesis >= 0 && pp.in.peek() == ')') { pp.parenthesis--; }
   if (pp.parenthesis != -1) { return; }
   //DBG("forall2ostfix 2")
   drop(pp);
   check(pp,is_right_parenthesis(pp),"no last right parenthesis found");
   get(pp);
   drop(pp);
   check(pp,is_semicolon(pp),"no last semicolon found");
   get(pp);
   pp.parenthesis--;
   pp.ker.is_forall = false;
#ifdef MFEM_USE_CUDA
   pp.out << "if (use_dev){";
   const char *ND = pp.ker.forall.d == 2 ? "2D" : "3D";
   pp.out << "\n\tCuWrap" << ND << "(" << pp.ker.forall.N.c_str() << ", ";
   pp.out << "[=] MFEM_DEVICE (int " << pp.ker.forall.e <<")";
   pp.out << pp.ker.forall.body.c_str() << ",";
   pp.out << pp.ker.forall.X.c_str() << ",";
   pp.out << pp.ker.forall.Y.c_str() << ",";
   pp.out << pp.ker.forall.Z.c_str() << ");";
   pp.out << "\n} else {";
   pp.out << "for (int k=0; k<" << pp.ker.forall.N.c_str() << ";k++) {";
   pp.out << "[&] (int " << pp.ker.forall.e <<")";
   pp.out << pp.ker.forall.body.c_str() << "(k);";
   pp.out << "}";
   pp.out << "}";
#else
   pp.out << "for (int " << pp.ker.forall.e << " = 0; "
          << pp.ker.forall.e << " < " << pp.ker.forall.N.c_str() << "; "
          << pp.ker.forall.e<<"++) {";
   pp.out << pp.ker.forall.body.c_str();
   pp.out << "}";
#endif
}

// *****************************************************************************
static bool token(const string &id, const char *token)
{
   if (strncmp(id.c_str(), "MFEM_", 5) != 0 ) { return false; }
   if (strcmp(id.c_str() + 5, token) != 0 ) { return false; }
   return true;
}

// *****************************************************************************
static void tokens(context_t &pp)
{
   if (peekn(pp, 4) != "MFEM") { return; }
   const string &id = get_id(pp);
   if (token(id, "JIT")) { return jit(pp); }
   if (token(id, "EMBED")) { return embed(pp); }
   if (token(id, "UNROLL")) { return unroll(pp); }
   if (token(id, "FORALL_2D")) { return forall(id,pp); }
   if (token(id, "FORALL_3D")) { return forall(id,pp); }
   if (pp.ker.is_embed) { pp.ker.embed += id; }
   // During the forall body, add MFEM_* id tokens
   if (pp.ker.is_forall) { pp.ker.forall.body += id; return; }
   pp.out << id;
}

// *****************************************************************************
inline bool eof(context_t &pp)
{
   const char c = get(pp);
   if (pp.in.eof()) { return true; }
   put(c,pp);
   return false;
}

// *****************************************************************************
int preprocess(context_t &pp)
{
   jitHeader(pp);
   pp.ker.is_jit = false;
   pp.ker.is_embed = false;
   pp.ker.is_forall = false;
   pp.ker.is_single_source = false;
   do
   {
      tokens(pp);
      comments(pp);
      jitPostfix(pp);
      embedPostfix(pp);
      forallPostfix(pp);
   }
   while (not eof(pp));
   return 0;
}

} // namespace jit

} // namespace mfem

#if defined(MFEM_JIT_MAIN) || defined(MFEM_JIT_SHOW_MAIN_CODE)
int main(const int argc, char* argv[])
{
   string input, output, file;

   if (argc <= 1) { return mfem::jit::help(argv); }

   for (int i = 1; i < argc; i++)
   {
      // -h lauches help
      if (argv[i] == string("-h")) { return mfem::jit::help(argv); }

      // -c wil launch ProcessFork in parallel mode, nothing otherwise
      if (argv[i] == string(MFEM_JIT_SHELL_COMMAND))
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

#endif // MFEM_JIT_MAIN

