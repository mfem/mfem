// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

///////////////////////////////////////////////////////////////////////////////
// Environment variables:
//   - MM_TRACE: trace functions: malloc, free, calloc, realloc, memalign
//   - MM_DEBUG: print debug information
//   - MM_CHECK: check memory manager
//   - MM_ALL: print full backtrace
//   - MM_ARGS: print function arguments
//   - MM_ORG: switch to org mode, print '*' instead of ' '

///////////////////////////////////////////////////////////////////////////////
// On macOS, we need to set the following flags to allow dynamic lookup,
// 'dsymutil ex1' is also required to generate the debug symbols
// - -g -O1 -fno-inline-functions -fno-omit-frame-pointer
// - -Wl,-undefined,dynamic_lookup

///////////////////////////////////////////////////////////////////////////////
// For example 1:
// - add '#include <miniapps/tools/mem_check.hpp>'
// - add 'mfem::MemoryManagerCheck::Init(argv[0]);'
// - make ex1 && dsymutil ex1
// - tput reset && MM_TRACE=1 MM_DEBUG=1 DYLD_INSERT_LIBRARIES=../miniapps/tools/mem_check.dylib ./ex1

#include <cassert>
#include <cstdlib>
#include <cxxabi.h>
#include <dlfcn.h>
#include <iostream>
#include <cstring>

#include <backtrace.h>
#include <backtrace-supported.h>

#if BACKTRACE_SUPPORTED != 1
#error "Backtrace not supported! See output file backtrace-supported.h for details."
#endif

#include "mem_check.hpp"

#include "config/config.hpp" // IWYU pragma: keep
#include "general/mem_manager.hpp"

// #undef NVTX_COLOR
// #define NVTX_COLOR nvtx::kLawnGreen
// #include "general/nvtx.hpp"
#define dbg(...)

///////////////////////////////////////////////////////////////////////////////
// static helper functions
static const char *strrnchr(const char *s, const unsigned char c, int n)
{
   assert(s);
   size_t len = strlen(s);
   char *p = (char *)s + len - 1;
   for (; n; n--, p--, len--)
   {
      for (; len; p--, len--) { if (*p == c) { break; } }
      if (!len) { return nullptr; }
      if (n == 1) { return p; }
   }
   return nullptr;
}

static const char *getFilename(const char *filename, const unsigned char delim,
                               const int n)
{
   assert(filename);
   const char *f = strrnchr(filename, delim, n);
   return f ? f + 1 : filename;
}

static void err_callback(void *data, const char *msg, int errnum)
{
   dbg("error: {} errnum: #{}", msg, errnum);
   assert(false);
}

static const char *cxx_demangle(const char *mangled_name)
{
   int status;
   assert(mangled_name);
   const char *demangled_name =
      abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
   const bool succeeded = status == 0;
   const bool memory_allocation_failure_occurred = status == -1;
   const bool one_argument_is_invalid = status == -3;
   assert(not one_argument_is_invalid);
   if (memory_allocation_failure_occurred)
   {
      printf("[demangle] memory_allocation_failure_occurred!");
      fflush(nullptr);
      assert(false);
   }
   return (succeeded) ? demangled_name : mangled_name;
}

///////////////////////////////////////////////////////////////////////////////
class backtrace_data
{
public:
   backtrace_state *state = nullptr;
   char *function = nullptr;
   const char *filename = nullptr;
   int lineno = -1;
   uintptr_t address = 0x0;
   bool dump = false;
   bool main = false;
   bool mm = false;
   bool mfem = false;
   bool got = false;
   int depth = 0;
   static constexpr int STACK_MAX = 32768;
   char stack[STACK_MAX];
   bool skip = false;

   backtrace_data(backtrace_state *state): state(state) {}

   ~backtrace_data() { if (function) { free((void*) function); } }

   void ini(const bool dmp = false)
   {
      if (function) { free((void*) function); }
      function = nullptr;
      lineno = -1;
      address = 0x0;
      dump = dmp;
      main = false;
      mm = false;
      mfem = false;
      got = false;
      depth = 0;
      stack[0] = 0;
      skip = false;
   }

   /// Returns 0 to continue tracing
   inline int continue_tracing() { return main ? 1 : 0; }

   void update(const char *demangled, uintptr_t PC,
               const char *path_name = nullptr, const int line = -1)
   {
      assert(demangled);
      // assert(path_name);
      // update the context depth
      depth += 1;
      dbg("{}", demangled);
      if (path_name) { dbg("{}:{}", path_name, line); }
      // check if we have reached the 'main' function
      main = !strncmp(demangled, "main", 4);
      dbg("main:\x1b[33m {}", main);
      // MFEM namespace test
      const bool mfem_namespace = strncmp(demangled, "mfem::", 6) == 0;
      dbg("mfem_namespace:\x1b[33m {}", mfem_namespace);
      mfem |= mfem_namespace;
      // Test if we are doing an allocation from inside the mfem::mem_manager
      // If it is the case, set 'skip' to true
      if (path_name && !skip)
      {
         const char *file = getFilename(path_name, '/', 2);
         dbg("file:\x1b[33m {}", file);
         const bool mm_hpp_file = strncmp(file, "general/mem_manager.hpp", 13) == 0;
         const bool mm_cpp_file = strncmp(file, "general/mem_manager.cpp", 13) == 0;
         // We want to filter out the allocations done in the mem_manager
         mm |= mm_cpp_file or mm_hpp_file;
         dbg("mem_manager:\x1b[33m {}", mm);
         // Skip where the 'New' and 'Delete' are done
         // Not sure these are the ones to skip!! ❌❌
         const size_t len = std::strlen(demangled);
         const char *new_suffix = ">::New(int, mfem::MemoryType)";
         const char *del_suffix = ">::Delete(int, mfem::MemoryType)";
         const size_t new_suffix_len = strlen(new_suffix);
         const size_t del_suffix_len = strlen(del_suffix);
         const bool New = (new_suffix_len <= len) &&
                          strncmp(demangled + len - new_suffix_len, new_suffix,
                                  new_suffix_len) == 0;
         const bool Delete = (del_suffix_len <= len) &&
                             strncmp(demangled + len - del_suffix_len, del_suffix,
                                     del_suffix_len) == 0;
         skip |= New or Delete;
         dbg("skip:\x1b[33m {}", skip);
      }
      // Record the stack
      static char path_file[PATH_MAX];
      const int n_char_printed =
         snprintf(path_file, PATH_MAX, "\n\t%s %s:%d", demangled,
                  path_name ? path_name : "???", line);
      assert(n_char_printed < STACK_MAX);
      dbg("path_file:\x1b[33m {}", path_file);
      strcat(stack, path_file);
      // if we already have caught the function name, return
      if (function != nullptr)
      {
         assert(filename); // we should have recorded the filename
         // assert(lineno >= 0); // and set the line number
         dbg("function:\x1b[33m {}, returning", function);
         return;
      }
      function = strdup(demangled);
      // assert(path_name);
      filename = path_name ? strdup(path_name) : strdup("???");
      address = PC, lineno = line;
      dbg("\x1b[31m[{}] {}:{}", filename, function, line);
   }
};

///////////////////////////////////////////////////////////////////////////////
class backrace_check
{
   backtrace_data *data {};
   backtrace_state *state {};
   // number of frames to skip:
   // '1' will go down to 'mm_alloc'
   // '3' should be sufficient
   static constexpr int SKIP = 3;

public:
   backrace_check()  = default;

   ~backrace_check() { delete data; }

   void ini(const char *argv0)
   {
      assert(argv0);
      state = backtrace_create_state(argv0,
                                     BACKTRACE_SUPPORTS_THREADS,
                                     err_callback, nullptr);
      data = new backtrace_data(state);
   }

   int backtrace(const bool dump = false)
   {
      if (!state or !data) { return EXIT_FAILURE; } // not ready
      data->ini(dump); // flush the data
      // int code =
      backtrace_simple(state, SKIP, simple_callback, err_callback, data);
      // printf("\x1b[33m[%d]\x1b[m",code);
      return EXIT_SUCCESS;
   }

private:
   static void sym_callback(void *data, uintptr_t pc, const char *symname,
                            uintptr_t symval, uintptr_t symsize)
   {
      if (!symname) { return; }
      auto *ctx = static_cast<backtrace_data *>(data);
      const char *demangled = cxx_demangle(symname);
      ctx->update(demangled, pc);
   }

   static int filter(const bool debug, const char *demangled)
   {
      if (!debug) { return 0; }
      assert(demangled);
      printf("\nFiltering OUT '%s'!\n", demangled);
      return 0;
   }

   static int full_callback(void *data, uintptr_t pc, const char *filename,
                            int lineno, const char *function)
   {
      dbg("filename: {}, lineno: {}, function: {}",
          filename ? filename : "?",
          lineno,
          function ? function : "?");
      static const bool all = getenv("MM_ALL");
      auto *ctx = static_cast<backtrace_data *>(data);
      const bool debug = ctx->dump or all;

      if (!function) // symbol hit
      {
         if (debug)
         {
            printf("\n[full_callback:symbol] filename:%s, lineno=%d, pc=0x%lx",
                   filename ? filename : "???", lineno, pc);
         }
         return backtrace_syminfo(ctx->state, pc, sym_callback, err_callback,
                                  data);
      }

      const char *demangled = cxx_demangle(function);
      assert(demangled);

      if (debug)
      {
         printf("\n\t\x1b[32m[full_callback:function] %s\x1b[m", demangled);
      }

      // Filtering
      if (strncmp("std::", demangled, 5) == 0) { return filter(all, demangled); }
      if (strncmp("__gnu_cxx::", demangled, 11) == 0) { return filter(all, demangled); }

      // Debug if ALL
      if (debug) { printf("\n\t"); }

      // Update context
      assert(filename);
      ctx->update(demangled, pc, filename, lineno);

      if (debug) { printf("%s:%d %s", filename, lineno, demangled); }
      return EXIT_SUCCESS;
   }

   static int simple_callback(void *data, uintptr_t pc)
   {
      auto *ctx = static_cast<backtrace_data *>(data);
      backtrace_pcinfo(ctx->state, pc, full_callback, err_callback, data);
      return ctx->continue_tracing(); // returns 0 to continue tracing
   }

public:
   bool mm() { return data->mm; }
   bool mfem() { return data->mfem; }
   bool skip() { return data->skip; }
   int depth() { return data->depth; }
   uintptr_t address() { return data->address; }
   const char* function() { return data->function; }
   const char* filename() { return data->filename; }
   int lineno() { return data->lineno; }
   char *stack() { return data->stack; }
};

///////////////////////////////////////////////////////////////////////////////
using mm_stack_t = std::unordered_map<const void *, const char *>;
static mm_stack_t *mm_stack_map = nullptr;

static backrace_check *bt_check = nullptr;
static bool trace = false, hooked = false, dlsymd = false;

namespace mfem
{

///////////////////////////////////////////////////////////////////////////////
// MemoryManagerCheck
MemoryManagerCheck::MemoryManagerCheck(const char *argv0)
{
   hooked = false;
   dbg();
   bt_check = new backrace_check();
   mm_stack_map = new mm_stack_t();
   bt_check->ini(argv0);
   hooked = true;
}

///////////////////////////////////////////////////////////////////////////////
MemoryManagerCheck::~MemoryManagerCheck()
{
   hooked = false;
   dbg();
   delete bt_check, bt_check = nullptr;
   delete mm_stack_map, mm_stack_map = nullptr;
   hooked = true;
}

} // namespace mfem

///////////////////////////////////////////////////////////////////////////////
static void mmCheck(const void *ptr, const bool is_new, const bool dump)
{
   static const bool all = getenv("MM_ALL") != nullptr;
   static const bool debug = getenv("MM_DEBUG") != nullptr;
   static const bool mfem_debug = getenv("MFEM_DEBUG") != nullptr;
   static const bool args = getenv("MM_ARGS") != nullptr;
   static const bool org_mode = getenv("MM_ORG") != nullptr;
   static const bool mm_check = getenv("MM_CHECK") != nullptr;

   const auto is_del = not is_new;
   const auto tab = org_mode ? "*" : "  ";

   // now backtracing if ready
   if (!bt_check) { return; }

   if (bt_check->backtrace(dump) != EXIT_SUCCESS) { return; }

   // we have the stack, checking the backtrace
   dbg();

   const bool from_memory_manager = bt_check->mm();     // memory manager
   dbg("from_memory_manager:\x1b[35m {}", from_memory_manager);
   const bool mfem_namespace = bt_check->mfem();    // mfem namespace
   dbg("mfem_namespace:\x1b[35m {}", mfem_namespace);
   const bool skip = bt_check->skip();
   dbg("skip:\x1b[35m {}", skip);
   const int depth = bt_check->depth();
   const int frames = depth - (org_mode ? -1 : 1);
   const auto address = bt_check->address();
   const char *function = bt_check->function() ? bt_check->function() : "???";
   const char *filename = bt_check->filename() ? bt_check->filename() : "???";
   const int lineno = bt_check->lineno();
   // dbg("\x1b[35m[{}] {}:{}", filename, function, lineno);

   const std::string demangled_function(function);
   const size_t first_parenthesis = demangled_function.find_first_of('(');
   const std::string no_args_demangled_function =
      demangled_function.substr(0, first_parenthesis);
   const std::string display_function =
      args ? demangled_function : no_args_demangled_function;
   dbg("display_function:\x1b[35m {}", display_function);

   const size_t first_3A = display_function.find_first_of(':');
   const size_t first_3C = display_function.find_first_of('<');
   const size_t first_5B = display_function.find_first_of('[');
   assert(first_3A <= (first_5B < 0) ? first_3A : first_5B);
   const size_t first_3AC = ((first_3A ^ first_3C) < 0)
                            ? std::max(first_3A, first_3C)
                            : std::min(first_3A, first_3C);
   const std::string root = (first_3A != first_3C)
                            ? display_function.substr(0, first_3AC)
                            : display_function;
   dbg("root:\x1b[35m {}", root);
   const int color = address % (256 - 46) + 46;

   if (debug && !mfem_debug)
   {
      if (all) { std::cout << std::endl; }
      // Generating tabs
      for (int k = 0; k < frames; ++k) { std::cout << tab; }
      // Bold outputing
      if (!org_mode) { std::cout << "\x1b[38;5;" << color << ";1m"; }
      else { std::cout << " "; }
      mfem::out << "[" << (filename ? filename : "")
                << ":" << lineno
                << ":" << display_function
                << "]\x1b[m";
   }

   if (skip and debug) { printf(" skip!"); }
   if (skip or not mm_check) { return; }

   assert(ptr);

   // should return false when the memory manager does not exist,
   // or is not configured and also when then maps are not valid.
   const bool known = mfem::mm.IsKnown((void *)ptr);

   dbg("known:\x1b[35m {}", known);

   if (debug) { printf(" %sMFEM", mfem_namespace ? "" : "Not "); }

   static auto error = [](const void *ptr)
   {
      assert(bt_check), assert(mm_stack_map);
      printf("\nStack:%s\n", bt_check->stack());
      printf("\nFirst:%s\n", mm_stack_map->at(ptr));
      fflush(nullptr);
      assert(false);
   };

   // all this logic needs to be adapted to the new memory manager
   if (mfem_namespace)
   {
      dbg("in mfem namespace");
      if (from_memory_manager)
      {
         dbg("in mfem namespace, from memory manager");
         if (known and is_new)
         {
            printf("\nTrying to 'insert' (%p), known by the MM!", ptr);
            return error(ptr);
         }
         else if (not known and is_del) // when device is not configured ?!
         {
            printf("\nTrying to 'erase' (%p), not known by the MM!", ptr);
            return error(ptr);
         }
         else
         {
            if (debug) { printf(", known: ok"); }
         }
      }
      else // not from memory manager
      {
         dbg("in mfem namespace, not from memory manager");
         if (debug) { printf(", !MM"); }
         if (known and is_new)
         {
            printf("\nTrying to 'new' (%p), known by the MM!", ptr);
            return error(ptr);
         }
         else if (known and is_del)
         {
            printf("\nTrying to 'delete' (%p), known by the MM!", ptr);
            return error(ptr);
         }
         else if (not known)
         {
            // Should we allow user to do allocations ?!
            printf("\nTrying to new/del (%p), not known by the MM!", ptr);
            return error(ptr);
         }
         else
         {
            if (debug) { printf(" unknown: ok/error ?"); }
            return error(ptr); // 🔥🔥🔥 return error for now
         }
      }
   }
   else // not mfem namespace
   {
      // from memory manager but not mfem namespace: possible ? 🔥
      if (from_memory_manager)
      {
         if (known and is_new)
         {
            printf("\nTrying to 'insert' (%p), known by the MM!", ptr);
            return error(ptr);
         }
         else if (not known and is_del)
         {
            printf("Trying to 'erase' (%p), not known by the MM!", ptr);
            return error(ptr);
         }
         else
         {
            if (debug) { printf(" known: ok"); }
         }
      }
      else // not from memory manager, not mfem namespace
      {
         if (known and is_new)
         {
            printf("\nTrying to 'new' (%p), known by the MM!", ptr);
            return error(ptr);
         }
         if (known and is_del)
         {
            printf("Trying to 'delete' (%p), known by the MM!", ptr);
            return error(ptr);
         }
         if (debug) { printf(", unknown: leave it"); } // 🔥 to the user
      }
   }

   static auto mmAdd = [](const void *ptr,
                          const bool is_new,
                          const char *stack)
   {
      assert(ptr);
      assert(stack);
      const bool known = mm_stack_map->find(ptr) != mm_stack_map->end();
      if (is_new and not known)
      {
         dbg("Add to stack map");
         mm_stack_map->emplace(ptr, strdup(stack));
      }
      if (known and !is_new) // 🔥 to check ?!
      {
         dbg("Remove from stack map");
         mm_stack_map->erase(ptr);
      }
   };

   mmAdd(ptr, is_new, bt_check->stack());
   fflush(nullptr);
   // assert(false);
}

///////////////////////////////////////////////////////////////////////////////
// Linux: LD_PRELOAD + dlsym(RTLD_NEXT, ...)
// MacOS: DYLD_INSERT_LIBRARIES + DYLD_INTERPOSE
// Apple macOS specifics to interpose functions, instead of dlsym NEXT
#ifdef __APPLE__
#define DYLD_INTERPOSE(_to,_from) \
   __attribute__((used)) static struct{ const void* to; const void* from; } \
   _interpose_##_from __attribute__ ((section ("__DATA,__interpose"))) = \
   { (const void*)(unsigned long)&_to, (const void*)(unsigned long)&_from };
#else
#define DYLD_INTERPOSE(...)
#endif

///////////////////////////////////////////////////////////////////////////////
// LD_PRELOADed functions: malloc, free, calloc, realloc and memalign
using free_t = void (void *);
using malloc_t = void *(size_t);
using calloc_t = void *(size_t, size_t);
using realloc_t = void *(void *, size_t);
#ifdef __APPLE__
using memalign_t = int (void **memptr, size_t alignment, size_t size);
#else
using memalign_t = void *(size_t, size_t);
#endif
using mm_t = std::unordered_map<void *, size_t>;

static free_t *_free = nullptr;
static malloc_t *_malloc = nullptr;
static calloc_t *_calloc = nullptr;
static realloc_t *_realloc = nullptr;
static memalign_t *_memalign = nullptr;

///////////////////////////////////////////////////////////////////////////////
// With APPLE, don't dlsym the functions, as it will use the interposed one
static void _init()
{
   if (getenv("MM_TRACE")) { trace = true; }
#ifdef __APPLE__ // use the system ones
   _free = (free_t *)free, assert(_free);
   _malloc = (malloc_t *)malloc, assert(_malloc);
   _calloc = (calloc_t *)calloc, assert(_calloc);
   _realloc = (realloc_t *)realloc, assert(_realloc);
   _memalign = (memalign_t *)posix_memalign, assert(_memalign);
#else
   _free = (free_t *)dlsym(RTLD_NEXT, "free"), assert(_free);
   _malloc = (malloc_t *)dlsym(RTLD_NEXT, "malloc"), assert(_malloc);
   _calloc = (calloc_t *)dlsym(RTLD_NEXT, "calloc"), assert(_calloc);
   _realloc = (realloc_t *)dlsym(RTLD_NEXT, "realloc"), assert(_realloc);
   _memalign = (memalign_t *)dlsym(RTLD_NEXT, "memalign"), assert(_memalign);
#endif
   assert(_free and _malloc and _calloc and _realloc and _memalign);
   hooked = true, dlsymd = true;
}

// ALLOC //////////////////////////////////////////////////////////////////////
#ifdef __APPLE__
void *mm_malloc(size_t size) // Red
#else
void *malloc(size_t size) // Red
#endif
{
   if (!_malloc) { _init(); }
   if (!hooked) { return _malloc(size); }
   hooked = false;
   void *ptr = _malloc(size);
   assert(ptr);
   if (trace) { printf("\n\x1b[31m[malloc] %p (%ld)\x1b[m", ptr, size); }
   mmCheck(ptr, true, false); // new, dont show full stack
   hooked = true;
   return ptr;
}
DYLD_INTERPOSE(mm_malloc, malloc)

// FREE ///////////////////////////////////////////////////////////////////////
void mm_free(void *ptr) // Green
{
   if (!_free) { _init(); }
   if (!hooked) { return _free(ptr); }
   if (!ptr) { return; }
   hooked = false;
   if (trace) { printf("\n\x1b[32m[free] %p\x1b[m", ptr); }
   mmCheck(ptr, false, false); // delete, dont show full stack
   _free(ptr);
   hooked = true;
}
DYLD_INTERPOSE(mm_free, free)

// CALLOC /////////////////////////////////////////////////////////////////////
void *mm_calloc(size_t nmemb, size_t size) // Yellow
{
   if (not dlsymd) // if we are not yet dlsym'ed, just do it ourselves
   {
      static const size_t MEM_MAX = 8192;
      static char mem[MEM_MAX];
      static size_t m = 0;
      const size_t bytes = nmemb * size;
      void *ptr = &mem[m];
      m += bytes;
      assert(m < MEM_MAX);
      for (size_t k = 0; k < bytes; k += 1) { *(((char *)ptr) + k) = 0; }
      return ptr;
   }
   if (!hooked) { return _calloc(nmemb, size); }
   hooked = false;
   void *ptr = _calloc(nmemb, size);
   if (trace) { printf("\n\x1b[33m[calloc] %p (%ld)\x1b[m", ptr, size); }
   mmCheck(ptr, true, false); // new, dont show full stack
   hooked = true;
   return ptr;
}
DYLD_INTERPOSE(mm_calloc, calloc)

// REALLOC ////////////////////////////////////////////////////////////////////
void *mm_realloc(void *ptr, size_t size) // Blue
{
   if (!_realloc) { _init(); }
   if (!hooked) { return _realloc(ptr, size); }
   hooked = false;
   void *nptr = _realloc(ptr, size);
   assert(nptr);
   if (trace) { printf("\n\x1b[34;7m[realloc] %p(%ld)\x1b[m", nptr, size); }
   mmCheck(nptr, true, false); // new, dont show full stack
   hooked = true;
   return nptr;
}
DYLD_INTERPOSE(mm_realloc, realloc)

// MEMALIGN ///////////////////////////////////////////////////////////////////
#ifdef __APPLE__
int mm_memalign(void **memptr, size_t alignment, size_t size) // Magenta
#else
void *memalign(size_t alignment, size_t size) // Magenta
#endif
{
   if (!_memalign) { _init(); }
   if (!hooked)
   {
#ifdef __APPLE__
      return _memalign(memptr, alignment, size);
#else
      return _memalign(alignment, size);
#endif
   }
   hooked = false;
#ifdef __APPLE__
   const int rtn = _memalign(memptr, alignment, size);
   void *ptr = *memptr;
#else
   void *ptr = _memalign(alignment, size);
#endif
   assert(ptr);
   if (trace) { printf("\n\x1b[35;7m[memalign] %p(%ld)\x1b[m", ptr, size); }
   mmCheck(ptr, true, false); // new, dont show full stack
   hooked = true;
#ifdef __APPLE__
   return rtn;
#else
   return ptr;
#endif
}
DYLD_INTERPOSE(mm_memalign, posix_memalign)
