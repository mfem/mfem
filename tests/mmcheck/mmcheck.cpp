// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include <dlfcn.h>
#include <cassert>
#include <unordered_map>
#include "mmcheck.hpp"

// *****************************************************************************
static const char *getFilename(const char *filename,
                               const char delim,
                               const int n)
{
   const char *f=strrnchr(filename, delim, n);
   return f?f+1:filename;
}

// *****************************************************************************
void mmBackTraceData::update(const char* demangled, uintptr_t PC,
                             const char* filename, const int line)
{
   hit = !strncmp(demangled,"main",4);
   depth+=1;

   // MFEM namespace test
   const bool mfm = strncmp(demangled,"mfem::",6)==0;
   if (debug) { printf("%s", mfm?"mfem::":""); }
   mfem |= mfm;

   {
      // MM test
      if (filename)
      {
         const char *file = getFilename(filename,'/',2);
         const bool mm_hpp_file = strncmp(file,"general/mm.hpp",13)==0;
         const bool mm_cpp_file = strncmp(file,"general/mm.cpp",13)==0;
         const bool mm_namespace = strncmp(demangled,"mfem::mm",8)==0;
         // We do want the allocations, but NOT the ones done in the MM!
         // Allocations inside 'mm.cpp' trig (mfem and mm_cpp)
         mm |= mm_cpp_file or mm_hpp_file or mm_namespace;
         if (debug) { printf("%s", mm?"mm::":""); }
         // Skip where the 'emplace's are done
         const bool alias = strncmp(demangled,"InsertAlias",11)==0;
         const bool erase = strncmp(demangled,"mfem::mm::Erase",15)==0;
         const bool insert = strncmp(demangled,"mfem::mm::Insert",16)==0;
         skip |= insert or erase or alias;
         if (debug) { printf("%s", skip?"skip::":""); }
      }
   }
   {
      // Record the stack
      char add[MMCHECK_MAX_SIZE];
      const int n_char_printed =
         snprintf(add, MMCHECK_MAX_SIZE, "\n\t%s %s:%d",
                  demangled,filename?filename:"???",line);
      assert(n_char_printed<MMCHECK_MAX_SIZE);
      strcat(stack,add);
   }
   if (function!=nullptr) { return; }
   function = strdup(demangled);
   filename = filename?strdup(filename):NULL;
   address = PC;
   lineno = line;
   if (!debug) { return; }
   printf("Setting function:%s, filename:%s:%d", function, filename, line);
}

// *****************************************************************************
static const char* cxx_demangle(const char* mangled_name)
{
   int status;
   const char *demangled_name =
      abi::__cxa_demangle(mangled_name, NULL, NULL, &status);
   const bool succeeded = status == 0;
   const bool memory_allocation_failure_occurred = status == -1;
   const bool one_argument_is_invalid = status == -3;
   assert(not one_argument_is_invalid);
   if (memory_allocation_failure_occurred)
   {
      printf("[demangle] memory_allocation_failure_occurred!");
      fflush(0);
      assert(false);
   }
   return (succeeded)?demangled_name:mangled_name;
}

// *****************************************************************************
static int filter(const bool dbg, const char* demangled)
{
   if (!dbg) { return 0; }
   //printf("\n[full_callback] Filtering OUT '%s'!\n",demangled);
   return 0;
}

// *****************************************************************************
static void sym_callback(void *data,
                         uintptr_t pc,
                         const char *symname,
                         uintptr_t symval,
                         uintptr_t symsize)
{
   if (!symname) { return; }
   mmBackTraceData *ctx=static_cast<mmBackTraceData*>(data);
   const char *demangled = cxx_demangle(symname);
   ctx->update(demangled,pc);
}

// *****************************************************************************
static void err_callback(void *data, const char *msg, int errnum)
{
   assert(false);
}

// *****************************************************************************
static int full_callback(void *data, uintptr_t pc,
                         const char *filename, int lineno,
                         const char *function)
{
   static const bool all = getenv("ALL");
   mmBackTraceData *ctx = static_cast<mmBackTraceData*>(data);
   const bool dbg = ctx->dump or all;
   if (!function)  // symbol hit
   {
      //if (dbg) printf("\n[full_callback] filename:%s, lineno=%d,
      //pc=0x%lx",filename, lineno, pc);
      return backtrace_syminfo(ctx->state, pc,
                               sym_callback, err_callback, data);
   }
   const char *demangled = cxx_demangle(function);
   // Filtering
   if (strncmp("std::",demangled,5)==0) { return filter(all,demangled); }
   if (strncmp("__gnu_cxx::",demangled,11)==0) { return filter(all,demangled); }
   if (dbg) { printf("\n\t"); }
   // Update context
   ctx->update(demangled, pc, filename, lineno);
   if (dbg)
   {
      printf("%s:%d %s", filename, lineno, demangled);
   }
   return 0;
}

// *****************************************************************************
static int simple_callback(void *data, uintptr_t pc)
{
   mmBackTraceData *ctx = static_cast<mmBackTraceData*>(data);
   backtrace_pcinfo(ctx->state, pc, full_callback, err_callback, data);
   return ctx->continue_tracing();
}

// *****************************************************************************
void mmBackTrace::ini(const char* argv0)
{
   state = backtrace_create_state(argv0, BACKTRACE_SUPPORTS_THREADS,
                                  err_callback, NULL);
   data = new mmBackTraceData(state);
}

// *****************************************************************************
int mmBackTrace::backtrace(const bool dump)
{
   if (state==NULL or data==NULL) { return -1; }
   data->flush(dump);
   backtrace_simple(state,
                    3, // skip frames to point on previous function call
                    simple_callback,
                    err_callback,
                    data);
   return 0;
}

// *****************************************************************************
// * Types
// *****************************************************************************
typedef std::unordered_map<const void*,const char*> mm_stack_t;

// *****************************************************************************
// * Statics
// *****************************************************************************
static mm_stack_t *mm = NULL;
static mmBackTrace *bt = NULL;

// *****************************************************************************
static void mmAdd(const void *ptr, const bool new_or_del, const char *stack)
{
   const bool is_new = new_or_del;
   const bool known = mm->find(ptr) != mm->end();
   if (is_new and not known)
   {
      mm->emplace(ptr, strdup(stack));
   }
   if (known)
   {
      mm->erase(ptr);
   }
}

// *****************************************************************************
static const char *stack(const void *ptr)
{
   return mm->at(ptr);
}

// *****************************************************************************
// * Checks can be done only if we have been initialized
// *****************************************************************************
void mmCheckIni(const char* argv0)
{
   bt = new mmBackTrace();
   mm = new mm_stack_t();
   bt->ini(argv0);
}

// *****************************************************************************
static void culprits(const void *ptr)
{
   printf("\nStack:%s\n",bt->stack());
   printf("\nFirst:%s\n", stack(ptr));
   fflush(0);
   assert(false);
}

// *****************************************************************************
void mmCheck(const void *ptr,
             const bool new_or_del,
             const bool dump)
{
   assert(ptr);
   const bool is_new = new_or_del;
   const bool is_del = not new_or_del;

   static const bool all = getenv("ALL")!=NULL;
   static const bool dbg = getenv("DBG")!=NULL;
   static const bool args = getenv("ARGS")!=NULL;

   // If we are in ORG_MODE, set tab_char to '*'
   static const bool org_mode = getenv("ORG")!=NULL;
   static const bool mm_assert = getenv("MM")!=NULL;
   const std::string tab_char = org_mode?"*":"  ";

   // now backtracing if initialized
   if (!bt) { return; }
   if (bt->backtrace(dump)!=0) { return; }

   const bool mm = bt->mm();
   const bool mfem = bt->mfem();
   const bool skip = bt->skip();
   const int depth = bt->depth();
   const int frames = depth-(org_mode?-1:1);
   const uintptr_t address = bt->address();
   const char *function = bt->function();
   const char *filename = bt->filename();
   const int lineno = bt->lineno();
   const std::string demangled_function(function);
   const int first_parenthesis = demangled_function.find_first_of('(');
   const std::string no_args_demangled_function =
      demangled_function.substr(0,first_parenthesis);
   const std::string display_function =
      (args)?demangled_function:no_args_demangled_function;
   const int first_3A = display_function.find_first_of(':');
   const int first_3C = display_function.find_first_of('<');
   const int first_5B = display_function.find_first_of('[');
   assert(first_3A<=(first_5B<0)?first_3A:first_5B);
   const int first_3AC = ((first_3A^first_3C)<0)?
                         std::max(first_3A,first_3C):std::min(first_3A,first_3C);
   const std::string root =
      (first_3A!=first_3C)?
      display_function.substr(0,first_3AC) : display_function;
   const int color = address%(256-46)+46;

   if (dbg)
   {
      if (all) { std::cout << std::endl; }
      // Generating tabs
      for (int k=0; k<frames; ++k) { std::cout<<tab_char; }
      // Outputing
      if (!org_mode)
      {
         std::cout << "\033[38;5;"<< color << ";1m";   // bold
      }
      else { std::cout << " "; }
      std::cout << "["<<filename<<":" << lineno << ":"
                << display_function << "]\033[m";
   }

   if (skip and dbg) { printf(" skip!"); }
   if (skip or not mm_assert) { return; }

   const bool known = mfem::mm::known((void*)ptr);
   if (dbg) { printf(" %sMFEM",mfem?"":"Not "); }
   if (mfem)
   {
      if (mm)
      {
         if (known and is_new)  // ******************************************
         {
            printf("\nTrying to 'insert' (%p), known by the MM!",ptr);
            return culprits(ptr);
         }
         else if (not known and is_del)   // ********************************
         {
            printf("\nTrying to 'erase' (%p), not known by the MM!",ptr);
            return culprits(ptr);
         }
         else
         {
            if (dbg) { printf(", known: ok"); }
         }
      }
      else
      {
         if (dbg) { printf(", !MM"); }
         if (known and is_new)  //  *****************************************
         {
            printf("\nTrying to 'new' (%p), known by the MM!",ptr);
            return culprits(ptr);
         }
         else if (known and is_del)   // ************************************
         {
            printf("\nTrying to 'delete' (%p), known by the MM!",ptr);
            return culprits(ptr);
         }
         else
         {
            if (dbg) { printf(" unknown: ok"); }
         }
      }
   }
   else   // not mfem
   {
      if (mm)  // MM but not mfem namespace
      {
         if (known and is_new)  // ******************************************
         {
            printf("\nTrying to 'insert' (%p), known by the MM!",ptr);
            return culprits(ptr);
         }
         else if (not known and is_del)   // ********************************
         {
            printf("Trying to 'erase' (%p), not known by the MM!",ptr);
            return culprits(ptr);
         }
         else
         {
            if (dbg) { printf(" known: ok"); }
         }
      }
      else   // not MM, not mfem namespace
      {
         if (known and is_new)  // ***************************************
         {
            printf("\nTrying to 'new' (%p), known by the MM!",ptr);
            return culprits(ptr);
         }
         if (known and is_del)  // *********************************
         {
            printf("Trying to 'delete' (%p), known by the MM!",ptr);
            return culprits(ptr);
         }
         if (dbg) { printf(", unknown: ok"); }
      }
   }
   mmAdd(ptr, new_or_del, bt->stack());
}

// *****************************************************************************
void mmCheckEnd()
{
   delete bt;
   delete mm;
}

// *****************************************************************************
// * LD_PRELOADed functions: malloc, free, calloc, realloc and memalign
// *****************************************************************************
typedef void *malloc_t(size_t);
typedef void *calloc_t(size_t, size_t);
typedef void free_t(void*);
typedef void *realloc_t(void*, size_t);
typedef void *memalign_t(size_t, size_t);
typedef std::unordered_map<void*,size_t> mm_t;

// *****************************************************************************
static bool dbg = false;
static bool hooked = false;
static bool dlsymd = false;

// *****************************************************************************
static free_t *_free = NULL;
static malloc_t *_malloc = NULL;
static calloc_t *_calloc = NULL;
static realloc_t *_realloc = NULL;
static memalign_t *_memalign = NULL;

// *****************************************************************************
static void _init(void)
{
   if (getenv("DBG")) { dbg = true; }
   _free = (free_t*) dlsym(RTLD_NEXT, "free");
   _calloc = (calloc_t*) dlsym(RTLD_NEXT, "calloc");
   _malloc = (malloc_t*) dlsym(RTLD_NEXT, "malloc");
   _realloc = (realloc_t*) dlsym(RTLD_NEXT, "realloc");
   _memalign = (memalign_t*) dlsym(RTLD_NEXT, "memalign");
   assert(_free and _malloc and _calloc and _realloc and _memalign);
   hooked = dlsymd = true;
}


// *****************************************************************************
void *malloc(size_t size)  // Red
{
   if (!_malloc) { _init(); }
   if (!hooked) { return _malloc(size); }
   hooked = false;
   void *ptr = _malloc(size);
   assert(ptr);
   if (dbg) { printf("\n\033[31m[malloc] %p (%ld)\033[m", ptr, size); }
   mmCheck(ptr, true, false); // new, dont show full stack
   hooked = true;
   return ptr;
}

// *****************************************************************************
void free(void *ptr)  // Green
{
   if (!_free) { _init(); }
   if (!hooked) { return _free(ptr); }
   if (!ptr) { return; }
   hooked = false;
   if (dbg) { printf("\n\033[32m[free] %p\033[m", ptr); }
   mmCheck(ptr, false, false); // delete, dont show full stack
   _free(ptr);
   hooked = true;
}

// *****************************************************************************
void *calloc(size_t nmemb, size_t size)  // Yellow
{
   if (not dlsymd)   // if we are not yet dlsym'ed, just do it ourselves
   {
      static const size_t MEM_MAX = 8192;
      static char mem[MEM_MAX];
      static size_t m = 0;
      const size_t bytes = nmemb*size;
      void *ptr = &mem[m];
      m += bytes;
      assert(m<MEM_MAX);
      for (size_t k=0; k<bytes; k+=1) { *(((char*)ptr)+k) = 0; }
      return ptr;
   }
   if (!hooked) { return _calloc(nmemb, size); }
   hooked = false;
   void *ptr = _calloc(nmemb, size);
   if (dbg) { printf("\n\033[33m[calloc] %p (%ld)\033[m", ptr, size); }
   mmCheck(ptr, true, false); // new, show full stack
   hooked = true;
   return ptr;
}

// *****************************************************************************
void *realloc(void *ptr, size_t size)  // Blue
{
   if (!_realloc) { _init(); }
   if (!hooked) { return _realloc(ptr, size); }
   hooked = false;
   void *nptr = _realloc(ptr, size);
   assert(nptr);
   if (dbg) { printf("\n\033[34;7m[realloc] %p(%ld)\033[m", nptr, size); }
   mmCheck(nptr, true, true); // new, show full stack
   hooked = true;
   return nptr;
}

// *****************************************************************************
void *memalign(size_t alignment, size_t size)  // Magenta
{
   if (!_memalign) { _init(); }
   if (!hooked) { return _memalign(alignment, size); }
   hooked = false;
   void *ptr = _memalign(alignment, size);
   assert(ptr);
   if (dbg) { printf("\n\033[35;7m[memalign] %p(%ld)\033[m", ptr, size); }
   mmCheck(ptr, true, true); // new, show full stack
   hooked = true;
   return ptr;
}
