// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "okstk.hpp"

// *****************************************************************************
// * structs
// *****************************************************************************
typedef std::unordered_map<const void*,const char*> mm_t;

// *****************************************************************************
// * statics
// *****************************************************************************
static mm_t *mm = NULL;
static stkBackTrace *bt = NULL;

// *****************************************************************************
const char *stack(const void *ptr){
   return mm->at(ptr);
}

// *****************************************************************************
void mmAdd(const void *ptr, const bool new_or_del, const char *stack){
   assert(mm);
   assert(ptr);
   const bool is_new = new_or_del;
   const bool known = mm->find(ptr) != mm->end();
   if (is_new){
      if (not known){
         mm->emplace(ptr, strdup(stack));
      }
   }else{
      if (known){
         mm->erase(ptr);
      }
   }
}

// *****************************************************************************
void backtraceIni(const char* argv0){
   bt = new stkBackTrace();
   mm = new mm_t();
   bt->ini(argv0);
}

// *****************************************************************************
void backtrace(const void *ptr,
               const bool new_or_del,
               const bool dump){
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
   if (!bt) return;
   if (bt->backtrace(dump)!=0) return;

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
   const std::string no_args_demangled_function = demangled_function.substr(0,first_parenthesis);
   const std::string display_function = (args)?demangled_function:no_args_demangled_function;  
   const int first_3A = display_function.find_first_of(':');
   const int first_3C = display_function.find_first_of('<');
   const int first_5B = display_function.find_first_of('[');
   assert(first_3A<=(first_5B<0)?first_3A:first_5B);
   const int first_3AC = ((first_3A^first_3C)<0)?
      std::max(first_3A,first_3C):std::min(first_3A,first_3C);
   const std::string root = (first_3A!=first_3C)?display_function.substr(0,first_3AC):display_function;
   const int color = address%(256-46)+46;

   if (dbg){
      if (all) std::cout << std::endl;
      // Generating tabs
      for(int k=0;k<frames;++k) std::cout<<tab_char;
      // Outputing 
      if (!org_mode)
         std::cout << "\033[38;5;"<< color << ";1m"; // bold
      else std::cout << " ";
      std::cout << "["<<filename<<":"<<lineno<<":"<<display_function<<"]\033[m ";
      // reset + normal color if !empty
      if (!org_mode)
         std::cout << "\033[38;5;"<<color<<"m";
      if (!org_mode) std::cout << "[m";
   }

   // If a pointer was given, use it to test if it is known by the MM
   if (mm_assert and not skip){
      const bool known = mfem::mm::known((void*)ptr);
      if (dbg) printf(" %sMFEM\033[m",mfem?"\033[1;32m":"\033[1;31mNot ");
      if (mfem){
         if (mm){
            if (known and is_new){ // ******************************************
               printf("\033[1;31m\nTrying to 'insert' a pointer (%p) that is known by the MM!\033[m",ptr);
               printf("\033[32m%s\033[m",bt->stack());
               printf("\nFirst:%s\n", stack(ptr));
               fflush(0);
               assert(false);
            }else if (not known and is_del){ // ********************************
               printf("\n\033[1;31m[MFEM,MM] Trying to 'erase' a pointer (%p) that is not known by the MM!\033[m",ptr);
               printf("\nStack:%s\n\033[m",bt->stack());
               printf("\nFirst:%s\n", stack(ptr));
               fflush(0);
               assert(false);                  
            }else{
               if (dbg) printf("\033[32m, known: ok\033[m");
            }
         }else{
            if (dbg) printf("\033[31m, !MM\033[m");
            if (known and is_new){ //  *****************************************
               printf("\033[1;31m\nTrying to 'new' a pointer (%p) that is known by the MM!\033[m",ptr);
               printf("\033[32m%s\n\033[m",bt->stack());
               printf("\nFirst:%s\n", stack(ptr));
               fflush(0);
               assert(false);
            }else if (known and is_del){ // ************************************
               printf("\033[1;31m\nTrying to 'delete' a pointer (%p) that is known by the MM!\033[m",ptr);
               printf("\033[32m%s\n\033[m",bt->stack()); 
               printf("\nFirst:%s\n", stack(ptr));
               fflush(0);
               assert(false);
            }else{
               if (dbg) printf("\033[31m, unknown: ok\033[m");
            }        
         }
      }else{ // not mfem
         if (mm){ // MM but not mfem namespace
            if (known and is_new){ // ******************************************
               printf("\033[1;31m\nTrying to 'insert' a pointer (%p) that is known by the MM!\033[m",ptr);
               printf("\033[32m%s\033[m",bt->stack());
               printf("\nFirst:%s\n", stack(ptr));
               fflush(0);
               assert(false);
            }else if (not known and is_del){ // ********************************
               printf("\033[1;31m[!MFEM,MM] Trying to 'erase' a pointer (%p) that is not known by the MM!",ptr);
               printf("\nStack:%s\033[m",bt->stack());
               printf("\nFirst added:%s\n", stack(ptr));
               fflush(0);
               assert(false);
            }else{
               if (dbg) printf("\033[32m, known: ok\033[m");
            }
         }else{ // not MM, not mfem namespace
            if (known){
               if (known and is_new){ // ***************************************
                  printf("\033[1;31m\nTrying to 'new' a pointer (%p) that is known by the MM!\033[m",ptr);
                  printf("\033[32m%s\n\033[m",bt->stack());
                  printf("\nFirst:%s\n", stack(ptr));
                  fflush(0);
                  assert(false);
               }else if (known and is_del){ // *********************************
                  printf("\033[1;31mTrying to 'delete' a pointer (%p) that is known by the MM!",ptr);
                  printf("\nStack:%s\033[m",bt->stack());
                  printf("\nFirst:%s\n", stack(ptr));
                  fflush(0);
                  assert(false);
               }else{
                  if (dbg) printf("\033[31m, unknown: ok\033[m");
               }
            }
         }
      }
      mmAdd(ptr, new_or_del, bt->stack());
   } 
   if (dbg) fflush(0);
   //if (out) std::cout << "\n";
   return;
}

// *****************************************************************************
void backtraceEnd(){
   //delete bt;
   //bt=NULL;  
   //delete mm;
   //mm=NULL; 
}
