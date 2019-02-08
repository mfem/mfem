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
// * 
// *****************************************************************************
static stkBackTrace *bt;
void stkIni(char* argv0){
   bt = new stkBackTrace();
  _stk("[stkIni] argv0=%s",argv0);
  bt->ini(argv0);
}

// ***************************************************************************
// * stk 
// ***************************************************************************
stk::stk(const bool _rip):
   ptr(NULL),new_or_delete(false),rip(_rip),options(""){}
stk::stk(const void *p,const bool nd, const bool _rip):
   ptr(p),new_or_delete(nd),rip(_rip),options(""){}
stk::stk(const char* opt,const bool _rip):
   ptr(NULL),new_or_delete(false),rip(_rip),options(opt){}
stk::~stk(){
  
  static std::unordered_map<std::string, int> known_colors;
 
  static std::unordered_map<uintptr_t,std::string> known_address;

  // If no STK varibale environment is set, just return
  static const bool stk = getenv("STK")!=NULL;
  static const bool out = getenv("OUT")!=NULL;
  static const bool args = getenv("ARGS")!=NULL;
  if (not stk) return;

  // If we are in ORG_MODE, set tab_char to '*'
  static const bool org_mode = getenv("ORG_MODE")!=NULL;
  static const bool mm_assert = getenv("MM")!=NULL;
  const std::string tab_char = org_mode?"*":"  ";
  
  // now backtracing if initialized
  if (!bt) return;
  if (bt->stk(rip)!=0) return;

  const bool mm = bt->mm();
  const bool mfem = bt->mfem();
  const int depth = bt->depth();
  assert(depth>0);
  const int frames = depth-(org_mode?-1:1);
  const uintptr_t address = bt->address();
  //printf("\033[33m[stk] address 0x%lx\033[m\n",address);fflush(0);
  const char *backtraced_function = bt->function();
  const char *filename = bt->filename();
  const int lineno = bt->lineno();
  const std::string demangled_function(backtraced_function);
   
  if (known_address.find(address)==known_address.end()){
    if (args){ // Get rid of arguments, or not
      const int first_parenthesis = demangled_function.find_first_of('(');
      const std::string function = demangled_function.substr(0,first_parenthesis);
      known_address[address]=function;
    }else known_address[address]=demangled_function;
  }

  const std::string display_function = known_address[address];
  
  const int first_3A = display_function.find_first_of(':');
  const int first_3C = display_function.find_first_of('<');
  const int first_5B = display_function.find_first_of('[');
  assert(first_3A<=(first_5B<0)?first_3A:first_5B);
  const int first_3AC = ((first_3A^first_3C)<0)?
    std::max(first_3A,first_3C):std::min(first_3A,first_3C);
  const std::string root = (first_3A!=first_3C)?display_function.substr(0,first_3AC):display_function;
        
  if (known_colors.find(display_function)==known_colors.end())
    known_colors[display_function] = known_colors.size()%(256-46)+46;
  const int color = known_colors[display_function];

  if (out){
     // Generating tabs
     for(int k=0;k<frames;++k) std::cout<<tab_char;
     // Outputing 
     if (!org_mode)
        std::cout << "\033[" << options
                  << ";38;5;"<< color
                  << ";1m"; // bold
     else std::cout << " ";
     std::cout << "["<<filename<<":"<<lineno<<":"<<display_function<<"]\033[m ";
     // reset + normal color if !empty
     if (!org_mode)
        std::cout << "\033[38;5;"<<color<<"m";
     std::cout << stream.str();
     if (!org_mode) std::cout << "[m";
     stream.clear();
  }

  // if MM was set, make sure all "mfem::*" calls go through the mm.*pp file
  if (mm_assert and mfem){
     if (out) printf("\033[1;32mMFEM\033[m");
     if (mm){
        if (out) printf("\033[1;32m, MM\033[m");
        if (ptr) {
           const bool is_new = new_or_delete;
           const bool is_del = not new_or_delete;
           const bool known = mfem::mm::known((void*)ptr);
           if (known and is_new){
              printf("\033[1;31m\nTrying to insert a pointer that is known by the MM!\033[m");
              assert(false);
           }else if (not known and is_del){
              //printf("\033[1;31m\nTrying to delete a pointer that is not known by the MM!\033[m");
              //assert(false);
           }else{
              if (out) printf("\033[32m, known: ok\033[m");
           }
        }
     }else{
        if (out) printf("\033[31m, !MM\033[m");
        // if a pointer has been given, use it to test if it known in the mm
        if (ptr) {
           const bool is_new = new_or_delete;
           const bool is_del = not new_or_delete;
           const bool known = mfem::mm::known((void*)ptr);
           if (known and is_new){
              printf("\033[1;31m\nTrying to insert a pointer that is known by the MM!\033[m");
              assert(false);
           }else if (known and is_del){
              printf("\033[1;31m\nTrying to delete a pointer that is known by the MM!\033[m");
              printf("\033[32mSTACK: %s\033[m",bt->stack()); 
              mfem::mm::dump((void*)ptr);
              fflush(0);
              exit(0);
           }else{
              if (out) printf("\033[31m, unknown: ok\033[m");
           }
        }
     }
  }else if (mm_assert and not mfem){
     if (out) printf("\033[1;31mNot MFEM\033[m");
     // if a pointer has been given, use it to test if it known in the mm
     if (ptr) {
        const bool known = mfem::mm::known((void*)ptr);
        if (known){
           printf("\033[1;31m\nTrying to delete a pointer that is known by the MM!\033[m");
           assert(false);
        }
     }
  }
  if (out) fflush(0);
  if (out) std::cout << "\n";
}
