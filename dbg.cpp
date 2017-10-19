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
#undef NDEBUG
#include <assert.h>
#include <execinfo.h>

#include <regex>
#include <sstream>
#include <iostream>
#include <cxxabi.h>
#include <unordered_map>

#include "dbg.h"
#include "dbg.hpp"

#include "dbgBackTrace.hpp"

#include <backtrace.h>

// *****************************************************************************
static const char *dbg_argv0 = NULL;
void dbgIni(char* argv0){
  _dbg("[dbgIni] argv0=%s",argv0);
  dbg_argv0=argv0;
}


// ***************************************************************************
// * dbg 
// ***************************************************************************
dbg::dbg(){}
dbg::~dbg(){
/*  static std::unordered_map<std::string, bool> wanted ={
    {"kernelArg_t", false},
    {"ptrRange_t", false},
    {"kernelArg", false},
    {"kernel_v", false},
    {"kernel", true},
    {"memory", false},
    {"memory_v", false},
    {"settings", false},
    {"properties", false},
    {"device_v", false},
    {"device", false},
    {"json", false},
    {"mode_v", false},
    {"hash_t",false},
    {"hash",false},
    {"withRefs",false},
    {"mode", true},
    {"sys", false},
    {"io", false},
    {"env", false},
    {"uvaToMemory",false},
    {"syncToDevice",false},
    {"serial",true},
    {"host",false},
    {"stopManaging",false},
    {"syncMemToDevice",false}
    };
  */
 static std::unordered_map<std::string, bool> wanted ={
    {"kernelArg_t", true},
    {"ptrRange_t", true},
    {"kernelArg", true},
    {"kernel_v", true},
    {"kernel", true},
    {"memory", true},
    {"memory_v", true},
    {"settings", true},
    {"properties", true},
    {"device_v", true},
    {"device", true},
    {"json", true},
    {"mode_v", true},
    {"hash_t",true},
    {"hash",true},
    {"withRefs",true},
    {"mode", true},
    {"sys", true},
    {"io", true},
    {"env", true},
    {"uvaToMemory",true},
    {"syncToDevice",true},
    {"serial",true},
    {"host",true},
    {"stopManaging",true},
    {"syncMemToDevice",true}
    };
    
  static std::unordered_map<std::string, int> known_colors ={
    {"occa",0},
    {"kernel",1},
    {"mode",2},
    {"mode_v",2},
    {"registerMode",2},
    {"cli",3},
    {"env",4},
    {"sys",5},
    {"serial",6}
  };
  static std::unordered_map<uintptr_t,std::string> known_functions;

  // If no DBG varibale environment is set, just return
  if (!getenv("DBG")) return;

  // If we are in ORG_MODE, set tab_char to '*'
  const bool org_mode = getenv("ORG_MODE")!=NULL;
  const std::string tab_char = org_mode?"*":"  ";

  // Get the backtrace depth and last function
  int frames = 0;
  int nb_addresses = 0;
  uintptr_t address = 0;
  std::string demangled_function;
  
  { // backtracing here
    dbgBackTrace bt(dbg_argv0,2);
  
    nb_addresses = bt.depth();
    assert(nb_addresses>=0);
    frames = nb_addresses-(org_mode?-1:1);
    address = bt.address();
    const char *backtraced_function = bt.function();
    //_dbg("[dbg] bt_function is '%s'",backtraced_function);
    demangled_function=std::string(backtraced_function);
  }
  
   
  if (known_functions.find(address)==known_functions.end()){
    const int first_parenthesis = demangled_function.find_first_of('(');
    //std::cout << "demangled_function: '"<<demangled_function<<"'\n";
    const std::string function = demangled_function.substr(0,first_parenthesis);
    //std::cout << "function: '"<<function<<"'\n";
    const std::string offunction =
      std::regex_replace(function,std::regex("occa::"),std::string());
    known_functions[address]=offunction;
  }

  const std::string display_function = known_functions[address];
  //std::cout << "display_function: '"<<display_function<<"'\n";

  //std::cout << "offunction: '"<<offunction<<"'\n";
  const int first_3A = display_function.find_first_of(':');
  //std::cout << "first_3A="<<first_3A<<"\n";
  const int first_3C = display_function.find_first_of('<');
  //std::cout << "first_3C="<<first_3C<<"\n";
  const int first_5B = display_function.find_first_of('[');
  //std::cout << "first_5B="<<first_5B<<"\n";
  assert(first_3A<=(first_5B<0)?first_3A:first_5B);
  const int first_3AC = ((first_3A^first_3C)<0)?
    std::max(first_3A,first_3C):
    std::min(first_3A,first_3C);
  // If first_3A==first_3C==-1, just take our offunction name
  std::string root = (first_3A!=first_3C)?display_function.substr(0,first_3AC):display_function;
  //std::cout << "root: '"<<root<<"'\n";
    
  // Look if this root is wanted or has to be filtered
  if (wanted.find(root)!=wanted.end()){
    if (!wanted[root]) {
      stream.clear();
      return;
    }
  }
        
  if (known_colors.find(root)==known_colors.end()){
    const int new_color = known_colors.size()%9;
    //std::cout<<"\033[7mNEW color: "<<root<<", new_color="<<new_color<<"\033[m\n";
    known_colors[root] = new_color;
  }
  //std::cout << "function: "<<function<<"\n";
  // Producinbg \t, skipping the 4 deepests
  for(int k=0;k<frames;++k)
    std::cout<<tab_char;
  const int color = 0x1d+known_colors[root];
  // Outputing 
  if (!org_mode) std::cout << "\033["<<color<<";1m"; // bold
  else std::cout << " ";
  std::cout << "["<<display_function<<"] ";
  // reset + normal color if !empty
  if (!org_mode)
    std::cout << "\033[m\033["<<color<<"m";
  std::cout << stream.str();
  if (!org_mode) std::cout << "[m";
  std::cout << "\n";
  stream.clear();
}
