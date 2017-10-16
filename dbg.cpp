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

#include "dbg.hpp"

#include "dbgBackTrace.hpp"

#include <backtrace.h>


// ***************************************************************************
// * dbg 
// ***************************************************************************
dbg::~dbg(){  
  static std::unordered_map<std::string, bool> wanted ={
    {"kernelArg_t", true},
    {"ptrRange_t", true},
    {"kernelArg", true},
    {"kernel_v", true},
    {"memory", true},
    {"memory_v", true},
    {"settings", true},
    {"properties", false},
    {"device_v", true},
    {"device", true},
    {"json", false},
    {"mode_v", true},
    {"hash_t",true},
    {"hash",true},
    {"withRefs",true},
    {"mode", true},
    {"sys", true},
    {"io", false},
    {"env", true},
    {"uvaToMemory",true},
    {"syncToDevice",true}
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
  static std::unordered_map<void*,std::string> known_functions;
    
  if (!getenv("OCCA_DBG")) return;
    
  const bool org_mode = getenv("OCCA_ORG")!=NULL;
  const std::string tab_char = org_mode?"*":"  ";

    
  // Get the backtrace depth and last function
  int frames = 0;
  int nb_addresses = 0;
  void* address = NULL;
  int status = 0;
  char* function = NULL;
    
  if (backTrace.is_inited()){
    backTrace.dbg();
    nb_addresses = backTrace.depth();
    assert(nb_addresses>=0);
    frames = nb_addresses-(org_mode?-1:1);
    address = backTrace.address();
    function = strdup(backTrace.function_name());
    assert(function);
    //std::cout << "[32;1mbacktrace:[m "<<backTrace.function_name() << " address="<<address<<std::endl;
  }else{
    // Looking how deep we are in the stack
    const int size = 64;
    void *buffer[size];
    nb_addresses = ::backtrace(buffer, size);
    assert(nb_addresses>=4);
    frames = nb_addresses-(org_mode?3:4);
    char **symbols = ::backtrace_symbols(buffer, size);
    address = buffer[1];
    //for(int k=0;k<nb_addresses;++k) std::cout << symbols[k] << std::endl;
    char* symbol = symbols[1];
    // Now within a shared lib, backtrace_symbols have lib name
    if (symbol[0]=='/'){
      char *first_parenthesis = strchr(symbol,'(');
      assert(first_parenthesis);
      symbol=1+first_parenthesis;
      char* next_plus = strchr(symbol,'+');
      assert(next_plus);
      *next_plus=0;
    }
    function = abi::__cxa_demangle(symbol,NULL,0, &status);
    assert(function);
  }

  std::string demangled_function(function);
  if (known_functions.find(address)==known_functions.end()){
    const int first_parenthesis = demangled_function.find_first_of('(');
    //std::cout << "demangled_function: '"<<demangled_function<<"'\n";
    const std::string function = demangled_function.substr(0,first_parenthesis);
    //std::cout << "function: '"<<function<<"'\n";
    const std::string offunction = std::regex_replace(function,std::regex("occa::"),std::string());
    known_functions[address]=offunction;
  }
  const std::string display_function = known_functions[address];
    

  //const std::string display_function = demangled_function;
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
