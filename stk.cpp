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
#include <assert.h>

#include <regex>
#include <iostream>
#include <unordered_map>

#include "stk.h"
#include "stk.hpp"
#include "stkBackTrace.hpp"

// *****************************************************************************
// * 
// *****************************************************************************
static stkBackTrace bt;
void stkIni(char* argv0){
  _stk("[stkIni] argv0=%s",argv0);
  bt.ini(argv0);
}

// ***************************************************************************
// * stk 
// ***************************************************************************
stk::stk():options(""){}
stk::stk(const char* opt):options(opt){}
stk::~stk(){
  
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
    
  static std::unordered_map<std::string, int> known_colors;
 
  static std::unordered_map<uintptr_t,std::string> known_address;

  // If no STK varibale environment is set, just return
  if (!getenv("STK")) return;

  // If we are in ORG_MODE, set tab_char to '*'
  const bool org_mode = getenv("ORG_MODE")!=NULL;
  const std::string tab_char = org_mode?"*":"  ";
  
  // now backtracing if initialized
  if (bt.stk()!=0) return;
  
  const int depth = bt.depth();
  assert(depth>0);
  const int frames = depth-(org_mode?-1:1);
  const uintptr_t address = bt.address();
  const char *backtraced_function = bt.function();
  const char *filename = bt.filename();
  const int lineno = bt.lineno();
  const std::string demangled_function(backtraced_function);
   
  if (known_address.find(address)==known_address.end()){
    if (!getenv("ARGS")){ // Get rid of arguments, or not
      const int first_parenthesis = demangled_function.find_first_of('(');
      const std::string function = demangled_function.substr(0,first_parenthesis);
      known_address[address]=function;
    }else known_address[address]=demangled_function;
  }

  const std::string display_function = known_address[address];


  //const std::string root = known_address[address];
  
  //std::cout << "offunction: '"<<offunction<<"'\n";
  const int first_3A = display_function.find_first_of(':');
  //std::cout << "first_3A="<<first_3A<<"\n";
  const int first_3C = display_function.find_first_of('<');
  //std::cout << "first_3C="<<first_3C<<"\n";
  const int first_5B = display_function.find_first_of('[');
  //std::cout << "first_5B="<<first_5B<<"\n";
  assert(first_3A<=(first_5B<0)?first_3A:first_5B);
  const int first_3AC = ((first_3A^first_3C)<0)?
    std::max(first_3A,first_3C):std::min(first_3A,first_3C);
  // If first_3A==first_3C==-1, just take our offunction name  
  const std::string root = (first_3A!=first_3C)?display_function.substr(0,first_3AC):display_function;
  //std::cout << "root: '"<<root<<"'\n";
  
      
  // Look if this root is wanted or has to be filtered
  if (wanted.find(root)!=wanted.end()){
    if (!wanted[root]) {
      stream.clear();
      return;
    }
  }
        
  if (known_colors.find(display_function)==known_colors.end())
    known_colors[display_function] = known_colors.size()%(256-46)+46;
  const int color = known_colors[display_function];

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
  std::cout << "\n";
  stream.clear();
}
