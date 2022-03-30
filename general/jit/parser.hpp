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

#include <list>
#include <string>
#include <iostream>

namespace mfem
{

namespace jit
{

struct argument_t
{
   int default_value = 0;
   std::string type, name;
   bool is_ptr = false, is_amp = false, is_const = false,
        is_restrict = false, is_tpl = false, has_default_value = false;
   std::list<int> range;
   bool operator==(const argument_t &arg) { return name == arg.name; }
   argument_t() {}
   argument_t(std::string id): name(id) {}
};

typedef std::list<argument_t>::iterator argument_it;

struct forall_t { int d; std::string e, N, X, Y, Z, body; };

struct kernel_t
{
   bool is_jit = false;
   bool is_embed = false;
   bool is_prefix = false;
   bool is_forall = false;
   bool is_single_source = false;
   std::string mfem_jit_cxx;         // holds MFEM_JIT_CXX
   std::string mfem_jit_build_flags; // holds MFEM_JIT_BUILD_FLAGS
   std::string mfem_source_dir;      // holds MFEM_SOURCE_DIR
   std::string mfem_install_dir;     // holds MFEM_INSTALL_DIR
   std::string name;                 // kernel name
   std::string space;                // kernel namespace
   // Templates: format, arguments and parameters
   std::string Tformat;              // template format, as in printf
   std::string Targs;                // template arguments, for hash and call
   std::string Tparams;              // template parameters, for the declaration
   std::string Tparams_src;          // template parameters, from original source
   // Arguments and parameter for the standard calls
   // We need two kinds of arguments because of the '& <=> *' transformation
   // (This might be no more the case as we are getting rid of Array/Vector).
   std::string params;
   std::string args;
   std::string args_wo_amp;
   std::string d2u, u2d;             // double to unsigned place holders
   std::string embed;                // source of the embed function
   struct forall_t forall;           // source of the lambda forall
   std::string prefix;
};

struct context_t
{
   kernel_t ker;
   std::istream &in;
   std::ostream &out;
   std::string &file;
   std::list<argument_t> args;
   int line, block, parenthesis;
public:
   context_t(std::istream& i, std::ostream& o, std::string &f)
      : in(i), out(o), file(f), line(1), block(-2), parenthesis(-2) {}
};

struct error_t
{
   int line;
   std::string file;
   const char *msg;
   error_t(int l, std::string f, const char *m): line(l), file(f), msg(m) {}
};

int preprocess(context_t &pp);

} // namespace jit

} // namespace mfem

