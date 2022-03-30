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

#include <regex>
#include <ciso646>
#include <cassert>

#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>

using std::regex;

#include "../../config/config.hpp"

#include "../error.hpp"
#include "../globals.hpp"

#include "jit.hpp"
#include "parser.hpp"

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 47
#include "../debug.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#define MFEM_JIT_STR(...) #__VA_ARGS__
#define MFEM_JIT_STRINGIFY(...) MFEM_JIT_STR(__VA_ARGS__)

namespace mfem
{

namespace jit
{

void check(context_t &pp, const bool test, const char *msg = nullptr)
{
   if (not test) { throw error_t(pp.line, pp.file,msg); }
}

void addComa(std::string &arg) { if (not arg.empty()) { arg += ",";  } }

void addArg(std::string &list, const char *arg) { addComa(list); list += arg; }

bool is_newline(const int ch) { return static_cast<unsigned char>(ch) == '\n'; }

bool good(context_t &pp) { pp.in.peek(); return pp.in.good(); }

char get(context_t &pp) { return static_cast<char>(pp.in.get()); }

int put(const char c, context_t &pp)
{
   if (is_newline(c)) { pp.line++; }
   if (pp.ker.is_embed) { pp.ker.embed += c; }
   // if we are storing the lambda body, just save it w/o output
   if (pp.ker.is_forall) { pp.ker.forall.body += c; return c;}
   // if we are not yet in the forall, just store all the prefix
   if (pp.ker.is_jit && pp.ker.is_prefix) { pp.ker.prefix += c; }
   pp.out.put(c);
   return c;
}

int put(context_t &pp) { return put(get(pp),pp); }

void skip_space(context_t &pp, std::string &out)
{
   while (isspace(pp.in.peek())) { out += get(pp); }
}

void skip_space(context_t &pp, bool keep=true)
{
   while (isspace(pp.in.peek())) { keep?put(pp):get(pp); }
}

void drop_space(context_t &pp) { while (isspace(pp.in.peek())) { get(pp); } }

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

void singleLineComments(context_t &pp, bool keep=true)
{
   while (!is_newline(pp.in.peek())) { keep?put(pp):get(pp); }
}

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

void comments(context_t &pp, bool keep=true)
{
   if (not is_comments(pp)) { return; }
   keep?put(pp):get(pp);
   if (keep?put(pp):get(pp) == '/') { return singleLineComments(pp,keep); }
   return blockComments(pp,keep);
}

void next(context_t &pp, bool keep=true)
{
   keep?skip_space(pp):drop_space(pp);
   comments(pp,keep);
}

void drop(context_t &pp) { next(pp, false); }

bool is_id(context_t &pp)
{
   const int c = pp.in.peek();
   return isalnum(c) or c == '_';
}

bool is_semicolon(context_t &pp)
{
   skip_space(pp);
   const int c = pp.in.peek();
   return c == ';';
}

std::string get_id(context_t &pp)
{
   std::string id;
   check(pp,is_id(pp),"name w/o alnum 1st letter");
   while (is_id(pp)) { id += get(pp); }
   return id;
}

bool is_digit(context_t &pp) { return isdigit(static_cast<char>(pp.in.peek())); }

int get_digit(context_t &pp)
{
   std::string digit;
   check(pp,is_digit(pp),"unknown number");
   while (is_digit(pp)) { digit += get(pp); }
   return atoi(digit.c_str());
}

std::string peekn(context_t &pp, const int n)
{
   int k = 0;
   assert(n < 64);
   static char c[64];
   for (k = 0; k <= n; k++) { c[k] = 0; }
   for (k = 0; k < n and good(pp); k++) { c[k] = get(pp); }
   std::string rtn(c);
   assert(!pp.in.fail());
   for (int l = 0; l < k; l++) { pp.in.unget(); }
   return rtn;
}

std::string peekid(context_t &pp)
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
   std::string rtn(c);
   for (int l = 0; l < k; l++) { pp.in.unget(); }
   return rtn;
}

void drop_name(context_t &pp) { while (is_id(pp)) { get(pp); } }

bool is_void(context_t &pp)
{
   skip_space(pp);
   const std::string void_peek = peekn(pp,4);
   assert(not pp.in.eof());
   if (void_peek == "void") { return true; }
   return false;
}

bool is_namespace(context_t &pp)
{
   skip_space(pp);
   const std::string namespace_peek = peekn(pp,2);
   assert(not pp.in.eof());
   if (namespace_peek == "::") { return true; }
   return false;
}

bool is_static(context_t &pp)
{
   skip_space(pp);
   const std::string void_peek = peekn(pp,6);
   assert(not pp.in.eof());
   if (void_peek == "static") { return true; }
   return false;
}

bool is_template(context_t &pp)
{
   skip_space(pp);
   const std::string void_peek = peekn(pp,8);
   assert(not pp.in.eof());
   if (void_peek == "template") { return true; }
   return false;
}

bool is_star(context_t &pp)
{
   skip_space(pp);
   if (pp.in.peek() == '*') { return true; }
   return false;
}

bool is_amp(context_t &pp)
{
   skip_space(pp);
   if (pp.in.peek() == '&') { return true; }
   return false;
}

bool is_left_parenthesis(context_t &pp)
{
   skip_space(pp);
   if (pp.in.peek() == '(') { return true; }
   return false;
}

bool is_right_parenthesis(context_t &pp)
{
   skip_space(pp);
   if (pp.in.peek() == ')') { return true; }
   return false;
}

bool is_coma(context_t &pp)
{
   skip_space(pp);
   if (pp.in.peek() == ',') { return true; }
   return false;
}

bool is_eq(context_t &pp)
{
   skip_space(pp);
   if (pp.in.peek() == '=') { return true; }
   return false;
}

// *****************************************************************************
void jitHeader(context_t &pp)
{
   pp.out << "#include \"general/jit/jit.hpp\"\n";
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
   pp.ker.mfem_jit_cxx = MFEM_JIT_STRINGIFY(MFEM_JIT_CXX);
   pp.ker.mfem_jit_build_flags = MFEM_JIT_STRINGIFY(MFEM_JIT_BUILD_FLAGS);
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
         //pp.ker.Targs += is_pointer?"_":"";
         pp.ker.Targs += name;
         // Tparams
         if (!has_default_value)
         {
            addComa(pp.ker.Tparams);
            pp.ker.Tparams += "const ";
            pp.ker.Tparams += is_double?"uint64_t":type;
            pp.ker.Tparams += " ";
            pp.ker.Tparams += is_double?"t":"";
            //pp.ker.Tparams += is_pointer?"_":"";
            pp.ker.Tparams += name;
         }
         if (is_double)
         {
            {
               pp.ker.d2u += "\n\tconst union_du_t union_";
               pp.ker.d2u += name;
               pp.ker.d2u += " = (union_du_t){u:t";
               //pp.ker.d2u += is_pointer?"_":"";
               pp.ker.d2u += name;
               pp.ker.d2u += "};";

               pp.ker.d2u += "\n\tconst double ";
               //pp.ker.d2u += is_pointer?"_":"";
               pp.ker.d2u += name;
               pp.ker.d2u += " = union_";
               pp.ker.d2u += name;
               pp.ker.d2u += ".d;";
            }
            {
               pp.ker.u2d += "\n\tconst uint64_t u";
               //pp.ker.u2d += is_pointer?"_":"";
               pp.ker.u2d += name;
               pp.ker.u2d += " = (union_du_t){";
               //pp.ker.u2d += is_pointer?"_":"";
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
         //pp.ker.args += is_pointer?"_":"";
         pp.ker.args += name;
         // other_arguments_wo_amp
         if (! pp.ker.args_wo_amp.empty()) {  pp.ker.args_wo_amp += ","; }
         //pp.ker.args_wo_amp += is_pointer?"_":"";
         pp.ker.args_wo_amp += name;
         // other_parameters
         if (not pp.ker.params.empty()) { pp.ker.params += ",";  }
         pp.ker.params += is_const?"const ":"";
         pp.ker.params += type;
         pp.ker.params += " *";
         //pp.ker.params += is_pointer?"_":"";
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
   pp.ker.is_prefix = true;
   pp.ker.prefix.clear();
   pp.out << "\n\tconst bool use_jit = Device::IsJITEnabled();";
   pp.out << "\n\tconst char *src=R\"_(";
   pp.out << "#include <cstdint>\n";
   pp.out << "#include <limits>\n";
   pp.out << "#include <cstring>\n";
   pp.out << "#include <stdbool.h>\n";

   //pp.out << "#define MJIT_FORALL\n";
   pp.out << "#include \""
          << pp.ker.mfem_install_dir
          << "/include/mfem/general/jit/jit.hpp\"\n";

   pp.out << "#include \""
          << pp.ker.mfem_install_dir
          << "/include/mfem/general/forall.hpp\"\n";

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
          << MFEM_JIT_PREFIX_CHAR << "%016lx("
          << "const bool use_dev, " << pp.ker.params << "){";
   pp.out << pp.ker.name << "_%016lx<" << pp.ker.Tformat << ">"
          << "(" << "use_dev, " << pp.ker.args_wo_amp << ");";
   pp.out << "})_\";";

   pp.out << "\n\tif (use_jit){";

   // typedef, hash map and launch
   pp.out << "\n\ttypedef void (*kernel_t)(const bool use_dev, "
          << pp.ker.params << ");";
   pp.out << "\n\tstatic std::unordered_map<size_t,jit::kernel<kernel_t>*> ks;";
   if (not pp.ker.u2d.empty()) { pp.out << "\n\t" << pp.ker.u2d; }
   pp.out << "\n\tconst char *cxx = " << pp.ker.mfem_jit_cxx << ";";
   pp.out << "\n\tconst char *mfem_build_flags = "
          << pp.ker.mfem_jit_build_flags <<  ";";
   pp.out << "\n\tconst char *mfem_source_dir = \""
          << pp.ker.mfem_source_dir <<  "\";";
   pp.out << "\n\tconst char *mfem_install_dir = \""
          << pp.ker.mfem_install_dir <<  "\";";
   pp.out << "\n\tconst size_t args_seed = std::hash<size_t>()(0);";
   pp.out << "\n\tconst size_t args_hash = jit::hash_args(args_seed,"
          << pp.ker.Targs << ");";
   pp.out << "\n\tif (!ks[args_hash]) ";
   pp.out << "ks[args_hash] = new jit::kernel<kernel_t>"
          << "(\"" << pp.ker.name << "\", "
          << "cxx, src, mfem_build_flags, mfem_source_dir, mfem_install_dir, "
          << pp.ker.Targs << ");";
   pp.out << "\n\tks[args_hash]->operator_void("
          << "Device::Allows(Backend::CUDA_MASK), "
          << pp.ker.args << ");";
   pp.out << "\n\treturn;";
   pp.out << "\n\t} // use_jit";
   pp.out << "\n";
   // Should check MFEM_USE_CUDA and push the right MFEM_FORALL
   pp.out << pp.ker.prefix;
   pp.out << "for (int " << pp.ker.forall.e << " = 0; "
          << pp.ker.forall.e << " < " << pp.ker.forall.N.c_str() << "; "
          << pp.ker.forall.e<<"++){";
   pp.out << pp.ker.forall.body.c_str();
   pp.out << "}\n";
   // Stop counting the blocks and flush the kernel status
   pp.block--;
   pp.ker.is_jit = false;
}

// *****************************************************************************
std::string arg_get_array_type(context_t &pp)
{
   std::string type;
   skip_space(pp);
   check(pp,pp.in.peek()=='<',"no '<' while in get_array_type");
   put(pp);
   type += "<";
   skip_space(pp);
   check(pp,is_id(pp),"no type found while in get_array_type");
   std::string id = get_id(pp);
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
      const std::string &id = peekid(pp);
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
      //const bool is_pointer = arg.is_ptr || arg.is_amp;
      //const bool underscore = is_pointer;
      pp.out << /*(underscore?"_":"") <<*/ id;
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
         std::string t_id("t_");
         t_id += id;
         std::transform(t_id.begin(), t_id.end(), t_id.begin(), ::toupper);
         // if we have a hit, fake it has_default_value to trig the args to <>
         if (pp.ker.Tparams_src.find(t_id) != std::string::npos)
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
// not used anymore
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
   pp.ker.is_prefix = false;
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
   const std::string void_return_type = get_id(pp);
   pp.out << void_return_type;
   // Get kernel's name or namespace
   pp.ker.name.clear();
   pp.ker.space.clear();
   next(pp);
   const std::string name = get_id(pp);
   pp.out << name;
   pp.ker.name = name;
   if (is_namespace(pp))
   {
      check(pp,pp.in.peek()==':',"no 1st ':' in namespaced kernel");
      put(pp);
      check(pp,pp.in.peek()==':',"no 2st ':' in namespaced kernel");
      put(pp);
      const std::string real_name = get_id(pp);
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
   // jitAmpFromPtr(pp);
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
   while ('(' != get(pp)) { assert(not pp.in.eof()); }
   drop(pp);
   std::string depth = get_id(pp);
   //DBG("(%s)",depth.c_str());
   drop(pp);
   check(pp,is_right_parenthesis(pp),"no last right parenthesis found");
   get(pp);
   drop(pp); // ')'
   //check(pp,is_newline(pp.in.peek()),"no newline after unroll");
   //put(pp);
   // check(pp,is_semicolon(pp),"no last semicolon found");
   // get(pp);
   // only if we are in a forall, we push the unrolling
   if (pp.ker.is_forall)
   {
      pp.ker.forall.body += "#pragma unroll ";
      pp.ker.forall.body += depth.c_str();
      pp.ker.forall.body += "\n";
   }
}

// *****************************************************************************
// * MFEM_FORALL_[2|3]D
// *****************************************************************************
void forall(const std::string &id, context_t &pp)
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
   // Switch from prefix capturing, to the forall one
   pp.ker.is_prefix = false;
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
static bool token(const std::string &id, const char *token)
{
   if (strncmp(id.c_str(), "MFEM_", 5) != 0 ) { return false; }
   if (strcmp(id.c_str() + 5, token) != 0 ) { return false; }
   return true;
}

// *****************************************************************************
static void tokens(context_t &pp)
{
   if (peekn(pp, 4) != "MFEM") { return; }
   const std::string &id = get_id(pp);
   if (token(id, "JIT")) { return jit(pp); }
   if (token(id, "EMBED")) { return embed(pp); }
   if (token(id, "UNROLL")) { return unroll(pp); }
   if (token(id, "FORALL_2D")) { return forall(id,pp); }
   if (token(id, "FORALL_3D")) { return forall(id,pp); }
   if (pp.ker.is_embed) { pp.ker.embed += id; }
   // During the kernel prefix, add MFEM_* id tokens
   if (pp.ker.is_prefix) { pp.ker.prefix += id; }
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
   //pp.ker.is_jit = false;
   //pp.ker.is_embed = false;
   //pp.ker.is_prefix = false;
   //pp.ker.is_forall = false;
   //pp.ker.is_single_source = false;
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

