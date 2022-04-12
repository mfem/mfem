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
#include <regex>
#include <string>
#include <ciso646>
#include <cassert>
#include <iostream>

#include "../../config/config.hpp"
#include "jit.hpp"

using std::regex;

namespace mfem
{

namespace jit
{

struct argument_t
{
   int default_value = 0;
   std::string type, name;
   bool is_ptr = false, is_amp = false, is_cst = false,
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
   bool is_prefix = false;
   bool is_forall = false;
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
   struct forall_t forall;           // source of the lambda forall
   std::string prefix;
};


struct error_t
{
   int line;
   std::string file;
   const char *msg;
   error_t(int l, std::string f, const char *m): line(l), file(f), msg(m) {}
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

void check(context_t &pp, const bool test, const char *msg = nullptr)
{
   if (!test) { throw error_t(pp.line, pp.file,msg); }
}

void addComa(std::string &arg) { if (!arg.empty()) { arg += ",";  } }

void addArg(std::string &args, const char *arg) { addComa(args); args += arg; }

bool is_newline(const char c) { return c == '\n'; }

bool good(context_t &pp) { pp.in.peek(); return pp.in.good(); }

char get(context_t &pp) { return static_cast<char>(pp.in.get()); }

char put(const char c, context_t &pp)
{
   if (is_newline(c)) { pp.line++; }
   // if we are storing the lambda body, just save it w/o output
   if (pp.ker.is_forall) { pp.ker.forall.body += c; return c;}
   // if we are not yet in the forall, just store in the prefix
   if (pp.ker.is_jit && pp.ker.is_prefix) { pp.ker.prefix += c; }
   pp.out.put(c);
   return c;
}

char put(context_t &pp) { return put(get(pp), pp); }

void skip_space(context_t &pp, std::string &out)
{
   while (std::isspace(pp.in.peek())) { out += get(pp); }
}

void skip_space(context_t &pp, bool keep = true)
{
   while (std::isspace(pp.in.peek())) { keep ? put(pp) : get(pp); }
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

void singleLineComments(context_t &pp, bool keep = true)
{
   while (!is_newline(pp.in.peek())) { keep ? put(pp) : get(pp); }
}

void blockComments(context_t &pp, bool keep = true)
{
   for (char c; pp.in.get(c);)
   {
      if (keep) { put(c,pp); }
      if (c == '*' && pp.in.peek() == '/')
      {
         keep ? put(pp) : get(pp);
         skip_space(pp, keep);
         return;
      }
   }
}

void Comments(context_t &pp, bool keep = true)
{
   if (!is_comments(pp)) { return; }
   keep ? put(pp) : get(pp);
   if (keep ? put(pp) : get(pp) == '/') { return singleLineComments(pp, keep); }
   return blockComments(pp, keep);
}

void next(context_t &pp, bool keep = true)
{
   keep ? skip_space(pp) : drop_space(pp);
   Comments(pp, keep);
}

void drop(context_t &pp) { next(pp, false); }

bool is_id(context_t &pp)
{
   const char c = pp.in.peek();
   return std::isalnum(c) or c == '_';
}

std::string get_id(context_t &pp)
{
   std::string id;
   check(pp, is_id(pp), "name w/o alnum 1st letter");
   while (is_id(pp)) { id += get(pp); }
   return id;
}

bool is_digit(context_t &pp) { return std::isdigit(static_cast<char>(pp.in.peek())); }

int get_digit(context_t &pp)
{
   std::string digits;
   check(pp, is_digit(pp), "unknown number");
   while (is_digit(pp)) { digits += get(pp); }
   return atoi(digits.c_str());
}

std::string peekn(context_t &pp, const int n)
{
   int k = 0;
   assert(n < 64);
   static char c[64];
   for (k = 0; k <= n; k++) { c[k] = 0; }
   for (k = 0; k < n && good(pp); k++) { c[k] = get(pp); }
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
      if (!is_id(pp)) { break; }
      c[k] = get(pp);
      assert(!pp.in.eof());
   }
   std::string rtn(c);
   for (int l = 0; l < k; l++) { pp.in.unget(); }
   return rtn;
}

void drop_name(context_t &pp) { while (is_id(pp)) { get(pp); } }

bool is_string(context_t &pp, const char *str)
{
   skip_space(pp);
   const std::string peek_str = peekn(pp, strlen(str));
   assert(not pp.in.eof());
   return peek_str == str;
}

bool is_template(context_t &pp) { return is_string(pp, "template"); }
bool is_static(context_t &pp) { return is_string(pp, "static"); }
bool is_namespace(context_t &pp) { return is_string(pp, "::"); }
bool is_void(context_t &pp) { return is_string(pp, "void"); }

template <unsigned char UCHR>
bool is_char(context_t &pp) { skip_space(pp); return pp.in.peek() == UCHR; }

bool is_eq(context_t &pp) { return is_char<'='>(pp); }
bool is_amp(context_t &pp) { return is_char<'&'>(pp); }
bool is_coma(context_t &pp) { return is_char<','>(pp); }
bool is_star(context_t &pp) { return is_char<'*'>(pp); }
bool is_semicolon(context_t &pp) { return is_char<';'>(pp); }
bool is_left_parenthesis(context_t &pp) { return is_char<'('>(pp); }
bool is_right_parenthesis(context_t &pp) { return is_char<')'>(pp); }

/**
 * @brief PrepareJitArgs
 * @param pp
 */
void PrepareJitArgs(context_t &pp)
{
   assert(pp.ker.is_jit);
   pp.ker.mfem_jit_cxx = "\"" MFEM_JIT_CXX "\"";
   pp.ker.mfem_jit_build_flags = "\"" MFEM_JIT_BUILD_FLAGS "\"";
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

   for (argument_it ia = pp.args.begin(); ia != pp.args.end() ; ia++)
   {
      const argument_t &arg = *ia;
      const bool is_cst = arg.is_cst;
      const bool is_amp = arg.is_amp;
      const bool is_ptr = arg.is_ptr;
      const bool is_ref = is_ptr or is_amp;
      const char *type = arg.type.c_str();
      const char *name = arg.name.c_str();
      const bool has_default_value = arg.has_default_value;
      // const and not reference/pointer => add it to the template args
      if (is_cst and not is_ref and has_default_value)
      {
         //DBG(" => 1")
         const bool is_double = strcmp(type,"double")==0;
         assert(!is_double);
         // Tformat
         addComa(pp.ker.Tformat);
         if (!has_default_value)
         {
            pp.ker.Tformat += is_double ? "0x%lx" : "%d";
         }
         else
         {
            pp.ker.Tformat += "%d";
         }
         // Targs
         addComa(pp.ker.Targs);
         pp.ker.Targs += is_double ? "u" : "";
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
      if (is_cst and not is_ref and not has_default_value)
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
      if (not is_cst and not is_ref)
      {
         //DBG(" => 3")
         addArg(pp.ker.args, name);
         addArg(pp.ker.args_wo_amp, name);
         addArg(pp.ker.params, type);
         pp.ker.params += " ";
         pp.ker.params += name;
      }
      //
      if (is_cst and not is_ref and has_default_value)
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
      if (is_ref)
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
         pp.ker.params += is_cst ? "const ":"";
         pp.ker.params += type;
         pp.ker.params += " *";
         //pp.ker.params += is_pointer?"_":"";
         pp.ker.params += name;
      }
   }
   addComa(pp.ker.Tparams);
   pp.ker.Tparams += pp.ker.Tparams_src;
}

/**
 * @brief GenerateJitPrefix
 * @param pp
 */
void GenerateJitPrefix(context_t &pp)
{
   assert(pp.ker.is_jit);
   pp.ker.is_prefix = true;
   pp.ker.prefix.clear();
   pp.out << "\n\tconst char *src=R\"_(";
   pp.out << "#include <cstdint>\n";
   pp.out << "#include <limits>\n";
   pp.out << "#include <cstring>\n";
   pp.out << "#include <stdbool.h>\n";

   pp.out << "#include \""
          << pp.ker.mfem_install_dir
          << "/include/mfem/general/jit/jit.hpp\"\n";

   pp.out << "#include \""
          << pp.ker.mfem_install_dir
          << "/include/mfem/general/forall.hpp\"\n";

   pp.out << "\nusing namespace mfem;\n";
   pp.out << "\ntemplate<" << pp.ker.Tparams << ">";
   pp.out << "\nvoid " << pp.ker.name << "_%016lx(";
   pp.out << "const bool use_dev,";
   pp.out << pp.ker.params << "){";
   if (not pp.ker.d2u.empty()) { pp.out << "\n\t" << pp.ker.d2u; }
   // Starts counting the block depth
   pp.block = 0;
}

/**
 * @brief JitPostfix
 * @param pp
 */
void JitPostfix(context_t &pp)
{
   if (!pp.ker.is_jit) { return; }
   if (pp.block >= 0 && pp.in.peek() == '{') { pp.block++; }
   if (pp.block >= 0 && pp.in.peek() == '}') { pp.block--; }
   // nothing to do while we have not went out of last block statement
   if (pp.block != -1) { return; }
   pp.out << "}\nextern \"C\" void "
          << MFEM_JIT_PREFIX_CHAR << "%016lx("
          << "const bool use_dev, " << pp.ker.params << "){";
   pp.out << pp.ker.name << "_%016lx<" << pp.ker.Tformat << ">"
          << "(" << "use_dev, " << pp.ker.args_wo_amp << ");";
   pp.out << "})_\";";

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
   pp.out << "\n";
   // Stop counting the blocks and flush the kernel status
   pp.block--;
   pp.ker.is_jit = false;
}

// *****************************************************************************
bool jitGetArgs(context_t &pp)
{
   bool empty = true;
   argument_t arg;
   pp.args.clear();

   skip_space(pp); // Go to first possible argument

   if (is_void(pp)) { drop_name(pp); return true; }

   for (int argc = 0; true; empty=false)
   {
      if (is_star(pp)) { arg.is_ptr = true; put(pp); continue; }
      if (is_amp(pp)) { arg.is_amp = true; put(pp); continue; }
      if (is_coma(pp)) { put(pp); continue; }
      if (is_left_parenthesis(pp)) { argc += 1; put(pp); continue; }
      const std::string &id = peekid(pp);
      drop_name(pp);
      // Qualifiers
      if (id=="const") { pp.out << id; arg.is_cst = true; continue; }
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
      pp.out << id;
      // focus on the name, we should have qual & type
      arg.name = id;
      // now check for a possible default value
      next(pp);
      if (is_eq(pp)) // found a default value after a '='
      {
         put(pp);
         next(pp);
         arg.has_default_value = true;
         assert(is_digit(pp)); // verify next token is a digit
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
      assert(not pp.in.eof());
      if (is_right_parenthesis(pp)) { argc -= 1; if (argc >= 0) { put(pp); continue; } }
      // end of the arguments
      if (argc < 0) { break; }
      check(pp, pp.in.peek()==',', "no coma while in args");
      put(pp);
   }
   // Prepare the kernel strings from the arguments
   PrepareJitArgs(pp);
   return empty;
}

// *****************************************************************************
// not used anymore
void jitAmpFromPtr(context_t &pp)
{
   for (argument_it ia = pp.args.begin(); ia != pp.args.end() ; ia++)
   {
      const argument_t a = *ia;
      const bool is_const = a.is_cst;
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

/**
 * @brief jit
 * @param pp
 */
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
   assert(is_void(pp)); // make sure the returm type is void
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
      assert(false);
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
   GenerateJitPrefix(pp);
   // Push the preprocessor #line directive
   pp.out << "\n#line " << pp.line
          << " \"" //<< pp.ker.mfem_source_dir << "/"
          << pp.file << "\"";
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

bool is_token(const std::string &id, const char *token)
{
   if (strncmp(id.c_str(), "MFEM_", 5) != 0 ) { return false; }
   if (strcmp(id.c_str() + 5, token) != 0 ) { return false; }
   return true;
}

void Tokens(context_t &pp)
{
   if (peekn(pp, 4) != "MFEM") { return; }
   const std::string &id = get_id(pp);
   if (is_token(id, "JIT")) { return jit(pp); }
   //if (is_token(id, "UNROLL")) { return unroll(pp); }
   //if (is_token(id, "FORALL_2D")) { return forall(id,pp); }
   //if (is_token(id, "FORALL_3D")) { return forall(id,pp); }
   // During the kernel prefix, add MFEM_* id tokens
   if (pp.ker.is_prefix) { pp.ker.prefix += id; }
   // During the forall body, add MFEM_* id tokens
   if (pp.ker.is_forall) { assert(false); pp.ker.forall.body += id; return; }
   pp.out << id;
}

bool eof(context_t &pp)
{
   const char c = get(pp);
   if (pp.in.eof()) { return true; }
   put(c,pp);
   return false;
}

int preprocess(std::istream &in, std::ostream &out, std::string &file)
{
   mfem::jit::context_t pp(in, out, file);
   try
   {
      do
      {
         Tokens(pp);
         Comments(pp);
         JitPostfix(pp);
         forallPostfix(pp);
      }
      while (!eof(pp));
   }
   catch (mfem::jit::error_t err)
   {
      std::cerr << std::endl << err.file << ":" << err.line << ":"
                << " mjit error"
                << (err.msg ? ": " : "")
                << (err.msg ? err.msg : "")
                << std::endl;
      //remove(out.c_str());
      return ~0;
   }
   return 0;
}

} // namespace jit

} // namespace mfem

