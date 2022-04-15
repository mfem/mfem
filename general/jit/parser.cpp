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

#include "../../config/config.hpp"

#ifdef MFEM_USE_JIT

#include <list>
#include <string>
#include <iostream>
#include <algorithm>

#include "jit.hpp"

#define MFEM_JIT_STR(...) #__VA_ARGS__
#define MFEM_JIT_STRINGIFY(...) MFEM_JIT_STR(__VA_ARGS__)

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
 * @brief Postfix
 * @param pp
 */
void Postfix(context_t &pp)
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
   pp.out << "\n\tks[args_hash]->operator()("
          << "Device::Allows(Backend::CUDA_MASK), "
          << pp.ker.args << ");";
   pp.out << "\n\treturn;";
   pp.out << "\n";
   // Stop counting the blocks and flush the kernel status
   pp.block--;
   pp.ker.is_jit = false;
}

/**
 * @brief JitTokenPrefix
 * @param pp
 */
void JitTokenPrefix(context_t &pp)
{
   assert(pp.ker.is_jit);
   pp.ker.is_prefix = true;
   pp.ker.prefix.clear();
   pp.out << "\n\tconst char *src=R\"_(";
   pp.out << "#include <cstdint>\n";
   pp.out << "#include <limits>\n";
   pp.out << "#include <cstring>\n";
   pp.out << "#include <stdbool.h>\n";
   pp.out << "#include \"" << pp.ker.mfem_install_dir
          << "/include/mfem/general/jit/jit.hpp\"\n";
   pp.out << "#include \"" << pp.ker.mfem_install_dir
          << "/include/mfem/general/forall.hpp\"\n";
   pp.out << "\nusing namespace mfem;\n";
   pp.out << "\ntemplate<" << pp.ker.Tparams << ">";
   pp.out << "\nvoid " << pp.ker.name << "_%016lx(";
   pp.out << "const bool use_dev,";
   pp.out << pp.ker.params << "){";
   // Starts counting the block depth
   pp.block = 0;
}

/**
 * @brief JitTokenArgsString
 * @param pp
 */
void JitTokenArgsString(context_t &pp)
{
   assert(pp.ker.is_jit);
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

   for (argument_it ia = pp.args.begin(); ia != pp.args.end() ; ia++)
   {
      const argument_t &arg = *ia;
      const bool is_cst = arg.is_cst;
      const bool is_ptr = arg.is_ptr;
      const char *type = arg.type.c_str();
      const char *name = arg.name.c_str();
      const bool has_default_value = arg.has_default_value;
      // const and not reference/pointer => add it to the template args
      if (! is_ptr && has_default_value)
      {
         //DBG(" => 1")
         addComa(pp.ker.Tformat);
         pp.ker.Tformat += "%d";
         addComa(pp.ker.Targs);
         pp.ker.Targs += name;
         if (!has_default_value)
         {
            addComa(pp.ker.Tparams);
            pp.ker.Tparams += "const ";
            pp.ker.Tparams += type;
            pp.ker.Tparams += " ";
            pp.ker.Tparams += name;
         }
      }

      //
      if (is_cst && !is_ptr && !has_default_value)
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
      if (!is_cst && !is_ptr)
      {
         //DBG(" => 3")
         addArg(pp.ker.args, name);
         addArg(pp.ker.args_wo_amp, name);
         addArg(pp.ker.params, type);
         pp.ker.params += " ";
         pp.ker.params += name;
      }
      //
      if (is_cst && !is_ptr && has_default_value)
      {
         //DBG(" => 4")
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
      if (is_ptr)
      {
         //DBG(" => 5")
         // other_arguments
         if (!pp.ker.args.empty()) { pp.ker.args += ","; }
         pp.ker.args += name;
         // other_arguments_wo_amp
         if (! pp.ker.args_wo_amp.empty()) {  pp.ker.args_wo_amp += ","; }
         pp.ker.args_wo_amp += name;
         // other_parameters
         if (not pp.ker.params.empty()) { pp.ker.params += ",";  }
         pp.ker.params += is_cst ? "const ":"";
         pp.ker.params += type;
         pp.ker.params += " *";
         pp.ker.params += name;
      }
   }
   addComa(pp.ker.Tparams);
   pp.ker.Tparams += pp.ker.Tparams_src;
}

/**
 * @brief JitTokenArgs
 * @param pp
 * @return
 */
bool JitTokenArgs(context_t &pp)
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
   JitTokenArgsString(pp);
   return empty;
}

/**
 * @brief JitToken
 * @param pp
 */
void JitToken(context_t &pp)
{
   pp.ker.is_jit = true;
   pp.ker.is_prefix = false;
   next(pp);
   // return type should be void for now, can hit a 'static' or 'template'
   const bool check_next_id = is_void(pp) or is_static(pp) or is_template(pp);
   // first check for the template
   check(pp, check_next_id, "kernel w/o void, static or template");
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
   next(pp);
   // check we are at the left parenthesis
   check(pp,pp.in.peek()=='(',"no 1st '(' in kernel");
   put(pp);
   // Get the arguments
   JitTokenArgs(pp);
   // Make sure we have hit the last ')' of the arguments
   check(pp,pp.in.peek()==')',"no last ')' in kernel");
   put(pp);
   next(pp);
   // Make sure we are about to start a compound statement
   check(pp,pp.in.peek()=='{',"no compound statement found");
   put(pp);
   // Generate the kernel prefix for this kernel
   JitTokenPrefix(pp);
   // Push the preprocessor #line directive
   pp.out << "\n#line " << pp.line
          << " \"" //<< pp.ker.mfem_source_dir << "/"
          << pp.file << "\"";
}

void Tokens(context_t &pp)
{
   struct
   {
      bool token(const std::string &id, const char *token)
      {
         if (strncmp(id.c_str(), "MFEM_", 5) != 0 ) { return false; }
         if (strcmp(id.c_str() + 5, token) != 0 ) { return false; }
         return true;
      }
   } is;
   if (peekn(pp, 4) != "MFEM") { return; }
   const std::string &id = get_id(pp);
   if (is.token(id, "JIT")) { return JitToken(pp); }
   // During the kernel prefix, add MFEM_* id tokens
   if (pp.ker.is_prefix) { pp.ker.prefix += id; }
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
   try { do { Tokens(pp); Comments(pp); Postfix(pp); } while (!eof(pp)); }
   catch (mfem::jit::error_t err)
   {
      std::cerr << std::endl << err.file << ":" << err.line << ":"
                << " mjit error"
                << (err.msg ? ": " : "")
                << (err.msg ? err.msg : "")
                << std::endl;
      return EXIT_FAILURE;
   }
   return EXIT_SUCCESS;
}

} // namespace jit

} // namespace mfem

#endif // MFEM_USE_JIT
