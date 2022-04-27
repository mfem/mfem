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
#include <cstring>
#include <cassert>
#include <iostream>
#include <algorithm>

#include "jit.hpp"

namespace mfem
{

struct JitPreProcessor
{

   struct argument_t
   {
      int default_value = 0;
      std::string type, name;
      bool is_ptr = false, is_cst = false,
           is_restrict = false, is_tpl = false, has_default_value = false;
      argument_t() {}
   };

   typedef std::list<argument_t>::iterator argument_it;

   struct kernel_t
   {
      bool is_jit = false;
      bool is_kernel = false;
      std::string name;                 // kernel name
      // Templates: format, arguments and parameters
      std::string Tformat;              // template format, as in printf
      std::string Targs;                // template arguments, for hash and call
      std::string Tparams;              // template parameters, for the declaration
      std::string Tparams_src;          // template parameters, original source
      // Arguments and parameter for the standard calls
      // We need two kinds of arguments because of the '& <=> *' transformation
      // (This might be no more the case as we are getting rid of Array/Vector).
      std::string params;
      std::string args;
      std::string args_wo_amp;
      std::string src;
   };

   struct error_t
   {
      int line;
      std::string file;
      const char *error;
      error_t(int l, std::string f, const char *e): line(l), file(f), error(e) {}
   };

   void check(const bool test, const char *error)
   {
      if (!test) { throw error_t(line, file, error); }
   }

   void add_coma(std::string &arg) { if (!arg.empty()) { arg += ",";  } }

   void add_arg(std::string &as, const char *a) { add_coma(as); as += a; }

   bool is_newline(const char c) { return c == '\n'; }

   bool good() { in.peek(); return in.good(); }

   char get() { return static_cast<char>(in.get()); }

   char put_char(const char c)
   {
      if (is_newline(c)) { line++; }
      if (ker.is_kernel) { ker.src += c; }
      else { out.put(c); }
      return c;
   }

   char put() { return put_char(get()); }

   void skip_space(bool keep = true)
   {
      while (std::isspace(in.peek())) { keep ? put() : get(); }
   }

   void drop_space() { while (isspace(in.peek())) { get(); } }

   bool is_comments()
   {
      if (in.peek() != '/') { return false; }
      in.get();
      assert(!in.eof());
      const int c = in.peek();
      in.unget();
      if (c == '/' || c == '*') { return true; }
      return false;
   }

   void single_line_comments(bool keep = true)
   {
      while (!is_newline(in.peek())) { keep ? put() : get(); }
   }

   void block_comments(bool keep = true)
   {
      for (char c; in.get(c);)
      {
         if (keep) { put(); }
         if (c == '*' && in.peek() == '/')
         {
            keep ? put() : get();
            skip_space(keep);
            return;
         }
      }
   }

   void comments(bool keep = true)
   {
      if (!is_comments()) { return; }
      keep ? put() : get();
      if (keep ? put() : get() == '/') { return single_line_comments(keep); }
      return block_comments(keep);
   }

   void next(bool keep = true)
   {
      keep ? skip_space() : drop_space();
      comments(keep);
   }

   bool is_id()
   {
      const char c = in.peek();
      return std::isalnum(c) || c == '_';
   }

   std::string get_id()
   {
      std::string id;
      check(is_id(), "name w/o alnum 1st letter");
      while (is_id()) { id += get(); }
      return id;
   }

   bool is_digit() { return std::isdigit(static_cast<char>(in.peek())); }

   int get_digit()
   {
      std::string digits;
      check(is_digit(), "unknown number");
      while (is_digit()) { digits += get(); }
      return atoi(digits.c_str());
   }

   std::string peek_n(const int n)
   {
      int k = 0;
      assert(n < 64);
      static char c[64];
      for (k = 0; k <= n; k++) { c[k] = 0; }
      for (k = 0; k < n && good(); k++) { c[k] = get(); }
      std::string rtn(c);
      assert(!in.fail());
      for (int l = 0; l < k; l++) { in.unget(); }
      return rtn;
   }

   std::string peek_id()
   {
      int k = 0;
      constexpr int n = 64;
      static char c[64];
      for (k = 0; k < n; k++) { c[k] = 0; }
      for (k = 0; k < n; k++)
      {
         if (!is_id()) { break; }
         c[k] = get();
         assert(!in.eof());
      }
      std::string str(c);
      for (int l = 0; l < k; l++) { in.unget(); }
      return str;
   }

   void drop_name() { while (is_id()) { get(); } }

   bool is_string(const char *str) { skip_space(); return peek_n(std::strlen(str)) == str; }
   bool is_template() { return is_string("template"); }
   bool is_static() { return is_string("static"); }
   bool is_namespace() { return is_string("::"); }
   bool is_void() { return is_string("void"); }

   bool is_char(const unsigned char c) { skip_space(); return in.peek() == c; }
   bool is_eq() { return is_char('='); }
   bool is_amp() { return is_char('&'); }
   bool is_coma() { return is_char(','); }
   bool is_star() { return is_char('*'); }
   bool is_semicolon() { return is_char(';'); }
   bool is_left_parenthesis() { return is_char('('); }
   bool is_right_parenthesis() { return is_char(')'); }

   bool args()
   {
      bool empty = true;
      argument_t arg;
      std::list<argument_t> args;
      args.clear();

      skip_space(); // Go to first possible argument

      if (is_void()) { drop_name(); return true; }

      for (int argc = 0; true; empty = false)
      {
         if (is_star()) { arg.is_ptr = true; put(); continue; }
         if (is_coma()) { put(); continue; }
         if (is_left_parenthesis()) { argc += 1; put(); continue; }
         const std::string &id = peek_id();
         drop_name();
         // Qualifiers
         if (id=="const") { out << id; arg.is_cst = true; continue; }
         if (id=="restrict") { out << id; arg.is_restrict = true; continue; }
         if (id=="__restrict") { out << id; arg.is_restrict = true; continue; }
         // Types
         if (id=="char") { out << id; arg.type = id; continue; }
         if (id=="int") { out << id; arg.type = id; continue; }
         if (id=="short") { out << id; arg.type = id; continue; }
         if (id=="unsigned") { out << id; arg.type = id; continue; }
         if (id=="long") { out << id; arg.type = id; continue; }
         if (id=="bool") { out << id; arg.type = id; continue; }
         if (id=="float") { out << id; arg.type = id; continue; }
         if (id=="double") { out << id; arg.type = id; continue; }
         if (id=="size_t") { out << id; arg.type = id; continue; }
         out << id;
         // focus on the name, we should have qual & type
         arg.name = id;
         // now check for a possible default value
         next();
         if (is_eq()) // found a default value after a '='
         {
            put();
            next();
            arg.has_default_value = true;
            assert(is_digit()); // verify next token is a digit
            arg.default_value = get_digit();
            out << arg.default_value;
         }
         else
         {
            // check if id has a T_id in ker.Tparams_src
            std::string t_id("t_");
            t_id += id;
            std::transform(t_id.begin(), t_id.end(), t_id.begin(), ::toupper);
            // if we have a hit, fake it has_default_value to trig the args to <>
            if (ker.Tparams_src.find(t_id) != std::string::npos)
            {
               arg.has_default_value = true;
               arg.default_value = 0;
            }
         }
         args.push_back(arg);
         arg = argument_t();
         assert(not in.eof());
         if (is_right_parenthesis()) { argc -= 1; if (argc >= 0) { put(); continue; } }
         // end of the arguments
         if (argc < 0) { break; }
         check(in.peek()==',', "no coma while in args");
         put();
      }

      // Prepare the kernel strings from the arguments
      assert(ker.is_jit);
      ker.Targs.clear();
      ker.Tparams.clear();
      ker.Tformat.clear();
      ker.args.clear();
      ker.params.clear();
      ker.args_wo_amp.clear();

      for (argument_it ia = args.begin(); ia != args.end() ; ia++)
      {
         //const argument_t &arg = *ia;
         arg = *ia;
         const bool is_cst = arg.is_cst;
         const bool is_ptr = arg.is_ptr;
         const char *type = arg.type.c_str();
         const char *name = arg.name.c_str();
         const bool has_default_value = arg.has_default_value;
         // const and not reference/pointer => add it to the template args
         if (! is_ptr && has_default_value)
         {
            //DBG(" => 1")
            add_coma(ker.Tformat);
            ker.Tformat += "%d";
            add_coma(ker.Targs);
            ker.Targs += name;
            if (!has_default_value)
            {
               add_coma(ker.Tparams);
               ker.Tparams += "const ";
               ker.Tparams += type;
               ker.Tparams += " ";
               ker.Tparams += name;
            }
         }

         //
         if (is_cst && !is_ptr && !has_default_value)
         {
            //DBG(" => 2")
            add_arg(ker.args, name);
            add_arg(ker.args_wo_amp, name);
            add_arg(ker.params, "const ");
            ker.params += type;
            ker.params += " ";
            ker.params += name;
         }

         // !const && !pointer => std args
         if (!is_cst && !is_ptr)
         {
            //DBG(" => 3")
            add_arg(ker.args, name);
            add_arg(ker.args_wo_amp, name);
            add_arg(ker.params, type);
            ker.params += " ";
            ker.params += name;
         }
         //
         if (is_cst && !is_ptr && has_default_value)
         {
            //DBG(" => 4")
            add_arg(ker.params, " const ");
            ker.params += type;
            ker.params += " ";
            ker.params += name;
            // other_arguments_wo_amp
            add_arg(ker.args_wo_amp, "0");
            // other_arguments
            add_arg(ker.args, "0");
         }

         // pointer
         if (is_ptr)
         {
            //DBG(" => 5")
            // other_arguments
            if (!ker.args.empty()) { ker.args += ","; }
            ker.args += name;
            // other_arguments_wo_amp
            if (! ker.args_wo_amp.empty()) {  ker.args_wo_amp += ","; }
            ker.args_wo_amp += name;
            // other_parameters
            if (not ker.params.empty()) { ker.params += ",";  }
            ker.params += is_cst ? "const ":"";
            ker.params += type;
            ker.params += " *";
            ker.params += name;
         }
      }
      add_coma(ker.Tparams);
      ker.Tparams += ker.Tparams_src;

      return empty;
   }

   void mfem_jit()
   {
      ker.is_jit = true;
      next();
      check(is_template(), "kernel is missing the 'template' token");

      // copy the 'template<...>' in Tparams_src
      out << get_id();
      next();
      check(in.peek()=='<',"no '<' in single source kernel!");
      put();
      ker.Tparams_src.clear();
      while (in.peek() != '>')
      {
         assert(not in.eof());
         char c = get();
         put_char(c);
         ker.Tparams_src += c;
      }
      put();

      // 'static' check
      if (is_static()) { out << get_id(); }
      next();
      check(is_void(), "kernel return type should be void");
      assert(is_void()); // make sure the returm type is void
      const std::string void_return_type = get_id();
      out << void_return_type;
      // Get kernel's name or namespace
      ker.name.clear();
      next();
      const std::string name = get_id();
      out << name;
      ker.name = name;
      next();
      // check we are at the left parenthesis
      check(in.peek()=='(',"no 1st '(' in kernel");
      put();
      // Get the arguments
      args();
      // Make sure we have hit the last ')' of the arguments
      check(in.peek()==')',"no last ')' in kernel");
      put();
      next();
      // Make sure we are about to start a compound statement
      check(in.peek()=='{',"no compound statement found");
      put();
      // Generate the kernel prefix for this kernel
      assert(ker.is_jit);
      ker.src.clear();
      out << "\n\tconst char *src = R\"_(";

      // switching from out to ker.src to compute the hash
      //ker.src += "#include <cstdint>\n";
      //ker.src += "#include <limits>\n";
      //ker.src += "#include <cstring>\n";

      ker.src += "#define MFEM_JIT_FORALL_COMPILATION\n";
      ker.src += "#define MFEM_DEVICE_HPP\n";

      ker.src += "#include \"";
      ker.src += MFEM_INSTALL_DIR;
      ker.src += "/include/mfem/general/forall.hpp\"\n";
      ker.src += "\nusing namespace mfem;\n";

      ker.src += "\ntemplate<";
      ker.src += ker.Tparams;
      ker.src += ">";
      ker.src += "\nvoid ";
      ker.src += ker.name;
      ker.src += "_%016lx(";
      ker.src += "const bool use_dev,";
      ker.src += ker.params;
      ker.src += "){";

      // Push the preprocessor #line directive
      ker.src += "\n#line ";
      ker.src += std::to_string(line);
      ker.src += " \"";
      ker.src += file;
      ker.src += "\"";

      // Starts counting the block depth
      block = 0;
      ker.is_kernel = true;
   }

   void postfix()
   {
      if (!ker.is_jit) { return; }
      if (block >= 0 && in.peek() == '{') { block++; }
      if (block >= 0 && in.peek() == '}') { block--; }
      // nothing to do while we have not went out of last block statement
      if (block != -1) { return; }
      ker.src += "}\nextern \"C\" void k%016lx(";
      ker.src += "const bool use_dev, ";
      ker.src += ker.params;
      ker.src += "){";
      ker.src += ker.name;
      ker.src += "_%016lx<";
      ker.src += ker.Tformat;
      ker.src += ">(use_dev, ";
      ker.src += ker.args_wo_amp;
      ker.src += ");}";

      out << ker.src; // output all kernel source, after having computed its hash
      out << ")_\";"; // eos
      ker.is_kernel = false;

      const size_t seed = Jit::Hash(std::hash<std::string> {}(ker.src),
                                    std::string(MFEM_JIT_CXX),
                                    std::string(MFEM_JIT_BUILD_FLAGS),
                                    std::string(MFEM_SOURCE_DIR),
                                    std::string(MFEM_INSTALL_DIR));
      // kernel hash
      out << "\n\tconst size_t hash = Jit::Hash("
          << "0x" << std::hex << seed << std::dec << "ul, "
          << ker.Targs << ");";

      // kernel typedef
      out << "\n\ttypedef void (*kernel_t)(bool use_dev," << ker.params << ");";

      // kernel map
      out << "\n\tstatic std::unordered_map<size_t, Jit::Kernel<kernel_t>*> ks;";

      // Add kernel in map if not already present
      out << "\n\tif (!ks[hash]){";
      out << "\n\t\tconst int n = 1 + " // source size
          << "snprintf(nullptr, 0, src, hash, hash, hash, " << ker.Targs << ");";
      out << "\n\t\tchar *Tsrc = new char[n];";
      out << "\n\t\tsnprintf(Tsrc, n, src, hash, hash, hash, "<< ker.Targs << ");";
      out << "\n\t\tstd::stringstream ss;"; // prepare symbol from computed hash
      out << "\n\t\tss << 'k' << std::setfill('0') ";
      out << "<< std::setw(16) << std::hex << (hash|0) << std::dec;";
      out << "\n\t\tconst int SYMBOL_SIZE = 1+16+1;";
      out << "\n\t\tchar *symbol = new char[SYMBOL_SIZE];";
      out << "memcpy(symbol, ss.str().c_str(), SYMBOL_SIZE);";
      out << "\n\t\tks[hash] = new Jit::Kernel<kernel_t>(hash, Tsrc, symbol);";
      out << "\n\t\tassert(ks[hash]);";
      out << "\n\t\tdelete[] Tsrc;";
      out << "\n\t}";

      // #warning should be CUDA dependent
      out << "\n\tconst bool use_dev = Device::Allows(Backend::CUDA_MASK);";

      out << "\n\tassert(ks[hash]);";
      out << "\n\tks[hash]->operator()(use_dev," << ker.args << ");\n";

      // Stop counting the blocks and flush the kernel status
      block--;
      ker.is_jit = false;
   }

   void tokens()
   {
      struct
      {
         bool token(const std::string &id, const char *token)
         {
            if (strncmp(id.c_str(), "MFEM_", 5) != 0 ) { return false; }
            if (strcmp(id.c_str() + 5, token) != 0 ) { return false; }
            return true;
         }
      } mfem;
      if (peek_n(4) != "MFEM") { return; }
      const std::string &id = get_id();
      if (mfem.token(id, "JIT")) { return mfem_jit(); }
      if (ker.is_kernel) { ker.src += id; }
      else { out << id; }
   }

   bool eof()
   {
      const char c = get();
      if (in.eof()) { return true; }
      put_char(c);
      return false;
   }

   JitPreProcessor(std::istream &in, std::ostream &out, std::string &file) :
      in(in), out(out), file(file), line(2), block(-2) {}

   int operator()()
   {
      try { do { tokens(); comments(); postfix(); } while (!eof()); }
      catch (error_t err)
      {
         std::cerr << std::endl << err.file << ":" << err.line << ":"
                   << " mjit error"
                   << (err.error ? ": " : "")
                   << (err.error ? err.error : "")
                   << std::endl;
         return EXIT_FAILURE;
      }
      return EXIT_SUCCESS;
   }

private:
   kernel_t ker;
   std::istream &in;
   std::ostream &out;
   std::string &file;
   int line, block;
};

int JitPreProcess(std::istream &in, std::ostream &out, std::string &file)
{
   return JitPreProcessor(in,out,file).operator()();
}

} // namespace mfem

#endif // MFEM_USE_JIT
