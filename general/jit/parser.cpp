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

#include "jit.hpp" // fot hash combine

namespace mfem
{

#define STOP { fflush(0); assert(false); }

struct JitPreProcessor
{
   struct dbg
   {
      static constexpr bool DEBUG = false;
      static constexpr uint8_t COLOR = 226;
      dbg(): dbg(COLOR) { }
      dbg(const uint8_t color)
      {
         if (!DEBUG) { return; }
         std::cout << "\033[38;5;" << std::to_string(color==0?COLOR:color) << "m";
      }
      ~dbg() { if (DEBUG) { std::cout << "\033[m"; std::cout.flush(); } }
      template <typename T> dbg& operator<<(const T &arg)
      { if (DEBUG) { std::cout << arg;} return *this; }
      template<typename T, typename... Args>
      inline void operator()(const T &arg, Args... args) const
      { operator<<(arg); operator()(args...); }
      template<typename T>
      inline void operator()(const T &arg) const { operator<<(arg); }
   };

   struct argument_t
   {
      int default_value = 0;
      std::string type, name;
      bool is_ptr = false, is_cst = false,
           is_restrict = false, is_tpl = false, has_default_value = false;
      argument_t() {}
   };

   typedef std::list<argument_t>::iterator argument_it;

   struct forall_t { int dim; std::string e, N, X, Y, Z, body; };

   struct fsm_t
   {
      const uint8_t fsm = 201;
      typedef void (fsm_t::*State) ();
      State state;
      void Next(State target) { state = target; }

      void Wait()    { dbg(fsm)<<"\n[JIT]";     Next(&fsm_t::Jit); }
      void Jit()     { dbg(fsm)<<"\n[TARG]";    Next(&fsm_t::Targs); }
      void Targs()   { dbg(fsm)<<"\n[SIGN]";    Next(&fsm_t::Sign); }
      void Sign()    { dbg(fsm)<<"\n[SOURCE]";  Next(&fsm_t::Source); }
      void Source()  { dbg(fsm)<<"\n[FORALL]";  Next(&fsm_t::Forall); }
      void Forall()  { dbg(fsm)<<"\n[BODY]";    Next(&fsm_t::Body); }
      void Body()    { dbg(fsm)<<"\n[POSTFIX]"; Next(&fsm_t::Postfix); }
      void Postfix() { dbg(fsm)<<"\n[WAIT]";    Next(&fsm_t::Wait); }

      fsm_t(): state(&fsm_t::Wait) { dbg(fsm)<<"\n[WAIT]"; }
      void advance() { (this->*state)(); }
   };

   struct kernel_t
   {
      fsm_t fsm;
      bool is_wait() { return fsm.state == &fsm_t::Wait; }
      bool is_jit() { return fsm.state == &fsm_t::Jit; }
      bool is_Targs() { return fsm.state == &fsm_t::Targs; }
      bool is_sign() { return fsm.state == &fsm_t::Sign; }
      bool is_source() { return fsm.state == &fsm_t::Source; }
      bool is_forall() { return fsm.state == &fsm_t::Forall; }
      bool is_body() { return fsm.state == &fsm_t::Body; }
      bool is_postfix() { return fsm.state == &fsm_t::Postfix; }
      void advance() { fsm.advance(); }
      std::string name;           // kernel name
      // Templates: format, arguments and parameters
      std::string Tformat;        // template format, as in printf
      std::string Targs;          // template arguments, for hash and call
      std::string Tparams;        // template parameters, for the declaration
      std::string Tparams_src;    // template parameters, original source
      // Arguments and parameter for the standard calls
      // We need two kinds of arguments because of the '& <=> *' transformation
      // (This might be no more the case as we are getting rid of Array/Vector).
      std::string params;
      std::string args;
      std::string args_wo_amp;
      struct forall_t forall;    // source of the lambda forall
      std::string source;
   };

   struct error_t
   {
      int line;
      std::string file;
      const char *error;
      error_t(int l, std::string f, const char *e): line(l), file(f), error(e) {}
   };

   kernel_t ker;
   std::istream &in;
   std::ostream &out;
   std::string &file;
   int line, block, parenthesis;

   JitPreProcessor(std::istream &in, std::ostream &out, std::string &file) :
      in(in), out(out), file(file),
      line(2), block(-2), parenthesis(-2) {}

   void check(const bool test, const char *error)
   {
      if (!test) { throw error_t(line, file, error); }
   }

   void add_coma(std::string &arg) { if (!arg.empty()) { arg += ",";  } }

   void add_arg(std::string &as, const char *a) { add_coma(as); as += a; }

   bool good() { in.peek(); return in.good(); }

   char get() { return static_cast<char>(in.get()); }

   char put(const char c)
   {
      if (ker.is_Targs()) { ker.Tparams_src += c; /* also out */} // line ?
      if (ker.is_forall()) { ker.forall.body += c; return c;}
      if (ker.is_body()) { ker.forall.body += c; return c;}
      if (ker.is_source()) { ker.source += c; return c;}
      out.put(c);
      if (c == '\n') { dbg(246) << line << "|"; line++; }
      return c;
   }

   char put() { return put(get()); }

   void skip_space(bool keep = true)
   {
      while (std::isspace(in.peek())) { keep ? put() : get(); }
   }

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
      dbg(123) << "single_line_comments";
      while (!is_linefeed()) { keep ? put() : get(); }
      dbg(123) << "\\n";
   }

   void block_comments(bool keep = true)
   {
      dbg(121) << "block_comments";
      while (peek_n(2) != "*/") { assert(!in.eof()); keep ? put(): get(); }
      const char c1 = keep ? put(): get(); assert(c1=='*');
      const char c2 = keep ? put(): get(); assert(c2=='/');
      dbg(121) << "*/";
   }

   void comments(bool keep = true)
   {
      while (is_comments())
      {
         const char c1 = keep ? put() : get();
         assert(c1=='/');
         const char c2 = keep ? put() : get();
         assert (c2 == '/' || c2 == '*');
         if (c2 == '/') { single_line_comments(keep); }
         else { block_comments(keep); }
         dbg()<<"skip_space<";
         skip_space(keep);
         dbg()<<">";
      }
   }

   void next(bool keep = true)
   {
      //dbg(229) << "[";
      skip_space(keep); comments(keep);
      //dbg(229) << "]";
   }

   void drop() { next(false); }

   bool is_id() { const char c = in.peek(); return std::isalnum(c) || c == '_'; }

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

   template<int M = 16> std::string peek_n(const int n)
   {
      int k = 0;
      assert(n < M);
      static char c[M];
      for (k = 0; k <= n; k++) { c[k] = 0; }
      for (k = 0; k < n && good(); k++) { c[k] = get(); }
      std::string rtn(c);
      if (!good()) { return rtn; }
      for (int l = 0; l < k; l++) { in.unget(); }
      return rtn;
   }

   template<int M = 16> std::string peek_id()
   {
      int k = 0;
      constexpr int n = M;
      static char c[M];
      for (k = 0; k < n; k++) { c[k] = 0; }
      for (k = 0; k < n; k++)
      {
         if (!is_id()) { break; }
         c[k] = get();
         assert(good());
         assert(!in.eof());
      }
      std::string str(c);
      for (int l = 0; l < k; l++) { in.unget(); }
      return str;
   }

   void drop_id() { while (is_id()) { get(); } }

   bool is_string(const char *str) { return peek_n(std::strlen(str)) == str; }
   bool is_template() { return is_string("template"); }
   bool is_static() { return is_string("static"); }
   bool is_void() { return is_string("void"); }

   bool is_char(const unsigned char c) { return in.peek() == c; }
   bool is_eq() { return is_char('='); }
   bool is_coma() { return is_char(','); }
   bool is_star() { return is_char('*'); }
   bool is_semicolon() { return is_char(';'); }
   bool is_left_parenthesis() { return is_char('('); }
   bool is_right_parenthesis() { return is_char(')'); }

   bool is_linefeed() { return is_char('\n'); }
   bool is_vertical_tab() { return is_char('\v'); }
   bool is_carriage_return() { return is_char('\r'); }

   bool mfem_jit_args()
   {
      bool empty = true;
      argument_t arg;
      std::list<argument_t> args;
      args.clear();

      for (int argc = 0; true; empty = false)
      {
         next();
         if (is_star()) { arg.is_ptr = true; put(); continue; }
         if (is_coma()) { put(); continue; }
         if (is_left_parenthesis()) { argc += 1; put(); continue; }
         const std::string &id = peek_id();
         drop_id();
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
         next();
         args.push_back(arg);
         arg = argument_t(); // prepare new argument
         assert(not in.eof());

         next();
         if (is_right_parenthesis())
         {
            argc -= 1;
            if (argc >= 0) // we had nested '('
            {
               put();
               continue;
            }
         }
         // end of the arguments
         if (argc < 0) { break; }
         check(in.peek()==',', "no coma while in args");
         put();
      }
      next();

      // Prepare the kernel strings from the arguments
      //assert(ker.is_prefix);
      ker.Targs.clear();
      ker.Tparams.clear();
      ker.Tformat.clear();
      ker.args.clear();
      ker.params.clear();
      ker.args_wo_amp.clear();

      for (argument_it arg_it = args.begin(); arg_it != args.end() ; arg_it++)
      {
         arg = *arg_it;
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

   void mfem_jit_prefix()
   {
      next();
      check(is_template(), "kernel is missing the 'template' token");

      // copy the 'template<...>' in Tparams_src
      out << get_id();
      next();
      check(in.peek()=='<',"no '<' in single source kernel!");
      put();
      ker.Tparams_src.clear();
      ker.advance(); // jit => Targ
      while (in.peek() != '>')
      {
         assert(good());
         next();
         put(); // will output in Tparams_src
      }
      ker.advance(); // Targ => sign
      const char Tc = put(); assert(Tc == '>');

      next(); // 'static' check
      if (is_static()) { out << get_id(); }

      next(); // 'void' check
      check(is_void(), "kernel return type should be void");

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
      mfem_jit_args();
      // Make sure we have hit the last ')' of the arguments
      check(in.peek()==')',"no last ')' in kernel");
      const char c = put(/*)*/);
      assert(c==')');
      next();
      // Make sure we are about to start a compound statement
      check(in.peek()=='{',"no compound statement found");
      put();
      // Generate the kernel prefix for this kernel
      ker.source.clear();
      out << "\n\tconst char *source = R\"_(";

      // switching from out to ker.source to compute the hash
      ker.source += "\n#define MFEM_JIT_FORALL_COMPILATION";
      ker.source += "\n#define MFEM_DEVICE_HPP";

      ker.source += "\n#include \"";
      ker.source += MFEM_INSTALL_DIR;
      ker.source += "/include/mfem/general/forall.hpp\"";
      ker.source += "\nusing namespace mfem;";

      ker.source += "\n\ntemplate<";
      ker.source += ker.Tparams;
      ker.source += ">";
      ker.source += "\nvoid ";
      ker.source += ker.name;
      ker.source += "_%016lx(";
      ker.source += "const bool use_dev,";
      ker.source += ker.params;
      ker.source += "){";

      // Push the preprocessor #line directive
      ker.source += "\n#line ";
      ker.source += std::to_string(line);
      ker.source += " \"";
      ker.source += file;
      ker.source += "\"";

      // Starts counting the block depth
      block = 0;
   }

   bool is_jit_postfix()
   {
      //if (ker.is_wait()) { return false; }
      //if (!ker.is_source()) { return false; }
      //if (ker.is_forall()) { return false; }
      if (block >= 0 && in.peek() == '{') { block++; }
      if (block >= 0 && in.peek() == '}') { block--; }
      // nothing to do while we have not went out of last block statement
      if (block == -1) { return true; }
      return false;
   }

   void mfem_jit_postfix()
   {
      ker.source += "}\nextern \"C\" void k%016lx(";
      ker.source += "const bool use_dev, ";
      ker.source += ker.params;
      ker.source += "){";
      ker.source += ker.name;
      ker.source += "_%016lx<";
      ker.source += ker.Tformat;
      ker.source += ">(use_dev, ";
      ker.source += ker.args_wo_amp;
      ker.source += ");}";

      // output all kernel source, after having computed its hash
      //#warning sources
      out << ker.source;
      out << ")_\";"; // eos

      const size_t seed = Jit::Hash(std::hash<std::string> {}(ker.source),
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
      out << "\n\tstatic std::unordered_map<size_t, Jit::Kernel<kernel_t>> ks;";

      // Add kernel in map if not already present
      out << "\n\tauto ks_iter = ks.find(hash);";
      out << "\n\tif (ks_iter == ks.end()){";
      out << "\n\t\tconst int n = 1 + " // source size
          << "snprintf(nullptr, 0, source, hash, hash, hash, " << ker.Targs << ");";
      out << "\n\t\tchar *Tsrc = new char[n];";
      out << "\n\t\tsnprintf(Tsrc, n, source, hash, hash, hash, "<< ker.Targs << ");";
      out << "\n\t\tstd::stringstream ss;"; // prepare symbol from computed hash
      out << "\n\t\tss << 'k' << std::setfill('0') ";
      out << "<< std::setw(16) << std::hex << (hash|0) << std::dec;";
      out << "\n\t\tconst int SYMBOL_SIZE = 1+16+1;";
      out << "\n\t\tchar *symbol = new char[SYMBOL_SIZE];";
      out << "\n\t\tmemcpy(symbol, ss.str().c_str(), SYMBOL_SIZE);";
      out << "\n\t\tauto res = ks.emplace(hash, Jit::Kernel<kernel_t>(hash, Tsrc, symbol));";
      out << "\n\t\tassert(res.second); // was not already in the map";
      out << "\n\t\tks_iter = ks.find(hash);";
      out << "\n\t\tassert(ks_iter != ks.end());";
      out << "\n\t\tdelete[] symbol;";
      out << "\n\t\tdelete[] Tsrc;";
      out << "\n\t}";

      // #warning should be CUDA dependent
      out << "\n\tconst bool use_dev = Device::Allows(Backend::CUDA_MASK);";
      out << "\n\tks_iter->second.operator()(use_dev," << ker.args << ");";

      // Stop counting the blocks and flush the kernel status
      block--;

      // Should count the difference!
      out << "\n//#line ";
      out << std::to_string(line);
      out << " \"";
      out << file;
      out << "\"\n";
   }

   void mfem_unroll()
   {
      assert (ker.is_body());
      while ('(' != get()) { assert(!in.eof()); }
      drop();
      std::string depth = get_id();
      dbg() << depth;

      drop();
      check(is_right_parenthesis(),"no last right parenthesis found");
      const char c = get(); // ')'
      assert(c==')');

      check(is_linefeed(),"no newline found");
      const char nl = get(); // '\n'
      assert(nl=='\n');

      ker.forall.body += "#pragma unroll ";
      ker.forall.body += depth.c_str();
      ker.forall.body += "\n";
   }

   void forall_prefix(const std::string &id)
   {
      const int dim = ker.forall.dim = id.c_str()[12] - 0x30;
      dbg() << id << ", dim:" << dim;

      ker.forall.body.clear();

      drop();
      check(is_left_parenthesis(),"no 1st '(' in MFEM_FORALL");
      get(); // drop '('

      drop();
      ker.forall.e = get_id();
      dbg() << ", iterator:'" << ker.forall.e << "'";

      drop();
      check(is_coma(),"no 1st coma in MFEM_FORALL");
      get(); // drop ','

      drop();
      check(is_id(),"no 1st id(N) in MFEM_FORALL");
      ker.forall.N = get_id();
      dbg() << ", N:'" << ker.forall.N << "'";

      drop();
      check(is_coma(),"no 2nd coma in MFEM_FORALL");
      get(); // drop ','

      drop();
      check(is_id(),"no 2st id (X) in MFEM_FORALL");
      ker.forall.X = get_id();
      dbg() << ", X:'" << ker.forall.X << "'";

      drop();
      check(is_coma(),"no 3rd coma in MFEM_FORALL");
      get(); // drop ','

      drop();
      check(is_id(),"no 3rd id (Y) in MFEM_FORALL");
      ker.forall.Y = get_id();
      dbg() << ", Y:'" << ker.forall.Y << "'";

      drop();
      check(is_coma(),"no 4th coma in MFEM_FORALL");
      get(); // drop ','

      drop();
      check(is_id(),"no 4th id (Z) in MFEM_FORALL");
      ker.forall.Z = get_id();
      dbg() << ", Z:'" << ker.forall.Z << "'";

      drop();
      check(is_coma(),"no last coma in MFEM_FORALL");
      get(); // drop ','

      // Starts counting the parentheses to wait for end of MFEM_FORALL'('
      parenthesis = 0;
   }

   bool is_forall_postfix()
   {
      if (parenthesis >= 0 && in.peek() == '(') { parenthesis++; }
      if (parenthesis >= 0 && in.peek() == ')') { parenthesis--; }
      if (parenthesis == -1) { return true; }
      //parenthesis--;
      return false;
   }

   void forall_postfix()
   {
      drop();
      check(is_right_parenthesis(),"no last right parenthesis found");
      get(); // drop ')'

      drop();
      check(is_semicolon(),"no last semicolon found");
      get(); // drop ';'
#ifdef MFEM_USE_CUDA
      out << "if (use_dev){";
      const char *ND = ker.forall.d == 2 ? "2D" : "3D";
      out << "\n\tCuWrap" << ND << "(" << ker.forall.N.c_str() << ", ";
      out << "[=] MFEM_DEVICE (int " << ker.forall.e <<")";
      out << ker.forall.body.c_str() << ",";
      out << ker.forall.X.c_str() << ",";
      out << ker.forall.Y.c_str() << ",";
      out << ker.forall.Z.c_str() << ");";
      out << "\n} else {";
      out << "for (int k=0; k<" << ker.forall.N.c_str() << ";k++) {";
      out << "[&] (int " << ker.forall.e <<")";
      out << ker.forall.body.c_str() << "(k);";
      out << "}";
      out << "}";
#else
      ker.source += "for (int ";
      ker.source +=  ker.forall.e ;
      ker.source += " = 0; ";
      ker.source += ker.forall.e;
      ker.source += " < ";
      ker.source += ker.forall.N.c_str();
      ker.source += "; ";
      ker.source += ker.forall.e;
      ker.source += "++) {";
      ker.source += ker.forall.body;
      ker.source += "}";
      /*
            ker.source += "(";
            ker.source +=  ker.forall.e;
            ker.source += ", ";
            ker.source += ker.forall.N;
            ker.source += ", ";
            ker.source += ker.forall.X;
            ker.source += ", ";
            ker.source += ker.forall.Y;
            ker.source += ", ";
            ker.source += ker.forall.Z;
            ker.source += ",";
            ker.source += ker.forall.body;
            ker.source += ");";*/
#endif
   }

   void tokens()
   {
      if (peek_n(4) == "MFEM")
      {
         const std::string &id = get_id();
         dbg(ker.is_jit() ? 154 : 0) << id;

         auto MFEM_token = [&](const std::string &id, const char *token)
         {
            if (strncmp(id.c_str(), "MFEM_", 5) != 0 ) { return false; }
            const int mn = strlen("MFEM_"), mt = strlen(token);
            if (strncmp(id.c_str() + mn, token, mt) != 0 ) { return false; }
            return true;
         };

         const bool jit = MFEM_token(id, "JIT");
         if (ker.is_wait() && jit) // MFEM_JIT
         {
            ker.advance(); // wait => jit
            mfem_jit_prefix();
            ker.advance(); // sign => source
         }

         const bool forall = MFEM_token(id, "FORALL");
         if (ker.is_source() && forall) // MFEM_FORALL
         {
            // Switch from prefix capturing, to the forall one
            ker.advance(); // source => forall
            forall_prefix(id); // start counting parenthesis
            ker.advance(); // forall => body
         }

         const bool unroll = MFEM_token(id, "UNROLL");
         if (ker.is_body() && unroll) { mfem_unroll(); }

         if (ker.is_wait()) { out << id; }
         if (ker.is_source() && !jit) { ker.source += id; }
         if (ker.is_forall()) { ker.forall.body += id; }
         if (ker.is_body() && !unroll && !forall) { ker.forall.body += id; }
      } // MFEM_*

      if (ker.is_body() && is_forall_postfix())
      {
         forall_postfix();
         ker.advance(); // body => postfix
      }

      if (ker.is_postfix() && is_jit_postfix())
      {
         mfem_jit_postfix();
         ker.advance(); // postfix => done
      }
   }

   bool eof()
   {
      const char c = get();
      if (in.eof()) { return true; }
      put(c);
      return false;
   }

   int operator()()
   {
      try { do { next(); tokens(); } while (!eof()); }
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
};

int JitPreProcess(std::istream &in, std::ostream &out, std::string &file)
{
   return JitPreProcessor(in,out,file).operator()();
}

} // namespace mfem

#endif // MFEM_USE_JIT
