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

#include <algorithm> // std::transform
#include <list>
#include <string>
#include <cassert>
#include <cstring> // std::strlen
#include <fstream>
#include <iostream>
#include <sstream>

#include "jit.hpp" // Hash, ToString, Find
#include "../device.hpp" // Backend::Id

struct Parser
{
   struct kernel_t
   {
      std::string name; // Kernel name
      std::string Targs, Tparams, Tformat; // Template arguments, parameters, format
      std::string Sargs, Sparams, Sparams0; // Symbol arguments, parameters
      struct { int dim; std::string e, N, X, Y, Z; std::ostringstream body; } forall;
      std::ostringstream src, dup;
      bool is_static, eq;

      struct fsm_t
      {
         typedef void (fsm_t::*State) ();
         State state;
         fsm_t(): state(&fsm_t::wait) { }
         void next(State next) { state = next; }
         void advance() { (this->*state)(); }
         // FSM states:
         void wait()    { next(&fsm_t::jit); }     // Waiting...
         void jit()     { next(&fsm_t::targs); }   // MFEM_JIT hit!
         void targs()   { next(&fsm_t::symbol); }  // Templated arguments
         void symbol()  { next(&fsm_t::params); }  // Symbol name
         void params()  { next(&fsm_t::body); }    // Parameters
         void body()    { next(&fsm_t::prefix); }  // Waiting for prefix
         void prefix()  { next(&fsm_t::forall); }  // Prefix code before forall
         void forall()  { next(&fsm_t::kernel); }  // Forall parameters
         void kernel()  { next(&fsm_t::postfix); } // Forall body
         void postfix() { next(&fsm_t::wait); }    // Postfix
      } fsm;

      void advance() { fsm.advance(); }
      bool is_wait()    { return fsm.state == &fsm_t::wait; }
      bool is_targs()   { return fsm.state == &fsm_t::targs; }
      bool is_params()  { return fsm.state == &fsm_t::params; }
      bool is_prefix()  { return fsm.state == &fsm_t::prefix; }
      bool is_forall()  { return fsm.state == &fsm_t::forall; }
      bool is_kernel()  { return fsm.state == &fsm_t::kernel; }
      bool is_postfix() { return fsm.state == &fsm_t::postfix; }

      bool put(const char c) // returns false if c should not be out.putted
      {
         if (is_targs()) { Tparams += c; }
         if (is_params()) { Sparams0 += c; if (!eq) { Sparams += c; } }
         if (is_prefix() || is_postfix()) { src << c; dup << c; return false;}
         if (is_forall() || is_kernel()) { forall.body << c; return false;}
         return true;
      }
   } ker;

   std::istream &in;
   std::ostream &out;
   std::string &file;
   int line, block, parenthesis;

   Parser(std::istream &in, std::ostream &out, std::string &file) :
      in(in), out(out), file(file), line(1), block(-1), parenthesis(-1) { }

   struct error_t
   {
      int line;
      std::string &file, message;
      error_t(int l, std::string &f, std::string &e): line(l), file(f), message(e) {}
   };
   void error(std::string msg) { throw error_t(line, file, msg);}
   void check(const bool tst, std::string msg = "") { if (!tst) { error(msg); }}

   template<typename T>
   void add_arg(std::string &s, const T a) { if (!s.empty()) { s += ","; } s += a; }

   bool good() { in.peek(); return in.good() && !in.eof(); }

   char get() { return static_cast<char>(in.get()); }

   char put(const char c)
   {
      if (c == '\n') { line++; } // always count the line number
      if (ker.put(c)) { out.put(c); } // output in kernel, and in out if needed
      return c;
   }

   char put() { check(good()); return put(get()); }

   bool is_space() { return good() && std::isspace(in.peek()); }

   void skip_space() { while (is_space()) { put(); } }

   bool is_comment()
   {
      if (!good()) { return false; }
      if (in.peek() != '/') { return false; }
      in.get();
      check(!in.eof(), "end of file found while in comment!");
      const int c = in.peek();
      in.unget();
      if (c == '/' || c == '*') { return true; }
      return false;
   }

   void comments()
   {
      check(good());
      while (is_comment())
      {
         check(put()=='/', "unknown comment");
         check(is_slash() || is_star(), "error in end-of-comment");
         if (put()=='/') { while (!is_linefeed()) { put(); } }
         else
         {
            while (peek_n(2) != "*/") { put(); }
            check(put() == '*', "unknown comment");
            check(put() == '/', "unknown comment");
         }
         skip_space();
      }
   }

   void next() { skip_space(); comments(); }

   bool is_id() { const char c = in.peek(); return std::isalnum(c) || c == '_'; }

   std::string get_id()
   {
      std::string id;
      check(is_id(), "id 1st character is not isalnum");
      while (is_id()) { id += get(); }
      return id;
   }

   template<typename L, int M = 32> std::string peek(L &&op, const int n = M-1)
   {
      int k = 0;
      check(n < M, "peek size error!");
      static char c[M];
      for (k = 0; k < n && good() && !in.eof() && op(); k++) { c[k] = get(); }
      std::string str((c[k]=0, c));
      for (int l = 0; l < k; l++) { in.unget(); }
      if (!good()) { return str; }
      return str;
   }
   std::string peek_id() { return peek([&]() { return is_id(); });}
   std::string peek_n(int n) { return peek([&]() { return true; }, n); }

   template<typename L>
   void next_check(L&&op, const char *msg = "") { next(); check(op(), msg); }

   bool is_string(const char *str) { return peek_n(std::strlen(str)) == str; }
   bool is_template() { return is_string("template"); }
   bool is_static() { return is_string("static"); }
   bool is_void() { return is_string("void"); }

   bool is_char(const unsigned char c) { return in.peek() == c; }
   bool is_linefeed() { return is_char('\n'); }
   bool is_slash() { return is_char('/'); }
   bool is_star() { return is_char('*'); }
   bool is_coma() { return is_char(','); }
   bool is_eq() { return is_char('='); }

   void mfem_jit_prefix()
   {
      ker.advance(); // wait => jit
      next_check([&]() {return is_template();}, "'template' token not found!");
      out << get_id();

      next_check([&]() {return put() == '<';}, "no '<' in kernel!");

      // preparing kernel's: Targs, Tparams & Tformat
      ker.advance(); // jit => Targ
      ker.Targs.clear();
      ker.Tparams.clear();
      ker.Tformat = "%d";
      auto to_lower = [](std::string &id)
      {
         std::transform(id.begin(), id.end(), id.begin(),
         [](unsigned char c) { return std::tolower(c); });
         return id;
      };
      while (good())
      {
         next();
         if (is_coma()) { ker.Tformat += ",%d";}
         if (peek_n(2) == "T_")
         {
            std::string id = peek_id();
            add_arg(ker.Targs, (id.erase(0,2), to_lower(id)));
         }
         if (in.peek() == '>') { break;}
         put(); // to ker.Tparams
      }
      ker.advance(); // Targ => Symbol
      check(put()=='>',"no '>' in kernel!");

      next(); // 'static' ?
      ker.is_static = is_static() ? (out << get_id(), true) : false;

      next_check([&]() { return is_void(); }, "kernel should return 'void'");
      out << get_id();

      next(); // kernel's name
      out << (ker.name = get_id());

      next_check([&]() { return put()=='(';}, "no first '(' in kernel");

      // Get the arguments
      ker.advance(); // Symbol => Params
      std::string id{};
      ker.eq = false;
      ker.Sargs.clear();
      ker.Sparams.clear();
      ker.Sparams0.clear();
      auto most = [](std::string &s) { return s = s.substr(0, s.size()-1); };
      auto last = [](std::string &s) { return s.substr(s.find_last_of('.')+1);};
      while (good() && in.peek() != ')')
      {
         if (is_comment())
         {
            comments();
            if (!id.empty() && id.back() != '.' && !ker.eq) { id += '.'; }
         }
         check(good());
         if (in.peek() == ')') { break; } // to handle only comments
         check(is_space() || is_id() || is_star() || is_eq() || is_coma());
         if (is_space() && !id.empty() && id.back() != '.' && !ker.eq) { id += '.'; }
         if (is_eq()) { ker.eq = true; id = id.substr(0, id.size()-1); }
         if (is_id() && !ker.eq) { id += in.peek(); }
         if (is_coma())
         {
            if (id.back() == '.') { most(id); }
            add_arg(ker.Sargs, last(id)); id.clear(); ker.eq = false;
         }
         put();
      }
      add_arg(ker.Sargs, last(id));
      ker.advance(); // Params => Body

      // Make sure we hit the last ')' of the arguments
      check(put()==')',"no last ')' in kernel");

      // Make sure we are about to start a compound statement
      next_check([&]() { return put()=='{';}, "no compound statement found");

      // Generate the kernel prefix for this kernel
      ker.advance(); // Body => Prefix
      ker.dup.clear();
      ker.dup.str(std::string());
      ker.forall.body.clear();
      ker.forall.body.str(std::string());
      ker.src.clear();
      ker.src.str(std::string());

      ker.src << "\n\tconst unsigned long backends = Device::Backends();";

      ker.src << "\n\tconst char *source = R\"_(";
      // defining 'MFEM_JIT_COMPILATION' to:
      //   - avoid MFEM_GPU_CHECK in cuda.hpp
      //   - pull <HYPRE_config.h> in mem_manager.hpp
      ker.src << "#define MFEM_JIT_COMPILATION"
              << "\n#include \"general/forall.hpp\"";

      // MFEM_FORALL_2D_JIT
      ker.src << "\n#define MFEM_FORALL_2D_JIT(i,N,X,Y,B,...)"
              << "ForallWrap<2>(true, backends, N,"
              <<"[=] MFEM_DEVICE (int i) {__VA_ARGS__},"
              <<"[&] MFEM_LAMBDA (int i) {__VA_ARGS__},"
              <<"X,Y,B)";

      // MFEM_FORALL_3D_JIT
      ker.src << "\n#define MFEM_FORALL_3D_JIT(i,N,X,Y,Z,...)"
              << "ForallWrap<3>(true, backends, N,"
              <<"[=] MFEM_DEVICE (int i) {__VA_ARGS__},"
              <<"[&] MFEM_LAMBDA (int i) {__VA_ARGS__},"
              <<"X,Y,Z)";

      ker.src << "\n#include \"general/jit/jit.hpp\""; // for Hash, Find

      ker.src << "\nusing namespace mfem;";

      ker.src << "\ntemplate<" << ker.Tparams << ">\nvoid " << ker.name << "_%016lx"
              << "(const unsigned long backends, " << ker.Sparams0 << "){";

      ker.src << "\n#line " << std::to_string(line) << " \"" << file << "\"\n";
      block = 0; // Start counting the block statements
   }

   void mfem_forall_prefix(const std::string &id)
   {
      // Switch from prefix capturing, to the forall one
      ker.advance(); // prefix => forall
      check(id.size() > 12, "Unknown MFEM_FORALL_?");
      ker.forall.dim = id.c_str()[12] - 0x30;
      check(ker.forall.dim == 2 || ker.forall.dim == 3, "FORALL dim error!");
      ker.forall.body.str(std::string());
      next_check([&]() {return get()=='(';}, "no 1st '(' in MFEM_FORALL");
      next(); ker.forall.e = get_id();
      next_check([&]() {return get()==',';}, "no 1st coma in MFEM_FORALL");
      next_check([&]() {return is_id();}, "no 1st id in MFEM_FORALL");
      ker.forall.N = get_id();
      next_check([&]() {return get()==',';}, "no 2nd coma in MFEM_FORALL");
      next_check([&]() {return is_id();}, "no 2nd id in MFEM_FORALL");
      ker.forall.X = get_id();
      next_check([&]() {return get()==',';}, "no 3rd coma in MFEM_FORALL");
      next_check([&]() {return is_id();}, "no 3rd id in MFEM_FORALL");
      ker.forall.Y = get_id();
      next_check([&]() {return get()==',';}, "no 4th coma in MFEM_FORALL");
      next_check([&]() {return is_id();}, "no 4th id in MFEM_FORALL");
      ker.forall.Z = get_id();
      next_check([&]() {return get()==',';}, "no last coma in MFEM_FORALL");
      parenthesis = 0; // Start counting MFEM_FORALL's parentheses
      ker.advance(); // forall => kernel
   }

   void mfem_forall_postfix()
   {
      check(get()==')',"no last right parenthesis found");
      next_check([&]() {return get()==';';}, "no last semicolon found");
      const char *ND = ker.forall.dim == 2 ? "2D" : "3D";
      ker.dup << "MFEM_FORALL_"<<ND<<"(";
      ker.src << "MFEM_FORALL_"<<ND<<"_JIT(";
      std::ostringstream forall;
      forall << ker.forall.e << ","
             << ker.forall.N << ", "
             << ker.forall.X << ","
             << ker.forall.Y << ","
             << ker.forall.Z << ", "
             << ker.forall.body.str() << ");";
      ker.src << forall.str();
      ker.dup << forall.str();
      ker.advance(/*kernel => postfix*/);
   }

   void mfem_jit_postfix() // output all kernel source, with updated hash
   {
      ker.src << "}\nextern \"C\" void k%016lx"
              << "(const unsigned long backends," << ker.Sparams0 << "){"
              << "\n\t" << ker.name << "_%016lx<"<< ker.Tformat << ">"
              << "(backends,"<< ker.Sargs << ");";
      size_t seed = // src is ready: compute its seed with all the MFEM context
         mfem::Jit::Hash(
            std::hash<std::string> {}(ker.src.str()),
            std::string(MFEM_JIT_CXX), std::string(MFEM_JIT_BUILD_FLAGS),
            std::string(MFEM_SOURCE_DIR), std::string(MFEM_INSTALL_DIR));

      out << ker.dup.str() << "}" // dump the first dup code
          << "\n"<< (ker.is_static ? "static " : "") // then generate the code
          << "void " << ker.name << "(" << ker.Sparams0 << ")"
          << "{" << ker.src.str() << "})_\";"; // end of src
      out << "\nconst size_t hash = Jit::Hash("
          << "0x" << std::hex << seed << std::dec << "ul," << ker.Targs << ");"
          << "\ntypedef void (*kernel_t)(unsigned long backends, " << ker.Sparams << ");"
          << "\nstatic std::unordered_map<size_t, Jit::Kernel<kernel_t>> kernels;"
          << "\nJit::Find(hash, source, kernels, " << ker.Targs << ")"
          << ".operator()(backends," << ker.Sargs << ");";
      out << "\n#line " << std::to_string(line) << " \"" << file << "\"\n";
      ker.advance(/*postfix => wait*/);
   }

   void tokens()
   {
      auto is_end_of = [&](int &c, const char i, const char o)
      {
         if (c >= 0 && in.peek() == i) { c++; }
         if (c >= 0 && in.peek() == o) { c--; }
         if (c < 0) { return true; }
         return false;
      };

      if (peek_n(4) == "MFEM")
      {
         const std::string &id = get_id();

         auto MFEM_token = [&](const std::string &id, const char *token)
         {
            const size_t mn = strlen("MFEM_"), mt = strlen(token);
            if (strlen(id.c_str()) != (mn + mt) ||
                strncmp(id.c_str(), "MFEM_", mn) != 0 ||
                strncmp(id.c_str() + mn, token, mt) != 0) { return false; }
            return true;
         };

         const bool JIT = MFEM_token(id, "JIT");
         if (ker.is_wait() && JIT) { mfem_jit_prefix(); }

         const bool forall =
            MFEM_token(id, "FORALL_2D") || MFEM_token(id, "FORALL_3D");
         if (ker.is_postfix() && forall)
         { error("Only one MFEM_FORALL is supported!"); }

         if (ker.is_prefix() && forall) { mfem_forall_prefix(id); }

         if (ker.is_wait()) { out << id; }
         if (ker.is_prefix() && !JIT) { ker.src << id; ker.dup << id; }
         if (ker.is_forall()) { ker.forall.body << id; }
         if (ker.is_kernel() && !forall) { ker.forall.body << id; }
      } // MFEM_*

      if (ker.is_kernel() && is_end_of(parenthesis,'(',')')) { mfem_forall_postfix(); }

      const bool is_end_blocks = is_end_of(block,'{','}');
      if (ker.is_prefix() && is_end_blocks) { ker.fsm.forall(); ker.fsm.kernel(); }
      if (ker.is_postfix() && is_end_blocks) { mfem_jit_postfix(); }
   }

   int operator()()
   {
      try { while (!in.eof()) { next(); if (good()) { tokens(); put(); } } }
      catch (error_t err)
      {
         std::cerr << std::endl << err.file << ":" << err.line << ":"
                   << " mjit error" << (!err.message.empty() ? ": " : "")
                   << (!err.message.empty() ? err.message : "") << std::endl;
         return EXIT_FAILURE;
      }
      return EXIT_SUCCESS;
   }
};

int main(const int argc, char* argv[])
{
   struct
   {
      int operator()(char* argv[])
      {
         std::cout << "mjit: ";
         std::cout << argv[0] << " [-h] [-o output] input" << std::endl;
         return EXIT_SUCCESS;
      }
   } Help;
   std::string input, output, file;

   if (argc <= 1) { return Help(argv); }
   for (int i = 1; i < argc; i++)
   {
      if (argv[i] == std::string("-h")) { return Help(argv); } // help
      if (argv[i] == std::string("-o")) { output = argv[++i]; continue; }
      assert(argv[i] && "Could not use last argument as input file!");
      file = input = argv[i];
   }
   const bool ofile = !output.empty();
   std::ifstream ifs(input.c_str(), std::ios::in | std::ios::binary);
   std::ofstream ofs(output.c_str(),
                     std::ios::out | std::ios::binary | std::ios::trunc);
   assert((!ifs.fail()) && "Could not open input!");
   assert(ifs.is_open() && "Could not open input!");
   if (ofile) { assert(ofs.is_open() && "Could not open output file!"); }
   const int status = Parser(ifs, ofile ? ofs : std::cout, file).operator()();
   ifs.close();
   ofs.close();
   return status;
}

#endif // MFEM_USE_JIT
