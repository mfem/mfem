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

#if !(defined(MFEM_CXX) && defined(MFEM_EXT_LIBS) &&\
      defined(MFEM_LINK_FLAGS) && defined(MFEM_BUILD_FLAGS))
#define MFEM_CXX ""
#define MFEM_EXT_LIBS ""
#define MFEM_LINK_FLAGS ""
#define MFEM_BUILD_FLAGS ""
#endif

struct dbg
{
   static constexpr bool DEBUG = true;
   static constexpr uint8_t COLOR = 226;
   dbg(): dbg(COLOR) { }
   dbg(const uint8_t color)
   {
      if (!DEBUG) { return; }
      std::cerr << "\033[38;5;" << std::to_string(color==0?COLOR:color) << "m";
   }
   ~dbg() { if (DEBUG) { std::cerr << "\033[m\n"; std::cerr.flush(); } }
   template <typename T> dbg& operator<<(const T &arg)
   { if (DEBUG) { std::cerr << arg; std::cerr.flush(); } return *this; }
   template<typename T, typename... Args>
   inline void operator()(const T &arg, Args... args) const
   { operator<<(arg); operator()(args...); }
   template<typename T>
   inline void operator()(const T &arg) const { operator<<(arg); }
};

struct Parser
{
   struct kernel_t
   {
      std::string name; // Kernel name
      std::string Targs, Tparams, Tformat; // Template arguments, parameters, format
      std::string Sargs, Sparams, Sparams0; // Symbol arguments, parameters
      std::string Tadds, Sargs_;
      struct { int dim; std::string e, N, X, Y, Z; std::ostringstream body; } forall;
      std::ostringstream src, dup;
      bool is_static, is_templated, eq, mv_to_targs;
      int lt;

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
      bool is_jit()     { return fsm.state == &fsm_t::jit; }
      bool is_wait()    { return fsm.state == &fsm_t::wait; }
      bool is_targs()   { return fsm.state == &fsm_t::targs; }
      bool is_symbol()  { return fsm.state == &fsm_t::symbol; }
      bool is_params()  { return fsm.state == &fsm_t::params; }
      bool is_prefix()  { return fsm.state == &fsm_t::prefix; }
      bool is_forall()  { return fsm.state == &fsm_t::forall; }
      bool is_kernel()  { return fsm.state == &fsm_t::kernel; }
      bool is_postfix() { return fsm.state == &fsm_t::postfix; }

      bool put(const char c) // returns false if c should not be out.putted
      {
         if (is_targs()) { Tparams += c; }
         if (is_params())
         {
            Sparams0 += c;
            if (!eq) { Sparams += c; }
            if (mv_to_targs) { Tparams += c; }
         }
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

   bool good() { return in.eof() ? false : (in.peek(), in.good()); }

   char get() { return static_cast<char>(in.get()); }

   char put(const char c)
   {
      if (c == '\n') { line++; } // always count the line number
      if (ker.put(c)) { out.put(c); } // output in kernel, and to out if needed
      return c;
   }

   char put() { check(good(),"!good put error"); return put(get()); }

   template<class T> std::string transform(std::string &s, T &&op)
   { return (std::transform(s.begin(), s.end(), s.begin(), op), s); }

   std::string to_lower(std::string s)
   { return transform(s, [](unsigned char c) { return std::tolower(c); }); }

   std::string to_upper(std::string s)
   { return transform(s, [](unsigned char c) { return std::toupper(c); }); }

   std::string most(std::string &s) { return s = s.substr(0, s.size()-1); };
   std::string head(std::string &s) { return s.substr(0, s.find_last_of('.'));};
   std::string last(std::string &s) { return s.substr(s.find_last_of('.')+1);};

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
      while (!in.eof() && good() && is_comment())
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
         check(good());
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
   void next_check(L &&op, const char *msg = "") { next(); check(op(), msg); }

   bool is_string(const char *str) { return peek_n(std::strlen(str)) == str; }
   bool is_volatile() { return is_string("volatile"); }
   bool is_template() { return is_string("template"); }
   bool is_static() { return is_string("static"); }
   bool is_void() { return is_string("void"); }

   bool is_char(const unsigned char c) { return in.peek() == c; }
   bool is_linefeed() { return is_char('\n'); }
   bool is_slash() { return is_char('/'); }
   bool is_star() { return is_char('*'); }
   bool is_coma() { return is_char(','); }
   bool is_amp() { return is_char('&'); }
   bool is_eq() { return is_char('='); }
   bool is_lt() { return is_char('<'); }
   bool is_gt() { return is_char('>'); }

   void mfem_jit_prefix()
   {
      (ker.advance(), check(ker.is_jit(), "wait => jit"));

      next();
      ker.Targs.clear();
      ker.Tparams.clear();
      ker.is_templated = is_template();
      if (ker.is_templated)
      {
         ker.Tformat = "%d";
         out << get_id();
         next_check([&]() {return put() == '<';}, "no '<' in kernel!");
         // preparing kernel's: Targs, Tparams & Tformat
         (ker.advance(), check(ker.is_targs(), "jit => Targ"));
         while (good())
         {
            next();
            if (is_coma()) { ker.Tformat += ",%d"; }
            if (peek_n(2) == "T_")
            {
               std::string id = peek_id();
               add_arg(ker.Targs, (id.erase(0,2), to_lower(id)));
            }
            if (in.peek() == '>') { break;}
            put(); // '>' to ker.Tparams
         }
         ker.advance(); // Targ => Symbol
         check(put()=='>',"no '>' in kernel!");
         check(ker.Targs.size() > 0, "No JIT templated parameter found (T_*)!");
      }
      else
      {
         ker.Tformat = "";
         (ker.advance(), check(ker.is_targs(), "jit => targs"));
         (ker.advance(), check(ker.is_symbol(), "targs => symbol"));
      }

      next(); // 'static' ?
      ker.is_static = is_static() ? (out << get_id(), true) : false;

      next_check([&]() { return is_void(); }, "kernel should return 'void'");
      out << get_id();

      next(); // kernel's name
      out << (ker.name = get_id());

      next_check([&]() { return put()=='(';}, "no first '(' in kernel");

      // Get the arguments
      ker.advance(); // Symbol => Params
      std::string id {};
      ker.lt = 0;
      ker.eq = false;
      ker.mv_to_targs = false;
      ker.Sargs.clear();
      ker.Sargs_.clear();
      ker.Sparams.clear();
      ker.Sparams0.clear();
      auto add_id = [&]()
      {
         add_arg(ker.Sargs, last(id));
         add_arg(ker.Sargs_, last(id) + (ker.mv_to_targs?"_":""));
         if (ker.mv_to_targs)
         {
            if (!ker.Tformat.empty()) { ker.Tformat += ","; }
            ker.Tformat += "%d";
            add_arg(ker.Targs, last(id));
            ker.Sparams0 += "_";
            ker.Tparams += ",";
         }
      };
      while (good() && in.peek() != ')')
      {
         if (is_comment())
         {
            comments();
            if (!id.empty() && id.back() != '.' && !ker.eq) { id += '.'; }
         }
         check(good());
         if (in.peek() == ')') { break; } // to handle only comments
         check(is_space() || is_id() || is_star() || is_amp() ||
               is_eq() || is_lt() || is_gt() || is_coma(),
               "while parsing the arguments.");
         if (is_lt()) { ker.lt++; }
         if (is_gt()) { ker.lt--; }
         if (is_space() && !id.empty() && id.back() != '.' && !ker.eq)
         {
            if (last(id) == "MFEM_JIT") { ker.mv_to_targs = true; id = head(id); }
            id += '.';
         }
         if (is_eq())
         {
            ker.eq = true;
            if (id.back() == '.') { most(id); }
            if (ker.Targs.find(last(id)) == std::string::npos)
            { check(false,std::string("Could not find T_")+to_upper(last(id))); }
         }
         if (is_id() && !ker.eq) { id += in.peek(); }
         if (is_coma() && ker.lt == 0)
         {
            if (id.back() == '.') { most(id); }
            add_id(), id.clear(), ker.eq = false, ker.mv_to_targs = false;
         }
         put();
      }
      add_id();
      check(!ker.Tparams.empty(), "MFEM_JIT function without JIT argument!");
      if (!ker.is_templated) { ker.Tparams += " bool done = true"; }

      ker.advance(); // Params => Body

      // Make sure we hit the last ')' of the arguments
      check(put()==')',"no last ')' in kernel");

      // Make sure we are about to start a compound statement
      next_check([&]() { return put()=='{';}, "no compound statement found");

      // Generate the kernel prefix for this kernel
      ker.advance(); // Body => Prefix
      ker.dup.clear(), ker.dup.str(std::string());
      ker.forall.body.clear(), ker.forall.body.str(std::string());
      ker.src.clear(), ker.src.str(std::string());

      ker.src << "\n\tconst unsigned long backends = Device::Backends();";
      ker.src << "\n\tconst char *source = R\"_(";
      // defining 'MFEM_JIT_COMPILATION' to avoid:
      //   - MFEM_GPU_CHECK in cuda.hpp
      //   - HYPRE_config.h in mem_manager.hpp
      ker.src << "\n#define MFEM_JIT_COMPILATION"
              << "\n#include \"general/forall.hpp\""
              << "\n#include \"linalg/kernels.hpp\"";
      // MFEM_FORALL_2D_JIT
      ker.src << "\n#define MFEM_FORALL_2D_JIT(i,N,X,Y,B,...)"
              << "ForallWrap<2>(backends, true, N,"
              <<"[=] MFEM_DEVICE (int i) {__VA_ARGS__},"
              <<"[&] MFEM_LAMBDA (int i) {__VA_ARGS__},"
              <<"X,Y,B)";
      // MFEM_FORALL_3D_JIT
      ker.src << "\n#define MFEM_FORALL_3D_JIT(i,N,X,Y,Z,...)"
              << "ForallWrap<3>(backends, true, N,"
              <<"[=] MFEM_DEVICE (int i) {__VA_ARGS__},"
              <<"[&] MFEM_LAMBDA (int i) {__VA_ARGS__},"
              <<"X,Y,Z)";
      ker.src << "\n#include \"general/jit/jit.hpp\""; // for Hash, Find
      ker.src << "\nusing namespace mfem;";
      ker.src << "\ntemplate<" << ker.Tparams << ">";
      ker.src << "\nvoid " << ker.name << "_%016lx"
              << "(const unsigned long backends, "
              << ker.Sparams0 << "){";

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
      forall << ker.forall.e << "," << ker.forall.N << ", "
             << ker.forall.X << "," << ker.forall.Y << "," << ker.forall.Z << ", "
             << ker.forall.body.str() << ");";
      ker.src << forall.str(), ker.dup << forall.str();
      ker.advance(/*kernel => postfix*/);
   }

   void mfem_jit_postfix() // output all kernel source, with updated hash
   {
      ker.src << "}\nextern \"C\" void k%016lx"
              << "(const unsigned long backends," << ker.Sparams0 << "){"
              << "\n\t" << ker.name << "_%016lx<" << ker.Tformat << ">"
              << "(backends,"<< ker.Sargs_ << ");";

      // defined on the command line
      const char *cxx = MFEM_CXX;
      const char *libs = MFEM_EXT_LIBS;
      const char *link = MFEM_LINK_FLAGS;
      const char *flags = MFEM_BUILD_FLAGS;

      size_t seed = // src is ready: compute its seed with all the MFEM context
         mfem::Jit::Hash(
            std::hash<std::string> {}(ker.src.str()),
            std::string(cxx), std::string(libs), std::string(flags),
            std::string(MFEM_SOURCE_DIR), std::string(MFEM_INSTALL_DIR));

      if (ker.is_templated) // dup code
      {
         out << ker.dup.str() << "}";
         out << "\n"<< (ker.is_static ? "static " : "") // then generate the code
             << "void " << ker.name << "(" << ker.Sparams0 << ")"
             << "{";
      }

      out << ker.src.str() << "})_\";"; // end of src

      out << "\nconst char *cxx = \""  << cxx << "\";";
      out << "\nconst char *flags = \""<< flags << "\";";
      out << "\nconst char *link = \"" << link << "\";";
      out << "\nconst char *libs = \"" << libs << "\";";
      out << "\nconst size_t hash = Jit::Hash("
          << "0x" << std::hex << seed << std::dec << "ul"
          << "," << ker.Targs << ");";
      out << "\ntypedef void (*kernel_t)"
          <<"(unsigned long backends, " << ker.Sparams << ");";
      out << "\nstatic std::unordered_map<size_t, Jit::Kernel<kernel_t>> kernels;"
          << "\nJit::Find(hash, \"" << ker.name  << "<" << ker.Tformat << ">"
          << "\", cxx, flags, link, libs, source, kernels" << ", " << ker.Targs
          <<  ").operator()(backends," << ker.Sargs << ");";
      out << "\n#line " << std::to_string(line) << " \"" << file << "\"\n";
      ker.advance(/*postfix => wait*/);
   }

   void token()
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
            const size_t mn = std::strlen("MFEM_"), mt = std::strlen(token);
            if (std::strlen(id.c_str()) != (mn + mt) ||
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
      try { while (good()) { put(); next(); token(); } }
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
   assert((!ifs.fail()) && "Could not open input file!");
   assert(ifs.is_open() && "Could not open input file!");
   if (ofile) { assert(ofs.is_open() && "Could not open output file!"); }
   const int status = Parser(ifs, ofile ? ofs : std::cout, file).operator()();
   ifs.close();
   ofs.close();
   return status;
}

#endif // MFEM_USE_JIT
