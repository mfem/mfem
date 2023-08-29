// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include "jit.hpp" // Hash, ToString, Find

#include <string>
#include <vector>
#include <cassert>
#include <cstring> // std::strlen
#include <fstream>
#include <algorithm> // std::transform

using namespace std;

#if !(defined(MFEM_CXX) && defined(MFEM_EXT_LIBS) &&\
      defined(MFEM_LINK_FLAGS) && defined(MFEM_BUILD_FLAGS))
#error MFEM_[CXX, EXT_LIBS, LINK_FLAGS, BUILD_FLAGS] must be defined!
#define MFEM_CXX
#define MFEM_EXT_LIBS
#define MFEM_LINK_FLAGS
#define MFEM_BUILD_FLAGS
#endif

#ifndef MFEM_SHARED_BUILD
#error Building with JIT must enable shared library build of MFEM
#endif

namespace mfem
{

namespace internal
{

namespace jit
{

struct Parser
{
   /*
   * The kernel_t struct holds the following information:
   *    - name,
   *    - templated format strings, arguments strings,
   *    - original source and its duplicate for the JIT'ed one,
   *    - various flags.
   */
   struct kernel_t
   {
      string name;
      string Targs, Tparams, Tformat; // Templated info: <%d,%d>
      string Sargs, Sparams, Sparams0; // Symbol call arguments
      string Tadds, Sargs_us;
      ostringstream src, dup;
      vector<string> includes;
      bool is_static, is_templated, is_eq, mv_to_targs;

      /*
       * The fsm_t struct is a minimal finite state machine mostly to
       * handle the 'filtering' towards the Templated and/or Symbol strings.
       * Each character emitted through 'put' is filtered when parsing a kernel.
       */
      struct fsm_t
      {
         using State = void (fsm_t::*)();
         State state;
         fsm_t(): state(&fsm_t::wait) { }
         void next(State next) { state = next; }
         void advance() { (this->*state)(); }
         void wait() { next(&fsm_t::jit); } // Waiting...
         void jit() { next(&fsm_t::targs); } // MFEM_JIT hit!
         void targs() { next(&fsm_t::symbol); } // Templated arguments
         void symbol() { next(&fsm_t::params); } // Symbol name
         void params() { next(&fsm_t::prefix); } // Parameters
         void prefix() { next(&fsm_t::body); } // Kernel body
         void body() { next(&fsm_t::wait); } // Kernel body
         void include() { next(&fsm_t::wait); } // Add include
         template<State S> bool is() { return state == S; }
      } fsm;

      void advance() { fsm.advance(); }
      void include() { fsm.next(&fsm_t::include); }

      bool is_wait()    { return fsm.is<&fsm_t::wait>(); }
      bool is_jit()     { return fsm.is<&fsm_t::jit>(); }
      bool is_targs()   { return fsm.is<&fsm_t::targs>(); }
      bool is_symbol()  { return fsm.is<&fsm_t::symbol>(); }
      bool is_params()  { return fsm.is<&fsm_t::params>(); }
      bool is_prefix()  { return fsm.is<&fsm_t::prefix>(); }
      bool is_body()    { return fsm.is<&fsm_t::body>(); }
      bool is_include() { return fsm.is<&fsm_t::include>(); }

      bool filter(const char c) // returns false if c should not be out.put'ed
      {
         if (is_targs()) { Tparams += c; }
         if (is_params())
         {
            Sparams0 += c;
            if (!is_eq) { Sparams += c; }
            if (mv_to_targs) { Tparams += c; }
         }
         if (is_body()) { src << c; dup << c; return false; }
         return true;
      }
   } ker;

   istream &in;
   ostream &out;
   string &filename;
   int line, block, parenthesis;

   Parser(istream &in, ostream &out, string &file) :
      in(in), out(out), filename(file), line(1), block(-1), parenthesis(-1) { }

   struct error_t
   {
      int line;
      string &file, message;
      error_t(int l, string &f, string &e): line(l), file(f), message(e) {}
   };
   void error(string msg) { throw error_t(line, filename, msg);}
   void check(const bool tst, string msg = "") { if (!tst) { error(msg); }}

   // pp_line returns a pre-processor line used to locate file and line number.
   string pp_line()
   {
      ostringstream oss {};
      oss << "\n#line " << to_string(line) << " \"" << filename << "\"\n";
      return oss.str();
   }

   template<typename T>
   void add_arg(string &s, const T a) { if (!s.empty()) { s += ","; } s += a; }

   bool good() { return in.eof() ? false : (in.peek(), in.good()); }

   char get() { return static_cast<char>(in.get()); }

   char put(const char c)
   {
      if (c == '\n') { line++; } // always count the line number
      if (ker.filter(c)) { out.put(c); } // kernel filter
      return c;
   }

   char put() { check(good(),"!good put error"); return put(get()); }

   // string utilities: transform, to_lower/upper, most, head & last
   template<class T> string apply(string &s, T &&op)
   { return (std::transform(s.begin(), s.end(), s.begin(), op), s); }
   string to_lower(string s)
   { return apply(s, [](unsigned char c) { return tolower(c); }); }
   string to_upper(string s)
   { return apply(s, [](unsigned char c) { return toupper(c); }); }
   string most(string &s) { return s = s.substr(0, s.size()-1); }
   string head(string &s) { return s.substr(0, s.find_last_of('.'));}
   string last(string &s) { return s.substr(s.find_last_of('.')+1);}

   bool is_space() { return good() && isspace(in.peek()); }

   void skip_space() { while (is_space()) { put(); } }

   void skip_string()
   {
      if (good() && is_quote()) { for (check(put()=='"'); !is_quote(); put()); }
   }

   string get_string()
   {
      string str;
      if (good() && is_quote())
      {
         for (check(put()=='"'); !is_quote(); str += put());
      }
      return str;
   }

   bool is_comment()
   {
      if (!good()) { return false; }
      if (in.peek() != '/') { return false; }
      in.get();
      check(good(), "end of file found while in comment!");
      const int c = in.peek();
      in.unget();
      if (c == '/' || c == '*') { return true; }
      return false;
   }

   void skip_comments()
   {
      while (good() && is_comment())
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

   void next() { skip_space(); skip_string(); skip_comments(); }

   bool is_id() { const char c = in.peek(); return isalnum(c) || c == '_'; }

   string get_id()
   {
      string id;
      check(is_id(), "id 1st character is not isalnum");
      while (is_id()) { id += get(); }
      return id;
   }

   template<typename L, int M = 32> string peek(L &&op, const int n = M-1)
   {
      int k = 0;
      check(n < M, "peek size error!");
      static char c[M];
      for (k = 0; k < n && good() && op(); k++) { c[k] = get(); }
      string str((c[k]=0, c));
      for (int l = 0; l < k; l++) { in.unget(); }
      if (!good()) { return str; }
      return str;
   }
   string peek_id() { return peek([&]() { return is_id(); });}
   string peek_n(int n) { return peek([&]() { return true; }, n); }

   template<typename L>
   void next_check(L &&op, const char *msg = "") { next(); check(op(), msg); }

   bool is_string(const char *str) { return peek_n(strlen(str)) == str; }
   bool is_template() { return is_string("template"); }
   bool is_include() { return is_string("include"); }
   bool is_static() { return is_string("static"); }
   bool is_void() { return is_string("void"); }

   bool is_char(const unsigned char c) { return in.peek() == c; }
   bool is_linefeed() { return is_char('\n'); }
   bool is_slash() { return is_char('/'); }
   bool is_star() { return is_char('*'); }
   bool is_coma() { return is_char(','); }
   bool is_hash() { return is_char('#'); }
   bool is_quote() { return is_char('"'); }
   bool is_amp() { return is_char('&'); }
   bool is_eq() { return is_char('='); }
   bool is_lt() { return is_char('<'); }
   bool is_gt() { return is_char('>'); }

   void mfem_jit_include()
   {
      check(ker.is_wait(), "mfem_jit_include not in jit state");
      put(); // '#'
      next();
      if (is_include())
      {
         out << get_id(); // include
         skip_space();
         check(is_quote());
         string inc = get_string();
         ker.includes.push_back(inc);
         check(is_quote());
         put(); // '"'
         next();
      }
      ker.include();
      check(ker.is_include());
   }

   /*
    * mfem_jit_kernel is the main parser part: it prepares the templated
    * format and argument strings, verifies the signature, parses and prepares
    * the arguments and sets the pre-processor line to the current source file
    * and line location.
    */
   void mfem_jit_kernel()
   {
      assert(ker.is_wait());
      next();

      if (is_hash()) { return mfem_jit_include(); } // #include ?

      (ker.advance(), check(ker.is_jit(), "wait => jit")); // FSM update
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
               string id = peek_id();
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
      check(ker.is_symbol(), "should be symbol!");

      next(); // 'static' ?
      ker.is_static = is_static() ? (out << get_id(), true) : false;

      next_check([&]() { return is_void(); }, "kernel should return 'void'");
      out << get_id();

      next(); // kernel's name
      out << (ker.name = get_id());

      next_check([&]() { return put()=='('; }, "no first '(' in kernel");

      // Get the arguments
      ker.advance(); // Symbol => Params
      string id {};
      int lt = 0;
      ker.is_eq = false;
      ker.mv_to_targs = false;
      ker.Sargs.clear();
      ker.Sargs_us.clear();
      ker.Sparams.clear();
      ker.Sparams0.clear();
      auto add_id = [&]()
      {
         add_arg(ker.Sargs, last(id));
         add_arg(ker.Sargs_us, last(id) + (ker.mv_to_targs?"_":""));
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
            skip_comments();
            if (!id.empty() && id.back() != '.' && !ker.is_eq) { id += '.'; }
         }
         check(good());
         if (in.peek() == ')') { break; } // to handle only comments
         check(is_space() || is_id() || is_star() || is_amp() ||
               is_eq() || is_lt() || is_gt() || is_coma(),
               "while parsing the arguments.");
         if (is_lt()) { lt++; }
         if (is_gt()) { lt--; }
         if (is_space() && !id.empty() && id.back() != '.' && !ker.is_eq)
         {
            if (last(id) == "MFEM_JIT") { ker.mv_to_targs = true; id = head(id); }
            id += '.';
         }
         if (is_eq())
         {
            ker.is_eq = true;
            if (id.back() == '.') { most(id); }
            if (ker.Targs.find(last(id)) == string::npos)
            { check(false, string("Could not find T_")+to_upper(last(id))); }
         }
         if (is_id() && !ker.is_eq) { id += in.peek(); }
         if (is_coma() && lt == 0)
         {
            if (id.back() == '.') { most(id); }
            add_id(), id.clear(), ker.is_eq = false, ker.mv_to_targs = false;
         }
         put();
      }
      add_id();
      check(!ker.Tparams.empty(), "MFEM_JIT function without JIT argument!");
      if (!ker.is_templated) { ker.Tparams += " bool done = true"; }

      ker.advance(); // Params => Prefix

      // Make sure we hit the last ')' of the arguments
      check(put()==')',"no last ')' in kernel");

      // Make sure we are about to start a compound statement
      next_check([&]() { return put()=='{';}, "no compound statement found");

      // Generate the kernel prefix for this kernel
      ker.dup.clear(), ker.dup.str(string());
      ker.src.clear(), ker.src.str(string());
      ker.advance(); // Prefix => Body

      ker.src << "\n\tconst char *source = R\"_(";
      // mfem, forall & jit includes are added from command line from jit.cpp
      ker.src << "\nusing namespace mfem;";
      for (auto inc: ker.includes) { ker.src << "\n#include \"" + inc +"\""; }
      ker.src << "\n\ntemplate<" << ker.Tparams << ">";
      ker.src << "\nvoid " << ker.name << "_%016lx"
              << "(" << ker.Sparams0 << "){";
      ker.src << pp_line();
      block = 0; // Start counting the block statements
   }

   /*
    * mfem_jit_postfix prepare:
    *   - the JIT inputs: compiler, flags, libraries from the build system,
    *   - computes the hash of the source, compiler, libraries, flags,
    *     MFEM_SOURCE_DIR and MFEM_INSTALL_DIR,
    *   - the per-kernel local unordered_map which holds the different kernel
    *     symbols that have already been loaded,
    *   - updates the pre-processor line.
    */
   void mfem_jit_postfix() // output all kernel source, with updated hash
   {
      ker.src << "}\nextern \"C\" void k%016lx"
              << "(" << ker.Sparams0 << "){"
              << "\n\t" << ker.name << "_%016lx<" << ker.Tformat << ">"
              << "("<< ker.Sargs_us << ");";

      // these settings are global to MFEM
      const char *cxx = "" MFEM_CXX;
      const char *libs = "-L" MFEM_INSTALL_DIR "/lib -lmfem " MFEM_EXT_LIBS;
      const char *link = "" MFEM_LINK_FLAGS;
      const char *flags = "" MFEM_BUILD_FLAGS;

      size_t seed = // src is ready: compute its seed with all the MFEM context
         mfem::JIT::Hash(hash<string> {}(ker.src.str()),
                         string(cxx), string(libs), string(flags),
                         string(MFEM_SOURCE_DIR), string(MFEM_INSTALL_DIR));

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
      // if needed, (c)make sets this directory for each file
      out << "\n#ifndef MFEM_JIT_INC_PATH\n#define MFEM_JIT_INC_PATH\n#endif";
      out << "\nconst char *dir = \"\" MFEM_JIT_INC_PATH;";
      out << "\nconst size_t hash = JIT::Hash("
          << "0x" << std::hex << seed << std::dec << "ul"
          << "," << ker.Targs << ");";
      out << "\ntypedef void (*kernel_t)"
          <<"(" << ker.Sparams << ");";
      out << "\nstatic std::unordered_map<size_t, JIT::Kernel<kernel_t>> kernels;"
          << "\nJIT::Find(hash, \"" << ker.name << "<" << ker.Tformat << ">"
          << "\", cxx, flags, link, libs, dir, source, kernels" << ", "
          << ker.Targs << ").Launch(" << ker.Sargs << ");";
      out << pp_line();
      ker.advance(/*postfix => wait*/);
   }

   /*
    * token triggers the parser for each MFEM_* encountered.
    * Depending on the FSM state, it also counts the blocks and parenthesis.
    */
   void token()
   {
      auto is_end_of = [&](int &chr, const char beg_chr, const char end_chr)
      {
         if (chr >= 0 && in.peek() == beg_chr) { chr++; }
         if (chr >= 0 && in.peek() == end_chr) { chr--; }
         if (chr < 0) { return true; }
         return false;
      };

      if (peek_n(4) == "MFEM")
      {
         const string id = get_id();
         auto is = [&](const string &id, const char *token)
         {
            const size_t mn = strlen("MFEM_"), mt = strlen(token);
            if (strlen(id.c_str()) != (mn + mt) ||
                strncmp(id.c_str(), "MFEM_", mn) != 0 ||
                strncmp(id.c_str() + mn, token, mt) != 0) { return false; }
            next(); // remove spaces after the MFEM_'token'
            return true;
         };
         const bool JIT = is(id, "JIT");
         if (ker.is_wait() && JIT) { mfem_jit_kernel(); }
         if (ker.is_body() && !JIT) { ker.src << id; ker.dup << id; }
         if (ker.is_wait()) { out << id; }
      } // MFEM_*

      if (ker.is_body() && is_end_of(block,'{','}')) { mfem_jit_postfix(); }
      if (ker.is_include())
      {
         ker.advance(); check(ker.is_wait());
         token(); // to handle immediate MFEM_JIT (Parse while does a put)
      }
   }

   /*
    * Parser operator which processes all the tokens.
    * EXIT_SUCCESS or EXIT_FAILURE
    */
   int Parse()
   {
      ker.includes.clear();
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

} // namespace jit

} // namespace internal

} // namespace mfem

int main(const int argc, char* argv[])
{
   string input, output, file;
   auto Help = [&]()
   {
      std::cout << "mjit: " << argv[0]
                << " [-h] [-o output_file] input_file" << std::endl;
      return EXIT_SUCCESS;
   };
   if (argc <= 1) { return Help(); }
   for (int i = 1; i < argc; i++)
   {
      if (argv[i] == string("-h")) { return Help(); } // help
      if (argv[i] == string("-o")) { output = argv[++i]; continue; }
      assert(argv[i] && "Could not use last argument as input file!");
      file = input = argv[i];
   }
   const bool empty = output.empty();
   const ios::openmode i_mode = ios::in  | ios::binary;
   const ios::openmode o_mode = ios::out | ios::binary | ios::trunc;
   ifstream ifs(input.c_str(), i_mode);
   ofstream ofs(output.c_str(), o_mode);
   assert((!ifs.fail()) && "Failed opening input file!");
   assert(ifs.is_open() && "Could not open input file!");
   assert((empty || ofs.is_open()) && "Could not open output file!");
   const int status =
      mfem::internal::jit::Parser(ifs, empty?std::cout:ofs, file).Parse();
   return ifs.close(), ofs.close(), status;
}

#endif // MFEM_USE_JIT
