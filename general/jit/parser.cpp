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
#include <sstream>
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

   struct kernel_t
   {
      struct fsm_t
      {
         static constexpr uint8_t color = 201;
         typedef void (fsm_t::*State) ();
         State state;
         fsm_t(): state(&fsm_t::Wait) { dbg(color)<<"\n[WAIT]"; }
         void Wait()    { dbg(color)<<"\n[JIT]";     Next(&fsm_t::Jit); }
         void Jit()     { dbg(color)<<"\n[TARGS]";   Next(&fsm_t::Targs); }
         void Targs()   { dbg(color)<<"\n[Symbol]";  Next(&fsm_t::Symbol); }
         void Symbol()  { dbg(color)<<"\n[PARAMS]";  Next(&fsm_t::Params); }
         void Params()  { dbg(color)<<"\n[BODY]";    Next(&fsm_t::Body); }
         void Body()    { dbg(color)<<"\n[PREFIX]";  Next(&fsm_t::Prefix); }
         void Prefix()  { dbg(color)<<"\n[FORALL]";  Next(&fsm_t::Forall); }
         void Forall()  { dbg(color)<<"\n[KERNEL]";  Next(&fsm_t::Kernel); }
         void Kernel()  { dbg(color)<<"\n[POSTFIX]"; Next(&fsm_t::Postfix); }
         void Postfix() { dbg(color)<<"\n[WAIT]";    Next(&fsm_t::Wait); }
         void Next(State next) { state = next; }
         void advance() { (this->*state)(); }
      } fsm;
      bool is_wait() { return fsm.state == &fsm_t::Wait; }
      bool is_targs() { return fsm.state == &fsm_t::Targs; }
      bool is_params() { return fsm.state == &fsm_t::Params; }
      bool is_prefix() { return fsm.state == &fsm_t::Prefix; }
      bool is_forall() { return fsm.state == &fsm_t::Forall; }
      bool is_kernel() { return fsm.state == &fsm_t::Kernel; }
      bool is_postfix() { return fsm.state == &fsm_t::Postfix; }
      void advance() { fsm.advance(); }
      std::string name;       // Kernel name
      // Template arguments, parameters, format
      std::string Targs, Tparams, Tformat;
      // Symbol arguments, parameters
      std::string Sargs, Sparams;
      struct { int dim; std::string e, N, X, Y, Z; std::ostringstream body; } forall;
      std::ostringstream source;
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
      in(in), out(out), file(file), line(1), block(-2), parenthesis(-2) { }

   void check(const bool test, const char *error)
   {
      if (!test) { throw error_t(line, file, error); }
   }

   void add_coma(std::string &arg) { if (!arg.empty()) { arg += ",";  } }

   template<typename T>
   void add_arg(std::string &as, const T a) { add_coma(as); as += a; }

   bool good() { in.peek(); return in.good(); }

   char get() { return static_cast<char>(in.get()); }

   char put(const char c)
   {
      if (ker.is_targs())  { ker.Tparams += c; }
      if (ker.is_params()) { ker.Sparams += c; }
      if (ker.is_prefix()) { ker.source << c; return c;}
      if (ker.is_forall()) { ker.forall.body << c; return c;}
      if (ker.is_kernel()) { ker.forall.body << c; return c;}
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

   void next(bool keep = true) { skip_space(keep); comments(keep); }

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
      for (k = 0; k < n && good() && !in.eof(); k++) { c[k] = get(); }
      std::string rtn((c[k]=0, c));
      if (!good()) { return rtn; }
      for (int l = 0; l < k; l++) { in.unget(); }
      return rtn;
   }

   template<int M = 16> std::string peek_id(const int n = M-1)
   {
      int k = 0;
      assert(n < M);
      static char c[M];
      for (k = 0; k < n && good() && is_id() && !in.eof(); k++) { c[k] = get(); }
      std::string str((c[k]=0, c));
      for (int l = 0; l < k; l++) { in.unget(); }
      return str;
   }

   void drop_id() { while (is_id()) { get(); } }

   bool is_string(const char *str) { return peek_n(std::strlen(str)) == str; }
   bool is_template() { return is_string("template"); }
   bool is_static() { return is_string("static"); }
   bool is_void() { return is_string("void"); }

   bool is_space() { return std::isspace(in.peek()); }
   bool is_alpha() { return std::isalpha(in.peek()); }

   bool is_char(const unsigned char c) { return in.peek() == c; }
   bool is_eq() { return is_char('='); }
   bool is_amp() { return is_char('&'); }
   bool is_coma() { return is_char(','); }
   bool is_star() { return is_char('*'); }
   bool is_linefeed() { return is_char('\n'); }
   bool is_semicolon() { return is_char(';'); }
   bool is_left_parenthesis() { return is_char('('); }
   bool is_right_parenthesis() { return is_char(')'); }

   void mfem_jit_prefix()
   {
      next();
      check(is_template(), "kernel is missing the 'template' token");

      // copy the 'template<...>' in Tparams_src
      out << get_id();
      next();
      check(in.peek()=='<',"no '<' in single source kernel!");
      put();
      ker.Targs.clear();
      ker.Tparams.clear();
      ker.advance(); // jit => Targ
      ker.Tformat = "%d";
      while (good() && in.peek() != '>')
      {
         next();
         if (is_coma()) { ker.Tformat += ",%d";}
         if (peek_n(2) == "T_")
         {
            std::string T_ID = peek_id(), id = T_ID;
            id.erase(0, 2);
            std::transform(id.begin(), id.end(), id.begin(),
            [](unsigned char c) { return std::tolower(c); });
            dbg() << T_ID << ":" << id;
            add_arg(ker.Targs, id);
         }
         put(); // will output in Tparams_src
      }
      ker.advance(); // Targ => Symbol
      const char Tc = put(); assert(Tc == '>');

      next(); // 'static' check
      if (is_static()) { out << get_id(); }

      next(); // 'void' check
      check(is_void(), "kernel return type should be void");
      const std::string void_return_type = get_id();
      out << void_return_type;

      next(); // Get kernel's name
      out << (ker.name = get_id());

      // check we are at the left parenthesis
      next();
      check(put()=='(', "no 1st '(' in kernel");
      ker.advance(); // Symbol => Params

      // Get the arguments
      bool eq = false;
      ker.Sargs.clear();
      std::string id;
      auto last = [](std::string &s) { return s.substr(s.find_last_of('.')+1);};
      while (good() && in.peek() != ')')
      {
         comments();
         assert(is_space() || is_id() || is_star() || is_amp() || is_eq() || is_coma());
         if (is_space() && !id.empty() && id.back() != '.' && !eq) { id += '.'; }
         if (is_eq()) { eq = true; id = id.substr(0, id.size()-1); }
         if (is_id() && !eq) { id += in.peek(); }
         if (is_coma()) { add_arg(ker.Sargs, last(id)); id.clear(); eq = false; }
         eq ? get() : put(); // remove the '= digit'
      }
      add_arg(ker.Sargs, last(id));
      dbg() << "\nker.Targs:" << ker.Targs;
      dbg() << "\nker.Tformat:" << ker.Tformat;
      dbg() << "\nker.Tparams:" << ker.Tparams;
      dbg() << "\nker.Sparams:" << ker.Sparams;
      dbg() << "\nker.Sargs:" << ker.Sargs; //STOP;
      ker.advance(); // Params => Body

      // Make sure we have hit the last ')' of the arguments
      check(put()==')',"no last ')' in kernel");

      next(); // Make sure we are about to start a compound statement
      check(put()=='{',"no compound statement found");

      // Generate the kernel prefix for this kernel
      ker.source.clear();
      ker.advance(); // Body => Prefix
      out << "\n\tconst char *source = R\"_(";

      // switching from out to ker.source to compute the hash
      ker.source << "#define MFEM_JIT_FORALL_COMPILATION";
      ker.source << "\n#define MFEM_DEVICE_HPP";

      ker.source << "\n#include \"";
      ker.source << MFEM_INSTALL_DIR;
      ker.source << "/include/mfem/general/forall.hpp\"";
      ker.source << "\nusing namespace mfem;";

      ker.source << "\n\ntemplate<";
      ker.source << ker.Tparams;
      ker.source << ">";
      ker.source << "\nvoid ";
      ker.source << ker.name;
      ker.source << "_%016lx(";
      ker.source << "const bool use_dev,";
      ker.source << ker.Sparams;
      ker.source << "){";

      // Push the preprocessor #line directive
      ker.source << "\n#line ";
      ker.source << std::to_string(line);
      ker.source << " \"";
      ker.source << file;
      ker.source << "\"";
   }

   void mfem_jit_postfix()
   {
      ker.source << "}\nextern \"C\" void k%016lx(";
      ker.source << "const bool use_dev,";
      ker.source << ker.Sparams;
      ker.source << "){";
      ker.source << ker.name;
      ker.source << "_%016lx<";
      ker.source << ker.Tformat;
      ker.source << ">(use_dev, ";
      ker.source << ker.Sargs;
      ker.source << ");}";

      // output all kernel source, after having computed its hash
      //#warning sources
      out << ker.source.str();
      out << ")_\";"; // eos

      const size_t seed = Jit::Hash(std::hash<std::string> {}(ker.source.str()),
                                    std::string(MFEM_JIT_CXX),
                                    std::string(MFEM_JIT_BUILD_FLAGS),
                                    std::string(MFEM_SOURCE_DIR),
                                    std::string(MFEM_INSTALL_DIR));
      // kernel hash
      out << "\nconst size_t hash = Jit::Hash("
          << "0x" << std::hex << seed << std::dec << "ul, "
          << ker.Targs << ");";

      // kernel typedef
      out << "\ntypedef void (*kernel_t)(bool use_dev," << ker.Sparams << ");";

      // kernel map
      out << "\nstatic std::unordered_map<size_t, Jit::Kernel<kernel_t>> ks;";

      // Add kernel in map if not already present
      out << "\nauto ks_iter = ks.find(hash);";
      out << "\nif (ks_iter == ks.end()){";
      out << "\n\tconst int n = 1 + " // source size
          << "snprintf(nullptr, 0, source, hash, hash, hash, " << ker.Targs << ");";
      out << "\n\tchar *Tsrc = new char[n];";
      out << "\n\tsnprintf(Tsrc, n, source, hash, hash, hash, "<< ker.Targs << ");";
      out << "\n\tstd::stringstream ss;"; // prepare symbol from computed hash
      out << "\n\tss << 'k' << std::setfill('0') ";
      out << "<< std::setw(16) << std::hex << (hash|0) << std::dec;";
      out << "\n\tconst int SYMBOL_SIZE = 1+16+1;";
      out << "\n\tchar *symbol = new char[SYMBOL_SIZE];";
      out << "\n\tmemcpy(symbol, ss.str().c_str(), SYMBOL_SIZE);";
      out << "\n\tauto res = ks.emplace(hash, Jit::Kernel<kernel_t>(hash, Tsrc, symbol));";
      out << "\n\tassert(res.second); // was not already in the map";
      out << "\n\tks_iter = ks.find(hash);";
      out << "\n\tassert(ks_iter != ks.end());";
      out << "\n\tdelete[] symbol;";
      out << "\n\tdelete[] Tsrc;";
      out << "\n}";

      // #warning should be CUDA dependent
      out << "\nconst bool use_dev = Device::Allows(Backend::CUDA_MASK);";
      out << "\nks_iter->second.operator()(use_dev," << ker.Sargs << ");";

      // Stop counting the blocks and flush the kernel status
      block--;

      out << "\n#line " << std::to_string(line) << " \"" << file << "\"\n";
   }

   void mfem_unroll()
   {
      assert (ker.is_kernel());
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

      ker.forall.body << "#pragma unroll ";
      ker.forall.body << depth.c_str();
      ker.forall.body << "\n";
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
      ker.source << "for (int ";
      ker.source <<  ker.forall.e ;
      ker.source << " = 0; ";
      ker.source << ker.forall.e;
      ker.source << " < ";
      ker.source << ker.forall.N.c_str();
      ker.source << "; ";
      ker.source << ker.forall.e;
      ker.source << "++)";
      ker.source << ker.forall.body.str();
      ker.source << "\n";
      //ker.source += "/**/}";
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
      auto is_postfix = [&](int &count, const char i, const char o)
      {
         if (count >= 0 && in.peek() == i) { count++; }
         if (count >= 0 && in.peek() == o) { count--; }
         if (count == -1) { return true; }
         return false;
      };

      if (peek_n(4) == "MFEM")
      {
         const std::string &id = get_id();
         //dbg(ker.is_jit() ? 154 : 0) << id;

         auto MFEM_token = [&](const std::string &id, const char *token)
         {
            if (strncmp(id.c_str(), "MFEM_", 5) != 0 ) { return false; }
            const int mn = strlen("MFEM_"), mt = strlen(token);
            if (strncmp(id.c_str() + mn, token, mt) != 0 ) { return false; }
            return true;
         };

         const bool JIT = MFEM_token(id, "JIT");
         if (ker.is_wait() && JIT) // MFEM_JIT
         {
            ker.advance(); // wait => jit
            mfem_jit_prefix();
            block = 0; // Starts counting the block depth
         }

         const bool forall = MFEM_token(id, "FORALL");
         if (ker.is_prefix() && forall) // MFEM_FORALL
         {
            // Switch from prefix capturing, to the forall one
            ker.advance(); // prefix => forall
            forall_prefix(id);
            // Starts counting the parentheses to wait for end of MFEM_FORALL'('
            parenthesis = 0;
            ker.advance(); // forall => kernel
         }

         const bool unroll = MFEM_token(id, "UNROLL");
         if (ker.is_kernel() && unroll) { mfem_unroll(); }

         if (ker.is_wait()) { out << id; }
         if (ker.is_prefix() && !JIT) { ker.source << id; }
         if (ker.is_forall()) { ker.forall.body << id; }
         if (ker.is_kernel() && !unroll && !forall) { ker.forall.body << id; }
      } // MFEM_*

      if (ker.is_kernel() && is_postfix(parenthesis,'(',')'))
      {
         forall_postfix();
         ker.advance(); // kernel => postfix
      }

      if (ker.is_postfix() && is_postfix(block,'{','}'))
      {
         mfem_jit_postfix();
         ker.advance(); // postfix => wait
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
