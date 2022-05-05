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
#include <cassert>
#include <iostream>
#include <sstream>

namespace mfem
{

struct JitPreProcessor
{
   /// \brief Terminal binary arguments hash combine function
   template <typename T> static inline
   size_t Hash(const size_t &h, const T &a) noexcept
   { return h ^ (std::hash<T> {}(a) + 0x9e3779b97f4a7c15ull + (h<<12) + (h>>4));}

   /// \brief Variadic hash combine function
   template<typename T, typename... Args> static inline
   size_t Hash(const size_t &h, const T &arg, Args... args) noexcept
   { return Hash(Hash(h, arg), args...); }

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
         static constexpr uint8_t c = 201;
         typedef void (fsm_t::*State) ();
         State state;
         fsm_t(): state(&fsm_t::wait) { dbg(c)<<"\n[WAIT]"; }
         void wait()    { dbg(c)<<"\n[JIT]";     next(&fsm_t::jit); }
         void jit()     { dbg(c)<<"\n[TARGS]";   next(&fsm_t::targs); }
         void targs()   { dbg(c)<<"\n[Symbol]";  next(&fsm_t::symbol); }
         void symbol()  { dbg(c)<<"\n[PARAMS]";  next(&fsm_t::params); }
         void params()  { dbg(c)<<"\n[BODY]";    next(&fsm_t::body); }
         void body()    { dbg(c)<<"\n[PREFIX]";  next(&fsm_t::prefix); }
         void prefix()  { dbg(c)<<"\n[FORALL]";  next(&fsm_t::forall); }
         void forall()  { dbg(c)<<"\n[KERNEL]";  next(&fsm_t::kernel); }
         void kernel()  { dbg(c)<<"\n[POSTFIX]"; next(&fsm_t::postfix); }
         void postfix() { dbg(c)<<"\n[WAIT]";    next(&fsm_t::wait); }
         void next(State next) { state = next; }
         void advance() { (this->*state)(); }
      } fsm;

      void advance() { fsm.advance(); }
      bool is_wait() { return fsm.state == &fsm_t::wait; }
      bool is_targs() { return fsm.state == &fsm_t::targs; }
      bool is_params() { return fsm.state == &fsm_t::params; }
      bool is_prefix() { return fsm.state == &fsm_t::prefix; }
      bool is_forall() { return fsm.state == &fsm_t::forall; }
      bool is_kernel() { return fsm.state == &fsm_t::kernel; }
      bool is_postfix() { return fsm.state == &fsm_t::postfix; }
      std::string name; // Kernel name
      std::string Targs, Tparams, Tformat,
          Ttest; // Template arguments, parameters, format
      std::string Sargs, Sparams0, Sparams; // Symbol arguments, parameters
      struct { int dim; std::string e, N, X, Y, Z; std::ostringstream body; } forall;
      std::ostringstream source, body;
      bool eq;
   };

   struct error_t
   {
      int line;
      std::string file;
      const char *message;
      error_t(int l, std::string f, const char *e): line(l), file(f), message(e) {}
   };

   kernel_t ker;
   std::istream &in;
   std::ostream &out;
   std::string &file;
   int line, block, parenthesis;

   JitPreProcessor(std::istream &in, std::ostream &out, std::string &file) :
      in(in), out(out), file(file), line(1), block(-1), parenthesis(-1) { }

   void error(const char *msg) { throw error_t(line, file, msg);}
   void check(const bool tst, const char *msg) { if (!tst) { error(msg); }}

   template<typename T>
   void add_arg(std::string &s, const T a) { if (!s.empty()) { s += ","; } s += a; }

   bool good() { in.peek(); return in.good() && !in.eof(); }

   char get() { return static_cast<char>(in.get()); }

   char put() { assert(good()); return put(get()); }

   char put(const char c)
   {
      if (c == '\n') { line++; }
      if (ker.is_targs())   { ker.Tparams += c; }
      if (ker.is_params())
      {
         ker.Sparams0 += c;
         if (!ker.eq) { ker.Sparams += c; }
      }
      if (ker.is_prefix())  { ker.source << c; ker.body << c; return c;}
      if (ker.is_postfix()) { ker.source << c; ker.body << c; return c;}
      if (ker.is_forall())  { ker.forall.body << c; return c;}
      if (ker.is_kernel())  { ker.forall.body << c; return c;}
      out.put(c);
      return c;
   }

   bool is_space() { return good() && std::isspace(in.peek()); }

   void skip_space() { while (is_space()) { put(); } }

   bool is_comment()
   {
      if (!good()) { return false; }
      assert(good());
      if (in.peek() != '/') { return false; }
      in.get();
      assert(!in.eof());
      const int c = in.peek();
      in.unget();
      if (c == '/' || c == '*') { return true; }
      return false;
   }

   void comments()
   {
      if (!good()) { return; }

      while (is_comment())
      {
         check(put()=='/', "unknown comment");
         assert(is_slash() || is_star());
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

   void next() { assert(good()); skip_space(); comments(); }

   bool is_id() { const char c = in.peek(); return std::isalnum(c) || c == '_'; }

   std::string get_id()
   {
      std::string id;
      check(is_id(), "name w/o alnum 1st letter");
      while (is_id()) { id += get(); }
      return id;
   }

   template<typename L, int M = 16> std::string peek(const int n, L &&op)
   {
      int k = 0;
      assert(n < M);
      static char c[M];
      for (k = 0; k < n && good() && !in.eof() && op(); k++) { c[k] = get(); }
      std::string str((c[k]=0, c));
      for (int l = 0; l < k; l++) { in.unget(); }
      if (!good()) { return str; }
      return str;
   }
   std::string peek_id(int n = 15) { return peek(n, [&]() { return is_id(); });}
   std::string peek_n(int n) { return peek(n, [&]() { return true; }); }

   template<typename L>
   void next_check(L&&op, const char *msg = "") { next(); check(op(),msg); }

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
      next(); // 'template' ?
      check(is_template(), "kernel is missing the 'template' token");
      out << get_id();

      next(); // '<' ?
      check(put()=='<',"no '<' in kernel!");

      // preparing kernel's: Targs, Tparams & Tformat
      ker.advance(); // jit => Targ
      ker.Targs.clear();
      ker.Tparams.clear();
      ker.Tformat = "%d";
      ker.Ttest.clear();
      auto to_lower = [](std::string &id)
      {
         std::transform(id.begin(), id.end(), id.begin(),
         [](unsigned char c) { return std::tolower(c); });
         return id;
      };
      while (good())
      {
         next();
         assert(good());
         if (is_coma()) { ker.Tformat += ",%d";}
         if (peek_n(2) == "T_")
         {
            std::string id = peek_id();
            {
               if (!ker.Ttest.empty()) { ker.Ttest += " && "; }
               ker.Ttest += id + "==0";
            }
            add_arg(ker.Targs, (id.erase(0, 2), to_lower(id)));
         }
         if (in.peek() == '>') { break;}
         assert(good());
         put(); // to ker.Tparams
         assert(good());
      }
      ker.advance(); // Targ => Symbol
      check(put()=='>',"no '>' in kernel!");

      next(); // 'static' ?
      if (is_static()) { out << get_id(); }

      next(); // 'void' ?
      check(is_void(), "kernel return type should be void");
      out << get_id();

      next(); // kernel's name
      out << (ker.name = get_id());

      next(); // '(' ?
      check(put()=='(', "no 1st '(' in kernel");

      // Get the arguments
      ker.advance(); // Symbol => Params
      std::string id;
      ker.eq = false;
      ker.Sargs.clear();
      ker.Sparams.clear();
      ker.Sparams0.clear();
      auto last = [](std::string &s) { return s.substr(s.find_last_of('.')+1);};
      while (good() && in.peek() != ')')
      {
         if (is_comment())
         {
            comments();
            if (!id.empty() && id.back() != '.' && !ker.eq) { id += '.'; }
         }
         assert(good());
         if (in.peek() == ')') { break; } // to handle only comments
         assert(is_space() || is_id() || is_star() || is_eq() || is_coma());
         if (is_space() && !id.empty() && id.back() != '.' && !ker.eq) { id += '.'; }
         if (is_eq()) { ker.eq = true; id = id.substr(0, id.size()-1); }
         if (is_id() && !ker.eq) { id += in.peek(); }
         if (is_coma()) { add_arg(ker.Sargs, last(id)); id.clear(); ker.eq = false; }
         put();
      }
      add_arg(ker.Sargs, last(id));
      ker.advance(); // Params => Body

      // Make sure we hit the last ')' of the arguments
      check(put()==')',"no last ')' in kernel");

      next(); // Make sure we are about to start a compound statement
      check(put()=='{',"no compound statement found");

      // Generate the kernel prefix for this kernel
      ker.advance(); // Body => Prefix
      ker.source.clear();
      ker.source.str(std::string());
      ker.body.clear();
      ker.body.str(std::string());
      // flush also ker.forall.body that can be missed with a kernel w/o forall
      ker.forall.body.clear();
      ker.forall.body.str(std::string());

      out << "\n\tconst bool use_jit = " << ker.Ttest << ";";
      out << "\nif (use_jit){";
      out << "\n\tconst char *source = R\"_(";

      // switching from out to ker.source to compute the hash,
      // defining 'MFEM_JIT_COMPILATION' to avoid MFEM_GPU_CHECK in cuda.hpp
      ker.source << "#include  <unordered_map>"
                 << "\n#define MFEM_JIT_COMPILATION"
                 << "\n#define MFEM_MEM_MANAGER_HPP" // will pull HYPRE_config.h
                 << "\n#define MFEM_DEVICE_HPP";

#ifdef MFEM_USE_CUDA
      // avoid mfem_cuda_error inside MFEM_GPU_CHECK
      ker.source << "\n#define MFEM_GPU_CHECK(...)";
#endif
      // used in forall.hpp
      ker.source << "\nstruct Backend { enum: unsigned long {"
                 << "CPU=1<<0, CUDA=1<<2, HIP=1<<3, DEBUG_DEVICE=1<<14};};"
                 << "namespace mfem{ struct Device{"
                 << "   static constexpr unsigned long backends = 0;"
                 << "   static inline bool Allows(unsigned long b_mask)"
                 << "      { return Device::backends & b_mask; }"
                 << "}/*Device*/;}/*mfem*/";


      ker.source << "\n#include \"" << MFEM_INSTALL_DIR
                 << "/include/mfem/general/forall.hpp\"";

      ker.source << "\n//#include \"" << MFEM_INSTALL_DIR
                 << "/include/mfem/general/jit/jit.hpp\"";

      ker.source << "\nusing namespace mfem;";

      ker.source << "\n\ntemplate<" << ker.Tparams << ">"
                 << "\nvoid " << ker.name << "_%016lx"
                 << "(const bool use_dev," << ker.Sparams0 << "){"
                 << "\n#line " << std::to_string(line) << " \"" << file << "\"\n";
   }

   void mfem_forall_prefix(const std::string &id)
   {
      ker.forall.dim = id.c_str()[12] - 0x30;
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
   }

   void mfem_kernel_unroll()
   {
      assert (ker.is_kernel());
      ker.forall.body << "#pragma unroll ";
      while (good() && get() != '(') { assert(!in.eof()); }
      next(); ker.forall.body << get_id();
      next_check([&]() {return get()==')';}, "no last right parenthesis found");
      check(get() == '\n',"no newline found");
      ker.forall.body << "\n";
   }

   void mfem_forall_postfix()
   {
      check(get()==')',"no last right parenthesis found");
      next_check([&]() {return get()==';';}, "no last semicolon found");

#ifdef MFEM_USE_CUDA
#define MFEM_DEV_PREFIX "Cu"
#endif

#ifdef MFEM_USE_HIP
#define MFEM_DEV_PREFIX "Hip"
#endif

#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
      const char *ND = ker.forall.dim == 2 ? "2D" : "3D";
      ker.source << "if (use_dev){"
                 << "\n" << MFEM_DEV_PREFIX << "Wrap" << ND
                 << "(" << ker.forall.N.c_str() << ", "
                 << "[=] MFEM_DEVICE (int " << ker.forall.e <<")"
                 << ker.forall.body.str() << ","
                 << ker.forall.X.c_str() << ","
                 << ker.forall.Y.c_str() << ","
                 << ker.forall.Z.c_str()
                 << (ker.forall.dim == 3 ? ",0":"") // grid
                 << ");"
                 << " } else { ";
#endif
      ker.source << "for (int k=0; k<" << ker.forall.N.c_str() << "; k++) {"
                 << "[&] (int " << ker.forall.e <<")"
                 << ker.forall.body.str() << "(k);"
                 << "}";

#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
      ker.source << "}";
#endif
   }

   void mfem_jit_postfix()
   {
      ker.source << "}\nextern \"C\" void k%016lx"
                 << "(const bool use_dev," << ker.Sparams0 << "){"
                 << ker.name << "_%016lx<"<< ker.Tformat << ">"
                 << "(use_dev,"<< ker.Sargs << ");}";

      const size_t seed = /*Jit::*/Hash(std::hash<std::string> {}(ker.source.str()),
                                        std::string(MFEM_JIT_CXX),
                                        std::string(MFEM_JIT_BUILD_FLAGS),
                                        std::string(MFEM_SOURCE_DIR),
                                        std::string(MFEM_INSTALL_DIR));

      // output all kernel source, after having computed its hash
      out << ker.source.str().c_str() << ")_\";"; // end of source

      out << "\nstruct Local {"
          << "static inline size_t Hash(const size_t &h, const size_t &a) noexcept"
          << "{ return h ^ (std::hash<size_t> {}(a) + 0x9e3779b97f4a7c15ull + (h<<12) + (h>>4));}"
          << "static inline size_t Hash(const size_t seed, const std::list<int> &args){"
          << "  size_t hash = seed;"
          << "  for (size_t arg : args) { hash = Hash(hash, static_cast<size_t>(arg)); }"
          << " return hash; }";
      out << "};";

      out << "\nconst size_t hash = Local::Hash("
          << "0x" << std::hex << seed << std::dec << "ul,"
          << "{" << ker.Targs << "});";

      // kernel typedef
      out << "\ntypedef void (*kernel_t)(bool use_dev," << ker.Sparams << ");";

      // kernel map
      out << "\nstatic std::unordered_map<size_t, mfem::Jit::Kernel<kernel_t>> ks;";

      // Add kernel in map if not already present
      out << "\nauto ks_iter = ks.find(hash);";
      out << "\nif (ks_iter == ks.end()){"
          << "\n\tconst int Nsrc = 1 + " // source size
          << "snprintf(nullptr, 0, source, hash, hash, hash, " << ker.Targs << ");"
          << "\n\tchar *Tsrc = new char[Nsrc];"
          << "\n\tsnprintf(Tsrc, Nsrc, source, hash, hash, hash, "<< ker.Targs << ");"
          << "\n\tstd::stringstream ss;" // prepare symbol from computed hash
          << "\n\tss << 'k' << std::setfill('0') "
          << "<< std::setw(16) << std::hex << (hash|0) << std::dec;"
          << "\n\tconst int SYMBOL_SIZE = 1+16+1;"
          << "\n\tchar *symbol = new char[SYMBOL_SIZE];"
          << "\n\tmemcpy(symbol, ss.str().c_str(), SYMBOL_SIZE);"
          << "\n\tauto res = ks.emplace(hash, Jit::Kernel<kernel_t>(hash,Tsrc,symbol));"
          << "\n\tassert(res.second); // was not already in the map"
          << "\n\tks_iter = ks.find(hash);"
          << "\n\tassert(ks_iter != ks.end());"
          << "\n\tdelete[] symbol;"
          << "\n\tdelete[] Tsrc;"
          << "\n}";

      out << "\nconst bool use_dev = Device::Allows(Backend::DEVICE_MASK);";
      out << "\nks_iter->second.operator()(use_dev," << ker.Sargs << ");";

      out << "\n} else { // not use_jit\n";
      out << ker.body.str();
      out << "}";

      out << "\n#line " << std::to_string(line) << " \"" << file << "\"\n";
      out.flush();
   }

   void tokens()
   {
      assert(good());
      auto is_end_of = [&](int &c, const char i, const char o)
      {
         assert(good());
         if (c >= 0 && in.peek() == i) { c++; }
         if (c >= 0 && in.peek() == o) { c--; }
         if (c < 0) { return true; }
         return false;
      };

      if (peek_n(4) == "MFEM")
      {
         assert(good());
         const std::string &id = get_id();
         assert(good());

         auto MFEM_token = [&](const std::string &id, const char *token)
         {
            if (strncmp(id.c_str(), "MFEM_", 5) != 0 ) { return false; }
            const size_t mn = strlen("MFEM_"), mt = strlen(token);
            if (strncmp(id.c_str() + mn, token, mt) != 0) { return false; }
            if (strlen(id.c_str()) != mn+mt) { return false; }
            return true;
         };

         const bool JIT = MFEM_token(id, "JIT");
         assert(good());
         if (ker.is_wait() && JIT) // MFEM_JIT
         {
            ker.advance(); // wait => jit
            mfem_jit_prefix();
            block = 0; // Start counting the block statements
            assert(good());
         }

         assert(good());

         const bool forall =
            MFEM_token(id, "FORALL_2D") || MFEM_token(id, "FORALL_3D");
         if (ker.is_postfix() && forall)
         { throw error_t(line, file, "Only one MFEM_FORALL is supported!"); }

         if (ker.is_prefix() && forall) // MFEM_FORALL
         {
            assert(good());
            // Switch from prefix capturing, to the forall one
            ker.advance(); // prefix => forall
            mfem_forall_prefix(id);
            parenthesis = 0; // Start counting MFEM_FORALL's parentheses
            ker.advance(); // forall => kernel
            assert(good());
         }

         assert(good());

         const bool unroll = MFEM_token(id, "UNROLL");
         if (ker.is_kernel() && unroll) { mfem_kernel_unroll(); }

         assert(good());

         if (ker.is_wait()) { out << id; }
         assert(good());
         if (ker.is_prefix() && !JIT) { ker.source << id; ker.body << id; }
         assert(good());
         if (ker.is_forall()) { ker.forall.body << id; }
         assert(good());
         if (ker.is_kernel() && !unroll && !forall) { ker.forall.body << id; }
         assert(good());
      } // MFEM_*

      assert(good());

      if (ker.is_kernel() && is_end_of(parenthesis,'(',')'))
      {
         mfem_forall_postfix();
         ker.advance(/*kernel => postfix*/);
         assert(good());
      }

      // no forall shortcut
      const bool is_end_of_block = is_end_of(block,'{','}');
      if (ker.is_prefix() && is_end_of_block)
      {
         ker.fsm.forall(); // shortcut
         ker.fsm.kernel();
         assert(good());
      }

      if (ker.is_postfix() && is_end_of_block)
      {
         assert(good());
         mfem_jit_postfix();
         ker.advance(/*postfix => wait*/);
         assert(good());
      }
   }

   int operator()()
   {
      try { while (!in.eof()) { next(); if (good()) { tokens(); put(); } } }
      catch (error_t err)
      {
         std::cerr << std::endl << err.file << ":" << err.line << ":"
                   << " mjit error" << (err.message ? ": " : "")
                   << (err.message ? err.message : "") << std::endl;
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
