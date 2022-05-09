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

#include <algorithm> // for std::transform
#include <list>
#include <string>
#include <cassert>
#include <cstring> // for std::strlen
#include <fstream>
#include <iostream>
#include <sstream>

#include "jit.hpp" // for Hash, ToHashString, Find

namespace mfem
{

struct Parser
{
   struct kernel_t
   {
      std::string name; // Kernel name
      std::string Targs, Tparams, Tformat, Ttest; // arguments, parameters, format
      std::string Sargs, Sparams0, Sparams; // Symbol arguments, parameters
      struct { int dim; std::string e, N, X, Y, Z; std::ostringstream body; } forall;
      std::ostringstream source, body;
      bool eq;

      struct fsm_t
      {
         bool put(const char c);
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

      bool put(const char c)
      {
         if (is_targs())   { Tparams += c; }
         if (is_params())  { Sparams0 += c; if (!eq) { Sparams += c; } }
         if (is_prefix())  { source << c; body << c; return true;}
         if (is_postfix()) { source << c; body << c; return true;}
         if (is_forall())  { forall.body << c; return true;}
         if (is_kernel())  { forall.body << c; return true;}
         return false;
      }
   };

   kernel_t ker;
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

   char put() { check(good()); return put(get()); }

   char put(const char c)
   {
      if (c == '\n') { line++; }
      if (ker.put(c)) { return c; }
      out.put(c);
      return c;
   }

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
      if (!good()) { return; }

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
      check(is_id(), "name w/o alnum 1st letter");
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
         if (is_coma()) { ker.Tformat += ",%d";}
         if (peek_n(2) == "T_")
         {
            std::string id = peek_id();
            /*{
               if (!ker.Ttest.empty()) { ker.Ttest += " && "; }
               ker.Ttest += id + "==0";
            }*/
            add_arg(ker.Targs, (id.erase(0, 2), to_lower(id)));
         }
         if (in.peek() == '>') { break;}
         put(); // to ker.Tparams
      }
      ker.advance(); // Targ => Symbol
      check(put()=='>',"no '>' in kernel!");

      next(); // 'static' ?
      if (is_static()) { out << get_id(); }

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
      //dbg(154) << "\nker.Sargs: " <<ker.Sargs;
      //dbg(154) << "\nker.Sparams:" << ker.Sparams;
      //dbg(154) << "\nker.Sparams0:" << ker.Sparams0;
      //assert(false);

      // Make sure we hit the last ')' of the arguments
      check(put()==')',"no last ')' in kernel");

      // Make sure we are about to start a compound statement
      next_check([&]() { return put()=='{';}, "no compound statement found");

      // Generate the kernel prefix for this kernel
      ker.advance(); // Body => Prefix
      ker.source.clear();
      ker.source.str(std::string());
      ker.body.clear();
      ker.body.str(std::string());
      // flush also ker.forall.body that can be missed with a kernel w/o forall
      ker.forall.body.clear();
      ker.forall.body.str(std::string());

      //out << "\n\tconstexpr bool USE_JIT = " << ker.Ttest << ";";
      ker.source << "\n\tconst bool use_dev = Device::Allows(Backend::DEVICE_MASK);";

      /*ut << "\nprintf(\"%s\", use_dev ? "
          << "\"\\033[32m USING_DEVICE\\033[m\":"
          << "\"\\033[31m USING_HOST\\033[m\");";*/

      //out << "\nif (USE_JIT){";

      //out << "\nprintf(\"\\033[35m USING_JIT\\033[m\");";

      ker.source << "\n\tconst char *source = R\"_(";

      // switching from out to ker.source to compute the hash,
      // defining 'MFEM_JIT_COMPILATION' to avoid MFEM_GPU_CHECK in cuda.hpp
      ker.source << "\n#define MFEM_JIT_COMPILATION"
                 << "\n#define MFEM_MEM_MANAGER_HPP" // pulls HYPRE_config.h
                 << "\n#define MFEM_DEVICE_HPP";

#ifdef MFEM_USE_CUDA
      // avoid mfem_cuda_error inside MFEM_GPU_CHECK
      ker.source << "\n#define MFEM_GPU_CHECK(...)";
#endif
      // used in forall.hpp
      ker.source << "\nstruct Backend { enum: unsigned long {"
                 << "CPU=1<<0, CUDA=1<<2, HIP=1<<3, DEBUG_DEVICE=1<<14};};"
                 << "namespace mfem{ struct Device{"
                 << "   static constexpr unsigned long CUDA = (1 << 2);"
                 << "   static constexpr unsigned long backends = CUDA;"
                 << "   static inline bool Allows(unsigned long b_mask)"
                 << "      { return Device::backends & b_mask; }};}";

      ker.source << "\n#include \"" << MFEM_INSTALL_DIR
                 << "/include/mfem/general/forall.hpp\"";

      ker.source << "\n#include \"" << MFEM_INSTALL_DIR
                 << "/include/mfem/general/jit/jit.hpp\""; // for Hash, Find

      ker.source << "\nusing namespace mfem;";

      ker.source << "\n\ntemplate<" << ker.Tparams << ">"
                 << "\nvoid " << ker.name << "_%016lx"
                 << "(const bool use_dev," << ker.Sparams0 << "){";

      ker.source << "\n#line " << std::to_string(line) << " \"" << file << "\"\n";
      block = 0; // Start counting the block statements
   }

   void mfem_forall_prefix(const std::string &id)
   {
      // Switch from prefix capturing, to the forall one
      ker.advance(); // prefix => forall
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

   void mfem_kernel_unroll()
   {
      check(ker.is_kernel(), "Unroll not in kernel error!");
      ker.forall.body << "#pragma unroll ";
      while (good() && get() != '(') { check(!in.eof()); }
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
      std::ostringstream device_forall;
      const char *ND = ker.forall.dim == 2 ? "2D" : "3D";
      device_forall  << "if (use_dev){"
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
      ker.source << device_forall.str();

      std::ostringstream mfem_forall;
      mfem_forall  << "MFEM_FORALL_" << ND
                   << "(" << ker.forall.e << ","
                   << ker.forall.N.c_str() << ","
                   << ker.forall.X.c_str() << ","
                   << ker.forall.Y.c_str() << ","
                   << ker.forall.Z.c_str() << ","
                   << ker.forall.body.str() <<  ");";
      ker.body << mfem_forall.str();
#endif
      std::ostringstream host_forall;
      host_forall << "for (int k=0; k<" << ker.forall.N.c_str() << "; k++) {"
                  << "[&] (int " << ker.forall.e <<")"
                  << ker.forall.body.str() << "(k);"
                  << "}";
      ker.source << host_forall.str();

#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
      ker.source << "}";
#endif
      ker.advance(/*kernel => postfix*/);
   }

   void mfem_jit_postfix() // output all kernel source, with updated hash
   {
      ker.source << "}\nextern \"C\" void k%016lx"
                 << "(const bool use_dev," << ker.Sparams0 << "){"
                 << ker.name << "_%016lx<"<< ker.Tformat << ">"
                 << "(use_dev,"<< ker.Sargs << ");}";

      const size_t seed = Jit::Hash(std::hash<std::string> {}(ker.source.str()),
                                    std::string(MFEM_JIT_CXX),
                                    std::string(MFEM_JIT_BUILD_FLAGS),
                                    std::string(MFEM_SOURCE_DIR),
                                    std::string(MFEM_INSTALL_DIR));

      out << ker.body.str() << "}";
      out << "\nvoid " << ker.name << "(" << ker.Sparams0 << ") {";
      out << ker.source.str().c_str() << ")_\";" // end of source
          << "\nconst size_t hash = Jit::Hash("
          << "0x" << std::hex << seed << std::dec << "ul," << ker.Targs << ");"
          << "\ntypedef void (*kernel_t)(bool use_dev," << ker.Sparams << ");"
          << "\nstatic std::unordered_map<size_t, Jit::Kernel<kernel_t>> km;"
          << "\nJit::Find(hash, source, km, " << ker.Targs << ")"
          << ".operator()(use_dev," << ker.Sargs << ");";
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

         const bool unroll = MFEM_token(id, "UNROLL");
         if (ker.is_kernel() && unroll) { mfem_kernel_unroll(); }

         if (ker.is_wait()) { out << id; }
         if (ker.is_prefix() && !JIT) { ker.source << id; ker.body << id; }
         if (ker.is_forall()) { ker.forall.body << id; }
         if (ker.is_kernel() && !unroll && !forall) { ker.forall.body << id; }
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

} // namespace mfem

int main(const int argc, char* argv[])
{
   std::string input, output, file;
   struct
   {
      int operator()(char* argv[])
      {
         std::cout << "mjit: ";
         std::cout << argv[0] << " [-h] [-o output] input" << std::endl;
         return EXIT_SUCCESS;
      }
   } Help;

   if (argc <= 1) { return Help(argv); }
   for (int i = 1; i < argc; i++)
   {
      if (argv[i] == std::string("-h")) { return Help(argv); } // help
      if (argv[i] == std::string("-o")) { output = argv[++i]; continue; }
      assert(argv[i]); // last argument should be the input file
      file = input = argv[i];
   }
   const bool ofile = !output.empty();
   std::ifstream ifs(input.c_str(), std::ios::in | std::ios::binary);
   std::ofstream ofs(output.c_str(),
                     std::ios::out | std::ios::binary | std::ios::trunc);
   assert(!ifs.fail());
   assert(ifs.is_open());
   if (ofile) { assert(ofs.is_open()); }
   const int status =
      mfem::Parser(ifs, ofile ? ofs : std::cout, file).operator()();;
   ifs.close();
   ofs.close();
   return status;
}

#endif // MFEM_USE_JIT
