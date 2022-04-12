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

#include <string>
#include <cassert>
#include <fstream>
#include <iostream>

#include "../../config/config.hpp"
#include "../globals.hpp"
#include "jit.hpp"

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 159
#include "../debug.hpp"

namespace mfem
{

namespace jit
{

static int help(char* argv[])
{
   std::cout << "MFEM " << MFEM_JIT_LIB_NAME << ": ";
   std::cout << argv[0] << " [-ch] [-o output] input" << std::endl;
   return ~0;
}

} // namespace jit

} // namespace mfem

int main(const int argc, char* argv[])
{
   std::string input, output, file;

   if (argc <= 1) { return mfem::jit::help(argv); }

   for (int i = 1; i < argc; i++)
   {
      // -h lauches help
      if (argv[i] == std::string("-h")) { return mfem::jit::help(argv); }

      // -c will launch ProcessFork in parallel mode, nothing otherwise
      if (argv[i] == std::string(MFEM_JIT_COMMAND_LINE_OPTION))
      {
#ifdef MFEM_USE_MPI
         dbg("Compilation requested, forking...");
         return mfem::jit::ProcessFork(argc, argv);
#else
         return 0;
#endif
      }
      // -o selects the output
      if (argv[i] == std::string("-o"))
      {
         output = argv[i+=1];
         continue;
      }
      // last argument should be the input file
      const char* last_dot = mfem::jit::strrnc(argv[i], '.');
      const size_t ext_size = last_dot ? strlen(last_dot) : 0;
      if (last_dot && ext_size > 0)
      {
         assert(file.size() == 0);
         file = input = argv[i];
      }
   }
   assert(!input.empty());
   const bool output_file = !output.empty();
   std::ifstream in(input.c_str(), std::ios::in | std::ios::binary);
   std::ofstream out(output.c_str(),
                     std::ios::out | std::ios::binary | std::ios::trunc);
   assert(!in.fail());
   assert(in.is_open());
   if (output_file) {assert(out.is_open());}
   std::ostream &mfem_out(std::cout);
   mfem::jit::preprocess(in, output_file ? out : mfem_out, file);
   in.close();
   out.close();
   return 0;
}
