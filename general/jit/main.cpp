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

#include <string>
#include <cassert>
#include <fstream>
#include <iostream>

#include "jit.hpp"

namespace mfem
{

int JitPreProcess(std::istream &in, std::ostream &out, std::string &file);

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
   } help;

   if (argc <= 1) { return help(argv); }

   for (int i = 1; i < argc; i++)
   {
      // -h lauches help
      if (argv[i] == std::string("-h")) { return help(argv); }

      // -o selects the output
      if (argv[i] == std::string("-o")) { output = argv[++i]; continue; }

      // last argument should be the input file
      assert(argv[i]);
      file = input = argv[i];
   }
   const bool ofile = !output.empty();
   std::ifstream ifs(input.c_str(), std::ios::in | std::ios::binary);
   std::ofstream ofs(output.c_str(),
                     std::ios::out | std::ios::binary | std::ios::trunc);
   assert(!ifs.fail());
   assert(ifs.is_open());
   if (ofile) { assert(ofs.is_open()); }
   const int status = mfem::JitPreProcess(ifs, ofile ? ofs : std::cout, file);
   ifs.close();
   ofs.close();
   return status;
}

#endif // MFEM_USE_JIT
