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

#ifndef MFEM_USE_MPI

#include <string>
using std::string;

#include "jit.hpp"
#include "tools.hpp"

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 226
#include "../debug.hpp"

namespace mfem
{

namespace jit
{

// The serial implementation does nothing special but the =system= command.
int System(const char *argv[])
{
   const int argc = argn(argv);
   if (argc < 2) { return EXIT_FAILURE; }
   string command(argv[1]);
   for (int k = 2; k < argc && argv[k]; k++)
   {
      command.append(" ");
      command.append(argv[k]);
   }
   const char *command_c_str = command.c_str();
   dbg(command_c_str);
   return ::system(command_c_str);
}

} // namespace jit

} // namespace mfem

#endif // MFEM_USE_MPI
