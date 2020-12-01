// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "dmumps_c.h"
#include <string>
#include <iostream>
#include <algorithm>

// Macros to expand a macro as a string
#define STR_EXPAND(s) #s
#define STR(s) STR_EXPAND(s)

int main()
{
#ifdef MUMPS_VERSION
   const char *ptr = STR(MUMPS_VERSION);
   std::string s(ptr);
   s.erase(std::remove(s.begin(), s.end(), '"'), s.end());
   s.erase(std::remove(s.begin(), s.end(), '.'), s.end());
   std::cout << s << "\n";
   return 0;
#else
   return -1;
#endif
}
