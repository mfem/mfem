// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "HYPRE_config.h"
#include <cstdio>

#ifdef HYPRE_RELEASE_VERSION
#define HYPRE_VERSION_STRING HYPRE_RELEASE_VERSION
#elif defined(HYPRE_PACKAGE_VERSION)
#define HYPRE_VERSION_STRING HYPRE_PACKAGE_VERSION
#endif

// Macros to expand a macro as a string
#define STR_EXPAND(s) #s
#define STR(s) STR_EXPAND(s)

// Convert the HYPRE_RELEASE_VERSION macro (string) to integer.
// Examples: "2.10.0b" --> 21000, "2.11.2"  --> 21102
int main()
{
#ifdef HYPRE_VERSION_STRING
   const char *ptr = STR(HYPRE_VERSION_STRING);
   if (*ptr == '"') { ptr++; }
   int version = 0;
   for (int i = 0; i < 3; i++, ptr++)
   {
      int pv = 0;
      for (char d; d = *ptr, '0' <= d && d <= '9'; ptr++)
      {
         pv = 10*pv + (d - '0');
         if (pv >= 100) { return 1; }
      }
      version = 100*version + pv;
   }
   printf("%i\n", version);
   return 0;
#else
   return 2;
#endif
}
