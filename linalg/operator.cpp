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

#include <iostream>
#include <iomanip>

#include "vector.hpp"
#include "operator.hpp"

namespace mfem
{

void Operator::PrintMatlab (std::ostream & out, int n, int m)
{
   using namespace std;
   if (n == 0) { n = width; }
   if (m == 0) { m = height; }

   Vector x(n), y(m);
   x = 0.0;

   int i, j;
   out << setiosflags(ios::scientific | ios::showpos);
   for (i = 0; i < n; i++)
   {
      if (i != 0)
      {
         x(i-1) = 0.0;
      }
      x(i) = 1.0;
      Mult(x,y);
      for (j = 0; j < m; j++)
         if (y(j))
         {
            out << j+1 << " " << i+1 << " " << y(j) << '\n';
         }
   }
}

}
