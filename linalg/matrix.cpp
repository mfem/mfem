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

// Implementation of class matrix

#include "matrix.hpp"
#include <iostream>
#include <iomanip>


namespace mfem
{

void Matrix::Print (std::ostream & out, int width_) const
{
   using namespace std;
   // output flags = scientific + show sign
   out << setiosflags(ios::scientific | ios::showpos);
   for (int i = 0; i < height; i++)
   {
      out << "[row " << i << "]\n";
      for (int j = 0; j < width; j++)
      {
         out << Elem(i,j) << " ";
         if ( !((j+1) % width_) )
         {
            out << '\n';
         }
      }
      out << '\n';
   }
   out << '\n';
}

}
