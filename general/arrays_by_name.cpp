// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "arrays_by_name.hpp"

namespace mfem
{

template <class T>
void ArraysByName<T>::Print(std::ostream &os, int width) const
{
   os << data.size() << '\n';
   for (auto const &it : data)
   {
      os << '"' << it.first << '"' << ' ' << it.second.Size() << ' ';
      it.second.Print(os, width > 0 ? width : it.second.Size());
   }
}

template <class T>
void ArraysByName<T>::Load(std::istream &in)
{
   int NumArrays;
   in >> NumArrays;

   std::string ArrayLine, ArrayName;
   for (int i=0; i < NumArrays; i++)
   {
      in >> std::ws;
      getline(in, ArrayLine);

      std::size_t q0 = ArrayLine.find('"');
      std::size_t q1 = ArrayLine.rfind('"');

      if (q0 != std::string::npos && q1 > q0)
      {
         // Locate set name between first and last double quote
         ArrayName = ArrayLine.substr(q0+1,q1-q0-1);
      }
      else
      {
         // If no double quotes found locate set name using white space
         q1 = ArrayLine.find(' ');
         ArrayName = ArrayLine.substr(0,q1-1);
      }

      // Prepare an input stream to read the rest of the line
      std::istringstream istr;
      istr.str(ArrayLine.substr(q1+1));

      data[ArrayName].Load(istr, 0);
   }

}

template class ArraysByName<int>;

} // namespace mfem
