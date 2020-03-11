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


// Abstract array data type

#include "array.hpp"
#include "../general/forall.hpp"
#include <fstream>

namespace mfem
{

template <class T>
void Array<T>::Print(std::ostream &out, int width) const
{
   for (int i = 0; i < size; i++)
   {
      out << data[i];
      if ( !((i+1) % width) || i+1 == size )
      {
         out << '\n';
      }
      else
      {
         out << " ";
      }
   }
}

template <class T>
void Array<T>::Save(std::ostream &out, int fmt) const
{
   if (fmt == 0)
   {
      out << size << '\n';
   }
   for (int i = 0; i < size; i++)
   {
      out << operator[](i) << '\n';
   }
}

template <class T>
void Array<T>::Load(std::istream &in, int fmt)
{
   if (fmt == 0)
   {
      int new_size;
      in >> new_size;
      SetSize(new_size);
   }
   for (int i = 0; i < size; i++)
   {
      in >> operator[](i);
   }
}

template <class T>
T Array<T>::Max() const
{
   MFEM_ASSERT(size > 0, "Array is empty with size " << size);

   T max = operator[](0);
   for (int i = 1; i < size; i++)
   {
      if (max < operator[](i))
      {
         max = operator[](i);
      }
   }

   return max;
}

template <class T>
T Array<T>::Min() const
{
   MFEM_ASSERT(size > 0, "Array is empty with size " << size);

   T min = operator[](0);
   for (int i = 1; i < size; i++)
   {
      if (operator[](i) < min)
      {
         min = operator[](i);
      }
   }

   return min;
}

// Partial Sum
template <class T>
void Array<T>::PartialSum()
{
   T sum = static_cast<T>(0);
   for (int i = 0; i < size; i++)
   {
      sum+=operator[](i);
      operator[](i) = sum;
   }
}

// Sum
template <class T>
T Array<T>::Sum()
{
   T sum = static_cast<T>(0);
   for (int i = 0; i < size; i++)
   {
      sum+=operator[](i);
   }

   return sum;
}

template <class T>
int Array<T>::IsSorted()
{
   T val_prev = operator[](0), val;
   for (int i = 1; i < size; i++)
   {
      val=operator[](i);
      if (val < val_prev)
      {
         return 0;
      }
      val_prev = val;
   }

   return 1;
}


template <class T>
void Array2D<T>::Load(const char *filename, int fmt)
{
   std::ifstream in;
   in.open(filename, std::ifstream::in);
   MFEM_VERIFY(in.is_open(), "File " << filename << " does not exist.");
   Load(in, fmt);
   in.close();
}

template <class T>
void Array2D<T>::Print(std::ostream &out, int width_)
{
   int height = this->NumRows();
   int width  = this->NumCols();

   for (int i = 0; i < height; i++)
   {
      out << "[row " << i << "]\n";
      for (int j = 0; j < width; j++)
      {
         out << (*this)(i,j);
         if ( (j+1) == width_ || (j+1) % width_ == 0 )
         {
            out << '\n';
         }
         else
         {
            out << ' ';
         }
      }
   }
}

template class Array<int>;
template class Array<double>;
template class Array2D<int>;
template class Array2D<double>;
}
