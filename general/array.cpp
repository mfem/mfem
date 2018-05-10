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


// Abstract array data type

#include "array.hpp"
#include <fstream>

namespace mfem
{

template <class T>
void Array<T>::Print(std::ostream &out, int width) const
{
   for (int i = 0; i < size; i++)
   {
      out << ((T*)data)[i];
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
      out << this->operator[](i) << '\n';
   }
}

template <class T>
void Array<T>::Load(std::istream &in, int fmt)
{
   if (fmt == 0)
   {
      int new_size;
      in >> new_size;
      this->SetSize(new_size);
   }
   for (int i = 0; i < size; i++)
   {
      in >> this->operator[](i);
   }
}

template <class T>
T Array<T>::Max() const
{
   MFEM_ASSERT(size > 0, "Array is empty with size " << size);

   T max = this->operator[](0);
   for (int i = 1; i < size; i++)
      if (max < this->operator[](i))
      {
         max = this->operator[](i);
      }

   return max;
}

template <class T>
T Array<T>::Min() const
{
   MFEM_ASSERT(size > 0, "Array is empty with size " << size);

   T min = this->operator[](0);
   for (int i = 1; i < size; i++)
      if (this->operator[](i) < min)
      {
         min = this->operator[](i);
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
      sum += this->operator[](i);
      this->operator[](i) = sum;
   }
}

// Sum
template <class T>
T Array<T>::Sum()
{
   T sum = static_cast<T>(0);
   for (int i = 0; i < size; i++)
   {
      sum += this->operator[](i);
   }

   return sum;
}

template <class T>
int Array<T>::IsSorted()
{
   T val_prev = this->operator[](0), val;
   for (int i = 1; i < size; i++)
   {
      val = this->operator[](i);
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
