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

#ifndef MFEM4_TENSOR_HPP
#define MFEM4_TENSOR_HPP

namespace mfem4
{

/** A lightweight N dimensional array.
 */
template<int DIM, typename T = double>
class Tensor
{
public:
   Tensor(int size0, int size1 = 1, int size2 = 1, int size3 = 1) // FIXME: C++11
      : data(NULL)
   {
      Resize(size0, size1, size2, size3);
   }

   Tensor(T *data, int size0, int size1 = 1, int size2 = 1, int size3 = 1) // FIXME: C++11
   {
      MakeRef(data, size0, size1, size2, size3);
   }

   void Resize(int size0, int size1 = 1, int size2 = 1, int size3 = 1)
   {
   }

   void MakeRef(int size0, int size1 = 1, int size2 = 1, int size3 = 1)
   {
   }

   T* operator()(int i)
   { return data[i]; }

   T* operator()(int i, int j)
   { return data[i + j*stride[1]]; }

   T* operator()(int i, int j, int k)
   { return data[i + j*stride[1] + k*stride[2]]; }

   T* operator()(int i, int j, int k, int l)
   { return data[i + j*stride[1] + k*stride[2] + l*stride[3]]; }

   const T* operator()(int i) const
   { return data[i]; }

   const T* operator()(int i, int j) const
   { return data[i + j*stride[1]]; }

   const T* operator()(int i, int j, int k) const
   { return data[i + j*stride[1] + k*stride[2]]; }

   const T* operator()(int i, int j, int k, int l) const
   { return data[i + j*stride[1] + k*stride[2] + l*stride[3]]; }

   T* GetData() { return data; }

   const T* GetData() const { return data; }

   ~Tensor()
   {
      if (own) { delete [] data; }
   }

protected:
   T* data;
   int size[DIM], stride[DIM];
   bool own;

   void Init(int size0, int size1, int size2, int size3)
   {
      //size
   }
};


class DeviceTensor : public Tensor
{
public:

};


} // namespace mfem4
