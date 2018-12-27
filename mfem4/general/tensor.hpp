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

#include "buffer.hpp"

namespace mfem4
{

/** A lightweight N dimensional array.
 */
template<int DIM, typename T = double>
class Tensor
{
public:
   Tensor() : data(NULL), buffer()
   {
      for (int i = 0; i < DIM; i++) { size[i] = stride[i] = 0; }
   }

   Tensor(int size0, int size1 = 1, int size2 = 1, int size3 = 1) // FIXME: C++11
      : data(NULL), buffer()
   {
      Resize(size0, size1, size2, size3);
   }

   /*Tensor(T *data, int size0, int size1 = 1, int size2 = 1, int size3 = 1) // FIXME: C++11
   {
      MakeRef(data, size0, size1, size2, size3);
   }*/

   void Resize(int size0, int size1 = 1, int size2 = 1, int size3 = 1)
   {
   }

   /*void MakeRef(int size0, int size1 = 1, int size2 = 1, int size3 = 1)
   {
   }*/

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

   int Size(int i) const { return size[i]; }

protected:
   T *data;
   int size[DIM], stride[DIM];
   Buffer buffer;

   void Init(int size0, int size1, int size2, int size3)
   {
      //size
   }
};


/** A counterpart to Tensor<DIM,T>, to be used on the device. Note that the
 *  two classes are nearly identical, only the 'data' and 'mirror' pointers
 *  are swapped. A transfer to the GPU occurs if tensor.mirror is NULL.
 */
template<int DIM, typename T = double>
class DeviceTensor : public Tensor
{
public:
   DeviceTensor(const Tensor<DIM, T> &tensor)
      : buffer(tensor.buffer)
   {
      std::memcpy(&tensor, this, sizeof(*this));
      /*if (gpu)
      {
         // TODO: allocate and copy to GPU mirror
         std::swap(data, mirror);
      }*/
   }


};


} // namespace mfem4

#endif // MFEM4_TENSOR_HPP
