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

#ifndef MFEM_BACKENDS_PA_VECTOR_HPP
#define MFEM_BACKENDS_PA_VECTOR_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "../base/vector.hpp"
#include "array.hpp"
#include "../../linalg/vector.hpp"
#include "../../linalg/densemat.hpp"
#include <cstring>

namespace mfem
{

namespace pa
{

template <typename T>
class Vector : public Array, public PVector
{
protected:
   //
   // Inherited fields
   //
   // DLayout layout;

   /**
       @name Virtual interface
    */
   ///@{

   virtual PVector *DoVectorClone(bool copy_data, void **buffer,
                                  int buffer_type_id) const;

   virtual void DoDotProduct(const PVector &x, void *result,
                             int result_type_id) const;

   virtual void DoAxpby(const void *a, const PVector &x,
                        const void *b, const PVector &y,
                        int ab_type_id);

   ///@}
   // End: Virtual interface

public:
   Vector(Layout &lt)
      : PArray(lt), Array(lt, sizeof(T)), PVector(lt)
   { }

   T* GetData() { return Array::GetTypedData<T>(); }
   const T* GetData() const { return Array::GetTypedData<T>(); }
   const mfem::Vector GetVectorView(const int offset, const int size) const
   {
      return mfem::Vector(static_cast<T*>(data) + offset, size);
   }
   const mfem::DenseMatrix GetMatrixView(const int offset, const int height, const int width) const
   {
      return mfem::DenseMatrix(static_cast<T*>(data) + offset, height, width);
   }
   const mfem::DenseTensor GetTensorView(const int offset, const int i, const int j, const int k) const
   {
      return mfem::DenseTensor(static_cast<T*>(data) + offset, i, j, k);
      // return mfem::DenseTensor().UseExternalData(data+offset,i,j,k);
   }

   mfem::Vector Wrap();

   const mfem::Vector Wrap() const;

};

template<typename T>
PVector* Vector<T>::DoVectorClone(bool copy_data, void **buffer,
                                  int buffer_type_id) const
{
   MFEM_ASSERT(buffer_type_id == ScalarId<T>::value, "The buffer has a different type.");
   Layout& lt = static_cast<Layout&>(*layout);
   Vector<T> *new_vector = new Vector<T>(lt);
   if (copy_data)
   {
      memcpy(new_vector->GetData(), data, layout->Size()*sizeof(T));
   }
   if (buffer)
   {
      *buffer = new_vector->GetData();
   }
   return new_vector;
}

template<typename T>
void Vector<T>::DoDotProduct(const PVector &x, void *result,
                             int result_type_id) const
{
   MFEM_ASSERT(result_type_id == ScalarId<T>::value, "The buffer has a different type.");
   const T* data_v1 = this->GetData();
   const T* data_v2 = x.As<Vector<T>>().GetData();
   T& result_d = *static_cast<T*>(result);
   result_d = 0;
   for (int i = 0; i < layout->Size(); ++i)
   {
      result_d += data_v1[i] * data_v2[i];
   }
}

template<typename T>
void Vector<T>::DoAxpby(const void *a, const PVector &x,
                        const void *b, const PVector &y,
                        int ab_type_id)
{
   MFEM_ASSERT(ab_type_id == ScalarId<T>::value, "The buffer has a different type.");
   const T& va = *static_cast<const T*>(a);
   const T& vb = *static_cast<const T*>(b);
   const T* vx = x.As<Vector<T>>().GetData();
   const T* vy = y.As<Vector<T>>().GetData();
   T* typed_data = GetData();
   for (int i = 0; i < layout->Size(); ++i)
   {
      typed_data[i] = va * vx[i] + vb * vy[i];
   }
}

template<typename T>
mfem::Vector Vector<T>::Wrap()
{
   return mfem::Vector(*this);
}

template<typename T>
const mfem::Vector Vector<T>::Wrap() const
{
   return mfem::Vector(*const_cast<Vector<T>*>(this));
}

} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#endif // MFEM_BACKENDS_PA_VECTOR_HPP
