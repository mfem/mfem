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

#include "array.hpp"
#include "../base/vector.hpp"
#include "../../linalg/vector.hpp"
#include "../../linalg/densemat.hpp"
#include <cstring>
#ifdef __NVCC__
#include <cuda_runtime.h>
#endif

namespace mfem
{

namespace pa
{

template <Location Device, typename T>
struct VectorType_t;

template <Location Device, typename T>
using VectorType = typename VectorType_t<Device,T>::type;

template <typename T>
class HostVector : public HostArray, public PVector
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
   HostVector(HostLayout& lt)
      : PArray(lt), PVector(lt), HostArray(lt, sizeof(T))
   { }

   T* GetData() { return HostArray::GetTypedData<T>(); }
   const T* GetData() const { return HostArray::GetTypedData<T>(); }
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

template <typename T>
struct VectorType_t<Host,T>{
   typedef HostVector<T> type;
};

template<typename T>
PVector* HostVector<T>::DoVectorClone(bool copy_data, void **buffer,
                                  int buffer_type_id) const
{
   MFEM_ASSERT(buffer_type_id == ScalarId<T>::value, "The buffer has a different type.");
   HostLayout& lt = static_cast<HostLayout&>(*layout);
   HostVector<T> *new_vector = new HostVector<T>(lt);
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
void HostVector<T>::DoDotProduct(const PVector &x, void *result,
                             int result_type_id) const
{
   MFEM_ASSERT(result_type_id == ScalarId<T>::value, "The buffer has a different type.");
   const T* data_v1 = this->GetData();
   const T* data_v2 = x.As<HostVector<T>>().GetData();
   T& result_d = *static_cast<T*>(result);
   result_d = 0;
   for (std::size_t i = 0; i < layout->Size(); ++i)
   {
      result_d += data_v1[i] * data_v2[i];
   }
}

template<typename T>
void HostVector<T>::DoAxpby(const void *a, const PVector &x,
                        const void *b, const PVector &y,
                        int ab_type_id)
{
   MFEM_ASSERT(ab_type_id == ScalarId<T>::value, "The buffer has a different type.");
   const T& va = *static_cast<const T*>(a);
   const T& vb = *static_cast<const T*>(b);
   if (va != 0.0 && vb != 0.0) {
      const T* vx = x.As<HostVector<T>>().GetData();
      const T* vy = y.As<HostVector<T>>().GetData();
      T* typed_data = GetData();
      for (std::size_t i = 0; i < layout->Size(); ++i)
      {
         typed_data[i] = va * vx[i] + vb * vy[i];
      }
   } else if (va == 0.0) {
      const T* vy = y.As<HostVector<T>>().GetData();
      T* typed_data = GetData();
      for (std::size_t i = 0; i < layout->Size(); ++i)
      {
         typed_data[i] = vb * vy[i];
      }
   } else if (vb == 0.0) {
      const T* vx = x.As<HostVector<T>>().GetData();
      T* typed_data = GetData();
      for (std::size_t i = 0; i < layout->Size(); ++i)
      {
         typed_data[i] = va * vx[i];
      }
   }
}

template<typename T>
mfem::Vector HostVector<T>::Wrap()
{
   return mfem::Vector(*this);
}

template<typename T>
const mfem::Vector HostVector<T>::Wrap() const
{
   return mfem::Vector(*const_cast<HostVector<T>*>(this));
}

#ifdef __NVCC__

template <typename T>
class CudaVector: public CudaArray, public PVector
{
protected:
   virtual PVector *DoVectorClone(bool copy_data, void **buffer,
                                  int buffer_type_id) const;

   virtual void DoDotProduct(const PVector &x, void *result,
                             int result_type_id) const;

   virtual void DoAxpby(const void *a, const PVector &x,
                        const void *b, const PVector &y,
                        int ab_type_id);
public:
   CudaVector(CudaLayout& lt)
      : PArray(lt), CudaArray(lt, sizeof(T)), PVector(lt)
   { }

   T* GetData() { return Array::GetTypedData<T>(); }
   const T* GetData() const { return Array::GetTypedData<T>(); }
};

template <typename T>
struct VectorType_t<CudaDevice,T>{
   typedef CudaVector<T> type;
};

template <typename T>
PVector* CudaVector<T>::DoVectorClone(bool copy_data, void **buffer,
                                   int buffer_type_id) const
{
   CudaLayout& lay = dynamic_cast<CudaLayout&>(*layout);
   CudaVector* new_vector = new CudaVector<T>(lay);
   if (copy_data)
   {
      cudaMemcpy(new_vector->data, this->data, lay.Size()*sizeof(T), cudaMemcpyDeviceToDevice);
   }
   if (buffer)
   {
      *buffer = NULL;
   }
   return new_vector;
}

template <typename T>
__global__ T dotProdKernel(const int size, const T* v1, const T* v2)
{
   return 0.0;//TODO
}

template<typename T>
void CudaVector<T>::DoDotProduct(const PVector &x, void *result,
                             int result_type_id) const
{
   MFEM_ASSERT(result_type_id == ScalarId<T>::value, "The buffer has a different type.");
   const T* data_v1 = this->GetData();
   const T* data_v2 = x.As<Vector<T>>().GetData();
   T& result_d = *static_cast<T*>(result);
   const int bsize = 512;
   const int vecsize = layout->Size();
   int gridsize = vecsize/bsize;
   if (bsize*gridsize < vecsize)
        gridsize += 1;
   result_d = dotProdKernel<T><<<gridsize,bsize>>>(vecsize,data_v1,data_v2);
}

template <typename T>
__global__ void axpbyKernel(const T a, const T* x, const T b, const T* y, const int size, T* result)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx>=size)
      return;
   result[idx] = a * x[idx] + b * y[idx];
}

template <typename T>
__global__ void axKernel(const T a, const T* x, const int size, T* result)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx>=size)
      return;
   result[idx] = a * x[idx];
}

template<typename T>
void CudaVector<T>::DoAxpby(const void *a, const PVector &x,
                        const void *b, const PVector &y,
                        int ab_type_id)
{
   MFEM_ASSERT(ab_type_id == ScalarId<T>::value, "The buffer has a different type.");
   //TODO assert for the sizes
   const T& va = *static_cast<const T*>(a);
   const T& vb = *static_cast<const T*>(b);
   const int bsize = 512;
   const int vecsize = layout->Size();
   int gridsize = vecsize/bsize;
   if (bsize*gridsize < vecsize)
        gridsize += 1;
   T* result = GetData();
   if (va != 0.0 && vb != 0.0) {
      const T* vx = x.As<Vector<T>>().GetData();
      const T* vy = y.As<Vector<T>>().GetData();
      axpbyKernel<T><<<gridsize,bsize>>>(a,vx,b,vy,vecsize,result);
   } else if (va == 0.0) {
      const T* vy = y.As<Vector<T>>().GetData();
      axKernel<T><<<gridsize,bsize>>>(b,vy,vecsize,result);
   } else if (vb == 0.0) {
      const T* vx = x.As<Vector<T>>().GetData();
      axKernel<T><<<gridsize,bsize>>>(a,vx,vecsize,result);
   }
}

#endif

} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#endif // MFEM_BACKENDS_PA_VECTOR_HPP
