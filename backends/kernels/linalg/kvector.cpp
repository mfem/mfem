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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

kvector::~kvector()
{
   if (!own) { return; }
   dbg("\033[33m[~v");
   kmalloc::operator delete (data);
}

// ***************************************************************************
double* kvector::alloc(const size_t sz)
{
   dbg("\033[33m[v");
   return (double*) kmalloc::operator new (sz);
}

// ***************************************************************************
void kvector::SetSize(const size_t sz, const void* ptr)
{
   own=true;
   size = sz;
   if (!data) { data = alloc(sz); }
   if (ptr) { rDtoD(data,ptr,bytes()); }
}

// ***************************************************************************
kvector::kvector(const size_t sz):
   size(sz),
   data(alloc(sz)),
   own(true) {assert(false);}
   
kvector::kvector(const size_t sz,double value):
   size(sz),data(alloc(sz)),own(true)
{assert(false);
   push(SkyBlue);
   *this=value;
   pop();
}

kvector::kvector(const kvector& v):
   size(0),data(NULL),own(true) { assert(false);SetSize(v.Size(), v); }

kvector::kvector(const kvector *v):size(v->size),
   data(v->data),
   own(false) {assert(false);}

kvector::kvector(kernels::array<double>& v):size(v.size()),
   data(v.ptr()),
   own(false) {assert(false);}

// Host 2 Device ***************************************************************
kvector::kvector(const mfem::Vector& v):size(v.Size()),
   data(alloc(size)),
   own(true)
{
   assert(v.GetData());
   kmemcpy::rHtoD(data,v.GetData(),size*sizeof(double));
}

// Device 2 Host ***************************************************************
kvector::operator mfem::Vector()
{
   if (!config::Get().Cuda()) { return mfem::Vector(data,size); }
   double *h_data= (double*) ::malloc(bytes());
   kmemcpy::rDtoH(h_data,data,bytes());
   mfem::Vector mfem_vector(h_data,size);
   mfem_vector.MakeDataOwner();
   return mfem_vector;
}

kvector::operator mfem::Vector() const
{
   if (!config::Get().Cuda()) { return mfem::Vector(data,size); }
   double *h_data= (double*) ::malloc(bytes());
   kmemcpy::rDtoH(h_data,data,bytes());
   mfem::Vector mfem_vector(h_data,size);
   mfem_vector.MakeDataOwner();
   return mfem_vector;
}

// ***************************************************************************
void kvector::Print(std::ostream& out, int width) const
{
   double *h_data = (double*) ::malloc(bytes());
   kmemcpy::rDtoH(h_data,data,bytes());
   for (size_t i=0; i<size; i+=1)
   {
      //printf("\n\t[%ld] %.15e",i,h_data[i]);
      printf("\n\t[%ld] %f",i,h_data[i]);
   }
   free(h_data);
}

// ***************************************************************************
kvector* kvector::GetRange(const size_t offset,
                                       const size_t entries) const
{
   static kvector ref;
   ref.size = entries;
   ref.data = (double*) ((unsigned char*)data + (offset*sizeof(double)));
   ref.own = false;
   return &ref;
}

// ***************************************************************************
kvector& kvector::operator=(const kvector& v)
{
   SetSize(v.Size(),v.data);
   own = false;
   return *this;
}

// ***************************************************************************
kvector& kvector::operator=(const mfem::Vector& v)
{
   size=v.Size();
   if (!config::Get().Cuda()) { SetSize(size,v.GetData()); }
   else { rHtoD(data,v.GetData(),bytes()); }
   own = false;
   return *this;
}

// ***************************************************************************
kvector& kvector::operator=(double value)
{
   vector_op_eq(size, value, data);
   return *this;
}

// ***************************************************************************
double kvector::operator*(const kvector& v) const
{
   return vector_dot(size, data, v.data);
}

// *****************************************************************************
kvector& kvector::operator-=(const kvector& v)
{
   vector_vec_sub(size, data, v.data);
   return *this;
}

// ***************************************************************************
kvector& kvector::operator+=(const kvector& v)
{
   vector_vec_add(size, data, v.data);
   return *this;
}

// ***************************************************************************
kvector& kvector::operator+=(const mfem::Vector& v)
{
   double *d_v_data;
   assert(v.GetData());
   if (!config::Get().Cuda()) { d_v_data=v.GetData(); }
   else { kmemcpy::rHtoD(d_v_data = alloc(size),v.GetData(),bytes()); }
   vector_vec_add(size, data, d_v_data);
   return *this;
}

// ***************************************************************************
kvector& kvector::operator*=(const double d)
{
   vector_vec_mul(size, data, d);
   return *this;
}

// ***************************************************************************
kvector& kvector::Add(const double alpha, const kvector& v)
{
   vector_axpy(Size(),alpha, data, v.data);
   return *this;
}

// ***************************************************************************
void kvector::Neg()
{
   vector_neg(Size(),ptr());
}

// *****************************************************************************
void kvector::SetSubVector(const kernels::array<int> &ess_tdofs,
                                 const double value,
                                 const int N)
{
   vector_set_subvector_const(N, value, data, ess_tdofs.ptr());
}


// ***************************************************************************
double kvector::Min() const
{
   return vector_min(Size(),(double*)data);
}

// ***************************************************************************
void add(const kvector& v1, const double alpha,
         const kvector& v2, kvector& out)
{
   vector_xpay(out.Size(),alpha,out.ptr(),v1.ptr(),v2.ptr());
}

// *****************************************************************************
void add(const double alpha,
         const kvector& v1,
         const double beta,
         const kvector& v2,
         kvector& out) { assert(false); }

// ***************************************************************************
void subtract(const kvector& v1,
              const kvector& v2,
              kvector& out)
{
   vector_xsy(out.Size(),out.ptr(),v1.ptr(),v2.ptr());
}

} // kernels

} // mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
