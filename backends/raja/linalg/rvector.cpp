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
#include "../raja.hpp"

namespace mfem {
  
RajaVector::~RajaVector(){
  if (!own) return;
  dbg("\033[33m[~v");
  rmalloc::operator delete(data);
}

// ***************************************************************************
double* RajaVector::alloc(const size_t sz) {
  dbg("\033[33m[v");
  return (double*) rmalloc::operator new(sz);
}

// ***************************************************************************
  void RajaVector::SetSize(const size_t sz, const void* ptr) {
  own=true;
  size = sz;
  if (!data) data = alloc(sz); 
  if (ptr) rDtoD(data,ptr,bytes());
}

// ***************************************************************************
RajaVector::RajaVector(const size_t sz):size(sz),data(alloc(sz)),own(true) {}
RajaVector::RajaVector(const size_t sz,double value):
  size(sz),data(alloc(sz)),own(true) {
  push(SkyBlue);
  *this=value;
  pop();
}

RajaVector::RajaVector(const RajaVector& v):
  size(0),data(NULL),own(true) { SetSize(v.Size(), v); }

RajaVector::RajaVector(const RajaVector *v):size(v->size),data(v->data),own(false){}
  
RajaVector::RajaVector(RajaArray<double>& v):size(v.size()),data(v.ptr()),own(false){}

// Host 2 Device ***************************************************************
RajaVector::RajaVector(const Vector& v):size(v.Size()),data(alloc(size)),own(true){
  assert(v.GetData());
  rmemcpy::rHtoD(data,v.GetData(),size*sizeof(double));
}

// Device 2 Host ***************************************************************
RajaVector::operator Vector() {
  if (!rconfig::Get().Cuda()) return Vector(data,size);
  double *h_data= (double*) ::malloc(bytes());
  rmemcpy::rDtoH(h_data,data,bytes());
  Vector mfem_vector(h_data,size);
  mfem_vector.MakeDataOwner();
  return mfem_vector;
}

RajaVector::operator Vector() const {
  if (!rconfig::Get().Cuda()) return Vector(data,size);
  double *h_data= (double*) ::malloc(bytes());
  rmemcpy::rDtoH(h_data,data,bytes());
  Vector mfem_vector(h_data,size);
  mfem_vector.MakeDataOwner();
  return mfem_vector;
}

// ***************************************************************************
void RajaVector::Print(std::ostream& out, int width) const {
  double *h_data = (double*) ::malloc(bytes());
  rmemcpy::rDtoH(h_data,data,bytes());
  for (size_t i=0; i<size; i+=1) 
    printf("\n\t[%ld] %.15e",i,h_data[i]);
  free(h_data);
}
  
// ***************************************************************************
RajaVector* RajaVector::GetRange(const size_t offset,
                                 const size_t entries) const {
  static RajaVector ref;
  ref.size = entries;
  ref.data = (double*) ((unsigned char*)data + (offset*sizeof(double)));
  ref.own = false;
  return &ref;
}

// ***************************************************************************
RajaVector& RajaVector::operator=(const RajaVector& v) {
  SetSize(v.Size(),v.data);
  own = false;
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::operator=(const Vector& v) {
  size=v.Size();
  if (!rconfig::Get().Cuda()) SetSize(size,v.GetData());
  else rHtoD(data,v.GetData(),bytes());
  own = false;
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::operator=(double value) {
  vector_op_eq(size, value, data);
  return *this;
}

// ***************************************************************************
double RajaVector::operator*(const RajaVector& v) const {
  return vector_dot(size, data, v.data);
}

// *****************************************************************************
RajaVector& RajaVector::operator-=(const RajaVector& v) {
  vector_vec_sub(size, data, v.data);
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::operator+=(const RajaVector& v) {
  vector_vec_add(size, data, v.data);
  return *this;
}
  
// ***************************************************************************
RajaVector& RajaVector::operator+=(const Vector& v) {
  double *d_v_data;
  assert(v.GetData());
  if (!rconfig::Get().Cuda()) d_v_data=v.GetData();
  else rmemcpy::rHtoD(d_v_data = alloc(size),v.GetData(),bytes());
  vector_vec_add(size, data, d_v_data);  
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::operator*=(const double d) {
  vector_vec_mul(size, data, d);
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::Add(const double alpha, const RajaVector& v) {
  vector_axpy(Size(),alpha, data, v.data);
  return *this;
}

// ***************************************************************************
void RajaVector::Neg() {
  vector_neg(Size(),ptr());
}

// *****************************************************************************
void RajaVector::SetSubVector(const RajaArray<int> &ess_tdofs,
                              const double value,
                              const int N) {
  vector_set_subvector_const(N, value, data, ess_tdofs.ptr());
}


// ***************************************************************************
double RajaVector::Min() const {
  return vector_min(Size(),(double*)data);
}

// ***************************************************************************
void add(const RajaVector& v1, const double alpha,
         const RajaVector& v2, RajaVector& out) {
  vector_xpay(out.Size(),alpha,out.ptr(),v1.ptr(),v2.ptr());
}

// *****************************************************************************
void add(const double alpha,
         const RajaVector& v1,
         const double beta,
         const RajaVector& v2,
         RajaVector& out) { assert(false); }

// ***************************************************************************
void subtract(const RajaVector& v1,
              const RajaVector& v2,
              RajaVector& out) {
  vector_xsy(out.Size(),out.ptr(),v1.ptr(),v2.ptr());
}

} // mfem
