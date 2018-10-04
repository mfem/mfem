// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef MFEM_KARRAY_HPP
#define MFEM_KARRAY_HPP

#include "../config/config.hpp"
#include "array.hpp"
#include "kmalloc.hpp"
#include "okina.hpp"

namespace mfem
{

template <class T, bool xyz = true> class karray;

// Partial Specializations for xyz==TRUE *************************************
template <class T> class karray<T,true> : public kmalloc<T>
{
private:
   T* data = NULL;
   size_t sz=0;
   size_t d[4]= {0};
public:
   karray():data(NULL),sz(0),d{0,0,0,0} {}
   karray(const size_t x) {allocate(x);}
   karray(const size_t x,const size_t y) {allocate(x,y);}
   karray(const karray<T,true> &r)
   {
      allocate(r.d[0], r.d[1], r.d[2], r.d[3]);
      mm::D2D(data,r.GetData(),r.bytes());
   }
   karray& operator=(const karray<T,true> &r)
   {
      allocate(r.d[0], r.d[1], r.d[2], r.d[3]);
      mm::D2D(data,r.GetData(),r.bytes());
      return *this;
   }
   karray& operator=(mfem::Array<T> &a)
   {
      mm::H2D(data,a.GetData(),a.Size()*sizeof(T));
      return *this;
   }
   karray& operator=(const mfem::Array<T> &a)
   {
      mm::H2D(data,a.GetData(),a.Size()*sizeof(T));
      return *this;
   }
   ~karray() {kmalloc<T>::operator delete (data);}
   inline size_t* dim() { return &d[0]; }
   inline T* ptr() { return data; }
   inline const T* GetData() const { return data; }
   inline const T* ptr() const { return data; }
   inline operator T* () { return data; }
   inline operator const T* () const { return data; }
   double operator* (const karray& a) const { return vector_dot(sz, data, a.data); }
   inline size_t size() const { return sz; }
   inline size_t Size() const { return sz; }
   inline size_t bytes() const { return size()*sizeof(T); }
   void allocate(const size_t X, const size_t Y =1,
                 const size_t Z =1, const size_t D =1,
                 const bool transposed = false)
   {
      d[0]=X; d[1]=Y; d[2]=Z; d[3]=D;
      sz=d[0]*d[1]*d[2]*d[3];
      data=(T*) kmalloc<T>::operator new (sz);
   }
   inline bool isInitialized(void)const {return true;}
   inline T& operator[](const size_t x) { return data[x]; }
   inline T& operator()(const size_t x, const size_t y)
   {
      return data[x + d[0]*y];
   }
   inline T& operator()(const size_t x, const size_t y, const size_t z)
   {
      return data[x + d[0]*(y + d[1]*z)];
   }
   void Print(std::ostream& out= std::cout, int width = 8) const
   {
      T *h_data = (double*) ::malloc(bytes());
      mm::D2H(h_data,data,bytes());
      for (size_t i=0; i<sz; i+=1)
         if (sizeof(T)==8) { printf("\n\t[%ld] %.15e",i,h_data[i]); }
         else { printf("\n\t[%ld] %d",i,h_data[i]); }
      free(h_data);
   }
};

// Partial Specializations for xyz==FALSE ************************************
template <class T> class karray<T,false> : public kmalloc<T>
{
private:
   static const int DIM = 4;
   T* data = NULL;
   size_t sz=0;
   size_t d[DIM]= {0};
public:
   karray():data(NULL),sz(0),d{0,0,0,0} {}
   karray(const size_t d0) {allocate(d0);}
   karray(const karray<T,false> &r)
   {
      allocate(r.d[0], r.d[1], r.d[2], r.d[3]);
      mm::D2D(data,r.GetData(),r.bytes());
   }
   karray& operator=(const karray<T,true> &r)
   {
      allocate(r.d[0], r.d[1], r.d[2], r.d[3]);
      mm::D2D(data,r.GetData(),r.bytes());
      return *this;
   }
   karray& operator=(mfem::Array<T> &a)
   {
      mm::H2D(data,a.GetData(),a.Size()*sizeof(T));
      return *this;
   }
   ~karray() {kmalloc<T>::operator delete (data);}
   inline size_t* dim() { return &d[0]; }
   inline T* ptr() { return data; }
   inline T* GetData() const { return data; }
   inline const T* ptr() const { return data; }
   inline operator T* () { return data; }
   inline operator const T* () const { return data; }
   double operator* (const karray& a) const { return vector_dot(sz, data, a.data); }
   inline size_t size() const { return sz; }
   inline size_t Size() const { return sz; }
   inline size_t bytes() const { return size()*sizeof(T); }
   void allocate(const size_t X, const size_t Y =1,
                 const size_t Z =1, const size_t D =1,
                 const bool transposed = false)
   {
      d[0]=X; d[1]=Y; d[2]=Z; d[3]=D;
      sz=d[0]*d[1]*d[2]*d[3];
      assert(sz>0);
      data=(T*) kmalloc<T>::operator new (sz);
      if (transposed) { std::swap(d[0],d[1]); }
      for (size_t i=1,b=d[0]; i<DIM; std::swap(d[i],b),++i)
      {
         d[i]*=d[i-1];
      }
      d[0]=1;
      if (transposed) { std::swap(d[0],d[1]); }
   }
   inline bool isInitialized(void)const {return true;}
   inline T& operator[](const size_t x) { return data[x]; }
   inline T& operator()(const size_t x, const size_t y)
   {
      return data[d[0]*x + d[1]*y];
   }
   inline T& operator()(const size_t x, const size_t y, const size_t z)
   {
      return data[d[0]*x + d[1]*y + d[2]*z];
   }
   void Print(std::ostream& out= std::cout, int width = 8) const
   {
      T *h_data = (double*) ::malloc(bytes());
      mm::D2H(h_data,data,bytes());
      for (size_t i=0; i<sz; i+=1)
         if (sizeof(T)==8) { printf("\n\t[%ld] %.15e",i,h_data[i]); }
         else { printf("\n\t[%ld] %d",i,h_data[i]); }
      free(h_data);
   }
};

} // mfem

#endif // MFEM_KARRAY_HPP
