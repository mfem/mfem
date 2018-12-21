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

#ifndef MFEM_KARRAY_HPP
#define MFEM_KARRAY_HPP

#include "../config/config.hpp"
#include "array.hpp"
#include "okina.hpp"

namespace mfem
{

template <class T, bool xyz = true> class karray;

// Partial Specializations for xyz==TRUE *************************************
template <class T> class karray<T,true>
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
      mm::memcpy(data,r,r.bytes());
   }
   karray& operator=(const karray<T,true> &r)
   {
      allocate(r.d[0], r.d[1], r.d[2], r.d[3]);
      mm::memcpy(data,r,r.bytes());
      return *this;
   }
   ~karray() {mm::free<T>(data);}
   inline size_t* dim() { return &d[0]; }
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
      data=(T*) mm::malloc<T>(sz);
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
   void Print(std::ostream& out= std::cout, int width = sizeof(T)) const
   {
      mm::pull(data,bytes());
      for (size_t i=0; i<sz; i+=1)
      {
         assert(width==4 or width==8);
         if (width==4) { printf("\n\t[%ld] %d",i,data[i]); }
         if (width==8) { printf("\n\t[%ld] %.15e",i,data[i]); }
      }
   }
};

// Partial Specializations for xyz==FALSE ************************************
template <class T> class karray<T,false>
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
      mm::memcpy(data,r.GetData(),r.bytes());
   }
   karray& operator=(const karray<T,true> &r)
   {
      allocate(r.d[0], r.d[1], r.d[2], r.d[3]);
      mm::memcpy(data,r.GetData(),r.bytes());
      return *this;
   }
   ~karray() {mm::free<T> (data);}
   inline size_t* dim() { return &d[0]; }
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
      data=(T*) mm::malloc<T>(sz);
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
   void Print(std::ostream& out= std::cout, int width = sizeof(T)) const
   {
      mm::pull(data,bytes());
      for (size_t i=0; i<sz; i+=1)
      {
         assert(width==4 or width==8);
         if (width==4) { printf("\n\t[%ld] %d",i,data[i]); }
         if (width==8) { printf("\n\t[%ld] %.15e",i,data[i]); }
      }
   }
};

} // mfem

#endif // MFEM_KARRAY_HPP
