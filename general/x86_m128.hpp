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
#ifndef MFEM_X86_M128_HPP
#define MFEM_X86_M128_HPP

// ****************************************************************************
// * SSE integer
// ****************************************************************************
struct __attribute__ ((aligned(8))) integer {
protected:
  __m128i vec;
public:
  // Constructors
  inline integer():vec(_mm_set_epi32(0,0,0,0)){}
  inline	integer(__m128i mm):vec(mm){}
  inline integer(int i):vec(_mm_set_epi32(0,0,i,i)){}
  inline integer(int i0, int i1){vec=_mm_set_epi32(0, 0, i1, i0);}
  // Convertors
  inline operator __m128i() const { return vec; }
  // Logical Operations
  inline integer& operator&=(const integer &a) { return *this = (integer) _mm_and_si128(vec,a); }
  inline integer& operator|=(const integer &a) { return *this = (integer) _mm_or_si128(vec,a); }
  inline integer& operator^=(const integer &a) { return *this = (integer) _mm_xor_si128(vec,a); }
  friend inline integer operator<(const integer &a, const integer &b)  { return _mm_cmpeq_epi32(a, b); }
  // Arithmetics
  friend inline integer operator +(const integer &a, const integer &b) { return _mm_add_epi32(a,b); }
  friend inline integer operator -(const integer &a, const integer &b) { return _mm_sub_epi32(a,b); }
  friend inline integer operator *(const integer &a, const integer &b) { return _mm_mul_epi32(a,b); }
  friend inline integer operator /(const integer &a, const integer &b) {
    return _mm_set_epi32(a[0]/b[0],a[1]/b[1],0,0);
  }
  friend inline integer operator %(const integer &a, const integer &b) {
    return _mm_set_epi32(a[0]%b[0],a[1]%b[1],0,0);
  }
  inline integer& operator +=(const integer &a) { return *this = (integer)_mm_add_epi32(vec,a); }
  inline integer& operator -=(const integer &a) { return *this = (integer)_mm_sub_epi32(vec,a); }   
  //friend inline __m128d operator==(const integer &a, const int i);
  inline const int& operator[](int i) const  {
    int *a=(int*)&vec;
    return a[i];
  }
  inline int& operator[](int i) {
    int *a=(int*)&vec;
    return a[i];
  }
};
// Logicals
//inline integer operator&(const integer &a, const integer &b) { return _mm_and_si128(a,b); }
//inline integer operator|(const integer &a, const integer &b) { return _mm_or_si128(a,b); }
//inline integer operator^(const integer &a, const integer &b) { return _mm_xor_si128(a,b); }

// ****************************************************************************
// * SSE real type class
// ****************************************************************************
struct __attribute__ ((aligned(16))) real {
 protected:
  __m128d vec;
 public:
  // Constructors
  inline real(): vec(_mm_setzero_pd()){}
  inline real(int i):vec(_mm_set1_pd((double)i)){}
  inline real(integer i):vec(_mm_set_pd(i[1],i[0])){}
  inline real(long i):vec(_mm_set1_pd((double)i)){}
  inline real(double d):vec(_mm_set1_pd(d)){}
  inline real(__m128d x):vec(x){}
  inline real(double *x):vec(_mm_load_pd(x)){}
  inline real(double d0, double d1):vec(_mm_set_pd(d1,d0)){}
  // Convertors
  inline operator __m128d() const { return vec; }
  // Arithmetics
  friend inline real operator +(const real &a, const real &b) { return _mm_add_pd(a,b); }
  friend inline real operator -(const real &a, const real &b) { return _mm_sub_pd(a,b); }
  friend inline real operator *(const real &a, const real &b) { return _mm_mul_pd(a,b); }
  //CLANG has a built-in candidate
#ifndef __clang_major__
  friend inline real operator /(const real &a, const real &b) { return _mm_div_pd(a,b); }
#endif
  inline real& operator +=(const real &a) { return *this = _mm_add_pd(vec,a); }
  inline real& operator -=(const real &a) { return *this = _mm_sub_pd(vec,a); }
  inline real& operator *=(const real &a) { return *this = _mm_mul_pd(vec,a); }
  inline real& operator /=(const real &a) { return *this = _mm_div_pd(vec,a); }
  // Unary +/- operators
  inline real operator -() const { return _mm_xor_pd (_mm_set1_pd(-0.0), *this); }
  inline real operator +() const { return vec; }
  // Mixed vector-scalar operations
  inline real& operator *=(const double &f) { return *this = _mm_mul_pd(vec,_mm_set1_pd(f)); }
  inline real& operator /=(const double &f) { return *this = _mm_div_pd(vec,_mm_set1_pd(f)); }
  inline real& operator +=(const double &f) { return *this = _mm_add_pd(vec,_mm_set1_pd(f)); }
  inline real& operator +=(double &f) { return *this = _mm_add_pd(vec,_mm_set1_pd(f)); }
  inline real& operator -=(const double &f) { return *this = _mm_sub_pd(vec,_mm_set1_pd(f)); }
  // Friends operators
  friend inline real operator+(const real &a, const double &f) { return _mm_add_pd(a, _mm_set1_pd(f)); }
  friend inline real operator-(const real &a, const double &f) { return _mm_sub_pd(a, _mm_set1_pd(f)); } 
  friend inline real operator*(const real &a, const double &f) { return _mm_mul_pd(a, _mm_set1_pd(f)); } 
  friend inline real operator/(const real &a, const double &f) { return _mm_div_pd(a, _mm_set1_pd(f)); }
  friend inline real operator+(const double &f, const real &a) { return _mm_add_pd(_mm_set1_pd(f),a); }
  friend inline real operator-(const double &f, const real &a) { return _mm_sub_pd(_mm_set1_pd(f),a); } 
  friend inline real operator*(const double &f, const real &a) { return _mm_mul_pd(_mm_set1_pd(f),a); } 
  friend inline real operator/(const double &f, const real &a) { return _mm_div_pd(_mm_set1_pd(f),a); }

  friend inline real sqrt(const real &a) { return _mm_sqrt_pd(a); }
  friend inline real min(const real &r, const real &s){ return _mm_min_pd(r,s);}
  friend inline real max(const real &r, const real &s){ return _mm_max_pd(r,s);}
  friend inline real cube_root(const real &a){return real(::cbrt(a[0]),::cbrt(a[1]));}
  friend inline real norm(const real &u){ return real(::fabs(u[0]),::fabs(u[1]));}
  // Compares: Mask is returned
  friend inline real cmp_eq(const real &a, const real &b)  { return _mm_cmpeq_pd(a, b); }
  friend inline real cmp_lt(const real &a, const real &b)  { return _mm_cmplt_pd(a, b); }
  friend inline real cmp_le(const real &a, const real &b)  { return _mm_cmple_pd(a, b); }
  friend inline real cmp_gt(const real &a, const real &b)  { return _mm_cmpgt_pd(a, b); }
  friend inline real cmp_ge(const real &a, const real &b)  { return _mm_cmpge_pd(a, b); }
  friend inline real cmp_neq(const real &a, const real &b)  { return _mm_cmpneq_pd(a, b); }
  friend inline real cmp_nlt(const real &a, const real &b)  { return _mm_cmpnlt_pd(a, b); }
  friend inline real cmp_nle(const real &a, const real &b)  { return _mm_cmpnle_pd(a, b); }
  friend inline real cmp_ngt(const real &a, const real &b)  { return _mm_cmpngt_pd(a, b); }
  friend inline real cmp_nge(const real &a, const real &b)  { return _mm_cmpnge_pd(a, b); }
  // Comparison operators
  friend inline real operator<(const real &a, const real& b) { return _mm_cmplt_pd(a, b); }
  friend inline real operator<(const real &a, double d) { return _mm_cmplt_pd(a, _mm_set1_pd(d)); }
  friend inline real operator>(const real &a, real& r) { return _mm_cmpgt_pd(a, r); }
  friend inline real operator>(const real &a, const real& r) { return _mm_cmpgt_pd(a, r); }
  friend inline real operator>(const real &a, double d) { return _mm_cmpgt_pd(a, _mm_set1_pd(d)); }
  friend inline real operator>=(const real &a, real& r) { return _mm_cmpge_pd(a, r); }
  friend inline real operator>=(const real &a, double d) { return _mm_cmpge_pd(a, _mm_set1_pd(d)); }
  friend inline real operator<=(const real &a, const real& r) { return _mm_cmple_pd(a, r); }
  friend inline real operator<=(const real &a, double d) { return _mm_cmple_pd(a, _mm_set1_pd(d)); }
  friend inline real operator==(const real &a, const real& r) { return _mm_cmpeq_pd(a, r); }
  friend inline real operator==(const real &a, double d) { return _mm_cmpeq_pd(a, _mm_set1_pd(d)); }
  friend inline real operator!=(const real &a, const real& r) { return _mm_cmpneq_pd(a, r); }
  friend inline real operator!=(const real &a, double d) { return _mm_cmpneq_pd(a, _mm_set1_pd(d)); }
  // [] operators
  inline const double& operator[](int i) const  {
    double *d= (double*)&vec;
    return d[i];
  }
  
  inline double& operator[](int i) {
    double *d = (double*)&vec;
    return d[i];
  }
};

#endif // MFEM_X86_M128_HPP
