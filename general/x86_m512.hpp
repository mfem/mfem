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
#ifndef MFEM_X86_M512_HPP
#define MFEM_X86_M512_HPP

// ****************************************************************************
// * AVX512 integer type class
// ****************************************************************************
struct __attribute__ ((aligned(64))) integer {
protected:
  __m512i vec;
public:
  // Constructors
  inline integer(){}
  inline	integer(__m512i mm):vec(mm){}
  inline integer(int i):vec(_mm512_set_epi64(i,i,i,i,i,i,i,i)){}
  inline integer(int i7, int i6, int i5, int i4,
                 int i3, int i2, int i1, int i0){vec=_mm512_set_epi64(i7,i6,i5,i4,i3,i2,i1,i0);}
  // Convertors
  inline operator __m512i() const { return vec; }
  // Logical Operations
  inline integer& operator&=(const integer &a) { return *this = (integer) _mm512_and_epi64(vec,a); }
  inline integer& operator|=(const integer &a) { return *this = (integer) _mm512_or_epi64(vec,a); }
  inline integer& operator^=(const integer &a) { return *this = (integer) _mm512_xor_epi64(vec,a); }
  inline integer& operator+=(const integer &a) { return *this = (integer)_mm512_add_epi64(vec,a); }
  inline integer& operator-=(const integer &a) { return *this = (integer)_mm512_sub_epi64(vec,a); }   
  // Friends operators
  //friend inline __mmask8 operator==(const integer &a, const int i);
  // [] operators
  inline const int& operator[](const int i) const  {
    int *dp = (int*)&vec;
    return *(dp+i);
  }
  inline int& operator[](const int i) {
    int *dp = (int*)&vec;
    return *(dp+i);
  }
};
// Logicals
//inline integer operator&(const integer &a, const integer &b) { return _mm512_and_epi64(a,b); }
//inline integer operator|(const integer &a, const integer &b) { return _mm512_or_epi64(a,b); }
//inline integer operator^(const integer &a, const integer &b) { return _mm512_xor_epi64(a,b); }
//inline __mmask8 operator==(const integer &a, const int i){
//  return _mm512_cmp_epi64_mask(a.vec,_mm512_set_epi64(i,i,i,i,i,i,i,i),_MM_CMPINT_EQ);
//}


// ****************************************************************************
// * AVX512 real type class
// ****************************************************************************
struct __attribute__ ((aligned(64))) real {
 protected:
  __m512d vec;
 public:
  // Constructors
  inline real(){}
  inline real(__m512i d):vec(_mm512_set_pd(d[0],d[1],d[2],d[3],
                                           d[4],d[5],d[6],d[7])){}
  inline real(integer d):vec(_mm512_set_pd(d[0],d[1],d[2],d[3],
                                           d[4],d[5],d[6],d[7])){}
  inline real(int d):vec(_mm512_set1_pd(d)){}
  inline real(double d):vec(_mm512_set1_pd(d)){}
  inline real(__m512d x):vec(x){}
  inline real(double *x):vec(_mm512_load_pd(x)){}
  inline real(double d7, double d6, double d5, double d4,
              double d3, double d2, double d1, double d0):
    vec(_mm512_set_pd(d7,d6,d5,d4,d3,d2,d1,d0)){}
  // Conversion operator
  inline operator __m512d() const { return vec; }
  // Arithmetics
  friend inline real operator +(const real &a, const real &b){ return _mm512_add_pd(a,b); }
  friend inline real operator -(const real &a, const real &b){ return _mm512_sub_pd(a,b); }
  friend inline real operator *(const real &a, const real &b){ return _mm512_mul_pd(a,b); }
#ifndef __clang_major__
  friend inline real operator /(const real &a, const real &b){ return _mm512_div_pd(a,b); }
#endif
  inline real& operator +=(const real &a){ return *this = _mm512_add_pd(vec,a); }
  inline real& operator -=(const real &a){ return *this = _mm512_sub_pd(vec,a); }
  inline real& operator *=(const real &a){ return *this = _mm512_mul_pd(vec,a); }
  inline real& operator /=(const real &a){ return *this = _mm512_div_pd(vec,a); }
  // Unary + or -
  inline real operator -() const { return real(0.0) - vec; }
  inline real operator -()       { return real(0.0) - vec; }
  inline real operator +()       { return vec; }
  // Mixed vector-scalar operations
  inline real& operator *=(const double &f){ return *this = _mm512_mul_pd(vec,_mm512_set1_pd(f)); }
  inline real& operator /=(const double &f){ return *this = _mm512_div_pd(vec,_mm512_set1_pd(f)); }
  inline real& operator +=(const double &f){ return *this = _mm512_add_pd(vec,_mm512_set1_pd(f)); }
  inline real& operator +=(double &f){ return *this = _mm512_add_pd(vec,_mm512_set1_pd(f)); }
  inline real& operator -=(const double &f){ return *this = _mm512_sub_pd(vec,_mm512_set1_pd(f)); }
  // Friends operators
  friend inline real operator +(const real &a, const double &f){ return _mm512_add_pd(a, _mm512_set1_pd(f)); }
  friend inline real operator -(const real &a, const double &f){ return _mm512_sub_pd(a, _mm512_set1_pd(f)); } 
  friend inline real operator *(const real &a, const double &f){ return _mm512_mul_pd(a, _mm512_set1_pd(f)); } 
  friend inline real operator /(const real &a, const double &f){ return _mm512_div_pd(a, _mm512_set1_pd(f)); }
  friend inline real operator +(const double &f, const real &a){ return _mm512_add_pd(_mm512_set1_pd(f),a); }
  friend inline real operator -(const double &f, const real &a){ return _mm512_sub_pd(_mm512_set1_pd(f),a); } 
  friend inline real operator *(const double &f, const real &a){ return _mm512_mul_pd(_mm512_set1_pd(f),a); } 
  friend inline real operator /(const double &f, const real &a){ return _mm512_div_pd(_mm512_set1_pd(f),a); }
  // Comparison operators
  friend inline __mmask8 operator==(const real &a, const real& r){ return _mm512_cmp_pd_mask(a,r,_CMP_EQ_OQ); }
  friend inline __mmask8 operator==(const real &a,      double d){ return _mm512_cmp_pd_mask(a,_mm512_set1_pd(d),_CMP_EQ_OQ); }
  friend inline __mmask8 operator< (const real &a, const real& b){ return _mm512_cmp_pd_mask(a,b,_CMP_LT_OS); }
  friend inline __mmask8 operator< (const real &a,      double d){ return _mm512_cmp_pd_mask(a,_mm512_set1_pd(d),_CMP_LT_OS); }
  friend inline __mmask8 operator<=(const real &a, const real& r){ return _mm512_cmp_pd_mask(a,r,_CMP_LE_OS); }
  friend inline __mmask8 operator<=(const real &a,      double d){ return _mm512_cmp_pd_mask(a,_mm512_set1_pd(d),_CMP_LE_OS); }
  friend inline __mmask8 operator> (const real &a,       real& r){ return _mm512_cmp_pd_mask(a,r,_CMP_NLE_US); }
  friend inline __mmask8 operator> (const real &a, const real& r){ return _mm512_cmp_pd_mask(a,r,_CMP_NLE_US); }
  friend inline __mmask8 operator> (const real &a,      double d){ return _mm512_cmp_pd_mask(a,_mm512_set1_pd(d),_CMP_NLE_US); }
  friend inline __mmask8 operator>=(const real &a,       real& r){ return _mm512_cmp_pd_mask(a,r,_CMP_NLT_US); }
  friend inline __mmask8 operator>=(const real &a,      double d){ return _mm512_cmp_pd_mask(a,_mm512_set1_pd(d),_CMP_NLT_US); }
  friend inline __mmask8 operator!=(const real &a, const real& r){ return _mm512_cmp_pd_mask(a,r,_CMP_NEQ_UQ); }
  friend inline __mmask8 operator!=(const real &a,      double d){ return _mm512_cmp_pd_mask(a,_mm512_set1_pd(d),_CMP_NEQ_UQ); }
  // [] operators
  inline const double& operator[](const int i) const  {
   double *dp = (double*)&vec;
    return *(dp+i);
  }
  inline double& operator[](const int i){
    double *dp = (double*)&vec;
    return *(dp+i);
  }
};

#endif // MFEM_X86_M512_HPP
