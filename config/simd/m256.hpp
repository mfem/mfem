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
#ifndef MFEM_X86_M256_HPP
#define MFEM_X86_M256_HPP

// ****************************************************************************
// * AVX integer type class
// ****************************************************************************
struct __attribute__ ((aligned(16))) integer {
protected:
  __attribute__ ((aligned(16))) __m128i vec;
public:
  // Constructors
  inline integer(){}
  inline	integer(__m128i mm):vec(mm){}
  inline integer(int i):vec(_mm_set_epi32(i,i,i,i)){}
  // Convertors
  inline operator __m128i() const { return vec; }
  // Logical Operations
  inline integer& operator&=(const integer &a) { return *this = (integer)_mm_and_si128(vec,a); }
  inline integer& operator|=(const integer &a) { return *this = (integer)_mm_or_si128(vec,a); }
  inline integer& operator^=(const integer &a) { return *this = (integer)_mm_xor_si128(vec,a); }
  inline integer& operator+=(const integer &a) { return *this = (integer)_mm_add_epi32(vec,a); }
  inline integer& operator-=(const integer &a) { return *this = (integer)_mm_sub_epi32(vec,a); }   
  // Friends operators
  //friend inline __m256d operator==(const integer &a, const int i);
  // [] operators
  inline const int& operator[](int i) const {
    const int *a=(int*)&vec;
    return a[i];
  }
  inline int& operator[](int i) {
    int *a=(int*)&vec;
    return a[i];
  }  
};

// ****************************************************************************
// * AVX real type class
// ****************************************************************************
struct __attribute__ ((aligned(32))) real {
 protected:
  __m256d vec;
 public:
  // Constructors
  inline real(){}
  inline real(int i):vec(_mm256_set1_pd((double)i)){}
  inline real(integer i):vec(_mm256_set_pd(i[3],i[2],i[1],i[0])){}
  inline real(long i):vec(_mm256_set1_pd((double)i)){}
  inline real(double d):vec(_mm256_set1_pd(d)){}
  inline real(__m256d x):vec(x){}
  inline real(double *x):vec(_mm256_load_pd(x)){}
  // Convertors
  inline operator __m256d() const { return vec; }
  // Arithmetics
  friend inline real operator +(const real &a, const real &b) { return _mm256_add_pd(a,b); }
  friend inline real operator -(const real &a, const real &b) { return _mm256_sub_pd(a,b); }
  friend inline real operator *(const real &a, const real &b) { return _mm256_mul_pd(a,b); }
  friend inline real operator /(const real &a, const real &b) { return _mm256_div_pd(a,b); }
  // Unary
  inline real operator -() const { return _mm256_xor_pd (_mm256_set1_pd(-0.0), *this); }
  inline real operator +() const { return vec; }
  // Assignment operations
  inline real& operator +=(const real &a) { return *this = _mm256_add_pd(vec,a); }
  inline real& operator -=(const real &a) { return *this = _mm256_sub_pd(vec,a); }
  inline real& operator *=(const real &a) { return *this = _mm256_mul_pd(vec,a); }
  inline real& operator /=(const real &a) { return *this = _mm256_div_pd(vec,a); }
  // Mixed vector-scalar assignment operations
  inline real& operator *=(const double &f) { return *this = _mm256_mul_pd(vec,_mm256_set1_pd(f)); }
  inline real& operator /=(const double &f) { return *this = _mm256_div_pd(vec,_mm256_set1_pd(f)); }
  inline real& operator +=(const double &f) { return *this = _mm256_add_pd(vec,_mm256_set1_pd(f)); }
  inline real& operator +=(double &f) { return *this = _mm256_add_pd(vec,_mm256_set1_pd(f)); }
  inline real& operator -=(const double &f) { return *this = _mm256_sub_pd(vec,_mm256_set1_pd(f)); }
  // Friends operator
  friend inline real operator +(const real &a, const double &f) { return _mm256_add_pd(a, _mm256_set1_pd(f)); }
  friend inline real operator -(const real &a, const double &f) { return _mm256_sub_pd(a, _mm256_set1_pd(f)); } 
  friend inline real operator *(const real &a, const double &f) { return _mm256_mul_pd(a, _mm256_set1_pd(f)); } 
  friend inline real operator /(const real &a, const double &f) { return _mm256_div_pd(a, _mm256_set1_pd(f)); }
  friend inline real operator +(const double &f, const real &a) { return _mm256_add_pd(_mm256_set1_pd(f),a); }
  friend inline real operator -(const double &f, const real &a) { return _mm256_sub_pd(_mm256_set1_pd(f),a); } 
  friend inline real operator *(const double &f, const real &a) { return _mm256_mul_pd(_mm256_set1_pd(f),a); } 
  friend inline real operator /(const double &f, const real &a) { return _mm256_div_pd(_mm256_set1_pd(f),a); }
  friend inline real sqrt(const real &a) { return _mm256_sqrt_pd(a); }
  friend inline real ceil(const real &a) { return _mm256_round_pd((a), _MM_FROUND_CEIL); }
  friend inline real floor(const real &a) { return _mm256_round_pd((a), _MM_FROUND_FLOOR); }
  friend inline real trunc(const real &a) { return _mm256_round_pd((a), _MM_FROUND_TO_ZERO); }
  friend inline real min(const real &r, const real &s){ return _mm256_min_pd(r,s);}
  friend inline real max(const real &r, const real &s){ return _mm256_max_pd(r,s);}
  //friend inline real round(const real &a) { return _mm256_svml_round_pd(a); }
  // Comparison operator
  friend inline real cmp_eq(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_EQ_OS); }
  friend inline real cmp_lt(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_LT_OS); }
  friend inline real cmp_le(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_LE_OS); }
  friend inline real cmp_gt(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_GT_OS); }
  friend inline real cmp_ge(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_GE_OS); }
  friend inline real cmp_neq(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_NEQ_US); }
  friend inline real cmp_nlt(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_NLT_US); }
  friend inline real cmp_nle(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_NLE_US); }
  friend inline real cmp_ngt(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_NGT_US); }
  friend inline real cmp_nge(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_NGE_US); }
  friend inline real operator<(const real &a, const real& b) { return _mm256_cmp_pd(a, b, _CMP_LT_OS); }
  friend inline real operator<(const real &a, double d) { return _mm256_cmp_pd(a, _mm256_set1_pd(d), _CMP_LT_OS); }
  friend inline real operator>(const real &a, real& r) { return _mm256_cmp_pd(a, r, _CMP_GT_OS); }
  friend inline real operator>(const real &a, const real& r) { return _mm256_cmp_pd(a, r, _CMP_GT_OS); }
  friend inline real operator>(const real &a, double d) { return _mm256_cmp_pd(a, _mm256_set1_pd(d), _CMP_GT_OS); }
  friend inline real operator>=(const real &a, real& r) { return _mm256_cmp_pd(a, r, _CMP_GE_OS); }
  friend inline real operator>=(const real &a, double d) { return _mm256_cmp_pd(a, _mm256_set1_pd(d), _CMP_GE_OS); }
  friend inline real operator<=(const real &a, const real& r) { return _mm256_cmp_pd(a, r, _CMP_LE_OS); }
  friend inline real operator<=(const real &a, double d) { return _mm256_cmp_pd(a, _mm256_set1_pd(d), _CMP_LE_OS); }
  friend inline real operator==(const real &a, const real& r) { return _mm256_cmp_pd(a, r, _CMP_EQ_OQ); }
  friend inline real operator==(const real &a, double d) { return _mm256_cmp_pd(a, _mm256_set1_pd(d), _CMP_EQ_OQ); }
  friend inline real operator!=(const real &a, const real& r) { return _mm256_cmp_pd(a, r, _CMP_NEQ_UQ); }
  friend inline real operator!=(const real &a, double d) { return _mm256_cmp_pd(a, _mm256_set1_pd(d), _CMP_NEQ_UQ); }
  // [] operators
  inline const double& operator[](int i) const  {
    const double *d = (double*)&vec;
    return *(d+i);
  }
  inline double& operator[](int i) {
    double *d = (double*)&vec;
    return *(d+i);
  }
};

#endif // MFEM_X86_M256_HPP
