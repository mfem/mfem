// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_WAVELETS_HPP
#define MFEM_WAVELETS_HPP

#include "linalg.hpp"

namespace mfem
{

////////////////////////////////////////////////////////////////////////////////
struct Wavelet : public Operator
{
   enum Type { HAAR = 0,
               DAUBECHIES,
               CDF53,
               CDF97,
               LEGENDRE2,
               LEGENDRE3
             };

   const Type type;
   const int m, odd;
   const bool lowpass;

   Wavelet(Type type, int s, bool lowpass = false):
      Operator(lowpass?(s+1)>>1:s,s),
      type(type),
      m(lowpass?height:(height+1)>>1),
      odd(width%2 &1),
      lowpass(lowpass) {}

   Wavelet(Type type, int h, int w):
      Operator(h,w),
      type(type),
      m(height),
      odd(width%2 &1),
      lowpass(true) { }

   static Operator* New(const Wavelet::Type&, int n, bool lowpass = false);
   static std::string GetType(const Wavelet::Type&);

   SparseMatrix *GetEvenMatrix(const double *coeffs, const int rowsize);
   SparseMatrix *GetOddMatrix(const double *coeffs, const int rowsize,
                              bool for_transpose);
   SparseMatrix *GetMatrix(const double *coeffs, const int rowsize);
   SparseMatrix *GetTransposedMatrix(const double *coeffs, const int rowsize);

   virtual SparseMatrix *GetMatrix();
   virtual SparseMatrix *GetTransposedMatrix();
   virtual double GetOddRowValue(const int i, const int j, const bool root);
};

////////////////////////////////////////////////////////////////////////////////
struct HaarWavelet : public Wavelet
{
   HaarWavelet(int n, bool lowpass = false);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   SparseMatrix *GetMatrix();
   SparseMatrix *GetTransposedMatrix();
   double GetOddRowValue(const int i, const int j, const bool root);
};

////////////////////////////////////////////////////////////////////////////////
/// \brief The DaubechiesWavelet of order 2
struct DaubechiesWavelet : public Wavelet
{
   DaubechiesWavelet(int n, bool lowpass = false);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   SparseMatrix *GetMatrix();
   SparseMatrix *GetTransposedMatrix();
   double GetOddRowValue(const int i, const int j, const bool root);
};

////////////////////////////////////////////////////////////////////////////////
struct CDF53Wavelet : public Wavelet
{
   CDF53Wavelet(int n);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

////////////////////////////////////////////////////////////////////////////////
class CDF97Wavelet : public Wavelet
{
   static constexpr double alpha = 1.58613434205992355842832;
   static constexpr double  beta = 0.05298011857296141462412;
   static constexpr double gamma = 0.88291107553093329591979;
   static constexpr double delta = 0.44350685204397115211560;
   static constexpr double kappa = 1.14960439886024115979508;
public:
   CDF97Wavelet(int n);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

////////////////////////////////////////////////////////////////////////////////
struct Legendre2Wavelet : public Wavelet
{
   Legendre2Wavelet(int n);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

////////////////////////////////////////////////////////////////////////////////
struct Legendre3Wavelet : public Wavelet
{
   Legendre3Wavelet(int n);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

} // namespace mfem

#endif // MFEM_WAVELETS_HPP
