// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "complex_coefficient.hpp"

namespace mfem
{

template <bool real_part>
ComplexCoefficient::PartCoefficient<real_part>::PartCoefficient(
   ComplexCoefficient &complex_coef)
   : complex_coef_(complex_coef)
{ }

template <bool real_part>
void ComplexCoefficient::PartCoefficient<real_part>::SetLocalTime(real_t t)
{
   Coefficient::SetTime(t);
}

template <bool real_part>
void ComplexCoefficient::PartCoefficient<real_part>::SetTime(real_t t)
{
   complex_coef_.SetTime(t);
}

template <bool real_part>
real_t ComplexCoefficient::PartCoefficient<real_part>::Eval(
   ElementTransformation &T, const IntegrationPoint &ip)
{
   const complex_t value = complex_coef_.Eval(T, ip);
   return real_part ? value.real() : value.imag();
}

ComplexCoefficient::ComplexCoefficient()
   : time(0.), re_part_coef_(*this), im_part_coef_(*this)
{ }

void ComplexCoefficient::SetTime(real_t t)
{
   time = t;
   re_part_coef_.SetLocalTime(t);
   im_part_coef_.SetLocalTime(t);
}

complex_t ComplexCoefficient::Eval(ElementTransformation &T,
                                   const IntegrationPoint &ip, real_t t)
{
   SetTime(t);
   return Eval(T, ip);
}

ComplexCoefficientFromRealImag::ComplexCoefficientFromRealImag(
   Coefficient &c_r, Coefficient &c_i)
   : real_coef_(c_r), imag_coef_(c_i)
{
   SetTime(c_r.GetTime());
}

void ComplexCoefficientFromRealImag::SetTime(real_t t)
{
   ComplexCoefficient::SetTime(t);
   real_coef_.SetTime(t);
   imag_coef_.SetTime(t);
}

complex_t ComplexCoefficientFromRealImag::Eval(ElementTransformation &T,
                                               const IntegrationPoint &ip)
{
   return complex_t(real_coef_.Eval(T, ip), imag_coef_.Eval(T, ip));
}

template <bool real_part>
ComplexVectorCoefficient::PartCoefficient<real_part>::PartCoefficient(
   ComplexVectorCoefficient &complex_vcoef)
   : VectorCoefficient(complex_vcoef.GetVDim()), complex_vcoef_(complex_vcoef)
#ifndef MFEM_THREAD_SAFE
   , value_(complex_vcoef.GetVDim())
#endif
{ }

template <bool real_part>
void ComplexVectorCoefficient::PartCoefficient<real_part>::SetLocalTime(
   real_t t)
{
   VectorCoefficient::SetTime(t);
}

template <bool real_part>
void ComplexVectorCoefficient::PartCoefficient<real_part>::SetTime(real_t t)
{
   complex_vcoef_.SetTime(t);
}

template <bool real_part>
void ComplexVectorCoefficient::PartCoefficient<real_part>::Eval(
   Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
{
#ifdef MFEM_THREAD_SAFE
   ComplexVector value(vdim);
#else
   ComplexVector &value = value_;
#endif
   complex_vcoef_.Eval(value, T, ip);
   V = real_part ? value.real() : value.imag();
}

ComplexVectorCoefficient::ComplexVectorCoefficient(int vd)
   : vdim(vd), time(0.), re_part_vcoef_(*this), im_part_vcoef_(*this)
{ }

void ComplexVectorCoefficient::SetTime(real_t t)
{
   time = t;
   re_part_vcoef_.SetLocalTime(t);
   im_part_vcoef_.SetLocalTime(t);
}

void ComplexVectorCoefficient::Eval(ComplexVector &V,
                                    ElementTransformation &T,
                                    const IntegrationPoint &ip, real_t t)
{
   SetTime(t);
   Eval(V, T, ip);
}

ComplexVectorCoefficientFromRealImag::ComplexVectorCoefficientFromRealImag(
   VectorCoefficient &v_r, VectorCoefficient &v_i)
   : ComplexVectorCoefficient(v_r.GetVDim()),
     real_vcoef_(v_r), imag_vcoef_(v_i)
#ifndef MFEM_THREAD_SAFE
   , V_r_(v_r.GetVDim()), V_i_(v_i.GetVDim())
#endif
{
   MFEM_VERIFY(v_r.GetVDim() == v_i.GetVDim(),
               "ComplexVectorCoefficientFromRealImag: incompatible "
               "vector dimensions of real and imaginary parts.");
   SetTime(v_r.GetTime());
}

void ComplexVectorCoefficientFromRealImag::SetTime(real_t t)
{
   ComplexVectorCoefficient::SetTime(t);
   real_vcoef_.SetTime(t);
   imag_vcoef_.SetTime(t);
}

void ComplexVectorCoefficientFromRealImag::Eval(
   ComplexVector &V, ElementTransformation &T, const IntegrationPoint &ip)
{
#ifdef MFEM_THREAD_SAFE
   Vector V_r(vdim), V_i(vdim);
#else
   Vector &V_r = V_r_, &V_i = V_i_;
#endif
   real_vcoef_.Eval(V_r, T, ip);
   imag_vcoef_.Eval(V_i, T, ip);
   V.SetSize(vdim);
   V.Set(V_r, V_i);
}

template <bool real_part>
ComplexMatrixCoefficient::PartCoefficient<real_part>::PartCoefficient(
   ComplexMatrixCoefficient &complex_mcoef)
   : MatrixCoefficient(complex_mcoef.GetHeight(), complex_mcoef.GetWidth()),
     complex_mcoef_(complex_mcoef)
#ifndef MFEM_THREAD_SAFE
   , value_(complex_mcoef.GetHeight(), complex_mcoef.GetWidth())
#endif
{ }

template <bool real_part>
void ComplexMatrixCoefficient::PartCoefficient<real_part>::SetLocalTime(
   real_t t)
{
   MatrixCoefficient::SetTime(t);
}

template <bool real_part>
void ComplexMatrixCoefficient::PartCoefficient<real_part>::SetTime(real_t t)
{
   complex_mcoef_.SetTime(t);
}

template <bool real_part>
void ComplexMatrixCoefficient::PartCoefficient<real_part>::Eval(
   DenseMatrix &M, ElementTransformation &T, const IntegrationPoint &ip)
{
#ifdef MFEM_THREAD_SAFE
   ComplexDenseMatrixData value(height, width);
#else
   ComplexDenseMatrixData &value = value_;
#endif
   complex_mcoef_.Eval(value, T, ip);
   M = real_part ? value.real() : value.imag();
}

ComplexMatrixCoefficient::ComplexMatrixCoefficient(int dim)
   : height(dim), width(dim), time(0.),
     re_part_mcoef_(*this), im_part_mcoef_(*this)
{ }

ComplexMatrixCoefficient::ComplexMatrixCoefficient(int h, int w)
   : height(h), width(w), time(0.),
     re_part_mcoef_(*this), im_part_mcoef_(*this)
{ }

void ComplexMatrixCoefficient::SetTime(real_t t)
{
   time = t;
   re_part_mcoef_.SetLocalTime(t);
   im_part_mcoef_.SetLocalTime(t);
}

void ComplexMatrixCoefficient::Eval(ComplexDenseMatrixData &K,
                                    ElementTransformation &T,
                                    const IntegrationPoint &ip, real_t t)
{
   SetTime(t);
   Eval(K, T, ip);
}

ComplexMatrixCoefficientFromRealImag::ComplexMatrixCoefficientFromRealImag(
   MatrixCoefficient &m_r, MatrixCoefficient &m_i)
   : ComplexMatrixCoefficient(m_r.GetHeight(), m_r.GetWidth()),
     real_mcoef_(m_r), imag_mcoef_(m_i)
#ifndef MFEM_THREAD_SAFE
   , M_r_(m_r.GetHeight(), m_r.GetWidth()),
     M_i_(m_i.GetHeight(), m_i.GetWidth())
#endif
{
   MFEM_VERIFY(m_r.GetHeight() == m_i.GetHeight() &&
               m_r.GetWidth() == m_i.GetWidth(),
               "ComplexMatrixCoefficientFromRealImag: incompatible "
               "matrix dimensions of real and imaginary parts.");
   SetTime(m_r.GetTime());
}

void ComplexMatrixCoefficientFromRealImag::SetTime(real_t t)
{
   ComplexMatrixCoefficient::SetTime(t);
   real_mcoef_.SetTime(t);
   imag_mcoef_.SetTime(t);
}

void ComplexMatrixCoefficientFromRealImag::Eval(
   ComplexDenseMatrixData &K, ElementTransformation &T,
   const IntegrationPoint &ip)
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix M_r(height, width), M_i(height, width);
#else
   DenseMatrix &M_r = M_r_, &M_i = M_i_;
#endif
   real_mcoef_.Eval(M_r, T, ip);
   imag_mcoef_.Eval(M_i, T, ip);
   K.SetSize(height, width);
   K.Set(M_r, M_i);
}

ComplexConstantCoefficient::ComplexConstantCoefficient(
   const complex_t z)
   : val(z), real_coef(z.real()), imag_coef(z.imag())
{ }

ComplexConstantCoefficient::ComplexConstantCoefficient(
   real_t z_r, real_t z_i)
   : real_coef(z_r), imag_coef(z_i)
{
   val = complex_t(z_r, z_i);
}

complex_t ComplexFunctionCoefficient::Eval(ElementTransformation & T,
                                           const IntegrationPoint & ip)
{
   real_t x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   if (Function)
   {
      return Function(transip);
   }
   else
   {
      return TDFunction(transip, GetTime());
   }
}

void ComplexVectorFunctionCoefficient::Eval(ComplexVector &V,
                                            ElementTransformation &T,
                                            const IntegrationPoint &ip)
{
   real_t x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   V.SetSize(vdim);
   if (Function)
   {
      Function(transip, V);
   }
   else
   {
      TDFunction(transip, GetTime(), V);
   }
   if (Q)
   {
      V *= Q->Eval(T, ip, GetTime());
   }
}

} // end namespace mfem
