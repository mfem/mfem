// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "complex_fem.hpp"
#include "../general/forall.hpp"

using namespace std;

namespace mfem
{

real_t
RealPartCoefficient::Eval(ElementTransformation &T,
                          const IntegrationPoint &ip)
{
   complex_t val = complex_coef_.Eval(T, ip);
   return val.real();
}

real_t
ImagPartCoefficient::Eval(ElementTransformation &T,
                          const IntegrationPoint &ip)
{
   complex_t val = complex_coef_.Eval(T, ip);
   return val.imag();
}

RealPartVectorCoefficient::RealPartVectorCoefficient(ComplexVectorCoefficient &
                                                     complex_vcoef)
   : VectorCoefficient(complex_vcoef.GetVDim()),
     complex_vcoef_(complex_vcoef),
     val_(vdim)
{}

void
RealPartVectorCoefficient::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   complex_vcoef_.Eval(val_, T, ip);
   V = val_.real();
}

ImagPartVectorCoefficient::ImagPartVectorCoefficient(ComplexVectorCoefficient &
                                                     complex_vcoef)
   : VectorCoefficient(complex_vcoef.GetVDim()),
     complex_vcoef_(complex_vcoef),
     val_(vdim)
{}

void
ImagPartVectorCoefficient::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   complex_vcoef_.Eval(val_, T, ip);
   V = val_.imag();
}

RealPartMatrixCoefficient::RealPartMatrixCoefficient(ComplexMatrixCoefficient &
                                                     complex_mcoef)
   : MatrixCoefficient(complex_mcoef.GetHeight(), complex_mcoef.GetWidth()),
     complex_mcoef_(complex_mcoef),
     val_(height, width)
{}

void
RealPartMatrixCoefficient::Eval(DenseMatrix &M, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   complex_mcoef_.Eval(val_, T, ip);
   M = val_.real();
}

ImagPartMatrixCoefficient::ImagPartMatrixCoefficient(ComplexMatrixCoefficient &
                                                     complex_mcoef)
   : MatrixCoefficient(complex_mcoef.GetHeight(), complex_mcoef.GetWidth()),
     complex_mcoef_(complex_mcoef),
     val_(height, width)
{}

void
ImagPartMatrixCoefficient::Eval(DenseMatrix &M, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   complex_mcoef_.Eval(val_, T, ip);
   M = val_.imag();
}

ComplexCoefficient::ComplexCoefficient()
   : time(0.),
     re_part_coef_(*this), im_part_coef_(*this),
     real_coef_(re_part_coef_), imag_coef_(im_part_coef_)
{ }

ComplexCoefficient::ComplexCoefficient(Coefficient &c_r,
                                       Coefficient &c_i)
   : time(c_r.GetTime()),
     re_part_coef_(*this), im_part_coef_(*this),
     real_coef_(c_r), imag_coef_(c_i)
{
   c_i.SetTime(time);
}

complex_t
ComplexCoefficient::Eval(ElementTransformation &T,
                         const IntegrationPoint &ip)
{
   // Avoid circular dependency
   MFEM_VERIFY(std::addressof(real_coef_) != std::addressof(re_part_coef_) &&
               std::addressof(imag_coef_) != std::addressof(im_part_coef_),
               "Classes dervied from ComplexCoefficient must either "
               "implement an Eval method or supply Coefficients "
               "for both the real and imaginary parts of the field.");

   return complex_t(real_coef_.Eval(T, ip), imag_coef_.Eval(T, ip));
}

ComplexVectorCoefficient::ComplexVectorCoefficient(VectorCoefficient &v_r,
                                                   VectorCoefficient &v_i)
   : vdim(v_r.GetVDim()), time(v_r.GetTime()),
     re_part_vcoef_(*this), im_part_vcoef_(*this),
     real_vcoef_(v_r), imag_vcoef_(v_i)
{
   MFEM_ASSERT(v_r.GetVDim() == v_i.GetVDim(), "ComplexVectorCoefficient"
               " - incompatible vector dimensions of real and imaginary parts.");

   v_i.SetTime(time);
}

void ComplexVectorCoefficient::Eval(ComplexVector &V, ElementTransformation &T,
                                    const IntegrationPoint &ip)
{
   // Avoid circular dependency
   MFEM_VERIFY(std::addressof(real_vcoef_) != std::addressof(re_part_vcoef_) &&
               std::addressof(imag_vcoef_) != std::addressof(im_part_vcoef_),
               "Classes dervied from ComplexVectorCoefficient must either "
               "implement an Eval method or supply VectorCoefficients "
               "for both the real and imaginary parts of the field.");

   V_r_.SetSize(vdim);
   V_i_.SetSize(vdim);

   real_vcoef_.Eval(V_r_, T, ip);
   imag_vcoef_.Eval(V_i_, T, ip);

   V.Set(V_r_, V_i_);
}

ComplexConstantCoefficient::ComplexConstantCoefficient(
   const complex_t z)
   : val(z), real_coef(z.real()), imag_coef(z.imag())
{
   real_coef_ = real_coef;
   imag_coef_ = imag_coef;
}

ComplexConstantCoefficient::ComplexConstantCoefficient(
   real_t z_r, real_t z_i)
   : real_coef(z_r), imag_coef(z_i)
{
   val = complex_t(z_r, z_i);

   real_coef_ = real_coef;
   imag_coef_ = imag_coef;
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

