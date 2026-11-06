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

#include "functionals.hpp"

namespace mfem
{

QuadraticFunctional::QuadraticFunctional(Operator &A_, const Vector *b_,
                                         real_t c_)
   : Operator(1, A_.Width()), A(A_), b(b_), c(c_), der(*this)
{
   MFEM_VERIFY(A.Height() == A.Width(), "QuadraticFunctional: A is "
               << A.Height() << " x " << A.Width() << ", expected square.");
   MFEM_VERIFY(!b || b->Size() == A.Width(), "QuadraticFunctional: b has "
               "size " << b->Size() << ", expected " << A.Width() << ".");
}

void QuadraticFunctional::Mult(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(x.Size() == width, "QuadraticFunctional: x has size "
               << x.Size() << ", expected " << width << ".");
   tmp.UseDevice(x.UseDevice());
   tmp.SetSize(width);
   A.Mult(x, tmp);                              // 1/2 x^T A x + b^T x
   real_t val = 0.5*(x*tmp) + (b ? (*b)*x : 0.0);
   y.SetSize(1);
   y = reduce(val) + c;
}

Operator &QuadraticFunctional::GetGradient(const Vector &x) const
{
   der.SetPoint(x);
   return der;
}

void QuadraticFunctional::Derivative::SetPoint(const Vector &x)
{
   MFEM_VERIFY(x.Size() == width, "QuadraticFunctional: x has size "
               << x.Size() << ", expected " << width << ".");
   g.UseDevice(x.UseDevice());
   g.SetSize(width);
   f.A.Mult(x, g);
   if (f.b) { g += *f.b; }
}

void QuadraticFunctional::Derivative::Mult(const Vector &v, Vector &y) const
{
   y.SetSize(1);
   y = f.reduce(g*v);
}

void QuadraticFunctional::Derivative::MultTranspose(const Vector &s,
                                                    Vector &dx) const
{
   dx.SetSize(width);
   dx.Set(s.HostRead()[0], g);
}

void LinearFunctional::Mult(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(x.Size() == width, "LinearFunctional: x has size "
               << x.Size() << ", expected " << width << ".");
   y.SetSize(1);
   y = reduce(b*x) + c;
}

void LinearFunctional::Derivative::Mult(const Vector &v, Vector &y) const
{
   y.SetSize(1);
   y = f.reduce(f.b*v);
}

void LinearFunctional::Derivative::MultTranspose(const Vector &s,
                                                 Vector &dx) const
{
   dx.SetSize(width);
   dx.Set(s.HostRead()[0], f.b);
}

} // namespace mfem
