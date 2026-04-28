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

#ifndef MFEM_UMANSKY_HPP
#define MFEM_UMANSKY_HPP

#include "mfem.hpp"

using namespace mfem;

class UmanskyBoundaryCoefficient : public Coefficient
{
public:
   UmanskyBoundaryCoefficient(real_t w, real_t h) : w_(w), h_(h) {}

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      real_t x[2];
      Vector transip(x, 2);

      T.Transform(ip, transip);

      if (fabs(w_ * x[1] - h_ * x[0]) < 1e-6 * sqrt(w_ * h_))
      {
         return 0.5;
      }
      else if (w_ * x[1] > h_ * x[0])
      {
         return 1.0;
      }
      else
      {
         return 0.0;
      }
   }

private:
   real_t w_, h_;
};

class AnisotropicDiffusionCoefficient : public MatrixCoefficient
{
public:
   AnisotropicDiffusionCoefficient(real_t w, real_t h, real_t Ak)
      : MatrixCoefficient(2), w_(w), h_(h), Ak_(Ak)
   { d2_ = w_ * w_ + h_ * h_; }

   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      M.SetSize(2);

      M(0,0) = 1.0 + (Ak_ - 1.0) * w_ * w_ / d2_;
      M(0,1) = (Ak_ - 1.0) * w_ * h_ / d2_;
      M(1,0) = M(0,1);
      M(1,1) = 1.0 + (Ak_ - 1.0) * h_ * h_ / d2_;
   }

private:
   real_t w_, h_, Ak_, d2_;
};

#endif // MFEM_UMANSKY_SOLVER
