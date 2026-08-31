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

#ifndef MFEM_TMOP_HDG
#define MFEM_TMOP_HDG

#include "../tmop.hpp"

namespace mfem
{

class HDG_TMOP_Integrator : public TMOP_Integrator
{
   Coefficient *Q;
   MatrixCoefficient *MQ;
   const GridFunction &u, &uhat;
   real_t td;

   DenseMatrix dshape1, dshape2, elfun1, elfun2;
   Vector shape1, shape2;
public:
   HDG_TMOP_Integrator(Coefficient &Q_, const GridFunction &u_,
                       const GridFunction &uhat_, real_t td_ = 0.5)
      : TMOP_Integrator(NULL, NULL), Q(&Q_), MQ(NULL), u(u_), uhat(uhat_), td(td_) { }

   HDG_TMOP_Integrator(MatrixCoefficient &MQ_, const GridFunction &u_,
                       const GridFunction &uhat_, real_t td_ = 0.5)
      : TMOP_Integrator(NULL, NULL), Q(NULL), MQ(&MQ_), u(u_), uhat(uhat_), td(td_) { }

   void SetTd(real_t td_) { td = td_; }

   real_t GetFaceEnergy(const FiniteElement &el1,
                        const FiniteElement &el2,
                        FaceElementTransformations &Tr,
                        const Vector &elfun) override;

   void AssembleFaceVector(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           const Vector &elfun, Vector &elvect) override;

   void AssembleFaceGrad(const FiniteElement &el1,
                         const FiniteElement &el2,
                         FaceElementTransformations &Tr,
                         const Vector &elfun, DenseMatrix &elmat) override;
};

} // namespace mfem

#endif // MFEM_TMOP_HDG
