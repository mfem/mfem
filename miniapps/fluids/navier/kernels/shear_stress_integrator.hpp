// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_NAVIER_STRESS_INTEGRATOR_HPP
#define MFEM_NAVIER_STRESS_INTEGRATOR_HPP

#include <mfem.hpp>
#include "kernel_helpers.hpp"

namespace mfem
{
namespace navier
{

// Bilinear form for S
class ShearStressIntegrator : public BilinearFormIntegrator
{
public:
   ShearStressIntegrator(GridFunctionCoefficient &n,
                         const IntegrationRule &ir, const int m = 0);

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;
   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override;

   void AssemblePA(const FiniteElementSpace &fes) override;
   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override { this->AssemblePA(trial_fes); }
   void AddMultPA(const Vector &x, Vector &y) const override;

   template <int d1d, int q1d, int dim = 2> static inline
   void ShearStressIntegratorApply2D(
      const int ne,
      const Array<real_t> &B_,
      const Array<real_t> &G_,
      const Array<real_t> &W_,
      const Vector &Jacobian_,
      const Vector &detJ_,
      const Vector &X_, Vector &Y_,
      const Vector &nu_
   );

   template <int d1d, int q1d, int dim = 3> static inline
   void ShearStressIntegratorApply3D(
      const int ne,
      const Array<real_t> &B_,
      const Array<real_t> &G_,
      const Array<real_t> &W_,
      const Vector &Jacobian_,
      const Vector &detJ_,
      const Vector &X_, Vector &Y_,
      const Vector &nu_
   );

protected:
   int dim;
   int vdim;

#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, pelmat;
#endif

   /// Number of elements in the mesh (rank local)
   int ne;
   const DofToQuad *maps = nullptr;
   const GeometricFactors *geom = nullptr;
   GridFunctionCoefficient &nu;
   // const IntegrationRule &ir;
   // Number of degrees of freedom in 1D
   int d1d;
   /// Number of quadrature points in 1D
   int q1d;
   Vector nu_wrap;
   int mode;
};

} // namespace navier
} // namespace mfem

#endif // MFEM_NAVIER_STRESS_INTEGRATOR_HPP