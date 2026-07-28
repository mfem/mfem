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

#include "../../general/forall.hpp"
#include "../../fem/kernels.hpp"
#include "../fem.hpp"

#include "lininteg_domain_kernels.hpp"

namespace mfem
{

VectorFEDomainLFIntegrator::Kernels::Kernels()
{
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 2, 1, 1>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 2, 2, 2>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 2, 3, 3>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 2, 4, 4>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 2, 5, 5>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 2, 6, 6>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 2, 7, 7>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 2, 8, 8>();

   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 3, 1, 1>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 3, 2, 2>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 3, 3, 3>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 3, 4, 4>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 3, 5, 5>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 3, 6, 6>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 3, 7, 7>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::DIV, 3, 8, 8>();

   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 1, 1>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 2, 2>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 3, 3>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 4, 4>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 5, 5>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 6, 6>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 7, 7>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 8, 8>();

   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 1, 2>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 2, 3>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 3, 4>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 4, 5>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 5, 6>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 6, 7>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 7, 8>();
   VectorFEDomainLFIntegrator::AddSpecialization<FiniteElement::CURL, 3, 8, 9>();
}

/// \cond DO_NOT_DOCUMENT
VectorFEDomainLFIntegrator::AssembleKernelType
VectorFEDomainLFIntegrator::AssembleKernels::Fallback(
   FiniteElement::DerivType TestType, int DIM, int, int)
{
   if (TestType == FiniteElement::DIV)
   {
      if (DIM == 2)
      {
         return HdivDLFAssemble2D<0, 0>;
      }
      if (DIM == 3)
      {
         return HdivDLFAssemble3D<0, 0>;
      }
   }
   else if (TestType == FiniteElement::CURL)
   {
      if (DIM == 3)
      {
         return HcurlDLFAssemble3D<0, 0>;
      }
   }
   MFEM_ABORT("");
}
/// \endcond DO_NOT_DOCUMENT

void VectorFEDomainLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                                const Array<int> &markers,
                                                Vector &b)
{
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int qorder = 2 * fe.GetOrder();
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);

   QuadratureSpace qs(*fes.GetMesh(), *ir);
   CoefficientVector coeff(QF, qs, CoefficientStorage::COMPRESSED);

   const FiniteElement::DerivType fe_type =
      static_cast<FiniteElement::DerivType>(fe.GetDerivType());

   Mesh &mesh = *fes.GetMesh();
   const int dim = mesh.Dimension();
   const FiniteElement *el = fes.GetTypicalFE();
   const auto *vel = dynamic_cast<const VectorTensorFiniteElement *>(el);
   MFEM_VERIFY(vel != nullptr, "Must be VectorTensorFiniteElement");
   const MemoryType mt = Device::GetDeviceMemoryType();
   const DofToQuad &maps_o = vel->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   const DofToQuad &maps_c = vel->GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int d = maps_c.ndof, q = maps_c.nqpt;
   constexpr int flags = GeometricFactors::JACOBIANS;
   const GeometricFactors *geom = mesh.GetGeometricFactors(*ir, flags, mt);

   AssembleKernels::Run(fe_type, dim, d, q, mesh.GetNE(), markers, geom->J,
                        ir->GetWeights(), maps_o.B, maps_c.B, coeff, b, d, q);
}

} // namespace mfem
