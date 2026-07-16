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

#include "../../fem/kernels.hpp"
#include "../../general/forall.hpp"
#include "../fem.hpp"

#include "bilininteg_pa_simplices_mma.hpp"
#include "lininteg_domain_kernels.hpp"
#include "lininteg_domain_simplices_mma.hpp"

/// \cond DO_NOT_DOCUMENT

namespace mfem
{

static void DLFEvalAssembleSimplexMma(const FiniteElementSpace &fes,
                                      const IntegrationRule *ir,
                                      const Array<int> &markers,
                                      const Vector &coeff,
                                      Vector &y)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   MFEM_VERIFY(CanUseSimplexMmaPA(fes), "");

   const FiniteElement &el = *fes.GetTypicalFE();
   const MemoryType mt = Device::GetDeviceMemoryType();
   const int map_type = el.GetMapType();
   const int p = el.GetOrder();
   const int dofs1D = p + 1;
   const int ndof = el.GetDof();
   const int nq = ir->GetNPoints();
   const int ne = mesh->GetNE();
   const int vdim = fes.GetVDim();

   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::FULL);
   MFEM_VERIFY(maps.ndof == ndof && maps.nqpt == nq, "");
   const Array<real_t> &P = maps.B;

   Vector nodes_e;
   int nd_n = 0, sdim = 0;
   internal::GetSimplexMeshNodesE(*mesh, mt, nodes_e, nd_n, sdim);
   MFEM_VERIFY(sdim == dim, "");
   const FiniteElement &nfe = *mesh->GetNodes()->FESpace()->GetTypicalFE();
   const DofToQuad &nmaps = nfe.GetDofToQuad(*ir, DofToQuad::FULL);
   MFEM_VERIFY(nmaps.ndof == nd_n && nmaps.nqpt == nq, "");

   // H1 GLL simplices (CanUseSimplexMmaPA) use VALUE map type: D = w*c*detJ.
   const bool by_val = true;
   MFEM_VERIFY(map_type == FiniteElement::VALUE,
               "Simplex MMA DomainLF requires VALUE map type");

   Vector D(nq * ne, mt);
   D.UseDevice(true);

   const int coeff_vdim = (coeff.Size() == vdim ||
                           coeff.Size() == vdim * nq * ne) ? vdim : 1;
   const bool coeff_const = (coeff.Size() == coeff_vdim);

   auto zero_unmarked = [&](Vector &dvec)
   {
      const auto M = markers.Read();
      auto Dv = Reshape(dvec.ReadWrite(), nq, ne);
      mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
      {
         if (M[e] != 0) { return; }
         for (int q = 0; q < nq; ++q) { Dv(q, e) = 0.0; }
      });
   };

   real_t *Y = y.ReadWrite();

   for (int vc = 0; vc < vdim; ++vc)
   {
      const int cc = (coeff_vdim == 1) ? 0 : vc;
      if (coeff_const)
      {
         Vector c1(1);
         c1.HostWrite()[0] = coeff.HostRead()[cc];
         c1.UseDevice(true);
         internal::PAMassSetupSimplexMmaFromNodes(
            dim, ne, nq, nd_n, by_val, ir->GetWeights(), nmaps.G, nodes_e, c1,
            D);
      }
      else if (coeff_vdim == 1)
      {
         internal::PAMassSetupSimplexMmaFromNodes(
            dim, ne, nq, nd_n, by_val, ir->GetWeights(), nmaps.G, nodes_e,
            coeff, D);
      }
      else
      {
         // coeff layout matches tensor DomainLF: (vdim, nq, ne)
         Vector c_e(nq * ne, mt);
         c_e.UseDevice(true);
         const auto C = Reshape(coeff.Read(), vdim, nq, ne);
         auto Ce = Reshape(c_e.Write(), nq, ne);
         mfem::forall(nq * ne, [=] MFEM_HOST_DEVICE (int idx)
         {
            const int e = idx / nq;
            const int q = idx - nq * e;
            Ce(q, e) = C(cc, q, e);
         });
         internal::PAMassSetupSimplexMmaFromNodes(
            dim, ne, nq, nd_n, by_val, ir->GetWeights(), nmaps.G, nodes_e, c_e,
            D);
      }

      zero_unmarked(D);
      DomainLFIntegrator::AssembleSimplexMmaKernels::Run(
         dim, dofs1D, nq, ne, P, D, Y, vdim, vc, dofs1D, nq);
   }
}

static void DLFEvalAssemble(const FiniteElementSpace &fes,
                            const IntegrationRule *ir,
                            const Array<int> &markers, const Vector &coeff,
                            Vector &y)
{
   if (CanUseSimplexMmaPA(fes))
   {
      DLFEvalAssembleSimplexMma(fes, ir, markers, coeff, y);
      return;
   }

   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const FiniteElement &el = *fes.GetTypicalFE();
   const MemoryType mt = Device::GetDeviceMemoryType();
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int d = maps.ndof, q = maps.nqpt;
   constexpr int flags = GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);
   const int map_type = fes.GetTypicalFE()->GetMapType();

   const int vdim = fes.GetVDim();
   const int ne = fes.GetMesh()->GetNE();
   const real_t *B = maps.B.Read();
   const int *M = markers.Read();
   const real_t *detJ = geom->detJ.Read();
   const real_t *W = ir->GetWeights().Read();
   real_t *Y = y.ReadWrite();
   DomainLFIntegrator::AssembleKernels::Run(dim, d, q, vdim, ne, d, q, map_type,
                                            M, B, detJ, W, coeff, Y);
}

void DomainLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                        const Array<int> &markers, Vector &b)
{
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int qorder = oa * fe.GetOrder() + ob;
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);

   QuadratureSpace qs(*fes.GetMesh(), *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   DLFEvalAssemble(fes, ir, markers, coeff, b);
}

void VectorDomainLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                              const Array<int> &markers,
                                              Vector &b)
{
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int qorder = 2 * fe.GetOrder();
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);

   QuadratureSpace qs(*fes.GetMesh(), *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   DLFEvalAssemble(fes, ir, markers, coeff, b);
}

DomainLFIntegrator::AssembleKernelType
DomainLFIntegrator::AssembleKernels::Fallback(int DIM, int, int)
{
   switch (DIM)
   {
      case 1:
         return DLFEvalAssemble1D<0, 0>;
      case 2:
         return DLFEvalAssemble2D<0, 0>;
      case 3:
         return DLFEvalAssemble3D<0, 0>;
   }
   MFEM_ABORT("");
}

DomainLFIntegrator::Kernels::Kernels()
{
   // 2D
   // Q = P+1
   DomainLFIntegrator::AddSpecialization<2, 1, 1>();
   DomainLFIntegrator::AddSpecialization<2, 2, 2>();
   DomainLFIntegrator::AddSpecialization<2, 3, 3>();
   DomainLFIntegrator::AddSpecialization<2, 4, 4>();
   DomainLFIntegrator::AddSpecialization<2, 5, 5>();
   // Q = P+2
   DomainLFIntegrator::AddSpecialization<2, 2, 3>();
   DomainLFIntegrator::AddSpecialization<2, 3, 4>();
   DomainLFIntegrator::AddSpecialization<2, 4, 5>();
   DomainLFIntegrator::AddSpecialization<2, 5, 6>();
   // 3D
   // Q = P+1
   DomainLFIntegrator::AddSpecialization<3, 1, 1>();
   DomainLFIntegrator::AddSpecialization<3, 2, 2>();
   DomainLFIntegrator::AddSpecialization<3, 3, 3>();
   DomainLFIntegrator::AddSpecialization<3, 4, 4>();
   DomainLFIntegrator::AddSpecialization<3, 5, 5>();
   // Q = P+2
   DomainLFIntegrator::AddSpecialization<3, 2, 3>();
   DomainLFIntegrator::AddSpecialization<3, 3, 4>();
   DomainLFIntegrator::AddSpecialization<3, 4, 5>();
   DomainLFIntegrator::AddSpecialization<3, 5, 6>();

   RegisterSimplexMmaKernels();
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
