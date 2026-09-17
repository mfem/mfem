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

#include "../../lininteg.hpp"
#include "mma.hpp"
#include "domain_lf.hpp"

namespace mfem
{


void DLFEvalAssembleSimplexMma(const FiniteElementSpace &fes,
                               const IntegrationRule *ir,
                               const Array<int> &markers,
                               const Vector &coeff,
                               Vector &y)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   MFEM_VERIFY(UsesSimplexMMA(fes), "");

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

   constexpr bool by_val = true;
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
         internal::PADetJSetupSimplexFromNodes(
            dim, ne, nq, nd_n, by_val, ir->GetWeights(), nmaps.G, nodes_e, c1,
            D);
      }
      else if (coeff_vdim == 1)
      {
         internal::PADetJSetupSimplexFromNodes(
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
         internal::PADetJSetupSimplexFromNodes(
            dim, ne, nq, nd_n, by_val, ir->GetWeights(), nmaps.G, nodes_e, c_e,
            D);
      }

      zero_unmarked(D);
      DomainLFIntegrator::AssembleSimplexMmaKernels::Run(
         dim, dofs1D, nq, ne, P, D, Y, vdim, vc);
   }
}

void VectorFEDLFAssembleSimplexMma(const FiniteElementSpace &fes,
                                   const IntegrationRule *ir,
                                   const Array<int> &markers,
                                   const Vector &coeff,
                                   Vector &y)
{
   const bool hcurl = UsesSimplexMmaHcurl(fes);
   const bool hdiv = UsesSimplexMmaHdiv(fes);
   MFEM_VERIFY(hcurl || hdiv, "VectorFE simplex LF requires ND or RT MMA");
   MFEM_VERIFY(!(hcurl && hdiv), "");

   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   const FiniteElement &el = *fes.GetTypicalFE();
   const MemoryType mt = Device::GetDeviceMemoryType();
   const int dofs1D = el.GetOrder() + 1;
   const int nd = el.GetDof();
   const int nq = ir->GetNPoints();
   const int ne = mesh->GetNE();
   const int sdim = dim;
   const bool nd_space = hcurl;

   Array<real_t> B;
   internal::BuildRefVShape(el, *ir, B);
   const bool dual_y = (el.GetDofTransformation() != nullptr);

   Vector nodes_e;
   const Array<real_t> *G = nullptr;
   int nd_n = 0;
   internal::GetSimplexSetupGeom(*mesh, *ir, mt, nodes_e, G, nd_n);

   Vector D(nq * sdim * ne, mt);
   D.UseDevice(true);

   const bool const_c = (coeff.Size() == sdim);
   const auto W = Reshape(ir->GetWeights().Read(), nq);
   const auto Gg = Reshape(G->Read(), nq, dim, nd_n);
   const auto E = Reshape(nodes_e.Read(), nd_n, dim, ne);
   const auto C = const_c ? Reshape(coeff.Read(), sdim, 1, 1)
                  : Reshape(coeff.Read(), sdim, nq, ne);
   auto Dv = Reshape(D.Write(), nq, sdim, ne);
   const auto M = markers.Read();

   if (dim == 2)
   {
      mfem::forall(nq * ne, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / nq;
         const int q = idx - nq * e;
         if (M[e] == 0)
         {
            Dv(q, 0, e) = 0.0;
            Dv(q, 1, e) = 0.0;
            return;
         }
         real_t J11, J21, J12, J22;
         internal::EvalSimplexJ2(E, Gg, q, e, nd_n, J11, J21, J12, J22);
         const real_t f0 = const_c ? C(0, 0, 0) : C(0, q, e);
         const real_t f1 = const_c ? C(1, 0, 0) : C(1, q, e);
         const real_t w = W(q);
         if (nd_space)
         {
            // D = w * adj(J) * f  (matches FA (J^{-T} hat)·f |detJ| w for detJ>0)
            Dv(q, 0, e) = w * ( J22 * f0 - J12 * f1);
            Dv(q, 1, e) = w * (-J21 * f0 + J11 * f1);
         }
         else
         {
            // D = w * J^T * f
            Dv(q, 0, e) = w * (J11 * f0 + J21 * f1);
            Dv(q, 1, e) = w * (J12 * f0 + J22 * f1);
         }
      });
   }
   else
   {
      mfem::forall(nq * ne, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / nq;
         const int q = idx - nq * e;
         if (M[e] == 0)
         {
            Dv(q, 0, e) = 0.0;
            Dv(q, 1, e) = 0.0;
            Dv(q, 2, e) = 0.0;
            return;
         }
         real_t J11, J21, J31, J12, J22, J32, J13, J23, J33;
         internal::EvalSimplexJ3(E, Gg, q, e, nd_n,
                                 J11, J21, J31, J12, J22, J32, J13, J23, J33);
         const real_t f0 = const_c ? C(0, 0, 0) : C(0, q, e);
         const real_t f1 = const_c ? C(1, 0, 0) : C(1, q, e);
         const real_t f2 = const_c ? C(2, 0, 0) : C(2, q, e);
         const real_t w = W(q);
         if (nd_space)
         {
            real_t C11, C12, C13, C21, C22, C23, C31, C32, C33;
            internal::CofactorsJ3(J11, J21, J31, J12, J22, J32, J13, J23, J33,
                                  C11, C12, C13, C21, C22, C23, C31, C32, C33);
            Dv(q, 0, e) = w * (C11 * f0 + C12 * f1 + C13 * f2);
            Dv(q, 1, e) = w * (C21 * f0 + C22 * f1 + C23 * f2);
            Dv(q, 2, e) = w * (C31 * f0 + C32 * f1 + C33 * f2);
         }
         else
         {
            Dv(q, 0, e) = w * (J11 * f0 + J21 * f1 + J31 * f2);
            Dv(q, 1, e) = w * (J12 * f0 + J22 * f1 + J32 * f2);
            Dv(q, 2, e) = w * (J13 * f0 + J23 * f1 + J33 * f2);
         }
      });
   }

   VectorFEDomainLFIntegrator::AssembleSimplexMmaKernels::Run(
      dim, dofs1D, nq, ne, nd, nq, sdim, B, D, y);
   if (dual_y)
   {
      internal::TransformDualEVector(fes, y);
   }
}

void DomainLFIntegrator::RegisterSimplexMmaKernels()
{
   // MMA specializations (separate lists per integrator — see fem/integ/mma/README.md).
   // Order: DIM, D1D, QND. Unregistered → Fallback runtime shell.
   // 2D
   AddSimplexMmaSpecialization<2,2,3>();
   AddSimplexMmaSpecialization<2,2,12>();

   AddSimplexMmaSpecialization<2,3,6>();
   AddSimplexMmaSpecialization<2,3,15>();
   AddSimplexMmaSpecialization<2,3,16>();

   AddSimplexMmaSpecialization<2,4,12>();
   AddSimplexMmaSpecialization<2,4,19>();
   AddSimplexMmaSpecialization<2,4,25>();

   AddSimplexMmaSpecialization<2,5,16>();
   AddSimplexMmaSpecialization<2,5,28>();
   AddSimplexMmaSpecialization<2,5,33>();

   AddSimplexMmaSpecialization<2,6,25>();
   AddSimplexMmaSpecialization<2,6,37>();
   AddSimplexMmaSpecialization<2,6,42>();

   AddSimplexMmaSpecialization<2,7,33>();
   AddSimplexMmaSpecialization<2,7,49>();
   AddSimplexMmaSpecialization<2,7,55>();

   AddSimplexMmaSpecialization<2,8,42>();
   AddSimplexMmaSpecialization<2,8,60>();

   // 3D
   AddSimplexMmaSpecialization<3,2,4>();
   AddSimplexMmaSpecialization<3,2,24>();

   AddSimplexMmaSpecialization<3,3,14>();
   AddSimplexMmaSpecialization<3,3,35>();
   AddSimplexMmaSpecialization<3,3,46>();

   AddSimplexMmaSpecialization<3,4,24>();
   AddSimplexMmaSpecialization<3,4,81>();

   AddSimplexMmaSpecialization<3,5,46>();
   AddSimplexMmaSpecialization<3,5,96>();
   AddSimplexMmaSpecialization<3,5,123>();

   AddSimplexMmaSpecialization<3,6,81>();
   AddSimplexMmaSpecialization<3,6,175>();

   AddSimplexMmaSpecialization<3,7,123>();
   AddSimplexMmaSpecialization<3,7,209>();
   AddSimplexMmaSpecialization<3,7,248>();

   AddSimplexMmaSpecialization<3,8,175>();
   AddSimplexMmaSpecialization<3,8,284>();
}

void VectorFEDomainLFIntegrator::RegisterSimplexMmaKernels()
{
   // Same (DIM,D1D,QND) table as DomainLF; extra RT D1D = GetOrder()+1 is
   // already covered (D1D up to 8). Unregistered → Fallback runtime shell.
   AddSimplexMmaSpecialization<2,2,3>();
   AddSimplexMmaSpecialization<2,2,12>();

   AddSimplexMmaSpecialization<2,3,6>();
   AddSimplexMmaSpecialization<2,3,15>();
   AddSimplexMmaSpecialization<2,3,16>();

   AddSimplexMmaSpecialization<2,4,12>();
   AddSimplexMmaSpecialization<2,4,19>();
   AddSimplexMmaSpecialization<2,4,25>();

   AddSimplexMmaSpecialization<2,5,16>();
   AddSimplexMmaSpecialization<2,5,28>();
   AddSimplexMmaSpecialization<2,5,33>();

   AddSimplexMmaSpecialization<2,6,25>();
   AddSimplexMmaSpecialization<2,6,37>();
   AddSimplexMmaSpecialization<2,6,42>();

   AddSimplexMmaSpecialization<2,7,33>();
   AddSimplexMmaSpecialization<2,7,49>();
   AddSimplexMmaSpecialization<2,7,55>();

   AddSimplexMmaSpecialization<2,8,42>();
   AddSimplexMmaSpecialization<2,8,60>();

   AddSimplexMmaSpecialization<3,2,4>();
   AddSimplexMmaSpecialization<3,2,24>();

   AddSimplexMmaSpecialization<3,3,14>();
   AddSimplexMmaSpecialization<3,3,35>();
   AddSimplexMmaSpecialization<3,3,46>();

   AddSimplexMmaSpecialization<3,4,24>();
   AddSimplexMmaSpecialization<3,4,81>();

   AddSimplexMmaSpecialization<3,5,46>();
   AddSimplexMmaSpecialization<3,5,96>();
   AddSimplexMmaSpecialization<3,5,123>();

   AddSimplexMmaSpecialization<3,6,81>();
   AddSimplexMmaSpecialization<3,6,175>();

   AddSimplexMmaSpecialization<3,7,123>();
   AddSimplexMmaSpecialization<3,7,209>();
   AddSimplexMmaSpecialization<3,7,248>();

   AddSimplexMmaSpecialization<3,8,175>();
   AddSimplexMmaSpecialization<3,8,284>();
}


} // namespace mfem
