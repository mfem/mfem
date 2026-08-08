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


} // namespace mfem
