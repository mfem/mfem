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

#include "../bilininteg.hpp"
#include "bilininteg_pa_mma.hpp"
#include "bilininteg_mass_pa_simplices_mma.hpp" // IWYU pragma: keep

namespace mfem
{

void MassIntegrator::AssembleSimplexMmaPA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   MFEM_VERIFY(mesh->SpaceDimension() == dim, "");

   const FiniteElement &el = *fes.GetTypicalFE();
   const Geometry::Type geom_t = (dim == 2) ? Geometry::TRIANGLE
                                 : Geometry::TETRAHEDRON;
   MFEM_VERIFY(el.GetGeomType() == geom_t, "");
   MFEM_VERIFY(IsSimplexMmaH1Element(el, dim), "");

   ElementTransformation *T0 = mesh->GetTypicalElementTransformation();
   const int map_type = el.GetMapType();
   const int p = el.GetOrder();
   dofs1D = p + 1;
   const int ndof = el.GetDof();

   const int q_order = IntRule ? IntRule->GetOrder()
                       : 2 * p + T0->OrderW() + 4;
   const IntegrationRule &ir =
      IntRule ? *IntRule : IntRules.Get(geom_t, q_order);
   nq = ir.GetNPoints();
   quad1D = 0;
   ne = mesh->GetNE();
   use_simplices_mma = true;
   maps = nullptr;

   simplex_mma_P.SetSize(nq * ndof, mt);
   {
      real_t *Ph = simplex_mma_P.HostWrite();
      Vector shape_ref(ndof);
      for (int q = 0; q < nq; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         el.CalcShape(ip, shape_ref);
         for (int i = 0; i < ndof; i++)
         {
            Ph[q + nq * i] = shape_ref(i);
         }
      }
   }

   // Assemble from restricted mesh nodes
   geom = nullptr;
   Vector nodes_e;
   int nd_n = 0, sdim = 0;
   internal::GetSimplexMeshNodesE(*mesh, mt, nodes_e, nd_n, sdim);
   MFEM_VERIFY(sdim == dim, "");
   const FiniteElement &nfe = *mesh->GetNodes()->FESpace()->GetTypicalFE();
   const DofToQuad &nmaps = nfe.GetDofToQuad(ir, DofToQuad::FULL);
   MFEM_VERIFY(nmaps.ndof == nd_n && nmaps.nqpt == nq, "");

   pa_data.SetSize(nq * ne, mt);

   QuadratureSpace qs(*mesh, ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);

   const bool by_val = map_type == FiniteElement::VALUE;
   internal::PADetJSetupSimplexFromNodes(
      dim, ne, nq, nd_n,
      by_val, ir.GetWeights(), nmaps.G, nodes_e, coeff, pa_data);
}

void MassIntegrator::RegisterSimplexMmaKernels()
{
   // Usage-based (DIM,D1D,QND): GetRule/bench q=2p, Stroud unit tests,
   // and AssembleSimplexMmaPA default / H1-smoke (2p+OrderW+4) over all
   // [MMA] meshes.

   // 2D
   AddSimplexMmaSpecialization<2,2,3>(); // GetRule, bench
   AddSimplexMmaSpecialization<2,2,4>(); // Stroud
   AddSimplexMmaSpecialization<2,2,9>(); // Stroud
   AddSimplexMmaSpecialization<2,2,12>(); // GetRule, default/smoke
   AddSimplexMmaSpecialization<2,2,16>(); // Stroud, default/smoke
   AddSimplexMmaSpecialization<2,2,25>(); // default/smoke
   AddSimplexMmaSpecialization<2,2,33>(); // default/smoke (curved)

   AddSimplexMmaSpecialization<2,3,6>(); // GetRule, bench
   AddSimplexMmaSpecialization<2,3,9>(); // Stroud
   AddSimplexMmaSpecialization<2,3,16>(); // GetRule, Stroud, default/smoke
   AddSimplexMmaSpecialization<2,3,25>(); // Stroud, default/smoke
   AddSimplexMmaSpecialization<2,3,33>(); // default/smoke
   AddSimplexMmaSpecialization<2,3,36>(); // Stroud (curved)
   AddSimplexMmaSpecialization<2,3,42>(); // default/smoke (curved)

   AddSimplexMmaSpecialization<2,4,12>(); // GetRule, bench
   AddSimplexMmaSpecialization<2,4,16>(); // GetRule, Stroud
   AddSimplexMmaSpecialization<2,4,25>(); // GetRule, Stroud, default/smoke

   AddSimplexMmaSpecialization<2,5,16>(); // GetRule, bench
   AddSimplexMmaSpecialization<2,5,33>(); // GetRule, default/smoke

   AddSimplexMmaSpecialization<2,6,25>(); // GetRule, bench
   AddSimplexMmaSpecialization<2,6,36>(); // Stroud
   AddSimplexMmaSpecialization<2,6,42>(); // GetRule, default/smoke
   AddSimplexMmaSpecialization<2,6,49>(); // Stroud
   AddSimplexMmaSpecialization<2,6,55>(); // default/smoke
   AddSimplexMmaSpecialization<2,6,64>(); // Stroud
   AddSimplexMmaSpecialization<2,6,67>(); // default/smoke
   AddSimplexMmaSpecialization<2,6,79>(); // default/smoke (curved)
   AddSimplexMmaSpecialization<2,6,81>(); // Stroud (curved)

   AddSimplexMmaSpecialization<2,7,33>(); // GetRule, bench
   AddSimplexMmaSpecialization<2,7,49>(); // Stroud
   AddSimplexMmaSpecialization<2,7,55>(); // GetRule, default/smoke
   AddSimplexMmaSpecialization<2,7,64>(); // Stroud
   AddSimplexMmaSpecialization<2,7,67>(); // default/smoke
   AddSimplexMmaSpecialization<2,7,79>(); // default/smoke
   AddSimplexMmaSpecialization<2,7,81>(); // Stroud
   AddSimplexMmaSpecialization<2,7,100>(); // default/smoke (curved)
   AddSimplexMmaSpecialization<2,7,126>(); // Stroud (curved)

   AddSimplexMmaSpecialization<2,8,42>(); // GetRule, bench

   // 3D
   AddSimplexMmaSpecialization<3,2,4>(); // GetRule, bench
   AddSimplexMmaSpecialization<3,2,8>(); // Stroud
   AddSimplexMmaSpecialization<3,2,14>(); // GetRule
   AddSimplexMmaSpecialization<3,2,24>(); // default/smoke

   AddSimplexMmaSpecialization<3,3,14>(); // GetRule, bench
   AddSimplexMmaSpecialization<3,3,27>(); // Stroud
   AddSimplexMmaSpecialization<3,3,35>(); // GetRule
   AddSimplexMmaSpecialization<3,3,46>(); // default/smoke

   AddSimplexMmaSpecialization<3,4,24>(); // GetRule, bench
   AddSimplexMmaSpecialization<3,4,59>(); // GetRule
   AddSimplexMmaSpecialization<3,4,81>(); // default/smoke

   AddSimplexMmaSpecialization<3,5,46>(); // GetRule, bench
   AddSimplexMmaSpecialization<3,5,96>(); // GetRule
   AddSimplexMmaSpecialization<3,5,123>(); // default/smoke

   AddSimplexMmaSpecialization<3,6,81>(); // GetRule, bench
   AddSimplexMmaSpecialization<3,6,145>(); // GetRule
   AddSimplexMmaSpecialization<3,6,175>(); // default/smoke
   AddSimplexMmaSpecialization<3,6,216>(); // Stroud

   AddSimplexMmaSpecialization<3,7,123>(); // GetRule, bench
   AddSimplexMmaSpecialization<3,7,209>(); // GetRule
   AddSimplexMmaSpecialization<3,7,248>(); // default/smoke

   AddSimplexMmaSpecialization<3,8,175>(); // GetRule, bench
   AddSimplexMmaSpecialization<3,8,284>(); // GetRule
}

} // namespace mfem
