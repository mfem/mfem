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

#include "../qfunction.hpp"
#include "bilininteg_mass_pa_simplices_mma.hpp"

namespace mfem
{

void MassIntegrator::AssemblePA_SimplexMma(const FiniteElementSpace &fes)
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
   const int nq1 = ir.GetNPoints();
   quad1D = nq1;
   this->nq = nq1;
   ne = mesh->GetNE();
   pa_simplex_mma = true;
   maps = nullptr;

   simplex_mma_P.SetSize(nq1 * ndof, mt);
   {
      real_t *Ph = simplex_mma_P.HostWrite();
      Vector shape_ref(ndof);
      for (int q = 0; q < nq1; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         el.CalcShape(ip, shape_ref);
         for (int i = 0; i < ndof; i++)
         {
            Ph[q + nq1 * i] = shape_ref(i);
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
   MFEM_VERIFY(nmaps.ndof == nd_n && nmaps.nqpt == nq1, "");

   pa_data.SetSize(nq1 * ne, mt);

   QuadratureSpace qs(*mesh, ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);

   const bool by_val = map_type == FiniteElement::VALUE;
   internal::PAMassSetupSimplexFromNodes(
      dim, ne, nq1, nd_n, by_val, ir.GetWeights(), nmaps.G, nodes_e, coeff,
      pa_data);
}

void MassIntegrator::RegisterSimplexMmaKernels()
{
   AddSimplexMmaSpecialization<2,2,3>();
   AddSimplexMmaSpecialization<2,2,7>();   // BP1tri p=1, q=2p+3
   AddSimplexMmaSpecialization<2,2,12>();
   AddSimplexMmaSpecialization<2,3,6>();
   AddSimplexMmaSpecialization<2,3,15>(); // GLL BP1tri p=2
   AddSimplexMmaSpecialization<2,3,16>();
   AddSimplexMmaSpecialization<2,4,7>();   // BP7tri p=3, q=2p-1
   AddSimplexMmaSpecialization<2,4,12>();
   AddSimplexMmaSpecialization<2,4,19>(); // GLL BP1tri p=3
   AddSimplexMmaSpecialization<2,4,25>();
   AddSimplexMmaSpecialization<2,5,15>();  // BP7tri p=4
   AddSimplexMmaSpecialization<2,5,16>();
   AddSimplexMmaSpecialization<2,5,28>(); // GLL BP1tri p=4
   AddSimplexMmaSpecialization<2,5,33>();
   AddSimplexMmaSpecialization<2,6,19>();  // BP7tri p=5
   AddSimplexMmaSpecialization<2,6,25>();
   AddSimplexMmaSpecialization<2,6,37>(); // GLL BP1tri p=5
   AddSimplexMmaSpecialization<2,6,42>();
   AddSimplexMmaSpecialization<2,7,28>();  // BP7tri p=6
   AddSimplexMmaSpecialization<2,7,33>();
   AddSimplexMmaSpecialization<2,7,49>(); // GLL BP1tri p=6
   AddSimplexMmaSpecialization<2,7,55>();
   AddSimplexMmaSpecialization<2,8,37>();  // BP7tri p=7
   AddSimplexMmaSpecialization<2,8,42>();  // BP5tri p=7, q=2p
   AddSimplexMmaSpecialization<2,8,60>(); // GLL BP1tri p=7

   // 3D (GLL tet): BP1tet uses IntRules order 2p; tests use 2p+4.
   AddSimplexMmaSpecialization<3,2,4>();   // BP p=1 order 2
   AddSimplexMmaSpecialization<3,2,14>();  // BP1/3tet p=1, q=2p+3
   AddSimplexMmaSpecialization<3,2,24>();  // test p=1 order 6
   AddSimplexMmaSpecialization<3,3,8>();   // BP7tet p=2, q=2p-1
   AddSimplexMmaSpecialization<3,3,14>();  // BP p=2 order 4
   AddSimplexMmaSpecialization<3,3,35>();
   AddSimplexMmaSpecialization<3,3,46>();  // test p=2 order 8
   AddSimplexMmaSpecialization<3,4,14>();  // BP7tet p=3
   AddSimplexMmaSpecialization<3,4,24>();  // BP p=3 order 6
   AddSimplexMmaSpecialization<3,4,59>();  // BP1/3tet p=3, q=2p+3
   AddSimplexMmaSpecialization<3,4,81>();  // test p=3 order 10
   AddSimplexMmaSpecialization<3,5,35>();  // BP7tet p=4
   AddSimplexMmaSpecialization<3,5,46>();  // BP p=4 order 8
   AddSimplexMmaSpecialization<3,5,96>();
   AddSimplexMmaSpecialization<3,5,123>(); // test p=4 order 12
   AddSimplexMmaSpecialization<3,6,59>();  // BP7tet p=5
   AddSimplexMmaSpecialization<3,6,81>();  // BP p=5 order 10
   AddSimplexMmaSpecialization<3,6,145>(); // BP1/3tet p=5, q=2p+3
   AddSimplexMmaSpecialization<3,6,175>(); // test p=5 order 14
   AddSimplexMmaSpecialization<3,7,96>();  // BP7tet p=6
   AddSimplexMmaSpecialization<3,7,123>(); // BP p=6 order 12
   AddSimplexMmaSpecialization<3,7,209>();
   AddSimplexMmaSpecialization<3,7,248>(); // test p=6 order 16
   AddSimplexMmaSpecialization<3,8,145>(); // BP7tet p=7
   AddSimplexMmaSpecialization<3,8,175>(); // BP p=7 order 14
   AddSimplexMmaSpecialization<3,8,284>();
}

} // namespace mfem
