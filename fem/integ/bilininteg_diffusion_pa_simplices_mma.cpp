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
#include "bilininteg_diffusion_pa_simplices_mma.hpp"

namespace mfem
{

void DiffusionIntegrator::AssemblePA_SimplexMma(const FiniteElementSpace &fes)
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

   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2;
   const int p = el.GetOrder();
   dofs1D = p + 1;
   const int ndof = el.GetDof();

   const IntegrationRule &ir = IntRule ? *IntRule : GetRule(el, el);
   const int nq1 = ir.GetNPoints();
   quad1D = nq1;
   ne = mesh->GetNE();
   pa_simplex_mma = true;
   maps = nullptr;

   simplex_mma_G.SetSize(nq1 * ndof * dim, mt);
   {
      real_t *Gh = simplex_mma_G.HostWrite();
      DenseMatrix dshape(ndof, dim);
      for (int q = 0; q < nq1; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         el.CalcDShape(ip, dshape);
         for (int d = 0; d < dim; d++)
         {
            for (int i = 0; i < ndof; i++)
            {
               Gh[q + nq1 * (i + ndof * d)] = dshape(i, d);
            }
         }
      }
   }

   // Assemble geometry directly from restricted mesh nodes (no QI /
   // GetGeometricFactors). Mult only reads the resulting pa_data.
   geom = nullptr;
   Vector nodes_e;
   int nd_n = 0, sdim = 0;
   internal::GetSimplexMeshNodesE(*mesh, mt, nodes_e, nd_n, sdim);
   MFEM_VERIFY(sdim == dim, "");
   const FiniteElement &nfe = *mesh->GetNodes()->FESpace()->GetTypicalFE();
   const DofToQuad &nmaps = nfe.GetDofToQuad(ir, DofToQuad::FULL);
   MFEM_VERIFY(nmaps.ndof == nd_n && nmaps.nqpt == nq1, "");

   QuadratureSpace qs(*mesh, ir);
   CoefficientVector coeff(qs, CoefficientStorage::COMPRESSED);
   if (MQ) { coeff.ProjectTranspose(*MQ); }
   else if (VQ) { coeff.Project(*VQ); }
   else if (Q) { coeff.Project(*Q); }
   else { coeff.SetConstant(1.0); }

   const int coeff_dim = coeff.GetVDim();
   symmetric = (coeff_dim != dims * dims);
   const int pa_size = symmetric ? symmDims : dims * dims;
   pa_data.SetSize(pa_size * nq1 * ne, mt);
   internal::PADiffusionSetupSimplexFromNodes(dim, coeff_dim, ne, nq1, nd_n,
                                              ir.GetWeights(), nmaps.G, nodes_e,
                                              coeff, pa_data);
}

void DiffusionIntegrator::RegisterSimplexMmaKernels()
{
   AddSimplexMmaSpecialization<2,2,1>();  // GetRule p=1
   AddSimplexMmaSpecialization<2,2,3>();
   AddSimplexMmaSpecialization<2,2,7>();   // BP3tri p=1, q=2p+3
   AddSimplexMmaSpecialization<2,2,12>();
   AddSimplexMmaSpecialization<2,3,3>();  // GetRule p=2
   AddSimplexMmaSpecialization<2,3,6>();
   AddSimplexMmaSpecialization<2,3,15>(); // BP3tri p=2
   AddSimplexMmaSpecialization<2,3,16>();
   AddSimplexMmaSpecialization<2,4,6>();  // GetRule p=3
   AddSimplexMmaSpecialization<2,4,7>();   // BP7tri p=3
   AddSimplexMmaSpecialization<2,4,12>();
   AddSimplexMmaSpecialization<2,4,19>(); // BP3tri p=3
   AddSimplexMmaSpecialization<2,4,25>();
   AddSimplexMmaSpecialization<2,5,12>(); // GetRule p=4
   AddSimplexMmaSpecialization<2,5,15>();  // BP7tri p=4
   AddSimplexMmaSpecialization<2,5,16>();
   AddSimplexMmaSpecialization<2,5,28>(); // BP3tri p=4
   AddSimplexMmaSpecialization<2,5,33>();
   AddSimplexMmaSpecialization<2,6,16>(); // GetRule p=5
   AddSimplexMmaSpecialization<2,6,19>();  // BP7tri p=5
   AddSimplexMmaSpecialization<2,6,25>();
   AddSimplexMmaSpecialization<2,6,37>(); // BP3tri p=5
   AddSimplexMmaSpecialization<2,6,42>();
   AddSimplexMmaSpecialization<2,7,25>(); // GetRule p=6
   AddSimplexMmaSpecialization<2,7,28>();  // BP7tri p=6
   AddSimplexMmaSpecialization<2,7,33>();
   AddSimplexMmaSpecialization<2,7,49>(); // BP3tri p=6
   AddSimplexMmaSpecialization<2,7,55>();
   AddSimplexMmaSpecialization<2,8,37>();  // BP7tri p=7
   AddSimplexMmaSpecialization<2,8,42>();  // BP5tri p=7
   AddSimplexMmaSpecialization<2,8,60>(); // BP3tri p=7

   // 3D (GLL tet)
   AddSimplexMmaSpecialization<3,2,1>();   // GetRule p=1
   AddSimplexMmaSpecialization<3,2,4>();
   AddSimplexMmaSpecialization<3,2,14>();  // BP3tet p=1, q=2p+3
   AddSimplexMmaSpecialization<3,2,24>();
   AddSimplexMmaSpecialization<3,3,4>();   // GetRule p=2
   AddSimplexMmaSpecialization<3,3,8>();   // BP7tet p=2
   AddSimplexMmaSpecialization<3,3,14>();
   AddSimplexMmaSpecialization<3,3,35>();
   AddSimplexMmaSpecialization<3,3,46>();
   AddSimplexMmaSpecialization<3,4,11>();  // GetRule p=3 approx
   AddSimplexMmaSpecialization<3,4,14>();  // BP7tet p=3
   AddSimplexMmaSpecialization<3,4,24>();
   AddSimplexMmaSpecialization<3,4,59>();  // BP3tet p=3, q=2p+3
   AddSimplexMmaSpecialization<3,4,81>();
   AddSimplexMmaSpecialization<3,5,24>();  // GetRule p=4
   AddSimplexMmaSpecialization<3,5,35>();  // BP7tet p=4
   AddSimplexMmaSpecialization<3,5,46>();
   AddSimplexMmaSpecialization<3,5,96>();
   AddSimplexMmaSpecialization<3,5,123>();
   AddSimplexMmaSpecialization<3,6,45>();  // GetRule p=5
   AddSimplexMmaSpecialization<3,6,59>();  // BP7tet p=5
   AddSimplexMmaSpecialization<3,6,81>();
   AddSimplexMmaSpecialization<3,6,145>(); // BP3tet p=5, q=2p+3
   AddSimplexMmaSpecialization<3,6,175>();
   AddSimplexMmaSpecialization<3,7,74>();  // GetRule p=6
   AddSimplexMmaSpecialization<3,7,96>();  // BP7tet p=6
   AddSimplexMmaSpecialization<3,7,123>();
   AddSimplexMmaSpecialization<3,7,209>(); // BP3tet p=6, q=2p+3
   AddSimplexMmaSpecialization<3,7,248>();
   AddSimplexMmaSpecialization<3,8,145>(); // BP7tet p=7
   AddSimplexMmaSpecialization<3,8,175>();
   AddSimplexMmaSpecialization<3,8,284>();
}

} // namespace mfem
