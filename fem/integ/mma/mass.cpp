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
#include "mma.hpp"
#include "mass.hpp"

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
   // MMA specializations (separate lists per integrator — see fem/integ/mma/README.md).
   // Order: DIM, D1D, QND. Unregistered → Fallback runtime shell.
   // 2D
   AddSimplexMmaSpecialization<2,2,3>();
   AddSimplexMmaSpecialization<2,2,4>();
   AddSimplexMmaSpecialization<2,2,9>();
   AddSimplexMmaSpecialization<2,2,12>();
   AddSimplexMmaSpecialization<2,2,16>();
   AddSimplexMmaSpecialization<2,2,25>();
   AddSimplexMmaSpecialization<2,2,33>();

   AddSimplexMmaSpecialization<2,3,6>();
   AddSimplexMmaSpecialization<2,3,9>();
   AddSimplexMmaSpecialization<2,3,16>();
   AddSimplexMmaSpecialization<2,3,25>();
   AddSimplexMmaSpecialization<2,3,33>();
   AddSimplexMmaSpecialization<2,3,36>();
   AddSimplexMmaSpecialization<2,3,42>();

   AddSimplexMmaSpecialization<2,4,12>();
   AddSimplexMmaSpecialization<2,4,16>();
   AddSimplexMmaSpecialization<2,4,25>();

   AddSimplexMmaSpecialization<2,5,16>();
   AddSimplexMmaSpecialization<2,5,33>();

   AddSimplexMmaSpecialization<2,6,25>();
   AddSimplexMmaSpecialization<2,6,36>();
   AddSimplexMmaSpecialization<2,6,42>();
   AddSimplexMmaSpecialization<2,6,49>();
   AddSimplexMmaSpecialization<2,6,55>();
   AddSimplexMmaSpecialization<2,6,64>();
   AddSimplexMmaSpecialization<2,6,67>();
   AddSimplexMmaSpecialization<2,6,79>();
   AddSimplexMmaSpecialization<2,6,81>();

   AddSimplexMmaSpecialization<2,7,33>();
   AddSimplexMmaSpecialization<2,7,49>();
   AddSimplexMmaSpecialization<2,7,55>();
   AddSimplexMmaSpecialization<2,7,64>();
   AddSimplexMmaSpecialization<2,7,67>();
   AddSimplexMmaSpecialization<2,7,79>();
   AddSimplexMmaSpecialization<2,7,81>();
   AddSimplexMmaSpecialization<2,7,100>();
   AddSimplexMmaSpecialization<2,7,126>();

   AddSimplexMmaSpecialization<2,8,42>();

   // 3D
   AddSimplexMmaSpecialization<3,2,4>();
   AddSimplexMmaSpecialization<3,2,8>();
   AddSimplexMmaSpecialization<3,2,14>();
   AddSimplexMmaSpecialization<3,2,24>();

   AddSimplexMmaSpecialization<3,3,14>();
   AddSimplexMmaSpecialization<3,3,27>();
   AddSimplexMmaSpecialization<3,3,35>();
   AddSimplexMmaSpecialization<3,3,46>();

   AddSimplexMmaSpecialization<3,4,24>();
   AddSimplexMmaSpecialization<3,4,59>();
   AddSimplexMmaSpecialization<3,4,81>();

   AddSimplexMmaSpecialization<3,5,46>();
   AddSimplexMmaSpecialization<3,5,96>();
   AddSimplexMmaSpecialization<3,5,123>();

   AddSimplexMmaSpecialization<3,6,81>();
   AddSimplexMmaSpecialization<3,6,145>();
   AddSimplexMmaSpecialization<3,6,175>();
   AddSimplexMmaSpecialization<3,6,216>();

   AddSimplexMmaSpecialization<3,7,123>();
   AddSimplexMmaSpecialization<3,7,209>();
   AddSimplexMmaSpecialization<3,7,248>();

   AddSimplexMmaSpecialization<3,8,175>();
   AddSimplexMmaSpecialization<3,8,284>();
}



void MassIntegrator::RegisterTensorsMmaKernels()
{
   // p = 3..7 (D1D = 4..8, Q1D = D1D+1). Keep in sync with diffusion tensors.
   AddTensorsMmaSpecialization<2,4,5>();
   AddTensorsMmaSpecialization<2,5,6>();
   AddTensorsMmaSpecialization<2,6,7>();
   AddTensorsMmaSpecialization<2,7,8>();
   AddTensorsMmaSpecialization<2,8,9>();

   AddTensorsMmaSpecialization<3,4,5>();
   AddTensorsMmaSpecialization<3,5,6>();
   AddTensorsMmaSpecialization<3,6,7>();
   AddTensorsMmaSpecialization<3,7,8>();
   AddTensorsMmaSpecialization<3,8,9>();
}

MassIntegrator::ApplyTensorsMmaKernelType
MassIntegrator::ApplyTensorsMmaPAKernels::Fallback(int dim, int, int)
{
   if (dim == 2) { return internal::MmaMassApplyTensors2D; }
   if (dim == 3) { return internal::MmaMassApplyTensors3D; }
   MFEM_ABORT("Tensors MMA mass PA is only implemented for dim 2 or 3");
   return nullptr;
}


} // namespace mfem
