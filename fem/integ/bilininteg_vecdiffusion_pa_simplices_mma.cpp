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
#include "mma/mma.hpp"
#include "bilininteg_vecdiffusion_pa_simplices_mma.hpp"
#include "bilininteg_diffusion_pa_simplices_mma.hpp" // PADiffusionSetupSimplexFromNodes

namespace mfem
{

void VectorDiffusionIntegrator::AssembleSimplexMmaPA(
   const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   maps = nullptr;
   geom = nullptr;

   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   sdim = mesh->SpaceDimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   MFEM_VERIFY(sdim == dim, "");

   const FiniteElement &el = *fes.GetTypicalFE();
   const Geometry::Type geom_t =
      (dim == 2) ? Geometry::TRIANGLE : Geometry::TETRAHEDRON;
   MFEM_VERIFY(el.GetGeomType() == geom_t, "");
   MFEM_VERIFY(IsSimplexMmaH1Element(el, dim), "");

   vdim = (vdim == -1) ? dim : vdim;
   MFEM_VERIFY(vdim == fes.GetVDim(), "vdim != fes.GetVDim()");

   const int p = el.GetOrder();
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2;

   const IntegrationRule &ir = IntRule ? *IntRule
                             : DiffusionIntegrator::GetRule(el, el);
   nq = ir.GetNPoints();
   const int dof = el.GetDof();

   dofs1D = p + 1;
   quad1D = 0;
   ne = mesh->GetNE();
   use_simplices_mma = true;
   use_tensors_mma = false;
   coeff_vdim = 1;

   simplex_mma_G.SetSize(nq * dof * dim, mt);
   {
      real_t *Gh = simplex_mma_G.HostWrite();
      dshape.SetSize(dof, dim);
      for (int q = 0; q < nq; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         el.CalcDShape(ip, dshape);
         for (int d = 0; d < dim; d++)
         {
            for (int i = 0; i < dof; i++)
            {
               Gh[q + nq * (i + dof * d)] = dshape(i, d);
            }
         }
      }
   }

   Vector nodes_e;
   int nd_n = 0, nodes_sdim = 0;
   internal::GetSimplexMeshNodesE(*mesh, mt, nodes_e, nd_n, nodes_sdim);
   MFEM_VERIFY(nodes_sdim == dim, "");
   const FiniteElement &nfe = *mesh->GetNodes()->FESpace()->GetTypicalFE();
   const DofToQuad &nmaps = nfe.GetDofToQuad(ir, DofToQuad::FULL);
   MFEM_VERIFY(nmaps.ndof == nd_n && nmaps.nqpt == nq, "");

   QuadratureSpace qs(*mesh, ir);
   CoefficientVector coeff(qs, CoefficientStorage::COMPRESSED);
   if (Q) { coeff.Project(*Q); }
   else { coeff.SetConstant(1.0); }
   MFEM_VERIFY(coeff.GetVDim() == 1, "simplex VectorDiffusion MMA: scalar Q");

   // Shared SYM metric pack (one copy for all vector components).
   pa_data.SetSize(symmDims * nq * ne, mt);
   internal::PADiffusionSetupSimplexFromNodes(dim, 1, ne, nq, nd_n,
                                              ir.GetWeights(), nmaps.G, nodes_e,
                                              coeff, pa_data);
}

void VectorDiffusionIntegrator::RegisterSimplexMmaKernels()
{
   // Match scalar DiffusionIntegrator simplex specializations.
   AddSimplexMmaSpecialization<2,2,1>();
   AddSimplexMmaSpecialization<2,2,4>();
   AddSimplexMmaSpecialization<2,2,9>();
   AddSimplexMmaSpecialization<2,2,12>();
   AddSimplexMmaSpecialization<2,2,16>();
   AddSimplexMmaSpecialization<2,2,25>();
   AddSimplexMmaSpecialization<2,2,33>();

   AddSimplexMmaSpecialization<2,3,3>();
   AddSimplexMmaSpecialization<2,3,9>();
   AddSimplexMmaSpecialization<2,3,16>();
   AddSimplexMmaSpecialization<2,3,25>();
   AddSimplexMmaSpecialization<2,3,33>();
   AddSimplexMmaSpecialization<2,3,36>();
   AddSimplexMmaSpecialization<2,3,42>();

   AddSimplexMmaSpecialization<2,4,6>();
   AddSimplexMmaSpecialization<2,4,16>();
   AddSimplexMmaSpecialization<2,4,25>();

   AddSimplexMmaSpecialization<2,5,12>();
   AddSimplexMmaSpecialization<2,5,33>();

   AddSimplexMmaSpecialization<2,6,16>();
   AddSimplexMmaSpecialization<2,6,36>();
   AddSimplexMmaSpecialization<2,6,42>();
   AddSimplexMmaSpecialization<2,6,49>();
   AddSimplexMmaSpecialization<2,6,55>();
   AddSimplexMmaSpecialization<2,6,64>();
   AddSimplexMmaSpecialization<2,6,67>();
   AddSimplexMmaSpecialization<2,6,79>();
   AddSimplexMmaSpecialization<2,6,81>();

   AddSimplexMmaSpecialization<2,7,25>();
   AddSimplexMmaSpecialization<2,7,49>();
   AddSimplexMmaSpecialization<2,7,55>();
   AddSimplexMmaSpecialization<2,7,64>();
   AddSimplexMmaSpecialization<2,7,67>();
   AddSimplexMmaSpecialization<2,7,79>();
   AddSimplexMmaSpecialization<2,7,81>();
   AddSimplexMmaSpecialization<2,7,100>();
   AddSimplexMmaSpecialization<2,7,126>();

   AddSimplexMmaSpecialization<2,8,33>();

   AddSimplexMmaSpecialization<3,2,1>();
   AddSimplexMmaSpecialization<3,2,8>();
   AddSimplexMmaSpecialization<3,2,24>();

   AddSimplexMmaSpecialization<3,3,4>();
   AddSimplexMmaSpecialization<3,3,27>();
   AddSimplexMmaSpecialization<3,3,46>();

   AddSimplexMmaSpecialization<3,4,14>();
   AddSimplexMmaSpecialization<3,4,81>();

   AddSimplexMmaSpecialization<3,5,24>();
   AddSimplexMmaSpecialization<3,5,123>();

   AddSimplexMmaSpecialization<3,6,46>();
   AddSimplexMmaSpecialization<3,6,175>();
   AddSimplexMmaSpecialization<3,6,216>();

   AddSimplexMmaSpecialization<3,7,81>();
   AddSimplexMmaSpecialization<3,7,248>();

   AddSimplexMmaSpecialization<3,8,123>();
}

VectorDiffusionIntegrator::ApplySimplexMmaKernelType
VectorDiffusionIntegrator::ApplySimplexMmaPAKernels::Fallback(int dim, int, int)
{
   if (dim == 2) { return internal::MmaVectorDiffusionApplySimplex2D; }
   if (dim == 3) { return internal::MmaVectorDiffusionApplySimplex3D; }
   MFEM_ABORT("Simplex MMA VectorDiffusion PA is only implemented for dim 2/3");
   return nullptr;
}

} // namespace mfem
