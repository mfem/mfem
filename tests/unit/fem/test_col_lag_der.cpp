// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "unit_tests.hpp"
#include "mfem.hpp"

#include "fem/qinterp/grad.hpp"

using namespace std;
using namespace mfem;

static IntegrationRule PermuteIR(const IntegrationRule *irule,
                                 const Array<int> &perm)
{
   const int np = irule->GetNPoints();
   MFEM_VERIFY(np == perm.Size(), "Invalid permutation size");
   IntegrationRule ir(np);
   ir.SetOrder(irule->GetOrder());

   for (int i = 0; i < np; i++)
   {
      IntegrationPoint &ip_new = ir.IntPoint(i);
      const IntegrationPoint &ip_old = irule->IntPoint(perm[i]);
      ip_new.Set(ip_old.x, ip_old.y, ip_old.z, ip_old.weight);
   }

   return ir;
}

TEST_CASE("Collocated Derivative Kernels", "[QuadratureInterpolator]")
{
   // Add some specializations for the kernels
   // DIM, LAYOUT, PHYS, VDIM, D1D, Q1D
   QuadratureInterpolator::GradKernels::Specialization
   <1, QVectorLayout::byNODES, false, 1, 2, 2>::Add();
   QuadratureInterpolator::GradKernels::Specialization
   <1, QVectorLayout::byNODES, true, 1, 2, 2>::Add();

   QuadratureInterpolator::CollocatedGradKernels::Specialization
   <1, QVectorLayout::byNODES, false, 1, 2>::Add();
   QuadratureInterpolator::CollocatedGradKernels::Specialization
   <1, QVectorLayout::byNODES, true, 1, 2>::Add();

   const auto mesh_fname = GENERATE(
                              "../../data/inline-segment.mesh",
                              "../../data/inline-quad.mesh",
                              "../../data/inline-hex.mesh",
                              "../../data/star.mesh",
                              "../../data/star-q3.mesh",
                              "../../data/fichera.mesh",
                              "../../data/fichera-q3.mesh",
                              "../../data/diag-segment-2d.mesh", // 1D mesh in 2D
                              "../../data/diag-segment-3d.mesh", // 1D mesh in 3D
                              "../../data/star-surf.mesh" // surface mesh
                           );
   int p = GENERATE(range(1,7)); // element order, 1 <= p < 7
   int vdim = GENERATE(1,2,3); // vector dimension for grid-function

   const int seed = 0x100001b3;
   Mesh mesh = Mesh::LoadFromFile(mesh_fname);
   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();

   CAPTURE(mesh_fname, dim, sdim, p, vdim);

   int nelem = mesh.GetNE();

   const H1_FECollection fec(p, dim);
   FiniteElementSpace fes(&mesh, &fec, vdim);
   FiniteElementSpace nfes(&mesh, &fec, sdim);

   GridFunction x(&fes);
   VectorFunctionCoefficient gfc(vdim, [](const Vector &x, Vector &p)
   {
      for (int i = 0; i < p.Size(); i++)
      {
         p(i) = 0.0;
         for (int j = 0; j < x.Size(); j++)
         {
            p(i) += std::pow(x(j), i+1.0);
         }
      }
   });
   x.ProjectCoefficient(gfc);

   GridFunction nodes(&nfes);
   mesh.SetNodalGridFunction(&nodes);
   {
      Array<int> dofs, vdofs;
      GridFunction rdm(&nfes);
      Vector h0(nfes.GetNDofs());
      rdm.Randomize(seed);
      rdm -= 0.5;
      h0 = infinity();
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         nfes.GetElementDofs(i, dofs);
         const real_t hi = mesh.GetElementSize(i);
         for (int j = 0; j < dofs.Size(); j++)
         {
            h0(dofs[j]) = std::min(h0(dofs[j]), hi);
         }
      }
      rdm.HostReadWrite();
      for (int i = 0; i < nfes.GetNDofs(); i++)
      {
         for (int d = 0; d < sdim; d++)
         {
            rdm(nfes.DofToVDof(i,d)) *= (0.25/p)*h0(i);
         }
      }
      for (int i = 0; i < nfes.GetNBE(); i++)
      {
         nfes.GetBdrElementVDofs(i, vdofs);
         for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
      }
      nodes -= rdm;
   }

   const FiniteElement &fe = *(fes.GetFE(0));
   const IntegrationRule irnodes = fe.GetNodes();

   const NodalFiniteElement *nfe = dynamic_cast<const NodalFiniteElement*>
                                   (&fe);
   const Array<int> &irordering = nfe->GetLexicographicOrdering();
   IntegrationRule ir = PermuteIR(&irnodes, irordering);

   int nqp = ir.GetNPoints();
   const DofToQuad maps = fe.GetDofToQuad(ir, DofToQuad::TENSOR);
   auto geom = mesh.GetGeometricFactors(ir, GeometricFactors::JACOBIANS);

   Vector evec_values;
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *n0_R = fes.GetElementRestriction(ordering);
   evec_values.SetSize(n0_R->Height());
   n0_R->Mult(x, evec_values);

   using GK = QuadratureInterpolator::GradKernels;
   using CGK = QuadratureInterpolator::CollocatedGradKernels;

   SECTION("Compare collocated kernels")
   {
      auto L = GENERATE(QVectorLayout::byNODES, QVectorLayout::byVDIM);
      auto P = GENERATE(true, false);

      const int nd = maps.ndof;
      const int nq = maps.nqpt;

      Vector qp_der(nelem*vdim*nqp*(P ? sdim : dim));
      GK::Run(dim, L, P, vdim, nd, nq, nelem, maps.B.Read(),
              maps.G.Read(), geom->J.Read(), evec_values.Read(),
              qp_der.Write(), sdim, vdim, nd, nq);

      Vector col_der(nelem*vdim*nqp*(P ? sdim : dim));
      CGK::Run(dim, L, P, vdim, nd, nelem, maps.G.Read(), geom->J.Read(),
               evec_values.Read(), col_der.Write(), sdim, vdim, nd);

      qp_der -= col_der;
      REQUIRE(qp_der.Normlinf() == MFEM_Approx(0.0, 1e-10, 1e-10));
   }
}
