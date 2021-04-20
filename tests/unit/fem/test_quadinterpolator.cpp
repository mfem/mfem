// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

static bool testQuadratureInterpolator(const int dim,
                                       const int p,
                                       const int q,
                                       const QVectorLayout q_layout,
                                       const int nx, const int ny, const int nz)
{
   const int vdim = dim;
   const int seed = 0x100001b3;
   const int ordering = Ordering::byNODES;

   REQUIRE((dim == 2 || dim == 3));
   Mesh mesh = dim == 1 ? Mesh::MakeCartesian1D(nx, Element::SEGMENT) :
               dim == 2 ? Mesh::MakeCartesian2D(nx,ny, Element::QUADRILATERAL):
               Mesh::MakeCartesian3D(nx,nx,nz, Element::HEXAHEDRON);

   mesh.SetCurvature(p, false, dim, ordering);

   const H1_FECollection fec(p, dim);
   FiniteElementSpace sfes(&mesh, &fec, 1, ordering);
   FiniteElementSpace vfes(&mesh, &fec, vdim, ordering);

   GridFunction x(&sfes);
   x.Randomize(seed);

   GridFunction nodes(&vfes);
   mesh.SetNodalFESpace(&vfes);
   mesh.SetNodalGridFunction(&nodes);
   {
      Array<int> dofs, vdofs;
      GridFunction rdm(&vfes);
      Vector h0(vfes.GetNDofs());
      rdm.Randomize(seed);
      rdm -= 0.5;
      h0 = infinity();
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         vfes.GetElementDofs(i, dofs);
         const double hi = mesh.GetElementSize(i);
         for (int j = 0; j < dofs.Size(); j++)
         {
            h0(dofs[j]) = std::min(h0(dofs[j]), hi);
         }
      }
      for (int i = 0; i < vfes.GetNDofs(); i++)
      {
         for (int d = 0; d < dim; d++)
         {
            rdm(vfes.DofToVDof(i,d)) *= h0(i);
         }
      }
      for (int i = 0; i < vfes.GetNBE(); i++)
      {
         vfes.GetBdrElementVDofs(i, vdofs);
         for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
      }
      nodes -= rdm;
   }

   const Geometry::Type GeomType = mesh.GetElementBaseGeometry(0);
   const IntegrationRule &ir = IntRules.Get(GeomType, q);
   const QuadratureInterpolator *sqi(sfes.GetQuadratureInterpolator(ir));
   const QuadratureInterpolator *vqi(vfes.GetQuadratureInterpolator(ir));

   const int NE(mesh.GetNE());
   const int NQ(ir.GetNPoints());
   const int ND(sfes.GetFE(0)->GetDof());
   REQUIRE(ND == vfes.GetFE(0)->GetDof());

   const ElementDofOrdering nat_ordering = ElementDofOrdering::NATIVE;
   const ElementDofOrdering lex_ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *SRN(sfes.GetElementRestriction(nat_ordering));
   const Operator *SRL(sfes.GetElementRestriction(lex_ordering));
   const Operator *VRN(vfes.GetElementRestriction(nat_ordering));
   const Operator *VRL(vfes.GetElementRestriction(lex_ordering));
   MFEM_VERIFY(SRN, "No element sn-restriction operator found!");
   MFEM_VERIFY(SRL, "No element sl-restriction operator found!");
   MFEM_VERIFY(VRN, "No element vn-restriction operator found!");
   MFEM_VERIFY(VRL, "No element vl-restriction operator found!");

   double val[2], der[2], det[2], pdr[2];

   {
      // Scalar
      sqi->SetOutputLayout(q_layout);
      Vector xe(1*ND*NE);
      REQUIRE(xe.Size() == SRN->Height());
      REQUIRE(SRN->Height() == SRL->Height());
      {
         // Full
         SRN->Mult(x, xe);
         sqi->DisableTensorProducts();
         Vector sq_val_f(NQ*NE), sq_der_f(dim*NQ*NE), sq_pdr_f(dim*NQ*NE);

         sqi->Values(xe, sq_val_f);
         val[0] = sq_val_f*sq_val_f;

         sqi->Derivatives(xe, sq_der_f);
         der[0] = sq_der_f*sq_der_f;

         sqi->PhysDerivatives(xe, sq_pdr_f);
         pdr[0] = sq_pdr_f*sq_pdr_f;
      }
      {
         // Tensor
         SRL->Mult(x, xe);
         sqi->EnableTensorProducts();
         Vector sq_val_t(NQ*NE), sq_der_t(dim*NQ*NE), sq_pdr_t(dim*NQ*NE);

         sqi->Values(xe, sq_val_t);
         val[1] = sq_val_t*sq_val_t;

         sqi->Derivatives(xe, sq_der_t);
         der[1] = sq_der_t*sq_der_t;

         sqi->PhysDerivatives(xe, sq_pdr_t);
         pdr[1] = sq_pdr_t*sq_pdr_t;
      }
      REQUIRE(val[0] == MFEM_Approx(val[1]));
      REQUIRE(der[0] == MFEM_Approx(der[1]));
      REQUIRE(pdr[0] == MFEM_Approx(pdr[1]));
   }

   {
      // Vector
      vqi->SetOutputLayout(q_layout);
      Vector ne(vdim*ND*NE);
      REQUIRE(ne.Size() == VRN->Height());
      REQUIRE(VRN->Height() == VRL->Height());
      {
         // Full
         VRN->Mult(nodes, ne);
         vqi->DisableTensorProducts();
         Vector vq_val_f(dim*NQ*NE), vq_der_f(vdim*dim*NQ*NE),
                vq_det_f(NQ*NE),
                vq_pdr_f(vdim*dim*NQ*NE);

         vqi->Values(ne, vq_val_f);
         val[0] = vq_val_f*vq_val_f;

         vqi->Derivatives(ne, vq_der_f);
         der[0] = vq_der_f*vq_der_f;

         vqi->Determinants(ne, vq_det_f);
         det[0] = vq_det_f*vq_det_f;

         vqi->PhysDerivatives(ne, vq_pdr_f);
         pdr[0] = vq_pdr_f*vq_pdr_f;
      }
      {
         // Tensor
         VRL->Mult(nodes, ne);
         vqi->EnableTensorProducts();
         Vector vq_val_t(dim*NQ*NE), vq_der_t(vdim*dim*NQ*NE),
                vq_det_t(NQ*NE),
                vq_pdr_t(vdim*dim*NQ*NE);

         vqi->Values(ne, vq_val_t);
         val[1] = vq_val_t*vq_val_t;

         vqi->Derivatives(ne, vq_der_t);
         der[1] = vq_der_t*vq_der_t;

         vqi->Determinants(ne, vq_det_t);
         det[1] = vq_det_t*vq_det_t;

         vqi->PhysDerivatives(ne, vq_pdr_t);
         pdr[1] = vq_pdr_t*vq_pdr_t;
      }
      REQUIRE(val[0] == MFEM_Approx(val[1]));
      REQUIRE(der[0] == MFEM_Approx(der[1]));
      REQUIRE(det[0] == MFEM_Approx(det[1]));
      REQUIRE(pdr[0] == MFEM_Approx(pdr[1]));
   }

   return true;
}

TEST_CASE("QuadratureInterpolator", "[QuadratureInterpolator]")
{
   const auto d = GENERATE(2,3); // dimension
   const auto p = GENERATE(range(1,8)); // element order
   const auto q = GENERATE_COPY(range(p+1,10)); // quadrature order
   const auto l = GENERATE(QVectorLayout::byNODES, QVectorLayout::byVDIM);
   const auto nx = GENERATE(3,4); // number of element in x
   const auto ny = GENERATE(5,6); // number of element in y
   const auto nz = GENERATE(1,3); // number of element in z
   REQUIRE(testQuadratureInterpolator(d, p, q, l, nx, ny, nz));
} // TEST_CASE "QuadratureInterpolator"
