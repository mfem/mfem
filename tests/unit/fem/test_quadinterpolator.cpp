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
                                       const int qpts,
                                       const QVectorLayout q_layout,
                                       const int nx, const int ny, const int nz)
{
   // Keep for debugging purposes:
   const bool verbose = false;
   if (verbose)
   {
      std::cout << "testQuadratureInterpolator(dim=" << dim
                << ",p=" << p
                << ",q=" << qpts
                << ",l=" << (q_layout == QVectorLayout::byNODES ?
                             "by_nodes" : "by_vdim")
                << ",nx=" << nx
                << ",ny=" << ny
                << ",nz=" << nz
                << ")" << std::endl;
   }

   const int vdim = dim;
   const int seed = 0x100001b3;
   const int ordering = Ordering::byNODES;

   REQUIRE((dim == 2 || dim == 3));
   Mesh mesh = dim == 1 ? Mesh::MakeCartesian1D(nx, Element::SEGMENT) :
               dim == 2 ? Mesh::MakeCartesian2D(nx,ny, Element::QUADRILATERAL):
               Mesh::MakeCartesian3D(nx,nx,nz, Element::HEXAHEDRON);

   const H1_FECollection fec(p, dim);
   FiniteElementSpace sfes(&mesh, &fec, 1, ordering);
   FiniteElementSpace vfes(&mesh, &fec, vdim, ordering);

   GridFunction x(&sfes);
   x.Randomize(seed);

   GridFunction nodes(&vfes);
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
            rdm(vfes.DofToVDof(i,d)) *= (0.25/p)*h0(i);
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
   const IntegrationRule &ir = IntRules.Get(GeomType, 2*qpts-1);
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

   const double rel_tol = 1e-12;

   {
      // Scalar
      sqi->SetOutputLayout(q_layout);
      Vector xe(1*ND*NE);
      REQUIRE(xe.Size() == SRN->Height());
      REQUIRE(SRN->Height() == SRL->Height());
      // Full results
      Vector sq_val_f(NQ*NE), sq_der_f(dim*NQ*NE), sq_pdr_f(dim*NQ*NE);
      // Tensor results
      Vector sq_val_t(NQ*NE), sq_der_t(dim*NQ*NE), sq_pdr_t(dim*NQ*NE);
      {
         // Full
         SRN->Mult(x, xe);
         sqi->DisableTensorProducts();

         sqi->Values(xe, sq_val_f);

         sqi->Derivatives(xe, sq_der_f);

         sqi->PhysDerivatives(xe, sq_pdr_f);
      }
      {
         // Tensor
         SRL->Mult(x, xe);
         sqi->EnableTensorProducts();

         sqi->Values(xe, sq_val_t);

         sqi->Derivatives(xe, sq_der_t);

         sqi->PhysDerivatives(xe, sq_pdr_t);
      }
      double norm, rel_error;

      norm = sq_val_f.Normlinf();
      sq_val_f -= sq_val_t;
      rel_error = sq_val_f.Normlinf()/norm;
      if (verbose)
      { std::cout << "sq_val rel. error = " << rel_error << std::endl; }
      REQUIRE(rel_error <= rel_tol);

      norm = sq_der_f.Normlinf();
      sq_der_f -= sq_der_t;
      rel_error = sq_der_f.Normlinf()/norm;
      if (verbose)
      { std::cout << "sq_der rel. error = " << rel_error << std::endl; }
      REQUIRE(rel_error <= rel_tol);

      norm = sq_pdr_f.Normlinf();
      sq_pdr_f -= sq_pdr_t;
      rel_error = sq_pdr_f.Normlinf()/norm;
      if (verbose)
      { std::cout << "sq_pdr rel. error = " << rel_error << std::endl; }
      REQUIRE(rel_error <= rel_tol);
   }

   {
      // Vector
      vqi->SetOutputLayout(q_layout);
      Vector ne(vdim*ND*NE);
      REQUIRE(ne.Size() == VRN->Height());
      REQUIRE(VRN->Height() == VRL->Height());
      // Full results
      Vector vq_val_f(dim*NQ*NE), vq_der_f(vdim*dim*NQ*NE),
             vq_det_f(NQ*NE),
             vq_pdr_f(vdim*dim*NQ*NE);
      // Tensor results
      Vector vq_val_t(dim*NQ*NE), vq_der_t(vdim*dim*NQ*NE),
             vq_det_t(NQ*NE),
             vq_pdr_t(vdim*dim*NQ*NE);
      {
         // Full
         VRN->Mult(nodes, ne);
         vqi->DisableTensorProducts();

         vqi->Values(ne, vq_val_f);

         vqi->Derivatives(ne, vq_der_f);

         vqi->Determinants(ne, vq_det_f);

         vqi->PhysDerivatives(ne, vq_pdr_f);
      }
      {
         // Tensor
         VRL->Mult(nodes, ne);
         vqi->EnableTensorProducts();

         vqi->Values(ne, vq_val_t);

         vqi->Derivatives(ne, vq_der_t);

         vqi->Determinants(ne, vq_det_t);

         vqi->PhysDerivatives(ne, vq_pdr_t);
      }
      double norm, rel_error;

      norm = vq_val_f.Normlinf();
      vq_val_f -= vq_val_t;
      rel_error = vq_val_f.Normlinf()/norm;
      if (verbose)
      { std::cout << "vq_val rel. error = " << rel_error << std::endl; }
      REQUIRE(rel_error <= rel_tol);

      norm = vq_der_f.Normlinf();
      vq_der_f -= vq_der_t;
      rel_error = vq_der_f.Normlinf()/norm;
      if (verbose)
      { std::cout << "vq_der rel. error = " << rel_error << std::endl; }
      REQUIRE(rel_error <= rel_tol);

      norm = vq_det_f.Normlinf();
      vq_det_f -= vq_det_t;
      rel_error = vq_det_f.Normlinf()/norm;
      if (verbose)
      { std::cout << "vq_det rel. error = " << rel_error << std::endl; }
      REQUIRE(rel_error <= rel_tol);

      norm = vq_pdr_f.Normlinf();
      vq_pdr_f -= vq_pdr_t;
      rel_error = vq_pdr_f.Normlinf()/norm;
      if (verbose)
      { std::cout << "vq_pdr rel. error = " << rel_error << std::endl; }
      REQUIRE(rel_error <= rel_tol);
   }

   return true;
}

TEST_CASE("QuadratureInterpolator", "[QuadratureInterpolator]")
{
   const auto d = GENERATE(2,3); // dimension
   const auto p = GENERATE(range(1,7)); // element order, 1 <= p < 7
   const auto q = GENERATE_COPY(p+1,p+2); // 1D quadrature points
   const auto l = GENERATE(QVectorLayout::byNODES, QVectorLayout::byVDIM);
   const auto nx = 3; // number of element in x
   const auto ny = 3; // number of element in y
   const auto nz = 3; // number of element in z
   testQuadratureInterpolator(d, p, q, l, nx, ny, nz);
} // TEST_CASE "QuadratureInterpolator"
