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
#include "fem/qinterp/dispatch.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::internal::quadrature_interpolator;

static void dummyfunction(const Vector &x, Vector &p)
{
   for (int i = 0; i < p.Size(); i++)
   {
      p(i) = 0.0;
      for (int j = 0; j < x.Size(); j++)
      {
         p(i) += std::pow(x(j), i+1.0);
      }
   }
}

TEST_CASE("CollocatedLagrangeDerivatives", "[CollocatedLagrangeDerivatives]")
{
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
   // int p = GENERATE(range(1,2)); // element order, 1 <= p < 7
   int vdim = GENERATE(1,2,3); // vector dimension for grid-function

   const int seed = 0x100001b3;
   Mesh mesh = Mesh::LoadFromFile(mesh_fname);
   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();

   // Keep for debugging purposes:
   if (verbose_tests)
   {
      std::cout << "testCollocatedLagrangeDerivatives(mesh=" << mesh_fname
                << ",dim=" << dim
                << ",sdim=" << sdim
                << ",p=" << p
                << ",vdim=" << vdim
                << ")" << std::endl;
   }

   int nelem = mesh.GetNE();

   const H1_FECollection fec(p, dim);
   FiniteElementSpace fes(&mesh, &fec, vdim);
   FiniteElementSpace nfes(&mesh, &fec, sdim);

   GridFunction x(&fes);
   VectorFunctionCoefficient gfc(vdim, dummyfunction);
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
   IntegrationRule ir = irnodes.Permute(irordering);

   int nqp = ir.GetNPoints();
   const DofToQuad maps = fe.GetDofToQuad(ir, DofToQuad::TENSOR);
   auto geom = mesh.GetGeometricFactors(ir, GeometricFactors::JACOBIANS);

   Vector evec_values;
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *n0_R = fes.GetElementRestriction(ordering);
   evec_values.SetSize(n0_R->Height());
   n0_R->Mult(x, evec_values);

   /// Check gradient
   /// Use quadrature kernels
   Vector qp_der(nelem*vdim*nqp*dim);
   TensorDerivatives<QVectorLayout::byNODES>(nelem, vdim, maps, evec_values,
                                             qp_der);

   /// Use collocated point kernels
   Vector col_der(nelem*vdim*nqp*dim);
   CollocatedTensorDerivatives<QVectorLayout::byNODES>(nelem, vdim, maps,
                                                       evec_values, col_der);
   qp_der -= col_der;
   REQUIRE(qp_der.Normlinf() == MFEM_Approx(0.0, 1e-10, 1e-10));

   /// Check gradient
   /// Use quadrature kernels
   Vector qp_phys_der(nelem*vdim*nqp*sdim);
   TensorPhysDerivatives<QVectorLayout::byNODES>(nelem, vdim, maps, *geom,
                                                 evec_values, qp_phys_der);

   /// Use collocated point kernels
   Vector col_phys_der(nelem*vdim*nqp*sdim);
   CollocatedTensorPhysDerivatives<QVectorLayout::byNODES>(nelem, vdim, maps,
                                        *geom, evec_values, col_phys_der);
   qp_phys_der -= col_phys_der;
   REQUIRE(qp_phys_der.Normlinf() == MFEM_Approx(0.0, 1e-10, 1e-10));

   /// Check but now use byVDIM layout
   /// Gradient
   TensorDerivatives<QVectorLayout::byVDIM>(nelem, vdim, maps, evec_values,
                                             qp_der);
   CollocatedTensorDerivatives<QVectorLayout::byVDIM>(nelem, vdim, maps,
                                                       evec_values, col_der);
   qp_der -= col_der;
   REQUIRE(qp_der.Normlinf() == MFEM_Approx(0.0, 1e-10, 1e-10));

   /// Physical gradient
   TensorPhysDerivatives<QVectorLayout::byVDIM>(nelem, vdim, maps, *geom,
                                                 evec_values, qp_phys_der);
   CollocatedTensorPhysDerivatives<QVectorLayout::byVDIM>(nelem, vdim, maps,
                                        *geom, evec_values, col_phys_der);
   qp_phys_der -= col_phys_der;
   REQUIRE(qp_phys_der.Normlinf() == MFEM_Approx(0.0, 1e-10, 1e-10));
}
