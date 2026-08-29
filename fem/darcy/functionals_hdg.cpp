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

#include "functionals_hdg.hpp"

namespace mfem
{

/// Integrate ut.n over one face, with the normal pointing from Elem1 to Elem2.
/** Returns the integral with no sign applied; the callers below decide the
    orientation. The normal comes from CalcOrtho(), which scales it by the face
    Jacobian, so the integration weight is the reference one and no separate
    face measure enters -- getting that wrong is a factor of the face area. */
static real_t FaceNormalFlux(const GridFunction &ut,
                             FaceElementTransformations &FTr, int ir_order)
{
   const int dim = FTr.GetSpaceDim();
   const IntegrationRule &ir = IntRules.Get(FTr.GetGeometryType(), ir_order);

   Vector nor(dim), val(dim);
   real_t sum = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      FTr.SetAllIntPoints(&ip);

      if (dim == 1)
      {
         nor(0) = 2.0 * FTr.GetElement1IntPoint().x - 1.0;
      }
      else
      {
         CalcOrtho(FTr.Jacobian(), nor);
      }

      // Read the flux from the Elem1 side. For a normally continuous ut the
      // side does not matter, and that is exactly the property being used.
      ut.GetVectorValue(*FTr.Elem1, FTr.GetElement1IntPoint(), val);

      sum += ip.weight * (val * nor);
   }
   return sum;
}

/// The quadrature order to integrate ut.n at, when the caller does not say.
static int DefaultIROrder(const GridFunction &ut, int ir_order)
{
   if (ir_order >= 0) { return ir_order; }
   const FiniteElementSpace *fes = ut.FESpace();
   return 2 * fes->GetMaxElementOrder() + 2;
}

real_t ComputeOutwardFlux(const GridFunction &ut, const Array<int> &elem_marker,
                          int ir_order)
{
   const FiniteElementSpace *fes = ut.FESpace();
   MFEM_VERIFY(fes, "the total flux has no finite element space");
   Mesh *mesh = fes->GetMesh();
   MFEM_VERIFY(elem_marker.Size() == mesh->GetNE(),
               "elem_marker must have one entry per element, got "
               << elem_marker.Size() << " for " << mesh->GetNE());

   const int iro = DefaultIROrder(ut, ir_order);

   real_t total = 0.0;
   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      int e1, e2;
      mesh->GetFaceElements(f, &e1, &e2);

      // Outside the mesh counts as outside the subdomain, which is what makes
      // the identity hold when the subdomain touches the mesh boundary.
      const bool in1 = (elem_marker[e1] != 0);
      const bool in2 = (e2 >= 0) ? (elem_marker[e2] != 0) : false;
      if (in1 == in2) { continue; }   // interior to the subdomain, or outside it

      FaceElementTransformations *FTr =
         mesh->GetFaceElementTransformations(f);
      if (!FTr) { continue; }

      // CalcOrtho() orients the normal from Elem1 towards Elem2, so it already
      // points out of the subdomain when Elem1 is the marked side.
      const real_t sgn = in1 ? 1.0 : -1.0;
      total += sgn * FaceNormalFlux(ut, *FTr, iro);
   }
   return total;
}

real_t ComputeBoundaryFlux(const GridFunction &ut,
                           const Array<int> &bdr_attr_marker, int ir_order)
{
   const FiniteElementSpace *fes = ut.FESpace();
   MFEM_VERIFY(fes, "the total flux has no finite element space");
   Mesh *mesh = fes->GetMesh();

   const int iro = DefaultIROrder(ut, ir_order);

   real_t total = 0.0;
   for (int b = 0; b < mesh->GetNBE(); b++)
   {
      const int attr = mesh->GetBdrAttribute(b);
      MFEM_VERIFY(attr <= bdr_attr_marker.Size(),
                  "boundary attribute " << attr << " exceeds the marker size "
                  << bdr_attr_marker.Size());
      if (bdr_attr_marker[attr-1] == 0) { continue; }

      FaceElementTransformations *FTr =
         mesh->GetBdrFaceTransformations(b);
      if (!FTr) { continue; }   // not a true boundary face on this rank

      // A boundary face has only Elem1, so CalcOrtho() already points out of
      // the domain.
      total += FaceNormalFlux(ut, *FTr, iro);
   }
   return total;
}

}
