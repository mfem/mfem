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

/// Add the integral of ut_e.n over one face, per field e, into @a sums.
/** @a sgn orients the result; the callers below decide it. The normal comes
    from CalcOrtho(), which scales it by the face Jacobian, so the integration
    weight is the reference one and no separate face measure enters -- getting
    that wrong is a factor of the face area.

    The flux is read field by field rather than through
    GridFunction::GetVectorValue(), which cannot do it: its H(div) branch does
    vshape.MultTranspose(loc_data, val) with vshape of GetDof() rows against a
    loc_data of GetDof()*vdim, so it consumes field 0's block and returns
    field 0's flux alone. Here vshape is applied to each field's block in turn.
    The blocks are field-outermost, which is what GetElementVDofs() produces
    locally whatever the space's Ordering is -- the same layout the
    reconstruction uses, and asserted under both orderings by the test. */
static void AddFaceNormalFlux(const GridFunction &ut,
                              FaceElementTransformations &FTr, int ir_order,
                              real_t sgn, Vector &sums)
{
   const FiniteElementSpace &fes = *ut.FESpace();
   const int dim = FTr.GetSpaceDim();
   const int neq = fes.GetVDim();
   const IntegrationRule &ir = IntRules.Get(FTr.GetGeometryType(), ir_order);

   // Read from the Elem1 side. For a normally continuous ut the side does not
   // matter, and that is exactly the property being used.
   const int el = FTr.Elem1No;
   const FiniteElement *fe = fes.GetFE(el);
   const int nd = fe->GetDof();

   Array<int> vdofs;
   DofTransformation doftrans;
   fes.GetElementVDofs(el, vdofs, doftrans);
   MFEM_VERIFY(neq == 1 || doftrans.IsIdentity(),
               "a total flux of " << neq << " fields on a space with a "
               "non-trivial DofTransformation is not supported: the element "
               "blocks are field-outermost and InvTransformPrimal() reads "
               "byVDIM data as interleaved, and the two have not been "
               "reconciled");

   Vector loc;
   ut.GetSubVector(vdofs, loc);
   doftrans.InvTransformPrimal(loc);

   DenseMatrix vshape(nd, dim);
   Vector nor(dim), val(dim);
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

      // SetAllIntPoints() has already put Elem1 at the matching point.
      fe->CalcVShape(*FTr.Elem1, vshape);

      for (int e = 0; e < neq; e++)
      {
         const Vector blk(loc.GetData() + e * nd, nd);
         vshape.MultTranspose(blk, val);
         sums(e) += sgn * ip.weight * (val * nor);
      }
   }
}

/** @brief Refuse a system from an overload that returns a single number.

    The per-field overloads take a Vector and are what a system wants; these
    two return one real_t and there is no honest thing for that to be when
    there are several fields. Returning field 0's would be the silent wrong
    answer this refusal exists to prevent, so it stays loud even now that the
    capability is beside it. */
static void VerifyOneField(const FiniteElementSpace *fes)
{
   MFEM_VERIFY(fes->GetVDim() == 1,
               "the total flux carries " << fes->GetVDim() << " fields; this "
               "overload returns one number and would have to pick a field. "
               "Use the overload taking a Vector, which fills one value per "
               "field");
}

/// The shared preamble of all four entry points.
static const FiniteElementSpace &FluxSpace(const GridFunction &ut)
{
   const FiniteElementSpace *fes = ut.FESpace();
   MFEM_VERIFY(fes, "the total flux has no finite element space");
   // A normally continuous flux is what these functionals are stated for, and
   // it is what the arithmetic needs: a scalar-range space would make
   // CalcVShape() meaningless here. The previous implementation required this
   // too, through GetVectorValue(), but only by producing a size mismatch.
   MFEM_VERIFY(fes->FEColl()->GetRangeType(fes->GetMesh()->Dimension()) ==
               FiniteElement::VECTOR,
               "the total flux must live in a vector-range (H(div)) space; "
               "DarcyForm::ReconstructTotalFlux() builds one");
   return *fes;
}

/// The quadrature order to integrate ut.n at, when the caller does not say.
static int DefaultIROrder(const GridFunction &ut, int ir_order)
{
   if (ir_order >= 0) { return ir_order; }
   const FiniteElementSpace *fes = ut.FESpace();
   return 2 * fes->GetMaxElementOrder() + 2;
}

void ComputeOutwardFlux(const GridFunction &ut, const Array<int> &elem_marker,
                        Vector &flux, int ir_order)
{
   const FiniteElementSpace &fes = FluxSpace(ut);
   Mesh *mesh = fes.GetMesh();
   MFEM_VERIFY(elem_marker.Size() == mesh->GetNE(),
               "elem_marker must have one entry per element, got "
               << elem_marker.Size() << " for " << mesh->GetNE());

   const int iro = DefaultIROrder(ut, ir_order);

   flux.SetSize(fes.GetVDim());
   flux = 0.0;
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
      AddFaceNormalFlux(ut, *FTr, iro, in1 ? 1.0 : -1.0, flux);
   }
}

real_t ComputeOutwardFlux(const GridFunction &ut, const Array<int> &elem_marker,
                          int ir_order)
{
   VerifyOneField(ut.FESpace());
   Vector flux;
   ComputeOutwardFlux(ut, elem_marker, flux, ir_order);
   return flux(0);
}

void ComputeBoundaryFlux(const GridFunction &ut,
                         const Array<int> &bdr_attr_marker,
                         Vector &flux, int ir_order)
{
   const FiniteElementSpace &fes = FluxSpace(ut);
   Mesh *mesh = fes.GetMesh();

   const int iro = DefaultIROrder(ut, ir_order);

   flux.SetSize(fes.GetVDim());
   flux = 0.0;
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
      AddFaceNormalFlux(ut, *FTr, iro, 1.0, flux);
   }
}

real_t ComputeBoundaryFlux(const GridFunction &ut,
                           const Array<int> &bdr_attr_marker, int ir_order)
{
   VerifyOneField(ut.FESpace());
   Vector flux;
   ComputeBoundaryFlux(ut, bdr_attr_marker, flux, ir_order);
   return flux(0);
}

}
