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

#include "postprocess_hdg.hpp"

#ifdef MFEM_USE_MPI
// ParMesh and ParFiniteElementSpace are named in the enriched-space branch
// below; without these the serial build compiles it away and only the MPI
// build notices.
#include "../pgridfunc.hpp"
#endif

namespace mfem
{

HDGPotentialPostprocessor::HDGPotentialPostprocessor(
   const GridFunction &flux, const GridFunction &potential)
   : q(&flux), p(&potential)
{
   const FiniteElementSpace *fes_p = p->FESpace();
   const FiniteElementSpace *fes_q = q->FESpace();
   MFEM_VERIFY(fes_p && fes_q, "a grid function has no finite element space");
   MFEM_VERIFY(fes_p->GetMesh() == fes_q->GetMesh(),
               "the flux and the potential are on different meshes");
   MFEM_VERIFY(fes_p->GetNE() > 0, "the space has no elements");

   neq = fes_p->GetVDim();

   // The flux layout is read from its space, not assumed: a scalar-range space
   // holds neq*dim components, an H(div) one holds neq. Getting this backwards
   // is how a block would be read past its end, so it is checked here once
   // rather than trusted in the element loop.
   const int dim = fes_p->GetMesh()->Dimension();
   const bool vector_range =
      (fes_q->GetFE(0)->GetRangeType() == FiniteElement::VECTOR);
   const int expect = vector_range ? neq : neq * dim;
   MFEM_VERIFY(fes_q->GetVDim() == expect,
               "the flux space has vdim " << fes_q->GetVDim() << ", expected "
               << expect << " for " << neq << " equation(s) in " << dim
               << "D on a " << (vector_range ? "vector" : "scalar")
               << "-range space");
}

void HDGPotentialPostprocessor::GetFluxBlock(
   const FiniteElement &fe, ElementTransformation &T,
   const IntegrationPoint &ip, const Vector &loc_q, int e, Vector &q_e) const
{
   const int dim = T.GetSpaceDim();
   const int ndof = fe.GetDof();
   q_e.SetSize(dim);

   if (fe.GetRangeType() == FiniteElement::VECTOR)
   {
      // H(div): the element is vector valued, so block e is one scalar
      // component -- ndof coefficients against the vector shape.
      DenseMatrix vshape(ndof, dim);
      fe.CalcVShape(T, vshape);
      const Vector blk(const_cast<real_t*>(loc_q.GetData()) + e * ndof, ndof);
      vshape.MultTranspose(blk, q_e);
   }
   else
   {
      // Scalar range: the element is scalar and block e is dim components of
      // vdim, each with its own ndof coefficients.
      Vector shape(ndof);
      fe.CalcPhysShape(T, shape);
      for (int d = 0; d < dim; d++)
      {
         const Vector cmp(const_cast<real_t*>(loc_q.GetData())
                          + (e * dim + d) * ndof, ndof);
         q_e(d) = cmp * shape;
      }
   }
}

void HDGPotentialPostprocessor::Compute(GridFunction &p_s) const
{
   const FiniteElementSpace *fes_p = p->FESpace();
   const FiniteElementSpace *fes_q = q->FESpace();
   Mesh *mesh = fes_p->GetMesh();
   const int dim = mesh->Dimension();

   // The enriched space, if the caller did not supply one.
   if (!p_s.FESpace())
   {
      const FiniteElementCollection *coll = fes_p->FEColl();
      FiniteElementCollection *s_coll = coll->Clone(coll->GetOrder() + 1);
      FiniteElementSpace *s_space;
#ifdef MFEM_USE_MPI
      ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
      if (pmesh)
      {
         s_space = new ParFiniteElementSpace(pmesh, s_coll, neq);
      }
      else
#endif
      {
         s_space = new FiniteElementSpace(mesh, s_coll, neq);
      }

      // One order above the potential *element by element*, not one order
      // above the collection, when the potential carries a degree per
      // element. Everything below already reads GetFE(z) from all three
      // spaces, so this is the only place the enrichment could have gone
      // uniform -- and a uniform enrichment over a p-adapted potential
      // silently postprocesses most elements at the wrong degree.
      if (fes_p->IsVariableOrder())
      {
         for (int z = 0; z < mesh->GetNE(); z++)
         {
            s_space->SetElementOrder(z, fes_p->GetElementOrder(z) + 1);
         }
         s_space->Update(false);
      }

      p_s.SetSpace(s_space);
      p_s.MakeOwner(s_coll);
   }
   const FiniteElementSpace *fes_s = p_s.FESpace();
   MFEM_VERIFY(fes_s->GetVDim() == neq,
               "the postprocessed potential has vdim " << fes_s->GetVDim()
               << ", expected " << neq);

   Array<int> vdofs_q, vdofs_p, vdofs_s;
   Vector loc_q, loc_p, q_e, shape_p, shape_s, rhs, sol;
   DenseMatrix dshape_s, A;
   DenseMatrixInverse Ai;

   for (int z = 0; z < mesh->GetNE(); z++)
   {
      const FiniteElement *fe_q = fes_q->GetFE(z);
      const FiniteElement *fe_p = fes_p->GetFE(z);
      const FiniteElement *fe_s = fes_s->GetFE(z);
      ElementTransformation *T = mesh->GetElementTransformation(z);

      const int nd_p = fe_p->GetDof();
      const int nd_s = fe_s->GetDof();

      fes_q->GetElementVDofs(z, vdofs_q);
      q->GetSubVector(vdofs_q, loc_q);
      fes_p->GetElementVDofs(z, vdofs_p);
      p->GetSubVector(vdofs_p, loc_p);
      fes_s->GetElementVDofs(z, vdofs_s);

      const int iro = (ir_order >= 0)
                      ? ir_order : (2 * fe_s->GetOrder() + T->OrderW());
      const IntegrationRule &ir = IntRules.Get(fe_s->GetGeomType(), iro);

      // The stiffness matrix and the mean row are the same for every equation
      // -- only the right-hand side differs -- so they are built once and the
      // factorisation is reused across the blocks.
      A.SetSize(nd_s);
      A = 0.0;
      Vector mass_s(nd_s), mass_p(nd_p);
      mass_s = 0.0;
      mass_p = 0.0;
      shape_s.SetSize(nd_s);
      shape_p.SetSize(nd_p);
      // CalcPhysDShape() multiplies into its argument without resizing it.
      dshape_s.SetSize(nd_s, dim);

      DenseMatrix rhs_all(nd_s, neq);
      rhs_all = 0.0;

      for (int k = 0; k < ir.GetNPoints(); k++)
      {
         const IntegrationPoint &ip = ir.IntPoint(k);
         T->SetIntPoint(&ip);
         const real_t w = ip.weight * T->Weight();

         fe_s->CalcPhysDShape(*T, dshape_s);
         AddMult_a_AAt(w, dshape_s, A);

         fe_s->CalcPhysShape(*T, shape_s);
         mass_s.Add(w, shape_s);
         fe_p->CalcPhysShape(*T, shape_p);
         mass_p.Add(w, shape_p);

         // -(iK q_e, grad v): the flux is minus the diffusivity times the
         // gradient, so this is the gradient of the potential, and the sign
         // is what makes the two equations of eq (25) consistent.
         Vector giq(dim);
         for (int e = 0; e < neq; e++)
         {
            GetFluxBlock(*fe_q, *T, ip, loc_q, e, q_e);

            if (iK)
            {
               DenseMatrix M(dim);
               iK->Eval(M, *T, ip);
               M.Mult(q_e, giq);
            }
            else
            {
               giq = q_e;
               if (ik) { giq *= ik->Eval(*T, ip); }
            }

            for (int i = 0; i < nd_s; i++)
            {
               real_t s = 0.0;
               for (int d = 0; d < dim; d++) { s += dshape_s(i, d) * giq(d); }
               rhs_all(i, e) -= w * s;
            }
         }
      }

      // Replace one equation of the local system by the mean constraint. The
      // problem is pure Neumann, so without this the matrix is singular; with
      // it the constant is the computed potential's element average, which is
      // where the superconvergence comes from.
      constexpr int i_c = 0;
      A.SetRow(i_c, 0.0);
      for (int j = 0; j < nd_s; j++) { A(i_c, j) = mass_s(j); }
      Ai.Factor(A);

      for (int e = 0; e < neq; e++)
      {
         rhs.SetSize(nd_s);
         for (int i = 0; i < nd_s; i++) { rhs(i) = rhs_all(i, e); }

         const Vector p_e(loc_p.GetData() + e * nd_p, nd_p);
         rhs(i_c) = mass_p * p_e;

         sol.SetSize(nd_s);
         Ai.Mult(rhs, sol);

         const Array<int> dofs_e(const_cast<int*>(vdofs_s.GetData()) + e * nd_s,
                                 nd_s);
         p_s.SetSubVector(dofs_e, sol);
      }
   }
}

}
