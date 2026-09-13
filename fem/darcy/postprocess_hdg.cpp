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

namespace
{

/** @brief The flux layout, checked once rather than trusted in an element
    loop. Returns true for an H(div) space, false for a scalar-range one.

    A scalar-range space (L2, H1) holds `neq*dim` components and a block is
    `dim` of them; an H(div) space holds `neq` and a block is one. Getting
    this backwards is how a block would be read past its end. */
bool CheckFluxLayout(const FiniteElementSpace &fes_q,
                     const FiniteElementSpace &fes_p, int neq)
{
   const int dim = fes_p.GetMesh()->Dimension();
   const bool vector_range =
      (fes_q.GetFE(0)->GetRangeType() == FiniteElement::VECTOR);
   const int expect = vector_range ? neq : neq * dim;
   MFEM_VERIFY(fes_q.GetVDim() == expect,
               "the flux space has vdim " << fes_q.GetVDim() << ", expected "
               << expect << " for " << neq << " equation(s) in " << dim
               << "D on a " << (vector_range ? "vector" : "scalar")
               << "-range space");
   return vector_range;
}

} // namespace

HDGPostprocessBlocks::HDGPostprocessBlocks(const FiniteElementSpace &fes_q_,
                                           const FiniteElementSpace &fes_p_,
                                           const FiniteElementSpace &fes_s_)
   : fes_q(&fes_q_), fes_p(&fes_p_), fes_s(&fes_s_)
{
   MFEM_VERIFY(fes_p->GetMesh() == fes_q->GetMesh() &&
               fes_p->GetMesh() == fes_s->GetMesh(),
               "the spaces are on different meshes");
   MFEM_VERIFY(fes_p->GetNE() > 0, "the space has no elements");

   neq = fes_p->GetVDim();
   MFEM_VERIFY(fes_s->GetVDim() == neq,
               "the enriched space has vdim " << fes_s->GetVDim()
               << ", expected " << neq);

   vector_flux = CheckFluxLayout(*fes_q, *fes_p, neq);
}

int HDGPostprocessBlocks::NumFluxDofs(int el) const
{
   if (assembled)
   {
      // Recovered from the offsets rather than from the space, so that
      // Apply() reaches no FiniteElementSpace at all. That is not tidiness:
      // FiniteElementSpace::GetFE() calls NURBSext->LoadFE(), which MUTATES
      // the element it hands back, so a threaded caller must not go near it.
      return (offs_B11[el+1] - offs_B11[el]) / (offs_s[el+1] - offs_s[el]);
   }
   const int nd_q = fes_q->GetFE(el)->GetDof();
   return vector_flux ? nd_q : nd_q * fes_p->GetMesh()->Dimension();
}

bool HDGPostprocessBlocks::IsAssembled() const
{
   return assembled && seq == fes_p->GetMesh()->GetSequence()
          && stamp == stamp_seen;
}

void HDGPostprocessBlocks::CheckFresh() const
{
   MFEM_VERIFY(assembled, "Assemble() has not been called");
   MFEM_VERIFY(seq == fes_p->GetMesh()->GetSequence(),
               "the mesh has changed since Assemble(); the blocks are stale");
   MFEM_VERIFY(stamp == stamp_seen,
               "a diffusion coefficient has moved since Assemble(); "
               "the blocks are stale -- call Assemble() again");
}

void HDGPostprocessBlocks::Assemble()
{
   Mesh *mesh = fes_p->GetMesh();
   const int NE = mesh->GetNE();
   const int dim = mesh->Dimension();

   // Sizes first, so the three arrays are allocated once. They are per
   // element rather than global constants because a variable-order space is
   // free to hand back a different count on every element.
   offs_B11.SetSize(NE + 1);
   offs_s.SetSize(NE + 1);
   offs_p.SetSize(NE + 1);
   offs_B11[0] = offs_s[0] = offs_p[0] = 0;
   for (int z = 0; z < NE; z++)
   {
      const int ns = fes_s->GetFE(z)->GetDof();
      const int nd = fes_p->GetFE(z)->GetDof();
      const int na = NumFluxDofs(z);
      offs_B11[z+1] = offs_B11[z] + ns * na;
      offs_s[z+1] = offs_s[z] + ns;
      offs_p[z+1] = offs_p[z] + nd;
   }
   B11_data.SetSize(offs_B11[NE]);
   c12_data.SetSize(offs_s[NE]);
   mass_data.SetSize(offs_p[NE]);

   IsoparametricTransformation Tr;
   Vector shape_s, shape_p, shape_q, e_ic;
   DenseMatrix dshape_s, A, psi, kpsi, G, Kmat;
   DenseMatrixInverse Ai;

   for (int z = 0; z < NE; z++)
   {
      const FiniteElement *fe_q = fes_q->GetFE(z);
      const FiniteElement *fe_p = fes_p->GetFE(z);
      const FiniteElement *fe_s = fes_s->GetFE(z);
      // The caller-allocated overload: Apply() is meant for a threaded loop
      // and Assemble() has no reason to reach for the shared transformation.
      mesh->GetElementTransformation(z, &Tr);

      const int nd_q = fe_q->GetDof();
      const int nd_p = fe_p->GetDof();
      const int nd_s = fe_s->GetDof();
      const int na = vector_flux ? nd_q : nd_q * dim;

      const int iro = (ir_order >= 0)
                      ? ir_order : (2 * fe_s->GetOrder() + Tr.OrderW());
      const IntegrationRule &ir = IntRules.Get(fe_s->GetGeomType(), iro);

      A.SetSize(nd_s);
      A = 0.0;
      G.SetSize(nd_s, na);
      G = 0.0;
      Vector mass_s(nd_s);
      Vector mass_p(mass_data.GetData() + offs_p[z], nd_p);
      mass_s = 0.0;
      mass_p = 0.0;
      shape_s.SetSize(nd_s);
      shape_p.SetSize(nd_p);
      // CalcPhysDShape() multiplies into its argument without resizing it.
      dshape_s.SetSize(nd_s, dim);
      psi.SetSize(na, dim);
      kpsi.SetSize(na, dim);
      if (!vector_flux) { shape_q.SetSize(nd_q); psi = 0.0; }

      for (int k = 0; k < ir.GetNPoints(); k++)
      {
         const IntegrationPoint &ip = ir.IntPoint(k);
         Tr.SetIntPoint(&ip);
         const real_t w = ip.weight * Tr.Weight();

         fe_s->CalcPhysDShape(Tr, dshape_s);
         AddMult_a_AAt(w, dshape_s, A);

         fe_s->CalcPhysShape(Tr, shape_s);
         mass_s.Add(w, shape_s);
         fe_p->CalcPhysShape(Tr, shape_p);
         mass_p.Add(w, shape_p);

         // The flux BASIS at this point, one row per local flux dof of one
         // equation. This is the only place the two layouts differ, and it is
         // the same distinction the value-wise GetFluxBlock() used to make:
         // an H(div) element is vector valued and a block is one component's
         // worth of dofs, a scalar one carries `dim` scalar components.
         if (vector_flux)
         {
            fe_q->CalcVShape(Tr, psi);
         }
         else
         {
            fe_q->CalcPhysShape(Tr, shape_q);
            for (int d = 0; d < dim; d++)
               for (int j = 0; j < nd_q; j++)
               {
                  psi(d * nd_q + j, d) = shape_q(j);
               }
         }

         // iK applied to each basis function, i.e. kpsi = psi * iK^T.
         if (iK)
         {
            Kmat.SetSize(dim);
            iK->Eval(Kmat, Tr, ip);
            MultABt(psi, Kmat, kpsi);
         }
         else
         {
            kpsi = psi;
            if (ik) { kpsi *= ik->Eval(Tr, ip); }
         }

         // G(i,j) = sum_k w (grad chi_i, iK psi_j). The right-hand side of
         // the local problem is -G u_e: the flux is minus the diffusivity
         // times the gradient, so this is the gradient of the potential, and
         // the sign is what makes the two equations consistent.
         AddMult_a_ABt(w, dshape_s, kpsi, G);
      }

      // Replace one equation of the local system by the mean constraint. The
      // problem is pure Neumann, so without this the matrix is singular; with
      // it the constant is the computed potential's element average, which is
      // where the superconvergence comes from. The same row of the right-hand
      // side belongs to the potential, so it is cleared from the flux block.
      constexpr int i_c = 0;
      A.SetRow(i_c, 0.0);
      for (int j = 0; j < nd_s; j++) { A(i_c, j) = mass_s(j); }
      Ai.Factor(A);
      G.SetRow(i_c, 0.0);

      DenseMatrix B11(B11_data.GetData() + offs_B11[z], nd_s, na);
      Ai.Mult(G, B11);
      B11.Neg();

      // B12 = (A^{-1} e_ic) mass_p^T, so only the column is stored: the rank
      // one property IS the storage, and a full block cannot drift from it.
      e_ic.SetSize(nd_s);
      e_ic = 0.0;
      e_ic(i_c) = 1.0;
      Vector c12(c12_data.GetData() + offs_s[z], nd_s);
      Ai.Mult(e_ic, c12);
   }

   assembled = true;
   seq = mesh->GetSequence();
   stamp_seen = stamp;
}

void HDGPostprocessBlocks::GetBlocks(int el, DenseMatrix &B11,
                                     DenseMatrix &B12) const
{
   CheckFresh();
   const int ns = NumNodes(el), nd = NumPotentialDofs(el);
   const int na = NumFluxDofs(el);

   B11.SetSize(ns, na);
   const DenseMatrix stored(const_cast<real_t*>(B11_data.GetData())
                            + offs_B11[el], ns, na);
   B11 = stored;

   const Vector c12(const_cast<real_t*>(c12_data.GetData()) + offs_s[el], ns);
   const Vector mass(const_cast<real_t*>(mass_data.GetData()) + offs_p[el],
                     nd);
   B12.SetSize(ns, nd);
   MultVWt(c12, mass, B12);
}

void HDGPostprocessBlocks::Apply(int el, const Vector &u_l, const Vector &p_l,
                                 Vector &gamma) const
{
   CheckFresh();
   const int ns = NumNodes(el), nd = NumPotentialDofs(el);
   const int na = NumFluxDofs(el);
   MFEM_ASSERT(u_l.Size() == neq * na, "the flux state is " << u_l.Size()
               << " long, expected " << neq * na);
   MFEM_ASSERT(p_l.Size() == neq * nd, "the potential state is " << p_l.Size()
               << " long, expected " << neq * nd);

   gamma.SetSize(neq * ns);

   const DenseMatrix B11(const_cast<real_t*>(B11_data.GetData())
                         + offs_B11[el], ns, na);
   const Vector c12(const_cast<real_t*>(c12_data.GetData()) + offs_s[el], ns);
   const Vector mass(const_cast<real_t*>(mass_data.GetData()) + offs_p[el],
                     nd);

   for (int e = 0; e < neq; e++)
   {
      const Vector u_e(const_cast<real_t*>(u_l.GetData()) + e * na, na);
      const Vector p_e(const_cast<real_t*>(p_l.GetData()) + e * nd, nd);
      Vector g_e(gamma.GetData() + e * ns, ns);
      B11.Mult(u_e, g_e);
      // The whole potential dependence, and the reason B12 is rank one: the
      // element average is the only thing the local problem is told.
      g_e.Add(mass * p_e, c12);
   }
}

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
   CheckFluxLayout(*fes_q, *fes_p, neq);
}

void HDGPotentialPostprocessor::Compute(GridFunction &p_s) const
{
   const FiniteElementSpace *fes_p = p->FESpace();
   const FiniteElementSpace *fes_q = q->FESpace();
   Mesh *mesh = fes_p->GetMesh();

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
      p_s.SetSpace(s_space);
      p_s.MakeOwner(s_coll);
   }
   const FiniteElementSpace *fes_s = p_s.FESpace();
   MFEM_VERIFY(fes_s->GetVDim() == neq,
               "the postprocessed potential has vdim " << fes_s->GetVDim()
               << ", expected " << neq);

   // One copy of the element algebra, and this is the other caller of it.
   // The blocks are formed and discarded here because Compute() is a
   // once-per-solution query; a caller that evaluates the same map repeatedly
   // -- an interpolatory source term does, every Newton step -- holds the
   // HDGPostprocessBlocks itself and pays the quadrature once.
   HDGPostprocessBlocks blocks(*fes_q, *fes_p, *fes_s);
   if (iK) { blocks.SetDiffusionInverse(*iK); }
   else if (ik) { blocks.SetDiffusionInverse(*ik); }
   blocks.SetIntegrationOrder(ir_order);
   blocks.Assemble();

   Array<int> vdofs_q, vdofs_p, vdofs_s;
   Vector loc_q, loc_p, gamma;

   for (int z = 0; z < mesh->GetNE(); z++)
   {
      fes_q->GetElementVDofs(z, vdofs_q);
      q->GetSubVector(vdofs_q, loc_q);
      fes_p->GetElementVDofs(z, vdofs_p);
      p->GetSubVector(vdofs_p, loc_p);
      fes_s->GetElementVDofs(z, vdofs_s);

      blocks.Apply(z, loc_q, loc_p, gamma);
      p_s.SetSubVector(vdofs_s, gamma);
   }
}

}
