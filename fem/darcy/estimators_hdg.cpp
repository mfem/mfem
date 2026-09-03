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

#include "estimators_hdg.hpp"
#include "darcyhybridization.hpp"

namespace mfem
{

void HDGErrorEstimator::ComputeEstimates()
{
   Mesh *mesh = sol_tr.FESpace()->GetMesh();
   const int dim = mesh->Dimension();
   const int NE = mesh->GetNE();

   error_estimates.SetSize(NE);
   error_estimates = 0.;

   Vector d_error_estimates;
   if (anisotropic && type == Type::Energy)
   {
      d_error_estimates.SetSize(NE * dim);
      d_error_estimates = 0.;
   }

   const int num_faces = mesh->GetNumFaces();

   for (int f = 0; f < num_faces; f++)
   {
      if (!mesh->FaceIsInterior(f)) { continue; }

      ComputeFaceEstimate(f, true, d_error_estimates);
   }

#ifdef MFEM_USE_MPI
   if (psol_tr)
   {
      const ParMesh *pmesh = psol_tr->ParFESpace()->GetParMesh();
      const int num_shared = pmesh->GetNSharedFaces();
      const_cast<ParGridFunction*>(psol_tr)->ExchangeFaceNbrData();

      for (int sf = 0; sf < num_shared; sf++)
      {
         const int sh_face = pmesh->GetSharedFace(sf);

         ComputeFaceEstimate(sh_face, false, d_error_estimates);
      }
   }
#endif

   const int num_nbe = mesh->GetNBE();
   for (int b = 0; b < num_nbe; b++)
   {
      if (excl_bdr.Size() > 0)
      {
         const int attr = mesh->GetBdrAttribute(b);
         if (attr <= excl_bdr.Size() && excl_bdr[attr-1] != 0) { continue; }
      }

      const int bdr_face = mesh->GetBdrElementFaceIndex(b);

      ComputeFaceEstimate(bdr_face, false, d_error_estimates);
   }

   total_error = error_estimates.Sum();
#ifdef MFEM_USE_MPI
   if (psol_tr)
   {
      MPI_Allreduce(MPI_IN_PLACE, &total_error, 1, MFEM_MPI_REAL_T, MPI_SUM,
                    psol_tr->ParFESpace()->GetComm());
   }
#endif

   if (type == Type::Energy)
   {
      for (int i = 0; i < NE; i++)
      {
         error_estimates(i) = std::sqrt(error_estimates(i));
      }
      total_error = std::sqrt(total_error);

      if (anisotropic)
      {
         aniso_flags.SetSize(NE);

         for (int i = 0; i < NE; i++)
         {
            const Vector d_en(d_error_estimates, i*dim, dim);
            const real_t en = d_en.Sum();

            // Note the flags are used to set the refinement type, which
            // assumes the element to be aligned with the coordinate axes
            // TODO: reorientation with the element
            const real_t thresh = 0.15 * 3.0/dim;
            int flag = 0;
            for (int k = 0; k < dim; k++)
            {
               if (d_en[k] > thresh * en) { flag |= (1 << k); }
            }

            aniso_flags[i] = flag;
         }
      }
   }

   current_sequence = sol_tr.FESpace()->GetMesh()->GetSequence();
}

void HDGErrorEstimator::ProjectOntoTrace(const FiniteElement &fe_tr,
                                         const FiniteElement &el,
                                         FaceElementTransformations &FTr,
                                         int side, const Vector &elfun,
                                         Vector &c)
{
   const int nt = fe_tr.GetDof();
   DenseMatrix M(nt);
   Vector b(nt), tr_shape(nt), el_shape(el.GetDof());
   M = 0.;
   b = 0.;

   // Degree 2*max(element, trace), the rule the HDG face integrators take --
   // the mass matrix is of degree 2*p_trace and a rule that cannot reach it
   // returns one that is rank-deficient, which is the same counting argument
   // as the note at the top of bilininteg_hdg.cpp.
   const int order = 2*std::max(el.GetOrder(), fe_tr.GetOrder());
   const IntegrationRule &ir = IntRules.Get(FTr.GetGeometryType(), order);

   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      FTr.SetAllIntPoints(&ip);

      fe_tr.CalcShape(ip, tr_shape);
      ElementTransformation *ElTr = (side != 0) ? FTr.Elem2 : FTr.Elem1;
      el.CalcPhysShape(*ElTr, el_shape);

      const real_t w = ip.weight * FTr.Weight();
      AddMult_a_VVt(w, tr_shape, M);
      b.Add(w * (el_shape * elfun), tr_shape);
   }

   DenseMatrixInverse Mi(M);
   c.SetSize(nt);
   Mi.Mult(b, c);
}

void HDGErrorEstimator::ProjectTraceDown(const FiniteElement &fe_hi,
                                         const FiniteElement &fe_lo,
                                         FaceElementTransformations &FTr,
                                         const Vector &tr_hi, Vector &tr_lo)
{
   const int n_hi = fe_hi.GetDof(), n_lo = fe_lo.GetDof();
   DenseMatrix M(n_lo);
   Vector b(n_lo), sh_hi(n_hi), sh_lo(n_lo);
   M = 0.;
   b = 0.;

   const int order = 2*std::max(fe_hi.GetOrder(), fe_lo.GetOrder());
   const IntegrationRule &ir = IntRules.Get(FTr.GetGeometryType(), order);

   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      FTr.SetAllIntPoints(&ip);
      fe_hi.CalcShape(ip, sh_hi);
      fe_lo.CalcShape(ip, sh_lo);

      const real_t w = ip.weight * FTr.Weight();
      AddMult_a_VVt(w, sh_lo, M);
      b.Add(w * (sh_hi * tr_hi), sh_lo);
   }

   DenseMatrixInverse Mi(M);
   tr_lo.SetSize(n_lo);
   Mi.Mult(b, tr_lo);
}

void HDGErrorEstimator::ComputeFaceEstimate(int face, bool side2,
                                            Vector &d_error_estimates)
{
   const FiniteElementSpace *fes_tr = sol_tr.FESpace();
   const FiniteElementSpace *fes_p = sol_p.FESpace();
   Mesh *mesh = fes_tr->GetMesh();
   const int dim = mesh->Dimension();
   Array<int> vdofs1, vdofs2, vdofs_tr;
   Vector p1, p2, tr, btr1, btr2;

   FaceElementTransformations &FTr = *mesh->GetFaceElementTransformations(face,
                                                                          side2 ? 31 : 21);

   fes_p->GetElementVDofs(FTr.Elem1No, vdofs1);
   sol_p.GetSubVector(vdofs1, p1);
   if (FTr.Elem2No >= 0)
   {
      fes_p->GetElementVDofs(FTr.Elem2No, vdofs2);
      sol_p.GetSubVector(vdofs2, p2);
   }

   const FiniteElement *fe_tr;
#ifdef MFEM_USE_MPI
   const int nfaces = mesh->GetNumFaces();
   if (psol_tr && face >= nfaces)
   {
      const ParFiniteElementSpace *pfes_tr = psol_tr->ParFESpace();
      fe_tr =  pfes_tr->GetFaceNbrFaceFE(face);
      pfes_tr->GetFaceNbrFaceVDofs(face, vdofs_tr);
      psol_tr->FaceNbrData().GetSubVector(vdofs_tr, tr);
   }
   else
#endif
   {
      // Through the hybridization when there is one, because a per-face trace
      // degree lives there and nowhere else -- the constraint space is uniform
      // at the ceiling whatever the faces carry.
      if (hyb)
      {
         fe_tr = hyb->TraceFE(face);
         hyb->TraceVDofs(face, vdofs_tr);
      }
      else
      {
         fe_tr = fes_tr->GetFaceElement(face);
         fes_tr->GetFaceVDofs(face, vdofs_tr);
      }
      sol_tr.GetSubVector(vdofs_tr, tr);
   }

   const FiniteElement &fe1 = *fes_p->GetFE(FTr.Elem1No);
   const FiniteElement &fe2 = (FTr.Elem2No >= 0)?(*fes_p->GetFE(FTr.Elem2No)):
                              (fe1);

   /* THE FACE'S DEGREE IS NOT ITS ELEMENT'S ORDER, and the two questions
      below are about the degree.

      The trace is stored in the constraint space's basis, which is uniform at
      the ceiling, and constrained to the face's own degree; so fe_tr is the
      ceiling's element on every face and reading a degree off it would call
      every face enriched the moment a ceiling is raised. The degree lives in
      the hybridization and nowhere else. On a face-neighbour face there is no
      such array and the element's order is the best available. */
   int tr_deg = fe_tr->GetOrder();
   if (hyb && face < mesh->GetNumFaces() &&
       hyb->GetTraceOrders().Size() > face)
   {
      tr_deg = hyb->GetTraceOrders()[face];
   }

   switch (type)
   {
      case Type::Residual:
      {
         constexpr int type = NonlinearFormIntegrator::HDGFaceType::CONSTR
                              | NonlinearFormIntegrator::HDGFaceType::FACE;

         bfi.AssembleHDGFaceVector(type, *fe_tr, fe1, FTr, tr, p1, btr1);
         error_estimates(FTr.Elem1No) += fabs(btr1.Sum());

         if (FTr.Elem2No >= 0)
         {
            bfi.AssembleHDGFaceVector(type | 1, *fe_tr, fe2, FTr, tr, p2, btr2);
            error_estimates(FTr.Elem2No) += fabs(btr2.Sum());
         }
      }
      break;
      case Type::Energy:
      {
         Vector d_en1, d_en2;

         const FiniteElement *fe_tr1 = fe_tr, *fe_tr2 = fe_tr;
         Vector cap1, cap2;
         if (cap_at_element)
         {
            const FiniteElementCollection *c_fec = fes_tr->FEColl();
            const Geometry::Type geom = mesh->GetFaceGeometry(face);
            if (tr_deg > fe1.GetOrder())
            {
               fe_tr1 = c_fec->GetFE(geom, fe1.GetOrder());
               ProjectTraceDown(*fe_tr, *fe_tr1, FTr, tr, cap1);
            }
            if (FTr.Elem2No >= 0 && tr_deg > fe2.GetOrder())
            {
               fe_tr2 = c_fec->GetFE(geom, fe2.GetOrder());
               ProjectTraceDown(*fe_tr, *fe_tr2, FTr, tr, cap2);
            }
         }
         const Vector &tr_1 = (fe_tr1 != fe_tr) ? cap1 : tr;
         const Vector &tr_2 = (fe_tr2 != fe_tr) ? cap2 : tr;

         /* The projected comparison is carried out by moving the difference
            into the trace space rather than by a second energy routine: with a
            zero element function the integrand is (0 - (λ - P_M p̂))², which is
            the number wanted, and the stabilization, its StabValue() hook and
            the anisotropic split are then exactly the ones the four assembly
            paths use. A second copy of that arithmetic here is how the energy
            estimator came to ignore an installed SetStabilization() once
            already.

            The one thing the hook does see differently is its @a v_q and
            @a tr_q arguments, which arrive shifted; a hook whose value depends
            on them is therefore evaluated off its own state under
            TraceComparison::Projected. */
         Vector tr1, tr2, z1, z2;
         const bool proj = (trcmp == TraceComparison::Projected);
         if (proj)
         {
            Vector c;
            ProjectOntoTrace(*fe_tr1, fe1, FTr, 0, p1, c);
            tr1.SetSize(tr_1.Size());
            subtract(tr_1, c, tr1);
            z1.SetSize(fe1.GetDof());
            z1 = 0.;

            if (FTr.Elem2No >= 0)
            {
               ProjectOntoTrace(*fe_tr2, fe2, FTr, 1, p2, c);
               tr2.SetSize(tr_2.Size());
               subtract(tr_2, c, tr2);
               z2.SetSize(fe2.GetDof());
               z2 = 0.;
            }
         }

         error_estimates(FTr.Elem1No) += bfi.ComputeHDGFaceEnergy(0, *fe_tr1, fe1, FTr,
                                                                  proj?tr1:tr_1, proj?z1:p1, (anisotropic)?(&d_en1):(NULL));

         if (FTr.Elem2No >= 0)
         {
            error_estimates(FTr.Elem2No) += bfi.ComputeHDGFaceEnergy(1, *fe_tr2, fe2, FTr,
                                                                     proj?tr2:tr_2, proj?z2:p2, (anisotropic)?(&d_en2):(NULL));
         }

         if (anisotropic)
         {
            /* A face richer than the element on one side keeps its MAGNITUDE
               -- that mismatch is real and the element is the one carrying it
               -- but contributes no DIRECTION, because the direction it would
               contribute is the wrong one. See SetSkipEnrichedDirection(). */
            const bool dir1 = !(skip_enriched_dir && tr_deg > fe1.GetOrder());
            const bool dir2 = !(skip_enriched_dir && FTr.Elem2No >= 0 &&
                                tr_deg > fe2.GetOrder());

            if (dir1)
            {
               for (int k = 0; k < dim; k++)
               {
                  d_error_estimates(FTr.Elem1No * dim + k) += d_en1(k);
               }
            }
            if (FTr.Elem2No >= 0 && dir2)
            {
               for (int k = 0; k < dim; k++)
               {
                  d_error_estimates(FTr.Elem2No * dim + k) += d_en2(k);
               }
            }
         }
      }
      break;
   }
}


PerssonPeraireSmoothness::PerssonPeraireSmoothness(const GridFunction &field,
                                                   real_t zero_tol_)
   : u(field), zero_tol(zero_tol_)
{
   const FiniteElementSpace *fes = u.FESpace();
   MFEM_VERIFY(fes, "the field has no finite element space");
   MFEM_VERIFY(fes->FEColl()->GetContType() ==
               FiniteElementCollection::DISCONTINUOUS,
               "The sensor truncates the expansion element by element, which "
               "only means something on a discontinuous space; this one is "
               "continuous across elements.");
}

const Vector &PerssonPeraireSmoothness::GetSensor()
{
   if (computed) { return S; }

   const FiniteElementSpace *fes = u.FESpace();
   const FiniteElementCollection *fec = fes->FEColl();
   Mesh *mesh = fes->GetMesh();
   const int NE = mesh->GetNE();
   const int vdim = fes->GetVDim();

   S.SetSize(NE);
   S = 0.0;

   MassIntegrator mass;
   DenseMatrix M_pp, M_qp, M_qq;
   Array<int> vdofs;
   Vector loc, u_e, b, c;
   Vector energy(NE);
   energy = 0.0;

   for (int e = 0; e < NE; e++)
   {
      const FiniteElement *fe_p = fes->GetFE(e);
      const int p = fe_p->GetOrder();
      fes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, loc);

      // Nothing to truncate to: the element resolves nothing, and reporting it
      // as maximally unresolved is what a driver should act on.
      if (p == 0) { S(e) = 1.0; energy(e) = -1.0; continue; }

      ElementTransformation *T = mesh->GetElementTransformation(e);
      const FiniteElement *fe_q = fec->GetFE(mesh->GetElementGeometry(e), p - 1);
      MFEM_VERIFY(fe_q && fe_q->GetOrder() == p - 1,
                  "the collection returned degree " << (fe_q ? fe_q->GetOrder() : -1)
                  << " when asked for " << p - 1);

      mass.AssembleElementMatrix(*fe_p, *T, M_pp);
      mass.AssembleElementMatrix(*fe_q, *T, M_qq);
      mass.AssembleElementMatrix2(*fe_p, *fe_q, *T, M_qp);   // (trial p, test q)

      const int ndof_p = fe_p->GetDof(), ndof_q = fe_q->GetDof();
      DenseMatrixInverse M_qq_inv(M_qq);

      // Per field, and the worst field wins: one unresolved component is
      // enough to say the element is not resolved.
      real_t worst = 0.0, tot = 0.0;
      for (int k = 0; k < vdim; k++)
      {
         u_e.SetSize(ndof_p);
         for (int i = 0; i < ndof_p; i++) { u_e(i) = loc(k * ndof_p + i); }

         // DenseMatrix::Mult does not resize its output, and with asserts off
         // an unsized one writes past the end -- segfault rather than a
         // message.
         c.SetSize(ndof_p);
         M_pp.Mult(u_e, c);
         const real_t nrm2 = u_e * c;                 // (u, u)_e
         b.SetSize(ndof_q);
         M_qp.Mult(u_e, b);                           // (u, phi_q)_e
         Vector proj(ndof_q);
         M_qq_inv.Mult(b, proj);
         const real_t prj2 = b * proj;                // (Pu, Pu)_e

         tot += nrm2;
         if (nrm2 > 0.0)
         {
            // Orthogonality of the projection: |u - Pu|^2 = |u|^2 - |Pu|^2.
            // Clamped because the two are equal to round-off wherever u is
            // already of degree p-1, and the difference can come back at -1e-18.
            const real_t num = std::max(nrm2 - prj2, real_t(0.0));
            worst = std::max(worst, num / nrm2);
         }
      }
      S(e) = worst;
      energy(e) = tot;
   }

   // An element carrying essentially nothing has no expansion to judge, and
   // 0/0 would otherwise come back as noise. Measured against the mean rather
   // than an absolute, so the floor follows the field's own scale.
   real_t mean = 0.0;
   int counted = 0;
   for (int e = 0; e < NE; e++)
   {
      if (energy(e) >= 0.0) { mean += energy(e); counted++; }
   }
   if (counted > 0) { mean /= counted; }
   for (int e = 0; e < NE; e++)
   {
      if (energy(e) >= 0.0 && energy(e) < zero_tol * mean) { S(e) = 0.0; }
   }

   computed = true;
   return S;
}

void PerssonPeraireSmoothness::GetLogSensor(Vector &s_e)
{
   const Vector &s = GetSensor();
   s_e.SetSize(s.Size());
   for (int e = 0; e < s.Size(); e++)
   {
      // A vanishing sensor is a perfectly resolved element; log10 of it is
      // minus infinity, and a large negative number is what a threshold test
      // wants instead.
      s_e(e) = (s(e) > 0.0) ? std::log10(s(e)) : -std::numeric_limits<real_t>::max();
   }
}

} // namespace mfem
