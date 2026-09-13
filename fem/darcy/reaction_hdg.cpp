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

#include "reaction_hdg.hpp"

#include <algorithm>

namespace mfem
{

HDGReactionIntegratorBase::HDGReactionIntegratorBase(
   const NodalReactionFunction &F_, HDGPostprocessBlocks &blocks_)
   : F(&F_), blocks(&blocks_)
{
   neq = blocks->GetNumEquations();
   MFEM_VERIFY(F->NumEquations() == neq,
               "the reaction law carries " << F->NumEquations()
               << " equation(s), the spaces carry " << neq);
}

void HDGReactionIntegratorBase::CheckFresh() const
{
   MFEM_VERIFY(assembled, "Assemble() has not been called");
   MFEM_VERIFY(blocks->IsAssembled(),
               "the postprocessing blocks are stale; Assemble() again");
}

void HDGReactionIntegratorBase::Assemble()
{
   if (!blocks->IsAssembled()) { blocks->Assemble(); }

   const FiniteElementSpace &fes_p = blocks->GetPotentialSpace();
   const FiniteElementSpace &fes_s = blocks->GetEnrichedSpace();
   Mesh *mesh = fes_p.GetMesh();
   const int NE = mesh->GetNE();
   const int dim = mesh->Dimension();

   offs_A9.SetSize(NE + 1);
   offs_x.SetSize(NE + 1);
   offs_A9[0] = offs_x[0] = 0;
   for (int z = 0; z < NE; z++)
   {
      const int nd = fes_p.GetFE(z)->GetDof();
      const int ns = fes_s.GetFE(z)->GetDof();
      offs_A9[z+1] = offs_A9[z] + nd * ns;
      offs_x[z+1] = offs_x[z] + ns * dim;
   }
   A9_data.SetSize(offs_A9[NE]);
   x_data.SetSize(offs_x[NE]);

   IsoparametricTransformation Tr;
   Vector shape_p, shape_s, xn(dim);

   for (int z = 0; z < NE; z++)
   {
      const FiniteElement *fe_p = fes_p.GetFE(z);
      const FiniteElement *fe_s = fes_s.GetFE(z);
      mesh->GetElementTransformation(z, &Tr);

      const int nd = fe_p->GetDof();
      const int ns = fe_s->GetDof();

      // A9 needs a rule of degree 2k+1; the postprocessing's own choice
      // covers it and a cheaper one would not.
      const int iro = (ir_order >= 0)
                      ? ir_order : (2 * fe_s->GetOrder() + Tr.OrderW());
      const IntegrationRule &ir = IntRules.Get(fe_s->GetGeomType(), iro);

      DenseMatrix A9(A9_data.GetData() + offs_A9[z], nd, ns);
      A9 = 0.0;
      shape_p.SetSize(nd);
      shape_s.SetSize(ns);

      for (int k = 0; k < ir.GetNPoints(); k++)
      {
         const IntegrationPoint &ip = ir.IntPoint(k);
         Tr.SetIntPoint(&ip);
         const real_t w = ip.weight * Tr.Weight();
         fe_p->CalcPhysShape(Tr, shape_p);
         fe_s->CalcPhysShape(Tr, shape_s);
         AddMult_a_VWt(w, shape_p, shape_s, A9);
      }

      // The node images. Z_h's nodes are fixed on the reference element, so
      // these are geometry: they belong here and not in the hot loop.
      DenseMatrix xs(x_data.GetData() + offs_x[z], ns, dim);
      const IntegrationRule &nodes = fe_s->GetNodes();
      for (int i = 0; i < ns; i++)
      {
         Tr.SetIntPoint(&nodes.IntPoint(i));
         Tr.Transform(nodes.IntPoint(i), xn);
         for (int d = 0; d < dim; d++) { xs(i, d) = xn(d); }
      }
   }

   assembled = true;
}

void HDGReactionIntegratorBase::GetA9(int el, DenseMatrix &A9) const
{
   CheckFresh();
   const int ns = blocks->NumNodes(el);
   const int nd = blocks->NumPotentialDofs(el);
   A9.SetSize(nd, ns);
   const DenseMatrix stored(const_cast<real_t*>(A9_data.GetData())
                            + offs_A9[el], nd, ns);
   A9 = stored;
}

void HDGReactionIntegratorBase::GetNodes(int el, DenseMatrix &x) const
{
   CheckFresh();
   const int ns = blocks->NumNodes(el);
   const int dim = (offs_x[el+1] - offs_x[el]) / ns;
   x.SetSize(ns, dim);
   const DenseMatrix stored(const_cast<real_t*>(x_data.GetData())
                            + offs_x[el], ns, dim);
   x = stored;
}

void HDGReactionIntegratorBase::Prepare(int el, const Vector &u_l,
                                        const Vector &p_l,
                                        Vector &gamma) const
{
   CheckFresh();
   blocks->Apply(el, u_l, p_l, gamma);
}

void HDGInterpolatoryReactionIntegrator::AssembleElementVector(
   const Array<const FiniteElement*> &el, ElementTransformation &Tr,
   const Array<const Vector*> &elfun, const Array<Vector*> &elvec)
{
   const int z = Tr.ElementNo;
   const int ns = blocks->NumNodes(z);
   const int nd = blocks->NumPotentialDofs(z);

   Vector gamma;
   Prepare(z, *elfun[0], *elfun[1], gamma);

   const DenseMatrix xs(const_cast<real_t*>(x_data.GetData()) + offs_x[z],
                        ns, (offs_x[z+1] - offs_x[z]) / ns);
   const DenseMatrix A9(const_cast<real_t*>(A9_data.GetData()) + offs_A9[z],
                        nd, ns);

   // F at every node, laid out equation outermost like gamma itself.
   Vector fvals(neq * ns), un(neq), Fn(neq), xn(xs.Width());
   for (int i = 0; i < ns; i++)
   {
      for (int d = 0; d < xs.Width(); d++) { xn(d) = xs(i, d); }
      for (int e = 0; e < neq; e++) { un(e) = gamma(e * ns + i); }
      F->Eval(xn, un, Fn);
      for (int e = 0; e < neq; e++) { fvals(e * ns + i) = Fn(e); }
   }

   // This term has no flux row. Emptying rather than zeroing is the contract
   // the element loop reads: a block left at size zero is one the integrator
   // did not write.
   elvec[0]->SetSize(0);
   elvec[1]->SetSize(neq * nd);
   // The caller's row, so the host accessor rather than GetData(); see
   // HDGPostprocessBlocks::Apply() for what the raw pointer costs. A pure
   // write: A9.Mult() overwrites each block and the loop covers them all.
   real_t *r_d = elvec[1]->HostWrite();
   for (int e = 0; e < neq; e++)
   {
      const Vector f_e(fvals.GetData() + e * ns, ns);
      Vector r_e(r_d + e * nd, nd);
      A9.Mult(f_e, r_e);
   }
}

void HDGInterpolatoryReactionIntegrator::AssembleElementGrad(
   const Array<const FiniteElement*> &el, ElementTransformation &Tr,
   const Array<const Vector*> &elfun, const Array2D<DenseMatrix*> &elmats)
{
   const int z = Tr.ElementNo;
   const int ns = blocks->NumNodes(z);
   const int nd = blocks->NumPotentialDofs(z);
   const int na = blocks->NumFluxDofs(z);

   Vector gamma;
   Prepare(z, *elfun[0], *elfun[1], gamma);

   const DenseMatrix xs(const_cast<real_t*>(x_data.GetData()) + offs_x[z],
                        ns, (offs_x[z+1] - offs_x[z]) / ns);
   const DenseMatrix A9(const_cast<real_t*>(A9_data.GetData()) + offs_A9[z],
                        nd, ns);
   DenseMatrix B11, B12;
   blocks->GetBlocks(z, B11, B12);

   // F'(gamma) at every node. For a system it is a neq x neq matrix per node,
   // so "diag(F')" is block structured: the (e,f) block of dR/dgamma is
   // A9 diag_i(J_i(e,f)), and the chain rule then multiplies by B11 or B12,
   // which do not couple the equations.
   // Flattened as (e*neq+f, node) rather than a container of matrices:
   // mfem::Array requires a trivial element type, and this is the layout the
   // loop below wants anyway.
   DenseMatrix Jall(neq * neq, ns), Ji(neq);
   Vector un(neq), xn(xs.Width());
   for (int i = 0; i < ns; i++)
   {
      for (int d = 0; d < xs.Width(); d++) { xn(d) = xs(i, d); }
      for (int e = 0; e < neq; e++) { un(e) = gamma(e * ns + i); }
      F->EvalJacobian(xn, un, Ji);
      for (int e = 0; e < neq; e++)
         for (int f = 0; f < neq; f++)
         {
            Jall(e * neq + f, i) = Ji(e, f);
         }
   }

   // A block may be NULL, and the (1,0) one currently always is: both of
   // DarcyHybridization's call sites do `grad_arr = NULL` and then fill only
   // (0,0), (0,1) and (1,1), on the stated grounds that the divergence form
   // is linear so B is exact in Bf_data. That is the block this term needs,
   // so until it exists the Jacobian assembled here is INCOMPLETE -- the
   // potential residual's dependence on the flux is dropped. Newton then
   // converges linearly rather than quadratically instead of failing, which
   // is why it has to be said here rather than discovered.
   if (elmats(0, 0)) { elmats(0, 0)->SetSize(0, 0); }
   if (elmats(0, 1)) { elmats(0, 1)->SetSize(0, 0); }
   if (elmats(1, 0))
   {
      elmats(1, 0)->SetSize(neq * nd, neq * na);
      *elmats(1, 0) = 0.0;
   }
   if (elmats(1, 1))
   {
      elmats(1, 1)->SetSize(neq * nd, neq * nd);
      *elmats(1, 1) = 0.0;
   }

   DenseMatrix AJ(nd, ns), blk10(nd, na), blk11(nd, nd);
   for (int e = 0; e < neq; e++)
      for (int f = 0; f < neq; f++)
      {
         for (int i = 0; i < nd; i++)
            for (int j = 0; j < ns; j++)
            {
               AJ(i, j) = A9(i, j) * Jall(e * neq + f, j);
            }
         if (elmats(1, 0))
         {
            Mult(AJ, B11, blk10);
            elmats(1, 0)->AddSubMatrix(e * nd, f * na, blk10);
         }
         if (elmats(1, 1))
         {
            Mult(AJ, B12, blk11);
            elmats(1, 1)->AddSubMatrix(e * nd, f * nd, blk11);
         }
      }
}

void HDGQuadratureReactionIntegrator::AssembleElementVector(
   const Array<const FiniteElement*> &el, ElementTransformation &Tr,
   const Array<const Vector*> &elfun, const Array<Vector*> &elvec)
{
   const int z = Tr.ElementNo;
   const int ns = blocks->NumNodes(z);
   const int nd = blocks->NumPotentialDofs(z);
   const FiniteElement *fe_p = el[1];
   const FiniteElement *fe_s = blocks->GetEnrichedSpace().GetFE(z);

   Vector gamma;
   Prepare(z, *elfun[0], *elfun[1], gamma);

   const int iro = (ir_order >= 0)
                   ? ir_order : (2 * fe_s->GetOrder() + Tr.OrderW());
   const IntegrationRule &ir = IntRules.Get(fe_s->GetGeomType(), iro);

   elvec[0]->SetSize(0);
   elvec[1]->SetSize(neq * nd);
   // Zeroed through the host pointer rather than with `*elvec[1] = 0.0`,
   // which honours the vector's UseDevice() flag and would put the zeros on
   // the DEVICE -- leaving the raw accumulation below writing a host page
   // the next reader has no reason to believe. See
   // HDGPostprocessBlocks::Apply().
   real_t *r_d = elvec[1]->HostWrite();
   std::fill(r_d, r_d + neq * nd, (real_t) 0.0);

   Vector shape_p(nd), shape_s(ns), un(neq), Fn(neq), xq(Tr.GetSpaceDim());
   for (int k = 0; k < ir.GetNPoints(); k++)
   {
      const IntegrationPoint &ip = ir.IntPoint(k);
      Tr.SetIntPoint(&ip);
      const real_t w = ip.weight * Tr.Weight();
      fe_p->CalcPhysShape(Tr, shape_p);
      fe_s->CalcPhysShape(Tr, shape_s);
      Tr.Transform(ip, xq);

      // u* AT the quadrature point, which is the one thing this integrator
      // does differently: the interpolatory form only ever sees u* at nodes.
      for (int e = 0; e < neq; e++)
      {
         const Vector g_e(gamma.GetData() + e * ns, ns);
         un(e) = g_e * shape_s;
      }
      F->Eval(xq, un, Fn);

      for (int e = 0; e < neq; e++)
      {
         Vector r_e(r_d + e * nd, nd);
         r_e.Add(w * Fn(e), shape_p);
      }
   }
}

void HDGQuadratureReactionIntegrator::AssembleElementGrad(
   const Array<const FiniteElement*> &el, ElementTransformation &Tr,
   const Array<const Vector*> &elfun, const Array2D<DenseMatrix*> &elmats)
{
   const int z = Tr.ElementNo;
   const int ns = blocks->NumNodes(z);
   const int nd = blocks->NumPotentialDofs(z);
   const int na = blocks->NumFluxDofs(z);
   const FiniteElement *fe_p = el[1];
   const FiniteElement *fe_s = blocks->GetEnrichedSpace().GetFE(z);

   Vector gamma;
   Prepare(z, *elfun[0], *elfun[1], gamma);

   DenseMatrix B11, B12;
   blocks->GetBlocks(z, B11, B12);

   const int iro = (ir_order >= 0)
                   ? ir_order : (2 * fe_s->GetOrder() + Tr.OrderW());
   const IntegrationRule &ir = IntRules.Get(fe_s->GetGeomType(), iro);

   // A block may be NULL; see the interpolatory integrator for which.
   if (elmats(0, 0)) { elmats(0, 0)->SetSize(0, 0); }
   if (elmats(0, 1)) { elmats(0, 1)->SetSize(0, 0); }
   if (elmats(1, 0))
   {
      elmats(1, 0)->SetSize(neq * nd, neq * na);
      *elmats(1, 0) = 0.0;
   }
   if (elmats(1, 1))
   {
      elmats(1, 1)->SetSize(neq * nd, neq * nd);
      *elmats(1, 1) = 0.0;
   }

   // dR_e/dgamma_f accumulated over quadrature, then chained through B11/B12.
   Array2D<DenseMatrix*> G(neq, neq);
   for (int e = 0; e < neq; e++)
      for (int f = 0; f < neq; f++)
      {
         G(e, f) = new DenseMatrix(nd, ns);
         *G(e, f) = 0.0;
      }

   Vector shape_p(nd), shape_s(ns), un(neq), xq(Tr.GetSpaceDim());
   DenseMatrix J(neq);
   for (int k = 0; k < ir.GetNPoints(); k++)
   {
      const IntegrationPoint &ip = ir.IntPoint(k);
      Tr.SetIntPoint(&ip);
      const real_t w = ip.weight * Tr.Weight();
      fe_p->CalcPhysShape(Tr, shape_p);
      fe_s->CalcPhysShape(Tr, shape_s);
      Tr.Transform(ip, xq);

      for (int e = 0; e < neq; e++)
      {
         const Vector g_e(gamma.GetData() + e * ns, ns);
         un(e) = g_e * shape_s;
      }
      F->EvalJacobian(xq, un, J);

      for (int e = 0; e < neq; e++)
         for (int f = 0; f < neq; f++)
         {
            AddMult_a_VWt(w * J(e, f), shape_p, shape_s, *G(e, f));
         }
   }

   DenseMatrix blk10(nd, na), blk11(nd, nd);
   for (int e = 0; e < neq; e++)
      for (int f = 0; f < neq; f++)
      {
         if (elmats(1, 0))
         {
            Mult(*G(e, f), B11, blk10);
            elmats(1, 0)->AddSubMatrix(e * nd, f * na, blk10);
         }
         if (elmats(1, 1))
         {
            Mult(*G(e, f), B12, blk11);
            elmats(1, 1)->AddSubMatrix(e * nd, f * nd, blk11);
         }
         delete G(e, f);
      }
}

}
