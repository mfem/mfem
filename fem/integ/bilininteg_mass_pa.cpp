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

#include "../../general/forall.hpp"
#include "../../general/globals.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "../ceed/integrators/mass/mass.hpp"
#include "../fe/fe_pos.hpp"
#include "../fe/fe_h1.hpp"
#include "bilininteg_mass_kernels.hpp"
#include "bilininteg_mass_pa_simplices.hpp"
#include "bilininteg_mass_pa_tmo.hpp"

namespace mfem
{

namespace
{

bool EnvFlag(const char *name)
{
   const char *e = GetEnv(name);
   return e && e[0] && e[0] != '0';
}

enum { TMO_OFF = 0, TMO_DUFFY = 1, TMO_TENSOR = 2, TMO_BERNSTEIN = 3 };

int SelectTmoMode()
{
   const bool duffy = EnvFlag("MFEM_USE_TMO_DUFFY");
   const bool tensor = EnvFlag("MFEM_USE_TMO_TENSOR");
   const bool bern = EnvFlag("MFEM_USE_TMO_BERNSTEIN");
   const int nset = int(duffy) + int(tensor) + int(bern);
   MFEM_VERIFY(nset <= 1,
               "Set only one of MFEM_USE_TMO_DUFFY, MFEM_USE_TMO_TENSOR, "
               "MFEM_USE_TMO_BERNSTEIN");
   if (duffy) { return TMO_DUFFY; }
   if (tensor) { return TMO_TENSOR; }
   if (bern) { return TMO_BERNSTEIN; }
   return TMO_OFF;
}

} // namespace

// PA Mass Integrator

void MassIntegrator::AssemblePA_TMO_Duffy(const FiniteElementSpace &fes)
{
   dbg();
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2, "MFEM_USE_TMO_DUFFY currently supports 2D triangles only");
   MFEM_VERIFY(fes.UsesRaggedTensorBasis(),
               "MFEM_USE_TMO_DUFFY requires Positive (Bernstein) H1 on simplices");
   MFEM_VERIFY(mesh->SpaceDimension() == 2, "MFEM_USE_TMO_DUFFY requires a 2D mesh");

   if (mesh->GetNodes())
   {
      MFEM_VERIFY(mesh->GetNodalFESpace()->GetMaxElementOrder() <= 1,
                  "MFEM_USE_TMO_DUFFY v1 requires affine (linear) meshes");
   }

   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(el.GetGeomType() == Geometry::TRIANGLE,
               "MFEM_USE_TMO_DUFFY currently supports triangles only");
   MFEM_VERIFY(dynamic_cast<const H1Pos_TriangleElement *>(&el),
               "MFEM_USE_TMO_DUFFY requires H1Pos_TriangleElement");

   ElementTransformation *T0 = mesh->GetTypicalElementTransformation();
   const int map_type = el.GetMapType();

   const IntegrationRule &ir =
      IntRule ? *IntRule : GetRule(el, el, *T0, true);
   maps = &el.GetDofToQuad(ir, DofToQuad::RAGGED_TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   const int Q1D = quad1D;
   const int nq1 = Q1D * Q1D;
   MFEM_VERIFY(ir.GetNPoints() == nq1, "TMO Duffy expects a tensor Stroud rule");
   this->nq = internal::tmo::NMIRRORS * nq1;
   const int nq_tmo = this->nq;
   ne = mesh->GetNE();
   pa_tmo = TMO_DUFFY;
   tmo_B.DeleteAll();
   tmo_P.DeleteAll();

   const real_t inv_m = real_t(1) / real_t(internal::tmo::NMIRRORS);
   IntegrationRule ir_tmo(nq_tmo);
   for (int k = 0; k < internal::tmo::NMIRRORS; k++)
   {
      for (int i2 = 0; i2 < Q1D; i2++)
      {
         for (int i1 = 0; i1 < Q1D; i1++)
         {
            const int q0 = i1 + Q1D * i2;
            const int q = i1 + Q1D * (i2 + Q1D * k);
            const IntegrationPoint &ip0 = ir.IntPoint(q0);
            real_t xi, eta;
            internal::tmo::MapSquareToTriangle(k, ip0.x, ip0.y, xi, eta);
            IntegrationPoint &ip = ir_tmo.IntPoint(q);
            ip.Set2(xi, eta);
            ip.weight = inv_m * ip0.weight;
         }
      }
   }

   geom = mesh->GetGeometricFactors(ir, GeometricFactors::DETERMINANTS, mt);

   pa_data.SetSize(nq_tmo * ne, mt);
   QuadratureSpace qs(*mesh, ir_tmo);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);

   const int NE = ne;
   const int NQ = nq_tmo;
   const int NQ1 = nq1;
   const bool const_c = coeff.Size() == 1;
   const bool by_val = map_type == FiniteElement::VALUE;
   {
      const auto W = Reshape(ir_tmo.GetWeights().Read(), NQ);
      const auto J = Reshape(geom->detJ.Read(), NQ1, NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1, 1)
                     : Reshape(coeff.Read(), NQ, NE);
      auto v = Reshape(pa_data.Write(), NQ, NE);
      mfem::forall(NQ, NE, [=] MFEM_HOST_DEVICE (int q, int e)
      {
         const real_t detJ = J(0, e);
         const real_t c = const_c ? C(0, 0) : C(q, e);
         v(q, e) = W(q) * c * (by_val ? detJ : real_t(1) / detJ);
      });
   }
}

void MassIntegrator::AssemblePA_TMO_Tensor(const FiniteElementSpace &fes)
{
   dbg();
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2, "MFEM_USE_TMO_TENSOR currently supports 2D triangles only");
   MFEM_VERIFY(!fes.UsesRaggedTensorBasis(),
               "MFEM_USE_TMO_TENSOR requires standard H1 (Gauss-Lobatto), not Positive");
   MFEM_VERIFY(mesh->SpaceDimension() == 2, "MFEM_USE_TMO_TENSOR requires a 2D mesh");

   if (mesh->GetNodes())
   {
      MFEM_VERIFY(mesh->GetNodalFESpace()->GetMaxElementOrder() <= 1,
                  "MFEM_USE_TMO_TENSOR v1 requires affine (linear) meshes");
   }

   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(el.GetGeomType() == Geometry::TRIANGLE,
               "MFEM_USE_TMO_TENSOR currently supports triangles only");
   const H1_TriangleElement *tel = dynamic_cast<const H1_TriangleElement *>(&el);
   MFEM_VERIFY(tel, "MFEM_USE_TMO_TENSOR requires H1_TriangleElement");

   ElementTransformation *T0 = mesh->GetTypicalElementTransformation();
   const int map_type = el.GetMapType();
   const int p = el.GetOrder();
   dofs1D = p + 1;
   const int ndof = el.GetDof();

   // Integrate on T via three parallelogram charts φ_k, using a triangle
   // rule as (s,t)∈T (weight 1/m). Full-square tensor Gauss is inexact for
   // the C0 even extension across s+t=1, so we stay on the T half.
   const int q_order = IntRule ? IntRule->GetOrder()
                       : 2 * p + T0->OrderW() + 4;
   const IntegrationRule &ir_T =
      IntRule ? *IntRule : IntRules.Get(Geometry::TRIANGLE, q_order);
   const int nq1 = ir_T.GetNPoints();
   quad1D = nq1; // for TENSOR: points per mirror (not a 1D count)
   this->nq = internal::tmo::NMIRRORS * nq1;
   const int nq_tmo = this->nq;
   ne = mesh->GetNE();
   pa_tmo = TMO_TENSOR;
   maps = nullptr;
   tmo_B.DeleteAll();

   const real_t inv_m = real_t(1) / real_t(internal::tmo::NMIRRORS);
   IntegrationRule ir_tmo(nq_tmo);
   tmo_P.SetSize(nq1 * ndof * internal::tmo::NMIRRORS, mt);
   {
      real_t *Ph = tmo_P.HostWrite();
      Vector shape_ref(ndof);
      IntegrationPoint ip;
      for (int k = 0; k < internal::tmo::NMIRRORS; k++)
      {
         for (int q1 = 0; q1 < nq1; q1++)
         {
            const IntegrationPoint &ip0 = ir_T.IntPoint(q1);
            real_t xf, yf;
            internal::tmo::MapSquareToParallelogram(k, ip0.x, ip0.y, xf, yf);
            ip.Set2(xf, yf);
            tel->CalcShape(ip, shape_ref);
            for (int i = 0; i < ndof; i++)
            {
               Ph[q1 + nq1 * (i + ndof * k)] = shape_ref(i);
            }
            const int q = q1 + nq1 * k;
            IntegrationPoint &ipt = ir_tmo.IntPoint(q);
            ipt.Set2(xf, yf);
            ipt.weight = inv_m * ip0.weight;
         }
      }
   }

   geom = mesh->GetGeometricFactors(ir_T, GeometricFactors::DETERMINANTS, mt);

   pa_data.SetSize(nq_tmo * ne, mt);
   QuadratureSpace qs(*mesh, ir_tmo);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);

   const int NE = ne;
   const int NQ = nq_tmo;
   const int NQ1 = nq1;
   const bool const_c = coeff.Size() == 1;
   const bool by_val = map_type == FiniteElement::VALUE;
   {
      const auto W = Reshape(ir_tmo.GetWeights().Read(), NQ);
      const auto J = Reshape(geom->detJ.Read(), NQ1, NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1, 1)
                     : Reshape(coeff.Read(), NQ, NE);
      auto v = Reshape(pa_data.Write(), NQ, NE);
      mfem::forall(NQ, NE, [=] MFEM_HOST_DEVICE (int q, int e)
      {
         const real_t detJ = J(0, e);
         const real_t c = const_c ? C(0, 0) : C(q, e);
         v(q, e) = W(q) * c * (by_val ? detJ : real_t(1) / detJ);
      });
   }
}

void MassIntegrator::AssemblePA_TMO_Bernstein(const FiniteElementSpace &fes)
{
   dbg();
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2,
               "MFEM_USE_TMO_BERNSTEIN currently supports 2D triangles only");
   MFEM_VERIFY(fes.UsesRaggedTensorBasis(),
               "MFEM_USE_TMO_BERNSTEIN requires Positive (Bernstein) H1");
   MFEM_VERIFY(mesh->SpaceDimension() == 2,
               "MFEM_USE_TMO_BERNSTEIN requires a 2D mesh");

   if (mesh->GetNodes())
   {
      MFEM_VERIFY(mesh->GetNodalFESpace()->GetMaxElementOrder() <= 1,
                  "MFEM_USE_TMO_BERNSTEIN v1 requires affine (linear) meshes");
   }

   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(el.GetGeomType() == Geometry::TRIANGLE,
               "MFEM_USE_TMO_BERNSTEIN currently supports triangles only");
   MFEM_VERIFY(dynamic_cast<const H1Pos_TriangleElement *>(&el),
               "MFEM_USE_TMO_BERNSTEIN requires H1Pos_TriangleElement");

   ElementTransformation *T0 = mesh->GetTypicalElementTransformation();
   const int map_type = el.GetMapType();
   const int p = el.GetOrder();
   dofs1D = p + 1;
   const int D1D = dofs1D;
   const int ndof = el.GetDof();
   const int nqdof = D1D * D1D;

   const int q_order = IntRule ? IntRule->GetOrder()
                       : 2 * p + T0->OrderW() + 4;
   const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, q_order);
   quad1D = ir1d.GetNPoints();
   const int Q1D = quad1D;
   const int nq1 = Q1D * Q1D;
   this->nq = internal::tmo::NMIRRORS * nq1;
   const int nq_tmo = this->nq;
   ne = mesh->GetNE();
   pa_tmo = TMO_BERNSTEIN;
   maps = nullptr;

   // 1D Bernstein B at Gauss quad pts via Positive quad element.
   IntegrationRule ir_sq(nq1);
   for (int j = 0; j < Q1D; j++)
   {
      for (int i = 0; i < Q1D; i++)
      {
         IntegrationPoint &ip = ir_sq.IntPoint(i + Q1D * j);
         ip.x = ir1d.IntPoint(i).x;
         ip.y = ir1d.IntPoint(j).x;
         ip.weight = ir1d.IntPoint(i).weight * ir1d.IntPoint(j).weight;
      }
   }
   H1Pos_QuadrilateralElement qel(p);
   const DofToQuad &qmaps = qel.GetDofToQuad(ir_sq, DofToQuad::TENSOR);
   tmo_B = qmaps.B;

   // Prolong P: V^{-1} E, where E samples even-extended triangle Bernstein at
   // uniform nodes (i/p,j/p), and V is the tensor-Bernstein Vandermonde.
   DenseMatrix V(nqdof), E(nqdof, ndof), Pk(nqdof, ndof);
   Vector bern_x(D1D), bern_y(D1D), shape_ref(ndof);
   tmo_P.SetSize(nqdof * ndof * internal::tmo::NMIRRORS, mt);
   {
      real_t *Ph = tmo_P.HostWrite();
      for (int k = 0; k < internal::tmo::NMIRRORS; k++)
      {
         for (int j = 0; j < D1D; j++)
         {
            const real_t tj = (p == 0) ? real_t(0) : real_t(j) / real_t(p);
            Poly_1D::CalcBernstein(p, tj, bern_y.GetData());
            for (int i = 0; i < D1D; i++)
            {
               const real_t si = (p == 0) ? real_t(0) : real_t(i) / real_t(p);
               Poly_1D::CalcBernstein(p, si, bern_x.GetData());
               const int iq = i + D1D * j;
               for (int b = 0; b < D1D; b++)
               {
                  for (int a = 0; a < D1D; a++)
                  {
                     V(iq, a + D1D * b) = bern_x(a) * bern_y(b);
                  }
               }
               real_t xf, yf;
               internal::tmo::EvenEvalPoint(k, si, tj, xf, yf);
               H1Pos_TriangleElement::CalcShape(p, xf, yf, shape_ref.GetData());
               for (int alpha = 0; alpha < ndof; alpha++)
               {
                  E(iq, alpha) = shape_ref(alpha);
               }
            }
         }
         DenseMatrixInverse Vinv(V);
         Vinv.Mult(E, Pk);
         for (int alpha = 0; alpha < ndof; alpha++)
         {
            for (int iq = 0; iq < nqdof; iq++)
            {
               Ph[iq + nqdof * (alpha + ndof * k)] = Pk(iq, alpha);
            }
         }
      }
   }

   // pa_data on the full square; weight 1/(2m) for even extension.
   const real_t wfac = real_t(1) / (real_t(2) * real_t(internal::tmo::NMIRRORS));
   IntegrationRule ir_tmo(nq_tmo);
   for (int k = 0; k < internal::tmo::NMIRRORS; k++)
   {
      for (int i2 = 0; i2 < Q1D; i2++)
      {
         for (int i1 = 0; i1 < Q1D; i1++)
         {
            const int q = i1 + Q1D * (i2 + Q1D * k);
            const real_t s = ir1d.IntPoint(i1).x;
            const real_t t = ir1d.IntPoint(i2).x;
            real_t xf, yf;
            internal::tmo::EvenEvalPoint(k, s, t, xf, yf);
            IntegrationPoint &ip = ir_tmo.IntPoint(q);
            ip.Set2(xf, yf);
            ip.weight = wfac * ir1d.IntPoint(i1).weight * ir1d.IntPoint(i2).weight;
         }
      }
   }

   const IntegrationRule &ir_geom = IntRules.Get(Geometry::TRIANGLE, q_order);
   geom = mesh->GetGeometricFactors(ir_geom, GeometricFactors::DETERMINANTS, mt);

   pa_data.SetSize(nq_tmo * ne, mt);
   QuadratureSpace qs(*mesh, ir_tmo);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);

   const int NE = ne;
   const int NQ = nq_tmo;
   const int NQ1 = ir_geom.GetNPoints();
   const bool const_c = coeff.Size() == 1;
   const bool by_val = map_type == FiniteElement::VALUE;
   {
      const auto W = Reshape(ir_tmo.GetWeights().Read(), NQ);
      const auto J = Reshape(geom->detJ.Read(), NQ1, NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1, 1)
                     : Reshape(coeff.Read(), NQ, NE);
      auto v = Reshape(pa_data.Write(), NQ, NE);
      mfem::forall(NQ, NE, [=] MFEM_HOST_DEVICE (int q, int e)
      {
         const real_t detJ = J(0, e);
         const real_t c = const_c ? C(0, 0) : C(q, e);
         v(q, e) = W(q) * c * (by_val ? detJ : real_t(1) / detJ);
      });
   }
}

void MassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   pa_tmo = TMO_OFF;
   tmo_P.DeleteAll();
   tmo_B.DeleteAll();

   const int mode = SelectTmoMode();
   if (mode == TMO_DUFFY)
   {
      dbg("[TMO Duffy] AssemblePA");
      MFEM_VERIFY(fes.GetMesh()->Dimension() == 2,
                  "MFEM_USE_TMO_DUFFY is only implemented for 2D triangles");
      AssemblePA_TMO_Duffy(fes);
      return;
   }
   if (mode == TMO_TENSOR)
   {
      dbg("[TMO Tensor] AssemblePA");
      MFEM_VERIFY(fes.GetMesh()->Dimension() == 2,
                  "MFEM_USE_TMO_TENSOR is only implemented for 2D triangles");
      AssemblePA_TMO_Tensor(fes);
      return;
   }
   if (mode == TMO_BERNSTEIN)
   {
      dbg("[TMO Bernstein] AssemblePA");
      MFEM_VERIFY(fes.GetMesh()->Dimension() == 2,
                  "MFEM_USE_TMO_BERNSTEIN is only implemented for 2D triangles");
      AssemblePA_TMO_Bernstein(fes);
      return;
   }

   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation *T0 = mesh->GetTypicalElementTransformation();
   const bool stroud = fes.UsesRaggedTensorBasis();
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, *T0, stroud);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed)
      {
         ceedOp = new ceed::MixedPAMassIntegrator(*this, fes, Q);
      }
      else
      {
         ceedOp = new ceed::PAMassIntegrator(fes, *ir, Q);
      }
      return;
   }
   int map_type = el.GetMapType();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::DETERMINANTS, mt);
   if (stroud)
   {
      maps = &el.GetDofToQuad(*ir, DofToQuad::RAGGED_TENSOR);
   }
   else
   {
      maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   }
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(ne*nq, mt);

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   {
      const int NE = ne;
      const int NQ = nq;
      const bool const_c = coeff.Size() == 1;
      const bool by_val = map_type == FiniteElement::VALUE;
      const auto W = Reshape(ir->GetWeights().Read(), NQ);
      const auto J = Reshape(geom->detJ.Read(), NQ, NE);
      const auto C =
         const_c ? Reshape(coeff.Read(), 1, 1) : Reshape(coeff.Read(), NQ, NE);
      auto v = Reshape(pa_data.Write(), NQ, NE);
      mfem::forall(NQ, NE, [=] MFEM_HOST_DEVICE(int q, int e)
      {
         const real_t detJ = J(q, e);
         const real_t coeff = const_c ? C(0, 0) : C(q, e);
         v(q, e) = W(q) * coeff * (by_val ? detJ : 1.0 / detJ);
      });
   }
}

void MassIntegrator::AssemblePABoundary(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   ne = mesh->GetNFbyType(FaceType::Boundary);
   if (ne == 0) { return; }
   const FiniteElement &el = *fes.GetBE(0);
   ElementTransformation *T0 = mesh->GetBdrElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, *T0);

   int map_type = el.GetMapType();
   dim = el.GetDim();
   nq = ir->GetNPoints();
   face_geom = mesh->GetFaceGeometricFactors(*ir, GeometricFactors::DETERMINANTS,
                                             FaceType::Boundary, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(ne*nq, mt);

   FaceQuadratureSpace qs(*mesh, *ir, FaceType::Boundary);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);

   const int NE = ne;
   const int NQ = nq;
   const bool const_c = coeff.Size() == 1;
   const bool by_val = map_type == FiniteElement::VALUE;
   {
      const auto W = Reshape(ir->GetWeights().Read(), NQ);
      const auto J = Reshape(face_geom->detJ.Read(), NQ, NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1, 1)
                     : Reshape(coeff.Read(), NQ, NE);
      auto v = Reshape(pa_data.Write(), NQ, NE);
      mfem::forall(NQ, NE, [=] MFEM_HOST_DEVICE(int q, int e)
      {
         const real_t detJ = J(q, e);
         const real_t coeff = const_c ? C(0, 0) : C(q, e);
         v(q, e) = W(q) * coeff * (by_val ? detJ : 1.0 / detJ);
      });
   }
}

void MassIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else if (pa_tmo)
   {
      MFEM_ABORT("AssembleDiagonalPA not implemented for TMO PA");
   }
   else
   {
      DiagonalPAKernels::Run(dim, dofs1D, quad1D, ne, maps->B, pa_data,
                             diag, dofs1D, quad1D);
   }
}

void MassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else if (pa_tmo == TMO_DUFFY)
   {
      dbg("[TMO Duffy] AddMultPA");
      const int D1D = dofs1D;
      const int Q1D = quad1D;
      const auto *rmaps = static_cast<const RaggedDofToQuad*>(maps);
      ApplyTmoPAKernels::Run(dim, D1D, Q1D, ne,
                             rmaps->lex_map,
                             rmaps->forward_map2d_mass,
                             rmaps->inverse_map2d_mass,
                             rmaps->forward_map3d_mass,
                             rmaps->inverse_map3d_mass,
                             rmaps->Ba1, rmaps->Ba2, rmaps->Ba3,
                             rmaps->Ba1t, rmaps->Ba2t, rmaps->Ba3t,
                             pa_data, x, y, D1D, Q1D);
   }
   else if (pa_tmo == TMO_TENSOR)
   {
      dbg("[TMO Tensor] AddMultPA");
      ApplyTmoTensorPAKernels::Run(dim, dofs1D, quad1D, ne, tmo_P,
                                   pa_data, x, y, dofs1D, quad1D);
   }
   else if (pa_tmo == TMO_BERNSTEIN)
   {
      dbg("[TMO Bernstein] AddMultPA");
      ApplyTmoBernsteinPAKernels::Run(dim, dofs1D, quad1D, ne, tmo_B, tmo_P,
                                      pa_data, x, y, dofs1D, quad1D);
   }
   else
   {
      const int D1D = dofs1D;
      const int Q1D = quad1D;
      const Vector &D = pa_data;
      const Array<real_t> &B = maps->B;
      const Array<real_t> &Bt = maps->Bt;

#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         if (dim == 2)
         {
            return internal::OccaPAMassApply2D(D1D,Q1D,ne,B,Bt,D,x,y);
         }
         if (dim == 3)
         {
            return internal::OccaPAMassApply3D(D1D,Q1D,ne,B,Bt,D,x,y);
         }
         MFEM_ABORT("OCCA PA Mass Apply unknown kernel!");
      }
#endif // MFEM_USE_OCCA

      if (fespace->UsesRaggedTensorBasis())
      {
         const auto *rmaps = static_cast<const RaggedDofToQuad*>(maps);

         const Array<real_t> &Ba1 = rmaps->Ba1;
         const Array<real_t> &Ba2 = rmaps->Ba2;
         const Array<real_t> &Ba3 = rmaps->Ba3;
         const Array<real_t> &Ba1t = rmaps->Ba1t;
         const Array<real_t> &Ba2t = rmaps->Ba2t;
         const Array<real_t> &Ba3t = rmaps->Ba3t;
         const Array<int> &lex_map = rmaps->lex_map;
         const Array<int> &forward_map2d = rmaps->forward_map2d_mass;
         const Array<int> &inverse_map2d = rmaps->inverse_map2d_mass;
         const Array<int> &forward_map3d = rmaps->forward_map3d_mass;
         const Array<int> &inverse_map3d = rmaps->inverse_map3d_mass;
         ApplySimplexPAKernels::Run(dim, D1D, Q1D, ne, lex_map, forward_map2d,
                                    inverse_map2d,
                                    forward_map3d, inverse_map3d, Ba1, Ba2, Ba3, Ba1t, Ba2t, Ba3t,
                                    D, x, y, D1D, Q1D);
      }
      else
      {
         ApplyPAKernels::Run(dim, D1D, Q1D, ne, B, Bt, D, x, y, D1D, Q1D);
      }
   }
}

void MassIntegrator::AddAbsMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      MFEM_ABORT("AddAbsMultPA not implemented with CEED!");
      ceedOp->AddMult(x, y);
   }
   else
   {
      MFEM_VERIFY(!fespace->UsesRaggedTensorBasis(),
                  "AbsMultPA not implemented for ragged tensor basis");
      MFEM_VERIFY(!pa_tmo, "AbsMultPA not implemented for TMO PA");
      Vector abs_pa_data(pa_data);
      abs_pa_data.Abs();
      Array<real_t> absB(maps->B);
      Array<real_t> absBt(maps->Bt);
      absB.Abs();
      absBt.Abs();

      ApplyPAKernels::Run(dim, dofs1D, quad1D, ne, absB, absBt, abs_pa_data,
                          x, y, dofs1D, quad1D);
   }
}

void MassIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   AddMultPA(x, y);
}

void MassIntegrator::AddAbsMultTransposePA(const Vector &x, Vector &y) const
{
   AddAbsMultPA(x, y);
}

} // namespace mfem
