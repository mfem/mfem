// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "qpoint.hpp"
#include "vector.hpp"
#include "../../../linalg/kernels.hpp"

#include <cmath>

namespace mfem
{

SimplexQPoint::SimplexQPoint(const FiniteElementSpace &fes,
                             const IntegrationRule &ir)
{
   Mesh &mesh = *fes.GetMesh();
   dim = mesh.Dimension();
   vdim = fes.GetVDim();
   ne = fes.GetNE();
   nq = ir.GetNPoints();
   nh = dim * (dim + 1) / 2;
   MFEM_VERIFY(!fes.IsVariableOrder() && MmaVolumetricMeshOk(mesh, dim) &&
               TypicalFeAvailable(mesh), "SimplexQPoint: needs a fixed-order"
               " space on a 2D or 3D mesh with a single element type");
   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(dynamic_cast<const H1_TriangleElement *>(&el) ||
               dynamic_cast<const H1_TetrahedronElement *>(&el),
               "SimplexQPoint: needs a nodal H1 triangle or tetrahedron space");
   nd = el.GetDof();

   // Affine elements only, otherwise dxi/dx varies in the element and the
   // Laplacian would need its derivatives.
   MFEM_VERIFY(IsAffine(mesh), "SimplexQPoint: affine simplices only, the "
               "mesh nodes are of order "
               << mesh.GetNodes()->FESpace()->GetMaxElementOrder()
               << " (curved or high-order mesh)");
   mesh.EnsureNodes(); // for GetGeometricFactors

   // Reference tables: B and G from DofToQuad::FULL, H from CalcHessian,
   // stacked as P(q, p, i), planes p = [value, DIM gradients, NH Hessians].
   maps = &el.GetDofToQuad(ir, DofToQuad::FULL);
   MFEM_VERIFY(maps->ndof == nd && maps->nqpt == nq, "invalid DofToQuad");
   const int np = 1 + dim + nh;
   H.SetSize(nq * nh * nd);
   P.SetSize(nq * np * nd);
   const auto B = Reshape(maps->B.HostRead(), nq, nd);
   const auto G = Reshape(maps->G.HostRead(), nq, dim, nd);
   auto Hh = Reshape(H.HostWrite(), nq, nh, nd);
   auto Ph = Reshape(P.HostWrite(), nq, np, nd);
   DenseMatrix hess(nd, nh);
   for (int q = 0; q < nq; ++q)
   {
      el.CalcHessian(ir.IntPoint(q), hess);
      for (int i = 0; i < nd; ++i)
      {
         Ph(q, 0, i) = B(q, i);
         for (int j = 0; j < dim; ++j) { Ph(q, 1 + j, i) = G(q, j, i); }
         for (int s = 0; s < nh; ++s)
         {
            Hh(q, s, i) = Ph(q, 1 + dim + s, i) = hess(i, s);
         }
      }
   }

   // J and det(J) come from the mesh GeometricFactors, they are copied since
   // the mesh may delete its factors (Mesh::NodesUpdated).
   const int flags = GeometricFactors::JACOBIANS |
                     GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = mesh.GetGeometricFactors(ir, flags);
   for (Vector *v : {&J, &InvJ, &DetJ, &WDetJ}) { v->UseDevice(true); }
   J = geom->J;
   DetJ = geom->detJ;
   InvJ.SetSize(J.Size());
   WDetJ.SetSize(DetJ.Size());

   const int D = dim, NQ = nq;
   const auto W = ir.GetWeights().Read();
   const auto Jq = Reshape(J.Read(), nq, dim, dim, ne);
   const auto dJ = Reshape(DetJ.Read(), nq, ne);
   auto Ji = Reshape(InvJ.Write(), nq, dim, dim, ne);
   auto WD = Reshape(WDetJ.Write(), nq, ne);
   mfem::forall(nq * ne, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int q = idx % NQ, e = idx / NQ;
      real_t A[9], Ai[9];
      for (int j = 0; j < D; ++j)
      {
         for (int i = 0; i < D; ++i) { A[i + D * j] = Jq(q, i, j, e); }
      }
      if (D == 2) { kernels::CalcInverse<2>(A, Ai); }
      else { kernels::CalcInverse<3>(A, Ai); }
      for (int j = 0; j < D; ++j)
      {
         for (int i = 0; i < D; ++i) { Ji(q, i, j, e) = Ai[i + D * j]; }
      }
      WD(q, e) = W[q] * dJ(q, e);
   });
}

bool SimplexQPoint::IsAffine(const Mesh &mesh)
{
   const GridFunction *nodes = mesh.GetNodes();
   return !nodes || nodes->FESpace()->GetMaxElementOrder() == 1;
}

void SimplexQPoint::Metric(Vector &G) const
{
   G.SetSize(nq * dim * dim * ne);
   G.UseDevice(true);
   // K = c (I + 1 1^T), see the declaration for the vertex-ordering argument
   const real_t c = (dim == 2) ? 1.0 / std::sqrt(3.0) : std::cbrt(2.0) / 2.0;
   const int D = dim, NQ = nq;
   const auto Ji = Reshape(InvJ.Read(), nq, dim, dim, ne);
   auto Gq = Reshape(G.Write(), nq, dim, dim, ne);
   mfem::forall(nq * ne, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int q = idx % NQ, e = idx / NQ;
      // s_k = sum_a dxi_a/dx_k = -d lambda_0/dx_k (the 1 1^T part of K)
      real_t s[3];
      for (int k = 0; k < D; ++k)
      {
         s[k] = 0.0;
         for (int a = 0; a < D; ++a) { s[k] += Ji(q, a, k, e); }
      }
      for (int l = 0; l < D; ++l)
      {
         for (int k = 0; k < D; ++k)
         {
            real_t m = 0.0; // ((dxi/dx)^T (dxi/dx))(k,l), the I part of K
            for (int a = 0; a < D; ++a)
            {
               m += Ji(q, a, k, e) * Ji(q, a, l, e);
            }
            Gq(q, k, l, e) = c * (m + s[k] * s[l]);
         }
      }
   });
}

void SimplexQPoint::Mult(const Vector &x, unsigned flags, Vector &u,
                         Vector &du, Vector &lap) const
{
   MFEM_VERIFY(x.Size() == nd * vdim * ne, "invalid E-vector size");
   const bool want_lap = flags & LAPLACIANS;
   const bool want_du = flags & GRADIENTS;
   // Interpolate only the planes needed: value, then gradients, then
   // Hessians, so a prefix of the rows of P is enough.
   const int np = want_lap ? 1 + dim + nh : want_du ? 1 + dim : 1;
   const int ldp = nq * (1 + dim + nh);
   const internal::mma::SimplexVecGemm in{&P, nq * np, ldp, nd, vdim};
   internal::mma::SimplexVecInterp(ne, in, x, U);

   auto Output = [](bool on, int size, Vector &v) -> real_t *
   {
      if (!on) { return nullptr; }
      v.SetSize(size);
      v.UseDevice(true);
      return v.Write();
   };
   real_t *pu = Output(flags & VALUES, nq * vdim * ne, u);
   real_t *pdu = Output(want_du, nq * vdim * dim * ne, du);
   real_t *plap = Output(want_lap, nq * vdim * ne, lap);

   const int D = dim, NQ = nq, VD = vdim;
   const auto Uq = Reshape(U.Read(), nq, np, vdim, ne);
   const auto Ji = Reshape(InvJ.Read(), nq, dim, dim, ne);
   mfem::forall(nq * vdim * ne, [=] MFEM_HOST_DEVICE (int idx)
   {
      // idx = q + NQ (c + VD e) is also the (NQ, VDIM, NE) output index
      const int q = idx % NQ, c = (idx / NQ) % VD, e = idx / (NQ * VD);
      if (pu) { pu[idx] = Uq(q, 0, c, e); }
      if (pdu) // du_c/dx_k = sum_j du_c/dxi_j dxi_j/dx_k
      {
         for (int k = 0; k < D; ++k)
         {
            real_t s = 0.0;
            for (int j = 0; j < D; ++j)
            {
               s += Uq(q, 1 + j, c, e) * Ji(q, j, k, e);
            }
            pdu[q + NQ * (c + VD * (k + D * e))] = s;
         }
      }
      if (plap) // sum_{a,b} H_ref(a,b) (InvJ InvJ^T)(a,b), a <= b stored
      {
         real_t s = 0.0;
         for (int a = 0, h = 0; a < D; ++a)
         {
            for (int b = a; b < D; ++b, ++h)
            {
               real_t m = 0.0;
               for (int k = 0; k < D; ++k)
               {
                  m += Ji(q, a, k, e) * Ji(q, b, k, e);
               }
               s += real_t(a == b ? 1 : 2) * m * Uq(q, 1 + D + h, c, e);
            }
         }
         plap[idx] = s;
      }
   });
}

void SimplexQPoint::AddMultTranspose(const Vector &dv, const Vector &dg,
                                     Vector &y) const
{
   const bool has_v = dv.Size() > 0, has_g = dg.Size() > 0;
   if (!has_v && !has_g) { return; }
   MFEM_VERIFY(!has_v || dv.Size() == nq * vdim * ne, "invalid dv size");
   MFEM_VERIFY(!has_g || dg.Size() == nq * vdim * dim * ne, "invalid dg size");
   MFEM_VERIFY(y.Size() == nd * vdim * ne, "invalid E-vector size");

   // Pull the physical-gradient data back to the reference gradients,
   // V(q,1+j) = sum_k dxi_j/dx_k dg(q,k), then y += P^T V in one product.
   const int np = has_g ? 1 + dim : 1;
   V.SetSize(nq * np * vdim * ne);
   V.UseDevice(true);
   const int D = dim, NQ = nq, VD = vdim;
   const real_t *pv = has_v ? dv.Read() : nullptr;
   const real_t *pg = has_g ? dg.Read() : nullptr;
   const auto Ji = Reshape(InvJ.Read(), nq, dim, dim, ne);
   auto Vq = Reshape(V.Write(), nq, np, vdim, ne);
   mfem::forall(nq * vdim * ne, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int q = idx % NQ, c = (idx / NQ) % VD, e = idx / (NQ * VD);
      Vq(q, 0, c, e) = pv ? pv[idx] : 0.0;
      if (!pg) { return; }
      for (int j = 0; j < D; ++j)
      {
         real_t s = 0.0;
         for (int k = 0; k < D; ++k)
         {
            s += Ji(q, j, k, e) * pg[q + NQ * (c + VD * (k + D * e))];
         }
         Vq(q, 1 + j, c, e) = s;
      }
   });
   const int ldp = nq * (1 + dim + nh);
   internal::mma::SimplexScatterAdd(nq * np, ldp, nd, vdim * ne, P, V, y);
}

void SimplexQPoint::AddDiagonal(const Vector &dvv, const Vector &dvg,
                                const Vector &dgg, Vector &diag) const
{
   const int n = nq * vdim * ne;
   MFEM_VERIFY(dvv.Size() == 0 || dvv.Size() == n, "invalid dvv size");
   MFEM_VERIFY(dvg.Size() == 0 || dvg.Size() == n * dim, "invalid dvg size");
   MFEM_VERIFY(dgg.Size() == 0 || dgg.Size() == n * dim * dim,
               "invalid dgg size");
   MFEM_VERIFY(diag.Size() == nd * vdim * ne, "invalid E-vector size");

   const int D = dim, ND = nd, NQ = nq, VD = vdim, NP = 1 + dim + nh;
   const real_t *pvv = dvv.Size() ? dvv.Read() : nullptr;
   const real_t *pvg = dvg.Size() ? dvg.Read() : nullptr;
   const real_t *pgg = dgg.Size() ? dgg.Read() : nullptr;
   const auto Pq = Reshape(P.Read(), nq, NP, nd);
   const auto Ji = Reshape(InvJ.Read(), nq, dim, dim, ne);
   real_t *Y = diag.ReadWrite();
   // One thread per diagonal entry (i,c,e), a loop over the points.
   mfem::forall(nd * vdim * ne, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % ND, c = (idx / ND) % VD, e = idx / (ND * VD);
      real_t s = 0.0;
      for (int q = 0; q < NQ; ++q)
      {
         const int qc = q + NQ * c, nqv = NQ * VD; // (q, c) and its stride
         const real_t b = Pq(q, 0, i);
         real_t g[3]; // physical gradient of phi_i
         for (int k = 0; k < D; ++k)
         {
            g[k] = 0.0;
            for (int j = 0; j < D; ++j)
            {
               g[k] += Pq(q, 1 + j, i) * Ji(q, j, k, e);
            }
         }
         if (pvv) { s += b * b * pvv[qc + nqv * e]; }
         for (int k = 0; k < D; ++k)
         {
            if (pvg) { s += b * g[k] * pvg[qc + nqv * (k + D * e)]; }
            if (!pgg) { continue; }
            for (int l = 0; l < D; ++l)
            {
               s += g[k] * g[l] * pgg[qc + nqv * (k + D * (l + D * e))];
            }
         }
      }
      Y[idx] += s;
   });
}

} // namespace mfem
