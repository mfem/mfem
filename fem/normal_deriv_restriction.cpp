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

#include "normal_deriv_restriction.hpp"
#include "fespace.hpp"
#include "pgridfunc.hpp"
#include "fe/face_map_utils.hpp"
#include "../general/forall.hpp"

namespace mfem
{

/// Compute the face index to volume index map "face_to_vol" in 2D
static void NormalDerivativeSetupFaceIndexMap2D(
   int nf, int d, const Array<int>& face_to_elem, Array<int>& face_to_vol)
{
   const auto f2e = Reshape(face_to_elem.HostRead(), 2, 2, nf);
   auto f2v = Reshape(face_to_vol.HostWrite(), d, 2, nf);

   for (int f = 0; f < nf; ++f)
   {
      const int fid0 = f2e(0, 1, f);
      const int fid1 = f2e(1, 1, f);
      for (int side = 0; side < 2; ++side)
      {
         const int el = f2e(side, 0, f);

         if (el < 0)
         {
            for (int p = 0; p < d; ++p)
            {
               f2v(p, side, f) = -1;
            }
         }
         else
         {
            for (int p = 0; p < d; ++p)
            {
               int i, j;
               internal::FaceIdxToVolIdx2D(p, d, fid0, fid1, side, i, j);

               f2v(p, side, f) = i + d * j;
            }
         }
      }
   }
}

/// Compute the face index to volume index map "face_to_vol" in 3D
static void NormalDerivativeSetupFaceIndexMap3D(
   int nf, int d, const Array<int>& face_to_elem, Array<int>& face_to_vol)
{
   const auto f2e = Reshape(face_to_elem.HostRead(), 2, 3, nf);
   auto f2v = Reshape(face_to_vol.HostWrite(), d*d, 2, nf);

   for (int f = 0; f < nf; ++f)
   {
      const int fid0 = f2e(0, 1, f);
      const int fid1 = f2e(1, 1, f);
      for (int side = 0; side < 2; ++side)
      {
         const int el = f2e(side, 0, f);
         const int orientation = f2e(side, 2, f);

         if (el < 0)
         {
            for (int p = 0; p < d*d; ++p)
            {
               f2v(p, side, f) = -1;
            }
         }
         else
         {
            for (int p = 0; p < d*d; ++p)
            {
               int i, j, k; // 3D lexicographic index of quad point p
               internal::FaceIdxToVolIdx3D(p, d, fid0, fid1, side, orientation, i, j, k);

               f2v(p, side, f) = i + d * (j + d * k);
            }
         }
      }
   }
}

L2NormalDerivativeFaceRestriction::L2NormalDerivativeFaceRestriction(
   const FiniteElementSpace &fes_,
   const ElementDofOrdering f_ordering,
   const FaceType face_type_)
   : fes(fes_),
     face_type(face_type_),
     dim(fes.GetMesh()->Dimension()),
     nf(fes.GetNFbyType(face_type)),
     ne(fes.GetNE())
{
   MFEM_VERIFY(f_ordering == ElementDofOrdering::LEXICOGRAPHIC,
               "Non-lexicographic ordering not currently supported in "
               "L2NormalDerivativeFaceRestriction.");

   Mesh &mesh = *fes.GetMesh();

   const FiniteElement &fe = *fes.GetTypicalFE();
   const int d = fe.GetDofToQuad(fe.GetNodes(), DofToQuad::TENSOR).ndof;

   if (dim == 2)
   {
      // (el0, el1, fid0, fid1)
      face_to_elem.SetSize(nf * 4);
      face_to_vol.SetSize(2 * nf * d);
   }
   else if (dim == 3)
   {
      // (el0, el1, fid0, fid1, or0, or1)
      face_to_elem.SetSize(nf * 6);
      face_to_vol.SetSize(2 * nf * d * d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension.");
   }
   auto f2e = Reshape(face_to_elem.HostWrite(), 2, (dim == 2) ? 2 : 3, nf);

   // Populate the face_to_elem array. The elem_indicator will be used to count
   // the number of elements that are adjacent to faces of the given type.
   Array<int> elem_indicator(ne);
   elem_indicator = 0;

   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);

      if (face.IsOfFaceType(face_type))
      {
         f2e(0, 0, f_ind) = face.element[0].index;
         f2e(0, 1, f_ind) = face.element[0].local_face_id;
         if (dim == 3)
         {
            f2e(0, 2, f_ind) = face.element[0].orientation;
         }

         elem_indicator[face.element[0].index] = 1;

         if (face_type == FaceType::Interior)
         {
            const int el_idx_1 = face.element[1].index;
            if (face.IsShared())
            {
               // Indicate shared face by index >= ne
               f2e(1, 0, f_ind) = ne + el_idx_1;
            }
            else
            {
               // Face is not shared
               f2e(1, 0, f_ind) = el_idx_1;
               elem_indicator[el_idx_1] = 1;
            }
            f2e(1, 1, f_ind) = face.element[1].local_face_id;

            if (dim == 3)
            {
               f2e(1, 2, f_ind) = face.element[1].orientation;
            }
         }
         else
         {
            f2e(1, 0, f_ind) = -1;
            f2e(1, 1, f_ind) = -1;

            if (dim == 3)
            {
               f2e(1, 2, f_ind) = -1;
            }
         }

         f_ind++;
      }
   }

   // evaluate face to vol map
   if (dim == 2)
   {
      NormalDerivativeSetupFaceIndexMap2D(nf, d, face_to_elem, face_to_vol);
   }
   else if (dim == 3)
   {
      NormalDerivativeSetupFaceIndexMap3D(nf, d, face_to_elem, face_to_vol);
   }

   // Number of elements adjacent to faces of face_type
   ne_type = elem_indicator.Sum();

   // In 2D: (el, f0,f1,f2,f3, s0,s1,s2,s3)
   // In 3D: (el, f0,f1,f2,f3,f4,f5, s0,s1,s2,s3,s4,s5)
   const int elem_data_sz = (dim == 2) ? 9 : 13;

   elem_to_face.SetSize(elem_data_sz * ne_type);
   elem_to_face = -1;

   auto e2f = Reshape(elem_to_face.HostWrite(), elem_data_sz, ne_type);
   elem_indicator.PartialSum();

   const int nsides = (face_type == FaceType::Interior) ? 2 : 1;
   const int side_begin = (dim == 2) ? 5 : 7;
   for (int f = 0; f < nf; ++f)
   {
      for (int side = 0; side < nsides; ++side)
      {
         const int el = f2e(side, 0, f);
         // Skip shared faces
         if (el < ne)
         {
            const int face_id = f2e(side, 1, f);

            const int e = elem_indicator[el] - 1;
            e2f(0, e) = el;
            e2f(1 + face_id, e) = f;
            e2f(side_begin + face_id, e) = side;
         }
      }
   }
}

void L2NormalDerivativeFaceRestriction::Mult(const Vector &x, Vector &y) const
{
   if (nf == 0) { return; }
   switch (dim)
   {
      case 2:
      {
         const int d1d = fes.GetElementOrder(0) + 1;
         switch (d1d)
         {
            case 1: Mult2D<1>(x, y); break;
            case 2: Mult2D<2>(x, y); break;
            case 3: Mult2D<3>(x, y); break;
            case 4: Mult2D<4>(x, y); break;
            case 5: Mult2D<5>(x, y); break;
            case 6: Mult2D<6>(x, y); break;
            case 7: Mult2D<7>(x, y); break;
            case 8: Mult2D<8>(x, y); break;
            default: Mult2D(x, y); break;
         }
      }
      break;
      case 3:
      {
         const int d1d = fes.GetElementOrder(0) + 1;
         switch (d1d)
         {
            case 1: Mult3D<1>(x, y); break;
            case 2: Mult3D<2>(x, y); break;
            case 3: Mult3D<3>(x, y); break;
            case 4: Mult3D<4>(x, y); break;
            case 5: Mult3D<5>(x, y); break;
            case 6: Mult3D<6>(x, y); break;
            case 7: Mult3D<7>(x, y); break;
            case 8: Mult3D<8>(x, y); break;
            default: Mult3D(x, y); break; // fallback
         }
         break;
      }
      default: MFEM_ABORT("Dimension not supported."); break;
   }
}

void L2NormalDerivativeFaceRestriction::AddMultTranspose(
   const Vector &x, Vector &y, const real_t a) const
{
   if (nf == 0) { return; }
   switch (dim)
   {
      case 2:
      {
         const int d1d = fes.GetElementOrder(0) + 1;
         switch (d1d)
         {
            case 1: AddMultTranspose2D<1>(x, y, a); break;
            case 2: AddMultTranspose2D<2>(x, y, a); break;
            case 3: AddMultTranspose2D<3>(x, y, a); break;
            case 4: AddMultTranspose2D<4>(x, y, a); break;
            case 5: AddMultTranspose2D<5>(x, y, a); break;
            case 6: AddMultTranspose2D<6>(x, y, a); break;
            case 7: AddMultTranspose2D<7>(x, y, a); break;
            case 8: AddMultTranspose2D<8>(x, y, a); break;
            default: AddMultTranspose2D(x, y, a); break;
         }
      }
      break;
      case 3:
      {
         const int d1d = fes.GetElementOrder(0) + 1;
         switch (d1d)
         {
            case 1: AddMultTranspose3D<1>(x, y, a); break;
            case 2: AddMultTranspose3D<2>(x, y, a); break;
            case 3: AddMultTranspose3D<3>(x, y, a); break;
            case 4: AddMultTranspose3D<4>(x, y, a); break;
            case 5: AddMultTranspose3D<5>(x, y, a); break;
            case 6: AddMultTranspose3D<6>(x, y, a); break;
            case 7: AddMultTranspose3D<7>(x, y, a); break;
            case 8: AddMultTranspose3D<8>(x, y, a); break;
            default: AddMultTranspose3D(x, y, a); break; // fallback
         }
         break;
      }
      default: MFEM_ABORT("Not yet implemented"); break;
   }
}

template <int T_D1D>
void L2NormalDerivativeFaceRestriction::Mult2D(const Vector &x, Vector &y) const
{
   const int vd = fes.GetVDim();
   const bool t = fes.GetOrdering() == Ordering::byVDIM;
   const int num_elem = ne;

   const FiniteElement &fe = *fes.GetTypicalFE();
   const DofToQuad &maps = fe.GetDofToQuad(fe.GetNodes(), DofToQuad::TENSOR);

   const int q = maps.nqpt;
   const int d = maps.ndof;

   Vector face_nbr_data = GetLVectorFaceNbrData(fes, x, face_type);
   const int ne_shared = face_nbr_data.Size() / d / d / vd;

   MFEM_VERIFY(q == d, "");
   MFEM_VERIFY(T_D1D == d || T_D1D == 0, "");

   // derivative of 1D basis function
   const auto G_ = Reshape(maps.G.Read(), q, d);
   // (el0, el1, fid0, fid1)
   const auto f2e = Reshape(face_to_elem.Read(), 2, 2, nf);

   const auto f2v = Reshape(face_to_vol.Read(), q, 2, nf);

   // if byvdim, d_x has shape (vdim, nddof, nddof, ne)
   // otherwise, d_x has shape (nddof, nddof, ne, vdim)
   const auto d_x = Reshape(x.Read(), t?vd:d, d, t?d:ne, t?ne:vd);
   const auto d_x_shared = Reshape(face_nbr_data.Read(),
                                   t?vd:d, d, t?d:ne_shared, t?ne_shared:vd);
   auto d_y = Reshape(y.Write(), q, vd, 2, nf);

   mfem::forall_2D(nf, 2, q, [=] MFEM_HOST_DEVICE (int f) -> void
   {
      constexpr int MD = (T_D1D) ? T_D1D : DofQuadLimits::MAX_D1D;

      MFEM_SHARED real_t G_s[MD*MD];
      DeviceMatrix G(G_s, q, d);

      MFEM_SHARED int E[2];
      MFEM_SHARED int FID[2];
      MFEM_SHARED int F2V[2][MD];

      if (MFEM_THREAD_ID(x) == 0)
      {
         MFEM_FOREACH_THREAD(j, y, d)
         {
            for (int i = 0; i < q; ++i)
            {
               G(i, j) = G_(i, j);
            }
         }
      }

      MFEM_FOREACH_THREAD(side, x, 2)
      {
         if (MFEM_THREAD_ID(y) == 0)
         {
            E[side] = f2e(side, 0, f);
            FID[side] = f2e(side, 1, f);
         }

         MFEM_FOREACH_THREAD(j, y, d)
         {
            F2V[side][j] = f2v(j, side, f);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(side, x, 2)
      {
         const int el = E[side];
         const bool shared = (el >= num_elem);
         const auto &d_x_e = shared ? d_x_shared : d_x;
         const int el_idx = shared ? el - num_elem : el;

         const int face_id = FID[side];

         MFEM_FOREACH_THREAD(p, y, q)
         {
            if (el < 0)
            {
               for (int c = 0; c < vd; ++c)
               {
                  d_y(p, c, side, f) = 0.0;
               }
            }
            else
            {
               const int ij = F2V[side][p];
               const int i = ij % q;
               const int j = ij / q;

               for (int c=0; c < vd; ++c)
               {
                  real_t grad_n = 0;
                  for (int kk=0; kk < d; ++kk)
                  {
                     const int k = (face_id == 0 || face_id == 2) ? i : kk;
                     const int l = (face_id == 0 || face_id == 2) ? kk : j;
                     const real_t g = (face_id == 0 || face_id == 2) ? G(j,l) : G(i,k);
                     grad_n += g * d_x_e(t?c:k, t?k:l, t?l:el_idx, t?el_idx:c);
                  }
                  d_y(p, c, side, f) = grad_n;
               }
            }
         }
      }
   });
}

template <int T_D1D>
void L2NormalDerivativeFaceRestriction::Mult3D(const Vector &x, Vector &y) const
{
   const int vd = fes.GetVDim();
   const bool t = fes.GetOrdering() == Ordering::byVDIM;
   const int num_elem = ne;

   const FiniteElement &fe = *fes.GetTypicalFE();
   const DofToQuad &maps = fe.GetDofToQuad(fe.GetNodes(), DofToQuad::TENSOR);

   const int q = maps.nqpt;
   const int d = maps.ndof;
   const int q2d = q * q;

   Vector face_nbr_data = GetLVectorFaceNbrData(fes, x, face_type);
   const int ne_shared = face_nbr_data.Size() / d / d / d / vd;

   MFEM_VERIFY(q == d, "");
   MFEM_VERIFY(T_D1D == d || T_D1D == 0, "");

   const auto G_ = Reshape(maps.G.Read(), q, d);
   // (el0, el1, fid0, fid1, or0, or1)
   const auto f2e = Reshape(face_to_elem.Read(), 2, 3, nf);
   const auto f2v = Reshape(face_to_vol.Read(), q2d, 2, nf);

   // t ? (vdim, d, d, d, ne) : (d, d, d, ne, vdim)
   const auto d_x = Reshape(x.Read(), t?vd:d, d, d, t?d:ne, t?ne:vd);
   const auto d_x_shared = Reshape(face_nbr_data.Read(),
                                   t?vd:d, d, d, t?d:ne_shared, t?ne_shared:vd);
   auto d_y = Reshape(y.Write(), q2d, vd, 2, nf);

   mfem::forall_2D(nf, q2d, 2, [=] MFEM_HOST_DEVICE (int f) -> void
   {
      static constexpr int MD = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;

      MFEM_SHARED real_t G_s[MD*MD];
      DeviceMatrix G(G_s, d, q);

      MFEM_SHARED int E[2];
      MFEM_SHARED int FID[2];
      MFEM_SHARED int F2V[2][MD*MD];

      // Load G matrix into shared memory
      if (MFEM_THREAD_ID(y) == 0)
      {
         MFEM_FOREACH_THREAD(j, x, d*q)
         {
            const int p = j % q;
            const int k = j / q;
            G(k, p) = G_(p, k);
         }
      }

      MFEM_FOREACH_THREAD(side, y, 2)
      {
         if (MFEM_THREAD_ID(x) == 0)
         {
            E[side] = f2e(side, 0, f);
            FID[side] = f2e(side, 1, f);
         }
         MFEM_FOREACH_THREAD(j, x, q2d)
         {
            F2V[side][j] = f2v(j, side, f);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(side, y, 2)
      {
         const int el = E[side];
         const bool shared = (el >= num_elem);
         const auto &d_x_e = shared ? d_x_shared : d_x;
         const int el_idx = shared ? el - num_elem : el;

         const int face_id = FID[side];

         // Is this face parallel to the x-y plane in reference coordinates?
         const bool xy_plane = (face_id == 0 || face_id == 5);
         const bool xz_plane = (face_id == 1 || face_id == 3);
         const bool yz_plane = (face_id == 2 || face_id == 4);

         MFEM_FOREACH_THREAD(p, x, q2d)
         {
            if (el_idx < 0)
            {
               for (int c = 0; c < vd; ++c)
               {
                  d_y(p, c, side, f) = 0.0;
               }
            }
            else
            {
               const int ijk = F2V[side][p];
               const int k = ijk / q2d;
               const int i = ijk % q;
               const int j = (ijk - q2d*k) / q;

               // the fixed 1D index of the normal component of the face
               // quadrature point
               const int g_row = yz_plane ? i : xz_plane ? j : k;

               for (int c = 0; c < vd; ++c)
               {
                  real_t grad_n = 0.0;

                  for (int kk = 0; kk < d; ++kk)
                  {
                     // (l, m, n) 3D lexicographic index of interior points used
                     // in evaluating normal derivatives
                     const int l = yz_plane ? kk : i;
                     const int m = xz_plane ? kk : j;
                     const int n = xy_plane ? kk : k;

                     const real_t g = G(kk, g_row);

                     grad_n += g * d_x_e(t?c:l, t?l:m, t?m:n, t?n:el_idx, t?el_idx:c);
                  }
                  d_y(p, c, side, f) = grad_n;
               }
            }
         }
      }
   });
}

template <int T_D1D>
void L2NormalDerivativeFaceRestriction::AddMultTranspose2D(
   const Vector &y, Vector &x, const real_t a) const
{
   const int vd = fes.GetVDim();
   const bool t = fes.GetOrdering() == Ordering::byVDIM;

   const FiniteElement &fe = *fes.GetTypicalFE();
   const DofToQuad &maps = fe.GetDofToQuad(fe.GetNodes(), DofToQuad::TENSOR);

   const int q = maps.nqpt;
   const int d = maps.ndof;

   // derivative of 1D basis function
   auto G_ = Reshape(maps.G.Read(), q, d);

   // entries of e2f: (el,f0,f1,f2,f3,s0,s1,s2,s3)
   auto e2f = Reshape(elem_to_face.Read(), 9, ne_type);

   auto f2v = Reshape(face_to_vol.Read(), d, 2, nf);

   // if byvdim, d_x has shape (vdim, nddof, nddof, ne)
   // otherwise, d_x has shape (nddof, nddof, ne, vdim)
   auto d_x = Reshape(x.ReadWrite(), t?vd:d, d, t?d:ne, t?ne:vd);
   auto d_y = Reshape(y.Read(), q, vd, 2, nf);

   mfem::forall_2D(ne_type, d, d, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MD = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;

      MFEM_SHARED real_t y_s[MD];
      MFEM_SHARED int pp[MD];
      MFEM_SHARED int jj;
      if (MFEM_THREAD_ID(x) == 0 && MFEM_THREAD_ID(y) == 0) { jj = 0; }

      MFEM_SHARED real_t BG[MD*MD];
      DeviceMatrix G(BG, q, d);

      MFEM_SHARED real_t x_s[MD*MD];
      DeviceMatrix xx(x_s, d, d);

      MFEM_SHARED int el; // global element index
      MFEM_SHARED int faces[4];
      MFEM_SHARED int sides[4];

      MFEM_FOREACH_THREAD(i,x,d)
      {
         MFEM_FOREACH_THREAD(p,y,q)
         {
            G(p,i) = a * G_(p,i);
            xx(p,i) = 0.0;
         }
      }

      if (MFEM_THREAD_ID(y) == 0)
      {
         if (MFEM_THREAD_ID(x) == 0)
         {
            el = e2f(0, e);
         }

         MFEM_FOREACH_THREAD(i, x, 4)
         {
            faces[i] = e2f(1 + i, e);
            sides[i] = e2f(5 + i, e);
         }
      }
      MFEM_SYNC_THREAD;

      for (int face_id=0; face_id < 4; ++face_id)
      {
         const int f = faces[face_id];

         if (f < 0) { continue; }

         const int side = sides[face_id];

         if (MFEM_THREAD_ID(y) == 0)
         {
            MFEM_FOREACH_THREAD(p,x,d)
            {
               y_s[p] = d_y(p, 0, side, f);

               const int ij = f2v(p, side, f);
               const int i = ij % q;
               const int j = ij / q;

               pp[(face_id == 0 || face_id == 2) ? i : j] = p;
               if (MFEM_THREAD_ID(x) == 0)
               {
                  jj = (face_id == 0 || face_id == 2) ? j : i;
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(k,x,d)
         {
            MFEM_FOREACH_THREAD(l,y,d)
            {
               const int p = (face_id == 0 || face_id == 2) ? pp[k] : pp[l];
               const int kk = (face_id == 0 || face_id == 2) ? l : k;
               const real_t g = G(jj, kk);
               xx(k,l) += g * y_s[p];
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(k,x,d)
      {
         MFEM_FOREACH_THREAD(l,y,d)
         {
            const int c = 0;
            d_x(t?c:k, t?k:l, t?l:el, t?el:c) += xx(k,l);
         }
      }
   });
}

template <int T_D1D>
void L2NormalDerivativeFaceRestriction::AddMultTranspose3D(
   const Vector &y, Vector &x, const real_t a) const
{
   const int vd = fes.GetVDim();
   const bool t = fes.GetOrdering() == Ordering::byVDIM;

   MFEM_VERIFY(vd == 1, "vdim > 1 not supported.");

   const FiniteElement &fe = *fes.GetTypicalFE();
   const DofToQuad &maps = fe.GetDofToQuad(fe.GetNodes(), DofToQuad::TENSOR);

   const int q = maps.nqpt;
   const int d = maps.ndof;
   const int q2d = q * q;

   MFEM_VERIFY(q == d, "");
   MFEM_VERIFY(T_D1D == d || T_D1D == 0, "");

   auto G_ = Reshape(maps.G.Read(), q, d);

   // (el, f0,f1,f2,f3,f4,f5, s0,s1,s2,s3,s4,s5)
   auto e2f = Reshape(elem_to_face.Read(), 13, ne_type);

   auto f2v = Reshape(face_to_vol.Read(), q2d, 2, nf);

   auto d_x = Reshape(x.ReadWrite(), t?vd:d, d, d, t?d:ne, t?ne:vd);
   const auto d_y = Reshape(y.Read(), q2d, vd, 2, nf);

   mfem::forall_2D(ne_type, q, q, [=] MFEM_HOST_DEVICE (int e) -> void
   {
      static constexpr int MD = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;

      MFEM_SHARED int pp[MD][MD];
      MFEM_SHARED real_t y_s[MD*MD];
      MFEM_SHARED int jj;
      if (MFEM_THREAD_ID(x) == 0 && MFEM_THREAD_ID(y) == 0) { jj = 0; }

      MFEM_SHARED real_t xx_s[MD*MD*MD];
      auto xx = Reshape(xx_s, d, d, d);

      MFEM_SHARED real_t G_s[MD*MD];
      DeviceMatrix G(G_s, q, d);

      MFEM_SHARED int el;
      MFEM_SHARED int faces[6];
      MFEM_SHARED int sides[6];

      // Load G into shared memory
      MFEM_FOREACH_THREAD(j, x, d)
      {
         MFEM_FOREACH_THREAD(i, y, q)
         {
            G(i, j) = a * G_(i, j);
            G(i, j) = a * G_(i, j);
            G(i, j) = a * G_(i, j);
         }
      }

      if (MFEM_THREAD_ID(y) == 0)
      {
         if (MFEM_THREAD_ID(x) == 0)
         {
            el = e2f(0, e); // global element index
         }

         MFEM_FOREACH_THREAD(i, x, 6)
         {
            faces[i] = e2f(1 + i, e);
            sides[i] = e2f(7 + i, e);
         }
      }

      MFEM_FOREACH_THREAD(k, x, d)
      {
         MFEM_FOREACH_THREAD(j, y, d)
         {
            for (int i = 0; i < d; ++i)
            {
               xx(i, j, k) = 0.0;
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int face_id = 0; face_id < 6; ++face_id)
      {
         const int f = faces[face_id];

         if (f < 0)
         {
            continue;
         }

         const int side = sides[face_id];

         // is this face parallel to the x-y plane in reference coordinates?
         const bool xy_plane = (face_id == 0 || face_id == 5);
         const bool xz_plane = (face_id == 1 || face_id == 3);

         MFEM_FOREACH_THREAD(p1, x, q)
         {
            MFEM_FOREACH_THREAD(p2, y, q)
            {
               const int p = p1 + q * p2;
               y_s[p] = d_y(p, 0, side, f);

               const int ijk = f2v(p, side, f);
               const int k = ijk / q2d;
               const int i = ijk % q;
               const int j = (ijk - q2d*k) / q;

               pp[(xy_plane || xz_plane) ? i : j][(xy_plane) ? j : k] = p;
               if (MFEM_THREAD_ID(x) == 0 && MFEM_THREAD_ID(y) == 0)
               {
                  jj = (xy_plane) ? k : (xz_plane) ? j : i;
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(n, x, d)
         {
            MFEM_FOREACH_THREAD(m, y, d)
            {
               for (int l = 0; l < d; ++l)
               {
                  const int p = (xy_plane) ? pp[l][m] : (xz_plane) ? pp[l][n] : pp[m][n];
                  const int kk = (xy_plane) ? n : (xz_plane) ? m : l;
                  const real_t g = G(jj, kk);
                  xx(l, m, n) += g * y_s[p];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // map back to global array
      MFEM_FOREACH_THREAD(n, x, d)
      {
         MFEM_FOREACH_THREAD(m, y, d)
         {
            for (int l = 0; l < d; ++l)
            {
               const int c = 0;
               d_x(t?c:l, t?l:m, t?m:n, t?n:el, t?el:c) += xx(l, m, n);
            }
         }
      }
   });
}

} // namespace mfem
