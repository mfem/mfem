// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include "pgridfunc.hpp"
#include "fe/face_map_utils.hpp"
#include "../general/forall.hpp"

namespace mfem
{

L2NormalDerivativeFaceRestriction::L2NormalDerivativeFaceRestriction(
   const FiniteElementSpace &fes_,
   const ElementDofOrdering ordering,
   const FaceType face_type_)
   : fes(fes_),
     face_type(face_type_),
     dim(fes.GetMesh()->Dimension()),
     nf(fes.GetNFbyType(face_type)),
     ne(fes.GetNE())
{
   Mesh &mesh = *fes.GetMesh();

   if (dim == 2)
   {
      // (el0, el1, fid0, fid1)
      face_to_elem.SetSize(nf * 4);
   }
   else if (dim == 3)
   {
      // (el0, el1, fid0, fid1, or0, or1)
      face_to_elem.SetSize(nf * 6);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension.");
   }
   auto f2e = Reshape(face_to_elem.HostWrite(), (dim == 2) ? 4 : 6, nf);

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
         f2e(0, f_ind) = face.element[0].index;
         f2e(2, f_ind) = face.element[0].local_face_id;
         if (dim == 3)
         {
            f2e(4, f_ind) = face.element[0].orientation;
         }

         elem_indicator[face.element[0].index] = 1;

         if (face_type == FaceType::Interior)
         {
            const int el_idx_1 = face.element[1].index;
            if (face.IsShared())
            {
               // Indicate shared face by index >= ne
               f2e(1, f_ind) = ne + el_idx_1;
            }
            else
            {
               // Face is not shared
               f2e(1, f_ind) = el_idx_1;
               elem_indicator[el_idx_1] = 1;
            }
            f2e(3, f_ind) = face.element[1].local_face_id;

            if (dim == 3)
            {
               f2e(5, f_ind) = face.element[1].orientation;
            }
         }
         else
         {
            f2e(1, f_ind) = -1;
            f2e(3, f_ind) = -1;

            if (dim == 3)
            {
               f2e(5, f_ind) = -1;
            }
         }

         f_ind++;
      }
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
         const int el = f2e(side, f);
         // Skip shared faces
         if (el < ne)
         {
            const int face_id = f2e(2 + side, f);

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
      case 2: Mult2D(x, y); break;
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
   const Vector &x, Vector &y, const double a) const
{
   if (nf == 0) { return; }
   switch (dim)
   {
      case 2: AddMultTranspose2D(x, y, a); break;
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

void L2NormalDerivativeFaceRestriction::Mult2D(const Vector &x, Vector &y) const
{
   int ne_shared = 0;
   const double *face_nbr_data = nullptr;

#ifdef MFEM_USE_MPI
   std::unique_ptr<ParGridFunction> x_gf;
   if (const auto *pfes = dynamic_cast<const ParFiniteElementSpace*>(&fes))
   {
      if (face_type == FaceType::Interior)
      {
         x_gf.reset(new ParGridFunction(const_cast<ParFiniteElementSpace*>(pfes),
                                        const_cast<Vector&>(x), 0));
         x_gf->ExchangeFaceNbrData();
         face_nbr_data = x_gf->FaceNbrData().Read();
         ne_shared = pfes->GetParMesh()->GetNFaceNeighborElements();
      }
   }
#endif

   const int vd = fes.GetVDim();
   const bool t = fes.GetOrdering() == Ordering::byVDIM;
   const int num_elem = ne;

   const FiniteElement &fe = *fes.GetFE(0);
   const DofToQuad &maps = fe.GetDofToQuad(fe.GetNodes(), DofToQuad::TENSOR);

   const int q = maps.nqpt;
   const int d = maps.ndof;

   MFEM_VERIFY(q == d, "");

   // derivative of 1D basis function
   const auto G_ = Reshape(maps.G.Read(), q, d);
   // (el0, el1, fid0, fid1)
   const auto f2e = Reshape(face_to_elem.Read(), 4, nf);

   // if byvdim, d_x has shape (vdim, nddof, nddof, ne)
   // otherwise, d_x has shape (nddof, nddof, ne, vdim)
   const auto d_x = Reshape(x.Read(), t?vd:d, d, t?d:ne, t?ne:vd);
   const auto d_x_shared = Reshape(face_nbr_data,
                                   t?vd:d, d, t?d:ne_shared, t?ne_shared:vd);
   auto d_y = Reshape(y.Write(), q, vd, 2, nf);

   mfem::forall_2D(nf, 2, q, [=] MFEM_HOST_DEVICE (int f) -> void
   {
      MFEM_SHARED double G_s[MAX_D1D*MAX_D1D];
      DeviceMatrix G(G_s, q, d);

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
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(side, x, 2)
      {
         const int el = f2e(side, f);
         const bool shared = (el >= num_elem);
         const auto &d_x_e = shared ? d_x_shared : d_x;
         const int el_idx = shared ? el - num_elem : el;

         const int face_id = f2e(2 + side, f);
         const int fid0 = f2e(2, f);
         const int fid1 = f2e(3, f);

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
               int i, j;
               internal::FaceIdxToVolIdx2D(p, q, fid0, fid1, side, i, j);
               for (int c=0; c < vd; ++c)
               {
                  double grad_n = 0;
                  for (int kk=0; kk < d; ++kk)
                  {
                     const int k = (face_id == 0 || face_id == 2) ? i : kk;
                     const int l = (face_id == 0 || face_id == 2) ? kk : j;
                     const double g = (face_id == 0 || face_id == 2) ? G(j,l) : G(i,k);
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
   static constexpr int MD = T_D1D ? T_D1D : MAX_D1D;

   int ne_shared = 0;
   const double *face_nbr_data = nullptr;

#ifdef MFEM_USE_MPI
   std::unique_ptr<ParGridFunction> x_gf;
   if (const auto *pfes = dynamic_cast<const ParFiniteElementSpace*>(&fes))
   {
      if (face_type == FaceType::Interior)
      {
         x_gf.reset(new ParGridFunction(const_cast<ParFiniteElementSpace*>(pfes),
                                        const_cast<Vector&>(x), 0));
         x_gf->ExchangeFaceNbrData();
         face_nbr_data = x_gf->FaceNbrData().Read();
         ne_shared = pfes->GetParMesh()->GetNFaceNeighborElements();
      }
   }
#endif

   const int vd = fes.GetVDim();
   const bool t = fes.GetOrdering() == Ordering::byVDIM;
   const int num_elem = ne;

   const FiniteElement &fe = *fes.GetFE(0);
   const DofToQuad &maps = fe.GetDofToQuad(fe.GetNodes(), DofToQuad::TENSOR);

   const int q = maps.nqpt;
   const int d = maps.ndof;
   const int q2d = q * q;

   MFEM_VERIFY(q == d, "");
   MFEM_VERIFY(T_D1D == d || T_D1D == 0, "");

   const auto G_ = Reshape(maps.G.Read(), q, d);
   // (el0, el1, fid0, fid1, or0, or1)
   const auto f2e = Reshape(face_to_elem.Read(), 6, nf);

   // t ? (vdim, d, d, d, ne) : (d, d, d, ne, vdim)
   const auto d_x = Reshape(x.Read(), t?vd:d, d, d, t?d:ne, t?ne:vd);
   const auto d_x_shared = Reshape(face_nbr_data,
                                   t?vd:d, d, d, t?d:ne_shared, t?ne_shared:vd);
   auto d_y = Reshape(y.Write(), q2d, vd, 2, nf);

   mfem::forall_2D(nf, 2, q2d, [=] MFEM_HOST_DEVICE (int f) -> void
   {
      MFEM_SHARED double G_s[MD*MD];
      DeviceMatrix G(G_s, q, d);

      // Load G matrix into shared memory
      if (MFEM_THREAD_ID(x) == 0)
      {
         MFEM_FOREACH_THREAD(j, y, d*q)
         {
            G[j] = G_[j];
         }
      }
      MFEM_SYNC_THREAD;

      const int fid0 = f2e(2, f);
      const int fid1 = f2e(3, f);

      MFEM_FOREACH_THREAD(side, x, 2)
      {
         const int el = f2e(side, f);
         const bool shared = (el >= num_elem);
         const auto &d_x_e = shared ? d_x_shared : d_x;
         const int el_idx = shared ? el - num_elem : el;

         const int face_id = f2e(2 + side, f);
         const int orientation = f2e(4 + side, f);

         // Is this face parallel to the x-y plane in reference coordinates?
         const bool xy_plane = (face_id == 0 || face_id == 5);
         const bool xz_plane = (face_id == 1 || face_id == 3);
         const bool yz_plane = (face_id == 2 || face_id == 4);

         MFEM_FOREACH_THREAD(p, y, q2d)
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
               int i, j, k; // 3D lexicographic index of quad point p
               internal::FaceIdxToVolIdx3D(p, q, fid0, fid1, side, orientation, i, j, k);

               for (int c = 0; c < vd; ++c)
               {
                  double grad_n = 0.0;

                  for (int kk = 0; kk < d; ++kk)
                  {
                     // (l, m, n) 3D lexicographic index of interior points used
                     // in evaluating normal derivatives
                     const int l = yz_plane ? kk : i;
                     const int m = xz_plane ? kk : j;
                     const int n = xy_plane ? kk : k;

                     // the fixed 1D index of the normal component of the face
                     // quadrature point
                     const int g_row = yz_plane ? i : xz_plane ? j : k;
                     const double g = G(g_row, kk);

                     grad_n += g * d_x_e(t?c:l, t?l:m, t?m:n, t?n:el_idx, t?el_idx:c);
                  }
                  d_y(p, c, side, f) = grad_n;
               }
            }
         }
      }
   });
}

void L2NormalDerivativeFaceRestriction::AddMultTranspose2D(
   const Vector &y, Vector &x, const double a) const
{
   const int vd = fes.GetVDim();
   const bool t = fes.GetOrdering() == Ordering::byVDIM;

   const FiniteElement &fe = *fes.GetFE(0);
   const DofToQuad &maps = fe.GetDofToQuad(fe.GetNodes(), DofToQuad::TENSOR);

   const int q = maps.nqpt;
   const int d = maps.ndof;

   // derivative of 1D basis function
   auto G_ = Reshape(maps.G.Read(), q, d);

   // entries of e2f: (el,f0,f1,f2,f3,s0,s1,s2,s3)
   auto e2f = Reshape(elem_to_face.Read(), 9, ne_type);
   // entries of f2e: (el0, el1, fid0, fid1)
   auto f2e = Reshape(face_to_elem.Read(), 4, nf);

   // if byvdim, d_x has shape (vdim, nddof, nddof, ne)
   // otherwise, d_x has shape (nddof, nddof, ne, vdim)
   auto d_x = Reshape(x.ReadWrite(), t?vd:d, d, t?d:ne, t?ne:vd);
   auto d_y = Reshape(y.Read(), q, vd, 2, nf);

   mfem::forall_2D(ne_type, d, d, [=] MFEM_HOST_DEVICE (int e)
   {
      const int el = e2f(0, e); // global element index

      MFEM_SHARED double y_s[MAX_D1D];
      MFEM_SHARED int pp[MAX_D1D];
      MFEM_SHARED double jj;
      MFEM_SHARED double BG[MAX_D1D*MAX_D1D];
      DeviceMatrix G(BG + d*q, q, d);

      MFEM_SHARED double x_s[MAX_D1D*MAX_D1D];
      DeviceMatrix xx(x_s, d, d);

      MFEM_FOREACH_THREAD(i,x,d)
      {
         MFEM_FOREACH_THREAD(p,y,q)
         {
            G(p,i) = a * G_(p,i);
            xx(p,i) = 0.0;
         }
      }
      MFEM_SYNC_THREAD;

      for (int face_id=0; face_id < 4; ++face_id)
      {
         const int f = e2f(1+face_id, e);

         if (f < 0) { continue; }

         const int side = e2f(5+face_id, e);
         const int fid0 = f2e(2, f);
         const int fid1 = f2e(3, f);

         if (MFEM_THREAD_ID(y) == 0)
         {
            MFEM_FOREACH_THREAD(p,x,d)
            {
               y_s[p] = d_y(p, 0, side, f);

               int i, j;
               internal::FaceIdxToVolIdx2D(p, q, fid0, fid1, side, i, j);

               if (face_id == 0 || face_id == 2)
               {
                  pp[i] = p;
                  jj = j;
               }
               else
               {
                  pp[j] = p;
                  jj = i;
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
               const double g = G(jj, kk);
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
   const Vector &y, Vector &x, const double a) const
{
   static constexpr int MD = T_D1D ? T_D1D : MAX_D1D;

   const int vd = fes.GetVDim();
   const bool t = fes.GetOrdering() == Ordering::byVDIM;

   MFEM_VERIFY(vd == 1, "vdim > 1 not supported.");

   const FiniteElement &fe = *fes.GetFE(0);
   const DofToQuad &maps = fe.GetDofToQuad(fe.GetNodes(), DofToQuad::TENSOR);

   const int q = maps.nqpt;
   const int d = maps.ndof;
   const int q2d = q * q;

   MFEM_VERIFY(q == d, "");
   MFEM_VERIFY(T_D1D == d || T_D1D == 0, "");

   auto G_ = Reshape(maps.G.Read(), q, d);

   // (el, f0,f1,f2,f3,f4,f5, s0,s1,s2,s3,s4,s5)
   auto e2f = Reshape(elem_to_face.Read(), 13, ne_type);
   // (el0, el1, fid0, fid1, or0, or1)
   auto f2e = Reshape(face_to_elem.Read(), 6, nf);

   auto d_x = Reshape(x.ReadWrite(), t?vd:d, d, d, t?d:ne, t?ne:vd);
   const auto d_y = Reshape(y.Read(), q2d, vd, 2, nf);

   mfem::forall_3D(ne_type, q, q, q, [=] MFEM_HOST_DEVICE (int e) -> void
   {
      MFEM_SHARED int pp[MD][MD];
      MFEM_SHARED double y_s[MD*MD];
      MFEM_SHARED int jj;
      MFEM_SHARED double xx_s[MD*MD*MD];
      auto xx = Reshape(xx_s, d, d, d);

      MFEM_SHARED double G_s[MD*MD];
      DeviceMatrix G(G_s, q, d);

      if (MFEM_THREAD_ID(z) == 0)
      {
         MFEM_FOREACH_THREAD(i, x, q)
         {
            MFEM_FOREACH_THREAD(j, y, d)
            {
               G(i, j) = a * G_(i, j);
            }
         }
      }

      MFEM_FOREACH_THREAD(i, x, d)
      {
         MFEM_FOREACH_THREAD(j, y, d)
         {
            MFEM_FOREACH_THREAD(k, z, d)
            {
               xx(i, j, k) = 0.0;
            }
         }
      }
      MFEM_SYNC_THREAD;

      const int el = e2f(0, e); // global element index

      for (int face_id = 0; face_id < 6; ++face_id)
      {
         const int f = e2f(1+face_id, e);

         if (f < 0)
         {
            continue;
         }

         const int side = e2f(7 + face_id, e);
         const int orientation = f2e(4 + side, f);
         const int fid0 = f2e(2, f);
         const int fid1 = f2e(3, f);

         // is this face parallel to the x-y plane in reference coordinates?
         const bool xy_plane = (face_id == 0 || face_id == 5);
         const bool xz_plane = (face_id == 1 || face_id == 3);

         if (MFEM_THREAD_ID(z) == 0)
         {
            MFEM_FOREACH_THREAD(p1, x, q)
            {
               MFEM_FOREACH_THREAD(p2, y, q)
               {
                  const int p = p1 + q * p2;
                  y_s[p] = d_y(p, 0, side, f);

                  int i, j, k;
                  internal::FaceIdxToVolIdx3D(p, q, fid0, fid1, side, orientation, i, j, k);


                  pp[(xy_plane || xz_plane) ? i : j][(xy_plane) ? j : k] = p;
                  if (MFEM_THREAD_ID(x) == 0 && MFEM_THREAD_ID(y) == 0)
                  {
                     jj = (xy_plane) ? k : (xz_plane) ? j : i;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(l, x, d)
         {
            MFEM_FOREACH_THREAD(m, y, d)
            {
               MFEM_FOREACH_THREAD(n, z, d)
               {
                  const int p = (xy_plane) ? pp[l][m] : (xz_plane) ? pp[l][n] : pp[m][n];
                  const int kk = (xy_plane) ? n : (xz_plane) ? m : l;
                  const double g = G(jj, kk);
                  xx(l, m, n) += g * y_s[p];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // map back to global array
      MFEM_FOREACH_THREAD(l, x, d)
      {
         MFEM_FOREACH_THREAD(m, y, d)
         {
            MFEM_FOREACH_THREAD(n, z, d)
            {
               const int c = 0;
               d_x(t?c:l, t?l:m, t?m:n, t?n:el, t?el:c) += xx(l, m, n);
            }
         }
      }
   });
}

} // namespace mfem
