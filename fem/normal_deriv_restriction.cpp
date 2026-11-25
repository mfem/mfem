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

#include "normal_deriv_restriction_kernels.hpp"
#include "kernel_dispatch.hpp"
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

/// \cond DO_NOT_DOCUMENT

namespace internal
{

NormalDerivativeMultKernelsType NormalDerivativeMultKernels::Fallback(int DIM,
                                                                      int)
{
   if (DIM == 2) { return &L2NormalDerivativeFaceRestriction::Mult2D<0>; }
   if (DIM == 3) { return &L2NormalDerivativeFaceRestriction::Mult3D<0>; }
   MFEM_ABORT("Dimension not supported.");
}

/// Register the existing specializations once, on first use.
static void RegisterNormalDerivativeMultKernels()
{
   struct Kernels
   {
      Kernels()
      {
         NormalDerivativeMultKernels::Specialization<2,1>::Add();
         NormalDerivativeMultKernels::Specialization<2,2>::Add();
         NormalDerivativeMultKernels::Specialization<2,3>::Add();
         NormalDerivativeMultKernels::Specialization<2,4>::Add();
         NormalDerivativeMultKernels::Specialization<2,5>::Add();
         NormalDerivativeMultKernels::Specialization<2,6>::Add();
         NormalDerivativeMultKernels::Specialization<2,7>::Add();
         NormalDerivativeMultKernels::Specialization<2,8>::Add();
         NormalDerivativeMultKernels::Specialization<3,1>::Add();
         NormalDerivativeMultKernels::Specialization<3,2>::Add();
         NormalDerivativeMultKernels::Specialization<3,3>::Add();
         NormalDerivativeMultKernels::Specialization<3,4>::Add();
         NormalDerivativeMultKernels::Specialization<3,5>::Add();
         NormalDerivativeMultKernels::Specialization<3,6>::Add();
         NormalDerivativeMultKernels::Specialization<3,7>::Add();
         NormalDerivativeMultKernels::Specialization<3,8>::Add();
      }
   };
   static Kernels kernels;
}

NormalDerivativeAddMultTransposeKernelsType
NormalDerivativeAddMultTransposeKernels::Fallback(int DIM, int)
{
   if (DIM == 2) { return &L2NormalDerivativeFaceRestriction::AddMultTranspose2D<0>; }
   if (DIM == 3) { return &L2NormalDerivativeFaceRestriction::AddMultTranspose3D<0>; }
   MFEM_ABORT("Dimension not supported.");
}

/// Register the existing specializations once, on first use.
static void RegisterNormalDerivativeAddMultTransposeKernels()
{
   struct Kernels
   {
      Kernels()
      {
         NormalDerivativeAddMultTransposeKernels::Specialization<2,1>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<2,2>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<2,3>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<2,4>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<2,5>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<2,6>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<2,7>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<2,8>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<3,1>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<3,2>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<3,3>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<3,4>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<3,5>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<3,6>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<3,7>::Add();
         NormalDerivativeAddMultTransposeKernels::Specialization<3,8>::Add();
      }
   };
   static Kernels kernels;
}

} // namespace internal

/// \endcond DO_NOT_DOCUMENT

void L2NormalDerivativeFaceRestriction::Mult(const Vector &x, Vector &y) const
{
   if (nf == 0) { return; }
   const int d1d = fes.GetElementOrder(0) + 1;
   internal::RegisterNormalDerivativeMultKernels();
   internal::NormalDerivativeMultKernels::Run(dim, d1d, *this, x, y);
}

void L2NormalDerivativeFaceRestriction::AddMultTranspose(
   const Vector &x, Vector &y, const real_t a) const
{
   if (nf == 0) { return; }
   const int d1d = fes.GetElementOrder(0) + 1;
   internal::RegisterNormalDerivativeAddMultTransposeKernels();
   internal::NormalDerivativeAddMultTransposeKernels::Run(
      dim, d1d, *this, x, y,
      a);
}

} // namespace mfem
