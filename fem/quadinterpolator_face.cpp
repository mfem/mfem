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

#include "quadinterpolator_face_kernels.hpp"
#include "kernel_dispatch.hpp"
#include "../general/annotation.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

/// Return the sign to apply to the normals on each face to point from e1 to e2.
static void GetSigns(const FiniteElementSpace &fes, const FaceType type,
                     Array<bool> &signs)
{
   const Mesh &mesh = *fes.GetMesh();
   const int dim = mesh.SpaceDimension();
   int face_id;
   int f_ind = 0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      face_id = face.element[0].local_face_id;
      if (face.IsNonconformingCoarse())
      {
         // We skip nonconforming coarse-fine faces as they are treated
         // by the corresponding nonconforming fine-coarse faces.
         continue;
      }
      else if ( face.IsOfFaceType(type) )
      {
         if (dim==2)
         {
            if (face_id==2 || face_id==3)
            {
               signs[f_ind] = true;
            }
            else
            {
               signs[f_ind] = false;
            }
         }
         else if (dim==3)
         {
            if (face_id==0 || face_id==3 || face_id==4)
            {
               signs[f_ind] = true;
            }
            else
            {
               signs[f_ind] = false;
            }
         }
         f_ind++;
      }
   }
}

FaceQuadratureInterpolator::FaceQuadratureInterpolator(
   const FiniteElementSpace &fes,
   const IntegrationRule &ir, FaceType type_)
   : type(type_), nf(fes.GetNFbyType(type)), signs(nf),
     q_layout(QVectorLayout::byNODES)
{
   fespace = &fes;
   IntRule = &ir;
   use_tensor_products = true; // not implemented yet (not used)

   if (fespace->GetNE() == 0) { return; }
   GetSigns(*fespace, type, signs);
   MFEM_VERIFY(SupportsFESpace(fes), "Unsupported finite element space");
}

bool FaceQuadratureInterpolator::SupportsFESpace(const FiniteElementSpace &fes)
{
   const FiniteElement *fe = fes.GetTypicalFE();
   const auto *sfe = dynamic_cast<const ScalarFiniteElement*>(fe);
   const auto *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   return sfe != nullptr && tfe != nullptr && (
             tfe->GetBasisType() == BasisType::GaussLobatto ||
             tfe->GetBasisType() == BasisType::Positive);
}

/// \cond DO_NOT_DOCUMENT

namespace internal
{

FaceEvalKernelsType FaceEvalKernels::Fallback(int DIM, int VDIM, int, int)
{
   if (DIM == 2)
   {
      if (VDIM == 1) { return FaceQuadratureInterpolator::Eval2D<1>; }
      if (VDIM == 2) { return FaceQuadratureInterpolator::Eval2D<2>; }
   }
   else if (DIM == 3)
   {
      if (VDIM == 1) { return FaceQuadratureInterpolator::Eval3D<1>; }
      if (VDIM == 3) { return FaceQuadratureInterpolator::Eval3D<3>; }
   }
   MFEM_ABORT("case not supported yet");
}

/// Register the existing specializations once, on first use.
static void RegisterFaceEvalKernels()
{
   struct Kernels
   {
      Kernels()
      {
         FaceEvalKernels::Specialization<2,1,1,1>::Add();
         FaceEvalKernels::Specialization<2,1,1,2>::Add();
         FaceEvalKernels::Specialization<2,1,2,2>::Add();
         FaceEvalKernels::Specialization<2,1,2,3>::Add();
         FaceEvalKernels::Specialization<2,1,3,3>::Add();
         FaceEvalKernels::Specialization<2,1,3,4>::Add();
         FaceEvalKernels::Specialization<2,1,4,4>::Add();
         FaceEvalKernels::Specialization<2,1,4,5>::Add();
         FaceEvalKernels::Specialization<2,1,4,6>::Add();
         FaceEvalKernels::Specialization<2,1,5,5>::Add();
         FaceEvalKernels::Specialization<2,1,5,6>::Add();
         FaceEvalKernels::Specialization<2,1,5,7>::Add();
         FaceEvalKernels::Specialization<2,1,5,8>::Add();
         FaceEvalKernels::Specialization<3,1,1,1>::Add();
         FaceEvalKernels::Specialization<3,1,1,2>::Add();
         FaceEvalKernels::Specialization<3,1,2,2>::Add();
         FaceEvalKernels::Specialization<3,1,2,3>::Add();
         FaceEvalKernels::Specialization<3,1,2,4>::Add();
         FaceEvalKernels::Specialization<3,1,3,3>::Add();
         FaceEvalKernels::Specialization<3,1,3,4>::Add();
         FaceEvalKernels::Specialization<3,1,4,4>::Add();
         FaceEvalKernels::Specialization<3,1,4,5>::Add();
         FaceEvalKernels::Specialization<3,1,4,6>::Add();
         FaceEvalKernels::Specialization<3,1,5,5>::Add();
         FaceEvalKernels::Specialization<3,1,5,6>::Add();
         FaceEvalKernels::Specialization<2,2,2,2>::Add();
         FaceEvalKernels::Specialization<2,2,2,3>::Add();
         FaceEvalKernels::Specialization<2,2,3,3>::Add();
         FaceEvalKernels::Specialization<2,2,3,4>::Add();
         FaceEvalKernels::Specialization<2,2,4,4>::Add();
         FaceEvalKernels::Specialization<2,2,4,5>::Add();
         FaceEvalKernels::Specialization<2,2,4,6>::Add();
         FaceEvalKernels::Specialization<2,2,5,5>::Add();
         FaceEvalKernels::Specialization<2,2,5,6>::Add();
         FaceEvalKernels::Specialization<2,2,5,7>::Add();
         FaceEvalKernels::Specialization<2,2,5,8>::Add();
         FaceEvalKernels::Specialization<3,3,2,2>::Add();
         FaceEvalKernels::Specialization<3,3,2,3>::Add();
         FaceEvalKernels::Specialization<3,3,2,4>::Add();
         FaceEvalKernels::Specialization<3,3,3,3>::Add();
         FaceEvalKernels::Specialization<3,3,3,4>::Add();
         FaceEvalKernels::Specialization<3,3,4,4>::Add();
         FaceEvalKernels::Specialization<3,3,4,5>::Add();
         FaceEvalKernels::Specialization<3,3,4,6>::Add();
         FaceEvalKernels::Specialization<3,3,5,5>::Add();
         FaceEvalKernels::Specialization<3,3,5,6>::Add();
      }
   };
   static Kernels kernels;
}

} // namespace internal

/// \endcond DO_NOT_DOCUMENT

void FaceQuadratureInterpolator::Mult(
   const Vector &e_vec, unsigned eval_flags,
   Vector &q_val, Vector &q_der, Vector &q_det, Vector &q_nor) const
{
   if (nf == 0) { return; }
   const int vdim = fespace->GetVDim();
   const int dim = fespace->GetMesh()->Dimension();
   const FiniteElement *fe = fespace->GetTypicalTraceElement();
   const IntegrationRule *ir = IntRule;
   const DofToQuad &maps = fe->GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int nd1d = maps.ndof;
   const int nq1d = maps.nqpt;
   internal::RegisterFaceEvalKernels();
   internal::FaceEvalKernels::Run(
      dim, vdim, nd1d, nq1d,
      nf, vdim, q_layout, maps, signs, e_vec,
      q_val, q_der, q_det, q_nor, eval_flags);
}

void FaceQuadratureInterpolator::Values(
   const Vector &e_vec, Vector &q_val) const
{
   Vector q_der, q_det, q_nor;
   Mult(e_vec, VALUES, q_val, q_der, q_det, q_nor);
}

} // namespace mfem
