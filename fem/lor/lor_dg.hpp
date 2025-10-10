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

#ifndef MFEM_LOR_DG
#define MFEM_LOR_DG

#include "lor_batched.hpp"

namespace mfem
{

// BatchedLORKernel specialization for DG spaces. Not user facing. See the
// classes BatchedLORAssembly and BatchedLORKernel .
class BatchedLOR_DG : BatchedLORKernel
{
   IntegrationRule ir_face; ///< Collocated Gauss-Lobatto face quadrature rule.
   real_t kappa; ///< DG penalty parameter.
   bool has_bdr_integ; ///< Is there a boundary integrator?
   const Array<int> *bdr_markers; ///< Boundary integrator markers.
public:
   template <int ORDER, int SDIM> void Assemble2D();
   template <int ORDER> void Assemble3D();
   BatchedLOR_DG(BilinearForm &a,
                 FiniteElementSpace &fes_ho_,
                 Vector &X_vert_,
                 Vector &sparse_ij_,
                 Array<int> &sparse_mapping_)
      : BatchedLORKernel(fes_ho_, X_vert_, sparse_ij_, sparse_mapping_),
        ir_face(GetLobattoIntRule(fes_ho_.GetMesh()->GetTypicalFaceGeometry(),
                                  fes_ho_.GetMaxElementOrder() + 1))
   {
      ProjectLORCoefficient<MassIntegrator>(a, c1);
      ProjectLORCoefficient<DiffusionIntegrator>(a, c2);

      if (auto *integ = GetInteriorFaceIntegrator<DGDiffusionIntegrator>(a))
      {
         kappa = integ->GetPenaltyParameter();
      }
      else
      {
         kappa = 0.0;
      }

      has_bdr_integ = false;
      auto *bdr_face_integs = a.GetBFBFI();
      for (int i = 0; i < bdr_face_integs->Size(); ++i)
      {
         if (auto *integ = dynamic_cast<DGDiffusionIntegrator*>((*bdr_face_integs)[i]))
         {
            kappa = integ->GetPenaltyParameter();
            bdr_markers = (*a.GetBFBFI_Marker())[i];
            has_bdr_integ = true;
            break;
         }
      }
   }

   /// @brief Compute and return the face info array.
   ///
   /// The face info array has shape (6, nf), where @a nf is the number of
   /// faces. For each face @a i, the column (:,i) has entries (e0, f0, o0, e1,
   /// f1, o1), where @a e is adjacent element, @a f is the local face index,
   /// and @a o is the orientation. For boundary and shared faces, (e1, f1, o1)
   /// are all set to -1.
   Array<int> GetFaceInfo() const;

   /// @brief Compute and return the boundary penalty factor.
   ///
   /// The returned vector has shape (nq, nf), where @a nq is the number of
   /// nodes per face, and @a nf is the number of faces.
   ///
   /// The boundary penalty factor is $J_f / h = J_f^2 / J_e$ (since $h = J_e /
   /// J_f$), where $J_f$ is the face Jacobian determinant, and $J_e$ is the
   /// element Jacobian determinant.
   Vector GetBdrPenaltyFactor() const;

   /// Assemble the face penalty terms in the matrix @a sparse_ij.
   void AssembleFaceTerms();
};

}

#include "lor_dg_impl.hpp"

#endif
