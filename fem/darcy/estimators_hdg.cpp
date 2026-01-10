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
      const int bdr_face = mesh->GetBdrElementFaceIndex(b);

      ComputeFaceEstimate(bdr_face, false, d_error_estimates);
   }

   total_error = error_estimates.Sum();

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
      fe_tr = fes_tr->GetFaceElement(face);
      fes_tr->GetFaceVDofs(face, vdofs_tr);
      sol_tr.GetSubVector(vdofs_tr, tr);
   }

   const FiniteElement &fe1 = *fes_p->GetFE(FTr.Elem1No);
   const FiniteElement &fe2 = (FTr.Elem2No >= 0)?(*fes_p->GetFE(FTr.Elem2No)):
                              (fe1);

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

         error_estimates(FTr.Elem1No) += bfi.ComputeHDGFaceEnergy(0, *fe_tr, fe1, FTr,
                                                                  tr, p1, (anisotropic)?(&d_en1):(NULL));

         if (FTr.Elem2No >= 0)
         {
            error_estimates(FTr.Elem2No) += bfi.ComputeHDGFaceEnergy(1, *fe_tr, fe2, FTr,
                                                                     tr, p2, (anisotropic)?(&d_en2):(NULL));
         }

         if (anisotropic)
         {
            for (int k = 0; k < dim; k++)
            {
               d_error_estimates(FTr.Elem1No * dim + k) += d_en1(k);
            }
            if (FTr.Elem2No >= 0)
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

} // namespace mfem
