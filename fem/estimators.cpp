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

#include "estimators.hpp"

namespace mfem
{

void ZienkiewiczZhuEstimator::ComputeEstimates()
{
   flux_space->Update(false);
   // In parallel, 'flux' can be a GridFunction, as long as 'flux_space' is a
   // ParFiniteElementSpace and 'solution' is a ParGridFunction.
   GridFunction flux(flux_space);

   if (!anisotropic) { aniso_flags.SetSize(0); }
   total_error = ZZErrorEstimator(integ, solution, flux, error_estimates,
                                  anisotropic ? &aniso_flags : NULL,
                                  flux_averaging,
                                  with_coeff);

   current_sequence = solution.FESpace()->GetMesh()->GetSequence();
}

void LSZienkiewiczZhuEstimator::ComputeEstimates()
{
   total_error = LSZZErrorEstimator(integ,
                                    solution,
                                    error_estimates,
                                    subdomain_reconstruction,
                                    with_coeff,
                                    tichonov_coeff);

   current_sequence = solution.FESpace()->GetMesh()->GetSequence();
}

#ifdef MFEM_USE_MPI

void L2ZienkiewiczZhuEstimator::ComputeEstimates()
{
   flux_space->Update(false);
   smooth_flux_space->Update(false);

   // TODO: move these parameters in the class, and add Set* methods.
   const double solver_tol = 1e-12;
   const int solver_max_it = 200;
   total_error = L2ZZErrorEstimator(integ, solution, *smooth_flux_space,
                                    *flux_space, error_estimates,
                                    local_norm_p, solver_tol, solver_max_it);

   current_sequence = solution.FESpace()->GetMesh()->GetSequence();
}

#endif // MFEM_USE_MPI

KellyErrorEstimator::KellyErrorEstimator(BilinearFormIntegrator& di_,
                                         GridFunction& sol_,
                                         FiniteElementSpace& flux_fespace_,
                                         const Array<int> &attributes_)
   : attributes(attributes_)
   , flux_integrator(&di_)
   , solution(&sol_)
   , flux_space(&flux_fespace_)
   , own_flux_fespace(false)
#ifdef MFEM_USE_MPI
   , isParallel(dynamic_cast<ParFiniteElementSpace*>(sol_.FESpace()))
#endif // MFEM_USE_MPI
{
   ResetCoefficientFunctions();
}

KellyErrorEstimator::KellyErrorEstimator(BilinearFormIntegrator& di_,
                                         GridFunction& sol_,
                                         FiniteElementSpace* flux_fespace_,
                                         const Array<int> &attributes_)
   : attributes(attributes_)
   , flux_integrator(&di_)
   , solution(&sol_)
   , flux_space(flux_fespace_)
   , own_flux_fespace(true)
#ifdef MFEM_USE_MPI
   , isParallel(dynamic_cast<ParFiniteElementSpace*>(sol_.FESpace()))
#endif // MFEM_USE_MPI
{
   ResetCoefficientFunctions();
}

KellyErrorEstimator::~KellyErrorEstimator()
{
   if (own_flux_fespace)
   {
      delete flux_space;
   }
}

void KellyErrorEstimator::ResetCoefficientFunctions()
{
   compute_element_coefficient = [](Mesh* mesh, const int e)
   {
      return 1.0;
   };

   compute_face_coefficient = [](Mesh* mesh, const int f,
                                 const bool shared_face)
   {
      auto FT = [&]()
      {
#ifdef MFEM_USE_MPI
         if (shared_face)
         {
            return dynamic_cast<ParMesh*>(mesh)->GetSharedFaceTransformations(f);
         }
#endif // MFEM_USE_MPI
         return mesh->GetFaceElementTransformations(f);
      }();
      const auto order = FT->GetFE()->GetOrder();

      // Poor man's face diameter.
      double diameter = 0.0;

      Vector p1(mesh->SpaceDimension());
      Vector p2(mesh->SpaceDimension());
      // NOTE: We have no direct access to vertices for shared faces,
      // so we fall back to compute the positions from the element.
      // This can also be modified to compute the diameter for non-linear
      // geometries by sampling along geometry-specific lines.
      auto vtx_intrule = Geometries.GetVertices(FT->GetGeometryType());
      const auto nip = vtx_intrule->GetNPoints();
      for (int i = 0; i < nip; i++)
      {
         // Evaluate flux vector at integration point
         auto fip1 = vtx_intrule->IntPoint(i);
         FT->Transform(fip1, p1);

         for (int j = 0; j < nip; j++)
         {
            auto fip2 = vtx_intrule->IntPoint(j);
            FT->Transform(fip2, p2);

            diameter = std::max<double>(diameter, p2.DistanceTo(p1));
         }
      }
      return diameter/(2.0*order);
   };
}

void KellyErrorEstimator::ComputeEstimates()
{
   // Remarks:
   // For some context you may have to consult the documentation of
   // the FaceInfo class [1]. Also, the FaceElementTransformations
   // documentation [2] may be helpful to grasp what is going on. Note
   // that the FaceElementTransformations also works in the non-
   // conforming case to transfer the Gauss points from the slave to
   // the master element.
   // [1]
   // https://github.com/mfem/mfem/blob/02d0bfe9c18ce049c3c93a6a4208080fcfc96991/mesh/mesh.hpp#L94
   // [2]
   // https://github.com/mfem/mfem/blob/02d0bfe9c18ce049c3c93a6a4208080fcfc96991/fem/eltrans.hpp#L435

   flux_space->Update(false);

   auto xfes = solution->FESpace();
   MFEM_ASSERT(xfes->GetVDim() == 1,
               "Estimation for vector-valued problems not implemented yet.");
   auto mesh = xfes->GetMesh();

   this->error_estimates.SetSize(xfes->GetNE());
   this->error_estimates = 0.0;

   // 1. Compute fluxes in discontinuous space
   GridFunction *flux =
#ifdef MFEM_USE_MPI
      isParallel ? new ParGridFunction(dynamic_cast<ParFiniteElementSpace*>
                                       (flux_space)) :
#endif // MFEM_USE_MPI
      new GridFunction(flux_space);

   *flux = 0.0;

   // We pre-sort the array to speed up the search in the following loops.
   if (attributes.Size())
   {
      attributes.Sort();
   }

   Array<int> xdofs, fdofs;
   Vector el_x, el_f;
   for (int e = 0; e < xfes->GetNE(); e++)
   {
      auto attr = xfes->GetAttribute(e);
      if (attributes.Size() && attributes.FindSorted(attr) == -1)
      {
         continue;
      }

      xfes->GetElementVDofs(e, xdofs);
      solution->GetSubVector(xdofs, el_x);

      ElementTransformation* Transf = xfes->GetElementTransformation(e);
      flux_integrator->ComputeElementFlux(*xfes->GetFE(e), *Transf, el_x,
                                          *flux_space->GetFE(e), el_f, true);

      flux_space->GetElementVDofs(e, fdofs);
      flux->AddElementVector(fdofs, el_f);
   }

   // 2. Add error contribution from local interior faces
   IntegrationPoint ip;
   Vector val(flux_space->GetVDim());
   Vector normal(mesh->SpaceDimension());
   Vector ref_normal(mesh->Dimension());
   for (int fi = 0; fi < mesh->GetNumFaces(); fi++)
   {
      if (mesh->FaceIsInterior(fi))
      {
         // Compute NC and face information
         int FaceElement1, FaceElement2, NCFace;
         mesh->GetFaceInfos(fi, &FaceElement1, &FaceElement2, &NCFace);
         mesh->GetFaceElements(fi, &FaceElement1, &FaceElement2);

         // We skip over master faces
         bool isNCSlave    = FaceElement2 >= 0 && NCFace >= 0;
         bool isConforming = FaceElement2 >= 0 && NCFace == -1;
         if (isConforming || isNCSlave)
         {
            if (attributes.Size() &&
                (attributes.FindSorted(mesh->GetAttribute(FaceElement1)) == -1
                 || attributes.FindSorted(mesh->GetAttribute(FaceElement2)) == -1))
            {
               continue;
            }

            // Computation of the transformations is not super cheap, so we delay it until here. All ops above are cheaper lookups.
            auto FT = mesh->GetFaceElementTransformations(fi);

            // TODO Refactor this computation into a user-facing function.
            auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * xfes->GetFaceOrder(fi));
            const auto nip = int_rule.GetNPoints();

            double jump_integral = 0.0;

            // Integral over local half face on the side of e₁
            // i.e. the numerical integration of ∫ flux ⋅ n dS₁
            for (int i = 0; i < nip; i++)
            {
               // Setup integration point
               auto &fip = int_rule.IntPoint(i);
               FT->Face->SetIntPoint(&fip);

               // Compute weighted normal - note that the normals match at each face integration point up to the sign!
               if (mesh->Dimension() == mesh->SpaceDimension())
               {
                  CalcOrtho(FT->Face->Jacobian(), normal);
               }
               else
               {
                  FT->Loc1.Transf.SetIntPoint(&fip);
                  CalcOrtho(FT->Loc1.Transf.Jacobian(), ref_normal);
                  auto &e1 = FT->GetElement1Transformation();
                  e1.AdjugateJacobian().MultTranspose(ref_normal, normal);
                  // We have to cancel the additional weighting from the
                  // reference to spatial transformation in the line above.
                  normal /= e1.Weight();
               }

               // Evaluate flux jump at IP on element 1
               FT->Loc1.Transform(fip, ip);
               flux->GetVectorValue(FT->Elem1No, ip, val);
               double jump = val * normal;

               // Evaluate flux jump at IP on element 2
               FT->Loc2.Transform(fip, ip);
               flux->GetVectorValue(FT->Elem2No, ip, val);
               jump -= val * normal;

               // Finalize integral
               // Since the jump is squared and the normal is weighted we have to cancel one normal weight here.
               jump_integral += jump*jump*fip.weight / FT->Face->Weight();
            }

            // A local face is shared between two local elements, so we
            // can get away with integrating the jump only once and add
            // it to both elements. To minimize communication, the jump
            // of shared faces is computed locally by each process.
            error_estimates(FT->Elem1No) += jump_integral;
            error_estimates(FT->Elem2No) += jump_integral;
         }
      } else { // We area t a boundary face
         // TODO implement boundary flux difference
      }
   }

   current_sequence = solution->FESpace()->GetMesh()->GetSequence();

#ifdef MFEM_USE_MPI
   if (!isParallel)
#endif // MFEM_USE_MPI
   {
      // Finalize element errors
      for (int e = 0; e < xfes->GetNE(); e++)
      {
         auto factor = compute_element_coefficient(mesh, e);
         // The sqrt belongs to the norm and hₑ to the indicator.
         error_estimates(e) = sqrt(factor * error_estimates(e));
      }

      total_error = error_estimates.Norml2();
      delete flux;
      return;
   }

#ifdef MFEM_USE_MPI

   // 3. Add error contribution from shared interior faces
   // Synchronize face data.

   ParGridFunction *pflux = dynamic_cast<ParGridFunction*>(flux);
   MFEM_VERIFY(pflux, "flux is not a ParGridFunction pointer");

   ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
   MFEM_VERIFY(pmesh, "mesh is not a ParMesh pointer");

   pflux->ExchangeFaceNbrData();

   for (int sfi = 0; sfi < pmesh->GetNSharedFaces(); sfi++)
   {
      auto FT = pmesh->GetSharedFaceTransformations(sfi, true);

      if (attributes.Size() &&
          (attributes.FindSorted(FT->Elem1->Attribute) == -1
           || attributes.FindSorted(FT->Elem2->Attribute) == -1))
      {
         continue;
      }

      // TODO Refactor this computation into a user-facing function.
      auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * xfes->GetFaceOrder(0));
      const auto nip = int_rule.GetNPoints();

      double jump_integral = 0.0;

      // Integral over local half face on the side of e₁
      // i.e. the numerical integration of ∫ flux ⋅ n dS₁
      for (int i = 0; i < nip; i++)
      {
         // Setup integration point
         auto &fip = int_rule.IntPoint(i);
         FT->Face->SetIntPoint(&fip);

         // Evaluate flux at IP on element 1
         FT->Loc1.Transform(fip, ip);
         flux->GetVectorValue(FT->Elem1No, ip, val);

         // Compute weighted normal - note that the normals match at each face integration point up to the sign!
         if (mesh->Dimension() == mesh->SpaceDimension())
         {
            CalcOrtho(FT->Face->Jacobian(), normal);
         }
         else
         {
            FT->Loc1.Transf.SetIntPoint(&fip);
            CalcOrtho(FT->Loc1.Transf.Jacobian(), ref_normal);
            auto &e1 = FT->GetElement1Transformation();
            e1.AdjugateJacobian().MultTranspose(ref_normal, normal);
            // We have to cancel the additional weighting from the
            // reference to spatial transformation in the line above.
            normal /= e1.Weight();
         }

         // Evaluate flux jump at IP on element 1
         FT->Loc1.Transform(fip, ip);
         flux->GetVectorValue(FT->Elem1No, ip, val);
         double jump = val * normal;

         // Evaluate flux jump at IP on element 2
         FT->Loc2.Transform(fip, ip);
         flux->GetVectorValue(FT->Elem2No, ip, val);
         jump -= val * normal;

         // Finalize integral
         // Since the jump is squared and the normal is weighted we have to cancel one normal weight here.
         jump_integral += jump*jump*fip.weight / FT->Face->Weight();
      }

      error_estimates(FT->Elem1No) += jump_integral;
      // We skip "error_estimates(FT->Elem2No) += jump_integral"
      // because the error is stored on the remote process and
      // recomputed there.
   }
   delete flux;

   // Finalize element errors
   for (int e = 0; e < xfes->GetNE(); e++)
   {
      auto factor = compute_element_coefficient(mesh, e);
      // The sqrt belongs to the norm and hₑ to the indicator.
      error_estimates(e) = sqrt(factor * error_estimates(e));
   }

   // Finish by computing the global error.
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(xfes);
   MFEM_VERIFY(pfes, "xfes is not a ParFiniteElementSpace pointer");

   double process_local_error = pow(error_estimates.Norml2(),2.0);
   MPI_Allreduce(&process_local_error, &total_error, 1, MPI_DOUBLE,
                 MPI_SUM, pfes->GetComm());
   total_error = sqrt(total_error);
#endif // MFEM_USE_MPI
}

void LpErrorEstimator::ComputeEstimates()
{
   MFEM_VERIFY(coef != NULL || vcoef != NULL,
               "LpErrorEstimator has no coefficient!  Call SetCoef first.");

   error_estimates.SetSize(sol->FESpace()->GetMesh()->GetNE());
   if (coef)
   {
      sol->ComputeElementLpErrors(local_norm_p, *coef, error_estimates);
   }
   else
   {
      sol->ComputeElementLpErrors(local_norm_p, *vcoef, error_estimates);
   }
#ifdef MFEM_USE_MPI
   total_error = error_estimates.Sum();
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(sol->FESpace());
   if (pfes)
   {
      auto process_local_error = total_error;
      MPI_Allreduce(&process_local_error, &total_error, 1, MPI_DOUBLE,
                    MPI_SUM, pfes->GetComm());
   }
#endif // MFEM_USE_MPI
   total_error = pow(total_error, 1.0/local_norm_p);
   current_sequence = sol->FESpace()->GetMesh()->GetSequence();
}

} // namespace mfem
