// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
   total_error = ZZErrorEstimator(*integ, *solution, flux, error_estimates,
                                  anisotropic ? &aniso_flags : NULL,
                                  flux_averaging,
                                  with_coeff);

   current_sequence = solution->FESpace()->GetMesh()->GetSequence();
}


#ifdef MFEM_USE_MPI

void L2ZienkiewiczZhuEstimator::ComputeEstimates()
{
   flux_space->Update(false);
   smooth_flux_space->Update(false);

   // TODO: move these parameters in the class, and add Set* methods.
   const double solver_tol = 1e-12;
   const int solver_max_it = 200;
   total_error = L2ZZErrorEstimator(*integ, *solution, *smooth_flux_space,
                                    *flux_space, error_estimates,
                                    local_norm_p, solver_tol, solver_max_it);

   current_sequence = solution->FESpace()->GetMesh()->GetSequence();
}

void KellyErrorEstimator::ComputeEstimates()
{
   // Remarks:
   // For some context you may have to consult the documentation of
   // the FaceInfo class [1]. Also, the FaceElementTransformations
   // documentation [2] may be helpful to grasp what is going on. Note
   // that the FaceElementTransformations also works in the non-
   // conforming case to transfer the gauss points from the slave to
   // the master element.
   // [1]
   // https://github.com/mfem/mfem/blob/02d0bfe9c18ce049c3c93a6a4208080fcfc96991/mesh/mesh.hpp#L94
   // [2]
   // https://github.com/mfem/mfem/blob/02d0bfe9c18ce049c3c93a6a4208080fcfc96991/fem/eltrans.hpp#L435
   flux_space->Update(false);

   auto xfes      = solution->ParFESpace();
   auto pmesh     = xfes->GetParMesh();

   this->error_estimates.SetSize(xfes->GetNE());
   this->error_estimates = 0.0;

   // 1. Compute fluxes in discontinuous space
   ParGridFunction flux(flux_space);
   flux = 0.0;

   Array<int> xdofs, fdofs;
   Vector el_x, el_f;

   for (int e = 0; e < xfes->GetNE(); e++)
   {
      auto attr = xfes->GetAttribute(e);
      if(attributes.Size() && attributes.Find(attr) == -1)
      {
         continue;
      }

      xfes->GetElementVDofs(e, xdofs);
      solution->GetSubVector(xdofs, el_x);

      ElementTransformation* Transf = xfes->GetElementTransformation(e);
      flux_integrator->ComputeElementFlux(*xfes->GetFE(e), *Transf, el_x,
                                          *xfes->GetFE(e), el_f, false);

      flux_space->GetElementVDofs(e, fdofs);
      flux.AddElementVector(fdofs, el_f);
   }

   // 2. Add error contribution from local interior faces
   ///@TODO how to obtain the "correct" rule?
   auto int_rules = IntegrationRules();
   for (int f = 0; f < pmesh->GetNumFaces(); f++)
   {
         auto FT = pmesh->GetFaceElementTransformations(f);

         ///@TODO how to obtain the "correct" rule?
         const auto int_rule =
            int_rules.Get(FT->FaceGeom, 2 * FT->Face->Order() - 1);
         const auto nip = int_rule.GetNPoints();

         if (pmesh->FaceIsInterior(f))
         {
            int Inf1, Inf2, NCFace;
            pmesh->GetFaceInfos(f, &Inf1, &Inf2, &NCFace);

            // Convention
            // * Conforming face: Face side with smaller element id handles
            // the integration
            // * Non-conforming face: The slave handles the integration.
            // See FaceInfo documentation for details.
            bool isNCSlave    = FT->Elem2No >= 0 && NCFace >= 0;
            bool isConforming = FT->Elem2No >= 0 && NCFace == -1;
            if ((FT->Elem1No < FT->Elem2No && isConforming) || isNCSlave)
            { 
               if(attributes.Size() && 
                     (attributes.Find(FT->Elem1->Attribute) == -1 
                     || attributes.Find(FT->Elem2->Attribute) == -1))
               {
                     continue;
               }

               IntegrationRule eir;
               Vector jumps(nip);

               // Integral over local half face on the side of e₁
               // i.e. the numerical integration of ∫ flux ⋅ n dS₁
               for (int i = 0; i < nip; i++)
               {
                     // Evaluate flux at IP
                     auto fip = int_rule.IntPoint(i);
                     IntegrationPoint ip;
                     FT->Loc1.Transform(fip, ip);

                     Vector val(flux_space->GetVDim());
                     flux.GetVectorValue(FT->Elem1No, ip, val);

                     // And build scalar product with normal
                     Vector normal(pmesh->Dimension());
                     FT->Face->SetIntPoint(&fip);
                     CalcOrtho(FT->Face->Jacobian(), normal);

                     jumps(i) = val * normal * fip.weight;
               }

               // Substract integral over half face of e₂
               // i.e. the numerical integration of ∫ flux ⋅ n dS₂
               for (int i = 0; i < nip; i++)
               {
                     // Evaluate flux vector at IP
                     auto fip = int_rule.IntPoint(i);
                     IntegrationPoint ip;
                     FT->Loc2.Transform(fip, ip);

                     Vector val(flux_space->GetVDim());
                     flux.GetVectorValue(FT->Elem2No, ip, val);

                     // And build scalar product with normal
                     Vector normal(pmesh->Dimension());
                     FT->Face->SetIntPoint(&fip);
                     CalcOrtho(FT->Face->Jacobian(), normal);

                     jumps(i) -= val * normal * fip.weight;
               }

               // Finalize "local" L₂ contribution
               for (int i = 0; i < nip; i++)
               {
                     jumps(i) *= jumps(i);
               }
               double jump_integral = jumps.Sum();

               // A local face is shared between two local elements, so we
               // can get away with integrating the jump only once and add
               // it to both elements. To minimize communication, the jump
               // of shared faces is computed locally by each process.
               error_estimates(FT->Elem1No) += jump_integral;
               error_estimates(FT->Elem2No) += jump_integral;
            }
         }
   }

   // 3. Add error contribution from shared interior faces
   // Synchronize face data.
   flux.ExchangeFaceNbrData();

   for (int sf = 0; sf < pmesh->GetNSharedFaces(); sf++)
   {
         auto FT = pmesh->GetSharedFaceTransformations(sf, true);
         if(attributes.Size() && 
            (attributes.Find(FT->Elem1->Attribute) == -1 
            || attributes.Find(FT->Elem2->Attribute) == -1))
         {
            continue;
         }

         ///@TODO how to obtain the "correct" rule?
         const auto int_rule =
            int_rules.Get(FT->FaceGeom, 2 * FT->Face->Order() - 1);
         const auto nip = int_rule.GetNPoints();

         IntegrationRule eir;
         Vector jumps(nip);

         // Integral over local half face on the side of e₁
         // i.e. the numerical integration of ∫ flux ⋅ n dS₁
         for (int i = 0; i < nip; i++)
         {
            // Evaluate flux vector at integration point
            auto fip = int_rule.IntPoint(i);
            IntegrationPoint ip;
            FT->Loc1.Transform(fip, ip);

            Vector val(flux_space->GetVDim());
            flux.GetVectorValue(FT->Elem1No, ip, val);

            Vector normal(pmesh->Dimension());
            FT->Face->SetIntPoint(&fip);
            CalcOrtho(FT->Face->Jacobian(), normal);

            jumps(i) = val * normal * fip.weight;
         }

         // Substract integral over non-local half face of e₂
         // i.e. the numerical integration of ∫ flux ⋅ n dS₂
         for (int i = 0; i < nip; i++)
         {
            // Evaluate flux vector at integration point
            auto fip = int_rule.IntPoint(i);
            IntegrationPoint ip;
            FT->Loc2.Transform(fip, ip);

            Vector val(flux_space->GetVDim());
            flux.GetVectorValue(FT->Elem2No, ip, val);

            // Evaluate gauss point
            Vector normal(pmesh->Dimension());
            FT->Face->SetIntPoint(&fip);
            CalcOrtho(FT->Face->Jacobian(), normal);

            jumps(i) -= val * normal * fip.weight;
         }

         // Finalize "local" L₂ contribution
         for (int i = 0; i < nip; i++)
         {
            jumps(i) *= jumps(i);
         }
         double jump_integral = jumps.Sum();

         error_estimates(FT->Elem1No) += jump_integral;
         // We skip "error_estimates(FT->Elem2No) += jump_integral"
         // because the error is stored on the remote process and
         // recomputed there.
   }

   // Finalize element errors
   for (int e = 0; e < xfes->GetNE(); e++)
   {
         // Obtain jacobian of the transformation.
         DenseMatrix J;
         // pmesh->GetElementJacobian(e, J); //..protected....
         Geometry::Type geom      = pmesh->GetElementBaseGeometry(e);
         ElementTransformation* T = pmesh->GetElementTransformation(e);
         T->SetIntPoint(&Geometries.GetCenter(geom));
         Geometries.JacToPerfJac(geom, T->Jacobian(), J);

         // Intuitively we must scale the error with the "element size".
         // hₑ is also denoted by hₖ in some papers.
         auto hₑ = pow(abs(J.Weight()), 1.0 / double(pmesh->Dimension()));
         // The sqrt belongs to the norm and hₑ to the indicator.
         error_estimates(e) = sqrt(hₑ * error_estimates(e) / (2*T->Order()));
   }

   current_sequence = solution->FESpace()->GetMesh()->GetSequence();

   // Finish by computing the global error.
   double process_local_error = error_estimates.Sum();
   MPI_Allreduce(&process_local_error, &total_error, 1, MPI_DOUBLE,
                  MPI_SUM, xfes->GetComm());
}

#endif // MFEM_USE_MPI

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
   current_sequence = sol->FESpace()->GetMesh()->GetSequence();
}

} // namespace mfem
