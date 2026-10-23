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

#include "estimators.hpp"
#include "complex_fem.hpp"

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

ComplexZienkiewiczZhuEstimator::ComplexZienkiewiczZhuEstimator(
   BilinearFormIntegrator &integ, ComplexGridFunction &solution_,
   FiniteElementSpace &flux_fes)
   : real_estimator(integ, solution_.real(), flux_fes),
     imag_estimator(integ, solution_.imag(), flux_fes),
     fespace(*solution_.FESpace())
{ }

ComplexZienkiewiczZhuEstimator::ComplexZienkiewiczZhuEstimator(
   BilinearFormIntegrator &integ, ComplexGridFunction &solution_,
   FiniteElementSpace *real_flux_fes, FiniteElementSpace *imag_flux_fes)
   : real_estimator(integ, solution_.real(), real_flux_fes),
     imag_estimator(integ, solution_.imag(), imag_flux_fes),
     fespace(*solution_.FESpace())
{ }

bool ComplexZienkiewiczZhuEstimator::MeshIsModified()
{
   const long sequence = fespace.GetMesh()->GetSequence();
   MFEM_ASSERT(sequence >= current_sequence, "improper mesh update sequence");
   return sequence > current_sequence;
}

void ComplexZienkiewiczZhuEstimator::ComputeEstimates()
{
   const Vector &real_errors = real_estimator.GetLocalErrors();
   const Vector &imag_errors = imag_estimator.GetLocalErrors();
   MFEM_VERIFY(real_errors.Size() == imag_errors.Size(),
               "incompatible real and imaginary ZZ estimates");
   error_estimates.SetSize(real_errors.Size());
   for (int i = 0; i < error_estimates.Size(); i++)
   {
      error_estimates(i) = hypot(real_errors(i), imag_errors(i));
   }
   current_sequence = fespace.GetMesh()->GetSequence();
}

real_t ComplexZienkiewiczZhuEstimator::GetTotalError() const
{
   real_t error_sq = error_estimates * error_estimates;
#ifdef MFEM_USE_MPI
   if (auto *pfes = dynamic_cast<const ParFiniteElementSpace*>(&fespace))
   {
      real_t global_error_sq = 0.0;
      MPI_Allreduce(&error_sq, &global_error_sq, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, pfes->GetComm());
      error_sq = global_error_sq;
   }
#endif
   return sqrt(error_sq);
}

const Vector &ComplexZienkiewiczZhuEstimator::GetLocalErrors()
{
   if (MeshIsModified()) { ComputeEstimates(); }
   return error_estimates;
}

void ComplexZienkiewiczZhuEstimator::Reset()
{
   current_sequence = -1;
   real_estimator.Reset();
   imag_estimator.Reset();
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
   const real_t solver_tol = 1e-12;
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
      real_t diameter = 0.0;

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

            diameter = std::max(diameter, p2.DistanceTo(p1));
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
   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      auto FT = mesh->GetFaceElementTransformations(f);

      auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * xfes->GetFaceOrder(f));
      const auto nip = int_rule.GetNPoints();

      if (mesh->FaceIsInterior(f))
      {
         int Inf1, Inf2, NCFace;
         mesh->GetFaceInfos(f, &Inf1, &Inf2, &NCFace);

         // Convention
         // * Conforming face: Face side with smaller element id handles
         // the integration
         // * Non-conforming face: The slave handles the integration.
         // See FaceInfo documentation for details.
         bool isNCSlave    = FT->Elem2No >= 0 && NCFace >= 0;
         bool isConforming = FT->Elem2No >= 0 && NCFace == -1;
         if ((FT->Elem1No < FT->Elem2No && isConforming) || isNCSlave)
         {
            if (attributes.Size() &&
                (attributes.FindSorted(FT->Elem1->Attribute) == -1
                 || attributes.FindSorted(FT->Elem2->Attribute) == -1))
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
               auto &fip = int_rule.IntPoint(i);
               IntegrationPoint ip;
               FT->Loc1.Transform(fip, ip);

               Vector val(flux_space->GetVDim());
               flux->GetVectorValue(FT->Elem1No, ip, val);

               // And build scalar product with normal
               Vector normal(mesh->SpaceDimension());
               FT->Face->SetIntPoint(&fip);
               if (mesh->Dimension() == mesh->SpaceDimension())
               {
                  CalcOrtho(FT->Face->Jacobian(), normal);
               }
               else
               {
                  Vector ref_normal(mesh->Dimension());
                  FT->Loc1.Transf.SetIntPoint(&fip);
                  CalcOrtho(FT->Loc1.Transf.Jacobian(), ref_normal);
                  auto &e1 = FT->GetElement1Transformation();
                  e1.AdjugateJacobian().MultTranspose(ref_normal, normal);
                  normal /= e1.Weight();
               }
               jumps(i) = val * normal * fip.weight * FT->Face->Weight();
            }

            // Subtract integral over half face of e₂
            // i.e. the numerical integration of ∫ flux ⋅ n dS₂
            for (int i = 0; i < nip; i++)
            {
               // Evaluate flux vector at IP
               auto &fip = int_rule.IntPoint(i);
               IntegrationPoint ip;
               FT->Loc2.Transform(fip, ip);

               Vector val(flux_space->GetVDim());
               flux->GetVectorValue(FT->Elem2No, ip, val);

               // And build scalar product with normal
               Vector normal(mesh->SpaceDimension());
               FT->Face->SetIntPoint(&fip);
               if (mesh->Dimension() == mesh->SpaceDimension())
               {
                  CalcOrtho(FT->Face->Jacobian(), normal);
               }
               else
               {
                  Vector ref_normal(mesh->Dimension());
                  FT->Loc1.Transf.SetIntPoint(&fip);
                  CalcOrtho(FT->Loc1.Transf.Jacobian(), ref_normal);
                  auto &e1 = FT->GetElement1Transformation();
                  e1.AdjugateJacobian().MultTranspose(ref_normal, normal);
                  normal /= e1.Weight();
               }

               jumps(i) -= val * normal * fip.weight * FT->Face->Weight();
            }

            // Finalize "local" L₂ contribution
            for (int i = 0; i < nip; i++)
            {
               jumps(i) *= jumps(i);
            }
            auto h_k_face = compute_face_coefficient(mesh, f, false);
            real_t jump_integral = h_k_face*jumps.Sum();

            // A local face is shared between two local elements, so we
            // can get away with integrating the jump only once and add
            // it to both elements. To minimize communication, the jump
            // of shared faces is computed locally by each process.
            error_estimates(FT->Elem1No) += jump_integral;
            error_estimates(FT->Elem2No) += jump_integral;
         }
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

   for (int sf = 0; sf < pmesh->GetNSharedFaces(); sf++)
   {
      auto FT = pmesh->GetSharedFaceTransformations(sf, true);
      if (attributes.Size() &&
          (attributes.FindSorted(FT->Elem1->Attribute) == -1
           || attributes.FindSorted(FT->Elem2->Attribute) == -1))
      {
         continue;
      }

      auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * xfes->GetFaceOrder(0));
      const auto nip = int_rule.GetNPoints();

      IntegrationRule eir;
      Vector jumps(nip);

      // Integral over local half face on the side of e₁
      // i.e. the numerical integration of ∫ flux ⋅ n dS₁
      for (int i = 0; i < nip; i++)
      {
         // Evaluate flux vector at integration point
         auto &fip = int_rule.IntPoint(i);
         IntegrationPoint ip;
         FT->Loc1.Transform(fip, ip);

         Vector val(flux_space->GetVDim());
         flux->GetVectorValue(FT->Elem1No, ip, val);

         Vector normal(mesh->SpaceDimension());
         FT->Face->SetIntPoint(&fip);
         if (mesh->Dimension() == mesh->SpaceDimension())
         {
            CalcOrtho(FT->Face->Jacobian(), normal);
         }
         else
         {
            Vector ref_normal(mesh->Dimension());
            FT->Loc1.Transf.SetIntPoint(&fip);
            CalcOrtho(FT->Loc1.Transf.Jacobian(), ref_normal);
            auto &e1 = FT->GetElement1Transformation();
            e1.AdjugateJacobian().MultTranspose(ref_normal, normal);
            normal /= e1.Weight();
         }

         jumps(i) = val * normal * fip.weight * FT->Face->Weight();
      }

      // Subtract integral over non-local half face of e₂
      // i.e. the numerical integration of ∫ flux ⋅ n dS₂
      for (int i = 0; i < nip; i++)
      {
         // Evaluate flux vector at integration point
         auto &fip = int_rule.IntPoint(i);
         IntegrationPoint ip;
         FT->Loc2.Transform(fip, ip);

         Vector val(flux_space->GetVDim());
         flux->GetVectorValue(FT->Elem2No, ip, val);

         // Evaluate Gauss point
         Vector normal(mesh->SpaceDimension());
         FT->Face->SetIntPoint(&fip);
         if (mesh->Dimension() == mesh->SpaceDimension())
         {
            CalcOrtho(FT->Face->Jacobian(), normal);
         }
         else
         {
            Vector ref_normal(mesh->Dimension());
            CalcOrtho(FT->Loc1.Transf.Jacobian(), ref_normal);
            auto &e1 = FT->GetElement1Transformation();
            e1.AdjugateJacobian().MultTranspose(ref_normal, normal);
            normal /= e1.Weight();
         }

         jumps(i) -= val * normal * fip.weight * FT->Face->Weight();
      }

      // Finalize "local" L₂ contribution
      for (int i = 0; i < nip; i++)
      {
         jumps(i) *= jumps(i);
      }
      auto h_k_face = compute_face_coefficient(mesh, sf, true);
      real_t jump_integral = h_k_face*jumps.Sum();

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

   real_t process_local_error = pow(error_estimates.Norml2(),2.0);
   MPI_Allreduce(&process_local_error, &total_error, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, pfes->GetComm());
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
      MPI_Allreduce(&process_local_error, &total_error, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, pfes->GetComm());
   }
#endif // MFEM_USE_MPI
   total_error = pow(total_error, 1.0/local_norm_p);
   current_sequence = sol->FESpace()->GetMesh()->GetSequence();
}


real_t GeneralErrorEstimator::GetTotalError() const
{
   real_t local_error_sq = elem_errors_ * elem_errors_;
#ifdef MFEM_USE_MPI
   if (auto *pmesh = dynamic_cast<ParMesh*>(mesh_))
   {
      real_t global_error_sq = 0.0;
      MPI_Allreduce(&local_error_sq, &global_error_sq, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, pmesh->GetComm());
      local_error_sq = global_error_sq;
   }
#endif
   return sqrt(local_error_sq);
}

/// Get a Vector with all element errors.
const Vector &GeneralErrorEstimator::GetLocalErrors()
{
   if (reset_ || current_sequence_ != mesh_->GetSequence())
   {
      ComputeEstimates();
   }
   return elem_errors_;
}

GeneralErrorEstimator::~GeneralErrorEstimator()
{
   for (auto *estimator : domain_estims_) { delete estimator; }
   for (auto *estimator : bdr_estims_) { delete estimator; }
   for (auto *estimator : face_estims_) { delete estimator; }
   for (auto *estimator : bdr_face_estims_) { delete estimator; }
}

void GeneralErrorEstimator::AddDomainEstimator(DomainErrorEstimator *dee)
{
   domain_estims_.Append(dee);
   domain_estims_marker_.Append(NULL); // NULL marker means apply everywhere
   Reset();
}

void GeneralErrorEstimator::AddDomainEstimator(DomainErrorEstimator *dee,
                                               Array<int> &elem_marker)
{
   domain_estims_.Append(dee);
   domain_estims_marker_.Append(&elem_marker);
   Reset();
}

void GeneralErrorEstimator::AddBdrEstimator(DomainErrorEstimator *dee)
{
   bdr_estims_.Append(dee);
   bdr_estims_marker_.Append(NULL);
   Reset();
}

void GeneralErrorEstimator::AddBdrEstimator(DomainErrorEstimator *dee,
                                            Array<int> &bdr_marker)
{
   bdr_estims_.Append(dee);
   bdr_estims_marker_.Append(&bdr_marker);
   Reset();
}

void GeneralErrorEstimator::AddInteriorFaceEstimator(FaceErrorEstimator *fee)
{
   face_estims_.Append(fee);
   Reset();
}

void GeneralErrorEstimator::AddBdrFaceEstimator(FaceErrorEstimator *fee)
{
   bdr_face_estims_.Append(fee);
   bdr_face_estims_marker_.Append(NULL); // NULL marker means apply everywhere
   Reset();
}

void GeneralErrorEstimator::AddBdrFaceEstimator(FaceErrorEstimator *fee,
                                                Array<int> &bdr_marker)

{
   bdr_face_estims_.Append(fee);
   bdr_face_estims_marker_.Append(&bdr_marker);
   Reset();
}

void GeneralErrorEstimator::ComputeEstimates()
{
   Mesh *mesh = mesh_;
   elem_errors_.SetSize(mesh_->GetNE());
   elem_errors_ = 0.0;

   // Preparation is separate from element/face traversal so data shared by
   // several estimators (for example a reconstructed residual field) is
   // updated only once per sweep.
   ErrorEstimatorContext context;
   for (auto *estimator : domain_estims_) { estimator->Prepare(context); }
   for (auto *estimator : bdr_estims_) { estimator->Prepare(context); }
   for (auto *estimator : face_estims_) { estimator->Prepare(context); }
   for (auto *estimator : bdr_face_estims_) { estimator->Prepare(context); }

   if (domain_estims_.Size())
   {
      for (int k = 0; k < domain_estims_.Size(); k++)
      {
         if (domain_estims_marker_[k] != NULL)
         {
            MFEM_VERIFY(domain_estims_marker_[k]->Size() ==
                        (mesh->attributes.Size() ? mesh->attributes.Max() : 0),
                        "invalid element marker for domain estimator #"
                        << k << ", counting from zero");
         }
      }
   }

   const int max_bdr_attr = mesh->bdr_attributes.Size() ?
                            mesh->bdr_attributes.Max() : 0;
   for (int k = 0; k < bdr_estims_marker_.Size(); k++)
   {
      if (bdr_estims_marker_[k])
      {
         MFEM_VERIFY(bdr_estims_marker_[k]->Size() == max_bdr_attr,
                     "invalid boundary marker for boundary estimator #" << k);
      }
   }
   for (int k = 0; k < bdr_face_estims_marker_.Size(); k++)
   {
      if (bdr_face_estims_marker_[k])
      {
         MFEM_VERIFY(bdr_face_estims_marker_[k]->Size() == max_bdr_attr,
                     "invalid boundary marker for boundary face estimator #" << k);
      }
   }

   for (int e = 0; e < mesh_->GetNE(); e++)
   {
      const int elem_attr = mesh->GetAttribute(e);
      ElementTransformation *eltrans = mesh_->GetElementTransformation(e);

      real_t elerr = 0.0;
      for (int k = 0; k < domain_estims_.Size(); k++)
      {
         if (domain_estims_marker_[k]) { domain_estims_marker_[k]->HostRead(); }
         if ((domain_estims_marker_[k] == NULL ||
              (*(domain_estims_marker_[k]))[elem_attr-1] == 1))
         {
            elerr += domain_estims_[k]->GetElementError(*eltrans);
         }
      }
      elem_errors_[e] += elerr;
   }

   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      FaceElementTransformations *tr = mesh->GetInteriorFaceTransformations(f);
      if (!tr) { continue; }
      for (auto *estimator : face_estims_)
      {
         // On a serial mesh, face estimators may evaluate material data at
         // element points in addition to face quadrature points. Refresh the
         // shared transformation before each term so this temporary state
         // cannot affect another term's traces. ParMesh uses a single cache
         // for both local and shared face transformations, so its state is
         // managed by the separate shared-face traversal below.
#ifdef MFEM_USE_MPI
         if (!dynamic_cast<ParMesh*>(mesh))
#endif
         {
            tr = mesh->GetInteriorFaceTransformations(f);
            if (!tr) { continue; }
         }
         real_t error1 = 0.0, error2 = 0.0;
         estimator->GetFaceError(*tr, error1, error2);
         elem_errors_(tr->Elem1No) += error1;
         elem_errors_(tr->Elem2No) += error2;
      }
   }

#ifdef MFEM_USE_MPI
   if (auto *pmesh = dynamic_cast<ParMesh*>(mesh_))
   {
      // A shared face is evaluated on both ranks. Each rank retains only its
      // local-side contribution, so every element indicator receives the jump
      // contribution exactly once without communicating element indicators.
      // Initialize face-neighbor geometry before obtaining any shared-face
      // transformations. Individual estimators then exchange their field data.
      pmesh->ExchangeFaceNbrData();
      for (auto *estimator : face_estims_)
      {
         estimator->ExchangeFaceNbrData();
      }

      for (int sf = 0; sf < pmesh->GetNSharedFaces(); sf++)
      {
         FaceElementTransformations *tr =
            pmesh->GetSharedFaceTransformations(sf, true);
         if (!tr) { continue; }

         for (auto *estimator : face_estims_)
         {
            real_t local_error = 0.0, neighbor_error = 0.0;
            estimator->GetFaceError(*tr, local_error, neighbor_error);
            elem_errors_(tr->Elem1No) += local_error;
         }
      }
   }
#endif

   for (int be = 0; be < mesh->GetNBE(); be++)
   {
      const int attr = mesh->GetBdrAttribute(be);
      ElementTransformation *tr = mesh_->GetBdrElementTransformation(be);
      for (int k = 0; k < bdr_estims_.Size(); k++)
      {
         const Array<int> *marker = bdr_estims_marker_[k];
         if (marker) { marker->HostRead(); }
         if (marker && (*marker)[attr - 1] == 0) { continue; }
         int el, info;
         mesh->GetBdrElementAdjacentElement(be, el, info);
         elem_errors_(el) += bdr_estims_[k]->GetElementError(*tr);
      }
   }

   for (int be = 0; be < mesh->GetNBE(); be++)
   {
      const int attr = mesh->GetBdrAttribute(be);
      FaceElementTransformations *tr = mesh->GetBdrFaceTransformations(be);
      if (!tr) { continue; }
      for (int k = 0; k < bdr_face_estims_.Size(); k++)
      {
         const Array<int> *marker = bdr_face_estims_marker_[k];
         if (marker) { marker->HostRead(); }
         if (marker && (*marker)[attr - 1] == 0) { continue; }
         elem_errors_(tr->Elem1No) += bdr_face_estims_[k]->GetFaceError(*tr);
      }
   }

   // Domain and face estimators return squared contributions. Finalizing here
   // gives GeneralErrorEstimator the same local-indicator convention as the
   // dedicated residual estimators.
   for (int e = 0; e < elem_errors_.Size(); e++)
   {
      MFEM_VERIFY(elem_errors_(e) >= 0.0,
                  "error-estimator contributions must be non-negative.");
      elem_errors_(e) = sqrt(elem_errors_(e));
   }

   current_sequence_ = mesh->GetSequence();
   reset_ = false;
}

} // namespace mfem
