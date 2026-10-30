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
   const real_t solver_tol = 1e-12;
   const int solver_max_it = 200;
   total_error = L2ZZErrorEstimator(integ, solution, *smooth_flux_space,
                                    *flux_space, error_estimates,
                                    local_norm_p, solver_tol, solver_max_it);

   current_sequence = solution.FESpace()->GetMesh()->GetSequence();
}

#endif // MFEM_USE_MPI

namespace
{

/// Maximal polynomial degree of the two elements sharing the face.
int MaxOrderOnFace(const FiniteElementSpace &fes,
                   const FaceElementTransformations &FT)
{
   auto order = [&fes](const int e)
   {
#ifdef MFEM_USE_MPI
      if (e >= fes.GetNE())
      {
         auto &pfes = static_cast<const ParFiniteElementSpace&>(fes);
         return pfes.GetFaceNbrFE(e - fes.GetNE())->GetOrder();
      }
#endif // MFEM_USE_MPI
      return fes.GetFE(e)->GetOrder();
   };
   return std::max(order(FT.Elem1No), order(FT.Elem2No));
}

/// Maximal distance between the vertices of @a T.
real_t VertexDiameter(ElementTransformation &T)
{
   const IntegrationRule *vertices =
      Geometries.GetVertices(T.GetGeometryType());
   Vector p1(T.GetSpaceDim());
   Vector p2(T.GetSpaceDim());
   real_t diameter = 0.0;
   for (int i = 0; i < vertices->GetNPoints(); i++)
   {
      T.Transform(vertices->IntPoint(i), p1);
      for (int j = i+1; j < vertices->GetNPoints(); j++)
      {
         T.Transform(vertices->IntPoint(j), p2);
         diameter = std::max(diameter, p2.DistanceTo(p1));
      }
   }
   return diameter;
}

} // anonymous namespace

KellyErrorEstimator::KellyErrorEstimator(BilinearFormIntegrator& di_,
                                         GridFunction& sol_,
                                         FiniteElementSpace& flux_fespace_,
                                         const Array<int> &attributes_,
                                         bool with_coeff_)
   : attributes(attributes_)
   , flux_integrator(&di_)
   , solution(&sol_)
   , flux_space(&flux_fespace_)
   , own_flux_fespace(false)
   , with_coeff(with_coeff_)
#ifdef MFEM_USE_MPI
   , isParallel(dynamic_cast<ParFiniteElementSpace*>(sol_.FESpace()))
#endif // MFEM_USE_MPI
{
   ResetCoefficientFunctions();
}

KellyErrorEstimator::KellyErrorEstimator(BilinearFormIntegrator& di_,
                                         GridFunction& sol_,
                                         FiniteElementSpace* flux_fespace_,
                                         const Array<int> &attributes_,
                                         bool with_coeff_)
   : attributes(attributes_)
   , flux_integrator(&di_)
   , solution(&sol_)
   , flux_space(flux_fespace_)
   , own_flux_fespace(true)
   , with_coeff(with_coeff_)
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

   const GridFunction *sol = solution;
   compute_face_coefficient = [sol](Mesh* mesh, const int f,
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

      const real_t diameter = mesh->Dimension() == 1
                              ? std::max(VertexDiameter(*FT->Elem1),
                                         VertexDiameter(*FT->Elem2))
                              : VertexDiameter(*FT);
      return diameter/(2.0*MaxOrderOnFace(*sol->FESpace(), *FT));
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
                                          *flux_space->GetFE(e), el_f,
                                          with_coeff);

      flux_space->GetElementVDofs(e, fdofs);
      flux->AddElementVector(fdofs, el_f);
   }

   // 2. Add error contribution from local interior faces
   const int dim = mesh->Dimension();
   const int sdim = mesh->SpaceDimension();
   Vector val(flux_space->GetVDim());
   Vector normal(sdim);
   Vector ref_normal(dim);
   // Computes ∫ ((flux₁ - flux₂) ⋅ n)² dS with the unit normal n.
   auto face_jump_integral = [&](FaceElementTransformations &FT)
   {
      const IntegrationRule &ir =
         IntRules.Get(FT.GetGeometryType(), 2*MaxOrderOnFace(*xfes, FT));
      real_t jump_integral = 0.0;
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &fip = ir.IntPoint(i);
         FT.SetAllIntPoints(&fip);

         if (dim == 1 && sdim == 1)
         {
            normal(0) = 1.0;
         }
         else if (dim == sdim)
         {
            CalcOrtho(FT.Jacobian(), normal);
            normal /= FT.Weight();
         }
         else
         {
            // Map the reference normal of element 1 to physical space. For
            // the non-square Jacobian J the adjugate is det(JᵀJ) J⁺.
            if (dim == 1)
            {
               ref_normal(0) = 2.0*FT.GetElement1IntPoint().x - 1.0;
            }
            else
            {
               FT.Loc1.Transf.SetIntPoint(&fip);
               CalcOrtho(FT.Loc1.Transf.Jacobian(), ref_normal);
               ref_normal /= FT.Weight();
            }
            ElementTransformation &T1 = FT.GetElement1Transformation();
            T1.AdjugateJacobian().MultTranspose(ref_normal, normal);
            normal /= T1.Weight();
         }

         flux->GetVectorValue(FT.Elem1No, FT.GetElement1IntPoint(), val);
         real_t jump = val * normal;
         flux->GetVectorValue(FT.Elem2No, FT.GetElement2IntPoint(), val);
         jump -= val * normal;

         jump_integral += jump*jump*fip.weight*FT.Weight();
      }
      return jump_integral;
   };

   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      // Boundary faces are ignored. Nonconforming master faces are not
      // interior either, as they are integrated via their slave faces.
      if (!mesh->FaceIsInterior(f)) { continue; }

      int e1, e2;
      mesh->GetFaceElements(f, &e1, &e2);
      if (attributes.Size() &&
          (attributes.FindSorted(mesh->GetAttribute(e1)) == -1
           || attributes.FindSorted(mesh->GetAttribute(e2)) == -1))
      {
         continue;
      }

      const real_t jump_integral =
         face_jump_integral(*mesh->GetFaceElementTransformations(f));

      // A local face is shared between two local elements, so we
      // can get away with integrating the jump only once and add
      // it to both elements. To minimize communication, the jump
      // of shared faces is computed locally by each process.
      const real_t h_k_face = compute_face_coefficient(mesh, f, false);
      error_estimates(e1) += h_k_face*jump_integral;
      error_estimates(e2) += h_k_face*jump_integral;
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

      const int e1 = FT->Elem1No;
      const real_t jump_integral = face_jump_integral(*FT);

      const real_t h_k_face = compute_face_coefficient(mesh, sf, true);
      error_estimates(e1) += h_k_face*jump_integral;
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

} // namespace mfem
