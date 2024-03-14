template<typename Kernel>
CustomKellyErrorEstimator<Kernel>::CustomKellyErrorEstimator(Kernel kernel_,
                                         BilinearFormIntegrator& di_,
                                         GridFunction& sol_,
                                         FiniteElementSpace& flux_fespace_,
                                         bool with_flux_,
                                         const Array<int> &attributes_)
   : kernel(kernel_)
   , attributes(attributes_)
   , flux_integrator(&di_)
   , solution(&sol_)
   , flux_space(&flux_fespace_)
   , own_flux_fespace(false)
   , with_flux(with_flux_)
#ifdef MFEM_USE_MPI
   , isParallel(dynamic_cast<ParFiniteElementSpace*>(sol_.FESpace()))
#endif // MFEM_USE_MPI
{
   ResetCoefficientFunctions();
}

template<typename Kernel>
CustomKellyErrorEstimator<Kernel>::CustomKellyErrorEstimator(Kernel kernel_,
                                         BilinearFormIntegrator& di_,
                                         GridFunction& sol_,
                                         FiniteElementSpace* flux_fespace_,
                                         bool with_flux_,
                                         const Array<int> &attributes_)
   : kernel(kernel_)
   , attributes(attributes_)
   , flux_integrator(&di_)
   , solution(&sol_)
   , flux_space(flux_fespace_)
   , own_flux_fespace(true)
   , with_flux(with_flux_)
#ifdef MFEM_USE_MPI
   , isParallel(dynamic_cast<ParFiniteElementSpace*>(sol_.FESpace()))
#endif // MFEM_USE_MPI
{
   ResetCoefficientFunctions();
}

template<typename T>
CustomKellyErrorEstimator<T>::~CustomKellyErrorEstimator()
{
   if (own_flux_fespace)
   {
      delete flux_space;
   }
}

template<typename T>
void CustomKellyErrorEstimator<T>::ResetCoefficientFunctions()
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

         for (int j = i+1; j < nip; j++)
         {
            auto fip2 = vtx_intrule->IntPoint(j);
            FT->Transform(fip2, p2);

            diameter = std::max<double>(diameter, p2.DistanceTo(p1));
         }
      }
      return diameter/(2.0*order);
   };
}

template<typename Kernel>
void CustomKellyErrorEstimator<Kernel>::ComputeEstimates()
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
                                          *flux_space->GetFE(e), el_f, with_flux);

      flux_space->GetElementVDofs(e, fdofs);
      flux->AddElementVector(fdofs, el_f);
   }

   // 2. Add error contribution from local interior faces
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

            auto FT = mesh->GetFaceElementTransformations(fi);

            const double kernel_value = kernel(FT, solution, flux);

            // A local face is shared between two local elements, so we
            // can get away with integrating the jump only once and add
            // it to both elements. To minimize communication, the jump
            // of shared faces is computed locally by each process.
            auto h_k_face = compute_face_coefficient(mesh, fi, false);
            error_estimates(FT->Elem1No) += h_k_face*kernel_value;
            error_estimates(FT->Elem2No) += h_k_face*kernel_value;
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
      // auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * xfes->GetFaceOrder(0)); // NOTE: This fails for DG
      auto &int_rule = IntRules.Get(FT->FaceGeom,
                                    2 * xfes->GetElementOrder(FT->Elem1No));
      const auto nip = int_rule.GetNPoints();

      const double kernel_value = kernel(FT, solution, flux);

      auto h_k_face = compute_face_coefficient(mesh, sfi, true);
      error_estimates(FT->Elem1No) += h_k_face*kernel_value;
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
