//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!! TODO Planned to be merged into #3693 !!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

template<typename Kernel>
class CustomKellyErrorEstimator final : public ErrorEstimator
{
public:
   /// Function type to compute the local coefficient hₑ of an element.
   using ElementCoefficientFunction =
      std::function<double(Mesh*, const int)>;
   /** @brief Function type to compute the local coefficient hₖ of a face. The
       third argument is true for shared faces and false for local faces. */
   using FaceCoefficientFunction =
      std::function<double(Mesh*, const int, const bool)>;

private:
   Kernel kernel;

   int current_sequence = -1;

   Vector error_estimates;

   double total_error = 0.0;

   Array<int> attributes;

   /** @brief A method to compute hₑ on per-element basis.

       This method weights the error approximation on the element level.

       Defaults to hₑ=1.0.
   */
   ElementCoefficientFunction compute_element_coefficient;

   /** @brief A method to compute hₖ on per-face basis.

       This method weights the error approximation on the face level. The
       background here is that classical Kelly error estimator implementations
       approximate the geometrical characteristic hₖ with the face diameter,
       which should be also be a possibility in this implementation.

       Defaults to hₖ=diameter/2p.
   */
   FaceCoefficientFunction compute_face_coefficient;

   BilinearFormIntegrator* flux_integrator; ///< Not owned.
   GridFunction* solution;               ///< Not owned.

   FiniteElementSpace*
   flux_space; /**< @brief Ownership based on own_flux_fes. */
   bool own_flux_fespace; ///< Ownership flag for flux_space.

   bool with_flux; ///< Use flux or gradient.
   bool isParallel;

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = solution->FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence,
                  "improper mesh update sequence");
      return (mesh_sequence > current_sequence);
   }

   /** @brief Compute the element error estimates.

       Algorithm outline:
       1. Compute flux field for each element
       2. Add error contribution from local interior faces
       3. Add error contribution from shared interior faces
       4. Finalize by computing hₖ and scale errors.
   */
   void ComputeEstimates();

public:
   /** @brief Construct a new CustomKellyErrorEstimator object for a scalar field.
       @param di_         The bilinearform to compute the interface flux.
       @param sol_        The solution field whose error is to be estimated.
       @param flux_fes_   The finite element space for the interface flux.
       @param attributes_ The attributes of the subdomain(s) for which the
                          error should be estimated. An empty array results in
                          estimating the error over the complete domain.
   */
   CustomKellyErrorEstimator(Kernel kernel_, BilinearFormIntegrator& di_, GridFunction& sol_,
                       FiniteElementSpace& flux_fes_,
                       bool with_flux,
                       const Array<int> &attributes_ = Array<int>());

   /** @brief Construct a new CustomKellyErrorEstimator object for a scalar field.
       @param di_         The bilinearform to compute the interface flux.
       @param sol_        The solution field whose error is to be estimated.
       @param flux_fes_   The finite element space for the interface flux.
       @param attributes_ The attributes of the subdomain(s) for which the
                          error should be estimated. An empty array results in
                          estimating the error over the complete domain.
   */
   CustomKellyErrorEstimator(Kernel kernel_, BilinearFormIntegrator& di_, GridFunction& sol_,
                       FiniteElementSpace* flux_fes_,
                       bool with_flux,
                       const Array<int> &attributes_ = Array<int>());

   ~CustomKellyErrorEstimator();

   /// Get a Vector with all element errors.
   const Vector& GetLocalErrors() override
   {
      if (MeshIsModified())
      {
         ComputeEstimates();
      }
      return error_estimates;
   }

   /// Reset the error estimator.
   void Reset() override { current_sequence = -1; };

   virtual double GetTotalError() const override { return total_error; }

   /** @brief Change the method to compute hₑ on a per-element basis.
       @param compute_element_coefficient_
                        A function taking a mesh and an element index to
                        compute the local hₑ for the element.
   */
   void SetElementCoefficientFunction(ElementCoefficientFunction
                                      compute_element_coefficient_)
   {
      compute_element_coefficient = compute_element_coefficient_;
   }

   /** @brief Change the method to compute hₖ on a per-element basis.
       @param compute_face_coefficient_
                        A function taking a mesh and a face index to
                        compute the local hₖ for the face.
   */
   void SetFaceCoefficientFunction(
      FaceCoefficientFunction
      compute_face_coefficient_)
   {
      compute_face_coefficient = compute_face_coefficient_;
   }

   /// Change the coefficients back to default as described above.
   void ResetCoefficientFunctions();
};

#include "new-kelly-estimator.cxx" // Template impl

struct DGFluxKernel {
   // Eval buffers
   IntegrationPoint ip;
   FiniteElementSpace* flux_space;
   Vector val;
   Vector normal;
   Vector ref_normal;

   DGFluxKernel(FiniteElementSpace* flux_space_)
   : flux_space(flux_space_)
   , val(Vector(flux_space_->GetVDim()))
   , normal(Vector(flux_space_->GetMesh()->SpaceDimension()))
   , ref_normal(Vector(flux_space_->GetMesh()->Dimension()))
   {
      
   }

   double operator()(FaceElementTransformations *FT, GridFunction *solution, GridFunction *flux) {
      auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * std::min(flux_space->GetElementOrder(FT->Elem1No), flux_space->GetElementOrder(FT->Elem2No)));
      const auto nip = int_rule.GetNPoints();

      double jump_integral = 0.0;

      // Numerical integration of ∫ [[flux ⋅ n]] dS
      for (int i = 0; i < nip; i++)
      {
         // Set up integration point
         auto &fip = int_rule.IntPoint(i);
         FT->Face->SetIntPoint(&fip);

         // Compute normal - note that the normals match at each face integration point up to the sign!
         if (flux_space->GetMesh()->Dimension() == flux_space->GetMesh()->SpaceDimension())
         {
            // This computes a weighted normal, so we divide by the weight to get back the normal.
            CalcOrtho(FT->Face->Jacobian(), normal);
            normal /= FT->Face->Weight();
         }
         else
         {
            // This computes a weighted normal, so we divide by the weight to get back the normal.
            FT->Loc1.Transf.SetIntPoint(&fip);
            FT->Loc1.Transform(fip, ip);
            CalcOrtho(FT->Loc1.Transf.Jacobian(), ref_normal);
            ref_normal /= FT->Face->Weight();
            auto &e1 = FT->GetElement1Transformation();
            e1.SetIntPoint(&ip);
            e1.AdjugateJacobian().MultTranspose(ref_normal, normal);
            // We have to cancel the additional weighting from the
            // reference to spatial transformation in the line above.
            normal /= e1.Weight();
         }

         // Evaluate flux jump at IP on element 1
         FT->Loc1.Transf.SetIntPoint(&fip);
         FT->Loc1.Transform(fip, ip);
         flux->GetVectorValue(FT->Elem1No, ip, val);
         double jump = val * normal;

         // Evaluate flux jump at IP on element 2
         FT->Loc2.Transf.SetIntPoint(&fip);
         FT->Loc2.Transform(fip, ip);
         flux->GetVectorValue(FT->Elem2No, ip, val);
         jump -= val * normal;

         // Finalize integral
         jump_integral += jump*jump*fip.weight * FT->Face->Weight();
      }

      return jump_integral;
   }
};


struct DGWeightedFluxKernel {
   // Eval buffers
   IntegrationPoint ip;
   FiniteElementSpace* flux_space;
   Vector val;
   Vector normal;
   Vector ref_normal;

   DGWeightedFluxKernel(FiniteElementSpace* flux_space_)
   : flux_space(flux_space_)
   , val(Vector(flux_space_->GetVDim()))
   , normal(Vector(flux_space_->GetMesh()->SpaceDimension()))
   , ref_normal(Vector(flux_space_->GetMesh()->Dimension()))
   {
   }

   double operator()(FaceElementTransformations *FT, GridFunction *solution, GridFunction *flux) {
      auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * std::min(flux_space->GetElementOrder(FT->Elem1No), flux_space->GetElementOrder(FT->Elem2No)));
      const auto nip = int_rule.GetNPoints();

      double jump_integral = 0.0;

      // Numerical integration of ∫ [[flux ⋅ n]] dS
      for (int i = 0; i < nip; i++)
      {
         // Set up integration point
         auto &fip = int_rule.IntPoint(i);
         FT->Face->SetIntPoint(&fip);

         // Compute normal - note that the normals match at each face integration point up to the sign!
         if (flux_space->GetMesh()->Dimension() == flux_space->GetMesh()->SpaceDimension())
         {
            // This computes a weighted normal, so we divide by the weight to get back the normal.
            CalcOrtho(FT->Face->Jacobian(), normal);
         }
         else
         {
            // This computes a weighted normal, so we divide by the weight to get back the normal.
            FT->Loc1.Transf.SetIntPoint(&fip);
            FT->Loc1.Transform(fip, ip);
            CalcOrtho(FT->Loc1.Transf.Jacobian(), ref_normal);
            auto &e1 = FT->GetElement1Transformation();
            e1.SetIntPoint(&ip);
            e1.AdjugateJacobian().MultTranspose(ref_normal, normal);
            // We have to cancel the additional weighting from the
            // reference to spatial transformation in the line above.
            normal /= e1.Weight();
         }

         // Evaluate flux jump at IP on element 1
         FT->Loc1.Transf.SetIntPoint(&fip);
         FT->Loc1.Transform(fip, ip);
         flux->GetVectorValue(FT->Elem1No, ip, val);
         double jump = val * normal;

         // Evaluate flux jump at IP on element 2
         FT->Loc2.Transf.SetIntPoint(&fip);
         FT->Loc2.Transform(fip, ip);
         flux->GetVectorValue(FT->Elem2No, ip, val);
         jump -= val * normal;

         // Finalize integral
         jump_integral += jump*jump*fip.weight * FT->Face->Weight();
      }

      return jump_integral;
   }
};
