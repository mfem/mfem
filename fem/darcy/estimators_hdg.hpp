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

#ifndef MFEM_ESTIMATORS_HDG
#define MFEM_ESTIMATORS_HDG

#include "../estimators.hpp"

namespace mfem
{

/// (Anisotropic) error estimator for hybridized Darcy-like mixed systems
/** HDGErrorEstimator is an error estimator for mixed systems with
    (anti)symmetric weak form hybridized as follows:
    \verbatim
        ┌           ┐┌   ┐   ┌    ┐
        | Mu ±Bᵀ Cᵀ || u |   | bu |
        | B   D  E  || p | = | bp |
        | C   G  H  || λ |   | br |
        └           ┘└   ┘   └    ┘
    \endverbatim
    where the notation follows DarcyHybridization.

    The idea behind HDGErrorEstimator is evaluation of the error |p̂-λ| between
    trace of the potential @a p̂ and the trace unknown @a λ, which is also used
    for stabilization of the scheme. Therefore, adaptive mesh refinement (AMR)
    based on this kind of estimator supports convergence of the scheme.

    The first estimator, Type::Residual, is quite general and evaluates the
    residuum of the potential constraint, i.e., |G p + H λ| integrated over the
    face elements. It requires from the integrator provided in constructor
    (HDGErrorEstimator()) only to implement the method
    BilinearFormIntegrator::AssembleHDGFaceVector().

    On the other hand, Type::Energy evaluates energy-like norm ||p̂-λ||² ~ pᵀDp
    +pᵀEλ -λᵀGp -λᵀHλ. For classical stabilization term τ(p̂-λ), this yields
    the expression (p̂-λ)ᵀτ(p̂-λ), which can be generalized to anisotropic
    cases, where the product can be evaluated component-wise in the reference
    space. This functionality requires the integrator to implement
    BilinearFormIntegrator::ComputeHDGFaceEnergy(), including its @p d_energy
    optional parameter for setting the anisotropic flags
    (see SetAnisotropic() and GetAnisotropicFlags()).
 */
class HDGErrorEstimator : public AnisotropicErrorEstimator
{
public:
   enum class Type
   {
      Residual,   ///< Residuum of the constraint |G p + H λ|
      Energy,     ///< Energy-like norm ~ sqrt(pᵀDp + pᵀEλ - λᵀGp - λᵀHλ)
   };

private:
   BilinearFormIntegrator &bfi;
   const GridFunction &sol_tr, &sol_p;
#ifdef MFEM_USE_MPI
   const ParGridFunction *psol_tr {};
#endif
   Type type;

   long current_sequence{-1};
   Vector error_estimates;
   real_t total_error{};
   bool anisotropic{};
   Array<int> aniso_flags;

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = sol_tr.FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();

   /// Compute the face error estimate
   void ComputeFaceEstimate(int face, bool side2, Vector &d_error_estimates);

public:
   /// Constructor
   /** @param integ     HDG face matrix integrator used for estimation
       @param solr      trace solution
       @param solp      potential solution
       @param type_     type of estimator
    */
   HDGErrorEstimator(BilinearFormIntegrator &integ, const GridFunction &solr,
                     const GridFunction &solp, Type type_ = Type::Energy)
      : bfi(integ), sol_tr(solr), sol_p(solp), type(type_) { }

#ifdef MFEM_USE_MPI
   /// Constructor
   /** @param integ     HDG face matrix integrator used for estimation
       @param solr      trace solution
       @param solp      potential solution
       @param type_     type of estimator
    */
   HDGErrorEstimator(BilinearFormIntegrator &integ, const ParGridFunction &solr,
                     const GridFunction &solp, Type type_ = Type::Energy)
      : bfi(integ), sol_tr(solr), sol_p(solp), psol_tr(&solr), type(type_) { }
#endif

   /// Enable/disable anisotropic estimates.
   /** To enable this option, the HDG integrator must support the @p d_energy
       parameter in its BilinearFormIntegrator::ComputeHDGFaceEnergy() method.
    */
   void SetAnisotropic(bool aniso = true) { anisotropic = aniso; }

   /// Return the total error from the last error estimate.
   real_t GetTotalError() const override { return total_error; }

   /// Get a Vector with all element errors.
   const Vector &GetLocalErrors() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /// Get an Array<int> with anisotropic flags for all mesh elements.
   /** Return an empty array when anisotropic estimates are not available or
       enabled. */
   const Array<int> &GetAnisotropicFlags() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return aniso_flags;
   }

   /// Reset the error estimator.
   void Reset() override { current_sequence = -1; }
};

} // namespace mfem

#endif // MFEM_ESTIMATORS_HDG
