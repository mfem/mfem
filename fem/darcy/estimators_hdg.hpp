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

/** @brief The boundary-datum term of a face-based error estimator: the
    mismatch between a computed field and the datum actually imposed on marked
    boundary faces.

    This is @f$\eta_5@f$ of Sánchez-Vizuet, Solano & Cerfon eq. (20), and it
    is the one term of that estimator HDGErrorEstimator above cannot express.
    Both of its types are built from an HDG face integrator and measure
    @f$|\hat p - \lambda|@f$ between the potential's trace and the trace
    unknown; @f$\eta_5@f$ instead compares a field against a **coefficient**,
    and on the extension path of the subdomain method that coefficient is
    TransferredDatumCoefficient, the datum @f$\varphi_h = g\circ a +
    L_e(\boldsymbol u_h)@f$ transferred along the paths. There is no integrator
    to build it from, which is why it is a class of its own rather than a third
    Type.

    **Why it matters, measured on the application that asked for it.** On the
    extension path the trace unknown on @f$\Gamma_h@f$ is pinned rather than
    free, so an estimator omitting this term compares the postprocessed
    potential against **zero**. The difference is then
    @f$O(\mathrm{dist}(\Gamma_h, \Gamma)) = O(h)@f$ and swamps the rest: at
    @f$k = 2@f$ the total read @f$\eta = 4.09\mathrm{e}{-1}@f$ against
    @f$\eta_1 = 2.12\mathrm{e}{-3}@f$, converging at about one half. An
    adaptive loop built on it runs, produces plausible pictures, and refines
    the wrong elements. Excluding those faces restores the rate and is an
    omission rather than a repair -- the term carries information and was
    being discarded.

    Each element's entry is @f$\sqrt{\int_F (u_h - \varphi)^2}@f$ summed over
    its marked faces, and the total is the square root of the sum, which is the
    convention HDGErrorEstimator's Type::Energy uses.

    @note The estimator evaluates the coefficient through the
    FaceElementTransformations and at the FACE integration point, because a
    path family may need the outward normal and an interpolated direction is
    not a function of the point alone. It also takes the weight and the field
    value BEFORE evaluating the coefficient, because
    PathLiftCoefficient::Eval() moves both transformations -- see the warning
    on that class. Getting that order wrong does not fail loudly; it silently
    integrates against a geometry that has walked off the face. */
class HDGDatumErrorEstimator : public ErrorEstimator
{
   const GridFunction &sol;
   Coefficient &datum;
   Array<int> bdr_marker;
   int ir_order;

   // Mutable, and ComputeEstimates() is const, so that GetTotalError() can
   // bring itself up to date. ErrorEstimator declares that method const, so an
   // estimator holding its state non-mutably can only return whatever
   // GetLocalErrors() last left there -- which is ZERO for a caller that asks
   // for the total and never asks for the local errors. That is a silent wrong
   // answer rather than an abort, it is what the first draft of this class
   // did, and the test caught it only because it checked a value rather than a
   // tolerance. HDGErrorEstimator above has the same shape and the same wart.
   mutable long current_sequence{-1};
   mutable Vector error_estimates;
   mutable real_t total_error{};

   bool MeshIsModified() const
   {
      long mesh_sequence = sol.FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   void ComputeEstimates() const;

public:
   /** @param sol_     the field to compare, typically the postprocessed
                       potential.
       @param datum_   the datum actually imposed on those faces.
       @param marker_  boundary attributes to integrate over, one entry per
                       attribute.
       @param ir_ord   quadrature order along a face; negative takes twice the
                       element order plus two. */
   HDGDatumErrorEstimator(const GridFunction &sol_, Coefficient &datum_,
                          const Array<int> &marker_, int ir_ord = -1)
      : sol(sol_), datum(datum_), bdr_marker(marker_), ir_order(ir_ord) { }

   /// Quadrature order along a face; negative restores the default.
   void SetIntegrationOrder(int order) { ir_order = order; Reset(); }

   real_t GetTotalError() const override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return total_error;
   }

   const Vector &GetLocalErrors() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   void Reset() override { current_sequence = -1; }
};

} // namespace mfem

#endif // MFEM_ESTIMATORS_HDG
