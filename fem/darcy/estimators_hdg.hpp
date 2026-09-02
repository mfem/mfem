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

/** @brief Persson & Peraire's smoothness sensor, per element.

    Writing the solution on an element in a hierarchical orthogonal basis and
    truncating it one degree,

        S_e = (u - u_hat, u - u_hat)_e / (u, u)_e,   s_e = log10 S_e

    which is Persson & Peraire, "Sub-Cell Shock Capturing for Discontinuous
    Galerkin Methods", AIAA 2006-112, eqs (5)-(7). It measures how much of the
    element's energy sits in its top degree: little, and the expansion is
    decaying and the solution is resolved; a lot, and it is not. The paper
    quotes `S_e ~ 1/p^4` for a smooth solution in 1D, which is what makes
    `s_0 ~ -4 log10(p)` the natural threshold.

    **This is a smoothness sensor and not an error estimator, and it is not one
    of those on purpose.** It says how well the element resolves what it holds,
    not how wrong that is; a badly under-resolved smooth region and a
    well-resolved discontinuity are opposite here and can be similar to an
    estimator. The intended use is the other half of an `hp` decision --
    an error estimator says *where* to spend, this says whether to spend it on
    `h` or on `p` -- so it deliberately does not derive from ErrorEstimator and
    cannot be handed to a ThresholdRefiner as if it were one.

    **Basis.** The paper writes the truncation as dropping coefficients, which
    needs the hierarchical orthogonal basis it assumes -- Legendre in 1D,
    Koornwinder on triangles -- and MFEM has no such basis: every BasisType is
    nodal or Bernstein. But eq (7) is an inner product, not a statement about
    coefficients, so it is computed here as the L2 projection onto the space
    one degree down, which is the same number in any basis. Orthogonality of
    that projection is also what lets the numerator be a difference of norms
    rather than a norm of a difference.

    **Variable order is the case this exists for**, so the degree is read per
    element and the truncation follows it. An element already at degree 0 has
    nothing to truncate to; it reports `S_e = 1`, the least smooth value, which
    is the right answer for a driver -- that element resolves nothing and
    should be refined either way.

    **Systems** are handled per field and the worst field wins, so one
    unresolved component is enough to mark the element. */
class PerssonPeraireSmoothness
{
   const GridFunction &u;
   Vector S;            ///< S_e per element
   real_t zero_tol;     ///< below this share of the norm an element reads as smooth
   bool computed{false};

public:
   /** @brief Sense @a field, which must be a discontinuous (L2) space: the
       truncation is elementwise and means nothing across a continuous one.
       @a zero_tol is the fraction of the field's mean energy below which an
       element is taken to carry nothing and reported as perfectly smooth,
       which keeps `S_e` from being 0/0 where the solution vanishes. */
   PerssonPeraireSmoothness(const GridFunction &field, real_t zero_tol = 1e-14);

   /// The raw sensor @a S_e of eq (7), one per element.
   const Vector &GetSensor();

   /// `log10` of it, the @a s_e the paper thresholds on.
   void GetLogSensor(Vector &s_e);

   /** @brief The paper's threshold, `s_0 ~ -4 log10(p)`, for degree @a p.
       Offered because it is the one number in the method that is a convention
       rather than a measurement, and it belongs next to the sensor that uses
       it rather than copied into every caller. */
   static real_t Threshold(int p) { return -4.0 * std::log10((p > 0) ? p : 1); }

   /// Recompute on the next query.
   void Reset() { computed = false; }
};

} // namespace mfem

#endif // MFEM_ESTIMATORS_HDG
