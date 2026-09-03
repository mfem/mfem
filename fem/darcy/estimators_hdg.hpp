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

class DarcyHybridization;

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

   /** @brief How the potential's trace is compared with λ on a face, which
       only becomes a question when the potential carries a HIGHER degree than
       the trace -- as it does when the estimate is built on the postprocessed
       potential, which is the case worth building it on.

       Write `M_h = P_k(e)` for the trace space on a face. A degree-`k+1`
       trace has a component no element of `M_h` can represent, and
       orthogonality splits the literal difference as

           ||p̂ - λ||² = ||P_M p̂ - λ||² + ||(I - P_M) p̂||²

       where the second piece is not an error: it survives when λ is the best
       trace of the exact solution. Comparing a `P_k(e)` function with a
       degree-`k+1` trace *as approximations to the same thing* means comparing
       them in `P_k(e)`, which is what Projected does.

       **Where the degrees agree the two are the same number, exactly**, since
       an element's trace on a face is then already in `M_h` and the projection
       is the identity -- so this changes nothing for a caller estimating on
       the computed potential, and Literal remains the default only so that the
       difference stays measurable rather than because it is ever wanted. */
   enum class TraceComparison
   {
      Literal,    ///< ||p̂ - λ||, as written
      Projected,  ///< ||P_M p̂ - λ||, inside the trace space
   };

private:
   BilinearFormIntegrator &bfi;
   const GridFunction &sol_tr, &sol_p;
#ifdef MFEM_USE_MPI
   const ParGridFunction *psol_tr {};
#endif
   Type type;

   Array<int> excl_bdr;
   const DarcyHybridization *hyb {};
   bool skip_enriched_dir {false};
   bool cap_at_element {false};
   TraceComparison trcmp{TraceComparison::Literal};
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

   /** @brief L2(e)-projection of a trace function from @a fe_hi onto the
       coarser trace element @a fe_lo, both on the same face. */
   static void ProjectTraceDown(const FiniteElement &fe_hi,
                                const FiniteElement &fe_lo,
                                FaceElementTransformations &FTr,
                                const Vector &tr_hi, Vector &tr_lo);

   /** @brief L2(e)-projection of one side's potential trace onto the trace
       element, returned as trace-basis coefficients. */
   static void ProjectOntoTrace(const FiniteElement &fe_tr,
                                const FiniteElement &el,
                                FaceElementTransformations &FTr, int side,
                                const Vector &elfun, Vector &c);

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

       **The directional split is only meaningful on the COMPUTED potential.**
       Handed the postprocessed one it points the wrong way and an adaptive
       loop then refines forever without touching the feature it is chasing.
       Measured on `anisodiff -p 5 -o 2 -hb -dg`, summing `d_energy` over the
       mesh, `d_0`/`d_1` with the layer in the `y` direction:

       | `-ks` | computed potential | postprocessed |
       |---|---|---|
       | 1 | 1.94 | 3.57 |
       | 100 | 4.2e-4 | 1.6e5 |
       | 10000 | 2.4e-5 | 1.9e7 |

       The computed potential puts the energy in `y`, correctly and more
       sharply as the layer sharpens. The postprocessed one puts it in `x`, by
       up to seven orders, and at `-ks 100` the total is nothing else: eta² is
       11.09 and `d_0` alone is 11.07.

       Three explanations are ruled out by measurement. It is not the degree
       gap against the trace -- projecting the postprocessed potential back to
       the potential's own degree leaves every flag where it was and moves eta
       by 0.5%. It is not the anisotropy -- the loop stalls the same way at
       `-ks 1` and on problem 6, a radially localised peak, both isotropic. And
       it is not the marking: under isotropic refinement the two estimates
       select the *same elements* for two cycles, agreeing to every printed
       digit, so only the direction differs.

       What is left is that the two are measuring different things.
       `p̂ - λ` on the computed potential is the scheme's own stabilization
       term; on a superconverged potential it is essentially λ's own error,
       which is a real quantity but not the element's, and attributing it to
       the direction NORMAL to the face is not the direction that would reduce
       it. The flag rule then converts a systematic bias into a wrong answer,
       because it is a hard threshold -- a direction is refined when it holds
       more than `0.15*3/dim` of the element's energy, 0.225 in 2D -- and at
       `-ks 1` the postprocessed estimate's `y` share is 0.219, missing it by
       3%, where the computed potential's is 0.34 and passes.

       **So take each from the field that answers it**, which needs no change
       here: build a second estimator on the computed potential, read
       GetAnisotropicFlags() from that one and GetLocalErrors() from the
       postprocessed one. It costs one more pass over the faces. Measured on
       the adaptive loop in `anisodiff`, which sat at 0.283893 through twelve
       cycles and 5352 dofs with both taken from the postprocessed field and
       reaches 2.5e-4 at M = 2217 with them separated -- and the postprocessed
       magnitude is then worth a further 1.4x in dofs, which it is not worth at
       all when it also supplies the direction. */
   void SetAnisotropic(bool aniso = true) { anisotropic = aniso; }

   /** @brief Boundary attributes whose faces the estimate leaves out. Marked
       with a nonzero entry per attribute; an empty array, the default, leaves
       nothing out and is what every caller had before this existed.

       **This is an omission, not a repair, and it exists because |p̂-λ| is an
       error only where λ is approximating the potential's trace.** It is not
       on a boundary face whose Dirichlet datum is imposed WEAKLY -- through
       `<T_D, v·n>` on the flux equation rather than by pinning λ -- because
       nothing there ties λ to the potential, and the constraint is not even
       assembled on such a face (MixedBilinearForm::AddBdrFaceIntegrator() is
       what puts it there, and a weak-datum caller does not add one). λ then
       settles somewhere unrelated to T and the term does not vanish with h.

       Measured on `anisodiff -p 5 -ks 1 -o 2 -hb -dg`, splitting the estimate
       by boundary attribute over nx = 8 and 16:

       | | interior | attrs 1,3 (datum) | attrs 2,4 (no datum) |
       |---|---|---|---|
       | nx = 8  | 5.68e-6 | 1.994 each | 1.39e-6 each |
       | nx = 16 | 2.49e-7 | 3.999 each | 4.25e-8 each |

       The interior converges at `h²`, which is optimal at `k = 2`, and so do
       the two boundaries carrying no datum. The two carrying one *double* --
       exactly `1/h`, one fixed amount per face -- so eta grows under
       refinement while the true error falls by 265x, and a refiner driven by
       it chases the boundary instead of the solution. Leaving those two
       attributes out restores `h²` for the total.

       Nothing needs excluding where the datum is essential (the trace pinned
       to it) or where it is zero: there λ and the potential's trace agree in
       the limit and the term is a real error. */
   void SetExcludedBoundary(const Array<int> &bdr_attr_marker)
   { bdr_attr_marker.Copy(excl_bdr); Reset(); }

   /** @brief Compare λ against the potential's trace literally, or against its
       projection into the trace space. See TraceComparison.

       **How much it is worth, measured rather than argued, and it is less than
       the splitting suggests.** On
       `anisodiff -p 5 -ks 1e2 -o 2 -hb -dg --postprocessed-estimate`, where
       the potential is degree 3 against a degree-2 trace, Projected moves eta
       from 3.499 to 3.327 -- 5% -- and changes not one marking decision: the
       same six elements are selected and the same six directional flags come
       back. So on this problem the component `M_h` cannot represent is a small
       part of the term rather than the whole of it, and dropping it is a
       correction to eta's *size* rather than to what it points at.

       **And it does not repair the one thing that looked like its doing.**
       With the postprocessed potential the directional split of
       GetAnisotropicFlags() points the wrong way, and that was the reason this
       option was written. It survives the projection unchanged: same six flags,
       same frozen loop, eta moved by 5%. The measurement that settled it is
       stronger still -- projecting the postprocessed potential back onto the
       potential's own DEGREE, which removes the gap entirely rather than
       removing one component of it, also leaves every flag where it was. So
       the degree gap is not what misdirects the split; see SetAnisotropic()
       for what is. */
   void SetTraceComparison(TraceComparison c) { trcmp = c; Reset(); }

   /** @brief Read the per-face trace degrees from @a hyb_, for `p`-adaptivity.

       **No longer required for the estimate to be right, and that is worth
       saying because it used to be.** While the surplus slots were RETIRED, a
       coarser face held a coarser basis's coefficients in the ceiling's
       storage, and reading it through the constraint space's own element
       evaluated the ceiling basis against them -- a different function, not
       an approximation of one, since the two bases are nodal at different
       points. Measured on `anisodiff -p 5 -ks 1e2 -o 2 -hb -dg`, one cycle,
       every face at degree 2, changing only the degree the constraint space
       was BUILT at: eta went 0.325, 5.92, 8.81 at ceilings 2, 3, 5, while the
       solution was the same to six digits. A factor of 27 from a parameter
       that must be inert.

       The surplus is CONSTRAINED now, so those coefficients are the ceiling's
       own and a generic reader is right by default. The unit test "The error
       estimate does not depend on the trace ceiling" pins both halves: the
       estimate does not move when the ceiling is raised, and it does not move
       when the estimator is told nothing.

       What this still buys is the DEGREE, which the constraint space does not
       carry and which two things below need: a face richer than its element
       is a real configuration, and both flags are about exactly that. Without
       the degrees a raised ceiling would make every face look enriched. */
   void SetHybridization(const DarcyHybridization &hyb_)
   {
      hyb = &hyb_;
      /* Both of the below are wanted exactly when per-face trace degrees
         exist, and per-face trace degrees exist exactly when there is a
         hybridization to read them from -- so this turns them on rather than
         leaving a caller to discover, from an adaptive loop that quietly goes
         nowhere, that it had to. Each is measured inert where no face outruns
         its element: the h-adaptive loop is identical to every printed digit
         with and without. Call either setter afterwards to measure the
         difference. */
      skip_enriched_dir = true;
      cap_at_element = true;
      Reset();
   }

   /** @brief Where a face's trace degree exceeds the element's, keep that
       face's contribution to the element's error but not to its DIRECTION.

       **The excess is real, belongs to that element, and points the wrong
       way.** A face richer than the element on one side carries modes that
       element cannot represent. On a conforming face they are zero, because a
       trace above *both* its neighbours is exactly redundant. Across a hanging
       node they are not: the master trace fits the several fine elements
       better than it fits the one coarse element, so `p̂ - λ` on the coarse
       side genuinely grows -- and under `p`-adaptivity that is every hanging
       node, because a family has to run at the ceiling degree
       (DarcyHybridization::SetTraceOrders()).

       As a magnitude that is right: the coarse element IS the mismatched one.
       As a direction it is exactly wrong. Refining an element in `y` puts
       hanging nodes on its *vertical* faces, the geometric split attributes a
       vertical face's energy to `x`, and the neighbour is then split in `x` --
       when what would actually match its neighbours is another `y`. So the
       loop alternates and never resolves anything. Measured on
       `anisodiff -p 5 -ks 1e2 -o 2 -hb -dg`, one identical hanging-node mesh,
       changing only the ceiling from 2 to 3, summed over the twelve elements
       next to a hanging node against the other 58:

       | | Σd₀ at ceiling 2 | at 3 | Σd₁ at 2 | at 3 |
       |---|---|---|---|---|
       | next to a hanging node | 1.11e-4 | **5.45e-2** | 6.55e-3 | 4.51e-3 |
       | everything else | 2.91e-5 | 2.83e-5 | 6.93e-3 | 3.60e-3 |

       A factor of 490, confined to those twelve elements, and entirely in
       `d₀`. Four of them flip from `y` to `x` with their estimate up by 17x.

       **Two other repairs were tried and measured to fail**, which is why this
       one is worth its lines. Dropping the face altogether stalls the hp loop
       at 1.7e-3 against 1.4e-6: it discards the part of `p̂ - λ` the element
       CAN see along with the part it cannot. And projecting λ down to the
       element's own degree before comparing -- which removes exactly the modes
       the element cannot represent -- moves eta by 2%, from 0.250 to 0.245,
       and changes no flag: the excess is not in λ's high modes at all, it is
       in where λ sits, and λ sits where the fine side puts it.

       With this, anisotropic refinement works under `p`-adaptivity, which it
       otherwise cannot: on the demonstrator it reaches 1.05e-4 at M = 921
       against the isotropic loop's 1351, and 1.8e-6 at M = 1302 against 2473,
       1.5 to 1.9 times fewer unknowns throughout. It then plateaus at 8.7e-7
       where the isotropic loop carries on to 5.9e-8, and that plateau is not
       understood.

       **How much this and SetCapTraceAtElement() are worth fell by three
       orders when the trace surplus stopped being retired**, and the reason is
       worth knowing rather than the number. Turning the cap off used to
       plateau the demonstrator at 8.7e-7; it now reaches 1.02e-9, against
       9.9e-10 with it on. Neither flag's own logic changed -- both still
       trigger on a face whose degree exceeds its element's -- but a
       hanging-node family is no longer forced to the CEILING degree, and that
       was where most faces richer than their elements came from. The flags
       still discriminate, and only just; a driver that leaves them off is now
       merely slightly worse rather than stopped. */
   void SetSkipEnrichedDirection(bool skip = true)
   { skip_enriched_dir = skip; Reset(); }

   /** @brief Where a face's trace degree exceeds the element's, compare that
       element against λ projected down to its own degree.

       **The magnitude half of what SetSkipEnrichedDirection() handles the
       direction half of**, and both are needed. A face richer than the element
       on one side carries modes that element cannot represent; charging them
       to it makes the estimate diverge from the truth, and -- because
       refinement makes it worse -- diverge further the more the loop acts on
       it.

       Measured on `anisodiff -p 5 -ks 1e2 -o 2 -hb -dg --hp-adaptivity` with
       the `max` face rule, comparing the estimate against the TRUE per-element
       error at the point where the loop had stopped moving. It was marking a
       cluster of degree-2 elements next to degree-5 ones, at `x ~ 0.63` in the
       middle of the domain:

       | cycle | η on the marked cluster | true error there | ratio |
       |---|---|---|---|
       | 22 | 4.7e-6 | 1.1e-8 | 443 |
       | 24 | 9.3e-6 | 5.4e-9 | 1700 |
       | 25 | 1.2e-5 | 3.8e-9 | 3000 |

       The estimate is wrong by three orders and getting worse, while the
       elements actually carrying the error -- five times more of it, at
       `x = 0.812` -- go unmarked. It is self-feeding: splitting those elements
       in `x` makes them narrower, `τ ~ 1/h` on their vertical faces grows, and
       η grows with it, so the refinement the estimate triggers is what makes
       the estimate bigger.

       With this the loop reaches 4.5e-10 where it had plateaued at 8.8e-7, and
       the `max` face rule becomes usable -- worth about 10% of the dofs at
       every matched error and an order deeper in the same cycle budget.

       **Where it does not bite, and why that is not a contradiction.** At a
       hanging-node family whose elements are all the same degree, projecting λ
       down moves eta by 2% and changes no flag: the excess there is not in λ's
       high modes but in where λ sits, and the direction half is what matters.
       At a genuine `p`-interface one element really is coarser than what λ
       carries, and then this is the whole of it. Both were measured; neither
       alone is enough. */
   void SetCapTraceAtElement(bool cap = true) { cap_at_element = cap; Reset(); }

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
