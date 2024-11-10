// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ERROR_ESTIMATORS
#define MFEM_ERROR_ESTIMATORS

#include <functional>

#include "../config/config.hpp"
#include "../linalg/vector.hpp"
#include "bilinearform.hpp"
#ifdef MFEM_USE_MPI
#include "pgridfunc.hpp"
#endif

namespace mfem
{

/** @brief Base class for all error estimators.
 */
class AbstractErrorEstimator
{
public:
   virtual ~AbstractErrorEstimator() {}
};


/** @brief Base class for all element based error estimators.

    At a minimum, an ErrorEstimator must be able compute one non-negative real
    (double) number for each element in the Mesh.
 */
class ErrorEstimator : public AbstractErrorEstimator
{
public:
   /// Return the total error from the last error estimate.
   /** @note This method is optional for derived classes to override and the
       base class implementation simply returns 0. */
   virtual real_t GetTotalError() const { return 0.0; }

   /// Get a Vector with all element errors.
   virtual const Vector &GetLocalErrors() = 0;

   /// Force recomputation of the estimates on the next call to GetLocalErrors.
   virtual void Reset() = 0;

   /// Destruct the error estimator
   virtual ~ErrorEstimator() { }
};


/** @brief The AnisotropicErrorEstimator class is the base class for all error
    estimators that compute one non-negative real (double) number and an
    anisotropic flag for every element in the Mesh.
 */
class AnisotropicErrorEstimator : public ErrorEstimator
{
public:
   /** @brief Get an Array<int> with anisotropic flags for all mesh elements.
       @return An empty array when anisotropic estimates are not available or
       enabled. */
   virtual const Array<int> &GetAnisotropicFlags() = 0;
};


/** @brief The ZienkiewiczZhuEstimator class implements the Zienkiewicz-Zhu
    error estimation procedure.

    [1] Zienkiewicz, O.C. and Zhu, J.Z., The superconvergent patch recovery
    and a posteriori error estimates. Part 1: The recovery technique.
    Int. J. Num. Meth. Engng. 33, 1331-1364 (1992).

    [2] Zienkiewicz, O.C. and Zhu, J.Z., The superconvergent patch recovery
    and a posteriori error estimates. Part 2: Error estimates and adaptivity.
    Int. J. Num. Meth. Engng. 33, 1365-1382 (1992).

    The required BilinearFormIntegrator must implement the methods
    ComputeElementFlux() and ComputeFluxEnergy().
 */
class ZienkiewiczZhuEstimator : public AnisotropicErrorEstimator
{
protected:
   long current_sequence;
   Vector error_estimates;
   real_t total_error;
   bool anisotropic;
   Array<int> aniso_flags;
   int flux_averaging; // see SetFluxAveraging()

   BilinearFormIntegrator &integ;
   GridFunction &solution;

   FiniteElementSpace *flux_space; /**< @brief Ownership based on own_flux_fes.
      Its Update() method is called automatically by this class when needed. */
   bool with_coeff;
   bool own_flux_fes; ///< Ownership flag for flux_space.

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = solution.FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();

public:
   /** @brief Construct a new ZienkiewiczZhuEstimator object.
       @param integ    This BilinearFormIntegrator must implement the methods
                       ComputeElementFlux() and ComputeFluxEnergy().
       @param sol      The solution field whose error is to be estimated.
       @param flux_fes The ZienkiewiczZhuEstimator assumes ownership of this
                       FiniteElementSpace and will call its Update() method when
                       needed.*/
   ZienkiewiczZhuEstimator(BilinearFormIntegrator &integ, GridFunction &sol,
                           FiniteElementSpace *flux_fes)
      : current_sequence(-1),
        total_error(),
        anisotropic(false),
        flux_averaging(0),
        integ(integ),
        solution(sol),
        flux_space(flux_fes),
        with_coeff(false),
        own_flux_fes(true)
   { }

   /** @brief Construct a new ZienkiewiczZhuEstimator object.
       @param integ    This BilinearFormIntegrator must implement the methods
                       ComputeElementFlux() and ComputeFluxEnergy().
       @param sol      The solution field whose error is to be estimated.
       @param flux_fes The ZienkiewiczZhuEstimator does NOT assume ownership of
                       this FiniteElementSpace; will call its Update() method
                       when needed. */
   ZienkiewiczZhuEstimator(BilinearFormIntegrator &integ, GridFunction &sol,
                           FiniteElementSpace &flux_fes)
      : current_sequence(-1),
        total_error(),
        anisotropic(false),
        flux_averaging(0),
        integ(integ),
        solution(sol),
        flux_space(&flux_fes),
        with_coeff(false),
        own_flux_fes(false)
   { }

   /** @brief Consider the coefficient in BilinearFormIntegrator to calculate
       the fluxes for the error estimator.*/
   void SetWithCoeff(bool w_coeff = true) { with_coeff = w_coeff; }

   /** @brief Enable/disable anisotropic estimates. To enable this option, the
       BilinearFormIntegrator must support the 'd_energy' parameter in its
       ComputeFluxEnergy() method. */
   void SetAnisotropic(bool aniso = true) { anisotropic = aniso; }

   /** @brief Set the way the flux is averaged (smoothed) across elements.

       When @a fa is zero (default), averaging is performed across interfaces
       between different mesh attributes. When @a fa is non-zero, the flux is
       not averaged across interfaces between different mesh attributes. */
   void SetFluxAveraging(int fa) { flux_averaging = fa; }

   /// Return the total error from the last error estimate.
   real_t GetTotalError() const override { return total_error; }

   /// Get a Vector with all element errors.
   const Vector &GetLocalErrors() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /** @brief Get an Array<int> with anisotropic flags for all mesh elements.
       Return an empty array when anisotropic estimates are not available or
       enabled. */
   const Array<int> &GetAnisotropicFlags() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return aniso_flags;
   }

   /// Reset the error estimator.
   void Reset() override { current_sequence = -1; }

   /** @brief Destroy a ZienkiewiczZhuEstimator object. Destroys, if owned, the
       FiniteElementSpace, flux_space. */
   virtual ~ZienkiewiczZhuEstimator()
   {
      if (own_flux_fes) { delete flux_space; }
   }
};


/** @brief The LSZienkiewiczZhuEstimator class implements the Zienkiewicz-Zhu
    error estimation procedure [1,2] using face-based patches [3].

    [1] Zienkiewicz, O.C. and Zhu, J.Z., The superconvergent patch recovery
    and a posteriori error estimates. Part 1: The recovery technique.
    Int. J. Num. Meth. Engng. 33, 1331-1364 (1992).

    [2] Zienkiewicz, O.C. and Zhu, J.Z., The superconvergent patch recovery
    and a posteriori error estimates. Part 2: Error estimates and adaptivity.
    Int. J. Num. Meth. Engng. 33, 1365-1382 (1992).

    [3] Bartels, S. and Carstensen, C., Each averaging technique yields reliable
    a posteriori error control in FEM on unstructured grids. Part II: Higher
    order FEM. Math. Comp. 71(239), 971-994 (2002)

    The required BilinearFormIntegrator must implement the method
    ComputeElementFlux().

   @note
   -  The present implementation ignores all single-element patches corresponding
      to boundary faces. This is appropriate for Dirichlet boundaries, but
      suboptimal for Neumann boundaries. Reference 3 shows that a constrained
      least-squares problem, where the reconstructed flux is constrained by the
      Neumann boundary data, is appropriate to handle this case.
      NOTE THAT THIS CONSTRAINED LS PROBLEM IS NOT YET IMPLEMENTED, so it is
      possible that the local error estimates for elements on a Neumann boundary
      are suboptimal.
   -  The global polynomial basis used for the flux reconstruction is, by default,
      aligned with the physical Cartesian axis. For patches with 2D elements, this
      has been improved on so that the basis is aligned with the physical patch
      orientation. Reorientation of the flux reconstruction basis is helpful to
      maintain symmetry in the refinement pattern and could be extended to 3D.
   -  This estimator is ONLY implemented IN SERIAL.
   -  Anisotropic refinement is NOT YET SUPPORTED.

 */
class LSZienkiewiczZhuEstimator : public ErrorEstimator
{
protected:
   long current_sequence;
   Vector error_estimates;
   real_t total_error;
   bool subdomain_reconstruction = true;
   real_t tichonov_coeff;

   BilinearFormIntegrator &integ;
   GridFunction &solution;
   bool with_coeff;

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = solution.FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();

public:
   /** @brief Construct a new LSZienkiewiczZhuEstimator object.
       @param integ    This BilinearFormIntegrator must implement only the
                       method ComputeElementFlux().
       @param sol      The solution field whose error is to be estimated.
   */
   LSZienkiewiczZhuEstimator(BilinearFormIntegrator &integ, GridFunction &sol)
      : current_sequence(-1),
        total_error(-1.0),
        subdomain_reconstruction(true),
        tichonov_coeff(0.0),
        integ(integ),
        solution(sol),
        with_coeff(false)
   { }

   /** @brief Consider the coefficient in BilinearFormIntegrator to calculate
       the fluxes for the error estimator.*/
   void SetWithCoeff(bool w_coeff = true) { with_coeff = w_coeff; }

   /** @brief Disable reconstructing the flux in patches spanning different
    *         subdomains. */
   void DisableReconstructionAcrossSubdomains() { subdomain_reconstruction = false; }

   /** @brief Solve a Tichonov-regularized least-squares problem for the
    *         reconstructed fluxes. This is especially helpful for when not
    *         using tensor product elements, which typically require fewer
    *         integration points and, therefore, may lead to an
    *         ill-conditioned linear system. */
   void SetTichonovRegularization(real_t tcoeff = 1.0e-8)
   {
      MFEM_VERIFY(tcoeff >= 0.0, "Tichonov coefficient cannot be negative");
      tichonov_coeff = tcoeff;
   }

   /// Return the total error from the last error estimate.
   real_t GetTotalError() const override { return total_error; }

   /// Get a Vector with all element errors.
   const Vector &GetLocalErrors() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /// Reset the error estimator.
   void Reset() override { current_sequence = -1; }

   virtual ~LSZienkiewiczZhuEstimator() { }
};


#ifdef MFEM_USE_MPI

/** @brief The L2ZienkiewiczZhuEstimator class implements the Zienkiewicz-Zhu
    error estimation procedure where the flux averaging is replaced by a global
    L2 projection (requiring a mass matrix solve).

    The required BilinearFormIntegrator must implement the methods
    ComputeElementFlux() and ComputeFluxEnergy().

    Implemented for the parallel case only.
 */
class L2ZienkiewiczZhuEstimator : public ErrorEstimator
{
protected:
   long current_sequence;
   int local_norm_p; ///< Local L_p norm to use, default is 1.
   Vector error_estimates;
   real_t total_error;

   BilinearFormIntegrator &integ;
   ParGridFunction &solution;

   ParFiniteElementSpace *flux_space; /**< @brief Ownership based on the flag
      own_flux_fes. Its Update() method is called automatically by this class
      when needed. */
   ParFiniteElementSpace *smooth_flux_space; /**< @brief Ownership based on the
      flag own_flux_fes. Its Update() method is called automatically by this
      class when needed.*/
   bool own_flux_fes; ///< Ownership flag for flux_space and smooth_flux_space.

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = solution.FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();

public:
   /** @brief Construct a new L2ZienkiewiczZhuEstimator object.
       @param integ    This BilinearFormIntegrator must implement the methods
                       ComputeElementFlux() and ComputeFluxEnergy().
       @param sol      The solution field whose error is to be estimated.
       @param flux_fes The L2ZienkiewiczZhuEstimator assumes ownership of this
                       FiniteElementSpace and will call its Update() method when
                       needed.
       @param smooth_flux_fes
                       The L2ZienkiewiczZhuEstimator assumes ownership of this
                       FiniteElementSpace and will call its Update() method when
                       needed. */
   L2ZienkiewiczZhuEstimator(BilinearFormIntegrator &integ,
                             ParGridFunction &sol,
                             ParFiniteElementSpace *flux_fes,
                             ParFiniteElementSpace *smooth_flux_fes)
      :  current_sequence(-1),
         local_norm_p(1),
         total_error(0.0),
         integ(integ),
         solution(sol),
         flux_space(flux_fes),
         smooth_flux_space(smooth_flux_fes),
         own_flux_fes(true)
   { }

   /** @brief Construct a new L2ZienkiewiczZhuEstimator object.
       @param integ    This BilinearFormIntegrator must implement the methods
                       ComputeElementFlux() and ComputeFluxEnergy().
       @param sol      The solution field whose error is to be estimated.
       @param flux_fes The L2ZienkiewiczZhuEstimator does NOT assume ownership
                       of this FiniteElementSpace; will call its Update() method
                       when needed.
       @param smooth_flux_fes
                       The L2ZienkiewiczZhuEstimator does NOT assume ownership
                       of this FiniteElementSpace; will call its Update() method
                       when needed. */
   L2ZienkiewiczZhuEstimator(BilinearFormIntegrator &integ,
                             ParGridFunction &sol,
                             ParFiniteElementSpace &flux_fes,
                             ParFiniteElementSpace &smooth_flux_fes)
      :  current_sequence(-1),
         local_norm_p(1),
         total_error(0.0),
         integ(integ),
         solution(sol),
         flux_space(&flux_fes),
         smooth_flux_space(&smooth_flux_fes),
         own_flux_fes(false)
   { }

   /** @brief Set the exponent, p, of the Lp norm used for computing the local
       element errors. Default value is 1. */
   void SetLocalErrorNormP(int p) { local_norm_p = p; }

   /// Return the total error from the last error estimate.
   real_t GetTotalError() const override { return total_error; }

   /// Get a Vector with all element errors.
   const Vector &GetLocalErrors() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /// Reset the error estimator.
   void Reset() override { current_sequence = -1; }

   /** @brief Destroy a L2ZienkiewiczZhuEstimator object. Destroys, if owned,
       the FiniteElementSpace, flux_space. */
   virtual ~L2ZienkiewiczZhuEstimator()
   {
      if (own_flux_fes) { delete flux_space; delete smooth_flux_space; }
   }
};

#endif // MFEM_USE_MPI


/** @brief The LpErrorEstimator class compares the solution to a known
    coefficient.

    This class can be used, for example, to adapt a mesh to a non-trivial
    initial condition in a time-dependent simulation. It can also be used to
    force refinement in the neighborhood of small features before switching to a
    more traditional error estimator.

    The LpErrorEstimator supports either scalar or vector coefficients and works
    both in serial and in parallel.
*/
class LpErrorEstimator : public ErrorEstimator
{
protected:
   long current_sequence;
   int local_norm_p;
   Vector error_estimates;

   real_t total_error = 0.0;

   Coefficient * coef;
   VectorCoefficient * vcoef;
   GridFunction * sol;

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = sol->FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();

public:
   /** @brief Construct a new LpErrorEstimator object for a scalar field.
       @param p    Integer which selects which Lp norm to use.
       @param sol  The GridFunction representation of the scalar field.
       Note: the coefficient must be set before use with the SetCoef method.
   */
   LpErrorEstimator(int p, GridFunction &sol)
      : current_sequence(-1), local_norm_p(p),
        error_estimates(0), coef(NULL), vcoef(NULL), sol(&sol) { }

   /** @brief Construct a new LpErrorEstimator object for a scalar field.
       @param p    Integer which selects which Lp norm to use.
       @param coef The scalar Coefficient to compare to the solution.
       @param sol  The GridFunction representation of the scalar field.
   */
   LpErrorEstimator(int p, Coefficient &coef, GridFunction &sol)
      : current_sequence(-1), local_norm_p(p),
        error_estimates(0), coef(&coef), vcoef(NULL), sol(&sol) { }

   /** @brief Construct a new LpErrorEstimator object for a vector field.
       @param p    Integer which selects which Lp norm to use.
       @param coef The vector VectorCoefficient to compare to the solution.
       @param sol  The GridFunction representation of the vector field.
   */
   LpErrorEstimator(int p, VectorCoefficient &coef, GridFunction &sol)
      : current_sequence(-1), local_norm_p(p),
        error_estimates(0), coef(NULL), vcoef(&coef), sol(&sol) { }

   /** @brief Set the exponent, p, of the Lp norm used for computing the local
       element errors. */
   void SetLocalErrorNormP(int p) { local_norm_p = p; }

   void SetCoef(Coefficient &A) { coef = &A; }
   void SetCoef(VectorCoefficient &A) { vcoef = &A; }

   /// Reset the error estimator.
   void Reset() override { current_sequence = -1; }

   /// Get a Vector with all element errors.
   const Vector &GetLocalErrors() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /// Destructor
   virtual ~LpErrorEstimator() {}
};


/** @brief The KellyErrorEstimator class provides a fast error indication
    strategy for smooth scalar parallel problems.

    The Kelly error indicator is based on the following papers:

    [1] Kelly, D. W., et al. "A posteriori error analysis and adaptive processes
    in the finite element method: Part I—Error analysis." International journal
    for numerical methods in engineering 19.11 (1983): 1593-1619.

    [2] De SR Gago, J. P., et al. "A posteriori error analysis and adaptive
    processes in the finite element method: Part II—Adaptive mesh refinement."
    International journal for numerical methods in engineering 19.11 (1983):
    1621-1656.

    It can be roughly described by:
        ||∇(u-uₕ)||ₑ ≅ √( C hₑ ∑ₖ (hₖ ∫ |J[∇uₕ]|²) dS )
    where "e" denotes an element, ||⋅||ₑ the corresponding local norm and k the
    corresponding faces. u is the analytic solution and uₕ the discretized
    solution. hₖ and hₑ are factors dependent on the face and element geometry.
    J is the jump function, i.e. the difference between the limits at each point
    for each side of the face. A custom method to compute hₖ can be provided. It
    is also possible to estimate the error only on a subspace by feeding this
    class an attribute array describing the subspace.

    @note This algorithm is appropriate only for problems with scalar diffusion
    coefficients (e.g. Poisson problems), because it measures only the flux of
    the gradient of the discrete solution. The current implementation also does
    not include the volume term present in Equation 75 of Kelly et al [1].
    Instead, it includes only the flux term recommended in Equation 82. The
    current implementation also does not include the constant factor "C". It
    furthermore assumes that the approximation error at the boundary is small
    enough, as the implementation ignores boundary faces.
*/
class KellyErrorEstimator final : public ErrorEstimator
{
public:
   /// Function type to compute the local coefficient hₑ of an element.
   using ElementCoefficientFunction =
      std::function<real_t(Mesh*, const int)>;
   /** @brief Function type to compute the local coefficient hₖ of a face. The
       third argument is true for shared faces and false for local faces. */
   using FaceCoefficientFunction =
      std::function<real_t(Mesh*, const int, const bool)>;

private:
   int current_sequence = -1;

   Vector error_estimates;

   real_t total_error = 0.0;

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

#ifdef MFEM_USE_MPI
   const bool isParallel;
#endif

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
   /** @brief Construct a new KellyErrorEstimator object for a scalar field.
       @param di_         The bilinearform to compute the interface flux.
       @param sol_        The solution field whose error is to be estimated.
       @param flux_fes_   The finite element space for the interface flux.
       @param attributes_ The attributes of the subdomain(s) for which the
                          error should be estimated. An empty array results in
                          estimating the error over the complete domain.
   */
   KellyErrorEstimator(BilinearFormIntegrator& di_, GridFunction& sol_,
                       FiniteElementSpace& flux_fes_,
                       const Array<int> &attributes_ = Array<int>());

   /** @brief Construct a new KellyErrorEstimator object for a scalar field.
       @param di_         The bilinearform to compute the interface flux.
       @param sol_        The solution field whose error is to be estimated.
       @param flux_fes_   The finite element space for the interface flux.
       @param attributes_ The attributes of the subdomain(s) for which the
                          error should be estimated. An empty array results in
                          estimating the error over the complete domain.
   */
   KellyErrorEstimator(BilinearFormIntegrator& di_, GridFunction& sol_,
                       FiniteElementSpace* flux_fes_,
                       const Array<int> &attributes_ = Array<int>());

   ~KellyErrorEstimator();

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

   real_t GetTotalError() const override { return total_error; }

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

} // namespace mfem

#endif // MFEM_ERROR_ESTIMATORS
