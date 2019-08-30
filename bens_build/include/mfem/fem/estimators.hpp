// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_ERROR_ESTIMATORS
#define MFEM_ERROR_ESTIMATORS

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
   /// Get a Vector with all element errors.
   virtual const Vector &GetLocalErrors() = 0;

   /// Force recomputation of the estimates on the next call to GetLocalErrors.
   virtual void Reset() = 0;

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

    The required BilinearFormIntegrator must implement the methods
    ComputeElementFlux() and ComputeFluxEnergy().
 */
class ZienkiewiczZhuEstimator : public AnisotropicErrorEstimator
{
protected:
   long current_sequence;
   Vector error_estimates;
   double total_error;
   bool anisotropic;
   Array<int> aniso_flags;
   int flux_averaging; // see SetFluxAveraging()

   BilinearFormIntegrator *integ; ///< Not owned.
   GridFunction *solution; ///< Not owned.

   FiniteElementSpace *flux_space; /**< @brief Ownership based on own_flux_fes.
      Its Update() method is called automatically by this class when needed. */
   bool own_flux_fes; ///< Ownership flag for flux_space.

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = solution->FESpace()->GetMesh()->GetSequence();
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
                       needed. */
   ZienkiewiczZhuEstimator(BilinearFormIntegrator &integ, GridFunction &sol,
                           FiniteElementSpace *flux_fes)
      : current_sequence(-1),
        total_error(),
        anisotropic(false),
        flux_averaging(0),
        integ(&integ),
        solution(&sol),
        flux_space(flux_fes),
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
        integ(&integ),
        solution(&sol),
        flux_space(&flux_fes),
        own_flux_fes(false)
   { }

   /** @brief Enable/disable anisotropic estimates. To enable this option, the
       BilinearFormIntegrator must support the 'd_energy' parameter in its
       ComputeFluxEnergy() method. */
   void SetAnisotropic(bool aniso = true) { anisotropic = aniso; }

   /** @brief Set the way the flux is averaged (smoothed) across elements.

       When @a fa is zero (default), averaging is performed globally. When @a fa
       is non-zero, the flux averaging is performed locally for each mesh
       attribute, i.e. the flux is not averaged across interfaces between
       different mesh attributes. */
   void SetFluxAveraging(int fa) { flux_averaging = fa; }

   /// Return the total error from the last error estimate.
   double GetTotalError() const { return total_error; }

   /// Get a Vector with all element errors.
   virtual const Vector &GetLocalErrors()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /** @brief Get an Array<int> with anisotropic flags for all mesh elements.
       Return an empty array when anisotropic estimates are not available or
       enabled. */
   virtual const Array<int> &GetAnisotropicFlags()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return aniso_flags;
   }

   /// Reset the error estimator.
   virtual void Reset() { current_sequence = -1; }

   /** @brief Destroy a ZienkiewiczZhuEstimator object. Destroys, if owned, the
       FiniteElementSpace, flux_space. */
   virtual ~ZienkiewiczZhuEstimator()
   {
      if (own_flux_fes) { delete flux_space; }
   }
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
   double total_error;

   BilinearFormIntegrator *integ; ///< Not owned.
   ParGridFunction *solution; ///< Not owned.

   ParFiniteElementSpace *flux_space; /**< @brief Ownership based on the flag
      own_flux_fes. Its Update() method is called automatically by this class
      when needed. */
   ParFiniteElementSpace *smooth_flux_space; /**< @brief Ownership based on the
      flag own_flux_fes. Its Update() method is called automatically by this
      class when needed.*/
   bool own_flux_fes; ///< Ownership flag for flux_space and smooth_flux_space.

   void Init(BilinearFormIntegrator &integ,
             ParGridFunction &sol,
             ParFiniteElementSpace *flux_fes,
             ParFiniteElementSpace *smooth_flux_fes)
   {
      current_sequence = -1;
      local_norm_p = 1;
      total_error = 0.0;
      this->integ = &integ;
      solution = &sol;
      flux_space = flux_fes;
      smooth_flux_space = smooth_flux_fes;
   }

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = solution->FESpace()->GetMesh()->GetSequence();
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
   { Init(integ, sol, flux_fes, smooth_flux_fes); own_flux_fes = true; }

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
   { Init(integ, sol, &flux_fes, &smooth_flux_fes); own_flux_fes = false; }

   /** @brief Set the exponent, p, of the Lp norm used for computing the local
       element errors. Default value is 1. */
   void SetLocalErrorNormP(int p) { local_norm_p = p; }

   /// Return the total error from the last error estimate.
   double GetTotalError() const { return total_error; }

   /// Get a Vector with all element errors.
   virtual const Vector &GetLocalErrors()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /// Reset the error estimator.
   virtual void Reset() { current_sequence = -1; }

   /** @brief Destroy a L2ZienkiewiczZhuEstimator object. Destroys, if owned,
       the FiniteElementSpace, flux_space. */
   virtual ~L2ZienkiewiczZhuEstimator()
   {
      if (own_flux_fes) { delete flux_space; delete smooth_flux_space; }
   }
};

#endif // MFEM_USE_MPI

} // namespace mfem

#endif // MFEM_ERROR_ESTIMATORS
