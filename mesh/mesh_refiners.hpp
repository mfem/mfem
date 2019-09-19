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


#ifndef REFINERS_HPP
#define REFINERS_HPP

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "mesh.hpp"
#include "mesh_operators.hpp"
#include "../fem/estimators.hpp"



namespace mfem
{
/** @brief Mesh refinement using Maximum Marking

  This class uses the given ErrorEstimator to estimate local element errors
  and then marks for refinement all elements i such that loc_err_i > threshold.
  The threshold is computed as
  \code
  threshold = gamma * Max_{i in Mesh} { local_error_i }
  \endcode
  where gamma is a configurable parameter
  */
class MaximumMarkingRefiner : public MeshOperator
{
protected:
   ErrorEstimator &estimator;

   long   max_elements;

   double gamma;
   double threshold;
   long num_marked_elements;

   Array<Refinement> marked_elements;
   long current_sequence;

   int non_conforming;
   int nc_limit;

   /** @brief Apply the operator to the mesh.
     @return STOP if a stopping criterion is satisfied or no elements were
     marked for refinement; REFINED + CONTINUE otherwise. */
   virtual int ApplyImpl(Mesh &mesh) override;

public:
   /// Construct a MaximumMarkingRefiner using the given ErrorEstimator.
   MaximumMarkingRefiner(ErrorEstimator &est);

   // default destructor (virtual)

   void SetMaxElements(long max_elem) { max_elements = max_elem; }

   /// Use nonconforming refinement, if possible (triangles, quads, hexes).
   void PreferNonconformingRefinement() { non_conforming = 1; }

   /** @brief Use conforming refinement, if possible (triangles, tetrahedra)
     -- this is the default. */
   void PreferConformingRefinement() { non_conforming = -1; }

   /** @brief Set the maximum ratio of refinement levels of adjacent elements
     (0 = unlimited). */
   void SetNCLimit(int nc_limit)
   {
      MFEM_ASSERT(nc_limit >= 0, "Invalid NC limit");
      this->nc_limit = nc_limit;
   }

   /// Get the number of marked elements in the last Apply() call.
   long GetNumMarkedElements() const { return num_marked_elements; }

   /// Set the confiugrable parameter gamma
   void SetGamma( double theta ) { gamma = theta; }
   /// Get the configured gamma
   double GetGamma() const { return gamma; }

   /// Get the threshold used in the last Apply() call.
   double GetThreshold() const { return threshold; }

   /// Reset the associated estimator.
   virtual void Reset() override
   {
      estimator.Reset();
      current_sequence = -1;
      num_marked_elements = 0;
   };
};

/** @brief Mesh refinement using Dörfler Marking

  This class uses the given ErrorEstimator to estimate local element errors
  and then marks for refinement a set M such that
  \code
  gamma Sum_{i in M} local_error_i^2 <= Sum_{i in Mesh} local_error_i^2
  \endcode
  where gamma is a configurable parameter.
  Currently we use the linear complexity QuickMark algorithm of Pfeiler and Praetorius ( arXiv:1907.13078 ).
  */
class DoerflerMarkingRefiner : public MeshOperator
{
public:
   using Iterator_t = std::vector<double>::iterator;
protected:
   ErrorEstimator &estimator;

   long   max_elements;

   double threshold;
   double gamma;
   long num_marked_elements;

   Array<Refinement> marked_elements;
   long current_sequence;

   int non_conforming;
   int nc_limit;


   /** @brief Compute the threshold using the linear complexity QuickMark algorithm of Pfeiler & Praetorius ( arXiv:1907.13078 )
    * @return Threshold x* such that the Dörfler set is precisely the set err > x*
    */
   const double compute_threshold ( const mfem::Vector & eta, double theta );
   const double xStarKernel( Iterator_t subX_begin, Iterator_t subX_end,
                             double goal );


   /** @brief Apply the operator to the mesh.
     @return STOP if a stopping criterion is satisfied or no elements were
     marked for refinement; REFINED + CONTINUE otherwise. */
   virtual int ApplyImpl(Mesh &mesh) override;

public:
   /// Construct a ThresholdRefiner using the given ErrorEstimator.
   DoerflerMarkingRefiner(ErrorEstimator &est);

   // default destructor (virtual)

   void SetMaxElements(long max_elem) { max_elements = max_elem; }

   /// Use nonconforming refinement, if possible (triangles, quads, hexes).
   void PreferNonconformingRefinement() { non_conforming = 1; }

   /** @brief Use conforming refinement, if possible (triangles, tetrahedra)
     -- this is the default. */
   void PreferConformingRefinement() { non_conforming = -1; }

   /** @brief Set the maximum ratio of refinement levels of adjacent elements
     (0 = unlimited). */
   void SetNCLimit(int nc_limit)
   {
      MFEM_ASSERT(nc_limit >= 0, "Invalid NC limit");
      this->nc_limit = nc_limit;
   }

   /// Get the number of marked elements in the last Apply() call.
   long GetNumMarkedElements() const { return num_marked_elements; }

   /// Set the confiugrable parameter gamma
   void SetGamma( double theta ) { gamma = theta; }
   /// Get the configured gamma
   double GetGamma() const { return gamma; }



   /// Get the threshold used in the last Apply() call.
   double GetThreshold() const { return threshold; }

   /// Reset the associated estimator.
   virtual void Reset() override
   {
      estimator.Reset();
      current_sequence = -1;
      num_marked_elements = 0;
   };
};

}
#endif // REFINERS_HPP

