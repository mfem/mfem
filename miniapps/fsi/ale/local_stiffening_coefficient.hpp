#pragma once

#include <mfem.hpp>
#include "dist_solver.hpp"

using namespace mfem;
using namespace common;

/**
 * @brief Abstract base class for local mesh stiffening coefficients in ALE formulations.
 *
 * This class provides functionality for determining mesh quality and potential remeshing
 * requirements based on various geometric and physical criteria. It serves as the base
 * for specific stiffening coefficient implementations.
 */
class LocalStiffeningCoefficient : public Coefficient
{
public:
   /**
    * @brief Constructor for LocalStiffeningCoefficient.
    *
    * @param coeff_ Pointer to the original coefficient (NOT OWNED)
    * @param fes_ Pointer to the parallel finite element space (NOT OWNED)
    * @param wall_attr_ Array of boundary attributes representing wall boundaries
    */
   LocalStiffeningCoefficient(ParFiniteElementSpace *fes_,
                              Array<int> &wall_attrs_);

   /**
    * @brief Destructor
    */
   ~LocalStiffeningCoefficient();

   /**
    * @brief Determines if remeshing is required based on a quality threshold.
    *
    * The base class implementation evaluates mesh quality using the skewness
    * infinity norm. Derived classes may override this method to use different
    * quality metrics.
    *
    * @param threshold Quality threshold value for remeshing decision
    * @return true if remeshing is required, false otherwise
    *
    * @note This method only determines if remeshing is needed; actual remeshing
    *       must be implemented separately and can be triggered based on the
    *       return value of this function.
    */
   virtual bool RequiresRemeshing(real_t threshold);

   /**
    * @brief Updates internal state of the coefficient.
    *
    * This pure virtual method must be implemented by derived classes to update
    * any internal data structures or computations when the mesh or solution changes.
    */
   virtual void Update() = 0;

   using Coefficient::Eval;

   /**
    * @brief Evaluates the stiffening coefficient at a given point.
    *
    * @param T Element transformation providing geometric information
    * @param ip Integration point where the coefficient is evaluated
    * @return Value of the stiffening coefficient
    */
   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip) = 0;

protected:
   ParMesh *pmesh = nullptr;   ///< Pointer to parallel mesh (NOT OWNED)
   Array<int> wall_attrs;       ///< Array of wall boundary attributes
   ParFiniteElementSpace *fes; ///< Pointer to finite element space (NOT OWNED)
   bool require_remeshing = false;
};

/**
 * @brief Jacobian-based mesh stiffening coefficient implementation.
 *
 * This class computes stiffening coefficient based on the Jacobian determinant.
 *
 * c = 1 / det(J)^stiffening_degree
 *
 * See:
 * Shamanskiy, Alexander, and Bernd Simeon. "Mesh moving techniques in fluid-structure interaction: robustness, accumulated distortion and computational efficiency." Computational Mechanics 67.2 (2021): 583-600.
 */
class JacobianBasedStiffeningCoefficient : public LocalStiffeningCoefficient
{
public:
   /**
    * @brief Constructor with stiffening degree parameter.
    *
    * @param fes_ Pointer to the parallel finite element space (NOT OWNED)
    * @param wall_attr_ Array of boundary attributes representing wall boundaries
    * @param stiffening_degree Degree of stiffening applied (default: 1.0)
    */
   JacobianBasedStiffeningCoefficient(ParFiniteElementSpace *fes_,
                                      Array<int> &wall_attr_, real_t stiffening_degree);

   /**
    * @brief Constructor with default stiffening degree.
    *
    * @param fes_ Pointer to the parallel finite element space (NOT OWNED)
    * @param wall_attr_ Array of boundary attributes representing wall boundaries
    */
   JacobianBasedStiffeningCoefficient(ParFiniteElementSpace *fes_,
                                      Array<int> &wall_attr_)
      : LocalStiffeningCoefficient(fes_, wall_attr_) {}

   /**
    * @brief Destructor
    */
   ~JacobianBasedStiffeningCoefficient();

   /// Inherit the base class remeshing functionality
   using LocalStiffeningCoefficient::RequiresRemeshing;

   /**
    * @brief Updates internal state (no-op for Jacobian-based implementation).
    *
    * This implementation requires no updates as Jacobian information is
    * computed on-demand during evaluation.
    */
   void Update() override {};

   using Coefficient::Eval;

   /**
    * @brief Evaluates the Jacobian-based stiffening coefficient.
    *
    * Computes stiffening based on the element's Jacobian determinant and
    * geometric quality measures.
    *
    * @param T Element transformation providing geometric information
    * @param ip Integration point where the coefficient is evaluated
    * @return Stiffening coefficient value
    */
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;

private:
   real_t stiffening_degree =
      1.0; ///< Degree of stiffening applied to distorted elements
};

/**
 * @brief Distance-based mesh stiffening coefficient implementation.
 *
 * This class computes stiffening coefficients based on the distance to wall
 * boundaries d.
 *
 * c = c(d)
 *
 * See:
 * Fabbri, Thomas, et al. "Design of a high fidelity fluid–structure interaction solver using LES on unstructured grid." Computers & Fluids 265 (2023): 105963.
 */
class DistanceBasedStiffeningCoefficient : public LocalStiffeningCoefficient
{
public:
   /**
    * @brief Constructor with full parameter specification.
    *
    * @param coeff_ Pointer to the coefficient for the stiffening calculation (NOT OWNED)
    * @param fes_ Pointer to the parallel finite element space (NOT OWNED)
    * @param wall_attr_ Array of boundary attributes representing wall boundaries
    * @param distance_solver_type Type of distance solver algorithm (default: 2)
    * @param filter_ls Whether to apply filtering to the level set function (default: false)
    */
   DistanceBasedStiffeningCoefficient(ParFiniteElementSpace *fes_,
                                      Array<int> &wall_attr_, int distance_solver_type = 0, bool filter_ls = false);

   /**
    * @brief Constructor with default parameters.
    *
    * @param fes_ Pointer to the parallel finite element space (NOT OWNED)
    * @param wall_attr_ Array of boundary attributes representing wall boundaries
    */
   DistanceBasedStiffeningCoefficient(ParFiniteElementSpace *fes_,
                                      Array<int> &wall_attr_)
      : DistanceBasedStiffeningCoefficient(fes_, wall_attr_, 2, false) {}

   /**
    * @brief Destructor
    */
   ~DistanceBasedStiffeningCoefficient();

   /// Inherit the base class remeshing functionality
   using LocalStiffeningCoefficient::RequiresRemeshing;

   /**
    * @brief Return computed distance GridFunction.
    */
   ParGridFunction *GetDistanceGf() { return &distance_gf; }

   /**
    * @brief Returns the level set GridFunction.
    */
   ParGridFunction *GetLevelSetGf() { return &ls_gf; }

   /**
    * @brief Sets the parameters for the distance-based stiffening coefficient.
    *
    * @param Rmin Minimum distance to wall boundaries
    * @param Rmax Maximum distance to wall boundaries
    * @param alpha Stiffening exponent
    * @param Amax Maximum stiffening coefficient value (near wall)
    * @param Amin Minimum stiffening coefficient value (far from wall)
    * @param stiffening_degree Power applied to the sigmoid result (default: 1.0)
    *
    * The coefficient ranges from Amax at Rmin to Amin at Rmax, and decreases sigmoidally:
    *
    * A(R) = pow( Amin + (Amax - Amin) / (1 + exp(alpha * (R - Rmid) / (Rmax - Rmin))), stiffening_degree )
    *
    * where Rmid is the middle of the transition zone Rmid = (Rmin + Rmax) / 2
    */
   void SetParameters(real_t Rmin, real_t Rmax, real_t alpha, real_t Amax,
                      real_t Amin, real_t stiffening_degree = 1.0);

   /**
    * @brief Updates the distance field computation.
    *
    * Recomputes the distance field to wall boundaries when the mesh geometry
    * or boundary conditions change.
    */
   void Update() override;

   using Coefficient::Eval;

   /**
    * @brief Evaluates the distance-based stiffening coefficient.
    *
    * Computes stiffening based on the distance to the nearest wall boundary,
    * with higher stiffening applied closer to walls.
    *
    * @param T Element transformation providing geometric information
    * @param ip Integration point where the coefficient is evaluated
    * @return Stiffening coefficient value based on wall distance
    */
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;

private:
   bool filter_ls;                   ///< Flag indicating whether to filter the level set function
   ParGridFunction
   ls_gf;            ///< Level set grid function for boundary representation
   ParGridFunction distance_gf;      ///< Grid function storing distance to walls
   GridFunctionCoefficient
   ls_coeff; ///< Coefficient wrapper for level set function
   GridFunctionCoefficient
   dist_coeff; ///< Coefficient wrapper for the distance (handles LOR meshes)
   DistanceSolver *distance_solver;  ///< Distance solver object (owned)
   real_t Rmin;
   real_t Rmax;
   real_t Rmid;
   real_t alpha;
   real_t Amax;
   real_t Amin;
   real_t stiffening_degree =
      1.0; ///< Power applied to the sigmoid stiffening value
};
