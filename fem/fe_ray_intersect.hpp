// Copyright (c) 2019, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


#ifndef MFEM_FE_RAY_INTERSECT_HPP_
#define MFEM_FE_RAY_INTERSECT_HPP_

namespace mfem
{

// Forward declarations
class Mesh;
class FiniteElement;

/**
 * @brief Sets the maximum number of newton iterations. Default set to 16.
 * @param [in] N the maximum number of newton iterations.
 *
 * @pre N >= 1
 */
void fe_ray_setMaxNewtonIterations( int N );

/**
 * @brief Sets the max number of iterations for the backtracking line search.
 * @param [in] N maxnumber iterations for the backtracking line search.
 */
void fe_ray_setMaxLineSearchIterations( int N );

/**
 * @brief Sets floating-point tolerance. Default is 1.e-12.
 * @param [in] TOL user-supplied tolerance.
 */
void fe_ray_setTolerance( double TOL );

/**
 * @brief Sets the coefficients used for the Wolfe conditions in the
 *  backtracking line search.
 *
 * @param [in] c1 coefficient for the Armijo condition
 * @param [in] c2 coefficient for the curvature condition
 *
 * @pre 0.0 < c1 < c2 < 1.0
 */
void fe_ray_setWolfeCoefficients( double c1, double c2 );

/**
 * @brief Checks if a given ray intersects with the specified surface mesh
 *  element and computes the parametric coordinates of the intersection point,
 *  with respect to the surface element and the ray.
 *
 * @param [in]  elementId the ID of the surface element in the given mesh.
 * @param [in]  mesh pointer to the associated surface mesh instance.
 * @param [in]  x0 array consisting of the ray's source point
 * @param [in]  n array consisting of the ray's normal
 * @param [out] r reference coordinates of intersection point w.r.t. the element
 * @param [out] t value at intersection w.r.t. the ray, R(t) = x0 + tn.
 *
 * @return status true if the solver converges, else false.
 *
 * @note The supplied arrays, x0 & r,  are expected to have length equal to at
 *  least the ambient space dimension, e.g., `mesh->SpaceDimension()`
 *
 *
 * @pre mesh != nullptr
 * @pre mesh->SpaceDimension() == mesh->Dimension()+1, i.e., surface mesh input
 * @pre 0 >= elementId < mesh->GetNE()
 * @pre x0 != nullptr
 * @pre n != nullptr
 * @pre r != nullptr
 */
bool fe_ray_solve( const int elementId, const Mesh* mesh,
                   const double* x0, const double* n,
                   double* r,
                   double& t );

/**
 * @brief Given the parametric coordinates of the intersection point, r, this
 *  method checks if the point intersect with the surface FE element.
 *
 * @param [in] r
 * @param [in] t
 * @param [in] ndims the physical problem dimension.
 *
 * @return status true if the ray intersects, otherwise false.
 *
 */
bool fe_ray_intersects( const FiniteElement* fe,
                        const double* r,
                        const double& t  );

} /* end mfem namespace */


#endif /* MFEM_FE_RAY_INTERSECT_HPP_ */
