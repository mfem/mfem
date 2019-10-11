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

#include "fe_ray_intersect.hpp"

#include "../mesh/mesh.hpp"     // for mfem::Mesh

#include "../fem/intrules.hpp"  // for mfem::IntegrationPoint
#include "../fem/fe.hpp"        // for mfem::FiniteElement
#include "../fem/fespace.hpp"   // for mfem::FiniteElementSpace
#include "../fem/eltrans.hpp"   // for mfem::ElementTransformation

namespace mfem
{

static struct params_t
{
  int     maxNewtonIterations     = 16;
  int     maxLineSearchIterations = 10;
  double  c1  = 1.e-4;
  double  c2  = 0.9;
  double  tol = 1.e-12;
} solver_parameters ;

//------------------------------------------------------------------------------
//                   INTERNAL HELPER METHODS
//------------------------------------------------------------------------------
namespace
{

} /* end anonymous namespace */

//------------------------------------------------------------------------------
//                FE/RAY INTERSECTION IMPLEMENTATION
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
void fe_ray_setMaxNewtonIterations( int N )
{
  MFEM_ASSERT( (N >= 1), "Max newton iterations, N, must be N >= 1" );
  solver_parameters.maxNewtonIterations = N;
}

//------------------------------------------------------------------------------
void fe_ray_setMaxLineSearchIterations( int N )
{
  MFEM_ASSERT( (N >= 1), "Max  line search iterations, N, must be N >= 1" );
  solver_parameters.maxLineSearchIterations = N;
}

//------------------------------------------------------------------------------
void fe_ray_setTolerance( double TOL )
{
  solver_parameters.tol = TOL;
}

//------------------------------------------------------------------------------
void fe_ray_setWolfeCoefficients( double c1, double c2 )
{
  // sanity checks
  MFEM_ASSERT( (c1 > 0.0 && c1 < 1.0), "Wolfe coefficients must be in (0,1)" );
  MFEM_ASSERT( (c2 > 0.0 && c2 < 1.0), "Wolfe coefficients must be in (0,1)" );
  MFEM_ASSERT( (c2 > c1 ),
  "Curvature coefficient, c2, must be greater than the Armijo coefficient, c1");

  solver_parameters.c1 = c1;
  solver_parameters.c2 = c2;
}

//------------------------------------------------------------------------------
bool fe_ray_solve( const int elementId,
                   const Mesh* mesh,
                   const double* x0,
                   const double* n,
                   double* r,
                   double& t
                   )
{
  // sanity checks
  MFEM_ASSERT( mesh != nullptr, "supplied mfem::Mesh is null" );
  MFEM_ASSERT( ( (elementId >= 0) && (elementId < mesh->GetNE()) ),
               "supplied elementId is out-of-bounds" );
  MFEM_ASSERT( x0 != nullptr, "supplied ray origin, x0, is null" );
  MFEM_ASSERT( n != nullptr, "supplied ray normal, n, is null" );
  MFEM_ASSERT( r != nullptr, "supplied output buffer, r, is null" );

  // STEP 0: get space and topological dimension, ensure surface mesh input
  const int sdim  = mesh->SpaceDimension(); // ambient space dimension
  const int tdim  = mesh->Dimension();      // topological dim of mesh elements
  MFEM_VERIFY( (sdim == tdim+1), "fe_ray_solve() operates on a surface mesh!" );

  // STEP 0: Get the finite element space
  const FiniteElementSpace* fes = mesh->GetNodalFESpace();
  MFEM_ASSERT( fes != nullptr,
            "supplied mesh does not have an associated Finite Element space" );

  // STEP 1: Get the FE instance & tranform for the supplied elementId
  const FiniteElement* fe  = fes->GetFE( elementId );
  ElementTransformation* T = fes->GetElementTransformation( elementId );
  MFEM_ASSERT( fe != nullptr , "null FE instance" );
  MFEM_ASSERT( T != nullptr, "null ElementTransformation" );

  const int ndofs = fe->GetDof();           // number of DoFs on the FE element
  bool converged = false;
  // TODO: implement this
  return converged;
}

//------------------------------------------------------------------------------
bool fe_ray_intersects( const FiniteElement* fe,
                        const double* r,
                        const double& t)
{
  MFEM_ASSERT( fe != nullptr, "supplied FiniteElement is null!" );
  MFEM_ASSERT( r != nullptr, "supplied reference coordinates are null!" );

  const int refdim = fe->GetDim();

  IntegrationPoint ip;


  return false;
//  const double LTOL = 0.0 - solver_parameters.tol;
//  const double HTOL = 1.0 + solver_parameters.tol;

//  Geometry G;
//  IntegrationPoint ip;
//  ip.Set( r, );
//  fe->GetGeomType()
//  bool on_element = true;
//  for ( int i=0; on_element && ( i < ndims ); ++i )
//  {
//    on_element = on_element && ( (r[ i ] > LTOL) && (r[ i ] < HTOL) );
//  }
//
//  const double& t       = r[ ndims ];
//  const bool intersects = on_element && ( t > LTOL );
//
//  return intersects;
}

} /* end mfem namespace */

