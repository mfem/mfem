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

#include "fe_ray_intersect.hpp"

#include "../mesh/mesh.hpp"     // for mfem::Mesh

#include "../fem/geom.hpp"      // for Geometry
#include "../fem/intrules.hpp"  // for IntegrationPoint & IntegrationRule
#include "../fem/fe.hpp"        // for FiniteElement
#include "../fem/fespace.hpp"   // for FiniteElementSpace
#include "../fem/eltrans.hpp"   // for ElementTransformation

namespace mfem
{

static struct params_t
{
   int     maxNewtonIterations     = 16;
   int     maxLineSearchIterations = 5;
   double  c1  = 10.e-4;
   double  c2  = 0.9;
   double  tol = 1.e-12;
} solver_parameters ;

//------------------------------------------------------------------------------
//                   INTERNAL HELPER METHODS
//------------------------------------------------------------------------------
namespace
{

template < typename T >
inline T clamp_value( T val, T min, T max )
{
   return ( (val < min) ? min : (val > max) ? max : val );
}

//------------------------------------------------------------------------------
template < typename T >
inline T clamp_min( T val, T min )
{
   return ( (val < min) ? min : val );
}

//------------------------------------------------------------------------------
void get_physical_nodes( const FiniteElement* fe,
                         ElementTransformation* T,
                         DenseMatrix& phys_nodes )
{
   MFEM_ASSERT( fe != nullptr, "supplied finite element is null" );
   MFEM_ASSERT( T != nullptr, "supplied element transformation is null" );

   const IntegrationRule& ref_nodes = fe->GetNodes();
   const int ndofs = ref_nodes.Size();
   const int ndims = T->GetSpaceDim();
   phys_nodes.SetSize( ndims, ndofs );
   T->Transform( ref_nodes, phys_nodes );
}

//------------------------------------------------------------------------------
bool check_convergence( const double* dx, int ndims, double TOL )
{
   double l1norm = 0.0;
   for ( int idim=0; idim < ndims; ++idim )
   {
      l1norm += abs( dx[idim] );
   }
   return ( l1norm < TOL );
}

//------------------------------------------------------------------------------
void update_jacobian( const double* sk, ElementTransformation* T ,
                      DenseMatrix& J )
{
   MFEM_ASSERT( sk != nullptr, "supplied solution vector, sk, is null!" );
   MFEM_ASSERT( T != nullptr, "supplied ElementTransformation object is null!" );
   MFEM_ASSERT( J.IsSquare(), "Jacobian must be a square matrix!" );

   IntegrationPoint ip( sk, T->GetDimension() );
   T->SetIntPoint( &ip );
   const DenseMatrix& Je = T->Jacobian();
   MFEM_ASSERT( Je.NumRows()==J.NumRows(), "Je.NumRows() != J.NumRows()" )

   for ( int icol=0; icol < Je.NumCols(); ++icol )
   {
      J.SetCol( icol, Je.GetColumn( icol ) );
   }
}

//------------------------------------------------------------------------------
void dFxda( const FiniteElement* fe,  ///!< Fe instance (in)
            const DenseMatrix& X,     ///!< FE phys. nodes (in)
            const double* sk,         ///!< current ref. coords (in)
            const double* n,          ///!< ray normal
            double* dfxda,            ///!< deriv. of fx w.r.t. alpha (out)
            double scalar=1.0         ///!< multiplier (optional)
          )
{
   // sanity checks
   MFEM_ASSERT( fe    != nullptr, "supplied FE instance is null!" );
   MFEM_ASSERT( sk    != nullptr, "supplied solution vector, sk, is null!" );
   MFEM_ASSERT( n     != nullptr, "pointer to ray normal is null!" );
   MFEM_ASSERT( dfxda != nullptr, "pointer to output vector is null!" );
   MFEM_ASSERT( X.NumRows()== (fe->GetDim()+1),
                "supplied FE is not a surface element!" );
   MFEM_ASSERT( X.NumCols()==fe->GetDof(),
                "supplied physical nodes matrix doesn't match FE number of dofs" );

   const int ndofs = fe->GetDof();
   const int ndims = X.NumRows();

   // compute shape functions at given point
   IntegrationPoint ip( sk, fe->GetDim() );
   Vector N( fe->GetDof() );
   fe->CalcShape( ip, N );

   const double& t = sk[ fe->GetDim() ];

   for ( int idim=0; idim < ndims; ++idim )
   {
      dfxda[ idim ] = 0.0;
      for ( int idof=0; idof < ndofs; ++idof )
      {
         dfxda[ idim ] += ( N[ idof ] * X(idim,idof) );
      }

      dfxda[ idim ] -= ( t*n[idim] );

   } // END for all dimensions

}

//------------------------------------------------------------------------------
void Fx( const FiniteElement* fe, ///!< FE instance (in)
         const DenseMatrix& X,    ///!< FE phys. nodes (in)
         const double* sk,        ///!< current ref. coords (in)
         const double* x0,        ///!< origin of the ray (in)
         const double* n,         ///!< ray normal (in)
         double* fx,               ///!< fx, computed (out)
         double scalar=1.0         ///!< multiplier (optional)
       )
{
   // sanity checks
   MFEM_ASSERT( fe != nullptr, "supplied FE instance is null!" );
   MFEM_ASSERT( sk != nullptr, "supplied solution vector, sk, is null!" );
   MFEM_ASSERT( x0 != nullptr, "pointer to ray origin buffer is null!" );
   MFEM_ASSERT( n  != nullptr, "pointer to ray normal is null!" );
   MFEM_ASSERT( fx != nullptr, "pointer to output vector is null!" );
   MFEM_ASSERT( X.NumRows()== (fe->GetDim()+1),
                "supplied FE is not a surface element!" );
   MFEM_ASSERT( X.NumCols()==fe->GetDof(),
                "supplied physical nodes matrix doesn't match FE number of dofs" );

   const int ndofs = fe->GetDof();
   const int ndims = X.NumRows();

   // compute shape functions at given point
   IntegrationPoint ip( sk, fe->GetDim() );
   Vector N( fe->GetDof() );
   fe->CalcShape( ip, N );

   const double& t = sk[ fe->GetDim() ];

   for ( int idim=0; idim < ndims; ++idim )
   {
      fx[ idim ] = 0.0;
      for ( int idof=0; idof < ndofs; ++idof )
      {
         fx[ idim ] += ( N[ idof ] * X(idim,idof) );
      }

      fx[ idim ] -= x0[ idim ];
      fx[ idim ] -= ( t*n[idim] );
      fx[ idim ] *= scalar;
   } // END for all dimensions

}

//------------------------------------------------------------------------------
bool wolfe_conditions( double ak,
                       const double* fxk,
                       const double* dfxk,
                       const double* fxn,
                       const double* dfxn,
                       int ndims )
{
   MFEM_ASSERT( (fxk != nullptr), "fxk is null" );
   MFEM_ASSERT( (dfxk != nullptr), "dfxk is null" );
   MFEM_ASSERT( (fxn != nullptr), "fxn is null" );
   MFEM_ASSERT( (dfxn != nullptr), "dfxk is null" );
   MFEM_ASSERT( (ndims==2 || ndims==3), "ndims must be 2 or 3" );

   // Wolfe coefficients used for sufficient decrease and curvature conditions
   const double& c1 = solver_parameters.c1;
   const double& c2 = solver_parameters.c2;

   double c1ak = c1*ak;

   bool armijo = true;
   bool curvature = true;
   for ( int idim=0; idim < ndims; ++idim )
   {
      armijo = armijo && ( fxn[ idim ] <= ( fxk[ idim ]+c1ak*dfxk[ idim ] ) );
      curvature = curvature && ( abs(dfxn[ idim ]) <= ( c2*abs(dfxk[ idim ]) ) );
   }

   return ( armijo && curvature );
}

//------------------------------------------------------------------------------
double backtracking_linesearch( const FiniteElement* fe, ///!< FE instance (in)
                                const DenseMatrix& X,    ///!< FE nodes (in)
                                const double* sk,        ///!< ref. coords (in)
                                const double* pk,        ///!< direction (in)
                                const double* x0,        ///!< ray origin (in)
                                const double* n          ///!< ray normal (in)
                              )
{
   // sanity checks
   // sanity checks
   MFEM_ASSERT( fe != nullptr, "supplied FE instance is null!" );
   MFEM_ASSERT( sk != nullptr, "supplied solution vector, sk, is null!" );
   MFEM_ASSERT( x0 != nullptr, "pointer to ray origin buffer is null!" );
   MFEM_ASSERT( n  != nullptr, "pointer to ray normal is null!" );
   MFEM_ASSERT( X.NumRows()== (fe->GetDim()+1),
                "supplied FE is not a surface element!" );
   MFEM_ASSERT( X.NumCols()==fe->GetDof(),
                "supplied physical nodes matrix doesn't match FE number of dofs" );

   const int ndims = X.NumRows();

   const int maxLineSearchIters = solver_parameters.maxLineSearchIterations;

   constexpr double alphalimit  = 0.1; /* limiter on smallest step size */
   constexpr double beta        = 0.5; /* shrink factor */

   // evaluate Fx and its derivative at sk, i.e., at alpha = 1
   double fxk[3];
   double dfxk[3];
   Fx( fe, X, sk, x0, n, fxk );
   dFxda( fe, X, sk, n, dfxk );

   // arrays to store Fx and derivatives at new position `sk +  alpha * pk`
   double fxn[3];
   double dfxn[3];
   double skn[3];

   double alpha  = 1.0;
   bool foundmin = false;
   for ( int iter=0; !foundmin && (iter < maxLineSearchIters); ++iter )
   {
      // Evaluate sk + alpha * pk at current alpha
      for ( int idim=0; idim < ndims; ++idim )
      {
         skn[ idim ] = sk[ idim ] + (alpha*pk[ idim ] );
      }

      // Let F(a) = Fx(sk+alpha*pk), evaluate f(a) and derivatives at new position
      Fx( fe, X, skn, x0, n, fxn );
      dFxda( fe, X, skn, n, dfxn );

      // check sufficient decrease (a.k.a, Armijo) and curvature conditions
      foundmin = wolfe_conditions( alpha, fxk, dfxk, fxn, dfxn, ndims );

      if ( !foundmin )
      {
         // shrink stepsize
         alpha *= beta;
      }

   } // END for all linesearch iterations

   return std::max( alpha, alphalimit );
}

} /* end anonymous namespace */

//------------------------------------------------------------------------------
//                FE/RAY INTERSECTION IMPLEMENTATION
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
void fe_ray_setMaxNewtonIterations( int N )
{
   MFEM_ASSERT( (N >= 1), "Max Newton iterations, N, must be N >= 1" );
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
   MFEM_VERIFY( sdim==2 || sdim==3, "supplied mesh must be either 2D or 3D" );
   MFEM_VERIFY( (sdim == tdim+1), "fe_ray_solve() operates on a surface mesh!" );

   // STEP 1: Get the finite element space
   const FiniteElementSpace* fes = mesh->GetNodalFESpace();
   MFEM_ASSERT( fes != nullptr,
                "supplied mesh does not have an associated Finite Element space" );

   // STEP 2: Get the FE instance & transform for the supplied elementId
   const FiniteElement* fe  = fes->GetFE( elementId );
   ElementTransformation* T = fes->GetElementTransformation( elementId );
   MFEM_ASSERT( fe != nullptr , "null FE instance" );
   MFEM_ASSERT( T != nullptr, "null ElementTransformation" );

   // STEP 3: Get the physical nodes (i.e., dofs) of the element
   DenseMatrix phys_nodes;
   get_physical_nodes( fe, T, phys_nodes );

   // STEP 4: create and initialize the Jacobian matrix. Note, the last column
   // of the Jacobian is set to the ray normal and is constant.
   DenseMatrix J( sdim );
   for ( int idim=0; idim < sdim; ++idim )
   {
      J( idim, tdim ) = -( n[idim] );
   }

   // STEP 5: initial guess
   double sk[ 3 ]; // solution vector, updated at each Newton step

   Geometry G;
   const IntegrationPoint& ip = G.GetCenter( fe->GetGeomType() );
   ip.Get( sk, tdim );
   sk[ tdim ] = 0.0; // can bracket this within [t1,t2] if necessary.


   // STEP 6: Newton iteration variables, updated at each Newton step
   double pk[ 3 ]; // Newton direction

   // STEP 7: start Newton iteration
   constexpr double NEGATE = -1.0;
   bool converged  = false;
   const int maxNewtonIters = solver_parameters.maxNewtonIterations;
   for (int iter=0; !converged && iter < maxNewtonIters; ++iter )
   {
      // update lhs Jacobian
      update_jacobian( sk, T, J );

      // update rhs, i.e., -fx
      Fx( fe, phys_nodes, sk, x0, n, pk, NEGATE );

      // Newton step: "J*pk = -fx", solve for pk
      // NOTE: on input, pk==-fx, and on output it store the delta
      int rc = LinearSolve( J, pk );
      if ( rc != 0 )
      {
         return false;
      }

      // check convergence
      converged = check_convergence( pk, sdim, solver_parameters.tol );

      // backtracking line-search
      double alpha = 1.0;
      if ( !converged )
      {
         alpha = backtracking_linesearch( fe, phys_nodes, sk, pk, x0, n );
      }

      // apply improvements
      for ( int idim=0; idim < sdim; ++idim )
      {
         sk[ idim ] += ( alpha * pk[ idim ] );
      }

      // bracket solution: restrict on the element and along the ray direction
      for ( int idim=0; idim < tdim; ++idim )
      {
         clamp_value< double >( sk[ idim ], 0.0, 1.0 );
      }
      clamp_min< double >( sk[ tdim ], 0.0 );

   } // END for all Newton iterations

   if ( converged )
   {
      memcpy( r, sk, sizeof(double)*tdim );
      t = sk[ tdim ];
   }

   return converged;
}

//------------------------------------------------------------------------------
bool fe_ray_intersects( const FiniteElement* fe,
                        const double* r,
                        const double& t)
{
   MFEM_ASSERT( fe != nullptr, "supplied FiniteElement is null!" );
   MFEM_ASSERT( r != nullptr, "supplied reference coordinates are null!" );

   const double& TOL = solver_parameters.tol;
   const double LTOL = 0.0 - TOL;

   IntegrationPoint ip( r, fe->GetDim() );
   const bool on_element = Geometry::CheckPoint( fe->GetGeomType(), ip, TOL );
   const bool intersects = on_element && ( t > LTOL );
   return intersects;
}

} /* end mfem namespace */

