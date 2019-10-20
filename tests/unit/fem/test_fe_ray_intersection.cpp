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

#include "mfem.hpp"
#include "catch.hpp"

#include <fstream>
#include <sstream>

using namespace mfem;

// NOTE: un-comment to generate VTK files for debugging
//#define FE_RAY_VTK_DEBUG

//------------------------------------------------------------------------------
// HELPER METHODS
//------------------------------------------------------------------------------
namespace
{

//------------------------------------------------------------------------------
void dump_ray( const double* x0, const double* n, int ndims,
               const std::string file )
{
#ifdef FE_RAY_VTK_DEBUG
   std::ostringstream oss;
   oss << file << "_" << ndims << "d.vtk";

   std::ofstream ofs( oss.str().c_str() );

   ofs << "# vtk DataFile Version 3.0\n";
   ofs << "Ray Data\n";
   ofs << "ASCII\n";
   ofs << "DATASET UNSTRUCTURED_GRID\n";

   ofs << "POINTS 2 double\n";
   ofs << x0[ 0 ] << " ";
   ofs << x0[ 1 ] << " ";
   ofs << ( (ndims==3)? x0[ 2 ] : 0.0 ) << std::endl;

   constexpr double t = 0.1;
   ofs << ( x0[ 0 ] + t*n[ 0 ] ) << " ";
   ofs << ( x0[ 1 ] + t*n[ 1 ] ) << " ";
   ofs << ( (ndims==3)? (x0[ 2 ] + t*n[ 2 ]) : 0.0 ) << std::endl;

   ofs << "CELLS 1 3\n";
   ofs << "2 0 1\n";

   ofs << "CELL_TYPES 1\n";
   ofs << "3\n";

   ofs << "CELL_DATA 1\n";
   ofs << "VECTORS normal double\n";
   ofs << n[ 0 ] << " " << n[ 1 ] << " ";
   ofs << ( (ndims==3)? n[ 2 ] : 0.0 ) << std::endl;

   ofs.close();
#endif
}

//------------------------------------------------------------------------------
void dump_line( const double* x0, const double* p, int ndims,
                const std::string file )
{
#ifdef FE_RAY_VTK_DEBUG
   std::ostringstream oss;
   oss << file << "_" << ndims << "d.vtk";

   std::ofstream ofs( oss.str().c_str() );

   ofs << "# vtk DataFile Version 3.0\n";
   ofs << "Intersection Line\n";
   ofs << "ASCII\n";
   ofs << "DATASET UNSTRUCTURED_GRID\n";

   ofs << "POINTS 2 double\n";
   ofs << x0[ 0 ] << " " << x0[ 1 ] << " ";
   ofs << ( (ndims==3)? x0[ 2 ] : 0.0 ) << std::endl;

   ofs << p[ 0 ]  << " " << p[ 1 ] << " ";
   ofs << ( (ndims==3)? p[ 2 ] : 0.0 )  << std::endl;

   ofs << "CELLS 1 3\n";
   ofs << "2 0 1\n";
   ofs << "CELL_TYPES 1\n";
   ofs << "3\n";

   ofs.close();
#endif
}

//------------------------------------------------------------------------------
void linspace( double x0, double x1, double* v, int N )
{
   MFEM_ASSERT( N > 0, "N > 0" );

   const double h = (x1-x0) / static_cast< double >( N-1 );

   for ( int i=0; i < N; ++i )
   {
      v[ i ] = x0 + i*h;
   }

}

//------------------------------------------------------------------------------
void mesh_trans2d( const mfem::Vector& x, mfem::Vector& p )
{
   p.SetSize( 2 );

   constexpr double A     = 2.0;
   constexpr double L     = 15.0;
   constexpr double INV_L = 1. / L;
   constexpr double K     = 2*M_PI * INV_L;
   constexpr double T     = 1.0;
   constexpr double vp    = T * INV_L;
   constexpr double f     = vp * INV_L;
   constexpr double w     = 2 * M_PI * f;
   constexpr double t     = 0.1;

   const double u = K*x[0] - w*t;
   const double v = K*x[0] + w*t;

   p[ 0 ]  = x[ 0 ];
   p[ 1 ] += A * std::sin( u ) + A * std::cos( v );
}

//------------------------------------------------------------------------------
void mesh_trans3d( const mfem::Vector& x, mfem::Vector& p )
{
   p.SetSize( 3 );

   constexpr double A     = 2.0;
   constexpr double L     = 15.0;
   constexpr double INV_L = 1. / L;
   constexpr double K     = 2*M_PI * INV_L;
   constexpr double T     = 1.0;
   constexpr double vp    = T * INV_L;
   constexpr double f     = vp * INV_L;
   constexpr double w     = 2 * M_PI * f;
   constexpr double t     = 0.1;

   const double u = K*x[0] - w*t;
   const double v = K*x[0] + w*t;
   p[ 0 ] = x[ 0 ];
   p[ 1 ] = x[ 1 ];
   p[ 2 ] += A * std::sin( u ) + A * std::cos( v );
}

//------------------------------------------------------------------------------
Mesh* get_mesh3d( int refinement_level, int order )
{
   MFEM_ASSERT( order >= 1, "order >= 1" );

   constexpr int SDIM = 3;
   constexpr int TDIM = SDIM - 1;
   constexpr int NV   = 4;
   constexpr int NE   = 1;

   Mesh* mesh = new Mesh( TDIM, NV, NE, 0, SDIM );

   const double v[ NV ][ SDIM ] =
   {
      {-5.0, 0.0, 0.0 }, // V0
      { 5.0, 0.0, 0.0 }, // V1
      { 5.0, 5.0, 0.0 }, // V2
      {-5.0, 5.0, 0.0 }, // V3
   };

   const int conn[ NE ][ 4 ] =
   {
      {0, 1, 2, 3}
   };

   // add the vertices
   for ( int i=0; i < NV; ++i )
   {
      mesh->AddVertex( v[ i ] );
   }

   // add the mesh elements
   for ( int i=0; i < NE; ++i )
   {
      mesh->AddQuad( conn[ i ], (i+1) );
   }

   for ( int ilevel=0; ilevel < refinement_level; ++ilevel )
   {
      mesh->UniformRefinement();
   }

   constexpr int GENERATE_EDGES = 1;
   constexpr int REFINE = 1;
   constexpr bool FIX_ORIENTATION = true;

   mesh->FinalizeQuadMesh( GENERATE_EDGES, REFINE, FIX_ORIENTATION );

   mesh->Transform( mesh_trans3d );

   mesh->SetNodalFESpace(
      new FiniteElementSpace( mesh, new H1_FECollection(order,TDIM), SDIM ) );

   return mesh;
}


//------------------------------------------------------------------------------
Mesh* get_mesh2d( int refinement_level, int order )
{
   MFEM_ASSERT( order >= 0, "order >= 1" );

   constexpr int SDIM = 2;
   constexpr int TDIM = SDIM - 1;

   constexpr double x0 = -5.0;
   constexpr double x1 =  5.0;

   const int nnodes = refinement_level + 2;
   const int nelems = nnodes - 1;
   double* v = new double[ nnodes ];
   linspace( x0, x1, v, nnodes );

   Mesh* mesh = new Mesh( TDIM, nnodes, nelems, 0, SDIM );

   for ( int i=0; i < nnodes; ++i )
   {
      double vertex[] = { v[ i ], 0.0 };
      mesh->AddVertex( vertex );
   }

   delete [] v;

   for ( int i=0; i < nelems; ++i )
   {
      int con[] = { i, i+1 };
      mesh->AddSegment( con, (i+1) );
   }

   constexpr int GENERATE_EDGES = 1;
   constexpr int REFINE = 1;
   constexpr bool FIX_ORIENTATION = true;

   mesh->FinalizeMesh( REFINE, FIX_ORIENTATION );

   mesh->Transform( mesh_trans2d );

   mesh->SetNodalFESpace(
      new FiniteElementSpace( mesh, new H1_FECollection(order,TDIM), SDIM ) );

   return mesh;
}

//------------------------------------------------------------------------------
template < int NDIMS >
Mesh* get_mesh( int refinement_level, int order );

template < >
Mesh* get_mesh< 2 >( int rl, int order ) { return get_mesh2d(rl,order); };

template < >
Mesh* get_mesh< 3 >( int rl, int order ) { return get_mesh3d(rl,order); };

//------------------------------------------------------------------------------
template < int NDIMS >
bool check_fe_ray( int fe_order,
                   const double* x0,
                   const double* n,
                   const int& element_found=-1,
                   const double* ip_expected=nullptr,
                   const double* r_expected=nullptr,
                   const double& t_expected=-1.0
                 )
{
   // STEP 0: create mesh
   constexpr int REFINEMENT_LEVELS = 5;
   Mesh* m = get_mesh< NDIMS >( REFINEMENT_LEVELS, fe_order );
   REQUIRE( m != nullptr );
   REQUIRE( m->GetNE() > 0 );
   REQUIRE( m->SpaceDimension()==NDIMS );

   dump_ray( x0, n, NDIMS, "test_ray" );

#ifdef FE_RAY_VTK_DEBUG
   if ( fe_order <= 2 )
   {
      std::ostringstream oss;
      oss << "mfem_mesh_" << NDIMS << "d_p" << fe_order << ".vtk";

      std::ofstream ofs( oss.str().c_str() );
      m->PrintVTK( ofs );
      ofs.close();
   }
#endif

   bool found = false;
   const int nelems = m->GetNE();
   for ( int ielem=0; ielem < nelems; ++ielem )
   {
      double ip[ 3];
      double r[ 3 ];
      double t;

      const FiniteElement* fe = m->GetNodalFESpace()->GetFE( ielem );
      bool converged = fe_ray_solve( ielem, m, x0, n, r, t );
      if ( converged && fe_ray_intersects( fe, r, t ) )
      {
         found = true;

         // ensure optional arguments are supplied
         REQUIRE( element_found != -1 );
         REQUIRE( ip_expected != nullptr );
         REQUIRE( r_expected != nullptr );
         REQUIRE( t_expected >= 0.0 );

         // check expected
         REQUIRE( element_found == ielem );
         REQUIRE( t == Approx(t_expected) );
         for ( int idim=0; idim < NDIMS-1; ++idim )
         {
            REQUIRE( r[idim] == Approx(r_expected[idim]) );
         }

         // compute physical point
         for ( int idim=0; idim < NDIMS; ++idim )
         {
            ip[ idim ] = x0[ idim ] + t*n[ idim ];
            REQUIRE( ip[ idim ] == Approx(ip_expected[idim]) );
         }

         break;
      }

   } // END for all elements

   delete m;
   return found;
}

} /* end anonymous namespace */

//------------------------------------------------------------------------------
// UNIT TESTS
//------------------------------------------------------------------------------
TEST_CASE( "Ray Surface Intersection 3D",
           "[fe_ray_intersection]")
{
   //  r=[0.520563, 0.241127] t=2.66268
   //  p=[2.66268, 3.16268, 2.66268]

   constexpr int NDIMS     = 3;
   constexpr int MAX_ORDER = 16;

   // Ray input
   const double x0[] = { 0.0, 0.5, 0.0 };     // ray origin
   const double n1[] = { 1.0, 1.0, 1.0 };     // ray normal for intersecting
   const double n2[] = { -1.0, -1.0, -1.0 };  // ray normal for non-intersecting

   // Expected results
   const int found_in = 48;
   const double r[]   = { 0.520563, 0.241127 };
   const double t     = 2.66268;
   const double ip[]  = { 2.66268, 3.16268, 2.66268 };

   for ( int iorder=1; iorder <= MAX_ORDER; ++iorder )
   {
      std::ostringstream oss;
      oss << "Testing 3D quad surface mesh of order [" << iorder << "]";
      SECTION( oss.str() )
      {

         SECTION( "Test with intersecting ray" )
         {
            REQUIRE( check_fe_ray< NDIMS >( iorder,x0,n1,found_in,ip,r,t) );
         }

         SECTION( "Test non-intersecting ray" )
         {
            REQUIRE_FALSE( check_fe_ray< NDIMS >( iorder,x0,n2 ) );
         }

      }

   } // END order loop

}

//------------------------------------------------------------------------------
TEST_CASE( "Ray Surface Intersection 2D",
           "[fe_ray_intersection]")
{
   //  r=[0.527751] t=2.54625
   //  p=[2.54625, 2.54625]

   constexpr int NDIMS     = 2;
   constexpr int MAX_ORDER = 16;

   // Ray input
   const double x0[] = {  0.0, 0.0  }; // ray origin
   const double n1[] = {  1.0, 1.0  }; // ray normal for intersecting
   const double n2[] = { -1.0, -1.0 }; // ray normal for non-intersecting

   // Expected results
   const int found_in = 4;
   const double r[]   = { 0.527751 };
   const double t     = 2.54625;
   const double ip[]  = { 2.54625, 2.54625 };

   for ( int iorder=1; iorder <= MAX_ORDER; ++iorder )
   {
      std::ostringstream oss;
      oss << "Testing 2D segment mesh of order [" << iorder << "]";
      SECTION( oss.str() )
      {

         SECTION( "Test with intersecting ray" )
         {
            REQUIRE( check_fe_ray< NDIMS >( iorder,x0,n1,found_in,ip,r,t) );
         }

         SECTION( "Test non-intersecting ray" )
         {
            REQUIRE_FALSE( check_fe_ray< NDIMS >( iorder,x0,n2 ) );
         }

      }

   } // END order loop
}
