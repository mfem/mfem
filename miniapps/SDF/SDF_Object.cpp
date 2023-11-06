// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

//#include "SDF_Mesh.hpp"
#include "SDF_Tools.hpp"
#include "SDF_Generator.hpp"

namespace mfem
{
namespace sdf
{
Triangle::Triangle(
   int                      aIndex,
   std::vector< Triangle_Vertex* >& aVertices )
   : mIndex( aIndex )
   , mVertices( aVertices )
   , mNodeCoords( 3, 3 )
   , mNodeIndices( 3 )
   , mCenter( 3 )
   , mNormal( 3 )
   , mPredictY( 3, 3 )
   , mPredictYRA( 3, 3 )
   , mPredictYRB( 3, 3 )
   , mMinCoord( 3 )
   , mMaxCoord( 3 )
{
   this->update_data();
}

//-------------------------------------------------------------------------------

void
Triangle::update_data()
{
   // step 1: copy node coordinates and determine center
   this->copy_node_coords_and_inds( mVertices );

   // help vector
   mfem::Vector tDirectionOfEdge( 3 );

   // step 2: calculate hesse normal form of plane
   this->calculate_hesse_normal_form( tDirectionOfEdge );

   // step 3: calculate barycentric data
   this->calculate_barycectric_data( tDirectionOfEdge );

   // step 4: calculate helpers for cross prediction
   this->calculate_prediction_helpers();
}

//-------------------------------------------------------------------------------

void Triangle::copy_node_coords_and_inds(
   std::vector< Triangle_Vertex* >& aVertices )
{
   // make sure that the length is correct
   if (not(aVertices.size() >= 3))
   {
      mfem_error( "Triangle() needs at least three vertices as input" );
   }

   // step 1: copy node coordinates into member variables
   //         and calculate center

   // reset center
   mCenter = 0.0;

   // loop over all nodes
   for ( unsigned int k = 0; k < 3; ++k )
   {
      // get vertex coordinates
      mfem::Vector tNodeCoords = aVertices[ k ]->get_coords();

      // copy coordinates into member matrix
      for ( unsigned int i = 0; i < 3; ++i )
      {
         mNodeCoords( i, k ) = tNodeCoords( i );
         mCenter( i ) += tNodeCoords( i );
      }

      // remember node indices
      mNodeIndices( k ) = aVertices[ k ]->get_index();
   }

   // identify minimum and maximum coordinate
   for (unsigned int i = 0; i < 3; ++i )
   {
      mMinCoord( i ) = min( mNodeCoords( i, 0 ),
                            mNodeCoords( i, 1 ),
                            mNodeCoords( i, 2 ) );

      mMaxCoord( i ) = max( mNodeCoords( i, 0 ),
                            mNodeCoords( i, 1 ),
                            mNodeCoords( i, 2 ) );
   }

   // divide center by three
   for ( unsigned int i = 0; i < 3; ++i )
   {
      mCenter( i ) /= 3.0;
   }
}

//-------------------------------------------------------------------------------

void
Triangle::calculate_hesse_normal_form( mfem::Vector & aDirectionOfEdge )
{
   // step 2: calculate plane of triangle
   mfem::Vector tDirection02( 3 );

   // help vectors: direction of sides 1 and 2
   for ( unsigned int i = 0; i < 3; ++i )
   {
      aDirectionOfEdge( i ) = mNodeCoords( i, 1 ) - mNodeCoords( i, 0 );
      tDirection02( i )     = mNodeCoords( i, 2 ) - mNodeCoords( i, 0 );
   }

   // norm of this triangle

   mNormal = cross( aDirectionOfEdge, tDirection02 );

   double tNorm = mNormal.Norml2();
   for ( unsigned int i = 0; i < 3; ++i )
   {
      mNormal( i ) /= tNorm;
   }

   // Hesse parameter of this triangle
   mHesse = mCenter * mNormal;
}

//-------------------------------------------------------------------------------

void
Triangle::calculate_barycectric_data( const mfem::Vector& aDirectionOfEdge )
{
   // calculate direction orthogonal to plane and edge
   mfem::Vector tDirectionOrtho = cross( mNormal, aDirectionOfEdge );

   // normalize tDirection10
   double tNorm10 = aDirectionOfEdge.Norml2();

   // normalize tDirectionOrtho
   double tNormOrtho = tDirectionOrtho.Norml2();

   // Projection matrix
   for ( unsigned int k = 0; k < 3; ++k )
   {
      mBarycentric.mProjectionMatrix( 0, k ) = aDirectionOfEdge( k ) / tNorm10;
      mBarycentric.mProjectionMatrix( 1, k ) = tDirectionOrtho( k ) / tNormOrtho;
      mBarycentric.mProjectionMatrix( 2, k ) = mNormal( k );
   }

   // node coordinates in triangle plane
   mBarycentric.mLocalNodeCoordsInPlane = 0.0;
   for ( unsigned int k = 0; k < 3; ++k )
   {
      for ( unsigned int j = 0; j < 3; ++j )
      {
         for ( unsigned int i = 0; i < 2; ++i )
         {
            mBarycentric.mLocalNodeCoordsInPlane( i, k ) +=
               mBarycentric.mProjectionMatrix( i, j )
               * ( mNodeCoords( j, k ) - mCenter( j ) );
         }
      }
   }

   // enlarge triangle
   mBarycentric.mLocalNodeCoordsInPlane( 0, 0 ) -= gSDFepsilon;
   mBarycentric.mLocalNodeCoordsInPlane( 1, 0 ) -= gSDFepsilon;
   mBarycentric.mLocalNodeCoordsInPlane( 0, 1 ) += gSDFepsilon;
   mBarycentric.mLocalNodeCoordsInPlane( 1, 1 ) -= gSDFepsilon;
   mBarycentric.mLocalNodeCoordsInPlane( 0, 2 ) += gSDFepsilon;

   // twice the area
   mBarycentric.mTwiceArea = ( mBarycentric.mLocalNodeCoordsInPlane( 0, 0 )
                               - mBarycentric.mLocalNodeCoordsInPlane( 0, 2 ) )
                             * ( mBarycentric.mLocalNodeCoordsInPlane( 1, 1 )
                                 - mBarycentric.mLocalNodeCoordsInPlane( 1, 2 ) )
                             - ( mBarycentric.mLocalNodeCoordsInPlane( 1, 0 )
                                 - mBarycentric.mLocalNodeCoordsInPlane( 1, 2 ) )
                             * ( mBarycentric.mLocalNodeCoordsInPlane( 0, 1 )
                                 - mBarycentric.mLocalNodeCoordsInPlane( 0, 2 ) );

   if (not(mBarycentric.mTwiceArea > 2 * gSDFepsilon))
   {
      mfem_error( "A degenerated triangle was found." );
   }

   mBarycentric.mInvTwiceArea = 1.0 / mBarycentric.mTwiceArea;

   // Edge directions
   for ( unsigned int k = 0; k < 3; ++k )
   {
      unsigned int i;
      unsigned int j;
      TrianglePermutation( k, i, j );
      for ( unsigned int l = 0; l < 2; ++l )
      {
         mBarycentric.mLocalEdgeDirectionVectors( l,
                                                  k ) = mBarycentric.mLocalNodeCoordsInPlane( l, j )
                                                        - mBarycentric.mLocalNodeCoordsInPlane( l, i );
      }
      double tMagnitude = 0.0;
      for ( unsigned int l = 0; l < 2; ++l )
      {
         tMagnitude += mBarycentric.mLocalEdgeDirectionVectors( l, k )
                       * mBarycentric.mLocalEdgeDirectionVectors( l, k );
      }
      mBarycentric.mLocalEdgeInverseMagnitudes( k ) = 1.0 / tMagnitude;
   }
}

//-------------------------------------------------------------------------------

void
Triangle::calculate_prediction_helpers()
{
   // values for cross prediction
   unsigned int i;
   unsigned int j;
   unsigned int p;
   unsigned int q;
   for ( unsigned int k = 0; k < 3; ++k )
   {
      TrianglePermutation( k, i, j );

      for ( unsigned int r = 0; r < 3; ++r )
      {
         TrianglePermutation( r, p, q );
         double tDelta = mNodeCoords( i, p ) - mNodeCoords( i, q );
         if ( std::abs( tDelta ) < gSDFepsilon )
         {
            if ( tDelta < 0 )
            {
               tDelta = -gSDFepsilon;
            }
            else
            {
               tDelta = gSDFepsilon;
            }
         }

         mPredictYRA( r, k ) = ( mNodeCoords( j, p ) - mNodeCoords( j, q ) ) / tDelta;

         mPredictY( r, k ) = mNodeCoords( j, p )
                             + mPredictYRA( r, k ) * ( mNodeCoords( i, r ) - mNodeCoords( i, p ) );

         mPredictYRB( r, k ) = mNodeCoords( j, p ) - mNodeCoords( i,
                                                                  p ) * mPredictYRA( r, k );
      }
   }
}

std::vector< Triangle_Vertex* > Triangle::get_vertex_pointers() const
{
   std::vector< Triangle_Vertex* > aVertices( 3, nullptr );

   for ( unsigned int k = 0; k < 3; ++k )
   {
      aVertices[ k ] = mVertices[ k ];
   }
   return aVertices;
}

//-------------------------------------------------------------------------------

// TODO MESHCLEANUP
void Triangle::remove_vertex_pointer( int aIndex )
{
   // std::cout<<"In SDF triangle"<<std::endl;
}

//-------------------------------------------------------------------------------

mfem::Vector Triangle::get_vertex_ids() const
{
   mfem::Vector aIDs( 3 );
   for ( unsigned int k = 0; k < 3; ++k )
   {
      aIDs[ k ] = mVertices[ k ]->get_id();
   }

   return aIDs;
}

//-------------------------------------------------------------------------------

mfem::Vector Triangle::get_vertex_inds() const
{
   mfem::Vector aINDs( 3 );

   for ( unsigned int k = 0; k < 3; ++k )
   {
      aINDs( k ) = mVertices[ k ]->get_index();
   }

   return aINDs;
}

//-------------------------------------------------------------------------------

mfem::DenseMatrix
Triangle::get_vertex_coords() const
{
   mfem::DenseMatrix aCoords( 3, 3 );

   for ( unsigned int k = 0; k < 3; ++k )
   {
      aCoords.SetRow( k, mVertices[ k ]->get_coords() );
   }

   return aCoords;
}

void Triangle::intersect_with_coordinate_axis(
   const mfem::Vector & aPoint,
   const int            aAxis,
   double             & aCoordinate,
   bool               & aError )
{
   if ( std::abs( mNormal( aAxis ) ) < gSDFepsilon )
   {
      aCoordinate = 0;
      aError      = true;
   }
   else
   {
      aCoordinate = aPoint( aAxis ) + ( mHesse - mNormal * aPoint ) / mNormal(
                       aAxis );
      aError      = false;
   }
}

//-------------------------------------------------------------------------------

bool Triangle::check_edge(
   const unsigned int   aEdge,
   const unsigned int   aAxis,
   const mfem::Vector & aPoint )
{
   unsigned int tI;
   unsigned int tJ;
   unsigned int tP;
   unsigned int tQ;

   // permutation parameter for axis
   TrianglePermutation( aAxis, tI, tJ );

   // permutation parameter for edge
   TrianglePermutation( aEdge, tP, tQ );

   // R
   double tPredictYR = mPredictYRA( aEdge,
                                    aAxis ) * aPoint( tI ) + mPredictYRB( aEdge, aAxis );

   // check of point is within all three projected edges
   return ( ( mPredictY( aEdge, aAxis ) > mNodeCoords( tJ, aEdge ) )
            && ( tPredictYR + gSDFepsilon > aPoint( tJ ) ) )
          || ( ( mPredictY( aEdge, aAxis ) < mNodeCoords( tJ, aEdge ) )
               && ( tPredictYR - gSDFepsilon < aPoint( tJ ) ) )
          || ( std::abs( ( mNodeCoords( tJ, tP ) - mNodeCoords( tJ, tQ ) )
                         * ( mNodeCoords( tI, tP ) - aPoint( tI ) ) )
               < gSDFepsilon );
}

//-------------------------------------------------------------------------------

mfem::Vector Triangle::get_barycentric_from_local_cartesian(
   const mfem::Vector& aLocalPoint )
{
   mfem::Vector aXi( 3 );

   // the first coordinate
   aXi( 0 ) = ( ( mBarycentric.mLocalNodeCoordsInPlane( 0, 1 )
                  - mBarycentric.mLocalNodeCoordsInPlane( 0, 2 ) )
                * ( mBarycentric.mLocalNodeCoordsInPlane( 1, 2 )
                    - aLocalPoint( 1 ) )
                - ( mBarycentric.mLocalNodeCoordsInPlane( 1, 1 )
                    - mBarycentric.mLocalNodeCoordsInPlane( 1, 2 ) )
                * ( mBarycentric.mLocalNodeCoordsInPlane( 0, 2 )
                    - aLocalPoint( 0 ) ) )
              * mBarycentric.mInvTwiceArea;

   // the second coordinate
   aXi( 1 ) = ( ( mBarycentric.mLocalNodeCoordsInPlane( 1, 0 )
                  - mBarycentric.mLocalNodeCoordsInPlane( 1, 2 ) )
                * ( mBarycentric.mLocalNodeCoordsInPlane( 0, 2 )
                    - aLocalPoint( 0 ) )
                - ( mBarycentric.mLocalNodeCoordsInPlane( 0, 0 )
                    - mBarycentric.mLocalNodeCoordsInPlane( 0, 2 ) )
                * ( mBarycentric.mLocalNodeCoordsInPlane( 1, 2 )
                    - aLocalPoint( 1 ) ) )
              * mBarycentric.mInvTwiceArea;

   // the third coordinate
   aXi( 2 ) = 1.0 - aXi( 0 ) - aXi( 1 );

   return aXi;
}

//-------------------------------------------------------------------------------

double Triangle::distance_point_to_edge_in_local_cartesian(
   const mfem::Vector& aLocalPoint,
   const unsigned int               aEdge )
{
   double tParam = 0;
   unsigned int i;
   unsigned int j;

   // permutation parameter of current edge
   TrianglePermutation( aEdge, i, j );

   // calculate projection of point on edge

   // tParam = 0: orthogonal intersects with point i
   // tParam = 1: orthogonal intersects with point j

   for ( unsigned int l = 0; l < 2; ++l )
   {
      tParam += ( aLocalPoint( l ) - mBarycentric.mLocalNodeCoordsInPlane( l, i ) )
                * mBarycentric.mLocalEdgeDirectionVectors( l, aEdge );
   }
   tParam *= mBarycentric.mLocalEdgeInverseMagnitudes( aEdge );

   mfem::Vector aDirection( 3 );

   if ( tParam < gSDFepsilon )
   {
      // snap to point i and set tParam = 0.0;
      aDirection( 0 ) = aLocalPoint( 0 ) - mBarycentric.mLocalNodeCoordsInPlane( 0,
                                                                                 i );
      aDirection( 1 ) = aLocalPoint( 1 ) - mBarycentric.mLocalNodeCoordsInPlane( 1,
                                                                                 i );
   }
   else if ( tParam > 1.0 - gSDFepsilon )
   {
      // snap to point j and set tParam = 1.0;
      aDirection( 0 ) = aLocalPoint( 0 ) - mBarycentric.mLocalNodeCoordsInPlane( 0,
                                                                                 j );
      aDirection( 1 ) = aLocalPoint( 1 ) - mBarycentric.mLocalNodeCoordsInPlane( 1,
                                                                                 j );
   }
   else
   {
      // find distance in plane
      aDirection( 0 ) = aLocalPoint( 0 ) - mBarycentric.mLocalNodeCoordsInPlane( 0,
                                                                                 i )
                        - tParam * mBarycentric.mLocalEdgeDirectionVectors( 0, aEdge );
      aDirection( 1 ) = aLocalPoint( 1 ) - mBarycentric.mLocalNodeCoordsInPlane( 1,
                                                                                 i )
                        - tParam * mBarycentric.mLocalEdgeDirectionVectors( 1, aEdge );
   }

   // add third dimension to distance
   aDirection( 2 ) = aLocalPoint( 2 );

   return aDirection.Norml2();
}

//-------------------------------------------------------------------------------

mfem::Vector Triangle::project_point_to_local_cartesian(
   const mfem::Vector& aPoint )
{
   // fixme: times operator does not work with eigen
   // return mBarycentric.mProjectionMatrix * ( aPoint - mCenter ) ;
   mfem::Vector aOut( 3 );
   aOut=0.0;;

   for ( unsigned int k = 0; k < 3; ++k )
   {
      for ( unsigned int i = 0; i < 3; ++i )
      {
         aOut( k ) += mBarycentric.mProjectionMatrix( k, i )
                      * ( aPoint( i ) - mCenter( i ) );
      }
   }

   return aOut;
}

//-------------------------------------------------------------------------------

double Triangle::get_distance_to_point(
   const mfem::Vector & aPoint )
{
   // step 1: Transform Point to in-plane coordinates
   mfem::Vector tLocalPointCoords = this->project_point_to_local_cartesian(
                                       aPoint );
   // step 2: calculate barycentric coordinates
   mfem::Vector tXi = this->get_barycentric_from_local_cartesian(
                         tLocalPointCoords );

   // step 3: check if we are inside the triangle
   if ( ( tXi( 0 ) >= -gSDFepsilon )
        && ( tXi( 1 ) >= -gSDFepsilon )
        && ( tXi( 2 ) >= -gSDFepsilon ) )
   {
      // the absolute value of the local z-coordinate is the distance
      return std::abs( tLocalPointCoords( 2 ) );
   }
   else
   {
      if ( tXi( 0 ) > 0 )
      {
         // this rules out edge 0
         double tDist1 = distance_point_to_edge_in_local_cartesian( tLocalPointCoords,
                                                                    1 );
         double tDist2 = distance_point_to_edge_in_local_cartesian( tLocalPointCoords,
                                                                    2 );
         return std::min( tDist1, tDist2 );
      }
      else if ( tXi( 1 ) > 0 )
      {
         // this rules out edge 1
         double tDist0 = distance_point_to_edge_in_local_cartesian( tLocalPointCoords,
                                                                    0 );
         double tDist2 = distance_point_to_edge_in_local_cartesian( tLocalPointCoords,
                                                                    2 );
         return std::min( tDist0, tDist2 );
      }
      else
      {
         // edge 2 must be the one to rule out
         double tDist0 = distance_point_to_edge_in_local_cartesian( tLocalPointCoords,
                                                                    0 );
         double tDist1 = distance_point_to_edge_in_local_cartesian( tLocalPointCoords,
                                                                    1 );
         return std::min( tDist0, tDist1 );
      }
   }
}

//-------------------------------------------------------------------------------

Triangle_Vertex::Triangle_Vertex(
   const int            aIndex,
   const mfem::Vector & aNodeCoords ) :
   mIndex( aIndex ),
   mNodeCoords( aNodeCoords ),
   mOriginalNodeCoords( aNodeCoords )
{ }

void Triangle_Vertex::rotate_node_coords( const mfem::DenseMatrix &
                                          aRotationMatrix )
{
   aRotationMatrix.Mult( mOriginalNodeCoords, mNodeCoords);
}

void Triangle_Vertex::reset_node_coords()
{
   mNodeCoords = mOriginalNodeCoords;
}
//-------------------------------------------------------------------------------

Object::Object ( const std::string & aFilePath)
{
    auto tFileExt = aFilePath.substr(aFilePath.find_last_of(".")+1,
                                     aFilePath.length());

    if (tFileExt == "obj")
    {
       this->load_from_object_file( aFilePath);
    }
    else if (tFileExt == "stl")
    {
       this->load_from_stl_file( aFilePath );
    }
    else
    {
       mfem_error("Object(), file type is not supported");
    }
}

Object::Object ( const std::string & aFilePath, double scalefac)
{
   // check the file extension
   auto tFileExt = aFilePath.substr(aFilePath.find_last_of(".")+1,
                                    aFilePath.length());

   if (tFileExt == "obj")
   {
      this->load_from_object_file( aFilePath, scalefac );
   }
   else if (tFileExt == "stl")
   {
      this->load_from_stl_file( aFilePath );
   }
   else
   {
      mfem_error("Object(), file type is not supported");
   }
}

Object::Object ( const std::string & aFilePath,
                 const mfem::Vector  aOffsets)
{
   if ( aOffsets.Size()>0)
   {
      mOffsets = aOffsets;
   }

   // check the file extension
   auto tFileExt = aFilePath.substr(aFilePath.find_last_of(".")+1,
                                    aFilePath.length());

   if (tFileExt == "obj")
   {
      this->load_from_object_file( aFilePath );
   }
   else if (tFileExt == "stl")
   {
      this->load_from_stl_file( aFilePath );
   }
   else
   {
      mfem_error("Object(), file type is not supported");
   }
}

//-------------------------------------------------------------------------------

void
Object::load_from_object_file( const std::string& aFilePath, double scalefac )
{
   // copy file into buffer
   std::vector<std::string> tBuffer;
   this->load_ascii_to_buffer( aFilePath, tBuffer );

   // step 1: count number of vertices and triangles in file

   // reset counter for vertices
   unsigned int tNumberOfVertices = 0;

   // reset counter for triangles
   unsigned int tNumberOfTriangles = 0;

   // get length of buffer
   unsigned int tBufferLength = tBuffer.size();

   // loop over all lines
   for ( unsigned int k=0; k<tBufferLength; ++k )
   {
      if ( tBuffer[ k ].substr( 0, 2 ) == "v ")
      {
         ++tNumberOfVertices;
      }
      else if ( tBuffer[ k ].substr( 0, 2 ) == "f ")
      {
         ++tNumberOfTriangles;
      }
   }

   std::cout << tNumberOfVertices << " " << tNumberOfTriangles << " k10nvnt\n";
   if (mOffsets.Size() == 0)
   {
       mOffsets.SetSize(3);
       mOffsets = 0.0;
   }

   // step 2: create vertices
   mVertices.resize( tNumberOfVertices, nullptr );

   // reset counter

   unsigned int tCount = 0;
   // loop over all lines
   int ftype1 = 0;
   int ftype2 = 0;
   double xmin = DBL_MAX,
          xmax = -DBL_MAX;
   double ymin = DBL_MAX,
           ymax = -DBL_MAX;
   double zmin = DBL_MAX,
          zmax = -DBL_MAX;
   for ( unsigned int k=0; k<tBufferLength; ++k )
   {
      if ( tBuffer[ k ].substr( 0, 2 ) == "v ")
      {
         // create matrix with coordinates
         mfem::Vector tNodeCoords( 3 );

         float tX[ 3 ];

         // read ascii data into coordinates
         std::sscanf(tBuffer[ k ].substr(
                        2,
                        tBuffer[ k ].length()).c_str(),
                     "%f %f %f",
                     &tX[ 0 ],
                     &tX[ 1 ],
                     &tX[ 2 ] );

//         std::cout << tX[0] << " " << tX[1] << " "  << tX[2] << " " << scalefac << " k10tx\n";
//         for (int d = 0; d < 3; d++)
//         {
//             tX[d] *= scalefac;
//         }
//         std::cout << tX[0] << " " << scalefac << " k10tx\n";

         // test coordinates for highpass
         for ( unsigned int i=0; i<3; ++i )
         {
            if ( std::abs( tX[ i ] ) > mMeshHighPass )
            {
               tNodeCoords[ i ] = tX[ i ] + mOffsets[ i ];
            }
            else
            {
               // use zero value
               tNodeCoords[ i ] = 0.0 + mOffsets[ i ] ;
            }
            xmax = std::fmax(xmax, tNodeCoords[0]);
            xmin = std::fmin(xmin, tNodeCoords[0]);
            ymax = std::fmax(ymax, tNodeCoords[1]);
            ymin = std::fmin(ymin, tNodeCoords[1]);
            zmax = std::fmax(zmax, tNodeCoords[2]);
            zmin = std::fmin(zmin, tNodeCoords[2]);
         }

         // create vertex
         mVertices[ tCount ] = new Triangle_Vertex( tCount, tNodeCoords );

         // increment counter
         ++tCount;
      }
      else if ( tBuffer[ k ].substr( 0, 2 ) == "vt")
      {
          ftype1 = 1;
      }
      else if ( tBuffer[ k ].substr( 0, 2 ) == "vn")
      {
          ftype2 = 1;
      }
   }

   double ld = xmax-xmin;
   double lmin = xmin;
   if (ymax-ymin > ld) {
       ld = ymax-ymin;
       lmin = ymin;
   }
   if (zmax-zmin > ld) {
       ld = zmax-zmin;
       lmin = zmin;
   }

   for (int i = 0; i < tCount; i++)
   {
       Vector &coords = mVertices[i]->get_coords();
//       std::cout << i << " k10v\n";
//       coords.Print();
       coords(0) = (coords(0)-xmin)/(ld);
       coords(1) = (coords(1)-ymin)/(ld);
       coords(2) = (coords(2)-zmin)/(ld);
       mVertices[i]->set_node_coords(coords);
//       Vector &coords2 = mVertices[i]->get_coords();
//       coords2.Print();
   }

//   MFEM_ABORT(" ");

   // step 3
   // create triangles

   // reset counter
   tCount = 0;

   // reserve memory
   mTriangles.resize( tNumberOfTriangles, nullptr );

   // temporary one-based Ids for triangle nodes 1, 2 and 3

   // loop over all lines
   for ( unsigned int k=0; k<tBufferLength; ++k )
   {
      if ( tBuffer[ k ].substr( 0, 2 ) == "f " )
      {
         // temporary container for vertices
         std::vector< Triangle_Vertex * > tNodes( 3, nullptr );

         int tNodeIndices[ 3 ];
         int dum[3];
         int dum2[3];
         // read triangle topology
//         bool f1 = false; //k10modify
         bool f1 = true;
         if (ftype1 + ftype2 == 0)
         {
             std::sscanf(tBuffer[ k ].substr(2,tBuffer[ k ].length()).c_str(),
                         "%u %u %u",
                         &tNodeIndices[ 0 ],
                         &tNodeIndices[ 1 ],
                         &tNodeIndices[ 2 ]);
         }
         else if (ftype1+ftype2 == 1)
         {
             std::sscanf(tBuffer[ k ].substr(2,tBuffer[ k ].length()).c_str(),
                         "%u/%u %u/%u %u/%u",
                         &tNodeIndices[ 0 ],&dum[ 0 ],
                         &tNodeIndices[ 1 ],&dum[ 1 ],
                         &tNodeIndices[ 2 ],&dum[ 2 ]);
         }
         else if (ftype1+ftype2 == 2)
         {
             std::sscanf(tBuffer[ k ].substr(2,tBuffer[ k ].length()).c_str(),
                         "%u/%u/%u %u/%u/%u %u/%u/%u",
                         &tNodeIndices[ 0 ],&dum[ 0 ],&dum2[ 0 ],
                         &tNodeIndices[ 1 ],&dum[ 1 ],&dum2[ 1 ],
                         &tNodeIndices[ 2 ],&dum[ 2 ],&dum2[ 2 ]);
         }
//         std::cout << tNodeIndices[0] << " " << tNodeIndices[1] << " " << tNodeIndices[2] << " k10ind\n";
         // assign vertices with triangle
         for ( int i=0; i<3; ++i )
         {
            // make sure that file is sane
            if (not(0 < tNodeIndices[ i ] && tNodeIndices[ i ] <= tNumberOfVertices))
            {
               std::cout << tNodeIndices[ i ] << " " << tNumberOfVertices << " k10debuginfo\n";
               mfem_error("Invalid vertex ID in object file" );
            }

            // copy vertex into cell
            tNodes[ i ] = mVertices[ tNodeIndices[ i ] - 1 ];
         }


         // create triangle pointer
         mTriangles[ tCount ] = new Triangle( tCount, tNodes );

         // increment counter
         ++tCount;
      }
   }
   std::cout << mTriangles.size() << " k10mtrianglessize\n";
}

//-------------------------------------------------------------------------------

Object::~Object()
{
   for ( auto tTriangle : mTriangles )
   {
      delete tTriangle;
   }

   for ( auto tVertex : mVertices )
   {
      delete tVertex;
   }
}

//-------------------------------------------------------------------------------
void
Object::load_ascii_to_buffer( const std::string& aFilePath,
                              std::vector<std::string>& aBuffer)
{
   // try to open ascii file
   std::ifstream tAsciiFile( aFilePath );
   std::string tLine;

   // load file into buffer, otherwise throw error
   if ( tAsciiFile )
   {
      // count number of lines
      unsigned int tBufferLength = 0;
      while ( !tAsciiFile.eof() )
      {
         std::getline(tAsciiFile, tLine);
         ++tBufferLength;
      }
      tAsciiFile.close();
      tAsciiFile.open(aFilePath);

      // load file into buffer
      aBuffer.reserve(tBufferLength);
      while (!tAsciiFile.eof())
      {
         std::getline(tAsciiFile, tLine);
         aBuffer.push_back(tLine);
      }
      tAsciiFile.close();
   }
   else
   {
      std::cerr << "Something went wrong while trying to load from " <<
                aFilePath << "." << std::endl;
   }
}

//-------------------------------------------------------------------------------

// note: this routines reads the ascii stl files
// for binary stl see https://en.wikipedia.org/wiki/STL_(file_format)
void
Object::load_from_stl_file( const std::string& aFilePath )
{
   // copy file into buffer
   std::vector<std::string> tBuffer;
   this->load_ascii_to_buffer( aFilePath, tBuffer );

   // get length of buffer
   unsigned int tBufferLength = tBuffer.size();

   // - - - - - - - - - - - - -
   // step 1: count triangles
   // - - - - - - - - - - - - -

   // initialize counter
   unsigned int tCount = 0;

   // loop over all lines
   for ( unsigned int k=0; k<tBufferLength; ++k )
   {
      // extract first word from string
      std::string tWord = clean( tBuffer [ k ] );

      if ( tWord.substr( 0, 5 ) == "facet" )
      {
         ++tCount;
      }
   }

   if (not(tCount > 0))
   {
      mfem_error( "Could not find any facets in this file. Maybe not an ASCII STL?" );
   }
   // remember number of triangles
   unsigned int tNumberOfTriangles = tCount;

   // - - - - - - - - - - - - -
   // step 2: create vertices
   // - - - - - - - - - - - - -

   mVertices.resize( 3*tCount, nullptr );

   // reset counter
   tCount = 0;

   // create matrix with coordinates
   mfem::Vector tNodeCoords( 3 );

   // loop over all lines
   for ( unsigned int k=0; k<tBufferLength; ++k )
   {
      // extract first word from string
      std::vector< std::string > tWords = string_to_words( tBuffer[ k ] );

      if ( tWords.size() > 0 )
      {
         if ( tWords[ 0 ] == "vertex" )
         {
            // parse words to coords
            tNodeCoords[ 0 ] = stod( tWords[ 1 ] );
            tNodeCoords[ 1 ] = stod( tWords[ 2 ] );
            tNodeCoords[ 2 ] = stod( tWords[ 3 ] );

            // create vertex
            mVertices[ tCount ] = new Triangle_Vertex( tCount, tNodeCoords );

            // increment vertex counter
            ++tCount;
         }
      }
   }

   // make sure that number of triangles is correct
   if (tCount != mVertices.size())
   {
      mfem_error(  "Number of vertices does not match" );
   }

   // - - - - - - - - - - - - -
   // step 3: create triangles
   // - - - - - - - - - - - - -

   // allocate memory
   mTriangles.resize( tNumberOfTriangles, nullptr );

   // reset counter
   tCount = 0;

   // temporary container for vertices
   std::vector< Triangle_Vertex * > tNodes( 3, nullptr );

   // create triangles
   for ( unsigned int k=0; k<tNumberOfTriangles; ++k )
   {
      tNodes[ 0 ] = mVertices[ tCount++ ];
      tNodes[ 1 ] = mVertices[ tCount++ ];
      tNodes[ 2 ] = mVertices[ tCount++ ];

      // create triangle pointer
      mTriangles[ k ] = new Triangle( k, tNodes );
   }
}

//-------------------------------------------------------------------------------

mfem::Vector Object::get_nodes_connected_to_element_loc_inds
( int aElementIndex ) const
{
   // get pointer to triangle
   return mTriangles[ aElementIndex ]->get_vertex_inds();
}
} /* namespace sdf */
} /* namespace mfem */
