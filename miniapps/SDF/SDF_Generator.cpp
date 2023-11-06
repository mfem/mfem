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

#include "SDF_Mesh.hpp"
#include "SDF_Tools.hpp"
#include "SDF_Generator.hpp"

namespace mfem
{
namespace sdf
{
Core::Core( mfem::sdf::Mesh & aMesh,
            mfem::sdf::Data & aData,
            bool   aVerbose ) :
   mMesh( aMesh ),
   mData( aData ),
   mVerbose( aVerbose )
{
   // fill unsure nodes list
   //uint tNumberOfNodes = aMesh.get_num_nodes();
}
//-------------------------------------------------------------------------------

void Core::calculate_raycast(
   mfem::Array<int> & aElementsAtSurface,
   mfem::Array<int> & aElementsInVolume )
{

   // call private routine
   this->calculate_raycast();

   // assign element containers
   aElementsAtSurface.SetSize( mData.mSurfaceElements );

   aElementsInVolume.SetSize( mData.mVolumeElements );

   // counters
   unsigned int tSurfaceCount = 0;
   unsigned int tVolumeCount = 0;

   // get number of elements
   unsigned int tNumberOfElements = mMesh.get_num_elems();

   // loop over all elements
   for ( unsigned int k=0; k<tNumberOfElements; ++k )
   {
      // get pointer to element
      Cell * tElement = mMesh.get_cell( k );

      if ( tElement->is_on_surface() )
      {
         aElementsAtSurface[ tSurfaceCount++ ] = k;
      }
      else if ( tElement->is_in_volume() )
      {
         aElementsInVolume[ tVolumeCount++ ] = k;
      }
   }

   // make sure that everything is OK
   if (tSurfaceCount != mData.mSurfaceElements)
   {
      mfem_error( "Number of surface elements does not match" );
   }

   if (tVolumeCount != mData.mVolumeElements)
   {
      mfem_error( "Number of volume elements does not match" );
   }
}

//-------------------------------------------------------------------------------

void Core::calculate_raycast(
   mfem::Array<int> & aElementsAtSurface )
{

   // call private routine
   this->calculate_raycast();

   // assign element containers
   aElementsAtSurface.SetSize( mData.mSurfaceElements );

   // counters
   unsigned int tSurfaceCount = 0;

   // get number of elements
   unsigned int tNumberOfElements = mMesh.get_num_elems();

   // loop over all elements
   for ( unsigned int k=0; k<tNumberOfElements; ++k )
   {
      // get pointer to element
      Cell * tElement = mMesh.get_cell( k );

      if ( tElement->is_on_surface() )
      {
         aElementsAtSurface[ tSurfaceCount++ ] = k;
      }
   }

   // make sure that everything is OK
   if (tSurfaceCount != mData.mSurfaceElements)
   {
      mfem_error( "Number of surface elements does not match" );
   }
}

//-------------------------------------------------------------------------------

void Core::calculate_raycast()
{
   // set unsure flag of all nodes to true
   unsigned int tNumberOfNodes = mMesh.get_num_nodes();

   for ( unsigned int k=0; k< tNumberOfNodes; ++k )
   {
      mMesh.get_vertex( k )->reset();
   }
   mData.mUnsureNodesCount = tNumberOfNodes;

   // flag that marks if rotation was called
   bool tRotation = false;

   while ( mData.mUnsureNodesCount > 0 )
   {
      // perform voxelizing algorithm in z-direction
      voxelize( 2 );
      if ( mData.mUnsureNodesCount > 0 )
      {
         // perform voxelizing algorithm in y-direction
         voxelize( 1 );
         if ( mData.mUnsureNodesCount > 0 )
         {
            // perform voxelizing algorithm in x-direction
            voxelize( 0 );
         }
      }

      if ( mData.mUnsureNodesCount > 0 )
      {
         tRotation = true;

         this->random_rotation();
      }
   }

   if ( tRotation )
   {
      this->undo_rotation();
   }

   // remainung nodes are pushed outside
   this->force_unsure_nodes_outside();

   // identify elements in surface, volume and candidates
   this->calculate_candidate_points_and_buffer_diagonal();

   if ( mVerbose )
   {

   }
}

//-------------------------------------------------------------------------------

void
Core::calculate_raycast_and_sdf( mfem::Vector & aSDF )
{
   this->calculate_raycast();

   std::vector< Vertex * >
   tCandidateList;  //========================================
   tCandidateList =
      this->set_candidate_list();  //===================================

   this->calculate_udf(tCandidateList);
   this->sweep();
   this->fill_sdf_with_values( aSDF );
}

//-------------------------------------------------------------------------------

void
Core::calculate_raycast_and_sdf(
   mfem::Vector     & aSDF,
   mfem::Array<int> & aElementsAtSurface,
   mfem::Array<int> & aElementsInVolume )
{
   this->calculate_raycast( aElementsAtSurface, aElementsInVolume );

   std::vector< Vertex * >
   tCandidateList;  //========================================
   tCandidateList =
      this->set_candidate_list();  //===================================

   this->calculate_udf(tCandidateList);
   this->sweep();
   this->fill_sdf_with_values( aSDF );
}

//-------------------------------------------------------------------------------

void Core::voxelize( const unsigned int aAxis )
{
   // reset unsure nodes counter
   mData.mUnsureNodesCount = 0;

   // get number of unsure nodes
   unsigned int tNumberOfNodes = mMesh.get_num_nodes();

   // fixme : check if this is neccessary
   for ( Triangle * tTriangle : mData.mTriangles )
   {
      tTriangle->unflag();
   }

   // loop over all nodes
   for ( unsigned int k=0; k<tNumberOfNodes; ++k )
   {
      if (  mMesh.get_vertex( k )->is_flagged() )
      {
         // get node coordinate
         const mfem::Vector & tPoint = mMesh.get_node_coordinate( k );

         // preselect triangles for intersection test
         if (aAxis == 0)
         {
            this->preselect_triangles_x( tPoint );
         }
         else if (aAxis == 1)
         {
            this->preselect_triangles_y( tPoint );
         }
         else
         {
            this->preselect_triangles_z( tPoint );
         }

         // from the candidate triangles, perform intersection
         if ( mData.mCandidateTriangles.Size() > 0 )
         {
            this->intersect_triangles( aAxis, tPoint );

            // intersect ray with triangles and check if node is inside
            if ( mData.mIntersectedTriangles.size() > 0 )
            {
               this->intersect_ray_with_triangles( aAxis, tPoint, k );

               this->check_if_node_is_inside( aAxis, k );
            }
         }
      }
   }
}

//-------------------------------------------------------------------------------

void Core::calculate_udf(std::vector< Vertex * > & aCandidateList)
{
   // get number of triangles
   unsigned int tNumberOfTriangles = mData.mTriangles.size();
   std::cout<<"number of triangles: "<<tNumberOfTriangles<<std::endl; //======
   // loop over all triangles
   for ( unsigned int k=0; k<tNumberOfTriangles; ++k )
   {
      // get pointer to triangle
      Triangle * tTriangle = mData.mTriangles[ k ];

      // get nodes withing triangle
      std::vector< Vertex * > tNodes;
//      {
//          for (int d = 0; d < 3; d++)
//          {
//              std::cout << k << " " << d << " " << tTriangle->get_min_coord( d ) << " " <<
//                           tTriangle->get_max_coord( d ) << " k111111" << std::endl;
//          }
//      }

      this->get_nodes_withing_bounding_box_of_triangle(
         tTriangle, tNodes, aCandidateList );


      // get number of nodes
      unsigned int tNumberOfNodes = tNodes.size();

      // calculate distance of this point to the triangle
      // and update udf value if it is smaller
      for ( unsigned int i=0; i<tNumberOfNodes; ++i )
      {
         // update UDF of this node
         tNodes[ i ]->update_udf( tTriangle );
//         std::cout << i << " k101\n";
      }

   } // end loop over all triangles

   if ( mVerbose )
   {

   }
}

//-------------------------------------------------------------------------------

void Core::preselect_triangles_x( const mfem::Vector & aPoint )
{
   // x: k = x, j = z, i = y

   // loop over all triangles in J-Direction
   unsigned int tCountJ = 0;
   for (unsigned int k = 0; k<mData.mNumberOfTriangles; ++k)
   {
      // check bounding box in J-direction
      if ( (aPoint(2) - mData.mTriangleMinCoordsZ(k)) * (mData.mTriangleMaxCoordsZ(
                                                            k) - aPoint(2)) > -gSDFepsilon )
      {
         // remember this triangle
         mData.mCandJ[tCountJ] = k;

         // increment counter
         ++tCountJ;
      }
   }

   // counter for triangles
   unsigned int tCount = 0;

   // reset candidate size
   mData.mCandidateTriangles.SetSize(mData.mNumberOfTriangles);

   // loop over remaining triangles in I-direction
   for (unsigned int k = 0; k<tCountJ; ++k)
   {
      // check bounding box in I-direction
      if ((aPoint(1) - mData.mTriangleMinCoordsY(mData.mCandJ[k]))*
          (mData.mTriangleMaxCoordsY(mData.mCandJ[k]) - aPoint(1)) > -gSDFepsilon )
      {
         mData.mCandidateTriangles[tCount] = mData.mCandJ[k];
         ++tCount;
      }
   }

   mData.mCandidateTriangles.SetSize(tCount);

}

//-------------------------------------------------------------------------------

void Core::preselect_triangles_y( const mfem::Vector & aPoint )
{
   // y: k = y, j = x, i = z

   // loop over all triangles in J-Direction
   unsigned int tCountJ = 0;
   for (unsigned int k = 0; k<mData.mNumberOfTriangles; ++k)
   {
      // check bounding box in J-direction
      if ((aPoint(0) - mData.mTriangleMinCoordsX(k)) *
          (mData.mTriangleMaxCoordsX(k) - aPoint(0)) > -gSDFepsilon )
      {
         // remember this triangle
         mData.mCandJ[tCountJ] = k;

         // increment counter
         ++tCountJ;
      }
   }

   // counter for triangles
   unsigned int tCount = 0;

   // reset candidate size
   mData.mCandidateTriangles.SetSize(mData.mNumberOfTriangles);

   // loop over remaining triangles in I-direction
   for (unsigned int k = 0; k<tCountJ; ++k)
   {
      // check bounding box in I-direction
      if ((aPoint(2) - mData.mTriangleMinCoordsZ(mData.mCandJ[k]))*
          (mData.mTriangleMaxCoordsZ(mData.mCandJ[k]) - aPoint(2)) > -gSDFepsilon )
      {
         mData.mCandidateTriangles[tCount] = mData.mCandJ[k];
         ++tCount;
      }
   }

   mData.mCandidateTriangles.SetSize(tCount);
}

//-------------------------------------------------------------------------------

void  Core::preselect_triangles_z( const mfem::Vector & aPoint )
{
   // z: k = z, j = y, i = x

   // loop over all triangles in J-Direction
   unsigned int tCountJ = 0;
   for (unsigned int k = 0; k<mData.mNumberOfTriangles; ++k)
   {
      // check bounding box in J-direction
      if ( (aPoint(1) - mData.mTriangleMinCoordsY(k)) *
           (mData.mTriangleMaxCoordsY(k) - aPoint(1)) > -gSDFepsilon )
      {
         // remember this triangle
         mData.mCandJ[tCountJ] = k;

         // increment counter
         ++tCountJ;
      }
   }

   // counter for triangles
   unsigned int tCount = 0;

   // reset candidate size
   mData.mCandidateTriangles.SetSize(mData.mNumberOfTriangles);

   // loop over remaining triangles in I-direction
   for (unsigned int k = 0; k<tCountJ; ++k)
   {
      // check bounding box in I-direction
      if ((aPoint(0) - mData.mTriangleMinCoordsX(mData.mCandJ[k]))*
          (mData.mTriangleMaxCoordsX(mData.mCandJ[k]) - aPoint(0)) > -gSDFepsilon )
      {
         mData.mCandidateTriangles[tCount] = mData.mCandJ[k];
         ++tCount;
      }
   }

   mData.mCandidateTriangles.SetSize(tCount);
}

//-------------------------------------------------------------------------------

void Core::intersect_triangles( const unsigned int aAxis,
                                const mfem::Vector & aPoint )
{
   // get number of candidate triangles
   unsigned int tNumberOfTriangles = mData.mCandidateTriangles.Size();

   // initialize counter for intersected triangles
   unsigned int tCount = 0;

   // loop over all candidates
   for ( unsigned int k=0; k<tNumberOfTriangles; ++k )
   {
      // get pointer to triangle
      Triangle * tTriangle
         = mData.mTriangles[ mData.mCandidateTriangles[ k ] ];

      if ( tTriangle->check_edge( 0, aAxis, aPoint ) )
      {
         if ( tTriangle->check_edge( 1, aAxis, aPoint ) )
         {
            if ( tTriangle->check_edge( 2, aAxis, aPoint ) )
            {
               tTriangle->flag();
               ++tCount;
            }
         }
      }
   }

   // resize container with intersected triangles
   mData.mIntersectedTriangles.resize( tCount, nullptr );

   // reset counter
   tCount = 0;

   // loop over all candidates
   for ( unsigned int k=0; k<tNumberOfTriangles; ++k )
   {
      // get pointer to triangle
      Triangle * tTriangle
         = mData.mTriangles[ mData.mCandidateTriangles[ k ] ];

      if ( tTriangle->is_flagged() )
      {
         // add triangle to list
         mData.mIntersectedTriangles[ tCount++ ] = tTriangle;

         // unflag triangle
         tTriangle->unflag();
      }
   }
}

//-------------------------------------------------------------------------------

void Core::intersect_ray_with_triangles(
   const unsigned int   aAxis,
   const mfem::Vector & aPoint,
   const unsigned int   aNodeIndex )
{
   // get number of triangles
   unsigned int tNumberOfTriangles = mData.mIntersectedTriangles.size();

   // initialize vector with coords in axis
   Array<double> tCoordsK( tNumberOfTriangles );

   unsigned int tCount = 0;

   bool tError;
   // loop over all intersected triangles and find intersection point
   for ( unsigned int k = 0; k<tNumberOfTriangles; ++k )
   {
      double tCoordK;

      // calculate intersection coordinate
      mData.mIntersectedTriangles[ k ]->intersect_with_coordinate_axis(
         aPoint,
         aAxis,
         tCoordK,
         tError );

      //tCoordK = std::max( std::min( tCoordK,  tMaxCoord ), tMinCoord );

      // error meant we would have divided by zero. This triangle is ignored
      // otherwise, the value is written into the result vector

      if ( ! tError )
      {
         tCoordsK[ tCount++ ] = std::round( tCoordK / gSDFepsilon )* gSDFepsilon;
      }
      else
      {
         break;
      }
   }

   if ( tError )
   {
      // this way, the matrix is ignored
      mData.mCoordsK.SetSize( 1 );
      mData.mCoordsK = 0.0;
   }
   else
   {
      // resize coord array
      tCoordsK.SetSize( tCount );

      // sort array
      tCoordsK.Sort();

      // make result unique
      unsigned int tCountUnique = 0;

      // set size of output array
      mData.mCoordsK.SetSize( tCount );

      double tMinCoord = mMesh.get_min_coord( aAxis );
      double tMaxCoord = mMesh.get_max_coord( aAxis );

      // set first entry
      if ( tMinCoord < tCoordsK[ 0 ] )
      {
         mData.mCoordsK( tCountUnique++ ) = tCoordsK[ 0 ];
      }

      // find unique entries
      for ( unsigned int k=1; k<tCount; ++k )
      {
         if (tCoordsK[ k ] > tMinCoord && tCoordsK[ k ] < tMaxCoord )
         {
            if ( std::abs( tCoordsK[ k ] - tCoordsK[ k-1 ] ) > 10*gSDFepsilon )
            {
               mData.mCoordsK( tCountUnique++ ) = tCoordsK[ k ];
            }
         }
      }

      // chop vector
      mData.mCoordsK.SetSize( tCountUnique );
   }
}

//-------------------------------------------------------------------------------

void Core::check_if_node_is_inside(
   const unsigned int aAxis,
   const unsigned int aNodeIndex )
{
   unsigned int tNumCoordsK = mData.mCoordsK.Size();

   bool tNodeIsInside = false;

   const mfem::Vector & aPoint = mMesh.get_node_coordinate( aNodeIndex );

   // only even number of intersections is considered
   if ( tNumCoordsK % 2 == 0)
   {
      for ( unsigned int k=0; k< tNumCoordsK / 2; ++k)
      {
         tNodeIsInside = ( aPoint( aAxis ) > mData.mCoordsK( 2 * k )) &&
                         ( aPoint( aAxis ) < mData.mCoordsK( 2 * k + 1 ));

         // break the loop if inside
         if ( tNodeIsInside )
         {
            break;
         }
      }

      // set the inside flag of this node to the corresponding value
      if ( tNodeIsInside )
      {
         mMesh.get_vertex( aNodeIndex )->set_inside_flag();
      }
      else
      {
         mMesh.get_vertex( aNodeIndex )->unset_inside_flag();
      }
      mMesh.get_vertex( aNodeIndex )->unflag();
   }
   else
   {
      // set unsure flag
      mMesh.get_vertex( aNodeIndex )->flag();

      // increment counter
      ++mData.mUnsureNodesCount;
   }
}

//-------------------------------------------------------------------------------

void Core::calculate_candidate_points_and_buffer_diagonal()
{
   // get number of elements
   unsigned int tNumberOfElements = mMesh.get_num_elems();

   // counter for elements near surface
   mData.mSurfaceElements = 0;

   // counter for elements in volume
   mData.mVolumeElements = 0;

   // reset buffer diagonal
   mData.mBufferDiagonal = 0;

   // search all elements for sign change
   for ( unsigned int e=0; e < tNumberOfElements; ++e )
   {
      // unflag this element

      Cell * tElement = mMesh.get_cell( e );

      // reset flags of this element
      tElement->unflag();
      tElement->unset_surface_flag();
      tElement->unset_volume_flag();

      // get pointer to nodes
      const std::vector< Vertex * > tNodes = tElement->get_vertices();

      // get number of nodes
      unsigned int tNumberOfNodes = tNodes.size();

      // get first sign
      bool tIsInside = tNodes[ 0 ]->is_inside();

      // assume element is not intersected
      bool tIsIntersected = false;

      // loop over all other nodes
      for ( unsigned int k=1; k<tNumberOfNodes; ++k )
      {
         // check of sign is the same
         if ( tNodes[ k ]->is_inside() != tIsInside )
         {
            // sign is not same
            tIsIntersected = true;

            // cancel loop
            break;
         }
      }

      // test if there is a sign change
      if ( tIsIntersected )
      {
         // flag this element as surface element
         tElement->set_surface_flag();
         tElement->unset_volume_flag();

         // increment counter
         ++mData.mSurfaceElements ;

         // update buffer diagonal
         mData.mBufferDiagonal = std::max(
                                    mData.mBufferDiagonal,
                                    tElement->get_buffer_diagonal() );

         // flag to indicate that the buffer of this element
         // has been calculated
         tElement->flag();

         // flag all nodes of this element as candidates
         for ( unsigned int k=0; k<tNumberOfNodes; ++k )
         {
            tNodes[ k ]->set_candidate_flag();
         }
//         std::cout << e << " "  << mData.mBufferDiagonal << " k10elintersect\n";
      }
      else if ( tIsInside )
      {
         // flag this element as volume element
         tElement->unset_surface_flag();
         tElement->set_volume_flag();

         // increment counter
         ++mData.mVolumeElements ;
      }
      else
      {
         // unflag element
         tElement->unset_surface_flag();
         tElement->unset_volume_flag();
      }
   }

   // add additional search depth
   for ( unsigned int d=1; d<mCandidateSearchDepth; ++d )
   {
      // loop over all elements
      for ( unsigned int e=0; e < tNumberOfElements; ++e )
      {
         // get pointer to element
         Cell * tElement = mMesh.get_cell( e );

         // test if element is not flagged
         if ( ! tElement->is_flagged() )
         {
            // get pointer to nodes
            const std::vector< Vertex * > tNodes = tElement->get_vertices();

            // get number of nodes
            unsigned int tNumberOfNodes = tNodes.size();

            bool tIsCandidate = false;

            // test if at least one node of this element is flagged
            // as candidate
            for ( unsigned int k=0; k<tNumberOfNodes; ++k )
            {
               if ( tNodes[ k ]->is_candidate() )
               {
                  tIsCandidate = true;
                  break;
               }
            }

            // test if candidtae flag is set
            if ( tIsCandidate )
            {
               // update buffer diagonal
               mData.mBufferDiagonal = std::max(
                                          mData.mBufferDiagonal,
                                          tElement->get_buffer_diagonal() );

               // flag this element
               tElement->flag();

               // flag all nodes of this element
               for ( unsigned int k=0; k<tNumberOfNodes; ++k )
               {
                  tNodes[ k ]->set_candidate_flag();
               }
            }
         } // end loop over all elements
      }
   } // end candidate search depth loop
}

//-------------------------------------------------------------------------------

std::vector< Vertex * > Core::set_candidate_list(  )
{
   unsigned int tNumberOfNodes = mMesh.get_num_nodes();
//   std::cout<<"number of nodes in mesh   : "<<tNumberOfNodes<<std::endl;
   std::vector< Vertex * > tCandidateVertices;

   for ( unsigned int k=0; k<tNumberOfNodes; k++ )
   {
      Vertex * tNode = mMesh.get_vertex( k );

      if ( tNode->is_candidate() )
      {
         tCandidateVertices.push_back(tNode);
      }
      else
      {
         continue;
      }
   }
   std::cout<<"number of candidate nodes : "<<tCandidateVertices.size()<<std::endl;
   return tCandidateVertices;
}

//-------------------------------------------------------------------------------

void Core::get_nodes_withing_bounding_box_of_triangle(
   Triangle                * aTriangle,
   std::vector< Vertex * > & aNodes,
   std::vector< Vertex * > &
   aCandList ) //===========================================
{
   // calculate minimum and maximum coordinate

   mfem::Vector tMinCoord( 3 );
   mfem::Vector tMaxCoord( 3 );

//   std::cout << mData.mBufferDiagonal << " k10bufdiag\n";

   for ( int i=0; i<3; ++i )
   {
      tMinCoord( i )  = aTriangle->get_min_coord( i ) - mData.mBufferDiagonal;
      tMaxCoord( i )  = aTriangle->get_max_coord( i ) + mData.mBufferDiagonal;
   }

   // why is this necessary?

//   for ( int i=0; i<3; ++i )
//   {
//      tMinCoord( i )  = std::max( tMinCoord( i ), mMesh.get_min_coord( i ) );
//      tMaxCoord( i )  = std::min( tMaxCoord( i ), mMesh.get_max_coord( i ) );
//   }

//   std::cout << mMesh.get_max_coord( 0 ) << " " <<
//                mMesh.get_max_coord( 1 ) << " " <<
//                mMesh.get_max_coord( 2 ) << " k101\n";
//   std::cout << mMesh.get_min_coord( 0 ) << " " <<
//                mMesh.get_min_coord( 1 ) << " " <<
//                mMesh.get_min_coord( 2 ) << " k101\n";
//   tMinCoord.Print();
//   tMaxCoord.Print();
//   MFEM_ABORT(" ");

   //            // get number of nodes on this mesh
   //            int tNumberOfNodes = mMesh.get_num_nodes();

   // number of candidate nodes
   int tNumberOfCandidates =
      aCandList.size(); //========================================

   // node counter
   int tCount = 0;

   // loop over all nodes of this mesh
   //            for( int k=0; k<tNumberOfNodes; ++k )
   //            {

   // loop over only the candidate nodes
   for ( int k=0; k<tNumberOfCandidates;
         k++ ) //========================================
   {
      // get pointer to node
      //                Vertex * tNode = mMesh.get_vertex( k );
      Vertex * tNode =
         aCandList[k];  //=================================================

      // unflag this node
      tNode->unflag();

      // test if node is a candidate
      //                if( tNode->is_candidate() )
      //                {
      // get coords of this node
      const mfem::Vector & tPoint = tNode->get_coords();

      // assume that node is in triangle
      bool tNodeIsWithinTriangle = true;

      for ( int i=0; i<3; ++i )
      {
         if ( tPoint( i ) < tMinCoord( i ) || tPoint( i ) > tMaxCoord( i ) )
         {
//             tPoint.Print();
//             for (int d = 0; d < 3; d++)
//             {
//                 std::cout << aTriangle->get_min_coord( d ) << " " <<
//                              aTriangle->get_max_coord( d ) << " k10" << std::endl;
//             }
//             tMinCoord.Print();
//             tMaxCoord.Print();
//             std::cout << " k10f\n";
//             MFEM_ABORT(" ");
            tNodeIsWithinTriangle = false;
            break;
         }
         /* tNodeIsWithinTriangle = tNodeIsWithinTriangle
                 && ( tPoint( i ) >= tMinCoord( i ) )
                 && ( tPoint( i ) <= tMaxCoord( i ) );
         if( ! tNodeIsWithinTriangle )
         {
             break;
         } */
      }

      // if node is in triangle
      if ( tNodeIsWithinTriangle )
      {
         // flag this node
         tNode->flag();

         // increment counter
         ++tCount;
      }
      //                } // end node is candidate
   } // end loop over all nodes

//   std:out << tCount << " k10tcound flagged\n";

   // reset output array
   aNodes.resize( tCount, nullptr );

   // reset counter
   tCount = 0;

   // loop over all nodes of this mesh
   //            for( int k=0; k<tNumberOfNodes; ++k )
   //            {

   // loop over only the candidate nodes+
   for ( int k=0; k<tNumberOfCandidates;
         k++ ) //========================================
   {
      // get pointer to node
      //                Vertex * tNode = mMesh.get_vertex( k );
      Vertex * tNode =
         aCandList[k]; //==================================================

      // test if node is flagged
      if ( tNode->is_flagged() )
      {
         aNodes[ tCount++ ] = tNode;
      }
   }
}

//-------------------------------------------------------------------------------

void Core::sweep()
{
   unsigned int tNumberOfVertices = mMesh.get_num_nodes();

   unsigned int tSweepCount = 1;

   while ( tSweepCount != 0 )
   {
      tSweepCount = 0;

      // loop over all vertices
      for ( unsigned int k=0; k<tNumberOfVertices; ++k )
      {
         // get vertex
         Vertex * tVertex = mMesh.get_vertex( k );

         // test if node has sdf
         if ( tVertex->has_sdf() )
         {
            // sweep this vertex
            tSweepCount += tVertex->sweep();
         }
      }

      if ( mVerbose )
      {

      }
   }
}

// -----------------------------------------------------------------------------

void Core::fill_sdf_with_values( mfem::Vector & aSDF )
{
   // get number of vertices
   unsigned int tNumberOfVertices = mMesh.get_num_nodes();

   // min and max value
   double tMinSDF = -1e-12; //std::numeric_limits<double>::max();
   double tMaxSDF = 1e-12;  //std::numeric_limits<double>::min();

   // allocate matrix
   aSDF.SetSize( tNumberOfVertices );

   // loop over all nodes and write real values
   for ( unsigned int k=0; k<tNumberOfVertices; ++k )
   {
      // get pointer to vertex
      Vertex * tVertex = mMesh.get_vertex( k );

      // test if vertex has SDF
      if ( tVertex->has_sdf() )
      {
         double tSDF =  tVertex->get_sdf();

         // write value
         aSDF( tVertex->get_index() ) = tSDF;

         if ( tVertex->is_inside() )
         {
            tMinSDF = std::min( tMinSDF, tSDF );
         }
         else
         {
            tMaxSDF = std::max( tMaxSDF, tSDF );
         }
      }
   }

   // if parallel, synchronize min and max values for SDF
   if ( mMesh.get_mfem_par_mesh()->GetNRanks() > 1 )
   {
      double tMinGlobCoord;
      double tMaxGlobCoord;
      MPI_Allreduce( &tMinSDF, &tMinGlobCoord, 1, MPI_DOUBLE, MPI_MIN,
                     MPI_COMM_WORLD );
      MPI_Allreduce( &tMaxSDF, &tMaxGlobCoord, 1, MPI_DOUBLE, MPI_MAX,
                     MPI_COMM_WORLD );

      tMinSDF = tMinGlobCoord;
      tMaxSDF = tMaxGlobCoord;
   }

   // loop over all nodes and write fake values
   for ( unsigned int k=0; k<tNumberOfVertices; ++k )
   {
      // get pointer to vertex
      Vertex * tVertex = mMesh.get_vertex( k );

      // test if vertex does not have an SDF
      if ( ! tVertex->has_sdf() )
      {
         if ( tVertex->is_inside() )
         {
            aSDF( tVertex->get_index() ) = tMinSDF;
         }
         else
         {
            aSDF( tVertex->get_index() ) = tMaxSDF;
         }
      }
   }

   //aSDF.Print();
}

// -----------------------------------------------------------------------------

void Core::save_to_vtk( const std::string & aFilePath )
{

}
// -----------------------------------------------------------------------------

void Core::save_unsure_to_vtk( const std::string & aFilePath )
{

}

// -----------------------------------------------------------------------------

void Core::force_unsure_nodes_outside()
{
   // get number of nodes on mesh
   unsigned int tNumberOfNodes = mMesh.get_num_nodes();

   // loop over all nodes
   for ( unsigned int k=0; k<tNumberOfNodes; ++k )
   {
      // get pointer to node
      mMesh.get_vertex( k )->unflag();
   }

   mData.mUnsureNodesCount = 0;
}

// -----------------------------------------------------------------------------

void Core::random_rotation()
{
   // generate random angle
   double tAngle = random_angle();

   // generate random normalized axis
   mfem::Vector tAxis = random_axis();

   // generate rotation matrix
   mfem::DenseMatrix tRotation = rotation_matrix( tAxis, tAngle );

   // rotate all vertices of triangle mesh
   for ( Triangle_Vertex * tVertex : mData.mVertices )
   {
      tVertex->rotate_node_coords( tRotation );
   }

   // update all triangles
   for ( Triangle * tTriangle : mData.mTriangles )
   {
      tTriangle->update_data();
   }

   // rotate unsure nodes
   unsigned int tNumberOfNodes = mMesh.get_num_nodes();

   // loop over all nodes
   for ( unsigned int k=0; k< tNumberOfNodes; ++k )
   {
      // test if node is unsore
      if ( mMesh.get_vertex( k )->is_flagged() )
      {
         mMesh.get_vertex( k )->rotate_coords( tRotation );
      }
   }
}

void Core::undo_rotation()
{
   // rotate all vertices of triangle mesh
   for ( Triangle_Vertex * tVertex : mData.mVertices )
   {
      tVertex->reset_node_coords();
   }

   // update all triangles
   for ( Triangle * tTriangle : mData.mTriangles )
   {
      tTriangle->update_data();
   }

   // rotate unsure nodes
   unsigned int tNumberOfNodes = mMesh.get_num_nodes();

   // loop over all nodes
   for ( unsigned int k=0; k< tNumberOfNodes; ++k )
   {
      mMesh.get_vertex( k )->reset_coords();
   }
}

// -----------------------------------------------------------------------------

Data::Data( Object & aObject ) :
   mTriangles( aObject.get_triangles() ),
   mVertices( aObject.get_vertices() ),
   mNumberOfTriangles( mTriangles.size() ),
   mTriangleMinCoordsX(mNumberOfTriangles),
   mTriangleMinCoordsY(mNumberOfTriangles),
   mTriangleMinCoordsZ(mNumberOfTriangles),
   mTriangleMaxCoordsX(mNumberOfTriangles),
   mTriangleMaxCoordsY(mNumberOfTriangles),
   mTriangleMaxCoordsZ(mNumberOfTriangles),
   mCandJ(mNumberOfTriangles),
   mCandidateTriangles(mNumberOfTriangles)

{
   this->init_triangles();
//    for ( unsigned int k=0; k<mNumberOfTriangles; ++k )
//    {
//       // get pointer to triangle
//       Triangle * tTriangle = mTriangles[ k ];
//       for (int d = 0; d < 3; d++)
//       {
//           std::cout << k << " " << d << " " << tTriangle->get_min_coord( d ) << " " <<
//                        tTriangle->get_max_coord( d ) << " k10info" << std::endl;
//       }
//    }
}

void Data::init_triangles()
{
   // copy triangle bounding box data
   for ( int k = 0; k < mNumberOfTriangles; ++k)
   {
      // minimum triangle coordinates for lower left point of bounding box
      mTriangleMinCoordsX( k )
         = mTriangles[ k ]->get_min_coord( 0 );

      mTriangleMinCoordsY( k )
         = mTriangles[ k ]->get_min_coord( 1 );

      mTriangleMinCoordsZ( k )
         = mTriangles[ k ]->get_min_coord( 2 );

      // maximum triangle coordinates for upper right point of bounding box
      mTriangleMaxCoordsX( k )
         = mTriangles[ k ]->get_max_coord( 0 );

      mTriangleMaxCoordsY( k )
         = mTriangles[ k ]->get_max_coord( 1 );

      mTriangleMaxCoordsZ ( k )
         = mTriangles[ k ]->get_max_coord( 2 );
   }

//   std::cout << mTriangleMinCoordsX.Min() << " " <<
//                mTriangleMinCoordsY.Min() << " " <<
//                mTriangleMinCoordsZ.Min() << " " <<
//                mTriangleMinCoordsX.Max() << " " <<
//                mTriangleMinCoordsY.Max() << " " <<
//                mTriangleMinCoordsZ.Max() << " k10c\n";
}


//-------------------------------------------------------------------------------

SDF_Generator::SDF_Generator(
   const std::string & aObjectPath,
   const bool aVerboseFlag ) :
   mObject( aObjectPath ),
   mVerboseFlag ( aVerboseFlag )
{ }

SDF_Generator::SDF_Generator(
   const std::string & aObjectPath,
   const mfem::Vector  aObjectOffset,
   const bool aVerboseFlag  ) :
   mObject( aObjectPath, aObjectOffset ),
   mVerboseFlag ( aVerboseFlag )
{ }

SDF_Generator::SDF_Generator(
   const std::string & aObjectPath,
   const bool aVerboseFlag,
   double scalefac) :
   mObject( aObjectPath, scalefac ),
   mVerboseFlag ( aVerboseFlag )
{ }

void SDF_Generator::DoAmrOnMeshBasedOnIntersections(mfem::ParMesh *pmesh,
                                                    int neighbors)
{
   mfem::sdf::Mesh tMesh( pmesh, mVerboseFlag );

   // create data container
   Data tData( mObject );

   // create core
   Core tCore( tMesh, tData, mVerboseFlag );
   tCore.set_candidate_search_depth(neighbors);

   // calculate SDF
   tCore.calculate_raycast( );



   mfem::L2_FECollection l2fec(0, pmesh->Dimension());
   mfem::ParFiniteElementSpace l2fespace(pmesh, &l2fec);
   mfem::ParGridFunction el_to_refine(&l2fespace);

   mfem::H1_FECollection lhfec(1, pmesh->Dimension());
   mfem::ParFiniteElementSpace lhfespace(pmesh, &lhfec);
   mfem::ParGridFunction lhx(&lhfespace);

   el_to_refine = 0.0;
   Array<int> refs;
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
       Cell * tElement = tMesh.get_cell( e );
       if (tElement->is_flagged())
       {
           el_to_refine(e) = 1.0;
       }
   }

   for (int inner_iter = 0; inner_iter < neighbors; inner_iter++)
   {
       el_to_refine.ExchangeFaceNbrData();
       GridFunctionCoefficient field_in_dg(&el_to_refine);
       lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);
       for (int e = 0; e < pmesh->GetNE(); e++)
       {
          Array<int> dofs;
          Vector x_vals;
          lhfespace.GetElementDofs(e, dofs);
          const IntegrationRule &ir = lhfespace.GetFE(e)->GetNodes();
//             irRules.Get(pmesh->GetElementGeometry(e), 1);
          lhx.GetValues(e, ir, x_vals);
          double max_val = x_vals.Max();
          if (max_val > 0)
          {
             el_to_refine(e) = 1.0;
          }
       }
   }

   Array<int> el_to_refine_list;
   for (int e = 0; e < el_to_refine.Size(); e++)
   {
      if (el_to_refine(e) > 0.0)
      {
         el_to_refine_list.Append(e);
      }
   }
//   std::cout << el_to_refine_list.Size() << " k10refsize\n";

   pmesh->GeneralRefinement(el_to_refine_list);
//   MFEM_ABORT(" ");

   //tCore.save_to_vtk( "sdf_mesh.vtk");
}

void SDF_Generator::calculate_sdf( mfem::ParGridFunction & aSDFGridFunc )
{
   mfem::sdf::Mesh tMesh( aSDFGridFunc.ParFESpace()->GetParMesh(), mVerboseFlag );

   // create data container
   Data tData( mObject );

   // create core
   Core tCore( tMesh, tData, mVerboseFlag );

   // calculate SDF
   tCore.calculate_raycast_and_sdf( aSDFGridFunc );

   //tCore.save_to_vtk( "sdf_mesh.vtk");
}

} /* namespace sdf */
} /* namespace mfem */
