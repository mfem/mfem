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
#include "SDF_Generator.hpp"

namespace mfem
{
namespace sdf
{
Vertex::Vertex( const int aIndex, double * aNodeCoords ) :
   mIndex( aIndex ),
   mNodeCoords( 3 ),
   mOriginalNodeCoords( 3 )
{
   // convert dynamic array to fixed array
   for ( int k=0; k<3; ++k )
   {
      mOriginalNodeCoords( k ) = aNodeCoords[ k ];
   }

   this->reset_coords();
}

void Vertex::update_udf( Triangle * aTriangle )
{
   // calculate distance to this point
   double tDistance = aTriangle->get_distance_to_point( mNodeCoords );

   std::cout<<"tDistance " <<tDistance<<std::endl;

   if ( tDistance < mSDF )
   {
      // remember value
      mSDF = tDistance;

      // set sdf flag
      mHasSDF = true;

      // remember triangle
      mClosestTriangle = aTriangle;
   }
}

void Vertex::insert_cell( mfem::sdf::Cell * aCell )
{
   mCells[ mCellCounter++ ] = aCell;
}

unsigned int Vertex::sweep()
{
   bool tSwept = false;

   // loop over all neighbors
   for ( Vertex * tNeighbor : mNeighbors )
   {
      // get pointer to triangle
      Triangle * tTriangle = tNeighbor->get_closest_triangle();

      if ( tTriangle != NULL )
      {
         // get distance to triangle of neighbor
         double tDistance
            = tTriangle->get_distance_to_point( mNodeCoords );

         if ( tDistance < mSDF )
         {
            tSwept = true;
            mSDF = tDistance;
            mClosestTriangle = tTriangle;
         }
      }
   }

   if ( tSwept )
   {
      return 1;
   }
   else
   {
      return 0;
   }
}

void Vertex::rotate_coords( const mfem::DenseMatrix & aRotationMatrix )
{
   aRotationMatrix.Mult(mOriginalNodeCoords,mNodeCoords);
}

void Vertex::reset_coords()
{
   mNodeCoords = mOriginalNodeCoords;
}

// -----------------------------------------------------------------------------

Cell::Cell(
   const int aIndex,
   const mfem::Array<int> & aIndices,
   std::vector< Vertex * >  & aAllVertices ) :
   mIndex( aIndex ),
   mID(0)
{
   // get number of vertices
   int tNumberOfVertices = aIndices.Size();

   // allocate member cell
   mVertices.resize( tNumberOfVertices, nullptr );

   // populate cell
   for ( int k=0; k<tNumberOfVertices; ++k )
   {
      mVertices[ k ] = aAllVertices[ aIndices[ k ] ];
   }
}

double Cell::get_buffer_diagonal()
{
   // calculate min and max values of this cell
   mfem::Vector tMinValue  = mVertices[ 0 ]->get_coords();
   mfem::Vector tMaxValue = mVertices[ 0 ]->get_coords();

   // get number of nodes
   int tNumberOfNodes = mVertices.size();

   // loop over all n
   for ( int k=1; k< tNumberOfNodes; ++k )
   {
      // get coordinates of node
      const mfem::Vector & tCoords =  mVertices[ k ]->get_coords();

      // get min and max coords
      for ( int i=0; i<3; ++i )
      {
         tMinValue( i ) = std::min( tMinValue( i ), tCoords( i ) );
         tMaxValue( i ) = std::max( tMaxValue( i ), tCoords( i ) );
      }
   }

   double aNorm = 0.0;
   for ( int i=0; i<3; ++i )
   {
      aNorm += std::pow( tMaxValue( i ) - tMinValue( i ), 2 );
   }
   return std::sqrt( aNorm );
}

// -----------------------------------------------------------------------------

Mesh::Mesh( mfem::ParMesh * aMesh, bool aVerbose ) :
   mMesh( aMesh ),
   mVerbose( aVerbose ),
   mMinCoord( 3 ),
   mMaxCoord( 3 )
{
   // determine interpolation order of mesh
   // pick first element
   int tNumVertices
      = aMesh->GetElement(0)->GetNVertices ();
   // determine interpolation order of mesh
   switch ( tNumVertices )
   {
      case ( 4 ) : // tet4
      case ( 6 ) : // penta 6
      case ( 8 ) : // hex8
      {
         mOrder = 1;
         break;
      }
      case ( 10 ) : // tet10
      case ( 20 ) : // hex20
      case ( 27 ) : // hex27
      {
         mOrder = 2;
         break;
      }
      default :
      {
         mfem_error( "Can't determine order of 3D cell");
      }
   }

   // get number of nodes GetNE ()
   unsigned int tNumberOfNodes = aMesh->GetNV() ;

   // reserve memory
   mVertices.resize( tNumberOfNodes, nullptr );

   // populate container
   for ( unsigned int k=0; k<tNumberOfNodes; ++k )
   {
      mVertices[ k ] = new Vertex( k, aMesh->GetVertex(k) );
   }

   // get number of elements
   unsigned int tNumberOfElements = aMesh->GetNE ();

   // reserve memory
   mCells.resize( tNumberOfElements, nullptr );

   // populate container
   for ( unsigned int k=0; k<tNumberOfElements; ++k )
   {
      mfem::Array<int> tVertexInds;
      aMesh->GetElement(k)->GetVertices(tVertexInds);

      // get cell indices from mesh
      mCells[ k ] = new Cell(
         k,
         tVertexInds,
         mVertices );
   }

   this->link_vertex_cells();
   this->link_vertex_neighbors();

   // find min and max coordinate
   for ( unsigned int i=0; i<3; ++i )
   {
      mMinCoord( i ) = std::numeric_limits<double>::max();
   }
   for ( unsigned int i=0; i<3; ++i )
   {
      mMaxCoord( i ) = std::numeric_limits<double>::min();
   }

   for ( Vertex * tVertex : mVertices )
   {
      mfem::Vector tPoint = tVertex->get_coords();
      for ( unsigned int i=0; i<3; ++i )
      {
         mMinCoord( i ) = std::min( tPoint( i ), mMinCoord( i ) );
         mMaxCoord( i ) = std::max( tPoint( i ), mMaxCoord( i ) );
      }
   }
//   mMinCoord.Print();
//   mMaxCoord.Print();

//   aMesh->GetBoundingBox(mMinCoord,mMaxCoord);
//   mMinCoord.Print();
//   mMaxCoord.Print();
//   std::cout << " internal mesh minmax coord\n";
//   MFEM_ABORT(" ");

   for ( unsigned int i=0; i<3; ++i )
   {
      double tMin = mMinCoord( i );
      double tMax = mMaxCoord( i );

      double tMinGlobCoord;
      double tMaxGlobCoord;
      MPI_Allreduce( &tMin, &tMinGlobCoord, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
      MPI_Allreduce( &tMax, &tMaxGlobCoord, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

      mMinCoord( i ) = tMinGlobCoord;
      mMaxCoord( i ) = tMaxGlobCoord;
   }

   // matrix that contains node IDs
   mNodeIDs.SetSize( tNumberOfNodes );
   for ( unsigned int k=0; k<tNumberOfNodes; ++k )
   {
      // mNodeIDs( k )
      //         = aMesh->get_glb_entity_id_from_entity_loc_index( k, EntityRank::NODE );
   }
}

Mesh::Mesh( std::shared_ptr< mfem::ParMesh > aMesh,
            bool aVerbose ) : Mesh ( aMesh.get(), aVerbose )
{

}

Mesh::~Mesh()
{
   // delete cell pointers
   for ( Cell * tCell : mCells )
   {
      delete tCell;
   }

   // delete vertex pointers
   for ( Vertex * tVertex : mVertices )
   {
      delete tVertex;
   }
}

void Mesh::link_vertex_cells()
{
   // count elements per node
   for ( Cell * tCell : mCells )
   {
      // get vertex pointers
      std::vector< Vertex* > tVertices = tCell->get_vertices();

      // count elements connected to this nodes
      for ( Vertex * tVertex : tVertices )
      {
         tVertex->increment_cell_counter();
      }
   }

   // initialize cell containers
   for ( Vertex * tVertex : mVertices )
   {
      tVertex->init_cell_container();
   }

   // insert cells
   for ( Cell * tCell : mCells )
   {
      // get vertex pointers
      std::vector< Vertex* > tVertices = tCell->get_vertices();

      // count elements connected to this nodes
      for ( Vertex * tVertex : tVertices )
      {
         tVertex->insert_cell( tCell );
      }
   }
}

void Mesh::link_vertex_neighbors()
{
   // loop over all vertices
   for ( Vertex * tVertex : mVertices )
   {
      // initialize counter
      unsigned int tCount = 0;

      // get nuber of cells
      unsigned int tNumberOfCells = tVertex->get_number_of_cells();

      // count number of cells
      for ( unsigned int k=0; k<tNumberOfCells; ++k )
      {
         tCount += tVertex->get_cell( k )->get_number_of_vertices();
      }

      // allocate array with indices
      mfem::Array<int> tIndices( tCount );

      // reset counter
      tCount = 0;
      for ( unsigned int k=0; k<tNumberOfCells; ++k )
      {
         // get pointer to cell
         Cell * tCell = tVertex->get_cell( k );

         // get number of vertices per element
         unsigned int tNumberOfVertices = tCell->get_number_of_vertices();

         // loop over all vertices
         for ( unsigned int i=0; i<tNumberOfVertices; ++i )
         {
            tIndices[ tCount++ ] = tCell->get_vertex( i )->get_index();
         }
      }

      // make indices unique
      tIndices.Sort();
      tIndices.Unique();

      // get my index
      unsigned int tIndex = tVertex->get_index();

      // reset counter
      tCount = 0;

      int tNumberOfVertices = tIndices.Size();

      // init container
      tVertex->init_neighbor_container( tNumberOfVertices-1 );

      // loop over all indices
      for ( int i=0; i<tNumberOfVertices; ++i )
      {
         // test that this is not me
         if ( tIndices[ i ] != tIndex )
         {
            // insert neighbor
            tVertex->insert_neighbor( mVertices[ tIndices[ i ] ], tCount++ );
         }
      }
   }
}

} /* namespace sdf */
} /* namespace mfem */
