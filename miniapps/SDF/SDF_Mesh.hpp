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

#ifndef MFEM_SDF_MESH_HPP
#define MFEM_SDF_MESH_HPP

#include "mfem.hpp"

namespace mfem
{
namespace sdf
{
class Cell;

class Triangle;

// -----------------------------------------------------------------------------
/**
 * The sdf vertex is a wrapper around an MTK vertex.
 * It contains a pointer to the MTK vertex and
 * has the ability to flag nodes
 */
class Vertex
{
   //! index
   const int      mIndex;

   //! flag telling if vertex is inside
   bool                mIsInside = false;

   //! flag telling if an SDF has been calculated for this vertex
   bool                mHasSDF = false;

   bool                mIsCandidate = false;

   bool                mFlag = true;

   // current node coords
   mfem::Vector   mNodeCoords;
   mfem::Vector   mOriginalNodeCoords;

   double                mSDF;
   Triangle *          mClosestTriangle = nullptr;

   unsigned int                mCellCounter = 0;

   std::vector< Cell * > mCells;

   std::vector< Vertex * > mNeighbors;
public:

   /**
    * constructor
    */
   Vertex( const int aIndex,
           double * aNodeCoordss );
   // -----------------------------------------------------------------------------

   /**
    * destructor
    */
   ~Vertex()
   {
      mCells.clear();
      mNeighbors.clear();
   };

   // -----------------------------------------------------------------------------

   const mfem::Vector & get_coords() const
   {
      return mNodeCoords;
   }

   // -----------------------------------------------------------------------------

   void set_inside_flag()
   {
      mIsInside = true;
   }

   void unset_inside_flag()
   {
      mIsInside = false;
   }

   bool is_inside() const
   {
      return mIsInside;
   }

   void set_candidate_flag()
   {
      mIsCandidate = true;
   }

   void unset_candidate_flag()
   {
      mIsCandidate = false;
   }

   bool is_candidate() const
   {
      return mIsCandidate;
   }

   void set_sdf_flag()
   {
      mHasSDF = true;
   }

   void unset_sdf_flag()
   {
      mHasSDF = false;
   }

   bool has_sdf() const
   {
      return mHasSDF;
   }

   int get_index() const
   {
      return mIndex;
   }

   void flag()
   {
      mFlag = true;
   }

   void unflag()
   {
      mFlag = false;
   }

   bool is_flagged() const
   {
      return mFlag;
   }

   void reset()
   {
      mHasSDF = false;
      mIsCandidate = false;
      mFlag = true;
      mSDF =  std::numeric_limits<double>::max();
      mClosestTriangle = nullptr;
      mIsInside = false;
   }

   void update_udf( Triangle *  aTriangle );

   void increment_cell_counter()
   {
      ++mCellCounter;
   }

   void init_cell_container()
   {
      mCells.resize( mCellCounter, nullptr );
      mCellCounter = 0;
   }

   void insert_cell( Cell * aCell );

   unsigned int get_number_of_cells() const
   {
      return mCellCounter;
   }

   Cell * get_cell( const unsigned int aIndex )
   {
      return mCells[ aIndex ];
   }

   void init_neighbor_container( const unsigned int aNumberOfNeighbors )
   {
      mNeighbors.resize( aNumberOfNeighbors, nullptr );
   }

   void insert_neighbor( Vertex * aNeighbor, const unsigned int aNeighborIndex )
   {
      mNeighbors[ aNeighborIndex ] = aNeighbor;
   }

   unsigned int get_number_of_neighbors() const
   {
      return mNeighbors.size();
   }

   Vertex * get_neighbor( const unsigned int aNeighborIndex )
   {
      return mNeighbors[ aNeighborIndex ];
   }

   Triangle * get_closest_triangle()
   {
      return mClosestTriangle;
   }

   unsigned int sweep();

   double get_sdf() const
   {
      if ( mIsInside )
      {
         return -mSDF;
      }
      else
      {
         return mSDF;
      }
   }

   void rotate_coords( const mfem::DenseMatrix & aRotationMatrix );

   void reset_coords();

};

//--------------------------------------------------------------------------
/**
 * a wrapper for the MTK cell
 */
class Cell
{
   const int mIndex;
   const int mID;

   // cell with MTK vertices
   std::vector< Vertex * > mVertices;

   // flag telling if element is in volume
   bool mElementIsInVolume = false;

   // flag telling if element is in surface
   bool mElementIsOnSurface= false;

   // general purpose flag
   bool mFlag = false;

public:

   Cell(   const int aIndex,
           const mfem::Array<int> & aIndices,
           std::vector< Vertex * >  & aAllVertices );

   unsigned int get_number_of_vertices() const
   {
      return mVertices.size();
   }

   Vertex * get_vertex( const unsigned int & aIndex )
   {
      return mVertices[ aIndex ];
   }

   std::vector< Vertex * > & get_vertices()
   {
      return mVertices;
   }

   void set_volume_flag()
   {
      mElementIsInVolume = true;
   }

   void unset_volume_flag()
   {
      mElementIsInVolume = false;
   }

   bool is_in_volume() const
   {
      return mElementIsInVolume;
   }

   void set_surface_flag()
   {
      mElementIsOnSurface = true;
   }

   void unset_surface_flag()
   {
      mElementIsOnSurface = false;
   }

   bool is_on_surface() const
   {
      return mElementIsOnSurface;
   }

   void flag()
   {
      mFlag = true;
   }

   void unflag()
   {
      mFlag = false;
   }

   bool is_flagged() const
   {
      return mFlag;
   }

   int get_index()
   {
      return mIndex;
   }

   int get_id()
   {
      mfem_error("not implemented");
      return mID;
   }

   double get_buffer_diagonal();
};

//------------------------------------------------------------------------------

/**
 * Wrapper around an MTK mesh
 */
class Mesh
{
   //! pointer to underlying mesh
   mfem::ParMesh * mMesh;

   //! vector with SDF Vertices
   std::vector< Vertex * > mVertices;

   //! vector with SDF Cells
   std::vector< Cell * > mCells;

   bool mVerbose;

   mfem::Vector mMinCoord;
   mfem::Vector mMaxCoord;

   mfem::Vector mNodeIDs;

   // interpolation order
   int mOrder;

public:

   /**
    * constructor
    */
   Mesh( std::shared_ptr< mfem::ParMesh > aMesh, bool aVerbose = false );

   Mesh( mfem::ParMesh * aMesh, bool aVerbose = false );

   /**
    * destructor
    */
   ~Mesh();

   /**
    * expose mesh pointer
    */
   mfem::ParMesh * get_mfem_par_mesh()
   {
      return mMesh;
   }

   unsigned int get_num_nodes() const
   {
      return mMesh->GetNV();
   }

   unsigned int get_num_elems() const
   {
      return mMesh->GetNE();
   }

   const mfem::Vector & get_node_coordinate( const unsigned int aIndex ) const
   {
      return mVertices[ aIndex ]->get_coords();
   }

   Vertex * get_vertex( const unsigned int & aIndex )
   {
      return mVertices[ aIndex ];
   }

   Cell * get_cell( const unsigned int & aIndex )
   {
      return mCells[ aIndex ];
   }

   double get_min_coord( const unsigned int aIndex ) const
   {
      return mMinCoord( aIndex );
   }

   double get_max_coord( const unsigned int aIndex ) const
   {
      return mMaxCoord( aIndex );
   }

   bool is_verbose() const
   {
      return mVerbose;
   }

   const mfem::Vector & get_node_ids() const
   {
      mfem_error("not implemented");
      return mNodeIDs;
   }

   /**
    * return the interpolation order of the mesh.
    * Needer for HDF5 output.
    * ( taken from first element on mesh, assuming that all elements
    *   are of the same order )
    */
   int get_order() const
   {
      return mOrder;
   }

private:

   void link_vertex_cells();

   void link_vertex_neighbors();
};
} /* namespace sdf */
} /* namespace mfem */

#endif
