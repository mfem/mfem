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

#ifndef MFEM_SDF_OBJECT_HPP
#define MFEM_SDF_OBJECT_HPP

#include "mfem.hpp"

namespace mfem
{
namespace sdf
{
class Triangle_Vertex;

class Triangle
{
   // index of this triangle
   const int mIndex;

   // cells with vertex pointers
   std::vector< Triangle_Vertex* > mVertices;
   std::vector< Triangle* >        mNeighbors;

   struct BarycentricData
   {
      mfem::DenseMatrix  mLocalEdgeDirectionVectors;
      mfem::Vector       mLocalEdgeInverseMagnitudes;
      mfem::DenseMatrix  mProjectionMatrix;
      mfem::DenseMatrix  mLocalNodeCoordsInPlane;
      double             mTwiceArea;
      double             mInvTwiceArea;
      BarycentricData()
         : mLocalEdgeDirectionVectors( 3, 3 )
         , mLocalEdgeInverseMagnitudes( 3 )
         , mProjectionMatrix( 3, 3 )
         , mLocalNodeCoordsInPlane( 2, 3 ) {};
      ~BarycentricData() = default;
   };

   BarycentricData mBarycentric;

   // container for node coordinates
   mfem::DenseMatrix mNodeCoords;

   // container for node indices
   mfem::Vector mNodeIndices;

   // container for center
   mfem::Vector mCenter;

   // container for normal
   mfem::Vector mNormal;

   double mHesse;

   mfem::DenseMatrix mPredictY;
   mfem::DenseMatrix mPredictYRA;
   mfem::DenseMatrix mPredictYRB;

   mfem::Vector mMinCoord;
   mfem::Vector mMaxCoord;

   bool mFlag = false;

public:

   Triangle( int                      aIndex,
             std::vector< Triangle_Vertex* >& aVertices );

   ~Triangle() {};

   void update_data();

   /**
    * @brief returns the minimum coordinate of the triangle
    *
    * @param[in] aDimension   0: x-coordinate
    *                         1: y-coordinate
    *                         2: z-coordinate
    */
   double get_min_coord( int aDimension ) const
   {
      return mMinCoord( aDimension );
   }

   /**
    * @brief returns the maximum coordinate of the triangle
    *
    * @param[in] aDimension   0: x-coordinate
    *                         1: y-coordinate
    *                         2: z-coordinate
    */
   double  get_max_coord( int aDimension ) const
   {
      return mMaxCoord( aDimension );
   }

   /**
    * @brief Returns the hesse distance of the plane describing the triangle
    */
   double get_hesse() const
   {
      return mHesse;
   }

   /**
    * @brief returns the normal vector of the triangle
    */
   const  mfem::Vector& get_normal() const
   {
      return mNormal;
   }

   /**
    * @brief returns the center of the triangle
    */
   const mfem::Vector& get_center() const
   {
      return mCenter;
   }

   /**
    * @brief returns the area of the triangle
    */
   double get_area() const
   {
      return 0.5 * mBarycentric.mTwiceArea;
   }

   /**
    * @brief intersects the line
    *
    *        g(i) = aPoint(i) + tParam*kronecker(i,aAxis)
    *        with the triangle and returns the coordinate of the axis
    * @param[in] aPoint
    * @param[in] aAxis
    */
   void intersect_with_coordinate_axis(
      const mfem::Vector & aPoint,
      const int            aAxis,
      double             & aCoordinate,
      bool               & aError );

   //-------------------------------------------------------------------------------

   bool check_edge(
      const unsigned int   aEdge,
      const unsigned int   aAxis,
      const mfem::Vector & aPoint );

   /**
    * @brief Projects a point on the local 2D coordinate system.
    *        The third entry contains a signed point-plane distance.
    * @param[in]  aPoint  point to project
    */
   mfem::Vector project_point_to_local_cartesian(
      const mfem::Vector& aPoint );

   /**
    * @brief Returns the barycentric coordinates for a point.
    *       Point must be in local cartesian coordinates.
    *
    * @param[in]  aLocalPoint  point to project
    */
   mfem::Vector
   get_barycentric_from_local_cartesian(
      const mfem::Vector& aLocalPoint );

   /**
    * @brief Returns the distance of a Point to an edge.
    *         Point must be in local cartesian coordinates.
    * @param[in]  aLocalPoint  point to be considered
    * @param[in]  aEdge        edge to be considered
    */
   double distance_point_to_edge_in_local_cartesian(
      const mfem::Vector& aLocalPoint,
      const unsigned int           aEdge );

   /**
    * @brief Returns the distance of a Point to the triangle.
    * @param[in]  aPoint  point to be considered
    *
    */
   double get_distance_to_point(const mfem::Vector & aPoint );

   int get_id() const
   {
      return mIndex + 1;
   }

   int  get_index() const
   {
      return mIndex;
   }

   int get_number_of_vertices() const
   {
      return 3;
   }

   int get_owner() const
   {
      return 0;
   }

   std::vector< Triangle_Vertex* > get_vertex_pointers() const;

   void remove_vertex_pointer( int aIndex );

   mfem::Vector get_vertex_ids() const;

   mfem::Vector get_vertex_inds() const;

   mfem::DenseMatrix get_vertex_coords() const;

   // mtk::Geometry_Type get_geometry_type() const
   // {
   //     return mtk::Geometry_Type::TRI;
   // }

   // mtk::Interpolation_Order get_interpolation_order() const
   // {
   //     return mtk::Interpolation_Order::LINEAR;
   // }

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

private:
   void copy_node_coords_and_inds( std::vector< Triangle_Vertex* >& aVertices );

   void calculate_hesse_normal_form( mfem::Vector & aDirectionOfEdge );

   void calculate_barycectric_data( const mfem::Vector & aDirectionOfEdge );

   void calculate_prediction_helpers();

}; // class Triangle
//-------------------------------------------------------------------------------

class Triangle_Vertex
{
   const int  mIndex;
   mfem::Vector       mNodeCoords;
   mfem::Vector mOriginalNodeCoords;

public:

   Triangle_Vertex( const int        aIndex,
                    const mfem::Vector & aNodeCoords );

   ~Triangle_Vertex() {};

   void rotate_node_coords( const DenseMatrix & aRotationMatrix );

   void reset_node_coords();

   mfem::Vector& get_coords()
   {
      return mNodeCoords;
   }

   void set_node_coords(Vector &coords)
   {
       mOriginalNodeCoords = coords;
      mNodeCoords = coords;
   }

   int get_id() const
   {
      return mIndex + 1;
   }

   int get_index() const
   {
      return mIndex;
   }
};

//-------------------------------------------------------------------------------

class Object
{
   double                            mMeshHighPass=1e-9;
   std::vector< Triangle_Vertex * >  mVertices;
   std::vector< Triangle * >         mTriangles;

   mfem::Vector mOffsets;

public:

   Object ( const std::string & aFilePath );

   Object ( const std::string & aFilePath,
            const mfem::Vector  aOffsets );

   Object ( const std::string & aFilePath, double scalefac );

   ~Object();

   std::vector< Triangle * > & get_triangles()
   {
      return mTriangles;
   }

   std::vector< Triangle_Vertex * > & get_vertices()
   {
      return mVertices;
   }

   mfem::Vector get_nodes_connected_to_element_loc_inds ( int aElementIndex )
   const;

private:

   /**
    * loads an ascii file and creates vertex and triangle objects
    */
   void
   load_from_object_file( const std::string& aFilePath, double scalefac = 1.0 );

   /**
    * loads an ascii file and creates vertex and triangle objects
    */
   void
   load_from_stl_file( const std::string& aFilePath );

   /**
    * loads an ASCII file into a buffer of strings.
    * Called through construction.
    */
   void
   load_ascii_to_buffer( const std::string& aFilePath,
                         std::vector<std::string>& aBuffer);
};
} /* namespace sdf */
} /* namespace mfem */

#endif
