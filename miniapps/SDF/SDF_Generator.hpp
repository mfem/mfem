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

#ifndef MFEM_SDF_HPP
#define MFEM_SDF_HPP

#include "mfem.hpp"
#include "SDF_Object.hpp"
#include "SDF_Mesh.hpp"
//#include "SDF_Tools.hpp"

namespace mfem
{
namespace sdf
{

//-------------------------------------------------------------------------------

struct Data
{
   //! cell with triangles
   std::vector< Triangle * >        & mTriangles;

   std::vector< Triangle_Vertex * > & mVertices;

   const unsigned int
   mNumberOfTriangles;            // !< number of triangles in object

   //!< counter for unsure nodes in voxelizing algorithm
   unsigned int                 mUnsureNodesCount;

   mfem::Vector
   mTriangleMinCoordsX;    //!< min coordinate x of triangle bounding box
   mfem::Vector
   mTriangleMinCoordsY;    //!< min coordinate y of triangle bounding box
   mfem::Vector
   mTriangleMinCoordsZ;    //!< min coordinate z of triangle bounding box
   mfem::Vector
   mTriangleMaxCoordsX;    //!< max coordinate x of triangle bounding box
   mfem::Vector
   mTriangleMaxCoordsY;    //!< max coordinate y of triangle bounding box
   mfem::Vector
   mTriangleMaxCoordsZ;    //!< max coordinate x of triangle bounding box
   mfem::Array<unsigned int>
   mCandJ;    //!< temporary variable needed for triangle preselection

   mfem::Vector
   mCoordsK;                //!< temporary variable needed for voxelizing
   mfem::Array<unsigned int> mCandidateTriangles;

   std::vector< Triangle * > mIntersectedTriangles;

   double mBufferDiagonal;

   // counter for volume elements
   unsigned int mVolumeElements = 0;

   // counter for surface elements
   unsigned int mSurfaceElements = 0;

public :

   Data( Object & aObject );

   //            void GetXYZBounds(Vector &xyzbounds);

   int GetNVolumeElements() { return mVolumeElements; }
   int GetNSurfaceElements() { return mSurfaceElements; }
   int GetNTriangles() { return mNumberOfTriangles; }
   Vector &GetMinCoordX() { return mTriangleMinCoordsX; }
   Vector &GetMinCoordY() { return mTriangleMinCoordsY; }
   Vector &GetMinCoordZ() { return mTriangleMinCoordsZ; }
   Vector &GetMaxCoordX() { return mTriangleMaxCoordsX; }
   Vector &GetMaxCoordY() { return mTriangleMaxCoordsY; }
   Vector &GetMaxCoordZ() { return mTriangleMaxCoordsZ; }

   ~Data() {};

private:

   void init_triangles ();

};

//-------------------------------------------------------------------------------

class Core
{
   mfem::sdf::Mesh & mMesh;
   mfem::sdf::Data & mData;

   unsigned int      mCandidateSearchDepth = 1;
   double            mCandidateSearchDepthEpsilon = 0.01;
   bool              mVerbose;

public :

   Core( mfem::sdf::Mesh & aMesh,
         mfem::sdf::Data & aData,
         bool              aVerbose=false );

   ~Core() {};

   void set_candidate_search_depth( const unsigned int aCandidateSearchDepth )
   {
      mCandidateSearchDepth = aCandidateSearchDepth;
   }

   void set_candidate_search_epsilon( const double aCandidateSearchEpsilon )
   {
      mCandidateSearchDepthEpsilon = aCandidateSearchEpsilon;
   }

   void calculate_raycast();

   void calculate_raycast( mfem::Array<int> & aElementsAtSurface );

   void calculate_raycast(
      mfem::Array<int> & aElementsAtSurface,
      mfem::Array<int> & aElementsInVolume );

   void calculate_raycast_and_sdf( mfem::Vector & aSDF );

   void calculate_raycast_and_sdf(
      mfem::Vector    & aSDF,
      mfem::Array<int> & aElementsAtSurface,
      mfem::Array<int> & aElementsInVolume );

   void save_to_vtk( const std::string & aFilePath );

   void save_unsure_to_vtk( const std::string & aFilePath );

   void calculate_intersecting_triangles();

private :

   void voxelize( const unsigned int aAxis );

   std::vector< Vertex * > set_candidate_list(  );

   void calculate_udf( std::vector< Vertex * > & aCandidateList);

   /**
    * Kehrwoche :
    * make sure that each vertex is really associated
    * to its closest triangle
    */
   void sweep();

   void fill_sdf_with_values( mfem::Vector & aSDF );

   void preselect_triangles_x( const mfem::Vector & aPoint );

   void preselect_triangles_y( const mfem::Vector & aPoint );

   void preselect_triangles_z( const mfem::Vector & aPoint );

   void intersect_triangles(
      const unsigned int aAxis,
      const mfem::Vector & aPoint );

   void intersect_ray_with_triangles(
      const unsigned int aAxis,
      const mfem::Vector & aPoint,
      const unsigned int aNodeIndex );

   void check_if_node_is_inside(
      const unsigned int aAxis,
      const unsigned int aNodeIndex );

   void calculate_candidate_points_and_buffer_diagonal();

   void get_nodes_withing_bounding_box_of_triangle(
      Triangle * aTriangle,
      std::vector< Vertex* >  & aNodes,
      std::vector< Vertex * > & aCandList );

   /**
    * performs a floodfill in order to fix unsure nodes
    */
   void force_unsure_nodes_outside();

   void random_rotation();

   void  undo_rotation();
};
//-------------------------------------------------------------------------------

class SDF_Generator
{
   //! file containing object data
   Object mObject;

   //! verbosity flag
   bool          mVerboseFlag = false;

public:

   /**
    * constructor with pointer
    */
   SDF_Generator( const std::string & aObjectPath,
                  const bool aVerboseFlag = true );

   SDF_Generator( const std::string & aObjectPath,
                  const bool aVerboseFlag = true,
                  double scalefac = 1.0);

   SDF_Generator( const std::string & aObjectPath,
                  const mfem::Vector  aObjectOffset,
                  const bool aVerboseFlag = true );

   /**
    * trivial destructor
    */
   ~SDF_Generator() {};


   /**
    * calculates the SDF for a given mesh
    */
   void calculate_sdf( mfem::ParGridFunction & aSDFGridFunc ) ;

   void DoAmrOnMeshBasedOnIntersections(mfem::ParMesh *pmesh,
                                        int neighbors);

   Object *GetObject() { return &mObject; }

};
} /* namespace sdf */
} /* namespace mfem */

#endif
