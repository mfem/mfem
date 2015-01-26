// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_PMESH
#define MFEM_PMESH

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../general/communication.hpp"
#include "mesh.hpp"
#include <iostream>

namespace mfem
{

/// Class for parallel meshes
class ParMesh : public Mesh
{
private:
   MPI_Comm MyComm;
   int NRanks, MyRank;

   Array<Element *> shared_edges;
   Array<Element *> shared_faces;

   /// Shared objects in each group.
   Table group_svert;
   Table group_sedge;
   Table group_sface;

   /// Shared to local index mapping.
   Array<int> svert_lvert;
   Array<int> sedge_ledge;
   Array<int> sface_lface;

   /// Return a number(0-1) identifying how the given edge has been split
   int GetEdgeSplittings(Element *edge, const DSTable &v_to_v, int *middle);
   /// Return a number(0-4) identifying how the given face has been split
   int GetFaceSplittings(Element *face, const DSTable &v_to_v, int *middle);

   void GetFaceNbrElementTransformation(
      int i, IsoparametricTransformation *ElTr);

   /// Refine quadrilateral mesh.
   virtual void QuadUniformRefinement();

   /// Refine a hexahedral mesh.
   virtual void HexUniformRefinement();

   virtual void NURBSUniformRefinement();

   void DeleteFaceNbrData();

public:
   ParMesh(MPI_Comm comm, Mesh &mesh, int *partitioning_ = NULL,
           int part_method = 1);

   MPI_Comm GetComm() { return MyComm; }
   int GetNRanks() { return NRanks; }
   int GetMyRank() { return MyRank; }

   GroupTopology gtopo;

   // Face-neighbor elements and vertices
   bool             have_face_nbr_data;
   Array<int>       face_nbr_group;
   Array<int>       face_nbr_elements_offset;
   Array<int>       face_nbr_vertices_offset;
   Array<Element *> face_nbr_elements;
   Array<Vertex>    face_nbr_vertices;
   // Local face-neighbor elements and vertices ordered by face-neighbor
   Table            send_face_nbr_elements;
   Table            send_face_nbr_vertices;

   int GetNGroups() { return gtopo.NGroups(); }

   // next 6 methods do not work for the 'local' group 0
   int GroupNVertices(int group) { return group_svert.RowSize(group-1); }
   int GroupNEdges(int group)    { return group_sedge.RowSize(group-1); }
   int GroupNFaces(int group)    { return group_sface.RowSize(group-1); }

   int GroupVertex(int group, int i)
   { return svert_lvert[group_svert.GetJ()[group_svert.GetI()[group-1]+i]]; }
   void GroupEdge(int group, int i, int &edge, int &o);
   void GroupFace(int group, int i, int &face, int &o);

   void ExchangeFaceNbrData();
   void ExchangeFaceNbrNodes();
   int GetNFaceNeighbors() const { return face_nbr_group.Size(); }
   int GetFaceNbrGroup(int fn) const { return face_nbr_group[fn]; }
   int GetFaceNbrRank(int fn) const;
   /** Similar to Mesh::GetFaceToElementTable with added face-neighbor elements
       with indices offset by the local number of elements. */
   Table *GetFaceToAllElementTable() const;

   /** Get the FaceElementTransformations for the given shared face (edge 2D).
       In the returned object, 1 and 2 refer to the local and the neighbor
       elements, respectively. */
   FaceElementTransformations *GetSharedFaceTransformations(int);

   /// Return the number of shared faces (3D), edges (2D), vertices (1D)
   int GetNSharedFaces() const;

   /// See the remarks for the serial version in mesh.hpp
   virtual void ReorientTetMesh();

   /// Refine the marked elements.
   virtual void LocalRefinement(const Array<int> &marked_el, int type = 3);

   /// Update the groups after tet refinement
   void RefineGroups(const DSTable &v_to_v, int *middle);

   /** Print the part of the mesh in the calling processor adding the interface
       as boundary (for visualization purposes) using the default format. */
   virtual void Print(std::ostream &out = std::cout) const;

   /** Print the part of the mesh in the calling processor adding the interface
       as boundary (for visualization purposes) using Netgen/Truegrid format .*/
   virtual void PrintXG(std::ostream &out = std::cout) const;

   /** Write the mesh to the stream 'out' on Process 0 in a form
       suitable for visualization: the mesh is written as a disjoint
       mesh and the shared boundary is added to the actual boundary;
       both the element and boundary attributes are set to the
       precessor number.  */
   void PrintAsOne(std::ostream &out = std::cout);

   /// Old mesh format (Netgen/Truegrid) version of 'PrintAsOne'
   void PrintAsOneXG(std::ostream &out = std::cout);

   /// Print various parallel mesh stats
   void PrintInfo(std::ostream &out = std::cout);

   virtual ~ParMesh();
};

}

#endif // MFEM_USE_MPI

#endif
