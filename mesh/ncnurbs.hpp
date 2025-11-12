// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_NCNURBS
#define MFEM_NCNURBS

#include "nurbs.hpp"

namespace mfem
{

/** @brief NCNURBSExtension extends NURBSExtension to support NC-patch NURBS
    meshes. */
class NCNURBSExtension : public NURBSExtension
{
public:
   /// Copy constructor: deep copy
   NCNURBSExtension(const NCNURBSExtension &orig);

   NCNURBSExtension(std::istream &input, bool spacing=false);

   void UniformRefinement(const Array<int> &rf) override;

protected:
   /** @brief Set the mesh and space offsets, and also count the global
   @a NumOfVertices and the global @a NumOfDofs. */
   void GenerateOffsets() override;

   /// Return true if @a edge is a master NC-patch edge.
   bool IsMasterEdge(int edge) const override
   { return masterEdges.count(edge) > 0; }

   /// Return true if @a face is a master NC-patch face.
   bool IsMasterFace(int face) const override
   { return masterFaces.count(face) > 0; }

   /// Given a pair of vertices, return the corresponding edge.
   int VertexPairToEdge(const std::pair<int, int> &vertices) const override
   { return v2e.at(vertices); }

   /** @brief Get the DOFs (dof = true) or vertices (dof = false) for
       master edge @a me. */
   void GetMasterEdgeDofs(bool dof, int me, Array<int> &dofs) const override;

   /** @brief Get the DOFs (dof = true) or vertices (dof = false) for
       master face @a mf. */
   void GetMasterFaceDofs(bool dof, int mf, Array2D<int> &dofs) const override;

   /// Load refinement factors for a list of knotvectors from file.
   void LoadFactorsForKV(const std::string &filename);

   /// Set consistent refinement factors on patch @a p.
   int SetPatchFactors(int p);

   /// Ensure consistent refinement factors on all knotvectors.
   void PropagateFactorsForKV(int rf_default);

   /// Refine with refinement factors loaded for some knotvectors specified in
   /// the given file, with default refinement factor @a rf elsewhere. The flag
   /// @a coarsened indicates whether each patch is a single element.
   void RefineWithKVFactors(int rf, const std::string &kvf_filename,
                            bool coarsened) override;

private:
   /// Global mesh offsets, meshOffsets == meshVertexOffsets
   Array<int> aux_e_meshOffsets, aux_f_meshOffsets;

   /// Global space offsets, spaceOffsets == dofOffsets
   Array<int> aux_e_spaceOffsets, aux_f_spaceOffsets;

   /// Represents a nonconforming edge not in patchTopo->ncmesh.
   struct AuxiliaryEdge
   {
      int parent; /// Signed parent edge index (sign encodes orientation)
      int v[2];   /// Vertex indices
      int ksi[2]; /// Knot-span indices of vertices in parent edge
   };

   /// Represents a nonconforming face not in patchTopo->ncmesh.
   struct AuxiliaryFace
   {
      int parent;  /// Parent face index
      int ori;     /// Orientation with respect to parent face
      int v[4];    /// Vertex indices
      int ksi0[2]; /// Lower knot-span indices in parent face
      int ksi1[2]; /// Upper knot-span indices in parent face
   };

   /** @brief Represents a pair of child and parent edges for a nonconforming
       patch topology. */
   struct EdgePairInfo
   {
      int v; /// Vertex index
      int ksi; /// Knot-span index of vertex
      int child, parent; /// Child and parent edge indices
      bool isSet; /// Whether this instance is set

      EdgePairInfo() : isSet(false) { }

      EdgePairInfo(int vertex, int knotIndex, int childEdge, int parentEdge)
         : v(vertex), ksi(knotIndex), child(childEdge), parent(parentEdge),
           isSet(true) { }

      /// Set the data members.
      void Set(int vertex, int knotIndex, int childEdge, int parentEdge)
      {
         v = vertex;
         ksi = knotIndex;
         child = childEdge;
         parent = parentEdge;
         isSet = true;
      }

      bool operator==(const EdgePairInfo& other) const
      {
         return v == other.v && ksi == other.ksi && child == other.child
                && parent == other.parent;
      }
   };

   /// Master edge data for a nonconforming patch topology.
   struct MasterEdgeInfo
   {
      std::vector<int> slaves; /// Slave edge indices on the master edge
      std::vector<int> vertices; /// Vertex indices on the master edge
      std::vector<int> ks; /// Knot-span indices of vertices on the master edge

      void Reverse()
      {
         std::reverse(slaves.begin(), slaves.end());
         std::reverse(vertices.begin(), vertices.end());
         std::reverse(ks.begin(), ks.end());
      }
   };

   /// Master face data for a nonconforming patch topology.
   struct MasterFaceInfo
   {
      std::vector<int> slaves; /// Slave face indices on the master face
      std::vector<int> slaveCorners; /// Corner vertices of slave faces
      std::array<int, 2> ne; /// Number of elements in each direction
      std::array<int, 2> s0; /// Cartesian shift, see Reorder2D
      bool rev; /// Whether dimensions are interchanged

      MasterFaceInfo() : rev(false)
      {
         for (int i=0; i<2; ++i)
         {
            ne[i] = 0;
            s0[i] = -1;
         }
      }

      MasterFaceInfo(int ne1, int ne2) : rev(false)
      {
         ne[0] = ne1;
         ne[1] = ne2;
         for (int i=0; i<2; ++i) { s0[i] = -1; }
      }
   };

   /// Slave face data for a nonconforming patch topology.
   struct SlaveFaceInfo
   {
      int index; /// Face index
      int ori; /// Orientation
      int ksi[2]; /// Knot-span indices in parent face of v0
      int ne[2]; /// Number of elements in each direction on child face
   };

   /** @brief Represents a pair of child and parent faces for a nonconforming
       patch topology. */
   struct FacePairInfo
   {
      int v0; /// Lower left corner vertex
      int parent; /// Parent face index
      SlaveFaceInfo info; /// Data for the child face
   };

   /// Auxiliary edges and faces for a nonconforming patch topology.
   std::vector<AuxiliaryEdge> auxEdges;
   std::vector<AuxiliaryFace> auxFaces;

   /** @brief Maps from vertex pairs to indices in auxEdges, auxFaces. Vertex
       pairs are sorted indices, with faces having 4 vertices represented by the
       minimum index and the index of the diagonally opposite vertex. */
   std::map<std::pair<int, int>, int> auxv2e, auxv2f;

   /// Map from sorted vertex pairs to edge indices.
   std::map<std::pair<int, int>, int> v2e;

   /// Sets of master edges and face in patchTopo->ncmesh.
   std::set<int> masterEdges, masterFaces;

   /// Array form of @a masterEdges.
   Array<int> masterEdgeIndex;

   /** @brief Arrays of slave edges or faces, with possible repetitions, ordered
       by position within their master entities. */
   std::vector<int> slaveEdges;
   std::vector<SlaveFaceInfo> slaveFaces;

   /// Arrays of unique indices in @a slaveEdges, @a slaveFaces.
   Array<int> slaveEdgesUnique, slaveFacesUnique;

   /// Maps from slaveEdges/slaveFaces to slaveEdgesUnique/slaveEdgesUnique.
   std::map<int,int> slaveEdgesToUnique, slaveFacesToUnique;

   /// Maps from masterEdges/masterFaces to their indices in an ordered list.
   std::map<int,int> masterEdgeToId, masterFaceToId;

   /// Master edge and face data for a nonconforming patch topology.
   std::vector<MasterEdgeInfo> masterEdgeInfo;
   std::vector<MasterFaceInfo> masterFaceInfo;

   /** @brief Get the DOF (dof = true) or vertex (dof = false) offset for the
     edge with index @a edge plus @a increment. */
   int GetEdgeOffset(bool dof, int edge, int increment) const;

   /** @brief Get the DOF (dof = true) or vertex (dof = false) offset for the
       face with index @a face plus @a increment. */
   int GetFaceOffset(bool dof, int face, int increment) const;

   /// Map from a parent entity vertex index pair to knotvectors on the entity.
   std::map<std::pair<int, int>, std::array<int, 2>> parentToKV;

   /// Update knot-span indices in @a auxEdges and @a auxFaces on refinement.
   void UpdateAuxiliaryKnotSpans(const Array<int> &rf);

   /// For the master edge with index @a mid, set offsets @a os for the number
   /// of mesh edges in each subedge (slave or auxiliary edge).
   void GetMasterEdgePieceOffsets(int mid, Array<int> &os);

   /// Return the number of mesh edges in auxiliary edge @a aux_edge.
   int AuxiliaryEdgeNE(int aux_edge);

   /** @brief Find the permutation @a perm of slave face entities, with entity
       perm[i] of the slave face being entity i in the master face ordering.

       @param[in] sf  Slave face index in @a patchTopo
       @param[in] n1  Number of slave face edges, first master face direction.
       @param[in] n2  Number of slave face edges, second master face direction.
       @param[in] v0  Bottom-left face vertex with respect to the master face.
       @param[in] e1  Local edge index, first direction of the slave face.
       @param[in] e2  Local edge index, second direction of the slave face. */
   void GetFaceOrdering(int sf, int n1, int n2, int v0, int e1, int e2,
                        Array<int> &perm) const;

   /// Find additional slave and auxiliary faces after ProcessVertexToKnot3D.
   void FindAdditionalFacesSA(
      std::map<std::pair<int, int>, int> &v2f,
      std::set<int> &addParentFaces,
      std::vector<FacePairInfo> &facePairs);

   /// Helper function for @a GenerateOffsets().
   void ProcessFacePairs(int start, int midStart,
                         const std::vector<std::array<int, 2>> &parentSize,
                         std::vector<int> &parentVerts,
                         const std::vector<FacePairInfo> &facePairs);

   /// Helper function for @a GenerateOffsets().
   void ProcessVertexToKnot2D(const VertexToKnotSpan &v2k,
                              std::set<int> &reversedParents,
                              std::vector<EdgePairInfo> &edgePairs);

   /// Helper function for @a GenerateOffsets().
   void ProcessVertexToKnot3D(const VertexToKnotSpan &v2k,
                              const std::map<std::pair<int, int>, int> &v2f,
                              std::vector<std::array<int, 2>> &parentSize,
                              std::vector<EdgePairInfo> &edgePairs,
                              std::vector<FacePairInfo> &facePairs,
                              std::vector<int> &parentFaces,
                              std::vector<int> &parentVerts);

   /// Helper function for @a GenerateOffsets().
   void SetDofToPatch() override;

   /// Helper functions for @a PropagateFactorsForKV().
   void GetAuxFaceToPatchTable(Array2D<int> &auxface2patch);
   void GetSlaveFaceToPatchTable(Array2D<int> &sface2patch);

   /// Helper function for @a UniformRefinement().
   void Refine(bool coarsened, const Array<int> *rf = nullptr);

   /// Get the two endpoints of the auxiliary edge with index @a auxEdge.
   void GetAuxEdgeVertices(int auxEdge, Array<int> &verts) const;

   /// Get the four vertices of the auxiliary face with index @a auxFace.
   void GetAuxFaceVertices(int auxFace, Array<int> &verts) const;

   /// Get the four edges of the auxiliary face with index @a auxFace.
   void GetAuxFaceEdges(int auxFace, Array<int> &edges) const;

   /// Helper function for @a SetPatchFactors().
   void SlaveEdgeToParent(int se, int parent, const Array<int> &os,
                          const std::vector<int> &parentVerts,
                          Array<int> &edges);

   /// Helper function for @a FindAdditionalFacesSA().
   void GetMasterEdgeEntities(int edge, Array<int> &edgeV, Array<int> &edgeE,
                              Array<int> &edgeVki);

   /// Helper function for @a Refine().
   void UpdateCoarseKVF();

   /** @brief Read the control points for coarse patches.

       This is useful for a mesh with a nonconforming patch topology, when
       non-nested refinement is done. In such cases, knot insertion is done on
       coarse structured patches with a single element. */
   void ReadCoarsePatchCP(std::istream &input) override;

   /// Print control points for coarse patches @a patchCP.
   void PrintCoarsePatches(std::ostream &os) override;

   std::vector<Array<int>> auxef; /// Auxiliary edge refinement factors
};

}

#endif
