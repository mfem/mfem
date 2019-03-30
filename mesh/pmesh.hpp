// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_PMESH
#define MFEM_PMESH

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../general/communication.hpp"
#include "../general/globals.hpp"
#include "mesh.hpp"
#include "pncmesh.hpp"
#include <iostream>

namespace mfem
{
#ifdef MFEM_USE_PUMI
class ParPumiMesh;
#endif

/// Class for parallel meshes
class ParMesh : public Mesh
{
protected:
   ParMesh() : MyComm(0), NRanks(0), MyRank(-1),
      have_face_nbr_data(false), pncmesh(NULL) {}

   MPI_Comm MyComm;
   int NRanks, MyRank;

   struct Vert3
   {
      int v[3];
      Vert3() { }
      Vert3(int v0, int v1, int v2) { v[0] = v0; v[1] = v1; v[2] = v2; }
      void Set(int v0, int v1, int v2) { v[0] = v0; v[1] = v1; v[2] = v2; }
      void Set(const int *w) { v[0] = w[0]; v[1] = w[1]; v[2] = w[2]; }
   };

   struct Vert4
   {
      int v[4];
      Vert4() { }
      Vert4(int v0, int v1, int v2, int v3)
      { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; }
      void Set(int v0, int v1, int v2, int v3)
      { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; }
      void Set(const int *w)
      { v[0] = w[0]; v[1] = w[1]; v[2] = w[2]; v[3] = w[3]; }
   };

   Array<Element *> shared_edges;
   // shared face id 'i' is:
   //   * triangle id 'i',                  if i < shared_trias.Size()
   //   * quad id 'i-shared_trias.Size()',  otherwise
   Array<Vert3> shared_trias;
   Array<Vert4> shared_quads;

   /// Shared objects in each group.
   Table group_svert;
   Table group_sedge;
   Table group_stria;  // contains shared triangle indices
   Table group_squad;  // contains shared quadrilateral indices

   /// Shared to local index mapping.
   Array<int> svert_lvert;
   Array<int> sedge_ledge;
   // sface ids: all triangles first, then all quads
   Array<int> sface_lface;

   /// Create from a nonconforming mesh.
   ParMesh(const ParNCMesh &pncmesh);

   // Convert the local 'meshgen' to a global one.
   void ReduceMeshGen();

   // Determine sedge_ledge and sface_lface.
   void FinalizeParTopo();

   // Mark all tets to ensure consistency across MPI tasks; also mark the
   // shared and boundary triangle faces using the consistently marked tets.
   virtual void MarkTetMeshForRefinement(DSTable &v_to_v);

   /// Return a number(0-1) identifying how the given edge has been split
   int GetEdgeSplittings(Element *edge, const DSTable &v_to_v, int *middle);
   /// Append codes identifying how the given face has been split to @a codes
   void GetFaceSplittings(const int *fv, const HashTable<Hashed2> &v_to_v,
                          Array<unsigned> &codes);

   bool DecodeFaceSplittings(HashTable<Hashed2> &v_to_v, const int *v,
                             const Array<unsigned> &codes, int &pos);

   void GetFaceNbrElementTransformation(
      int i, IsoparametricTransformation *ElTr);

   ElementTransformation* GetGhostFaceTransformation(
      FaceElementTransformations* FETr, Element::Type face_type,
      Geometry::Type face_geom);

   /// Update the groups after triangle refinement
   void RefineGroups(const DSTable &v_to_v, int *middle);

   /// Update the groups after tetrahedron refinement
   void RefineGroups(int old_nv, const HashTable<Hashed2> &v_to_v);

   void UniformRefineGroups2D(int old_nv);

   // f2qf can be NULL if all faces are quads or there are no quad faces
   void UniformRefineGroups3D(int old_nv, int old_nedges,
                              const DSTable &old_v_to_v,
                              const STable3D &old_faces,
                              Array<int> *f2qf);

   void ExchangeFaceNbrData(Table *gr_sface, int *s2l_face);

   /// Refine a mixed 2D mesh uniformly.
   virtual void UniformRefinement2D();

   /// Refine a mixed 3D mesh uniformly.
   virtual void UniformRefinement3D();

   virtual void NURBSUniformRefinement();

   /// This function is not public anymore. Use GeneralRefinement instead.
   virtual void LocalRefinement(const Array<int> &marked_el, int type = 3);

   /// This function is not public anymore. Use GeneralRefinement instead.
   virtual void NonconformingRefinement(const Array<Refinement> &refinements,
                                        int nc_limit = 0);

   virtual bool NonconformingDerefinement(Array<double> &elem_error,
                                          double threshold, int nc_limit = 0,
                                          int op = 1);
   void DeleteFaceNbrData();

   bool WantSkipSharedMaster(const NCMesh::Master &master) const;

   /// Fills out partitioned Mesh::vertices
   int BuildLocalVertices(const Mesh& global_mesh, const int *partitioning,
                          Array<int> &vert_global_local);

   /// Fills out partitioned Mesh::elements
   int BuildLocalElements(const Mesh& global_mesh, const int *partitioning,
                          const Array<int> &vert_global_local);

   /// Fills out partitioned Mesh::boundary
   int BuildLocalBoundary(const Mesh& global_mesh, const int *partitioning,
                          const Array<int> &vert_global_local,
                          Array<bool>& activeBdrElem,
                          Table* &edge_element);

   void FindSharedFaces(const Mesh &mesh, const int* partition,
                        Array<int>& face_group,
                        ListOfIntegerSets& groups);

   int FindSharedEdges(const Mesh &mesh, const int* partition,
                       Table* &edge_element, ListOfIntegerSets& groups);

   int FindSharedVertices(const int *partition, Table* vertex_element,
                          ListOfIntegerSets& groups);

   void BuildFaceGroup(int ngroups, const Mesh &mesh,
                       const Array<int>& face_group,
                       int &nstria, int &nsquad);

   void BuildEdgeGroup(int ngroups, const Table& edge_element);

   void BuildVertexGroup(int ngroups, const Table& vert_element);

   void BuildSharedFaceElems(int ntri_faces, int nquad_faces,
                             const Mesh &mesh, int *partitioning,
                             const STable3D *faces_tbl,
                             const Array<int> &face_group,
                             const Array<int> &vert_global_local);

   void BuildSharedEdgeElems(int nedges, Mesh &mesh,
                             const Array<int> &vert_global_local,
                             const Table *edge_element);

   void BuildSharedVertMapping(int nvert, const Table* vert_element,
                               const Array<int> &vert_global_local);


public:
   /** Copy constructor. Performs a deep copy of (almost) all data, so that the
       source mesh can be modified (e.g. deleted, refined) without affecting the
       new mesh. If 'copy_nodes' is false, use a shallow (pointer) copy for the
       nodes, if present. */
   explicit ParMesh(const ParMesh &pmesh, bool copy_nodes = true);

   ParMesh(MPI_Comm comm, Mesh &mesh, int *partitioning_ = NULL,
           int part_method = 1);

   /// Read a parallel mesh, each MPI rank from its own file/stream.
   /** The @a refine parameter is passed to the method Mesh::Finalize(). */
   ParMesh(MPI_Comm comm, std::istream &input, bool refine = true);

   /// Create a uniformly refined (by any factor) version of @a orig_mesh.
   /** @param[in] orig_mesh  The starting coarse mesh.
       @param[in] ref_factor The refinement factor, an integer > 1.
       @param[in] ref_type   Specify the positions of the new vertices. The
                             options are BasisType::ClosedUniform or
                             BasisType::GaussLobatto.

       The refinement data which can be accessed with GetRefinementTransforms()
       is set to reflect the performed refinements.

       @note The constructed ParMesh is linear, i.e. it does not have nodes. */
   ParMesh(ParMesh *orig_mesh, int ref_factor, int ref_type);

   virtual void Finalize(bool refine = false, bool fix_orientation = false);

   MPI_Comm GetComm() const { return MyComm; }
   int GetNRanks() const { return NRanks; }
   int GetMyRank() const { return MyRank; }

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

   ParNCMesh* pncmesh;

   int GetNGroups() const { return gtopo.NGroups(); }

   ///@{ @name These methods require group > 0
   int GroupNVertices(int group) { return group_svert.RowSize(group-1); }
   int GroupNEdges(int group)    { return group_sedge.RowSize(group-1); }
   int GroupNTriangles(int group) { return group_stria.RowSize(group-1); }
   int GroupNQuadrilaterals(int group) { return group_squad.RowSize(group-1); }

   int GroupVertex(int group, int i)
   { return svert_lvert[group_svert.GetRow(group-1)[i]]; }
   void GroupEdge(int group, int i, int &edge, int &o);
   void GroupTriangle(int group, int i, int &face, int &o);
   void GroupQuadrilateral(int group, int i, int &face, int &o);
   ///@}

   void GenerateOffsets(int N, HYPRE_Int loc_sizes[],
                        Array<HYPRE_Int> *offsets[]) const;

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
   FaceElementTransformations *
   GetSharedFaceTransformations(int sf, bool fill2 = true);

   /// Return the number of shared faces (3D), edges (2D), vertices (1D)
   int GetNSharedFaces() const;

   /// Return the local face index for the given shared face.
   int GetSharedFace(int sface) const;

   /// See the remarks for the serial version in mesh.hpp
   virtual void ReorientTetMesh();

   /// Utility function: sum integers from all processors (Allreduce).
   virtual long ReduceInt(int value) const;

   /// Load balance the mesh. NC meshes only.
   void Rebalance();

   /** Print the part of the mesh in the calling processor adding the interface
       as boundary (for visualization purposes) using the mfem v1.0 format. */
   virtual void Print(std::ostream &out = mfem::out) const;

   /** Print the part of the mesh in the calling processor adding the interface
       as boundary (for visualization purposes) using Netgen/Truegrid format .*/
   virtual void PrintXG(std::ostream &out = mfem::out) const;

   /** Write the mesh to the stream 'out' on Process 0 in a form suitable for
       visualization: the mesh is written as a disjoint mesh and the shared
       boundary is added to the actual boundary; both the element and boundary
       attributes are set to the processor number.  */
   void PrintAsOne(std::ostream &out = mfem::out);

   /// Old mesh format (Netgen/Truegrid) version of 'PrintAsOne'
   void PrintAsOneXG(std::ostream &out = mfem::out);

   /// Returns the minimum and maximum corners of the mesh bounding box. For
   /// high-order meshes, the geometry is refined first "ref" times.
   void GetBoundingBox(Vector &p_min, Vector &p_max, int ref = 2);

   void GetCharacteristics(double &h_min, double &h_max,
                           double &kappa_min, double &kappa_max);

   /// Print various parallel mesh stats
   virtual void PrintInfo(std::ostream &out = mfem::out);

   /// Save the mesh in a parallel mesh format.
   void ParPrint(std::ostream &out) const;

   virtual int FindPoints(DenseMatrix& point_mat, Array<int>& elem_ids,
                          Array<IntegrationPoint>& ips, bool warn = true,
                          InverseElementTransformation *inv_trans = NULL);

   /// Debugging method
   void PrintSharedEntities(const char *fname_prefix) const;

   virtual ~ParMesh();

   friend class ParNCMesh;
#ifdef MFEM_USE_PUMI
   friend class ParPumiMesh;
#endif
};

}

#endif // MFEM_USE_MPI

#endif
