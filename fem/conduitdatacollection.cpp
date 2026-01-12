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

#include "../config/config.hpp"

#ifdef MFEM_USE_CONDUIT

#include "fem.hpp"
#include "../general/text.hpp"
#ifdef MFEM_USE_MPI
#include "../mesh/pmesh.hpp"
#endif

#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_blueprint_mesh_utils.hpp>
#include <conduit_blueprint_mesh_utils_iterate_elements.hpp>

#include <string>
#include <sstream>

// These macros control output of Blueprint debugging files.
#define DEBUG_SHARED_EDGES
#define DEBUG_SHARED_FACES

#if defined(DEBUG_SHARED_EDGES) || defined(DEBUG_SHARED_FACES)
#include <conduit_relay.hpp>
#include <conduit_relay_io_blueprint.hpp>
#endif

using namespace conduit;

namespace mfem
{
#ifdef MFEM_USE_MPI
namespace
{
//---------------------------------------------------------------------------//
// Utility functions
//---------------------------------------------------------------------------//

#ifdef DEBUG_SHARED_EDGES
/**
  @brief Make the MFEM mesh edges into a Blueprint mesh and make a field that
         labels them so we can visualize the edges in VisIt.

  @param mesh The input MFEM mesh.
  @param[out] n_mesh A Conduit node that will contain a mesh that represents the MFEM mesh's edges.
 */
void MeshEdges(const mfem::Mesh &mesh, conduit::Node &n_mesh)
{
   n_mesh.reset();  // Clear any existing data

   const int dim = mesh.Dimension();
   const int nv = mesh.GetNV();
   const int ne = mesh.GetNEdges();

   // -----------------------------
   // 1) Coordinates (coordsets)
   // -----------------------------
   // Single explicit coordset "coords"
   conduit::Node &n_coordset = n_mesh["coordsets/mfem_coords"];
   n_coordset["type"] = "explicit";

   conduit::Node &vx = n_coordset["values/x"];
   conduit::Node *vy = nullptr;
   conduit::Node *vz = nullptr;

   if (dim > 1) { vy = &n_coordset["values/y"]; }
   if (dim > 2) { vz = &n_coordset["values/z"]; }

   vx.set(conduit::DataType::float64(nv));
   double *vx_ptr = vx.value();

   double *vy_ptr = nullptr;
   double *vz_ptr = nullptr;
   if (vy)
   {
      vy->set(conduit::DataType::float64(nv));
      vy_ptr = vy->value();
   }
   if (vz)
   {
      vz->set(conduit::DataType::float64(nv));
      vz_ptr = vz->value();
   }

   // Linear mesh, so vertex coordinates are the geometry
   mfem::Vector vtx(dim);
   for (int i = 0; i < nv; i++)
   {
      const auto v = mesh.GetVertex(i);
      vx_ptr[i] = v[0];
      if (dim > 1)
      {
         vy_ptr[i] = v[1];
      }
      if (dim > 2)
      {
         vz_ptr[i] = v[2];
      }
   }

   // -----------------------------
   // 2) Topology of edges
   // -----------------------------
   conduit::Node &n_topo = n_mesh["topologies/mfem_edges"];
   n_topo["type"] = "unstructured";
   n_topo["coordset"] = "mfem_coords";
   n_topo["elements/shape"] = "line";  // each edge is a 2-vertex line

   conduit::Node &n_conn = n_topo["elements/connectivity"];
   n_conn.set(conduit::DataType::int32(2 * ne));
   int *conn_ptr = n_conn.value();

   for (int e = 0; e < ne; e++)
   {
      mfem::Array<int> v;
      mesh.GetEdgeVertices(e, v);
      conn_ptr[2 * e + 0] = v[0];
      conn_ptr[2 * e + 1] = v[1];
   }

   // -----------------------------
   // 3) Edge ID field
   // -----------------------------
   conduit::Node &n_field = n_mesh["fields/mfem_edge_id"];
   n_field["association"] = "element";
   n_field["topology"] = "mfem_edges";
   n_field["volume_dependent"] = "false";  // optional metadata

   n_field["values"].set(conduit::DataType::int32(ne));
   int *f_ptr = n_field["values"].value();

   for (int e = 0; e < ne; e++)
   {
      f_ptr[e] = e;  // edge number
   }

   // -----------------------------
   // 4) (Optional) Blueprint verify
   // -----------------------------
   conduit::Node info;
   if (!conduit::blueprint::mesh::verify(n_mesh, info))
   {
      mfem::err << "Blueprint verify failed in MeshEdges:\n" << info.to_yaml() <<
                std::endl;
   }
}
#endif

/**
  @brief Computes a zone ordering array for the input topology that sorts the zones spatially.

  @param n_topo A node that contains the blueprint topology to be ordered.

  @return A vector containing zone numbers in their spatially-sorted order.
 */
std::vector<conduit::index_t> spatial_ordering(const conduit::Node &n_topo)
{
   // Compute centroids on the new topology, to make a new point mesh topology.
   conduit::Node n_centroids;
   namespace topoutils = conduit::blueprint::mesh::utils::topology;
   conduit::blueprint::mesh::topology::unstructured::generate_centroids(
      n_topo,
      n_centroids["topologies/centroid"],
      n_centroids["coordsets/centroid_coords"],
      n_centroids["s2dmap"],
      n_centroids["d2smap"]);

   // Make mesh info for the centroids topology (for the extents).
   topoutils::MeshInfo info;
   topoutils::compute_mesh_info(n_centroids["topologies/centroid"], info);
   // Make mesh info for the input topology since it will have edges.
   topoutils::MeshInfo info2;
   topoutils::compute_mesh_info(n_topo, info2);
   // Put the edge lengths into the centroids mesh info.
   info.minEdgeLength = info2.minEdgeLength / 4.;
   info.maxEdgeLength = info2.maxEdgeLength / 4.;
   info.minDiagonalLength = info2.minDiagonalLength / 4.;
   info.maxDiagonalLength = info2.maxDiagonalLength / 4.;

   // Make a quantizer that will use the mesh info.
   topoutils::MeshInfoCollection infoCollection;
   infoCollection.begin();
   infoCollection.add(0, info);
   infoCollection.end();
   topoutils::Quantizer Q(infoCollection.getMergedMeshInfo());

   // Get some accessors for the centroid coordset.
   const conduit::Node &n_centroid_coord_values =
      n_centroids["coordsets/centroid_coords/values"];
   const auto dimension = n_centroid_coord_values.number_of_children();
   const auto x = n_centroid_coord_values["x"].as_float64_accessor();
   const auto y = n_centroid_coord_values["y"].as_float64_accessor();
   const auto centroid_nzones = x.number_of_elements();
   std::vector<typename topoutils::Quantizer::QuantizedIndex> keys(
      centroid_nzones);
   std::vector<conduit::index_t> order(centroid_nzones);

   // Use the quantizer to make a key for each coordinate in the centroids coordset.
   // This makes a spatially-ordered key.
   typename topoutils::Quantizer::Coordinate coord(dimension);
   if (dimension == 3)
   {
      const auto z = n_centroid_coord_values["z"].as_float64_accessor();
      for (size_t i = 0; i < centroid_nzones; i++)
      {
         coord[0] = x[i];
         coord[1] = y[i];
         coord[2] = z[i];
         keys[i] = Q.quantize(coord);
         order[i] = i;
      }
   }
   else
   {
      for (size_t i = 0; i < centroid_nzones; i++)
      {
         coord[0] = x[i];
         coord[1] = y[i];
         keys[i] = Q.quantize(coord);
         order[i] = i;
      }
   }
   // Sort the order according to the name.
   std::sort(order.begin(), order.end(), [&](auto a, auto b) { return keys[a] < keys[b]; });
   return order;
}

/**
  @brief Iterates over all faces for each zone in the input topology where all
         of the face's nodes are selected in the @a n_selectedNodes array. For these
         selected faces, a function is invoked that lets the function append to the
         new topology being created in @a n_output.

  @tparam FuncType A callable type that accepts various connectivity-related vectors.

  @param n_topo A node that contains the source Blueprint topology.
  @param n_selectedNodes A node that contains indices for the selected nodes.
  @param[out] n_output A node that contains the new topology. It needs to be different
                       from the node that contains n_topo.
  @param func The callable type that is invoked for faces whose nodes are all selected.
 */
template <typename FuncType>
void extract_topology_with_selected_nodes(const conduit::Node &n_topo,
                                          const conduit::Node &n_selectedNodes,
                                          conduit::Node &n_output,
                                          FuncType &&func)
{
   // Get the coordset to get the number of nodes.
   const conduit::Node *n_coordset =
      conduit::blueprint::mesh::utils::find_reference_node(n_topo, "coordset");
   assert(n_coordset != nullptr);
   const auto nnodes = conduit::blueprint::mesh::coordset::length(*n_coordset);

   // Make a mask for the points in the coordset to indicate which are selected in n_selectedNodes.
   std::vector<int> pointIsSelected(nnodes, 0);
   const auto values = n_selectedNodes.as_int_accessor();
   for (conduit::index_t i = 0; i < values.number_of_elements(); i++)
   {
      pointIsSelected[values[i]] = 1;
   }

   // Make a new topology from the selected faces if all points in a zone's face are selected.
   std::vector<int> conn, sizes, offsets;
   std::vector<conduit::index_t> originalElement;
   int counts[] = {0, 0, 0 /*line*/, 0 /*tri*/, 0 /*quad*/, 0 /*polygon*/};
   const auto nzones = conduit::blueprint::mesh::topology::length(n_topo);
   conduit::blueprint::mesh::utils::topology::iterate_elements(
      n_topo,
      [&](conduit::blueprint::mesh::utils::topology::entity &e)
   {
      if (e.subelement_ids.empty())
      {
         // Iterate over the shapes in the face and determine whether the face gets made.
         for (conduit::index_t fi = 0; fi < e.shape.num_faces(); fi++)
         {
            conduit::index_t nFaceIds = 0;
            const auto *faceIds = e.shape.get_face(fi, nFaceIds);
            bool allPointsSelected = true;
            for (int i = 0; i < nFaceIds && allPointsSelected; i++)
            {
               const auto id = e.element_ids[faceIds[i]];
               allPointsSelected &= pointIsSelected[id];
            }
            // Make a face
            if (allPointsSelected)
            {
               constexpr int MAX_NODES_PER_FACE = 4;
               conduit::index_t thisFaceIds[MAX_NODES_PER_FACE];
               for (int i = 0; i < nFaceIds; i++)
               {
                  thisFaceIds[i] = e.element_ids[faceIds[i]];
               }
               const int index =
                  func(e.entity_id, thisFaceIds, nFaceIds, conn, sizes, offsets, originalElement);

               counts[index]++;
            }
         }
      }
      else
      {
         for (const auto &face : e.subelement_ids)
         {
            bool allPointsSelected = true;
            for (const auto &id : face) { allPointsSelected &= pointIsSelected[id]; }

            // Make a face
            if (allPointsSelected)
            {
               const int index = func(e.entity_id,
                                      face.data(),
                                      face.size(),
                                      conn,
                                      sizes,
                                      offsets,
                                      originalElement);

               counts[index]++;
            }
         }
      }
   });

   // Look in counts to determine the spatial dimension.
   int shapeDimension = 0;
   shapeDimension = std::max(shapeDimension, 1 * ((counts[2] > 0) ? 1 : 0));
   shapeDimension = std::max(shapeDimension, 2 * ((counts[3] > 0) ? 1 : 0));
   shapeDimension = std::max(shapeDimension, 2 * ((counts[4] > 0) ? 1 : 0));
   shapeDimension = std::max(shapeDimension, 2 * ((counts[5] > 0) ? 1 : 0));

   // Make the output mesh.
   n_output["coordsets/" + n_coordset->name()].set_external(*n_coordset);
   conduit::Node &n_new_topo = n_output["topologies/" + n_topo.name()];
   n_new_topo["coordset"] = n_coordset->name();
   n_new_topo["type"] = "unstructured";
   if (shapeDimension == 1)
   {
      n_new_topo["elements/shape"] = "line";
   }
   else if (shapeDimension == 2)
   {
      n_new_topo["elements/shape"] = "polygonal";
   }
   else
   {
      assert("Unhandled shape type.");
   }
   n_new_topo["elements/connectivity"].set(conn);
   n_new_topo["elements/sizes"].set(sizes);
   n_new_topo["elements/offsets"].set(offsets);

   // Figure out a spatial ordering for the face topology. Save it as a field.
   const auto order = spatial_ordering(n_new_topo);

   // Reorder the connectivity so it is easier to think about.
   std::vector<int> newconn, newsizes, newoffsets;
   std::vector<conduit::index_t> newOriginalElement;
   newconn.reserve(conn.size());
   newsizes.reserve(sizes.size());
   newoffsets.reserve(offsets.size());
   newOriginalElement.reserve(newOriginalElement.size());
   for (size_t i = 0; i < sizes.size(); i++)
   {
      const auto size = sizes[order[i]];
      const auto offset = offsets[order[i]];

      newoffsets.push_back(newconn.size());
      newsizes.push_back(size);
      for (int j = 0; j < size; j++)
      {
         newconn.push_back(conn[offset + j]);
      }

      // Make the original element be a different order
      newOriginalElement.push_back(originalElement[order[i]]);
   }
   // Update the connectivity, sizes, offsets with reordered versions.
   n_new_topo["elements/connectivity"].set(newconn);
   n_new_topo["elements/sizes"].set(newsizes);
   n_new_topo["elements/offsets"].set(newoffsets);

   newOriginalElement.swap(originalElement);

   // Save a field that relates the face topology to the original topology zone indices.
   n_output["fields/originalElement/topology"] = n_topo.name();
   n_output["fields/originalElement/association"] = "element";
   n_output["fields/originalElement/values"].set(originalElement);
}
}  // end namespace

//---------------------------------------------------------------------------//
// class ConduitParMeshBuilder implementation
//---------------------------------------------------------------------------//

/**
  @brief Builds a ParMesh from a Mesh and the supplied adjset.
 */
class ConduitParMeshBuilder
{
public:
   /**
     Builds a ParMesh from a Mesh, keeping all zones in Mesh on the current
     processor, preserving its decomposition, and initializing ParMesh
     communication data from the supplied adjset.

     @param comm The MPI communicator
     @param mesh The mesh that represents the Blueprint domain on the current processor.
     @param n_adjset The adjset for the domain.

     @return A new ParMesh instance initialized by the adjset. The caller must free it.
    */
   static ParMesh *Build(MPI_Comm comm, mfem::Mesh &mesh,
                         const conduit::Node &n_adjset);

private:
   /**
     @brief Initializes ParMesh's gtopo member from the adjacency set.

     @param n_adjset The node that contains the adjacency set.
    */
   static void InitGroupTopology(ParMesh *pmesh, const conduit::Node &n_adjset);

   /**
     @brief Initializes ParMesh's shared vertices members from the adjacency set.

     @param n_adjset The node that contains the adjacency set.
    */
   static void InitSharedVertices(ParMesh *pmesh, const conduit::Node &n_adjset);

   /**
     @brief Initializes ParMesh's shared faces members from the adjacency set.

     @param n_adjset The node that contains the adjacency set.
    */
   static void InitSharedFaces(ParMesh *pmesh, const conduit::Node &n_adjset);

   /**
     @brief Initializes the ParMesh's shared edges members from the adjacency set.

     @param n_adjset The node that contains the adjacency set.
    */
   static void InitSharedEdges(ParMesh *pmesh, const conduit::Node &n_adjset);

   /**
     @brief Makes a new Blueprint topology containing faces based on the adjacency set group.

     @param n_adjset The node that contains the adjacency set.
     @param group The index of the adjacency set group (starts at 0).
     @param[out] n_output A new face topology for the adjset group.
    */
   static void GetSelectedFaceTopologyFromAdjset(const conduit::Node &n_adjset,
                                                 int group,
                                                 conduit::Node &n_output);

   /**
     @brief Makes a new Blueprint topology containing edges based on the adjacency set group.

     @param n_adjset The node that contains the adjacency set.
     @param group The index of the adjacency set group (starts at 0).
     @param[out] n_output A new edge topology for the adjset group.
    */
   static void GetSelectedEdgeTopologyFromAdjset(
      const conduit::Node &n_adjset,
      int group,
      conduit::Node &n_output);

   /**
     @brief Builds an mfem::Table using values stored in @a values.

     @param t The Table to build.
     @param values A vector of vectors where each vector contains a row of table data.
    */
   static void BuildTable(mfem::Table &t,
                          const std::vector<std::vector<int>> &values);

   /**
     @brief Gets a vector of unique vertex numbers for a face. The face vertices are
            returned in numerical order, which might not necessarily result in a good
            polygonal face.

     @param faceID The faceID in the MFEM mesh.

     @return A vector of face vertices.
    */
   static std::vector<int> GetFaceVertices(ParMesh *pmesh, int faceID);
};

//---------------------------------------------------------------------------//
ParMesh *
ConduitParMeshBuilder::Build(MPI_Comm comm, mfem::Mesh &mesh,
                             const conduit::Node &n_adjset)
{
   // Keep all zones on this rank.
   ParMesh *pmesh = new ParMesh(comm, mesh, std::vector<int>(mesh.GetNE(),
                                                             mfem::Mpi::WorldRank()).data());

   // Finish initializing the shared communication members that did not get set
   // due to the partition keeping all zones on this rank (above).
   InitGroupTopology(pmesh, n_adjset);
   InitSharedVertices(pmesh, n_adjset);
   InitSharedEdges(pmesh, n_adjset);
   InitSharedFaces(pmesh, n_adjset);

   return pmesh;
}

//---------------------------------------------------------------------------//
void ConduitParMeshBuilder::InitGroupTopology(ParMesh *pmesh,
                                              const conduit::Node &n_adjset)
{
   const conduit::Node &n_groups = n_adjset.fetch_existing("groups");
   const int mpitag = 823;
   mfem::ListOfIntegerSets groups;
   // Add the first group.
   mfem::IntegerSet local {mfem::Mpi::WorldRank()};
   groups.Insert(local);
   const auto numGroups = static_cast<int>(n_groups.number_of_children());
   for (int group_id = 0; group_id < numGroups; ++group_id)
   {
      const conduit::Node &n_group = n_groups[group_id];
      const auto neighbors = n_group["neighbors"].as_int_accessor();
      const auto group_size = static_cast<int>(neighbors.number_of_elements());

      mfem::IntegerSet newGroup;
      mfem::Array<int> &array = newGroup;
      array.Reserve(group_size + 1);
      array.Append(mfem::Mpi::WorldRank());
      for (int index = 0; index < group_size; ++index)
      {
         array.Append(neighbors[index]);
      }
      groups.Insert(newGroup);
   }
   mfem::GroupTopology g;
   g.SetComm(MPI_COMM_WORLD);
   g.Create(groups, mpitag);
   pmesh->gtopo = g;
}

//---------------------------------------------------------------------------//
void ConduitParMeshBuilder::InitSharedVertices(ParMesh *pmesh,
                                               const conduit::Node &n_adjset)
{
   const conduit::Node &n_groups = n_adjset.fetch_existing("groups");
   const auto numGroups = static_cast<int>(n_groups.number_of_children());

   std::vector<std::vector<int>> groups;
   groups.resize(numGroups);
   int maxVertex = -1;
   for (int g = 0; g < numGroups; ++g)
   {
      const conduit::Node &n_group = n_groups[g];
      const auto values = n_group["values"].as_int_accessor();
      const auto n = static_cast<int>(values.number_of_elements());
      groups[g].reserve(n);
      for (int c = 0; c < n; c++)
      {
         groups[g].push_back(values[c]);
         maxVertex = std::max(maxVertex, values[c]);
      }
   }
   BuildTable(pmesh->group_svert, groups);
   if (mfem::Mpi::WorldRank() == 0)
   {
      pmesh->group_svert.Print(mfem::out);
   }

   // Build a map of shared vertices to local.
   pmesh->svert_lvert = mfem::Array<int>(maxVertex + 1);
   std::iota(pmesh->svert_lvert.GetData(),
             pmesh->svert_lvert.GetData() + maxVertex + 1, 0);
}

//---------------------------------------------------------------------------//
void ConduitParMeshBuilder::GetSelectedFaceTopologyFromAdjset(
   const conduit::Node &n_adjset,
   int group,
   conduit::Node &n_output)
{
   const conduit::Node *n_topo =
      conduit::blueprint::mesh::utils::find_reference_node(n_adjset, "topology");
   assert(n_topo != nullptr);

   // Use the group to pull out faces that use the nodes selected in the adjset.
   const conduit::Node &n_groups = n_adjset.fetch_existing("groups");
   extract_topology_with_selected_nodes(
      *n_topo,
      n_groups[group]["values"],
      n_output,
      // Invoke this on each shape to make a new face in the output topo from the supplied face.
      [](conduit::index_t elementNumber,
         const conduit::index_t *faceIds,
         conduit::index_t nFaceIds,
         std::vector<int> &connectivity,
         std::vector<int> &sizes,
         std::vector<int> &offsets,
         std::vector<conduit::index_t> &originalElement)
   {
      offsets.push_back(connectivity.size());
      sizes.push_back(nFaceIds);
      for (conduit::index_t i = 0; i < nFaceIds; i++)
      {
         connectivity.push_back(faceIds[i]);
      }
      originalElement.push_back(elementNumber);
      return std::max(nFaceIds, conduit::index_t {5});
   });
}

//---------------------------------------------------------------------------//
void ConduitParMeshBuilder::GetSelectedEdgeTopologyFromAdjset(
   const conduit::Node &n_adjset,
   int group,
   conduit::Node &n_output)
{
   const conduit::Node *n_topo =
      conduit::blueprint::mesh::utils::find_reference_node(n_adjset, "topology");
   assert(n_topo != nullptr);

   // Use the group to pull out edges that use the nodes selected in the adjset.
   const conduit::Node &n_groups = n_adjset.fetch_existing("groups");

   std::set<std::pair<conduit::index_t, conduit::index_t>> usedEdges;

   extract_topology_with_selected_nodes(
      *n_topo,
      n_groups[group]["values"],
      n_output,
      // Invoke this on each shape to make new edges in the output topo.
      [&usedEdges](conduit::index_t elementNumber,
                   const conduit::index_t *faceIds,
                   conduit::index_t nFaceIds,
                   std::vector<int> &connectivity,
                   std::vector<int> &sizes,
                   std::vector<int> &offsets,
                   std::vector<conduit::index_t> &originalElement)
   {
      for (conduit::index_t i = 0; i < nFaceIds; i++)
      {
         const auto p0 = faceIds[i];
         const auto p1 = faceIds[(i + 1) % nFaceIds];
         const auto key = std::make_pair(std::min(p0, p1), std::max(p0, p1));
         if (usedEdges.find(key) == usedEdges.end())
         {
            usedEdges.insert(key);

            offsets.push_back(connectivity.size());
            sizes.push_back(2);
            connectivity.push_back(p0);
            connectivity.push_back(p1);
            originalElement.push_back(elementNumber);
         }
      }
      return 2;
   });
}

//---------------------------------------------------------------------------//
void
ConduitParMeshBuilder::InitSharedFaces(ParMesh *pmesh,
                                       const conduit::Node &n_adjset)
{
   const conduit::Node &n_groups = n_adjset.fetch_existing("groups");
   const auto numGroups = static_cast<int>(n_groups.number_of_children());

   std::vector<std::vector<int>> triGroups, quadGroups;
   std::vector<std::vector<int>> triGroupIDs, quadGroupIDs;
   triGroups.resize(numGroups);
   quadGroups.resize(numGroups);
   triGroupIDs.resize(numGroups);
   quadGroupIDs.resize(numGroups);
   int triFaces = 0, quadFaces = 0;
   for (conduit::index_t g = 0; g < numGroups; g++)
   {
      auto &triGroup = triGroups[g];
      auto &quadGroup = quadGroups[g];
      auto &triGroupID = triGroupIDs[g];
      auto &quadGroupID = quadGroupIDs[g];

      // Get the faces from this group's adjset
      conduit::Node n_group_faces;
      GetSelectedFaceTopologyFromAdjset(n_adjset, g, n_group_faces);

#ifdef DEBUG_SHARED_FACES
      std::stringstream ss;
      ss << "group_faces_rank_" << mfem::Mpi::WorldRank() << "_group_" << g;
      std::string filename(ss.str());
      conduit::relay::io::blueprint::save_mesh(n_group_faces, filename, "hdf5");
      conduit::relay::io::save(n_group_faces, filename + ".yaml", "yaml");
#endif

      // Make accessors for the new topology and its fields.
      conduit::Node &n_topo = n_group_faces["topologies"][0];
      const auto shape = n_topo["elements/shape"].as_string();
      const auto connectivity = n_topo["elements/connectivity"].as_int_ptr();
      const auto sizes = n_topo["elements/sizes"].as_int_ptr();
      const auto numFaces = n_topo["elements/sizes"].dtype().number_of_elements();
      const auto offsets = n_topo["elements/offsets"].as_int_ptr();
      const auto originalElement =
         n_group_faces["fields/originalElement/values"].as_index_t_ptr();

      // Go through the topology zones (faces) and figure out their matching
      // MFEM face numbers. Put them in their correct group too.
      for (int i = 0; i < numFaces; i++)
      {
         const auto zoneIndex = i;
         const auto offset = offsets[zoneIndex];
         const auto size = sizes[zoneIndex];
         const auto origZone = originalElement[zoneIndex];

         // Sort the face's ids
         const auto zoneConn = connectivity + offset;
         std::vector<int> sortedZoneVerts(zoneConn, zoneConn + size);
         std::sort(sortedZoneVerts.begin(), sortedZoneVerts.end());

         // Get the MFEM mesh faces for the original zone. Get the vertices for
         // each face and see if it equals the adjset-generated face. If so, save
         // the MFEM face Id in a vector.
         mfem::Array<int> elemFaces, elemOrientation;
         pmesh->GetElementFaces(origZone, elemFaces, elemOrientation);
         int mfemFace = -1;
         for (int fi = 0; fi < elemFaces.Size(); fi++)
         {
            const auto faceID = elemFaces[fi];
            std::vector<int> ids = GetFaceVertices(pmesh, elemFaces[fi]);

            if (ids == sortedZoneVerts)
            {
               if (ids.size() == 3)
               {
                  mfemFace = faceID;
                  triGroup.push_back(triFaces);
                  triGroupID.push_back(faceID);
                  triFaces++;
               }
               else if (ids.size() == 4)
               {
                  mfemFace = faceID;
                  quadGroup.push_back(quadFaces);
                  quadGroupID.push_back(faceID);
                  quadFaces++;
               }
               break;
            }
         }
         MFEM_ASSERT(mfemFace != -1, "Face not found");
      }
   }

   // Make the tables from the data.
   if (triFaces > 0)
   {
      BuildTable(pmesh->group_stria, triGroups);

      int stri_counter = 0;
      pmesh->shared_trias.SetSize(triFaces);
      for (int g = 0; g < numGroups; g++)
      {
         auto &triGroupID = triGroupIDs[g];
         for (int i = 0; i < triGroupID.size(); i++)
         {
            const auto faceID = triGroupID[i];
            const mfem::Element *face = pmesh->GetFace(faceID);
            const int *fv = face->GetVertices();
            pmesh->shared_trias[stri_counter++].Set(fv);
         }
      }
   }
   if (quadFaces > 0)
   {
      BuildTable(pmesh->group_squad, quadGroups);

      int squad_counter = 0;
      pmesh->shared_quads.SetSize(quadFaces);
      for (int g = 0; g < numGroups; g++)
      {
         auto &quadGroupID = quadGroupIDs[g];
         for (int i = 0; i < quadGroupID.size(); i++)
         {
            const auto faceID = quadGroupID[i];
            const mfem::Element *face = pmesh->GetFace(faceID);
            const int *fv = face->GetVertices();
            pmesh->shared_quads[squad_counter++].Set(fv);
         }
      }
   }

   // Build a map of shared face indices to local face IDs.
   const auto totalFaces = triFaces + quadFaces;
   if (totalFaces > 0)
   {
      pmesh->sface_lface = mfem::Array<int>(totalFaces);
      int pos = 0;
      int stri_counter = 0;
      int squad_counter = 0;
      for (int g = 0; g < numGroups; g++)
      {
         auto &triGroupID = triGroupIDs[g];
         for (int i = 0; i < triGroupID.size(); i++) { pmesh->sface_lface[pos++] = triGroupID[i]; }
      }
      for (int g = 0; g < numGroups; g++)
      {
         auto &quadGroupID = quadGroupIDs[g];
         for (int i = 0; i < quadGroupID.size(); i++) { pmesh->sface_lface[pos++] = quadGroupID[i]; }
      }
   }
}

//---------------------------------------------------------------------------//
void ConduitParMeshBuilder::InitSharedEdges(ParMesh *pmesh,
                                            const conduit::Node &n_adjset)
{
   const conduit::Node &n_groups = n_adjset.fetch_existing("groups");
   const auto numGroups = static_cast<int>(n_groups.number_of_children());

#ifdef DEBUG_SHARED_EDGES
   // Save the MFEM mesh edges to a Blueprint file.
   conduit::Node n_mfem;
   MeshEdges(*pmesh, n_mfem);
   std::stringstream ss1;
   ss1 << "mfem_edges_rank_" << mfem::Mpi::WorldRank();
   std::string filename(ss1.str());
   conduit::relay::io::blueprint::save_mesh(n_mfem, filename, "hdf5");
   conduit::relay::io::save(n_mfem, filename + ".yaml", "yaml");
#endif

   std::vector<std::vector<int>> edgeGroups;
   std::vector<std::vector<int>> edgeGroupIDs;
   std::vector<int> elemIds;
   edgeGroups.resize(numGroups);
   edgeGroupIDs.resize(numGroups);
   int edgeCount = 0;
   for (conduit::index_t g = 0; g < numGroups; g++)
   {
      auto &edgeGroup = edgeGroups[g];
      auto &edgeGroupID = edgeGroupIDs[g];

      // Get the edges from this group's adjset
      conduit::Node n_group_edges;
      GetSelectedEdgeTopologyFromAdjset(n_adjset, g, n_group_edges);

#ifdef DEBUG_SHARED_EDGES
      std::stringstream ss;
      ss << "group_edges_rank_" << mfem::Mpi::WorldRank() << "_group_" << g;
      std::string filename(ss.str());
      conduit::relay::io::blueprint::save_mesh(n_group_edges, filename, "hdf5");
      conduit::relay::io::save(n_group_edges, filename + ".yaml", "yaml");
#endif

      // Make accessors for the new topology and its fields.
      conduit::Node &n_topo = n_group_edges["topologies"][0];
      const auto shape = n_topo["elements/shape"].as_string();
      const auto connectivity = n_topo["elements/connectivity"].as_int_ptr();
      const auto sizes = n_topo["elements/sizes"].as_int_ptr();
      const auto numFaces = n_topo["elements/sizes"].dtype().number_of_elements();
      const auto offsets = n_topo["elements/offsets"].as_int_ptr();

      const auto originalElement =
         n_group_edges["fields/originalElement/values"].as_index_t_ptr();

      // Go through the topology zones (edges) in order
      for (int i = 0; i < numFaces; i++)
      {
         const auto offset = offsets[i];
         const auto size = sizes[i];
         const auto origZone = originalElement[i];

         // Sort the face's ids
         const auto zoneConn = connectivity + offset;
         std::vector<int> sortedZoneVerts(zoneConn, zoneConn + size);
         std::sort(sortedZoneVerts.begin(), sortedZoneVerts.end());

         // Get the MFEM mesh faces for the original zone. Get the vertices for
         // each face and see if it equals the adjset-generated face. If so, save
         // the MFEM face Id in a vector.
         mfem::Array<int> elemEdges, elemOrientation;
         pmesh->GetElementEdges(origZone, elemEdges, elemOrientation);
         int mfemEdge = -1;
         for (int ei = 0; ei < elemEdges.Size(); ei++)
         {
            mfem::Array<int> ids;
            const auto edgeID = elemEdges[ei];
            pmesh->GetEdgeVertices(edgeID, ids);
            std::vector<int> sortedIds(ids.GetData(), ids.GetData() + ids.Size());
            std::sort(sortedIds.begin(), sortedIds.end());
            if (sortedZoneVerts == sortedIds)
            {
               mfemEdge = edgeID;
               edgeGroup.push_back(edgeCount);
               edgeGroupID.push_back(edgeID);
               edgeCount++;
               elemIds.push_back(origZone);
               break;
            }
         }
         MFEM_ASSERT(mfemEdge != -1, "Unable to match edge.");
      }
   }

   // Make the tables from the data.
   if (edgeCount > 0)
   {
      BuildTable(pmesh->group_sedge, edgeGroups);

      int counter = 0;
      pmesh->shared_edges.SetSize(edgeCount);
      pmesh->sedge_ledge = mfem::Array<int>(edgeCount);
      for (int g = 0; g < numGroups; g++)
      {
         auto &edgeGroupID = edgeGroupIDs[g];
         for (int i = 0; i < edgeGroupID.size(); i++)
         {
            pmesh->shared_edges[counter] = pmesh->GetElement(elemIds[counter])->Duplicate(
                                              pmesh);

            pmesh->sedge_ledge[counter] = edgeGroupID[i];

            counter++;
         }
      }
   }
}

//---------------------------------------------------------------------------//
void ConduitParMeshBuilder::BuildTable(mfem::Table &t,
                                       const std::vector<std::vector<int>> &values)
{
   const auto numGroups = static_cast<int>(values.size());

   // Build I
   int *I = new int[numGroups + 1];
   I[0] = 0;
   for (int g = 0; g < numGroups; ++g)
   {
      const auto n = static_cast<int>(values[g].size());
      I[g + 1] = I[g] + n;
   }
   const int nnz = I[numGroups];

   // Build J
   int *J = new int[nnz];
   int pos = 0;
   for (int g = 0; g < numGroups; ++g)
   {
      const auto &row = values[g];
      const auto n = static_cast<int>(row.size());
      for (int c = 0; c < n; c++)
      {
         J[pos++] = row[c];
      }
   }

   // Build the Table from I and J
   t.SetIJ(I, J, numGroups);
}

//---------------------------------------------------------------------------//
std::vector<int> ConduitParMeshBuilder::GetFaceVertices(ParMesh *pmesh,
                                                        int faceID)
{
   mfem::Array<int> edges, orientation;
   pmesh->GetFaceEdges(faceID, edges, orientation);

   // Go through the edges and make a unique list of sorted vertices.
   std::vector<int> ids;
   for (int ei = 0; ei < edges.Size(); ei++)
   {
      mfem::Array<int> edgeVerts;
      pmesh->GetEdgeVertices(edges[ei], edgeVerts);
      for (int vi = 0; vi < edgeVerts.Size(); vi++)
      {
         if (std::find(ids.begin(), ids.end(), edgeVerts[vi]) == ids.end())
         {
            ids.push_back(edgeVerts[vi]);
         }
      }
   }
   std::sort(ids.begin(), ids.end());
   return ids;
}
#endif

//---------------------------------------------------------------------------//
// class ConduitDataCollection implementation
//---------------------------------------------------------------------------//

//------------------------------
// begin public methods
//------------------------------

//---------------------------------------------------------------------------//
ConduitDataCollection::ConduitDataCollection(const std::string& coll_name,
                                             Mesh *mesh)
   : DataCollection(coll_name, mesh),
     relay_protocol("hdf5")
{
   appendRankToFileName = true; // always include rank in file names
   cycle = 0;                   // always include cycle in directory names
}

#ifdef MFEM_USE_MPI
//---------------------------------------------------------------------------//
ConduitDataCollection::ConduitDataCollection(MPI_Comm comm,
                                             const std::string& coll_name,
                                             Mesh *mesh)
   : DataCollection(coll_name, mesh),
     relay_protocol("hdf5")
{
   m_comm = comm;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);
   appendRankToFileName = true; // always include rank in file names
   cycle = 0;                   // always include cycle in directory names
}
#endif

//---------------------------------------------------------------------------//
ConduitDataCollection::~ConduitDataCollection()
{
   // empty
}

//---------------------------------------------------------------------------//
void ConduitDataCollection::Save()
{
   std::string dir_name = MeshDirectoryName();
   int err_ = create_directory(dir_name, mesh, myid);
   if (err_)
   {
      MFEM_ABORT("Error creating directory: " << dir_name);
   }

   Node n_mesh;
   // future? If moved into Mesh class
   // mesh->toConduitBlueprint(n_mesh);
   MeshToBlueprintMesh(mesh,n_mesh);

   Node verify_info;
   if (!blueprint::mesh::verify(n_mesh,verify_info))
   {
      MFEM_ABORT("Conduit Mesh Blueprint Verify Failed:\n"
                 << verify_info.to_json());
   }

   // wrap all grid functions
   FieldMapConstIterator itr;
   for ( itr = field_map.begin(); itr != field_map.end(); itr++)
   {
      std::string name = itr->first;
      GridFunction *gf = itr->second;
      // don't save mesh nodes twice ...
      if (  gf != mesh->GetNodes())
      {
         // future? If moved into GridFunction class
         //gf->toConduitBlueprint(n_mesh["fields"][it->first]);
         GridFunctionToBlueprintField(gf,
                                      n_mesh["fields"][name]);
      }
   }

   // wrap all quadrature functions
   QFieldMapConstIterator qf_itr;
   for ( qf_itr = q_field_map.begin(); qf_itr != q_field_map.end(); qf_itr++)
   {
      std::string name = qf_itr->first;
      QuadratureFunction *qf = qf_itr->second;
      QuadratureFunctionToBlueprintField(qf,
                                         n_mesh["fields"][name]);
   }

   // save mesh data
   SaveMeshAndFields(myid,
                     n_mesh,
                     relay_protocol);

   if (myid == 0)
   {
      // save root file
      SaveRootFile(num_procs,
                   n_mesh,
                   relay_protocol);
   }
}

//---------------------------------------------------------------------------//
void ConduitDataCollection::Load(int cycle)
{
   DeleteAll();
   this->cycle = cycle;

   // Note: We aren't currently using much info from the root file ...
   // with cycle, we can use implicit mfem conduit file layout

   Node n_root;
   LoadRootFile(n_root);
   relay_protocol = n_root["protocol/name"].as_string();

   // for MPI case, we assume that we have # of mpi tasks
   //  == number of domains

   int num_domains = n_root["number_of_trees"].to_int();

   if (num_procs != num_domains)
   {
      error = READ_ERROR;
      MFEM_WARNING("num_procs must equal num_domains");
      return;
   }

   // load the mesh and fields
   LoadMeshAndFields(myid,relay_protocol);

   // TODO: am I properly wielding this?
   own_data = true;

}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::SetProtocol(const std::string &protocol)
{
   relay_protocol = protocol;
}

// Conduit data type id for the MFEM precision
constexpr conduit::index_t mfem_precision_conduit_id =
#if defined(MFEM_USE_DOUBLE)
   CONDUIT_NATIVE_DOUBLE_ID;
#elif defined(MFEM_USE_SINGLE)
   CONDUIT_NATIVE_FLOAT_ID;
#else
#error Unknown MFEM precision
#endif

//------------------------------
// begin static public methods
//------------------------------

//---------------------------------------------------------------------------//
mfem::Mesh *
ConduitDataCollection::BlueprintMeshToMesh(const Node &n_mesh,
                                           const std::string &main_topology_name,
                                           bool zero_copy
#ifdef MFEM_USE_MPI
                                           , MPI_Comm comm
#endif
                                          )
{
   // n_conv holds converted data (when necessary for mfem api)
   // if n_conv is used ( !n_conv.dtype().empty() ) we
   // now that some data allocation was necessary, so we
   // can't return a mesh that zero copies the conduit data
   Node n_conv;

   //
   // we need to find the topology and its coordset.
   //

   std::string topo_name = main_topology_name;
   // if topo name is not set, look for first topology
   if (topo_name == "")
   {
      topo_name = n_mesh["topologies"].schema().child_name(0);
   }

   MFEM_ASSERT(n_mesh.has_path("topologies/" + topo_name),
               "Expected topology named \"" + topo_name + "\" "
               "(node is missing path \"topologies/" + topo_name + "\")");

   // find the coord set
   std::string coords_name =
      n_mesh["topologies"][topo_name]["coordset"].as_string();


   MFEM_ASSERT(n_mesh.has_path("coordsets/" + coords_name),
               "Expected topology named \"" + coords_name + "\" "
               "(node is missing path \"coordsets/" + coords_name + "\")");

   const Node &n_coordset = n_mesh["coordsets"][coords_name];
   const Node &n_coordset_vals = n_coordset["values"];

   // get the number of dims of the coordset
   int ndims = n_coordset_vals.number_of_children();

   // get the number of points
   int num_verts = n_coordset_vals[0].dtype().number_of_elements();
   // get vals for points
   const real_t *verts_ptr = NULL;

   // the mfem mesh constructor needs coords with interleaved (aos) type
   // ordering, even for 1d + 2d we always need 3 real_t (double/float) b/c it
   // uses Array<Vertex> and Vertex is a pod of 3 real_t. we check for this
   // case, if we don't have it we convert the data

   if (ndims == 3 &&
       n_coordset_vals[0].dtype().id() == mfem_precision_conduit_id &&
       blueprint::mcarray::is_interleaved(n_coordset_vals) )
   {
      // already interleaved mcarray of 3 real_t (double/float),
      // return ptr to beginning
      verts_ptr = n_coordset_vals[0].value();
   }
   else
   {
      Node n_tmp;
      // check all vals, if we don't have real_t (double/float) convert
      // to real_t
      NodeConstIterator itr = n_coordset_vals.children();
      while (itr.has_next())
      {
         const Node &c_vals = itr.next();
         std::string c_name = itr.name();

         if ( c_vals.dtype().id() == mfem_precision_conduit_id )
         {
            // zero copy current coords
            n_tmp[c_name].set_external(c_vals);
         }
         else
         {
            // convert
            c_vals.to_data_type(mfem_precision_conduit_id, n_tmp[c_name]);
         }
      }

      // check if we need to add extra dims to get
      // proper interleaved array
      if (ndims < 3)
      {
         // add dummy z
         n_tmp["z"].set(DataType(mfem_precision_conduit_id, num_verts));
      }

      if (ndims < 2)
      {
         // add dummy y
         n_tmp["y"].set(DataType(mfem_precision_conduit_id, num_verts));
      }

      Node &n_conv_coords_vals = n_conv["coordsets"][coords_name]["values"];
      blueprint::mcarray::to_interleaved(n_tmp,
                                         n_conv_coords_vals);
      verts_ptr = n_conv_coords_vals[0].value();
   }



   const Node &n_mesh_topo    = n_mesh["topologies"][topo_name];
   std::string mesh_ele_shape = n_mesh_topo["elements/shape"].as_string();

   mfem::Geometry::Type mesh_geo = ShapeNameToGeomType(mesh_ele_shape);
   int num_idxs_per_ele = Geometry::NumVerts[mesh_geo];

   const Node &n_mesh_conn = n_mesh_topo["elements/connectivity"];

   const int *elem_indices = NULL;
   // mfem requires ints, we could have int64s, etc convert if necessary
   if (n_mesh_conn.dtype().is_int() &&
       n_mesh_conn.is_compact() )
   {
      elem_indices = n_mesh_topo["elements/connectivity"].value();
   }
   else
   {
      Node &n_mesh_conn_conv=
         n_conv["topologies"][topo_name]["elements/connectivity"];
      n_mesh_conn.to_int_array(n_mesh_conn_conv);
      elem_indices = n_mesh_conn_conv.value();
   }

   int num_mesh_ele        =
      n_mesh_topo["elements/connectivity"].dtype().number_of_elements();
   num_mesh_ele            = num_mesh_ele / num_idxs_per_ele;


   const int *bndry_indices = NULL;
   int num_bndry_ele        = 0;
   // init to something b/c the mesh constructor will use this for a
   // table lookup, even if we don't have boundary info.
   mfem::Geometry::Type bndry_geo = mfem::Geometry::POINT;

   if ( n_mesh_topo.has_child("boundary_topology") )
   {
      std::string bndry_topo_name = n_mesh_topo["boundary_topology"].as_string();

      // In VisIt, we encountered a case were a mesh specified a boundary
      // topology, but the boundary topology was omitted from the blueprint
      // index, so it's data could not be obtained.
      //
      // This guard prevents an error in that case, allowing the mesh to be
      // created without boundary info

      if (n_mesh["topologies"].has_child(bndry_topo_name))
      {
         const Node &n_bndry_topo    = n_mesh["topologies"][bndry_topo_name];
         std::string bndry_ele_shape = n_bndry_topo["elements/shape"].as_string();

         bndry_geo = ShapeNameToGeomType(bndry_ele_shape);
         int num_idxs_per_bndry_ele = Geometry::NumVerts[bndry_geo];

         const Node &n_bndry_conn = n_bndry_topo["elements/connectivity"];
         // mfem requires ints, we could have int64s, etc convert if necessary
         if ( n_bndry_conn.dtype().is_int() &&
              n_bndry_conn.is_compact())
         {
            bndry_indices = n_bndry_conn.value();
         }
         else
         {
            Node &n_bndry_conn_conv =
               n_conv["topologies"][bndry_topo_name]["elements/connectivity"];
            n_bndry_conn.to_int_array(n_bndry_conn_conv);
            bndry_indices = (n_bndry_conn_conv).value();

         }

         num_bndry_ele =
            n_bndry_topo["elements/connectivity"].dtype().number_of_elements();
         num_bndry_ele = num_bndry_ele / num_idxs_per_bndry_ele;
      }
   }
   else
   {
      // Skipping Boundary Element Data
   }

   const int *mesh_atts  = NULL;
   const int *bndry_atts = NULL;

   // These variables are used in debug code below.
   // int num_mesh_atts_entires = 0;
   // int num_bndry_atts_entires = 0;

   // the attribute fields could have several names
   // for the element attributes check for first occurrence of field with
   // name containing "_attribute", that doesn't contain "boundary"
   std::string main_att_name = "";

   const Node &n_fields = n_mesh["fields"];
   NodeConstIterator itr = n_fields.children();

   while ( itr.has_next() && main_att_name == "" )
   {
      itr.next();
      std::string fld_name = itr.name();
      if ( fld_name.find("boundary")   == std::string::npos &&
           fld_name.find("_attribute") != std::string::npos )
      {
         main_att_name = fld_name;
      }
   }

   if ( main_att_name != "" )
   {
      const Node &n_mesh_atts_vals = n_fields[main_att_name]["values"];

      // mfem requires ints, we could have int64s, etc convert if necessary
      if (n_mesh_atts_vals.dtype().is_int() &&
          n_mesh_atts_vals.is_compact() )
      {
         mesh_atts = n_mesh_atts_vals.value();
      }
      else
      {
         Node &n_mesh_atts_vals_conv = n_conv["fields"][main_att_name]["values"];
         n_mesh_atts_vals.to_int_array(n_mesh_atts_vals_conv);
         mesh_atts = n_mesh_atts_vals_conv.value();
      }

      // num_mesh_atts_entires = n_mesh_atts_vals.dtype().number_of_elements();
   }
   else
   {
      // Skipping Mesh Attribute Data
   }

   // for the boundary attributes check for first occurrence of field with
   // name containing "_attribute", that also contains "boundary"
   std::string bnd_att_name = "";
   itr = n_fields.children();

   while ( itr.has_next() && bnd_att_name == "" )
   {
      itr.next();
      std::string fld_name = itr.name();
      if ( fld_name.find("boundary")   != std::string::npos &&
           fld_name.find("_attribute") != std::string::npos )
      {
         bnd_att_name = fld_name;
      }
   }

   if ( bnd_att_name != "" )
   {
      // Info: "Getting Boundary Attribute Data"
      const Node &n_bndry_atts_vals =n_fields[bnd_att_name]["values"];

      // mfem requires ints, we could have int64s, etc convert if necessary
      if ( n_bndry_atts_vals.dtype().is_int() &&
           n_bndry_atts_vals.is_compact())
      {
         bndry_atts = n_bndry_atts_vals.value();
      }
      else
      {
         Node &n_bndry_atts_vals_conv = n_conv["fields"][bnd_att_name]["values"];
         n_bndry_atts_vals.to_int_array(n_bndry_atts_vals_conv);
         bndry_atts = n_bndry_atts_vals_conv.value();
      }

      // num_bndry_atts_entires = n_bndry_atts_vals.dtype().number_of_elements();

   }
   else
   {
      // Skipping Boundary Attribute Data
   }

   // Info: "Number of Vertices: " << num_verts  << endl
   //         << "Number of Mesh Elements: "    << num_mesh_ele   << endl
   //         << "Number of Boundary Elements: " << num_bndry_ele  << endl
   //         << "Number of Mesh Attribute Entries: "
   //         << num_mesh_atts_entires << endl
   //         << "Number of Boundary Attribute Entries: "
   //         << num_bndry_atts_entires << endl);

   // Construct MFEM Mesh Object with externally owned data
   // Note: if we don't have a gf, we need to provide the proper space dim
   //       if nodes gf is attached later, it resets the space dim based
   //       on the gf's fes.
   Mesh *mesh = new Mesh(// from coordset
      const_cast<real_t*>(verts_ptr),
      num_verts,
      // from topology
      const_cast<int*>(elem_indices),
      mesh_geo,
      // from mesh_attribute field
      const_cast<int*>(mesh_atts),
      num_mesh_ele,
      // from boundary topology
      const_cast<int*>(bndry_indices),
      bndry_geo,
      // from boundary_attribute field
      const_cast<int*>(bndry_atts),
      num_bndry_ele,
      ndims, // dim
      ndims); // space dim

   // Attach Nodes Grid Function, if it exists
   if (n_mesh_topo.has_child("grid_function"))
   {
      std::string nodes_gf_name = n_mesh_topo["grid_function"].as_string();

      // fetch blueprint field for the nodes gf
      const Node &n_mesh_gf = n_mesh["fields"][nodes_gf_name];
      // create gf
      mfem::GridFunction *nodes = BlueprintFieldToGridFunction(mesh,
                                                               n_mesh_gf);
      // attach to mesh
      mesh->NewNodes(*nodes,true);
   }


   if (zero_copy && !n_conv.dtype().is_empty())
   {
      //Info: "Cannot zero-copy since data conversions were necessary"
      zero_copy = false;
   }

   Mesh *res = NULL;

   if (zero_copy)
   {
      res = mesh;
   }
   else
   {
      // the mesh above contains references to external data, to get a
      // copy independent of the conduit data, we use:
      res = new Mesh(*mesh,true);
      delete mesh;
   }

#ifdef MFEM_USE_MPI
   // If an adjset is found for the topology then we will try to make a ParMesh
   // from the Mesh.
   if (n_mesh.has_path("adjsets"))
   {
      const conduit::Node &n_adjsets = n_mesh["adjsets"];
      for (conduit::index_t i = 0; i < n_adjsets.number_of_children(); i++)
      {
         if (n_adjsets[i]["topology"].as_string() == main_topology_name)
         {
            auto *pmesh = ConduitParMeshBuilder::Build(comm, *res, n_adjsets[i]);
            if (pmesh != nullptr)
            {
               delete res;
               res = pmesh;
            }
            break;
         }
      }
   }
#endif

   return res;
}

//---------------------------------------------------------------------------//
mfem::GridFunction *
ConduitDataCollection::BlueprintFieldToGridFunction(Mesh *mesh,
                                                    const Node &n_field,
                                                    bool zero_copy)
{
   // n_conv holds converted data (when necessary for mfem api)
   // if n_conv is used ( !n_conv.dtype().empty() ) we
   // know that some data allocation was necessary, so we
   // can't return a gf that zero copies the conduit data
   Node n_conv;

   const real_t *vals_ptr = NULL;

   int vdim = 1;

   Ordering::Type ordering = Ordering::byNODES;

   if (n_field["values"].dtype().is_object())
   {
      vdim = n_field["values"].number_of_children();

      // need to check that we have real_t (double/float) and
      // cover supported layouts

      if ( n_field["values"][0].dtype().id() == mfem_precision_conduit_id )
      {
         // check for contig
         if (n_field["values"].is_contiguous())
         {
            // conduit mcarray contig  == mfem byNODES
            vals_ptr = n_field["values"].child(0).value();
         }
         // check for interleaved
         else if (blueprint::mcarray::is_interleaved(n_field["values"]))
         {
            // conduit mcarray interleaved == mfem byVDIM
            ordering = Ordering::byVDIM;
            vals_ptr = n_field["values"].child(0).value();
         }
         else
         {
            // for mcarray generic case --  default to byNODES
            // and provide values w/ contiguous (soa) ordering
            blueprint::mcarray::to_contiguous(n_field["values"],
                                              n_conv["values"]);
            vals_ptr = n_conv["values"].child(0).value();
         }
      }
      else // convert to real_t (double/float) and use contig
      {
         Node n_tmp;
         // check all vals, if we don't have real_t (double/float) convert
         // to real_t
         NodeConstIterator itr = n_field["values"].children();
         while (itr.has_next())
         {
            const Node &c_vals = itr.next();
            std::string c_name = itr.name();

            if ( c_vals.dtype().id() == mfem_precision_conduit_id )
            {
               // zero copy current coords
               n_tmp[c_name].set_external(c_vals);
            }
            else
            {
               // convert
               c_vals.to_data_type(mfem_precision_conduit_id, n_tmp[c_name]);
            }
         }

         // for mcarray generic case --  default to byNODES
         // and provide values w/ contiguous (soa) ordering
         blueprint::mcarray::to_contiguous(n_tmp,
                                           n_conv["values"]);
         vals_ptr = n_conv["values"].child(0).value();
      }
   }
   else
   {
      if (n_field["values"].dtype().id() == mfem_precision_conduit_id &&
          n_field["values"].is_compact())
      {
         vals_ptr = n_field["values"].value();
      }
      else
      {
         n_field["values"].to_data_type(mfem_precision_conduit_id,
                                        n_conv["values"]);
         vals_ptr = n_conv["values"].value();
      }
   }

   if (zero_copy && !n_conv.dtype().is_empty())
   {
      //Info: "Cannot zero-copy since data conversions were necessary"
      zero_copy = false;
   }

   // we need basis name to create the proper mfem fec
   std::string fec_name;
   if (n_field.has_path("basis"))
   {
      fec_name = n_field["basis"].as_string();
   }
   else
   {
      namespace bputils = conduit::blueprint::mesh::utils;
      // There was no basis so pick an appropriate one for typical Blueprint data.
      int dim = 2;
      const auto association = n_field["association"].as_string();
      const conduit::Node *n_topo =
         bputils::find_reference_node(n_field, "topology");
      if (n_topo != nullptr)
      {
         bputils::ShapeType shape(*n_topo);
         dim = shape.dim;
      }
      std::stringstream ss;
      if (association == "element")
      {
         ss << "L2_" << dim << "D_P0";
      }
      else
      {
         ss << "H1_" << dim << "D_P1";
      }
      fec_name = ss.str();
      mfem::out << "Picked fec_name: " << fec_name << std::endl;
   }

   GridFunction *res = NULL;
   mfem::FiniteElementCollection *fec = FiniteElementCollection::New(
                                           fec_name.c_str());
   mfem::FiniteElementSpace *fes = new FiniteElementSpace(mesh,
                                                          fec,
                                                          vdim,
                                                          ordering);

   if (zero_copy)
   {
      res = new GridFunction(fes,const_cast<real_t*>(vals_ptr));
   }
   else
   {
      // copy case, this constructor will alloc the space for the GF data
      res = new GridFunction(fes);
      // create an mfem vector that wraps the conduit data
      Vector vals_vec(const_cast<real_t*>(vals_ptr),fes->GetVSize());
      // copy values into the result
      (*res) = vals_vec;
   }

   // TODO: I believe the GF already has ownership of fes, so this should be all
   // we need to do to avoid leaking objs created here?
   res->MakeOwner(fec);

   return res;
}

//---------------------------------------------------------------------------//
mfem::QuadratureFunction *
ConduitDataCollection::BlueprintFieldToQuadratureFunction(Mesh *mesh,
                                                          const Node &n_field,
                                                          bool zero_copy)
{
   // n_conv holds converted data (when necessary for mfem api)
   // if n_conv is used ( !n_conv.dtype().empty() ) we
   // know that some data allocation was necessary, so we
   // can't return a qf that zero copies the conduit data
   Node n_conv;

   const real_t *vals_ptr = NULL;
   int vdim = 1;

   if (n_field["values"].dtype().is_object())
   {
      vdim = n_field["values"].number_of_children();

      // need to check that we have real_t (double/float) and
      // cover supported layouts
      if ( n_field["values"][0].dtype().id() == mfem_precision_conduit_id )
      {
         // quad funcs use what mfem calls byVDIM
         // and what conduit calls interleaved
         // check for interleaved
         if (blueprint::mcarray::is_interleaved(n_field["values"]))
         {
            // conduit mcarray interleaved == mfem byVDIM
            vals_ptr = n_field["values"].child(0).value();
         }
         else
         {
            // for mcarray generic case --  default to byVDIM
            // aka interleaved
            blueprint::mcarray::to_interleaved(n_field["values"],
                                               n_conv["values"]);
            vals_ptr = n_conv["values"].child(0).value();
         }
      }
      else // convert to real_t (double/float) and use interleaved
      {
         Node n_tmp;
         // check all vals, if we don't have real_t (double/float) convert
         // to real_t
         NodeConstIterator itr = n_field["values"].children();
         while (itr.has_next())
         {
            const Node &c_vals = itr.next();
            std::string c_name = itr.name();

            if ( c_vals.dtype().id() == mfem_precision_conduit_id )
            {
               // zero copy current coords
               n_tmp[c_name].set_external(c_vals);
            }
            else
            {
               // convert
               c_vals.to_data_type(mfem_precision_conduit_id, n_tmp[c_name]);
            }
         }

         // for mcarray generic case --  default to byVDIM
         // aka interleaved
         blueprint::mcarray::to_interleaved(n_tmp,
                                            n_conv["values"]);
         vals_ptr = n_conv["values"].child(0).value();
      }
   }
   else // scalar case
   {
      if (n_field["values"].dtype().id() == mfem_precision_conduit_id &&
          n_field["values"].is_compact())
      {
         vals_ptr = n_field["values"].value();
      }
      else
      {
         n_field["values"].to_data_type(mfem_precision_conduit_id,
                                        n_conv["values"]);
         vals_ptr = n_conv["values"].value();
      }
   }

   if (zero_copy && !n_conv.dtype().is_empty())
   {
      //Info: "Cannot zero-copy since data conversions were necessary"
      zero_copy = false;
   }

   // we need basis name to create the proper mfem quad space and quad func
   // the pattern used to encode the quad space params is:
   // QF_{ORDER}_{VDIM}
   // ORDER is the degree of the polynomials for the quad rule
   // VDIM  is the number of components at each quad point (scalar, vector, etc)

   int qf_order = 0;
   int qf_vdim  = 0;
   std::string qf_name = n_field["basis"].as_string();
   const char *qf_name_cstr = qf_name.c_str();
   if (!strncmp(qf_name_cstr, "QF_", 3))
   {
      // parse {ORDER}
      qf_order = atoi(qf_name_cstr + 3);
      // find second `_`
      const char *qf_vdim_cstr = strstr(qf_name_cstr+3,"_");
      if (qf_vdim_cstr == NULL)
      {
         MFEM_ABORT("Error parsing quadrature function description string: "
                    << qf_name << std::endl
                    << "Expected: QF_{ORDER}_{VDIM}");
      }
      // parse {VDIM}
      qf_vdim  = atoi(qf_vdim_cstr+1);
   }
   else
   {
      MFEM_ABORT("Error parsing quadrature function description string: "
                 << qf_name << std::endl
                 << "Expected: QF_{ORDER}_{VDIM}");
   }
   MFEM_VERIFY(qf_vdim == vdim, "vector dimension mismatch: vdim = " << vdim
               << ", qf_vdim = " << qf_vdim);

   mfem::QuadratureSpace *quad_space = new mfem::QuadratureSpace(mesh, qf_order);
   mfem::QuadratureFunction *res = new mfem::QuadratureFunction();

   if (zero_copy)
   {
      res->SetSpace(quad_space, const_cast<real_t*>(vals_ptr), vdim);
      res->SetOwnsSpace(true);
   }
   else
   {
      res->SetSpace(quad_space, vdim);
      res->SetOwnsSpace(true);
      // copy case, this constructor will alloc the space for the quad data
      // create an mfem vector that wraps the conduit data
      Vector vals_vec(const_cast<real_t*>(vals_ptr),res->Size());
      // copy values into the result
      (*res) = vals_vec;
   }

   return res;
}



//---------------------------------------------------------------------------//
void
ConduitDataCollection::MeshToBlueprintMesh(Mesh *mesh,
                                           Node &n_mesh,
                                           const std::string &coordset_name,
                                           const std::string &main_topology_name,
                                           const std::string &boundary_topology_name,
                                           const std::string &main_adjset_name)
{
   int dim = mesh->SpaceDimension();

   MFEM_ASSERT(dim >= 1 && dim <= 3, "invalid mesh dimension");

   ////////////////////////////////////////////
   // Setup main coordset
   ////////////////////////////////////////////

   // Assumes  mfem::Vertex has the layout of a real_t (double/float) array.

   // this logic assumes an mfem vertex is always 3 real_t (double/float) wide
   int stride = sizeof(mfem::Vertex);
   int num_vertices = mesh->GetNV();

   MFEM_ASSERT( ( stride == 3 * sizeof(real_t) ),
                "Unexpected stride for Vertex");

   Node &n_mesh_coords = n_mesh["coordsets"][coordset_name];
   n_mesh_coords["type"] =  "explicit";


   real_t *coords_ptr = mesh->GetVertex(0);

   n_mesh_coords["values/x"].set_external(coords_ptr,
                                          num_vertices,
                                          0,
                                          stride);

   if (dim >= 2)
   {
      n_mesh_coords["values/y"].set_external(coords_ptr,
                                             num_vertices,
                                             sizeof(real_t),
                                             stride);
   }
   if (dim >= 3)
   {
      n_mesh_coords["values/z"].set_external(coords_ptr,
                                             num_vertices,
                                             sizeof(real_t) * 2,
                                             stride);
   }

   ////////////////////////////////////////////
   // Setup main topo
   ////////////////////////////////////////////

   Node &n_topo = n_mesh["topologies"][main_topology_name];

   n_topo["type"]  = "unstructured";
   n_topo["coordset"] = coordset_name;

   Element::Type ele_type = mesh->GetElementType(0);

   std::string ele_shape = ElementTypeToShapeName(ele_type);

   n_topo["elements/shape"] = ele_shape;

   GridFunction *gf_mesh_nodes = mesh->GetNodes();

   if (gf_mesh_nodes != NULL)
   {
      n_topo["grid_function"] =  "mesh_nodes";
   }

   // connectivity
   // TODO: generic case, i don't think we can zero-copy (mfem allocs
   // an array per element) so we alloc our own contig array and
   // copy out. Some other cases (sidre) may actually have contig
   // allocation but I am  not sure how to detect this case from mfem
   int num_ele = mesh->GetNE();
   int geom = mesh->GetTypicalElementGeometry();
   int idxs_per_ele = Geometry::NumVerts[geom];
   int num_conn_idxs =  num_ele * idxs_per_ele;

   n_topo["elements/connectivity"].set(DataType::c_int(num_conn_idxs));

   int *conn_ptr = n_topo["elements/connectivity"].value();

   for (int i=0; i < num_ele; i++)
   {
      const Element *ele = mesh->GetElement(i);
      const int *ele_verts = ele->GetVertices();

      memcpy(conn_ptr, ele_verts, idxs_per_ele * sizeof(int));

      conn_ptr += idxs_per_ele;
   }

   if (gf_mesh_nodes != NULL)
   {
      GridFunctionToBlueprintField(gf_mesh_nodes,
                                   n_mesh["fields/mesh_nodes"],
                                   main_topology_name);
   }

   ////////////////////////////////////////////
   // Setup mesh attribute
   ////////////////////////////////////////////

   Node &n_mesh_att = n_mesh["fields/element_attribute"];

   n_mesh_att["association"] = "element";
   n_mesh_att["topology"] = main_topology_name;
   n_mesh_att["values"].set(DataType::c_int(num_ele));

   int_array att_vals = n_mesh_att["values"].value();
   for (int i = 0; i < num_ele; i++)
   {
      att_vals[i] = mesh->GetAttribute(i);
   }

   ////////////////////////////////////////////
   // Setup bndry topo "boundary"
   ////////////////////////////////////////////

   // guard vs if we have boundary elements
   if (mesh->HasBoundaryElements())
   {
      n_topo["boundary_topology"] = boundary_topology_name;

      Node &n_bndry_topo = n_mesh["topologies"][boundary_topology_name];

      n_bndry_topo["type"]     = "unstructured";
      n_bndry_topo["coordset"] = coordset_name;

      int num_bndry_ele = mesh->GetNBE();

      Element *BE0 = NULL; // representative boundary element
      if (num_bndry_ele > 0) { BE0 = mesh->GetBdrElement(0); }

      // must initialize this to something, pick POINT if no boundary elements
      Element::Type bndry_ele_type   = (BE0) ? BE0->GetType() : Element::POINT;
      std::string bndry_ele_shape    = ElementTypeToShapeName(bndry_ele_type);
      n_bndry_topo["elements/shape"] = bndry_ele_shape;

      // must initialize this to something, pick POINT if no boundary elements
      int bndry_geom          = (BE0) ? BE0->GetGeometryType() : Geometry::POINT;
      int bndry_idxs_per_ele  = Geometry::NumVerts[bndry_geom];
      int num_bndry_conn_idxs = num_bndry_ele * bndry_idxs_per_ele;

      n_bndry_topo["elements/connectivity"].set(DataType::c_int(num_bndry_conn_idxs));

      int *bndry_conn_ptr = n_bndry_topo["elements/connectivity"].value();

      for (int i=0; i < num_bndry_ele; i++)
      {
         const Element *bndry_ele = mesh->GetBdrElement(i);
         const int *bndry_ele_verts = bndry_ele->GetVertices();

         memcpy(bndry_conn_ptr, bndry_ele_verts, bndry_idxs_per_ele  * sizeof(int));

         bndry_conn_ptr += bndry_idxs_per_ele;
      }

      ////////////////////////////////////////////
      // Setup bndry mesh attribute
      ////////////////////////////////////////////

      Node &n_bndry_mesh_att = n_mesh["fields/boundary_attribute"];

      n_bndry_mesh_att["association"] = "element";
      n_bndry_mesh_att["topology"] = boundary_topology_name;
      n_bndry_mesh_att["values"].set(DataType::c_int(num_bndry_ele));

      int_array bndry_att_vals = n_bndry_mesh_att["values"].value();
      for (int i = 0; i < num_bndry_ele; i++)
      {
         bndry_att_vals[i] = mesh->GetBdrAttribute(i);
      }
   }

   ////////////////////////////////////////////
   // Setup adjsets
   ////////////////////////////////////////////

#ifdef MFEM_USE_MPI
   ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
   if (pmesh)
   {
      ////////////////////////////////////////////
      // Setup main adjset
      ////////////////////////////////////////////

      Node &n_adjset = n_mesh["adjsets"][main_adjset_name];

      n_adjset["association"] = "vertex";
      n_adjset["topology"] = main_topology_name;
      n_adjset["groups"].set(DataType::object());

      const GroupTopology &pmesh_gtopo = pmesh->gtopo;
      const int local_rank = pmesh->GetMyRank();
      const int num_groups = pmesh_gtopo.NGroups();
      // NOTE: skip the first group since its the local-only group
      for (int i = 1; i < num_groups; i++)
      {
         const int num_group_nbrs = pmesh_gtopo.GetGroupSize(i);
         const int *group_nbrs = pmesh_gtopo.GetGroup(i);
         const int num_group_verts = pmesh->GroupNVertices(i);

         // NOTE: 'neighbor' values are local to this processor, but Blueprint
         // expects global domain identifiers, so we collapse this layer of
         // indirection
         Array<int> group_ranks(num_group_nbrs);
         std::string group_name = "group";
         {
            for (int j = 0; j < num_group_nbrs; j++)
            {
               group_ranks[j] = pmesh_gtopo.GetNeighborRank(group_nbrs[j]);
            }
            group_ranks.Sort();
            for (int j = 0; j < num_group_nbrs; j++)
            {
               group_name += "_" + std::to_string(group_ranks[j]);
            }

            // NOTE: Blueprint only wants remote ranks in its neighbor list,
            // so we remove the local rank after the canonicalized Blueprint
            // group name is formed
            group_ranks.DeleteFirst(local_rank);
         }
         Node &n_group = n_adjset["groups"][group_name];

         n_group["neighbors"].set(group_ranks.GetData(), group_ranks.Size());
         n_group["values"].set(DataType::c_int(num_group_verts));

         int_array group_vals = n_group["values"].value();
         for (int j = 0; j < num_group_verts; j++)
         {
            group_vals[j] = pmesh->GroupVertex(i, j);
         }
      }

      // NOTE: We don't create an adjset for face neighbor data because
      // these faces aren't listed in the 'boundary_topology_name' topology
      // (this topology only covers the faces between 'main_topology_name'
      // elements and void). To include a face neighbor data adjset, this
      // function would need to export a topology with either (1) all faces
      // in the mesh topology or (2) all boundary faces, including neighbors.

      ////////////////////////////////////////////
      // Setup distributed state
      ////////////////////////////////////////////

      Node &n_domid = n_mesh["state/domain_id"];
      n_domid.set(local_rank);
   }
#endif
}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::GridFunctionToBlueprintField(mfem::GridFunction *gf,
                                                    Node &n_field,
                                                    const std::string &main_topology_name)
{
   n_field["basis"] = gf->FESpace()->FEColl()->Name();
   n_field["topology"] = main_topology_name;

   int vdim  = gf->FESpace()->GetVDim();
   int ndofs = gf->FESpace()->GetNDofs();

   if (vdim == 1) // scalar case
   {
      n_field["values"].set_external(const_cast<real_t *>(gf->HostRead()),
                                     ndofs);
   }
   else // vector case
   {
      // deal with striding of all components

      Ordering::Type ordering = gf->FESpace()->GetOrdering();

      int entry_stride = (ordering == Ordering::byNODES ? 1 : vdim);
      int vdim_stride  = (ordering == Ordering::byNODES ? ndofs : 1);

      index_t offset = 0;
      index_t stride = sizeof(real_t) * entry_stride;

      for (int d = 0;  d < vdim; d++)
      {
         std::ostringstream oss;
         oss << "v" << d;
         std::string comp_name = oss.str();
         n_field["values"][comp_name].set_external(const_cast<real_t *>(gf->HostRead()),
                                                   ndofs,
                                                   offset,
                                                   stride);
         offset +=  sizeof(real_t) * vdim_stride;
      }
   }

}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::QuadratureFunctionToBlueprintField(
   mfem::QuadratureFunction *qf,
   Node &n_field,
   const std::string &main_topology_name)
{
   // For quadrature functions, use basis pattern:
   //   QF_{ORDER}_{VDIM}

   int qf_vdim  = qf->GetVDim();
   int qf_order = qf->GetSpace()->GetOrder();
   int qf_size  = qf->GetSpace()->GetSize();

   {
      std::ostringstream oss;
      oss << "QF_" << qf_order << "_" << qf_vdim;

      n_field["basis"] = oss.str();
      n_field["topology"] = main_topology_name;
   }

   if (qf_vdim == 1) // scalar case
   {
      n_field["values"].set_external(const_cast<real_t *>(qf->HostRead()),
                                     qf_size);
   }
   else // vector case
   {
      // deal with striding of all components
      // quadrature functions are always byVDIM
      // or what conduit calls interleaved

      index_t offset = 0;
      index_t stride = sizeof(real_t) * qf_vdim;

      for (int d = 0;  d < qf_vdim; d++)
      {
         std::ostringstream oss;
         oss << "v" << d;
         std::string comp_name = oss.str();
         n_field["values"][comp_name].set_external(const_cast<real_t *>(qf->HostRead()),
                                                   qf_size,
                                                   offset,
                                                   stride);
         offset += sizeof(real_t);
      }
   }

}



//------------------------------
// end static public methods
//------------------------------

//------------------------------
// end public methods
//------------------------------

//------------------------------
// begin protected methods
//------------------------------

//---------------------------------------------------------------------------//
std::string
ConduitDataCollection::RootFileName()
{
   std::string res = prefix_path + name + "_" +
                     to_padded_string(cycle, pad_digits_cycle) +
                     ".root";
   return res;
}

//---------------------------------------------------------------------------//
std::string
ConduitDataCollection::MeshFileName(int domain_id,
                                    const std::string &relay_protocol_)
{
   std::string res = prefix_path +
                     name  +
                     "_" +
                     to_padded_string(cycle, pad_digits_cycle) +
                     "/domain_" +
                     to_padded_string(domain_id, pad_digits_rank) +
                     "." +
                     relay_protocol_;

   return res;
}

//---------------------------------------------------------------------------//
std::string
ConduitDataCollection::MeshDirectoryName()
{
   std::string res = prefix_path +
                     name +
                     "_" +
                     to_padded_string(cycle, pad_digits_cycle);
   return res;
}

//---------------------------------------------------------------------------//
std::string
ConduitDataCollection::MeshFilePattern(const std::string &relay_protocol_)
{
   std::ostringstream oss;
   oss << name
       << "_"
       << to_padded_string(cycle, pad_digits_cycle)
       << "/domain_%0"
       << pad_digits_rank
       << "d."
       << relay_protocol_;

   return oss.str();
}


//---------------------------------------------------------------------------//
void
ConduitDataCollection::SaveRootFile(int num_domains,
                                    const Node &n_mesh,
                                    const std::string &relay_protocol_)
{
   // default to json root file, except for hdf5 case
   std::string root_proto = "json";

   if (relay_protocol_ == "hdf5")
   {
      root_proto = relay_protocol_;
   }

   Node n_root;
   // create blueprint index
   Node &n_bp_idx = n_root["blueprint_index"];

   blueprint::mesh::generate_index(n_mesh,
                                   "",
                                   num_domains,
                                   n_bp_idx["mesh"]);

   // there are cases where the data backing the gf fields doesn't
   // accurately represent the number of components in physical space,
   // so we loop over all gfs and fix those that are incorrect

   FieldMapConstIterator itr;
   for ( itr = field_map.begin(); itr != field_map.end(); itr++)
   {
      std::string gf_name = itr->first;
      GridFunction *gf = itr->second;

      Node &idx_gf_ncomps = n_bp_idx["mesh/fields"][gf_name]["number_of_components"];
      // check that the number_of_components in the index matches what we expect
      // correct if necessary
      if ( idx_gf_ncomps.to_int() != gf->VectorDim() )
      {
         idx_gf_ncomps = gf->VectorDim();
      }
   }
   // add extra header info
   n_root["protocol/name"]    =  relay_protocol_;
   n_root["protocol/version"] = "0.3.1";


   // we will save one file per domain, so trees == files
   n_root["number_of_files"]  = num_domains;
   n_root["number_of_trees"]  = num_domains;
   n_root["file_pattern"]     = MeshFilePattern(relay_protocol_);
   n_root["tree_pattern"]     = "";

   // Add the time, time step, and cycle
   n_root["blueprint_index/mesh/state/time"] = time;
   n_root["blueprint_index/mesh/state/time_step"] = time_step;
   n_root["blueprint_index/mesh/state/cycle"] = cycle;

   relay::io::save(n_root, RootFileName(), root_proto);
}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::SaveMeshAndFields(int domain_id,
                                         const Node &n_mesh,
                                         const std::string &relay_protocol_)
{
   relay::io::save(n_mesh, MeshFileName(domain_id, relay_protocol_));
}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::LoadRootFile(Node &root_out)
{
   if (myid == 0)
   {
      // assume root file is json, unless hdf5 is specified
      std::string root_protocol = "json";

      if ( relay_protocol.find("hdf5") != std::string::npos )
      {
         root_protocol = "hdf5";
      }


      relay::io::load(RootFileName(), root_protocol, root_out);
#ifdef MFEM_USE_MPI
      // broadcast contents of root file other ranks
      // (conduit relay mpi  would simplify, but we would need to link another
      // lib for mpi case)

      // create json string
      std::string root_json = root_out.to_json();
      // string size +1 for null term
      int json_str_size = root_json.size() + 1;

      // broadcast json string buffer size
      int mpi_status = MPI_Bcast((void*)&json_str_size, // ptr
                                 1, // size
                                 MPI_INT, // type
                                 0, // root
                                 m_comm); // comm

      if (mpi_status != MPI_SUCCESS)
      {
         MFEM_ABORT("Broadcast of root file json string size failed");
      }

      // broadcast json string
      mpi_status = MPI_Bcast((void*)root_json.c_str(), // ptr
                             json_str_size, // size
                             MPI_CHAR, // type
                             0, // root
                             m_comm); // comm

      if (mpi_status != MPI_SUCCESS)
      {
         MFEM_ABORT("Broadcast of root file json string failed");
      }

#endif
   }

#ifdef MFEM_USE_MPI
   else
   {
      // recv json string buffer size via broadcast
      int json_str_size = -1;
      int mpi_status = MPI_Bcast(&json_str_size, // ptr
                                 1, // size
                                 MPI_INT, // type
                                 0, // root
                                 m_comm); // comm

      if (mpi_status != MPI_SUCCESS)
      {
         MFEM_ABORT("Broadcast of root file json string size failed");
      }

      // recv json string buffer via broadcast
      char *json_buff = new char[json_str_size];
      mpi_status = MPI_Bcast(json_buff,  // ptr
                             json_str_size, // size
                             MPI_CHAR, // type
                             0, // root
                             m_comm); // comm

      if (mpi_status != MPI_SUCCESS)
      {
         MFEM_ABORT("Broadcast of root file json string failed");
      }

      // reconstruct root file contents
      Generator g(std::string(json_buff),"json");
      g.walk(root_out);
      // cleanup temp buffer
      delete [] json_buff;
   }
#endif
}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::LoadMeshAndFields(int domain_id,
                                         const std::string &relay_protocol_)
{
   // Note: This path doesn't use any info from the root file
   // it uses the implicit mfem ConduitDataCollection layout

   Node n_mesh;
   relay::io::load( MeshFileName(domain_id, relay_protocol_), n_mesh);


   Node verify_info;
   if (!blueprint::mesh::verify(n_mesh,verify_info))
   {
      MFEM_ABORT("Conduit Mesh Blueprint Verify Failed:\n"
                 << verify_info.to_json());
   }

   mesh = BlueprintMeshToMesh(n_mesh);

   field_map.clear();

   NodeConstIterator itr = n_mesh["fields"].children();

   std::string nodes_gf_name = "";

   const Node &n_topo = n_mesh["topologies/main"];
   if (n_topo.has_child("grid_function"))
   {
      nodes_gf_name = n_topo["grid_function"].as_string();
   }

   while (itr.has_next())
   {
      const Node &n_field = itr.next();
      std::string field_name = itr.name();

      // skip mesh nodes gf since they are already processed
      // skip attribute fields, they aren't grid functions
      if ( field_name != nodes_gf_name &&
           field_name.find("_attribute") == std::string::npos
         )
      {
         GridFunction *gf = BlueprintFieldToGridFunction(mesh, n_field);
         field_map.Register(field_name, gf, true);
      }
   }
}

//------------------------------
// end protected methods
//------------------------------

//------------------------------
// begin static private methods
//------------------------------

//---------------------------------------------------------------------------//
std::string
ConduitDataCollection::ElementTypeToShapeName(Element::Type element_type)
{
   // Adapted from SidreDataCollection

   // Note -- the mapping from Element::Type to string is based on
   //   enum Element::Type { POINT, SEGMENT, TRIANGLE, QUADRILATERAL,
   //                        TETRAHEDRON, HEXAHEDRON };
   // Note: -- the string names are from conduit's blueprint

   switch (element_type)
   {
      case Element::POINT:          return "point";
      case Element::SEGMENT:        return "line";
      case Element::TRIANGLE:       return "tri";
      case Element::QUADRILATERAL:  return "quad";
      case Element::TETRAHEDRON:    return "tet";
      case Element::HEXAHEDRON:     return "hex";
      case Element::WEDGE:
      default: ;
   }

   return "unknown";
}

//---------------------------------------------------------------------------//
mfem::Geometry::Type
ConduitDataCollection::ShapeNameToGeomType(const std::string &shape_name)
{
   // Note: must init to something to avoid invalid memory access
   // in the mfem mesh constructor
   mfem::Geometry::Type res = mfem::Geometry::POINT;

   if (shape_name == "point")
   {
      res = mfem::Geometry::POINT;
   }
   else if (shape_name == "line")
   {
      res =  mfem::Geometry::SEGMENT;
   }
   else if (shape_name == "tri")
   {
      res =  mfem::Geometry::TRIANGLE;
   }
   else if (shape_name == "quad")
   {
      res =  mfem::Geometry::SQUARE;
   }
   else if (shape_name == "tet")
   {
      res =  mfem::Geometry::TETRAHEDRON;
   }
   else if (shape_name == "hex")
   {
      res =  mfem::Geometry::CUBE;
   }
   else
   {
      MFEM_ABORT("Unsupported Element Shape: " << shape_name);
   }

   return res;
}

//------------------------------
// end static private methods
//------------------------------

} // end namespace mfem

#endif
