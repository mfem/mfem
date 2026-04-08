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

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_CONDUIT)

namespace
{

std::string FormatPoint(const real_t *pt, int dim)
{
   std::ostringstream os;
   os.setf(std::ios::fixed);
   os.precision(12);
   for (int d = 0; d < dim; ++d)
   {
      if (d > 0) { os << ','; }
      os << pt[d];
   }
   return os.str();
}

std::string Join(const std::vector<std::string> &items)
{
   std::ostringstream os;
   for (size_t i = 0; i < items.size(); ++i)
   {
      if (i > 0) { os << " | "; }
      os << items[i];
   }
   return os.str();
}

int SignatureToken(const std::string &text)
{
   uint32_t hash = 2166136261u;
   for (unsigned char c : text)
   {
      hash ^= c;
      hash *= 16777619u;
   }
   return static_cast<int>(hash & 0x7fffffff);
}

void SetBoundaryAttributes(Mesh &mesh, real_t sx, real_t sy)
{
   const real_t scale = std::max<real_t>(1.0, std::max(sx, sy));
   const real_t tol = 1.0e-12 * scale;
   Array<int> vertices;

   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      mesh.GetBdrElementVertices(be, vertices);

      real_t xc = 0.0;
      real_t yc = 0.0;
      for (int j = 0; j < vertices.Size(); ++j)
      {
         const real_t *v = mesh.GetVertex(vertices[j]);
         xc += v[0];
         yc += v[1];
      }
      xc /= vertices.Size();
      yc /= vertices.Size();

      const bool on_left = std::abs(xc) <= tol;
      const bool on_bottom = std::abs(yc) <= tol;
      const bool on_right = std::abs(xc - sx) <= tol;
      const bool on_top = std::abs(yc - sy) <= tol;

      const bool on_boundary = on_left || on_bottom || on_right || on_top;
      REQUIRE(on_boundary);
      mesh.SetBdrAttribute(be, (on_left || on_bottom) ? 1 : 2);
   }

   mesh.SetAttributes();
}

void BuildCartesianPartitioning(Mesh &mesh, Array<int> &partitioning)
{
   int dims[2] = { 0, 0 };
   MPI_Dims_create(Mpi::WorldSize(), 2, dims);

   int *raw_partitioning = mesh.CartesianPartitioning(dims);
   partitioning.SetSize(mesh.GetNE());
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      partitioning[e] = raw_partitioning[e];
   }
   delete [] raw_partitioning;
}

std::vector<std::string> CollectBoundaryAttributeSignature(const ParMesh &pmesh)
{
   std::map<int, int> counts;
   for (int be = 0; be < pmesh.GetNBE(); ++be)
   {
      counts[pmesh.GetBdrAttribute(be)]++;
   }

   std::vector<std::string> signature;
   signature.reserve(counts.size());
   for (const auto &entry : counts)
   {
      std::ostringstream os;
      os << entry.first << ':' << entry.second;
      signature.push_back(os.str());
   }
   return signature;
}

std::vector<std::string> CollectGroupSignature(const ParMesh &pmesh)
{
   std::vector<std::string> signature;
   signature.reserve(std::max(0, pmesh.GetNGroups() - 1));
   for (int g = 1; g < pmesh.GetNGroups(); ++g)
   {
      std::ostringstream os;
      os << "g=" << g
         << ",master=" << pmesh.gtopo.GetGroupMasterRank(g)
         << ",v=" << pmesh.GroupNVertices(g)
         << ",e=" << pmesh.GroupNEdges(g)
         << ",t=" << pmesh.GroupNTriangles(g)
         << ",q=" << pmesh.GroupNQuadrilaterals(g);
      signature.push_back(os.str());
   }
   return signature;
}

std::vector<std::string> CollectSharedFaceSignature(ParMesh &pmesh)
{
   std::vector<std::string> signature;
   signature.reserve(pmesh.GetNSharedFaces());
   Array<int> vertices;
   std::vector<std::string> point_signature;

   for (int sf = 0; sf < pmesh.GetNSharedFaces(); ++sf)
   {
      const int local_face = pmesh.GetSharedFace(sf);
      pmesh.GetFaceVertices(local_face, vertices);

      point_signature.clear();
      point_signature.reserve(vertices.Size());
      for (int j = 0; j < vertices.Size(); ++j)
      {
         point_signature.push_back(
            FormatPoint(pmesh.GetVertex(vertices[j]), pmesh.SpaceDimension()));
      }
      std::sort(point_signature.begin(), point_signature.end());

      std::ostringstream os;
      os << "verts=" << Join(point_signature);
      signature.push_back(os.str());
   }

   std::sort(signature.begin(), signature.end());
   return signature;
}

std::vector<std::string> CollectMeshSummary(ParMesh &pmesh)
{
   std::vector<std::string> summary;
   summary.push_back("NV=" + std::to_string(pmesh.GetNV()));
   summary.push_back("NE=" + std::to_string(pmesh.GetNE()));
   summary.push_back("NBE=" + std::to_string(pmesh.GetNBE()));
   summary.push_back("groups=" + std::to_string(pmesh.GetNGroups()));
   summary.push_back("shared_faces=" + std::to_string(pmesh.GetNSharedFaces()));
   return summary;
}

std::vector<std::string> CollectFESpaceSignature(ParMesh &pmesh)
{
   const int dim = pmesh.Dimension();

   L2_FECollection l2_fec(1, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace l2_pfes(&pmesh, &l2_fec);
   l2_pfes.ExchangeFaceNbrData();

   H1_FECollection h1_fec(1, dim);
   ParFiniteElementSpace h1_pfes(&pmesh, &h1_fec);

   std::vector<std::string> signature;
   {
      std::ostringstream os;
      os << "L2:V=" << l2_pfes.GetVSize()
         << ",TV=" << l2_pfes.GetTrueVSize()
         << ",GV=" << l2_pfes.GlobalVSize();
      signature.push_back(os.str());
   }
   {
      std::ostringstream os;
      os << "H1:V=" << h1_pfes.GetVSize()
         << ",TV=" << h1_pfes.GetTrueVSize()
         << ",GV=" << h1_pfes.GlobalVSize();
      signature.push_back(os.str());
   }
   return signature;
}

std::string CheckSharedVertexCommunication(ParMesh &pmesh)
{
   GroupCommunicator svert_comm(pmesh.gtopo);
   pmesh.GetSharedVertexCommunicator(svert_comm);
   const GroupTopology &gtopo = svert_comm.GetGroupTopology();
   const Table &table = svert_comm.GroupLDofTable();

   int num_shared_vertices = 0;
   for (int g = 1; g < pmesh.GetNGroups(); ++g)
   {
      num_shared_vertices += pmesh.GroupNVertices(g);
   }

   if (num_shared_vertices > 0)
   {
      Array<int> tokens(num_shared_vertices);
      tokens = -1;

      for (int g = 1; g < gtopo.NGroups(); ++g)
      {
         if (!gtopo.IAmMaster(g)) { continue; }
         const int *row = table.GetRow(g);
         for (int j = 0; j < table.RowSize(g); ++j)
         {
            const int sv = row[j];
            tokens[sv] = SignatureToken(
                            "v:" + FormatPoint(pmesh.GetVertex(pmesh.GroupVertex(g, j)),
                                               pmesh.SpaceDimension()));
         }
      }

      svert_comm.Bcast<int>(tokens, 1);

      for (int g = 1; g < gtopo.NGroups(); ++g)
      {
         const int *row = table.GetRow(g);
         for (int j = 0; j < table.RowSize(g); ++j)
         {
            const int sv = row[j];
            const int expected = SignatureToken(
                                    "v:" +
                                    FormatPoint(pmesh.GetVertex(pmesh.GroupVertex(g, j)),
                                                pmesh.SpaceDimension()));
            REQUIRE(tokens[sv] == expected);
         }
      }

      Array<int> counts(num_shared_vertices);
      counts = 1;
      svert_comm.Reduce<int>(counts, GroupCommunicator::Sum);
      svert_comm.Bcast<int>(counts, 1);

      for (int g = 1; g < gtopo.NGroups(); ++g)
      {
         const int expected = gtopo.GetGroupSize(g);
         const int *row = table.GetRow(g);
         for (int j = 0; j < table.RowSize(g); ++j)
         {
            REQUIRE(counts[row[j]] == expected);
         }
      }
   }

   std::ostringstream os;
   os << "shared_vertices=" << num_shared_vertices;
   return os.str();
}

std::string CheckSharedEdgeCommunication(ParMesh &pmesh)
{
   GroupCommunicator sedge_comm(pmesh.gtopo);
   pmesh.GetSharedEdgeCommunicator(sedge_comm);
   const GroupTopology &gtopo = sedge_comm.GetGroupTopology();
   const Table &table = sedge_comm.GroupLDofTable();

   int num_shared_edges = 0;
   for (int g = 1; g < pmesh.GetNGroups(); ++g)
   {
      num_shared_edges += pmesh.GroupNEdges(g);
   }

   if (num_shared_edges > 0)
   {
      Array<int> tokens(num_shared_edges);
      tokens = -1;
      Array<int> edge_vertices;

      for (int g = 1; g < gtopo.NGroups(); ++g)
      {
         if (!gtopo.IAmMaster(g)) { continue; }
         const int *row = table.GetRow(g);
         for (int j = 0; j < table.RowSize(g); ++j)
         {
            const int se = row[j];
            pmesh.GetEdgeVertices(pmesh.GroupEdge(g, j), edge_vertices);
            std::vector<std::string> pts;
            pts.reserve(edge_vertices.Size());
            for (int k = 0; k < edge_vertices.Size(); ++k)
            {
               pts.push_back(
                  FormatPoint(pmesh.GetVertex(edge_vertices[k]), pmesh.SpaceDimension()));
            }
            std::sort(pts.begin(), pts.end());
            tokens[se] = SignatureToken("e:" + Join(pts));
         }
      }

      sedge_comm.Bcast<int>(tokens, 1);

      for (int g = 1; g < gtopo.NGroups(); ++g)
      {
         const int *row = table.GetRow(g);
         for (int j = 0; j < table.RowSize(g); ++j)
         {
            const int se = row[j];
            pmesh.GetEdgeVertices(pmesh.GroupEdge(g, j), edge_vertices);
            std::vector<std::string> pts;
            pts.reserve(edge_vertices.Size());
            for (int k = 0; k < edge_vertices.Size(); ++k)
            {
               pts.push_back(
                  FormatPoint(pmesh.GetVertex(edge_vertices[k]), pmesh.SpaceDimension()));
            }
            std::sort(pts.begin(), pts.end());
            REQUIRE(tokens[se] == SignatureToken("e:" + Join(pts)));
         }
      }

      Array<int> counts(num_shared_edges);
      counts = 1;
      sedge_comm.Reduce<int>(counts, GroupCommunicator::Sum);
      sedge_comm.Bcast<int>(counts, 1);

      for (int g = 1; g < gtopo.NGroups(); ++g)
      {
         const int expected = gtopo.GetGroupSize(g);
         const int *row = table.GetRow(g);
         for (int j = 0; j < table.RowSize(g); ++j)
         {
            REQUIRE(counts[row[j]] == expected);
         }
      }
   }

   std::ostringstream os;
   os << "shared_edges=" << num_shared_edges;
   return os.str();
}

std::string CheckFaceNeighborExchange(ParMesh &pmesh)
{
   pmesh.ExchangeFaceNbrData();

   int checked = 0;
   const IntegrationPoint face_center = Geometries.GetCenter(Geometry::SEGMENT);

   for (int sf = 0; sf < pmesh.GetNSharedFaces(); ++sf)
   {
      auto *tr = pmesh.GetSharedFaceTransformations(sf, true);
      REQUIRE(tr != nullptr);
      REQUIRE(tr->Elem2 != nullptr);
      REQUIRE((tr->GetConfigurationMask() & FaceElementTransformations::HAVE_LOC2) != 0);

      tr->SetAllIntPoints(&face_center);

      Vector x_face(pmesh.SpaceDimension());
      Vector x_elem1(pmesh.SpaceDimension());
      Vector x_elem2(pmesh.SpaceDimension());
      tr->Face->Transform(face_center, x_face);
      tr->Elem1->Transform(tr->GetElement1IntPoint(), x_elem1);
      tr->Elem2->Transform(tr->GetElement2IntPoint(), x_elem2);

      for (int d = 0; d < pmesh.SpaceDimension(); ++d)
      {
         REQUIRE(std::abs(x_face[d] - x_elem1[d]) < 1.0e-10);
         REQUIRE(std::abs(x_face[d] - x_elem2[d]) < 1.0e-10);
      }
      checked++;
   }

   std::ostringstream os;
   os << "shared_faces=" << checked
      << ",face_neighbors=" << pmesh.GetNFaceNeighbors()
      << ",face_nbr_elements=" << pmesh.GetNFaceNeighborElements();
   return os.str();
}

std::string AssembleDiffusion(ParMesh &pmesh)
{
   H1_FECollection fec(1, pmesh.Dimension());
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParBilinearForm a(&pfes);
   ConstantCoefficient one(1.0);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();
   a.Finalize();

   std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());

   std::ostringstream os;
   os << "rows=" << A->GetGlobalNumRows()
      << ",cols=" << A->GetGlobalNumCols();
   return os.str();
}

} // namespace

TEST_CASE("ConduitDataCollectionParMeshRoundTrip",
          "[Parallel][Conduit][Blueprint][ConduitDataCollection]")
{
   CAPTURE(Mpi::WorldSize(), Mpi::WorldRank());

   Mesh serial_mesh = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL,
                                            true, 1.0, 1.0);
   SetBoundaryAttributes(serial_mesh, 1.0, 1.0);

   Array<int> partitioning;
   BuildCartesianPartitioning(serial_mesh, partitioning);

   ParMesh original(MPI_COMM_WORLD, serial_mesh, partitioning.GetData());
   original.SetPrintShared(false);

   conduit::Node n_mesh;
   ConduitDataCollection::MeshToBlueprintMesh(&original, n_mesh);

   std::unique_ptr<Mesh> roundtrip_base(
      ConduitDataCollection::BlueprintMeshToMesh(n_mesh, "main", false,
                                                 MPI_COMM_WORLD));
   ParMesh *roundtrip = dynamic_cast<ParMesh *>(roundtrip_base.get());
   REQUIRE(roundtrip != nullptr);
   roundtrip->SetPrintShared(false);

   const std::vector<std::string> original_mesh_summary =
      CollectMeshSummary(original);
   const std::vector<std::string> roundtrip_mesh_summary =
      CollectMeshSummary(*roundtrip);
   INFO("original mesh summary: " << Join(original_mesh_summary));
   INFO("roundtrip mesh summary: " << Join(roundtrip_mesh_summary));
   REQUIRE(original_mesh_summary == roundtrip_mesh_summary);

   const std::vector<std::string> original_bdr =
      CollectBoundaryAttributeSignature(original);
   const std::vector<std::string> roundtrip_bdr =
      CollectBoundaryAttributeSignature(*roundtrip);
   INFO("original boundary signature: " << Join(original_bdr));
   INFO("roundtrip boundary signature: " << Join(roundtrip_bdr));
   REQUIRE(original_bdr == roundtrip_bdr);

   const std::vector<std::string> original_groups =
      CollectGroupSignature(original);
   const std::vector<std::string> roundtrip_groups =
      CollectGroupSignature(*roundtrip);
   INFO("original group signature: " << Join(original_groups));
   INFO("roundtrip group signature: " << Join(roundtrip_groups));
   REQUIRE(original_groups == roundtrip_groups);

   const std::vector<std::string> original_shared_faces =
      CollectSharedFaceSignature(original);
   const std::vector<std::string> roundtrip_shared_faces =
      CollectSharedFaceSignature(*roundtrip);
   INFO("original shared-face signature: " << Join(original_shared_faces));
   INFO("roundtrip shared-face signature: " << Join(roundtrip_shared_faces));
   REQUIRE(original_shared_faces == roundtrip_shared_faces);

   const std::string original_vertex_comm = CheckSharedVertexCommunication(original);
   const std::string roundtrip_vertex_comm = CheckSharedVertexCommunication(*roundtrip);
   INFO("original shared-vertex summary: " << original_vertex_comm);
   INFO("roundtrip shared-vertex summary: " << roundtrip_vertex_comm);
   REQUIRE(original_vertex_comm == roundtrip_vertex_comm);

   const std::string original_edge_comm = CheckSharedEdgeCommunication(original);
   const std::string roundtrip_edge_comm = CheckSharedEdgeCommunication(*roundtrip);
   INFO("original shared-edge summary: " << original_edge_comm);
   INFO("roundtrip shared-edge summary: " << roundtrip_edge_comm);
   REQUIRE(original_edge_comm == roundtrip_edge_comm);

   const std::string original_face_nbr = CheckFaceNeighborExchange(original);
   const std::string roundtrip_face_nbr = CheckFaceNeighborExchange(*roundtrip);
   INFO("original face-neighbor summary: " << original_face_nbr);
   INFO("roundtrip face-neighbor summary: " << roundtrip_face_nbr);
   REQUIRE(original_face_nbr == roundtrip_face_nbr);

   const std::vector<std::string> original_fespaces =
      CollectFESpaceSignature(original);
   const std::vector<std::string> roundtrip_fespaces =
      CollectFESpaceSignature(*roundtrip);
   INFO("original FE-space signature: " << Join(original_fespaces));
   INFO("roundtrip FE-space signature: " << Join(roundtrip_fespaces));
   REQUIRE(original_fespaces == roundtrip_fespaces);

   const std::string original_assembly = AssembleDiffusion(original);
   const std::string roundtrip_assembly = AssembleDiffusion(*roundtrip);
   INFO("original diffusion assembly summary: " << original_assembly);
   INFO("roundtrip diffusion assembly summary: " << roundtrip_assembly);
   REQUIRE(original_assembly == roundtrip_assembly);
}

#endif // MFEM_USE_MPI && MFEM_USE_CONDUIT
