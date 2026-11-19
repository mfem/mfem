// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#include <chrono>
#include <thread>
#include <vector>

namespace mfem
{
#ifdef MFEM_USE_MPI

TEST_CASE("ParMeshGlobalIndices",  "[Parallel], [ParMesh]")
{
   const int ne = 5;

   for (int dimension = 1; dimension < 4; ++dimension)
   {
      for (int amr=0; amr < 1 + (dimension > 1); ++amr)
      {
         Mesh mesh;
         if (dimension == 1)
         {
            mesh = Mesh::MakeCartesian1D(ne, 1.0);
         }
         else if (dimension == 2)
         {
            if (amr)
            {
               const char *mesh_file = "../../data/amr-quad.mesh";
               mesh = Mesh::LoadFromFile(mesh_file, 1, 1);
            }
            else
            {
               mesh = Mesh::MakeCartesian2D(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
            }
         }
         else
         {
            if (amr)
            {
               const char *mesh_file = "../../data/amr-hex.mesh";
               mesh = Mesh::LoadFromFile(mesh_file, 1, 1);
            }
            else
            {
               mesh = Mesh::MakeCartesian3D(ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
            }
         }


         ParMesh pmesh(MPI_COMM_WORLD, mesh);

         int globalN = 0;

         enum EntityType { VERTEX, EDGE, FACE, ELEMENT };
         // Loop over all types of mesh entities
         for (int e=EntityType::VERTEX; e<=EntityType::ELEMENT; ++e)
         {
            if (amr && dimension > 1 && e != EntityType::ELEMENT)
            {
               continue;
            }

            Array<HYPRE_BigInt> gi;

            switch (e)
            {
               case EntityType::VERTEX:
                  globalN = mesh.GetNV();
                  pmesh.GetGlobalVertexIndices(gi);
                  break;
               case EntityType::EDGE:
                  globalN = dimension == 1 ? mesh.GetNV() : mesh.GetNEdges();
                  pmesh.GetGlobalEdgeIndices(gi);
                  break;
               case EntityType::FACE:
                  globalN = mesh.GetNumFaces();
                  pmesh.GetGlobalFaceIndices(gi);
                  break;
               case EntityType::ELEMENT:
                  globalN = mesh.GetNE();
                  pmesh.GetGlobalElementIndices(gi);
                  break;
            }

            // Verify that the local entities do not share a global index.
            {
               std::set<HYPRE_BigInt> localGI;
               for (int i=0; i<gi.Size(); ++i)
               {
                  localGI.insert(gi[i]);
               }

               REQUIRE(localGI.size() == (std::size_t) gi.Size());
            }

            // Verify that the global indices range from 0 to globalN-1.
            {
               HYPRE_BigInt localMin = gi.Size() > 0 ? gi.Min() :
                                       std::numeric_limits<HYPRE_BigInt>::max();
               HYPRE_BigInt localMax = gi.Size() > 0 ? gi.Max() :
                                       std::numeric_limits<HYPRE_BigInt>::min();

               HYPRE_BigInt globalMin, globalMax;
               MPI_Allreduce(&localMin, &globalMin, 1, HYPRE_MPI_BIG_INT, MPI_MIN,
                             MPI_COMM_WORLD);
               MPI_Allreduce(&localMax, &globalMax, 1, HYPRE_MPI_BIG_INT, MPI_MAX,
                             MPI_COMM_WORLD);

               REQUIRE((globalMin == 0 && globalMax == globalN-1));
            }
         }
      }
   }
}

TEST_CASE("ParMeshSharedFaces", "[Parallel], [ParMesh]")
{
   const char *mesh_file = "../../data/fichera-amr.mesh";

   Mesh mesh(mesh_file);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   pmesh.ExchangeFaceNbrData();

   const int nshared = pmesh.GetNSharedFaces();
   int local_ghosts_nonmatching = 0;
   for (int sf = 0; sf < nshared; sf++)
   {
      const int f = pmesh.GetSharedFace(sf);
      FaceElementTransformations *ftr =
         pmesh.GetSharedFaceTransformationsByLocalIndex(f, false);
      if (f != ftr->ElementNo) { local_ghosts_nonmatching++; }
   }

   int global_ghosts_nonmatching = 0;
   MPI_Allreduce(&local_ghosts_nonmatching, &global_ghosts_nonmatching, 1, MPI_INT,
                 MPI_SUM, pmesh.GetComm());

   REQUIRE(global_ghosts_nonmatching == 0);
}

namespace simplicial
{

double exact(const Vector &xvec)
{
   // The exact solution is linear and is harmonic
   return xvec[0] + xvec[1] + xvec[2];
}

void SolveDiffusionProblem(ParMesh &mesh, Vector &x_out)
{
   H1_FECollection fec(1, mesh.Dimension());
   ParFiniteElementSpace fes(&mesh, &fec);

   // Right-hand side is zero since exact solution is harmonic
   ParLinearForm b(&fes);
   b.Assemble();

   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   Array<int> ess_tdof_list, ess_bdr;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // Use the exact solution as boundary conditions
   ParGridFunction x(&fes);
   FunctionCoefficient exact_coeff(exact);
   x.ProjectBdrCoefficient(exact_coeff, ess_bdr);

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   // X = 0.0;
   cg.Mult(B, X);
   x_out = X;
}

}

// Test case: the numbering of a parallel non-conforming mesh must not depend
//            on the order in which ParNCMesh receives the messages of
//            Refine() and of Rebalance() with a custom partition. The mesh is
//            a row of slabs owned by the ranks a, x, b, x, r (then the other
//            ranks), so that a, b and r are not neighbors of one another.
//            Delaying a, or b, before an operation makes its messages arrive
//            after those of the other one, at x for the refinement and at r
//            for the rebalance, where a and b move their elements to r. The
//            two runs must give the same mesh. (Rebalance() with the default
//            partition and derefinement communicate collectively before
//            sending, which absorbs the delay, so they are not tested here.)
TEST_CASE("ParNCMesh numbering is independent of message order",
          "[Parallel], [ParMesh], [NCMesh]")
{
   const int rank = Mpi::WorldRank();
   const int nranks = Mpi::WorldSize();
   if (nranks < 4) { return; }
   constexpr int x = 0, a = 1, b = 2, r = 3;

   const auto type = GENERATE(Element::HEXAHEDRON, Element::TETRAHEDRON);
   CAPTURE(type);

   // The vertices of the elements and of the boundary elements, the shared
   // entities of the groups, and the edge and face indices in the lists of
   // the ParNCMesh, which include the ghost edges and faces.
   auto Numbering = [](ParMesh &pmesh)
   {
      std::vector<int> num;
      Array<int> v;
      for (int i = 0; i < pmesh.GetNE(); i++)
      {
         pmesh.GetElementVertices(i, v);
         num.insert(num.end(), v.begin(), v.end());
      }
      for (int i = 0; i < pmesh.GetNBE(); i++)
      {
         pmesh.GetBdrElementVertices(i, v);
         num.insert(num.end(), v.begin(), v.end());
         num.push_back(pmesh.GetBdrAttribute(i));
      }
      for (int g = 1; g < pmesh.GetNGroups(); g++)
      {
         const int *group = pmesh.gtopo.GetGroup(g);
         num.insert(num.end(), group, group + pmesh.gtopo.GetGroupSize(g));
         int ent, ori;
         for (int i = 0; i < pmesh.GroupNVertices(g); i++)
         {
            num.push_back(pmesh.GroupVertex(g, i));
         }
         for (int i = 0; i < pmesh.GroupNEdges(g); i++)
         {
            pmesh.GroupEdge(g, i, ent, ori);
            num.insert(num.end(), {ent, ori});
         }
         for (int i = 0; i < pmesh.GroupNTriangles(g); i++)
         {
            pmesh.GroupTriangle(g, i, ent, ori);
            num.insert(num.end(), {ent, ori});
         }
         for (int i = 0; i < pmesh.GroupNQuadrilaterals(g); i++)
         {
            pmesh.GroupQuadrilateral(g, i, ent, ori);
            num.insert(num.end(), {ent, ori});
         }
      }
      auto AddIndices = [&num](const NCMesh::NCList &list)
      {
         for (const auto &id : list.conforming) { num.push_back(id.index); }
         for (const auto &id : list.masters) { num.push_back(id.index); }
         for (const auto &id : list.slaves) { num.push_back(id.index); }
      };
      if (pmesh.GetNE())
      {
         AddIndices(pmesh.pncmesh->GetEdgeList());
         AddIndices(pmesh.pncmesh->GetFaceList());
      }
      return num;
   };

   auto Vertices = [](const ParMesh &pmesh)
   {
      std::vector<real_t> x;
      for (int i = 0; i < pmesh.GetNV(); i++)
      {
         const real_t *v = pmesh.GetVertex(i);
         x.insert(x.end(), v, v + pmesh.SpaceDimension());
      }
      return x;
   };

   struct Result
   {
      std::vector<int> refined, rebalanced;
      std::vector<real_t> vertices;
   };

   // Refine and rebalance a mesh, delaying 'delayed_rank' at the start of
   // both operations.
   auto Run = [&](int delayed_rank)
   {
      auto Delay = [&]()
      {
         MPI_Barrier(MPI_COMM_WORLD);
         if (rank == delayed_rank)
         {
            // long compared to the time the other ranks need to send
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
         }
      };

      const int nslabs = nranks + 1;
      const int slab_rank[] = {a, x, b, x, r};
      Mesh mesh = Mesh::MakeCartesian3D(nslabs, 1, 1, type, nslabs, 1.0, 1.0,
                                        false);
      mesh.EnsureNCMesh(true);
      Array<int> partition(mesh.GetNE());
      Vector center;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         mesh.GetElementCenter(i, center);
         const int slab = static_cast<int>(center(0));
         partition[i] = (slab < 5) ? slab_rank[slab] : slab - 1;
      }
      ParMesh pmesh(MPI_COMM_WORLD, mesh, partition.GetData());

      Result res;

      // Refine() sends before any other communication: at x, the ghost
      // refinements from the delayed rank arrive last.
      Delay();
      pmesh.UniformRefinement();
      res.refined = Numbering(pmesh);

      // Move the elements of a and b to r. Before sending, the ranks only wait
      // for their neighbors: the delayed rank sends late, the other one of a
      // and b does not.
      Array<int> new_partition(pmesh.GetNE());
      new_partition = (rank == a || rank == b) ? r : rank;
      Delay();
      pmesh.Rebalance(new_partition);
      res.rebalanced = Numbering(pmesh);
      res.vertices = Vertices(pmesh);

      return res;
   };

   const Result run_a = Run(a);
   const Result run_b = Run(b);

   // reduce over all ranks, so that they pass or fail together
   auto AllRanks = [](bool local)
   {
      int global = local;
      MPI_Allreduce(MPI_IN_PLACE, &global, 1, MPI_INT, MPI_LAND,
                    MPI_COMM_WORLD);
      return global != 0;
   };
   // ghost refinements in ParNCMesh::Refine()
   const bool same_refined = AllRanks(run_a.refined == run_b.refined);
   // decoding of the elements in ParNCMesh::RedistributeElements()
   const bool same_rebalanced = AllRanks(run_a.rebalanced == run_b.rebalanced &&
                                         run_a.vertices == run_b.vertices);
   CHECK(same_refined);
   CHECK(same_rebalanced);
}

TEST_CASE("ParMeshMakeSimplicial", "[Parallel], [ParMesh]")
{
   // Test that the parallel mesh obtained by ParMesh::MakeSimplicial is valid.
   // This test solves a Poisson problem on a 3x3x3 hex mesh, and on the tet
   // mesh obtained by splitting the hexes into tets. The finite element space
   // is linear in both cases, and the exact solution is also linear, so it will
   // be recovered exactly in both cases. The vertices of both meshes are the
   // same, so we check that the resulting discrete solutions are identical up
   // to solver tolerance.

   Mesh mesh = Mesh::MakeCartesian3D(3, 3, 3, Element::HEXAHEDRON);
   if (GENERATE(false,true))
   {
      mesh.SetCurvature(2);
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   ParMesh pmesh_tet = ParMesh::MakeSimplicial(pmesh);

   Vector x, x_tet;
   simplicial::SolveDiffusionProblem(pmesh, x);
   simplicial::SolveDiffusionProblem(pmesh_tet, x_tet);

   x -= x_tet;
   REQUIRE(x.Normlinf() == MFEM_Approx(0.0));
}

TEST_CASE("ParMeshMakeSimplicial interior boundary", "[Parallel], [ParMesh]")
{
   if (Mpi::WorldSize() < 2) { return; }

   const int order = GENERATE(1, 2);
   const int rank = Mpi::WorldRank();

   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL,
                                     true, 2.0, 1.0, false);
   // Add an interior boundary that points downwards, opposite to the
   // orientation of the left quad's edge. This is valid for an interior
   // boundary, even when its adjacent elements belong to different ranks.
   mesh.AddBdrSegment(4, 1, 5);
   mesh.FinalizeTopology();

   if (order > 1) { mesh.SetCurvature(order); }

   // Extra ranks have empty partitions.
   int partitioning[] = {0, 1};
   ParMesh pmesh(MPI_COMM_WORLD, mesh, partitioning);
   REQUIRE(pmesh.GetNSharedFaces() == (rank < 2 ? 1 : 0));

   ParMesh simplex_mesh = ParMesh::MakeSimplicial(pmesh);
   REQUIRE(simplex_mesh.GetNE() == (rank < 2 ? 2 : 0));
   REQUIRE(simplex_mesh.GetNBE() == pmesh.GetNBE());
   REQUIRE(simplex_mesh.GetNSharedFaces() == (rank < 2 ? 1 : 0));
   REQUIRE(simplex_mesh.CheckElementOrientation(false) == 0);
   for (int i = 0; i < simplex_mesh.GetNE(); i++)
   {
      REQUIRE(simplex_mesh.GetElementGeometry(i) == Geometry::TRIANGLE);
   }
   for (int i = 0; i < simplex_mesh.GetNBE(); i++)
   {
      REQUIRE(simplex_mesh.GetBdrAttribute(i) == pmesh.GetBdrAttribute(i));
   }
}

#endif // MFEM_USE_MPI

} // namespace mfem
