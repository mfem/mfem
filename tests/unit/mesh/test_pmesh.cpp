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

#include <algorithm>
#include <cmath>
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

TEST_CASE("ParMesh nodes after a conforming refinement are the same in any "
          "partition", "[Parallel], [ParMesh]")
{
   // After a conforming refinement each node takes the largest of the values
   // of the fine elements that contain it, on all the ranks: a curved mesh
   // distributed cyclically and refined near a point, then uniformly, has the
   // same nodes, bit for bit and element for element, as the serial mesh.
   const int nranks = Mpi::WorldSize();
   for (const int dim : {2, 3})
   {
      INFO(dim << "D");
      Mesh serial = (dim == 2) ?
                    Mesh::MakeCartesian2D(4, 4, Element::TRIANGLE) :
                    Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
      serial.Finalize(true);
      serial.SetCurvature(3);
      serial.Transform([](const Vector &x, Vector &y)
      {
         y = x;
         y(0) += 0.03*std::sin(3.0*x(1));
         y(1) += 0.02*std::cos(2.0*x(0));
      });
      Array<int> part(serial.GetNE());
      for (int e = 0; e < serial.GetNE(); e++) { part[e] = e % nranks; }
      ParMesh pmesh(MPI_COMM_WORLD, serial, part.GetData());
      // The nodal values of each element, of all the ranks, sorted.
      auto Nodes = [](Mesh &mesh)
      {
         const GridFunction &nodes = *mesh.GetNodes();
         std::vector<real_t> flat;
         Array<int> vdofs;
         int len = 0;
         for (int e = 0; e < mesh.GetNE(); e++)
         {
            nodes.FESpace()->GetElementVDofs(e, vdofs);
            len = vdofs.Size();
            for (int i : vdofs) { flat.push_back(nodes(i)); }
         }
         if (ParMesh *pm = dynamic_cast<ParMesh *>(&mesh))
         {
            int n = static_cast<int>(flat.size()), size;
            MPI_Comm_size(pm->GetComm(), &size);
            std::vector<int> counts(size), offsets(size + 1, 0);
            MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT,
                          pm->GetComm());
            for (int r = 0; r < size; r++)
            {
               offsets[r+1] = offsets[r] + counts[r];
            }
            std::vector<real_t> all(offsets[size]);
            MPI_Allgatherv(flat.data(), n, MPITypeMap<real_t>::mpi_type,
                           all.data(), counts.data(), offsets.data(),
                           MPITypeMap<real_t>::mpi_type, pm->GetComm());
            flat.swap(all);
            MPI_Allreduce(MPI_IN_PLACE, &len, 1, MPI_INT, MPI_MAX,
                          pm->GetComm());
         }
         std::vector<std::vector<real_t>> rows;
         for (std::size_t i = 0; i + len <= flat.size() && len > 0; i += len)
         {
            rows.emplace_back(flat.begin() + i, flat.begin() + i + len);
         }
         std::sort(rows.begin(), rows.end());
         return rows;
      };
      for (Mesh *m :
           {
              static_cast<Mesh *>(&serial),
              static_cast<Mesh *>(&pmesh)
           })
      {
         Array<int> marked;
         Vector c;
         for (int e = 0; e < m->GetNE(); e++)
         {
            m->GetElementCenter(e, c);
            if (c.Norml2() < 0.6) { marked.Append(e); }
         }
         m->GeneralRefinement(marked, 0);
      }
      CHECK(Nodes(pmesh) == Nodes(serial));
      pmesh.UniformRefinement();
      serial.UniformRefinement();
      CHECK(Nodes(pmesh) == Nodes(serial));
   }
}

TEST_CASE("ParMesh nodes refined with a matrix update operator",
          "[Parallel], [ParMesh]")
{
   // The largest values of the nodes after a conforming refinement come from
   // the default update operator. With a matrix (SetUpdateOperatorType()) they
   // do not: each rank applies the matrix to its coarse nodes.
   Mesh serial = Mesh::MakeCartesian2D(3, 3, Element::TRIANGLE);
   serial.Finalize(true);
   serial.SetCurvature(3);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   FiniteElementSpace &fes = *pmesh.GetNodes()->FESpace();
   fes.SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);
   const Vector coarse(*pmesh.GetNodes());
   pmesh.UniformRefinement();
   const SparseMatrix *T =
      dynamic_cast<const SparseMatrix *>(fes.GetUpdateOperator());
   REQUIRE(T != nullptr);
   Vector y(T->Height());
   T->Mult(coarse, y);
   y -= *pmesh.GetNodes();
   CHECK(y.Normlinf() == 0.0);
}

#endif // MFEM_USE_MPI

} // namespace mfem
