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
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
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

namespace tet_marking
{

/// Prisms over equilateral triangles (a sheared Cartesian mesh of n^3 wedges),
/// split into tetrahedra and marked: their horizontal faces have three edges
/// of the same length, so their marking depends on the order of the edges.
Mesh EquilateralPrisms(int n)
{
   Mesh wedges = Mesh::MakeCartesian3D(n, n, n, Element::WEDGE);
   for (int v = 0; v < wedges.GetNV(); v++)
   {
      real_t *x = wedges.GetVertex(v);
      x[0] -= 0.5*x[1];
      x[1] *= std::sqrt(0.75);
   }
   Mesh mesh = Mesh::MakeSimplicial(wedges);
   mesh.Finalize(true);
   return mesh;
}

/// Check that the members of each group list the same shared edges and
/// triangles, in the same order and with the same vertex order (for the
/// triangles, their marked edge first): the master broadcasts the global
/// vertex numbers of its entities, the others compare. The sizes of the lists
/// are compared first.
bool SharedEntitiesAgree(ParMesh &pmesh)
{
   H1_FECollection h1(1, pmesh.Dimension());
   ParFiniteElementSpace fes(&pmesh, &h1);
   fes.Dof_TrueDof_Matrix();
   const int ng = pmesh.GetNGroups();
   auto Compare = [&](const std::function<void(int, std::vector<int> &)> &fill)
   {
      GroupCommunicator gc(pmesh.gtopo);
      Table &group_ldof = gc.GroupLDofTable();
      std::vector<std::vector<int>> per_group(ng);
      group_ldof.MakeI(ng);
      for (int g = 1; g < ng; g++)
      {
         fill(g, per_group[g]);
         group_ldof.AddColumnsInRow(g, static_cast<int>(per_group[g].size()));
      }
      group_ldof.MakeJ();
      Array<int> values;
      for (int g = 1; g < ng; g++)
      {
         for (int x : per_group[g])
         {
            group_ldof.AddConnection(g, values.Size());
            values.Append(x);
         }
      }
      group_ldof.ShiftUpI();
      gc.Finalize();
      Array<int> master(values);
      gc.Bcast(master);
      int ok = 1;
      for (int k = 0; k < values.Size(); k++)
      {
         ok &= (values[k] == master[k]);
      }
      MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_LAND, pmesh.GetComm());
      return bool(ok);
   };
   const bool same_sizes = Compare([&](int g, std::vector<int> &x)
   {
      x = {pmesh.GroupNVertices(g), pmesh.GroupNEdges(g),
           pmesh.GroupNTriangles(g)
          };
   });
   if (!same_sizes) { return false; }
   auto Id = [&](int v)
   {
      return static_cast<int>(fes.GetGlobalTDofNumber(v));
   };
   return Compare([&](int g, std::vector<int> &x)
   {
      Array<int> v;
      for (int i = 0; i < pmesh.GroupNVertices(g); i++)
      {
         x.push_back(Id(pmesh.GroupVertex(g, i)));
      }
      for (int i = 0; i < pmesh.GroupNEdges(g); i++)
      {
         int edge, o;
         pmesh.GroupEdge(g, i, edge, o);
         pmesh.GetEdgeVertices(edge, v);
         if (o < 0) { std::swap(v[0], v[1]); }
         for (int vi : v) { x.push_back(Id(vi)); }
      }
      for (int i = 0; i < pmesh.GroupNTriangles(g); i++)
      {
         int face, o;
         pmesh.GroupTriangle(g, i, face, o);
         pmesh.GetFaceVertices(face, v);
         int sv[3];
         const int *perm = Geometry::Constants<Geometry::TRIANGLE>::Orient[o];
         for (int j = 0; j < 3; j++) { sv[perm[j]] = v[j]; }
         for (int j = 0; j < 3; j++) { x.push_back(Id(sv[j])); }
      }
   });
}

/// Refine conformally the elements whose center has x < 0.5.
void RefineLeftHalf(ParMesh &pmesh)
{
   Array<int> marked;
   Vector c;
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      pmesh.GetElementCenter(e, c);
      if (c(0) < 0.5) { marked.Append(e); }
   }
   pmesh.GeneralRefinement(marked, 0);
}

/// A mesh of tetrahedra, marked: prisms over equilateral triangles, or a
/// Cartesian mesh of 4^3 cubes split into tetrahedra; with its vertices moved
/// by up to a fraction of the edge length (its edges of equal length then
/// become nearly equal), and nodes of the given order, straight or deformed.
Mesh TetMesh(bool prisms, real_t perturbation, int order, bool deform)
{
   Mesh mesh = prisms ? EquilateralPrisms(4) :
               Mesh::MakeCartesian3D(4, 4, 4, Element::TETRAHEDRON);
   for (int v = 0; v < mesh.GetNV(); v++)
   {
      real_t *x = mesh.GetVertex(v);
      for (int d = 0; d < 3; d++)
      {
         x[d] += 0.25*perturbation*std::sin(12.9898*v + 78.233*d);
      }
   }
   mesh.Finalize(true);
   if (order > 0) { mesh.SetCurvature(order); }
   if (deform)
   {
      mesh.Transform([](const Vector &x, Vector &y)
      {
         y = x;
         y(0) += 0.03*std::sin(3.0*x(1)) + 0.02*x(0)*x(2);
         y(1) += 0.02*std::cos(2.0*x(0));
         y(2) += 0.01*std::sin(2.0*x(0) + x(1));
      });
   }
   return mesh;
}

/// Check, on each rank, that the two marked tetrahedra of each face mark the
/// same edge on it, and that the refinement edge of each boundary triangle is
/// the marked edge of its face in the adjacent tetrahedron, if marked.
bool LocalFacesAgree(ParMesh &pmesh)
{
   auto Marked = [&pmesh](int e) -> const Tetrahedron *
   {
      auto tet = static_cast<const Tetrahedron *>(pmesh.GetElement(e));
      return tet->GetRefinementFlag() ? tet : nullptr;
   };
   auto SameEdge = [](const int *a, const int *b)
   {
      return (a[0] == b[0] && a[1] == b[1]) || (a[0] == b[1] && a[1] == b[0]);
   };
   int ok = 1;
   for (int f = 0; f < pmesh.GetNumFaces(); f++)
   {
      int e[2], inf[2], mv[2][3], n = 0;
      pmesh.GetFaceElements(f, &e[0], &e[1]);
      pmesh.GetFaceInfos(f, &inf[0], &inf[1]);
      for (int k = 0; k < 2; k++)
      {
         const Tetrahedron *tet = (e[k] >= 0) ? Marked(e[k]) : nullptr;
         if (tet) { tet->GetMarkedFace(inf[k]/64, mv[n++]); }
      }
      if (n == 2) { ok &= SameEdge(mv[0], mv[1]); }
   }
   for (int b = 0; b < pmesh.GetNBE(); b++)
   {
      int e, inf, mv[3];
      pmesh.GetBdrElementAdjacentElement(b, e, inf);
      const Tetrahedron *tet = Marked(e);
      if (!tet) { continue; }
      tet->GetMarkedFace(inf/64, mv);
      ok &= SameEdge(pmesh.GetBdrElement(b)->GetVertices(), mv);
   }
   MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_LAND, pmesh.GetComm());
   return bool(ok);
}

/// The largest change of a smooth field interpolated in the space when the
/// values of its shared dofs are replaced by those of their owners: round-off
/// if the space is conforming.
real_t ConformityError(ParMesh &pmesh, const FiniteElementCollection &fec)
{
   VectorFunctionCoefficient u(3, [](const Vector &x, Vector &y)
   {
      y(0) = 1.0 + x(1)*x(2);
      y(1) = 2.0 - x(0)*x(2);
      y(2) = 3.0 + x(0)*x(1);
   });
   ParFiniteElementSpace fes(&pmesh, &fec);
   ParGridFunction x(&fes), y(&fes);
   x.ProjectCoefficient(u);
   Vector t(fes.GetTrueVSize());
   x.GetTrueDofs(t);
   y.SetFromTrueDofs(t);
   y -= x;
   real_t diff = y.Normlinf();
   MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 pmesh.GetComm());
   return diff;
}

/// Check that ND and RT spaces of orders 1 and 2 are conforming.
bool NDAndRTConforming(ParMesh &pmesh)
{
   real_t worst = 0.0;
   for (int p = 1; p <= 2; p++)
   {
      ND_FECollection nd(p, 3);
      RT_FECollection rt(p - 1, 3);
      worst = std::max({worst, ConformityError(pmesh, nd),
                        ConformityError(pmesh, rt)});
   }
   return worst < 1e-10;
}

/// Refine conformally the elements whose center is closer than r to p.
void RefineNear(Mesh &mesh, const Vector &p, real_t r)
{
   Array<int> marked;
   Vector c;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementCenter(e, c);
      if (c.DistanceTo(p) < r) { marked.Append(e); }
   }
   mesh.GeneralRefinement(marked, 0);
}

/// The elements of @a mesh, each as the positions of its vertices in their
/// order (its marked edge first) and, for a tetrahedron, its refinement flag;
/// of all the ranks of a ParMesh; sorted: the same for the same mesh, whatever
/// its partition and its numbering. The positions are those the marking uses:
/// the vertex coordinates, the vertex DOFs of the nodes, or, for nodes without
/// vertex DOFs, the images of the reference vertices by the element.
std::vector<std::vector<real_t>> Description(Mesh &mesh)
{
   const int sdim = mesh.SpaceDimension(), dim = mesh.Dimension();
   const int len = (dim + 1)*sdim + (dim == 3 ? 1 : 0);
   const GridFunction *nodes = mesh.GetNodes();
   const FiniteElementSpace *fes = nodes ? nodes->FESpace() : nullptr;
   std::vector<real_t> flat;
   Array<int> vdofs;
   DenseMatrix x;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      const Element *el = mesh.GetElement(e);
      const int *v = el->GetVertices();
      if (fes && fes->GetNVDofs() == 0)
      {
         mesh.GetElementTransformation(e)->Transform(
            *Geometries.GetVertices(el->GetGeometryType()), x);
      }
      for (int k = 0; k < dim + 1; k++)
      {
         for (int d = 0; d < sdim; d++)
         {
            if (!fes) { flat.push_back(mesh.GetVertex(v[k])[d]); }
            else if (fes->GetNVDofs() == 0) { flat.push_back(x(d, k)); }
            else
            {
               fes->GetVertexVDofs(v[k], vdofs);
               flat.push_back((*nodes)(vdofs[d]));
            }
         }
      }
      if (dim == 3)
      {
         const Tetrahedron *t = static_cast<const Tetrahedron *>(el);
         flat.push_back(t->GetRefinementFlag());
      }
   }
   ParMesh *pmesh = dynamic_cast<ParMesh *>(&mesh);
   if (pmesh)
   {
      MPI_Comm comm = pmesh->GetComm();
      int n = static_cast<int>(flat.size()), nranks;
      MPI_Comm_size(comm, &nranks);
      std::vector<int> counts(nranks), offsets(nranks + 1, 0);
      MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
      for (int r = 0; r < nranks; r++)
      {
         offsets[r+1] = offsets[r] + counts[r];
      }
      std::vector<real_t> all(offsets[nranks]);
      const MPI_Datatype type = MPITypeMap<real_t>::mpi_type;
      MPI_Allgatherv(flat.data(), n, type, all.data(), counts.data(),
                     offsets.data(), type, comm);
      flat.swap(all);
   }
   std::vector<std::vector<real_t>> d;
   for (std::size_t i = 0; i < flat.size(); i += len)
   {
      d.emplace_back(flat.begin() + i, flat.begin() + i + len);
   }
   std::sort(d.begin(), d.end());
   return d;
}

/// The number of elements, on all the ranks, whose marked edge, between their
/// local vertices 0 and 1, is shorter than one of their other edges by more
/// than round-off. The lengths are those of the chords between the vertices
/// as mapped by each element.
int MarkedNotLongest(ParMesh &pmesh)
{
   int n = 0;
   IsoparametricTransformation T;
   DenseMatrix x;
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      const Element *el = pmesh.GetElement(e);
      pmesh.GetElementTransformation(e, &T);
      T.Transform(*Geometries.GetVertices(el->GetGeometryType()), x);
      real_t longest = 0.0, marked = 0.0;
      for (int k = 0; k < el->GetNEdges(); k++)
      {
         const int *ev = el->GetEdgeVertices(k);
         real_t l = 0.0;
         for (int d = 0; d < x.Height(); d++)
         {
            l += (x(d, ev[0]) - x(d, ev[1]))*(x(d, ev[0]) - x(d, ev[1]));
         }
         longest = std::max(longest, l);
         if (ev[0] + ev[1] == 1) { marked = l; }
      }
      n += (marked < longest*(1.0 - 1e-12));
   }
   MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_SUM, pmesh.GetComm());
   return n;
}

/// Waits until all the ranks of @a comm call this, or aborts the run if they
/// do not within @a seconds: a rank kept in a collective call that the others
/// did not make would otherwise hang the test. @a comm must not be used by the
/// code before the call (e.g. a duplicate of the communicator of the mesh).
void AbortUnlessAllArrive(MPI_Comm comm, double seconds)
{
   MPI_Request request;
   MPI_Ibarrier(comm, &request);
   const double start = MPI_Wtime();
   int arrived = 0;
   while (!arrived)
   {
      MPI_Test(&request, &arrived, MPI_STATUS_IGNORE);
      if (!arrived && MPI_Wtime() - start > seconds)
      {
         mfem::err << "Not all the ranks arrived within " << seconds
                   << " s: aborting." << std::endl;
         MPI_Abort(comm, 1);
      }
   }
}

/// The largest distance, on all the ranks, between the vertices of the
/// elements and the images of their reference vertices by the nodes.
real_t NodesAtVerticesGap(ParMesh &pmesh)
{
   real_t gap = 0.0;
   IsoparametricTransformation T;
   DenseMatrix x;
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      const Element *el = pmesh.GetElement(e);
      pmesh.GetElementTransformation(e, &T);
      T.Transform(*Geometries.GetVertices(el->GetGeometryType()), x);
      const int *v = el->GetVertices();
      for (int k = 0; k < el->GetNVertices(); k++)
      {
         const real_t *p = pmesh.GetVertex(v[k]);
         real_t d = 0.0;
         for (int i = 0; i < x.Height(); i++)
         {
            d += (x(i, k) - p[i])*(x(i, k) - p[i]);
         }
         gap = std::max(gap, std::sqrt(d));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &gap, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 pmesh.GetComm());
   return gap;
}

/// The largest distance between the points of the shared faces as mapped
/// from their two sides, through the face neighbor data; infinity if these
/// data do not describe the shared faces.
real_t SharedFaceGap(ParMesh &pmesh)
{
   real_t gap = 0.0;
   try
   {
      pmesh.ExchangeFaceNbrData();
      Vector x1, x2;
      for (int f = 0; f < pmesh.GetNSharedFaces(); f++)
      {
         FaceElementTransformations *T = pmesh.GetSharedFaceTransformations(f);
         const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(), 3);
         for (int j = 0; j < ir.GetNPoints(); j++)
         {
            T->SetAllIntPoints(&ir.IntPoint(j));
            T->Elem1->Transform(T->GetElement1IntPoint(), x1);
            T->Elem2->Transform(T->GetElement2IntPoint(), x2);
            gap = std::max(gap, x1.DistanceTo(x2));
         }
      }
   }
   catch (std::exception &)
   {
      gap = std::numeric_limits<real_t>::infinity();
   }
   MPI_Allreduce(MPI_IN_PLACE, &gap, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 pmesh.GetComm());
   return gap;
}

/// The largest change of the nodes, and of a smooth field interpolated in the
/// space of the nodes, when the values of their shared dofs are replaced by
/// those of their owners: round-off if the space of the nodes is conforming.
real_t NodalSpaceError(ParMesh &pmesh)
{
   ParGridFunction &nodes = *static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace &fes = *nodes.ParFESpace();
   VectorFunctionCoefficient u(3, [](const Vector &x, Vector &y)
   {
      y(0) = 1.0 + x(1)*x(2);
      y(1) = 2.0 - x(0)*x(2);
      y(2) = 3.0 + x(0)*x(1);
   });
   ParGridFunction field(&fes);
   field.ProjectCoefficient(u);
   real_t diff = 0.0;
   for (ParGridFunction *x : {&nodes, &field})
   {
      Vector t(fes.GetTrueVSize());
      x->GetTrueDofs(t);
      ParGridFunction y(&fes);
      y.SetFromTrueDofs(t);
      y -= *x;
      diff = std::max(diff, y.Normlinf());
   }
   MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 pmesh.GetComm());
   return diff;
}

} // namespace tet_marking

TEST_CASE("ParMesh marks tetrahedra alike on all ranks",
          "[Parallel], [ParMesh]")
{
   // The marking of the shared triangles of these prisms depends on the global
   // order of their edges of equal length, which all ranks must agree on.
   // Marking after a uniform refinement (which leaves the tetrahedra
   // unmarked), then local refinement.
   Mesh serial = tet_marking::EquilateralPrisms(4);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   REQUIRE(tet_marking::SharedEntitiesAgree(pmesh));
   pmesh.UniformRefinement();
   pmesh.Finalize(true);
   REQUIRE(tet_marking::SharedEntitiesAgree(pmesh));
   tet_marking::RefineLeftHalf(pmesh);
   CHECK(tet_marking::SharedEntitiesAgree(pmesh));
}

TEST_CASE("ParMesh marks shared triangles from the lengths of their master",
          "[Parallel], [ParMesh]")
{
   // Each rank stretches its copy of the mesh along x by its own factor, so
   // that the ranks compute different lengths for the same shared edges, as
   // round-off does after the refinement of a curved mesh, on a larger scale
   // here. The ranks must still mark their shared triangles alike.
   Mesh serial = tet_marking::EquilateralPrisms(4);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   const real_t s = 1.0 + 1e-3*Mpi::WorldRank();
   for (int v = 0; v < pmesh.GetNV(); v++) { pmesh.GetVertex(v)[0] *= s; }
   pmesh.Finalize(true);
   REQUIRE(tet_marking::SharedEntitiesAgree(pmesh));
   tet_marking::RefineLeftHalf(pmesh);
   CHECK(tet_marking::SharedEntitiesAgree(pmesh));
}

TEST_CASE("ParMesh marked again updates its face neighbors",
          "[Parallel], [ParMesh]")
{
   // A Nedelec space of order 2 has the mesh exchange the data of its face
   // neighbors, which include the orientations of the shared faces in the
   // neighbor elements. Marking the tetrahedra again changes these
   // orientations: the spaces built after it must not use the old ones.
   Mesh serial = tet_marking::EquilateralPrisms(4);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   pmesh.UniformRefinement();
   {
      ND_FECollection nd(2, 3);
      ParFiniteElementSpace fes(&pmesh, &nd);
      fes.Dof_TrueDof_Matrix();
   }
   pmesh.Finalize(true);
   CHECK(tet_marking::NDAndRTConforming(pmesh));
}

TEST_CASE("ParMesh finalized without a new marking keeps its face neighbors",
          "[Parallel], [ParMesh]")
{
   // A DG form uses the face neighbor data of its space. Finalize() without a
   // new marking or orientation fix keeps the vertex orders, and so must keep
   // these data: the form assembled again gives the same matrix.
   Mesh serial = Mesh::MakeCartesian3D(3, 3, 3, Element::TETRAHEDRON);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   L2_FECollection fec(1, 3);
   ParFiniteElementSpace fes(&pmesh, &fec);
   auto Assemble = [&fes]()
   {
      ParBilinearForm a(&fes);
      ConstantCoefficient one(1.0);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, -1.0, 2.0));
      a.Assemble();
      a.Finalize();
      std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());
      ParGridFunction u(&fes);
      FunctionCoefficient f([](const Vector &p)
      {
         return p(0) + 2.0*p(1)*p(2);
      });
      u.ProjectCoefficient(f);
      Vector U(A->Width()), V(A->Height());
      u.GetTrueDofs(U);
      A->Mult(U, V);
      return InnerProduct(fes.GetComm(), V, V);
   };
   const real_t before = Assemble();
   pmesh.Finalize();
   CHECK(Assemble() == before);
}

TEST_CASE("ParMesh marks tetrahedra consistently through refinements",
          "[Parallel], [ParMesh]")
{
   // Edges of equal length, of nearly equal length, and curved meshes, whose
   // ranks hold node values that differ by round-off after a refinement.
   // After a uniform refinement, a new marking, and rounds of local refinement
   // near a boundary, the ranks agree on their shared entities, the faces and
   // the boundary triangles agree with the marking of their tetrahedra, and
   // ND and RT spaces are conforming.
   struct Case
   {
      const char *name;
      bool prisms;
      real_t eps;
      int order;
      bool deform;
   };
   const Case cases[] =
   {
      {"prisms", true, 0.0, 0, false},
      {"prisms, perturbed", true, 3e-7, 0, false},
      {"prisms, nodes of order 3", true, 0.0, 3, false},
      {"cubes, nodes of order 2", false, 0.0, 2, false},
      {"cubes, curved, nodes of order 2", false, 0.0, 2, true},
      {"cubes, curved, nodes of order 3", false, 0.0, 3, true}
   };
   const Case &c = cases[GENERATE(0, 1, 2, 3, 4, 5)];
   INFO(c.name);
   Mesh serial = tet_marking::TetMesh(c.prisms, c.eps, c.order, c.deform);
   Vector pmin, pmax, p(3);
   serial.GetBoundingBox(pmin, pmax);
   p(0) = 0.5*(pmin(0) + pmax(0));
   p(1) = 0.5*(pmin(1) + pmax(1));
   p(2) = pmin(2);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   serial.Clear();
   auto Check = [&pmesh](bool marked)
   {
      REQUIRE(tet_marking::SharedEntitiesAgree(pmesh));
      if (marked) { REQUIRE(tet_marking::LocalFacesAgree(pmesh)); }
      REQUIRE(tet_marking::NDAndRTConforming(pmesh));
   };
   Check(true);
   pmesh.UniformRefinement();
   Check(false);
   pmesh.Finalize(true);
   Check(true);
   for (int r = 0; r < 3; r++)
   {
      tet_marking::RefineNear(pmesh, p, 0.3*std::pow(0.5, r));
      Check(true);
   }
}

TEST_CASE("ParMesh marks shared triangles alike when their vertices differ by "
          "round-off", "[Parallel], [ParMesh]")
{
   // Two tetrahedra share an equilateral face, so its three edges have the
   // same squared length up to round-off. The second rank moves the apex of
   // the face by 1 to 6 units in the last place, as round-off does after the
   // refinement of a curved mesh: the two ranks must still mark the face
   // alike, which they do with the lengths that the group master computes.
   if (Mpi::WorldSize() < 2) { return; }
   int differ = 0;
   for (int ulps = 1; ulps <= 6; ulps++)
   {
      Mesh serial(3, 5, 2);
      const real_t p[5][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
         {0.5, std::sqrt(real_t(0.75)), 0.0}, {0.5, 0.3, 1.0}, {0.5, 0.3, -1.0}
      };
      for (int v = 0; v < 5; v++) { serial.AddVertex(p[v]); }
      serial.AddTet(0, 1, 2, 3);
      serial.AddTet(1, 0, 2, 4);
      serial.FinalizeTopology();
      serial.Finalize(true);
      int partitioning[2] = {0, 1};
      ParMesh pmesh(MPI_COMM_WORLD, serial, partitioning);
      if (Mpi::WorldRank() == 1)
      {
         for (int v = 0; v < pmesh.GetNV(); v++)
         {
            real_t *x = pmesh.GetVertex(v);
            if (x[2] == 0.0 && x[1] > 0.0)
            {
               for (int k = 0; k < ulps; k++)
               {
                  x[1] = std::nextafter(x[1], real_t(1));
               }
            }
         }
      }
      pmesh.Finalize(true);
      differ += !tet_marking::SharedEntitiesAgree(pmesh);
   }
   INFO(differ << " of 6 perturbations marked differently");
   CHECK(differ == 0);
}

TEST_CASE("ParMesh marked again keeps ND_3 and RT_3 spaces conforming",
          "[Parallel], [ParMesh]")
{
   // Distributed, refined uniformly and marked again, without any local
   // refinement: Nedelec and Raviart-Thomas spaces built then on
   // ND_FECollection(3) and RT_FECollection(3) (whose GetOrder() is 4) must
   // be conforming. The order of the shared edges must be the same on all
   // ranks, not only within each group: a shared triangle can have edges of
   // different groups, and its ranks would then mark it differently.
   Mesh serial = tet_marking::EquilateralPrisms(4);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   pmesh.UniformRefinement();
   pmesh.Finalize(true);
   CHECK(tet_marking::SharedEntitiesAgree(pmesh));
   ND_FECollection nd(3, 3);
   RT_FECollection rt(3, 3);
   CHECK(tet_marking::ConformityError(pmesh, nd) < 1e-10);
   CHECK(tet_marking::ConformityError(pmesh, rt) < 1e-10);
}

TEST_CASE("ParMesh marks meshes with discontinuous nodes by their lengths",
          "[Parallel], [ParMesh]")
{
   // The nodes of a periodic mesh are discontinuous: a vertex on a periodic
   // boundary has a position in each of its elements. With cells long in one
   // direction, each element marked again on the ranks is marked by its
   // longest edge, and the ranks agree on their shared entities. A straight
   // mesh with discontinuous nodes is marked as the same mesh without nodes.
   const int dim = GENERATE(2, 3);
   INFO(dim << "D");
   {
      Mesh base = (dim == 2) ?
                  Mesh::MakeCartesian2D(4, 3, Element::TRIANGLE, false, 1.0,
                                        9.0) :
                  Mesh::MakeCartesian3D(3, 3, 3, Element::TETRAHEDRON, 1.0,
                                        4.0, 13.0);
      std::vector<Vector> translations;
      for (int d = 0; d < dim; d++)
      {
         Vector t(dim);
         t = 0.0;
         t(d) = (d == 0) ? 1.0 : (d == 1) ? ((dim == 2) ? 9.0 : 4.0) : 13.0;
         translations.push_back(t);
      }
      Mesh serial = Mesh::MakePeriodic(
                       base, base.CreatePeriodicVertexMapping(translations));
      ParMesh pmesh(MPI_COMM_WORLD, serial);
      pmesh.Finalize(true);
      CHECK(tet_marking::MarkedNotLongest(pmesh) == 0);
      CHECK(tet_marking::SharedEntitiesAgree(pmesh));
      if (dim == 3) { CHECK(tet_marking::LocalFacesAgree(pmesh)); }
   }
   auto Make = [dim]()
   {
      Mesh mesh = (dim == 2) ?
                  Mesh::MakeCartesian2D(6, 6, Element::TRIANGLE) :
                  Mesh::MakeCartesian3D(4, 4, 4, Element::TETRAHEDRON);
      for (int v = 0; v < mesh.GetNV(); v++)
      {
         real_t *x = mesh.GetVertex(v);
         for (int d = 0; d < dim; d++)
         {
            x[d] += 0.015*std::sin(12.9898*v + 78.233*d);
         }
      }
      return mesh;
   };
   Mesh plain = Make(), nodal = Make();
   nodal.SetCurvature(2, true);
   int *part = plain.GeneratePartitioning(Mpi::WorldSize());
   ParMesh pplain(MPI_COMM_WORLD, plain, part), pnodal(MPI_COMM_WORLD, nodal,
                                                       part);
   delete [] part;
   pplain.Finalize(true);
   pnodal.Finalize(true);
   int same = (pplain.GetNE() == pnodal.GetNE());
   for (int e = 0; same && e < pplain.GetNE(); e++)
   {
      const int *a = pplain.GetElement(e)->GetVertices();
      const int n = pplain.GetElement(e)->GetNVertices();
      same = std::equal(a, a + n, pnodal.GetElement(e)->GetVertices());
   }
   MPI_Allreduce(MPI_IN_PLACE, &same, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
   CHECK(same == 1);
   CHECK(tet_marking::MarkedNotLongest(pnodal) == 0);
}

TEST_CASE("ParMesh marks discontinuous nodes inverted on some ranks",
          "[Parallel], [ParMesh]")
{
   // Finalize(true, true) fixes the inverted elements of each rank, then marks
   // the elements, reading discontinuous nodes in the vertex orders that they
   // follow until it reorders them: the ranks must make the same collective
   // calls, whether they have inverted elements, none, or no element at all.
   // Two disconnected tetrahedra with L2 nodes of order 2, on ranks 0 and 1
   // (the other ranks have none), the one of rank 0 inverted, or, as a
   // control, none. If a rank does not return from Finalize(), the others
   // abort the run instead of waiting for it.
   const bool invert = GENERATE(true, false);
   INFO((invert ? "one tetrahedron inverted" : "none inverted"));
   Mesh mesh(3, 8, 2);
   for (int i = 0; i < 2; i++)
   {
      const real_t a = 3.0*i;
      mesh.AddVertex(a, 0.0, 0.0);
      mesh.AddVertex(a + 1.0, 0.0, 0.0);
      mesh.AddVertex(a, 2.0, 0.0);
      mesh.AddVertex(a, 0.0, 3.0);
   }
   if (invert) { mesh.AddTet(1, 0, 2, 3); }
   else { mesh.AddTet(0, 1, 2, 3); }
   mesh.AddTet(4, 5, 6, 7);
   mesh.FinalizeTopology();
   mesh.SetCurvature(2, true);
   REQUIRE(mesh.CheckElementOrientation(false) == (invert ? 1 : 0));
   int part[2] = {0, std::min(1, Mpi::WorldSize() - 1)};
   ParMesh pmesh(MPI_COMM_WORLD, mesh, part);
   MPI_Comm watch;
   MPI_Comm_dup(MPI_COMM_WORLD, &watch);
   pmesh.Finalize(true, true);
   tet_marking::AbortUnlessAllArrive(watch, 60.0);
   MPI_Comm_free(&watch);
   int wrong = pmesh.CheckElementOrientation(false);
   MPI_Allreduce(MPI_IN_PLACE, &wrong, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK(wrong == 0);
   CHECK(tet_marking::MarkedNotLongest(pmesh) == 0);
   CHECK(tet_marking::NodesAtVerticesGap(pmesh) < 1e-12);
}

TEST_CASE("ParMesh marks and refines the same for any partition",
          "[Parallel], [ParMesh]")
{
   // The marking depends on the geometry only: a mesh distributed with a
   // METIS, a cyclic or a random partition, marked again, refined uniformly,
   // marked again and refined three times near a point is, element for element
   // (vertices in their order, marked edge first, refinement flags), the
   // serial mesh refined so. Meshes with many edges of equal length, in 2D and
   // 3D; straight, with nodes of order 1, with curved nodes of order 2, and
   // periodic with discontinuous nodes of order 2 (2D and 3D).
   const int nranks = Mpi::WorldSize();
   struct M { const char *name; std::function<Mesh()> make; bool refine; };
   auto Curved2D = []
   {
      Mesh m = Mesh::MakeCartesian2D(6, 6, Element::TRIANGLE);
      m.Finalize(true);
      m.SetCurvature(2);
      m.Transform([](const Vector &x, Vector &y)
      {
         y = x;
         y(0) += 0.03*std::sin(3.0*x(1));
         y(1) += 0.02*std::cos(2.0*x(0));
      });
      return m;
   };
   const std::vector<M> meshes =
   {
      {
         "2D Cartesian", []
         {
            Mesh m = Mesh::MakeCartesian2D(6, 6, Element::TRIANGLE);
            m.Finalize(true);
            return m;
         }, true
      },
      {"3D prisms", [] { return tet_marking::EquilateralPrisms(3); }, true},
      {
         "3D cubes", []
         {
            Mesh m = Mesh::MakeCartesian3D(3, 3, 3, Element::TETRAHEDRON);
            m.Finalize(true);
            return m;
         }, true
      },
      {
         "3D prisms, nodes of order 1", []
         {
            Mesh m = tet_marking::EquilateralPrisms(3);
            m.SetCurvature(1);
            return m;
         }, true
      },
      {"2D curved, order 2", Curved2D, true},
      {
         "3D curved, order 2", []
         { return tet_marking::TetMesh(true, 0.0, 2, true); }, true
      },
      {
         "3D periodic prisms, discontinuous nodes of order 2", []
         {
            Mesh m = tet_marking::EquilateralPrisms(3);
            m.Transform([](const Vector &x, Vector &y) { y = x; y *= 0.7; });
            std::vector<Vector> t(3, Vector(3));
            t[0] = 0.0; t[0](0) = 0.7;
            t[1] = 0.0; t[1](0) = -0.35; t[1](1) = 0.7*std::sqrt(0.75);
            t[2] = 0.0; t[2](2) = 0.7;
            Mesh periodic =
            Mesh::MakePeriodic(m, m.CreatePeriodicVertexMapping(t));
            periodic.SetCurvature(2, true);
            return periodic;
         }, true
      },
      {
         "2D periodic triangles, discontinuous nodes of order 2", []
         {
            Mesh m = Mesh::MakeCartesian2D(6, 6, Element::TRIANGLE);
            for (int v = 0; v < m.GetNV(); v++)
            {
               real_t *x = m.GetVertex(v);
               x[0] -= 0.5*x[1];
               x[1] *= std::sqrt(real_t(0.75));
            }
            m.Finalize(true);
            std::vector<Vector> t(2, Vector(2));
            t[0] = 0.0; t[0](0) = 1.0;
            t[1] = 0.0; t[1](0) = -0.5; t[1](1) = std::sqrt(real_t(0.75));
            Mesh periodic =
            Mesh::MakePeriodic(m, m.CreatePeriodicVertexMapping(t));
            periodic.SetCurvature(2, true);
            return periodic;
         }, true
      }
   };
   for (const M &m : meshes)
   {
      for (int kind = 0; kind < 3; kind++)
      {
         INFO(m.name << ", " << (kind == 0 ? "METIS" : kind == 1 ? "cyclic" :
                                 "random") << " partition");
         Mesh serial = m.make();
         Array<int> part(serial.GetNE());
         for (int e = 0; e < serial.GetNE(); e++)
         {
            std::uint64_t h = (e + 1)*0x9E3779B97F4A7C15ull;
            h ^= h >> 31;
            part[e] = (kind == 1) ? e % nranks : int(h % nranks);
         }
         ParMesh pmesh(MPI_COMM_WORLD, serial,
                       kind == 0 ? nullptr : part.GetData());
         auto Same = [&]()
         {
            return tet_marking::Description(pmesh) ==
                   tet_marking::Description(serial);
         };
         pmesh.Finalize(true);
         serial.Finalize(true);
         CHECK(Same());
         if (!m.refine) { continue; }
         pmesh.UniformRefinement();
         pmesh.Finalize(true);
         serial.UniformRefinement();
         serial.Finalize(true);
         CHECK(Same());
         for (int round = 0; round < 3; round++)
         {
            Vector c(serial.Dimension());
            c = 0.3 + 0.1*round;
            tet_marking::RefineNear(pmesh, c, 0.3);
            tet_marking::RefineNear(serial, c, 0.3);
            INFO("round " << round);
            CHECK(Same());
         }
      }
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

TEST_CASE("ParMesh reoriented updates its face neighbors",
          "[Parallel], [ParMesh]")
{
   // ReorientTetMesh() changes the vertex order of the tetrahedra and numbers
   // the faces again. The face neighbor data exchanged before, here by a
   // Nedelec space of order 3, must not be used after it: new Nedelec and
   // Raviart-Thomas spaces are conforming, and the transformations of the
   // shared faces map their points to the same places from both sides.
   Mesh serial = tet_marking::EquilateralPrisms(4);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   pmesh.UniformRefinement();
   {
      ND_FECollection nd(3, 3);
      ParFiniteElementSpace fes(&pmesh, &nd);
      fes.Dof_TrueDof_Matrix();
   }
#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
   pmesh.ReorientTetMesh();
#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic pop
#endif
   CHECK(tet_marking::NDAndRTConforming(pmesh));
   CHECK(tet_marking::SharedFaceGap(pmesh) < 1e-12);
}

TEST_CASE("ParMesh marked again keeps the space of its nodes conforming",
          "[Parallel], [ParMesh]")
{
   // A new marking, and a reorientation, number the edges and faces again and
   // rebuild the space of the nodes: it must be built with the shared edges
   // and faces of the new numbering. With nodes of orders 2 to 4, the nodes
   // and a field interpolated in their space equal their restriction to the
   // true dofs prolongated back, after a uniform refinement and a new marking,
   // and after a reorientation.
   const int order = GENERATE(2, 3, 4);
   INFO("nodes of order " << order);
   Mesh serial = tet_marking::TetMesh(true, 0.0, order, false);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   serial.Clear();
   pmesh.UniformRefinement();
   pmesh.Finalize(true);
   CHECK(tet_marking::NodalSpaceError(pmesh) < 1e-12);
#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
   pmesh.ReorientTetMesh();
#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic pop
#endif
   CHECK(tet_marking::NodalSpaceError(pmesh) < 1e-12);
}

#endif // MFEM_USE_MPI

} // namespace mfem
