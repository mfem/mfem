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
#include <climits>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace mfem
{

#ifdef MFEM_USE_MPI

namespace pmesh_rebalance
{

// Tests of ParMesh::Rebalance(const Array<int> &) for conforming meshes of
// triangles and tetrahedra. Most tests run on communicators of 1, 2, 3 and all
// ranks; the longer ones on all the ranks only, so that each launch of the
// parallel tests stays short (they are launched on several numbers of ranks).
// Every rebalance that completes checks what a rebalance must not change
// (RebalanceChecked()). Partitions are functions of the element centers, so
// the same partition can be applied to a ParMesh and to the serial mesh it
// comes from.

// UNIFORM: refined uniformly once more at the end, so that the tetrahedra are
// unmarked; REMARKED: the same, then marked again (Finalize(true)), as before
// a local refinement; ROTATED: some boundary triangles have a refinement edge
// (vertices 0-1) that is not the marked edge of their face.
enum class Kind { STRAIGHT, CURVED, PERIODIC, CRACKED, UNIFORM, REMARKED,
                  ROTATED
                };

using Rule = std::function<int(const Vector &center, int nranks)>;

bool AllOf(MPI_Comm comm, bool local)
{
   int l = local ? 1 : 0, g = 0;
   MPI_Allreduce(&l, &g, 1, MPI_INT, MPI_LAND, comm);
   return g == 1;
}

/// Run @a f on communicators of increasing size: 1, 2, 3 and all ranks, or,
/// if @a all_sizes is false, all the ranks only. An exception thrown by @a f
/// (a failed REQUIRE, or an error of MFEM on all the ranks of a communicator)
/// is rethrown after the last size, so that the ranks outside the
/// communicator do not wait for the others.
void ForEachComm(const std::function<void(MPI_Comm)> &f, bool all_sizes = true)
{
   const int size = Mpi::WorldSize(), rank = Mpi::WorldRank();
   std::vector<int> sizes = {1, 2, 3, size};
   if (!all_sizes) { sizes = {size}; }
   std::sort(sizes.begin(), sizes.end());
   sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
   std::exception_ptr error;
   for (int n : sizes)
   {
      if (n > size) { continue; }
      MPI_Comm comm;
      MPI_Comm_split(MPI_COMM_WORLD, (rank < n) ? 0 : MPI_UNDEFINED, rank,
                     &comm);
      if (comm != MPI_COMM_NULL)
      {
         try { f(comm); }
         catch (...) { if (!error) { error = std::current_exception(); } }
         MPI_Comm_free(&comm);
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }
   if (error) { std::rethrow_exception(error); }
}

/// Smooth deformation used to curve the meshes.
void Deform(const Vector &x, Vector &y)
{
   y = x;
   y(0) += 0.04*std::sin(M_PI*x(1)) + 0.02*x(0)*x(1);
   y(1) += 0.03*std::sin(2.0*M_PI*x(0));
   if (x.Size() == 3) { y(2) += 0.03*x(0)*x(1) + 0.02*std::sin(M_PI*x(0)); }
}

/// Add boundary elements on the interior faces between the elements of
/// attributes 1 and 2, with attribute 10. With @a flip, every other one has
/// the orientation of the second element.
void AddInterfaceBoundary(Mesh &mesh, bool flip)
{
   int count = 0;
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      int e1, e2;
      mesh.GetFaceElements(f, &e1, &e2);
      if (e2 < 0) { continue; }
      const int a1 = mesh.GetAttribute(e1), a2 = mesh.GetAttribute(e2);
      if (std::min(a1, a2) != 1 || std::max(a1, a2) != 2) { continue; }
      Array<int> v;
      mesh.GetFaceVertices(f, v);
      if (flip && (count++ % 2)) { std::swap(v[0], v[1]); }
      Element *be = mesh.NewElement((mesh.Dimension() == 2) ?
                                    Geometry::SEGMENT : Geometry::TRIANGLE);
      be->SetVertices(v);
      be->SetAttribute(10);
      mesh.AddBdrElement(be);
   }
}

/// Duplicate the vertices of the plane x = 1/2 with y < 1/2 for the elements
/// with x > 1/2 (a crack ending at y = 1/2), with boundary elements on both
/// sides of the crack (attributes 11 and 12).
Mesh Crack(Mesh &mesh)
{
   const int dim = mesh.Dimension();
   const real_t eps = 1e-10;
   Array<int> dup(mesh.GetNV());
   dup = -1;
   int nv = mesh.GetNV();
   for (int i = 0; i < mesh.GetNV(); i++)
   {
      const real_t *x = mesh.GetVertex(i);
      if (std::abs(x[0] - 0.5) < eps && x[1] < 0.5 - eps) { dup[i] = nv++; }
   }
   Mesh cracked(dim, nv, mesh.GetNE(), 0, dim);
   for (int i = 0; i < mesh.GetNV(); i++)
   {
      cracked.AddVertex(mesh.GetVertex(i));
   }
   for (int i = 0; i < mesh.GetNV(); i++)
   {
      if (dup[i] >= 0) { cracked.AddVertex(mesh.GetVertex(i)); }
   }
   Vector c;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementCenter(e, c);
      Array<int> v;
      mesh.GetElementVertices(e, v);
      if (c(0) > 0.5)
      {
         for (int &vi : v) { if (dup[vi] >= 0) { vi = dup[vi]; } }
      }
      Element *el = cracked.NewElement(mesh.GetElementGeometry(e));
      el->SetVertices(v);
      el->SetAttribute(mesh.GetAttribute(e));
      cracked.AddElement(el);
   }
   for (int b = 0; b < mesh.GetNBE(); b++)
   {
      if (mesh.GetBdrAttribute(b) == 10) { continue; } // on the crack
      Array<int> v;
      mesh.GetBdrElementVertices(b, v);
      int e, info;
      mesh.GetBdrElementAdjacentElement(b, e, info);
      mesh.GetElementCenter(e, c);
      if (c(0) > 0.5)
      {
         for (int &vi : v) { if (dup[vi] >= 0) { vi = dup[vi]; } }
      }
      Element *be = cracked.NewElement(mesh.GetBdrElementGeometry(b));
      be->SetVertices(v);
      be->SetAttribute(mesh.GetBdrAttribute(b));
      cracked.AddBdrElement(be);
   }
   cracked.FinalizeTopology(false);
   // The two sides of the crack.
   const int nbe = cracked.GetNBE();
   for (int f = 0; f < cracked.GetNumFaces(); f++)
   {
      int e1, e2;
      cracked.GetFaceElements(f, &e1, &e2);
      if (e2 >= 0) { continue; }
      Array<int> v;
      cracked.GetFaceVertices(f, v);
      bool on_crack = true;
      for (int vi : v)
      {
         const real_t *x = cracked.GetVertex(vi);
         on_crack = on_crack && std::abs(x[0] - 0.5) < eps && x[1] < 0.5 + eps;
      }
      if (!on_crack) { continue; }
      cracked.GetElementCenter(e1, c);
      Element *be = cracked.NewElement((dim == 2) ? Geometry::SEGMENT :
                                       Geometry::TRIANGLE);
      be->SetVertices(v);
      be->SetAttribute((c(0) < 0.5) ? 11 : 12);
      cracked.AddBdrElement(be);
   }
   MFEM_VERIFY(cracked.GetNBE() > nbe, "no crack");
   cracked.FinalizeTopology(false);
   cracked.Finalize(true);
   return cracked;
}

/// A serial test mesh with several element and boundary attributes, interior
/// boundary elements, attribute sets, nodes of the given order and ordering,
/// and refinement flags varied by local refinement.
Mesh MakeMesh(int dim, int order, Kind kind,
              Ordering::Type ordering = Ordering::byVDIM)
{
   Mesh mesh = (dim == 2) ?
               Mesh::MakeCartesian2D(6, 4, Element::TRIANGLE, true) :
               Mesh::MakeCartesian3D(4, 2, 2, Element::TETRAHEDRON);
   Vector c;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementCenter(e, c);
      mesh.SetAttribute(e, 1 + (c(0) > 0.5) + 2*(c(1) > 0.5));
   }
   mesh.SetAttributes();
   if (kind != Kind::PERIODIC)
   {
      // In 2D, the ParMesh constructor gives an interior boundary element to
      // its element of lowest index, and the rebalance to the element its
      // normal points out of: they agree for the orientation of the faces.
      AddInterfaceBoundary(mesh, dim == 3);
      mesh.FinalizeTopology(false);
      mesh.Finalize(true);
   }
   if (kind == Kind::CRACKED) { mesh = Crack(mesh); }
   if (kind == Kind::PERIODIC)
   {
      Vector t(dim);
      t = 0.0;
      t(0) = 1.0;
      std::vector<Vector> translations = {t};
      mesh = Mesh::MakePeriodic(mesh,
                                mesh.CreatePeriodicVertexMapping(translations));
   }
   mesh.SetAttributes();
   mesh.attribute_sets.CreateAttributeSet("left");
   mesh.attribute_sets.AddToAttributeSet("left", 1);
   mesh.attribute_sets.AddToAttributeSet("left", 3);
   mesh.bdr_attribute_sets.CreateAttributeSet("ends");
   mesh.bdr_attribute_sets.AddToAttributeSet("ends", 1);
   mesh.bdr_attribute_sets.AddToAttributeSet("ends", 2);

   if (kind == Kind::PERIODIC || order > 1 || kind == Kind::CURVED)
   {
      mesh.SetCurvature(order, kind == Kind::PERIODIC, dim, ordering);
   }
   if (kind == Kind::CURVED) { mesh.Transform(Deform); }

   // Local refinement around two points, so that the refinement flags of the
   // tetrahedra and the vertex orders of the triangles vary.
   for (int pass = 0; pass < 2; pass++)
   {
      Array<int> marked;
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         mesh.GetElementCenter(e, c);
         real_t r2 = 0.0;
         for (int d = 0; d < dim; d++)
         {
            const real_t p = (pass == 0) ? 0.3 : 0.7;
            r2 += (c(d) - p)*(c(d) - p);
         }
         if (r2 < 0.06) { marked.Append(e); }
      }
      mesh.GeneralRefinement(marked, 0);
   }
   if (kind == Kind::UNIFORM || kind == Kind::REMARKED)
   {
      mesh.UniformRefinement();
      if (kind == Kind::REMARKED) { mesh.Finalize(true); }
   }
   if (kind == Kind::ROTATED)
   {
      // Rotate the vertices of every other boundary triangle: the orientation
      // stays, the refinement edge changes.
      for (int b = 0; b < mesh.GetNBE(); b += 2)
      {
         int *v = mesh.GetBdrElement(b)->GetVertices();
         if (dim == 3) { std::rotate(v, v + 1, v + 3); }
      }
   }
   return mesh;
}

// Partition rules, as functions of the element centers.
int Slab(real_t x, int n) { return std::min(n - 1, std::max(0, int(x*n))); }

const std::vector<std::pair<std::string, Rule>> &InitialRules()
{
   static const std::vector<std::pair<std::string, Rule>> rules =
   {
      // Nearly all elements on rank 0, a few on the last rank, others empty.
      {
         "unbalanced", [](const Vector &c, int n)
         { return (c(0) > 0.9 && c(1) > 0.8) ? n - 1 : 0; }
      },
      {"x slabs", [](const Vector &c, int n) { return Slab(c(0), n); }}
   };
   return rules;
}

const std::vector<std::pair<std::string, Rule>> &TargetRules()
{
   static const std::vector<std::pair<std::string, Rule>> rules =
   {
      // Every element moves away from its x slab.
      {
         "shifted slabs", [](const Vector &c, int n)
         { return (Slab(c(0), n) + 1) % n; }
      },
      {
         "reversed y slabs", [](const Vector &c, int n)
         { return n - 1 - Slab(c(1), n); }
      },
      {
         "hash", [](const Vector &c, int n)
         {
            const real_t z = (c.Size() > 2) ? c(2) : 0.0;
            const real_t s = std::sin(12.9898*c(0) + 78.233*c(1) + 37.719*z);
            return int(std::floor(std::abs(43758.5453*s))) % n;
         }
      },
      {"last rank", [](const Vector &, int n) { return n - 1; }}
   };
   return rules;
}

Array<int> SerialPartition(Mesh &mesh, const Rule &rule, int nranks)
{
   Array<int> part(mesh.GetNE());
   Vector c;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementCenter(e, c);
      part[e] = rule(c, nranks);
   }
   return part;
}

Array<int> LocalPartition(ParMesh &pmesh, const Rule &rule)
{
   return SerialPartition(pmesh, rule, pmesh.GetNRanks());
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

using Descriptors = std::vector<std::vector<real_t>>;

/// Vertex coordinates rounded to 1e-9: the vertex coordinates of a curved mesh
/// are averages of the nodes over the local elements, which depend on the
/// distribution to round-off.
void AppendVertex(const Mesh &mesh, int v, std::vector<real_t> &x)
{
   for (int k = 0; k < mesh.SpaceDimension(); k++)
   {
      x.push_back(std::round(mesh.GetVertex(v)[k]*1e9));
   }
}

/// Attribute, refinement flag, and vertex coordinates (in the element's
/// vertex order) or node values of the local element @a e.
std::vector<real_t> ElementDescriptor(ParMesh &pmesh, int e)
{
   Array<int> vdofs;
   Vector vals;
   std::vector<real_t> x;
   {
      const Element *el = pmesh.GetElement(e);
      x.push_back(el->GetAttribute());
      const bool tet = (el->GetType() == Element::TETRAHEDRON);
      x.push_back(tet ? static_cast<const Tetrahedron *>(el)->
                  GetRefinementFlag() : 0);
      if (pmesh.GetNodes())
      {
         // The node values, in the element's DOF order, include its vertices
         // in its vertex order. (The vertex coordinates of a curved mesh are
         // averages of the nodes over the local elements.)
         pmesh.GetNodes()->FESpace()->GetElementVDofs(e, vdofs);
         pmesh.GetNodes()->GetSubVector(vdofs, vals);
         x.insert(x.end(), vals.begin(), vals.end());
      }
      else
      {
         const int *v = el->GetVertices();
         for (int j = 0; j < el->GetNVertices(); j++)
         {
            for (int k = 0; k < pmesh.SpaceDimension(); k++)
            {
               x.push_back(pmesh.GetVertex(v[j])[k]);
            }
         }
      }
   }
   return x;
}

/// The descriptors of the local elements, sorted.
Descriptors ElementDescriptors(ParMesh &pmesh)
{
   Descriptors d(pmesh.GetNE());
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      d[e] = ElementDescriptor(pmesh, e);
   }
   std::sort(d.begin(), d.end());
   return d;
}

/// Attribute and vertex coordinates (in order) of each local boundary element.
Descriptors BoundaryDescriptors(const Mesh &mesh)
{
   Descriptors d(mesh.GetNBE());
   for (int b = 0; b < mesh.GetNBE(); b++)
   {
      const Element *be = mesh.GetBdrElement(b);
      d[b].push_back(be->GetAttribute());
      const int *v = be->GetVertices();
      for (int j = 0; j < be->GetNVertices(); j++)
      {
         AppendVertex(mesh, v[j], d[b]);
      }
   }
   std::sort(d.begin(), d.end());
   return d;
}

/// Gather the descriptors of all ranks on rank 0 of @a comm, sorted.
Descriptors Gather(MPI_Comm comm, const Descriptors &local)
{
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);
   std::vector<real_t> flat;
   for (const auto &x : local)
   {
      flat.push_back(real_t(x.size()));
      flat.insert(flat.end(), x.begin(), x.end());
   }
   int n = static_cast<int>(flat.size());
   std::vector<int> counts(size), displs(size + 1, 0);
   MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
   for (int i = 0; i < size; i++) { displs[i+1] = displs[i] + counts[i]; }
   std::vector<real_t> all(rank == 0 ? displs[size] : 0);
   MPI_Gatherv(flat.data(), n, MPITypeMap<real_t>::mpi_type, all.data(),
               counts.data(), displs.data(), MPITypeMap<real_t>::mpi_type, 0,
               comm);
   Descriptors d;
   for (std::size_t k = 0; k < all.size();)
   {
      const std::size_t m = static_cast<std::size_t>(all[k++]);
      d.emplace_back(all.begin() + k, all.begin() + k + m);
      k += m;
   }
   std::sort(d.begin(), d.end());
   return d;
}

/// For each element, (rank[e], index[e]) followed by its descriptor, gathered
/// on rank 0 and sorted: with the rank and index of each element before a
/// rebalance, and with those given by GetRebalanceOldElements() after it, the
/// two agree if the previous location of each element is exact.
Descriptors Located(ParMesh &pmesh, const Array<int> &rank,
                    const Array<int> &index)
{
   Descriptors d(pmesh.GetNE());
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      d[e] = {real_t(rank[e]), real_t(index[e])};
      const std::vector<real_t> x = ElementDescriptor(pmesh, e);
      d[e].insert(d[e].end(), x.begin(), x.end());
   }
   return Gather(pmesh.GetComm(), d);
}

/// The boundary elements of @a a and @a b agree: all of them on each rank in
/// 3D, and in 2D those on the exterior boundary on each rank and all of them
/// together (the ParMesh constructor gives an interior boundary edge to the
/// element of lowest index, the rebalance to the element it points out of).
bool SameBoundary(ParMesh &a, ParMesh &b)
{
   Descriptors da = BoundaryDescriptors(a), db = BoundaryDescriptors(b);
   if (a.Dimension() == 3) { return da == db; }
   auto Exterior = [](const Descriptors &d)
   {
      Descriptors e;
      for (const auto &x : d) { if (x[0] <= 4) { e.push_back(x); } }
      return e;
   };
   const bool same_all = (Gather(a.GetComm(), da) == Gather(b.GetComm(), db));
   return Exterior(da) == Exterior(db) && same_all;
}

/// The groups of the shared entities: ranks, and sorted coordinates of the
/// shared vertices, edges and triangles of each group.
Descriptors SharedDescriptors(ParMesh &pmesh)
{
   Descriptors d;
   auto Coords = [&pmesh](const Array<int> &v, std::vector<real_t> &x)
   {
      Descriptors verts(v.Size());
      for (int i = 0; i < v.Size(); i++)
      {
         AppendVertex(pmesh, v[i], verts[i]);
      }
      std::sort(verts.begin(), verts.end());
      for (auto &y : verts) { x.insert(x.end(), y.begin(), y.end()); }
   };
   for (int g = 1; g < pmesh.GetNGroups(); g++)
   {
      std::vector<real_t> x;
      for (int k = 0; k < pmesh.gtopo.GetGroupSize(g); k++)
      {
         x.push_back(pmesh.gtopo.GetNeighborRank(pmesh.gtopo.GetGroup(g)[k]));
      }
      std::sort(x.begin(), x.end());
      Descriptors ents;
      Array<int> v;
      for (int i = 0; i < pmesh.GroupNVertices(g); i++)
      {
         std::vector<real_t> y = {0};
         v.SetSize(1);
         v[0] = pmesh.GroupVertex(g, i);
         Coords(v, y);
         ents.push_back(y);
      }
      for (int i = 0; i < pmesh.GroupNEdges(g); i++)
      {
         std::vector<real_t> y = {1};
         pmesh.GetEdgeVertices(pmesh.GroupEdge(g, i), v);
         Coords(v, y);
         ents.push_back(y);
      }
      for (int i = 0; i < pmesh.GroupNTriangles(g); i++)
      {
         std::vector<real_t> y = {2};
         pmesh.GetFaceVertices(pmesh.GroupTriangle(g, i), v);
         Coords(v, y);
         ents.push_back(y);
      }
      std::sort(ents.begin(), ents.end());
      for (auto &y : ents) { x.insert(x.end(), y.begin(), y.end()); }
      d.push_back(x);
   }
   std::sort(d.begin(), d.end());
   return d;
}

/// Check that all members of each group list the shared vertices, edges and
/// triangles in the same order, with the same vertex order: the master
/// broadcasts the coordinates of its entities, the others compare.
bool SharedEntitiesAgree(ParMesh &pmesh)
{
   const int ng = pmesh.GetNGroups();
   GroupCommunicator gc(pmesh.gtopo);
   Table &group_ldof = gc.GroupLDofTable();
   std::vector<real_t> values;
   Array<int> v;
   group_ldof.MakeI(ng);
   std::vector<std::vector<real_t>> per_group(ng);
   for (int g = 1; g < ng; g++)
   {
      std::vector<real_t> &x = per_group[g];
      for (int i = 0; i < pmesh.GroupNVertices(g); i++)
      {
         AppendVertex(pmesh, pmesh.GroupVertex(g, i), x);
      }
      for (int i = 0; i < pmesh.GroupNEdges(g); i++)
      {
         int edge, o;
         pmesh.GroupEdge(g, i, edge, o);
         pmesh.GetEdgeVertices(edge, v);
         if (o < 0) { std::swap(v[0], v[1]); }
         for (int vi : v) { AppendVertex(pmesh, vi, x); }
      }
      for (int i = 0; i < pmesh.GroupNTriangles(g); i++)
      {
         int face, o;
         pmesh.GroupTriangle(g, i, face, o);
         pmesh.GetFaceVertices(face, v);
         int sv[3];
         const int *perm = Geometry::Constants<Geometry::TRIANGLE>::Orient[o];
         for (int j = 0; j < 3; j++) { sv[perm[j]] = v[j]; }
         for (int j = 0; j < 3; j++) { AppendVertex(pmesh, sv[j], x); }
      }
      group_ldof.AddColumnsInRow(g, static_cast<int>(x.size()));
   }
   group_ldof.MakeJ();
   for (int g = 1; g < ng; g++)
   {
      for (std::size_t k = 0; k < per_group[g].size(); k++)
      {
         group_ldof.AddConnection(g, static_cast<int>(values.size()));
         values.push_back(per_group[g][k]);
      }
   }
   group_ldof.ShiftUpI();
   gc.Finalize();
   Array<real_t> mine(values.data(), static_cast<int>(values.size()));
   Array<real_t> master(mine);
   gc.Bcast(master);
   bool ok = true;
   for (int k = 0; k < mine.Size(); k++) { ok = ok && (mine[k] == master[k]); }
   return AllOf(pmesh.GetComm(), ok);
}

struct Counts
{
   long long ne, nv, nedges, nfaces, nbe;
   std::vector<long long> nbe_attr;
   real_t volume;
   bool operator==(const Counts &o) const
   {
      return ne == o.ne && nv == o.nv && nedges == o.nedges &&
             nfaces == o.nfaces && nbe == o.nbe && nbe_attr == o.nbe_attr &&
             std::abs(volume - o.volume) <= 1e-12*std::abs(o.volume);
   }
};

Counts SerialCounts(Mesh &mesh)
{
   Counts c;
   const int dim = mesh.Dimension();
   H1_FECollection h1(1, dim);
   ND_FECollection nd(1, dim);
   c.ne = mesh.GetNE();
   c.nv = FiniteElementSpace(&mesh, &h1).GetTrueVSize();
   c.nedges = FiniteElementSpace(&mesh, &nd).GetTrueVSize();
   c.nfaces = (dim == 3) ? mesh.GetNFaces() : c.nedges;
   c.nbe = mesh.GetNBE();
   c.nbe_attr.assign(mesh.bdr_attributes.Max() + 1, 0);
   for (int b = 0; b < mesh.GetNBE(); b++)
   {
      c.nbe_attr[mesh.GetBdrAttribute(b)]++;
   }
   c.volume = 0.0;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      c.volume += mesh.GetElementVolume(e);
   }
   return c;
}

Counts ParallelCounts(ParMesh &pmesh)
{
   Counts c;
   const int dim = pmesh.Dimension();
   MPI_Comm comm = pmesh.GetComm();
   H1_FECollection h1(1, dim);
   ND_FECollection nd(1, dim);
   RT_FECollection rt(0, dim);
   c.ne = pmesh.GetGlobalNE();
   c.nv = ParFiniteElementSpace(&pmesh, &h1).GlobalTrueVSize();
   c.nedges = ParFiniteElementSpace(&pmesh, &nd).GlobalTrueVSize();
   c.nfaces = (dim == 3) ?
              ParFiniteElementSpace(&pmesh, &rt).GlobalTrueVSize() : c.nedges;
   c.nbe = pmesh.ReduceInt(pmesh.GetNBE());
   c.nbe_attr.assign(pmesh.bdr_attributes.Max() + 1, 0);
   for (int b = 0; b < pmesh.GetNBE(); b++)
   {
      c.nbe_attr[pmesh.GetBdrAttribute(b)]++;
   }
   MPI_Allreduce(MPI_IN_PLACE, c.nbe_attr.data(),
                 static_cast<int>(c.nbe_attr.size()), MPI_LONG_LONG, MPI_SUM,
                 comm);
   c.volume = 0.0;
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      c.volume += pmesh.GetElementVolume(e);
   }
   MPI_Allreduce(MPI_IN_PLACE, &c.volume, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, comm);
   return c;
}

/// Check the true sizes of H1, ND, RT and L2 spaces of orders 1-3 against
/// the serial mesh, and that the interpolant of a polynomial has no jump
/// across the shared entities: P R x = x.
/// What a rebalance must not change: the global numbers of elements,
/// vertices, edges and faces (each shared entity counted on the master of its
/// group), of boundary elements per attribute, and the global true sizes of
/// scalar H1_p, ND_p, RT_{p-1} and L2_{p-1} spaces, p = 1, 2, 3 (other orders
/// and vector spaces: the tests of the transfer).
std::vector<long long> Invariants(ParMesh &pmesh)
{
   const int dim = pmesh.Dimension();
   long long c[4] = {pmesh.GetNE(), pmesh.GetNV(), pmesh.GetNEdges(),
                     (dim == 3) ? pmesh.GetNFaces() : 0
                    };
   for (int g = 1; g < pmesh.GetNGroups(); g++)
   {
      if (pmesh.gtopo.IAmMaster(g)) { continue; }
      c[1] -= pmesh.GroupNVertices(g);
      c[2] -= pmesh.GroupNEdges(g);
      if (dim == 3) { c[3] -= pmesh.GroupNTriangles(g); }
   }
   MPI_Allreduce(MPI_IN_PLACE, c, 4, MPI_LONG_LONG, MPI_SUM, pmesh.GetComm());
   std::vector<long long> inv(c, c + 4);
   const int nattr = pmesh.bdr_attributes.Size() ?
                     pmesh.bdr_attributes.Max() : 0;
   std::vector<long long> nbe(nattr + 1, 0);
   for (int b = 0; b < pmesh.GetNBE(); b++) { nbe[pmesh.GetBdrAttribute(b)]++; }
   MPI_Allreduce(MPI_IN_PLACE, nbe.data(), nattr + 1, MPI_LONG_LONG, MPI_SUM,
                 pmesh.GetComm());
   inv.insert(inv.end(), nbe.begin(), nbe.end());
   for (int p = 1; p <= 3; p++)
   {
      H1_FECollection h1(p, dim);
      ND_FECollection nd(p, dim);
      RT_FECollection rt(p - 1, dim);
      L2_FECollection l2(p - 1, dim);
      for (FiniteElementCollection *fec :
           std::vector<FiniteElementCollection *> {&h1, &nd, &rt, &l2})
      {
         inv.push_back(ParFiniteElementSpace(&pmesh, fec).GlobalTrueVSize());
      }
   }
   return inv;
}

/// The largest |P R x - x| of a projected field over ND_3 and RT_2 spaces: zero
/// for spaces that conform across the ranks.
real_t ConformityResidual(ParMesh &pmesh)
{
   const int dim = pmesh.Dimension();
   ND_FECollection nd(3, dim);
   RT_FECollection rt(2, dim);
   VectorFunctionCoefficient coeff(dim, [](const Vector &x, Vector &y)
   {
      y.SetSize(x.Size());
      for (int d = 0; d < x.Size(); d++)
      {
         y(d) = 1.0 + x((d + 1) % x.Size()) - 0.5*x(d)*x(d);
      }
   });
   real_t res = 0.0;
   for (FiniteElementCollection *fec :
        std::vector<FiniteElementCollection *> {&nd, &rt})
   {
      ParFiniteElementSpace fes(&pmesh, fec);
      ParGridFunction x(&fes), y(&fes);
      x.ProjectCoefficient(coeff);
      Vector tv(fes.GetTrueVSize());
      x.GetTrueDofs(tv);
      y.SetFromTrueDofs(tv);
      y -= x;
      res = std::max(res, y.Normlinf());
   }
   MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 pmesh.GetComm());
   return res;
}

/// The interior penalty (DG) diffusion energy of a discontinuous field with a
/// jump across every face (an offset per element attribute).
real_t DGEnergy(ParMesh &pmesh)
{
   L2_FECollection fec(2, pmesh.Dimension());
   ParFiniteElementSpace fes(&pmesh, &fec);
   ParGridFunction x(&fes);
   FunctionCoefficient f([](const Vector &p)
   {
      return 0.7 + std::sin(2.0*M_PI*p(0)) + 0.3*p(1)*p(1);
   });
   x.ProjectCoefficient(f);
   Array<int> dofs;
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      fes.GetElementDofs(e, dofs);
      for (int d : dofs) { x(d) += 0.001*pmesh.GetAttribute(e); }
   }
   ConstantCoefficient one(1.0);
   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, -1.0, 20.0));
   a.Assemble();
   a.Finalize();
   std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());
   Vector X(fes.GetTrueVSize()), AX(fes.GetTrueVSize());
   x.GetTrueDofs(X);
   A->Mult(X, AX);
   return InnerProduct(pmesh.GetComm(), X, AX);
}

/// The two elements of each shared face map the points of the face to the
/// same physical points.
bool SharedFacePointsAgree(ParMesh &pmesh)
{
   pmesh.ExchangeFaceNbrData();
   bool ok = true;
   Vector p1, p2;
   for (int sf = 0; sf < pmesh.GetNSharedFaces(); sf++)
   {
      FaceElementTransformations *t = pmesh.GetSharedFaceTransformations(sf);
      const IntegrationRule &ir = IntRules.Get(t->GetGeometryType(), 6);
      for (int k = 0; k < ir.GetNPoints(); k++)
      {
         t->SetAllIntPoints(&ir.IntPoint(k));
         t->Elem1->Transform(t->GetElement1IntPoint(), p1);
         t->Elem2->Transform(t->GetElement2IntPoint(), p2);
         ok = ok && p1.DistanceTo(p2) < 1e-12;
      }
   }
   return AllOf(pmesh.GetComm(), ok);
}

// ---------------------------------------------------------------------------
// Transfer of grid functions
// ---------------------------------------------------------------------------

/// A space of the transfer tests.
struct SpaceCase
{
   std::string name;
   std::function<FiniteElementCollection *(int dim)> fec;
   int vdim; // 0: the dimension
   Ordering::Type ordering;
   /// Fields set through scalar components: for ND and RT spaces with
   /// vdim > 1, whose parallel prolongation is not used by the tests.
   bool components = false;
};

std::vector<SpaceCase> SpaceCases()
{
   std::vector<SpaceCase> c;
   for (int p = 1; p <= 4; p++)
   {
      c.push_back({"H1_" + std::to_string(p), [p](int d)
      { return new H1_FECollection(p, d); }, 1, Ordering::byNODES});
   }
   c.push_back({"H1_2^d by nodes", [](int d)
   { return new H1_FECollection(2, d); }, 0, Ordering::byNODES});
   c.push_back({"H1_3^d by vdim", [](int d)
   { return new H1_FECollection(3, d); }, 0, Ordering::byVDIM});
   for (int p = 0; p <= 3; p++)
   {
      c.push_back({"L2_" + std::to_string(p), [p](int d)
      { return new L2_FECollection(p, d); }, 1, Ordering::byNODES});
   }
   c.push_back({"L2_1^d by vdim", [](int d)
   { return new L2_FECollection(1, d); }, 0, Ordering::byVDIM});
   for (int p = 1; p <= 4; p++)
   {
      c.push_back({"ND_" + std::to_string(p), [p](int d)
      { return new ND_FECollection(p, d); }, 1, Ordering::byNODES});
   }
   for (int p = 0; p <= 3; p++)
   {
      c.push_back({"RT_" + std::to_string(p), [p](int d)
      { return new RT_FECollection(p, d); }, 1, Ordering::byNODES});
   }
   return c;
}

/// The spaces of the test of transferred projections: SpaceCases() with fewer
/// orders (each order of SpaceCases(): the tests of projections and round
/// trips).
std::vector<SpaceCase> TransferSpaceCases()
{
   std::vector<SpaceCase> all = SpaceCases(), c;
   for (const SpaceCase &sc : all)
   {
      for (const char *name :
           {"H1_3", "H1_2^d by nodes", "L2_2",
            "L2_1^d by vdim", "ND_2", "ND_3", "RT_1", "RT_2"
           })
      {
         if (sc.name == name) { c.push_back(sc); }
      }
   }
   return c;
}

/// A smooth function, not in any of the spaces (component k of a field).
real_t Smooth(const Vector &x, int k)
{
   const real_t z = (x.Size() == 3) ? x(2) : 0.0;
   return std::sin(1.3*x(0) + 0.7*k) * std::cos(0.9*x(1) - 0.3*k) +
          0.5*z*x(0) + 0.1*k;
}

/// A polynomial of degree p (component k of a field).
real_t Poly(const Vector &x, int k, int p)
{
   real_t v = 1.0 + 0.5*k, t = 1.0;
   for (int i = 1; i <= p; i++)
   {
      t *= (0.3*x(0) - 0.2*x(1) + ((x.Size() == 3) ? 0.4*x(2) : 0.0) +
            0.1*k);
      v += t/i;
   }
   return v;
}

/// The values at the points of a quadrature rule of each local element of
/// @a x, after the element's location (rank[e], index[e]); gathered on rank
/// 0 and sorted, as in Located().
Descriptors FieldValues(ParGridFunction &x, const Array<int> &rank,
                        const Array<int> &index)
{
   ParMesh &pmesh = *x.ParFESpace()->GetParMesh();
   Descriptors d(pmesh.GetNE());
   Vector v;
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      d[e] = {real_t(rank[e]), real_t(index[e])};
      ElementTransformation *T = pmesh.GetElementTransformation(e);
      const IntegrationRule &ir =
         IntRules.Get(pmesh.GetElementBaseGeometry(e), 5);
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         T->SetIntPoint(&ir.IntPoint(i));
         x.GetVectorValue(*T, ir.IntPoint(i), v);
         d[e].insert(d[e].end(), v.begin(), v.end());
      }
   }
   return Gather(pmesh.GetComm(), d);
}

/// The reference coefficients of each local element of @a x: the values of
/// its DOFs in the element, with the signs and DOF transformations of its
/// edges and faces undone, which determine the field on the element whatever
/// the numbering and the orientation of its edges and faces; after the
/// element's location (rank[e], index[e]), gathered on rank 0 and sorted, as
/// in Located().
Descriptors Coefficients(ParGridFunction &x, const Array<int> &rank,
                         const Array<int> &index)
{
   const ParFiniteElementSpace &fes = *x.ParFESpace();
   ParMesh &pmesh = *fes.GetParMesh();
   Descriptors d(pmesh.GetNE());
   Array<int> vdofs;
   DofTransformation doftrans;
   Vector v;
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      fes.GetElementVDofs(e, vdofs, doftrans);
      x.GetSubVector(vdofs, v);
      doftrans.InvTransformPrimal(v);
      d[e] = {real_t(rank[e]), real_t(index[e])};
      d[e].insert(d[e].end(), v.begin(), v.end());
   }
   return Gather(pmesh.GetComm(), d);
}

/// The same lists, with values equal to a relative @a tol.
bool Close(const Descriptors &a, const Descriptors &b, real_t tol)
{
   if (a.size() != b.size()) { return false; }
   for (std::size_t i = 0; i < a.size(); i++)
   {
      if (a[i].size() != b[i].size()) { return false; }
      for (std::size_t j = 0; j < a[i].size(); j++)
      {
         if (std::abs(a[i][j] - b[i][j]) > tol*(1.0 + std::abs(a[i][j])))
         {
            return false;
         }
      }
   }
   return true;
}

/// A tetrahedral mesh whose elements have their vertices in random even
/// permutations (so that their faces have all six orientations), unmarked,
/// with an attribute per element.
Mesh PermutedTetMesh()
{
   Mesh mesh = Mesh::MakeCartesian3D(3, 3, 2, Element::TETRAHEDRON);
   const int even[12][4] = {{0,1,2,3}, {0,2,3,1}, {0,3,1,2}, {1,0,3,2},
      {1,2,0,3}, {1,3,2,0}, {2,0,1,3}, {2,1,3,0}, {2,3,0,1}, {3,0,2,1},
      {3,1,0,2}, {3,2,1,0}
   };
   std::uint64_t state = 2024;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      state = state*6364136223846793005ull + 1442695040888963407ull;
      const int *perm = even[(state >> 33) % 12];
      int *v = mesh.GetElement(e)->GetVertices();
      const int w[4] = {v[0], v[1], v[2], v[3]};
      for (int j = 0; j < 4; j++) { v[j] = w[perm[j]]; }
      static_cast<Tetrahedron *>(mesh.GetElement(e))->SetRefinementFlag(0);
      mesh.SetAttribute(e, 1 + e);
   }
   mesh.FinalizeTopology();
   mesh.Finalize(false, false);
   mesh.SetAttributes();
   return mesh;
}

/// Whether the faces have every relative orientation, counted over the
/// ranks: the second element of an interior face sees it in each of the three
/// orientations opposite to the first's (codes 1, 3, 5), and, on more than
/// one rank, the shared triangles have all six orientations with respect to
/// the local faces.
bool AllFaceOrientations(ParMesh &pmesh)
{
   int count[12] = {};
   Array<int> faces, ori;
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      pmesh.GetElementFaces(e, faces, ori);
      for (int o : ori) { if (o >= 0 && o < 6) { count[o]++; } }
   }
   for (int g = 1; g < pmesh.GetNGroups(); g++)
   {
      for (int i = 0; i < pmesh.GroupNTriangles(g); i++)
      {
         int face, o;
         pmesh.GroupTriangle(g, i, face, o);
         if (o >= 0 && o < 6) { count[6 + o]++; }
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, count, 12, MPI_INT, MPI_SUM, pmesh.GetComm());
   bool all = count[1] && count[3] && count[5];
   for (int o = 0; o < 6 && pmesh.GetNRanks() > 1; o++)
   {
      all = all && count[6 + o];
   }
   return all;
}

/// Rebalance(partition), checking that the input is consistent across the
/// ranks, so that a bad input is not taken for a failure of the rebalance,
/// and that the rebalance keeps the invariants above and the consistency.
/// (With discontinuous nodes, as on a periodic mesh, the coordinates of a
/// shared vertex can differ between its ranks: there, the check of the input
/// made by the rebalance itself applies.)
void RebalanceChecked(ParMesh &pmesh, const Array<int> &partition)
{
   const GridFunction *nodes = pmesh.GetNodes();
   const bool coords = !nodes || nodes->FESpace()->FEColl()->GetContType() !=
                       FiniteElementCollection::DISCONTINUOUS;
   if (coords) { REQUIRE(SharedEntitiesAgree(pmesh)); }
   const std::vector<long long> before = Invariants(pmesh);
   pmesh.Rebalance(partition);
   CHECK(Invariants(pmesh) == before);
   if (coords) { CHECK(SharedEntitiesAgree(pmesh)); }
}

bool SpacesConform(ParMesh &pmesh, Mesh &serial, int max_order = 3)
{
   const int dim = pmesh.Dimension();
   bool ok = true;
   // Periodic in x with period 1, for the periodic meshes.
   auto scalar = [](const Vector &x)
   {
      const real_t c = std::cos(2.0*M_PI*x(0));
      real_t s = 1.0 + c - 2.0*x(1)*x(1) + c*x(1);
      if (x.Size() > 2) { s += x(2)*c - x(2)*x(2)*x(2); }
      return s;
   };
   auto vector = [dim](const Vector &x, Vector &y)
   {
      const real_t c = std::cos(2.0*M_PI*x(0)), s = std::sin(2.0*M_PI*x(0));
      y.SetSize(dim);
      y(0) = 1.0 + x(1) - s*x(1);
      y(1) = 2.0 - c + x(1)*x(1);
      if (dim > 2) { y(2) = s*x(1) - x(2); }
   };
   FunctionCoefficient sc(scalar);
   VectorFunctionCoefficient vc(dim, vector);
   for (int p = 1; p <= max_order; p++)
   {
      H1_FECollection h1(p, dim);
      ND_FECollection nd(p, dim);
      RT_FECollection rt(p - 1, dim);
      L2_FECollection l2(p, dim);
      for (FiniteElementCollection *fec :
           std::vector<FiniteElementCollection *> {&h1, &nd, &rt, &l2})
      {
         ParFiniteElementSpace pfes(&pmesh, fec);
         FiniteElementSpace sfes(&serial, fec);
         const bool size_ok = (pfes.GlobalTrueVSize() == sfes.GetTrueVSize());
         CHECK(size_ok);
         ParGridFunction x(&pfes), y(&pfes);
         if (fec == &h1 || fec == &l2) { x.ProjectCoefficient(sc); }
         else { x.ProjectCoefficient(vc); }
         Vector tv(pfes.GetTrueVSize());
         x.ParallelProject(tv);
         y.Distribute(tv);
         y -= x;
         real_t err = y.Normlinf(), ref = x.Normlinf();
         MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_MAX, pmesh.GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &ref, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_MAX, pmesh.GetComm());
         CHECK(err <= 1e-12*ref);
         ok = ok && size_ok && err <= 1e-12*ref;
      }
   }
   return ok;
}

/// Solve -div(a grad u) = f with u = 0 on the exterior boundary, H1 of order
/// 2, and curl curl E + E = g, ND of order 2. Return the energies (f, u) and
/// (g, E).
std::pair<real_t, real_t> Solve(ParMesh &pmesh)
{
   const int dim = pmesh.Dimension();
   Vector attr_coef(pmesh.attributes.Max());
   for (int i = 0; i < attr_coef.Size(); i++) { attr_coef(i) = 1.0 + i; }
   PWConstCoefficient a(attr_coef);
   FunctionCoefficient f([](const Vector &x) { return 1.0 + x(0)*x(1); });
   VectorFunctionCoefficient g(dim, [](const Vector &x, Vector &y)
   {
      y = 1.0;
      y(0) += x(1);
   });
   real_t energy[2];
   {
      H1_FECollection fec(2, dim);
      ParFiniteElementSpace fes(&pmesh, &fec);
      Array<int> ess_bdr(pmesh.bdr_attributes.Max()), ess_tdofs;
      ess_bdr = 0;
      for (int b = 0; b < std::min(2*dim, ess_bdr.Size()); b++)
      {
         ess_bdr[b] = 1;
      }
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
      ParLinearForm b(&fes);
      b.AddDomainIntegrator(new DomainLFIntegrator(f));
      b.Assemble();
      ParBilinearForm form(&fes);
      form.AddDomainIntegrator(new DiffusionIntegrator(a));
      form.Assemble();
      ParGridFunction x(&fes);
      x = 0.0;
      OperatorPtr A;
      Vector B, X;
      form.FormLinearSystem(ess_tdofs, x, b, A, X, B);
      HypreBoomerAMG amg(*A.As<HypreParMatrix>());
      amg.SetPrintLevel(0);
      CGSolver cg(pmesh.GetComm());
      cg.SetRelTol(1e-13);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(amg);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      form.RecoverFEMSolution(X, b, x);
      energy[0] = b(x);
   }
   {
      ND_FECollection fec(2, dim);
      ParFiniteElementSpace fes(&pmesh, &fec);
      ParLinearForm b(&fes);
      b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(g));
      b.Assemble();
      ParBilinearForm form(&fes);
      ConstantCoefficient one(1.0);
      form.AddDomainIntegrator(new CurlCurlIntegrator(a));
      form.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      form.Assemble();
      ParGridFunction x(&fes);
      x = 0.0;
      Array<int> ess_tdofs;
      OperatorPtr A;
      Vector B, X;
      form.FormLinearSystem(ess_tdofs, x, b, A, X, B);
      HypreAMS ams(*A.As<HypreParMatrix>(), &fes);
      ams.SetPrintLevel(0);
      CGSolver cg(pmesh.GetComm());
      cg.SetRelTol(1e-13);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(ams);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      form.RecoverFEMSolution(X, b, x);
      energy[1] = b(x);
   }
   return {energy[0], energy[1]};
}

bool SameAttributeSets(const ParMesh &a, const Mesh &b)
{
   return a.attribute_sets.GetAttributeSetNames() ==
          b.attribute_sets.GetAttributeSetNames() &&
          a.bdr_attribute_sets.GetAttributeSetNames() ==
          b.bdr_attribute_sets.GetAttributeSetNames() &&
          a.attribute_sets.GetAttributeSet("left") ==
          b.attribute_sets.GetAttributeSet("left") &&
          a.bdr_attribute_sets.GetAttributeSet("ends") ==
          b.bdr_attribute_sets.GetAttributeSet("ends");
}

struct Case
{
   int dim, order;
   Kind kind;
   Ordering::Type ordering = Ordering::byVDIM;
};

std::vector<Case> Cases()
{
   return {{2, 1, Kind::STRAIGHT}, {2, 2, Kind::CURVED}, {2, 3, Kind::CURVED},
      {2, 2, Kind::PERIODIC}, {2, 3, Kind::PERIODIC, Ordering::byNODES},
      {2, 1, Kind::CRACKED}, {2, 1, Kind::UNIFORM}, {3, 1, Kind::STRAIGHT},
      {3, 1, Kind::CURVED}, {3, 2, Kind::CURVED},
      {3, 3, Kind::CURVED, Ordering::byNODES}, {3, 4, Kind::CURVED},
      {3, 2, Kind::PERIODIC}, {3, 3, Kind::PERIODIC, Ordering::byNODES},
      {3, 1, Kind::CRACKED}, {3, 1, Kind::UNIFORM}, {3, 2, Kind::REMARKED},
      {3, 1, Kind::ROTATED}};
}

/// The cases of Cases() for the longer tests: straight, curved, periodic and
/// cracked meshes in 2D and 3D, and in 3D marked again and rotated.
std::vector<Case> FewerCases()
{
   return {{2, 1, Kind::STRAIGHT}, {2, 3, Kind::CURVED},
      {2, 3, Kind::PERIODIC, Ordering::byNODES}, {2, 1, Kind::CRACKED},
      {3, 1, Kind::STRAIGHT}, {3, 2, Kind::CURVED},
      {3, 3, Kind::PERIODIC, Ordering::byNODES}, {3, 1, Kind::CRACKED},
      {3, 2, Kind::REMARKED}, {3, 1, Kind::ROTATED}};
}

std::string Name(const Case &c)
{
   const char *kinds[] = {"straight", "curved", "periodic", "cracked",
                          "uniformly refined", "marked again", "rotated"
                         };
   return std::to_string(c.dim) + "D " + kinds[int(c.kind)] + " order " +
          std::to_string(c.order) +
          ((c.ordering == Ordering::byNODES) ? " by nodes" : "");
}

/// A ParMesh whose description of its shared entities a test can corrupt, on
/// one rank, to make the parallel mesh inconsistent.
class CorruptibleParMesh : public ParMesh
{
public:
   using ParMesh::ParMesh;

   /// Rotate the vertices of a shared triangle: its refinement edge is no
   /// longer the one the other side gives it.
   bool RotateSharedTriangle()
   {
      if (shared_trias.Size() == 0) { return false; }
      int *v = shared_trias[0].v;
      std::rotate(v, v + 1, v + 3);
      return true;
   }

   /// Swap two shared edges in the list of a group: the other members list
   /// them in the other order.
   bool SwapSharedEdges()
   {
      for (int g = 0; g < group_sedge.Size(); g++)
      {
         if (group_sedge.RowSize(g) >= 2)
         {
            int *row = group_sedge.GetRow(g);
            std::swap(row[0], row[1]);
            return true;
         }
      }
      return false;
   }

   /// Reverse a shared edge: the other members give it the other direction.
   bool ReverseSharedEdge()
   {
      if (shared_edges.Size() == 0) { return false; }
      int *v = shared_edges[0]->GetVertices();
      std::swap(v[0], v[1]);
      return true;
   }

   /// Swap the first two vertices of a shared triangle: it keeps its
   /// refinement edge, but the other members give it the other orientation.
   bool FlipSharedTriangle()
   {
      if (shared_trias.Size() == 0) { return false; }
      std::swap(shared_trias[0].v[0], shared_trias[0].v[1]);
      return true;
   }

   /// Drop the last shared vertex of a group whose master is another rank
   /// from the list of this rank: the other members list one vertex more.
   bool DropSharedVertex()
   {
      int drop = -1;
      for (int g = 1; g < GetNGroups() && drop < 0; g++)
      {
         if (!gtopo.IAmMaster(g) && group_svert.RowSize(g-1) > 0)
         {
            drop = g-1;
         }
      }
      if (drop < 0) { return false; }
      Table svert;
      svert.MakeI(group_svert.Size());
      for (int r = 0; r < group_svert.Size(); r++)
      {
         svert.AddColumnsInRow(r, group_svert.RowSize(r) - (r == drop));
      }
      svert.MakeJ();
      for (int r = 0; r < group_svert.Size(); r++)
      {
         for (int j = 0; j < group_svert.RowSize(r) - (r == drop); j++)
         {
            svert.AddConnection(r, group_svert.GetRow(r)[j]);
         }
      }
      svert.ShiftUpI();
      group_svert.Swap(svert);
      return true;
   }

   /** Share the last @a n vertices among all the ranks, as one group with no
       edges and faces; for a mesh with no other shared entities (all the
       elements on one rank) whose last vertices, of no element, are at the
       same points on every rank. */
   void ShareLastVertices(int n)
   {
      const int nranks = GetNRanks();
      ListOfIntegerSets groups;
      IntegerSet group;
      int rank = GetMyRank();
      group.Recreate(1, &rank);
      groups.Insert(group);
      Array<int> all(nranks);
      for (int i = 0; i < nranks; i++) { all[i] = i; }
      if (nranks > 1)
      {
         group.Recreate(nranks, all.GetData());
         groups.Insert(group);
      }
      gtopo.Create(groups, 3971);
      const int ng = (nranks > 1) ? 1 : 0;
      group_svert.SetDims(ng, ng ? n : 0);
      svert_lvert.SetSize(ng ? n : 0);
      if (ng)
      {
         group_svert.GetI()[0] = 0;
         group_svert.GetI()[1] = n;
         for (int i = 0; i < n; i++)
         {
            group_svert.GetJ()[i] = i;
            svert_lvert[i] = GetNV() - n + i;
         }
      }
      for (Table *t : {&group_sedge, &group_stria, &group_squad})
      {
         t->SetDims(ng, 0);
         for (int i = 0; i <= ng; i++) { t->GetI()[i] = 0; }
      }
   }
};

/// A ParMesh whose exchange of rows after a rebalance, used by the transfer of
/// grid functions, can be called with rows of any size.
class RowExchangeParMesh : public ParMesh
{
public:
   using ParMesh::ParMesh;
   using ParMesh::ExchangeRebalanceRows;
};

} // namespace pmesh_rebalance

using namespace pmesh_rebalance;

TEST_CASE("ParMesh Rebalance conforming: same mesh as a direct distribution",
          "[Parallel], [ParMesh], [Rebalance]")
{
   for (const Case &cs : Cases())
   {
      Mesh serial = MakeMesh(cs.dim, cs.order, cs.kind, cs.ordering);
      const Counts serial_counts = SerialCounts(serial);
      ForEachComm([&](MPI_Comm comm)
      {
         int nranks;
         MPI_Comm_size(comm, &nranks);
         for (const auto &init : InitialRules())
         {
            for (const auto &target : TargetRules())
            {
               INFO(Name(cs) << ", " << nranks << " ranks, from " << init.first
                    << " to " << target.first);
               Array<int> p0 = SerialPartition(serial, init.second, nranks);
               Array<int> p1 = SerialPartition(serial, target.second, nranks);
               ParMesh pmesh(comm, serial, p0.GetData());
               Array<int> partition = LocalPartition(pmesh, target.second);
               RebalanceChecked(pmesh, partition);
               ParMesh ref(comm, serial, p1.GetData());

               CHECK(AllOf(comm, ElementDescriptors(pmesh) ==
                           ElementDescriptors(ref)));
               // The vertex coordinates of a periodic mesh are averages of the
               // nodes of the local elements on the two sides of the periodic
               // boundary: there, only the checks of the finite element spaces
               // and of the refinement apply.
               if (cs.kind != Kind::PERIODIC)
               {
                  CHECK(AllOf(comm, SameBoundary(pmesh, ref)));
                  CHECK(AllOf(comm, SharedDescriptors(pmesh) ==
                              SharedDescriptors(ref)));
                  CHECK(SharedEntitiesAgree(pmesh));
               }
               CHECK(ParallelCounts(pmesh) == serial_counts);
               CHECK(SameAttributeSets(pmesh, serial));
               CHECK(pmesh.GetLastOperation() == Mesh::REBALANCE);
            }
         }
      }, false);
   }
}

TEST_CASE("ParMesh Rebalance conforming: conforming spaces and solutions",
          "[Parallel], [ParMesh], [Rebalance]")
{
   for (const Case &cs : FewerCases())
   {
      Mesh serial = MakeMesh(cs.dim, cs.order, cs.kind, cs.ordering);
      ForEachComm([&](MPI_Comm comm)
      {
         int nranks;
         MPI_Comm_size(comm, &nranks);
         INFO(Name(cs) << ", " << nranks << " ranks");
         const Rule &init = InitialRules()[0].second;
         const Rule &target = TargetRules()[2].second;
         Array<int> p0 = SerialPartition(serial, init, nranks);
         Array<int> p1 = SerialPartition(serial, target, nranks);
         ParMesh pmesh(comm, serial, p0.GetData());
         Array<int> partition = LocalPartition(pmesh, target);
         RebalanceChecked(pmesh, partition);
         CHECK(SpacesConform(pmesh, serial));
         if (cs.kind != Kind::PERIODIC)
         {
            ParMesh ref(comm, serial, p1.GetData());
            const auto e = Solve(pmesh), e_ref = Solve(ref);
            CHECK(std::abs(e.first - e_ref.first) <=
                  1e-9*std::abs(e_ref.first));
            CHECK(std::abs(e.second - e_ref.second) <=
                  1e-9*std::abs(e_ref.second));
         }
      }, false);
   }
}

TEST_CASE("ParMesh Rebalance conforming: local refinement afterwards",
          "[Parallel], [ParMesh], [Rebalance]")
{
   for (const Case &cs : FewerCases())
   {
      // Local refinement needs marked tetrahedra and boundary triangles whose
      // refinement edge is the marked edge of their face.
      if (cs.kind == Kind::UNIFORM || cs.kind == Kind::ROTATED) { continue; }
      ForEachComm([&](MPI_Comm comm)
      {
         int nranks;
         MPI_Comm_size(comm, &nranks);
         INFO(Name(cs) << ", " << nranks << " ranks");
         Mesh serial = MakeMesh(cs.dim, cs.order, cs.kind, cs.ordering);
         Array<int> p0 = SerialPartition(serial, InitialRules()[0].second,
                                         nranks);
         ParMesh pmesh(comm, serial, p0.GetData());
         // Rebalance, refine near a point, three times; the same refinements
         // of the serial mesh give the reference.
         for (int round = 0; round < 3; round++)
         {
            const Rule &target = TargetRules()[round % 3].second;
            Array<int> partition = LocalPartition(pmesh, target);
            RebalanceChecked(pmesh, partition);
            Vector point(cs.dim);
            point = 0.25 + 0.2*round;
            auto Mark = [&](Mesh &mesh)
            {
               Array<int> marked;
               Vector c;
               for (int e = 0; e < mesh.GetNE(); e++)
               {
                  mesh.GetElementCenter(e, c);
                  if (c.DistanceTo(point) < 0.3) { marked.Append(e); }
               }
               return marked;
            };
            pmesh.GeneralRefinement(Mark(pmesh), 0);
            serial.GeneralRefinement(Mark(serial), 0);
            INFO("round " << round);
            CHECK(ParallelCounts(pmesh) == SerialCounts(serial));
            if (cs.kind != Kind::PERIODIC)
            {
               CHECK(SharedEntitiesAgree(pmesh));
            }
         }
         CHECK(SpacesConform(pmesh, serial, 2));
      }, false);
   }
}

TEST_CASE("ParMesh Rebalance conforming: round trip",
          "[Parallel], [ParMesh], [Rebalance]")
{
   for (const Case &cs : Cases())
   {
      Mesh serial = MakeMesh(cs.dim, cs.order, cs.kind, cs.ordering);
      ForEachComm([&](MPI_Comm comm)
      {
         int nranks;
         MPI_Comm_size(comm, &nranks);
         INFO(Name(cs) << ", " << nranks << " ranks");
         const Rule &init = InitialRules()[1].second;
         Array<int> p0 = SerialPartition(serial, init, nranks);
         ParMesh pmesh(comm, serial, p0.GetData());
         const Descriptors elems = ElementDescriptors(pmesh);
         const Descriptors shared = SharedDescriptors(pmesh);
         {
            // Rebalancing to the same partition keeps the mesh. In 2D, it can
            // move an interior boundary edge on a shared edge to the other
            // side (see SameBoundary); not in 3D.
            Array<int> same(pmesh.GetNE());
            same = pmesh.GetMyRank();
            const Descriptors bdr0 = BoundaryDescriptors(pmesh);
            RebalanceChecked(pmesh, same);
            CHECK(AllOf(comm, ElementDescriptors(pmesh) == elems));
            if (cs.kind != Kind::PERIODIC)
            {
               CHECK(AllOf(comm, SharedDescriptors(pmesh) == shared));
            }
            if (cs.dim == 3)
            {
               CHECK(AllOf(comm, BoundaryDescriptors(pmesh) == bdr0));
            }
         }
         const Descriptors bdr = BoundaryDescriptors(pmesh);
         for (const Rule *rule :
              {&TargetRules()[2].second, &TargetRules()[1].second, &init})
         {
            Array<int> partition = LocalPartition(pmesh, *rule);
            Array<int> old_rank, old_index;
            const int ne_old = pmesh.GetNE();
            Array<int> rank(ne_old), index(ne_old);
            rank = pmesh.GetMyRank();
            for (int e = 0; e < ne_old; e++) { index[e] = e; }
            const Descriptors located = Located(pmesh, rank, index);
            RebalanceChecked(pmesh, partition);
            pmesh.GetRebalanceOldElements(old_rank, old_index);
            REQUIRE(old_rank.Size() == pmesh.GetNE());
            REQUIRE(old_index.Size() == pmesh.GetNE());
            // The previous rank and index of each element are exact.
            CHECK(AllOf(comm, Located(pmesh, old_rank, old_index) == located));
            bool ok = true;
            for (int i = 1; i < old_rank.Size(); i++)
            {
               // in the order of the old ranks, then of the old indices
               ok = ok && (old_rank[i-1] < old_rank[i] ||
                           (old_rank[i-1] == old_rank[i] &&
                            old_index[i-1] < old_index[i]));
            }
            CHECK(ok);
            CHECK(pmesh.ReduceInt(ne_old) == pmesh.GetGlobalNE());
         }
         CHECK(AllOf(comm, ElementDescriptors(pmesh) == elems));
         if (cs.kind != Kind::PERIODIC)
         {
            CHECK(AllOf(comm, BoundaryDescriptors(pmesh) == bdr));
            CHECK(AllOf(comm, SharedDescriptors(pmesh) == shared));
            CHECK(SharedEntitiesAgree(pmesh));
         }
      });
   }
}

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("ParMesh Rebalance conforming: old elements forgotten by Load",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // GetRebalanceOldElements() answers for the last rebalance of the mesh, not
   // after Load(), which starts the sequence of the mesh again: a refinement
   // of the loaded mesh brings it back to its value after the rebalance.
   Mesh serial = Mesh::MakeCartesian2D(4, 4, Element::TRIANGLE);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   Array<int> partition(pmesh.GetNE()), old_rank, old_index;
   partition = (pmesh.GetMyRank() + 1) % pmesh.GetNRanks();
   RebalanceChecked(pmesh, partition);
   const long sequence = pmesh.GetSequence();
   pmesh.GetRebalanceOldElements(old_rank, old_index);
   CHECK(old_rank.Size() == pmesh.GetNE());
   std::stringstream saved;
   pmesh.ParPrint(saved);
   pmesh.Load(saved, 1, 0);
   CHECK_THROWS_AS(pmesh.GetRebalanceOldElements(old_rank, old_index),
                   ErrorException);
   while (pmesh.GetSequence() < sequence) { pmesh.UniformRefinement(); }
   REQUIRE(pmesh.GetSequence() == sequence);
   CHECK_THROWS_AS(pmesh.GetRebalanceOldElements(old_rank, old_index),
                   ErrorException);
}
#endif

TEST_CASE("ParMesh Rebalance conforming: node values moved exactly",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // The nodes move with the transfer operator of their space: each element
   // keeps its node values exactly (as numbers). Random perturbations make
   // them use every bit.
   Mesh serial = MakeMesh(3, 3, Kind::CURVED);
   GridFunction &nodes = *serial.GetNodes();
   uint64_t state = 12345;
   for (int i = 0; i < nodes.Size(); i++)
   {
      state = state*6364136223846793005ull + 1442695040888963407ull;
      nodes(i) += 1e-4*(static_cast<real_t>(state >> 11)*0x1.0p-53 - 0.5);
   }
   ForEachComm([&](MPI_Comm comm)
   {
      int nranks;
      MPI_Comm_size(comm, &nranks);
      INFO(nranks << " ranks");
      const Rule &init = InitialRules()[1].second;
      Array<int> p0 = SerialPartition(serial, init, nranks);
      ParMesh pmesh(comm, serial, p0.GetData());
      const Descriptors elems = ElementDescriptors(pmesh);
      for (const Rule *rule : {&TargetRules()[2].second, &init})
      {
         Array<int> partition = LocalPartition(pmesh, *rule);
         RebalanceChecked(pmesh, partition);
      }
      CHECK(AllOf(comm, ElementDescriptors(pmesh) == elems));
   });
}

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("ParMesh Rebalance conforming: unsupported meshes",
          "[Parallel], [ParMesh], [Rebalance]")
{
   Mesh quads = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   ParMesh pmesh(MPI_COMM_WORLD, quads);
   Array<int> partition(pmesh.GetNE());
   partition = 0;
   REQUIRE_THROWS_AS(pmesh.Rebalance(partition), ErrorException);
   REQUIRE_THROWS_AS(pmesh.Rebalance(), ErrorException);

   Mesh tris = Mesh::MakeCartesian2D(4, 4, Element::TRIANGLE);
   ParMesh ptris(MPI_COMM_WORLD, tris);
   partition.SetSize(ptris.GetNE());
   partition = Mpi::WorldSize();
   REQUIRE_THROWS_AS(ptris.Rebalance(partition), ErrorException);
}
#endif

TEST_CASE("ParMesh Rebalance conforming: caches, DG and printing",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // The caches of the mesh are filled before each rebalance: face neighbor
   // data, element and face tables, global element numbers and geometric
   // factors. After it, a space updated with Update(false), the faces between
   // the ranks, a DG operator, the serial mesh and the printed mesh are those
   // of the same mesh.
   for (int dim : {2, 3})
   {
      for (bool discont : {false, true})
      {
         Mesh serial = (dim == 2) ?
                       Mesh::MakeCartesian2D(4, 3, Element::TRIANGLE) :
                       Mesh::MakeCartesian3D(3, 2, 2, Element::TETRAHEDRON);
         for (int e = 0; e < serial.GetNE(); e++)
         {
            serial.SetAttribute(e, e + 1);
         }
         serial.SetAttributes();
         serial.SetCurvature(3, discont, dim, Ordering::byNODES);
         serial.Transform(Deform);
         ForEachComm([&](MPI_Comm comm)
         {
            int nranks;
            MPI_Comm_size(comm, &nranks);
            INFO(dim << "D, " << (discont ? "L2" : "H1") << " nodes, "
                 << nranks << " ranks");
            Array<int> p0 = SerialPartition(serial, InitialRules()[1].second,
                                            nranks);
            ParMesh pmesh(comm, serial, p0.GetData());
            const real_t energy = DGEnergy(pmesh);
            const GridFunction *nodes = pmesh.GetNodes();
            const FiniteElementSpace *nodes_fes = nodes->FESpace();
            H1_FECollection h1(3, dim);
            ParFiniteElementSpace fes(&pmesh, &h1);
            ParGridFunction u(&fes);
            // In the space: a linear function of the coordinates, which are in
            // the nodal space of the same order.
            FunctionCoefficient f([](const Vector &x)
            {
               return x(0) + 2.0*x(1) + ((x.Size() == 3) ? 3.0*x(2) : 0.0);
            });
            for (int round = 0; round < 3; round++)
            {
               INFO("round " << round);
               pmesh.ExchangeFaceNbrData();
               pmesh.ElementToElementTable();
               delete pmesh.GetFaceToElementTable();
               pmesh.GetNFbyType(FaceType::Interior);
               pmesh.GetGlobalElementNum(0); // collective
               if (pmesh.GetNE())
               {
                  pmesh.GetGeometricFactors(
                     IntRules.Get(pmesh.GetTypicalElementGeometry(), 4),
                     GeometricFactors::DETERMINANTS);
               }
               const long sequence = pmesh.GetSequence();
               Array<int> partition(pmesh.GetNE());
               partition = nranks - 1;
               if (round < 2)
               {
                  const Rule &rule = TargetRules()[round].second;
                  partition = LocalPartition(pmesh, rule);
               }
               RebalanceChecked(pmesh, partition);
               CHECK(pmesh.GetSequence() == sequence + 1);
               CHECK(pmesh.GetNodes() == nodes);
               CHECK(pmesh.GetNodes()->FESpace() == nodes_fes);
               fes.Update(false);
               u.Update();
               u.ProjectCoefficient(f);
               CHECK(u.ComputeL2Error(f) < 1e-11);
               CHECK(SharedFacePointsAgree(pmesh));
               CHECK(std::abs(DGEnergy(pmesh) - energy) <=
                     1e-10*std::abs(energy));

               Mesh gathered = pmesh.GetSerialMesh(0);
               std::stringstream one;
               one.precision(17);
               pmesh.PrintAsOne(one);
               int rank;
               MPI_Comm_rank(comm, &rank);
               if (rank == 0)
               {
                  ParMesh self(MPI_COMM_SELF, gathered);
                  CHECK(std::abs(DGEnergy(self) - energy) <=
                        1e-10*std::abs(energy));
                  Mesh printed(one);
                  CHECK(printed.GetNE() == serial.GetNE());
               }
               // Reading a ParMesh needs elements on every rank.
               if (AllOf(comm, pmesh.GetNE() > 0))
               {
                  std::stringstream stream;
                  stream.precision(17);
                  pmesh.ParPrint(stream);
                  ParMesh read(comm, stream, false);
                  CHECK(std::abs(DGEnergy(read) - energy) <=
                        1e-10*std::abs(energy));
               }
            }
         });
      }
   }
}

TEST_CASE("ParMesh Rebalance conforming: after uniform and local refinement",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // Uniform refinement of the parallel mesh, which leaves the tetrahedra
   // unmarked, marking again (Finalize(true)), then local refinements and
   // rebalances: the mesh is consistent across the ranks at every step, and
   // the spaces conform.
   for (int dim : {2, 3})
   {
      Mesh serial = MakeMesh(dim, 1, Kind::STRAIGHT);
      ForEachComm([&](MPI_Comm comm)
      {
         int nranks;
         MPI_Comm_size(comm, &nranks);
         INFO(dim << "D, " << nranks << " ranks");
         Array<int> p0 = SerialPartition(serial, InitialRules()[0].second,
                                         nranks);
         ParMesh pmesh(comm, serial, p0.GetData());
         pmesh.UniformRefinement();
         pmesh.Finalize(true);
         REQUIRE(SharedEntitiesAgree(pmesh));
         for (int round = 0; round < 2; round++)
         {
            INFO("round " << round);
            Array<int> marked;
            Vector c;
            for (int e = 0; e < pmesh.GetNE(); e++)
            {
               pmesh.GetElementCenter(e, c);
               if (c(0) + c(1) < 0.6 + 0.4*round) { marked.Append(e); }
            }
            pmesh.GeneralRefinement(marked, 0);
            REQUIRE(SharedEntitiesAgree(pmesh));
            Array<int> partition =
               LocalPartition(pmesh, TargetRules()[round].second);
            RebalanceChecked(pmesh, partition);
            CHECK(ConformityResidual(pmesh) < 1e-11);
         }
      }, false);
   }
}

/// The spaces of SpaceCases() on @a pmesh, and on each two grid functions:
/// the projections of Smooth() and of Poly() of degree @a p.
struct Fields
{
   std::vector<std::unique_ptr<FiniteElementCollection>> fec;
   std::vector<std::unique_ptr<ParFiniteElementSpace>> fes;
   std::vector<std::unique_ptr<ParGridFunction>> x;

   Fields(ParMesh &pmesh, const std::vector<SpaceCase> &cases, int p)
   {
      const int dim = pmesh.Dimension();
      for (const SpaceCase &c : cases)
      {
         fec.emplace_back(c.fec(dim));
         fes.emplace_back(new ParFiniteElementSpace(&pmesh, fec.back().get(),
                                                    c.vdim ? c.vdim : dim,
                                                    c.ordering));
         for (int k = 0; k < 2; k++)
         {
            x.emplace_back(new ParGridFunction(fes.back().get()));
            Project(*x.back(), k ? p : -1);
         }
      }
   }

   /// Project Smooth() (degree < 0) or Poly() of that degree.
   static void Project(GridFunction &x, int degree)
   {
      const FiniteElementSpace &fes = *x.FESpace();
      const int n = (fes.GetVDim() > 1 || fes.FEColl()->GetRangeType(
                        fes.GetMesh()->Dimension()) == FiniteElement::VECTOR) ?
                    fes.GetMesh()->SpaceDimension() : 1;
      VectorFunctionCoefficient f(n, [degree](const Vector &y, Vector &v)
      {
         for (int k = 0; k < v.Size(); k++)
         {
            v(k) = (degree < 0) ? Smooth(y, k) : Poly(y, k, degree);
         }
      });
      FunctionCoefficient g([degree](const Vector &y)
      { return (degree < 0) ? Smooth(y, 0) : Poly(y, 0, degree); });
      if (n > 1) { x.ProjectCoefficient(f); }
      else { x.ProjectCoefficient(g); }
   }

   void Update()
   {
      for (auto &f : fes) { f->Update(); }
      for (auto &y : x) { y->Update(); }
   }
};

/// The L2 norm of @a x (computed on its own mesh).
real_t Norm(GridFunction &x)
{
   const FiniteElementSpace &fes = *x.FESpace();
   const int n = (fes.FEColl()->GetRangeType(fes.GetMesh()->Dimension()) ==
                  FiniteElement::VECTOR) ? fes.GetMesh()->SpaceDimension() :
                 fes.GetVDim();
   Vector zero(n);
   zero = 0.0;
   VectorConstantCoefficient z(zero);
   ConstantCoefficient zs(0.0);
   return (n > 1) ? x.ComputeL2Error(z) : x.ComputeL2Error(zs);
}

TEST_CASE("ParMesh Rebalance conforming: grid functions transferred",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // After a rebalance, Update() of each space and then of each grid function
   // on it gives the same field, redistributed: at the points of every
   // element, matched through its previous location, the values are those
   // of the same element before, and the norms are those of the serial mesh.
   // The spaces of TransferSpaceCases(), scalar and vector with both
   // orderings, several on one mesh; curved meshes and, in 3D, a mesh whose
   // faces have every orientation; four rebalances in a row, the last to a
   // single rank. (The other orders: the tests of projections and round
   // trips; all the coefficients of fields: the test of random fields.)
   const std::vector<SpaceCase> cases = TransferSpaceCases();
   for (int dim : {2, 3})
   {
      std::vector<std::pair<std::string, Mesh>> meshes;
      meshes.emplace_back("curved", MakeMesh(dim, 2, Kind::CURVED));
      if (dim == 3) { meshes.emplace_back("permuted", PermutedTetMesh()); }
      for (auto &m : meshes)
      {
         Mesh &serial = m.second;
         std::vector<real_t> ref;
         {
            for (const SpaceCase &c : cases)
            {
               std::unique_ptr<FiniteElementCollection> fec(c.fec(dim));
               FiniteElementSpace fes(&serial, fec.get(), c.vdim ? c.vdim : dim,
                                      c.ordering);
               GridFunction x(&fes);
               for (int k = 0; k < 2; k++)
               {
                  Fields::Project(x, k ? 2 : -1);
                  ref.push_back(Norm(x));
               }
            }
         }
         ForEachComm([&](MPI_Comm comm)
         {
            int nranks, rank;
            MPI_Comm_size(comm, &nranks);
            MPI_Comm_rank(comm, &rank);
            Array<int> p0 = SerialPartition(serial, InitialRules()[1].second,
                                            nranks);
            ParMesh pmesh(comm, serial, p0.GetData());
            Fields f(pmesh, cases, 2);
            for (std::size_t r = 0; r < TargetRules().size(); r++)
            {
               INFO(dim << "D " << m.first << ", " << nranks << " ranks, to "
                    << TargetRules()[r].first);
               Array<int> rank_(pmesh.GetNE()), index(pmesh.GetNE());
               rank_ = rank;
               for (int e = 0; e < pmesh.GetNE(); e++) { index[e] = e; }
               std::vector<Descriptors> before;
               for (auto &x : f.x)
               {
                  before.push_back(FieldValues(*x, rank_, index));
               }
               Array<int> partition =
                  LocalPartition(pmesh, TargetRules()[r].second);
               RebalanceChecked(pmesh, partition);
               // (After the last rule, all elements are on one rank.)
               if (m.first == "permuted" && r + 1 < TargetRules().size())
               {
                  CHECK(AllFaceOrientations(pmesh));
               }
               f.Update();
               Array<int> old_rank, old_index;
               pmesh.GetRebalanceOldElements(old_rank, old_index);
               for (std::size_t i = 0; i < f.x.size(); i++)
               {
                  INFO(cases[i/2].name << (i % 2 ? ", polynomial" :
                                           ", smooth"));
                  CHECK(AllOf(comm, Close(FieldValues(*f.x[i], old_rank,
                                                      old_index), before[i],
                                          1e-12)));
                  const real_t norm = Norm(*f.x[i]);
                  CHECK(std::abs(norm - ref[i]) <= 1e-11*std::abs(ref[i]));
               }
            }
         }, false);
      }
   }
}

TEST_CASE("ParMesh Rebalance conforming: grid functions equal new projections",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // On a straight mesh, a polynomial of the order of the space is in it:
   // after the transfer, its DOF values are those of a projection on the new
   // mesh, so every DOF, with its sign, lands where it belongs.
   const std::vector<SpaceCase> cases = SpaceCases();
   for (int dim : {2, 3})
   {
      Mesh serial = (dim == 3) ? PermutedTetMesh() :
                    MakeMesh(dim, 1, Kind::STRAIGHT);
      ForEachComm([&](MPI_Comm comm)
      {
         int nranks;
         MPI_Comm_size(comm, &nranks);
         Array<int> p0 = SerialPartition(serial, InitialRules()[0].second,
                                         nranks);
         ParMesh pmesh(comm, serial, p0.GetData());
         for (std::size_t r = 0; r < TargetRules().size(); r++)
         {
            INFO(dim << "D, " << nranks << " ranks, to "
                 << TargetRules()[r].first);
            std::vector<std::unique_ptr<FiniteElementCollection>> fec;
            std::vector<std::unique_ptr<ParFiniteElementSpace>> fes;
            std::vector<std::unique_ptr<ParGridFunction>> x;
            std::vector<int> degree;
            for (const SpaceCase &c : cases)
            {
               fec.emplace_back(c.fec(dim));
               const int vdim = c.vdim ? c.vdim : dim;
               fes.emplace_back(new ParFiniteElementSpace(
                                   &pmesh, fec.back().get(), vdim, c.ordering));
               // The polynomial degree the space contains.
               const int p = fec.back()->GetOrder();
               const bool nd =
                  dynamic_cast<ND_FECollection *>(fec.back().get());
               degree.push_back(nd ? p - 1 : p);
               x.emplace_back(new ParGridFunction(fes.back().get()));
               Fields::Project(*x.back(), degree.back());
            }
            Array<int> partition =
               LocalPartition(pmesh, TargetRules()[r].second);
            RebalanceChecked(pmesh, partition);
            for (auto &s : fes) { s->Update(); }
            for (std::size_t i = 0; i < x.size(); i++)
            {
               INFO(cases[i].name);
               x[i]->Update();
               ParGridFunction fresh(fes[i].get());
               Fields::Project(fresh, degree[i]);
               fresh -= *x[i];
               real_t diff = fresh.Normlinf();
               MPI_Allreduce(MPI_IN_PLACE, &diff, 1,
                             MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
               CHECK(diff <= 1e-12);
            }
         }
      });
   }
}

TEST_CASE("ParMesh Rebalance conforming: grid functions round trip",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // There and back: on each rank, every element has again its DOF values in
   // the element's own (reference) layout, exactly.
   const std::vector<SpaceCase> cases = SpaceCases();
   for (int dim : {2, 3})
   {
      Mesh serial = (dim == 3) ? PermutedTetMesh() :
                    MakeMesh(dim, 2, Kind::CURVED);
      for (int e = 0; e < serial.GetNE(); e++)
      {
         serial.SetAttribute(e, 1 + e);
      }
      serial.SetAttributes();
      ForEachComm([&](MPI_Comm comm)
      {
         int nranks;
         MPI_Comm_size(comm, &nranks);
         INFO(dim << "D, " << nranks << " ranks");
         const Rule &init = InitialRules()[1].second;
         Array<int> p0 = SerialPartition(serial, init, nranks);
         ParMesh pmesh(comm, serial, p0.GetData());
         Fields f(pmesh, cases, 3);
         // The copies of each shared DOF agree (true DOFs), so that the
         // transfer is exact.
         for (auto &x : f.x)
         {
            Vector tv;
            x->GetTrueDofs(tv);
            x->SetFromTrueDofs(tv);
         }
         // The element values, by element (its attribute).
         auto Values = [&](std::size_t i)
         {
            std::map<int, std::vector<real_t>> v;
            Vector dv;
            for (int e = 0; e < pmesh.GetNE(); e++)
            {
               f.x[i]->GetElementDofValues(e, dv);
               v[pmesh.GetAttribute(e)] = std::vector<real_t>(dv.begin(),
                                                              dv.end());
            }
            return v;
         };
         std::vector<std::map<int, std::vector<real_t>>> before;
         for (std::size_t i = 0; i < f.x.size(); i++)
         {
            before.push_back(Values(i));
         }
         for (const Rule *rule : {&TargetRules()[2].second, &init})
         {
            Array<int> partition = LocalPartition(pmesh, *rule);
            RebalanceChecked(pmesh, partition);
            f.Update();
         }
         for (std::size_t i = 0; i < f.x.size(); i++)
         {
            INFO(cases[i/2].name);
            CHECK(AllOf(comm, Values(i) == before[i]));
         }
      });
   }
}

TEST_CASE("ParMesh Rebalance conforming: isolated vertices kept",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // A vertex of no element stays on its rank, with its coordinates and node
   // values, and keeps its DOFs in the H1 spaces, with their values.
   for (int dim : {2, 3})
   {
      Mesh serial = (dim == 2) ?
                    Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE) :
                    Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
      ForEachComm([&](MPI_Comm comm)
      {
         int nranks, rank;
         MPI_Comm_size(comm, &nranks);
         MPI_Comm_rank(comm, &rank);
         INFO(dim << "D, " << nranks << " ranks");
         Array<int> init(serial.GetNE());
         init = 0;
         ParMesh pmesh(comm, serial, init.GetData());
         const real_t x[3] = {real_t(2 + rank), 3.0, 4.0};
         const int iv = pmesh.AddVertex(x);
         pmesh.SetCurvature(2);
         GridFunction &nodes = *pmesh.GetNodes();
         Array<int> vdofs;
         nodes.FESpace()->GetVertexVDofs(iv, vdofs);
         for (int d = 0; d < dim; d++) { nodes(vdofs[d]) = x[d]; }
         H1_FECollection h1(2, dim);
         ParFiniteElementSpace fes(&pmesh, &h1);
         ParGridFunction u(&fes);
         FunctionCoefficient f([](const Vector &y) { return y(0) + 2.0*y(1); });
         u.ProjectCoefficient(f);
         u(iv) = 7.0 + rank;
         for (int round = 0; round < 2; round++)
         {
            Array<int> partition(pmesh.GetNE());
            for (int e = 0; e < pmesh.GetNE(); e++)
            {
               partition[e] = (round == 0) ? nranks - 1 : e % nranks;
            }
            RebalanceChecked(pmesh, partition);
            fes.Update();
            u.Update();
         }
         CHECK(u.ComputeL2Error(f) < 1e-12);
         // Exactly one vertex of no element on each rank, the one it had.
         Array<bool> used(pmesh.GetNV());
         used = false;
         for (int e = 0; e < pmesh.GetNE(); e++)
         {
            const Element *el = pmesh.GetElement(e);
            for (int j = 0; j < el->GetNVertices(); j++)
            {
               used[el->GetVertices()[j]] = true;
            }
         }
         int n_isolated = 0;
         bool same = true;
         for (int i = 0; i < pmesh.GetNV(); i++)
         {
            if (used[i]) { continue; }
            n_isolated++;
            same = same && u(i) == 7.0 + rank;
            pmesh.GetNodes()->FESpace()->GetVertexVDofs(i, vdofs);
            for (int d = 0; d < dim; d++)
            {
               same = same && pmesh.GetVertex(i)[d] == x[d] &&
                      (*pmesh.GetNodes())(vdofs[d]) == x[d];
            }
         }
         CHECK(AllOf(comm, n_isolated == 1 && same));
      });
   }
}

/// Set @a x from pseudo-random true DOFs in [-1, 1), functions of their global
/// numbers and of @a seed: any field of the space, whose copies of each shared
/// DOF agree.
void RandomTrueDofs(ParGridFunction &x, int seed)
{
   ParFiniteElementSpace &fes = *x.ParFESpace();
   Vector t(fes.GetTrueVSize());
   const std::uint64_t first = fes.GetMyTDofOffset();
   for (int i = 0; i < t.Size(); i++)
   {
      std::uint64_t h = (first + i + 1)*0x9E3779B97F4A7C15ull +
                        std::uint64_t(seed)*0xBF58476D1CE4E5B9ull;
      h ^= h >> 31;
      h *= 0x94D049BB133111EBull;
      h ^= h >> 29;
      t(i) = real_t(h >> 11)/real_t(1ull << 52) - 1.0;
   }
   x.SetFromTrueDofs(t);
}

/// Whether x = P R x: the copies of each shared DOF of @a x are equal, or
/// differ by at most @a tol.
bool SingleValued(ParGridFunction &x, real_t tol = 0.0)
{
   ParFiniteElementSpace &fes = *x.ParFESpace();
   Vector t(fes.GetTrueVSize());
   x.GetTrueDofs(t);
   ParGridFunction y(&fes);
   y.SetFromTrueDofs(t);
   bool same = (y.Size() == x.Size());
   for (int i = 0; same && i < y.Size(); i++)
   {
      same = (std::abs(y(i) - x(i)) <= tol);
   }
   return AllOf(fes.GetComm(), same);
}

/// The spaces of the test with random fields: H1 of orders 1 and 4, ND of
/// orders 1, 2 and 4, L2 of orders 0 and 4, RT of orders 0, 1 and 4, H1 and L2
/// spaces with vdim 3 in both orderings, and RT and ND spaces with vdim 2 and
/// 3, set through their scalar components.
std::vector<SpaceCase> RandomSpaceCases()
{
   std::vector<SpaceCase> c;
   for (int p : {1, 4})
   {
      c.push_back({"H1_" + std::to_string(p), [p](int d)
      { return new H1_FECollection(p, d); }, 1, Ordering::byNODES});
   }
   for (int p : {1, 2, 4})
   {
      c.push_back({"ND_" + std::to_string(p), [p](int d)
      { return new ND_FECollection(p, d); }, 1, Ordering::byNODES});
   }
   for (int p : {0, 4})
   {
      c.push_back({"L2_" + std::to_string(p), [p](int d)
      { return new L2_FECollection(p, d); }, 1, Ordering::byNODES});
   }
   for (int p : {0, 1, 4})
   {
      c.push_back({"RT_" + std::to_string(p), [p](int d)
      { return new RT_FECollection(p, d); }, 1, Ordering::byNODES});
   }
   c.push_back({"H1_2 x 3 by nodes", [](int d)
   { return new H1_FECollection(2, d); }, 3, Ordering::byNODES});
   c.push_back({"L2_2 x 3 by vdim", [](int d)
   { return new L2_FECollection(2, d); }, 3, Ordering::byVDIM});
   c.push_back({"RT_1 x 2 by vdim", [](int d)
   { return new RT_FECollection(1, d); }, 2, Ordering::byVDIM, true});
   c.push_back({"ND_2 x 3 by nodes", [](int d)
   { return new ND_FECollection(2, d); }, 3, Ordering::byNODES, true});
   return c;
}

TEST_CASE("ParMesh Rebalance conforming: random fields transferred",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // Fields set from random true DOFs, so any field of their space, in the
   // spaces of RandomSpaceCases() (those with vdim > 1 of ND and RT from
   // random scalar fields, one per component). After each of four rebalances
   // in a row (from a distribution with empty ranks; the last to a single
   // rank), and Update() of the spaces and the fields: the reference
   // coefficients of every element, matched through its previous location,
   // are those before, and x = P R x (for ND and RT with vdim > 1, for each
   // component, which is also the scalar field transferred). A fifth
   // rebalance back to the first distribution gives each element (an
   // attribute each) its first coefficients. Meshes: 2D straight and curved;
   // in 3D, tetrahedra whose faces have every orientation and no refinement
   // flags, marked tetrahedra, and marked tetrahedra of which those of one
   // rank are then unmarked; cracked meshes in 2D and 3D. On all the ranks,
   // and on all the ranks in the reverse order. Last, on the marked straight
   // meshes, a local refinement keeps the norm of each field (vdim 1 and H1,
   // L2), and x = P R x up to round-off.
   const std::vector<SpaceCase> cases = RandomSpaceCases();
   struct M
   {
      std::string name;
      int dim;
      std::function<Mesh()> make;
      bool refine, unmark_last_rank;
   };
   const std::vector<M> meshes =
   {
      {
         "2D straight", 2, [] { return MakeMesh(2, 1, Kind::STRAIGHT); },
         true, false
      },
      {
         "2D curved", 2, [] { return MakeMesh(2, 2, Kind::CURVED); },
         false, false
      },
      {
         "3D, every orientation, no flags", 3,
         [] { return PermutedTetMesh(); }, false, false
      },
      {
         "3D marked", 3, [] { return MakeMesh(3, 1, Kind::STRAIGHT); },
         true, false
      },
      {
         // The ParMesh constructor orients each shared triangle from the
         // marking of its tetrahedron: the flags are cleared afterwards.
         "3D, the tetrahedra of the last rank unmarked", 3,
         [] { return MakeMesh(3, 1, Kind::STRAIGHT); }, false, true
      },
      {
         "2D cracked", 2, [] { return MakeMesh(2, 1, Kind::CRACKED); },
         false, false
      },
      {
         "3D cracked", 3, [] { return MakeMesh(3, 1, Kind::CRACKED); },
         false, false
      }
   };
   for (const M &m : meshes)
   {
      Mesh serial = m.make();
      for (int e = 0; e < serial.GetNE(); e++)
      {
         serial.SetAttribute(e, 1 + e);
      }
      serial.SetAttributes();
      const char *order = "";
      auto Body = [&](MPI_Comm comm)
      {
         int nranks, rank;
         MPI_Comm_size(comm, &nranks);
         MPI_Comm_rank(comm, &rank);
         const Rule &init = InitialRules()[0].second;
         Array<int> p0 = SerialPartition(serial, init, nranks);
         ParMesh pmesh(comm, serial, p0.GetData());
         if (m.unmark_last_rank && nranks > 1 && rank == nranks - 1)
         {
            for (int e = 0; e < pmesh.GetNE(); e++)
            {
               Element *el = pmesh.GetElement(e);
               static_cast<Tetrahedron *>(el)->SetRefinementFlag(0);
            }
         }
         std::vector<std::unique_ptr<FiniteElementCollection>> fec;
         std::vector<std::unique_ptr<ParFiniteElementSpace>> fes, comp_fes;
         std::vector<std::unique_ptr<ParGridFunction>> x;
         // The scalar components of the fields set through them.
         std::vector<std::vector<std::unique_ptr<ParGridFunction>>> comp(
            cases.size());
         for (std::size_t i = 0; i < cases.size(); i++)
         {
            const SpaceCase &c = cases[i];
            fec.emplace_back(c.fec(m.dim));
            fes.emplace_back(new ParFiniteElementSpace(&pmesh, fec.back().get(),
                                                       c.vdim, c.ordering));
            x.emplace_back(new ParGridFunction(fes.back().get()));
            comp_fes.emplace_back(c.components ?
                                  new ParFiniteElementSpace(&pmesh,
                                                            fec.back().get()) :
                                  nullptr);
            if (!c.components)
            {
               RandomTrueDofs(*x.back(), int(i));
               continue;
            }
            for (int k = 0; k < c.vdim; k++)
            {
               comp[i].emplace_back(new ParGridFunction(comp_fes[i].get()));
               RandomTrueDofs(*comp[i].back(), 100*int(i) + k);
               for (int j = 0; j < comp_fes[i]->GetNDofs(); j++)
               {
                  (*x[i])(fes[i]->DofToVDof(j, k)) = (*comp[i][k])(j);
               }
            }
         }
         // Component k of field i, in the scalar space.
         auto Component = [&](std::size_t i, int k)
         {
            ParGridFunction y(comp_fes[i].get());
            for (int j = 0; j < comp_fes[i]->GetNDofs(); j++)
            {
               y(j) = (*x[i])(fes[i]->DofToVDof(j, k));
            }
            return y;
         };
         auto ByElement = [&](std::size_t i)
         {
            Array<int> attr(pmesh.GetNE()), zero(pmesh.GetNE());
            for (int e = 0; e < pmesh.GetNE(); e++)
            {
               attr[e] = pmesh.GetAttribute(e);
            }
            zero = 0;
            return Coefficients(*x[i], attr, zero);
         };
         std::vector<Descriptors> first;
         for (std::size_t i = 0; i < x.size(); i++)
         {
            first.push_back(ByElement(i));
         }
         std::vector<const Rule *> rules;
         for (const auto &r : TargetRules()) { rules.push_back(&r.second); }
         rules.push_back(&init);
         for (std::size_t r = 0; r < rules.size(); r++)
         {
            INFO(m.name << ", " << nranks << " ranks" << order
                 << ", rebalance " << r + 1);
            Array<int> rank_(pmesh.GetNE()), index(pmesh.GetNE());
            rank_ = rank;
            for (int e = 0; e < pmesh.GetNE(); e++) { index[e] = e; }
            std::vector<Descriptors> before;
            for (auto &y : x)
            {
               before.push_back(Coefficients(*y, rank_, index));
            }
            Array<int> partition = LocalPartition(pmesh, *rules[r]);
            RebalanceChecked(pmesh, partition);
            // (The last two rebalances put all the elements on one rank, then
            // most of them on rank 0.)
            if (m.name == "3D, every orientation, no flags" &&
                r + 2 < rules.size())
            {
               CHECK(AllFaceOrientations(pmesh));
            }
            for (auto &f : fes) { f->Update(); }
            for (auto &y : x) { y->Update(); }
            for (auto &f : comp_fes) { if (f) { f->Update(); } }
            for (auto &c : comp) { for (auto &y : c) { y->Update(); } }
            Array<int> old_rank, old_index;
            pmesh.GetRebalanceOldElements(old_rank, old_index);
            for (std::size_t i = 0; i < x.size(); i++)
            {
               INFO(cases[i].name);
               CHECK(AllOf(comm, Close(Coefficients(*x[i], old_rank, old_index),
                                       before[i], 1e-12)));
               if (!cases[i].components) { CHECK(SingleValued(*x[i])); }
               for (std::size_t k = 0; k < comp[i].size(); k++)
               {
                  ParGridFunction y = Component(i, int(k));
                  CHECK(SingleValued(y));
                  y -= *comp[i][k];
                  CHECK(AllOf(comm, y.Normlinf() <= 1e-12));
               }
               if (r + 1 == rules.size())
               {
                  CHECK(AllOf(comm, Close(ByElement(i), first[i], 1e-12)));
               }
            }
         }
         if (m.refine)
         {
            // (Norm() is for fields of vdim 1, or of H1 or L2 spaces.)
            std::vector<real_t> norm;
            for (std::size_t i = 0; i < x.size(); i++)
            {
               norm.push_back(cases[i].components ? 0.0 : Norm(*x[i]));
            }
            Array<int> marked;
            Vector c;
            for (int e = 0; e < pmesh.GetNE(); e++)
            {
               pmesh.GetElementCenter(e, c);
               if (c(0) < 0.5) { marked.Append(e); }
            }
            pmesh.GeneralRefinement(marked, 0);
            for (auto &f : fes) { f->Update(); }
            for (auto &y : x) { y->Update(); }
            for (std::size_t i = 0; i < x.size(); i++)
            {
               if (cases[i].components) { continue; }
               INFO(m.name << ", " << nranks << " ranks" << order
                    << ", refined, " << cases[i].name);
               CHECK(std::abs(Norm(*x[i]) - norm[i]) <= 1e-11*norm[i]);
               // The ranks interpolate the new shared DOFs from different
               // elements: their copies agree up to round-off.
               CHECK(SingleValued(*x[i], 1e-12));
            }
         }
      };
      ForEachComm(Body, false);
      // All the ranks, numbered in the reverse order of MPI_COMM_WORLD.
      if (Mpi::WorldSize() > 1)
      {
         MPI_Comm reversed;
         MPI_Comm_split(MPI_COMM_WORLD, 0,
                        Mpi::WorldSize() - 1 - Mpi::WorldRank(), &reversed);
         order = " in reverse order";
         Body(reversed);
         MPI_Comm_free(&reversed);
      }
   }
}

TEST_CASE("ParMesh Rebalance conforming: grid function on the nodal space",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // The rebalance updates the nodal space itself. A grid function on that
   // space follows with Update() alone: at the points of every element,
   // matched through its previous location, its values are those before, and
   // x = P R x. H1 nodes of orders 2 and 3, and the L2 nodes of a periodic
   // mesh, through two rebalances.
   for (int dim : {2, 3})
   {
      for (const std::pair<Kind, int> &ko :
           {
              std::make_pair(Kind::CURVED, 2), std::make_pair(Kind::CURVED, 3),
              std::make_pair(Kind::PERIODIC, 2)
           })
      {
         Mesh serial = MakeMesh(dim, ko.second, ko.first);
         ForEachComm([&](MPI_Comm comm)
         {
            int nranks, rank;
            MPI_Comm_size(comm, &nranks);
            MPI_Comm_rank(comm, &rank);
            Array<int> p0 = SerialPartition(serial, InitialRules()[1].second,
                                            nranks);
            ParMesh pmesh(comm, serial, p0.GetData());
            FiniteElementSpace *nodes_fes = pmesh.GetNodes()->FESpace();
            ParFiniteElementSpace &nfes =
               *static_cast<ParFiniteElementSpace *>(nodes_fes);
            ParGridFunction g(&nfes);
            RandomTrueDofs(g, 3);
            for (int r : {0, 2})
            {
               const char *kind = (ko.first == Kind::PERIODIC) ? " periodic" :
                                  " curved";
               INFO(dim << "D, order " << ko.second << kind << ", " << nranks
                    << " ranks, to " << TargetRules()[r].first);
               Array<int> rank_(pmesh.GetNE()), index(pmesh.GetNE());
               rank_ = rank;
               for (int e = 0; e < pmesh.GetNE(); e++) { index[e] = e; }
               const Descriptors before = FieldValues(g, rank_, index);
               Array<int> partition =
                  LocalPartition(pmesh, TargetRules()[r].second);
               RebalanceChecked(pmesh, partition);
               g.Update();
               Array<int> old_rank, old_index;
               pmesh.GetRebalanceOldElements(old_rank, old_index);
               CHECK(AllOf(comm, Close(FieldValues(g, old_rank, old_index),
                                       before, 1e-12)));
               CHECK(SingleValued(g));
            }
         });
      }
   }
}

TEST_CASE("ParMesh Rebalance conforming: nodes equal their transfer",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // The rebalance updates the nodal space with its transfer operator, which
   // gives the nodes. Applied to the nodes before, the operator gives the
   // nodes after (equal as numbers: the comparison does not tell -0 from 0),
   // when the nodes before are single-valued: curved meshes with H1 nodes
   // and periodic meshes with L2 nodes, of orders 1 to 5, through four
   // rebalances.
   for (int dim : {2, 3})
   {
      for (Kind kind : {Kind::CURVED, Kind::PERIODIC})
      {
         for (int order = 1; order <= 5; order++)
         {
            Mesh serial = MakeMesh(dim, order, kind);
            ForEachComm([&](MPI_Comm comm)
            {
               int nranks;
               MPI_Comm_size(comm, &nranks);
               Array<int> p0 = SerialPartition(serial, InitialRules()[1].second,
                                               nranks);
               ParMesh pmesh(comm, serial, p0.GetData());
               for (std::size_t r = 0; r < TargetRules().size(); r++)
               {
                  const char *name = (kind == Kind::PERIODIC) ? " periodic" :
                                     " curved";
                  INFO(dim << "D, order " << order << name << ", " << nranks
                       << " ranks, to " << TargetRules()[r].first);
                  ParGridFunction &nodes =
                     *static_cast<ParGridFunction *>(pmesh.GetNodes());
                  CHECK(SingleValued(nodes));
                  const Vector old(nodes);
                  Array<int> partition =
                     LocalPartition(pmesh, TargetRules()[r].second);
                  RebalanceChecked(pmesh, partition);
                  const Operator *T = nodes.FESpace()->GetUpdateOperator();
                  const bool have_T = AllOf(comm, T != nullptr);
                  CHECK(have_T);
                  if (!have_T) { continue; }
                  Vector y(T->Height());
                  T->Mult(old, y);
                  bool same = (y.Size() == nodes.Size());
                  for (int i = 0; same && i < y.Size(); i++)
                  {
                     same = (y(i) == nodes(i));
                  }
                  CHECK(AllOf(comm, same));
               }
            });
         }
      }
   }
}

TEST_CASE("ParMesh Rebalance conforming: nodes of a mesh refined in parallel",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // After a parallel refinement of a curved mesh, the ranks hold copies of
   // the new shared nodes that can differ by round-off: each rank computes
   // them from its own elements. The nodes still follow the elements with the
   // transfer operator of their space, and each element keeps its geometry up
   // to that round-off, through four rebalances.
   for (int dim : {2, 3})
   {
      for (int order : {2, 3})
      {
         Mesh serial = MakeMesh(dim, order, Kind::CURVED);
         ForEachComm([&](MPI_Comm comm)
         {
            int nranks, rank;
            MPI_Comm_size(comm, &nranks);
            MPI_Comm_rank(comm, &rank);
            Array<int> p0 = SerialPartition(serial, InitialRules()[1].second,
                                            nranks);
            ParMesh pmesh(comm, serial, p0.GetData());
            pmesh.UniformRefinement();
            for (std::size_t r = 0; r < TargetRules().size(); r++)
            {
               INFO(dim << "D, order " << order << ", " << nranks
                    << " ranks, to " << TargetRules()[r].first);
               ParGridFunction &nodes =
                  *static_cast<ParGridFunction *>(pmesh.GetNodes());
               Array<int> rank_(pmesh.GetNE()), index(pmesh.GetNE());
               rank_ = rank;
               for (int e = 0; e < pmesh.GetNE(); e++) { index[e] = e; }
               const Descriptors before = FieldValues(nodes, rank_, index);
               const Vector old(nodes);
               Array<int> partition =
                  LocalPartition(pmesh, TargetRules()[r].second);
               RebalanceChecked(pmesh, partition);
               const Operator *T = nodes.FESpace()->GetUpdateOperator();
               REQUIRE(AllOf(comm, T != nullptr));
               Vector y(T->Height());
               T->Mult(old, y);
               bool same = (y.Size() == nodes.Size());
               for (int i = 0; same && i < y.Size(); i++)
               {
                  same = (y(i) == nodes(i));
               }
               CHECK(AllOf(comm, same));
               Array<int> old_rank, old_index;
               pmesh.GetRebalanceOldElements(old_rank, old_index);
               CHECK(AllOf(comm, Close(FieldValues(nodes, old_rank, old_index),
                                       before, 1e-13)));
            }
         });
      }
   }
}

TEST_CASE("ParMesh Rebalance conforming: cached true vectors",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // A true vector cached on all the ranks (SetTrueVector()), in a grid
   // function and in the nodes, with ranks that have no element before and
   // after the rebalance. The update of a grid function refreshes its cache on
   // the ranks where it was not empty: that must not be the first use of the
   // new prolongation, whose construction is collective (the space builds it
   // in its Update(), as it had one before).
   const int nranks = Mpi::WorldSize();
   for (const bool curved : {false, true})
   {
      INFO((curved ? "curved" : "straight"));
      Mesh serial = Mesh::MakeCartesian2D(3, 3, Element::TRIANGLE);
      if (curved) { serial.SetCurvature(2); }
      Array<int> p0(serial.GetNE());
      p0 = 0;
      ParMesh pmesh(MPI_COMM_WORLD, serial, p0.GetData());
      H1_FECollection fec(2, 2);
      ParFiniteElementSpace fes(&pmesh, &fec);
      ParGridFunction x(&fes);
      FunctionCoefficient f([](const Vector &p)
      { return 1.0 + p(0) + 2.0*p(1)*p(1); });
      x.ProjectCoefficient(f);
      x.SetTrueVector();
      ParGridFunction *nodes =
         curved ? static_cast<ParGridFunction *>(pmesh.GetNodes()) : nullptr;
      if (nodes) { nodes->SetTrueVector(); }
      // The cached true vector of u, if kept, is that of its new space.
      auto CacheValid = [](const ParGridFunction &u, int old_true_size)
      {
         if (old_true_size == 0 || u.ParFESpace()->GetTrueVSize() == 0)
         {
            return true;
         }
         Vector t;
         u.GetTrueDofs(t);
         const Vector &cached = u.GetTrueVector();
         bool same = (cached.Size() == t.Size());
         for (int i = 0; same && i < t.Size(); i++)
         {
            same = (cached(i) == t(i));
         }
         return same;
      };
      // All the elements to the last rank, then back to rank 0.
      for (const int target : {nranks - 1, 0})
      {
         const int old_size = fes.GetTrueVSize();
         const int old_nodes_size =
            nodes ? nodes->ParFESpace()->GetTrueVSize() : 0;
         Array<int> partition(pmesh.GetNE());
         partition = target;
         RebalanceChecked(pmesh, partition);
         fes.Update();
         x.Update();
         ParGridFunction y(&fes);
         y.ProjectCoefficient(f);
         y -= x;
         CHECK(AllOf(MPI_COMM_WORLD, y.Normlinf() < 1e-12));
         CHECK(AllOf(MPI_COMM_WORLD, CacheValid(x, old_size)));
         if (nodes)
         {
            CHECK(AllOf(MPI_COMM_WORLD, CacheValid(*nodes, old_nodes_size)));
         }
      }
   }
}

TEST_CASE("ParMesh Rebalance conforming: fields at vertices of no element",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // Five vertices of no element on every rank, the last three shared by all
   // the ranks, the first two not; at first all the elements are on rank 0.
   // Random H1 fields of orders 1 and 3, without nodes and with H1 nodes of
   // order 3, keep their values at these vertices through three rebalances,
   // and x = P R x. The ranks without elements have one group, of all the
   // ranks: in an MFEM_DEBUG build, the build of the groups checks there too
   // the sizes of their tables counted before the mesh changes.
   for (int dim : {2, 3})
   {
      Mesh serial = (dim == 3) ?
                    Mesh::MakeCartesian3D(1, 1, 1, Element::TETRAHEDRON) :
                    Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE);
      for (int nodes : {0, 1})
      {
         ForEachComm([&](MPI_Comm comm)
         {
            int nranks, rank;
            MPI_Comm_size(comm, &nranks);
            MPI_Comm_rank(comm, &rank);
            Array<int> init(serial.GetNE());
            init = 0;
            CorruptibleParMesh pmesh(comm, serial, init.GetData());
            auto Point = [dim](int i, real_t *x)
            {
               for (int d = 0; d < dim; d++) { x[d] = 10.0*(d + 1) + i; }
            };
            for (int i = 0; i < 5; i++)
            {
               real_t x[3];
               Point(i, x);
               pmesh.AddVertex(x);
            }
            pmesh.Finalize(false, false);
            pmesh.ShareLastVertices(3);
            if (nodes)
            {
               pmesh.SetCurvature(3, false, dim, Ordering::byNODES);
               // The nodes of a vertex of no element are not set from an
               // element: set them to the vertex.
               Array<int> vd;
               for (int i = 0; i < 5; i++)
               {
                  const int v = pmesh.GetNV() - 5 + i;
                  pmesh.GetNodes()->FESpace()->GetVertexVDofs(v, vd);
                  real_t x[3];
                  Point(i, x);
                  for (int d = 0; d < dim; d++)
                  {
                     (*pmesh.GetNodes())(vd[d]) = x[d];
                  }
               }
            }
            H1_FECollection h1(1, dim), h3(3, dim);
            ParFiniteElementSpace f1(&pmesh, &h1), f3(&pmesh, &h3);
            ParGridFunction x1(&f1), x3(&f3);
            RandomTrueDofs(x1, 11);
            RandomTrueDofs(x3, 12);
            // The values at the vertices of no element, found by their points.
            auto Values = [&]()
            {
               std::vector<real_t> v(10, 0.0);
               Array<int> dofs;
               for (int i = 0; i < pmesh.GetNV(); i++)
               {
                  real_t x[3];
                  const real_t *p = pmesh.GetVertex(i);
                  for (int k = 0; k < 5; k++)
                  {
                     Point(k, x);
                     bool here = true;
                     for (int d = 0; d < dim; d++)
                     {
                        here = here && p[d] == x[d];
                     }
                     if (!here) { continue; }
                     f1.GetVertexDofs(i, dofs);
                     v[k] = x1(dofs[0]);
                     f3.GetVertexDofs(i, dofs);
                     v[5 + k] = x3(dofs[0]);
                  }
               }
               return v;
            };
            const std::vector<real_t> before = Values();
            for (real_t b : before) { CHECK(b != 0.0); }
            for (int pass = 0; pass < 3; pass++)
            {
               INFO(dim << "D, " << (nodes ? "nodes" : "no nodes") << ", "
                    << nranks << " ranks, rebalance " << pass + 1);
               Array<int> partition(pmesh.GetNE());
               partition = (pass == 0) ? rank : (pass == 1) ? nranks - 1 : 0;
               RebalanceChecked(pmesh, partition);
               for (ParFiniteElementSpace *f : {&f1, &f3}) { f->Update(); }
               x1.Update();
               x3.Update();
               CHECK(AllOf(comm, Values() == before));
               CHECK(SingleValued(x1));
               CHECK(SingleValued(x3));
            }
         });
      }
   }
}

TEST_CASE("ParMesh Rebalance conforming: mixed refinement flags",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // Tetrahedra marked on some ranks and not on others: a valid mesh, with
   // conforming spaces, which a rebalance must keep valid (its shared faces
   // must be described the same way by both of their ranks).
   Mesh serial = Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
   ForEachComm([&](MPI_Comm comm)
   {
      int nranks, rank;
      MPI_Comm_size(comm, &nranks);
      MPI_Comm_rank(comm, &rank);
      INFO(nranks << " ranks");
      Array<int> init(serial.GetNE());
      for (int i = 0; i < init.Size(); i++) { init[i] = i % nranks; }
      ParMesh pmesh(comm, serial, init.GetData());
      if (rank % 2)
      {
         for (int e = 0; e < pmesh.GetNE(); e++)
         {
            static_cast<Tetrahedron *>(pmesh.GetElement(e))->
            SetRefinementFlag(0);
         }
      }
      // A valid input: the ranks agree on their shared entities and the
      // spaces conform.
      REQUIRE(SharedEntitiesAgree(pmesh));
      REQUIRE(ConformityResidual(pmesh) < 1e-11);
      Array<int> old_rank(pmesh.GetNE()), old_index(pmesh.GetNE());
      old_rank = rank;
      for (int e = 0; e < pmesh.GetNE(); e++) { old_index[e] = e; }
      const Descriptors located = Located(pmesh, old_rank, old_index);
      Array<int> partition(pmesh.GetNE());
      partition = (rank + 1) % nranks;
      RebalanceChecked(pmesh, partition);
      CHECK(SharedEntitiesAgree(pmesh));
      CHECK(ConformityResidual(pmesh) < 1e-11);
      // The elements keep their vertex orders and refinement flags.
      pmesh.GetRebalanceOldElements(old_rank, old_index);
      CHECK(AllOf(comm, Located(pmesh, old_rank, old_index) == located));
   });
}

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("ParMesh Rebalance conforming: a vertex of all the ranks",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // The elements around a vertex are dealt to all the ranks, from rank 0,
   // then all go to the last rank: the group of the vertex has all the ranks,
   // and its master holds a description of it for each of them. The tables
   // of the groups are counted before the mesh is replaced; in an MFEM_DEBUG
   // build, the build of the groups checks that they and the sizes of their
   // tables are those counted.
   const int nranks = Mpi::WorldSize();
   for (const int dim : {2, 3})
   {
      INFO(dim << "D");
      Mesh serial;
      if (dim == 2)
      {
         // A fan of triangles around the origin.
         const int n = std::max(3, nranks);
         serial = Mesh(2, n + 1, n, n);
         serial.AddVertex(0.0, 0.0);
         for (int i = 0; i < n; i++)
         {
            const real_t a = 2.0*M_PI*i/n;
            serial.AddVertex(std::cos(a), std::sin(a));
         }
         for (int i = 0; i < n; i++)
         {
            serial.AddTriangle(0, 1 + i, 1 + (i + 1) % n);
            serial.AddBdrSegment(1 + i, 1 + (i + 1) % n);
         }
         serial.FinalizeTopology();
         serial.Finalize(true, true);
      }
      else
      {
         serial = Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
      }
      // The vertex at the center, and the ranks of the elements around it.
      int center = -1;
      for (int v = 0; v < serial.GetNV(); v++)
      {
         real_t d = 0.0;
         for (int k = 0; k < dim; k++)
         {
            const real_t c = (dim == 2) ? 0.0 : 0.5;
            d += std::abs(serial.GetVertex(v)[k] - c);
         }
         if (d < 1e-12) { center = v; }
      }
      REQUIRE(center >= 0);
      Array<int> target(serial.GetNE()), v;
      target = 0;
      for (int e = 0, k = 0; e < serial.GetNE(); e++)
      {
         serial.GetElementVertices(e, v);
         if (v.Find(center) >= 0) { target[e] = k++ % nranks; }
      }
      Array<int> p0(serial.GetNE());
      p0 = 0;
      ParMesh pmesh(MPI_COMM_WORLD, serial, p0.GetData());
      // Rank 0 holds all the elements, in the order of the serial mesh.
      Array<int> partition(pmesh.GetNE());
      for (int e = 0; e < pmesh.GetNE(); e++) { partition[e] = target[e]; }
      RebalanceChecked(pmesh, partition);
      int widest = 1;
      for (int g = 1; g < pmesh.gtopo.NGroups(); g++)
      {
         widest = std::max(widest, pmesh.gtopo.GetGroupSize(g));
      }
      CHECK(AllOf(MPI_COMM_WORLD, widest == nranks));
      partition.SetSize(pmesh.GetNE());
      partition = nranks - 1;
      RebalanceChecked(pmesh, partition);
      CHECK(pmesh.gtopo.NGroups() == 1);
   }
}

TEST_CASE("ParMesh Rebalance conforming: inconsistent meshes refused",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // A parallel mesh whose ranks describe their shared entities differently
   // is refused, on all ranks, before anything moves: its face neighbor data,
   // exchanged before, are kept too.
   if (Mpi::WorldSize() < 2) { return; }
   const int dim = GENERATE(2, 3);
   Mesh serial = MakeMesh(dim, 1, Kind::STRAIGHT);
   const char *name[] = {"refinement edge", "edge order", "vertex count",
                         "edge direction", "face orientation"
                        };
   for (int corruption = 0; corruption < 5; corruption++)
   {
      CorruptibleParMesh pmesh(MPI_COMM_WORLD, serial);
      pmesh.ExchangeFaceNbrData();
      const int nbr_elems = pmesh.GetNFaceNeighborElements();
      // The vertex count is changed on rank 1: a member, not the master, of a
      // group it shares with rank 0.
      const int rank = (corruption == 2) ? 1 : 0;
      bool corrupted = false;
      if (Mpi::WorldRank() == rank)
      {
         switch (corruption)
         {
            case 0: corrupted = pmesh.RotateSharedTriangle(); break;
            case 1: corrupted = pmesh.SwapSharedEdges(); break;
            case 2: corrupted = pmesh.DropSharedVertex(); break;
            case 3: corrupted = pmesh.ReverseSharedEdge(); break;
            case 4: corrupted = pmesh.FlipSharedTriangle(); break;
         }
      }
      MPI_Bcast(&corrupted, 1, MFEM_MPI_CXX_BOOL, rank, MPI_COMM_WORLD);
      if (!corrupted) { continue; }
      INFO(name[corruption] << ", dimension " << dim);
      const int ne = pmesh.GetNE();
      Array<int> partition(ne);
      partition = (pmesh.GetMyRank() + 1) % pmesh.GetNRanks();
      CHECK_THROWS_AS(pmesh.Rebalance(partition), ErrorException);
      CHECK(pmesh.GetNE() == ne);
      CHECK(pmesh.GetNFaceNeighborElements() == nbr_elems);
   }
}
#endif

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("ParMesh Rebalance conforming: ParSubMesh refused",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // A ParSubMesh keeps maps to its parent mesh, which a rebalance would
   // invalidate: it must be refused, on all ranks, before anything moves.
   Mesh tets = Mesh::MakeCartesian3D(4, 2, 2, Element::TETRAHEDRON);
   for (int e = 0; e < tets.GetNE(); e++)
   {
      Vector c;
      tets.GetElementCenter(e, c);
      tets.SetAttribute(e, c(0) < 0.5 ? 1 : 2);
   }
   tets.SetAttributes();
   ParMesh pmesh(MPI_COMM_WORLD, tets);
   Array<int> domain(1);
   domain[0] = 1;
   ParSubMesh sub = ParSubMesh::CreateFromDomain(pmesh, domain);
   const int ne = sub.GetNE();
   const int nv = sub.GetNV();
   Array<int> partition(ne);
   partition = 0;
   REQUIRE_THROWS_AS(sub.Rebalance(partition), ErrorException);
   // Unchanged.
   CHECK(sub.GetNE() == ne);
   CHECK(sub.GetNV() == nv);
}
#endif

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("ParMesh Rebalance conforming: rows refused before they are filled",
          "[Parallel], [ParMesh], [Rebalance]")
{
   // The sizes of the rows that follow the elements of a rebalance are checked
   // in 64 bits, on all ranks, before any row is written into a message: a
   // row of INT_MAX entries, which does not fit a message with its length,
   // and a row longer than an int can count are refused without a call to
   // the function that fills the rows. Rows that fit are exchanged afterwards,
   // in the order of the previous ranks and indices of the elements.
   ForEachComm([](MPI_Comm comm)
   {
      Mesh serial = Mesh::MakeCartesian2D(3, 3, Element::TRIANGLE);
      RowExchangeParMesh pmesh(comm, serial);
      const int old_ne = pmesh.GetNE(), rank = pmesh.GetMyRank();
      Array<int> partition(old_ne);
      partition = 0;
      RebalanceChecked(pmesh, partition);

      // The long row is on one rank only: the first that has elements.
      int first = (old_ne > 0) ? rank : pmesh.GetNRanks();
      MPI_Allreduce(MPI_IN_PLACE, &first, 1, MPI_INT, MPI_MIN, comm);
      bool filled = false;
      auto fill_none = [&filled](int, std::int64_t *) { filled = true; };
      Array<int> offsets;
      Array<std::int64_t> row_sizes(old_ne), rows;
      for (std::int64_t size : {std::int64_t(INT_MAX), std::int64_t(1) << 40})
      {
         row_sizes = 0;
         if (rank == first) { row_sizes[old_ne - 1] = size; }
         CHECK_THROWS_AS(pmesh.ExchangeRebalanceRows(row_sizes, fill_none,
                                                     offsets, rows),
                         ErrorException);
      }
      CHECK_FALSE(filled);

      row_sizes = 2;
      pmesh.ExchangeRebalanceRows(row_sizes, [&](int i, std::int64_t *row)
      {
         row[0] = rank;
         row[1] = i;
      }, offsets, rows);
      // All the elements are on rank 0 now.
      const int nranks = pmesh.GetNRanks();
      std::vector<int> old_counts(nranks);
      MPI_Allgather(&old_ne, 1, MPI_INT, old_counts.data(), 1, MPI_INT, comm);
      std::vector<std::int64_t> expected;
      for (int r = 0; rank == 0 && r < nranks; r++)
      {
         for (int i = 0; i < old_counts[r]; i++)
         {
            expected.push_back(r);
            expected.push_back(i);
         }
      }
      bool same = offsets.Size() == pmesh.GetNE() + 1 &&
                  rows.Size() == int(expected.size()) &&
                  std::equal(expected.begin(), expected.end(), rows.begin());
      for (int e = 0; same && e <= pmesh.GetNE(); e++)
      {
         same = offsets[e] == 2*e;
      }
      CHECK(AllOf(comm, same));
   });
}
#endif

#endif // MFEM_USE_MPI

} // namespace mfem
