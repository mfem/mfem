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

#ifndef MFEM_MESH_TEST_UTILS
#define MFEM_MESH_TEST_UTILS

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>

namespace mfem
{

/**
 * @brief Which type of FiniteElementCollection to use
 */
enum class FECType
{
   H1,
   ND,
   RT,
   L2
};

/**
 * @brief Create a FiniteElementCollection
 *
 * @param fectype the type of FEC to create
 * @param p The polynomial order
 * @param dim The dimension
 * @return FiniteElementCollection*
 */
FiniteElementCollection *create_fec(FECType fectype, int p, int dim);


/**
 * @brief Helper function for performing an H1 Poisson solve on a serial mesh,
 * with homogeneous essential boundary conditions. Optionally can disable a
 * boundary.
 *
 * @param mesh The SERIAL mesh to perform the Poisson solve on
 * @param order The polynomial order of the basis
 * @param disabled_boundary_attribute Optional boundary attribute to NOT apply
 * homogeneous Dirichlet boundary condition on. Default of -1 means no boundary
 * is disabled.
 * @return int The number of DOF that are fixed by the essential boundary
 * condition.
 */
int CheckPoisson(Mesh &mesh, int order, int disabled_boundary_attribute = -1);

/**
 * @brief Helper for counting the number of essential degrees of freedom within
 * a mesh.
 *
 * @tparam FECollection FiniteElementCollection to define the space.
 * @tparam TDOF Whether or not to count true degrees of freedom (TDOF) or
 * local (vector) degrees of freedom (VDOF).
 * @param mesh The mesh to perform the test upon.
 * @param order The polynomial order of the basis.
 * @param attribute The attribute of the boundary to count the essential DOFs
 * on.
 * @return int The number of essential DOFs.
 */
template <typename FECollection, bool TDOF = true>
int CountEssentialDof(Mesh &mesh, int order, int attribute);

/**
 * @brief Build a mesh with a central tetrahedron surrounded by four
 * tetrahedra, one sharing each face with the central tetrahedron.
 *
 * @return Mesh
 */
Mesh TetStarMesh();

/**
 * @brief Create a mesh of a cube with an internal boundary separating the
 * domain into two halves.
 * @details Depending on @a split, the cube will be divided into two volumes
 * with different volume attributes.
 *
 * @param tet_mesh Whether or not to split the generated mesh into tetrahedra.
 * @param split Whether to introduce the internal boundary.
 * @param three_dim Whether to generate a 3D mesh.
 * @return Mesh
 */
Mesh DividingPlaneMesh(bool tet_mesh = true, bool split = true,
                       bool three_dim = true);

/**
 * @brief Create a mesh of two tetrahedra that share one triangular face at x=0.
 *
 * @param orientation The orientation of the shared triangular face viewed from
 * the second tetrahedra. Options: 1, 3 or 5.
 * @param add_extbdr Whether or not to define boundary elements on the external
 * faces.
 * @return Mesh
 */
Mesh OrientedTriFaceMesh(int orientation, bool add_extbdr = false);

/**
 * @brief Create a mesh of a cylinder using Prisms, Cubes or Tetrahedra.
 *
 * @param el_type Geometry type used, PRISM, CUBE, and TETRAHEDRON are the only
 * valid options.
 * @param quadratic Whether the mesh should be quadratic.
 * @param variant If using prisms, there are 3 different variants of the vertex
 * numbering, specify 0, 1 or 2 to choose between.
 * @return Mesh
 */
Mesh CylinderMesh(Geometry::Type el_type, bool quadratic, int variant = 0);



/**
 * @brief Helper to refine a single element attached to a boundary attribute
 *
 * @param mesh Mesh to refine
 * @param vattr Volume attribute to check for elements
 * @param battr Boundary attribute refined element should be attached to
 * @param backwards Whether to iterate over the faces in reverse order
 */
void RefineSingleAttachedElement(Mesh &mesh, int vattr, int battr,
                                 bool backwards = true);

/**
 * @brief Helper to refine a single element not attached to a boundary
 *
 * @param mesh Mesh to refine
 * @param vattr Volume attribute to check for elements
 * @param battr Boundary attribute refined element should not be attached to
 * @param backwards Whether to iterate over the elements in reverse order
 */
void RefineSingleUnattachedElement(Mesh &mesh, int vattr, int battr,
                                   bool backwards = true);

/**
 * @brief Create a 4 x 2 x 2 mesh of the unit cube, or a 4 x 2 mesh of the unit
 * square, with attribute 1 for x < 0.5 and 2 otherwise. Boundary elements are
 * added on the interior faces x = 0.25 (attribute 7) and x = 0.5 (attribute 8).
 * All the boundary elements are rotated with respect to their faces, except
 * the segments which have no rotation, and every other one on the interior
 * faces is reversed.
 *
 * @param type The element type, HEXAHEDRON, TETRAHEDRON, QUADRILATERAL or
 * TRIANGLE
 * @param nonconforming Make the mesh nonconforming (simplices included)
 * @param curved Give the mesh third order nodes and curve it
 * @return Mesh
 */
Mesh RotatedBoundaryElementsMesh(Element::Type type, bool nonconforming = false,
                                 bool curved = false);

/**
 * @brief Check the boundary elements of a submesh extracted from a domain.
 * There shall be at most one boundary element per face. Except on the local
 * interior faces of the submesh, the boundary elements shall keep the
 * orientation of their faces. When the vertex order of the matching boundary
 * element of the parent is a rotation of the one of the submesh, or a
 * reflection on a local interior face, both orders shall be equal, and both
 * boundary elements shall map the same reference points to the same physical
 * points. On meshes without nodes, the boundary values of a cubic function
 * shall be exact with an order 3 space. The checks do not stop at the first
 * failure, so that all the ranks reach the collective calls of the caller.
 *
 * @tparam SubMeshT SubMesh or ParSubMesh
 * @param submesh The submesh
 * @return The number of boundary elements compared to the ones of the parent
 */
template <typename SubMeshT>
int CheckSubMeshBoundaryElements(SubMeshT &submesh)
{
   const auto &parent = *submesh.GetParent();
   const auto parent_face_to_be = parent.GetFaceToBdrElMap();
   const auto &parent_face_ids = submesh.Dimension() == 3 ?
                                 submesh.GetParentFaceIDMap() :
                                 submesh.GetParentEdgeIDMap();
   const auto &parent_vertex_ids = submesh.GetParentVertexIDMap();
   const auto face_to_be = submesh.GetFaceToBdrElMap();
   bool one_per_face = true, orientation_kept = true;
   bool same_order = true, same_points = true;
   int ncompared = 0;
   for (int i = 0; i < submesh.GetNBE(); i++)
   {
      int f, o;
      submesh.GetBdrElementFace(i, &f, &o);
      one_per_face = one_per_face && (face_to_be[f] == i);
      const bool keep_orientation = !submesh.GetFaceInformation(f).IsLocal();
      orientation_kept = orientation_kept && (!keep_orientation || o % 2 == 0);
      const int pbe = parent_face_to_be[parent_face_ids[f]];
      if (pbe == -1) { continue; }
      Array<int> sv, pv;
      submesh.GetBdrElementVertices(i, sv);
      parent.GetBdrElementVertices(pbe, pv);
      CHECK(sv.Size() == pv.Size());
      if (sv.Size() != pv.Size()) { continue; }
      const int n = sv.Size();
      for (int j = 0; j < n; j++) { sv[j] = parent_vertex_ids[sv[j]]; }
      bool same_cycle = false;
      for (int k = 0; !same_cycle && k < n; k++)
      {
         // A segment has no rotation but the identity.
         bool rotation = n > 2 || k == 0, reflection = !keep_orientation;
         for (int j = 0; j < n; j++)
         {
            rotation = rotation && (pv[j] == sv[(k + j) % n]);
            reflection = reflection && (pv[j] == sv[(k + n - j) % n]);
         }
         same_cycle = rotation || reflection;
      }
      if (!same_cycle) { continue; }
      for (int j = 0; j < n; j++)
      {
         same_order = same_order && (sv[j] == pv[j]);
      }
      IsoparametricTransformation T, PT;
      submesh.GetBdrElementTransformation(i, &T);
      parent.GetBdrElementTransformation(pbe, &PT);
      const auto &ir = IntRules.Get(submesh.GetBdrElementGeometry(i), 6);
      Vector x, px;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         T.Transform(ir.IntPoint(j), x);
         PT.Transform(ir.IntPoint(j), px);
         x -= px;
         same_points = same_points && (x.Normlinf() < 1e-12);
      }
      ncompared++;
   }
   CHECK(one_per_face);
   CHECK(orientation_kept);
   CHECK(same_order);
   CHECK(same_points);
   if (submesh.GetNodes() != nullptr) { return ncompared; }
   H1_FECollection fec(3, submesh.Dimension());
   FiniteElementSpace fes(&submesh, &fec);
   FunctionCoefficient q([](const Vector &x)
   {
      const real_t z = x.Size() == 3 ? x(2) : real_t(1);
      return 1 + x(0) * x(0) * x(0) + 2 * x(0) * x(1) * z + 3 * x(1) * z * z;
   });
   GridFunction g(&fes);
   g.ProjectCoefficient(q);
   real_t error = 0;
   for (int i = 0; i < submesh.GetNBE(); i++)
   {
      Array<int> dofs;
      fes.GetBdrElementDofs(i, dofs);
      Vector values, shape(dofs.Size());
      g.GetSubVector(dofs, values);
      const auto &fe = *fes.GetBE(i);
      auto &T = *submesh.GetBdrElementTransformation(i);
      const auto &ir = IntRules.Get(fe.GetGeomType(), 6);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         T.SetIntPoint(&ip);
         fe.CalcShape(ip, shape);
         error = std::max(error, std::abs(shape * values - q.Eval(T, ip)));
      }
   }
   CHECK(error < 1e-12);
   return ncompared;
}


#ifdef MFEM_USE_MPI

/**
 * @brief Test GetVectorValue on face neighbor elements for nonconforming meshes
 *
 * @param smesh The serial mesh to start from
 * @param nc_level Depth of refinement on processor boundaries
 * @param skip Refine every "skip" processor boundary element
 * @param use_ND Whether to use Nedelec elements (which are sensitive to
 * orientation)
 */
void TestVectorValueInVolume(Mesh &smesh, int nc_level, int skip, bool use_ND);

/**
 * @brief Helper function for performing an H1 Poisson solve on a parallel mesh,
 * with homogeneous essential boundary conditions. Optionally can disable a
 * boundary.
 *
 * @param mesh The PARALLEL mesh to perform the Poisson solve on
 * @param order The polynomial order of the basis
 * @param disabled_boundary_attribute Optional boundary attribute to NOT apply
 * homogeneous Dirichlet boundary condition on. Default of -1 means no boundary
 * is disabled.
 * @return int The number of DOF that are fixed by the essential boundary
 * condition.
 */
void CheckPoisson(ParMesh &pmesh, int order,
                  int disabled_boundary_attribute = -1);

/**
 * @brief Check that a ParMesh generates the same number of boundary elements as
 * the serial mesh.
 *
 * @param smesh Serial mesh to be built from and compared against
 * @param partition Optional partition
 * @return std::unique_ptr<ParMesh> Pointer to the mesh in question.
 */
std::unique_ptr<ParMesh> CheckParMeshNBE(Mesh &smesh,
                                         const std::unique_ptr<int[]> &partition = nullptr);

/**
 * @brief Helper function to track if a face index is internal
 *
 * @param pmesh The mesh containing the face
 * @param f The face index
 * @param local_to_shared A map from local faces to shared faces
 * @return true the face is between domain attributes (and owned by this rank)
 * @return false the face is not between domain attributes or not owned by this
 * rank
 */
bool CheckFaceInternal(ParMesh& pmesh, int f,
                       const std::map<int, int> &local_to_shared);

/**
 * @brief Given a parallel and a serial mesh, perform an L2 projection and check
 * the solutions match exactly.
 * @details The result of an L2 projection on a parallel mesh and serial mesh
 * should be identical irrespective of partitioning. This check can be used for
 * generating data for comparison of a parallel and serial mesh.
 *
 * @param pmesh The parallel mesh to use in the AB test
 * @param smesh The serial mesh to use in the AB test
 * @param order The polynomial order to use in the projection
 * @param exact_soln A function that returns the exact solution at a given point
 * @return std::array<double, 2> Pair of error on the serial mesh and the
 * parallel mesh. Should be within numerical tolerance of each other.
 */
std::array<real_t, 2> CheckL2Projection(ParMesh& pmesh, Mesh& smesh, int order,
                                        std::function<real_t(Vector const&)> exact_soln);


/**
 * @brief Helper for counting the number of essential local degrees of freedom
 * within a parallel mesh.
 *
 * @tparam FECollection FiniteElementCollection to define the space.
 * @tparam TDOF Whether or not to count true degrees of freedom (TDOF) or
 * local (vector) degrees of freedom (VDOF).
 * @param mesh The mesh to perform the test upon.
 * @param order The polynomial order of the basis.
 * @param attribute The attribute of the boundary to count the essential DOFs
 * on.
 * @return int The number of essential DOFs.
 */
template <typename FECollection, bool TDOF = true>
int CountEssentialDof(ParMesh &mesh, int order, int attribute);

/**
 * @brief Helper for counting the number of essential degrees of freedom within
 * a parallel mesh, and summing over all processors.
 *
 * @tparam FECollection FiniteElementCollection to define the space.
 * @tparam TDOF Whether or not to count true degrees of freedom (TDOF) or
 * local (vector) degrees of freedom (VDOF).
 * @param mesh The mesh to perform the test upon.
 * @param order The polynomial order of the basis.
 * @param attribute The attribute of the boundary to count the essential DOFs
 * on.
 * @return int The number of essential DOFs.
 */
template <typename FECollection, bool TDOF = true>
int ParCountEssentialDof(ParMesh &mesh, int order, int attribute);

/**
 * @brief Helper for checking the identity RP = I on a ParFiniteElementSpace
 *
 * @return true The identity holds
 * @return false The identity does not hold
 */
bool CheckRPIdentity(const ParFiniteElementSpace& pfespace);

#endif

} // namespace mfem

#endif // MFEM_MESH_TEST_UTILS
