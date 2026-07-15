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
#include "mesh/mesh_test_utils.hpp"

#include <cmath>

using namespace mfem;

namespace domain_int
{

static double a_ = 5.0;
static double b_ = 3.0;
static double c_ = 2.0;

double integral(int dim)
{
   if (dim == 1)
   {
      return a_;
   }
   else if (dim == 2)
   {
      return a_ * b_;
   }
   else
   {
      return a_ * b_ * c_;
   }
}

enum FEType
{
   H1_FEC = 0,
   ND_FEC,
   RT_FEC,
   L2V_FEC,
   L2I_FEC,
};

enum MeshType
{
   SEGMENT = 0,
   QUADRILATERAL = 1,
   TRIANGLE2A = 2,
   TRIANGLE2B = 3,
   TRIANGLE2C = 4,
   TRIANGLE4 = 5,
   MIXED2D = 6,
   HEXAHEDRON = 7,
   HEXAHEDRON2A = 8,
   HEXAHEDRON2B = 9,
   HEXAHEDRON2C = 10,
   HEXAHEDRON2D = 11,
   WEDGE2 = 12,
   TETRAHEDRA = 13,
   WEDGE4 = 14,
   MIXED3D6 = 15,
   MIXED3D8 = 16,
   PYRAMID = 17
};

Mesh * GetMesh(MeshType type);

TEST_CASE("Domain Integration (Scalar Field)",
          "[H1_FECollection]"
          "[L2_FECollection]"
          "[GridFunction]"
          "[LinearForm]")
{
   int order = 2;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::PYRAMID; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt);
      int  dim = mesh->Dimension();
      mesh->UniformRefinement();

      ConstantCoefficient oneCoef(1.0);

      for (int ft = (int)FEType::H1_FEC; ft <= (int)FEType::L2I_FEC; ft++)
      {
         if (ft == (int)FEType::ND_FEC || ft == (int)FEType::RT_FEC)
         { continue; }

         SECTION("Integral of field " + std::to_string(ft) +
                 " on mesh type " + std::to_string(mt) )
         {

            FiniteElementCollection *fec = NULL;
            switch ((FEType)ft)
            {
               case FEType::H1_FEC:
                  fec = new H1_FECollection(order, dim);
                  break;
               case FEType::L2V_FEC:
                  fec = new L2_FECollection(order-1, dim);
                  break;
               case FEType::L2I_FEC:
                  fec = new L2_FECollection(order-1, dim,
                                            BasisType::GaussLegendre,
                                            FiniteElement::INTEGRAL);
                  break;
               default:
                  MFEM_ABORT("Invalid vector FE type");
            }
            FiniteElementSpace fespace(mesh, fec);

            GridFunction u(&fespace);
            u.ProjectCoefficient(oneCoef);

            LinearForm b(&fespace);
            b.AddDomainIntegrator(new DomainLFIntegrator(oneCoef));
            b.Assemble();

            double id = b(u);

            if (dim == 1)
            {
               REQUIRE(id == MFEM_Approx( 5.0));
            }
            else if (dim == 2)
            {
               REQUIRE(id == MFEM_Approx(15.0));
            }
            else
            {
               REQUIRE(id == MFEM_Approx(30.0));
            }

            delete fec;
         }
      }

      delete mesh;
   }
}

TEST_CASE("Domain Integration (Vector Field)",
          "[ND_FECollection]"
          "[RT_FECollection]"
          "[GridFunction]"
          "[LinearForm]")
{
   int order = 1;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt);
      int  dim = mesh->Dimension();
      int sdim = mesh->SpaceDimension();
      mesh->UniformRefinement();

      Vector f1(sdim); f1 = 1.0;
      Vector fx(sdim); fx = 0.0; fx[0] = 1.0;
      Vector fy(sdim); fy = 0.0;
      if (sdim > 1) { fy[1] = 1.0; }
      Vector fz(sdim); fz = 0.0;
      if (sdim > 2) { fz[2] = 1.0; }

      VectorConstantCoefficient f1Coef(f1);
      VectorConstantCoefficient fxCoef(fx);
      VectorConstantCoefficient fyCoef(fy);
      VectorConstantCoefficient fzCoef(fz);

      for (int ft = (int)FEType::ND_FEC; ft <= (int)FEType::RT_FEC; ft++)
      {
         if (dim == 1 && ft == (int)FEType::RT_FEC) { continue; }
         if (mt == (int)MeshType::WEDGE2 || mt == (int)MeshType::WEDGE4 ||
             mt == (int)MeshType::MIXED3D6 || mt == (int)MeshType::MIXED3D8)
         { continue; }

         SECTION("Integral of field " + std::to_string(ft) +
                 " on mesh type " + std::to_string(mt) )
         {

            FiniteElementCollection *fec = NULL;
            switch ((FEType)ft)
            {
               case FEType::ND_FEC:
                  fec = new ND_FECollection(order, dim);
                  break;
               case FEType::RT_FEC:
                  fec = new RT_FECollection(order-1, dim);
                  break;
               default:
                  MFEM_ABORT("Invalid vector FE type");
            }
            FiniteElementSpace fespace(mesh, fec);

            GridFunction u(&fespace);
            u.ProjectCoefficient(f1Coef);

            LinearForm bx(&fespace);
            LinearForm by(&fespace);
            LinearForm bz(&fespace);
            bx.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fxCoef));
            by.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fyCoef));
            bz.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fzCoef));
            bx.Assemble();
            by.Assemble();
            bz.Assemble();

            double ix = bx(u);
            double iy = by(u);
            double iz = bz(u);

            if (dim == 1)
            {
               REQUIRE(ix == MFEM_Approx( 5.0));
            }
            else if (dim == 2)
            {
               REQUIRE(ix == MFEM_Approx(15.0));
               REQUIRE(iy == MFEM_Approx(15.0));
            }
            else
            {
               REQUIRE(ix == MFEM_Approx(30.0));
               REQUIRE(iy == MFEM_Approx(30.0));
               REQUIRE(iz == MFEM_Approx(30.0));
            }

            delete fec;
         }
      }

      delete mesh;
   }
}

// ---------------------------------------------------------------------------
// Delta-coefficient domain integrators. A delta on a shared mesh entity is
// split uniformly over the elements containing its center.

/** @brief Return a 3D mesh of one hex split into 24 tetrahedra.

    The mesh is translated so that the hex center, a vertex shared by all 24
    tetrahedra, is located at the origin. */
Mesh MakeCenteredHex24Tets()
{
   Mesh mesh = Mesh::MakeCartesian3DWith24TetsPerHex(1, 1, 1, 2.0, 2.0, 2.0);
   for (int v = 0; v < mesh.GetNV(); v++)
   {
      real_t *vp = mesh.GetVertex(v);
      for (int d = 0; d < 3; d++) { vp[d] -= 1.0; }
   }
   return mesh;
}

/** @brief Evaluate a 3D linear vector field used by the delta tests.

    The field is not in the lowest-order Nedelec space, so its ND projection has
    a discontinuous normal trace across element faces. */
void GenericField(const Vector &x, Vector &v)
{
   v.SetSize(3);
   v(0) = 0.1 + 0.2 * x(0) - 0.3 * x(1) + 0.15 * x(2);
   v(1) = -0.2 + 0.4 * x(1) + 0.25 * x(2) - 0.1 * x(0);
   v(2) = 0.3 + 0.5 * x(2) + 0.2 * x(0) - 0.35 * x(1);
}

/** @brief Evaluate GenericField with a nonzero constant x-component offset.

    The offset makes x-directed delta actions nonzero by construction in the
    parallel NC tests, while the varying part of GenericField still exercises
    the split over elements with different one-sided projected values. */
void ShiftedGenericField(const Vector &x, Vector &v)
{
   GenericField(x, v);
   v(0) += 1.0;
}

/** @brief Evaluate the 3D scalar linear field used by scalar delta tests. */
real_t ScalarLinearField(const Vector &x)
{
   return 0.25 + 0.5 * x(0) - 0.125 * x(1) + 0.75 * x(2);
}

/** @brief Return the index of the vertex located at the origin, or -1. */
int FindOriginVertex(const Mesh &mesh)
{
   for (int v = 0; v < mesh.GetNV(); v++)
   {
      const real_t *vp = mesh.GetVertex(v);
      if (std::abs(vp[0]) < 1e-12 && std::abs(vp[1]) < 1e-12 &&
          std::abs(vp[2]) < 1e-12)
      {
         return v;
      }
   }
   return -1;
}

/** @brief Assemble a reference vector delta at the origin.

    The reference uniformly distributes the vector delta over every element
    sharing the origin vertex, using weight 1/N per element.

    @param fes Finite element space on the 3D test mesh. Not modified.
    @param dir Vector delta direction.
    @param expected Output vector, resized and overwritten with the reference
    load vector. */
void AssembleUniformReference(const FiniteElementSpace &fes, const Vector &dir,
                              Vector &expected)
{
   Mesh &mesh = *fes.GetMesh();
   const int v0 = FindOriginVertex(mesh);
   REQUIRE(v0 >= 0);

   Table *vte = mesh.GetVertexToElementTable();
   const int N = vte->RowSize(v0);
   const int *els = vte->GetRow(v0);
   REQUIRE(N > 1);

   expected.SetSize(fes.GetVSize());
   expected = 0.0;

   Vector center(mesh.SpaceDimension());
   center = 0.0;

   for (int j = 0; j < N; j++)
   {
      const int e = els[j];
      ElementTransformation &Trans = *fes.GetElementTransformation(e);
      InverseElementTransformation inv_tr(&Trans);
      IntegrationPoint ip;
      const int res = inv_tr.Transform(center, ip);
      REQUIRE(res == InverseElementTransformation::Inside);
      Trans.SetIntPoint(&ip);

      const FiniteElement &fe = *fes.GetFE(e);
      DenseMatrix vshape(fe.GetDof(), mesh.SpaceDimension());
      fe.CalcPhysVShape(Trans, vshape);

      Vector elemvect(fe.GetDof());
      vshape.Mult(dir, elemvect);
      elemvect *= 1.0 / static_cast<real_t>(N);

      Array<int> vdofs;
      fes.GetElementVDofs(e, vdofs);
      expected.AddElementVector(vdofs, elemvect);
   }

   delete vte;
}

TEST_CASE("Domain Integration (Vector Delta on Shared Vertex)",
          "[ND_FECollection]"
          "[LinearForm]"
          "[DeltaCoefficient]")
{
   Mesh mesh = MakeCenteredHex24Tets();

   ND_FECollection fec(1, 3);
   FiniteElementSpace fes(&mesh, &fec);

   Vector dir(3);
   dir = 0.0;
   dir[2] = 1.0;
   VectorDeltaCoefficient vdc(dir, 0.0, 0.0, 0.0, 1.0);

   LinearForm rhs(&fes);
   rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc));
   rhs.Assemble();

   Vector expected;
   AssembleUniformReference(fes, dir, expected);

   Vector diff(rhs);
   diff -= expected;
   const real_t err = diff.Norml2();
   const real_t ref = expected.Norml2();
   INFO("rhs.Norml2 = " << rhs.Norml2() << ", ref.Norml2 = " << ref
        << ", err = " << err);
   REQUIRE(ref > 0.0);
   REQUIRE(err / ref < 1e-12);
}

// A vector delta on a face shared by two elements, oriented along the face
// NORMAL. The Nedelec tangential trace is continuous across the face so only
// the normal component is discontinuous: the assembled functional applied to a
// field with a discontinuous normal trace must equal the symmetric average of
// the two one-sided values.
TEST_CASE("Domain Integration (Vector Delta on Shared Face)",
          "[ND_FECollection]"
          "[LinearForm]"
          "[DeltaCoefficient]")
{
   Mesh mesh = OrientedTriFaceMesh(1);

   ND_FECollection fec(1, 3);
   FiniteElementSpace fes(&mesh, &fec);

   Vector x0(3);
   x0(0) = 0.0; x0(1) = 1.0 / 3.0; x0(2) = 1.0 / 3.0;  // shared-face centroid
   Vector dir(3);
   dir = 0.0; dir(0) = 1.0;                             // normal to the face

   VectorDeltaCoefficient vdc(dir, x0(0), x0(1), x0(2), 1.0);
   LinearForm rhs(&fes);
   rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc));
   rhs.Assemble();

   VectorFunctionCoefficient Fc(3, GenericField);
   GridFunction g(&fes);
   g.ProjectCoefficient(Fc);

   Vector g0(3), g1(3);
   {
      IntegrationPoint ip;
      InverseElementTransformation inv0(fes.GetElementTransformation(0));
      REQUIRE(inv0.Transform(x0, ip) == InverseElementTransformation::Inside);
      g.GetVectorValue(0, ip, g0);
      InverseElementTransformation inv1(fes.GetElementTransformation(1));
      REQUIRE(inv1.Transform(x0, ip) == InverseElementTransformation::Inside);
      g.GetVectorValue(1, ip, g1);
   }

   const real_t side0 = dir * g0;
   const real_t side1 = dir * g1;
   INFO("side0 = " << side0 << ", side1 = " << side1);
   REQUIRE(std::abs(side0 - side1) > 1e-3);   // the two sides genuinely differ

   const real_t Lg = rhs * g;
   const real_t sym = 0.5 * (side0 + side1);
   INFO("Lg = " << Lg << ", symmetric avg = " << sym
        << ", one-sided = " << side0 << "/" << side1);
   REQUIRE(Lg == MFEM_Approx(sym));
   REQUIRE(std::abs(Lg - side0) > 1e-4);
   REQUIRE(std::abs(Lg - side1) > 1e-4);
}

/** @brief Return representative points on the triangular shared face.

    The shared face is `(x=0)` with vertices `{(0,0,0),(0,1,0),(0,0,1)}`.
    After refinement makes it a master/slave interface, the selected points
    exercise distinct entity classes: 0 a master/conforming vertex, 1 a hanging
    vertex (edge midpoint), 2 a hanging sub-edge on a coarse edge, 3 an interior
    hanging edge between two slave sub-faces, 4 a corner slave sub-face interior,
    and 5 the center slave sub-face interior. */
Vector SharedFacePoint(int entity_type)
{
   Vector p(3);
   p = 0.0;
   switch (entity_type)
   {
      case 0:  p(1) = 0.0;       p(2) = 0.0;       break;
      case 1:  p(1) = 0.5;       p(2) = 0.0;       break;
      case 2:  p(1) = 0.25;      p(2) = 0.0;       break;
      case 3:  p(1) = 0.25;      p(2) = 0.25;      break;
      case 4:  p(1) = 1.0 / 6.0; p(2) = 1.0 / 6.0; break;
      default: p(1) = 1.0 / 3.0; p(2) = 1.0 / 3.0; break;
   }
   return p;
}

/** @brief Check whether a point is contained in a 3D test element.

    The independent check uses barycentric coordinates for tetrahedra and
    bounding boxes for the axis-aligned test hexahedra.

    @param mesh Mesh containing element @a e. Not modified.
    @param e Element index.
    @param pt Physical point to test.
    @param ip Output reference point when the element geometry is supported. */
bool ElementContainsByHand(Mesh &mesh, int e, const Vector &pt,
                           IntegrationPoint &ip)
{
   const real_t tol = 1e-9;
   Array<int> v;
   mesh.GetElementVertices(e, v);
   const Geometry::Type geom = mesh.GetElementBaseGeometry(e);
   if (geom == Geometry::TETRAHEDRON)
   {
      // Barycentric coordinates by Cramer's rule: solve [c0 c1 c2] l = pt-v0
      // with columns ck = v_{k+1} - v0. l are the reference coordinates.
      const real_t *v0 = mesh.GetVertex(v[0]);
      real_t c[3][3], b[3];
      for (int d = 0; d < 3; d++)
      {
         for (int k = 0; k < 3; k++)
         {
            c[d][k] = mesh.GetVertex(v[k + 1])[d] - v0[d];
         }
         b[d] = pt(d) - v0[d];
      }
      auto det3 = [](real_t a0, real_t a1, real_t a2,
                     real_t b0, real_t b1, real_t b2,
                     real_t d0, real_t d1, real_t d2)
      {
         return a0 * (b1 * d2 - b2 * d1) - a1 * (b0 * d2 - b2 * d0)
                + a2 * (b0 * d1 - b1 * d0);
      };
      const real_t D = det3(c[0][0], c[0][1], c[0][2],
                            c[1][0], c[1][1], c[1][2],
                            c[2][0], c[2][1], c[2][2]);
      const real_t l0 = det3(b[0], c[0][1], c[0][2],
                             b[1], c[1][1], c[1][2],
                             b[2], c[2][1], c[2][2]) / D;
      const real_t l1 = det3(c[0][0], b[0], c[0][2],
                             c[1][0], b[1], c[1][2],
                             c[2][0], b[2], c[2][2]) / D;
      const real_t l2 = det3(c[0][0], c[0][1], b[0],
                             c[1][0], c[1][1], b[1],
                             c[2][0], c[2][1], b[2]) / D;
      ip.x = l0; ip.y = l1; ip.z = l2;
      return l0 >= -tol && l1 >= -tol && l2 >= -tol &&
             (l0 + l1 + l2) <= 1.0 + tol;
   }
   else if (geom == Geometry::CUBE)
   {
      real_t mn[3], mx[3];
      for (int d = 0; d < 3; d++) { mn[d] = mx[d] = mesh.GetVertex(v[0])[d]; }
      for (int i = 1; i < v.Size(); i++)
      {
         const real_t *xi = mesh.GetVertex(v[i]);
         for (int d = 0; d < 3; d++)
         {
            mn[d] = std::min(mn[d], xi[d]);
            mx[d] = std::max(mx[d], xi[d]);
         }
      }
      bool inside = true;
      real_t r[3];
      for (int d = 0; d < 3; d++)
      {
         inside = inside && (pt(d) >= mn[d] - tol) &&
                  (pt(d) <= mx[d] + tol);
         r[d] = (pt(d) - mn[d]) / (mx[d] - mn[d]);
      }
      ip.x = r[0]; ip.y = r[1]; ip.z = r[2];
      return inside;
   }
   MFEM_ABORT("by-hand containment supports only TET and CUBE");
   return false;
}

/** @brief Assemble an independent vector-delta reference by brute force.

    Every element is checked geometrically with `ElementContainsByHand`; the
    vector delta is then distributed uniformly over the containing elements.

    @param fes Finite element space on the 3D test mesh.
    @param center Physical delta center.
    @param dir Vector delta direction.
    @param expected Output vector, resized and overwritten with the reference
    load vector.
    @param attr Optional element attribute restriction; zero means all
    attributes are active. */
void AssembleDeltaReferenceBruteForce(FiniteElementSpace &fes,
                                      const Vector &center, const Vector &dir,
                                      Vector &expected, int attr = 0)
{
   Mesh &mesh = *fes.GetMesh();
   std::vector<int> containing;
   std::vector<IntegrationPoint> ips;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      if (attr != 0 && mesh.GetAttribute(e) != attr) { continue; }
      IntegrationPoint ip;
      if (ElementContainsByHand(mesh, e, center, ip))
      {
         containing.push_back(e);
         ips.push_back(ip);
      }
   }
   const int N = static_cast<int>(containing.size());
   REQUIRE(N > 0);

   expected.SetSize(fes.GetVSize());
   expected = 0.0;
   for (size_t j = 0; j < containing.size(); j++)
   {
      const int e = containing[j];
      ElementTransformation &Trans = *fes.GetElementTransformation(e);
      Trans.SetIntPoint(&ips[j]);
      const FiniteElement &fe = *fes.GetFE(e);
      DenseMatrix vshape(fe.GetDof(), mesh.SpaceDimension());
      fe.CalcPhysVShape(Trans, vshape);
      Vector elemvect(fe.GetDof());
      vshape.Mult(dir, elemvect);
      elemvect *= 1.0 / static_cast<real_t>(N);
      Array<int> vdofs;
      fes.GetElementVDofs(e, vdofs);
      expected.AddElementVector(vdofs, elemvect);
   }
}

TEST_CASE("Domain Integration (Vector Delta on Hex Shared Face)",
          "[ND_FECollection]"
          "[LinearForm]"
          "[DeltaCoefficient]")
{
   // Two hexahedra sharing a face (a unit cube split in X at x=0.5).
   const int entity_type = GENERATE(0, 1, 2);
   Vector x0(3);
   x0(0) = 0.5;
   switch (entity_type)
   {
      case 0:  x0(1) = 0.5;  x0(2) = 0.5;  break;  // face centroid
      case 1:  x0(1) = 0.5;  x0(2) = 0.0;  break;  // shared-face edge midpoint
      default: x0(1) = 0.0;  x0(2) = 0.0;  break;  // shared-face corner vertex
   }
   Vector dir(3);
   dir = 0.0; dir(0) = 1.0;

   Mesh mesh = DividingPlaneMesh(false, false, true);
   REQUIRE(mesh.GetNE() == 2);
   REQUIRE(mesh.Conforming());

   ND_FECollection fec(1, 3);
   FiniteElementSpace fes(&mesh, &fec);

   VectorDeltaCoefficient vdc(dir, x0(0), x0(1), x0(2), 1.0);
   LinearForm rhs(&fes);
   rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc));
   rhs.Assemble();

   Vector expected;
   AssembleDeltaReferenceBruteForce(fes, x0, dir, expected);

   Vector diff(rhs);
   diff -= expected;
   const real_t err = diff.Norml2();
   const real_t ref = expected.Norml2();
   INFO("entity_type = " << entity_type << ", rhs.Norml2 = " << rhs.Norml2()
        << ", ref.Norml2 = " << ref << ", err = " << err);
   REQUIRE(ref > 0.0);
   REQUIRE(err / ref < 1e-12);
}

TEST_CASE("Domain Integration (Vector Delta near Shared Face)",
          "[ND_FECollection]"
          "[LinearForm]"
          "[DeltaCoefficient]")
{
   // The center is close to the shared face but is still in only one element.
   Mesh mesh = DividingPlaneMesh(false, false, true);

#ifdef MFEM_USE_SINGLE
   constexpr real_t offset = 1e-3;
#else
   constexpr real_t offset = 1e-7;
#endif
   Vector x0(3);
   x0(0) = 0.5 + offset; x0(1) = 0.5; x0(2) = 0.5;
   Vector dir(3);
   dir = 0.0; dir(0) = 1.0;

   ND_FECollection fec(1, 3);
   FiniteElementSpace fes(&mesh, &fec);

   VectorDeltaCoefficient vdc(dir, x0(0), x0(1), x0(2), 1.0);
   LinearForm rhs(&fes);
   rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc));
   rhs.Assemble();

   Vector expected;
   AssembleDeltaReferenceBruteForce(fes, x0, dir, expected);

   Vector diff(rhs);
   diff -= expected;
   const real_t err = diff.Norml2();
   const real_t ref = expected.Norml2();
   INFO("rhs.Norml2 = " << rhs.Norml2() << ", ref.Norml2 = " << ref
        << ", err = " << err);
   REQUIRE(ref > 0.0);
   REQUIRE(err / ref < 1e-12);
}

TEST_CASE("Domain Integration (Vector Delta Respects Element Marker)",
          "[ND_FECollection]"
          "[LinearForm]"
          "[DeltaCoefficient]")
{
   Mesh mesh = DividingPlaneMesh(false, true, true);
   REQUIRE(mesh.attributes.Max() == 2);

   Vector x0(3);
   x0(0) = 0.5; x0(1) = 0.5; x0(2) = 0.5;
   Vector dir(3);
   dir = 0.0; dir(0) = 1.0;

   ND_FECollection fec(1, 3);
   FiniteElementSpace fes(&mesh, &fec);

   Array<int> attr_marker(mesh.attributes.Max());
   attr_marker = 0;
   attr_marker[0] = 1;

   VectorDeltaCoefficient vdc(dir, x0(0), x0(1), x0(2), 1.0);
   LinearForm rhs(&fes);
   rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc), attr_marker);
   rhs.Assemble();

   Vector expected;
   AssembleDeltaReferenceBruteForce(fes, x0, dir, expected, 1);

   Vector diff(rhs);
   diff -= expected;
   const real_t err = diff.Norml2();
   const real_t ref = expected.Norml2();
   INFO("rhs.Norml2 = " << rhs.Norml2() << ", ref.Norml2 = " << ref
        << ", err = " << err);
   REQUIRE(ref > 0.0);
   REQUIRE(err / ref < 1e-12);
}

TEST_CASE("Domain Integration (RT Vector Delta on Shared Face)",
          "[RT_FECollection]"
          "[LinearForm]"
          "[DeltaCoefficient]")
{
   Mesh mesh = OrientedTriFaceMesh(1);

   RT_FECollection fec(0, 3);
   FiniteElementSpace fes(&mesh, &fec);

   Vector x0(3);
   x0(0) = 0.0; x0(1) = 1.0 / 3.0; x0(2) = 1.0 / 3.0;
   Vector dir(3);
   dir = 0.0; dir(1) = 1.0;

   VectorDeltaCoefficient vdc(dir, x0(0), x0(1), x0(2), 1.0);
   LinearForm rhs(&fes);
   rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc));
   rhs.Assemble();

   Vector expected;
   AssembleDeltaReferenceBruteForce(fes, x0, dir, expected);

   Vector diff(rhs);
   diff -= expected;
   const real_t err = diff.Norml2();
   const real_t ref = expected.Norml2();
   INFO("rhs.Norml2 = " << rhs.Norml2() << ", ref.Norml2 = " << ref
        << ", err = " << err);
   REQUIRE(ref > 0.0);
   REQUIRE(err / ref < 1e-12);
}

TEST_CASE("Domain Integration (Scalar Delta on Shared Entity)",
          "[H1_FECollection]"
          "[LinearForm]"
          "[DeltaCoefficient]")
{
   const int entity_type = GENERATE(0, 1, 2);
   Vector x0(3);
   x0(0) = 0.5;
   switch (entity_type)
   {
      case 0:  x0(1) = 0.5;  x0(2) = 0.5;  break;  // face centroid
      case 1:  x0(1) = 0.5;  x0(2) = 0.0;  break;  // edge midpoint
      default: x0(1) = 0.0;  x0(2) = 0.0;  break;  // vertex
   }

   Mesh mesh = DividingPlaneMesh(false, false, true);
   REQUIRE(mesh.GetNE() == 2);
   REQUIRE(mesh.Conforming());

   H1_FECollection fec(2, 3);
   FiniteElementSpace fes(&mesh, &fec);

   DeltaCoefficient dc(x0(0), x0(1), x0(2), 1.0);
   LinearForm rhs(&fes);
   rhs.AddDomainIntegrator(new DomainLFIntegrator(dc));
   rhs.Assemble();

   FunctionCoefficient q(ScalarLinearField);
   GridFunction g(&fes);
   g.ProjectCoefficient(q);

   INFO("entity_type = " << entity_type << ", rhs(g) = " << rhs(g)
        << ", exact = " << ScalarLinearField(x0));
   REQUIRE(rhs(g) == MFEM_Approx(ScalarLinearField(x0)));
}

TEST_CASE("Domain Integration (Vector Delta on Nonconforming Interface)",
          "[ND_FECollection]"
          "[LinearForm]"
          "[DeltaCoefficient]")
{
   // Refine the shared face at representative vertex, edge, and sub-face
   // locations, then compare with the brute-force reference.
   const int entity_type = GENERATE(0, 1, 2, 3, 4, 5);
   const int levels = GENERATE(1, 2);

   const Vector x0 = SharedFacePoint(entity_type);
   Vector dir(3);
   dir = 0.0; dir(0) = 1.0;

   Mesh mesh = OrientedTriFaceMesh(1);
   mesh.EnsureNCMesh(true);
   for (int l = 0; l < levels; l++)
   {
      int target = -1;
      real_t best = infinity();
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         InverseElementTransformation inv(mesh.GetElementTransformation(e));
         IntegrationPoint ip;
         inv.Transform(x0, ip);
         if (Geometry::CheckPoint(mesh.GetElementBaseGeometry(e), ip, 1e-6))
         {
            const real_t vol = mesh.GetElementVolume(e);
            if (vol < best) { best = vol; target = e; }
         }
      }
      REQUIRE(target >= 0);
      Array<int> refine_list(1);
      refine_list[0] = target;
      mesh.GeneralRefinement(refine_list);
   }
   REQUIRE(mesh.Nonconforming());

   ND_FECollection fec(1, 3);
   FiniteElementSpace fes(&mesh, &fec);

   VectorDeltaCoefficient vdc(dir, x0(0), x0(1), x0(2), 1.0);
   LinearForm rhs(&fes);
   rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc));
   rhs.Assemble();

   Vector expected;
   AssembleDeltaReferenceBruteForce(fes, x0, dir, expected);

   Vector diff(rhs);
   diff -= expected;
   const real_t err = diff.Norml2();
   const real_t ref = expected.Norml2();
   INFO("entity_type = " << entity_type << ", levels = " << levels <<
        ", rhs.Norml2 = "
        << rhs.Norml2() << ", ref.Norml2 = " << ref << ", err = " << err);
   REQUIRE(ref > 0.0);
   REQUIRE(err / ref < 1e-12);
}

/** @brief Return representative points on the hexahedral shared face.

    The shared face is `(x=0.5)` in the two-hex mesh. After refinement makes it
    a master/slave interface, the selected points exercise distinct entity
    classes: 0 a master/conforming vertex, 1 a hanging vertex on a coarse edge,
    2 the hanging face-center vertex, 3 a hanging sub-edge on a coarse edge, 4
    an interior hanging edge, and 5 a slave sub-face interior. */
Vector HexSharedFacePoint(int entity_type)
{
   Vector p(3);
   p(0) = 0.5;
   switch (entity_type)
   {
      case 0:  p(1) = 0.0;  p(2) = 0.0;  break;
      case 1:  p(1) = 0.5;  p(2) = 0.0;  break;
      case 2:  p(1) = 0.5;  p(2) = 0.5;  break;
      case 3:  p(1) = 0.25; p(2) = 0.0;  break;
      case 4:  p(1) = 0.5;  p(2) = 0.25; break;
      default: p(1) = 0.25; p(2) = 0.25; break;
   }
   return p;
}

TEST_CASE("Domain Integration (Vector Delta on Hex Nonconforming Interface)",
          "[ND_FECollection]"
          "[LinearForm]"
          "[DeltaCoefficient]")
{
   // Refine the shared quad face at representative vertex, edge, and sub-face
   // locations, then compare with the brute-force reference.
   const int entity_type = GENERATE(0, 1, 2, 3, 4, 5);
   const int levels = GENERATE(1, 2);

   const Vector x0 = HexSharedFacePoint(entity_type);
   Vector dir(3);
   dir = 0.0; dir(0) = 1.0;

   Mesh mesh = DividingPlaneMesh(false, false, true);
   mesh.EnsureNCMesh();
   for (int l = 0; l < levels; l++)
   {
      int target = -1;
      real_t best = infinity();
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         InverseElementTransformation inv(mesh.GetElementTransformation(e));
         IntegrationPoint ip;
         inv.Transform(x0, ip);
         if (Geometry::CheckPoint(mesh.GetElementBaseGeometry(e), ip, 1e-6))
         {
            const real_t vol = mesh.GetElementVolume(e);
            if (vol < best) { best = vol; target = e; }
         }
      }
      REQUIRE(target >= 0);
      Array<int> refine_list(1);
      refine_list[0] = target;
      mesh.GeneralRefinement(refine_list);
   }
   REQUIRE(mesh.Nonconforming());

   ND_FECollection fec(1, 3);
   FiniteElementSpace fes(&mesh, &fec);

   VectorDeltaCoefficient vdc(dir, x0(0), x0(1), x0(2), 1.0);
   LinearForm rhs(&fes);
   rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc));
   rhs.Assemble();

   Vector expected;
   AssembleDeltaReferenceBruteForce(fes, x0, dir, expected);

   Vector diff(rhs);
   diff -= expected;
   const real_t err = diff.Norml2();
   const real_t ref = expected.Norml2();
   INFO("entity_type = " << entity_type << ", levels = " << levels <<
        ", rhs.Norml2 = "
        << rhs.Norml2() << ", ref.Norml2 = " << ref << ", err = " << err);
   REQUIRE(ref > 0.0);
   REQUIRE(err / ref < 1e-12);
}

#ifdef MFEM_USE_MPI

TEST_CASE("Domain Integration in Parallel (Scalar Field)",
          "[H1_FECollection]"
          "[L2_FECollection]"
          "[ParGridFunction]"
          "[ParLinearForm]"
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int order = 3;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt);
      int dim = mesh->Dimension();
      while (mesh->GetNE() < num_procs)
      {
         mesh->UniformRefinement();
      }
      ParMesh pmesh(MPI_COMM_WORLD, *mesh);
      delete mesh;

      ConstantCoefficient oneCoef(1.0);

      for (int ft = (int)FEType::H1_FEC; ft <= (int)FEType::L2I_FEC; ft++)
      {
         if (ft == (int)FEType::ND_FEC || ft == (int)FEType::RT_FEC)
         { continue; }

         SECTION("Integral of field " + std::to_string(ft) +
                 " on mesh type " + std::to_string(mt) )
         {
            FiniteElementCollection *fec = NULL;
            switch ((FEType)ft)
            {
               case FEType::H1_FEC:
                  fec = new H1_FECollection(order, dim);
                  break;
               case FEType::L2V_FEC:
                  fec = new L2_FECollection(order-1, dim);
                  break;
               case FEType::L2I_FEC:
                  fec = new L2_FECollection(order-1, dim,
                                            BasisType::GaussLegendre,
                                            FiniteElement::INTEGRAL);
                  break;
               default:
                  MFEM_ABORT("Invalid vector FE type");
            }
            ParFiniteElementSpace fespace(&pmesh, fec);

            ParGridFunction u(&fespace);
            u.ProjectCoefficient(oneCoef);

            ParLinearForm b(&fespace);
            b.AddDomainIntegrator(new DomainLFIntegrator(oneCoef));
            b.Assemble();

            double id = b(u);

            if (dim == 1)
            {
               REQUIRE(id == MFEM_Approx( 5.0));
            }
            else if (dim == 2)
            {
               REQUIRE(id == MFEM_Approx(15.0));
            }
            else
            {
               REQUIRE(id == MFEM_Approx(30.0));
            }

            delete fec;
         }
      }
   }
}

TEST_CASE("Domain Integration in Parallel (Vector Field)",
          "[ParGridFunction]"
          "[ParLinearForm]"
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int order = 3;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt);
      int  dim = mesh->Dimension();
      int sdim = mesh->SpaceDimension();
      while (mesh->GetNE() < num_procs)
      {
         mesh->UniformRefinement();
      }
      ParMesh pmesh(MPI_COMM_WORLD, *mesh);
      delete mesh;

      Vector f1(sdim); f1 = 1.0;
      Vector fx(sdim); fx = 0.0; fx[0] = 1.0;
      Vector fy(sdim); fy = 0.0;
      if (sdim > 1) { fy[1] = 1.0; }
      Vector fz(sdim); fz = 0.0;
      if (sdim > 2) { fz[2] = 1.0; }

      VectorConstantCoefficient f1Coef(f1);
      VectorConstantCoefficient fxCoef(fx);
      VectorConstantCoefficient fyCoef(fy);
      VectorConstantCoefficient fzCoef(fz);

      for (int ft = (int)FEType::ND_FEC; ft <= (int)FEType::RT_FEC; ft++)
      {
         if (dim == 1 && ft == (int)FEType::RT_FEC) { continue; }
         if (mt == (int)MeshType::WEDGE2 || mt == (int)MeshType::WEDGE4 ||
             mt == (int)MeshType::MIXED3D6 || mt == (int)MeshType::MIXED3D8)
         { continue; }

         SECTION("Integral of field " + std::to_string(ft) +
                 " on mesh type " + std::to_string(mt) )
         {
            FiniteElementCollection *fec = NULL;
            switch ((FEType)ft)
            {
               case FEType::ND_FEC:
                  fec = new ND_FECollection(order, dim);
                  break;
               case FEType::RT_FEC:
                  fec = new RT_FECollection(order-1, dim);
                  break;
               default:
                  MFEM_ABORT("Invalid vector FE type");
            }
            ParFiniteElementSpace fespace(&pmesh, fec);

            ParGridFunction u(&fespace);
            u.ProjectCoefficient(f1Coef);

            ParLinearForm bx(&fespace);
            ParLinearForm by(&fespace);
            ParLinearForm bz(&fespace);
            bx.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fxCoef));
            by.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fyCoef));
            bz.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fzCoef));
            bx.Assemble();
            by.Assemble();
            bz.Assemble();

            double ix = bx(u);
            double iy = by(u);
            double iz = bz(u);

            if (dim == 1)
            {
               REQUIRE(ix == MFEM_Approx( 5.0));
            }
            else if (dim == 2)
            {
               REQUIRE(ix == MFEM_Approx(15.0));
               REQUIRE(iy == MFEM_Approx(15.0));
            }
            else
            {
               REQUIRE(ix == MFEM_Approx(30.0));
               REQUIRE(iy == MFEM_Approx(30.0));
               REQUIRE(iz == MFEM_Approx(30.0));
            }

            delete fec;
         }
      }
   }
}

TEST_CASE("Domain Integration in Parallel (Vector Delta on Shared Vertex)",
          "[ParLinearForm]"
          "[DeltaCoefficient]"
          "[Parallel]")
{
   // Compare the parallel true-vector norm with the serial reference.
   Mesh mesh = MakeCenteredHex24Tets();

   Vector dir(3);
   dir = 0.0;
   dir[2] = 1.0;

   ND_FECollection fec_serial(1, 3);
   FiniteElementSpace fes_serial(&mesh, &fec_serial);
   Vector ref;
   AssembleUniformReference(fes_serial, dir, ref);
   const real_t ref_norm = ref.Norml2();
   REQUIRE(ref_norm > 0.0);

   VectorFunctionCoefficient field(3, GenericField);
   GridFunction g_serial(&fes_serial);
   g_serial.ProjectCoefficient(field);
   const real_t ref_action = ref * g_serial;

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   ND_FECollection fec(1, 3);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParGridFunction g(&pfes);
   g.ProjectCoefficient(field);

   VectorDeltaCoefficient vdc(dir, 0.0, 0.0, 0.0, 1.0);

   ParLinearForm rhs(&pfes);
   rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc));
   rhs.Assemble();

   Vector tv(pfes.GetTrueVSize());
   rhs.ParallelAssemble(tv);
   const real_t par_norm = GlobalLpNorm(2.0, tv.Norml2(), MPI_COMM_WORLD);
   const real_t par_action = rhs(g);

   INFO("nranks = " << Mpi::WorldSize() << ", ref = " << ref_norm
        << ", par = " << par_norm << ", ref_action = " << ref_action
        << ", par_action = " << par_action);
   REQUIRE(par_norm == MFEM_Approx(ref_norm));
   REQUIRE(par_action == MFEM_Approx(ref_action));
}

TEST_CASE("Domain Integration in Parallel (Vector Delta, Scattered Partition)",
          "[ParLinearForm]"
          "[DeltaCoefficient]"
          "[Parallel]")
{
   // Scatter the origin's element star across ranks.
   Mesh mesh = MakeCenteredHex24Tets();

   Vector dir(3);
   dir = 0.0;
   dir[2] = 1.0;

   ND_FECollection fec_serial(1, 3);
   FiniteElementSpace fes_serial(&mesh, &fec_serial);
   Vector ref;
   AssembleUniformReference(fes_serial, dir, ref);
   const real_t ref_norm = ref.Norml2();
   REQUIRE(ref_norm > 0.0);

   VectorFunctionCoefficient field(3, GenericField);
   GridFunction g_serial(&fes_serial);
   g_serial.ProjectCoefficient(field);
   const real_t ref_action = ref * g_serial;

   const int nranks = Mpi::WorldSize();
   Array<int> part(mesh.GetNE());
   for (int e = 0; e < mesh.GetNE(); e++) { part[e] = e % nranks; }
   ParMesh pmesh(MPI_COMM_WORLD, mesh, part.GetData());

   ND_FECollection fec(1, 3);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParGridFunction g(&pfes);
   g.ProjectCoefficient(field);

   VectorDeltaCoefficient vdc(dir, 0.0, 0.0, 0.0, 1.0);
   ParLinearForm rhs(&pfes);
   rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc));
   rhs.Assemble();

   Vector tv(pfes.GetTrueVSize());
   rhs.ParallelAssemble(tv);
   const real_t par_norm = GlobalLpNorm(2.0, tv.Norml2(), MPI_COMM_WORLD);
   const real_t par_action = rhs(g);

   INFO("nranks = " << nranks << ", ref = " << ref_norm
        << ", par = " << par_norm << ", ref_action = " << ref_action
        << ", par_action = " << par_action);
   REQUIRE(par_norm == MFEM_Approx(ref_norm));
   REQUIRE(par_action == MFEM_Approx(ref_action));
}

TEST_CASE("Domain Integration in Parallel (Vector Delta on Shared Face)",
          "[ParLinearForm]"
          "[DeltaCoefficient]"
          "[Parallel]")
{
   // Put the two elements containing the face delta on different ranks.
   const int nranks = Mpi::WorldSize();

   Mesh mesh = OrientedTriFaceMesh(1);

   Vector x0(3);
   x0(0) = 0.0; x0(1) = 1.0 / 3.0; x0(2) = 1.0 / 3.0;
   Vector dir(3);
   dir = 0.0; dir(0) = 1.0;

   real_t sym_ref;
   {
      ND_FECollection fec(1, 3);
      FiniteElementSpace fes(&mesh, &fec);
      VectorFunctionCoefficient Fc(3, GenericField);
      GridFunction g(&fes);
      g.ProjectCoefficient(Fc);
      Vector g0(3), g1(3);
      IntegrationPoint ip;
      InverseElementTransformation inv0(fes.GetElementTransformation(0));
      REQUIRE(inv0.Transform(x0, ip) == InverseElementTransformation::Inside);
      g.GetVectorValue(0, ip, g0);
      InverseElementTransformation inv1(fes.GetElementTransformation(1));
      REQUIRE(inv1.Transform(x0, ip) == InverseElementTransformation::Inside);
      g.GetVectorValue(1, ip, g1);
      sym_ref = 0.5 * ((dir * g0) + (dir * g1));
      REQUIRE(std::abs((dir * g0) - (dir * g1)) > 1e-3);
   }

   Array<int> partitioning(mesh.GetNE());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      partitioning[e] = (nranks >= 2) ? e % nranks : 0;
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh, partitioning.GetData());

   ND_FECollection fec(1, 3);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   VectorFunctionCoefficient Fc(3, GenericField);
   ParGridFunction g(&pfes);
   g.ProjectCoefficient(Fc);

   VectorDeltaCoefficient vdc(dir, x0(0), x0(1), x0(2), 1.0);
   ParLinearForm rhs(&pfes);
   rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc));
   rhs.Assemble();

   const real_t Lg = rhs(g);
   INFO("nranks = " << nranks << ", Lg = " << Lg << ", sym_ref = " << sym_ref);
   REQUIRE(Lg == MFEM_Approx(sym_ref));
}

TEST_CASE("Domain Integration in Parallel (Vector Delta on NC Interface)",
          "[ParLinearForm]"
          "[DeltaCoefficient]"
          "[Parallel]")
{
   // Partition the coarse master and fine slaves onto different ranks.
   const int entity_type = GENERATE(0, 1, 2, 3, 4, 5);
   const Vector x0 = SharedFacePoint(entity_type);
   Vector dir(3);
   dir = 0.0; dir(0) = 1.0;

   Mesh mesh = OrientedTriFaceMesh(1);
   mesh.EnsureNCMesh(true);
   {
      Array<int> refine_list(1);
      refine_list[0] = 0;
      mesh.GeneralRefinement(refine_list);
   }
   REQUIRE(mesh.Nonconforming());

   ND_FECollection fec(1, 3);
   VectorFunctionCoefficient field(3, ShiftedGenericField);

   // Reference: same assembly path on an unpartitioned COMM_SELF ParMesh.
   auto action = [&](MPI_Comm comm, int *part) -> real_t
   {
      ParMesh pmesh(comm, mesh, part);
      ParFiniteElementSpace pfes(&pmesh, &fec);
      ParGridFunction g(&pfes);
      g.ProjectCoefficient(field);
      VectorDeltaCoefficient vdc(dir, x0(0), x0(1), x0(2), 1.0);
      ParLinearForm rhs(&pfes);
      rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc));
      rhs.Assemble();
      Vector tv(pfes.GetTrueVSize());
      rhs.ParallelAssemble(tv);
      Vector gt(pfes.GetTrueVSize());
      g.GetTrueDofs(gt);
      return InnerProduct(comm, tv, gt);
   };

   const real_t serial_action = action(MPI_COMM_SELF, nullptr);

   // Keep the coarse master and fine slaves on different ranks.
   const int nranks = Mpi::WorldSize();
   real_t vmax = 0.0;
   for (int e = 0; e < mesh.GetNE(); e++)
   { vmax = std::max(vmax, mesh.GetElementVolume(e)); }
   Array<int> part(mesh.GetNE());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      part[e] = (mesh.GetElementVolume(e) > 0.5 * vmax) ? (nranks - 1) : 0;
   }
   const real_t par_action = action(MPI_COMM_WORLD, part.GetData());

   INFO("entity_type = " << entity_type << ", nranks = " << nranks << ", serial = "
        << serial_action << ", parallel = " << par_action);
   REQUIRE(par_action == MFEM_Approx(serial_action));
}

TEST_CASE("Domain Integration in Parallel (Vector Delta on Hex NC Interface)",
          "[ParLinearForm]"
          "[DeltaCoefficient]"
          "[Parallel]")
{
   // Partition the coarse master and fine slaves onto different ranks.
   const int entity_type = GENERATE(0, 1, 2, 3, 4, 5);
   const Vector x0 = HexSharedFacePoint(entity_type);
   Vector dir(3);
   dir = 0.0; dir(0) = 1.0;

   Mesh mesh = DividingPlaneMesh(false, false, true);
   mesh.EnsureNCMesh();
   {
      Array<int> refine_list(1);
      refine_list[0] = 0;
      mesh.GeneralRefinement(refine_list);
   }
   REQUIRE(mesh.Nonconforming());

   ND_FECollection fec(1, 3);
   VectorFunctionCoefficient field(3, ShiftedGenericField);

   auto action = [&](MPI_Comm comm, int *part) -> real_t
   {
      ParMesh pmesh(comm, mesh, part);
      ParFiniteElementSpace pfes(&pmesh, &fec);
      ParGridFunction g(&pfes);
      g.ProjectCoefficient(field);
      VectorDeltaCoefficient vdc(dir, x0(0), x0(1), x0(2), 1.0);
      ParLinearForm rhs(&pfes);
      rhs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vdc));
      rhs.Assemble();
      Vector tv(pfes.GetTrueVSize());
      rhs.ParallelAssemble(tv);
      Vector gt(pfes.GetTrueVSize());
      g.GetTrueDofs(gt);
      return InnerProduct(comm, tv, gt);
   };

   const real_t serial_action = action(MPI_COMM_SELF, nullptr);

   const int nranks = Mpi::WorldSize();
   real_t vmax = 0.0;
   for (int e = 0; e < mesh.GetNE(); e++)
   { vmax = std::max(vmax, mesh.GetElementVolume(e)); }
   Array<int> part(mesh.GetNE());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      part[e] = (mesh.GetElementVolume(e) > 0.5 * vmax) ? (nranks - 1) : 0;
   }
   const real_t par_action = action(MPI_COMM_WORLD, part.GetData());

   INFO("entity_type = " << entity_type << ", nranks = " << nranks << ", serial = "
        << serial_action << ", parallel = " << par_action);
   REQUIRE(par_action == MFEM_Approx(serial_action));
}

#endif // MFEM_USE_MPI

Mesh * GetMesh(MeshType type)
{
   Mesh * mesh = NULL;

   switch (type)
   {
      case SEGMENT:
         mesh = new Mesh(1, 2, 1);
         mesh->AddVertex(0.0);
         mesh->AddVertex(a_);

         mesh->AddSegment(0, 1);

         mesh->AddBdrPoint(0);
         mesh->AddBdrPoint(1);
         break;
      case QUADRILATERAL:
         mesh = new Mesh(2, 4, 1);
         mesh->AddVertex(0.0, 0.0);
         mesh->AddVertex(a_, 0.0);
         mesh->AddVertex(a_, b_);
         mesh->AddVertex(0.0, b_);

         mesh->AddQuad(0, 1, 2, 3);
         break;
      case TRIANGLE2A:
         mesh = new Mesh(2, 4, 2);
         mesh->AddVertex(0.0, 0.0);
         mesh->AddVertex(a_, 0.0);
         mesh->AddVertex(a_, b_);
         mesh->AddVertex(0.0, b_);

         mesh->AddTriangle(0, 1, 2);
         mesh->AddTriangle(2, 3, 0);
         break;
      case TRIANGLE2B:
         mesh = new Mesh(2, 4, 2);
         mesh->AddVertex(0.0, 0.0);
         mesh->AddVertex(a_, 0.0);
         mesh->AddVertex(a_, b_);
         mesh->AddVertex(0.0, b_);

         mesh->AddTriangle(1, 2, 0);
         mesh->AddTriangle(3, 0, 2);
         break;
      case TRIANGLE2C:
         mesh = new Mesh(2, 4, 2);
         mesh->AddVertex(0.0, 0.0);
         mesh->AddVertex(a_, 0.0);
         mesh->AddVertex(a_, b_);
         mesh->AddVertex(0.0, b_);

         mesh->AddTriangle(2, 0, 1);
         mesh->AddTriangle(0, 2, 3);
         break;
      case TRIANGLE4:
         mesh = new Mesh(2, 5, 4);
         mesh->AddVertex(0.0, 0.0);
         mesh->AddVertex(a_, 0.0);
         mesh->AddVertex(a_, b_);
         mesh->AddVertex(0.0, b_);
         mesh->AddVertex(0.5 * a_, 0.5 * b_);

         mesh->AddTriangle(0, 1, 4);
         mesh->AddTriangle(1, 2, 4);
         mesh->AddTriangle(2, 3, 4);
         mesh->AddTriangle(3, 0, 4);
         break;
      case MIXED2D:
         mesh = new Mesh(2, 6, 4);
         mesh->AddVertex(0.0, 0.0);
         mesh->AddVertex(a_, 0.0);
         mesh->AddVertex(a_, b_);
         mesh->AddVertex(0.0, b_);
         mesh->AddVertex(0.5 * b_, 0.5 * b_);
         mesh->AddVertex(a_ - 0.5 * b_, 0.5 * b_);

         mesh->AddQuad(0, 1, 5, 4);
         mesh->AddTriangle(1, 2, 5);
         mesh->AddQuad(2, 3, 4, 5);
         mesh->AddTriangle(3, 0, 4);
         break;
      case HEXAHEDRON:
         mesh = new Mesh(3, 8, 1);
         mesh->AddVertex(0.0, 0.0, 0.0);
         mesh->AddVertex(a_, 0.0, 0.0);
         mesh->AddVertex(a_, b_, 0.0);
         mesh->AddVertex(0.0, b_, 0.0);
         mesh->AddVertex(0.0, 0.0, c_);
         mesh->AddVertex(a_, 0.0, c_);
         mesh->AddVertex(a_, b_, c_);
         mesh->AddVertex(0.0, b_, c_);

         mesh->AddHex(0, 1, 2, 3, 4, 5, 6, 7);
         break;
      case HEXAHEDRON2A:
      case HEXAHEDRON2B:
      case HEXAHEDRON2C:
      case HEXAHEDRON2D:
         mesh = new Mesh(3, 12, 2);
         mesh->AddVertex(0.0, 0.0, 0.0);
         mesh->AddVertex(0.5 * a_, 0.0, 0.0);
         mesh->AddVertex(a_, 0.0, 0.0);
         mesh->AddVertex(a_, b_, 0.0);
         mesh->AddVertex(0.5 * a_, b_, 0.0);
         mesh->AddVertex(0.0, b_, 0.0);
         mesh->AddVertex(0.0, 0.0, c_);
         mesh->AddVertex(0.5 * a_, 0.0, c_);
         mesh->AddVertex(a_, 0.0, c_);
         mesh->AddVertex(a_, b_, c_);
         mesh->AddVertex(0.5 * a_, b_, c_);
         mesh->AddVertex(0.0,b_, c_);

         mesh->AddHex(0, 5, 11, 6, 1, 4, 10, 7);

         switch (type)
         {
            case HEXAHEDRON2A: // Face Orientation 1
               mesh->AddHex(4, 10, 7, 1, 3, 9, 8, 2);
               break;
            case HEXAHEDRON2B: // Face Orientation 3
               mesh->AddHex(10, 7, 1, 4, 9, 8, 2, 3);
               break;
            case HEXAHEDRON2C: // Face Orientation 5
               mesh->AddHex(7, 1, 4, 10, 8, 2, 3, 9);
               break;
            case HEXAHEDRON2D: // Face Orientation 7
               mesh->AddHex(1, 4, 10, 7, 2, 3, 9, 8);
               break;
            default:
               // Cannot happen
               break;
         }
         break;
      case WEDGE2:
         mesh = new Mesh(3, 8, 2);
         mesh->AddVertex(0.0, 0.0, 0.0);
         mesh->AddVertex(a_, 0.0, 0.0);
         mesh->AddVertex(a_, b_, 0.0);
         mesh->AddVertex(0.0, b_, 0.0);
         mesh->AddVertex(0.0, 0.0, c_);
         mesh->AddVertex(a_, 0.0, c_);
         mesh->AddVertex(a_, b_, c_);
         mesh->AddVertex(0.0, b_, c_);

         mesh->AddWedge(0, 1, 2, 4, 5, 6);
         mesh->AddWedge(0, 2, 3, 4, 6, 7);
         break;
      case TETRAHEDRA:
         mesh = new Mesh(3, 8, 5);
         mesh->AddVertex(0.0, 0.0, 0.0);
         mesh->AddVertex(a_, 0.0, 0.0);
         mesh->AddVertex(a_, b_, 0.0);
         mesh->AddVertex(0.0, b_, 0.0);
         mesh->AddVertex(0.0, 0.0, c_);
         mesh->AddVertex(a_, 0.0, c_);
         mesh->AddVertex(a_, b_, c_);
         mesh->AddVertex(0.0, b_, c_);

         mesh->AddTet(0, 2, 7, 5);
         mesh->AddTet(6, 7, 2, 5);
         mesh->AddTet(4, 7, 5, 0);
         mesh->AddTet(1, 0, 5, 2);
         mesh->AddTet(3, 7, 0, 2);
         break;
      case WEDGE4:
         mesh = new Mesh(3, 10, 4);
         mesh->AddVertex(0.0, 0.0, 0.0);
         mesh->AddVertex(a_, 0.0, 0.0);
         mesh->AddVertex(a_, b_, 0.0);
         mesh->AddVertex(0.0, b_, 0.0);
         mesh->AddVertex(0.5 * a_, 0.5 * b_, 0.0);
         mesh->AddVertex(0.0, 0.0, c_);
         mesh->AddVertex(a_, 0.0, c_);
         mesh->AddVertex(a_, b_, c_);
         mesh->AddVertex(0.0, b_, c_);
         mesh->AddVertex(0.5 * a_, 0.5 * b_, c_);

         mesh->AddWedge(0, 1, 4, 5, 6, 9);
         mesh->AddWedge(1, 2, 4, 6, 7, 9);
         mesh->AddWedge(2, 3, 4, 7, 8, 9);
         mesh->AddWedge(3, 0, 4, 8, 5, 9);
         break;
      case MIXED3D6:
         mesh = new Mesh(3, 12, 6);
         mesh->AddVertex(0.0, 0.0, 0.0);
         mesh->AddVertex(a_, 0.0, 0.0);
         mesh->AddVertex(a_, b_, 0.0);
         mesh->AddVertex(0.0, b_, 0.0);
         mesh->AddVertex(0.5 * c_, 0.5 * c_, 0.5 * c_);
         mesh->AddVertex(a_ - 0.5 * c_, 0.5 * c_, 0.5 * c_);
         mesh->AddVertex(a_ - 0.5 * c_, b_ - 0.5 * c_, 0.5 * c_);
         mesh->AddVertex(0.5 * c_, b_ - 0.5 * c_, 0.5 * c_);
         mesh->AddVertex(0.0, 0.0, c_);
         mesh->AddVertex(a_, 0.0, c_);
         mesh->AddVertex(a_, b_, c_);
         mesh->AddVertex(0.0, b_, c_);

         mesh->AddHex(0, 1, 2, 3, 4, 5, 6, 7);
         mesh->AddWedge(0, 4, 8, 1, 5, 9);
         mesh->AddWedge(1, 5, 9, 2, 6, 10);
         mesh->AddWedge(2, 6, 10, 3, 7, 11);
         mesh->AddWedge(3, 7, 11, 0, 4, 8);
         mesh->AddHex(4, 5, 6, 7, 8, 9, 10, 11);
         break;
      case MIXED3D8:
         mesh = new Mesh(3, 10, 8);
         mesh->AddVertex(0.0, 0.0, 0.0);
         mesh->AddVertex(a_, 0.0, 0.0);
         mesh->AddVertex(a_, b_, 0.0);
         mesh->AddVertex(0.0, b_, 0.0);
         mesh->AddVertex(0.25 * a_, 0.5 * b_, 0.5 * c_);
         mesh->AddVertex(0.75 * a_, 0.5 * b_, 0.5 * c_);
         mesh->AddVertex(0.0, 0.0, c_);
         mesh->AddVertex(a_, 0.0, c_);
         mesh->AddVertex(a_, b_, c_);
         mesh->AddVertex(0.0, b_, c_);

         mesh->AddWedge(0, 3, 4, 1, 2, 5);
         mesh->AddWedge(3, 9, 4, 2, 8, 5);
         mesh->AddWedge(9, 6, 4, 8, 7, 5);
         mesh->AddWedge(6, 0, 4, 7, 1, 5);
         mesh->AddTet(0, 3, 9, 4);
         mesh->AddTet(0, 9, 6, 4);
         mesh->AddTet(1, 7, 2, 5);
         mesh->AddTet(8, 2, 7, 5);
         break;
      case PYRAMID:
         mesh = new Mesh(3, 9, 6);
         mesh->AddVertex(0.0, 0.0, 0.0);
         mesh->AddVertex(a_, 0.0, 0.0);
         mesh->AddVertex(a_, b_, 0.0);
         mesh->AddVertex(0.0, b_, 0.0);
         mesh->AddVertex(0.5 * a_, 0.5 * b_, 0.5 * c_);
         mesh->AddVertex(0.0, 0.0, c_);
         mesh->AddVertex(a_, 0.0, c_);
         mesh->AddVertex(a_, b_, c_);
         mesh->AddVertex(0.0, b_, c_);

         mesh->AddPyramid(0, 1, 2, 3, 4);
         mesh->AddPyramid(0, 5, 6, 1, 4);
         mesh->AddPyramid(1, 6, 7, 2, 4);
         mesh->AddPyramid(2, 7, 8, 3, 4);
         mesh->AddPyramid(3, 8, 5, 0, 4);
         mesh->AddPyramid(8, 7, 6, 5, 4);
         break;
   }
   mesh->FinalizeTopology();

   return mesh;
}

} // namespace domain_int
