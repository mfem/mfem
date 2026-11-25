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
#include "fem/normal_deriv_restriction.hpp"
#include "unit_tests.hpp"

using namespace mfem;

Mesh MakeCartesianMesh(int nx, int dim)
{
   if (dim == 2)
   {
      return Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL, true);
   }
   else
   {
      return Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON);
   }
}

namespace face_restriction_test { enum class SpaceType {RT, ND}; }

TEST_CASE("Vector FE Face Restriction", "[FaceRestriction]")
{
   using namespace face_restriction_test;

   const auto space_type = GENERATE(SpaceType::RT, SpaceType::ND);
   const int dim = GENERATE(2, 3);
   const int nx = 3;
   const int order = 4;

   CAPTURE(dim);

   Mesh mesh = MakeCartesianMesh(nx, dim);

   int ndof_per_face;
   std::unique_ptr<FiniteElementCollection> fec;
   if (space_type == SpaceType::RT)
   {
      fec.reset(new RT_FECollection(order-1, dim));
      ndof_per_face = int(pow(order, dim-1));
   }
   else
   {
      fec.reset(new ND_FECollection(order, dim));
      ndof_per_face = (dim - 1)*order*int(pow(order + 1, dim - 2));
   }

   FiniteElementSpace fes(&mesh, fec.get());

   auto ordering = ElementDofOrdering::LEXICOGRAPHIC;
   auto ftype = FaceType::Boundary;
   const int nfaces = fes.GetNFbyType(FaceType::Boundary);
   const FaceRestriction *face_restr =
      fes.GetFaceRestriction(ordering, ftype);

   REQUIRE(face_restr != nullptr);

   Array<int> bdr_dofs;
   fes.GetBoundaryTrueDofs(bdr_dofs);

   // Set gf to have random values on the boundary, zero on the interior
   GridFunction gf(&fes);
   gf.Randomize(0);
   gf.SetSubVectorComplement(bdr_dofs, 0.0);

   // Mapping to face E-vector and back to L-vector should give back the
   // original grid function.
   Vector face_vec(face_restr->Height());
   REQUIRE(face_vec.Size() == nfaces*ndof_per_face);
   face_restr->Mult(gf, face_vec);
   face_vec.HostReadWrite();

   if (space_type == SpaceType::ND && dim == 3)
   {
      // Adjust for multiplicity. In all other cases, each boundary DOF is
      // unique (not shared between faces). In the case of 3D ND elements, some
      // boundary DOFs are shared between two faces (i.e. those that lie on
      // element edges).
      //
      // This adjustment will ensure that the original vector is recovered after
      // multiplying by the transpose of the face restriction operator.
      //
      // Note that this assumes the mesh contains only hexahedral elements.
      const int n = order*(order+1);
      for (int f = 0; f < fes.GetNFbyType(ftype); ++f)
      {
         for (int d = 0; d < 2; ++d)
         {
            const int mx = (d == 0) ? order : order + 1;
            const int my = (d == 0) ? order + 1 : order;
            for (int i = 0; i < n; ++i)
            {
               const int ix = i % mx;
               const int iy = i / mx;
               if ((d == 0 && (iy == 0 || iy == my - 1)) ||
                   (d == 1 && (ix == 0 || ix == mx - 1)))
               {
                  face_vec[f*ndof_per_face + d*n + i] *= 0.5;
               }
            }
         }
      }
   }

   GridFunction gf2(&fes);
   face_restr->MultTranspose(face_vec, gf2);

   gf2 -= gf;
   REQUIRE(gf2.Normlinf() == MFEM_Approx(0.0));
}

#ifdef MFEM_USE_MPI

TEST_CASE("L2 Face Restriction", "[FaceRestriction][Parallel]")
{
   const int dim = GENERATE(2, 3);
   constexpr int nx = 3;
   constexpr int order = 2;
   constexpr int vdim = 2;
   const Ordering::Type ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);

   Mesh serial_mesh = MakeCartesianMesh(nx, dim);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);

   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&mesh, &fec, vdim, ordering);

   auto *R = fes.GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC,
                                    FaceType::Interior);

   Vector vals({1.0, 2.0});
   VectorConstantCoefficient coeff(vals);

   ParGridFunction gf(&fes);
   gf.ProjectCoefficient(coeff);

   Vector face_vec(R->Height());
   R->Mult(gf, face_vec);

   const int nf = mesh.GetNFbyType(FaceType::Interior);
   const int face_dofs = fes.GetTypicalTraceElement()->GetDof();
   auto h_face_vec = Reshape(face_vec.HostRead(), face_dofs, vdim, 2, nf);

   for (int f = 0; f < nf; ++f)
   {
      for (int m = 0; m < 2; ++m)
      {
         for (int c = 0; c < vdim; ++c)
         {
            for (int i = 0; i < face_dofs; ++i)
            {
               REQUIRE(h_face_vec(i, c, m, f) == vals[c]);
            }
         }
      }
   }
}

#endif

TEST_CASE("Normal derivative restriction specializations and fallback",
          "[FaceRestriction][KernelDispatch][GPU]")
{
   const int dim = GENERATE(2, 3);
   // D1D=2 and D1D=8 are registered; D1D=9 exercises the runtime fallback.
   const int order = GENERATE(1, 7, 8);
   const auto ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);
   const auto face_type = GENERATE(FaceType::Interior, FaceType::Boundary);
   CAPTURE(dim, order, ordering, face_type);

   Mesh mesh = MakeCartesianMesh(2, dim);
   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 1, ordering);
   L2NormalDerivativeFaceRestriction restriction(
      fes, ElementDofOrdering::LEXICOGRAPHIC, face_type);
   FunctionCoefficient coefficient([dim](const Vector &x)
   {
      real_t sum = 0.0;
      for (int j = 0; j < dim; ++j) { sum += x[j]; }
      return sum;
   });
   GridFunction x(&fes);
   x.ProjectCoefficient(coefficient);
   const int nf = fes.GetNFbyType(face_type);
   const int nd = dim == 2 ? order + 1 : (order + 1)*(order + 1);
   Vector rx(nd*2*nf);
   restriction.Mult(x, rx);
   const auto derivatives = Reshape(rx.HostRead(), nd, 1, 2, nf);
   for (int f = 0; f < nf; ++f)
   {
      for (int side = 0; side < 2; ++side)
      {
         const real_t expected = face_type == FaceType::Boundary && side == 1 ?
                                 0.0 : 0.5;
         for (int d = 0; d < nd; ++d)
         {
            REQUIRE(derivatives(d,0,side,f) == MFEM_Approx(expected));
         }
      }
   }

   // Check the transpose and its accumulation factor independently through
   // the adjoint identity, starting from a nonzero destination.
   Vector y(rx.Size()), z(fes.GetVSize()), initial(fes.GetVSize());
   y.Randomize(1);
   initial.Randomize(2);
   z = initial;
   const real_t a = -0.75;
   restriction.AddMultTranspose(y, z, a);
   z -= initial;
   REQUIRE((x*z) == MFEM_Approx(a*(rx*y)));
}
