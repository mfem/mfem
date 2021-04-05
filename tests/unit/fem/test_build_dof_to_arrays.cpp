// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

namespace for_dof
{

static double a_ = M_PI;
static double b_ = M_PI / sqrt(2.0);
static double c_ = M_PI / 2.0;

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
   MIXED3D8 = 16
};

Mesh * GetMesh(MeshType type);

enum BasisType
{
   H1 = 0, ND = 1, RT = 2, L2 = 3
};

TEST_CASE("Build Dof To Arrays",
          "[BuildDofToArrays]"
          "[FiniteElementSpace]")
{
   int order = 3;

   Array<int> dofs;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt);
      int  dim = mesh->Dimension();
      if (dim < 3 ||
          mt == MeshType::HEXAHEDRON ||
          mt == MeshType::WEDGE2     ||
          mt == MeshType::TETRAHEDRA ||
          mt == MeshType::WEDGE4     ||
          mt == MeshType::MIXED3D8 )
      {
         mesh->UniformRefinement();
      }
      if (dim == 3)
      {
         mesh->ReorientTetMesh();
      }

      for (int bt = (int)BasisType::H1; bt <= (int)BasisType::L2; bt++)
      {
         if (dim == 1 && bt == (int)BasisType::ND) { continue; }
         if (dim == 1 && bt == (int)BasisType::RT) { continue; }
         // FIX THIS: ND Wedge elements are not yet implemented
         if ((mt == (int)MeshType::WEDGE2 ||
              mt == (int)MeshType::MIXED3D6 ||
              mt == (int)MeshType::MIXED3D8) &&
             bt == (int)BasisType::ND) { continue; }

         int num_elem_fails = 0;
         int num_rang_fails = 0;
         int num_ldof_fails = 0;

         SECTION("Mesh Type: " + std::to_string(mt) +
                 ", Basis Type: " + std::to_string(bt))
         {
            FiniteElementCollection * fec = NULL;
            if (bt == (int)BasisType::H1)
            {
               fec = new H1_FECollection(order, dim);
            }
            else if (bt == (int)BasisType::ND)
            {
               fec = new ND_FECollection(order, dim);
            }
            else if (bt == (int)BasisType::RT)
            {
               fec = new RT_FECollection(order-1, dim);
            }
            else
            {
               fec = new L2_FECollection(order-1, dim);
            }
            FiniteElementSpace fespace(mesh, fec);
            HYPRE_Int size = fespace.GetTrueVSize();

            fespace.BuildDofToArrays();

            for (int i = 0; i<size; i++)
            {
               int e = fespace.GetElementForDof(i);
               int l = fespace.GetLocalDofForDof(i);

               if (e < 0 || e >= mesh->GetNE()) { num_elem_fails++; }

               fespace.GetElementDofs(e, dofs);

               if (l < 0 || l >= dofs.Size()) { num_rang_fails++; }

               int ldof = (dofs[l] >= 0) ? dofs[l] : (-1 - dofs[l]);

               if (i != ldof) { num_ldof_fails++; }
            }
            delete fec;
         }
         REQUIRE(num_elem_fails == 0);
         REQUIRE(num_rang_fails == 0);
         REQUIRE(num_ldof_fails == 0);
      }
   }
}

#ifdef MFEM_USE_MPI
#
TEST_CASE("Build Dof To Arrays (Parallel)",
          "[BuildDofToArrays]"
          "[ParFiniteElementSpace]"
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int order = 3;

   Array<int> dofs;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt);
      int  dim = mesh->Dimension();
      if (dim < 3 ||
          mt == MeshType::HEXAHEDRON ||
          mt == MeshType::WEDGE2     ||
          mt == MeshType::TETRAHEDRA ||
          mt == MeshType::WEDGE4     ||
          mt == MeshType::MIXED3D8 )
      {
         mesh->UniformRefinement();
      }
      while (mesh->GetNE() < num_procs)
      {
         mesh->UniformRefinement();
      }
      ParMesh pmesh(MPI_COMM_WORLD, *mesh);
      delete mesh;
      if (dim == 3)
      {
         pmesh.ReorientTetMesh();
      }

      for (int bt = (int)BasisType::H1; bt <= (int)BasisType::L2; bt++)
      {
         if (dim == 1 && bt == (int)BasisType::ND) { continue; }
         if (dim == 1 && bt == (int)BasisType::RT) { continue; }
         // FIX THIS: ND Wedge elements are not yet implemented
         if ((mt == (int)MeshType::WEDGE2 ||
              mt == (int)MeshType::MIXED3D6 ||
              mt == (int)MeshType::MIXED3D8) &&
             bt == (int)BasisType::ND) { continue; }

         int num_elem_fails = 0;
         int num_rang_fails = 0;
         int num_ldof_fails = 0;

         SECTION("Mesh Type: " + std::to_string(mt) +
                 ", Basis Type: " + std::to_string(bt))
         {
            FiniteElementCollection * fec = NULL;
            if (bt == (int)BasisType::H1)
            {
               fec = new H1_FECollection(order, dim);
            }
            else if (bt == (int)BasisType::ND)
            {
               fec = new ND_FECollection(order, dim);
            }
            else if (bt == (int)BasisType::RT)
            {
               fec = new RT_FECollection(order-1, dim);
            }
            else
            {
               fec = new L2_FECollection(order-1, dim);
            }
            ParFiniteElementSpace fespace(&pmesh, fec);
            HYPRE_Int size = fespace.GetTrueVSize();

            fespace.BuildDofToArrays();

            for (int i = 0; i<size; i++)
            {
               int e = fespace.GetElementForDof(i);
               int l = fespace.GetLocalDofForDof(i);

               if (e < 0 || e >= pmesh.GetNE()) { num_elem_fails++; }

               fespace.GetElementDofs(e, dofs);

               if (l < 0 || l >= dofs.Size()) { num_rang_fails++; }

               int ldof = (dofs[l] >= 0) ? dofs[l] : (-1 - dofs[l]);

               if (i != ldof) { num_ldof_fails++; }
            }
            delete fec;
         }
         REQUIRE(num_elem_fails == 0);
         REQUIRE(num_rang_fails == 0);
         REQUIRE(num_ldof_fails == 0);
      }
   }
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
   }
   mesh->FinalizeTopology();

   return mesh;
}

} // namespace for_dof

} // namespace mfem
