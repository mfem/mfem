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
#include "common_get_mesh.hpp"

using namespace mfem_test_fem;

namespace mfem
{

namespace build_dof_to_arrays
{

static double a_ = M_PI;
static double b_ = M_PI / sqrt(2.0);
static double c_ = M_PI / 2.0;

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
      Mesh *mesh = GetMesh((MeshType)mt, a_, b_, c_);
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
            int size = fespace.GetTrueVSize();

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
      Mesh *mesh = GetMesh((MeshType)mt, a_, b_, c_);
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

} // namespace build_dof_to_arrays

} // namespace mfem
