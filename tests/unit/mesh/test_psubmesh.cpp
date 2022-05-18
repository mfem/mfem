// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

using namespace mfem;

#ifdef MFEM_USE_MPI

namespace ParSubMeshTests
{
enum FECType
{
   H1,
   L2
};

FiniteElementCollection *create_fec(FECType fectype, int p, int dim)
{
   switch (fectype)
   {
      case H1:
         return new H1_FECollection(p, dim);
         break;
      case L2:
         return new L2_FECollection(p, dim);
         break;
   }

   return nullptr;
}

TEST_CASE("ParSubMesh", "[Parallel],[ParSubMesh]")
{
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   // Circle: sideset 1
   // Domain boundary: sideset 2
   Mesh *serial_parent_mesh = new Mesh("multidomain.mesh");
   ParMesh parent_mesh(MPI_COMM_WORLD, *serial_parent_mesh);
   delete serial_parent_mesh;

   Array<int> cylinder_domain_attributes(1);
   cylinder_domain_attributes[0] = 1;

   Array<int> outer_domain_attributes(1);
   outer_domain_attributes[0] = 2;

   Array<int> cylinder_surface_attributes(1);
   cylinder_surface_attributes[0] = 9;

   auto cylinder_submesh = ParSubMesh::CreateFromDomain(parent_mesh,
                                                        cylinder_domain_attributes);

   auto outer_submesh = ParSubMesh::CreateFromDomain(parent_mesh,
                                                     outer_domain_attributes);

   auto cylinder_surface_submesh = ParSubMesh::CreateFromBoundary(parent_mesh,
                                                                  cylinder_surface_attributes);

   auto fec_type = GENERATE(FECType::H1, FECType::L2);
   FiniteElementCollection *fec = create_fec(fec_type, 2, 3);

   ParFiniteElementSpace parent_fes(&parent_mesh, fec);
   ParGridFunction parent_gf(&parent_fes);

   ParFiniteElementSpace cylinder_fes(&cylinder_submesh, fec);
   ParGridFunction cylinder_gf(&cylinder_fes);

   ParFiniteElementSpace outer_fes(&outer_submesh, fec);
   ParGridFunction outer_gf(&outer_fes);

   ParFiniteElementSpace cylinder_surface_fes(&cylinder_surface_submesh, fec);
   ParGridFunction cylinder_surface_gf(&cylinder_surface_fes);

   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      double z = coords(2);
      return y + 0.05 * sin(x * 2.0 * M_PI) + z;
   });

   parent_gf.ProjectCoefficient(coeff);

   Vector tmp;

   ParGridFunction parent_gf_ex(&parent_fes);
   parent_gf_ex.ProjectCoefficient(coeff);

   ParGridFunction cylinder_gf_ex(&cylinder_fes);
   cylinder_gf_ex.ProjectCoefficient(coeff);

   ParGridFunction cylinder_surface_gf_ex(&cylinder_surface_fes);
   if (fec_type != FECType::L2)
   {
      cylinder_surface_gf_ex.ProjectCoefficient(coeff);
   }

   ParGridFunction outer_gf_ex(&outer_fes);
   outer_gf_ex.ProjectCoefficient(coeff);

   auto CHECK_GLOBAL_NORM = [](Vector &v)
   {
      double norm_local = v.Norml2(), norm_global = 0.0;
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      REQUIRE(norm_global < 1e-8);
   };

   SECTION("ParentToSubMesh")
   {
      SECTION("Volume to matching volume")
      {
         ParSubMesh::Transfer(parent_gf, cylinder_gf);
         tmp = cylinder_gf_ex;
         tmp -= cylinder_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      if (fec_type != FECType::L2)
      {
         SECTION("Volume to matching surface")
         {
            ParSubMesh::Transfer(parent_gf, cylinder_surface_gf);
            tmp = cylinder_surface_gf_ex;
            tmp -= cylinder_surface_gf;
            CHECK_GLOBAL_NORM(tmp);
         }
      }
   }
   SECTION("SubMeshToParent")
   {
      SECTION("Volume to matching volume")
      {
         parent_gf.ProjectCoefficient(coeff);
         cylinder_gf.ProjectCoefficient(coeff);
         ParSubMesh::Transfer(cylinder_gf, parent_gf);
         tmp = parent_gf_ex;
         tmp -= parent_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Volume to matching volume")
      {
         outer_gf.ProjectCoefficient(coeff);
         ParSubMesh::Transfer(outer_gf, parent_gf);
         tmp = parent_gf_ex;
         tmp -= parent_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      if (fec_type != FECType::L2)
      {
         SECTION("Surface to matching surface in volume")
         {
            cylinder_surface_gf.ProjectCoefficient(coeff);
            ParSubMesh::Transfer(cylinder_surface_gf, parent_gf);
            tmp = parent_gf_ex;
            tmp -= parent_gf;
            CHECK_GLOBAL_NORM(tmp);
         }
      }
   }
   SECTION("SubMeshToSubMesh")
   {
      SECTION("Volume to matching volume")
      {
         cylinder_gf.ProjectCoefficient(coeff);
         outer_gf.ProjectCoefficient(coeff);
         outer_gf_ex.ProjectCoefficient(coeff);
         ParSubMesh::Transfer(cylinder_gf, outer_gf);
         tmp = outer_gf_ex;
         tmp -= outer_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Volume to matching surface on volume")
      {
         cylinder_gf.ProjectCoefficient(coeff);
         outer_gf.ProjectCoefficient(coeff);
         cylinder_gf_ex.ProjectCoefficient(coeff);
         ParSubMesh::Transfer(outer_gf, cylinder_gf);
         tmp = cylinder_gf_ex;
         tmp -= cylinder_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      if (fec_type != FECType::L2)
      {
         SECTION("Volume to matching surface")
         {
            cylinder_gf.ProjectCoefficient(coeff);
            cylinder_surface_gf_ex.ProjectCoefficient(coeff);
            ParSubMesh::Transfer(cylinder_gf, cylinder_surface_gf);
            tmp = cylinder_surface_gf_ex;
            tmp -= cylinder_surface_gf;
            CHECK_GLOBAL_NORM(tmp);
         }
      }
   }
}

} // namespace ParSubMeshTests

#endif // MFEM_USE_MPI
