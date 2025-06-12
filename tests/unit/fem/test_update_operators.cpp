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

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

#ifdef MFEM_USE_MPI

TEST_CASE("Parallel Update Operators", "[Parallel]")
{
   const int dim = GENERATE(2, 3);

   Mesh mesh = [&]() -> Mesh
   {
      switch (dim)
      {
         case 2:
         {
            const int ne_1d = 9;
            const Element::Type elem_type = Element::QUADRILATERAL;
            const int gen_edges = 1;
            const bool sfc_ordering = true;
            return Mesh::MakeCartesian2D(ne_1d, ne_1d, elem_type, gen_edges,
                                         1.0, 1.0, sfc_ordering);
         }
         case 3:
         {
            const int ne_1d = 5;
            const Element::Type elem_type = Element::HEXAHEDRON;
            const bool sfc_ordering = true;
            return Mesh::MakeCartesian3D(ne_1d, ne_1d, ne_1d, elem_type,
                                         1.0, 1.0, 1.0, sfc_ordering);
         }
         default: MFEM_ABORT("unsupported dimension: " << dim);
      }
   }();

   INFO("dimension: " << dim);
   INFO("geomtry type: " << Geometry::Name[mesh.GetElementGeometry(0)]);

   const bool nc_simplices = true;
   mesh.EnsureNCMesh(nc_simplices);

   const int sdim = mesh.SpaceDimension();
   Vector bbox_min, bbox_max, bbox_center(sdim), bbox_scale(sdim);
   const int mesh_order = !mesh.GetNodes() ? 1 :
                          mesh.GetNodalFESpace()->GetMaxElementOrder();
   mesh.GetBoundingBox(bbox_min, bbox_max, 2*mesh_order);
   add(0.5_r, bbox_min, bbox_max, bbox_center);
   subtract(bbox_max, bbox_min, bbox_scale);
   bbox_scale *= bbox_scale.Norml2();

   // Create ParMesh using the default partitioning algorithm:
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   const int order = 3;
   H1_FECollection fec(order, dim);
   const int vdim = 1;
   const int ordering = Ordering::byNODES;
   ParFiniteElementSpace pfes(&pmesh, &fec, vdim, ordering);

   // Function that can be exactly represented in the FE space:
   FunctionCoefficient fun_c([&](const Vector &x) -> real_t
   {
      real_t u[17]; // max order is 16
      switch (sdim)
      {
         case 2:
         {
            const real_t y1 = (x[0] - bbox_center[0])/bbox_scale[0];
            const real_t y2 = (x[1] - bbox_center[1])/bbox_scale[1];
            const real_t a1 = -30*(M_PI/180);
            const real_t a2 = +40*(M_PI/180);
            const real_t x1 = 1.0_r*(cos(a1)*y1 + sin(a1)*y2);
            const real_t x2 = 0.8_r*(cos(a2)*y1 + sin(a2)*y2);
            Poly_1D::CalcLegendre(order, x1 + bbox_center[0], u);
            const real_t u1 = u[order];
            Poly_1D::CalcLegendre(order, x2 + bbox_center[1], u);
            const real_t u2 = u[order];
            return 0.37_r*u1 + 0.63_r*u2;
         }
         case 3:
         {
            const real_t y1 = (x[0] - bbox_center[0])/bbox_scale[0];
            const real_t y2 = (x[1] - bbox_center[1])/bbox_scale[1];
            const real_t y3 = (x[2] - bbox_center[2])/bbox_scale[2];
            const real_t m[3][3] =
            {
               {+0.8537, +0.2876, -0.1139},
               {-0.0705, +0.9391, +0.1570},
               {+0.1238, +0.0643, +0.6715}
            };
            const real_t x1 = m[0][0]*y1 + m[0][1]*y2 + m[0][2]*y3;
            const real_t x2 = m[1][0]*y1 + m[1][1]*y2 + m[1][2]*y3;
            const real_t x3 = m[2][0]*y1 + m[2][1]*y2 + m[2][2]*y3;
            Poly_1D::CalcLegendre(order, x1 + bbox_center[0], u);
            const real_t u1 = u[order];
            Poly_1D::CalcLegendre(order, x2 + bbox_center[1], u);
            const real_t u2 = u[order];
            Poly_1D::CalcLegendre(order, x3 + bbox_center[2], u);
            const real_t u3 = u[order];
            return 0.33_r*u1 + 0.42_r*u2 + 0.25*u3;
         }
         default: MFEM_ABORT("unsupported dimension: " << dim);
      }
   });

   ParGridFunction sol(&pfes);
   Vector t_sol_prev, t_sol;
   sol.ProjectCoefficient(fun_c);
   sol.GetTrueDofs(t_sol_prev);

   // Keep the next section for debugging visually 'fun_c':
#if 0
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize()
               << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << sol << std::flush;
   }
#endif

   OperatorPtr T(Operator::Hypre_ParCSR);

   // Lambda that checks the update operator after mesh modification:
   auto update_and_check = [&]()
   {
      pfes.GetTrueUpdateOperator(T);
      REQUIRE(T.Ptr() != nullptr);
      sol.Update();
      t_sol.SetSize(pfes.TrueVSize());
      T->Mult(t_sol_prev, t_sol);
      sol.SetFromTrueDofs(t_sol);
      REQUIRE(sol.ComputeMaxError(fun_c) < 1e-14);
      t_sol_prev = t_sol;
   };

   // Perform 'num_iter' iterations with the following steps:
   // 1. random refinements
   // 2. rebalancing
   // 3. random de-refinements
   // 4. rebalancing
   srand(1805092903);
   const int num_iter = 2;
   for (int iter = 1; iter <= num_iter; iter++)
   {
      INFO("refinement iteration " << iter);

      INFO("  before refinement\n" <<
           "    num of elements: " << pmesh.GetGlobalNE());
      pmesh.RandomRefinement(0.6);
      INFO("  after refinement\n" <<
           "    num of elements: " << pmesh.GetGlobalNE());
      update_and_check();

      INFO("  rebalancing");
      pmesh.Rebalance();
      update_and_check();

      INFO("  de-refinement");
      Vector elem_error(pmesh.GetNE());
      for (int el = 0; el < elem_error.Size(); el++)
      {
         elem_error[el] = rand_real();
      }
      const int nc_limit = 0;
      const int op = 0; // combine errors from children by taking a min
      const bool mesh_derefined =
         pmesh.DerefineByError(elem_error, 0.3, nc_limit, op);
      INFO("  after de-refinement\n" <<
           "    num of elements: " << pmesh.GetGlobalNE());
      // Require that the mesh was derefined in order to actually test the
      // de-refinement update operator:
      REQUIRE(mesh_derefined);
      // Objects should be updated only when the mesh was actually de-refined:
      if (mesh_derefined)
      {
         update_and_check();
      }

      INFO("  rebalancing");
      pmesh.Rebalance();
      update_and_check();
   }
}

#endif // #ifdef MFEM_USE_MPI
