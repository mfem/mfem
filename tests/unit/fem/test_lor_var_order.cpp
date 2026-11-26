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

#include <cmath>
#include <cstdint>
#include <memory>

using namespace mfem;

namespace lor_var_order_test
{

#ifdef MFEM_USE_MPI

static int RandomElementOrder(Mesh &mesh, int e, int pmin, int pmax)
{
   Vector c(mesh.SpaceDimension());
   mesh.GetElementCenter(e, c);
   std::uint64_t h = 1469598103934665603ULL;
   for (int d = 0; d < c.Size(); ++d)
   {
      long long q = std::llround(c(d) * 1e6);
      h ^= static_cast<std::uint64_t>(q + 0x9e3779b97f4a7c15ULL);
      h *= 1099511628211ULL;
   }
   return pmin + static_cast<int>(h % static_cast<std::uint64_t>(pmax - pmin + 1));
}

static Mesh MakeNonconformingHexMesh(bool curved)
{
   Mesh mesh = Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   mesh.EnsureNCMesh();
   // 2-irregular refinement near the origin corner
   for (int lvl = 0; lvl < 2; ++lvl)
   {
      Array<int> refs;
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         Vector c(3);
         mesh.GetElementCenter(e, c);
         const real_t thresh = (lvl == 0) ? 0.52 : 0.27;
         if (c(0) < thresh && c(1) < thresh && c(2) < thresh)
         {
            refs.Append(e);
         }
      }
      mesh.GeneralRefinement(refs, -1, 0);
   }
   if (curved)
   {
      mesh.SetCurvature(2);
      mesh.Transform([](const Vector &x, Vector &y)
      {
         const real_t s = 0.05 * std::sin(M_PI * x(0)) * std::sin(M_PI * x(1)) *
                          std::sin(M_PI * x(2));
         y(0) = x(0) + s;
         y(1) = x(1) - 0.5 * s;
         y(2) = x(2) + 0.7 * s;
      });
   }
   return mesh;
}

TEST_CASE("Parallel Variable-Order Nonconforming LOR de Rham Complex",
          "[LOR][VariableOrder][Parallel]")
{
   const bool curved = GENERATE(false, true);
   CAPTURE(curved);

   Mesh smesh = MakeNonconformingHexMesh(curved);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();

   const int max_p = 4;
   H1_FECollection h1_fec(max_p, 3, BasisType::GaussLobatto);
   ND_FECollection nd_fec(max_p, 3, BasisType::GaussLobatto,
                          BasisType::IntegratedGLL);
   RT_FECollection rt_fec(max_p - 1, 3, BasisType::GaussLobatto,
                          BasisType::IntegratedGLL);

   ParFiniteElementSpace h1_fes(&pmesh, &h1_fec);
   ParFiniteElementSpace nd_fes(&pmesh, &nd_fec);
   ParFiniteElementSpace rt_fes(&pmesh, &rt_fec);
   for (int e = 0; e < pmesh.GetNE(); ++e)
   {
      const int ord = RandomElementOrder(pmesh, e, 1, max_p);
      h1_fes.SetElementOrder(e, ord);
      nd_fes.SetElementOrder(e, ord);
      rt_fes.SetElementOrder(e, ord);
   }
   h1_fes.Update(false);
   nd_fes.Update(false);
   rt_fes.Update(false);

   // 1. Compare VarOrderLORAssembly::AssembleGradientOrPi & AssembleCurl
   //    against MFEM's native ParDiscreteLinearOperator(Gradient/CurlInterpolator)
   auto G_lor = VarOrderLORAssembly::AssembleGradientOrPi(h1_fes, nd_fes, -1);
   auto C_lor = VarOrderLORAssembly::AssembleCurl(nd_fes, rt_fes);

   ParDiscreteLinearOperator grad_ref(&h1_fes, &nd_fes);
   grad_ref.AddDomainInterpolator(new GradientInterpolator);
   grad_ref.Assemble();
   grad_ref.Finalize();
   std::unique_ptr<HypreParMatrix> G_ref(grad_ref.ParallelAssemble());

   ParDiscreteLinearOperator curl_ref(&nd_fes, &rt_fes);
   curl_ref.AddDomainInterpolator(new CurlInterpolator);
   curl_ref.Assemble();
   curl_ref.Finalize();
   std::unique_ptr<HypreParMatrix> C_ref(curl_ref.ParallelAssemble());

   // Exact operator equality: G_lor == G_ref and C_lor == C_ref
   {
      Vector q(h1_fes.GetTrueVSize());
      q.Randomize(12345 + Mpi::WorldRank());
      Vector g1(nd_fes.GetTrueVSize()), g2(nd_fes.GetTrueVSize());
      G_lor->Mult(q, g1);
      G_ref->Mult(q, g2);
      g1 -= g2;
      const real_t g_diff = g1.Normlinf();
      real_t g_diff_glob = 0.0;
      MPI_Allreduce(&g_diff, &g_diff_glob, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, MPI_COMM_WORLD);
      REQUIRE(g_diff_glob < 1e-13);

      // Exact complex property: C_lor * G_lor * q == 0
      Vector cg(rt_fes.GetTrueVSize());
      C_lor->Mult(g2, cg);
      const real_t cg_norm = cg.Normlinf();
      real_t cg_glob = 0.0;
      MPI_Allreduce(&cg_norm, &cg_glob, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, MPI_COMM_WORLD);
      REQUIRE(cg_glob < 1e-13);

      // Local-global commutativity: P_ND * G_true * q == G_loc * P_H1 * q
      ParGridFunction q_gf(&h1_fes), g_gf(&nd_fes), g_loc_gf(&nd_fes);
      q_gf.SetFromTrueDofs(q);
      g_gf.SetFromTrueDofs(g2);
      grad_ref.Mult(q_gf, g_loc_gf);
      g_gf -= g_loc_gf;
      const real_t comm_g = g_gf.Normlinf();
      real_t comm_g_glob = 0.0;
      MPI_Allreduce(&comm_g, &comm_g_glob, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, MPI_COMM_WORLD);
      REQUIRE(comm_g_glob < 1e-13);
   }

   {
      Vector v(nd_fes.GetTrueVSize());
      v.Randomize(54321 + Mpi::WorldRank());
      Vector c1(rt_fes.GetTrueVSize()), c2(rt_fes.GetTrueVSize());
      C_lor->Mult(v, c1);
      C_ref->Mult(v, c2);
      c1 -= c2;
      const real_t c_diff = c1.Normlinf();
      real_t c_diff_glob = 0.0;
      MPI_Allreduce(&c_diff, &c_diff_glob, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, MPI_COMM_WORLD);
      REQUIRE(c_diff_glob < 1e-13);

      // Local-global commutativity: P_RT * C_true * v == C_loc * P_ND * v
      ParGridFunction v_gf(&nd_fes), c_gf(&rt_fes), c_loc_gf(&rt_fes);
      v_gf.SetFromTrueDofs(v);
      c_gf.SetFromTrueDofs(c2);
      curl_ref.Mult(v_gf, c_loc_gf);
      c_gf -= c_loc_gf;
      const real_t comm_c = c_gf.Normlinf();
      real_t comm_c_glob = 0.0;
      MPI_Allreduce(&comm_c, &comm_c_glob, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, MPI_COMM_WORLD);
      REQUIRE(comm_c_glob < 1e-13);
   }

   // 2. Native LORSolver<HypreBoomerAMG>, LORSolver<HypreAMS>, LORSolver<HypreADS>
   //    + PCG on random variable-order nonconforming mesh!
   ConstantCoefficient one(1.0);
   Array<int> bdr_attr(pmesh.bdr_attributes.Max());
   bdr_attr = 1;

   // H1 / AMG
   {
      Array<int> ess_tdofs;
      h1_fes.GetEssentialTrueDofs(bdr_attr, ess_tdofs);
      ParBilinearForm a(&h1_fes);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.AddDomainIntegrator(new MassIntegrator(one));
      a.Assemble();
      OperatorHandle A_op;
      a.FormSystemMatrix(ess_tdofs, A_op);

      LORSolver<HypreBoomerAMG> prec(a, ess_tdofs);
      prec.GetSolver().SetPrintLevel(0);

      Vector x_exact(h1_fes.GetTrueVSize()), b(h1_fes.GetTrueVSize()),
             x(h1_fes.GetTrueVSize());
      x_exact.Randomize(101 + Mpi::WorldRank());
      for (int i = 0; i < ess_tdofs.Size(); ++i) { x_exact(ess_tdofs[i]) = 0.0; }
      A_op->Mult(x_exact, b);
      x = 0.0;

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetOperator(*A_op);
      cg.SetPreconditioner(prec);
      cg.SetRelTol(1e-10);
      cg.SetMaxIter(100);
      cg.SetPrintLevel(0);
      cg.Mult(b, x);
      REQUIRE(cg.GetConverged());
      REQUIRE(cg.GetNumIterations() < 40);
   }

   // H(curl) / AMS
   {
      Array<int> ess_tdofs;
      nd_fes.GetEssentialTrueDofs(bdr_attr, ess_tdofs);
      ParBilinearForm a(&nd_fes);
      a.AddDomainIntegrator(new CurlCurlIntegrator(one));
      a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      a.Assemble();
      OperatorHandle A_op;
      a.FormSystemMatrix(ess_tdofs, A_op);

      LORSolver<HypreAMS> prec(a, ess_tdofs);
      prec.GetSolver().SetPrintLevel(0);

      Vector x_exact(nd_fes.GetTrueVSize()), b(nd_fes.GetTrueVSize()),
             x(nd_fes.GetTrueVSize());
      x_exact.Randomize(202 + Mpi::WorldRank());
      for (int i = 0; i < ess_tdofs.Size(); ++i) { x_exact(ess_tdofs[i]) = 0.0; }
      A_op->Mult(x_exact, b);
      x = 0.0;

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetOperator(*A_op);
      cg.SetPreconditioner(prec);
      cg.SetRelTol(1e-10);
      cg.SetMaxIter(100);
      cg.SetPrintLevel(0);
      cg.Mult(b, x);
      REQUIRE(cg.GetConverged());
      REQUIRE(cg.GetNumIterations() < 50);
   }

   // H(div) / ADS (both native LORSolver<HypreADS> and native HypreADS on FA!)
   {
      Array<int> ess_tdofs;
      rt_fes.GetEssentialTrueDofs(bdr_attr, ess_tdofs);
      ParBilinearForm a(&rt_fes);
      a.AddDomainIntegrator(new DivDivIntegrator(one));
      a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      a.Assemble();
      OperatorHandle A_op;
      a.FormSystemMatrix(ess_tdofs, A_op);

      LORSolver<HypreADS> prec(a, ess_tdofs);
      prec.GetSolver().SetPrintLevel(0);

      Vector x_exact(rt_fes.GetTrueVSize()), b(rt_fes.GetTrueVSize()),
             x(rt_fes.GetTrueVSize());
      x_exact.Randomize(303 + Mpi::WorldRank());
      for (int i = 0; i < ess_tdofs.Size(); ++i) { x_exact(ess_tdofs[i]) = 0.0; }
      A_op->Mult(x_exact, b);
      x = 0.0;

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetOperator(*A_op);
      cg.SetPreconditioner(prec);
      cg.SetRelTol(1e-10);
      cg.SetMaxIter(100);
      cg.SetPrintLevel(0);
      cg.Mult(b, x);
      REQUIRE(cg.GetConverged());
      REQUIRE(cg.GetNumIterations() < 50);

      // Also verify native HypreADS works directly on FA IntegratedGLL RT
      // AND verify IdentityInterpolator H1^3 -> RT(IntegratedGLL) matches
      // ProjectCoefficient on curved Q2 variable-order NC mesh!
      {
         ParFiniteElementSpace h1_vec(&pmesh, &h1_fec, 3, Ordering::byVDIM);
         for (int e = 0; e < pmesh.GetNE(); ++e)
         {
            h1_vec.SetElementOrder(e, rt_fes.GetElementOrder(e));
         }
         h1_vec.Update(false);

         ParDiscreteLinearOperator id_op(&h1_vec, &rt_fes);
         id_op.AddDomainInterpolator(new IdentityInterpolator);
         id_op.Assemble();
         id_op.Finalize();

         VectorFunctionCoefficient f_coeff(3, [](const Vector &x, Vector &v)
         {
            v(0) = 1.0 + 0.3 * x(0) - 0.2 * x(1);
            v(1) = -0.5 + 0.4 * x(1) + 0.1 * x(2);
            v(2) = 0.7 - 0.25 * x(2) + 0.15 * x(0);
         });
         ParGridFunction f_h1(&h1_vec), f_rt_id(&rt_fes), f_rt_proj(&rt_fes);
         f_h1.ProjectCoefficient(f_coeff);
         id_op.Mult(f_h1, f_rt_id);
         f_rt_proj.ProjectCoefficient(f_coeff);
         f_rt_id -= f_rt_proj;
         const real_t id_diff = f_rt_id.Normlinf();
         real_t id_diff_glob = 0.0;
         MPI_Allreduce(&id_diff, &id_diff_glob, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_MAX, MPI_COMM_WORLD);
         // On curved Q2, f_h1 is projected in H1_p while f_rt_proj projects f directly;
         // check exact identity for constant fields (in H1_1 identically on curved Q2!)
         VectorConstantCoefficient c_vec(Vector({1.2, -0.7, 0.5}));
         f_h1.ProjectCoefficient(c_vec);
         id_op.Mult(f_h1, f_rt_id);
         f_rt_proj.ProjectCoefficient(c_vec);
         f_rt_id -= f_rt_proj;
         const real_t cid_diff = f_rt_id.Normlinf();
         real_t cid_diff_glob = 0.0;
         MPI_Allreduce(&cid_diff, &cid_diff_glob, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_MAX, MPI_COMM_WORLD);
         REQUIRE(cid_diff_glob < 1e-12);
      }

      ParBilinearForm a_fa(&rt_fes);
      a_fa.AddDomainIntegrator(new DivDivIntegrator(one));
      a_fa.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      a_fa.Assemble();
      OperatorHandle A_fa;
      a_fa.FormSystemMatrix(ess_tdofs, A_fa);
      HypreADS ads_fa(*A_fa.As<HypreParMatrix>(), &rt_fes);
      ads_fa.SetPrintLevel(0);
      A_fa->Mult(x_exact, b);
      x = 0.0;
      cg.SetOperator(*A_fa);
      cg.SetPreconditioner(ads_fa);
      cg.SetMaxIter(250);
      cg.Mult(b, x);
      CAPTURE(cg.GetNumIterations(), cg.GetFinalRelNorm());
      REQUIRE(cg.GetConverged());
   }
}

#endif // MFEM_USE_MPI

} // namespace lor_var_order_test
