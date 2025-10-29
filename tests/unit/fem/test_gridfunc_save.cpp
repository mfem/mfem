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

using namespace mfem;

TEST_CASE("GridFunction Save", "[GridFunction]")
{
   const int order = 3;

   Mesh mesh = Mesh::MakeCartesian3D(1, 2, 3, Element::PYRAMID,
                                     1.0, 2.0, 3.0);

   H1_FECollection fec_h1(order, mesh.Dimension());
   ND_FECollection fec_nd(order, mesh.Dimension());
   RT_FECollection fec_rt(order-1, mesh.Dimension());
   L2_FECollection fec_l2(order-1, mesh.Dimension());

   FiniteElementSpace fes_h1(&mesh, &fec_h1);
   FiniteElementSpace fes_nd(&mesh, &fec_nd);
   FiniteElementSpace fes_rt(&mesh, &fec_rt);
   FiniteElementSpace fes_l2(&mesh, &fec_l2);

   GridFunction gf_h1(&fes_h1);
   GridFunction gf_nd(&fes_nd);
   GridFunction gf_rt(&fes_rt);
   GridFunction gf_l2(&fes_l2);

   gf_h1.Randomize(1);
   gf_nd.Randomize(1);
   gf_rt.Randomize(1);
   gf_l2.Randomize(1);

   Vector zeroVec(3); zeroVec = 0.0;
   ConstantCoefficient zeroCoef(0.0);
   VectorConstantCoefficient zeroVecCoef(zeroVec);

   const double norm_h1 = gf_h1.ComputeL2Error(zeroCoef);
   const double norm_nd = gf_nd.ComputeL2Error(zeroVecCoef);
   const double norm_rt = gf_rt.ComputeL2Error(zeroVecCoef);
   const double norm_l2 = gf_l2.ComputeL2Error(zeroCoef);

   std::ofstream ofs_h1("real_h1.gf"); ofs_h1.precision(8);
   std::ofstream ofs_nd("real_nd.gf"); ofs_nd.precision(8);
   std::ofstream ofs_rt("real_rt.gf"); ofs_rt.precision(8);
   std::ofstream ofs_l2("real_l2.gf"); ofs_l2.precision(8);

   gf_h1.Save(ofs_h1); ofs_h1.close();
   gf_nd.Save(ofs_nd); ofs_nd.close();
   gf_rt.Save(ofs_rt); ofs_rt.close();
   gf_l2.Save(ofs_l2); ofs_l2.close();

   std::ifstream ifs_h1("real_h1.gf");
   std::ifstream ifs_nd("real_nd.gf");
   std::ifstream ifs_rt("real_rt.gf");
   std::ifstream ifs_l2("real_l2.gf");

   GridFunction gf_h1_read(&mesh, ifs_h1); ifs_h1.close();
   GridFunction gf_nd_read(&mesh, ifs_nd); ifs_nd.close();
   GridFunction gf_rt_read(&mesh, ifs_rt); ifs_rt.close();
   GridFunction gf_l2_read(&mesh, ifs_l2); ifs_l2.close();

   gf_h1_read -= gf_h1;
   gf_nd_read -= gf_nd;
   gf_rt_read -= gf_rt;
   gf_l2_read -= gf_l2;

   const double diff_h1 = gf_h1_read.ComputeL2Error(zeroCoef);
   const double diff_nd = gf_nd_read.ComputeL2Error(zeroVecCoef);
   const double diff_rt = gf_rt_read.ComputeL2Error(zeroVecCoef);
   const double diff_l2 = gf_l2_read.ComputeL2Error(zeroCoef);

   REQUIRE(diff_h1 < 1e-8 * norm_h1);
   REQUIRE(diff_nd < 1e-8 * norm_nd);
   REQUIRE(diff_rt < 1e-8 * norm_rt);
   REQUIRE(diff_l2 < 1e-8 * norm_l2);

   // Clean up
   REQUIRE(std::remove("real_h1.gf") == 0);
   REQUIRE(std::remove("real_nd.gf") == 0);
   REQUIRE(std::remove("real_rt.gf") == 0);
   REQUIRE(std::remove("real_l2.gf") == 0);
}

#ifdef MFEM_USE_MPI

TEST_CASE("ParGridFunction Save", "[ParGridFunction][Parallel]")
{
   const int num_procs = Mpi::WorldSize();
   const int my_rank = Mpi::WorldRank();
   const int order = 3;

   const int nx = (int)rint(cbrt(real_t(num_procs)));
   const int ny = (int)rint(2.0 * cbrt(real_t(num_procs)));
   const int nz = (int)rint(3.0 * cbrt(real_t(num_procs)));
   Mesh mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::PYRAMID,
                                     1.0, 2.0, 3.0);

   // Define a parallel mesh by a partitioning of the serial mesh.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   H1_FECollection fec_h1(order, mesh.Dimension());
   ND_FECollection fec_nd(order, mesh.Dimension());
   RT_FECollection fec_rt(order-1, mesh.Dimension());
   L2_FECollection fec_l2(order-1, mesh.Dimension());

   ParFiniteElementSpace pfes_h1(&pmesh, &fec_h1);
   ParFiniteElementSpace pfes_nd(&pmesh, &fec_nd);
   ParFiniteElementSpace pfes_rt(&pmesh, &fec_rt);
   ParFiniteElementSpace pfes_l2(&pmesh, &fec_l2);

   ParGridFunction pgf_h1(&pfes_h1);
   ParGridFunction pgf_nd(&pfes_nd);
   ParGridFunction pgf_rt(&pfes_rt);
   ParGridFunction pgf_l2(&pfes_l2);

   pgf_h1.Randomize(1);
   pgf_nd.Randomize(1);
   pgf_rt.Randomize(1);
   pgf_l2.Randomize(1);

   // Ensure that the L-DOFs are set consistently on all ranks
   pgf_h1.SetTrueVector(); pgf_h1.SetFromTrueVector();
   pgf_nd.SetTrueVector(); pgf_nd.SetFromTrueVector();
   pgf_rt.SetTrueVector(); pgf_rt.SetFromTrueVector();
   pgf_l2.SetTrueVector(); pgf_l2.SetFromTrueVector();

   Vector zeroVec(3); zeroVec = 0.0;
   ConstantCoefficient zeroCoef(0.0);
   VectorConstantCoefficient zeroVecCoef(zeroVec);

   const double norm_h1 = pgf_h1.ComputeL2Error(zeroCoef);
   const double norm_nd = pgf_nd.ComputeL2Error(zeroVecCoef);
   const double norm_rt = pgf_rt.ComputeL2Error(zeroVecCoef);
   const double norm_l2 = pgf_l2.ComputeL2Error(zeroCoef);

   std::ostringstream name_h1, name_nd, name_rt, name_l2;
   name_h1 << "real_gf_h1." << std::setfill('0') << std::setw(6) << my_rank;
   name_nd << "real_gf_nd." << std::setfill('0') << std::setw(6) << my_rank;
   name_rt << "real_gf_rt." << std::setfill('0') << std::setw(6) << my_rank;
   name_l2 << "real_gf_l2." << std::setfill('0') << std::setw(6) << my_rank;

   std::ofstream ofs_h1(name_h1.str().c_str()); ofs_h1.precision(8);
   std::ofstream ofs_nd(name_nd.str().c_str()); ofs_nd.precision(8);
   std::ofstream ofs_rt(name_rt.str().c_str()); ofs_rt.precision(8);
   std::ofstream ofs_l2(name_l2.str().c_str()); ofs_l2.precision(8);

   pgf_h1.Save(ofs_h1); ofs_h1.close();
   pgf_nd.Save(ofs_nd); ofs_nd.close();
   pgf_rt.Save(ofs_rt); ofs_rt.close();
   pgf_l2.Save(ofs_l2); ofs_l2.close();

   std::ifstream ifs_h1(name_h1.str().c_str());
   std::ifstream ifs_nd(name_nd.str().c_str());
   std::ifstream ifs_rt(name_rt.str().c_str());
   std::ifstream ifs_l2(name_l2.str().c_str());

   ParGridFunction pgf_h1_read(&pmesh, ifs_h1); ifs_h1.close();
   ParGridFunction pgf_nd_read(&pmesh, ifs_nd); ifs_nd.close();
   ParGridFunction pgf_rt_read(&pmesh, ifs_rt); ifs_rt.close();
   ParGridFunction pgf_l2_read(&pmesh, ifs_l2); ifs_l2.close();

   pgf_h1_read -= pgf_h1;
   pgf_nd_read -= pgf_nd;
   pgf_rt_read -= pgf_rt;
   pgf_l2_read -= pgf_l2;

   const double diff_h1 = pgf_h1_read.ComputeL2Error(zeroCoef);
   const double diff_nd = pgf_nd_read.ComputeL2Error(zeroVecCoef);
   const double diff_rt = pgf_rt_read.ComputeL2Error(zeroVecCoef);
   const double diff_l2 = pgf_l2_read.ComputeL2Error(zeroCoef);

   if (my_rank == 0)
   {
      REQUIRE(diff_h1 < 1e-8 * norm_h1);
      REQUIRE(diff_nd < 1e-8 * norm_nd);
      REQUIRE(diff_rt < 1e-8 * norm_rt);
      REQUIRE(diff_l2 < 1e-8 * norm_l2);
   }

   // Clean up
   REQUIRE(std::remove(name_h1.str().c_str()) == 0);
   REQUIRE(std::remove(name_nd.str().c_str()) == 0);
   REQUIRE(std::remove(name_rt.str().c_str()) == 0);
   REQUIRE(std::remove(name_l2.str().c_str()) == 0);
}

#endif // MFEM_USE_MPI
