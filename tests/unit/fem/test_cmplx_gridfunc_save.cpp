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

TEST_CASE("ComplexGridFunction Save", "[ComplexGridFunction]")
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

   ComplexGridFunction gf_h1(&fes_h1);
   ComplexGridFunction gf_nd(&fes_nd);
   ComplexGridFunction gf_rt(&fes_rt);
   ComplexGridFunction gf_l2(&fes_l2);

   gf_h1.Randomize(1);
   gf_nd.Randomize(1);
   gf_rt.Randomize(1);
   gf_l2.Randomize(1);

   Vector zeroVec(3); zeroVec = 0.0;
   ConstantCoefficient zeroCoef(0.0);
   VectorConstantCoefficient zeroVecCoef(zeroVec);

   const double norm_h1 = gf_h1.ComputeL2Error(zeroCoef, zeroCoef);
   const double norm_nd = gf_nd.ComputeL2Error(zeroVecCoef, zeroVecCoef);
   const double norm_rt = gf_rt.ComputeL2Error(zeroVecCoef, zeroVecCoef);
   const double norm_l2 = gf_l2.ComputeL2Error(zeroCoef, zeroCoef);

   std::ofstream ofs_h1("cmplx_h1.gf"); ofs_h1.precision(8);
   std::ofstream ofs_nd("cmplx_nd.gf"); ofs_nd.precision(8);
   std::ofstream ofs_rt("cmplx_rt.gf"); ofs_rt.precision(8);
   std::ofstream ofs_l2("cmplx_l2.gf"); ofs_l2.precision(8);

   gf_h1.Save(ofs_h1);
   gf_nd.Save(ofs_nd);
   gf_rt.Save(ofs_rt);
   gf_l2.Save(ofs_l2);

   std::ifstream ifs_h1("cmplx_h1.gf");
   std::ifstream ifs_nd("cmplx_nd.gf");
   std::ifstream ifs_rt("cmplx_rt.gf");
   std::ifstream ifs_l2("cmplx_l2.gf");

   ComplexGridFunction gf_h1_read(&mesh, ifs_h1);
   ComplexGridFunction gf_nd_read(&mesh, ifs_nd);
   ComplexGridFunction gf_rt_read(&mesh, ifs_rt);
   ComplexGridFunction gf_l2_read(&mesh, ifs_l2);

   gf_h1_read -= gf_h1;
   gf_nd_read -= gf_nd;
   gf_rt_read -= gf_rt;
   gf_l2_read -= gf_l2;

   const double diff_h1 = gf_h1_read.ComputeL2Error(zeroCoef, zeroCoef);
   const double diff_nd = gf_nd_read.ComputeL2Error(zeroVecCoef, zeroVecCoef);
   const double diff_rt = gf_rt_read.ComputeL2Error(zeroVecCoef, zeroVecCoef);
   const double diff_l2 = gf_l2_read.ComputeL2Error(zeroCoef, zeroCoef);

   REQUIRE(diff_h1 < 1e-8 * norm_h1);
   REQUIRE(diff_nd < 1e-8 * norm_nd);
   REQUIRE(diff_rt < 1e-8 * norm_rt);
   REQUIRE(diff_l2 < 1e-8 * norm_l2);

   // Clean up
   REQUIRE(std::remove("cmplx_h1.gf") == 0);
   REQUIRE(std::remove("cmplx_nd.gf") == 0);
   REQUIRE(std::remove("cmplx_rt.gf") == 0);
   REQUIRE(std::remove("cmplx_l2.gf") == 0);
}

#ifdef MFEM_USE_MPI

TEST_CASE("ParComplexGridFunction Save", "[ParComplexGridFunction][Parallel]")
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

   ParComplexGridFunction pgf_h1(&pfes_h1);
   ParComplexGridFunction pgf_nd(&pfes_nd);
   ParComplexGridFunction pgf_rt(&pfes_rt);
   ParComplexGridFunction pgf_l2(&pfes_l2);

   pgf_h1.Randomize(1);
   pgf_nd.Randomize(1);
   pgf_rt.Randomize(1);
   pgf_l2.Randomize(1);

   // Ensure that the L-DOFs are set consistently on all ranks
   pgf_h1.real().SetTrueVector(); pgf_h1.real().SetFromTrueVector();
   pgf_h1.imag().SetTrueVector(); pgf_h1.imag().SetFromTrueVector();
   pgf_nd.real().SetTrueVector(); pgf_nd.real().SetFromTrueVector();
   pgf_nd.imag().SetTrueVector(); pgf_nd.imag().SetFromTrueVector();
   pgf_rt.real().SetTrueVector(); pgf_rt.real().SetFromTrueVector();
   pgf_rt.imag().SetTrueVector(); pgf_rt.imag().SetFromTrueVector();
   pgf_l2.real().SetTrueVector(); pgf_l2.real().SetFromTrueVector();
   pgf_l2.imag().SetTrueVector(); pgf_l2.imag().SetFromTrueVector();

   Vector zeroVec(3); zeroVec = 0.0;
   ConstantCoefficient zeroCoef(0.0);
   VectorConstantCoefficient zeroVecCoef(zeroVec);

   const double norm_h1 = pgf_h1.ComputeL2Error(zeroCoef, zeroCoef);
   const double norm_nd = pgf_nd.ComputeL2Error(zeroVecCoef, zeroVecCoef);
   const double norm_rt = pgf_rt.ComputeL2Error(zeroVecCoef, zeroVecCoef);
   const double norm_l2 = pgf_l2.ComputeL2Error(zeroCoef, zeroCoef);

   std::ostringstream name_h1, name_nd, name_rt, name_l2;
   name_h1 << "cmplx_gf_h1." << std::setfill('0') << std::setw(6) << my_rank;
   name_nd << "cmplx_gf_nd." << std::setfill('0') << std::setw(6) << my_rank;
   name_rt << "cmplx_gf_rt." << std::setfill('0') << std::setw(6) << my_rank;
   name_l2 << "cmplx_gf_l2." << std::setfill('0') << std::setw(6) << my_rank;

   std::ofstream ofs_h1(name_h1.str().c_str()); ofs_h1.precision(8);
   std::ofstream ofs_nd(name_nd.str().c_str()); ofs_nd.precision(8);
   std::ofstream ofs_rt(name_rt.str().c_str()); ofs_rt.precision(8);
   std::ofstream ofs_l2(name_l2.str().c_str()); ofs_l2.precision(8);

   pgf_h1.Save(ofs_h1);
   pgf_nd.Save(ofs_nd);
   pgf_rt.Save(ofs_rt);
   pgf_l2.Save(ofs_l2);

   MPI_Barrier(MPI_COMM_WORLD);

   std::ifstream ifs_h1(name_h1.str().c_str());
   std::ifstream ifs_nd(name_nd.str().c_str());
   std::ifstream ifs_rt(name_rt.str().c_str());
   std::ifstream ifs_l2(name_l2.str().c_str());

   ParComplexGridFunction pgf_h1_read(&pmesh, ifs_h1);
   ParComplexGridFunction pgf_nd_read(&pmesh, ifs_nd);
   ParComplexGridFunction pgf_rt_read(&pmesh, ifs_rt);
   ParComplexGridFunction pgf_l2_read(&pmesh, ifs_l2);

   pgf_h1_read -= pgf_h1;
   pgf_nd_read -= pgf_nd;
   pgf_rt_read -= pgf_rt;
   pgf_l2_read -= pgf_l2;

   const double diff_h1 = pgf_h1_read.ComputeL2Error(zeroCoef, zeroCoef);
   const double diff_nd = pgf_nd_read.ComputeL2Error(zeroVecCoef, zeroVecCoef);
   const double diff_rt = pgf_rt_read.ComputeL2Error(zeroVecCoef, zeroVecCoef);
   const double diff_l2 = pgf_l2_read.ComputeL2Error(zeroCoef, zeroCoef);

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
