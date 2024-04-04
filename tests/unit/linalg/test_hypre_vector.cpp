// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

#ifdef MFEM_USE_MPI

// TODO: modify the tests here to support HYPRE_USING_GPU?
#ifndef HYPRE_USING_GPU

namespace internal
{

std::vector<HYPRE_BigInt> GenerateColumnPartitioning(const int world_size,
                                                     const int rank, const int size_per_rank)
{
   std::vector<HYPRE_BigInt> col;
   if (HYPRE_AssumedPartitionCheck())
   {
      int offset = rank*size_per_rank;
      col = {offset, offset + size_per_rank};
   }
   else
   {
      col.resize(world_size+1);
      for (int i=0; i<=world_size; ++i)
      {
         col[i] = i*size_per_rank;
      }
   }
   return col;
}

} // namespace internal

TEST_CASE("HypreParVector I/O", "[Parallel], [HypreParVector]")
{
   // Create a test vector (two entries per rank) with entries increasing
   // sequentially. Write the vector to a file, read it into another vector, and
   // make sure we get the same answer.

   int world_size, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int size_per_rank = 2;

   HYPRE_BigInt glob_size = world_size*size_per_rank;
   std::vector<HYPRE_BigInt> col = internal::GenerateColumnPartitioning(world_size,
                                                                        rank, size_per_rank);

   // Initialize vector and write to files
   HypreParVector v1(MPI_COMM_WORLD, glob_size, col.data());
   for (int i=0; i<size_per_rank; ++i)
   {
      v1(i) = rank*size_per_rank + i;
   }
   v1.Print("vec");

   // Check that we read the same vector from disk
   HypreParVector v2;
   v2.Read(MPI_COMM_WORLD, "vec");
   v2 -= v1;
   REQUIRE(InnerProduct(v2, v2) == MFEM_Approx(0.0));

   // Clean up
   std::string prefix = "vec.";
   std::string suffix = std::to_string(rank);
   remove((prefix + suffix).c_str());
   remove((prefix + "INFO." + suffix).c_str());
}

TEST_CASE("HypreParVector Move Constructor", "[Parallel], [HypreParVector]")
{
   int world_size, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int size_per_rank = 2;

   HYPRE_BigInt glob_size = world_size*size_per_rank;
   std::vector<HYPRE_BigInt> col = internal::GenerateColumnPartitioning(world_size,
                                                                        rank, size_per_rank);

   // Initialize vector
   HypreParVector v1(MPI_COMM_WORLD, glob_size, col.data());
   for (int i=0; i<size_per_rank; ++i)
   {
      v1(i) = rank*size_per_rank + i;
   }

   // Make a copy so we can compare it later
   HypreParVector v1_copy(v1);

   // Also save off the data pointer
   real_t* v1_data = v1.GetData();

   // Now move from the original and compare
   HypreParVector v1_move(std::move(v1));

   REQUIRE(v1.GetOwnership() == 0);
   REQUIRE(v1_move.GetOwnership() == 1);
   REQUIRE(v1_move.Size() == v1_copy.Size());
   REQUIRE(v1_move.GlobalSize() == v1_copy.GlobalSize());
   REQUIRE(v1.Size() == 0);
   REQUIRE(v1.GetData() == nullptr);

   REQUIRE(v1_move.GetData() == v1_data);

   // The full HypreParVectors should also be the same
   v1_copy -= v1_move;
   REQUIRE(InnerProduct(v1_copy, v1_copy) == MFEM_Approx(0.0));
}

TEST_CASE("HypreParVector Move Assignment", "[Parallel], [HypreParVector]")
{
   int world_size, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int size_per_rank = 2;

   HYPRE_BigInt glob_size = world_size*size_per_rank;
   std::vector<HYPRE_BigInt> col = internal::GenerateColumnPartitioning(world_size,
                                                                        rank, size_per_rank);

   // Initialize vector
   HypreParVector v1(MPI_COMM_WORLD, glob_size, col.data());
   for (int i=0; i<size_per_rank; ++i)
   {
      v1(i) = rank*size_per_rank + i;
   }

   // Make a copy so we can compare it later
   HypreParVector v1_copy(v1);

   // Also save off the data pointer
   real_t* v1_data = v1.GetData();

   // Now move from the original and compare
   HypreParVector v1_move;
   v1_move = std::move(v1);

   REQUIRE(v1.GetOwnership() == 0);
   REQUIRE(v1_move.GetOwnership() == 1);
   REQUIRE(v1_move.Size() == v1_copy.Size());
   REQUIRE(v1_move.GlobalSize() == v1_copy.GlobalSize());
   REQUIRE(v1.Size() == 0);
   REQUIRE(v1.GetData() == nullptr);

   REQUIRE(v1_move.GetData() == v1_data);

   // The full HypreParVectors should also be the same
   v1_copy -= v1_move;
   REQUIRE(InnerProduct(v1_copy, v1_copy) == MFEM_Approx(0.0));
}

#endif // HYPRE_USING_GPU

#endif // MFEM_USE_MPI

} // namespace mfem
