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

#include "unit_tests.hpp"
#include "mfem.hpp"
#include <sstream>

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

TEST_CASE("HypreParVector reuses registered raw buffers",
          "[HypreParVector][GPU]")
{
   auto &inst = MemoryManager::Instance();
   const auto d_mt = IsDeviceMemory(inst.GetDeviceMemoryType()) ?
                     inst.GetDeviceMemoryType() : MemoryType::DEVICE_DEBUG;
   Vector base(8, MemoryType::HOST);
   base.GetMemory().SetDeviceMemoryType(d_mt);
   auto host = base.HostWrite();
   for (int i = 0; i < 8; ++i) { host[i] = i; }
   const auto device = base.Read(true);
   hypre_Vector local = {};
   hypre_VectorSize(&local) = 4;
   hypre_VectorData(&local) = host + 2;
#if MFEM_HYPRE_VERSION >= 21800
   hypre_VectorMemoryLocation(&local) = HYPRE_MEMORY_HOST;
#endif
#ifdef HYPRE_USING_GPU
   SECTION("Borrow a device pointer")
   {
      hypre_VectorData(&local) = const_cast<real_t *>(device) + 2;
      hypre_VectorMemoryLocation(&local) = HYPRE_MEMORY_DEVICE;
   }
   SECTION("Borrow a host pointer") {}
#endif
   // Borrow local HYPRE structures; these wrapper operations need no MPI calls.
   hypre_ParVector native = {};
   hypre_ParVectorLocalVector(&native) = &local;
   std::ostringstream sink;
   const int registrations = inst.PrintPtrs(sink);
   {
      HypreParVector wrapped(reinterpret_cast<HYPRE_ParVector>(&native));
      REQUIRE(wrapped.Read(true) == device + 2);
      REQUIRE(inst.PrintPtrs(sink) == registrations);
      auto ptr = wrapped.Write(true);
      forall(4, [=] MFEM_HOST_DEVICE(int i) { ptr[i] = 90.0 + i; });
      REQUIRE_FALSE(base.GetMemory().HostIsValid());
      REQUIRE(base.HostRead()[2] == 90.0);

      wrapped.SetData(base.HostReadWrite() + 4);
      REQUIRE(wrapped.Read(true) == device + 4);
      REQUIRE(inst.PrintPtrs(sink) == registrations);
   }
   Vector checked(base.HostReadWrite() + 2, 4, true);
   REQUIRE(checked.Read(true) == device + 2);
   REQUIRE(inst.PrintPtrs(sink) == registrations);
   REQUIRE(base.HostRead()[2] == 90.0);
}

TEST_CASE("HypreParMatrix reuses registered CSR buffers",
          "[HypreParMatrix][GPU]")
{
   auto &inst = MemoryManager::Instance();
   const auto d_mt = IsDeviceMemory(inst.GetDeviceMemoryType()) ?
                     inst.GetDeviceMemoryType() : MemoryType::DEVICE_DEBUG;
   Array<HYPRE_Int> rows(3, MemoryType::HOST), cols(2, MemoryType::HOST),
         offd_rows(3, MemoryType::HOST);
   Array<real_t> values(2, MemoryType::HOST);
   rows[0] = 0; rows[1] = 1; rows[2] = 2;
   cols[0] = 0; cols[1] = 1;
   offd_rows = 0;
   values[0] = 1.0; values[1] = 2.0;
   rows.GetMemory().SetDeviceMemoryType(d_mt);
   cols.GetMemory().SetDeviceMemoryType(d_mt);
   offd_rows.GetMemory().SetDeviceMemoryType(d_mt);
   values.GetMemory().SetDeviceMemoryType(d_mt);
   const auto device_rows = rows.Read(true);
   const auto device_cols = cols.Read(true);
   const auto device_offd_rows = offd_rows.Read(true);
   const auto device_values = values.Read(true);
   hypre_CSRMatrix diag = {}, offd = {};
   diag.num_rows = diag.num_cols = offd.num_rows = 2;
   diag.num_nonzeros = 2;
   diag.i = const_cast<HYPRE_Int *>(rows.HostRead());
   diag.j = const_cast<HYPRE_Int *>(cols.HostRead());
   diag.data = const_cast<real_t *>(values.HostRead());
   offd.i = const_cast<HYPRE_Int *>(offd_rows.HostRead());
#if MFEM_HYPRE_VERSION >= 21800
   diag.memory_location = offd.memory_location = HYPRE_MEMORY_HOST;
#endif
   hypre_ParCSRMatrix native = {};
   native.diag = &diag;
   native.offd = &offd;
   native.last_row_index = native.last_col_diag = 1;
   std::ostringstream sink;
   const int registrations = inst.PrintPtrs(sink);
   {
      HypreParMatrix wrapped(&native, false);
      REQUIRE(wrapped.GetDiagMemoryI().Read(true) == device_rows);
      REQUIRE(wrapped.GetDiagMemoryJ().Read(true) == device_cols);
      REQUIRE(wrapped.GetOffdMemoryI().Read(true) == device_offd_rows);
      REQUIRE(wrapped.GetDiagMemoryData().Read(true) == device_values);
      REQUIRE(inst.PrintPtrs(sink) == registrations);
      auto ptr = wrapped.GetDiagMemoryData().Write(true);
      forall(2, [=] MFEM_HOST_DEVICE(int i) { ptr[i] = 90.0 + i; });
      REQUIRE_FALSE(values.GetMemory().HostIsValid());
      REQUIRE(values.HostRead()[0] == 90.0);
   }
   Memory<real_t> checked;
   checked.CheckedWrap(const_cast<real_t *>(values.HostRead()), 2, false);
   REQUIRE(checked.Read(true) == device_values);
   REQUIRE(inst.PrintPtrs(sink) == registrations);
   REQUIRE(values.HostRead()[1] == 91.0);
}

TEST_CASE("ParNormlp", "[Parallel], [HypreParVector], [GPU]")
{
   int world_size, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int size_per_rank = 2;

   // global vector contains negative integers -[0, 1, ..., world_size*size_per_rank-1]
   // our rank only holds a portion
   Vector v(size_per_rank);
   for (int i=0; i<size_per_rank; ++i)
   {
      v(i) = -(rank * size_per_rank + i);
   }
   v.UseDevice(true);
   int N = world_size*size_per_rank;
   REQUIRE(ParNormlp(v, 1, MPI_COMM_WORLD) == MFEM_Approx((N * (N - 1)) / 2.));
   REQUIRE(ParNormlp(v, 2, MPI_COMM_WORLD) ==
           MFEM_Approx(sqrt((N * (N - 1) * (2 * N - 1)) / 6.)));
   REQUIRE(ParNormlp(v, 3, MPI_COMM_WORLD) ==
           MFEM_Approx(cbrt((N * N * (N - 1) * (N - 1)) / 4.)));
   // no easily computed exact answer, just make sure that p=1.5 is finite
   REQUIRE(std::isfinite(ParNormlp(v, 1.5, MPI_COMM_WORLD)));
   REQUIRE(ParNormlp(v, std::numeric_limits<real_t>::infinity(),
                     MPI_COMM_WORLD) == N - 1);
}

#endif // MFEM_USE_MPI

} // namespace mfem
