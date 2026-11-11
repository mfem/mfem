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

#include <cstdint>
#include <limits>
#include <vector>

using namespace mfem;

#ifdef MFEM_USE_MPI

// Make the device copy the only valid one. On a real GPU, also clobber the
// stale host copy so a router that reads host memory fails the checks.
template <typename T>
static void MoveToDevice(Array<T> &a, T poison)
{
   a.GetMemory().UseDevice(true);
   a.ReadWrite();
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      std::fill_n(a.GetData(), a.Size(), poison);
   }
}

static void MoveToDevice(Vector &v)
{
   v.UseDevice(true);
   v.ReadWrite();
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      std::fill_n(v.GetData(), v.Size(),
                  std::numeric_limits<real_t>::quiet_NaN());
   }
}

static int TargetRank(unsigned long long gid, int size)
{
   return static_cast<int>((gid * 2654435761ull) % size);
}

TEST_CASE("CrystalRouter device-resident fields",
          "[CrystalRouter][Parallel][GPU]")
{
   const int size = Mpi::WorldSize();
   const int rank = Mpi::WorldRank();

   // Uneven local counts; every value is a function of the global id
   const int n = 40 + 7*rank;
   Array<unsigned long long> keys(n);  // vdim 1
   Array<int> ai(2*n);                 // vdim 2, byVDIM
   Vector v(3*n);                      // vdim 3, byNODES
   ParticleVector pv(2, Ordering::byVDIM, n);
   Array<unsigned int> rank_list(n);

   for (int i = 0; i < n; i++)
   {
      const unsigned long long gid = 1000ull*rank + i;
      keys[i] = gid;
      for (int c = 0; c < 2; c++) { ai[2*i + c] = static_cast<int>(2*gid + c); }
      for (int c = 0; c < 3; c++) { v[c*n + i] = gid + 0.25*c; }
      for (int c = 0; c < 2; c++) { pv(i, c) = -static_cast<real_t>(gid) - c; }
      rank_list[i] = TargetRank(gid, size);
   }

   MoveToDevice(keys, ~0ull);
   MoveToDevice(ai, -1);
   MoveToDevice(v);
   MoveToDevice(pv);
   MoveToDevice(rank_list, ~0u);

   CrystalRouter router(MPI_COMM_WORLD);
   router.Route(rank_list,
   {
      CR::Array(keys),
      CR::Array(ai, 2, Ordering::byVDIM),
      CR::Vector(v, 3, Ordering::byNODES),
      CR::ParticleVector(pv)
   });

   keys.HostRead();
   ai.HostRead();
   v.HostRead();
   pv.HostRead();

   const int m = keys.Size();
   REQUIRE(ai.Size() == 2*m);
   REQUIRE(v.Size() == 3*m);
   REQUIRE(pv.GetNumParticles() == m);

   int wrong = 0;
   for (int i = 0; i < m; i++)
   {
      const unsigned long long gid = keys[i];
      bool ok = TargetRank(gid, size) == rank;
      for (int c = 0; c < 2; c++) { ok = ok && ai[2*i + c] == static_cast<int>(2*gid + c); }
      for (int c = 0; c < 3; c++) { ok = ok && v[c*m + i] == gid + 0.25*c; }
      for (int c = 0; c < 2; c++) { ok = ok && pv(i, c) == -static_cast<real_t>(gid) - c; }
      if (!ok) { wrong++; }
   }
   MPI_Allreduce(MPI_IN_PLACE, &wrong, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK(wrong == 0);

   int global_m = m;
   MPI_Allreduce(MPI_IN_PLACE, &global_m, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK(global_m == 40*size + 7*size*(size - 1)/2);
}

// expect[p] >= 0: old particle id; expect[p] = -k-1: arrival row k
struct CrFillCase { std::vector<int> leaving, dst, tail, expect; int n_arr; };

static CR::internal::ArrayField<int> CrWrap(Array<int> &a, int vd, Ordering::Type o)
{ return CR::internal::ArrayField<int>(a, vd, o); }
static CR::internal::VectorField CrWrap(Vector &v, int vd, Ordering::Type o)
{ return CR::internal::VectorField(v, vd, o); }
static void CrMove(Array<int> &a) { MoveToDevice(a, -1); }
static void CrMove(Vector &v) { MoveToDevice(v); }

// C: Array<int> or Vector, T: its element type
template <typename C, typename T>
static void CrRunColumnCases(int vdim, Ordering::Type ord)
{
   static const std::vector<CrFillCase> fill_cases = {
      {{1, 3}, {1, 3},    {5}, {0, -1, 2, 5, 4},             1}, // shrink, tail fills a hole
      {{4, 5}, {},        {},  {0, 1, 2, 3},                 0}, // shrink, leavers at the end
      {{0, 5}, {0},       {4}, {4, 1, 2, 3},                 0}, // shrink, no arrivals
      {{2},    {2},       {},  {0, 1, -1, 3, 4, 5},          1}, // same count: in place
      {{2},    {2, 6, 7}, {},  {0, 1, -1, 3, 4, 5, -2, -3},  3}, // grow past capacity
   };
   const int n = 6, off = 8;
   const std::size_t stride = (off + vdim*sizeof(T) + 7) / 8 * 8;

   for (const CrFillCase &fc : fill_cases)
   {
      C col(n*vdim);
      for (int i = 0; i < n; i++)
      {
         for (int c = 0; c < vdim; c++)
         {
            col.HostWrite()[CR::internal::ElemIndex(i, c, n, vdim, ord)] = T(10*i + c);
         }
      }
      CrMove(col);
      auto f = CrWrap(col, vdim, ord);

      Array<int> leaving(static_cast<int>(fc.leaving.size()));
      for (int k = 0; k < leaving.Size(); k++) { leaving[k] = fc.leaving[k]; }

      // Pack
      Array<std::uint64_t> rows(static_cast<int>(leaving.Size()*stride / sizeof(std::uint64_t)));
      char *wr = reinterpret_cast<char*>(f.OnDevice() ? rows.Write(true) : rows.HostWrite());
      f.Pack(leaving, wr, stride, off);
      const char *rd = reinterpret_cast<const char*>(rows.HostRead());
      for (int k = 0; k < leaving.Size(); k++)
      {
         for (int c = 0; c < vdim; c++)
         {
            REQUIRE(reinterpret_cast<const T*>(rd + k*stride + off)[c]
                    == T(10*fc.leaving[k] + c));
         }
      }

      // Fill
      Array<int> dst(static_cast<int>(fc.dst.size())), tail(static_cast<int>(fc.tail.size()));
      for (int k = 0; k < dst.Size(); k++) { dst[k] = fc.dst[k]; }
      for (int k = 0; k < tail.Size(); k++) { tail[k] = fc.tail[k]; }
      Array<std::uint64_t> arr(static_cast<int>(fc.n_arr*stride / sizeof(std::uint64_t)));
      if (fc.n_arr > 0)
      {
         char *aw = reinterpret_cast<char*>(arr.HostWrite());
         for (int k = 0; k < fc.n_arr; k++)
         {
            for (int c = 0; c < vdim; c++)
            {
               reinterpret_cast<T*>(aw + k*stride + off)[c] = T(100*(k+1) + c);
            }
         }
      }
      const char *ab = nullptr;
      if (fc.n_arr > 0)
      {
         ab = reinterpret_cast<const char*>(f.OnDevice() ? arr.Read(true) : arr.HostRead());
      }
      const int n_new = static_cast<int>(fc.expect.size());
      const int cap_before = col.Capacity();
      const T *data_before = col.GetData();
      f.Unpack(dst, tail, ab, fc.n_arr, n_new, stride, off);

      REQUIRE(col.Size() == n_new*vdim);
      REQUIRE(col.Capacity() >= cap_before);
      const T *h = col.HostRead();
      for (int p = 0; p < n_new; p++)
      {
         for (int c = 0; c < vdim; c++)
         {
            const int e = fc.expect[p];
            REQUIRE(h[CR::internal::ElemIndex(p, c, n_new, vdim, ord)]
                    == T(e >= 0 ? 10*e + c : 100*(-e) + c));
         }
      }
      if (n_new == n) { REQUIRE(col.GetData() == data_before); }
   }
}

TEST_CASE("CrystalRouter column pack and fill",
          "[CrystalRouter][Parallel][GPU]")
{
   for (Ordering::Type ord : {Ordering::byNODES, Ordering::byVDIM})
   {
      CrRunColumnCases<Vector, real_t>(1, ord);
      CrRunColumnCases<Vector, real_t>(3, ord);
      CrRunColumnCases<Array<int>, int>(2, ord);
   }
}

#endif // MFEM_USE_MPI
