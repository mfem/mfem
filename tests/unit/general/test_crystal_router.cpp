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
   // Fix: get the host pointer before the move. Reason: GetData() is not
   // valid while only the device copy is valid.
   T *h = a.GetData();
   a.GetMemory().UseDevice(true);
   a.ReadWrite();
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      std::fill_n(h, a.Size(), poison);
   }
}

static void MoveToDevice(Vector &v)
{
   // Fix: get the host pointer before the move. Reason: GetData() is not
   // valid while only the device copy is valid.
   real_t *h = v.GetData();
   v.UseDevice(true);
   v.ReadWrite();
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      std::fill_n(h, v.Size(), std::numeric_limits<real_t>::quiet_NaN());
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

   // Fix: read through host pointers and a const reference. Reason: non-const
   // operator[] is not valid while a device copy is valid.
   const unsigned long long *h_keys = keys.HostRead();
   const int *h_ai = ai.HostRead();
   const real_t *h_v = v.HostRead();
   pv.HostRead();
   const ParticleVector &h_pv = pv;

   const int m = keys.Size();
   REQUIRE(ai.Size() == 2*m);
   REQUIRE(v.Size() == 3*m);
   REQUIRE(pv.GetNumParticles() == m);

   int wrong = 0;
   for (int i = 0; i < m; i++)
   {
      const unsigned long long gid = h_keys[i];
      bool ok = TargetRank(gid, size) == rank;
      for (int c = 0; c < 2; c++)
      {
         ok = ok && h_ai[2*i + c] == static_cast<int>(2*gid + c);
      }
      for (int c = 0; c < 3; c++) { ok = ok && h_v[c*m + i] == gid + 0.25*c; }
      for (int c = 0; c < 2; c++)
      {
         ok = ok && h_pv(i, c) == -static_cast<real_t>(gid) - c;
      }
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

static CR::internal::ArrayField<int> CrWrap(Array<int> &a, int vd,
                                            Ordering::Type o)
{ return CR::internal::ArrayField<int>(a, vd, o); }
static CR::internal::VectorField CrWrap(Vector &v, int vd, Ordering::Type o)
{ return CR::internal::VectorField(v, vd, o); }
static void CrMove(Array<int> &a) { MoveToDevice(a, -1); }
static void CrMove(Vector &v) { MoveToDevice(v); }

// C: Array<int> or Vector, T: its element type
template <typename C, typename T>
static void CrRunColumnCases(int vdim, Ordering::Type ord)
{
   static const std::vector<CrFillCase> fill_cases =
   {
      // shrink, tail fills a hole
      {{1, 3}, {1, 3},    {5}, {0, -1, 2, 5, 4},             1},
      // shrink, leavers at the end
      {{4, 5}, {},        {},  {0, 1, 2, 3},                 0},
      // shrink, no arrivals
      {{0, 5}, {0},       {4}, {4, 1, 2, 3},                 0},
      // same count: in place
      {{2},    {2},       {},  {0, 1, -1, 3, 4, 5},          1},
      // grow past capacity
      {{2},    {2, 6, 7}, {},  {0, 1, -1, 3, 4, 5, -2, -3},  3},
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
            const int j = CR::internal::ElemIndex(i, c, n, vdim, ord);
            col.HostWrite()[j] = T(10*i + c);
         }
      }
      // Fix: get the host pointer before the move. Reason: GetData() is not
      // valid while a device copy is valid.
      const T *data_before = col.GetData();
      CrMove(col);
      auto f = CrWrap(col, vdim, ord);

      Array<int> leaving(static_cast<int>(fc.leaving.size()));
      for (int k = 0; k < leaving.Size(); k++) { leaving[k] = fc.leaving[k]; }

      // Pack
      Array<std::uint64_t> rows(static_cast<int>(leaving.Size()*stride / sizeof(
                                                    std::uint64_t)));
      char *wr = reinterpret_cast<char*>(f.OnDevice() ? rows.Write(
                                            true) : rows.HostWrite());
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
      Array<int> dst(static_cast<int>(fc.dst.size())),
            tail(static_cast<int>(fc.tail.size()));
      for (int k = 0; k < dst.Size(); k++) { dst[k] = fc.dst[k]; }
      for (int k = 0; k < tail.Size(); k++) { tail[k] = fc.tail[k]; }
      Array<std::uint64_t> arr(static_cast<int>(fc.n_arr*stride / sizeof(
                                                   std::uint64_t)));
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
         ab = reinterpret_cast<const char*>(f.OnDevice() ? arr.Read(
                                               true) : arr.HostRead());
      }
      const int n_new = static_cast<int>(fc.expect.size());
      const int cap_before = col.Capacity();
      f.Unpack(dst, tail, ab, fc.n_arr, n_new, stride, off);

      REQUIRE(col.Size() == n_new*vdim);
      REQUIRE(col.Capacity() >= cap_before);
      if (n_new*vdim > cap_before) { REQUIRE(col.Capacity() >= 2*cap_before); }
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
      if (n_new == n) { REQUIRE(h == data_before); }
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

// gid = 1000*rank + i. keys = gid; ai (int, vdim 3, byNODES) = 3*gid + c;
// v (real_t, vdim 2, byVDIM) = gid + 0.5*c
static void MakeColumns(int n, Array<unsigned long long> &keys, Array<int> &ai,
                        Vector &v)
{
   const int rank = Mpi::WorldRank();
   keys.SetSize(n);
   ai.SetSize(3*n);
   v.SetSize(2*n);
   for (int i = 0; i < n; i++)
   {
      const unsigned long long gid = 1000ull*rank + i;
      keys[i] = gid;
      for (int c = 0; c < 3; c++) { ai[c*n + i] = static_cast<int>(3*gid + c); }
      for (int c = 0; c < 2; c++) { v[2*i + c] = gid + 0.5*c; }
   }
}

// particles on this rank whose ai or v values do not match their key
static int CountWrong(const Array<unsigned long long> &keys,
                      const Array<int> &ai, const Vector &v)
{
   const int m = keys.Size();
   if (ai.Size() != 3*m || v.Size() != 2*m) { return std::max(m, 1); }
   int wrong = 0;
   for (int i = 0; i < m; i++)
   {
      const unsigned long long gid = keys[i];
      bool ok = true;
      for (int c = 0; c < 3; c++)
      {
         ok = ok && ai[c*m + i] == static_cast<int>(3*gid + c);
      }
      for (int c = 0; c < 2; c++) { ok = ok && v[2*i + c] == gid + 0.5*c; }
      if (!ok) { wrong++; }
   }
   return wrong;
}

// MoveToDevice. A column with 0 entries only gets the device flag: it is never
// read or written, but the router still takes the device path for it.
template <typename T>
static void CrDev(Array<T> &a, T poison)
{
   if (a.Size() > 0) { MoveToDevice(a, poison); }
   else { a.GetMemory().UseDevice(true); }
}
static void CrDev(Vector &v)
{
   if (v.Size() > 0) { MoveToDevice(v); }
   else { v.UseDevice(true); }
}

// HostRead, skipped for a column with 0 entries
// Fix: return the host pointer. Reason: the tests read through it, non-const
// operator[] is not valid while a device copy is valid.
template <typename T>
static const T *CrHost(const Array<T> &a)
{ return a.Size() > 0 ? a.HostRead() : nullptr; }
static const real_t *CrHost(const Vector &v)
{ return v.Size() > 0 ? v.HostRead() : nullptr; }

static int CrSum(int x)
{
   MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   return x;
}

TEST_CASE("CrystalRouter empty ranks and one receiver",
          "[CrystalRouter][Parallel][GPU]")
{
   const int size = Mpi::WorldSize();
   const int rank = Mpi::WorldRank();
   const int n = rank % 2 ? 0 : 30 + rank;
   int global = 0;
   for (int r = 0; r < size; r += 2) { global += 30 + r; }

   Array<unsigned long long> keys;
   Array<int> ai;
   Vector v;
   MakeColumns(n, keys, ai, v);
   Array<unsigned int> rank_list(n);
   rank_list = static_cast<unsigned int>(size - 1);
   CrDev(keys, ~0ull); CrDev(ai, -1); CrDev(v);

   CrystalRouter router(MPI_COMM_WORLD);
   router.Route(rank_list,
   {
      CR::Array(keys), CR::Array(ai, 3, Ordering::byNODES),
      CR::Vector(v, 2, Ordering::byVDIM)
   });

   CrHost(keys); CrHost(ai); CrHost(v);
   const int wrong = CrSum(CountWrong(keys, ai, v));
   CHECK(wrong == 0);
   CHECK(keys.Size() == (rank == size - 1 ? global : 0));
}

TEST_CASE("CrystalRouter equal swap keeps columns in place",
          "[CrystalRouter][Parallel][GPU]")
{
   const int size = Mpi::WorldSize();
   const int rank = Mpi::WorldRank();
   const int n = 30;

   Array<unsigned long long> keys;
   Array<int> ai;
   Vector v;
   MakeColumns(n, keys, ai, v);
   Array<unsigned int> rank_list(n);
   for (int i = 0; i < n; i++)
   {
      rank_list[i] = i % 3 == 0 ? (rank + 1) % size : rank;
   }
   // Fix: get the host pointers before the move, compare with CrHost().
   // Reason: GetData() is not valid while a device copy is valid.
   const auto *pk = keys.GetData();
   const auto *pa = ai.GetData();
   const auto *pv = v.GetData();
   CrDev(keys, ~0ull); CrDev(ai, -1); CrDev(v);

   CrystalRouter router(MPI_COMM_WORLD);
   router.Route(rank_list,
   {
      CR::Array(keys), CR::Array(ai, 3, Ordering::byNODES),
      CR::Vector(v, 2, Ordering::byVDIM)
   });

   const unsigned long long *h_keys = CrHost(keys);
   const int *h_ai = CrHost(ai);
   const real_t *h_v = CrHost(v);
   const int wrong = CrSum(CountWrong(keys, ai, v));
   CHECK(wrong == 0);
   CHECK(keys.Size() == n);
   CHECK(ai.Size() == 3*n);
   CHECK(v.Size() == 2*n);
   CHECK(h_keys == pk);
   CHECK(h_ai == pa);
   CHECK(h_v == pv);
   if (keys.Size() == n)
   {
      for (int i = 0; i < n; i++)
      {
         if (i % 3 != 0) { CHECK(h_keys[i] == 1000ull*rank + i); }
         else
         {
            const int src = (rank + size - 1) % size;
            CHECK(h_keys[i] / 1000 == static_cast<unsigned long long>(src));
         }
      }
   }
}

TEST_CASE("CrystalRouter all particles stay",
          "[CrystalRouter][Parallel][GPU]")
{
   const int rank = Mpi::WorldRank();
   const int n = 25 + rank;

   Array<unsigned long long> keys;
   Array<int> ai;
   Vector v;
   MakeColumns(n, keys, ai, v);
   Array<unsigned int> rank_list(n);
   rank_list = static_cast<unsigned int>(rank);
   // Fix: get the host pointers before the move, compare with CrHost().
   // Reason: GetData() is not valid while a device copy is valid.
   const auto *pk = keys.GetData();
   const auto *pa = ai.GetData();
   const auto *pv = v.GetData();
   CrDev(keys, ~0ull); CrDev(ai, -1); CrDev(v);

   CrystalRouter router(MPI_COMM_WORLD);
   router.Route(rank_list,
   {
      CR::Array(keys), CR::Array(ai, 3, Ordering::byNODES),
      CR::Vector(v, 2, Ordering::byVDIM)
   });

   const unsigned long long *h_keys = CrHost(keys);
   const int *h_ai = CrHost(ai);
   const real_t *h_v = CrHost(v);
   const int wrong = CrSum(CountWrong(keys, ai, v));
   CHECK(wrong == 0);
   CHECK(keys.Size() == n);
   CHECK(h_keys == pk);
   CHECK(h_ai == pa);
   CHECK(h_v == pv);
   if (keys.Size() == n)
   {
      for (int i = 0; i < n; i++) { CHECK(h_keys[i] == 1000ull*rank + i); }
   }
}

TEST_CASE("CrystalRouter mixed element sizes",
          "[CrystalRouter][Parallel][GPU]")
{
   const int size = Mpi::WorldSize();
   const int rank = Mpi::WorldRank();
   const int n = 40 + 3*rank;
   int total = 0;
   for (int r = 0; r < size; r++) { total += 40 + 3*r; }

   Array<unsigned long long> keys;
   Array<int> ai;
   Vector v;
   MakeColumns(n, keys, ai, v);
   Array<int> tag(n);
   Array<unsigned int> rank_list(n);
   for (int i = 0; i < n; i++)
   {
      const unsigned long long gid = 1000ull*rank + i;
      tag[i] = static_cast<int>(gid % 7);
      rank_list[i] = TargetRank(gid, size);
   }
   CrDev(keys, ~0ull); CrDev(ai, -1); CrDev(v); CrDev(tag, -1);

   CrystalRouter router(MPI_COMM_WORLD);
   router.Route(rank_list,
   {
      CR::Array(tag), CR::Array(keys), CR::Array(ai, 3, Ordering::byNODES),
      CR::Vector(v, 2, Ordering::byVDIM)
   });

   CrHost(ai); CrHost(v);
   // Fix: read keys and tag through host pointers. Reason: see CrHost().
   const unsigned long long *h_keys = CrHost(keys);
   const int *h_tag = CrHost(tag);
   const int wrong = CrSum(CountWrong(keys, ai, v));
   int bad_tag = tag.Size() == keys.Size() ? 0 : std::max(keys.Size(), 1);
   if (bad_tag == 0)
   {
      for (int i = 0; i < keys.Size(); i++)
      {
         if (h_tag[i] != static_cast<int>(h_keys[i] % 7)) { bad_tag++; }
      }
   }
   bad_tag = CrSum(bad_tag);
   const int global = CrSum(keys.Size());
   CHECK(wrong == 0);
   CHECK(bad_tag == 0);
   CHECK(global == total);
}

TEST_CASE("CrystalRouter host and device columns",
          "[CrystalRouter][Parallel][GPU]")
{
   const int size = Mpi::WorldSize();
   const int rank = Mpi::WorldRank();
   const int n = 40 + 7*rank;
   int total = 0;
   for (int r = 0; r < size; r++) { total += 40 + 7*r; }

   Array<unsigned long long> keys;
   Array<int> ai;
   Vector v;
   MakeColumns(n, keys, ai, v);
   Array<unsigned int> rank_list(n);
   for (int i = 0; i < n; i++)
   {
      rank_list[i] = TargetRank(1000ull*rank + i, size);
   }
   CrDev(keys, ~0ull); CrDev(ai, -1);   // v stays host-only

   CrystalRouter router(MPI_COMM_WORLD);
   router.Route(rank_list,
   {
      CR::Array(keys), CR::Array(ai, 3, Ordering::byNODES),
      CR::Vector(v, 2, Ordering::byVDIM)
   });

   // before any HostRead of v: the router must not have moved it
   const bool v_host = v.GetMemory().HostIsValid();
   const bool v_dev = v.GetMemory().DeviceIsValid();
   CrHost(keys); CrHost(ai); CrHost(v);
   const int wrong = CrSum(CountWrong(keys, ai, v));
   const int global = CrSum(keys.Size());
   CHECK(v_host);
   CHECK(!v_dev);
   CHECK(wrong == 0);
   CHECK(global == total);
}

TEST_CASE("CrystalRouter reuse one router",
          "[CrystalRouter][Parallel][GPU]")
{
   const int size = Mpi::WorldSize();
   const int rank = Mpi::WorldRank();
   const int n = 30 + rank;
   int total = 0;
   for (int r = 0; r < size; r++) { total += 30 + r; }

   Array<unsigned long long> keys;
   Array<int> ai;
   Vector v;
   MakeColumns(n, keys, ai, v);
   CrDev(keys, ~0ull); CrDev(ai, -1); CrDev(v);

   CrystalRouter router(MPI_COMM_WORLD);

   // Call 1: equal swap
   Array<unsigned int> rank_list(n);
   for (int i = 0; i < n; i++)
   {
      rank_list[i] = i % 3 == 0 ? (rank + 1) % size : rank;
   }
   router.Route(rank_list,
   {
      CR::Array(keys), CR::Array(ai, 3, Ordering::byNODES),
      CR::Vector(v, 2, Ordering::byVDIM)
   });
   CrHost(keys); CrHost(ai); CrHost(v);
   const int w1 = CrSum(CountWrong(keys, ai, v));

   // Call 2: everything to the last rank, more rows than call 1
   Array<unsigned int> rank_list2(keys.Size());
   rank_list2 = static_cast<unsigned int>(size - 1);
   router.Route(rank_list2,
   {
      CR::Array(keys), CR::Array(ai, 3, Ordering::byNODES),
      CR::Vector(v, 2, Ordering::byVDIM)
   });
   CrHost(keys); CrHost(ai); CrHost(v);
   const int w2 = CrSum(CountWrong(keys, ai, v));
   const int global = CrSum(keys.Size());

   CHECK(w1 == 0);
   CHECK(w2 == 0);
   CHECK(global == total);
   CHECK(keys.Size() == (rank == size - 1 ? total : 0));
}

// Fix: test the index list after a device pack. Reason: Route() read
// leaving_idx with operator[] while only the device copy was usable.
TEST_CASE("CrystalRouter host index list after device pack",
          "[CrystalRouter][Parallel][GPU]")
{
   const int size = Mpi::WorldSize();
   const int rank = Mpi::WorldRank();
   const int n = 20 + rank;
   int total = 0;
   for (int r = 0; r < size; r++) { total += 20 + r; }

   Array<unsigned long long> keys;
   Array<int> ai;
   Vector v;
   MakeColumns(n, keys, ai, v);
   // each rank sends its even particles: at least one when size > 1
   Array<unsigned int> rank_list(n);
   for (int i = 0; i < n; i++)
   {
      rank_list[i] = i % 2 == 0 ? (rank + 1) % size : rank;
   }
   CrDev(keys, ~0ull); CrDev(ai, -1); CrDev(v);

   CrystalRouter router(MPI_COMM_WORLD);
   router.Route(rank_list,
   {
      CR::Array(keys), CR::Array(ai, 3, Ordering::byNODES),
      CR::Vector(v, 2, Ordering::byVDIM)
   });

   CrHost(ai); CrHost(v);
   const unsigned long long *h_keys = CrHost(keys);
   // an even particle must be on the rank after its source rank
   int misplaced = 0;
   for (int i = 0; i < keys.Size(); i++)
   {
      const int src = static_cast<int>(h_keys[i] / 1000);
      const int want = h_keys[i] % 2 == 0 ? (src + 1) % size : src;
      if (want != rank) { misplaced++; }
   }
   misplaced = CrSum(misplaced);
   const int wrong = CrSum(CountWrong(keys, ai, v));
   const int global = CrSum(keys.Size());
   CHECK(wrong == 0);
   CHECK(misplaced == 0);
   CHECK(global == total);
}

// Fix: test a small send with arrivals after a large arrival. Reason: with no
// buffer reset in Route(), call 2 faults on the debug device (-d debug).
TEST_CASE("CrystalRouter reuse after a large arrival",
          "[CrystalRouter][Parallel][GPU]")
{
   const int size = Mpi::WorldSize();
   const int rank = Mpi::WorldRank();
   const int n = 4000;

   Array<unsigned long long> keys;
   Array<int> ai;
   Vector v;
   MakeColumns(n, keys, ai, v);
   CrDev(keys, ~0ull); CrDev(ai, -1); CrDev(v);

   CrystalRouter router(MPI_COMM_WORLD);

   // Call 1: each rank r > 0 sends 2000 particles to rank 0, all others stay.
   // The buffer of rank 0 ends with 2000*(size - 1) rows.
   Array<unsigned int> rank_list1(n);
   for (int i = 0; i < n; i++) { rank_list1[i] = i < 2000 ? 0 : rank; }
   router.Route(rank_list1,
   {
      CR::Array(keys), CR::Array(ai, 3, Ordering::byNODES),
      CR::Vector(v, 2, Ordering::byVDIM)
   });
   const int global1 = CrSum(keys.Size());

   // Call 2: rank 0 sends one particle to the last rank, each rank r > 0
   // sends 1500 particles to rank 0. Rank 0 packs one row, then its buffer
   // grows to many pages, but stays below its size at the end of call 1.
   const int n2 = keys.Size();
   Array<unsigned int> rank_list2(n2);
   for (int i = 0; i < n2; i++) { rank_list2[i] = i < 1500 ? 0 : rank; }
   if (rank == 0 && n2 > 0)
   {
      rank_list2[0] = static_cast<unsigned int>(size - 1);
   }
   router.Route(rank_list2,
   {
      CR::Array(keys), CR::Array(ai, 3, Ordering::byNODES),
      CR::Vector(v, 2, Ordering::byVDIM)
   });
   CrHost(keys); CrHost(ai); CrHost(v);
   // the particles are not on their TargetRank here: check only the data
   const int wrong = CrSum(CountWrong(keys, ai, v));
   const int global2 = CrSum(keys.Size());

   CHECK(global1 == n*size);
   CHECK(global2 == n*size);
   CHECK(wrong == 0);
   if (rank == 0)
   {
      CHECK(keys.Size() == (size > 1 ? n + 3500*(size - 1) - 1 : n));
   }
}

#endif // MFEM_USE_MPI
