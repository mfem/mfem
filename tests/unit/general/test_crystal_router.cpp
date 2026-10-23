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

#include <limits>

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

#endif // MFEM_USE_MPI
