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

#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

using namespace std;
using namespace mfem;

static constexpr int SpaceDim = 3;
static const Array<int> FieldVDims({2,3,1,5});
static constexpr int NumTags = 3;

static constexpr int N = 100;
static constexpr int N_rm = 37;
static_assert(N_rm < N);

void InitializeRandom(Particle &p, int seed)
{
   std::mt19937 gen(seed);
   std::uniform_real_distribution<real_t> real_dist;
   std::uniform_int_distribution<> int_dist;

   for (int i = 0; i < p.GetDim(); i++)
   {
      p.Coords()[i] = real_dist(gen);
   }
   for (int f = 0; f < p.GetNFields(); f++)
   {
      for (int c = 0; c < p.GetFieldVDim(f); c++)
      {
         p.FieldValue(f,c) = real_dist(gen);
      }
   }

   for (int t = 0; t < p.GetNTags(); t++)
   {
      p.Tag(t) = int_dist(gen);
   }
}

void TestAddRemove(Ordering::Type ordering)
{
   // Initialize a vector of random particles
   int seed = 17;
   std::vector<Particle> particles;

   for (int i = 0; i < N; i++)
   {
      particles.emplace_back(SpaceDim, FieldVDims, NumTags);
      InitializeRandom(particles[i], seed);
      seed++;
   }

   // Generate random set of unique indices to remove particles from
   int rm_seed = 2;
   std::array<int, N> indices;
   std::iota(indices.begin(), indices.end(), 0);
   std::shuffle(indices.begin(), indices.end(),
                std::default_random_engine(rm_seed));
   Array<int> indices_rm(N_rm);
   for (int i = 0; i < N_rm; i++)
   {
      indices_rm[i] = indices[i];
   }
   indices_rm.Sort();

   // Create new vector of particles after removal
   std::vector<Particle> particles_rm = particles;
   for (int i = 0; i < N_rm; i++)
   {
      particles_rm.erase(particles_rm.begin() + indices_rm[i] - i);
   }

   SECTION(std::string("Ordering: ") +
           (ordering == Ordering::byNODES ? "byNODES" : "byVDIM"))
   {
      ParticleSet pset(0, SpaceDim, FieldVDims, NumTags, ordering);

      SECTION("Add Particle object")
      {
         for (int i = 0; i < N; i++)
         {
            pset.AddParticle(particles[i]);
         }
         REQUIRE(static_cast<int>(particles.size()) == pset.GetNParticles());


         int add_err_count = 0;
         for (int i = 0; i < N; i++)
         {
            Particle p = pset.GetParticle(i);
            if (particles[i] != p)
            {
               add_err_count++;
            }
         }
         REQUIRE(add_err_count == 0);

         SECTION("Remove particles")
         {
            pset.RemoveParticles(indices_rm);
            REQUIRE(static_cast<int>(particles_rm.size()) ==
                    pset.GetNParticles());

            int rm_err_count = 0;
            for (std::size_t i = 0; i < particles_rm.size(); i++)
            {
               Particle p = pset.GetParticle(i);
               if (particles_rm[i] != p)
               {
                  rm_err_count++;
               }
            }
            REQUIRE(rm_err_count == 0);
         }
      }

      SECTION("Add particles and set")
      {
         Array<int> new_idxs;
         pset.AddParticles(N, &new_idxs);

         for (int i = 0; i < new_idxs.Size(); i++)
         {
            pset.SetParticle(new_idxs[i], particles[i]);
         }
         REQUIRE(static_cast<int>(particles.size()) == pset.GetNParticles());


         int add_err_count = 0;
         for (int i = 0; i < N; i++)
         {
            Particle p = pset.GetParticle(i);
            if (particles[i] != p)
            {
               add_err_count++;
            }
         }
         REQUIRE(add_err_count == 0);
      }
   }
}

TEST_CASE("Adding + Removing Particles",
          "[ParticleSet]")
{

   TestAddRemove(Ordering::byNODES);
   TestAddRemove(Ordering::byVDIM);
}

TEST_CASE("Get Particle Reference", "[ParticleSet]")
{
   int seed = 17;
   Particle p1(SpaceDim, FieldVDims, NumTags);
   InitializeRandom(p1, seed);

   ParticleSet pset(1, SpaceDim, FieldVDims, NumTags, Ordering::byVDIM);
   Particle p2_copy_0 = pset.GetParticle(0);
   InitializeRandom(p2_copy_0, seed+1);
   pset.SetParticle(0, p2_copy_0);

   Particle p2_ref = pset.GetParticleRef(0);
   InitializeRandom(p2_ref, seed);

   Particle p2_copy_f = pset.GetParticle(0);

   REQUIRE(p1 == p2_copy_f);

}

#if defined(MFEM_USE_MPI)

// Make the device copy the only valid one. On a real GPU, also clobber the
// stale host copy so a router that reads host memory fails the checks.
template <typename T>
static void MoveToDevice(Array<T> &a, T poison)
{
   a.GetMemory().UseDevice(true);
   // Fix: take the host pointer before the move to the device. Reason:
   // GetData() fails a debug check when the host copy is not valid.
   T *h = a.GetData();
   a.ReadWrite();
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      std::fill_n(h, a.Size(), poison);
   }
}

static void MoveToDevice(Vector &v)
{
   v.UseDevice(true);
   // Fix: take the host pointer before the move to the device. Reason:
   // GetData() fails a debug check when the host copy is not valid.
   real_t *h = v.GetData();
   v.ReadWrite();
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      std::fill_n(h, v.Size(), std::numeric_limits<real_t>::quiet_NaN());
   }
}

void TestRedistributeNative(Ordering::Type ordering)
{
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();

   constexpr int target_tag = 0;

   std::vector<Particle> all_particles;
   int seed = 17;
   std::mt19937 gen(seed);
   std::uniform_int_distribution<> rank_dist(0, size - 1);

   for (int i = 0; i < N; i++)
   {
      all_particles.emplace_back(SpaceDim, FieldVDims, NumTags);
      Particle &p = all_particles.back();

      InitializeRandom(p, seed);

      p.Tag(target_tag) = rank_dist(gen);

      seed++;
   }

   int N_rank = N/size + ( rank < N % size ? 1 : 0);

   SECTION(std::string("Ordering: ") +
           (ordering == Ordering::byNODES ? "byNODES" : "byVDIM"))
   {
      ParticleSet pset(MPI_COMM_WORLD, 0, SpaceDim, FieldVDims,
                       NumTags, ordering);

      for (int i = 0; i < N_rank; i++)
      {
         pset.AddParticle(all_particles[i*size+rank]);
      }

      Array<unsigned int> rank_list(pset.GetNParticles());
      for (int i = 0; i < pset.GetNParticles(); i++)
      {
         rank_list[i] = static_cast<unsigned int>(pset.Tag(target_tag)[i]);
      }

      // GetIDs() is const-only; the test needs it device-resident too
      auto &ids = const_cast<Array<ParticleSet::IDType>&>(pset.GetIDs());
      MoveToDevice(ids, ~0ull);
      MoveToDevice(pset.Coords());
      for (int f = 0; f < pset.GetNFields(); f++) { MoveToDevice(pset.Field(f)); }
      for (int t = 0; t < pset.GetNTags(); t++) { MoveToDevice(pset.Tag(t), -1); }
      MoveToDevice(rank_list, ~0u);

      pset.Redistribute(rank_list);

      CHECK(static_cast<int>(pset.GetGlobalNParticles()) == N);

      // Fix: no manual sync; read tags and ids by HostRead() pointers. Reason:
      // GetParticle() syncs now, and non-const operator[] fails a debug check.
      const int *h_tag = pset.Tag(target_tag).HostRead();
      int wrong_proc_count = 0;
      for (int i = 0; i < pset.GetNParticles(); i++)
      {
         if (rank != h_tag[i])
         {
            wrong_proc_count++;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &wrong_proc_count, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      CHECK(wrong_proc_count == 0);

      const ParticleSet::IDType *h_ids = pset.GetIDs().HostRead();
      int wrong_particle_count = 0;
      for (int i = 0; i < pset.GetNParticles(); i++)
      {
         // a poisoned id would index out of bounds
         if (h_ids[i] >= static_cast<ParticleSet::IDType>(N))
         {
            wrong_particle_count++;
            continue;
         }
         Particle &actual_p = all_particles[h_ids[i]];
         Particle pset_p = pset.GetParticle(i);

         if (actual_p != pset_p)
         {
            wrong_particle_count++;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &wrong_particle_count, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      CHECK(wrong_particle_count == 0);
   }
}

TEST_CASE("Particle Redistribution (native)", "[ParticleSet][Parallel][GPU]")
{
   TestRedistributeNative(Ordering::byNODES);
   TestRedistributeNative(Ordering::byVDIM);
}

// Move all columns to the device, then send each particle to the next rank.
static void RedistributeOnDevice(ParticleSet &pset)
{
   Array<unsigned int> rank_list(pset.GetNParticles());
   rank_list = static_cast<unsigned int>((Mpi::WorldRank() + 1) %
                                         Mpi::WorldSize());

   // GetIDs() is const-only; the test needs it device-resident too
   auto &ids = const_cast<Array<ParticleSet::IDType>&>(pset.GetIDs());
   MoveToDevice(ids, ~0ull);
   MoveToDevice(pset.Coords());
   for (int f = 0; f < pset.GetNFields(); f++) { MoveToDevice(pset.Field(f)); }
   for (int t = 0; t < pset.GetNTags(); t++) { MoveToDevice(pset.Tag(t), -1); }
   MoveToDevice(rank_list, ~0u);

   pset.Redistribute(rank_list);
}

// Fix: test for the host syncs in GetParticle(), SetParticle() and
// AddParticles(). Reason: no test used them on device-only columns.
TEST_CASE("ParticleSet accessors after device Redistribute",
          "[ParticleSet][Parallel][GPU]")
{
   const int size = Mpi::WorldSize();
   const int rank = Mpi::WorldRank();
   constexpr int src_tag = 1; // holds the index in all_particles

   // Same list on all ranks. Rank r adds the particles r, r+size, ... so the
   // id of a particle is also its index in this list.
   std::vector<Particle> all_particles;
   for (int i = 0; i < N; i++)
   {
      all_particles.emplace_back(SpaceDim, FieldVDims, NumTags);
      InitializeRandom(all_particles[i], 17 + i);
      all_particles[i].Tag(src_tag) = i;
   }

   ParticleSet pset(MPI_COMM_WORLD, 0, SpaceDim, FieldVDims, NumTags);
   for (int i = rank; i < N; i += size) { pset.AddParticle(all_particles[i]); }
   const ParticleSet::IDType next_id = rank + size*pset.GetNParticles();

   RedistributeOnDevice(pset);

   // No manual host sync from here: the accessors must sync
   const int np = pset.GetNParticles();
   int wrong_get = 0, wrong_set = 0;
   for (int i = 0; i < np; i++)
   {
      Particle p = pset.GetParticle(i);
      const int src = p.Tag(src_tag);
      if (src < 0 || src >= N || p != all_particles[src]) { wrong_get++; }

      const int new_tag = -2 - i; // the source tags are not negative
      p.Tag(0) = new_tag;
      pset.SetParticle(i, p);
      if (pset.GetParticle(i).Tag(0) != new_tag) { wrong_set++; }
   }
   CHECK(wrong_get == 0);
   CHECK(wrong_set == 0);

   // A device access drops a host write that was made with no sync, so the
   // new tags must stay after a move to the device and back
   pset.Tag(0).ReadWrite();
   const int *h_tag0 = pset.Tag(0).HostRead();
   int wrong_sync = 0;
   for (int i = 0; i < np; i++)
   {
      if (h_tag0[i] != -2 - i) { wrong_sync++; }
   }
   CHECK(wrong_sync == 0);

   Particle extra(SpaceDim, FieldVDims, NumTags);
   InitializeRandom(extra, 17 + N);
   pset.AddParticle(extra);
   REQUIRE(pset.GetNParticles() == np + 1);
   CHECK(pset.GetParticle(np) == extra);

   // AddParticle() must keep the ids that were valid only on the device
   const ParticleSet::IDType *h_ids = pset.GetIDs().HostRead();
   const int *h_src = pset.Tag(src_tag).HostRead();
   int wrong_ids = 0;
   for (int i = 0; i < np; i++)
   {
      if (h_ids[i] != static_cast<ParticleSet::IDType>(h_src[i]))
      {
         wrong_ids++;
      }
   }
   CHECK(wrong_ids == 0);
   CHECK(h_ids[np] == next_id);
}

// Fix: test for the host sync of ids and tags in PrintCSV(). Reason: no test
// printed columns that are valid only on the device.
TEST_CASE("ParticleSet PrintCSV after device Redistribute",
          "[ParticleSet][Parallel][GPU]")
{
   ParticleSet pset(MPI_COMM_WORLD, 0, SpaceDim, FieldVDims, NumTags);
   for (int i = 0; i < 10; i++)
   {
      Particle p(SpaceDim, FieldVDims, NumTags);
      InitializeRandom(p, 17 + i);
      pset.AddParticle(p);
   }

   RedistributeOnDevice(pset);

   // No host sync before PrintCSV(): it must sync ids and tags
   const char *fname = "test_particleset_print.csv";
   pset.PrintCSV(fname);

   // The rows of rank 0 follow the header line. The columns are: id, rank,
   // coords, fields, tags.
   if (Mpi::Root())
   {
      const ParticleSet::IDType *h_ids = pset.GetIDs().HostRead();
      const int ncols = 2 + SpaceDim + FieldVDims.Sum() + NumTags;
      std::ifstream ifs(fname);
      std::string line;
      std::getline(ifs, line); // header
      int wrong_rows = 0;
      for (int i = 0; i < pset.GetNParticles(); i++)
      {
         std::vector<std::string> cols;
         if (std::getline(ifs, line))
         {
            std::istringstream iss(line);
            for (std::string c; std::getline(iss, c, ','); )
            {
               cols.push_back(c);
            }
         }
         bool ok = static_cast<int>(cols.size()) == ncols &&
                   std::stoull(cols[0]) == h_ids[i];
         for (int t = 0; ok && t < NumTags; t++)
         {
            ok = std::stoi(cols[ncols - NumTags + t]) ==
                 pset.Tag(t).HostRead()[i];
         }
         if (!ok) { wrong_rows++; }
      }
      CHECK(wrong_rows == 0);
      ifs.close();
      REQUIRE(std::remove(fname) == 0);
   }
}

#endif // MFEM_USE_MPI

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

static constexpr int N_e = 10;

template<typename T>
int CheckArrayEquality(const Array<T> &arr1, const Array<T> &arr2)
{
   MFEM_VERIFY(arr1.Size() == arr2.Size(), "arr1 and arr2 are not the same size!");

   int wrong_ct = 0;
   for (int i = 0; i < arr1.Size(); i++)
   {
      if (arr1[i] != arr2[i])
      {
         wrong_ct++;
      }
   }
   return wrong_ct;
}

void TestRedistribute(Ordering::Type ordering)
{
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();

   // Create a 3D hex mesh
   Mesh m = Mesh::MakeCartesian3D(N_e, N_e, N_e, Element::Type::HEXAHEDRON);

   // Generate a master list of all particles ; ID is the index
   // (This should be same on all ranks)
   // Ensure that all particles fall within an element (none on elem bdr)
   std::vector<Particle> all_particles;
   int seed = 17;
   std::mt19937 gen(seed);
   std::uniform_int_distribution<> int_dist(0, m.GetNE()-1);

   for (int i = 0; i < N; i++)
   {
      all_particles.emplace_back(SpaceDim, FieldVDims, NumTags);
      Particle &p = all_particles.back();

      // Initialize a particle with random coords, fields, and tags
      // Coords are [0.0, 1.0]
      InitializeRandom(p, seed);

      // Seed for a particular element on the mesh
      int elem = int_dist(gen);
      ElementTransformation &T = *m.GetElementTransformation(elem);

      // Rescale the coords to fall within [0.1,0.9] (of the to-be reference space of element)
      for (int d = 0; d < SpaceDim; d++)
      {
         p.Coords()[d] = 0.1 + p.Coords()[d]*0.8;
      }

      // Transform reference space coords to global
      IntegrationPoint ip;
      ip.Set(p.Coords().GetData(), SpaceDim);
      T.Transform(ip, p.Coords());

      seed++;
   }

   int N_rank = N/size + ( rank < N % size ? 1 : 0);

   ParMesh pmesh(MPI_COMM_WORLD, m);
   pmesh.EnsureNodes();

   SECTION(std::string("Ordering: ") +
           (ordering == Ordering::byNODES ? "byNODES" : "byVDIM"))
   {
      // Add the particles uniquely to each rank particleset
      ParticleSet pset(MPI_COMM_WORLD, 0, SpaceDim, FieldVDims,
                       NumTags, ordering);

      for (int i = 0; i < N_rank; i++)
      {
         pset.AddParticle(all_particles[i*size+rank]);
      }

      // Find points
      FindPointsGSLIB finder(MPI_COMM_WORLD);
      finder.Setup(pmesh);
      finder.FindPoints(pset.Coords(), ordering);

      // Ensure no code 1 nor 2 (all particles are within elements)
      int code_1_count = 0;
      int code_2_count = 0;
      const Array<unsigned int> &code = finder.GetCode();
      for (int i = 0; i < code.Size(); i++)
      {
         if (code[i] == 1)
         {
            code_1_count++;
         }

         if (code[i] == 2)
         {
            code_2_count++;
         }
      }
      CHECK(code_1_count == 0);
      CHECK(code_2_count == 0);

      // Redistribute
      pset.Redistribute(finder.GetProc());

      // Find again
      finder.FindPoints(pset.Coords(), ordering);

      const Array<unsigned int> &procs = finder.GetProc();

      int wrong_proc_count = 0;
      for (int i = 0; i < procs.Size(); i++)
      {
         if (rank != procs[i])
         {
            wrong_proc_count++;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &wrong_proc_count, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      CHECK(wrong_proc_count == 0);

      // Check that coordinates + fields + tags are all still correct
      int wrong_particle_count = 0;
      for (int i = 0; i < pset.GetNParticles(); i++)
      {
         Particle &actual_p = all_particles[pset.GetIDs()[i]];
         Particle pset_p = pset.GetParticle(i);

         if (actual_p != pset_p)
         {
            wrong_particle_count++;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &wrong_particle_count, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      CHECK(wrong_particle_count == 0);
   }
}

TEST_CASE("Particle Redistribution", "[ParticleSet][Parallel]")
{
   TestRedistribute(Ordering::byNODES);
   TestRedistribute(Ordering::byVDIM);
}

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB
