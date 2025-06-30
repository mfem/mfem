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

using namespace std;
using namespace mfem;

static constexpr int SpaceDim = 3;
static constexpr int NumScalars = 2;
static const Array<int> VectorVDims({2,3,1,5});

static constexpr int N = 3;
static constexpr int N_rm = 1;

void InitializeRandom(Particle &p, int seed, const Vector &pos_min, const Vector &pos_max)
{
   std::mt19937 gen(seed);
   std::uniform_real_distribution<> real_dist(0.0,1.0);

   for (int i = 0; i < pos_min.Size(); i++)
   {
      p.GetCoords()[i] = pos_min[i] + (pos_max[i] - pos_min[i])*real_dist(gen);
   }

   for (int s = 0; s < p.GetNumScalars(); s++)
      p.GetScalar(s) = real_dist(gen);
   
   for (int v = 0; v < p.GetNumVectors(); v++)
   {
      for (int c = 0; c < p.GetVDim(v); c++)
      {
         p.GetVector(v)[c] = real_dist(gen);
      }
   }
}

template<Ordering::Type VOrdering>
void TestParticleSet()
{
   // Initialize a vector of random particles
   int seed = 17;
   std::vector<Particle> particles;

   Vector pos_min(SpaceDim), pos_max(SpaceDim);
   pos_min = 0.0;
   pos_max = 1.0;
   for (int i = 0; i < N; i++)
   {
      particles.emplace_back(SpaceDim, NumScalars, VectorVDims);
      InitializeRandom(particles[i], seed, pos_min, pos_max);
      seed++;
   }

   // Generate random set of unique indices to remove particles from
   int rm_seed = 2;
   std::array<int, N> indices;
   std::iota(indices.begin(), indices.end(), 0);
   std::shuffle(indices.begin(), indices.end(), std::default_random_engine(rm_seed));
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

   SECTION(std::string("Ordering: ") + (VOrdering == Ordering::byNODES ? "byNODES" : "byVDIM"))
   {
      ParticleSet<VOrdering> pset(SpaceDim, NumScalars, VectorVDims);

      SECTION("Add")
      {
         for (int i = 0; i < N; i++)
            pset.AddParticle(particles[i]);
         REQUIRE(particles.size() == pset.GetNP());


         int add_err_count = 0;
         for (int i = 0; i < N; i++)
         {
            if (particles[i] != pset.GetParticleData(i))
               add_err_count++;
         }
         REQUIRE(add_err_count == 0);

         SECTION("Remove")
         {
            pset.RemoveParticles(indices_rm);
            REQUIRE(particles_rm.size() == pset.GetNP());


            int rm_err_count = 0;
            for (int i = 0; i < particles_rm.size(); i++)
            {
               if (particles_rm[i] != pset.GetParticleData(i))
                  rm_err_count++;
            }
            REQUIRE(rm_err_count == 0);
         }
      }
   }
}

TEST_CASE("Adding + Removing Particles",
          "[ParticleSet]")
{

   TestParticleSet<Ordering::byNODES>();
   TestParticleSet<Ordering::byVDIM>();
}

/*
TEST_CASE("Particle Redistribution", "[ParticleSet]" "[Parallel]")
{

   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();

   Mesh m = Mesh::MakeCartesian2D(N, N, Element::Type::QUADRILATERAL);
   ParMesh pmesh(MPI_COMM_WORLD, m);

   // Generate particles randomly on entire mesh domain, for each rank
   ParticleSet<SampleParticle, Ordering::byVDIM> pset(MPI_COMM_WORLD);
   int seed = rank;
   for (int i = 0; i < N; i++)
   {
      SampleParticle p;
      InitializeRandom(p, seed);
      pset.AddParticle(p);
      seed += size;
   }

   // Find points
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   pmesh.EnsureNodes();
   finder.Setup(pmesh);
   finder.FindPoints(pset.GetSetCoords(), pset.GetOrdering());

   // Redistribute
   pset.Redistribute(finder.GetProc());

   // Find again
   finder.FindPoints(pset.GetSetCoords(), pset.GetOrdering());

   // Ensure that all points lie on their proc
   const Array<unsigned int> &procs = finder.GetProc();
   int err_count = 0;
   for (int i = 0; i < procs.Size(); i++)
   {
      if (rank != procs[i])
      {
         err_count++;
      }
   }
   REQUIRE(err_count == 0);

}
*/