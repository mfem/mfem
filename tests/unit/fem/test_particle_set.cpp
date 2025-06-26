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


#include "../../../fem/particleset.hpp"

using namespace std;
using namespace mfem;

using SampleParticle = Particle<2,2,3>;
static constexpr int N = 10;
static constexpr int N_rm = 32;

void InitializeRandom(SampleParticle &p, int seed)
{
   p.GetCoords().Randomize(seed);
   seed++;

   for (int s = 0; s < SampleParticle::GetNumScalars(); s++)
      p.GetScalar(s) = rand_real();
   
   for (int v = 0; v < SampleParticle::GetNumVectors(); v++)
   {
      p.GetVector(v).Randomize(seed);
      seed++;
   }
}

template<Ordering::Type VOrdering>
void TestParticleSet()
{
   // Initialize a vector of random particles
   int seed = 17;
   std::vector<SampleParticle> particles(N);
   for (int i = 0; i < N; i++)
   {
      InitializeRandom(particles[i], seed);
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
   std::vector<SampleParticle> particles_rm = particles;
   for (int i = 0; i < N_rm; i++)
   {
      particles_rm.erase(particles_rm.begin() + indices_rm[i] - i);
   }

   SECTION(std::string("Ordering: ") + (VOrdering == Ordering::byNODES ? "byNODES" : "byVDIM"))
   {
      ParticleSet<SampleParticle, VOrdering> pset;

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
         pset.PrintCSV("ParticleData");
      }
   }
}

TEST_CASE("Adding + Removing Particles",
          "[ParticleSet]")
{

   TestParticleSet<Ordering::byNODES>();
   TestParticleSet<Ordering::byVDIM>();
}


// To be removed -- just for testing right now
TEST_CASE("Parallel Particles I/O", "[ParticleSet]" "[Parallel]")
{
   ParticleSet<SampleParticle, Ordering::byVDIM> pset(MPI_COMM_WORLD);
   
   int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   int seed = 17*rank;
   for (int i = 0; i < N; i++)
   {
      SampleParticle p;
      InitializeRandom(p, seed);
      pset.AddParticle(p);
      seed++;
   }

   pset.PrintCSV("Test.csv");


}