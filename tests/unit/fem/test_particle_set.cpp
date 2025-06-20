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


#include "../../../fem/particle_set.hpp"

using namespace std;
using namespace mfem;

using SampleParticle = Particle<2,2,3,2,1,5>;
static constexpr int N = 100;


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

TEST_CASE("Add Particles",
          "[ParticleSet]")
{
   // Initialize a random set of particles
   int seed = 17;

   SampleParticle particles[N];
   for (int i = 0; i < N; i++)
   {
      InitializeRandom(particles[i], seed);
      seed++;
   }

   SECTION("byNODES")
   {
      ParticleSet<SampleParticle, Ordering::byNODES> pset;
      for (int i = 0; i < N; i++)
         pset.AddParticle(particles[i]);

      int err_count = 0;

      for (int i = 0; i < N; i++)
      {
         // mfem::out << "Particle " << i << ":\n";
         // particles[i].Print();
         // cout << "\n";
         // pset.GetParticleData(i).Print();
         // mfem::out << "\n\n";
         if (particles[i] != pset.GetParticleData(i))
            err_count++;
      }

      REQUIRE(err_count == 0);
   }

   SECTION("byVDIM")
   {
      ParticleSet<SampleParticle, Ordering::byVDIM> pset;
      for (int i = 0; i < N; i++)
         pset.AddParticle(particles[i]);

      int err_count = 0;

      for (int i = 0; i < N; i++)
      {
         if (particles[i] != pset.GetParticleData(i))
            err_count++;
      }

      REQUIRE(err_count == 0);
   }
}