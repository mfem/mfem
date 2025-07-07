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

static constexpr int N = 100;
static constexpr int N_rm = 37;
static constexpr int N_e = 10;

void InitializeRandom(Particle &p, int seed)
{
   std::mt19937 gen(seed);
   std::uniform_real_distribution<> real_dist(0.0,1.0);

   for (int i = 0; i < p.GetSpaceDim(); i++)
   {
      p.GetCoords()[i] = real_dist(gen);
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

   for (int i = 0; i < N; i++)
   {
      particles.emplace_back(SpaceDim, NumScalars, VectorVDims);
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


TEST_CASE("Particle Redistribution", "[ParticleSet]" "[Parallel]")
{

   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();

   Mesh m = Mesh::MakeCartesian3D(N_e, N_e, N_e, Element::Type::QUADRILATERAL);
   ParMesh pmesh(MPI_COMM_WORLD, m);

   // Generate a master list of all particles ; ID is the index
   std::vector<Particle> all_particles;
   int seed = 17;
   for (int i = 0; i < N*size; i++)
   {
      all_particles.emplace_back(SpaceDim, NumScalars, VectorVDims);
      InitializeRandom(all_particles.back(), seed);
   }

   // Add the particles properly (based on index being ID) to the particle sets
   ParticleSet<Ordering::byVDIM> pset(MPI_COMM_WORLD, SpaceDim, NumScalars, VectorVDims);
   for (int i = 0; i < N; i++)
   {
      pset.AddParticle(all_particles[i*size+rank]);
   }


   // // Print Particles:
   // for (int r = 0; r < size; r++)
   // {
   //    if (r == rank)
   //    {
   //       cout << "-------------------------------------------\n RANK " << r << endl;
   //       for (int i = 0; i < pset.GetNP(); i++)
   //       {
   //          cout << "\nParticle ID " << pset.GetIDs()[i] << endl;
   //          Particle p = pset.GetParticleData(i);
   //          p.Print();
   //       }
   //    }
   //    MPI_Barrier(MPI_COMM_WORLD);
   // }


   // Find points
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   pmesh.EnsureNodes();
   finder.Setup(pmesh);
   finder.FindPoints(pset.GetSetCoords(), pset.GetOrdering());

   // Redistribute
   pset.Redistribute(finder.GetProc());


   // for (int r = 0; r < size; r++)
   // {
   //    if (r == rank)
   //    {
   //       cout << "-------------------------------------------\n RANK " << r << endl;
   //       for (int i = 0; i < pset.GetNP(); i++)
   //       {
   //          cout << "\nParticle ID " << pset.GetIDs()[i] << endl;
   //          Particle p = pset.GetParticleData(i);
   //          p.Print();
   //       }
   //    }
   //    MPI_Barrier(MPI_COMM_WORLD);
   // }

   // Find again
   finder.FindPoints(pset.GetSetCoords(), pset.GetOrdering());

   // Ensure that all points lie on their correct proc
   const Array<unsigned int> &procs = finder.GetProc();
   int wrong_proc_count = 0;
   for (int i = 0; i < procs.Size(); i++)
   {
      if (rank != procs[i])
      {
         wrong_proc_count++;
      }
   }
   REQUIRE(wrong_proc_count == 0);

   // Check that coordinates, scalars, + vectors are all still correct
   int wrong_coords_count = 0;
   int wrong_scalars_count = 0;
   int wrong_vectors_count = 0;
   for (int i = 0; i < pset.GetNP(); i++)
   {
      Particle &actual_p = all_particles[pset.GetIDs()[i]];
      Particle pset_p = pset.GetParticleData(i);

      
      for (int d = 0; d < SpaceDim; d++)
      {
         if (actual_p.GetCoords()[d] != pset_p.GetCoords()[d])
         {
            wrong_coords_count++;
            break;
         }
      }

      for (int s = 0; s < NumScalars; s++)
      {
         if (actual_p.GetScalar(s) != actual_p.GetScalar(s))
         {
            wrong_scalars_count++;
            break;
         }
      }

      for (int v = 0; v < VectorVDims.Size(); v++)
      {
         for (int c = 0; c < VectorVDims[v]; c++)
         {
            if (actual_p.GetVector(v)[c] != pset_p.GetVector(v)[c])
            {
               wrong_vectors_count++;
               v = VectorVDims.Size();
               break;
            }
         }
      }
   }

   REQUIRE(wrong_coords_count == 0);
   REQUIRE(wrong_scalars_count == 0);
   REQUIRE(wrong_vectors_count == 0);
   
}
