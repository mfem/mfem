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
static const Array<int> FieldVDims({2,3,1,5});
static constexpr int NumTags = 3;

static constexpr int N = 100;
static constexpr int N_rm = 37;
static_assert(N_rm < N);

void InitializeRandom(Particle &p, int seed)
{
   std::mt19937 gen(seed);
   std::uniform_real_distribution<> real_dist(0.0,1.0);
   std::uniform_int_distribution<> int_dist;

   for (int i = 0; i < p.GetDim(); i++)
   {
      p.Coords()[i] = real_dist(gen);
   }
   for (int f = 0; f < p.GetNF(); f++)
   {
      for (int c = 0; c < p.FieldVDim(f); c++)
      {
         p.FieldValue(f,c) = real_dist(gen);
      }
   }

   for (int t = 0; t < p.GetNT(); t++)
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

   SECTION(std::string("Ordering: ") + (ordering == Ordering::byNODES ? "byNODES" :
                                        "byVDIM"))
   {
      ParticleSet pset(0, SpaceDim, FieldVDims, NumTags, ordering);

      SECTION("Add Particle object")
      {
         for (int i = 0; i < N; i++)
         {
            pset.AddParticle(particles[i]);
         }
         REQUIRE(static_cast<int>(particles.size()) == pset.GetNP());


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
            REQUIRE(static_cast<int>(particles_rm.size()) == pset.GetNP());

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
         REQUIRE(static_cast<int>(particles.size()) == pset.GetNP());


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

   // NOTE: This test could fail if a point falls on an element boundary
   SECTION(std::string("Ordering: ") + (ordering == Ordering::byNODES ? "byNODES" :
                                        "byVDIM"))
   {
      // Add the particles uniquely to each rank particleset
      ParticleSet pset(MPI_COMM_WORLD, 0, SpaceDim, FieldVDims, NumTags, ordering);

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
      for (int i = 0; i < pset.GetNP(); i++)
      {
         Particle &actual_p = all_particles[pset.GetIDs()[i]];
         Particle pset_p = pset.GetParticle(i);

         if (actual_p != pset_p)
         {
            wrong_particle_count++;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &wrong_proc_count, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      CHECK(wrong_particle_count == 0);
   }

}

TEST_CASE("Particle Redistribution", "[ParticleSet]" "[Parallel]")
{
   TestRedistribute(Ordering::byNODES);
   TestRedistribute(Ordering::byVDIM);
}

#endif // MFEM_USE_MPI && MFEM_USE_GSLIB 