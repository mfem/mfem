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

static constexpr int Dim = 3;
static constexpr int DataVDim = 7;

struct Particle
{
   Vector coords;
   Vector data;
   int id;
   Particle() : coords(Dim), data(DataVDim), id() {}; 
};

static constexpr int N = 100;
static constexpr int N_rm = 37;
static constexpr int N_e = 10;

// Better than Randomize:
void InitializeRandom(int seed, Vector &v)
{
   std::mt19937 gen(seed);
   std::uniform_real_distribution<> real_dist(0.0,1.0);

   for (int i = 0; i < v.Size(); i++)
   {
      v[i] = real_dist(gen);
   }
}

void CheckEquality(const ParticleSpace &pspace, const ParticleFunction &pf, const std::vector<Particle> &particles)
{
   REQUIRE(particles.size() == pspace.GetNP());

   // Verify particles added properly
   int err_coords_count = 0;
   int err_data_count = 0;
   int err_ids_count = 0;
   Vector pv(Dim);
   Vector pd(DataVDim);
   for (int i = 0; i < particles.size(); i++)
   {
      pspace.GetCoords().GetParticleData(i, pv);
      for (int d = 0; d < Dim; d++)
      {
         if (particles[i].coords[d] != pv[d])
         {
            err_coords_count++;
         }
      }
      pf.GetParticleData(i, pd);
      for (int vd = 0; vd < DataVDim; vd++)
      {
         if (particles[i].data[vd] != pd[vd])
         {
            err_data_count++;
         }
      }

      if (particles[i].id != pspace.GetID(i))
      {
         err_ids_count++;
      }

   }
   REQUIRE(err_coords_count == 0);
   REQUIRE(err_data_count == 0);
   REQUIRE(err_ids_count == 0);

}

void TestAddRemove(Ordering::Type ordering)
{
   // Initialize set of particles to add
   int seed = 17;
   std::vector<Particle> particles;
   for (int i = 0; i < N; i++)
   {
      particles.emplace_back();
      InitializeRandom(seed, particles.back().coords);
      seed++;
      InitializeRandom(seed, particles.back().data);
      seed++;

      particles.back().id = i;
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

   // Create new set of particles after removal
   std::vector<Particle> particles_rm = particles;
   for (int i = 0; i < N_rm; i++)
   {
      particles_rm.erase(particles_rm.begin() + indices_rm[i] - i);
   }

   SECTION(std::string("Ordering: ") + (ordering == Ordering::byNODES ? "byNODES" : "byVDIM"))
   {
      // Initialize an empty ParticleSpace w/o mesh
      ParticleSpace pspace(Dim, 0, ordering);
      ParticleFunction &pf = pspace.CreateParticleFunction(DataVDim);

      SECTION("Add")
      {
         // Add each particle individually
         Array<int> new_indices;
         for (int i = 0; i < N; i++)
         {
            pspace.AddParticles(particles[i].coords, &new_indices);
            pf.SetParticleData(new_indices, particles[i].data);
         }

         CheckEquality(pspace, pf, particles);

         
         SECTION("Remove")
         {
            // Remove particles
            pspace.RemoveParticles(indices_rm);
            CheckEquality(pspace, pf, particles_rm);
         }
      }
   }
}

TEST_CASE("Adding + Removing Particles",
          "[ParticleSet]")
{

   TestAddRemove(Ordering::byNODES);
   TestAddRemove(Ordering::byVDIM);
}


#if defined(MFEM_USE_MPI)

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

   Mesh m = Mesh::MakeCartesian3D(N_e, N_e, N_e, Element::Type::HEXAHEDRON);
   ParMesh pmesh(MPI_COMM_WORLD, m);
   pmesh.EnsureNodes();

   // Generate a master list of all particles ; ID is the index
   ParticleMeta pmeta(SpaceDim, NumProps, StateVDims);
   std::vector<Particle> all_particles;
   int seed = 17;
   for (int i = 0; i < N*size; i++)
   {
      all_particles.emplace_back(pmeta);
      InitializeRandom(all_particles.back(), seed);
      seed++;
   }

   SECTION(std::string("Ordering: ") + (ordering == Ordering::byNODES ? "byNODES" : "byVDIM"))
   {
      SECTION("With rank_list")
      {
         // Add the particles uniquely to each rank particleset
         ParticleSet pset(MPI_COMM_WORLD, pmeta, ordering);
         for (int i = 0; i < N; i++)
         {
            pset.AddParticle(all_particles[i*size+rank]);
         }

         // Find points
         FindPointsGSLIB finder(MPI_COMM_WORLD);
         finder.Setup(pmesh);
         finder.FindPoints(pset.GetAllCoords(), pset.GetOrdering());

         // Redistribute
         pset.Redistribute(finder.GetProc());

         // Find again
         finder.FindPoints(pset.GetAllCoords(), pset.GetOrdering());

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
         int wrong_props_count = 0;
         int wrong_state_count = 0;
         for (int i = 0; i < pset.GetNP(); i++)
         {
            Particle &actual_p = all_particles[pset.GetIDs()[i]];
            Particle pset_p(pmeta);
            pset.GetParticle(i, pset_p);

            
            for (int d = 0; d < pmeta.SpaceDim(); d++)
            {
               if (actual_p.GetCoords()[d] != pset_p.GetCoords()[d])
               {
                  wrong_coords_count++;
                  break;
               }
            }

            for (int s = 0; s < pmeta.NumProps(); s++)
            {
               if (actual_p.GetProperty(s) != actual_p.GetProperty(s))
               {
                  wrong_props_count++;
                  break;
               }
            }

            for (int v = 0; v < pmeta.NumStateVars(); v++)
            {
               for (int c = 0; c < pmeta.StateVDim(v); c++)
               {
                  if (actual_p.GetStateVar(v)[c] != pset_p.GetStateVar(v)[c])
                  {
                     wrong_state_count++;
                     v = pmeta.NumStateVars();
                     break;
                  }
               }
            }
         }

         REQUIRE(wrong_coords_count == 0);
         REQUIRE(wrong_props_count == 0);
         REQUIRE(wrong_state_count == 0);
      }
      SECTION("With FindPointsGSLIB")
      {
         // Add the particles uniquely to each rank particleset
         ParticleSet pset1(MPI_COMM_WORLD, pmeta, ordering);
         ParticleSet pset2(MPI_COMM_WORLD, pmeta, ordering);
         for (int i = 0; i < N; i++)
         {
            pset1.AddParticle(all_particles[i*size+rank]);
            pset2.AddParticle(all_particles[i*size+rank]);
         }


         FindPointsGSLIB finder1(MPI_COMM_WORLD);
         FindPointsGSLIB finder2(MPI_COMM_WORLD);
         finder1.Setup(pmesh);
         finder2.Setup(pmesh);
         
         finder1.FindPoints(pset1.GetAllCoords(), pset1.GetOrdering());
         finder2.FindPoints(pset2.GetAllCoords(), pset2.GetOrdering());

         pset1.Redistribute(finder1); // Redistribute points and FindPointsGSLIB data

         pset2.Redistribute(finder2.GetProc()); // Redistribute points
         finder2.FindPoints(pset2.GetAllCoords(), pset2.GetOrdering()); // Re-find FindPointsGSLIB data


         // // All Code, Elem, Proc, and ReferencePositions in finder1 and finder2 should now be equivalent
         MPI_Barrier(MPI_COMM_WORLD);
         int wrong_codes_count = CheckArrayEquality(finder1.GetCode(), finder2.GetCode());
         int wrong_elems_count = CheckArrayEquality(finder1.GetGSLIBElem(), finder2.GetGSLIBElem());
         int wrong_mfem_elems_count = CheckArrayEquality(finder1.GetElem(), finder2.GetElem());
         int wrong_procs_count = CheckArrayEquality(finder1.GetProc(), finder2.GetProc());
         bool correct_ref_coords = finder1.GetGSLIBReferencePosition().DistanceTo(finder2.GetGSLIBReferencePosition()) == MFEM_Approx(0.0);
         bool correct_mfem_ref_coords = finder1.GetReferencePosition().DistanceTo(finder2.GetReferencePosition()) == MFEM_Approx(0.0);

         REQUIRE(wrong_codes_count == 0);
         REQUIRE(wrong_elems_count == 0);
         REQUIRE(wrong_mfem_elems_count == 0);
         REQUIRE(wrong_procs_count == 0);
         REQUIRE(correct_ref_coords);
         REQUIRE(correct_mfem_ref_coords);
      }
   }
}

TEST_CASE("Particle Redistribution", "[ParticleSet]" "[Parallel]")
{
   TestRedistribute(Ordering::byNODES);
   TestRedistribute(Ordering::byVDIM);
   
}
#endif // MFEM_USE_MPI