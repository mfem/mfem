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
//
//    -----------------------------------------------------------------------
//    Particles Redistribute (native): all-to-all transfer via CrystalRouter
//    -----------------------------------------------------------------------
//
// This miniapp highlights the ParticleSet::Redistribute feature, which uses
// the native MFEM CrystalRouter for all-to-all communication across ranks.
// Unlike particles_redist, it has no GSLIB dependency: rather than locating
// particles on a mesh, every particle is assigned a random target rank. That
// target is stored in a tag so it travels with the particle through the
// router, letting us verify afterwards that each particle landed on its
// assigned rank.
//
// Compile with: make particles_redist_native
//
// Sample runs:
//   mpirun -np 4 particles_redist_native
//   mpirun -np 4 particles_redist_native -npt 5000
//   mpirun -np 4 particles_redist_native -npt 7500 -m ../../data/star.mesh

#include "mfem.hpp"
#include "../common/particles_extras.hpp"

#include <random>

using namespace std;
using namespace mfem;
using namespace mfem::common;

// Count, per rank, how many local particles carry a target-rank tag equal to
// this rank ("owned") vs. destined elsewhere.
void PrintRankCounts(const ParticleSet &pset);

int main (int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int rank = Mpi::WorldRank();
   int size = Mpi::WorldSize();
   Hypre::Init();

   const char *mesh_file = "../../data/rt-2d-q3.mesh";
   int npt = 1000;
   int seed = 0;
   bool visualization = true;
   int visport = 19916;
   char vishost[] = "localhost";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use for the particle-placement bounding box.");
   args.AddOption(&npt, "-npt", "--num-particles",
                  "Number of particles to initialize per rank.");
   args.AddOption(&seed, "-s", "--seed", "Base random seed (offset by rank).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (rank == 0) { args.PrintOptions(cout); }

   // Create mesh
   Mesh mesh(mesh_file);
   int space_dim = mesh.Dimension();

   // Create parallel particle set
   // ParticleSet(MPI_Comm comm_, int rank_num_particles, int dim,
   //             const Array<int> &field_vdims, int num_tags);
   ParticleSet pset(MPI_COMM_WORLD, npt, space_dim, Array<int>(), 1);

   Vector pos_min, pos_max;
   mesh.GetBoundingBox(pos_min, pos_max);


   // Set particles randomly on entire mesh domain, for each rank (with seed)
   std::mt19937 gen(seed + rank);
   std::uniform_real_distribution<real_t> point_dist;
   std::uniform_int_distribution<int> rank_dist(0, size - 1);

   Array<unsigned int> ranks(pset.GetNParticles());
   for (int i = 0; i < pset.GetNParticles(); i++)
   {
      Particle p = pset.GetParticleRef(i);
      for (int d = 0; d < pset.GetDim(); d++)
      {
         p.Coords()[d] = pos_min[d] + point_dist(gen)*(pos_max[d] - pos_min[d]);
      }
      int target = rank_dist(gen);
      ranks[i] = static_cast<unsigned int>(target);
   }


   // Particle size for visualization
   real_t psize = Distance(pos_min, pos_max)*2e-3;
   if (visualization)
   {
      socketstream sock;
      Vector rank_vector(pset.GetNParticles());
      rank_vector = rank;
      VisualizeParticles(sock, vishost, visport,
                         pset, rank_vector, psize,
                         "Particle Owning Rank (Pre-Redistribute)",
                         0, 0, 400, 400, pset.GetDim() == 2 ? "Rjbcb" : "cb");
   }

   if (rank == 0)
   {
      mfem::out << "Pre-Redistribute:" << endl;
   }
   PrintRankCounts(pset);

   // Redistribute using the native CrystalRouter (no GSLIB)
   pset.Redistribute(ranks);

   if (rank == 0)
   {
      mfem::out << "\nPost-Redistribute:\n";
   }
   PrintRankCounts(pset);

   if (visualization)
   {
      socketstream sock;
      Vector rank_vector(pset.GetNParticles());
      rank_vector = rank;
      VisualizeParticles(sock, vishost, visport, pset, rank_vector, psize,
                         "Particle Owning Rank (Post-Redistribute)",
                         410, 0, 400, 400, pset.GetDim() == 2 ? "Rjbcb" : "cb");
   }

   return 0;
}




void PrintRankCounts(const ParticleSet &pset)
{
   int rank = Mpi::WorldRank(),
       size = Mpi::WorldSize();

   int on_rank = 0, off_rank = 0;
   for (int i = 0; i < pset.GetNParticles(); i++)
   {
      if (pset.Tag(0)[i] == rank)
      {
         on_rank++;
      }
      else
      {
         off_rank++;
      }
   }

   std::vector<int> all_on_rank(size), all_off_rank(size);
   MPI_Gather(&on_rank, 1, MPI_INT, all_on_rank.data(), 1, MPI_INT, 0,
              MPI_COMM_WORLD);
   MPI_Gather(&off_rank, 1, MPI_INT, all_off_rank.data(), 1, MPI_INT, 0,
              MPI_COMM_WORLD);
   if (rank == 0)
   {
      for (int r = 0; r < size; r++)
      {
         mfem::out << "Rank " << r << " holds "
                   << all_on_rank[r] << " particles destined for it, "
                   << all_off_rank[r] << " destined elsewhere\n";
      }
   }
}
