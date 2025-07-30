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
//    ----------------------------------------------------------------------
//    Particle Redistribute: Transfer particles to ranks they are located on
//    ----------------------------------------------------------------------
//
// This miniapp highlights the ParticleSet::Redistribute feature. Particles
// are initialized onto an input mesh by all ranks, and then re-distributed
// so that particles are held by the ranks of which they are actually physically
// located in on the ParMesh.
//
// Compile with: make particle-redistribute
//
// Sample runs:
//   mpirun -np 4 particle-redistribute -npt 1000
//   mpirun -np 4 particle-redistribute -npt 5000 -m ../../data/star.mesh
//   mpirun -np 4 particle-redistribute -npt 7500 -m ../../data/toroid-hex.mesh
//   mpirun -np 4 particle-redistribute -npt 7500 -m ../../data/fichera-q3.mesh

#include "mfem.hpp"
#include "../common/particles_extras.hpp"

#include <random>

using namespace std;
using namespace mfem;
using namespace mfem::common;

void PrintOnOffRankCounts(const Array<unsigned int> &procs, MPI_Comm comm)
{
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   int on_rank = 0, off_rank = 0;
   for (int i = 0; i < procs.Size(); i++)
   {
      if (procs[i] == rank)
      {
         on_rank++;
      }
      else
      {
         off_rank++;
      }
   }

   std::vector<int> all_on_rank(size), all_off_rank(size);
   MPI_Gather(&on_rank, 1, MPI_INT, all_on_rank.data(), 1, MPI_INT, 0, comm);
   MPI_Gather(&off_rank, 1, MPI_INT, all_off_rank.data(), 1, MPI_INT, 0, comm);
   if (rank == 0)
   {
      for (int r = 0; r < size; r++)
      {
         mfem::out << "Rank " << r << " owns " << all_on_rank[r] << " within it, " << all_off_rank[r] << " particles outside it\n";
      }
   }
}

template<typename T>
Vector ArrayToVector(const Array<T> &fields)
{
   Vector v(fields.Size());
   for (int i = 0; i < fields.Size(); i++)
   {
      v[i] = static_cast<real_t>(fields[i]);
   }
   return std::move(v);
}

int main (int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../../data/rt-2d-q3.mesh";
   int npt = 1;
   bool visualization = true;
   int visport = 19916;
   char vishost[] = "localhost";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use");
   args.AddOption(&npt, "-npt", "--num-particles", "Number of particles to initialize on global mesh bounding box.");
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
   // TODO: This should be in FindPointsGSLIB:
   MFEM_ASSERT(mesh.SpaceDimension() == mesh.Dimension(), "FindPointsGSLIB requires that the mesh space dimension + reference element dimension are the same");
   int space_dim = mesh.Dimension();

   ParticleSet pset(MPI_COMM_WORLD, npt, space_dim);

   Vector pos_min, pos_max;
   mesh.GetBoundingBox(pos_min, pos_max);

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   // Set particles randomly on entire mesh domain, for each rank
   std::mt19937 gen(rank);
   std::uniform_real_distribution<> real_dist(0.0,1.0);
   for (int i = 0; i < pset.GetNP(); i++)
   {
      Particle p = pset.GetParticleRef(i);
      for (int d = 0; d < pset.GetDim(); d++)
      {
         p.Coords()[d] = pos_min[d] + real_dist(gen)*(pos_max[d] - pos_min[d]);
      }
   }

   // Find points
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   pmesh.EnsureNodes();
   finder.Setup(pmesh);
   finder.FindPoints(pset.Coords(), pset.Coords().GetOrdering());

   // Remove points not in domain
   const Array<int> rm_idxs = finder.GetPointsNotFoundIndices();
   pset.RemoveParticles(rm_idxs);

   int num_removed = rm_idxs.Size();
   MPI_Allreduce(MPI_IN_PLACE, &num_removed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   if (rank == 0)
   {
      mfem::out << endl << "Removed " << num_removed << " particles that were not within the mesh" << endl << endl;
   }

   // Re-find w/ new particles to re-get Proc array
   finder.FindPoints(pset.Coords(), pset.Coords().GetOrdering());
   if (rank == 0)
   {
      mfem::out << "Pre-Redistribute:" << endl;
   }

   PrintOnOffRankCounts(finder.GetProc(), MPI_COMM_WORLD);

   real_t psize = Distance(pos_min, pos_max)*2e-3;
   if (visualization)
   {
      socketstream sock;
      Vector rank_vector(pset.GetNP());
      rank_vector = rank;
      VisualizeParticles(sock, vishost, visport, pset, rank_vector, psize, "Particle Owning Rank (Pre-Redistribute)", 0, 0, 400, 400, "bc");
   }

   // Redistribute
   pset.Redistribute(finder.GetProc());

   // Find again
   finder.FindPoints(pset.Coords(), pset.Coords().GetOrdering());

   // Remove particles not in domain

   if (rank == 0)
   {
      mfem::out << "\nPost-Redistribute:\n";
   }
   PrintOnOffRankCounts(finder.GetProc(), MPI_COMM_WORLD);

   if (visualization)
   {
      socketstream sock;
      Vector rank_vector(pset.GetNP());
      rank_vector = rank;
      VisualizeParticles(sock, "localhost", visport, pset, rank_vector, psize, "Particle Owning Rank (Post-Redistribute)", 410, 0, 400, 400, "bc");
   }

   return 0;
}
