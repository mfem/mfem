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
// are initialized onto a square mesh by all ranks, and then re-distributed
// so that particles are held by the ranks of which they are actually physically
// located in on the ParMesh.
//
// Compile with: make particle-redistribute
//
// Sample runs:
//   mpirun -np 4 particle-redistribute -n 1000
//   mpirun -np 4 particle-redistribute -n 5000 -m ../../data/star.mesh
//   mpirun -np 4 particle-redistribute -n 7500 -m ../../data/toroid-hex.mesh
//   mpirun -np 4 particle-redistribute -n 7500 -m ../../data/fichera-q3.mesh

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

   MPI_Barrier(comm);
   for (int i = 0; i < size; i++)
   {
      if (i == rank)
      {
         mfem::out << "Rank " << i << " owns " << on_rank << " within it, " << off_rank << " particles outside it\n";
      }
      MPI_Barrier(comm);
   }
}

void Add3DPoint(const Vector &center, real_t s, Mesh &m)
{

   Vector v[8];

   for (int i = 0; i < 8; i++)
   {
      v[i].SetSize(3);
   }

   for (int i = 0; i < center.Size(); i++)
   {
      v[0][i] = center[i];
   }

   Vector v_s(3); v_s = s;
   v[0] -= v_s;

   v[1] = v[0];
   v[1][0] += 2*s;

   v[2] = v[1];
   v[2][1] += 2*s;

   v[3] = v[2];
   v[3][0] -= 2*s;

   Vector v_s_z({0.0,0.0,s});
   for (int i = 4; i < 8; i++)
   {
      add(1.0, v[i-4], 2.0, v_s_z, v[i]);
   }

   for (int i = 0; i < 8; i++)
   {
      m.AddVertex(v[i]);
   }

   int vi[8];
   for (int i = 0; i < 8; i++)
   {
      vi[i] = i + (m.GetNE())*8;
   }
   m.AddHex(vi);
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
   int N = 10;
   bool visualization = true;
   int visport = 19916;
   char vishost[] = "localhost";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use");
   args.AddOption(&npt, "-n", "--npt", "Number of particles to initialize on global mesh bounding box by each rank.");
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
   MFEM_ASSERT(mesh.SpaceDimension() == mesh.Dimension(), "FindPointsGSLIB requires that the mesh space dimension + reference element dimension are the same");
   int space_dim = mesh.Dimension();

   ParticleSet<Ordering::byVDIM> pset(MPI_COMM_WORLD, space_dim, 0, Array<int>());

   Vector pos_min, pos_max;
   mesh.GetBoundingBox(pos_min, pos_max);

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   // Generate particles randomly on entire mesh domain, for each rank
   int seed = rank;
   for (int i = 0; i < npt; i++)
   {
      Particle p(space_dim, 0, {});
      InitializeRandom(p, seed, pos_min, pos_max);
      pset.AddParticle(p);
      seed += size;
   }

   // Find points
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   pmesh.EnsureNodes();
   finder.Setup(pmesh);
   finder.FindPoints(pset.GetSetCoords(), pset.GetOrdering());

   // Remove points not in domain
   Array<int> rm_idxs;
   const Array<unsigned int> &codes = finder.GetCode();
   for (int i = 0; i < pset.GetNP(); i++)
   {
      if (codes[i] == 2)
      {
         rm_idxs.Append(i);
      }
   }
   pset.RemoveParticles(rm_idxs);

   // Re-find w/ new particles to re-get Proc array
   finder.FindPoints(pset.GetSetCoords(), pset.GetOrdering());
   if (rank == 0)
   {
      mfem::out << "Pre-Redistribute:\n";
   }

   PrintOnOffRankCounts(finder.GetProc(), MPI_COMM_WORLD);

   real_t psize = Distance(pos_min, pos_max)*2e-3;
   if (visualization)
   {
      socketstream sock;
      Vector rank_vector(pset.GetNP());
      rank_vector = rank;
      ParticleVisualizer::VisualizeParticles(sock, "localhost", visport, pset, rank_vector, psize, "Particle Owning Rank (Pre-Redistribute)", 0, 0, 400, 400, "bc");
   }

   // Redistribute
   pset.Redistribute(finder.GetProc());

   // Find again
   finder.FindPoints(pset.GetSetCoords(), pset.GetOrdering());

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
      ParticleVisualizer::VisualizeParticles(sock, "localhost", visport, pset, rank_vector, psize, "Particle Owning Rank (Post-Redistribute)", 410, 0, 400, 400, "bc");
   }

   return 0;
}
