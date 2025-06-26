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

#include "fem/particleset.hpp"

using namespace std;
using namespace mfem;

using SampleParticle = Particle<2,2>;

static constexpr int N = 10;
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

template<typename T>
void SetVectorFromArray(const Array<T> &dat, Vector &v)
{
   v.SetSize(dat.Size());
   for (int i = 0; i < dat.Size(); i++)
   {
      v[i] = dat[i];
   }
}

int main (int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   int npt = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&npt, "-n", "--npt", "Number of particles to initialize on each rank.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (rank == 0) { args.PrintOptions(cout); }

   Mesh m = Mesh::MakeCartesian2D(N, N, Element::Type::QUADRILATERAL);
   ParMesh pmesh(MPI_COMM_WORLD, m);

   L2_FECollection l2fec(1,2);
   ParFiniteElementSpace pfes(&pmesh, &l2fec);
   ParGridFunction rank_gf(&pfes);
   rank_gf = rank;

   ParaViewDataCollection pvdc("Redistribute", &pmesh);
   pvdc.RegisterField("Rank", &rank_gf);
   pvdc.Save();

   // Generate particles randomly on entire mesh domain, for each rank
   ParticleSet<SampleParticle, Ordering::byVDIM> pset(MPI_COMM_WORLD);
   int seed = rank;
   for (int i = 0; i < npt; i++)
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

   // Output initial particles
   SetVectorFromArray(finder.GetCode(), pset.GetSetScalar(0));
   SetVectorFromArray(finder.GetProc(), pset.GetSetScalar(1));
   pset.PrintCSV("Particles_0.csv");

   // Redistribute
   pset.Redistribute(finder.GetProc());

   // Find again
   finder.FindPoints(pset.GetSetCoords(), pset.GetOrdering());


   // Output redistributed particles
   SetVectorFromArray(finder.GetCode(), pset.GetSetScalar(0));
   SetVectorFromArray(finder.GetProc(), pset.GetSetScalar(1));
   pset.PrintCSV("Particles_1.csv");

   return 0;
}
