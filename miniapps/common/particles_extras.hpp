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

#ifndef MFEM_PARTICLES_EXTRAS
#define MFEM_PARTICLES_EXTRAS

#include "mfem.hpp"
#include "fem_extras.hpp"
#include "pfem_extras.hpp"

namespace mfem
{
namespace common
{

void InitializeRandom(Particle &p, int seed, const Vector &pos_min, const Vector &pos_max);

/// Add a point to a given Mesh, represented as a hex sixed \p scale
void Add3DPoint(const Vector &center, Mesh &m, real_t scale=2e-3);

/// Plot a point cloud of particles, represented as hexes, colored by \p scalar_field
template<Ordering::Type VOrdering>
void VisualizeParticles(socketstream &sock, const char* vishost, int visport,
                                const ParticleSet<VOrdering> &pset, const Vector &scalar_field, real_t psize, 
                                const char* title, int x = 0, int y = 0, int w = 400, int h = 400,
                                const char* keys=nullptr);

class ParticleTrajVisualizer
{
private:
    socketstream sock;
    Mesh trajectories;

#ifdef MFEM_USE_MPI
    MPI_Comm comm;
#endif // MFEM_USE_MPI

public:

    ParticleVisualizer(int visport)
    : sock("localhost", visport) { }

#ifdef MFEM_USE_MPI
    ParticleVisualizer(MPI_Comm comm_, int visport)
    : sock("localhost", visport), comm(comm_) {}
#endif // MFEM_USE_MPI


    /// Plot particle trajectories, accounting for previous calls to this function, removed particles, and redistributed particles in parallel
    template<Ordering::Type VOrdering>
    void PlotParticleTrajectories(const ParticleSet<VOrdering> &pset);

};

} // namespace common
} // namespace mfem


#endif // MFEM_PARTICLES_EXTRAS