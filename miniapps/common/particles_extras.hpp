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

/// Add a point to a given Mesh, represented as a hex sixed \p scale
void Add3DPoint(const Vector &center, Mesh &m, real_t scale=2e-3);

/// Plot a point cloud of particles, represented as hexes, colored by \p scalar_field
void VisualizeParticles(socketstream &sock, const char* vishost, int visport,
                                const ParticleSet &pset, const Vector &scalar_field, real_t psize, 
                                const char* title, int x = 0, int y = 0, int w = 400, int h = 400,
                                const char* keys=nullptr);

class ParticleTrajectories
{
protected:

    const ParticleSet &pset;

    socketstream sock;
    std::vector<Array<unsigned int>> segment_ids; /// Track particle IDs that exist at the segment start.
    std::vector<Mesh> segment_meshes; /// Each segment is stored as a Mesh snapshot
    
    int tail_size;
    int x, y, w, h;
    const char *title, *keys;
    
    bool newly_opened;
#ifdef MFEM_USE_MPI
    MPI_Comm comm;
#endif // MFEM_USE_MPI

    void AddSegmentStart();

    void SetSegmentEnd();

public:

    ParticleTrajectories(const ParticleSet &particles, int tail_size_, const char *vishost, int visport, const char *title_, int x_=0, int y_=0, int w_=400, int h_=400, const char *keys_=nullptr);

    void Visualize();

};

} // namespace common
} // namespace mfem


#endif // MFEM_PARTICLES_EXTRAS