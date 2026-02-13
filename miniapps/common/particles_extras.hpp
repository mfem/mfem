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
#include <list>
#ifdef MFEM_USE_MPI
#include "pfem_extras.hpp"
#else
#include "fem_extras.hpp"
#endif // MFEM_USE_MPI

namespace mfem
{
namespace common
{

/// Add a point to a given Mesh, represented as a hex sized \p scale
void Add3DPoint(const Vector &center, Mesh &m, real_t scale=2e-3);

/// Add a point to a given Mesh, represented as a quad sized \p scale
void Add2DPoint(const Vector &center, Mesh &m, real_t scale=2e-3);

/** @brief Plot particles in ParticleSet \p pset, represented as quads/hexes of
 *  size \p psize and colored by \p scalar_field .
 */
void VisualizeParticles(socketstream &sock, const char* vishost, int visport,
                        const ParticleSet &pset,
                        const Vector &scalar_field, real_t psize,
                        const char* title,
                        int x = 0, int y = 0, int w = 400, int h = 400,
                        const char* keys=nullptr);

/// Helper class for easily visualizing particle trajectories using GLVis
class ParticleTrajectories
{
protected:
   const ParticleSet &pset;
   Mesh *mesh = nullptr; // optional edge mesh to visualize along with particles
   Mesh *mesh_bb = nullptr; // optional bounding box mesh for visualization

   socketstream sock;
   /// Track particle IDs that exist at the segment start.
   std::list<Array<ParticleSet::IDType>> segment_ids;
   /// Each segment is stored as a Mesh snapshot
   std::list<Mesh> segment_meshes;

   int tail_size;
   int x, y, w, h;
   const char *title, *keys;
   const char *vishost;
   int visport;

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif // MFEM_USE_MPI

   // Mark the start of a new segment for all particles.
   void AddSegmentStart();

   // Mark the end of the current segment for all particles.
   void SetSegmentEnd();

public:
   /** @brief Setup up the particle trajectory for visualization.
    *
    * @details Visualize particle trajectory by connecting their positions at
    * each timestep with line segments. The trajectory "tail" length is
    * controlled by \p tail_size_. If tail_size_ = 0, entire particle
    * trajectory is visualized. Optionally, the mesh can also be visualized
    * along the particles by calling AddMeshForVisualization().
    *
    * Note this is a helper utility for quick visualization with GLVis and
    * not necessarily optimized for large number of particles or long tails.
    * Consider using the output from ParticleSet::PrintCSV() with ParaView for
    * more complex visualization needs.
    */
   ParticleTrajectories(const ParticleSet &particles, int tail_size_,
                        const char *vishost_, int visport_, const char *title_,
                        int x_=0, int y_=0, int w_=400, int h_=400,
                        const char *keys_=nullptr);

   /// Add a mesh to be visualized along with the particle trajectories.
   void AddMeshForVisualization(Mesh *mesh_)
   {
      MFEM_VERIFY(mesh_->Dimension() == 1,
                  "Mesh dimension must be 1 to match the particle trajectory.");
      mesh = mesh_;
   }

   /// Visualize the particle trajectories (and mesh if provided).
   void Visualize();

   /// Set the bounding box for visualization.
   void SetVisualizationBoundingBox(const Vector &xmin, const Vector &xmax);

   /// Destructor
   ~ParticleTrajectories()
   {
      delete mesh_bb;
   }
};


} // namespace common
} // namespace mfem


#endif // MFEM_PARTICLES_EXTRAS
