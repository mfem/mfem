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

#include "particles_extras.hpp"


namespace mfem
{
namespace common
{

void InitializeRandom(Particle &p, int seed, const Vector &pos_min, const Vector &pos_max)
{
   std::mt19937 gen(seed);
   std::uniform_real_distribution<> real_dist(0.0,1.0);

   for (int i = 0; i < p.GetSpaceDim(); i++)
   {
      p.GetCoords()[i] = pos_min[i] + (pos_max[i] - pos_min[i])*real_dist(gen);
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


void Add3DPoint(const Vector &center, Mesh &m, real_t s)
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


template<Ordering::Type VOrdering>
void VisualizeParticles(socketstream &sock, const char* vishost, int visport,
                                    const ParticleSet<VOrdering> &pset, const Vector &scalar_field, real_t psize, 
                                    const char* title, int x, int y, int w, int h, const char* keys)
{                
   L2_FECollection l2fec(1,3);
   Mesh particles_mesh(3, pset.GetNP()*8, pset.GetNP(), 0, 3);

   for (int i = 0; i < pset.GetNP(); i++)
   {
      Particle p = pset.GetParticleData(i);
      const Vector &pcoords = p.GetCoords();
      Add3DPoint(pcoords, particles_mesh, psize);
   }
   particles_mesh.FinalizeMesh();

   FiniteElementSpace fes(&particles_mesh, &l2fec, 1);
   GridFunction gf(&fes);

   for (int i = 0; i < pset.GetNP(); i++)
   {
      for (int j = 0; j < 8; j++)
      {
         gf[j+i*8] = scalar_field[i];
      }
   }

#ifdef MFEM_USE_MPI

   int myid, num_procs;
   MPI_Comm_rank(pset.GetComm(), &myid);
   MPI_Comm_size(pset.GetComm(), &num_procs);

   bool newly_opened = false;

   if (!sock.is_open() || !sock)
   {
      sock.open(vishost, visport);
      sock.precision(8);
      newly_opened = true;
   }

   sock << "parallel " << num_procs << " " << myid << "\n";
   sock << "solution\n" << particles_mesh << gf << std::flush;

   if (myid == 0)
   {
      if (newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n";
         if ( keys ) { sock << "keys " << keys << "\n"; }
         else { sock << "keys maaAc\n"; }
         sock << std::endl;
      }
   }

   
#else

   VisualizeField(sock, vishost, visport, gf, title, x, y, w, h, keys, false);

#endif // MFEM_USE_MPI
}


ParticleTrajectories::ParticleTrajectories(const char* vishost, int visport, const char* title, int x, int y, int w, int h, const char* keys)
{
   sock.open(vishost, visport);
   sock.precision(8);
   // sock << "window_title '" << title << "'\n"
   //       << "window_geometry "
   //       << x << " " << y << " " << w << " " << h << "\n";
   // if ( keys ) { sock << "keys " << keys << "\n"; }
   // else { sock << "keys maaAc\n"; }
   // sock << std::endl;
}

template<Ordering::Type VOrdering>
void ParticleTrajectories::AddSegmentStart(const ParticleSet<VOrdering> &pset)
{
   MFEM_ASSERT(segment_completed, "SetSegmentEnd must be called after each AddSegmentStart.");

   // Create a new mesh for all particle segments for this timestep
   segment_meshes.emplace_back(1, pset.GetNP()*2, pset.GetNP(), 0, pset.GetSpaceDim());

   // Add segment start particle IDs
   segment_ids.emplace_back(pset.GetIDs());

   // Add all particle starting vertices
   for (int i = 0; i < pset.GetNP(); i++)
   {
      Particle p = pset.GetParticleData(i);
      segment_meshes.back().AddVertex(p.GetCoords());
   }

   segment_completed = false;
}

template<Ordering::Type VOrdering>
void ParticleTrajectories::SetSegmentEnd(const ParticleSet<VOrdering> &pset)
{
   MFEM_ASSERT(!segment_completed, "AddSegmentStart must be called prior to SetSegmentEnd.");

   const Array<unsigned int> &end_ids = pset.GetIDs();

   // Add all endpoint vertices + segments for all particles
   int num_start = segment_ids.back().Size();
   for (int i = 0; i < num_start; i++)
   {
      // If this particle's initial position was set in AddSegmentStart, set the vertex to its now current location
      int pidx = end_ids.Find(segment_ids.back()[i]);
      if (pidx != -1)
      {
         Particle p = pset.GetParticleData(pidx);
         segment_meshes.back().AddVertex(p.GetCoords());
      }
      else // Otherwise set its end vertex == start vertex
      {
         segment_meshes.back().AddVertex(segment_meshes.back().GetVertex(i));
      }
      segment_meshes.back().AddSegment(i, i+num_start);
   }
   segment_meshes.back().FinalizeMesh();

   segment_completed = true;

}


void ParticleTrajectories::Visualize()
{
   // Create a mesh of all the trajectory segments
   std::vector<Mesh*> all_meshes;
   for (Mesh &m : segment_meshes)
      all_meshes.push_back(&m);

   Mesh trajectories(all_meshes.data(), all_meshes.size());

   // Update socketstream
#ifdef MFEM_USE_MPI
   int myid, num_procs;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);

   sock << "parallel " << num_procs << " " << myid << "\n";
#endif // MFEM_USE_MPI
   sock << "mesh\n";
   trajectories.Print(sock);

}

template void VisualizeParticles<Ordering::byNODES>(socketstream &sock, const char* vishost, int visport,
                                    const ParticleSet<Ordering::byNODES> &pset, const Vector &scalar_field, real_t psize, 
                                    const char* title, int x, int y, int w, int h, const char* keys);
template void VisualizeParticles<Ordering::byVDIM>(socketstream &sock, const char* vishost, int visport,
                                    const ParticleSet<Ordering::byVDIM> &pset, const Vector &scalar_field, real_t psize, 
                                    const char* title, int x, int y, int w, int h, const char* keys);

template void ParticleTrajectories::AddSegmentStart<Ordering::byNODES>(const ParticleSet<Ordering::byNODES> &pset);
template void ParticleTrajectories::AddSegmentStart<Ordering::byVDIM>(const ParticleSet<Ordering::byVDIM> &pset);

template void ParticleTrajectories::SetSegmentEnd<Ordering::byNODES>(const ParticleSet<Ordering::byNODES> &pset);
template void ParticleTrajectories::SetSegmentEnd<Ordering::byVDIM>(const ParticleSet<Ordering::byVDIM> &pset);

} // namespace common

} // namespace mfem