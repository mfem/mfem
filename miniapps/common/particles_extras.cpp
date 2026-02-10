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

void Add3DPoint(const Vector &center, Mesh &m, real_t scale)
{
   real_t s = 0.5*scale;
   Vector v[8];

   for (int i = 0; i < 8; i++)
   {
      v[i].SetSize(3);
      v[i] = 0.0;
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

   Vector v_s_z(3);
   v_s_z = 0.0;
   v_s_z[2] = s;
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

void Add2DPoint(const Vector &center, Mesh &m, real_t scale)
{
   real_t s = 0.5*scale;
   Vector v[4];

   for (int i = 0; i < 4; i++)
   {
      v[i].SetSize(2);
      v[i] = 0.0;
   }

   for (int i = 0; i < center.Size(); i++)
   {
      v[0][i] = center[i];
   }

   Vector v_s(2); v_s = s;
   v[0] -= v_s;

   v[1] = v[0];
   v[1][0] += 2*s;

   v[2] = v[1];
   v[2][1] += 2*s;

   v[3] = v[2];
   v[3][0] -= 2*s;

   for (int i = 0; i < 4; i++)
   {
      m.AddVertex(v[i]);
   }

   int vi[4];
   for (int i = 0; i < 4; i++)
   {
      vi[i] = i + (m.GetNE())*4;
   }
   m.AddQuad(vi);
}


void VisualizeParticles(socketstream &sock, const char* vishost, int visport,
                        const ParticleSet &pset, const Vector &scalar_field,
                        real_t psize,
                        const char* title, int x, int y, int w, int h, const char* keys)
{
   const int dim = pset.GetDim();
   MFEM_VERIFY(dim == 2 || dim == 3,
               "ParticleSet dimension must be 2 or 3 for visualization.");
   const int nv = dim == 2 ? 4 : 8;

   L2_FECollection l2fec(1,dim);
   Mesh particles_mesh(dim, pset.GetNParticles()*nv, pset.GetNParticles(),
                       0, dim);

   for (int i = 0; i < pset.GetNParticles(); i++)
   {
      Vector pcoords;
      pset.Coords().GetValues(i, pcoords);
      if (dim == 2)
      {
         Add2DPoint(pcoords, particles_mesh, psize);
      }
      else
      {
         Add3DPoint(pcoords, particles_mesh, psize);
      }
   }
   particles_mesh.FinalizeMesh();

   FiniteElementSpace fes(&particles_mesh, &l2fec, 1);
   GridFunction gf(&fes);

   for (int i = 0; i < pset.GetNParticles(); i++)
   {
      for (int j = 0; j < nv; j++)
      {
         gf[j+i*nv] = scalar_field[i];
      }
   }

#ifdef MFEM_USE_MPI
   VisualizeField(sock, vishost, visport, gf, pset.GetComm(), title,
                  x, y, w, h, keys, false);
#else
   VisualizeField(sock, vishost, visport, gf, title, x, y, w, h, keys, false);
#endif // MFEM_USE_MPI
}


ParticleTrajectories::ParticleTrajectories(const ParticleSet &particles,
                                           int tail_size_, const char *vishost_,
                                           int visport_, const char *title_,
                                           int x_, int y_, int w_, int h_,
                                           const char *keys_)
   : pset(particles), tail_size(tail_size_),
     x(x_), y(y_), w(w_), h(h_), title(title_), keys(keys_),
     vishost(vishost_), visport(visport_)
#ifdef MFEM_USE_MPI
   ,comm(particles.GetComm())
#endif // MFEM_USE_MPI
{
   AddSegmentStart();
}

void ParticleTrajectories::AddSegmentStart()
{
   // Create a new mesh for all particle segments for this timestep
   segment_meshes.emplace_front(1, pset.GetNParticles()*2,
                                pset.GetNParticles(),
                                0, pset.GetDim());

   // Add segment start particle IDs
   segment_ids.emplace_front(pset.GetIDs());

   if (tail_size > 0 && static_cast<int>(segment_meshes.size()) > tail_size)
   {
      segment_meshes.pop_back();
      segment_ids.pop_back();
   }

   // Add all particle starting vertices
   for (int i = 0; i < pset.GetNParticles(); i++)
   {
      Vector pcoords;
      pset.Coords().GetValues(i, pcoords);
      segment_meshes.front().AddVertex(pcoords);
   }
}

void ParticleTrajectories::SetSegmentEnd()
{
   const Array<ParticleSet::IDType> &end_ids = pset.GetIDs();

   // Add all endpoint vertices + segments for all particles that were in
   // SetSegmentStart
   int num_start = segment_ids.front().Size();
   for (int i = 0; i < num_start; i++)
   {
      // If this particle's initial position was set in AddSegmentStart,
      // set the vertex to its now current location
      int pidx = end_ids.Find(segment_ids.front()[i]);
      if (pidx != -1)
      {
         Vector pcoords;
         pset.Coords().GetValues(pidx, pcoords);
         segment_meshes.front().AddVertex(pcoords);
      }
      else // Otherwise set its end vertex == start vertex
      {
         segment_meshes.front().AddVertex(segment_meshes.front().GetVertex(i));
      }
      segment_meshes.front().AddSegment(i, i+num_start);
   }
   segment_meshes.front().FinalizeMesh();
}


void ParticleTrajectories::Visualize()
{
   SetSegmentEnd();

   // Create a mesh of all the trajectory segments
   std::vector<Mesh*> all_meshes;
   for (Mesh &m : segment_meshes)
   {
      all_meshes.push_back(&m);
   }
   if (mesh)
   {
      all_meshes.push_back(mesh);
   }
   if (mesh_bb)
   {
      all_meshes.push_back(mesh_bb);
   }

   Mesh trajectories(all_meshes.data(), all_meshes.size());
   bool vis = trajectories.GetNE() > 0;
#ifdef MFEM_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &vis, 1, MFEM_MPI_CXX_BOOL,
                 MPI_LOR, pset.GetComm());
#endif // MFEM_USE_MPI
   if (!vis) // if all rank have 0 elements, skip visualization
   {
      AddSegmentStart();
      return;
   }


#ifdef MFEM_USE_MPI
   VisualizeMesh(sock, vishost, visport, trajectories, comm,
                 title, x, y, w, h, keys);
#else
   VisualizeMesh(sock, vishost, visport, trajectories,
                 title, x, y, w, h, keys);
#endif

   AddSegmentStart();
}

void ParticleTrajectories::SetVisualizationBoundingBox(const Vector &xmin,
                                                       const Vector &xmax)
{
   MFEM_VERIFY(xmin.Size() == pset.GetDim() &&
               xmax.Size() == pset.GetDim(),
               "Bounding box dimension must match ParticleSet dimension.");

   // Create a box mesh for visualization
   if (mesh_bb)
   {
      delete mesh_bb;
      mesh_bb = nullptr;
   }

   if (pset.GetDim() == 2)
   {
      int dim = 2;
      int nvert = 4;
      int nelem = 4;
      mesh_bb = new Mesh(1, nvert, nelem, 0, dim);
      Vector v0(dim), v1(dim), v2(dim), v3(dim);
      v0 = xmin;
      v1 = xmax;
      v2[0] = xmax[0]; v2[1] = xmin[1];
      v3[0] = xmin[0]; v3[1] = xmax[1];

      mesh_bb->AddVertex(v0);
      mesh_bb->AddVertex(v1);
      mesh_bb->AddVertex(v2);
      mesh_bb->AddVertex(v3);

      int vi[2] = {0,1};
      mesh_bb->AddSegment(vi);
      vi[0] = 1; vi[1] = 2;
      mesh_bb->AddSegment(vi);
      vi[0] = 2; vi[1] = 3;
      mesh_bb->AddSegment(vi);
      vi[0] = 3; vi[1] = 0;
      mesh_bb->AddSegment(vi);
      mesh_bb->FinalizeMesh();
   }
   else // dim == 3
   {
      int dim = 3;
      int nvert = 8;
      int nelem = 12;
      mesh_bb = new Mesh(1, nvert, nelem, 0, dim);
      Vector v(dim);

      // Vertices
      v[0] = xmin[0]; v[1] = xmin[1]; v[2] = xmin[2];
      mesh_bb->AddVertex(v); // 0: 000
      v[0] = xmax[0]; v[1] = xmin[1]; v[2] = xmin[2];
      mesh_bb->AddVertex(v); // 1: 100
      v[0] = xmax[0]; v[1] = xmax[1]; v[2] = xmin[2];
      mesh_bb->AddVertex(v); // 2: 110
      v[0] = xmin[0]; v[1] = xmax[1]; v[2] = xmin[2];
      mesh_bb->AddVertex(v); // 3: 010

      v[0] = xmin[0]; v[1] = xmin[1]; v[2] = xmax[2];
      mesh_bb->AddVertex(v); // 4: 001
      v[0] = xmax[0]; v[1] = xmin[1]; v[2] = xmax[2];
      mesh_bb->AddVertex(v); // 5: 101
      v[0] = xmax[0]; v[1] = xmax[1]; v[2] = xmax[2];
      mesh_bb->AddVertex(v); // 6: 111
      v[0] = xmin[0]; v[1] = xmax[1]; v[2] = xmax[2];
      mesh_bb->AddVertex(v); // 7: 011

      // Segments
      int vi[2];
      // Bottom face
      vi[0] = 0; vi[1] = 1; mesh_bb->AddSegment(vi);
      vi[0] = 1; vi[1] = 2; mesh_bb->AddSegment(vi);
      vi[0] = 2; vi[1] = 3; mesh_bb->AddSegment(vi);
      vi[0] = 3; vi[1] = 0; mesh_bb->AddSegment(vi);

      // Top face
      vi[0] = 4; vi[1] = 5; mesh_bb->AddSegment(vi);
      vi[0] = 5; vi[1] = 6; mesh_bb->AddSegment(vi);
      vi[0] = 6; vi[1] = 7; mesh_bb->AddSegment(vi);
      vi[0] = 7; vi[1] = 4; mesh_bb->AddSegment(vi);

      // Vertical edges
      vi[0] = 0; vi[1] = 4; mesh_bb->AddSegment(vi);
      vi[0] = 1; vi[1] = 5; mesh_bb->AddSegment(vi);
      vi[0] = 2; vi[1] = 6; mesh_bb->AddSegment(vi);
      vi[0] = 3; vi[1] = 7; mesh_bb->AddSegment(vi);

      mesh_bb->FinalizeMesh();
   }
}

} // namespace common
} // namespace mfem
