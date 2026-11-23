// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mesh_generator.hpp"
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

namespace mfem
{
namespace
{
// Transform geometry nodes and refresh vertices used for patch classification.
void TransformContactMesh(Mesh &mesh,
                          std::function<void(const Vector &, Vector &)> map)
{
   mesh.SetCurvature(1);
   mesh.Transform(map);
   // Mesh::Transform updates its nodal grid function but not the vertex array.
   Vector values;
   for (int component = 0; component < mesh.SpaceDimension(); component++)
   {
      mesh.GetNodes()->GetNodalValues(values, component + 1);
      for (int vertex = 0; vertex < mesh.GetNV(); vertex++)
      {
         mesh.GetVertex(vertex)[component] = values[vertex];
      }
   }
}
}

namespace
{

// An octant shell built from the three outer faces of a cube. Shared cube
// edges use shared vertices; separate shells deliberately have disjoint DOFs.
std::unique_ptr<Mesh> MakeBeamSphereShell(int angular_cells,
                                          int radial_cells, bool inner_shell)
{
   const int n = angular_cells, side = n + 1;
   auto flat = [=](int i, int j, int k) { return (i*side + j)*side + k; };
   std::vector<int> surface_index(side*side*side, -1);
   std::vector<std::array<int, 3>> directions;
   for (int i = 0; i <= n; i++)
   {
      for (int j = 0; j <= n; j++)
      {
         for (int k = 0; k <= n; k++)
         {
            if (i != n && j != n && k != n) { continue; }
            surface_index[flat(i,j,k)] = directions.size();
            directions.push_back({i,j,k});
         }
      }
   }
   // The inner insert matches two intersecting spherical arcs in the input
   // mesh: R=.8 centered at zero, and R=.85 centered at z=-.075. Its inner
   // surface has R=.7 about that shifted center. Solve along a ray so radial
   // lines stay continuous across octant cube-face seams and symmetry planes.
   auto point = [=](real_t i, real_t j, real_t k, real_t layer, Vector &x)
   {
      const real_t norm = std::sqrt(i*i + j*j + k*k);
      const real_t dz = k/norm;
      auto shifted_radius = [=](real_t radius)
      {
         return -0.075*dz + std::sqrt(radius*radius - 0.075*0.075*(1-dz*dz));
      };
      const real_t inner = inner_shell ? shifted_radius(0.7) : 0.8;
      const real_t outer = inner_shell ? std::min(real_t(0.8),
                                                  shifted_radius(0.85)) : 1.0;
      const real_t radius = inner + (outer-inner)*layer/radial_cells;
      // Rotate the positive octant into x<=0, y>=0, z>=0; determinant +1.
      x[0] = -radius*j/norm;
      x[1] = radius*i/norm;
      x[2] = radius*k/norm;
   };
   auto mesh = std::make_unique<Mesh>(3, 0, 0);
   std::vector<std::vector<int>> vertex(radial_cells + 1,
                                        std::vector<int>(directions.size()));
   Vector x(3);
   for (int layer = 0; layer <= radial_cells; layer++)
   {
      for (int d = 0; d < static_cast<int>(directions.size()); d++)
      {
         const auto &p = directions[d];
         point(p[0], p[1], p[2], layer, x);
         vertex[layer][d] = mesh->AddVertex(x.GetData());
      }
   }
   auto face_point = [=](int face, int u, int v)
   {
      if (face == 0) { return std::array<int,3> {n,u,v}; }
      if (face == 1) { return std::array<int,3> {u,n,v}; }
      return std::array<int,3> {u,v,n};
   };
   for (int face = 0; face < 3; face++)
   {
      for (int u = 0; u < n; u++)
      {
         for (int v = 0; v < n; v++)
         {
            std::array<std::array<int,3>,4> p = {{
                  face_point(face,u,v),
                  face_point(face,u+1,v), face_point(face,u+1,v+1), face_point(face,u,v+1)
               }
            };
            if (face == 1) { std::swap(p[1], p[3]); }
            for (int layer = 0; layer < radial_cells; layer++)
            {
               int lower[4], upper[4];
               for (int c = 0; c < 4; c++)
               {
                  const int d = surface_index[flat(p[c][0],p[c][1],p[c][2])];
                  lower[c] = vertex[layer][d]; upper[c] = vertex[layer+1][d];
               }
               mesh->AddHex(lower[0], lower[1], lower[2], lower[3],
                            upper[0], upper[1], upper[2], upper[3], inner_shell ? 3 : 2);
               if (layer == 0)
               {
                  mesh->AddBdrQuad(lower[3],lower[2],lower[1],lower[0], inner_shell ? 10 : 8);
               }
               if (layer == radial_cells-1)
               {
                  // Only the part of the outer shell facing the beam is a
                  // contact patch; the remainder is a free outer surface.
                  const real_t i = face == 0 ? n : u+0.5;
                  const real_t j = face == 1 ? n : face == 0 ? u+0.5 : v+0.5;
                  const real_t k = face == 2 ? n : v+0.5;
                  const int attr = inner_shell ? 9 :
                                   j/std::sqrt(i*i+j*j+k*k) <= 0.6 ? 7 : 12;
                  mesh->AddBdrQuad(upper[0],upper[1],upper[2],upper[3], attr);
               }
               for (int edge = 0; edge < 4; edge++)
               {
                  const int next = (edge+1)%4;
                  for (int axis = 0; axis < 3; axis++)
                  {
                     if (p[edge][axis] == 0 && p[next][axis] == 0)
                     {
                        const int attr = axis == 0 ? 2 : axis == 1 ? 1 : 4;
                        mesh->AddBdrQuad(lower[edge],lower[next],upper[next],upper[edge],attr);
                     }
                  }
               }
            }
         }
      }
   }
   mesh->FinalizeHexMesh(1, 1);
   mesh->SetCurvature(1);
   return mesh;
}

}


Mesh *MakeContactProblemMesh(int prob_name, bool hexahedra,
                             int beam_cross_cells, int beam_axial_cells,
                             int sphere_angular_cells, int sphere_radial_cells)
{
   MFEM_VERIFY(prob_name >= twoblock && prob_name <= beamsphere,
               "Unknown contact problem: " << prob_name);
   std::unique_ptr<Mesh> combined_mesh;
   if (prob_name == beamsphere)
   {
      const int nc = beam_cross_cells < 0 ? 8 : beam_cross_cells;
      const int ny = beam_axial_cells < 0 ? 24 : beam_axial_cells;
      const int ns = sphere_angular_cells < 0 ? 6 : sphere_angular_cells;
      const int nr = sphere_radial_cells < 0 ? 2 : sphere_radial_cells;
      const int inner_nr = sphere_radial_cells < 0 ? 1 : sphere_radial_cells;
      MFEM_VERIFY(nc > 0 && ny > 0 && ns > 0 && nr > 0,
                  "Beam/sphere mesh needs positive cell counts.");
      auto beam = std::make_unique<Mesh>(Mesh::MakeCartesian3D(
                                            nc, ny, nc, Element::HEXAHEDRON, 1, 2.5, 1));
      for (int b = 0; b < beam->GetNBE(); b++)
      {
         const Element &boundary = *beam->GetBdrElement(b);
         int attr;
         switch (boundary.GetAttribute())
         {
            case 1: attr = 3; break; // top pressure, z=2
            case 2: attr = 2; break; // y=0 symmetry
            case 4: attr = 5; break; // fixed beam end, y=2.5
            case 5: attr = 1; break; // x=0 symmetry
            default:
            {
               real_t y = 0;
               for (int v = 0; v < boundary.GetNVertices(); v++)
               { y += beam->GetVertex(boundary.GetVertices()[v])[1]; }
               attr = y/boundary.GetNVertices() < 1.25 ? 6 : 11;
            }
         }
         beam->SetBdrAttribute(b,attr);
      }
      beam->SetAttributes();
      TransformContactMesh(*beam, [](const Vector &logical, Vector &physical)
      {
         const real_t s = logical[0], t = logical[2];
         physical.SetSize(3);
         physical[0] = -0.7*s*std::sqrt(1-t*t/2);
         physical[1] = logical[1];
         physical[2] = 2-t*std::sqrt(1-s*s/2);
      });
      auto sphere = MakeBeamSphereShell(ns,nr,false);
      auto insert = MakeBeamSphereShell(ns,inner_nr,true);
      Mesh *parts[] = {beam.get(),sphere.get(),insert.get()};
      combined_mesh = std::make_unique<Mesh>(parts,3);
   }
   else
   {
      constexpr real_t scale = 9.0/3.6;
      constexpr real_t lx0 = 3.6*scale, ly0 = 1.6*scale, lz0 = 1.0*scale;
      constexpr int nx0 = 24, ny0 = 12, nz0 = 6;
      Mesh mesh0 = Mesh::MakeCartesian3D(nx0, ny0, nz0, Element::HEXAHEDRON,
                                         lx0, ly0, lz0);

      mesh0.SetCurvature(1);

      // adjust boundary attributes
      for (int i = 0; i<mesh0.GetNBE(); i++)
      {
         int battr = mesh0.GetBdrElement(i)->GetAttribute();
         int new_battr;
         switch (battr)
         {
            case 1: new_battr = 2; break;
            case 6: new_battr = 3; break;
            default: new_battr = 1; break;
         }
         mesh0.SetBdrAttribute(i, new_battr);
      }
      mesh0.SetAttributes();
      if (prob_name == twoblock)
      {
         constexpr real_t lx = 1.0, ly = 1.0, lz = 1.0;
         const int nx = 5, ny = nx, nz = nx;
         Mesh mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, lx, ly, lz);

         mesh.SetCurvature(1);

         // adjust element attributes
         for (int i = 0; i<mesh.GetNE(); i++)  { mesh.GetElement(i)->SetAttribute(2); }
         mesh.SetAttributes();

         // adjust boundary attributes
         for (int i = 0; i<mesh.GetNBE(); i++)
         {
            int battr = mesh.GetBdrElement(i)->GetAttribute();
            int new_battr;
            switch (battr)
            {
               case 1: new_battr = 4; break;
               case 6: new_battr = 6; break;
               default: new_battr = 5; break;
            }
            mesh.SetBdrAttribute(i, new_battr);
         }
         mesh.SetAttributes();

         Vector shift(3);
         shift(0) = (ly0-ly)/3;
         shift(1) = (ly0-ly)/2;
         shift(2) = lz0;

         auto shift_map = [&](const Vector &x, Vector &y)
         {
            y.SetSize(3);
            y = x;
            y+=shift;
         };

         TransformContactMesh(mesh, shift_map);

         Mesh *mesh_array2[] = {&mesh, &mesh0};
         combined_mesh = std::make_unique<Mesh>(mesh_array2, 2);


      }
      else if (prob_name == ironing)
      {
         constexpr real_t lx = 1.6*scale, ly = 2.0*scale, lz = 0.2*scale;
         const int nx = 16, ny = 20,
                   nz = 2;
         Mesh mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, lx, ly, lz);

         mesh.SetCurvature(1);

         // adjust element attributes
         for (int i = 0; i<mesh.GetNE(); i++) { mesh.GetElement(i)->SetAttribute(2); }
         mesh.SetAttributes();

         // adjust boundary attributes
         for (int i = 0; i<mesh.GetNBE(); i++)
         {
            int battr = mesh.GetBdrElement(i)->GetAttribute();
            int new_battr;
            switch (battr)
            {
               case 1: new_battr = 6; break;
               case 6: new_battr = 4; break;
               default: new_battr = 5; break;
            }
            mesh.SetBdrAttribute(i, new_battr);
         }
         mesh.SetAttributes();

         constexpr real_t theta_arc = M_PI/2;
         constexpr real_t rin = lx / theta_arc - lz;
         constexpr real_t rout = lx / theta_arc + lz;

         constexpr real_t xshift = lx0/4;
         constexpr real_t yshift = 0.5*(ly0-ly);
         constexpr real_t zshift = rout + lz0;

         auto bend_map = [&](const Vector &x, Vector &y)
         {
            const real_t theta = -((x(0) / lx + 0.5) * theta_arc);
            const real_t r = rin + (x(2) + lz);
            y.SetSize(3);
            y(0) = xshift + r * std::cos(theta);
            y(1) = yshift + x(1);
            y(2) = zshift + r * std::sin(theta);
         };

         TransformContactMesh(mesh, bend_map);
         Mesh *mesh_array2[] = {&mesh, &mesh0};
         combined_mesh = std::make_unique<Mesh>(mesh_array2, 2);
      }

   }

   if (hexahedra) { return combined_mesh.release(); }
   // Preserve geometry and boundary attributes while splitting hexes.
   // Triangular contact faces avoid Tribol's warped-quad inverse-map failures.
   return new Mesh(Mesh::MakeSimplicial(*combined_mesh));
}

Mesh * MakeTwistingStripsMesh(real_t length, real_t width, real_t thickness,
                              real_t separation, real_t initial_angle,
                              bool wrap_core, real_t core_radius, real_t core_gap,
                              Element::Type element_type, int strip_length_cells)
{
   std::array<std::unique_ptr<Mesh>, 2> bodies;
   const real_t reference_angle = initial_angle * (M_PI/180.0);
   const int length_cells = strip_length_cells;
   const int width_cells = wrap_core ? twisted_strips::width_cells :
                           twisted_strips::stacked_width_cells;
   for (int body = 0; body < 2; body++)
   {
      if (wrap_core && body == 1)
      {
         bodies[body].reset(MakeTwistedCylinderMesh(length, core_radius, 0.0,
                                                    12, strip_length_cells,
                                                    element_type));
         Mesh &core = *bodies[body];
         for (int i = 0; i < core.GetNE(); i++)
         {
            core.GetElement(i)->SetAttribute(2);
         }
         for (int i = 0; i < core.GetNBE(); i++)
         {
            const int attr = core.GetBdrElement(i)->GetAttribute();
            core.SetBdrAttribute(i, attr == twist::left_end || attr == twist::right_end ?
                                 twisted_strips::core_ends : twisted_strips::second_surface);
         }
         core.SetAttributes();
         continue;
      }
      bodies[body] = std::make_unique<Mesh>(Mesh::MakeCartesian3D(
                                               length_cells,
                                               twisted_strips::thickness_cells,
                                               width_cells,
                                               element_type,
                                               real_t(length_cells),
                                               thickness, width));
      Mesh &mesh = *bodies[body];
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         mesh.GetElement(i)->SetAttribute(body + 1);
      }
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         const int original_attr = mesh.GetBdrElement(i)->GetAttribute();
         // Only the facing broad surfaces form the strip-to-strip interface.
         // Including the opposite backs creates false negative mortar gaps
         // through the solid thickness, even for flat touching strips.
         const bool facing = body == 0 ? original_attr == 4 : original_attr == 2;
         const int attr = original_attr == 5 ? twist::left_end :
                          original_attr == 3 ? twist::right_end :
                          wrap_core || facing ? twisted_strips::first_surface + body :
                          twisted_strips::free_surface;
         mesh.SetBdrAttribute(i, attr);
      }
      mesh.SetAttributes();
      const real_t offset = wrap_core ? -(core_radius + thickness/2 + core_gap) :
                            (body == 0 ? -0.5 : 0.5) * separation;
      auto ribbon_map = [=](const Vector &logical, Vector &physical)
      {
         const real_t fraction = logical[0] / length_cells;
         const real_t angle = reference_angle * (fraction - 0.5);
         const real_t cosine = std::cos(angle), sine = std::sin(angle);
         // In two-strip mode width lies along y and thickness along z:
         // body 1 is below body 2, with independent coincident faces at z=0.
         // The sign on width preserves the Cartesian mesh orientation.
         const real_t y = wrap_core ? offset + logical[1] - thickness/2 :
                          width/2 - logical[2];
         const real_t z = wrap_core ? logical[2] - width/2 :
                          offset + logical[1] - thickness/2;
         physical.SetSize(3);
         physical[0] = length * fraction;
         physical[1] = cosine*y - sine*z;
         physical[2] = sine*y + cosine*z;
      };
      TransformContactMesh(mesh, ribbon_map);
   }
   Mesh *parts[] = {bodies[0].get(), bodies[1].get()};
   return new Mesh(parts, 2);
}

// A solid cylinder with disjoint contact bands and a small stress-free bow to
// select a buckling direction when the ends move toward each other.
Mesh * MakeTwistedCylinderMesh(real_t length, real_t radius, real_t initial_bow,
                               int cross_section_cells, int length_cells,
                               Element::Type element_type)
{
   auto mesh = std::make_unique<Mesh>(Mesh::MakeCartesian3D(
                                         length_cells, cross_section_cells,
                                         cross_section_cells,
                                         element_type,
                                         real_t(length_cells),
                                         2.0, 2.0));
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const Element &face = *mesh->GetBdrElement(i);
      int attr = cylinder_twist::free_surface;
      if (face.GetAttribute() == 5) { attr = twist::left_end; }
      else if (face.GetAttribute() == 3) { attr = twist::right_end; }
      else
      {
         real_t path = 0.0;
         for (int j = 0; j < face.GetNVertices(); j++)
         {
            path += mesh->GetVertex(face.GetVertices()[j])[0];
         }
         const int cell = static_cast<int>(path / face.GetNVertices());
         const int band_cell = cell % cylinder_twist::cells_per_band;
         // Leave a cell at both ends of each band outside contact. Refined
         // children inherit these markers, preserving the separation.
         if (band_cell > 0 && band_cell < cylinder_twist::cells_per_band - 1)
         {
            attr = cylinder_twist::first_contact_band +
                   cell / cylinder_twist::cells_per_band;
         }
      }
      mesh->SetBdrAttribute(i, attr);
   }
   mesh->SetAttributes();
   auto cylinder_map = [=](const Vector &logical, Vector &physical)
   {
      const real_t fraction = logical[0] / length_cells;
      const real_t bow = std::sin(M_PI * fraction);
      const real_t y = logical[1] - 1.0, z = logical[2] - 1.0;
      physical.SetSize(3);
      physical[0] = length * fraction;
      // Map the square cross-section into a disk with a filled hex core.
      physical[1] = radius * y * std::sqrt(1.0 - z*z/2);
      physical[2] = radius * z * std::sqrt(1.0 - y*y/2) +
                    initial_bow * bow * bow;
   };
   TransformContactMesh(*mesh, cylinder_map);
   return mesh.release();
}

void twist::PredictMotion(const ParGridFunction &reference,
                          ParGridFunction &displacement,
                          real_t angle_increment, real_t shortening,
                          real_t previous_shortening, real_t length)
{
   const auto &fes = *displacement.ParFESpace();
   const real_t *reference_values = reference.HostRead();
   real_t *values = displacement.HostReadWrite();
   const real_t axial_scale = (length - shortening) / (length -
                                                       previous_shortening);
   for (int dof = 0; dof < fes.GetNDofs(); dof++)
   {
      const int x_dof = fes.DofToVDof(dof, 0);
      const int y_dof = fes.DofToVDof(dof, 1);
      const int z_dof = fes.DofToVDof(dof, 2);
      const real_t position = reference_values[x_dof] / length - 0.5;
      const real_t angle = angle_increment * position;
      const real_t cosine = std::cos(angle), sine = std::sin(angle);
      const real_t x = reference_values[x_dof] + values[x_dof];
      const real_t y = reference_values[y_dof] + values[y_dof];
      const real_t z = reference_values[z_dof] + values[z_dof];
      values[x_dof] = length/2 + axial_scale*(x - length/2) - reference_values[x_dof];
      values[y_dof] = cosine*y - sine*z - reference_values[y_dof];
      values[z_dof] = sine*y + cosine*z - reference_values[z_dof];
   }
}

// Map one Cartesian solid along the lower arm, around the bend, and along the
// upper arm. Shared vertices at both transitions keep the body connected.
Mesh * MakeFoldedStripMesh()
{
   constexpr int arm_cells = 24, bend_cells = 24;
   constexpr int width_cells = 4, thickness_cells = 4;
   constexpr int path_cells = 2 * arm_cells + bend_cells;
   auto mesh = std::make_unique<Mesh>(Mesh::MakeCartesian3D(
                                         path_cells, width_cells, thickness_cells,
                                         Element::HEXAHEDRON, real_t(path_cells),
                                         folded_strip::width,
                                         folded_strip::thickness));

   // Mark patches before mapping. The inner bend remains a free surface,
   // keeping the two contact patches disjoint.
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const Element *boundary = mesh->GetBdrElement(i);
      const int *vertices = boundary->GetVertices();
      real_t path = 0.0;
      for (int j = 0; j < boundary->GetNVertices(); j++)
      {
         path += mesh->GetVertex(vertices[j])[0];
      }
      path /= boundary->GetNVertices();
      int attr = folded_strip::free_surface;
      if (boundary->GetAttribute() == 1)
      {
         if (path < arm_cells) { attr = folded_strip::lower_contact; }
         if (path > arm_cells + bend_cells)
         {
            attr = folded_strip::upper_contact;
         }
      }
      if (boundary->GetAttribute() == 6)
      {
         if (path > arm_cells + bend_cells) { attr = folded_strip::load; }
         if (std::abs(path - (arm_cells + bend_cells / 2)) < 2.0)
         {
            attr = folded_strip::support;
         }
      }
      mesh->SetBdrAttribute(i, attr);
   }
   mesh->SetAttributes();

   auto bend_map = [](const Vector &logical, Vector &physical)
   {
      const real_t path = logical[0];
      const real_t radius = folded_strip::radius + logical[2] -
                            folded_strip::thickness / 2;
      physical.SetSize(3);
      physical[1] = logical[1] - folded_strip::width / 2;
      if (path <= arm_cells)
      {
         physical[0] = folded_strip::length * (1.0 - path / arm_cells);
         physical[2] = -radius;
      }
      else if (path <= arm_cells + bend_cells)
      {
         const real_t theta = -M_PI / 2 -
                              M_PI * (path - arm_cells) / bend_cells;
         physical[0] = radius * std::cos(theta);
         physical[2] = radius * std::sin(theta);
      }
      else
      {
         physical[0] = folded_strip::length *
                       (path - arm_cells - bend_cells) / arm_cells;
         physical[2] = radius;
      }
   };
   TransformContactMesh(*mesh, bend_map);
   return mesh.release();
}

// Circular elastic ring extruded across its width. Station indices wrap around
// to share the seam vertices, producing one closed connected body. Clockwise
// stations, increasing width and increasing radius give positive hex orientation.
Mesh * MakeRingMesh()
{
   constexpr int angular_cells = 64, width_cells = 2, radial_cells = 2;
   constexpr real_t radius = 1.0, thickness = 0.12, width = 0.6;
   auto mesh = std::make_unique<Mesh>(3, 0, 0);
   const auto vertex = [](int i, int j, int k)
   {
      return ((i % angular_cells) * (width_cells + 1) + j)
             * (radial_cells + 1) + k;
   };
   for (int i = 0; i < angular_cells; i++)
   {
      const real_t theta = 2.0 * M_PI * i / angular_cells;
      for (int j = 0; j <= width_cells; j++)
      {
         for (int k = 0; k <= radial_cells; k++)
         {
            const real_t r = radius + thickness * (real_t(k)/radial_cells - 0.5);
            mesh->AddVertex(r * std::cos(theta),
                            width * (real_t(j)/width_cells - 0.5),
                            -r * std::sin(theta));
         }
      }
   }
   for (int i = 0; i < angular_cells; i++)
   {
      for (int j = 0; j < width_cells; j++)
      {
         for (int k = 0; k < radial_cells; k++)
         {
            const int v[8] = {vertex(i,j,k), vertex(i+1,j,k),
                              vertex(i+1,j+1,k), vertex(i,j+1,k),
                              vertex(i,j,k+1), vertex(i+1,j,k+1),
                              vertex(i+1,j+1,k+1), vertex(i,j+1,k+1)
                             };
            mesh->AddHex(v, 1);
            if (k == 0)
            {
               const int attr = i >= 8 && i < 24 ? 3 :
                                i >= 40 && i < 56 ? 4 : 2;
               mesh->AddBdrQuad(v[3], v[2], v[1], v[0], attr);
            }
            if (k == radial_cells - 1)
            {
               const int attr = i >= 14 && i < 18 ? 1 :
                                i >= 44 && i < 52 ? 6 : 2;
               mesh->AddBdrQuad(v[4], v[5], v[6], v[7], attr);
            }
            if (j == 0) { mesh->AddBdrQuad(v[0], v[1], v[5], v[4], 2); }
            if (j == width_cells - 1)
            {
               mesh->AddBdrQuad(v[3], v[7], v[6], v[2], 2);
            }
         }
      }
   }
   mesh->FinalizeHexMesh(1, 1);
   return mesh.release();
}

// Open annular clip and a separate elastic pin. Their vertices are deliberately
// disjoint: Tribol couples the supported pin to the moving clip.
Mesh * MakeSnapFitMesh(bool hexahedra)
{
   constexpr int angular_cells = snap_fit::angular_cells;
   constexpr int width_cells = 1, radial_cells = 4;
   constexpr real_t tip_angle = snap_fit::tip_angle;
   const Element::Type element_type = hexahedra ? Element::HEXAHEDRON :
                                      Element::TETRAHEDRON;
   auto clip = std::make_unique<Mesh>(Mesh::MakeCartesian3D(
                                         angular_cells, width_cells, radial_cells,
                                         element_type, real_t(angular_cells),
                                         snap_fit::width, snap_fit::thickness));
   for (int i = 0; i < clip->GetNBE(); i++)
   {
      const Element &face = *clip->GetBdrElement(i);
      real_t path = 0.0;
      for (int j = 0; j < face.GetNVertices(); j++)
      {
         path += clip->GetVertex(face.GetVertices()[j])[0];
      }
      path /= face.GetNVertices();
      const real_t theta = snap_fit::Angle(path);
      int attr = snap_fit::free_surface;
      if (face.GetAttribute() == 1)
      {
         attr = snap_fit::clip_contact;
      }
      if (face.GetAttribute() == 3 || face.GetAttribute() == 5)
      {
         attr = snap_fit::clip_contact;
      }
      if (face.GetAttribute() == 6)
      {
         // The outer lip patches also touch the pin during initial entry.
         if (theta < M_PI/3 || theta > 5*M_PI/3)
         {
            attr = snap_fit::clip_contact;
         }
         if (std::abs(path - angular_cells/2) < 2.0)
         {
            attr = snap_fit::grip;
         }
         if (std::abs(theta - M_PI/2) < M_PI/9)
         {
            attr = snap_fit::clip_lower_load;
         }
         if (std::abs(theta - 3*M_PI/2) < M_PI/9)
         {
            attr = snap_fit::clip_upper_load;
         }
      }
      clip->SetBdrAttribute(i, attr);
   }
   clip->SetAttributes();
   auto clip_map = [](const Vector &logical, Vector &physical)
   {
      const real_t theta = snap_fit::Angle(logical[0]);
      const real_t r = snap_fit::inner_radius + logical[2];
      // Round each lip with a semicircular end cap. Blend the cap into the
      // neighboring stations so contact normals turn smoothly around the nose.
      const real_t half = snap_fit::thickness/2;
      const real_t radial_offset = logical[2] - half;
      const real_t blend = std::max(real_t(0.0), real_t(1.0) -
                                    std::min(theta - tip_angle,
                                             2*M_PI - tip_angle - theta)/0.3);
      const real_t nose = std::sqrt(std::max(real_t(0.0),
                                             half*half - radial_offset*radial_offset))*blend;
      const real_t tangent = theta < M_PI ? -nose : nose;
      physical.SetSize(3);
      physical[0] = -snap_fit::offset + r*std::cos(theta) - tangent*std::sin(theta);
      physical[1] = logical[1] - snap_fit::width/2;
      physical[2] = -r*std::sin(theta) - tangent*std::cos(theta);
   };
   TransformContactMesh(*clip, clip_map);

   // Tetrahedra give triangular contact faces with Tribol's direct inverse map.
   // Hexahedral contact faces use its quadrilateral Newton iteration.
   // A square-to-disk map gives a solid pin without a singular axial cell.
   // Match the pin's boundary stations to the clip's inner arc. Equal radii
   // then give coincident facets when seated, including on the coarse mesh.
   constexpr int pin_cells = (snap_fit::angular_cells + snap_fit::mouth_cells)/4;
   auto pin = std::make_unique<Mesh>(Mesh::MakeCartesian3D(
                                        pin_cells, width_cells, pin_cells,
                                        element_type, 2.0,
                                        snap_fit::width, 2.0));
   for (int i = 0; i < pin->GetNE(); i++)
   {
      pin->GetElement(i)->SetAttribute(snap_fit::pin_body);
   }
   for (int i = 0; i < pin->GetNBE(); i++)
   {
      const int attr = pin->GetBdrElement(i)->GetAttribute();
      pin->SetBdrAttribute(i, attr == 2 || attr == 4 ?
                           snap_fit::pin_ends : snap_fit::pin_contact);
   }
   pin->SetAttributes();
   auto pin_map = [](const Vector &logical, Vector &physical)
   {
      const real_t x = logical[0] - 1.0, z = logical[2] - 1.0;
      physical.SetSize(3);
      physical[1] = logical[1] - snap_fit::width/2;
      const real_t square_radius = std::max(std::abs(x), std::abs(z));
      if (square_radius == 0.0)
      {
         physical[0] = physical[2] = 0.0;
         return;
      }
      const real_t u = x/square_radius, v = z/square_radius;
      real_t station;
      if (u >= std::abs(v))
      {
         station = v <= 0.0 ? -pin_cells*v/2 : 4*pin_cells - pin_cells*v/2;
      }
      else if (-v >= std::abs(u)) { station = pin_cells - pin_cells*u/2; }
      else if (-u >= std::abs(v)) { station = 2*pin_cells + pin_cells*v/2; }
      else { station = 3*pin_cells + pin_cells*u/2; }
      // Boundary grid vertices have integer station indices. Snap roundoff
      // away so both bodies use exactly the same angle and radius there.
      if (std::abs(station - std::round(station)) < 1e-12)
      { station = std::round(station); }
      const real_t theta = snap_fit::Angle(station);
      const real_t r = snap_fit::pin_radius * (square_radius == 1.0 ? 1.0 :
                                               std::sqrt(x*x + z*z - x*x*z*z));
      physical[0] = r*std::cos(theta);
      physical[2] = -r*std::sin(theta);
   };
   TransformContactMesh(*pin, pin_map);
   for (int i = 0; i < pin->GetNBE(); i++)
   {
      const Element &face = *pin->GetBdrElement(i);
      if (face.GetAttribute() != snap_fit::pin_ends) { continue; }
      real_t x = 0.0, z = 0.0;
      for (int j = 0; j < face.GetNVertices(); j++)
      {
         const real_t *vertex = pin->GetVertex(face.GetVertices()[j]);
         x += vertex[0];
         z += vertex[2];
      }
      x /= face.GetNVertices();
      z /= face.GetNVertices();
      if (x*x + z*z < 0.25*0.25*snap_fit::pin_radius*snap_fit::pin_radius)
      {
         pin->SetBdrAttribute(i, snap_fit::pin_support);
      }
   }
   pin->SetAttributes();
   Mesh *parts[] = {clip.get(), pin.get()};
   return new Mesh(parts, 2);
}

void snap_fit::GetBodyDofs(ParFiniteElementSpace &fes, int attr,
                           Array<int> &dofs)
{
   Array<int> marker(fes.GetNDofs()), element_dofs;
   marker = 0;
   for (int i = 0; i < fes.GetNE(); i++)
   {
      if (fes.GetParMesh()->GetAttribute(i) != attr) { continue; }
      fes.GetElementDofs(i, element_dofs);
      for (int j : element_dofs) { marker[j] = 1; }
   }
   dofs.SetSize(0);
   for (int i = 0; i < marker.Size(); i++)
   {
      if (marker[i]) { dofs.Append(i); }
   }
}

void snap_fit::TranslateClip(ParGridFunction &displacement,
                             const Array<int> &dofs, real_t distance)
{
   const auto &fes = *displacement.ParFESpace();
   real_t *values = displacement.HostReadWrite();
   for (int dof : dofs) { values[fes.DofToVDof(dof, 0)] += distance; }
}

void snap_fit::PrintState(const ParGridFunction &displacement,
                          const Array<int> &clip_dofs, const Array<int> &pin_dofs)
{
   const auto &fes = *displacement.ParFESpace();
   const real_t *values = displacement.HostRead();
   real_t local[2] = {0.0, 0.0}, global[2];
   for (int dof : clip_dofs)
   {
      local[0] = std::max(local[0], std::abs(values[fes.DofToVDof(dof, 2)]));
   }
   for (int dof : pin_dofs)
   {
      for (int c = 0; c < 3; c++)
      {
         local[1] = std::max(local[1], std::abs(values[fes.DofToVDof(dof, c)]));
      }
   }
   MPI_Allreduce(local, global, 2, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 fes.GetComm());
   if (Mpi::Root())
   {
      cout << "Snap-fit: maximum transverse deflection = " << global[0]
           << ", maximum pin displacement = " << global[1] << endl;
   }
}

// Generate nine disconnected spherical octant shells. Distinct surface meshes
// at each shared radius are intentional: Tribol owns the eight interfaces.
Mesh * MakeOnionMesh()
{
   // Keep the element count close to the former mesh while giving neighboring
   // shells different surface resolutions for nonmatching mortar interfaces.
   constexpr int angular_cells[9] = {16, 17, 17, 17, 18, 18, 19, 19, 19};
   Mesh *mesh = new Mesh(3, 0, 0);

   for (int shell = 0; shell < 9; ++shell)
   {
      const int n = angular_cells[shell];
      const int radial_cells = shell < 4 ? 2 : 1;
      const int side = n + 1;
      const auto flat = [side](int i, int j, int k)
      {
         return (i * side + j) * side + k;
      };

      // Three square faces of the positive octant of a cube map to a sphere.
      // The shared edges use one vertex, while different shells do not share
      // vertices, including where their radii coincide.
      std::vector<int> surface_index(side * side * side, -1);
      std::vector<std::array<int, 3>> directions;
      for (int i = 0; i <= n; ++i)
      {
         for (int j = 0; j <= n; ++j)
         {
            for (int k = 0; k <= n; ++k)
            {
               if (i != n && j != n && k != n) { continue; }
               surface_index[flat(i, j, k)] = directions.size();
               directions.push_back({i, j, k});
            }
         }
      }

      std::vector<std::vector<int>> vertex(radial_cells + 1,
                                           std::vector<int>(directions.size()));
      for (int layer = 0; layer <= radial_cells; ++layer)
      {
         const real_t radius = 5.5 + 0.5 * shell +
                               0.5 * layer / radial_cells;
         for (int d = 0; d < static_cast<int>(directions.size()); ++d)
         {
            const auto &p = directions[d];
            const real_t scale = radius / std::sqrt(real_t(p[0] * p[0] +
                                                           p[1] * p[1] +
                                                           p[2] * p[2]));
            vertex[layer][d] = mesh->AddVertex(scale * p[0], scale * p[1],
                                               scale * p[2]);
         }
      }

      const auto face_point = [n](int face, int u, int v)
      {
         if (face == 0) { return std::array<int, 3> {n, u, v}; }
         if (face == 1) { return std::array<int, 3> {u, n, v}; }
         return std::array<int, 3> {u, v, n};
      };
      for (int face = 0; face < 3; ++face)
      {
         for (int u = 0; u < n; ++u)
         {
            for (int v = 0; v < n; ++v)
            {
               std::array<std::array<int, 3>, 4> p = {{
                     face_point(face, u, v), face_point(face, u + 1, v),
                     face_point(face, u + 1, v + 1), face_point(face, u, v + 1)
                  }
               };
               if (face == 1) { std::swap(p[1], p[3]); }

               for (int layer = 0; layer < radial_cells; ++layer)
               {
                  int inner[4], outer[4];
                  for (int c = 0; c < 4; ++c)
                  {
                     const int id = surface_index[flat(p[c][0], p[c][1], p[c][2])];
                     inner[c] = vertex[layer][id];
                     outer[c] = vertex[layer + 1][id];
                  }
                  mesh->AddHex(inner[0], inner[1], inner[2], inner[3],
                               outer[0], outer[1], outer[2], outer[3], shell + 1);
                  if (layer == 0)
                  {
                     mesh->AddBdrQuad(inner[3], inner[2], inner[1], inner[0],
                                      2 * shell + 1);
                  }
                  if (layer == radial_cells - 1)
                  {
                     mesh->AddBdrQuad(outer[0], outer[1], outer[2], outer[3],
                                      2 * shell + 2);
                  }
                  for (int edge = 0; edge < 4; ++edge)
                  {
                     const int next = (edge + 1) % 4;
                     for (int axis = 0; axis < 3; ++axis)
                     {
                        if (p[edge][axis] == 0 && p[next][axis] == 0)
                        {
                           mesh->AddBdrQuad(inner[edge], inner[next],
                                            outer[next], outer[edge], 21 + axis);
                        }
                     }
                  }
               }
            }
         }
      }
   }

   mesh->FinalizeHexMesh(1, 1);
   return mesh;
}

}
