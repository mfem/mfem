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

#ifndef MFEM_CONTACT_MESH_GENERATOR_HPP
#define MFEM_CONTACT_MESH_GENERATOR_HPP

#include "mfem.hpp"
#include <cmath>

namespace mfem
{

enum ContactProblemId { twoblock = 0, ironing, beamsphere };
/// Construct a generated first-order problem mesh (problems 0–2).
/// Cell counts apply to beam/sphere only; -1 selects the default resolution.
Mesh *MakeContactProblemMesh(int id, bool hexahedra,
                             int beam_cross_cells = -1,
                             int beam_axial_cells = -1,
                             int sphere_angular_cells = -1,
                             int sphere_radial_cells = -1);

enum ContactExampleId { onion = 3, foldedstrip, ring, snapfit, twistedcylinder, twistedstrips };
Mesh *MakeOnionMesh();
Mesh *MakeFoldedStripMesh();
Mesh *MakeRingMesh();
Mesh *MakeSnapFitMesh(bool hexahedra);
Mesh *MakeTwistedCylinderMesh(real_t length, real_t radius, real_t bow,
                              int cross_section_cells, int length_cells,
                              Element::Type type);
Mesh *MakeTwistingStripsMesh(real_t length, real_t width, real_t thickness,
                             real_t separation, real_t initial_angle,
                             bool wrap_core, real_t radius, real_t gap,
                             Element::Type type, int axial_cells);
namespace folded_strip
{
constexpr real_t length = 6.0;
constexpr real_t width = 0.9;
constexpr real_t thickness = 0.3;
constexpr real_t radius = 0.72;
constexpr real_t gap = 2 * radius - thickness;
constexpr int support = 1;
constexpr int load = 2;
constexpr int lower_contact = 3;
constexpr int upper_contact = 4;
constexpr int free_surface = 5;
}

namespace twist
{
constexpr int left_end = 1;
constexpr int right_end = 2;

void PredictMotion(const ParGridFunction &reference,
                   ParGridFunction &displacement, real_t angle_increment,
                   real_t shortening, real_t previous_shortening, real_t length);
}

namespace twisted_strips
{
constexpr int length_cells = 48;
constexpr int thickness_cells = 2;
constexpr int width_cells = 6;
constexpr int stacked_width_cells = 4;
constexpr int first_surface = 3;
constexpr int second_surface = 4;
constexpr int core_ends = 5;
constexpr int free_surface = 6;
}

namespace cylinder_twist
{
constexpr int bands = 6;
constexpr int cells_per_band = 6;
constexpr int length_cells = bands * cells_per_band;
constexpr int cross_section_cells = 6;
constexpr int free_surface = 3;
constexpr int first_contact_band = 4;
}

namespace snap_fit
{
constexpr real_t inner_radius = 1.0;
constexpr real_t thickness = 0.2;
constexpr real_t width = 0.6;
constexpr real_t pin_radius = inner_radius;
constexpr real_t offset = 1.9;
constexpr int angular_cells = 64;
constexpr int mouth_cells = 16;
constexpr real_t tip_angle = 35.0 * M_PI / 180.0;
constexpr int clip_body = 1;
constexpr int pin_body = 2;
constexpr int grip = 1;
constexpr int clip_contact = 2;
constexpr int free_surface = 3;
constexpr int pin_ends = 4;
constexpr int pin_contact = 5;
constexpr int clip_lower_load = 8;
constexpr int clip_upper_load = 9;
constexpr int pin_support = 10;

inline real_t Angle(real_t index)
{
   if (index <= angular_cells)
   {
      return tip_angle + (2*M_PI - 2*tip_angle)*index/angular_cells;
   }
   return 2*M_PI - tip_angle + 2*tip_angle*(index - angular_cells)/mouth_cells;
}

void GetBodyDofs(ParFiniteElementSpace &fes, int attr, Array<int> &dofs);
void TranslateClip(ParGridFunction &displacement, const Array<int> &dofs,
                   real_t distance);
void PrintState(const ParGridFunction &displacement,
                const Array<int> &clip_dofs,
                const Array<int> &pin_dofs);
}

}

#endif
