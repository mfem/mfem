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

// Additional contact examples: onion, strip, ring, clip, cylinder, and strips.
// Build: make contact_examples. See docs/EXAMPLES.md for geometry, commands, and limitations.
#include "contact_driver.hpp"
#include "mesh_generator.hpp"
#include <cmath>
#include <iostream>
using namespace std;
using namespace mfem;

namespace
{
class ExampleProblem : public ContactDriverProblem
{
   int id = twistedstrips;
   real_t strip_pressure = 0.06, ring_pressure = 2.5;
   real_t clip_travel = snap_fit::offset, clip_pressure = 40,
          pin_young_modulus = 500;
   bool hex_mesh = false;
   real_t twist_angle = 180, twist_shortening = 0.2, twist_length = 3,
          twist_radius = 0.5;
   real_t strip_width = 0.6, strip_thickness = 0.02;
   real_t strip_young_modulus = 6000, strip_poisson_ratio = 0.3;
   int strip_length_cells = -1;
   real_t strip_separation = -1, wrap_angle = 360, core_gap = 0.02;
   bool wrap_core = false;
   bool strips_case = false, cylinder_case = false, snap_fit_case = false;
   const char *contact_name = "";
   Array<int> clip_dofs, pin_dofs;
   real_t previous_clip_travel = 0, previous_twist_fraction = 0;
public:
   explicit ExampleProblem(ContactDriverOptions &options) : ContactDriverProblem(
         options) { }
   void AddOptions(OptionsParser &args) override
   {
      args.AddOption(&id, "-prob", "--problem-number",
                     "Example: 3 onion, 4 folded strip, 5 ring, 6 elastic C-clip/pin, "
                     "7 twisting cylinder, 8 twisting elastic strips (default).");
      args.AddOption(&strip_pressure, "-u-pressure", "--strip-pressure",
                     "Maximum downward pressure on the folded strip's upper arm.");
      args.AddOption(&ring_pressure, "-ring-pressure", "--ring-pressure",
                     "Maximum inward reference pressure on the ring's upper outer arc.");
      args.AddOption(&clip_travel, "-clip-travel", "--clip-travel",
                     "Maximum insertion displacement of the snap-fit clip (default 1.9).");
      args.AddOption(&clip_pressure, "-clip-pressure", "--clip-pressure",
                     "Maximum clip closing pressure after half insertion (default 40).");
      args.AddOption(&pin_young_modulus, "-pin-young-modulus", "--pin-young-modulus",
                     "Young's modulus of the elastic snap-fit pin (default 500).");
      args.AddOption(&hex_mesh, "-hex", "--hexahedra", "-tet",
                     "--tetrahedra",
                     "Use hexahedra or tetrahedra for both contact bodies "
                     "(problems 6/8; default: tetrahedra).");
      args.AddOption(&twist_angle, "-twist-angle", "--twist-angle",
                     "Total relative end twist in degrees (360 for two strips, 180 otherwise; signed). "
                     "The ends rotate by minus/plus half this angle.");
      args.AddOption(&twist_shortening, "-twist-shortening", "--twist-shortening",
                     "Total axial shortening (0 for two strips, 0.2 otherwise). "
                     "Each end moves inward by half this distance.");
      args.AddOption(&twist_length, "-twist-length", "--twist-length",
                     "Reference body length (problems 7/8; default 3).");
      args.AddOption(&twist_radius, "-twist-radius", "--twist-radius",
                     "Reference cylinder radius (default 0.5).");
      args.AddOption(&strip_width, "-strip-width", "--strip-width",
                     "Width of each strip in problem 8 (default 0.6).");
      args.AddOption(&strip_thickness, "-strip-thickness", "--strip-thickness",
                     "Thickness of each strip in problem 8 (default 0.02).");
      args.AddOption(&strip_young_modulus, "-strip-young-modulus",
                     "--strip-young-modulus",
                     "Elastic strip Young's modulus in the simulation's units (default 6000).");
      args.AddOption(&strip_poisson_ratio, "-strip-poisson-ratio",
                     "--strip-poisson-ratio",
                     "Elastic strip Poisson ratio (default 0.3).");
      args.AddOption(&strip_length_cells, "-strip-nx", "--strip-length-cells",
                     "Axial strip cells (default 48).");
      args.AddOption(&strip_separation, "-strip-separation", "--strip-separation",
                     "Reference strip centerline separation (default: strip thickness, touching).");
      args.AddOption(&wrap_angle, "-wrap-angle", "--wrap-angle",
                     "Initial stress-free wrap in degrees (0 for two strips, 360 for a core).");
      args.AddOption(&wrap_core, "-wrap-core", "--wrap-core", "-two-strips",
                     "--two-strips", "Wrap a strip around a stationary cylinder "
                     "or twist two initially stacked touching strips (default: two strips).");
      args.AddOption(&core_gap, "-core-gap", "--core-gap",
                     "Initial radial strip-to-core clearance (default 0.02).");
   }
   bool ResolveOptions(int argc, char *argv[]) override
   {
      if (id < 3 || id > 8)
      {
         if (Mpi::Root()) { mfem::err << "contact_examples supports 3..8. Use contact for 0, 1, 2.\n"; }
         return false;
      }
      strips_case = id == twistedstrips; cylinder_case = id == twistedcylinder;
      snap_fit_case = id == snapfit;
      const bool twisting = cylinder_case || strips_case;
      auto supplied = [&](std::initializer_list<const char *> names)
      { return ContactOptionSupplied(argc, argv, names); };
      if (opts.sref == -1) { opts.sref = strips_case ? 0 : 1; }
      if (opts.nsteps == -1) { opts.nsteps = twisting ? 72 : snap_fit_case ? 32 : id == onion ? 1 : 20; }
      if (opts.msteps == -1) { opts.msteps = snap_fit_case ? 32 : 0; }
      if (opts.tribol_ratio == -1) { opts.tribol_ratio = UpdatesGeometry() ? 2 : 8; }
      opts.nonlinear = UpdatesGeometry();
      if (strip_length_cells == -1) { strip_length_cells = 48; }
      if (!supplied({"-strip-separation", "--strip-separation"})) { strip_separation = strip_thickness; }
      if (strips_case && !wrap_core)
      {
         if (!supplied({"-wrap-angle", "--wrap-angle"})) { wrap_angle = 0; }
         if (!supplied({"-twist-angle", "--twist-angle"})) { twist_angle = 360; }
         if (!supplied({"-twist-shortening", "--twist-shortening"})) { twist_shortening = 0; }
      }
      contact_name = strips_case ? (wrap_core ? "Elastic strip on core" : wrap_angle
                                    == 0 ?
                                    "Stacked elastic strips" : "Wrapped elastic strips") :
                     cylinder_case ? "Cylinder" : snap_fit_case ? "Snap-fit" : id == ring ? "Ring" :
                     id == foldedstrip ? "Folded strip" : "Onion";
      if (twisting && (!IsFinite(twist_angle) || !IsFinite(twist_length) ||
                       twist_length <= 0 ||
                       !IsFinite(twist_shortening) || twist_shortening < 0 ||
                       twist_shortening >= twist_length ||
                       !IsFinite(twist_radius) || twist_radius <= 0))
      {
         if (Mpi::Root()) { mfem::err << "Require finite angle, positive length/radius and 0 <= shortening < length.\n"; }
         return false;
      }
      if (strips_case && (!IsFinite(strip_width) || strip_width <= 0 ||
                          !IsFinite(strip_thickness) || strip_thickness <= 0 ||
                          !IsFinite(strip_young_modulus) || strip_young_modulus <= 0 ||
                          !IsFinite(strip_poisson_ratio) || strip_poisson_ratio <= -1 ||
                          strip_poisson_ratio >= 0.5 ||
                          strip_length_cells < 1 || !IsFinite(wrap_angle) || !IsFinite(core_gap) ||
                          core_gap < 0 ||
                          (!wrap_core && (!IsFinite(strip_separation) ||
                                          strip_separation < strip_thickness))))
      {
         if (Mpi::Root()) { mfem::err << "Require valid strip dimensions, elastic constants, wrap, clearance and axial cells.\n"; }
         return false;
      }
      if (snap_fit_case && (!IsFinite(clip_travel) || clip_travel < 0 ||
                            !IsFinite(clip_pressure) || clip_pressure < 0 ||
                            !IsFinite(pin_young_modulus) || pin_young_modulus <= 0))
      {
         if (Mpi::Root()) { mfem::err << "Require nonnegative clip travel/pressure and positive pin modulus.\n"; }
         return false;
      }
      if ((id == foldedstrip && (!IsFinite(strip_pressure) || strip_pressure < 0)) ||
          (id == ring && (!IsFinite(ring_pressure) || ring_pressure < 0))) { return false; }
      return true;
   }
   std::string Name() const override { return contact_name; }
   bool UpdatesGeometry() const override { return id != onion; }
   real_t MaxCorrection() const override
   {
      return strips_case ? strip_thickness/3 : cylinder_case ? twist_radius/5 :
             snap_fit_case ? snap_fit::thickness/3 : id == ring ? 0.04 : 0.1;
   }
   real_t MaxDamping() const override { return snap_fit_case || cylinder_case ? 0.5 : 1; }
   std::unique_ptr<Mesh> MakeMesh() const override
   {
      switch (id)
      {
         case onion: return std::unique_ptr<Mesh>(MakeOnionMesh());
         case foldedstrip: return std::unique_ptr<Mesh>(MakeFoldedStripMesh());
         case ring: return std::unique_ptr<Mesh>(MakeRingMesh());
         case snapfit: return std::unique_ptr<Mesh>(MakeSnapFitMesh(hex_mesh));
         case twistedcylinder:
            return std::unique_ptr<Mesh>(MakeTwistedCylinderMesh(twist_length, twist_radius,
                                                                 0.05,
                                                                 cylinder_twist::cross_section_cells, cylinder_twist::length_cells,
                                                                 Element::HEXAHEDRON));
         case twistedstrips:
            return std::unique_ptr<Mesh>(MakeTwistingStripsMesh(twist_length, strip_width,
                                                                strip_thickness,
                                                                strip_separation, wrap_angle, wrap_core, twist_radius, core_gap,
                                                                hex_mesh ? Element::HEXAHEDRON : Element::TETRAHEDRON, strip_length_cells));
      }
      MFEM_ABORT("Unknown example");
      return nullptr;
   }
   void Configure(ParMesh &, Vector &E, Vector &nu,
                  Array<int> &ess_bdr_attr, Array<int> &ess_bdr_attr_comp) const override
   {
      switch (id)
      {
         case onion:
            // symmetry planes x = 0, y = 0, z = 0
            ess_bdr_attr.Append(21); ess_bdr_attr_comp.Append(0);
            ess_bdr_attr.Append(22); ess_bdr_attr_comp.Append(1);
            ess_bdr_attr.Append(23); ess_bdr_attr_comp.Append(2);
            E = 1.e3;
            nu = 0.33;
            break;
         case foldedstrip:
            // Clamp a short outer bend patch, leaving both arm tips free.
            ess_bdr_attr.Append(folded_strip::support);
            ess_bdr_attr_comp.Append(-1);
            E = 1.e3;
            nu = 0.3;
            break;
         case ring:
            // Clamp a short lower outer arc, leaving the inner arc free to contact.
            ess_bdr_attr.Append(1); ess_bdr_attr_comp.Append(-1);
            E = 1.e3;
            nu = 0.3;
            break;
         case snapfit:
            ess_bdr_attr.Append(snap_fit::grip); ess_bdr_attr_comp.Append(-1);
            // Plane-strain clip motion keeps the extruded contact faces planar.
            ess_bdr_attr.Append(snap_fit::clip_contact); ess_bdr_attr_comp.Append(1);
            ess_bdr_attr.Append(snap_fit::free_surface); ess_bdr_attr_comp.Append(1);
            ess_bdr_attr.Append(snap_fit::clip_lower_load); ess_bdr_attr_comp.Append(1);
            ess_bdr_attr.Append(snap_fit::clip_upper_load); ess_bdr_attr_comp.Append(1);
            // Pin faces remain in plane strain. Small central end-face patches
            // anchor the axis while the outer pin deforms under the closing clip.
            ess_bdr_attr.Append(snap_fit::pin_ends); ess_bdr_attr_comp.Append(1);
            ess_bdr_attr.Append(snap_fit::pin_contact); ess_bdr_attr_comp.Append(1);
            ess_bdr_attr.Append(snap_fit::pin_support); ess_bdr_attr_comp.Append(-1);
            E = 1.e3;
            E[snap_fit::pin_body - 1] = pin_young_modulus;
            nu = 0.3;
            break;
         case twistedcylinder:
         case twistedstrips:
            ess_bdr_attr.Append(twist::left_end);
            ess_bdr_attr_comp.Append(-1);
            ess_bdr_attr.Append(twist::right_end);
            ess_bdr_attr_comp.Append(-1);
            E = 1.e3;
            nu = 0.3;
            if (strips_case)
            {
               E[0] = strip_young_modulus;
               nu[0] = strip_poisson_ratio;
               if (!wrap_core)
               {
                  E = strip_young_modulus;
                  nu = strip_poisson_ratio;
               }
            }
            if (strips_case && wrap_core)
            {
               ess_bdr_attr.Append(twisted_strips::second_surface);
               ess_bdr_attr_comp.Append(-1);
               ess_bdr_attr.Append(twisted_strips::core_ends);
               ess_bdr_attr_comp.Append(-1);
            }
            break;
         default:
            MFEM_ABORT("Should be unreachable");
            break;
      }
   }
   std::vector<ContactInterface> Interfaces() const override
   {
      std::vector<ContactInterface> contact_interfaces;
      switch (id)
      {
         case onion:
            // interfaces between consecutive shells: outer surface of shell k
            // (mortar, attr 2k) and inner surface of shell k+1 (non-mortar,
            // attr 2k+1)
            for (int a = 2; a <= 16; a += 2)
            {
               contact_interfaces.push_back({{a}, {a+1}});
            }
            break;
         case foldedstrip:
            // Disjoint facing patches of ONE connected body. The bend is excluded
            // so neighboring faces can never become spurious self-contact pairs.
            contact_interfaces.push_back({{folded_strip::lower_contact},
               {folded_strip::upper_contact}});
            break;
         case ring:
            // Opposite inner arcs of one ring; lateral arcs separate the patches
            // so neighboring elements are never paired against themselves.
            contact_interfaces.push_back({{3}, {4}});
            break;
         case snapfit:
            // Put the pressure field on the fixed, smooth non-mortar surface.
            // The clip's lips slide across it without rotating the pressure basis.
            contact_interfaces.push_back({{
                  snap_fit::clip_contact,
                  snap_fit::clip_lower_load, snap_fit::clip_upper_load
               },
               {snap_fit::pin_contact}});
            break;
         case twistedcylinder:
            // Pair separated longitudinal bands. Within-band and adjacent-band
            // faces are excluded, including all topological neighbors.
            for (int band = 0; band < cylinder_twist::bands - 2; band++)
            {
               for (int other = band + 2; other < cylinder_twist::bands; other++)
               {
                  contact_interfaces.push_back(
                  {
                     {cylinder_twist::first_contact_band + band},
                     {cylinder_twist::first_contact_band + other}});
               }
            }
            break;
         case twistedstrips:
            contact_interfaces.push_back({{twisted_strips::first_surface},
               {twisted_strips::second_surface}});
            break;
         default:
            MFEM_ABORT("Should be unreachable");
            break;
      }
      return contact_interfaces;
   }
   std::string OutputName() const override
   {
      switch (id)
      {
         case onion: return "nested_shells_hex";
         case foldedstrip: return "folded_strip_hex";
         case ring: return "ring_hex";
         case snapfit: return hex_mesh ? "clip_pin_hex" : "clip_pin_tet";
         case twistedcylinder: return "cylinder_hex";
         case twistedstrips:
            return std::string(wrap_core ? "strips_core" : "strips") +
                   (hex_mesh ? "_hex" : "_tet");
      }
      MFEM_ABORT("Unknown example");
      return "";
   }
   void PrintParameters() const override
   {
      if (strips_case)
      {
         cout << "Elastic strip material: Neo-Hookean, E = " << strip_young_modulus
              << ", nu = " << strip_poisson_ratio
              << ", thickness = " << strip_thickness
              << ", membrane E*h = " << strip_young_modulus*strip_thickness
              << ", plate bending scale = "
              << strip_young_modulus*std::pow(strip_thickness,3)/
              (12*(1-strip_poisson_ratio*strip_poisson_ratio))
              << ", displacement order = 1" << endl;
      }
   }
   void Initialize(ElasticityOperator &prob) override
   {
      auto &fes = *prob.GetFESpace();
      if (snap_fit_case)
      {
         snap_fit::GetBodyDofs(fes, snap_fit::clip_body, clip_dofs);
         snap_fit::GetBodyDofs(fes, snap_fit::pin_body, pin_dofs);
      }
      if (strips_case && wrap_core)
      {
         snap_fit::GetBodyDofs(fes, 2, pin_dofs);
         fes.DofsToVDofs(pin_dofs);
         Array<int> core_tdofs;
         for (int dof : pin_dofs)
         {
            const int true_dof = fes.GetLocalTDofNumber(dof);
            if (true_dof >= 0) { core_tdofs.Append(true_dof); }
         }
         prob.AddEssentialTrueDofs(core_tdofs);
      }

   }
   bool InitialClearance(real_t &clearance) const override
   {
      clearance = strip_separation - strip_thickness;
      return strips_case && !wrap_core && wrap_angle == 0;
   }
   void PrintState(const ParGridFunction &displacement) const override
   { if (snap_fit_case) { snap_fit::PrintState(displacement, clip_dofs, pin_dofs); } }
   void ApplyLoad(int i, ElasticityOperator &prob, ParGridFunction &x_gf,
                  const ParGridFunction &reference) override
   {
      const int dim = prob.GetMesh()->Dimension();
      Vector ess_values(dim); ess_values = 0.0;
      Array<int> ess_bdr(prob.GetMesh()->bdr_attributes.Max()); ess_bdr = 0;
      const int myid = Mpi::WorldRank();
      const real_t p = 30;
      ConstantCoefficient f(p);
      switch (id)
      {
         case onion:
            // equal pressure on the innermost (attr 1) and outermost
            // (attr 18) surfaces
            f.constant = -p*(i+1)/opts.nsteps;
            if (ess_bdr.Size())
            {
               ess_bdr = 0; ess_bdr[0] = 1;
            }
            prob.SetNeumannPressureData(f,ess_bdr);
            if (ess_bdr.Size())
            {
               ess_bdr = 0; ess_bdr[17] = 1;
            }
            prob.SetNeumannPressureData(f,ess_bdr);
            break;
         case foldedstrip:
         case ring:
         {
            const real_t load_fraction = i < opts.nsteps ? real_t(i+1)/opts.nsteps
                                         : 1.0 - real_t(i+1-opts.nsteps)/opts.msteps;
            f.constant = -(id == ring ? ring_pressure : strip_pressure)
                         * load_fraction;
            const int load_attr = id == ring ? 6 : folded_strip::load;
            ess_bdr = 0; ess_bdr[load_attr - 1] = 1;
            prob.SetNeumannPressureData(f, ess_bdr);
            if (myid == 0 && opts.print_level >= 2)
            {
               cout << contact_name << " pressure = " << -f.constant << endl;
            }
            break;
         }
         case snapfit:
         {
            const real_t fraction = i < opts.nsteps ? real_t(i+1)/opts.nsteps
                                    : 1.0 - real_t(i+1-opts.nsteps)/opts.msteps;
            const real_t travel = clip_travel * fraction;
            const real_t closing_fraction = std::min(real_t(1.0),
                                                     std::max(real_t(0.0), (travel - snap_fit::offset/2) / (snap_fit::offset/2)));
            const real_t closing_pressure =
               clip_pressure * closing_fraction;
            ess_bdr = 0; ess_bdr[snap_fit::grip - 1] = 1;
            ess_values[0] = travel;
            prob.SetDisplacementDirichletData(ess_values, ess_bdr);
            // Predict a rigid translation before linearizing the material and
            // contact. Updating only the grip would stretch/invert the rear
            // elements instead of carrying forward the previous clip shape.
            snap_fit::TranslateClip(x_gf, clip_dofs, travel - previous_clip_travel);
            ess_bdr = 0;
            ess_bdr[snap_fit::clip_lower_load - 1] = 1;
            ess_bdr[snap_fit::clip_upper_load - 1] = 1;
            f.constant = -closing_pressure;
            prob.SetNeumannPressureData(f, ess_bdr);
            previous_clip_travel = travel;
            if (myid == 0 && opts.print_level >= 2)
            {
               cout << "Snap-fit clip travel = " << travel
                    << ", closing pressure = " << closing_pressure << endl;
            }
            break;
         }
         case twistedcylinder:
         case twistedstrips:
         {
            const real_t fraction = i < opts.nsteps ? real_t(i+1)/opts.nsteps
                                    : 1.0 - real_t(i+1-opts.nsteps)/opts.msteps;
            const real_t angle = twist_angle * (M_PI/180.0) * fraction;
            const real_t shortening = twist_shortening * fraction;
            auto end_displacement = [=](const Vector &point, Vector &delta)
            {
               const real_t position = point[0] / twist_length - 0.5;
               const real_t end_angle = angle * position;
               const real_t cosine = std::cos(end_angle), sine = std::sin(end_angle);
               delta.SetSize(3);
               delta[0] = -shortening * position;
               delta[1] = (cosine - 1.0)*point[1] - sine*point[2];
               delta[2] = sine*point[1] + (cosine - 1.0)*point[2];
            };
            VectorFunctionCoefficient twist_bc(dim, end_displacement);
            ess_bdr = 0;
            ess_bdr[twist::left_end - 1] = 1;
            ess_bdr[twist::right_end - 1] = 1;
            prob.SetDisplacementDirichletData(twist_bc, ess_bdr);
            // Carry the previous shape through a distributed incremental
            // rotation before forming the tangent at the new grip position.
            twist::PredictMotion(
               reference, x_gf,
               twist_angle * (M_PI/180.0) * (fraction - previous_twist_fraction),
               shortening, twist_shortening * previous_twist_fraction,
               twist_length);
            if (strips_case && wrap_core) { x_gf.SetSubVector(pin_dofs, 0.0); }
            previous_twist_fraction = fraction;
            if (myid == 0 && opts.print_level >= 2)
            {
               const real_t end_degrees = twist_angle * fraction / 2;
               cout << contact_name << " end rotations = " << -end_degrees << " and "
                    << end_degrees << " degrees, axial shortening = "
                    << shortening << endl;
            }
            break;
         }
         default:
            MFEM_ABORT("Should be unreachable");
            break;
      }

   }
};
}

int main(int argc, char *argv[])
{
   ContactDriverOptions options;
   ExampleProblem problem(options);
   return RunContactDriver(argc, argv, options, problem);
}
