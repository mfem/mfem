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

// Additional contact examples: onion, folded strip, ring, clip, twisting strips,
// and a rope tightening around an elastic cylinder.
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
   real_t clip_travel = snap_fit::offset, pin_young_modulus = 500;
   bool hex_mesh = false;
   real_t twist_angle = 360, twist_shortening = 0, twist_length = 3;
   real_t strip_width = 0.6, strip_thickness = 0.02;
   real_t strip_young_modulus = 6000, strip_poisson_ratio = 0.3;
   int strip_length_cells = -1;
   real_t strip_separation = -1, wrap_angle = 0;
   real_t rope_radius = 0.08, rope_wrap_angle = 720, rope_tighten_angle = 90,
          rope_pull = 0.2, rope_gap = 0.01;
   real_t rope_young_modulus = 3000, rope_poisson_ratio = 0.3;
   real_t cylinder_length = 3, cylinder_radius = 0.5,
          cylinder_young_modulus = 1000, cylinder_poisson_ratio = 0.3;
   int rope_length_cells = 48, cylinder_axial_cells = 12, cylinder_cross_cells = 6;
   bool strips_case = false, rope_case = false, snap_fit_case = false;
   const char *contact_name = "";
   Array<int> clip_dofs, pin_dofs, rope_dofs, cylinder_surface_dofs;
   Vector rope_reference;
   real_t previous_clip_travel = 0, previous_twist_fraction = 0,
          previous_rope_fraction = 0;
public:
   explicit ExampleProblem(ContactDriverOptions &options) : ContactDriverProblem(
         options) { }
   void AddOptions(OptionsParser &args) override
   {
      args.AddOption(&id, "-prob", "--problem-number",
                     "Example: 3 onion, 4 folded strip, 5 ring, 6 elastic C-clip/pin, "
                     "7 twisting elastic strips (default), 8 rope tightening around an elastic cylinder.");
      args.AddOption(&strip_pressure, "-u-pressure", "--strip-pressure",
                     "Maximum downward pressure on the folded strip's upper arm.");
      args.AddOption(&ring_pressure, "-ring-pressure", "--ring-pressure",
                     "Maximum inward reference pressure on the ring's upper outer arc.");
      args.AddOption(&clip_travel, "-clip-travel", "--clip-travel",
                     "Maximum insertion displacement of the snap-fit clip (default 1.9).");
      args.AddOption(&pin_young_modulus, "-pin-young-modulus", "--pin-young-modulus",
                     "Young's modulus of the elastic snap-fit pin (default 500).");
      args.AddOption(&hex_mesh, "-hex", "--hexahedra", "-tet",
                     "--tetrahedra",
                     "Use hexahedra or tetrahedra for both contact bodies "
                     "(problems 6–8; default: tetrahedra).");
      args.AddOption(&twist_angle, "-twist-angle", "--twist-angle",
                     "Relative end twist for problem 7 (default 360 degrees; signed). "
                     "The ends rotate by minus/plus half this angle.");
      args.AddOption(&twist_shortening, "-twist-shortening", "--twist-shortening",
                     "Total axial shortening for problem 7 (default 0). "
                     "Each end moves inward by half this distance.");
      args.AddOption(&twist_length, "-twist-length", "--twist-length",
                     "Reference strip length in problem 7 (default 3).");
      args.AddOption(&strip_width, "-strip-width", "--strip-width",
                     "Width of each strip in problem 7 (default 0.6).");
      args.AddOption(&strip_thickness, "-strip-thickness", "--strip-thickness",
                     "Thickness of each strip in problem 7 (default 0.02).");
      args.AddOption(&strip_young_modulus, "-strip-young-modulus",
                     "--strip-young-modulus",
                     "Elastic strip Young's modulus in the simulation's units (default 6000).");
      args.AddOption(&strip_poisson_ratio, "-strip-poisson-ratio",
                     "--strip-poisson-ratio",
                     "Elastic strip Poisson ratio (default 0.3).");
      args.AddOption(&strip_length_cells, "-strip-nx", "--strip-length-cells",
                     "Axial strip cells (default 48).");
      args.AddOption(&strip_separation, "-strip-separation", "--strip-separation",
                     "Strip centerline separation in problem 7 (default: strip thickness, touching).");
      args.AddOption(&wrap_angle, "-wrap-angle", "--wrap-angle",
                     "Initial stress-free strip twist in problem 7 (default 0 degrees).");
      args.AddOption(&rope_radius, "-rope-radius", "--rope-radius",
                     "Circular rope radius in problem 8 (default 0.08).");
      args.AddOption(&rope_wrap_angle, "-rope-wrap-angle", "--rope-wrap-angle",
                     "Stress-free helical wrap in problem 8 (default 720 degrees).");
      args.AddOption(&rope_tighten_angle, "-rope-tighten-angle",
                     "--rope-tighten-angle",
                     "Additional winding at full load, shared by the two grips (default 90 degrees).");
      args.AddOption(&rope_pull, "-rope-pull", "--rope-pull",
                     "Total outward axial separation of the rope grips (default 0.2).");
      args.AddOption(&rope_gap, "-rope-gap", "--rope-gap",
                     "Initial rope-to-cylinder clearance (default 0.01).");
      args.AddOption(&rope_length_cells, "-rope-nx", "--rope-length-cells",
                     "Cells along the helical rope centerline (default 48).");
      args.AddOption(&rope_young_modulus, "-rope-young-modulus",
                     "--rope-young-modulus",
                     "Neo-Hookean rope Young's modulus (default 3000).");
      args.AddOption(&rope_poisson_ratio, "-rope-poisson-ratio",
                     "--rope-poisson-ratio",
                     "Rope Poisson ratio (default 0.3).");
      args.AddOption(&cylinder_length, "-cylinder-length", "--cylinder-length",
                     "Elastic cylinder length in problem 8 (default 3).");
      args.AddOption(&cylinder_radius, "-cylinder-radius", "--cylinder-radius",
                     "Elastic cylinder radius in problem 8 (default 0.5).");
      args.AddOption(&cylinder_axial_cells, "-cylinder-nx", "--cylinder-axial-cells",
                     "Cylinder cells along its axis in problem 8 (default 12).");
      args.AddOption(&cylinder_cross_cells, "-cylinder-nc", "--cylinder-cross-cells",
                     "Cylinder cells per cross-section direction in problem 8 (default 6).");
      args.AddOption(&cylinder_young_modulus, "-cylinder-young-modulus",
                     "--cylinder-young-modulus",
                     "Neo-Hookean cylinder Young's modulus (default 1000; decrease for a softer cylinder).");
      args.AddOption(&cylinder_poisson_ratio, "-cylinder-poisson-ratio",
                     "--cylinder-poisson-ratio",
                     "Cylinder Poisson ratio (default 0.3).");
   }
   bool ResolveOptions(int argc, char *argv[]) override
   {
      if (id < onion || id > ropecylinder)
      {
         if (Mpi::Root()) { mfem::err << "contact_examples supports 3..8. Use contact for 0, 1, 2.\n"; }
         return false;
      }
      rope_case = id == ropecylinder;
      strips_case = id == twistedstrips;
      snap_fit_case = id == snapfit;
      auto supplied = [&](std::initializer_list<const char *> names)
      { return ContactOptionSupplied(argc, argv, names); };
      if (opts.load_steps == -1) { opts.load_steps = rope_case ? 48 : strips_case ? 72 : snap_fit_case ? 32 : id == onion ? 1 : 20; }
      opts.nonlinear = id != onion;
      if (opts.tribol_ratio == -1) { opts.tribol_ratio = opts.nonlinear ? 2 : 8; }
      if (strip_length_cells == -1) { strip_length_cells = 48; }
      if (!supplied({"-strip-separation", "--strip-separation"})) { strip_separation = strip_thickness; }
      contact_name = rope_case ? "Rope/cylinder" : strips_case ?
                     (wrap_angle == 0 ? "Stacked elastic strips" : "Wrapped elastic strips") :
                     snap_fit_case ? "Snap-fit" : id == ring ? "Ring" :
                     id == foldedstrip ? "Folded strip" : "Onion";
      if (strips_case && (!IsFinite(twist_angle) || !IsFinite(twist_length) ||
                          twist_length <= 0 ||
                          !IsFinite(twist_shortening) || twist_shortening < 0 ||
                          twist_shortening >= twist_length))
      {
         if (Mpi::Root()) { mfem::err << "Require finite angle, positive strip length and 0 <= shortening < length.\n"; }
         return false;
      }
      if (strips_case && (!IsFinite(strip_width) || strip_width <= 0 ||
                          !IsFinite(strip_thickness) || strip_thickness <= 0 ||
                          !IsFinite(strip_young_modulus) || strip_young_modulus <= 0 ||
                          !IsFinite(strip_poisson_ratio) || strip_poisson_ratio <= -1 ||
                          strip_poisson_ratio >= 0.5 ||
                          strip_length_cells < 1 || !IsFinite(wrap_angle) ||
                          !IsFinite(strip_separation) || strip_separation < strip_thickness))
      {
         if (Mpi::Root()) { mfem::err << "Require valid strip dimensions, elastic constants, initial twist, separation and axial cells.\n"; }
         return false;
      }
      if (rope_case)
      {
         auto positive = [](real_t value) { return IsFinite(value) && value > 0; };
         auto poisson = [](real_t value) { return IsFinite(value) && value > -1 && value < 0.5; };
         if (!positive(rope_radius) || !positive(cylinder_radius) ||
             !positive(cylinder_length) ||
             !positive(rope_young_modulus) || !positive(cylinder_young_modulus) ||
             !poisson(rope_poisson_ratio) || !poisson(cylinder_poisson_ratio) ||
             !IsFinite(rope_wrap_angle) || rope_wrap_angle < 180 ||
             !IsFinite(rope_tighten_angle) || rope_tighten_angle < 0 ||
             !IsFinite(rope_pull) || rope_pull < 0 || !IsFinite(rope_gap) || rope_gap < 0 ||
             rope_length_cells < std::ceil(rope_wrap_angle/15) ||
             cylinder_axial_cells < 2 || cylinder_cross_cells < 2)
         {
            if (Mpi::Root()) { mfem::err << "Require valid rope/cylinder dimensions and elastic constants, wrap >= 180 degrees, nonnegative tightening/pull/gap, at least one rope cell per 15 degrees, and at least two cylinder cells per direction.\n"; }
            return false;
         }
         const real_t span = rope_cylinder::axial_fraction*cylinder_length;
         const real_t winding = rope_wrap_angle*(M_PI/180);
         const real_t center_radius = cylinder_radius + rope_radius + rope_gap;
         // Tube radius must be smaller than the curvature radius; neighboring
         // turns and supported cylinder end caps must remain separated.
         const real_t curvature = center_radius*winding*winding /
                                  (span*span + center_radius*center_radius*winding*winding);
         if (rope_radius >= (cylinder_length - span)/2 ||
             rope_radius*curvature >= 0.5 ||
             span*360/rope_wrap_angle <= 2.2*rope_radius)
         {
            if (Mpi::Root()) { mfem::err << "Rope is too thick for the helix pitch, curvature or cylinder end clearance.\n"; }
            return false;
         }
      }
      if (snap_fit_case && (!IsFinite(clip_travel) || clip_travel < 0 ||
                            !IsFinite(pin_young_modulus) || pin_young_modulus <= 0))
      {
         if (Mpi::Root()) { mfem::err << "Require nonnegative clip travel and positive pin modulus.\n"; }
         return false;
      }
      if ((id == foldedstrip && (!IsFinite(strip_pressure) || strip_pressure < 0)) ||
          (id == ring && (!IsFinite(ring_pressure) || ring_pressure < 0))) { return false; }
      return true;
   }
   std::string Name() const override { return contact_name; }
   real_t MaxCorrection() const override
   {
      return rope_case ? rope_radius/3 : strips_case ? strip_thickness/3 :
             snap_fit_case ? snap_fit::thickness/3 : id == ring ? 0.04 : 0.1;
   }
   real_t MaxDamping() const override { return snap_fit_case || rope_case ? 0.5 : 1; }
   std::unique_ptr<Mesh> MakeMesh() const override
   {
      switch (id)
      {
         case onion: return std::unique_ptr<Mesh>(MakeOnionMesh());
         case foldedstrip: return std::unique_ptr<Mesh>(MakeFoldedStripMesh());
         case ring: return std::unique_ptr<Mesh>(MakeRingMesh());
         case snapfit: return std::unique_ptr<Mesh>(MakeSnapFitMesh(hex_mesh));
         case twistedstrips:
            return std::unique_ptr<Mesh>(MakeTwistingStripsMesh(twist_length, strip_width,
                                                                strip_thickness,
                                                                strip_separation, wrap_angle,
                                                                hex_mesh ? Element::HEXAHEDRON : Element::TETRAHEDRON, strip_length_cells));
         case ropecylinder:
            return std::unique_ptr<Mesh>(MakeRopeCylinderMesh(cylinder_length,
                                                              cylinder_radius,
                                                              rope_radius, rope_wrap_angle, rope_gap,
                                                              rope_length_cells, cylinder_axial_cells,
                                                              cylinder_cross_cells,
                                                              hex_mesh ? Element::HEXAHEDRON : Element::TETRAHEDRON));
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
            // Pin faces remain in plane strain. Small central end-face patches
            // anchor the axis while the outer pin deforms under contact forces.
            ess_bdr_attr.Append(snap_fit::pin_ends); ess_bdr_attr_comp.Append(1);
            ess_bdr_attr.Append(snap_fit::pin_contact); ess_bdr_attr_comp.Append(1);
            ess_bdr_attr.Append(snap_fit::pin_support); ess_bdr_attr_comp.Append(-1);
            E = 1.e3;
            E[snap_fit::pin_body - 1] = pin_young_modulus;
            nu = 0.3;
            break;
         case twistedstrips:
            ess_bdr_attr.Append(twist::left_end); ess_bdr_attr_comp.Append(-1);
            ess_bdr_attr.Append(twist::right_end); ess_bdr_attr_comp.Append(-1);
            E = strip_young_modulus;
            nu = strip_poisson_ratio;
            break;
         case ropecylinder:
            ess_bdr_attr.Append(rope_cylinder::left_grip); ess_bdr_attr_comp.Append(-1);
            ess_bdr_attr.Append(rope_cylinder::right_grip); ess_bdr_attr_comp.Append(-1);
            // Only the cylinder end caps are supported; its middle is elastic.
            ess_bdr_attr.Append(rope_cylinder::cylinder_support);
            ess_bdr_attr_comp.Append(-1);
            E[rope_cylinder::rope_body - 1] = rope_young_modulus;
            E[rope_cylinder::cylinder_body - 1] = cylinder_young_modulus;
            nu[rope_cylinder::rope_body - 1] = rope_poisson_ratio;
            nu[rope_cylinder::cylinder_body - 1] = cylinder_poisson_ratio;
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
            // Keep the mortar pressure basis on the pin contact surface.
            contact_interfaces.push_back({{snap_fit::clip_contact},
               {snap_fit::pin_contact}});
            break;
         case twistedstrips:
            contact_interfaces.push_back({{twisted_strips::first_surface},
               {twisted_strips::second_surface}});
            break;
         case ropecylinder:
            contact_interfaces.push_back({{rope_cylinder::rope_contact},
               {rope_cylinder::cylinder_contact}});
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
         case twistedstrips: return hex_mesh ? "strips_hex" : "strips_tet";
         case ropecylinder: return hex_mesh ? "rope_hex" : "rope_tet";
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
      if (rope_case)
      {
         cout << "Rope/cylinder: Neo-Hookean, rope E = " << rope_young_modulus
              << ", nu = " << rope_poisson_ratio << ", radius = " << rope_radius
              << "; cylinder E = " << cylinder_young_modulus << ", nu = " <<
              cylinder_poisson_ratio
              << ", radius = " << cylinder_radius << ", length = " << cylinder_length
              << "; initial wrap = " << rope_wrap_angle << " degrees, clearance = " <<
              rope_gap << endl;
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
      if (rope_case)
      {
         snap_fit::GetBodyDofs(fes, rope_cylinder::rope_body, rope_dofs);
         rope_reference = *fes.GetParMesh()->GetNodes();
         Array<int> marker(fes.GetNDofs()), dofs;
         marker = 0;
         for (int i = 0; i < fes.GetNBE(); i++)
         {
            if (fes.GetParMesh()->GetBdrAttribute(i) != rope_cylinder::cylinder_contact) { continue; }
            fes.GetBdrElementDofs(i, dofs);
            for (int dof : dofs) { marker[dof] = 1; }
         }
         for (int dof = 0; dof < marker.Size(); dof++)
         {
            if (marker[dof]) { cylinder_surface_dofs.Append(dof); }
         }
      }
   }
   bool InitialClearance(real_t &clearance) const override
   {
      clearance = strip_separation - strip_thickness;
      return strips_case && wrap_angle == 0;
   }
   void PrintState(const ParGridFunction &displacement) const override
   {
      if (snap_fit_case) { snap_fit::PrintState(displacement, clip_dofs, pin_dofs); }
      if (rope_case)
      {
         const auto &fes = *displacement.ParFESpace();
         const real_t *u = displacement.HostRead();
         const real_t *r = rope_reference.HostRead();
         real_t local_inward = 0, inward;
         for (int dof : cylinder_surface_dofs)
         {
            const int y = fes.DofToVDof(dof, 1), z = fes.DofToVDof(dof, 2);
            local_inward = std::max(local_inward, std::hypot(r[y], r[z]) -
                                    std::hypot(r[y] + u[y], r[z] + u[z]));
         }
         MPI_Allreduce(&local_inward, &inward, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_MAX, fes.GetComm());
         if (Mpi::Root())
         {
            cout << "Cylinder: maximum inward surface displacement = " << inward
                 << " (" << 100*inward/cylinder_radius << "% of reference radius)" << endl;
         }
      }
   }
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
            f.constant = -p*(i+1)/opts.load_steps;
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
            const real_t load_fraction = real_t(i+1)/opts.load_steps;
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
            const real_t fraction = real_t(i+1)/opts.load_steps;
            const real_t travel = clip_travel * fraction;
            ess_bdr = 0; ess_bdr[snap_fit::grip - 1] = 1;
            ess_values[0] = travel;
            prob.SetDisplacementDirichletData(ess_values, ess_bdr);
            // Predict a rigid translation before linearizing the material and
            // contact. Updating only the grip would stretch/invert the rear
            // elements instead of carrying forward the previous clip shape.
            snap_fit::TranslateClip(x_gf, clip_dofs, travel - previous_clip_travel);
            previous_clip_travel = travel;
            if (myid == 0 && opts.print_level >= 2)
            {
               cout << "Snap-fit clip travel = " << travel << endl;
            }
            break;
         }
         case twistedstrips:
         {
            const real_t fraction = real_t(i+1)/opts.load_steps;
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
         case ropecylinder:
         {
            const real_t fraction = real_t(i+1)/opts.load_steps;
            const real_t increment = fraction - previous_rope_fraction;
            const real_t angle = rope_tighten_angle*(M_PI/180)*fraction;
            const real_t pull = rope_pull*fraction;
            const real_t span = rope_cylinder::axial_fraction*cylinder_length;
            const auto &fes = *x_gf.ParFESpace();
            const real_t *r = reference.HostRead();
            real_t *u = x_gf.HostReadWrite();
            // Predict winding/pulling on rope nodes only. Cylinder nodes retain
            // their previous elastic deformation. The grip values below are exact.
            for (int dof : rope_dofs)
            {
               const int x = fes.DofToVDof(dof, 0), y = fes.DofToVDof(dof, 1),
                         z = fes.DofToVDof(dof, 2);
               const real_t position = std::max(real_t(-0.5), std::min(real_t(0.5),
                                                                       (r[x] - cylinder_length/2)/span));
               const real_t rotation = rope_tighten_angle*(M_PI/180)*increment*position;
               const real_t cosine = std::cos(rotation), sine = std::sin(rotation);
               const real_t py = r[y] + u[y], pz = r[z] + u[z];
               u[x] += rope_pull*increment*position;
               u[y] = cosine*py - sine*pz - r[y];
               u[z] = sine*py + cosine*pz - r[z];
            }
            for (int end = 0; end < 2; end++)
            {
               const real_t sign = end == 0 ? -1 : 1;
               auto grip_displacement = [=](const Vector &point, Vector &delta)
               {
                  const real_t rotation = sign*angle/2;
                  const real_t cosine = std::cos(rotation), sine = std::sin(rotation);
                  delta.SetSize(3);
                  delta[0] = sign*pull/2;
                  delta[1] = (cosine - 1)*point[1] - sine*point[2];
                  delta[2] = sine*point[1] + (cosine - 1)*point[2];
               };
               VectorFunctionCoefficient grip_bc(dim, grip_displacement);
               ess_bdr = 0;
               ess_bdr[(end == 0 ? rope_cylinder::left_grip : rope_cylinder::right_grip) - 1] =
                  1;
               prob.SetDisplacementDirichletData(grip_bc, ess_bdr);
               x_gf.ProjectBdrCoefficient(grip_bc, ess_bdr);
            }
            previous_rope_fraction = fraction;
            if (myid == 0 && opts.print_level >= 2)
            {
               cout << "Rope tightening: added winding = " << rope_tighten_angle*fraction
                    << " degrees, outward grip separation = " << pull << endl;
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
