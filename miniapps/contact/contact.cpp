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

// Contact problems: 0 two-block, 1 ironing, 2 beam-sphere.
// Build: make contact. Run: mpirun -np 1 ./contact -prob 0 -sr 0 -no-vis.
// See docs/README.md for solver options, first-order geometry, and visualization.
#include "contact_driver.hpp"
#include "mesh_generator.hpp"
#include <iostream>
using namespace mfem;

namespace
{
class ContactProblem : public ContactDriverProblem
{
   int id = 0;
   int beam_cross_cells = -1, beam_axial_cells = -1;
   int sphere_angular_cells = -1, sphere_radial_cells = -1;
   bool hex_mesh = true;
public:
   explicit ContactProblem(ContactDriverOptions &options) : ContactDriverProblem(
         options) { }
   void AddOptions(OptionsParser &args) override
   {
      args.AddOption(&id, "-prob", "--problem-number",
                     "Contact problem: 0 two-block, 1 ironing, 2 beam-sphere.");
      args.AddOption(&hex_mesh, "-hex", "--hexahedra", "-tet", "--tetrahedra",
                     "Generated problem elements: default hexes for 0/1, tets for 2.");
      args.AddOption(&beam_cross_cells, "-beam-nc", "--beam-cross-cells",
                     "Generated beam cross-section cells per direction (-1: 8).");
      args.AddOption(&beam_axial_cells, "-beam-ny", "--beam-axial-cells",
                     "Generated beam axial cells (-1: 24).");
      args.AddOption(&sphere_angular_cells, "-sphere-n", "--sphere-angular-cells",
                     "Generated sphere cells per octant cube-face edge (-1: 6).");
      args.AddOption(&sphere_radial_cells, "-sphere-nr", "--sphere-radial-cells",
                     "Generated sphere radial cells in each shell (-1: 2 outer / 1 inner).");
   }
   bool ResolveOptions(int argc, char *argv[]) override
   {
      if (id < 0 || id > 2)
      {
         if (Mpi::Root()) { mfem::err << "contact supports problems 0, 1, 2. Use contact_examples for 3..8.\n"; }
         return false;
      }
      if (!ContactOptionSupplied(argc, argv,
   {"-hex", "--hexahedra", "-tet", "--tetrahedra"}))
      { hex_mesh = id != beamsphere; }
      if (opts.sref == -1) { opts.sref = 1; }
      if (opts.nsteps == -1) { opts.nsteps = 1; }
      if (opts.msteps == -1) { opts.msteps = 0; }
      if (opts.tribol_ratio == -1) { opts.tribol_ratio = 8; }
      for (int cells :
           {
              beam_cross_cells, beam_axial_cells,
              sphere_angular_cells, sphere_radial_cells
           })
      {
         if (cells == 0 || cells < -1)
         {
            if (Mpi::Root()) { mfem::err << "Mesh cell counts must be positive or -1 for automatic resolution.\n"; }
            return false;
         }
      }
      if (id != beamsphere) { opts.nonlinear = false; }
      return true;
   }
   std::unique_ptr<Mesh> MakeMesh() const override
   {
      return std::unique_ptr<Mesh>(MakeContactProblemMesh(
                                      id, hex_mesh, beam_cross_cells, beam_axial_cells,
                                      sphere_angular_cells, sphere_radial_cells));
   }
   std::string Name() const override
   { return id == twoblock ? "Two-block" : id == ironing ? "Ironing" : "Beam-sphere"; }
   std::string OutputName() const override
   {
      const std::string name = id == twoblock ? "blocks" :
                               id == ironing ? "ironing" : "beam_sphere";
      return name + (hex_mesh ? "_hex" : "_tet");
   }
   void Configure(ParMesh &, Vector &E, Vector &nu,
                  Array<int> &ess_bdr_attr, Array<int> &ess_bdr_attr_comp) const override
   {
      switch (id)
      {
         case twoblock:
         case ironing:
            ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(-1);
            ess_bdr_attr.Append(6); ess_bdr_attr_comp.Append(-1);
            E[0] = 1.0;  E[1] = 1e3;
            nu[0] = 0.499;  nu[1] = 0.0;
            break;
         case beamsphere:
            ess_bdr_attr.Append(1); ess_bdr_attr_comp.Append(0);
            ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(1);
            ess_bdr_attr.Append(4); ess_bdr_attr_comp.Append(2);
            ess_bdr_attr.Append(5); ess_bdr_attr_comp.Append(-1);
            E = 1.e3;
            nu = 0.4;
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
         case twoblock:
         case  ironing:
            contact_interfaces.push_back({{3}, {4}});
            break;
         case beamsphere:
            contact_interfaces.push_back({{6, 9}, {7, 8}});
            break;
         default:
            MFEM_ABORT("Should be unreachable");
            break;
      }
      return contact_interfaces;
   }
   void ApplyLoad(int i, ElasticityOperator &prob, ParGridFunction &,
                  const ParGridFunction &) override
   {
      const int dim = prob.GetMesh()->Dimension();
      Vector ess_values(dim); ess_values = 0.0;
      Array<int> ess_bdr(prob.GetMesh()->bdr_attributes.Max()); ess_bdr = 0;
      const real_t p = 40.0;
      ConstantCoefficient f(p);
      switch (id)
      {
         case twoblock:
         case ironing:
            if (ess_bdr.Size())
            {
               ess_bdr = 0; ess_bdr[5] = 1;
            }
            if (i < opts.nsteps)
            {
               ess_values[2] = -1.0/1.4*(i+1)/opts.nsteps;
            }
            else
            {
               ess_values[0] = 3.0/1.4*(i+1-opts.nsteps)/opts.msteps;
               ess_values[2] = -1.0/1.4;
            }
            prob.SetDisplacementDirichletData(ess_values, ess_bdr);
            break;
         case beamsphere:
            if (ess_bdr.Size())
            {
               ess_bdr = 0; ess_bdr[2] = 1;
            }
            f.constant = -p*(i+1)/opts.nsteps;
            prob.SetNeumannPressureData(f,ess_bdr);
            break;
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
   ContactProblem problem(options);
   return RunContactDriver(argc, argv, options, problem);
}
