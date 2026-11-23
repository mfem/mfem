// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
// This file is part of MFEM; see https://mfem.org and CONTRIBUTING.md.

#ifndef MFEM_CONTACT_DRIVER_HPP
#define MFEM_CONTACT_DRIVER_HPP

#include "ip.hpp"
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace mfem
{

/// Solver, discretization, and output options shared by both contact drivers.
struct ContactDriverOptions
{
   int sref = -1, pref = 0;
   int nsteps = -1, msteps = -1;
   int opt_maxit = 100, print_level = 2;
   int contact_max_updates = 10;
   real_t contact_update_tol = 1e-6, tribol_ratio = -1.0;
   bool nonlinear = false, elasticity_fa = false;
   bool allow_inexact_contact_updates = true;
   bool visualization = true, paraview = false, amgf = false;
   const char *amgf_fsolver = "auto", *linear_solver = "cg";
   int amg_coarsen_type = 8, amg_agg_levels = 0;
   int amg_interp_type = 6, amg_relax_type = 16;
   bool amg_print_options = false, hypre_gpu_aware_mpi = false;
   const char *device_config = "cpu";
};

/// A problem supplies geometry and loading; the shared driver owns the solve.
class ContactDriverProblem
{
protected:
   ContactDriverOptions &opts;
public:
   explicit ContactDriverProblem(ContactDriverOptions &options) : opts(options) { }
   virtual ~ContactDriverProblem() = default;
   virtual void AddOptions(OptionsParser &args) = 0;
   /// Validate problem options and resolve mode-dependent driver defaults.
   virtual bool ResolveOptions(int argc, char *argv[]) = 0;
   virtual std::unique_ptr<Mesh> MakeMesh() const = 0;
   virtual void Configure(ParMesh &mesh, Vector &E, Vector &nu,
                          Array<int> &attrs, Array<int> &components) const = 0;
   virtual std::vector<ContactInterface> Interfaces() const = 0;
   virtual std::string Name() const = 0;
   /// Short geometry/cell-type name; driver adds a unique UTC timestamp.
   virtual std::string OutputName() const = 0;
   virtual bool UpdatesGeometry() const { return false; }
   virtual real_t MaxCorrection() const { return 0.1; }
   virtual real_t MaxDamping() const { return 1.0; }
   virtual void Initialize(ElasticityOperator &problem) { }
   virtual void ApplyLoad(int step, ElasticityOperator &problem,
                          ParGridFunction &displacement,
                          const ParGridFunction &reference) = 0;
   virtual void PrintParameters() const { }
   virtual void PrintState(const ParGridFunction &displacement) const { }
   /// Request the initially touching contact diagnostic.
   virtual bool InitialClearance(real_t &clearance) const { return false; }
};

/// Shared MPI initialization, option parsing, nonlinear/IP solve, and output.
int RunContactDriver(int argc, char *argv[], ContactDriverOptions &options,
                     ContactDriverProblem &problem);

/// Test whether the user supplied any spelling of an option.
bool ContactOptionSupplied(int argc, char *argv[],
                           std::initializer_list<const char *> names);


}
#endif
