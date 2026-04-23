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
//
//    ------------------------------------------------------------------
//    Maxwell Miniapp:  Simple Full-Wave Electromagnetic Simulation Code
//    ------------------------------------------------------------------
//
// This miniapp solves a simple 3D full-wave electromagnetic problem using the
// coupled, first-order equations:
//
//                 epsilon dE/dt = Curl 1/mu B - sigma E - J
//                         dB/dt = - Curl E
//
// The permittivity function is that of the vacuum with an optional dielectric
// sphere. The permeability function is that of the vacuum with an optional
// diamagnetic or paramagnetic spherical shell. The optional conductivity
// function is also a user-defined sphere.
//
// The optional current density is a pulse of current in the shape of a cylinder
// with a time dependence resembling the derivative of a Gaussian distribution.
//
// Boundary conditions can be 'natural' meaning zero tangential current,
// 'Dirichlet' which sets the time-derivative of the tangential components of E,
// or 'absorbing' (we use a simple Sommerfeld first order absorbing boundary
// condition).
//
// We discretize the electric field with H(Curl) finite elements (Nedelec edge
// elements) and the magnetic flux with H(Div) finite elements (Raviart-Thomas
// elements).
//
// The symplectic time integration algorithm used below is designed to conserve
// energy unless lossy materials or absorbing boundary conditions are used.
// When losses are expected, the algorithm uses an implicit method which
// includes the loss operators in the left hand side of the linear system.
//
// For increased accuracy the time integration order can be set to 2, 3, or 4
// (the default is 1st order).
//
// See: J.C. Nedelec, Mixed Finite Elements in R^3, Numerische Mathematik, 35
// (1980), 315-341. https://doi.org/10.1007/BF01396415
//
// Sample runs:
//
// clang-format off
//
//   Current source in a sphere with absorbing boundary conditions:
//     mpirun -np 4 maxwell -m ../../data/ball-nurbs.mesh -rs 2 -abcs '-1' -dp '-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5'
//
//   Current source in a metal sphere with dielectric and conducting materials:
//     mpirun -np 4 maxwell -m ../../data/ball-nurbs.mesh -rs 2 -dbcs '-1' -dp '-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5' -cs '0.0 0.0 -0.5 .2 3e6' -ds '0.0 0.0 0.5 .2 10'
//
//   Current source in a metal box:
//     mpirun -np 4 maxwell -m ../../data/fichera.mesh -rs 3 -ts 0.25 -tf 10 -dbcs '-1' -dp '-0.5 -0.5 0.0 -0.5 -0.5 1.0 0.1 1 .5 1'
//
//   Current source with a mixture of absorbing and reflecting boundaries:
//     mpirun -np 4 maxwell -m ../../data/fichera.mesh -rs 3 -ts 0.25 -tf 10 -dp '-0.5 -0.5 0.0 -0.5 -0.5 1.0 0.1 1 .5 1' -dbcs '4 8 19 21' -abcs '5 18'
//
//   By default the sources and fields are all zero:
//   * mpirun -np 4 maxwell
//
// clang-format on

#include "mfem.hpp"

#include "../common/mesh_extras.hpp"
#include "electromagnetics.hpp"

#include <fstream>
#include <iostream>

using namespace mfem;

// Prints the program's logo to the given output stream
void display_banner(std::ostream &os);

real_t dielectric_sphere(const Vector &x, const Vector &ds_params);
real_t magnetic_shell(const Vector &x, const Vector &ms_params);
real_t conductive_sphere(const Vector &x, const Vector &cs_params);
// dE/dt Boundary Condition: The following function returns zero but any time
// dependent function could be used.
void dEdtBCFunc(const Vector &x, real_t t, Vector &E);
// The following functions return zero but they could be modified to set initial
// conditions for the electric and magnetic fields
void EFieldFunc(const Vector &x, Vector &E);
void BFieldFunc(const Vector &x, Vector &B);

/// assumes x is 3D
MFEM_HOST_DEVICE void dipole_pulse(const real_t *x, real_t t, real_t *j,
                                   const real_t *dp_params, real_t tscale)
{
   constexpr int ndims = 3;
   real_t v[ndims];
   real_t xu[ndims];
   real_t h = 0;
   for (int i = 0; i < ndims; ++i)
   {
      j[i] = 0;
      xu[i] = x[i] - dp_params[i];
      v[i] = dp_params[ndims + i] - dp_params[i];
      h += v[i] * v[i];
   }
   if (h == 0)
   {
      return;
   }
   for (int i = 0; i < ndims; ++i)
   {
      v[i] /= h;
   }
   real_t r = dp_params[2 * ndims];
   real_t a = dp_params[2 * ndims + 1] * tscale;
   real_t b = dp_params[2 * ndims + 2] * tscale;
   real_t c = dp_params[2 * ndims + 3] * tscale;

   real_t xv = 0;
   for (int i = 0; i < ndims; ++i)
   {
      xv += xu[i] * v[i];
   }
   real_t xp = 0;
   for (int i = 0; i < ndims; ++i)
   {
      xu[i] -= xv * v[i];
      xp += xu[i] * xu[i];
   }
   if (xv >= 0 && xv <= h && xp <= r)
   {
      real_t mag = a * (t - b) * exp(-0.5 * pow((t - b) / c, 2)) / (c * c);
      for (int i = 0; i < ndims; ++i)
      {
         j[i] = v[i] * mag;
      }
   }
}

/// since current density is (potentially) time varying, evaluate it on the GPU
/// in a special linear form integrator
struct CurrentIntegrator : public LinearFormIntegrator
{
   ParMesh *pmesh;
   // parameters state
   const Vector *dp_params;
   // for getting the coordinates evaluated at quadrature points
   const GeometricFactors *geo = nullptr;
   real_t tscale = 1;
   real_t t = 0;
   CurrentIntegrator(ParMesh &pm, const Vector &dp, real_t ts,
                     const IntegrationRule *ir = nullptr)
      : LinearFormIntegrator(ir), pmesh(&pm), dp_params(&dp), tscale(ts)
   {}

   bool SupportsDevice() const override { return true; }

   void AssembleDevice(const FiniteElementSpace &fes, const Array<int> &markers,
                       Vector &b) override;

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override;
   using LinearFormIntegrator::AssembleRHSElementVect;
};

///
/// LHS Operator for the CG solver of AmpereOperator
///
struct AmpereAction : public Operator
{
   // not owned
   ParMesh *pmesh = nullptr;
   ParFiniteElementSpace *hcurl_space = nullptr;
   ParFiniteElementSpace *hdiv_space = nullptr;
   Coefficient *sigma_coeff = nullptr;
   VectorCoefficient *dEdtbc_coeff = nullptr;
   /// int epsilon dE/dt . E* dV
   ParBilinearForm hcurl_mass;
   /// int sigma E . E* dV
   std::unique_ptr<ParBilinearForm> loss_term;
   /// int J . E* dV
   std::unique_ptr<ParLinearForm> j_term;
   CurrentIntegrator *current_integrator;
   /// int B / mu . Curl E* dV
   ParMixedBilinearForm weak_curl;

   std::unique_ptr<Coefficient> eps_coeff;
   std::unique_ptr<Coefficient> inv_mu_coeff;
   std::unique_ptr<Coefficient> abc_coeff;

   // Array of 0's and 1's marking the location of absorbing surfaces
   Array<int> abc_marker;
   // Array of 0's and 1's marking the location of Dirichlet boundaries
   Array<int> dbc_marker;

   // Dirichlet dofs
   Array<int> dbc_dofs;

   // current dt, needed if there are loss terms
   mutable real_t dt = 0;

   AmpereAction(ParMesh &pmesh_, ParFiniteElementSpace &hcurl,
                ParFiniteElementSpace &hdiv, size_t assembly_type,
                Coefficient &eps_coeff, Coefficient &inv_mu_coeff,
                Coefficient *sigma, VectorCoefficient *dEdtbc, Array<int> &abcs,
                Array<int> &dbcs, CurrentIntegrator *current_integrator_);

   // int epsilon x . E* dV + 0.5 sigma dt x . E* dV + oint 0.5 dt
   // sqrt(epsilon0 / mu0) x . E* dS -> y
   // note: the loss terms which are multiplied by 0.5 dt so in combination with
   // the RHS the discrete form is F((E_{n+1} + E_{n})/2), where F is the loss
   // operator
   void Mult(const Vector &x, Vector &y) const override;

   void EliminateRHS(Vector &rhs) const;
};

/// weak form:
/// int epsilon dE/dt . E* dV = int B / mu . Curl E* - (sigma E + J) . E* dV -
/// oint sqrt(epsilon0 / mu0) E . E* dS
///   E* are test functions in H(curl)
///   E in loss terms is evaluated as an average of E_{n+1} and E_{n}
struct AmpereOperator : public TimeDependentOperator
{
   AmpereAction action;
   CGSolver solver;

   mutable ParGridFunction rhs;
   mutable std::unique_ptr<ParGridFunction> dedt;

   // not owned, needed to compute loss terms
   ParGridFunction *E_gf = nullptr;

   /// assumes ownership of current_integrator (if any)
   AmpereOperator(ParMesh &pmesh_, ParFiniteElementSpace &hcurl,
                  ParFiniteElementSpace &hdiv, size_t assembly_type,
                  Coefficient &eps_coeff, Coefficient &inv_mu_coeff,
                  Coefficient *sigma, VectorCoefficient *dEdtbc,
                  Array<int> &abcs, Array<int> &dbcs,
                  CurrentIntegrator *current_integrator, ParGridFunction &E);

   void Mult(const Vector &B, Vector &dEdt) const override;

   void ImplicitSolve(const real_t dt, const Vector &B, Vector &dEdt) override;
};

/// Faraday's equation maps -curl E from H(curl) to H(div) (exact in 3D)
/// dB/dt = - Curl E
struct FaradayOperator : public Operator
{
   // not owned
   ParMesh *pmesh = nullptr;
   ParFiniteElementSpace *hcurl_space = nullptr;
   ParFiniteElementSpace *hdiv_space = nullptr;

   ParDiscreteLinearOperator curl_op;
   std::unique_ptr<HypreParMatrix> neg_curl;

   // temporary workspace vectors
   mutable std::unique_ptr<HypreParVector> E_work;
   mutable std::unique_ptr<HypreParVector> dBdt_work;

   FaradayOperator(ParMesh &pmesh_, ParFiniteElementSpace &hcurl,
                   ParFiniteElementSpace &hdiv, size_t assembly_type);

   void Mult(const Vector &E, Vector &dBdt) const override;
};

int main(int argc, char *argv[])
{
   using namespace std::literals::string_literals;

   Mpi::Init();
   Hypre::Init();

   if (Mpi::Root())
   {
      display_banner(std::cout);
   }

   // Parse command-line options.
   const char *mesh_file = "../../data/ball-nurbs.mesh";
   int sOrder = 1;
   int tOrder = 1;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   int visport = 19916;
   bool visualization = true;
   bool visit = true;
   real_t dt = 1.0e-12;
   real_t dtsf = 0.95;
   real_t ti = 0.0;
   real_t ts = 1.0;
   real_t tf = 40.0;

   // Permittivity Function
   // Center, Radius, and Permittivity of dielectric sphere
   Vector ds_params(0);
   // Permeability Function
   // Center, Inner and Outer Radii, and Permeability of magnetic shell
   Vector ms_params(0);

   // Conductivity Function
   // Center, Radius, and Conductivity of conductive sphere
   Vector cs_params(0);

   // Current Density Function
   // Axis Start, Axis End, Rod Radius, Total Current of Rod, and Frequency
   Vector dp_params(0);

   // Scale factor between input time units and seconds
   real_t tscale = 1e-9; // Input time in nanosecond

   Array<int> abcs;
   Array<int> dbcs;
   const char *device_config = "cpu";
   // partial: 0
   // full: 1
   size_t assembly_type = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&sOrder, "-so", "--spatial-order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&tOrder, "-to", "--temporal-order",
                  "Time integration order.");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&dtsf, "-sf", "--dt-safety-factor",
                  "Used to reduce the time step below the upper bound.");
   args.AddOption(&ti, "-ti", "--initial-time",
                  "Beginning of time interval to simulate (ns).");
   args.AddOption(&tf, "-tf", "--final-time",
                  "End of time interval to simulate (ns).");
   args.AddOption(&ts, "-ts", "--snapshot-time",
                  "Time between snapshots (ns).");
   args.AddOption(&ds_params, "-ds", "--dielectric-sphere-params",
                  "Center, Radius, and Permittivity of Dielectric Sphere");
   args.AddOption(&ms_params, "-ms", "--magnetic-shell-params",
                  "Center, Inner Radius, Outer Radius, and Permeability "
                  "of Magnetic Shell");
   args.AddOption(&cs_params, "-cs", "--conductive-sphere-params",
                  "Center, Radius, and Conductivity of Conductive Sphere");
   args.AddOption(&dp_params, "-dp", "--dipole-pulse-params",
                  "Axis End Points, Radius, Amplitude, "
                  "Pulse Center (ns), Pulse Width (ns)");
   args.AddOption(&abcs, "-abcs", "--absorbing-bc-surf",
                  "Absorbing Boundary Condition Surfaces");
   args.AddOption(&dbcs, "-dbcs", "--dirichlet-bc-surf",
                  "Dirichlet Boundary Condition Surfaces");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visualization",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOptionChoice(&assembly_type, "-a", "--assembly",
                        "Operator assembly level",
                        std::vector<std::string>({"partial"s, "full"s}));
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(std::cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(std::cout);
   }

   Device device(device_config);
   if (Mpi::Root())
   {
      device.Print();
   }
   // Read the (serial) mesh from the given mesh file on all processors.  We can
   // handle triangular, quadrilateral, tetrahedral, hexahedral, surface and
   // volume meshes with the same code.
   std::unique_ptr<Mesh> mesh;
   {
      std::ifstream imesh(mesh_file);
      if (!imesh)
      {
         if (Mpi::Root())
         {
            std::cerr << "\nCan not open mesh file: " << mesh_file << '\n'
                      << std::endl;
         }
         return 2;
      }
      mesh.reset(new Mesh(imesh, 1, 1));
   }
   // Project a NURBS mesh to a piecewise-quadratic curved mesh
   if (mesh->NURBSext)
   {
      mesh->UniformRefinement();
      if (serial_ref_levels > 0)
      {
         serial_ref_levels--;
      }

      mesh->SetCurvature(2);
   }

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }
   // Define a parallel mesh by a partitioning of the serial mesh. Refine this
   // mesh further in parallel to increase the resolution. Once the parallel
   // mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   mesh.reset();

   // Refine this mesh in parallel to increase the resolution.
   for (int l = 0; l < parallel_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   // finite element collections and spaces
   ND_FECollection hcurl_fec(sOrder, pmesh.Dimension());
   RT_FECollection hdiv_fec(sOrder - 1, pmesh.Dimension());
   ParFiniteElementSpace hcurl_space(&pmesh, &hcurl_fec);
   ParFiniteElementSpace hdiv_space(&pmesh, &hdiv_fec);

   const HYPRE_BigInt glob_hcurl_size = hcurl_space.GlobalTrueVSize();
   const HYPRE_BigInt glob_hdiv_size = hdiv_space.GlobalTrueVSize();
   if (Mpi::Root())
   {
      std::cout << "Number of H(Curl) dofs: " << glob_hcurl_size << std::endl;
      std::cout << "Number of H(Div) dofs: " << glob_hdiv_size << std::endl;
   }

   ParGridFunction E_gf(&hcurl_space), B_gf(&hdiv_space);
   // initial conditions
   {
      VectorFunctionCoefficient e0(3, EFieldFunc);
      E_gf.ProjectCoefficient(e0);
   }
   {
      VectorFunctionCoefficient b0(3, BFieldFunc);
      E_gf.ProjectCoefficient(b0);
   }

   // Use separate operators for Ampere and Faraday equations for each part of
   // the Hamiltonian operators.
   std::unique_ptr<Coefficient> eps_coeff;
   std::unique_ptr<Coefficient> inv_mu_coeff;
   std::unique_ptr<Coefficient> sigma_coeff;
   std::unique_ptr<VectorCoefficient> dEdtbc_coeff;
   CurrentIntegrator *current_integrator = nullptr;
   if (ds_params.Size() > 0)
   {
      eps_coeff.reset(new FunctionCoefficient([&](const Vector &x)
      { return dielectric_sphere(x, ds_params); }));
   }
   else
   {
      eps_coeff.reset(new ConstantCoefficient(electromagnetics::epsilon0_));
   }
   if (ms_params.Size() > 0)
   {
      inv_mu_coeff.reset(new FunctionCoefficient([&](const Vector &x)
      { return 1_r / magnetic_shell(x, ms_params); }));
   }
   else
   {
      inv_mu_coeff.reset(new ConstantCoefficient(1_r / electromagnetics::mu0_));
   }
   if (cs_params.Size() > 0)
   {
      sigma_coeff.reset(new FunctionCoefficient([&](const Vector &x)
      { return conductive_sphere(x, cs_params); }));
   }
   if (dbcs.Size() > 0)
   {
      dEdtbc_coeff.reset(new VectorFunctionCoefficient(3, dEdtBCFunc));
   }
   if (dp_params.Size() > 0)
   {
      // TODO: current source
      current_integrator = new CurrentIntegrator(pmesh, dp_params, tscale);
   }
   AmpereOperator ampere(pmesh, hcurl_space, hdiv_space, assembly_type,
                         *eps_coeff, *inv_mu_coeff, sigma_coeff.get(),
                         dEdtbc_coeff.get(), abcs, dbcs, current_integrator,
                         E_gf);
   FaradayOperator faraday(pmesh, hcurl_space, hdiv_space, assembly_type);

   // Create the ODE solver
   SIAVSolver siaSolver(tOrder);
   siaSolver.Init(faraday, ampere);
   // TODO: initialize visualization
   // TODO: run sim
   Vector dEdt(hcurl_space.GetVSize());
   ampere.ImplicitSolve(dt, B_gf, dEdt);
   return 0;
}

// Print the Maxwell ascii logo to the given ostream
void display_banner(std::ostream &os)
{
   os << "     ___    ____                                      " << std::endl
      << "    /   |  /   /                           __   __    " << std::endl
      << "   /    |_/ _ /__  ___  _____  _  __ ____ |  | |  |   " << std::endl
      << "  /         \\__  \\ \\  \\/  /\\ \\/ \\/ // __ \\|  | |  |   "
      << std::endl
      << " /   /|_/   // __ \\_>    <  \\     /\\  ___/|  |_|  |__ "
      << std::endl
      << "/___/  /_  /(____  /__/\\_ \\  \\/\\_/  \\___  >____/____/ "
      << std::endl
      << "         \\/       \\/      \\/             \\/           "
      << std::endl;
}

AmpereAction::AmpereAction(ParMesh &pmesh_, ParFiniteElementSpace &hcurl,
                           ParFiniteElementSpace &hdiv, size_t assembly_type,
                           Coefficient &eps_coeff, Coefficient &inv_mu_coeff,
                           Coefficient *sigma, VectorCoefficient *dEdtbc,
                           Array<int> &abcs, Array<int> &dbcs,
                           CurrentIntegrator *current_integrator_)
   : Operator(hcurl.GetVSize(), hcurl.GetVSize()), pmesh(&pmesh_),
     hcurl_space(&hcurl), hdiv_space(&hdiv), hcurl_mass(&hcurl),
     weak_curl(&hdiv, &hcurl), sigma_coeff(sigma), dEdtbc_coeff(dEdtbc),
     current_integrator(current_integrator_)
{
   hcurl_mass.AddDomainIntegrator(new VectorFEMassIntegrator(&eps_coeff));
   weak_curl.AddDomainIntegrator(
      new MixedVectorWeakCurlIntegrator(inv_mu_coeff));
   if (dbcs.Size() > 0)
   {
      common::AttrToMarker(pmesh_.bdr_attributes.Max(), dbcs, dbc_marker);
      hcurl.GetEssentialTrueDofs(dbc_marker, dbc_dofs);
   }
   if (sigma || abcs.Size() > 0)
   {
      loss_term.reset(new ParBilinearForm(&hcurl));
      if (sigma)
      {
         loss_term->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
      }
      if (abcs.Size() > 0)
      {
         common::AttrToMarker(pmesh_.bdr_attributes.Max(), abcs, abc_marker);
         abc_coeff.reset(new ConstantCoefficient(
                            sqrt(electromagnetics::epsilon0_ / electromagnetics::mu0_)));
         loss_term->AddBoundaryIntegrator(
            new VectorFEMassIntegrator(abc_coeff.get()), abc_marker);
      }
   }
   if (current_integrator)
   {
      j_term.reset(new ParLinearForm(hcurl_space));
      j_term->AddDomainIntegrator(current_integrator);
      j_term->UseFastAssembly(true);
   }
   switch (assembly_type)
   {
      case 1:
         // full assembly
         // TODO
         break;
      case 0:
         // partial assembly
         hcurl_mass.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         weak_curl.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         if (loss_term)
         {
            loss_term->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            loss_term->Assemble();
         }
         hcurl_mass.Assemble();
         weak_curl.Assemble();
         break;
   }
}

AmpereOperator::AmpereOperator(
   ParMesh &pmesh_, ParFiniteElementSpace &hcurl, ParFiniteElementSpace &hdiv,
   size_t assembly_type, Coefficient &eps_coeff, Coefficient &inv_mu_coeff,
   Coefficient *sigma, VectorCoefficient *dEdtbc, Array<int> &abcs,
   Array<int> &dbcs, CurrentIntegrator *current_integrator, ParGridFunction &E)
   : TimeDependentOperator(hcurl.GetVSize(), hdiv.GetVSize()),
     action(pmesh_, hcurl, hdiv, assembly_type, eps_coeff, inv_mu_coeff, sigma,
            dEdtbc, abcs, dbcs, current_integrator),
     rhs(&hcurl), E_gf(&E)
{
   if (action.loss_term)
   {
      type = IMPLICIT;
   }
   else
   {
      type = EXPLICIT;
   }
   // TODO: full assembly
   solver.SetOperator(action);
   // TODO: make configurable parameters
   solver.SetAbsTol(0);
   solver.SetRelTol(1e-6);
   solver.SetMaxIter(300);
}

void AmpereAction::Mult(const Vector &x, Vector &y) const
{
   // TODO: full assembly version
   hcurl_mass.Mult(x, y);
   if (loss_term)
   {
      loss_term->AddMult(x, y, 0.5_r * dt);
   }
   // bcs
   if (dbc_dofs.Size() > 0)
   {
      y.SetSubVector(dbc_dofs, 0_r);
   }
}

void AmpereAction::EliminateRHS(Vector& b) const
{
   // for partial assembly
   if (dbc_dofs.Size() > 0)
   {
      b.SetSubVector(dbc_dofs, 0_r);
   }
}

void AmpereOperator::Mult(const Vector &B, Vector &dEdt) const
{
   // TODO
}

void AmpereOperator::ImplicitSolve(const real_t dt, const Vector &B,
                                   Vector &dEdt)
{
   // TODO: full assembly version
   action.dt = dt;
   if (!dedt)
   {
      dedt.reset(new ParGridFunction);
      dedt->MakeRef(action.hcurl_space, dEdt, 0);
      *dedt = 0_r;
   }
   else
   {
      dedt->MakeRef(dEdt, 0);
   }
   // compute RHS
   action.weak_curl.Mult(B, rhs);
   if (action.loss_term)
   {
      action.loss_term->AddMult(*E_gf, rhs, -1);
   }

   if (action.j_term)
   {
      action.current_integrator->t = this->t;
      action.j_term->Assemble();
      rhs -= *action.j_term;
   }
   if (action.dEdtbc_coeff)
   {
      action.dEdtbc_coeff->SetTime(t);
      // TODO: projecting bdr coefficient happens on the CPU
      dedt->ProjectBdrCoefficientTangent(*action.dEdtbc_coeff, action.dbc_marker);
   }
   // dirichlet bc's handled directly by action
   action.EliminateRHS(rhs);
   solver.Mult(rhs, *dedt);
   dedt->SyncAliasMemory(dEdt);
}

/// TODO: for now this always does full assembly into a hypre matrix.
/// CurlInterpolator from H(curl) to H(div) doesn't support partial assembly
/// yet.
// #define FARADAY_HAS_PA

FaradayOperator::FaradayOperator(ParMesh &pmesh_, ParFiniteElementSpace &hcurl,
                                 ParFiniteElementSpace &hdiv,
                                 size_t assembly_type)
   : pmesh(&pmesh_), hcurl_space(&hcurl), hdiv_space(&hdiv),
     curl_op(&hcurl, &hdiv)
{
   curl_op.AddDomainInterpolator(new CurlInterpolator);
#ifdef FARADAY_HAS_PA
   switch (assembly_type)
   {
      case 1:
#endif
         // full assembly
         curl_op.Assemble();
         curl_op.Finalize();
         neg_curl.reset(curl_op.ParallelAssemble());
         *neg_curl *= -1_r;
         E_work.reset(hcurl_space->NewTrueDofVector());
         dBdt_work.reset(hdiv_space->NewTrueDofVector());
#ifdef FARADAY_HAS_PA
         break;
      case 0:
         // partial assembly
         curl_op.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         curl_op.Assemble();
         curl_op.Finalize();
         break;
   }
#endif
}

void FaradayOperator::Mult(const Vector &E, Vector &dBdt) const
{
   if (neg_curl)
   {
      // convert E from L-dofs to T-dofs
      const Operator *R = hcurl_space->GetRestrictionOperator();
      const Operator *P = hdiv_space->GetProlongationMatrix();
      R->Mult(E, *E_work);
      // mult by neg_curl
      neg_curl->Mult(*E_work, *dBdt_work);
      // convert result from T-dofs to L-dofs -> dBdt
      P->Mult(*dBdt_work, dBdt);
   }
   else
   {
      // partial assembly
      curl_op.Mult(E, dBdt);
      dBdt *= -1_r;
   }
}

// A sphere with constant permittivity.
// The sphere has a center, radius, and permittivity specified on the command
// line.
real_t dielectric_sphere(const Vector &x, const Vector &ds_params)
{
   real_t r2 = 0.0;

   for (int i = 0; i < x.Size(); i++)
   {
      r2 += (x(i) - ds_params(i)) * (x(i) - ds_params(i));
   }

   if (sqrt(r2) <= ds_params(x.Size()))
   {
      return ds_params(x.Size() + 1) * electromagnetics::epsilon0_;
   }
   return electromagnetics::epsilon0_;
}

// A spherical shell with constant permeability.  The sphere has center, inner
// and outer radii, and relative permeability specified on the command line and
// stored in ms_params_.
real_t magnetic_shell(const Vector &x, const Vector &ms_params)
{
   real_t r2 = 0.0;

   for (int i = 0; i < x.Size(); i++)
   {
      r2 += (x(i) - ms_params(i)) * (x(i) - ms_params(i));
   }

   if (sqrt(r2) >= ms_params(x.Size()) && sqrt(r2) <= ms_params(x.Size() + 1))
   {
      return electromagnetics::mu0_ * ms_params(x.Size() + 2);
   }
   return electromagnetics::mu0_;
}

// A sphere with constant conductivity.  The sphere has a radius, center, and
// conductivity specified on the command line and stored in ls_params_.
real_t conductive_sphere(const Vector &x, const Vector &cs_params)
{
   real_t r2 = 0;

   for (int i = 0; i < x.Size(); i++)
   {
      r2 += (x(i) - cs_params(i)) * (x(i) - cs_params(i));
   }

   if (sqrt(r2) <= cs_params(x.Size()))
   {
      return cs_params(x.Size() + 1);
   }
   return 0;
}

void dEdtBCFunc(const Vector &x, real_t t, Vector &dE)
{
   dE.SetSize(3);
   dE = 0_r;
}

void EFieldFunc(const Vector &x, Vector &E)
{
   E.SetSize(3);
   E = 0_r;
}
void BFieldFunc(const Vector &x, Vector &B)
{
   B.SetSize(3);
   B = 0_r;
}

/// assumes fes is an H(curl) space
void CurrentIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                       const Array<int> &markers, Vector &b)
{
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int qorder = 2 * fe.GetOrder();
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);
}

void CurrentIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                               ElementTransformation &Tr,
                                               Vector &elvect)
{
   // TODO
}
