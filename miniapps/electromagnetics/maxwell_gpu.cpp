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
//     mpirun -np 4 maxwell-gpu -m ../../data/ball-nurbs.mesh -rs 2 -abcs '-1' -dp '-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5'
//
//   Current source in a metal sphere with dielectric and conducting materials:
//     mpirun -np 4 maxwell-gpu -m ../../data/ball-nurbs.mesh -rs 2 -dbcs '-1' -dp '-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5' -cs '0.0 0.0 -0.5 .2 3e6' -ds '0.0 0.0 0.5 .2 10'
//
//   Current source in a metal box:
//     mpirun -np 4 maxwell-gpu -m ../../data/fichera.mesh -rs 3 -ts 0.25 -tf 10 -dbcs '-1' -dp '-0.5 -0.5 0.0 -0.5 -0.5 1.0 0.1 1 .5 1'
//
//   Current source with a mixture of absorbing and reflecting boundaries:
//     mpirun -np 4 maxwell-gpu -m ../../data/fichera.mesh -rs 3 -ts 0.25 -tf 10 -dp '-0.5 -0.5 0.0 -0.5 -0.5 1.0 0.1 1 .5 1' -dbcs '4 8 19 21' -abcs '5 18'
//
//   By default the sources and fields are all zero:
//   * mpirun -np 4 maxwell
//
// clang-format on

#include "mfem.hpp"

#include "fem/kernels.hpp"

#include "../common/mesh_extras.hpp"
#include "electromagnetics.hpp"

#include <fstream>
#include <iostream>
#include <chrono>

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

int SnapTimeStep(real_t tmax, real_t dtmax, real_t &dt);

/// assumes x is 3D
MFEM_HOST_DEVICE void dipole_pulse(const real_t x_, const real_t y_,
                                   const real_t z_, real_t t, real_t *j,
                                   const real_t *dp_params, real_t tscale)
{
   real_t x[3] = {x_, y_, z_};
   constexpr int ndims = 3;
   real_t v[ndims];
   real_t xu[ndims];
   real_t h = 0;
   for (int i = 0; i < ndims; ++i)
   {
      j[i] = 0;
      xu[i] = x[i] - dp_params[i];
   }
   for (int i = 0; i < ndims; ++i)
   {
      v[i] = dp_params[ndims + i] - dp_params[i];
      h += v[i] * v[i];
   }
   if (h == 0)
   {
      return;
   }
   h = sqrt(h);
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
   xp = sqrt(xp);
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
   // not owned
   ParMesh *pmesh;
   const GeometricFactors *geom = nullptr;
   const DofToQuad *maps_o = nullptr;
   const DofToQuad *maps_c = nullptr;
   // parameters state
   const Vector *dp_params;
   real_t tscale = 1;
   real_t t = 0;
   CurrentIntegrator(ParMesh &pm, const Vector &dp, real_t ts,
                     const IntegrationRule *ir = nullptr);

   bool SupportsDevice() const override { return true; }

   void AssembleDevice(const FiniteElementSpace &fes, const Array<int> &markers,
                       Vector &y) override;

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override;
   using LinearFormIntegrator::AssembleRHSElementVect;

   using AssembleKernelType = void (*)(const int ne, const int d1d,
                                       const int q1d, const int *markers,
                                       const real_t *Bo, const real_t *Bc,
                                       const real_t *J, const real_t *W,
                                       const real_t *coords,
                                       const real_t *dp_params, real_t *y,
                                       real_t t, real_t tscale);

   /// call when new dofmap or geometric factors are needed
   void Update()
   {
      geom = nullptr;
      maps_o = nullptr;
      maps_c = nullptr;
   }

   /// parameters: use T_D1D, T_Q1D
   MFEM_REGISTER_KERNELS(AssembleKernels, AssembleKernelType, (int, int));
   struct Kernels
   {
      Kernels();
   };

   template <int D1D, int Q1D> static void AddSpecialization()
   {
      AssembleKernels::Specialization<D1D, Q1D>::Add();
   }
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

   struct AmperePA : public Operator
   {
      AmpereAction *action;
      AmperePA(AmpereAction &a)
         : Operator(a.hcurl_space->GetVSize()), action(&a)
      {}

      void Mult(const Vector &x, Vector &y) const override;

      /// Get the parallel finite element space prolongation matrix
      const Operator *GetProlongation() const override
      {
         return action->hcurl_space->GetProlongationMatrix();
      }
      /// Get the transpose of GetRestriction, useful for matrix-free RAP
      virtual const Operator *GetRestrictionTranspose() const
      {
         return action->hcurl_space->GetRestrictionTransposeOperator();
      }
      /// Get the parallel finite element space restriction matrix
      const Operator *GetRestriction() const override
      {
         return action->hcurl_space->GetRestrictionOperator();
      }
   };

   std::unique_ptr<AmperePA> ampere_pa;
   OperatorHandle linsys;

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

   // ldofs
   mutable ParGridFunction rhs;
   // tdofs
   mutable Vector rhs_tdofs;
   // tdofs
   mutable Vector X;
   // ldofs
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
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   if (Mpi::Root())
   {
      display_banner(std::cout);
   }

   // Parse command-line options.
   // const char *mesh_file = "../../data/ball-nurbs.mesh";
   const char *mesh_file = "../../data/periodic-cube.mesh";
   int sOrder = 1;
   int tOrder = 1;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   int visport = 19916;
   bool visualization = true;
   bool visit = true;
   real_t dt = 1.0e-12;
   real_t dtsf = 0.95;
   real_t ti = 0;
   real_t ts = 1;
   real_t tf = 40;

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
   ti *= tscale;
   tf *= tscale;
   ts *= tscale;

   MFEM_VERIFY(ds_params.Size() == 0 || ds_params.Size() == 5, "");
   MFEM_VERIFY(ms_params.Size() == 0 || ms_params.Size() == 6, "");
   MFEM_VERIFY(cs_params.Size() == 0 || cs_params.Size() == 5, "");
   MFEM_VERIFY(dp_params.Size() == 0 || dp_params.Size() == 10, "");

   Device device(device_config);
   if (Mpi::Root())
   {
      device.Print();
   }
   KernelReporter::Enable();
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
      B_gf.ProjectCoefficient(b0);
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

   // TODO: compute dtmax
   real_t dtmax = 0.03e-9;
   int nsteps = SnapTimeStep(tf - ti, dtsf * dtmax, dt);
   if ( Mpi::Root() )
   {
      std::cout << "Number of Time Steps:  " << nsteps << std::endl;
      std::cout << "Time Step Size:        " << dt / tscale << "ns"
                << std::endl;
   }

   // Create the ODE solver
   SIAVSolver siaSolver(tOrder);
   siaSolver.Init(faraday, ampere);
   // TODO: initialize visualization

   socketstream E_sock, B_sock;

   if (visualization)
   {
      char vishost[] = "localhost";
      E_sock.open(vishost, visport);
      B_sock.open(vishost, visport);
      if (!E_sock || !B_sock)
      {
         visualization = false;
         if (Mpi::Root())
         {
            std::cout << "GLVis visualization disabled" << std::endl;
         }
      }
      else
      {
         E_sock << "parallel " << num_procs << " " << myid << "\n";
         E_sock.precision(8);
         E_sock << "solution\n"
                << pmesh << E_gf << std::flush;
         E_sock << " window_title 'Electric Field (E)'";
         E_sock << " window_geometry 0 0 1024 768" << " keys cm" << std::endl;

         B_sock << "parallel " << num_procs << " " << myid << "\n";
         B_sock.precision(8);
         B_sock << "solution\n"
                << pmesh << B_gf << std::flush;
         B_sock << " window_title 'Magnetic Flux Density (B)'";
         B_sock << " window_geometry 0 0 1024 768" << " keys cm" << std::endl;
      }
   }
   real_t t = ti;
   std::chrono::high_resolution_clock timer;
   int it = 1;
   while (t < tf)
   {
      if (myid == 0)
      {
         std::cout << "t = " << t << std::endl;
      }
      auto start = timer.now();
      siaSolver.Run(B_gf, E_gf, t, dt, std::max(t + dt, ti + ts * it));
      auto stop = timer.now();

      if (Mpi::Root())
      {
         std::cout << "walltime: "
                   << std::chrono::duration_cast<std::chrono::duration<double>>(
                      stop - start)
                   .count()
                   << std::endl;
      }

      if (visualization)
      {
         E_sock << "parallel " << num_procs << " " << myid << "\n";
         E_sock << "solution\n" << pmesh << E_gf << std::flush;
         B_sock << "parallel " << num_procs << " " << myid << "\n";
         B_sock << "solution\n" << pmesh << B_gf << std::endl;
      }
      ++it;
   }
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
   : Operator(hcurl.GetTrueVSize()), pmesh(&pmesh_), hcurl_space(&hcurl),
     hdiv_space(&hdiv), hcurl_mass(&hcurl), weak_curl(&hdiv, &hcurl),
     sigma_coeff(sigma), dEdtbc_coeff(dEdtbc),
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
   // TODO: full assembly
   ampere_pa.reset(new AmperePA(*this));
   Operator *oper;
   ampere_pa->FormSystemOperator(dbc_dofs, oper);
   linsys.Reset(oper);
}

AmpereOperator::AmpereOperator(
   ParMesh &pmesh_, ParFiniteElementSpace &hcurl, ParFiniteElementSpace &hdiv,
   size_t assembly_type, Coefficient &eps_coeff, Coefficient &inv_mu_coeff,
   Coefficient *sigma, VectorCoefficient *dEdtbc, Array<int> &abcs,
   Array<int> &dbcs, CurrentIntegrator *current_integrator, ParGridFunction &E)
   : TimeDependentOperator(hcurl.GetVSize(), hdiv.GetVSize()),
     action(pmesh_, hcurl, hdiv, assembly_type, eps_coeff, inv_mu_coeff, sigma,
            dEdtbc, abcs, dbcs, current_integrator),
     solver(hcurl.GetParMesh()->GetComm()), rhs(&hcurl),
     rhs_tdofs(hcurl.GetTrueVSize()), X(hcurl.GetTrueVSize()), E_gf(&E)
{
   rhs.UseDevice(true);
   rhs_tdofs.UseDevice(true);
   X.UseDevice(true);
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
   solver.SetRelTol(1e-12);
   solver.SetMaxIter(300);
   // solver.SetPrintLevel(1);
}

void AmpereAction::Mult(const Vector &x, Vector &y) const
{
   linsys->Mult(x, y);
   // bcs
   if (dbc_dofs.Size() > 0)
   {
      y.SetSubVector(dbc_dofs, 0_r);
   }
}

void AmpereAction::AmperePA::Mult(const Vector &x, Vector &y) const
{
   // TODO: full assembly version
   action->hcurl_mass.Mult(x, y);
   if (action->loss_term)
   {
      action->loss_term->AddMult(x, y, 0.5_r * action->dt);
   }
}

void AmpereAction::EliminateRHS(Vector &b) const
{
   // for partial assembly
   if (dbc_dofs.Size() > 0)
   {
      b.SetSubVector(dbc_dofs, 0_r);
   }
}

void AmpereOperator::Mult(const Vector &B, Vector &dEdt) const
{
   const_cast<AmpereOperator *>(this)->ImplicitSolve(0, B, dEdt);
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
   const Operator *P = action.hcurl_space->GetProlongationMatrix();
   if (P)
   {
      P->MultTranspose(rhs, rhs_tdofs);
   }
   else
   {
      // TODO: can just set reference
      rhs_tdofs = rhs;
   }

   if (action.dEdtbc_coeff)
   {
      action.dEdtbc_coeff->SetTime(t);
      // TODO: projecting bdr coefficient happens on the CPU
      dedt->ProjectBdrCoefficientTangent(*action.dEdtbc_coeff,
                                         action.dbc_marker);
   }
   // dirichlet bc's handled directly by action
   action.EliminateRHS(rhs_tdofs);
   action.hcurl_space->GetRestrictionOperator()->Mult(*dedt, X);
   solver.Mult(rhs_tdofs, X);
   MFEM_VERIFY(solver.GetConverged(), "");
   if (P)
   {
      P->Mult(X, dEdt);
   }
   else
   {
      dEdt = X;
   }
}

/// TODO: for now this always does full assembly into a hypre matrix.
/// CurlInterpolator from H(curl) to H(div) doesn't support partial assembly
/// yet.
// #define FARADAY_HAS_PA

FaradayOperator::FaradayOperator(ParMesh &pmesh_, ParFiniteElementSpace &hcurl,
                                 ParFiniteElementSpace &hdiv,
                                 size_t assembly_type)
   : Operator(hdiv.GetVSize(), hcurl.GetVSize()), pmesh(&pmesh_),
     hcurl_space(&hcurl), hdiv_space(&hdiv), curl_op(&hcurl, &hdiv)
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
   E(0) = 0;
   E(1) = 0;
   E(2) = 0;
}
void BFieldFunc(const Vector &x, Vector &B)
{
   B.SetSize(3);
   B(0) = 0;
   B(1) = 0;
   B(2) = 0;
}

CurrentIntegrator::Kernels::Kernels()
{
   CurrentIntegrator::AddSpecialization<1, 1>();
   CurrentIntegrator::AddSpecialization<1, 2>();
   CurrentIntegrator::AddSpecialization<2, 2>();
   CurrentIntegrator::AddSpecialization<2, 3>();
   CurrentIntegrator::AddSpecialization<3, 3>();
   CurrentIntegrator::AddSpecialization<3, 4>();
   CurrentIntegrator::AddSpecialization<4, 4>();
   CurrentIntegrator::AddSpecialization<4, 5>();
}

CurrentIntegrator::CurrentIntegrator(ParMesh &pm, const Vector &dp, real_t ts,
                                     const IntegrationRule *ir)
   : LinearFormIntegrator(ir), pmesh(&pm), dp_params(&dp), tscale(ts)
{
   static Kernels kernels{};
}

template <int T_D1D, int T_Q1D>
void CurrentIntegratorKernel(const int ne, const int d, const int q,
                             const int *markers, const real_t *bo,
                             const real_t *bc, const real_t *j_,
                             const real_t *weights, const real_t *coords,
                             const real_t *dp_params, real_t *Y, real_t t,
                             real_t tscale)
{
   {
      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;
      MFEM_VERIFY(q <= Q, "");
      MFEM_VERIFY(d <= D, "");
      MFEM_VERIFY(d <= q, "");
   }
   constexpr int vdim = 3;
   const auto BO = Reshape(bo, q, d - 1);
   const auto BC = Reshape(bc, q, d);
   const auto J = Reshape(j_, q, q, q, vdim, vdim, ne);
   const auto W = Reshape(weights, q, q, q);
   const auto C = Reshape(coords, q, q, q, vdim, ne);

   // TODO: batching

   mfem::forall_3D(ne, q, q, vdim, [=] MFEM_HOST_DEVICE(int e)
   {
      if (markers[e] == 0)
      {
         // ignore
         return;
      }

      constexpr int vdim = 3;
      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::HCURL_MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::HCURL_MAX_D1D;

      // D-1 could be zero, dummy have space for one
      MFEM_SHARED real_t sBot[D > 1 ? Q * (D - 1) : 1];
      MFEM_SHARED real_t sBct[Q * D];

      // Bo and Bc into shared memory
      const DeviceMatrix Bot(sBot, d - 1, q);
      // nvcc can't first-capture in an if constexpr
      auto Bo = BO;
      if constexpr (D > 1)
      {
         kernels::internal::LoadB<D - 1, Q>(d - 1, q, Bo, sBot);
      }
      const DeviceMatrix Bct(sBct, d, q);
      kernels::internal::LoadB<D, Q>(d, q, BC, sBct);

      MFEM_SHARED real_t sm0[vdim * Q * Q * Q];
      MFEM_SHARED real_t sm1[vdim * Q * Q * Q];
      DeviceTensor<4> QQQ(sm1, q, q, q, vdim);
      DeviceTensor<4> DQQ(sm0, d, q, q, vdim);
      DeviceTensor<4> DDQ(sm1, d, d, q, vdim);

      MFEM_FOREACH_THREAD(vd, z, vdim)
      {
         MFEM_FOREACH_THREAD(y, y, q)
         {
            MFEM_FOREACH_THREAD(x, x, q)
            {
               for (int z = 0; z < q; ++z)
               {
                  real_t curr[3];
                  dipole_pulse(C(x, y, z, 0, e), C(x, y, z, 1, e),
                               C(x, y, z, 2, e), t, curr, dp_params, tscale);

                  const real_t J11 = J(x, y, z, 0, 0, e);
                  const real_t J21 = J(x, y, z, 1, 0, e);
                  const real_t J31 = J(x, y, z, 2, 0, e);
                  const real_t J12 = J(x, y, z, 0, 1, e);
                  const real_t J22 = J(x, y, z, 1, 1, e);
                  const real_t J32 = J(x, y, z, 2, 1, e);
                  const real_t J13 = J(x, y, z, 0, 2, e);
                  const real_t J23 = J(x, y, z, 1, 2, e);
                  const real_t J33 = J(x, y, z, 2, 2, e);
                  // adj(J)
                  const real_t A11 = (J22 * J33) - (J23 * J32);
                  const real_t A12 = (J32 * J13) - (J12 * J33);
                  const real_t A13 = (J12 * J23) - (J22 * J13);
                  const real_t A21 = (J31 * J23) - (J21 * J33);
                  const real_t A22 = (J11 * J33) - (J13 * J31);
                  const real_t A23 = (J21 * J13) - (J11 * J23);
                  const real_t A31 = (J21 * J32) - (J31 * J22);
                  const real_t A32 = (J31 * J12) - (J11 * J32);
                  const real_t A33 = (J11 * J22) - (J12 * J21);
                  const real_t A[9] = {A11, A12, A13, A21, A22,
                                       A23, A31, A32, A33
                                      };
                  QQQ(x, y, z, vd) = W(x, y, z) * (A[vd * vdim] * curr[0] +
                                                   A[vd * vdim + 1] * curr[1] +
                                                   A[vd * vdim + 2] * curr[2]);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply Bt operator
      MFEM_FOREACH_THREAD(vd, z, vdim)
      {
         const int nx = (vd == 0) ? d - 1 : d;
         DeviceMatrix Btx = (vd == 0) ? Bot : Bct;
         MFEM_FOREACH_THREAD(qy, y, q)
         {
            MFEM_FOREACH_THREAD(dx, x, nx)
            {
               real_t u[Q];
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz)
               {
                  u[qz] = 0.0;
               }
               MFEM_UNROLL(Q)
               for (int qx = 0; qx < q; ++qx)
               {
                  MFEM_UNROLL(Q)
                  for (int qz = 0; qz < q; ++qz)
                  {
                     u[qz] += QQQ(qx, qy, qz, vd) * Btx(dx, qx);
                  }
               }
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz)
               {
                  DQQ(dx, qy, qz, vd) = u[qz];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd, z, vdim)
      {
         const int nx = (vd == 0) ? d - 1 : d;
         const int ny = (vd == 1) ? d - 1 : d;
         DeviceMatrix Bty = (vd == 1) ? Bot : Bct;
         MFEM_FOREACH_THREAD(dy, y, ny)
         {
            MFEM_FOREACH_THREAD(dx, x, nx)
            {
               real_t u[Q];
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz)
               {
                  u[qz] = 0.0;
               }
               MFEM_UNROLL(Q)
               for (int qy = 0; qy < q; ++qy)
               {
                  MFEM_UNROLL(Q)
                  for (int qz = 0; qz < q; ++qz)
                  {
                     u[qz] += DQQ(dx, qy, qz, vd) * Bty(dy, qy);
                  }
               }
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz)
               {
                  DDQ(dx, dy, qz, vd) = u[qz];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(vd, z, vdim)
      {
         const int nx = (vd == 0) ? d - 1 : d;
         const int ny = (vd == 1) ? d - 1 : d;
         const int nz = (vd == 2) ? d - 1 : d;
         DeviceTensor<5> Yxyz(Y, nx, ny, nz, vdim, ne);
         DeviceMatrix Btz = (vd == 2) ? Bot : Bct;
         MFEM_FOREACH_THREAD(dy, y, ny)
         {
            MFEM_FOREACH_THREAD(dx, x, nx)
            {
               real_t u[D];
               MFEM_UNROLL(D)
               for (int dz = 0; dz < nz; ++dz)
               {
                  u[dz] = 0.0;
               }
               MFEM_UNROLL(Q)
               for (int qz = 0; qz < q; ++qz)
               {
                  MFEM_UNROLL(D)
                  for (int dz = 0; dz < nz; ++dz)
                  {
                     u[dz] += DDQ(dx, dy, qz, vd) * Btz(dz, qz);
                  }
               }
               MFEM_UNROLL(D)
               for (int dz = 0; dz < nz; ++dz)
               {
                  Yxyz(dx, dy, dz, vd, e) += u[dz];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

template <int T_D1D, int T_Q1D>
CurrentIntegrator::AssembleKernelType
CurrentIntegrator::AssembleKernels::Kernel()
{
   return CurrentIntegratorKernel<T_D1D, T_Q1D>;
}

CurrentIntegrator::AssembleKernelType
CurrentIntegrator::AssembleKernels::Fallback(int d1d, int q1d)
{
   return CurrentIntegratorKernel<0, 0>;
}

/// assumes fes is an H(curl) space
void CurrentIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                       const Array<int> &markers, Vector &y)
{
   const FiniteElement *fe = fes.GetTypicalFE();
   const VectorTensorFiniteElement *vfe =
      dynamic_cast<const VectorTensorFiniteElement *>(fe);
   MFEM_VERIFY(vfe != nullptr, "Must be a VectorTensorFiniteElement");
   const int qorder = 2 * fe->GetOrder();
   const Geometry::Type gtype = vfe->GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);
   const MemoryType mt = Device::GetDeviceMemoryType();
   if (!geom)
   {
      geom = pmesh->GetGeometricFactors(
                *ir, GeometricFactors::COORDINATES | GeometricFactors::JACOBIANS, mt);
   }
   if (!maps_o)
   {
      maps_o = &vfe->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   }
   if (!maps_c)
   {
      maps_c = &vfe->GetDofToQuad(*ir, DofToQuad::TENSOR);
   }
   const int d = maps_c->ndof, q = maps_c->nqpt;
   const int ne = fes.GetMesh()->GetNE();

   const real_t *Bo = maps_o->B.Read();
   const real_t *Bc = maps_c->B.Read();
   const int *M = markers.Read();
   const real_t *coords = geom->X.Read();
   const real_t *J = geom->J.Read();
   const real_t *W = ir->GetWeights().Read();
   real_t *Y = y.ReadWrite();
   AssembleKernels::Run(d, q, ne, d, q, M, Bo, Bc, J, W, coords,
                        dp_params->Read(), Y, t, tscale);
}

void CurrentIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                               ElementTransformation &Tr,
                                               Vector &elvect)
{
   // TODO
   MFEM_ABORT("Not implemented yet");
}

int
SnapTimeStep(real_t tmax, real_t dtmax, real_t & dt)
{
   real_t dsteps = tmax/dtmax;

   int nsteps = static_cast<int>(pow(10,(int)ceil(log10(dsteps))));

   for (int i=1; i<=5; i++)
   {
      int a = (int)ceil(log10(dsteps/pow(5.0,i)));
      int nstepsi = (int)pow(5,i)*max(1,(int)pow(10,a));

      nsteps = min(nsteps,nstepsi);
   }

   dt = tmax / nsteps;

   return nsteps;
}
