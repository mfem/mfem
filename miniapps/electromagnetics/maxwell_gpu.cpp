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
// Sample runs:
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

#include "mfem.hpp"
#include "electromagnetics.hpp"

#include <fstream>
#include <iostream>

using namespace mfem;

// Prints the program's logo to the given output stream
void display_banner(std::ostream & os);

struct AmpereOperator : public TimeDependentOperator
{
   // not owned
   ParMesh *pmesh;
   ParFiniteElementSpace *hcurl_space;
   ParFiniteElementSpace *hdiv_space;
   AmpereOperator(ParMesh &pmesh_, ParFiniteElementSpace &hcurl,
                  ParFiniteElementSpace &hdiv, size_t assembly_type);

   void Mult(const Vector &B, Vector &dEdt) const override;

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;
};

struct FaradayOperator : public Operator
{
   // not owned
   ParMesh *pmesh;
   ParFiniteElementSpace *hcurl_space;
   ParFiniteElementSpace *hdiv_space;

   FaradayOperator(ParMesh &pmesh_, ParFiniteElementSpace &hcurl,
                   ParFiniteElementSpace &hdiv, size_t assembly_type);

   void Mult(const Vector &E, Vector &dBdt) const override;
};

int main(int argc, char *argv[])
{
   using namespace std::literals::string_literals;

   Mpi::Init();
   Hypre::Init();

   if ( Mpi::Root() ) { display_banner(std::cout); }

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
   Vector ds_params_(0); // Center, Radius, and Permittivity
   // of dielectric sphere
   // Permeability Function
   static Vector ms_params_(0); // Center, Inner and Outer Radii, and
   // Permeability of magnetic shell

   // Conductivity Function
   Vector cs_params_(0); // Center, Radius, and Conductivity
   // of conductive sphere

   // Current Density Function
   Vector dp_params_(0); // Axis Start, Axis End, Rod Radius,
   // Total Current of Rod, and Frequency

   // Scale factor between input time units and seconds
   real_t tScale_ = 1e-9; // Input time in nanosecond

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
   args.AddOption(&ds_params_, "-ds", "--dielectric-sphere-params",
                  "Center, Radius, and Permittivity of Dielectric Sphere");
   args.AddOption(&ms_params_, "-ms", "--magnetic-shell-params",
                  "Center, Inner Radius, Outer Radius, and Permeability "
                  "of Magnetic Shell");
   args.AddOption(&cs_params_, "-cs", "--conductive-sphere-params",
                  "Center, Radius, and Conductivity of Conductive Sphere");
   args.AddOption(&dp_params_, "-dp", "--dipole-pulse-params",
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
   if (Mpi::Root()) { device.Print(); }
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
      if (serial_ref_levels > 0) { serial_ref_levels--; }

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

   if (Mpi::Root())
   {
      std::cout << "Number of H(Curl) dofs: "
                << hcurl_space.GlobalTrueVSize() << std::endl;
      std::cout << "Number of H(Div) dofs: "
                << hdiv_space.GlobalTrueVSize() << std::endl;
   }

   // Electric field
   ParGridFunction Efield(&hcurl_space);
   // Maxnetic flux
   ParGridFunction Bflux(&hdiv_space);

   // Use separate operators for Ampere and Faraday equations for each part of
   // the Hamiltonian operators.
   AmpereOperator ampere(pmesh, hcurl_space, hdiv_space, assembly_type);
   FaradayOperator faraday(pmesh, hcurl_space, hdiv_space, assembly_type);

   // Create the ODE solver
   SIAVSolver siaSolver(tOrder);
   siaSolver.Init(faraday, ampere);
   // TODO: initialize visualization
   // TODO: run sim
   return 0;
}

// Print the Maxwell ascii logo to the given ostream
void display_banner(std::ostream & os)
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

AmpereOperator::AmpereOperator(ParMesh &pmesh_, ParFiniteElementSpace &hcurl,
                               ParFiniteElementSpace &hdiv,
                               size_t assembly_type)
   : TimeDependentOperator(hcurl.GetVSize(), hdiv.GetVSize()), pmesh(&pmesh_),
     hcurl_space(&hcurl), hdiv_space(&hdiv)
{
}

void AmpereOperator::Mult(const Vector& B, Vector &dEdt) const
{
   // TODO
}

void AmpereOperator::ImplicitSolve(const real_t dt, const Vector &x, Vector &k)
{
   // TODO
}

FaradayOperator::FaradayOperator(ParMesh &pmesh_, ParFiniteElementSpace &hcurl,
                                 ParFiniteElementSpace &hdiv,
                                 size_t assembly_type)
   : Operator(hdiv.GetVSize(), hcurl.GetVSize()), pmesh(&pmesh_),
     hcurl_space(&hcurl), hdiv_space(&hdiv)
{
}

void FaradayOperator::Mult(const Vector& E, Vector &dBdt) const
{
   // TODO
}
