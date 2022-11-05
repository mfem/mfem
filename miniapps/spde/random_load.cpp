// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

// ===========================================================================
//
//        Mini-App: surrogate model for imperfect spde.
//
//  Details: refer to README
//
//  Runs:
//    mpirun -np 4 ./miniapps/spde/main
//
// ===========================================================================

#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include "mfem.hpp"

#include "material_metrics.hpp"
#include "spde_solver.hpp"
#include "transformation.hpp"
#include "util.hpp"
#include "visualizer.hpp"

using namespace std;
using namespace mfem;

enum TopologicalSupport { kParticles, kOctetTruss };

int main(int argc, char *argv[]) {
  // 0. Initialize MPI.
  Mpi::Init(argc, argv);
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  // 1. Parse command-line options.
  const char *mesh_file = "../../data/inline-quad.mesh";
  int order = 1;
  int num_refs = 3;
  int num_parallel_refs = 3;
  double nu = 1.0;
  double l1 = 1;
  double l2 = 1;
  double l3 = 1;
  double e1 = 0;
  double e2 = 0;
  double e3 = 0;
  bool random_seed = true;
  bool compute_boundary_integrals = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&num_refs, "-r", "--refs", "Number of uniform refinements");
  args.AddOption(&num_parallel_refs, "-rp", "--refs-parallel",
                 "Number of uniform refinements");
  args.AddOption(&nu, "-nu", "--nu", "Fractional exponent nu (smoothness)");
  args.AddOption(&l1, "-l1", "--l1",
                 "First component of diagonal core of theta");
  args.AddOption(&l2, "-l2", "--l2",
                 "Second component of diagonal core of theta");
  args.AddOption(&l3, "-l3", "--l3",
                 "Third component of diagonal core of theta");
  args.AddOption(&e1, "-e1", "--e1", "First euler angle for rotation of theta");
  args.AddOption(&e2, "-e2", "--e2",
                 "Second euler angle for rotation of theta");
  args.AddOption(&e3, "-e3", "--e3", "Third euler angle for rotation of theta");
  args.AddOption(&random_seed, "-rs", "--random-seed", "-no-rs",
                 "--no-random-seed", "Enable or disable random seed.");
  args.AddOption(&compute_boundary_integrals, "-cbi",
                 "--compute-boundary-integrals", "-no-cbi",
                 "--no-compute-boundary-integrals",
                 "Enable or disable computation of boundary integrals.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  if (Mpi::Root()) {
    args.PrintOptions(cout);
  }

  // 2. Read the mesh from the given mesh file.
  Mesh orig_mesh(mesh_file, 1, 1);
  int dim = orig_mesh.Dimension();
  std::vector<Vector> translations(1);
  translations[0].SetSize(2);
  translations[0][0] = 1; 
  translations[0][1] = 0; 
  std::vector<int> mapping = orig_mesh.CreatePeriodicVertexMapping(translations);

  Mesh mesh = Mesh::MakePeriodic(orig_mesh,mapping);

  mesh.RemoveInternalBoundaries();

  for (int i = 0; i<mesh.GetNBE(); i++)
  {
    int attr = mesh.GetBdrAttribute(i);
    if (attr == 3) mesh.SetBdrAttribute(i,2);
  }
  mesh.SetAttributes();

  bool is_3d = (dim == 3);

  // 4. Refine the mesh to increase the resolution.
  for (int i = 0; i < num_refs; i++) {
    mesh.UniformRefinement();
  }
  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  ParMesh orig_pmesh(MPI_COMM_WORLD, orig_mesh);
  mesh.Clear();
  orig_mesh.Clear();
  for (int i = 0; i < num_parallel_refs; i++) {
    pmesh.UniformRefinement();
    orig_pmesh.UniformRefinement();
  }

  // 5. Define a finite element space on the mesh.
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fespace(&pmesh, &fec);
  ParFiniteElementSpace orig_fespace(&pmesh, &fec);
  HYPRE_BigInt size = fespace.GlobalTrueVSize();
  HYPRE_BigInt orig_size = orig_fespace.GlobalTrueVSize();
  if (Mpi::Root()) {
    const Array<int> boundary(pmesh.bdr_attributes);
    cout << "Number of finite element unknowns periodic mesh: " << size << "\n";
    cout << "Number of finite element unknowns: " << orig_size << "\n";
    cout << "Boundary attributes: ";
    boundary.Print(cout, 6);
  }

  // ========================================================================
  // III. Generate random imperfections via fractional PDE
  // ========================================================================

  /// III.1 Define the fractional PDE solution
  ParGridFunction u(&fespace);
  ParGridFunction orig_u(&orig_fespace);
  u = 0.0;

  // III.2 Define the boundary conditions.
  spde::Boundary bc;
  // bc.AddHomogeneousBoundaryCondition(1,spde::BoundaryType::kDirichlet);
  bc.AddInhomogeneousDirichletBoundaryCondition(2,1.0);
  bc.AddHomogeneousBoundaryCondition(1,spde::BoundaryType::kDirichlet);
  if (Mpi::Root()) {
    bc.PrintInfo();
    bc.VerifyDefinedBoundaries(pmesh);
  }

  // III.3 Solve the SPDE problem
  spde::SPDESolver solver(nu, bc, &fespace, l1, l2, l3, e1, e2, e3);
  if (random_seed) {
    solver.GenerateRandomField(u);
  } else {
    const int seed = 0;
    solver.GenerateRandomField(u, seed);
  }

  /// III.4 Verify boundary conditions
  if (compute_boundary_integrals) {
    bc.ComputeBoundaryError(u);
  }

  char vishost[] = "localhost";
  int  visport   = 19916;
  
  Array<int> vdofs;
  Vector dofs;
  for (int i = 0; i<4; i++)
  {
    bc.AddInhomogeneousDirichletBoundaryCondition(2,(double(i)));
    solver.GenerateRandomField(u);

    for (int i = 0; i<pmesh.GetNE(); i++)
    {
      fespace.GetElementVDofs(i,vdofs);
      u.GetSubVector(vdofs,dofs);
      orig_fespace.GetElementVDofs(i,vdofs);
      orig_u.SetSubVector(vdofs,dofs);  
    }
    socketstream sol_sock(vishost, visport);
    sol_sock << "parallel " << num_procs << " " << myid << "\n";
    sol_sock.precision(8);
    sol_sock << "solution\n" << pmesh << u << flush;

    socketstream sol1_sock(vishost, visport);
    sol1_sock << "parallel " << num_procs << " " << myid << "\n";
    sol1_sock.precision(8);
    sol1_sock << "solution\n" << pmesh << orig_u << flush;
  }
}
