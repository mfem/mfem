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
//     --------------------------------------------------------------------
//      Solid mechanics problem using NEML2 to handle constitutive updates
//     --------------------------------------------------------------------
//
// Compile with: make dfem-neml2
//
// Sample runs:  dfem-neml2
//               dfem-neml2 -d cpu
//               dfem-neml2 -d cuda
//               mpirun -np 4 dfem-neml2
//               mpirun -np 4 dfem-neml2 -d cpu
//               mpirun -np 4 dfem-neml2 -d cuda
//
// Description:  This example code demonstrates the use of MFEM to solve the
//               balance of linear momentum equation in 2D under plane strain,
//               small deformation assumptions with the constitutive model
//               provided by NEML2.

#include "operators.hpp"

using namespace mfem;
using namespace mfem::future;

int main(int argc, char *argv[]) {
  // Initialize MPI and HYPRE
  Mpi::Init();
  Hypre::Init();

  // Parse command-line options
  const char *device_config = "cpu";
  int derivative_type = static_cast<int>(op::AUTODIFF);

  OptionsParser args(argc, argv);
  args.AddOption(&device_config, "-d", "--device",
                 "Device configuration string, see Device::Configure().");
  args.AddOption(&derivative_type, "-der", "--derivative-type",
                 "Derivative computation type: 0=AutomaticDifferentiation,"
                 " 1=HandCoded, 2=FiniteDifference");
  args.ParseCheck();

  // Enable hardware devices such as GPUs, and programming models such as CUDA
  Device device(device_config);
  if (Mpi::Root())
    device.Print();

  // Create a 3D mesh on the square domain [0,1]^3
  constexpr int dim = 3;
  Mesh mesh = Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON);

  // Define a parallel mesh
  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  pmesh.SetCurvature(1);

  // Define a parallel finite element space on the parallel mesh
  H1_FECollection fec(1, /*dim=*/dim);
  ParFiniteElementSpace fe_space(&pmesh, &fec, /*vdim=*/dim, Ordering::byNODES);

  // Set up the integration rule
  const auto &ir = IntRules.Get(pmesh.GetTypicalElementGeometry(), 3);

  // The NEML2 constitutive model
  neml2::set_default_dtype(neml2::kFloat64);
  constexpr auto constitutive_model_path =
      MFEM_SOURCE_DIR "/miniapps/neml2/neml2_model.i";
  auto cmodel = neml2::load_model(constitutive_model_path, "elasticity");

  // Send the constitutive model to the appropriate device
  auto options = neml2::TensorOptions().dtype(neml2::kFloat64);
  if (device.Allows(Backend::CUDA))
    options = options.device(neml2::kCUDA);
  else if (device.Allows(Backend::CPU))
    options = options.device(neml2::kCPU);
  else
    MFEM_ABORT("Unsupported device backend for NEML2");
  cmodel->to(options);

#ifdef MFEM_USE_ENZYME
  // The finite element operator for linear momentum balance
  auto op = std::make_unique<op::LinearMomentumBalance<real_t>>(
      fe_space, ir, cmodel, static_cast<op::DerivativeType>(derivative_type));
#else
  // When Enzyme is not available, use the dual type for automatic
  // differentiation
  using mfem::future::dual;
  using dual_t = dual<real_t, real_t>;
  // The finite element operator for linear momentum balance
  auto op = std::make_unique<op::LinearMomentumBalance<dual_t>>(
      fe_space, ir, cmodel, static_cast<op::DerivativeType>(derivative_type));
#endif

  // Essential boundary condition (fixed)
  Array<int> fixed_bnd(pmesh.bdr_attributes.Max());
  fixed_bnd = 0;
  fixed_bnd[2] = 1;
  Vector uz(3);
  VectorConstantCoefficient zero_disp(uz);

  // Essential boundary condition (displaced)
  Array<int> displaced_bnd(pmesh.bdr_attributes.Max());
  displaced_bnd = 0;
  displaced_bnd[4] = 1;
  Vector ug(3);
  ug(0) = 0.001;
  ug(1) = 0.0;
  ug(2) = 0.0;
  VectorConstantCoefficient prescribed_disp(ug);

  // Set the essential boundary conditions on the operator
  op->SetEssBdrConditions(
      {{&fixed_bnd, &zero_disp}, {&displaced_bnd, &prescribed_disp}});

  // Initial condition (also make it satisfy the EBCs)
  ParGridFunction u(&fe_space);
  u = 0;
  u.ProjectBdrCoefficient(zero_disp, fixed_bnd);
  u.ProjectBdrCoefficient(prescribed_disp, displaced_bnd);

  // Linear solver
  CGSolver krylov(MPI_COMM_WORLD);
  krylov.SetAbsTol(0.0);
  krylov.SetRelTol(1e-12);
  krylov.SetMaxIter(200);
  krylov.SetPrintLevel(2);

  // Nonlinear solver
  NewtonSolver newton(MPI_COMM_WORLD);
  newton.SetOperator(*op);
  newton.SetAbsTol(1e-8);
  newton.SetRelTol(1e-6);
  newton.SetMaxIter(10);
  newton.SetSolver(krylov);
  newton.SetPrintLevel(1);

  // Solve
  Vector X0(fe_space.GetTrueVSize());
  Vector X(fe_space.GetTrueVSize());
  fe_space.GetRestrictionMatrix()->Mult(u, X0);
  newton.Mult(X0, X);
  fe_space.GetProlongationMatrix()->Mult(X, u);

  // Save the solution in parallel using ParaView format
  ParaViewDataCollection dc("dfem-neml2-output", &pmesh);
  dc.RegisterField("disp", &u);
  dc.SetCycle(0);
  dc.Save();

  return 0;
}
