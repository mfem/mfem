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

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE
   Mpi::Init();
   Hypre::Init();
   int myid = Mpi::WorldRank();

   // Parse command-line options
   const char *device_config = "cpu";
   bool enable_pcamg = false;
   constexpr auto petscrc_file = MFEM_SOURCE_DIR "/miniapps/neml2/"
                                                 "petscopts";
   int n = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&enable_pcamg, "-pcamg", "--pcamg", "-no-pcamg", "--no-pcamg",
                  "Enable AMG as a preconditioner when using automatic "
                  "differentiation.");
   args.AddOption(&n, "-n", "--n",
                  "The number of elements in one dimension. The total number "
                  "will be a tensor product of this");
   args.ParseCheck();

   // Enable hardware devices such as GPUs, and programming models such as CUDA
   Device device(device_config);
   if (Mpi::Root())
   {
      device.Print();
   }

   MFEMInitializePetsc(nullptr, nullptr, petscrc_file, nullptr);

   // Create a 3D mesh on the square domain [0,1]^3
   constexpr int dim = 3;
   Mesh mesh = Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON);

   // Define a parallel mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   pmesh.SetCurvature(1);

   // Define a parallel finite element space on the parallel mesh
   H1_FECollection fec(1, /*dim=*/dim);
   ParFiniteElementSpace fe_space(&pmesh, &fec, /*vdim=*/dim,
                                  Ordering::byNODES);
   if (myid == 0)
   {
      std::cout << "Number of finite element unknowns: "
                << fe_space.GlobalTrueVSize() << std::endl;
   }

   // Set up the integration rule
   const auto &ir = IntRules.Get(pmesh.GetTypicalElementGeometry(), 3);

   // The NEML2 constitutive model
   neml2::set_default_dtype(neml2::kFloat64);
   constexpr auto constitutive_model_path = MFEM_SOURCE_DIR "/miniapps/neml2/"
                                                            "neml2_model.i";
   auto cmodel = neml2::load_model(constitutive_model_path, "elasticity");

   // Send the constitutive model to the appropriate device
   auto options = neml2::TensorOptions().dtype(neml2::kFloat64);
   if (device.Allows(Backend::CUDA))
   {
      options = options.device(neml2::kCUDA);
   }
   else if (device.Allows(Backend::CPU))
   {
      options = options.device(neml2::kCPU);
   }
   else
   {
      MFEM_ABORT("Unsupported device backend for NEML2");
   }
   cmodel->to(options);

   // Essential boundary condition (fixed)
   Array<int> essential_bnd(pmesh.bdr_attributes.Max());
   Array<int> fixed_bnd(pmesh.bdr_attributes.Max());
   fixed_bnd = 0;
   fixed_bnd[2] = 1;
   essential_bnd = 0;
   essential_bnd[2] = 1;
   Vector uz(3);
   uz(0) = 0.;
   uz(1) = 0.;
   uz(2) = 0.;
   VectorConstantCoefficient zero_disp(uz);

   // Essential boundary condition (displaced)
   Array<int> displaced_bnd(pmesh.bdr_attributes.Max());
   displaced_bnd = 0;
   displaced_bnd[4] = 1;
   essential_bnd[4] = 1;
   Vector ug(3);
   ug(0) = 0.001;
   ug(1) = 0.0;
   ug(2) = 0.0;
   VectorConstantCoefficient prescribed_disp(ug);

   // Initial condition (also make it satisfy the EBCs)
   ParGridFunction u(&fe_space);
   u = 0;
   u.ProjectBdrCoefficient(zero_disp, fixed_bnd);
   u.ProjectBdrCoefficient(prescribed_disp, displaced_bnd);

   // Setup the parallel nonlinear form
   ParNonlinearForm f(&fe_space);
   f.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   f.AddDomainIntegrator(new NEML2StressDivergenceIntegrator(cmodel));
   Array<int> ess_tdof_list;
   fe_space.GetEssentialTrueDofs(essential_bnd, ess_tdof_list);
   f.SetEssentialTrueDofs(ess_tdof_list);

   f.Setup();

   // Nonlinear solver
   PetscNonlinearSolver newton(MPI_COMM_WORLD, f);
   newton.SetAbsTol(1e-8);
   newton.SetRelTol(1e-6);
   newton.SetMaxIter(10);
   newton.SetPrintLevel(1);

   // Solve
   Vector R;
   // Use the current state of u as the initial guess
   newton.iterative_mode = true;
   newton.Mult(R, u);

   // Save the solution in parallel using ParaView format
   ParaViewDataCollection dc("dfem-neml2-output", &pmesh);
   dc.RegisterField("disp", &u);
   dc.SetCycle(0);
   dc.Save();

   return 0;
}
