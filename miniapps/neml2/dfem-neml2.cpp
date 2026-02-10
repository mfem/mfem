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
#include <string>

using namespace mfem;
using namespace mfem::future;

// I think we're going to have duplicate nonlinear forms on the fine level but maybe that's fine?
class NEML2Multigrid : public GeometricMultigrid
{
 private:
   std::shared_ptr<neml2::Model> cmodel;
   HypreBoomerAMG *amg;
   std::vector<ParNonlinearForm> pnlfs;
   std::vector<Vector> coarser_solutions;
   const Vector &fine_solution;

 public:
   // Constructs a diffusion multigrid for the ParFiniteElementSpaceHierarchy
   // and the array of essential boundaries
   NEML2Multigrid(ParFiniteElementSpaceHierarchy &fespaces, Array<int> &ess_bdr,
                  std::shared_ptr<neml2::Model> cmodel_,
                  const Vector &fine_solution_)
       : GeometricMultigrid(fespaces, ess_bdr), cmodel(cmodel_),
         fine_solution(fine_solution_)
   {
      const auto num_levels = fespaces.GetNumLevels();
      const auto num_coarser_levels = num_levels - 1;

      // Build all the nonlinear forms first
      pnlfs.reserve(num_levels);
      for (int level = 0; level < fespaces.GetNumLevels(); ++level)
      {
         ConstructNonlinearForm(fespaces.GetFESpaceAtLevel(level),
                                static_cast<bool>(level));
      }
      MFEM_ASSERT(pnlfs.back().Height() == fine_solution.Size(),
                  "The size of the fine level nonlinear form should match the "
                  "size of our current solution");

      // Now we must construct the solution at the different levels
      coarser_solutions.resize(num_coarser_levels);
      if (num_coarser_levels)
      {
         auto create_coarser_solution = [this](const int level,
                                               const Vector &finer_solution)
         {
            auto &coarse_solution = coarser_solutions[level];
            coarse_solution.SetSize(pnlfs[level].Height());
            prolongations[level]->MultTranspose(finer_solution,
                                                coarse_solution);
         };
         create_coarser_solution(num_coarser_levels - 1, fine_solution);
         for (int level = num_coarser_levels - 2; level >= 0; --level)
         {
            create_coarser_solution(level, coarser_solutions[level + 1]);
         }
      }

      ConstructCoarseOperatorAndSolver(fespaces.GetFESpaceAtLevel(0));

      for (int level = 1; level < fespaces.GetNumLevels(); ++level)
      {
         ConstructOperatorAndSmoother(fespaces.GetFESpaceAtLevel(level), level);
      }

      // No longer need the gradient evaluation points
      coarser_solutions.clear();
   }

   ~NEML2Multigrid() override
   {
      delete amg;
   }

 private:
   void ConstructNonlinearForm(ParFiniteElementSpace &fespace,
                               bool partial_assembly)
   {
      auto &form = pnlfs.emplace_back(&fespace);
      if (partial_assembly)
      {
         form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      else
      {
         form.SetAssemblyLevel(AssemblyLevel::FULL);
      }
      form.AddDomainIntegrator(new NEML2StressDivergenceIntegrator(cmodel,
                                                                   0.01)); // hardcoded time
      form.SetEssentialTrueDofs(*essentialTrueDofs[pnlfs.size() - 1]);
      form.Setup();
   }

   void ConstructCoarseOperatorAndSolver(ParFiniteElementSpace &coarse_fespace)
   {
      const auto *const coarse_pnlf = &pnlfs[0];
      const auto &coarse_solution = coarser_solutions.size() ? coarser_solutions[0]
                                                             : fine_solution;
      auto &coarse_operator = coarse_pnlf->GetGradient(coarse_solution);
      auto *const hypreCoarseMat = dynamic_cast<HypreParMatrix *>(&coarse_operator);
      MFEM_ASSERT(hypreCoarseMat,
                  "We should have created a parallel hypre csr matrix");

      amg = new HypreBoomerAMG(*hypreCoarseMat);
      amg->SetPrintLevel(-1);

      CGSolver *pcg = new CGSolver(MPI_COMM_WORLD);
      pcg->SetPrintLevel(-1);
      pcg->SetMaxIter(10);
      pcg->SetRelTol(sqrt(1e-4));
      pcg->SetAbsTol(0.0);
      pcg->SetOperator(*hypreCoarseMat);
      pcg->SetPreconditioner(*amg);

      AddLevel(hypreCoarseMat, pcg, false, true);
   }

   void ConstructOperatorAndSmoother(ParFiniteElementSpace &fespace, int level)
   {
      const auto *const pnlf = &pnlfs[level];
      const auto &level_soln = level == coarser_solutions.size() ? fine_solution
                                                                 : coarser_solutions[level];
      auto &opr = pnlf->GetGradient(level_soln);
      Vector diag(fespace.GetTrueVSize());
      opr.AssembleDiagonal(diag);

      Solver *smoother = new OperatorChebyshevSmoother(opr, diag,
                                                       *essentialTrueDofs[level],
                                                       2,
                                                       fespace.GetParMesh()->GetComm());

      AddLevel(&opr, smoother, false, true);
   }
};

class NEML2MultigridPreconditionerFactory : public PetscPreconditionerFactory
{
 private:
 public:
   NEML2MultigridPreconditionerFactory(ParFiniteElementSpaceHierarchy &fespaces_,
                                       Array<int> &ess_bdr_,
                                       std::shared_ptr<neml2::Model> cmodel_,
                                       const Vector &fine_solution_)
       : PetscPreconditionerFactory(), fespaces(fespaces_), ess_bdr(ess_bdr_),
         cmodel(cmodel_), fine_solution(fine_solution_)
   {
   }

   // Since all the operator construction currently happens in the constructor, let's just rebuild this every time for the moment. That definitely won't be optimal but for a liner constitutive model we only plan to construct this once anyway
   mfem::Solver *NewPreconditioner(const mfem::OperatorHandle &) override
   {
      multigrid = std::make_unique<NEML2Multigrid>(fespaces, ess_bdr, cmodel,
                                                   fine_solution);
      return multigrid.get();
   }

   virtual ~NEML2MultigridPreconditionerFactory() = default;

 private:
   ParFiniteElementSpaceHierarchy &fespaces;
   Array<int> &ess_bdr;
   std::shared_ptr<neml2::Model> cmodel;
   const Vector &fine_solution;
   std::unique_ptr<NEML2Multigrid> multigrid;
};

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE
   Mpi::Init();
   Hypre::Init();
   int myid = Mpi::WorldRank();

   // Parse command-line options
   const char *device_config = "cpu";
   constexpr auto petscrc_file = MFEM_SOURCE_DIR "/miniapps/neml2/"
                                                 "petscopts";
   int geometric_refinements = 0;
   int order_refinements = 1;
   int n = 5;
   std::string neml2_input = "elasticity.i";
   std::string neml2_model = "model";

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&n, "-n", "--n",
                  "The number of elements in one dimension. The total number "
                  "will be a tensor product of this");
   args.AddOption(&geometric_refinements, "-gr", "--geometric-refinements",
                  "Number of geometric refinements done prior to order "
                  "refinements.");
   args.AddOption(&order_refinements, "-or", "--order-refinements",
                  "Number of order refinements. Finest level in the hierarchy "
                  "has order 2^{or}.");
   args.AddOption(&neml2_input, "-i", "--input",
                  "Path to the NEML2 input file.");
   args.AddOption(&neml2_model, "-m", "--model",
                  "Name of the NEML2 model to use.");
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
   auto *const fec = new H1_FECollection(1, /*dim=*/dim);
   auto *const coarse_fe_space = new ParFiniteElementSpace(&pmesh, fec,
                                                           /*vdim=*/dim,
                                                           Ordering::byNODES);
   Array<FiniteElementCollection *> collections;
   collections.Append(fec);
   ParFiniteElementSpaceHierarchy fespaces(&pmesh, coarse_fe_space, false,
                                           true);
   for (int level = 0; level < geometric_refinements; ++level)
   {
      fespaces.AddUniformlyRefinedLevel(dim, Ordering::byNODES);
   }
   for (int level = 0; level < order_refinements; ++level)
   {
      collections.Append(new H1_FECollection((int)std::pow(2, level + 1), dim));
      fespaces.AddOrderRefinedLevel(collections.Last(), dim, Ordering::byNODES);
   }
   auto &finest_fe_space = fespaces.GetFinestFESpace();

   HYPRE_BigInt size = finest_fe_space.GlobalTrueVSize();
   if (myid == 0)
   {
      std::cout << "Number of finite element unknowns: " << size << std::endl;
   }

   // Set up the integration rule
   const auto &ir = IntRules.Get(pmesh.GetTypicalElementGeometry(), 2);

   // The NEML2 constitutive model
   neml2::set_default_dtype(neml2::kFloat64);
   const auto constitutive_model_path = std::string(MFEM_SOURCE_DIR) +
                                        "/miniapps/neml2/" + neml2_input;
   auto cmodel = neml2::load_model(constitutive_model_path, neml2_model);

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
   ParGridFunction u(&finest_fe_space);
   u = 0;
   u.ProjectBdrCoefficient(zero_disp, fixed_bnd);
   u.ProjectBdrCoefficient(prescribed_disp, displaced_bnd);

   // Setup the parallel nonlinear form
   ParNonlinearForm f(&finest_fe_space);
   const bool fully_assemble_jacobian = (geometric_refinements == 0) &&
                                        (order_refinements == 0);
   f.SetAssemblyLevel(fully_assemble_jacobian ? AssemblyLevel::FULL
                                              : AssemblyLevel::PARTIAL);
   f.AddDomainIntegrator(new NEML2StressDivergenceIntegrator(cmodel,
                                                             0.01)); // hard-coded time
   Array<int> ess_tdof_list;
   finest_fe_space.GetEssentialTrueDofs(essential_bnd, ess_tdof_list);
   f.SetEssentialTrueDofs(ess_tdof_list);
   f.Setup();

   // Setup vector to solve for
   Vector &X = u.GetTrueVector();

   // Nonlinear solver
   PetscNonlinearSolver newton(MPI_COMM_WORLD, f);
   newton.SetAbsTol(1e-8);
   newton.SetRelTol(1e-6);
   newton.SetMaxIter(10);
   newton.SetPrintLevel(1);
   if (!fully_assemble_jacobian)
   {
      newton.SetJacobianType(Operator::PETSC_MATSHELL);
   }
   // Use the current state of u as the initial guess
   newton.iterative_mode = true;
   // Indicate that the Jacobian is a shell matrix
   //
   // Set multigrid preconditioner factory. MFEM PETSc wrapper code will try to destroy this during the nonlinear solver destruction so allow them to do so
   auto *const pre_factory = new NEML2MultigridPreconditionerFactory(fespaces,
                                                                     essential_bnd,
                                                                     cmodel, X);
   newton.SetPreconditionerFactory(pre_factory);

   // Solve
   Vector R;
   newton.Mult(R, X);
   u.SetFromTrueVector();

   // Save the solution in parallel using ParaView format
   ParaViewDataCollection dc("dfem-neml2-output", &pmesh);
   dc.RegisterField("disp", &u);
   dc.SetCycle(0);
   dc.Save();

   for (int level = 0; level < collections.Size(); ++level)
   {
      delete collections[level];
   }
   return 0;
}
