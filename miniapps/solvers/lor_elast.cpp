// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
//                --------------------------------------------
//                Elasticity LOR Block Preconditioning Miniapp
//                --------------------------------------------
//
// Description:
//    The purpose of this miniapp is to demonstrate how to effectively
//    precondition vector valued PDEs, such as elasticity, on GPUs.
//
//    Using 3D elasticity as an example, the linear operator can be broken into
//    vector components and has the form:
//
//    A = [A_00  A_01  A_02]
//        [A_10  A_11  A_12]
//        [A_20  A_21  A_22]
//
//    Traditional AMG requires having the entire matrix in memory which can be
//    prohibitive on GPUs. An effective preconditioning strategy for materials
//    which are not nearly incompressible is to use
//    P^{-1} = diag(AMG(A_00), AMG(A_11), AMG(A_22)) where AMG(A) is the AMG
//    approximation inv(A) [1]. This requires storing 3 blocks instead of 9,
//    but this is still prohibitive for high order discretizations on GPUs. This
//    is alleviated here by performing AMG on the low-order refined
//    operators instead.
//
//    This miniapp solves the same beam problem described in Example 2. Run
//    times of the new solver (partial assembly with block diagonal LOR-AMG)
//    can be compared with the LEGACY approach of assembling the full matrix
//    and preconditioning with AMG.
//
//    For the partial assembly approach, the operator actions and component
//    matrix assembly are supported on GPUs.
//
//    The LEGACY approach should be performed with "-vdim" ordering while PARTIAL
//    requires "-nodes".
//
//    There is also an option "-ss" or "--sub-solve" for the partial assembly
//    version which replaces P^{-1} with diag(inv(A_00), inv(A_11), inv(A_22))
//    where the action of inv(A_ii) is performed with a CG inner solve that
//    is preconditioned with AMG(A_ii). This seems to give order independent
//    conditioning of the outer CG solve, but is much slower than performing
//    a single AMG iteration per block.
//
//    This miniapp supports beam-tri.mesh, beam-quad.mesh, and beam-hex.mesh.
//    beam-tet.mesh can be run if MFEM is build with
//    ElementRestriction::MaxNbNbr set to 32 instead of 16.
//
//    This miniapp shows how to test if the derived component integrators are
//    correct using BlockFESpaceOperator. If "-ca" (for componentwise action)
//    is used, a block operator where each block is a component of the
//    elasticity operator is used for A rather than the vector version.
//    This yields the same answer, but is less efficient. "-ca" can be called
//    with "-pa" for a version where each component is partially assembled, or
//    without where each component is called with full assembly, although the
//    latter may only work for order 1 on GPUs.
//
// Sample runs:
//
//       ./lor_elast -m ../../data/beam-tri.mesh
//       ./lor_elast -m ../../data/beam-quad.mesh
//       ./lor_elast -m ../../data/beam-hex.mesh
//       mpirun -np 4 ./lor_elast -m ../../data/beam-hex.mesh -l 5 -vdim
//       mpirun -np 4 ./lor_elast -m ../../data/beam-hex.mesh -l 5 -vdim -elast
//       ./lor_elast --device cuda -m ../../data/beam-hex.mesh -l 4 -o 2 -pa
//       ./lor_elast --device cuda -m ../../data/beam-hex.mesh -l 4 -o 2 -pa -pv
//       ./lor_elast --device cuda -m ../../data/beam-hex.mesh -l 4 -o 2 -pa -ss
//       ./lor_elast --device cuda -m ../../data/beam-hex.mesh -l 4 -o 2 -pa -ca
//       ./lor_elast --device cuda -m ../../data/beam-hex.mesh -l 5 -ca
//
// References:
//    [1] Mihajlović, M.D. and Mijalković, S., "A component decomposition
//        preconditioning for 3D stress analysis problems", Numerical Linear
//        Algebra with Applications, 2002.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "block_fespace_operator.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/beam-tri.mesh";
   int order = 1;
   bool pa = false;
   bool visualization = false;
   bool amg_elast = 0;
   bool reorder_space = true;
   const char *device_config = "cpu";
   int ref_levels = 0;
   bool sub_solve = false;
   bool componentwise_action = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&amg_elast, "-elast", "--amg-for-elasticity", "-sys",
                  "--amg-for-systems",
                  "Use the special AMG elasticity solver (GM/LN approaches), "
                  "or standard AMG for systems (unknown approach).");
   args.AddOption(&sub_solve, "-ss", "--sub-solve", "-no-ss",
                  "--no-sub-solve",
                  "Blocks are solved with a few CG iterations instead of a single AMG application.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&reorder_space, "-nodes", "--by-nodes", "-vdim", "--by-vdim",
                  "Use byNODES ordering of vector space instead of byVDIM");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&ref_levels, "-l","--reflevels",
                  "How many mesh refinements to perform.");
   args.AddOption(&componentwise_action, "-ca", "--component-action", "-no-ca",
                  "--no-component-action",
                  "Uses partial assembly with a block operator of components instead of the monolithic vector integrator.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable ParaView and GLVis output.");
   args.ParseCheck();

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }
   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   if (mesh.attributes.Max() < 2 || mesh.bdr_attributes.Max() < 2)
   {
      if (Mpi::Root())
      {
         cerr << "\nInput mesh should have at least two materials and "
              << "two boundary attributes! (See schematic in ex2.cpp)\n"
              << endl;
      }
      return 3;
   }

   // 5. Refine the serial mesh on all processors to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   // 7. Define a parallel finite element spaces on the parallel mesh. Here we
   //    use vector finite elements, i.e. dim copies of a scalar finite element
   //    space. If using partial assembly, also assemble the low order refined
   //    (LOR) fespace.
   H1_FECollection fec(order, dim);
   const Ordering::Type fes_ordering =
      reorder_space ? Ordering::byNODES : Ordering::byVDIM;
   ParFiniteElementSpace fespace(&pmesh, &fec, dim, fes_ordering);
   ParFiniteElementSpace scalar_fespace(&pmesh, &fec, 1, fes_ordering);
   unique_ptr<ParLORDiscretization> lor_disc;
   unique_ptr<ParFiniteElementSpace> scalar_lor_fespace;
   if (pa || componentwise_action)
   {
      lor_disc.reset(new ParLORDiscretization(fespace));
      ParFiniteElementSpace &lor_space = lor_disc->GetParFESpace();
      const FiniteElementCollection &lor_fec = *lor_space.FEColl();
      ParMesh &lor_mesh = *lor_space.GetParMesh();
      scalar_lor_fespace.reset(
         new ParFiniteElementSpace(&lor_mesh, &lor_fec, 1, fes_ordering));
   }
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of finite element unknowns: " << size << endl
           << "Assembling: " << flush;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined by
   //    marking only boundary attribute 1 from the mesh as essential and
   //    converting it to a list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system. In this case, b_i equals the
   //    boundary integral of f*phi_i where f represents a "pull down" force on
   //    the Neumann part of the boundary and phi_i are the basis functions in
   //    the finite element fespace. The force is defined by the object f, which
   //    is a vector of Coefficient objects. The fact that f is non-zero on
   //    boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component.
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(pmesh.bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   ParLinearForm b(&fespace);
   b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   if (Mpi::Root())
   {
      cout << "r.h.s. ... " << flush;
   }
   b.Assemble();

   // 10. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator with piece-wise
   //     constants coefficient lambda and mu.
   Vector lambda(pmesh.attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(pmesh.attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);
   ElasticityIntegrator integrator(lambda_func, mu_func);

   ParBilinearForm a(&fespace);
   if (pa || componentwise_action)
   {
      a.SetAssemblyLevel(
         AssemblyLevel::PARTIAL);
   }
   a.AddDomainIntegrator(&integrator);
   a.UseExternalIntegrators();

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (Mpi::Root()) { cout << "matrix ... " << flush; }
   StopWatch total_timer;
   StopWatch assembly_timer;
   assembly_timer.Start();
   total_timer.Start();
   a.Assemble();
   OperatorPtr A;
   Vector B, X;
   Operator *a_lhs = componentwise_action ? nullptr : &a;
   if (!componentwise_action)
   {
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   }
   if (Mpi::Root())
   {
      cout << "done." << endl;
      cout << "Size of linear system: " << fespace.GlobalTrueVSize() << endl;
   }

   Array<int> block_offsets(dim + 1);
   block_offsets[0] = 0;
   BlockDiagonalPreconditioner *blockDiag;
   unique_ptr<Solver> prec = nullptr;

   // 13. For partial assembly, assemble forms on LOR space. Construct the block
   //     diagonal preconditioner by fully assembling the component bilinear
   //     on the LOR space. If additionally "-ss" is enabled, create the
   //     block CG solvers and the high order, partially assembled components.
   vector<unique_ptr<ParBilinearForm>> bilinear_forms;
   vector<unique_ptr<HypreParMatrix>> lor_block;
   // amg_blocks stores preconditioners of lor_block.
   vector<unique_ptr<HypreBoomerAMG>> amg_blocks;
   // cg_blocks only gets used if -ss is enabled.
   vector<unique_ptr<CGSolver>> cg_blocks;
   // diag_ho only used if -hoa enabled. The high order partial assembled operators
   // with the essential dofs eliminated and constrained to one.
   vector<unique_ptr<ParBilinearForm>> ho_bilinear_form_blocks;
   vector<unique_ptr<ConstrainedOperator>> diag_ho;
   // If -ca is used, component bilinear forms are stored in pa_components, and
   // pointers to fespaces.
   vector<unique_ptr<ParBilinearForm>> pa_components;
   vector<const FiniteElementSpace*> fespaces;
   // get block essential boundary info.
   // need to allocate here since constrained operator will not own essential dofs.
   Array<int> ess_tdof_list_block_ho, ess_bdr_block_ho(pmesh.bdr_attributes.Max());
   ess_bdr_block_ho = 0;
   ess_bdr_block_ho[0] = 1;
   ElasticityIntegrator lor_integrator(lambda_func, mu_func);
   if (pa || componentwise_action)
   {
      // 13(a) Create the diagonal LOR matrices and corresponding AMG preconditioners.
      lor_integrator.AssemblePA(lor_disc->GetParFESpace());
      for (int j = 0; j < dim; j++)
      {
         ElasticityComponentIntegrator *block = new ElasticityComponentIntegrator(
            lor_integrator, j, j);
         // create the LOR matrix and corresponding AMG preconditioners.
         bilinear_forms.emplace_back(new ParBilinearForm(scalar_lor_fespace.get()));
         bilinear_forms[j]->SetAssemblyLevel(AssemblyLevel::FULL);
         bilinear_forms[j]->EnableSparseMatrixSorting(Device::IsEnabled());
         bilinear_forms[j]->AddDomainIntegrator(block);
         bilinear_forms[j]->Assemble();

         // get block essential boundary info
         Array<int> ess_tdof_list_block, ess_bdr_block(pmesh.bdr_attributes.Max());
         ess_bdr_block = 0;
         ess_bdr_block[0] = 1;
         scalar_lor_fespace->GetEssentialTrueDofs(ess_bdr_block, ess_tdof_list_block);
         lor_block.emplace_back(bilinear_forms[j]->ParallelAssemble());
         lor_block[j]->EliminateBC(ess_tdof_list_block,
                                   Operator::DiagonalPolicy::DIAG_ONE);
         amg_blocks.emplace_back(new HypreBoomerAMG);
         amg_blocks[j]->SetStrengthThresh(0.25);
         amg_blocks[j]->SetRelaxType(16);  // Chebyshev
         amg_blocks[j]->SetOperator(*lor_block[j]);
         block_offsets[j+1] = amg_blocks[j]->Height();
         // 13(b) If needed, create the block components for operator action.
         if (componentwise_action)
         {
            for (int i = 0; i < dim; i++)
            {
               ElasticityComponentIntegrator *action_block = new ElasticityComponentIntegrator(
                  integrator, i, j);
               if (i == j)
               {
                  fespaces.emplace_back(&scalar_fespace);
               }
               pa_components.emplace_back(new ParBilinearForm(&scalar_fespace));
               pa_components[i + dim*j]->SetAssemblyLevel(pa ? AssemblyLevel::PARTIAL :
                                                          AssemblyLevel::FULL);
               pa_components[i + dim*j]->EnableSparseMatrixSorting(Device::IsEnabled());
               pa_components[i + dim*j]->AddDomainIntegrator(action_block);
               pa_components[i + dim*j]->Assemble();
            }
         }
      }
      block_offsets.PartialSum();
      // 13(c) If needed, create CG solvers for diagonal sub-systems.
      if (sub_solve)
      {
         // create diagonal high order partial assembly operators
         for (int i = 0; i < dim; i++)
         {
            ElasticityComponentIntegrator *block = new ElasticityComponentIntegrator(
               integrator, i, i);
            scalar_fespace.GetEssentialTrueDofs(ess_bdr_block_ho, ess_tdof_list_block_ho);
            ho_bilinear_form_blocks.emplace_back(new ParBilinearForm(&scalar_fespace));
            ho_bilinear_form_blocks[i]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            ho_bilinear_form_blocks[i]->AddDomainIntegrator(block);
            ho_bilinear_form_blocks[i]->Assemble();
            const auto *prolong = scalar_fespace.GetProlongationMatrix();
            auto *rap = new RAPOperator(*prolong, *ho_bilinear_form_blocks[i], *prolong);
            diag_ho.emplace_back(new ConstrainedOperator(rap, ess_tdof_list_block_ho, true,
                                                         Operator::DiagonalPolicy::DIAG_ONE));
         }
         // create CG solvers
         for (int i = 0; i < dim; i++)
         {
            cg_blocks.emplace_back(new CGSolver(MPI_COMM_WORLD));
            cg_blocks[i]->iterative_mode = false;
            cg_blocks[i]->SetOperator(*diag_ho[i]);
            cg_blocks[i]->SetPreconditioner(*amg_blocks[i]);
            cg_blocks[i]->SetMaxIter(30);
            cg_blocks[i]->SetRelTol(1e-8);
         }
      }
      blockDiag = new BlockDiagonalPreconditioner(block_offsets);
      for (int i = 0; i < dim; i++)
      {
         if (sub_solve)
         {
            blockDiag->SetDiagonalBlock(i, cg_blocks[i].get());
         }
         else
         {
            blockDiag->SetDiagonalBlock(i, amg_blocks[i].get());
         }
      }
      prec.reset(blockDiag);
   }
   else
   {
      // 13(d) If not using PA, configure preconditioner on global matrix.
      auto *amg = new HypreBoomerAMG(*A.As<HypreParMatrix>());
      if (amg_elast && !a.StaticCondensationIsEnabled())
      {
         amg->SetElasticityOptions(&fespace);
      }
      else
      {
         amg->SetSystemsOptions(dim, reorder_space);
      }
      prec.reset(amg);
   }
   // 13(e) For componentwise action, create block operator and form linear system.
   unique_ptr<Operator> A_components = nullptr;
   unique_ptr<BlockFESpaceOperator> pa_blocks;
   if (componentwise_action)
   {
      pa_blocks.reset(new BlockFESpaceOperator(fespaces));
      for (int j = 0; j < dim; j++)
      {
         for (int i = 0; i < dim; i++)
         {
            pa_blocks->SetBlock(i,j,pa_components[i + dim*j].get());
         }
      }
      Operator *A_temp;
      pa_blocks->FormLinearSystem(ess_tdof_list, x, b, A_temp, X, B);
      A_components.reset(A_temp);
      a_lhs = pa_blocks.get();
   }
   assembly_timer.Stop();

   // 14. Create the global CG solver, solve, and recover solution.
   CGSolver solver(MPI_COMM_WORLD);
   solver.SetRelTol(1e-8);
   solver.SetMaxIter(2500);
   solver.SetPrintLevel(1);
   if (prec) { solver.SetPreconditioner(*prec); }
   solver.SetOperator(A_components ? *A_components : *A);

   StopWatch linear_solve_timer;
   linear_solve_timer.Start();
   solver.Mult(B, X);
   linear_solve_timer.Stop();

   a_lhs->RecoverFEMSolution(X, b, x);
   total_timer.Stop();

   // Print run times
   if (Mpi::Root())
   {
      cout << "Elapsed Times\n";
      cout << "Assembly (s) = " << assembly_timer.RealTime() << endl;
      cout << "Linear Solve (s) = " << linear_solve_timer.RealTime() << endl;
      cout << "Total Solve (s) " << total_timer.RealTime() << endl;
   }

   // 15. For non-NURBS meshes, make the mesh curved based on the finite element
   //     space. This means that we define the mesh elements through a fespace
   //     based transformation of the reference element.  This allows us to save
   //     the displaced mesh as a curved mesh when using high-order finite
   //     element displacement field. We assume that the initial mesh (read from
   //     the file) is not higher order curved mesh compared to the chosen FE
   //     space.
   pmesh.SetNodalFESpace(&fespace);

   // 16. If visualization is enabled, Save in parallel the displaced mesh and
   //     the inverted solution (which gives the backward displacements to the
   //     original grid). This output can be viewed later using GLVis: "glvis
   //     -np <np> -m mesh -g sol".
   //
   //     Also, save the displacement, with displaced mesh, to VTK.
   if (visualization)
   {
      GridFunction *nodes = pmesh.GetNodes();
      *nodes += x;
      x *= -1;

      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << Mpi::WorldRank();
      sol_name << "sol." << setfill('0') << setw(6) << Mpi::WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);

      ParaViewDataCollection pd("LOR_Elasticity", &pmesh);
      pd.SetPrefixPath("ParaView");
      pd.RegisterField("displacement", &x);
      pd.SetLevelsOfDetail(order);
      pd.SetDataFormat(VTKFormat::BINARY);
      pd.SetHighOrderOutput(true);
      pd.SetCycle(0);
      pd.SetTime(0.0);
      pd.Save();
   }

   return 0;
}
