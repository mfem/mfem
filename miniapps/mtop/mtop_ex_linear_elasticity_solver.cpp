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
// Driver coverage:
//   1. PA elasticity solve and inverse-operator consistency.
//   2. Functional, state gradient, adjoint gradient, and Taylor test.
//   3. Vector Dirichlet data plus internally assembled volume/traction loads.
//   4. Optional ParaView output.

#include "linear_elasticity_solver.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize parallel runtimes and parse driver options.
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file =
      MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_quad.mesh";
   const char *device_config = "cpu";
   int order = 2;
   int serial_refinements = 0;
   int parallel_refinements = 1;
   bool paraview = false;
   bool use_by_vdim = false;
   const char *preconditioner = "jacobi";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&device_config, "-d", "--device", "Device configuration.");
   args.AddOption(&order, "-o", "--order", "H1 polynomial degree.");
   args.AddOption(&serial_refinements, "-rs", "--serial-refinements",
                  "Number of serial refinements.");
   args.AddOption(&parallel_refinements, "-rp", "--parallel-refinements",
                  "Number of parallel refinements.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview", "Write the solution for ParaView.");
   args.AddOption(&use_by_vdim, "-vdim", "--by-vdim", "-nodes",
                  "--by-nodes", "Use byVDIM instead of byNODES for plain "
                  "and monolithic auxiliary AMG.");
   args.AddOption(&preconditioner, "-pc", "--preconditioner",
                  "Preconditioner: jacobi, amg, lor-diagonal, or "
                  "lor-monolithic.");
   args.ParseCheck();

   LinearElasticitySolver::PreconditionerType preconditioner_type;
   if (std::string(preconditioner) == "jacobi")
   {
      preconditioner_type =
         LinearElasticitySolver::PreconditionerType::Jacobi;
   }
   else if (std::string(preconditioner) == "amg")
   {
      preconditioner_type =
         LinearElasticitySolver::PreconditionerType::AMG;
   }
   else if (std::string(preconditioner) == "lor-diagonal")
   {
      preconditioner_type =
         LinearElasticitySolver::PreconditionerType::LORDiagonalAMG;
   }
   else if (std::string(preconditioner) == "lor-monolithic")
   {
      preconditioner_type =
         LinearElasticitySolver::PreconditionerType::LORMonolithicAMG;
   }
   else
   {
      MFEM_ABORT("Unknown preconditioner '" << preconditioner << "'.");
   }

   // 2. Build and refine the mesh. Serial refinement occurs before
   // partitioning; parallel refinement operates on the distributed mesh.
   Device device(device_config);
   Mesh mesh(mesh_file, 1, 1);
   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh.UniformRefinement();
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   for (int i = 0; i < parallel_refinements; ++i)
   {
      pmesh.UniformRefinement();
   }

   // 3. Construct the high-order vector H1 space. Elasticity PA requires
   // byNODES. The ordering option controls the auxiliary space used by plain
   // AMG and monolithic LOR/AMG.
   H1_FECollection fec(order, pmesh.Dimension());
   ParFiniteElementSpace fes(&pmesh, &fec, pmesh.Dimension(),
                             Ordering::byNODES);
   const long long total_elements = pmesh.GetGlobalNE();
   const HYPRE_BigInt total_dofs = fes.GlobalTrueVSize();
   const HYPRE_BigInt total_nodes = total_dofs/fes.GetVDim();
   if (Mpi::Root())
   {
      std::cout << "Total elements: " << total_elements << std::endl;
      std::cout << "Total nodes: " << total_nodes << std::endl;
      std::cout << "Total DOFs: " << total_dofs << std::endl;
      std::cout << "Auxiliary AMG ordering: "
                << (use_by_vdim ? "byVDIM" : "byNODES") << std::endl;
      std::cout << "Preconditioner: " << preconditioner << std::endl;
   }

   // 4. Configure material parameters, solver tolerances, preconditioner, and
   // homogeneous displacement constraints on every boundary attribute.
   LinearElasticitySolver solver(fes);
   solver.SetPreconditionerType(preconditioner_type);
   solver.SetMonolithicLOROrdering(use_by_vdim ? Ordering::byVDIM :
                                   Ordering::byNODES);
   solver.SetLambda(2.0);
   solver.SetMu(3.0);
   solver.SetRelTol(1.0e-13);
   solver.SetMaxIter(1000);
   for (int id = 1; id <= pmesh.bdr_attributes.Max(); ++id)
   {
      solver.AddBoundaryID(id);
   }

   // Assemble explicitly so operator and preconditioner costs are measured
   // separately from the first linear solve.
   solver.Assemble();
   double initial_assembly_time = solver.GetAssemblyTime();
   double initial_prec_time = solver.GetPrecAssemblyTime();
   double initial_assembly_time_max = 0.0;
   double initial_prec_time_max = 0.0;
   MPI_Reduce(&initial_assembly_time, &initial_assembly_time_max, 1,
              MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&initial_prec_time, &initial_prec_time_max, 1,
              MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

   // 5. Verify K^{-1}K on a random admissible true-DOF vector. Essential
   // entries are zeroed before applying the constrained PA operator K.
   Vector exact(fes.GetTrueVSize());
   exact.Randomize(1729);
   const Array<int> &ess_tdofs = solver.GetEssentialTrueDofs();
   for (int i = 0; i < ess_tdofs.Size(); ++i) { exact[ess_tdofs[i]] = 0.0; }

   Vector rhs(fes.GetTrueVSize());
   solver.GetOperator()->Mult(exact, rhs);
   Vector computed;
   StopWatch homogeneous_timer;
   homogeneous_timer.Start();
   solver.Mult(rhs, computed);
   homogeneous_timer.Stop();
   const int homogeneous_iterations = solver.GetNumIterations();
   real_t homogeneous_time = homogeneous_timer.RealTime();
   real_t homogeneous_time_max = 0.0;
   MPI_Reduce(&homogeneous_time, &homogeneous_time_max, 1, MFEM_MPI_REAL_T,
              MPI_MAX, 0, MPI_COMM_WORLD);
   computed -= exact;

   const real_t local_error = computed.Norml2();
   const real_t local_norm = exact.Norml2();
   real_t error2 = local_error*local_error;
   real_t norm2 = local_norm*local_norm;
   real_t global_error2 = 0.0;
   real_t global_norm2 = 0.0;
   MPI_Allreduce(&error2, &global_error2, 1, MFEM_MPI_REAL_T, MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&norm2, &global_norm2, 1, MFEM_MPI_REAL_T, MPI_SUM,
                 MPI_COMM_WORLD);
   const real_t relative_error =
      std::sqrt(global_error2/std::max(global_norm2, real_t(1.0e-30)));

   // 6. Define J(u) = 1/2 int |u|^2 and verify its reduced derivative with
   // respect to the amplitude alpha of a fixed volume load. The state
   // derivative is M u, the adjoint satisfies K^T p = M u, and
   // dJ/dalpha = p^T f.
   ParBilinearForm mass(&fes);
   mass.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass.AddDomainIntegrator(new VectorMassIntegrator);
   mass.Assemble();
   Array<int> no_constraints;
   OperatorHandle mass_operator;
   mass_operator.SetType(Operator::ANY_TYPE);
   // No essential elimination is applied to M: constrained state variations
   // are handled by the adjoint solve itself.
   mass.FormSystemMatrix(no_constraints, mass_operator);

   // Assemble the unit-amplitude true-DOF load f for a constant force in the
   // last spatial direction.
   Vector load_direction_value(pmesh.Dimension());
   load_direction_value = 0.0;
   load_direction_value(pmesh.Dimension() - 1) = 1.0;
   VectorConstantCoefficient load_direction(load_direction_value);
   ParLinearForm unit_load_form(&fes);
   unit_load_form.AddDomainIntegrator(
      new VectorDomainLFIntegrator(load_direction));
   unit_load_form.Assemble();
   Vector unit_load(fes.GetTrueVSize());
   unit_load_form.ParallelAssemble(unit_load);

   const real_t alpha = 1.0;
   Vector alpha_load(unit_load);
   alpha_load *= alpha;
   Vector functional_state;
   // Forward state equation: K u(alpha) = alpha f.
   solver.Mult(alpha_load, functional_state);
   Vector state_gradient(fes.GetTrueVSize());
   // Functional state derivative: dJ/du = M u.
   mass_operator->Mult(functional_state, state_gradient);
   const real_t functional = 0.5*InnerProduct(
                                MPI_COMM_WORLD, functional_state,
                                state_gradient);

   Vector adjoint;
   // Adjoint equation: K^T p = dJ/du, with zero adjoint Dirichlet data.
   solver.MultTranspose(state_gradient, adjoint);
   // Chain rule for the scalar load amplitude: dJ/dalpha = p^T f.
   const real_t reduced_gradient = InnerProduct(MPI_COMM_WORLD, adjoint,
                                                unit_load);
   const real_t state_gradient_norm = std::sqrt(InnerProduct(
                                                   MPI_COMM_WORLD,
                                                   state_gradient,
                                                   state_gradient));

   const int num_taylor_steps = 4;
   real_t h = 1.0e-1;
   real_t previous_first = -1.0;
   real_t previous_second = -1.0;
   real_t final_second_remainder = 0.0;
   real_t initial_second_remainder = 0.0;
   if (Mpi::Root())
   {
      std::cout << "Functional J(u): " << functional << std::endl;
      std::cout << "State-gradient norm: " << state_gradient_norm << std::endl;
      std::cout << "Adjoint reduced gradient dJ/dalpha: "
                << reduced_gradient << std::endl;
      std::cout << "Taylor remainder check:\n"
                << std::setw(14) << "h"
                << std::setw(22) << "|J(a+h)-J(a)|"
                << std::setw(10) << "rate"
                << std::setw(28) << "|J(a+h)-J(a)-h dJ|"
                << std::setw(10) << "rate" << std::endl;
   }
   // A correct adjoint gradient gives O(h) for the uncorrected difference and
   // O(h^2) after subtracting h*dJ/dalpha.
   for (int step = 0; step < num_taylor_steps; ++step)
   {
      Vector perturbed_load(unit_load);
      perturbed_load *= alpha + h;
      Vector perturbed_state;
      Vector perturbed_state_gradient(fes.GetTrueVSize());
      solver.Mult(perturbed_load, perturbed_state);
      mass_operator->Mult(perturbed_state, perturbed_state_gradient);
      const real_t perturbed_functional = 0.5*InnerProduct(
                                             MPI_COMM_WORLD,
                                             perturbed_state,
                                             perturbed_state_gradient);
      const real_t first_remainder =
         std::abs(perturbed_functional - functional);
      const real_t second_remainder =
         std::abs(perturbed_functional - functional - h*reduced_gradient);
      if (step == 0) { initial_second_remainder = second_remainder; }
      final_second_remainder = second_remainder;
      const real_t first_rate = step ?
                                std::log(previous_first/first_remainder)/std::log(10.0) : 0.0;
      const real_t second_rate = step ?
                                 std::log(previous_second/second_remainder)/std::log(10.0) : 0.0;
      if (Mpi::Root())
      {
         std::cout << std::scientific << std::setprecision(6)
                   << std::setw(14) << h
                   << std::setw(22) << first_remainder
                   << std::fixed << std::setprecision(2)
                   << std::setw(10) << first_rate
                   << std::scientific << std::setprecision(6)
                   << std::setw(28) << second_remainder
                   << std::fixed << std::setprecision(2)
                   << std::setw(10) << second_rate << std::endl;
      }
      previous_first = first_remainder;
      previous_second = second_remainder;
      h *= 0.1;
   }
   // Three decade reductions in h should reduce the quadratic remainder by
   // approximately six orders of magnitude from its first recorded value.
   const bool taylor_passed = final_second_remainder <
                              2.0e-5*initial_second_remainder;
   if (Mpi::Root())
   {
      std::cout << std::defaultfloat << std::setprecision(6);
      std::cout << "Taylor check: "
                << (taylor_passed ? "PASSED" : "FAILED") << std::endl;
   }

   // 7. Exercise vector-valued Dirichlet data and internally assembled volume
   // and boundary loads with a rigid translation, which has zero strain.
   solver.ClearBoundaryConditions();
   Vector translation_value(2);
   translation_value(0) = 0.125;
   translation_value(1) = -0.25;
   VectorConstantCoefficient translation(translation_value);
   for (int id = 1; id <= pmesh.bdr_attributes.Max(); ++id)
   {
      solver.AddDisplacementBC(id, translation);
   }
   Vector zero_value(2);
   zero_value = 0.0;
   VectorConstantCoefficient zero_vector(zero_value);
   solver.AddVolumeLoad(1, zero_vector);
   solver.AddBoundaryLoad(1, zero_vector);
   ParGridFunction displacement(&fes);
   StopWatch prescribed_timer;
   prescribed_timer.Start();
   // Solve() assembles all registered loads, applies prescribed displacement
   // values, solves the constrained system, and recovers the grid function.
   solver.Solve(displacement);
   prescribed_timer.Stop();
   const int prescribed_iterations = solver.GetNumIterations();
   double prescribed_assembly_time = solver.GetAssemblyTime();
   double prescribed_prec_time = solver.GetPrecAssemblyTime();
   double prescribed_assembly_time_max = 0.0;
   double prescribed_prec_time_max = 0.0;
   MPI_Reduce(&prescribed_assembly_time, &prescribed_assembly_time_max, 1,
              MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&prescribed_prec_time, &prescribed_prec_time_max, 1,
              MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   real_t prescribed_time = prescribed_timer.RealTime();
   real_t prescribed_time_max = 0.0;
   MPI_Reduce(&prescribed_time, &prescribed_time_max, 1, MFEM_MPI_REAL_T,
              MPI_MAX, 0, MPI_COMM_WORLD);
   const real_t boundary_error = displacement.ComputeL2Error(translation);

   // 8. Report correctness and maximum wall times across MPI ranks.
   if (Mpi::Root())
   {
      std::cout << "PA elasticity relative solve error: "
                << relative_error << std::endl;
      std::cout << "Initial assembly time: " << initial_assembly_time_max
                << " seconds" << std::endl;
      std::cout << "Initial preconditioner assembly time: "
                << initial_prec_time_max << " seconds" << std::endl;
      std::cout << "Homogeneous solve iterations: "
                << homogeneous_iterations << std::endl;
      std::cout << "Homogeneous solve time: " << homogeneous_time_max
                << " seconds" << std::endl;
      std::cout << "Prescribed displacement error: "
                << boundary_error << std::endl;
      std::cout << "Prescribed assembly time: "
                << prescribed_assembly_time_max << " seconds" << std::endl;
      std::cout << "Prescribed preconditioner assembly time: "
                << prescribed_prec_time_max << " seconds" << std::endl;
      std::cout << "Prescribed solve iterations: "
                << prescribed_iterations << std::endl;
      std::cout << "Prescribed solve time: " << prescribed_time_max
                << " seconds" << std::endl;
   }

   // 9. Optionally write the final prescribed-displacement solution.
   if (paraview)
   {
      ParaViewDataCollection paraview_dc(
         "mtop_linear_elasticity_solver", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("displacement", &displacement);
      paraview_dc.Save();
   }

   // Treat every numerical verification as part of the executable test.
   return relative_error < 1.0e-10 && boundary_error < 1.0e-10 &&
          taylor_passed ?
          EXIT_SUCCESS : EXIT_FAILURE;
}
