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
//                   --------------------------------------
//                    MG Abs L(1)-Jacobi smoothers miniapp
//                   --------------------------------------
//
// (See lpq-jacobi.cpp first)
//
// This miniapp illustrates the use of an absolute value L(1)-Jacobi smoother.
// We use a multigrid approach (cf. ex26(p)). The global solver and
// the coarse level solver are user-selected. The current options are SLI and
// PCG. The intermediate levels are directy smoothed with the absolute value
// L(1)-Jacobi preconditioner. The systems to solve correspond to a mass matrix,
// a difussion system, an elasticity system, and a definite Maxwell system.
//
// The preconditioner can be defined at run-time. Similarly, the mesh can be
// modified by a Kershaw transformation at run-time. Relative tolerance and
// maximum number of iterations can be modified as well.
//
// Compile with: make mg-lpq-jacobi
//
// Sample runs: mpirun -np 4 ./mg-abs-l1-jacobi
//              mpirun -np 4 ./mg-abs-l1-jacobi -s 1 -i 3
//              mpirun -np 4 ./mg-abs-l1-jacobi -m meshes/icf.mesh -f 0.5
//              mpirun -np 4 ./mg-abs-l1-jacobi -rs 2 -rp 0
//              mpirun -np 4 ./mg-abs-l1-jacobi -t 1e5 -ni 100 -vis
//              mpirun -np 4 ./mg-abs-l1-jacobi -m meshes/beam-tet.mesh -Ky 0.5 -Kz 0.5

#include "ds-common.hpp"

using namespace std;
using namespace mfem;
using namespace ds_common;

int main(int argc, char *argv[])
{
   /// 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   /// 2. Parse command line options.
   string mesh_file = "meshes/cube.mesh";
   // System properties
   int order = 1;
   SolverType solver_type = cg;
   IntegratorType integrator_type = diffusion;
   int assembly_type_int = 4;
   AssemblyLevel assembly_type;
   // Number of refinements
   int refine_serial = 0;
   int refine_parallel = 0;
   // Number of geometric and order levels
   int geometric_levels = 1;
   int order_levels = 1;
   // Solver parameters
   real_t rel_tol = 1e-10;
   real_t max_iter = 3000;
   // Kershaw Transformation
   real_t eps_y = 0.0;
   real_t eps_z = 0.0;
   // Other options
   string device_config = "cpu";
   bool use_monitor = false;
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&geometric_levels, "-gl", "--geometric-levels",
                  "Number of geometric refinements (levels) done prior to order refinements.");
   args.AddOption(&order_levels, "-ol", "--order-levels",
                  "Number of order refinements (levels). "
                  "Finest level in the hierarchy has order 2^{or}.");
   args.AddOption((int*)&solver_type, "-s", "--solver",
                  "Solvers to be considered:"
                  "\n\t0: Stationary Linear Iteration"
                  "\n\t1: Preconditioned Conjugate Gradient");
   args.AddOption((int*)&integrator_type, "-i", "--integrator",
                  "Integrators to be considered:"
                  "\n\t0: MassIntegrator"
                  "\n\t1: DiffusionIntegrator"
                  "\n\t2: ElasticityIntegrator"
                  "\n\t3: CurlCurlIntegrator + VectorFEMassIntegrator");
   args.AddOption(&assembly_type_int, "-a", "--assembly",
                  "Assembly level to be considered:"
                  "\n\t0: LEGACY"
                  "\n\t1: LEGACYFULL (Deprecated)"
                  "\n\t2: FULL"
                  "\n\t3: ELEMENT"
                  "\n\t4: PARTIAL"
                  "\n\t5: NONE");
   args.AddOption(&refine_serial, "-rs", "--refine-serial",
                  "Number of serial refinements");
   args.AddOption(&refine_parallel, "-rp", "--refine-parallel",
                  "Number of parallel refinements");
   args.AddOption(&rel_tol, "-t", "--tolerance",
                  "Relative tolerance for the iterative solver");
   args.AddOption(&max_iter, "-ni", "--iterations",
                  "Maximum number of iterations");
   args.AddOption(&eps_y, "-Ky", "--Kershaw-y",
                  "Kershaw transform factor, eps_y in (0,1]");
   args.AddOption(&eps_z, "-Kz", "--Kershaw-z",
                  "Kershaw transform factor, eps_z in (0,1]");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_monitor, "-mon", "--monitor", "-no-mon",
                  "--no-monitor",
                  "Enable or disable Data Monitor.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   MFEM_ASSERT((0 <= solver_type) && (solver_type < num_solvers), "");
   MFEM_ASSERT((0 <= integrator_type) && (integrator_type < num_integrators), "");
   MFEM_ASSERT((0 <= assembly_type_int) && (assembly_type_int < 6), "");
   MFEM_ASSERT(geometric_levels >= 0, "geometric_level needs to be non-negative");
   MFEM_ASSERT(order_levels >= 0, "order_level needs to be non-negative");
   MFEM_ASSERT((0.0 <= eps_y) && (eps_y <= 1.0), "eps_y in [0,1]");
   MFEM_ASSERT((0.0 <= eps_z) && (eps_z <= 1.0), "eps_z in [0,1]");

   kappa = freq * M_PI;

   ostringstream file_name;
   if (use_monitor)
   {
      file_name << "MGABS-"
                << "G" << geometric_levels
                << "O" << order_levels
                << "O" << order
                << "I" << (int) integrator_type
                << "S" << (int) solver_type
                << "A" << assembly_type_int
                << ".csv";
   }

   string assembly_description;
   switch (assembly_type_int)
   {
      case 0:
         assembly_type = AssemblyLevel::LEGACY;
         assembly_description = "Using Legacy type of assembly level...";
         break;
      case 1:
         assembly_type = AssemblyLevel::LEGACYFULL;
         assembly_description =
            "Using Legacy Full type of assembly level... (Deprecated)";
         break;
      case 2:
         assembly_type = AssemblyLevel::FULL;
         assembly_description = "Using Full type of assembly level...";
         break;
      case 3:
         assembly_type = AssemblyLevel::ELEMENT;
         assembly_description = "Using Element type of assembly level...";
         break;
      case 4:
         assembly_type = AssemblyLevel::PARTIAL;
         assembly_description = "Using Partial type of assembly level...";
         break;
      case 5:
         assembly_type = AssemblyLevel::NONE;
         assembly_description = "Using matrix-free type of assembly level...";
         break;
      default:
         MFEM_ABORT("Unsupported option!");
   }

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   /// 3. Read the serial mesh from the given mesh file.
   ///    For convinience, the meshes are available in
   ///    ./meshes, and the number of serial and parallel
   ///    refinements are user-defined.
   Mesh *serial_mesh = new Mesh(mesh_file);
   for (int ls = 0; ls < refine_serial; ls++)
   {
      serial_mesh->UniformRefinement();
   }

   /// 4. Define a parallel mesh by a partitioning of the serial mesh.
   ///    Number of parallel refinements given by the user. If defined,
   ///    apply Kershaw transformation.
   ParMesh *mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   for (int lp = 0; lp < refine_parallel; lp++)
   {
      mesh->UniformRefinement();
   }

   dim = mesh->Dimension();
   space_dim = mesh->SpaceDimension();

   bool cond_z = (dim < 3)?true:(eps_z != 0.0); // lazy check
   if (eps_y != 0.0 && cond_z)
   {
      if (dim < 3) { eps_z = 0.0; }
      common::KershawTransformation kershawT(dim, eps_y, eps_z);
      mesh->Transform(kershawT);
   }

   /// 5. Define a finite element space on the mesh. We use different spaces
   ///    and collections for different systems.
   ///    - H1-conforming Lagrange elements for the H1-mass matrix and the
   ///      diffusion problem.
   ///    - Vector H1-conforming Lagrange elements for the elasticity problem.
   ///    - H(curl)-conforming Nedelec elements for the definite Maxwell problem.
   FiniteElementCollection *fec;
   ParFiniteElementSpace *coarse_fes;
   switch (integrator_type)
   {
      case mass: case diffusion:
         fec = new H1_FECollection(order, dim);
         coarse_fes = new ParFiniteElementSpace(mesh, fec);
         break;
      case elasticity:
         fec = new H1_FECollection(order, dim);
         coarse_fes= new ParFiniteElementSpace(mesh, fec, dim);
         break;
      case maxwell:
         fec = new ND_FECollection(order, dim);
         coarse_fes= new ParFiniteElementSpace(mesh, fec);
         break;
      default:
         mfem_error("Invalid integrator type! Check FiniteElementCollection");
   }

   if (order > 1)
   {
      if (Mpi::Root())
      {
         mfem::out << "Warning! Polynomial order provided. "
                   << "Ignoring order level..." << endl;
      }
      order_levels = 0;
   }

   /// 6. Define a finite element space hierarchy for the multigrid solver.
   ///    Define a FEC array for the order-refinement levels. Add the refinements
   ///    to the hierarchy.
   Array<FiniteElementCollection*> fec_array;
   fec_array.Append(fec);
   // Transfer ownership of mesh and coarse_fes to fes_hierarchy
   ParFiniteElementSpaceHierarchy* fes_hierarchy = new
   ParFiniteElementSpaceHierarchy(mesh, coarse_fes, true, true);

   for (int lg = 0; lg < geometric_levels; ++lg)
   {
      fes_hierarchy->AddUniformlyRefinedLevel();
   }
   for (int lo = 0; lo < order_levels; ++lo)
   {
      switch (integrator_type)
      {
         case mass: case diffusion: case elasticity:
            fec_array.Append(new H1_FECollection(std::pow(2, lo + 1), dim));
            break;
         case maxwell:
            fec_array.Append(new ND_FECollection(std::pow(2, lo + 1), dim));
            break;
         default:
            mfem_error("Invalid integrator type! "
                       "Check FiniteElementCollection for order refinements...");
      }
      fes_hierarchy->AddOrderRefinedLevel(fec_array.Last());
   }

   HYPRE_BigInt sys_size = fes_hierarchy->GetFinestFESpace().GlobalTrueVSize();
   if (Mpi::Root())
   {
      mfem::out << "Number of unknowns: " << sys_size << endl;
      mfem::out << assembly_description << endl;
   }

   /// 7. Extract the list of the essential boundary DoFs. We mark all boundary
   ///    attibutes as essential. LpqGeometricMultigrid will determine
   ///    the DoFs per level.
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   if (mesh->bdr_attributes.Size()) { ess_bdr = 1; }

   /// 8. Define the linear system. Set up the linear form b(.).
   ///    The linear form has the standar form (f,v).
   ///    Define the matrices and vectors associated to the forms, and project
   ///    the required boundary data into the GridFunction solution.
   ParLinearForm *b = new ParLinearForm(&fes_hierarchy->GetFinestFESpace());
   LinearFormIntegrator *lfi = nullptr;

   // These pointers are not owned by the integrators
   FunctionCoefficient *scalar_u = nullptr;
   FunctionCoefficient *scalar_f = nullptr;
   VectorFunctionCoefficient *vector_u = nullptr;
   VectorFunctionCoefficient *vector_f = nullptr;

   ConstantCoefficient one(1.0);

   // These variables will define the linear system
   ParGridFunction x(&fes_hierarchy->GetFinestFESpace());
   OperatorPtr A;
   Vector B, X;

   x = 0.0;

   switch (integrator_type)
   {
      case mass:
         scalar_u = new FunctionCoefficient(diffusion_solution);
         lfi = new DomainLFIntegrator(*scalar_u);
         x.ProjectBdrCoefficient(*scalar_u, ess_bdr);
         break;
      case diffusion:
         scalar_u = new FunctionCoefficient(diffusion_solution);
         scalar_f = new FunctionCoefficient(diffusion_source);
         lfi = new DomainLFIntegrator(*scalar_f);
         x.ProjectBdrCoefficient(*scalar_u, ess_bdr);
         break;
      case elasticity:
         vector_u = new VectorFunctionCoefficient(space_dim, elasticity_solution);
         vector_f = new VectorFunctionCoefficient(space_dim, elasticity_source);
         lfi = new VectorDomainLFIntegrator(*vector_f);
         x.ProjectBdrCoefficient(*vector_u, ess_bdr);
         break;
      case maxwell:
         vector_u = new VectorFunctionCoefficient(space_dim, maxwell_solution);
         vector_f = new VectorFunctionCoefficient(space_dim, maxwell_source);
         lfi = new VectorFEDomainLFIntegrator(*vector_f);
         x.ProjectBdrCoefficientTangent(*vector_u, ess_bdr);
         break;
      default:
         mfem_error("Invalid integrator type! Check ParLinearForm");
   }
   b->AddDomainIntegrator(lfi);
   b->Assemble();

   /// 9. Define a geometric multigrid solver. The bilinear form
   ///    a(.,.) is assembled internally. Set up the type of cycles
   ///    and form the linear system.
   auto mg = new AbsL1GeometricMultigrid(*fes_hierarchy,
                                         ess_bdr,
                                         integrator_type,
                                         solver_type,
                                         assembly_type);
   mg->SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);
   mg->FormFineLinearSystem(x, *b, A, X, B);

   A.SetOperatorOwner(mg->GetOwnershipLevelOperators());

   Solver *solver = nullptr;
   DataMonitor *monitor = nullptr;

   switch (solver_type)
   {
      case sli:
         solver = new SLISolver(MPI_COMM_WORLD);
         break;
      case cg:
         solver = new CGSolver(MPI_COMM_WORLD);
         break;
      default:
         mfem_error("Invalid solver type!");
   }
   solver->SetOperator(*A.Ptr());

   IterativeSolver *it_solver = dynamic_cast<IterativeSolver*>(solver);
   if (it_solver)
   {
      it_solver->SetRelTol(rel_tol);
      it_solver->SetMaxIter(max_iter);
      it_solver->SetPrintLevel(1);
      it_solver->SetPreconditioner(*mg);
      if (use_monitor)
      {
         monitor = new DataMonitor(file_name.str(), MONITOR_DIGITS);
         it_solver->SetMonitor(*monitor);
      }
   }

   solver->Mult(B, X);

   /// 10. Recover the solution x as a grid function. Send the data by socket
   ///     to a GLVis server.
   mg->RecoverFineFEMSolution(X, *b, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *fes_hierarchy->GetFinestFESpace().GetParMesh()
               << x << flush;
   }

   /// 11. Compute and print the L^2 norm of the error
   {
      real_t error = 0.0;
      switch (integrator_type)
      {
         case mass: case diffusion:
            error = x.ComputeL2Error(*scalar_u);
            break;
         case elasticity: case maxwell:
            error = x.ComputeL2Error(*vector_u);
            break;
         default:
            mfem_error("Invalid integrator type! Check ComputeL2Error");
      }
      if (Mpi::Root())
      {
         mfem::out << "\n|| u_h - u ||_{L^2} = " << error << "\n" << endl;
      }
   }

   /// 12. Free the memory used
   delete mg;
   delete solver;
   delete b;
   if (monitor) { delete monitor; }
   if (scalar_u) { delete scalar_u; }
   if (scalar_f) { delete scalar_f; }
   if (vector_u) { delete vector_u; }
   if (vector_f) { delete vector_f; }
   for (int level = 0; level < fec_array.Size(); ++level)
   {
      delete fec_array[level];
   }
   delete fes_hierarchy;

   return 0;
}