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
//                 -----------------------------------------
//                   Absolute L(1)-Jacobi smoothers miniapp
//                 -----------------------------------------
//
// This miniapp illustrates the implementation of an (slightly generalized)
// absolute-L(1) Jacobi preconditioner. This preconditioner is tested in different
// settings. We use Stationary Linear Iterations and Preconditioned Conjugate
// Gradient as the main solvers.
// TODO(Gabriel): mass, diffusion, and maxwell so far...
// We consider a H1-mass matrix, a diffusion matrix, a elasticity system, and a
// definite Maxwell system.
//
// The preconditioner can be defined at run-time. Similarly, the mesh can be
// modified by a Kershaw transformation at run-time. Relative tolerance and
// maximum number of iterations can be modified as well.
//
// Compile with: make l1-partial
//
// Sample runs: mpirun -np 4 ./l1-partial
//              mpirun -np 4 ./l1-partial -s 1 -i 3
//              mpirun -np 4 ./l1-partial -m meshes/icf.mesh -f 0.5
//              mpirun -np 4 ./l1-partial -rs 2 -rp 0
//              mpirun -np 4 ./l1-partial -t 1e5 -ni 100 -vis
//              mpirun -np 4 ./l1-partial -m meshes/beam-tet.mesh -Ky 0.5 -Kz 0.5

#include "lpq-common.hpp"

using namespace std;
using namespace mfem;
using namespace lpq_common;

int main(int argc, char *argv[])
{
   /// 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   /// 2. Parse command line options.
   string mesh_file = "meshes/cube.mesh";
   // System properties
   int order = 1;
   SolverType solver_type = sli;
   IntegratorType integrator_type = mass;
   LpqType pc_type = global;
   int assembly_type_int = 4;
   AssemblyLevel assembly_type;
   // Number of refinements
   int refine_serial = 0;
   int refine_parallel = 0;
   // Preconditioner parameters
   double p_order = 1.0;
   double q_order = 0.0;
   // Solver parameters
   double rel_tol = 1e-10;
   double max_iter = 3000;
   // Kershaw Transformation
   double eps_y = 0.0;
   double eps_z = 0.0;
   // Other options
   string device_config = "cpu";
   bool use_pc = true;
   bool use_monitor = false;
   bool visualization = true;

   // Construct argument parser
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
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
   args.AddOption((int*)&pc_type, "-pc", "--preconditioner",
                  "Preconditioners to be considered:"
                  "\n\t0: No preconditioner"
                  "\n\t1: L(p,q)-Jacobi preconditioner"
                  "\n\t2: Element L(p,q)-Jacobi preconditioner");
   args.AddOption(&refine_serial, "-rs", "--refine-serial",
                  "Number of serial refinements");
   args.AddOption(&refine_parallel, "-rp", "--refine-parallel",
                  "Number of parallel refinements");
   args.AddOption(&p_order, "-p", "--p-order",
                  "P-order for L(p,q)-Jacobi preconditioner");
   args.AddOption(&q_order, "-q", "--q-order",
                  "Q-order for L(p,q)-Jacobi preconditioner");
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
   args.AddOption(&use_pc, "-pc", "--preconditioner", "-no-pc",
                  "--no-preconditioner",
                  "Enable or disable Absolute L1 Jacobi preconditioner.");
   args.AddOption(&use_monitor, "-mon", "--monitor", "-no-mon",
                  "--no-monitor",
                  "Enable or disable Data Monitor.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   MFEM_ASSERT(p_order > 0.0, "p needs to be positive");
   MFEM_ASSERT((0 <= solver_type) && (solver_type < num_solvers), "");
   MFEM_ASSERT((0 <= integrator_type) && (integrator_type < num_integrators), "");
   MFEM_ASSERT((0 <= assembly_type_int) && (assembly_type_int < 6), "");
   MFEM_ASSERT((0 <= pc_type) && (pc_type < num_lpq_pc), "");
   MFEM_ASSERT(0.0 < eps_y <= 1.0, "eps_y in (0,1]");
   MFEM_ASSERT(0.0 < eps_z <= 1.0, "eps_z in (0,1]");

   kappa = freq * M_PI;

   ostringstream file_name;
   if (use_monitor)
   {
      file_name << "ABS-"
                << "O" << order
                << "I" << (int) integrator_type
                << "S" << (int) solver_type
                << "A" << assembly_type_int
                << ".csv";
   }

   switch (assembly_type_int)
   {
      case 0:
         assembly_type = AssemblyLevel::LEGACY;
         break;
      case 1:
         assembly_type = AssemblyLevel::LEGACYFULL;
         break;
      case 2:
         assembly_type = AssemblyLevel::FULL;
         break;
      case 3:
         assembly_type = AssemblyLevel::ELEMENT;
         break;
      case 4:
         assembly_type = AssemblyLevel::PARTIAL;
         break;
      case 5:
         assembly_type = AssemblyLevel::NONE;
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
   ///    Number of parallel refinements given by the user.
   ParMesh *mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   for (int lp = 0; lp < refine_parallel; lp++)
   {
      mesh->UniformRefinement();
   }

   dim = mesh->Dimension();
   space_dim = mesh->SpaceDimension();

   bool cond_z = (dim < 3)?true:(eps_z != 0); // lazy check
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
   /// TODO(Gabriel): Elasticity not implemented yet for partial assembly
   ///    - Vector H1-conforming Lagrange elements for the elasticity problem.
   ///    - H(curl)-conforming Nedelec elements for the definite Maxwell problem.
   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;

   switch (integrator_type)
   {
      case mass: case diffusion:
         fec = new H1_FECollection(order, dim);
         fespace = new ParFiniteElementSpace(mesh, fec);
         break;
      case elasticity:
         fec = new H1_FECollection(order, dim);
         fespace = new ParFiniteElementSpace(mesh, fec, dim);
         break;
      case maxwell:
         fec = new ND_FECollection(order, dim);
         fespace = new ParFiniteElementSpace(mesh, fec);
         break;
      default:
         mfem_error("Invalid integrator type! Check FiniteElementCollection");
   }

   HYPRE_BigInt sys_size = fespace->GlobalTrueVSize();
   if (Mpi::Root())
   {
      mfem::out << "Number of unknowns: " << sys_size << endl;
   }

   /// 6. Extract the list of the essential boundary DoFs. We mark all boundary
   ///    attibutes as essential. Then we get the list of essential DoFs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   /// 7. Define the linear system. Set up the bilinear form a(.,.) and the
   ///    linear form b(.). The current implemented systems are the following:
   ///    - (u,v), i.e., L2-projection.
   ///    - (grad(u), grad(v)), i.e., diffusion operator.
   ///    - (div(u), div(v)) + (e(u),e(v)), i.e., elasticity operator.
   ///    - (curl(u), curl(v)) + (u,v), i.e., definite Maxwell operator.
   ///    The linear form has the standard form (f,v).
   ///    Define the matrices and vectors associated to the forms, and project
   ///    the required boundary data into the GridFunction solution.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   ParLinearForm *b = new ParLinearForm(fespace);

   // These pointers are owned by the forms
   LinearFormIntegrator *lfi = nullptr;
   BilinearFormIntegrator *bfi = nullptr;
   // Required for a static_cast
   SumIntegrator *sum_bfi = nullptr;

   // These pointers are not owned by the integrators
   FunctionCoefficient *scalar_u = nullptr;
   FunctionCoefficient *scalar_f = nullptr;
   VectorFunctionCoefficient *vector_u = nullptr;
   VectorFunctionCoefficient *vector_f = nullptr;

   ConstantCoefficient one(1.0);

   // These variables will define the linear system
   ParGridFunction x(fespace), y(fespace);
   OperatorPtr A;
   Vector B, X;

   x = 0.0;

   switch (integrator_type)
   {
      case mass:
         scalar_u = new FunctionCoefficient(diffusion_solution);
         lfi = new DomainLFIntegrator(*scalar_u);
         bfi = new MassIntegrator(one);
         x.ProjectBdrCoefficient(*scalar_u, ess_bdr);
         break;
      case diffusion:
         scalar_u = new FunctionCoefficient(diffusion_solution);
         scalar_f = new FunctionCoefficient(diffusion_source);
         lfi = new DomainLFIntegrator(*scalar_f);
         bfi = new DiffusionIntegrator(one);
         x.ProjectBdrCoefficient(*scalar_u, ess_bdr);
         break;
      case elasticity:
         vector_u = new VectorFunctionCoefficient(space_dim, elasticity_solution);
         vector_f = new VectorFunctionCoefficient(space_dim, elasticity_source);
         lfi = new VectorDomainLFIntegrator(*vector_f);
         bfi = new ElasticityIntegrator(one, one);
         x.ProjectBdrCoefficient(*vector_u, ess_bdr);
         break;
      case maxwell:
         vector_u = new VectorFunctionCoefficient(space_dim, maxwell_solution);
         vector_f = new VectorFunctionCoefficient(space_dim, maxwell_source);
         lfi = new VectorFEDomainLFIntegrator(*vector_f);
         bfi = new SumIntegrator();
         sum_bfi = static_cast<SumIntegrator*>(bfi);
         sum_bfi->AddIntegrator(new CurlCurlIntegrator(one));
         sum_bfi->AddIntegrator(new VectorFEMassIntegrator(one));
         x.ProjectBdrCoefficientTangent(*vector_u, ess_bdr);
         break;
      default:
         mfem_error("Invalid integrator type! Check ParLinearForm");
   }

   a->SetAssemblyLevel(assembly_type);
   a->AddDomainIntegrator(bfi);
   a->Assemble();

   b->AddDomainIntegrator(lfi);
   b->Assemble();

   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   /// 8. Construct the preconditioner. Uses AbsMult to construct an appoximation
   ///    of the diagonal of the matrix.

   Solver *jacobi = nullptr;
   Vector ones(sys_size);
   Vector diag(sys_size);

   switch (pc_type)
   {
      case none:
         break;
      case global:
         ones = 1.0;
         A->AbsMult(ones, diag);
         jacobi = new OperatorJacobiSmoother(diag, ess_tdof_list);
         break;
      case element:
         AssembleElementLpqJacobiDiag(*a, p_order, q_order, diag);
         jacobi = new OperatorJacobiSmoother(diag, ess_tdof_list);
         break;
      default:
         mfem_error("Invalid preconditioner type!");
   }

   /// 9. Construct the solver. The implemented solvers are the following:
   ///    - Stationary Linear Iteration
   ///    - Preconditioned Conjugate Gradient
   ///    Then, solve the system with the used-selected solver.
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
   solver->SetOperator(*A);

   IterativeSolver *it_solver = dynamic_cast<IterativeSolver *>(solver);
   if (it_solver)
   {
      it_solver->SetRelTol(rel_tol);
      it_solver->SetMaxIter(max_iter);
      it_solver->SetPrintLevel(1);
      if (use_monitor)
      {
         monitor = new DataMonitor(file_name.str(), MONITOR_DIGITS);
         it_solver->SetMonitor(*monitor);
      }
      if (jacobi)
      {
         it_solver->SetPreconditioner(*jacobi);
      }
   }

   solver->Mult(B, X);

   /// 10. Recover the solution x as a grid function. Send the data by socket
   ///     to a GLVis server.
   a->RecoverFEMSolution(X, *b, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);

      sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   /// 11. Compute and print the L^2 norm of the error, print elapsed times
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
   delete solver;
   if (jacobi) { delete jacobi; }
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;
   if (monitor) { delete monitor; }
   if (scalar_u) { delete scalar_u; }
   if (scalar_f) { delete scalar_f; }
   if (vector_u) { delete vector_u; }
   if (vector_f) { delete vector_f; }

   return 0;
}
