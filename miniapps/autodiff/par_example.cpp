// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
//                       MFEM AD Example  - Parallel Version
//
// Compile with: make par_example
//
// Sample runs:
//   mpirun -np 2 paradiff -m ../data/beam-quad.mesh -pp 3.8
//   mpirun -np 2 paradiff -m ../data/beam-tri.mesh  -pp 7.2
//   mpirun -np 2 paradiff -m ../data/beam-hex.mesh
//   mpirun -np 2 paradiff -m ../data/beam-tet.mesh
//   mpirun -np 2 paradiff -m ../data/beam-wedge.mesh
//
// Description:  This examples solves a quasi-static nonlinear
//               p-Laplacian problem with zero Dirichlet boundary
//               conditions applied on all defined boundaries
//
//           The example demonstrates the use of nonlinear operators combined
//           with automatic differentiation (AD). The definitions of the
//           integrators are written in the example.hpp.  Selecting integrator=0
//           will use the manually implemented integrator.  Selecting
//           integrator=1,2 will utilize one of the AD integrators.
//
//        We recommend viewing examples 1 and 19, before playing with this
//        example.

#include "example.hpp"

using namespace mfem;

///Non-linear solver for the p-Laplacian problem.
class ParNLSolverPLaplacian
{
public:
   ///Constructor Input: imesh - FE mesh, finite element space,
   /// power for the p-Laplacian, external load (source, input),
   /// regularization parameter
   ParNLSolverPLaplacian(MPI_Comm comm, ParMesh& imesh,
                         ParFiniteElementSpace& ifespace,
                         double powerp=2,
                         Coefficient* load=nullptr,
                         double regularizationp=1e-7)
   {
      lcomm = comm;

      //default parameters for
      //the Newton solver
      newton_rtol = 1e-4;
      newton_atol = 1e-8;
      newton_iter = 10;

      //linear solver
      linear_rtol = 1e-7;
      linear_atol = 1e-15;
      linear_iter = 500;

      print_level = 0;

      //set the mesh
      mesh=&imesh;

      //set the fespace
      fespace=&ifespace;

      //set the parameters
      plap_epsilon=new ConstantCoefficient(regularizationp);
      plap_power=new ConstantCoefficient(powerp);
      if (load==nullptr)
      {
         plap_input=new ConstantCoefficient(1.0);
         input_ownership=true;
      }
      else
      {
         plap_input=load;
         input_ownership=false;
      }

      nf=nullptr;
      ns=nullptr;
      gmres=nullptr;
      prec=nullptr;

      //set the default integrator
      integ=0; //hand coded
   }

   ~ParNLSolverPLaplacian()
   {
      if (nf!=nullptr) { delete nf;}
      if (ns!=nullptr) { delete ns;}
      if (prec!=nullptr) { delete prec;}
      if (gmres!=nullptr) { delete gmres;}
      if (input_ownership) { delete plap_input;}
      delete plap_epsilon;
      delete plap_power;
   }

   ///Set the integrator.
   /// 0 - hand coded, 1 - AD based (compute only Heassian by AD),
   /// 2 - AD based (compute residual and Hessian by AD)
   void SetIntegrator(int intr)
   {
      integ=intr;
   }

   //set relative tolerance for the Newton solver
   void SetNRRTol(double rtol)
   {
      newton_rtol=rtol;
   }

   //set absolute tolerance for the Newton solver
   void SetNRATol(double atol)
   {
      newton_atol=atol;
   }

   //set max iterations for the NR solver
   void SetMaxNRIter(int miter)
   {
      newton_iter=miter;
   }

   void SetLSRTol(double rtol)
   {
      linear_rtol=rtol;
   }

   void SetLSATol(double atol)
   {
      linear_atol=atol;
   }

   //set max iterations for the linear solver
   void SetMaxLSIter(int miter)
   {
      linear_iter=miter;
   }

   //set the print level
   void SetPrintLevel(int plev)
   {
      print_level=plev;
   }

   ///The state vector is used as initial condition for the NR solver.
   /// On return the statev holds the solution to the problem.
   void Solve(Vector& statev)
   {
      if (nf==nullptr)
      {
         AllocSolvers();
      }
      Vector b; //RHS is zero
      ns->Mult(b, statev);
   }

   ///Compute the energy
   double GetEnergy(Vector& statev)
   {
      if (nf==nullptr)
      {
         //allocate the solvers
         AllocSolvers();
      }
      return nf->GetEnergy(statev);
   }

private:
   void AllocSolvers()
   {
      if (nf!=nullptr) { delete nf;}
      if (ns!=nullptr) { delete ns;}
      if (gmres!=nullptr) { delete gmres;}
      if (prec!=nullptr) { delete prec;}

      // Define the essential boundary attributes
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;

      nf = new ParNonlinearForm(fespace);
      if (integ==0)
      {
         nf->AddDomainIntegrator(new pLaplace(*plap_power,*plap_epsilon,*plap_input));
      }
      else if (integ==1)
      {
         nf->AddDomainIntegrator(new pLaplaceAD<mfem::QVectorFuncAutoDiff<MyVFunctor,4,4,3>>(*plap_power,*plap_epsilon,*plap_input));
      }
      else if (integ==2)
      {
         nf->AddDomainIntegrator(new pLaplaceAD<mfem::QFunctionAutoDiff<MyQFunctor,4,3>>(*plap_power,*plap_epsilon,*plap_input));
      }

      nf->SetEssentialBC(ess_bdr);

      prec = new HypreBoomerAMG();
      prec->SetPrintLevel(print_level);

      gmres = new GMRESSolver(lcomm);
      gmres->SetAbsTol(linear_atol);
      gmres->SetRelTol(linear_rtol);
      gmres->SetMaxIter(linear_iter);
      gmres->SetPrintLevel(print_level);
      gmres->SetPreconditioner(*prec);

      ns = new NewtonSolver(lcomm);

      ns->iterative_mode = true;
      ns->SetSolver(*gmres);
      ns->SetOperator(*nf);
      ns->SetPrintLevel(print_level);
      ns->SetRelTol(newton_rtol);
      ns->SetAbsTol(newton_atol);
      ns->SetMaxIter(newton_iter);
   }

   double newton_rtol;
   double newton_atol;
   int newton_iter;

   double linear_rtol;
   double linear_atol;
   int linear_iter;

   int print_level;

   //power of the p-laplacian
   Coefficient* plap_power;
   //regularization parammeter
   Coefficient* plap_epsilon;
   //load(input) paramater
   Coefficient* plap_input;
   bool input_ownership;

   MPI_Comm lcomm;

   ParMesh *mesh;
   ParFiniteElementSpace *fespace;

   ParNonlinearForm *nf;

   HypreBoomerAMG *prec;
   GMRESSolver *gmres;
   NewtonSolver *ns;
   int integ;

};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI
   int num_procs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   // Define Caliper ConfigManager
#ifdef MFEM_USE_CALIPER
   cali::ConfigManager mgr;
#endif
   // Caliper instrumentation
   MFEM_PERF_FUNCTION;

   // 2. Parse command-line options
   const char *mesh_file = "../../data/beam-tet.mesh";
   int ser_ref_levels = 3;
   int par_ref_levels = 1;
   int order = 1;
   bool visualization = true;
   double newton_rel_tol = 1e-4;
   double newton_abs_tol = 1e-6;
   int newton_iter = 10;
   int print_level = 0;
   double pp = 2.0;
   int integrator = 2; //use AD
   const char* cali_config = "runtime-report";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&newton_rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter,
                  "-it",
                  "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.AddOption(&pp,
                  "-pp",
                  "--power-parameter",
                  "Power parameter (>=2.0) for the p-Laplacian.");
   args.AddOption((&print_level), "-prt", "--print-level", "Print level.");
   args.AddOption(&integrator,
                  "-int",
                  "--integrator",
                  "Integrator 0: standard; 1: AD for Hessian; 2: AD for residual and Hessian");
   args.AddOption(&cali_config, "-p", "--caliper",
                  "Caliper configuration string.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   StopWatch *timer = new StopWatch();

   // Caliper configuration
#ifdef MFEM_USE_CALIPER
   mgr.add(cali_config);
   mgr.start();
#endif
   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define the load for the p-Laplacian
   ConstantCoefficient load(1.00);

   // 7. Define the finite element spaces for the solution
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fec, 1, Ordering::byVDIM);
   HYPRE_Int glob_size = fespace.GlobalTrueVSize();
   if (myrank == 0)
   {
      std::cout << "Number of finite element unknowns: " << glob_size
                << std::endl;
   }

   // 8. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;
   HypreParVector *sv = x.GetTrueDofs();

   // 9. Define ParaView DataCollection
   ParaViewDataCollection *dacol = new ParaViewDataCollection("Example",
                                                              pmesh);
   dacol->SetLevelsOfDetail(order);
   dacol->RegisterField("sol", &x);

   // 10. Define the NR solver
   ParNLSolverPLaplacian* nr;

   // 11. Start with linear diffusion - solvable for any initial guess
   nr=new ParNLSolverPLaplacian(MPI_COMM_WORLD,*pmesh, fespace, 2.0, &load);
   nr->SetIntegrator(integrator);
   nr->SetMaxNRIter(newton_iter);
   nr->SetNRATol(newton_abs_tol);
   nr->SetNRRTol(newton_rel_tol);
   nr->SetPrintLevel(print_level);
   timer->Clear();
   timer->Start();
   nr->Solve(*sv);
   timer->Stop();
   if (myrank==0)
   {
      std::cout << "[pp=2] The solution time is: " << timer->RealTime()
                << std::endl;
   }
   // Compute the energy
   double energy = nr->GetEnergy(*sv);
   if (myrank==0)
   {
      std::cout << "[pp=2] The total energy of the system is E=" << energy
                << std::endl;
   }
   delete nr;
   x.SetFromTrueDofs(*sv);
   dacol->SetTime(2.0);
   dacol->SetCycle(2);
   dacol->Save();


   // 12. Continue with powers higher than 2
   for (int i = 3; i < pp; i++)
   {
      nr=new ParNLSolverPLaplacian(MPI_COMM_WORLD,*pmesh, fespace, (double)i, &load);
      nr->SetIntegrator(integrator);
      nr->SetMaxNRIter(newton_iter);
      nr->SetNRATol(newton_abs_tol);
      nr->SetNRRTol(newton_rel_tol);
      nr->SetPrintLevel(print_level);
      timer->Clear();
      timer->Start();
      nr->Solve(*sv);
      timer->Stop();
      if (myrank==0)
      {
         std::cout << "[pp="<<i<<"] The solution time is: " << timer->RealTime()
                   << std::endl;
      }
      // Compute the energy
      double energy = nr->GetEnergy(*sv);
      if (myrank==0)
      {
         std::cout << "[pp="<<i<<"] The total energy of the system is E=" << energy
                   << std::endl;
      }
      delete nr;
      x.SetFromTrueDofs(*sv);
      dacol->SetTime((double)i);
      dacol->SetCycle(i);
      dacol->Save();
   }

   // 13. Continue with the final power
   if (std::abs(pp - 2.0) > std::numeric_limits<double>::epsilon())
   {
      nr=new ParNLSolverPLaplacian(MPI_COMM_WORLD,*pmesh, fespace, pp, &load);
      nr->SetIntegrator(integrator);
      nr->SetMaxNRIter(newton_iter);
      nr->SetNRATol(newton_abs_tol);
      nr->SetNRRTol(newton_rel_tol);
      nr->SetPrintLevel(print_level);
      timer->Clear();
      timer->Start();
      nr->Solve(*sv);
      timer->Stop();
      if (myrank==0)
      {
         std::cout << "[pp="<<pp<<"] The solution time is: " << timer->RealTime()
                   << std::endl;
      }
      // Compute the energy
      double energy = nr->GetEnergy(*sv);
      if (myrank==0)
      {
         std::cout << "[pp="<<pp<<"] The total energy of the system is E=" << energy
                   << std::endl;
      }
      delete nr;
      x.SetFromTrueDofs(*sv);
      dacol->SetTime(pp);
      if (pp < 2.0)
      {
         dacol->SetCycle(std::floor(pp));
      }
      else
      {
         dacol->SetCycle(std::ceil(pp));
      }
      dacol->Save();
   }

   // 14. Free the used memory
   delete dacol;
   delete sv;
   delete pmesh;
   delete timer;

   // Flush output before MPI_finalize
#ifdef MFEM_USE_CALIPER
   mgr.flush();
#endif
   MPI_Finalize();

   return 0;
}
