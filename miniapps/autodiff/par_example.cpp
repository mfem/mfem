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
//                       MFEM AD Example  - Parallel Version
//
// Compile with: make par_example
//
// Sample runs:  mpirun -np 2 par_example -m ../data/beam-quad.mesh -pp 3.8
//               mpirun -np 2 par_example -m ../data/beam-tri.mesh  -pp 7.2
//               mpirun -np 2 par_example -m ../data/beam-hex.mesh
//               mpirun -np 2 par_example -m ../data/beam-tet.mesh
//               mpirun -np 2 par_example -m ../data/beam-wedge.mesh
//
// Description:  This examples solves a quasi-static nonlinear p-Laplacian
//               problem with zero Dirichlet boundary conditions applied on all
//               defined boundaries
//
//               The example demonstrates the use of nonlinear operators
//               combined with automatic differentiation (AD). The integrators
//               are defined in example.hpp. Selecting integrator = 0 will use
//               the manually implemented integrator. Selecting integrator = 1
//               or 2 will utilize one of the AD integrators.
//
//               We recommend viewing examples 1 and 19, before viewing this
//               example.

#include "example.hpp"

using namespace mfem;

enum IntegratorType
{
   HandCodedIntegrator  = 0,
   ADJacobianIntegrator = 1,
   ADHessianIntegrator  = 2
};

/// Non-linear solver for the p-Laplacian problem.
class ParNLSolverPLaplacian
{
public:
   /// Constructor Input: imesh - FE mesh, finite element space, power for the
   /// p-Laplacian, external load (source, input), regularization parameter
   ParNLSolverPLaplacian(MPI_Comm comm, ParMesh& imesh,
                         ParFiniteElementSpace& ifespace,
                         double powerp=2,
                         Coefficient* load=nullptr,
                         double regularizationp=1e-7)
   {
      lcomm = comm;

      // default parameters for the Newton solver
      newton_rtol = 1e-4;
      newton_atol = 1e-8;
      newton_iter = 10;

      // linear solver
      linear_rtol = 1e-7;
      linear_atol = 1e-15;
      linear_iter = 500;

      print_level = 0;

      // set the mesh
      mesh=&imesh;

      // set the fespace
      fespace=&ifespace;

      // set the parameters
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

      nlform=nullptr;
      nsolver=nullptr;
      gmres=nullptr;
      prec=nullptr;

      // set the default integrator
      integ=IntegratorType::HandCodedIntegrator;
   }

   ~ParNLSolverPLaplacian()
   {
      delete nlform;
      delete nsolver;
      delete prec;
      delete gmres;
      if (input_ownership) { delete plap_input;}
      delete plap_epsilon;
      delete plap_power;
   }

   /// Set the integrator.
   /// 0 - hand coded, 1 - AD based (compute only Hessian by AD),
   /// 2 - AD based (compute residual and Hessian by AD)
   void SetIntegrator(IntegratorType intr)
   {
      integ=intr;
   }

   // set relative tolerance for the Newton solver
   void SetNRRTol(double rtol)
   {
      newton_rtol=rtol;
   }

   // set absolute tolerance for the Newton solver
   void SetNRATol(double atol)
   {
      newton_atol=atol;
   }

   // set max iterations for the NR solver
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

   // set max iterations for the linear solver
   void SetMaxLSIter(int miter)
   {
      linear_iter=miter;
   }

   // set the print level
   void SetPrintLevel(int plev)
   {
      print_level=plev;
   }

   /// The state vector is used as initial condition for the NR solver. On
   /// return the statev holds the solution to the problem.
   void Solve(Vector& statev)
   {
      if (nlform==nullptr)
      {
         AllocSolvers();
      }
      Vector b; // RHS is zero
      nsolver->Mult(b, statev);
   }

   /// Compute the energy
   double GetEnergy(Vector& statev)
   {
      if (nlform==nullptr)
      {
         // allocate the solvers
         AllocSolvers();
      }
      return nlform->GetEnergy(statev);
   }

private:
   void AllocSolvers()
   {
      if (nlform!=nullptr) { delete nlform;}
      if (nsolver!=nullptr) { delete nsolver;}
      if (gmres!=nullptr) { delete gmres;}
      if (prec!=nullptr) { delete prec;}

      // Define the essential boundary attributes
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;

      nlform = new ParNonlinearForm(fespace);
      if (integ==IntegratorType::HandCodedIntegrator)
      {
         nlform->AddDomainIntegrator(new pLaplace(*plap_power,*plap_epsilon,
                                                  *plap_input));
      }
      else if (integ==IntegratorType::ADJacobianIntegrator)
      {
         // The template integrator is based on automatic differentiation. For
         // ADJacobianIntegrator the residual (vector function) at an
         // integration point is implemented as a functor by MyResidualFunctor.
         // The vector function has a return size of four(4), four state
         // arguments, and three(3) parameters. MyResidualFunctor is a template
         // argument to the actual template class performing the differentiation
         // - in this case, QVectorFuncAutoDiff. The derivatives are used in the
         // integration loop in the integrator pLaplaceAD.
         nlform->AddDomainIntegrator(new
                                     pLaplaceAD<mfem::QVectorFuncAutoDiff<MyResidualFunctor,4,4,3>>(*plap_power,
                                           *plap_epsilon,*plap_input));
      }
      else if (integ==IntegratorType::ADHessianIntegrator)
      {
         // The main difference from the previous case is that the user has to
         // implement only a functional evaluation at an integration point. The
         // implementation is in MyEnergyFunctor, which takes four state
         // arguments and three parameters. The residual vector is the first
         // derivative of the energy/functional with respect to the state
         // variables, and the Hessian is the second derivative. Automatic
         // differentiation is used for evaluating both of them.
         nlform->AddDomainIntegrator(new
                                     pLaplaceAD<mfem::QFunctionAutoDiff<MyEnergyFunctor,4,3>>(*plap_power,
                                           *plap_epsilon,*plap_input));
      }

      nlform->SetEssentialBC(ess_bdr);

      prec = new HypreBoomerAMG();
      prec->SetPrintLevel(print_level);

      gmres = new GMRESSolver(lcomm);
      gmres->SetAbsTol(linear_atol);
      gmres->SetRelTol(linear_rtol);
      gmres->SetMaxIter(linear_iter);
      gmres->SetPrintLevel(print_level);
      gmres->SetPreconditioner(*prec);

      nsolver = new NewtonSolver(lcomm);

      nsolver->iterative_mode = true;
      nsolver->SetSolver(*gmres);
      nsolver->SetOperator(*nlform);
      nsolver->SetPrintLevel(print_level);
      nsolver->SetRelTol(newton_rtol);
      nsolver->SetAbsTol(newton_atol);
      nsolver->SetMaxIter(newton_iter);
   }

   double newton_rtol;
   double newton_atol;
   int newton_iter;

   double linear_rtol;
   double linear_atol;
   int linear_iter;

   int print_level;

   // power of the p-laplacian
   Coefficient* plap_power;
   // regularization parameter
   Coefficient* plap_epsilon;
   // load(input) parameter
   Coefficient* plap_input;
   // flag indicating the ownership of plap_input
   bool input_ownership;

   MPI_Comm lcomm;

   ParMesh *mesh;
   ParFiniteElementSpace *fespace;

   ParNonlinearForm *nlform;

   HypreBoomerAMG *prec;
   GMRESSolver *gmres;
   NewtonSolver *nsolver;
   IntegratorType integ;

};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myrank = Mpi::WorldRank();
   Hypre::Init();
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

   double pp = 2.0; // p-Laplacian power

   IntegratorType integrator = IntegratorType::ADHessianIntegrator;
   int int_integrator = integrator;
   // HandCodedIntegrator  = 0 - do not use AD (hand coded)
   // ADJacobianIntegrator = 1 - use AD for Hessian only
   // ADHessianIntegrator  = 2 - use AD for Residual and Hessian

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
   args.AddOption(&int_integrator,
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
      return 1;
   }
   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }
   integrator = static_cast<IntegratorType>(int_integrator);

   StopWatch *timer = new StopWatch();

   // Caliper configuration
#ifdef MFEM_USE_CALIPER
   mgr.add(cali_config);
   mgr.start();
#endif
   // 3. Read the (serial) mesh from the given mesh file on all processors. We
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
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
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
      energy = nr->GetEnergy(*sv);
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
      energy = nr->GetEnergy(*sv);
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
         dacol->SetCycle(static_cast<int>(std::floor(pp)));
      }
      else
      {
         dacol->SetCycle(static_cast<int>(std::ceil(pp)));
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

   return 0;
}
