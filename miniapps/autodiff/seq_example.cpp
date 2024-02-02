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
//                       MFEM AD Example - Serial Version
//
// Compile with: make seq_example
//
// Sample runs:  seq_example -m ../data/beam-quad.mesh -pp 3.5
//               seq_example -m ../data/beam-tri.mesh  -pp 4.6
//               seq_example -m ../data/beam-hex.mesh
//               seq_example -m ../data/beam-tet.mesh
//               seq_example -m ../data/beam-wedge.mesh
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
class NLSolverPLaplacian
{
public:
   /// Constructor Input: imesh - FE mesh, finite element space, power for the
   /// p-Laplacian, external load (source, input), regularization parameter
   NLSolverPLaplacian(Mesh& imesh, FiniteElementSpace& ifespace,
                      double powerp=2,
                      Coefficient* load=nullptr,
                      double regularizationp=1e-7)
   {
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

      // set the nonlinear form
      nlform=nullptr;
      lsolver=nullptr;
      prec=nullptr;
      nsolver=nullptr;

      // set the default integrator
      integ=IntegratorType::HandCodedIntegrator; // hand coded
   }

   ~NLSolverPLaplacian()
   {
      if (nlform!=nullptr) { delete nlform;}
      if (nsolver!=nullptr) { delete nsolver;}
      if (prec!=nullptr) { delete prec;}
      if (lsolver!=nullptr) { delete lsolver;}
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
      if (nsolver!=nullptr) {delete nsolver;}
      if (prec!=nullptr) {delete prec;}
      if (lsolver!=nullptr) { delete lsolver;}

      // Define the essential boundary attributes
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;

      nlform = new NonlinearForm(fespace);

      if (integ==IntegratorType::HandCodedIntegrator)
      {
         // standard hand coded integrator
         nlform->AddDomainIntegrator(new pLaplace(*plap_power,*plap_epsilon,
                                                  *plap_input));
      }
      else if (integ==IntegratorType::ADJacobianIntegrator)
      {
         // The template integrator is based on automatic differentiation. For
         // ADJacobianIntegrator the residual (vector function) at an
         // integration point is implemented as a functor by MyVFunctor. The
         // vector function has a return size of four(4), four state arguments,
         // and three(3) parameters. MyVFunctor is a template argument to the
         // actual template class performing the differentiation - in this case,
         // QVectorFuncAutoDiff. The derivatives are used in the integration
         // loop in the integrator pLaplaceAD.
         nlform->AddDomainIntegrator(new
                                     pLaplaceAD<mfem::QVectorFuncAutoDiff<MyResidualFunctor,4,4,3>>(*plap_power,
                                           *plap_epsilon,*plap_input));
      }
      else // IntegratorType::ADHessianIntegrator
      {
         // The main difference from the previous case is that the user has to
         // implement only a functional evaluation at an integration point. The
         // implementation is in MyQFunctor, which takes four state arguments
         // and three parameters. The residual vector is the first derivative of
         // the energy/functional with respect to the state variables, and the
         // Hessian is the second derivative. Automatic differentiation is used
         // for evaluating both of them.
         nlform->AddDomainIntegrator(new
                                     pLaplaceAD<mfem::QFunctionAutoDiff<MyEnergyFunctor,4,3>>(*plap_power,
                                           *plap_epsilon,*plap_input));
      }

      nlform->SetEssentialBC(ess_bdr);

#ifdef MFEM_USE_SUITESPARSE
      prec = new UMFPackSolver();
#else
      prec = new GSSmoother();
#endif

      // allocate the linear solver
      lsolver=new CGSolver();
      lsolver->SetRelTol(linear_rtol);
      lsolver->SetAbsTol(linear_atol);
      lsolver->SetMaxIter(linear_iter);
      lsolver->SetPrintLevel(print_level);
      lsolver->SetPreconditioner(*prec);

      // allocate the NR solver
      nsolver = new NewtonSolver();
      nsolver->iterative_mode = true;
      nsolver->SetSolver(*lsolver);
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

   // reference to the mesh
   Mesh* mesh;
   // reference to the fespace
   FiniteElementSpace *fespace;

   // nonlinear form for the p-laplacian
   NonlinearForm *nlform;
   CGSolver *lsolver; // linear solver
   Solver *prec; // preconditioner for the linear solver
   NewtonSolver *nsolver; // NR solver
   IntegratorType integ;

   // power of the p-laplacian
   Coefficient* plap_power;
   // regularization parameter
   Coefficient* plap_epsilon;
   // load(input) parameter
   Coefficient* plap_input;
   // flag indicating the ownership of plap_input
   bool input_ownership;
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options
   const char *mesh_file = "../../data/beam-tet.mesh";
   int ser_ref_levels = 3;
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
   StopWatch *timer = new StopWatch();
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
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
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);
   integrator = static_cast<IntegratorType>(int_integrator);

   // 2. Read the (serial) mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 4. Define the load parameter for the p-Laplacian
   ConstantCoefficient load(1.00);

   // 5. Define the finite element spaces for the solution
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(mesh, &fec, 1, Ordering::byVDIM);
   int glob_size = fespace.GetTrueVSize();

   std::cout << "Number of finite element unknowns: " << glob_size << std::endl;

   // 6. Define the solution grid function
   GridFunction x(&fespace);
   x = 0.0;

   // 7. Define the solution true vector
   Vector sv(fespace.GetTrueVSize());
   sv = 0.0;

   // 8. Define ParaView DataCollection
   ParaViewDataCollection *dacol = new ParaViewDataCollection("Example", mesh);
   dacol->SetLevelsOfDetail(order);
   dacol->RegisterField("sol", &x);

   // 9. Define the nonlinear p-Laplacian solver
   NLSolverPLaplacian* nr;

   // 10. Start with linear diffusion - solvable for any initial guess
   nr=new NLSolverPLaplacian(*mesh, fespace, 2.0, &load);
   nr->SetIntegrator(integrator);
   nr->SetMaxNRIter(newton_iter);
   nr->SetNRATol(newton_abs_tol);
   nr->SetNRRTol(newton_rel_tol);
   timer->Clear();
   timer->Start();
   nr->Solve(sv);
   timer->Stop();
   std::cout << "[pp=2] The solution time is: " << timer->RealTime()
             << std::endl;
   // Compute the energy
   double energy = nr->GetEnergy(sv);
   std::cout << "[pp=2] The total energy of the system is E=" << energy
             << std::endl;
   delete nr;
   x.SetFromTrueDofs(sv);
   dacol->SetTime(2.0);
   dacol->SetCycle(2);
   dacol->Save();


   // 11. Continue with powers higher than 2
   for (int i = 3; i < pp; i++)
   {
      nr=new NLSolverPLaplacian(*mesh, fespace, (double)i, &load);
      nr->SetIntegrator(integrator);
      nr->SetMaxNRIter(newton_iter);
      nr->SetNRATol(newton_abs_tol);
      nr->SetNRRTol(newton_rel_tol);
      timer->Clear();
      timer->Start();
      nr->Solve(sv);
      timer->Stop();
      std::cout << "[pp=" << i
                << "] The solution time is: " << timer->RealTime() << std::endl;
      energy = nr->GetEnergy(sv);
      std::cout << "[pp="<< i<<"] The total energy of the system is E=" << energy
                << std::endl;
      delete nr;
      x.SetFromTrueDofs(sv);
      dacol->SetTime(i);
      dacol->SetCycle(i);
      dacol->Save();
   }

   // 12. Continue with the final power
   if (std::abs(pp - 2.0) > std::numeric_limits<double>::epsilon())
   {
      nr=new NLSolverPLaplacian(*mesh, fespace, pp, &load);
      nr->SetIntegrator(integrator);
      nr->SetMaxNRIter(newton_iter);
      nr->SetNRATol(newton_abs_tol);
      nr->SetNRRTol(newton_rel_tol);
      timer->Clear();
      timer->Start();
      nr->Solve(sv);
      timer->Stop();
      std::cout << "[pp=" << pp
                << "] The solution time is: " << timer->RealTime() << std::endl;
      energy = nr->GetEnergy(sv);
      std::cout << "[pp="<<pp<<"] The total energy of the system is E=" << energy
                << std::endl;
      delete nr;
      x.SetFromTrueDofs(sv);
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

   // 13. Free the memory
   delete dacol;
   delete mesh;
   delete timer;
   return 0;
}
