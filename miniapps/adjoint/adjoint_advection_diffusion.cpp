// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
//          ----------------------------------------------------------
//          Advection-Diffusion Miniapp:  Parallel MFEM CVODES Example
//          ----------------------------------------------------------
//
// Compile with: make adjoint_advection_diffusion
//
// Sample runs:  adjoint_advection_diffusion -dt 0.01 -tf 2.5
//               adjoint_advection_diffusion -dt 0.005
//
// Description:  This example is a port of cvodes/parallel/cvsAdvDiff_ASAp_non_p
//               example that is part of SUNDIALS. The goal is to demonstrate
//               how to use the adjoint SUNDIALS CVODES interface with MFEM.
//               Below is an excerpt description from the aforementioned file.
//
// Example problem:
//
// The following is a simple example problem, with the program for its solution
// by CVODE. The problem is the semi-discrete form of the advection-diffusion
// equation in 1-D:
//
//   du/dt = p1 * d^2u / dx^2 + p2 * du / dx
//
// on the interval 0 <= x <= 2, and the time interval 0 <= t <= 5. Homogeneous
// Dirichlet boundary conditions are posed, and the initial condition is:
//
//   u(x,t=0) = x(2-x)exp(2x).
//
// The nominal values of the two parameters are: p1=1.0, p2=0.5. The PDE is
// discretized on a uniform grid of size MX+2 with central differencing, and
// with boundary values eliminated, leaving an ODE system of size NEQ = MX.
//
// The program solves the problem with the option for nonstiff systems: ADAMS
// method and functional iteration. It uses scalar relative and absolute
// tolerances. In addition to the solution, sensitivities with respect to p1 and
// p2 as well as with respect to initial conditions are computed for the
// quantity:
//
//    g(t, u, p) = int_x u(x,t) at t = 5
//
// These sensitivities are obtained by solving the adjoint system:
//
//    dv/dt = -p1 * d^2 v / dx^2 + p2 * dv / dx
//
// with homogeneous Dirichlet boundary conditions and the final condition:
//
//    v(x,t=5) = 1.0
//
// Then, v(x, t=0) represents the sensitivity of g(5) with respect to u(x, t=0)
// and the gradient of g(5) with respect to p1, p2 is
//
//    (dg/dp)^T = [  int_t int_x (v * d^2u / dx^2) dx dt ]
//                [  int_t int_x (v * du / dx) dx dt     ]
//
// This version uses MPI for user routines.
// Execute with number of processors = N, with 1 <= N <= MX.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

#ifndef MFEM_USE_SUNDIALS
#error This example requires that MFEM is built with MFEM_USE_SUNDIALS=YES
#endif

using namespace std;
using namespace mfem;

// Implement the adjoint rate equations in AdvDiffSUNDIALS
class AdvDiffSUNDIALS : public TimeDependentAdjointOperator
{
public:
   AdvDiffSUNDIALS(int ydot_dim, int ybdot_dim, Vector p,
                   ParFiniteElementSpace *fes, Array<int> & ess_tdof) :
      TimeDependentAdjointOperator(ydot_dim, ybdot_dim),
      p_(p),
      ess_tdof_list(ess_tdof),
      pfes(fes),
      Mf(NULL),
      M_solver(fes->GetComm())
   {
      int skip_zeros = 0;

      cout << "Essential tdofs: " << endl;
      ess_tdof_list.Print();

      m = new ParBilinearForm(pfes);
      m->AddDomainIntegrator(new MassIntegrator());
      m->Assemble(skip_zeros);
      m->Finalize(skip_zeros);

      // Define coefficients
      mp0 = new ConstantCoefficient(-p_[0]);
      p0 = new ConstantCoefficient(p_[0]);
      Vector p2vec(fes->GetParMesh()->SpaceDimension());
      p2vec = p_[1];
      p2 = new VectorConstantCoefficient(p2vec);

      k = new ParBilinearForm(pfes);
      k->AddDomainIntegrator(new DiffusionIntegrator(*mp0));
      k->AddDomainIntegrator(new ConvectionIntegrator(*p2));
      k->Assemble(skip_zeros);
      k->Finalize(skip_zeros);

      k1 = new ParBilinearForm(pfes);
      k1->AddDomainIntegrator(new DiffusionIntegrator(*p0));
      k1->AddDomainIntegrator(new ConvectionIntegrator(*p2));
      k1->Assemble(skip_zeros);
      k1->Finalize(skip_zeros);

      M = m->ParallelAssemble();
      HypreParMatrix *temp = M->EliminateRowsCols(ess_tdof_list);
      delete temp;

      K = k->ParallelAssemble();
      temp = K->EliminateRowsCols(ess_tdof_list);
      delete temp;

      K_adj = k1->ParallelAssemble();
      temp = K_adj->EliminateRowsCols(ess_tdof_list);
      delete temp;

      M_prec.SetType(HypreSmoother::Jacobi);
      M_solver.SetPreconditioner(M_prec);
      M_solver.SetOperator(*M);

      M_solver.SetRelTol(1e-14);
      M_solver.SetAbsTol(0.0);
      M_solver.SetMaxIter(1000);
      M_solver.SetPrintLevel(0);

   }

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void AdjointRateMult(const Vector &y, Vector &yB,
                                Vector &yBdot) const;

   virtual int SUNImplicitSetup(const Vector &y,
                                const Vector &fy, int jok, int *jcur,
                                double gamma);

   virtual int SUNImplicitSolve(const Vector &b, Vector &x, double tol);

   virtual void QuadratureSensitivityMult(const Vector &y, const Vector &yB,
                                          Vector &qbdot) const;

   ~AdvDiffSUNDIALS()
   {
      delete m;
      delete k;
      delete k1;
      delete M;
      delete K;
      delete K_adj;
      delete Mf;
      delete p0;
      delete mp0;
      delete p2;
   }

protected:
   Vector p_;
   Array<int> ess_tdof_list;
   ParFiniteElementSpace *pfes;

   // Internal matrices
   ParBilinearForm *m;
   ParBilinearForm *k;
   ParBilinearForm *k1;

   HypreParMatrix *M;
   HypreParMatrix *K;
   HypreParMatrix *K_adj;

   HypreParMatrix *Mf;

   CGSolver M_solver;
   HypreSmoother M_prec;
   ConstantCoefficient *p0;
   ConstantCoefficient *mp0;
   VectorConstantCoefficient *p2;
};

// Initial conditions for the problem
double u_init(const Vector &x)
{
   return x[0]*(2. - x[0])*exp(2.*x[0]);
}

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   double t_final = 2.5;
   double dt = 0.01;
   int mx = 20;
   bool step_mode = true;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);

   args.AddOption(&mx, "-m", "--mx", "The number of mesh elements in the x-dir");
   args.AddOption(&ser_ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&step_mode, "-a", "--adams", "-no-a","--no-adams",
                  "A switch to toggle between CV_ADAMS, and CV_BDF stepping modes");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Create a small 1D mesh with a length of 2. This mesh corresponds with the
   // cvsAdvDiff_ASA_p_non_p example.
   Mesh *mesh = new Mesh(mx+1, 2.);

   // Refine the mesh to increase the resolution. In this example we do
   // 'ref_levels' of uniform refinement, where 'ref_levels' is a
   // command-line parameter. If the mesh is of NURBS type, we convert it to
   // a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // Finite Element Spaces
   H1_FECollection fec(1, pmesh->SpaceDimension());
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   // Set up material properties, and primal and adjoint variables
   // p are the fixed material properties
   Vector p(2);
   p[0] = 1.0;
   p[1] = 0.5;

   // U is the size of the solution/primal vector
   // Set U with the initial conditions
   ParGridFunction u(fes);
   FunctionCoefficient u0(u_init);
   u.ProjectCoefficient(u0);

   cout << "Init u: " << endl;
   u.Print();

   // TimeDependentOperators need to be TrueDOF Size
   HypreParVector *U = u.GetTrueDofs();

   // Get boundary conditions
   Array<int> ess_tdof_list;
   Array<int> essential_attr(pmesh->bdr_attributes.Size());
   essential_attr[0] = 1;
   essential_attr[1] = 1;
   fes->GetEssentialTrueDofs(essential_attr, ess_tdof_list);

   // Setup the TimeDependentAdjointOperator and the CVODESSolver
   AdvDiffSUNDIALS adv(U->Size(), U->Size(), p, fes, ess_tdof_list);

   // Set the initial time to the TimeDependentAdjointOperator
   double t = 0.0;
   adv.SetTime(t);

   // Create the CVODES solver corresponding to the selected step method
   CVODESSolver *cvodes = new CVODESSolver(fes->GetComm(),
                                           step_mode ? CV_ADAMS : CV_BDF);
   cvodes->Init(adv);
   cvodes->UseSundialsLinearSolver();
   cvodes->SetMaxNSteps(5000);

   // Relative and absolute tolerances for CVODES
   double reltol = 1e-8, abstol = 1e-6;
   cvodes->SetSStolerances(reltol, abstol);

   // Initialize adjoint problem settings
   int checkpoint_steps = 50; // steps between checkpoints
   cvodes->InitAdjointSolve(checkpoint_steps, CV_HERMITE);

   // Perform time-integration for the problem (looping over the time
   // iterations, ti, with a time-step dt).
   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = max(dt, t_final - t);
      cvodes->Step(*U, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done && myid == 0)
      {
         cvodes->PrintInfo();
      }
   }

   u = *U;
   if (myid == 0)
   {
      cout << "Final Solution: " << t << endl;
   }

   cout << "u (" << myid << "):" << endl;
   u.Print();
   cout << flush;
   MPI_Barrier(MPI_COMM_WORLD);

   // Calculate the quadrature int_x u dx at t = 5
   // Since it's only a spatial quadrature we evaluate it at t=5
   ParLinearForm obj(fes);
   ConstantCoefficient one(1.0);
   obj.AddDomainIntegrator(new DomainLFIntegrator(one));
   obj.Assemble();

   double g = obj(u);
   if (myid == 0)
   {
      cout << "g: " << g << endl;
   }

   // Solve the adjoint problem. v is the adjoint solution
   ParGridFunction v(fes);
   v = 1.;
   v.SetSubVector(ess_tdof_list, 0.0);
   HypreParVector *V = v.GetTrueDofs();

   // Initialize quadrature sensitivity values to zero
   Vector qBdot(p.Size());
   qBdot = 0.;

   t = t_final;
   cvodes->InitB(adv);
   cvodes->InitQuadIntegrationB(qBdot, 1.e-6, 1.e-6);
   cvodes->UseSundialsLinearSolverB();
   cvodes->SetSStolerancesB(reltol, abstol);

   // Results at time TBout1
   double dt_real = max(dt, t);
   cvodes->StepB(*V, t, dt_real);
   if (myid == 0)
   {
      cout << "t: " << t << endl;
   }

   cout << "v (" << myid << "):" << endl;
   V->Vector::Print();
   cout << flush;
   MPI_Barrier(MPI_COMM_WORLD);

   // Evaluate the sensitivity
   cvodes->EvalQuadIntegrationB(t, qBdot);

   MPI_Barrier(MPI_COMM_WORLD);
   if (myid == 0)
   {
      cout << "sensitivity:" << endl;
      qBdot.Print();
   }

   // Free the used memory.
   delete fes;
   delete pmesh;
   delete U;
   delete V;
   delete cvodes;

   MPI_Finalize();
   return 0;
}

// AdvDiff rate equation
void AdvDiffSUNDIALS::Mult(const Vector &x, Vector &y) const
{
   Vector z(x.Size());
   Vector x1(x);

   // Set boundary conditions to zero
   x1.SetSubVector(ess_tdof_list, 0.0);

   K->Mult(x1, z);

   y = 0.;
   M_solver.Mult(z, y);
}

// AdvDiff Rate equation setup
int AdvDiffSUNDIALS::SUNImplicitSetup(const Vector &y,
                                      const Vector &fy,
                                      int jok, int *jcur, double gamma)
{
   // Mf = M(I - gamma J) = M - gamma * M * J
   // J = df/dy => K
   *jcur = 1; // We've updated the jacobian

   delete Mf;
   Mf = Add(1., *M, -gamma, *K);
   HypreParMatrix *temp = Mf->EliminateRowsCols(ess_tdof_list);
   delete temp;
   return 0;
}

// AdvDiff Rate equation solve
int AdvDiffSUNDIALS::SUNImplicitSolve(const Vector &b, Vector &x, double tol)
{
   Vector z(b.Size());
   M->Mult(b,z);

   CGSolver solver(pfes->GetComm());
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi);
   solver.SetPreconditioner(prec);
   solver.SetOperator(*Mf);
   solver.SetRelTol(1E-14);
   solver.SetMaxIter(1000);

   solver.Mult(z, x);

   return (0);
}

// AdvDiff adjoint rate equation
void AdvDiffSUNDIALS::AdjointRateMult(const Vector &y, Vector & yB,
                                      Vector &yBdot) const
{
   Vector z(yB.Size());

   // Set boundary conditions to zero
   yB.SetSubVector(ess_tdof_list, 0.0);
   K_adj->Mult(yB, z);
   M_solver.Mult(z, yBdot);
}

// AdvDiff quadrature sensitivity rate equation
void AdvDiffSUNDIALS::QuadratureSensitivityMult(const Vector &y,
                                                const Vector &yB,
                                                Vector &qBdot) const
{
   // Now we have both the adjoint, yB, and y, at the same point in time
   // We calculate
   /*
    * to u(x, t=0) and the gradient of g(5) with respect to p1, p2 is
    *    (dg/dp)^T = [  int_t int_x (v * d^2u / dx^2) dx dt ]
    *                [  int_t int_x (v * du / dx) dx dt     ]
    */

   ParBilinearForm dp1(pfes);
   ConstantCoefficient mone(-1.);
   dp1.AddDomainIntegrator(new DiffusionIntegrator(mone));
   dp1.Assemble();
   dp1.Finalize();

   HypreParMatrix * dP1 = dp1.ParallelAssemble();
   HypreParMatrix *temp = dP1->EliminateRowsCols(ess_tdof_list);
   delete temp;

   Vector b1(y.Size());
   dP1->Mult(y, b1);
   delete dP1;

   ParBilinearForm dp2(pfes);
   Vector p2vec(pfes->GetParMesh()->SpaceDimension()); p2vec = 1.;
   VectorConstantCoefficient dp2_coef(p2vec);
   dp2.AddDomainIntegrator(new ConvectionIntegrator(dp2_coef));
   dp2.Assemble();
   dp2.Finalize();

   HypreParMatrix * dP2 = dp2.ParallelAssemble();
   temp = dP2->EliminateRowsCols(ess_tdof_list);
   delete temp;

   Vector b2(y.Size());
   dP2->Mult(y, b2);
   delete dP2;

   double dp1_result = InnerProduct(pfes->GetComm(), yB, b1);
   double dp2_result = InnerProduct(pfes->GetComm(), yB, b2);

   qBdot[0] = -dp1_result;
   qBdot[1] = -dp2_result;
}
