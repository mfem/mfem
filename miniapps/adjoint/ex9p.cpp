// This is a 2D analog of the AdvDiff_ASA_p_non_p.c SUNDIALS CVODES example

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

#ifndef MFEM_USE_SUNDIALS
#error This example requires that MFEM is built with MFEM_USE_SUNDIALS=YES
#endif

using namespace std;
using namespace mfem;

/** Reimplement AdvDiff problem here */
class AdvDiffSUNDIALS : public TimeDependentAdjointOperator
{
public:
   AdvDiffSUNDIALS(int ydot_dim, int ybdot_dim, Vector p,
                   ParFiniteElementSpace *fes) :
      TimeDependentAdjointOperator(ydot_dim, ybdot_dim),
      p_(p),
      M(NULL), K(NULL), K_adj(NULL),
      Mf(NULL), MK(NULL),
      m(NULL), k(NULL),
      pfes(fes),
      M_solver(fes->GetComm())
   {
      int skip_zeros = 0;
      ParMesh * pmesh = pfes->GetParMesh();

      // Boundary conditions for this problem
      Array<int> essential_attr(pmesh->bdr_attributes.Size());
      essential_attr[0] = 1;
      essential_attr[1] = 1;

      fes->GetEssentialTrueDofs(essential_attr, ess_tdof_list);

      cout << "Essential tdofs: " << endl;
      ess_tdof_list.Print();

      m = new ParBilinearForm(pfes);
      m->AddDomainIntegrator(new MassIntegrator());
      m->Assemble(skip_zeros);
      m->Finalize(skip_zeros);

      k = new ParBilinearForm(pfes);
      k->AddDomainIntegrator(
	   new DiffusionIntegrator(
		  *(new ConstantCoefficient(-p_[0]))));
      Vector p2(fes->GetParMesh()->SpaceDimension());
      p2 = p_[1];
      k->AddDomainIntegrator(
	   new ConvectionIntegrator(
		  *(new VectorConstantCoefficient(p2))));
      k->Assemble(skip_zeros);
      k->Finalize(skip_zeros);

      k1 = new ParBilinearForm(pfes);
      k1->AddDomainIntegrator(
	   new DiffusionIntegrator(
		  *(new ConstantCoefficient(p_[0]))));
      k1->AddDomainIntegrator(
	   new ConvectionIntegrator(
		  *(new VectorConstantCoefficient(p2))));
      k1->Assemble(skip_zeros);
      k1->Finalize(skip_zeros);

      M = m->ParallelAssemble();
      M->EliminateRowsCols(ess_tdof_list);

      K = k->ParallelAssemble();
      K->EliminateRowsCols(ess_tdof_list);

      K_adj = k1->ParallelAssemble();
      K_adj->EliminateRowsCols(ess_tdof_list);

      MK = ParMult(M, K);

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


protected:
   Vector p_;
   Array<int> ess_tdof_list;
   ParFiniteElementSpace *pfes;

   // Internal matrices
   ParBilinearForm * m;
   ParBilinearForm * k;
   ParBilinearForm * k1;

   HypreParMatrix *M;
   HypreParMatrix *K;
   HypreParMatrix *K_adj;

   HypreParMatrix *Mf;
   HypreParMatrix *MK;
   HypreParMatrix *I;

   CGSolver M_solver;
   HypreSmoother M_prec;
};

double u_init(const Vector &x)
{
   return x[0]*(2. - x[0])*exp(2.*x[0]);
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Parse command-line options.
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 1;
   int ode_solver_type = 4;
   double t_final = 2.5;
   double dt = 0.01;
   int mx = 20;

   // Relative and absolute tolerances for CVODES
   double reltol = 1e-8, abstol = 1e-6;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);

   args.AddOption(&ser_ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - CVODES (adaptive order) implicit Adams,\n\t"
                  "            2 - ARKODE default (4th order) explicit,\n\t"
                  "            3 - ARKODE RK8.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   // check for vaild ODE solver option
   if (ode_solver_type < 1 || ode_solver_type > 4)
   {
      cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      return 3;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   //   Mesh *mesh = new Mesh(mx+1,1,Element::QUADRILATERAL, true, 2.0, 1.0);
   Mesh *mesh = new Mesh(mx+1, 2.);
   int dim = 2;

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
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

   //6. Finite Element Spaces

   H1_FECollection fec(1, dim);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   // 7. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and define the ODE solver used for time integration.

   Vector p(2);
   p[0] = 1.0;
   p[1] = 0.5;

   // u is the size of the solution vector
   ParGridFunction u(fes);
   FunctionCoefficient u0(u_init);
   u.ProjectCoefficient(u0);

   cout << "Init u: " << endl;
   u.Print();

   // TimeDependentOperators need to be TrueDOF Size
   HypreParVector *U = u.GetTrueDofs();

   int adj_size = U->Size() + (myid == 0 ? p.Size() : 0);

   AdvDiffSUNDIALS adv(U->Size(), adj_size, p, fes);

   double t = 0.0;
   adv.SetTime(t);

   // Create the time integrator
   ODESolver *ode_solver = NULL;
   CVODESolver *cvode = NULL;
   CVODESSolver *cvodes = NULL;
   ARKStepSolver *arkode = NULL;

   int steps = 50;

   Array<int> ess_tdof_list;

   Array<int> essential_attr(pmesh->bdr_attributes.Size());
   essential_attr[0] = 1;
   essential_attr[1] = 1;

   fes->GetEssentialTrueDofs(essential_attr, ess_tdof_list);


   switch (ode_solver_type)
   {
      case 4:
         cvodes = new CVODESSolver(fes->GetComm(), CV_ADAMS);
         cvodes->Init(adv);
         cvodes->UseSundialsLinearSolver();
         cvodes->SetSStolerances(reltol, abstol);
         cvodes->InitAdjointSolve(steps, CV_HERMITE);
         ode_solver = cvodes; break;
   }

   // 8. Perform time-integration (looping over the time iterations, ti,
   //    with a time-step dt).
   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = max(dt, t_final - t);
      ode_solver->Step(*U, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done )
      {
         if (cvodes) { cvodes->PrintInfo(); }

      }
   }

   u = *U;
   cout << "Final Solution: " << t << endl;
   u.Print();

   // Calculate int_x u dx at t = 5

   ParLinearForm obj(fes);
   ConstantCoefficient one(1.0);
   obj.AddDomainIntegrator(new DomainLFIntegrator(one));
   obj.Assemble();

   double g = obj(u);
   if (myid == 0)
   {
      cout << "g: " << g << endl;
   }

   if (cvodes)
   {

      // backward portion of the problem
      ParGridFunction v(fes);
      v = 1.;
      v.SetSubVector(ess_tdof_list, 0.0);
      HypreParVector *V = v.GetTrueDofs();

      // Add additional space for integrated parameter
      Vector V_final(adj_size);
      for (int i = 0; i < v.Size(); i++)
      {
         V_final[i] = v[i];
      }

      if (myid == 0 )
      {
         for (int i = 0 ; i < p.Size(); i++)
         {
            V_final[adj_size - p.Size() + i] = 0.;
         }
      }

      V_final.Print();

      t = t_final;
      cvodes->InitB(adv);
      cvodes->UseSundialsLinearSolverB();
      cvodes->SetSStolerancesB(reltol, abstol);

      // Results at time TBout1
      double dt_real = max(dt, t);
      cvodes->StepB(V_final, t, dt_real);
      cout << "t: " << t << endl;
      cout << "v:" << endl;
      V_final.Print();

   }

   // 10. Free the used memory.
   delete U;

   MPI_Finalize();
   return 0;
}

// AdvDiff Implementation
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

int AdvDiffSUNDIALS::SUNImplicitSetup(const Vector &y,
                                      const Vector &fy,
                                      int jok, int *jcur, double gamma)
{
   // Mf = M(I - gamma J) = M - gamma * M * J
   // J = df/dy => K
  *jcur = 1; // We've updated the jacobian

   delete Mf;
   Mf = Add(1., *M, -gamma, *K);
   Mf->EliminateRowsCols(ess_tdof_list);
   return 0;
}

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

void AdvDiffSUNDIALS::AdjointRateMult(const Vector &y, Vector & yB,
                                      Vector &yBdot) const
{

   int x1_size = (pfes->GetMyRank() == 0) ? yB.Size() - 2 : yB.Size();
   Vector z(x1_size);

   Vector x1(yB.GetData(), x1_size);

   // Set boundary conditions to zero
   Vector yB1(x1);
   yB1.SetSubVector(ess_tdof_list, 0.0);

   K_adj->Mult(yB1, z);

   Vector yBdot1(yBdot.GetData(), x1_size);
   M_solver.Mult(z, yBdot1);

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
   dP1->EliminateRowsCols(ess_tdof_list);

   Vector b1(y.Size());
   dP1->Mult(y, b1);
   delete dP1;

   ParBilinearForm dp2(pfes);
   Vector p2(pfes->GetParMesh()->SpaceDimension()); p2 = 1.;
   dp2.AddDomainIntegrator(new ConvectionIntegrator(*(new
                                                      VectorConstantCoefficient(p2))));
   dp2.Assemble();
   dp2.Finalize();

   HypreParMatrix * dP2 = dp2.ParallelAssemble();
   dP2->EliminateRowsCols(ess_tdof_list);

   Vector b2(y.Size());
   dP2->Mult(y, b2);
   delete dP2;

   double dp1_result = InnerProduct(pfes->GetComm(), yB1, b1);
   double dp2_result = InnerProduct(pfes->GetComm(), yB1, b2);

   if (pfes->GetMyRank() == 0)
   {
      yBdot[x1_size] = dp1_result;
      yBdot[x1_size + 1] = dp2_result;
   }


}
