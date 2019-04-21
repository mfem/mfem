//                       MFEM Example 16 - Parallel Version
//
// Compile with: make ex16p
//
// Sample runs:  mpirun -np 4 ex16p
//               mpirun -np 4 ex16p -m ../data/inline-tri.mesh
//               mpirun -np 4 ex16p -m ../data/disc-nurbs.mesh -tf 2
//               mpirun -np 4 ex16p -s 1 -a 0.0 -k 1.0
//               mpirun -np 4 ex16p -s 2 -a 1.0 -k 0.0
//               mpirun -np 8 ex16p -s 3 -a 0.5 -k 0.5 -o 4
//               mpirun -np 4 ex16p -s 14 -dt 1.0e-4 -tf 4.0e-2 -vs 40
//               mpirun -np 16 ex16p -m ../data/fichera-q2.mesh
//               mpirun -np 16 ex16p -m ../data/fichera-mixed.mesh
//               mpirun -np 16 ex16p -m ../data/escher-p2.mesh
//               mpirun -np 8 ex16p -m ../data/beam-tet.mesh -tf 10 -dt 0.1
//               mpirun -np 4 ex16p -m ../data/amr-quad.mesh -o 4 -rs 0 -rp 0
//               mpirun -np 4 ex16p -m ../data/amr-hex.mesh -o 2 -rs 0 -rp 0
//
// Description:  This example solves a time dependent linear heat equation
//               problem of the form du/dt = C(u), with a linear diffusion
//               operator C(u) = \nabla \cdot (\kappa\nabla u).
//
//               The example demonstrates the use of nonlinear operators (the
//               class ConductionOperator defining C(u)), as well as their
//               implicit time integration. Note that implementing the method
//               ConductionOperator::ImplicitSolve is the only requirement for
//               high-order implicit (SDIRK) time integration.
//
//               We recommend viewing examples 2, 9 and 10 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/** After spatial discretization, the conduction model can be written as:
 *
 *     du/dt = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with diffusivity depending on u:
 *  (\kappa + \alpha u).
 *
 *  Class ConductionOperator represents the right-hand side of the above ODE.
 */
class ConductionOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   // Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.
   Array<int> ess_bdr_tdofs; // this list remains empty for pure Neumann b.c.

   ParBilinearForm *m;
   ParBilinearForm *s;
   ParBilinearForm *a;

   HypreParMatrix Mmat;
   HypreParMatrix Amat;
   double current_dt;

   HypreDiagScale * M_prec; // Preconditioner for the mass matrix M
   HypreBoomerAMG * A_prec; // Preconditioner for the implicit solver

   CGSolver M_solver;    // Krylov solver for inverting the mass matrix M
   CGSolver A_solver;    // Implicit solver for A = M + dt K

   ConstantCoefficient k_coeff;
   ConstantCoefficient dt_coeff;
   ProductCoefficient dtk_coeff;

   mutable ParGridFunction dudt_gf;
   mutable Vector rhs; // auxiliary vector
   mutable Vector RHS; // auxiliary vector
   mutable Vector dUdt; // auxiliary vector

   bool new_dt;
   void initA(double dt);
   void initImplicitSolve();

public:
   ConductionOperator(ParFiniteElementSpace &f, double kappa);

   // virtual void Mult(const Vector &u, Vector &du_dt) const;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &u, Vector &k);

   /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
   // void SetParameters(const Vector &u);

   virtual ~ConductionOperator();
};

static double kappa_  = 1.0;
static double TInf_   = 1.0;
static double TMax_   = 10.0;
static double TWidth_ = 0.125;

static vector<Vector> aVec_;

double Temperature(const Vector &x, double t);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/periodic-centered-segment.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 2;
   int ode_solver_type = 3;
   double t_final = 1.0;
   double dt = -1.0;
   double tol = -1.0;
   bool visualization = true;
   bool visit = false;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&tol, "-tol", "--tolerance",
                  "Solution tolerance.");
   args.AddOption(&kappa_, "-k", "--kappa",
                  "Kappa coefficient offset.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      MPI_Finalize();
      return 1;
   }

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   aVec_.resize(dim);
   if (strstr(mesh_file, "segment") != NULL)
   {
      aVec_[0].SetSize(1);
      aVec_[0][0] = 2.0;
   }
   else if (strstr(mesh_file, "square") != NULL)
   {
      aVec_[0].SetSize(2);
      aVec_[1].SetSize(2);
      aVec_[0][0] = 2.0; aVec_[0][1] = 0.0;
      aVec_[1][0] = 0.0; aVec_[1][1] = 2.0;
   }
   else if (strstr(mesh_file, "hexagon") != NULL)
   {
      aVec_[0].SetSize(2);
      aVec_[1].SetSize(2);
      aVec_[0][0] = 1.5; aVec_[0][1] = 0.5 * sqrt(3.0);
      aVec_[1][0] = 0.0; aVec_[1][1] = sqrt(3.0);
   }
   else if (strstr(mesh_file, "cube") != NULL)
   {
      aVec_[0].SetSize(3);
      aVec_[1].SetSize(3);
      aVec_[2].SetSize(3);
      aVec_[0][0] = 2.0; aVec_[0][1] = 0.0; aVec_[0][2] = 0.0;
      aVec_[1][0] = 0.0; aVec_[1][1] = 2.0; aVec_[1][2] = 0.0;
      aVec_[2][0] = 0.0; aVec_[2][1] = 0.0; aVec_[2][2] = 2.0;
   }


   // 4. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_solver = new BackwardEulerSolver; break;
      case 2:  ode_solver = new SDIRK23Solver(2); break;
      case 3:  ode_solver = new SDIRK33Solver; break;
      // Explicit methods
      case 11: ode_solver = new ForwardEulerSolver; break;
      case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 13: ode_solver = new RK3SSPSolver; break;
      case 14: ode_solver = new RK4Solver; break;
      case 15: ode_solver = new GeneralizedAlphaSolver(0.5); break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 7. Define the vector finite element space representing the current and the
   //    initial temperature, u_ref.
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   int fe_size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of temperature unknowns: " << fe_size << endl;
   }

   ParGridFunction next_u_gf(&fespace);
   ParGridFunction u_gf(&fespace);

   GridFunctionCoefficient uGFCoef(&u_gf);

   // 8. Set the initial conditions for u. All boundaries are considered
   //    natural.
   ConstantCoefficient zeroCoef(0.0);
   ConstantCoefficient oneCoef(1.0);
   FunctionCoefficient uCoef(Temperature);
   uCoef.SetTime(0.0);
   u_gf.ProjectCoefficient(uCoef);

   double err0 = u_gf.ComputeL2Error(uCoef);
   double nrm0 = u_gf.ComputeL2Error(zeroCoef);
   if (myid == 0)
   {
      cout << "Relative error in initial condition: " << err0 / nrm0
           << '\t' << err0 << '\t' << nrm0 << endl;
   }
   if (tol < 0.0)
   {
      tol = err0 /  nrm0;
   }

   {
      double h_min, h_max, kappa_min, kappa_max;
      pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

      double dt_cfl = h_min * h_min / kappa_;

      if (dt < 0.0)
      {
         dt = dt_cfl;
      }

      cout << "dt = " << dt << ", dt_cfl = " << dt_cfl << endl;
   }

   // 9. Initialize the conduction operator and the VisIt visualization.
   ConductionOperator oper(fespace, kappa_);

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "ex16p-pi-control-mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex16p-pi-control-init." << setfill('0') << setw(6) << myid;
      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(precision);
      pmesh->Print(omesh);
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u_gf.Save(osol);
   }

   VisItDataCollection visit_dc("Example16-PI-Control-Parallel", pmesh);
   visit_dc.RegisterField("temperature", &u_gf);
   if (visit)
   {
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      sout << "parallel " << num_procs << " " << myid << endl;
      int good = sout.good(), all_good;
      MPI_Allreduce(&good, &all_good, 1, MPI_INT, MPI_MIN, pmesh->GetComm());
      if (!all_good)
      {
         sout.close();
         visualization = false;
         if (myid == 0)
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *pmesh << u_gf;
         sout << "pause\n";
         sout << flush;
         if (myid == 0)
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
      }
   }

   // 10. Perform time-integration (looping over the time iterations, ti, with a
   //     time-step dt).
   ode_solver->Init(oper);
   double t = 0.0;

   ofstream ofs_err("ex16p_pi_control_err.out");
   ofstream ofs_dt("ex16p_pi_control_dt.out");

   double dt1 = dt; // dt_{n-1}
   double dt2 = dt; // dt_{n-2}
   double r1 = err0 / (tol * nrm0); // err_{n-1}
   double r2 = r1; // err_{n-2}

   bool last_step = false;
   // for (int ti = 1; !last_step; ti++)
   int ti = 1;
   int ts = 1;
   while (t < t_final)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      {
         double nrm = u_gf.ComputeL2Error(zeroCoef);

         bool reject = true;
         while (reject)
         {
            double next_t = t;
            next_u_gf = u_gf;

            cout << "Trial dt: " << dt << endl;

            ode_solver->Step(next_u_gf, next_t, dt);
            ts++;

            double e0 = next_u_gf.ComputeL2Error(uGFCoef);
            // double r0 = e0 / (nrm + nrm0);
            double r0 = e0 / nrm;

            if (r0 <= 1.2 * tol)
            {
               cout << "Accepting new dt based on relative difference " << r0
                    << " and tolerance " << tol << endl;
               // Gustafsson
               // double dt_temp = dt * pow(tol * dt / r0, 0.066666666666667) *
               //   pow(dt * r1 / (r0 * dt1), 0.13);
               // Modified Gustafsson
               double dt_temp = dt * pow(tol / r0, 0.066666666666667) *
                                pow(dt * r1 / (r0 * dt1), 0.13);
               // Valli
               // double dt_temp = dt * pow(1.0 / r0, 0.175) *
               //  pow(r1 / r0, 0.075) * pow(r1 * r1 / (r0 * r2), 0.01);

               r2 = r1;
               r1 = r0;

               dt2 = dt1;
               dt1 = dt;
               dt = min(2.0 * dt, dt_temp);

               t = next_t;

               u_gf = next_u_gf;
               reject = false;
            }
            else
            {
               cout << "Rejecting new dt based on relative difference " << r0
                    << " and tolerance " << tol
                    << ", " << r0 << " > " << 1.2 * tol << endl;
               // Gustafsson
               // double dt_temp = dt * pow(tol * dt/ r0, 0.2);
               // Modified Gustafsson
               double dt_temp = dt * pow(tol / r0, 0.2);
               // Valli
               // double dt_temp = dt * pow(1.0 / r0, 0.175) *
               //   pow(r1 / r0, 0.075) * pow(r1 * r1 / (r0 * r2), 0.01);
               dt = min(2.0 * dt, dt_temp);
            }
            ofs_dt << ts << '\t' << ti << '\t' << t << '\t' << dt << endl;
         }

         ti++;
      }

      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid == 0)
         {
            cout << "step " << ti << ", t = " << t << endl;
         }

         uCoef.SetTime(t);
         double nrm1 = u_gf.ComputeL2Error(oneCoef);
         double err1 = u_gf.ComputeL2Error(uCoef);
         if (myid == 0)
         {
            cout << t << " L2 Relative Error of Solution: " << err1 / nrm1
                 << endl;
            ofs_err << t << '\t' << err1 / nrm1 << '\t'
                    << err1 << '\t' << nrm1 << endl;
         }

         if (visualization)
         {
            sout << "parallel " << num_procs << " " << myid << "\n";
            sout << "solution\n" << *pmesh << u_gf << flush;
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
   }

   // 11. Save the final solution in parallel. This output can be viewed later
   //     using GLVis: "glvis -np <np> -m ex16-mesh -g ex16-final".
   {
      ostringstream sol_name;
      sol_name << "ex16p-pi-control-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u_gf.Save(osol);
   }
   ofs_err.close();
   ofs_dt.close();

   // 12. Free the used memory.
   delete ode_solver;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

ConductionOperator::ConductionOperator(ParFiniteElementSpace &f, double kap)
   : TimeDependentOperator(f.GetVSize(), 0.0),
     fespace(f), m(NULL), s(NULL), a(NULL),
     current_dt(0.0),
     M_prec(NULL), A_prec(NULL),
     M_solver(f.GetComm()), A_solver(f.GetComm()),
     k_coeff(kap), dt_coeff(current_dt), dtk_coeff(dt_coeff, k_coeff),
     rhs(height),
     RHS(f.GetTrueVSize()),
     dUdt(f.GetTrueVSize()),
     new_dt(true)
{
   const double rel_tol = 1e-8;

   m = new ParBilinearForm(&fespace);
   m->AddDomainIntegrator(new MassIntegrator());
   m->Assemble();
   // m->FormSystemMatrix(ess_tdof_list, Mmat);

   s = new ParBilinearForm(&fespace);
   s->AddDomainIntegrator(new DiffusionIntegrator(k_coeff));
   s->Assemble();

   a = new ParBilinearForm(&fespace);
   a->AddDomainIntegrator(new MassIntegrator());
   a->AddDomainIntegrator(new DiffusionIntegrator(dtk_coeff));
   // A->Assemble(0); // keep sparsity pattern of M and K the same
   // A->FormSystemMatrix(ess_tdof_list, Kmat);

   M_prec = new HypreDiagScale(Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
   M_solver.SetOperator(Mmat);
   M_solver.SetPreconditioner(*M_prec);

   A_solver.iterative_mode = false;
   A_solver.SetRelTol(rel_tol);
   A_solver.SetAbsTol(0.0);
   A_solver.SetMaxIter(100);
   A_solver.SetPrintLevel(0);
}
/*
void ConductionOperator::Mult(const Vector &u, Vector &du_dt) const
{
   // Compute:
   //    du_dt = M^{-1}*-K(u)
   // for du_dt
   Kmat.Mult(u, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, du_dt);
}
*/
void ConductionOperator::initA(double dt)
{
   new_dt = (dt != current_dt);
   if (new_dt)
   {
      current_dt = dt;
      dt_coeff.constant = dt;

      a->Update();
      a->Assemble();
   }
}

void ConductionOperator::initImplicitSolve()
{
   if (new_dt)
   {
      delete A_prec;
      A_prec = new HypreBoomerAMG(Amat);
      A_prec->SetPrintLevel(0);
      A_solver.SetPreconditioner(*A_prec);
      A_solver.SetOperator(Amat);
   }
}

void ConductionOperator::ImplicitSolve(const double dt,
                                       const Vector &u, Vector &du_dt)
{
   /*
   // Solve the equation:
   //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
   // for du_dt
   if (!T)
   {
      T = Add(1.0, Mmat, dt, Kmat);
      current_dt = dt;
      T_solver.SetOperator(*T);
   }
   MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt
   Kmat.Mult(u, z);
   z.Neg();
   T_solver.Mult(z, du_dt);
   */

   s->Mult(u, rhs);
   rhs *= -1.0;

   this->initA(dt);

   dudt_gf.MakeRef(&fespace, du_dt);

   a->FormLinearSystem(ess_bdr_tdofs, dudt_gf, rhs, Amat, dUdt, RHS);

   this->initImplicitSolve();

   A_solver.Mult(RHS, dUdt);

   a->RecoverFEMSolution(dUdt, rhs, dudt_gf);
}
/*
void ConductionOperator::SetParameters(const Vector &u)
{
   delete K;
   delete T;
   T = NULL; // re-compute T on the next ImplicitSolve
}
*/
ConductionOperator::~ConductionOperator()
{
   delete M_prec;
   delete A_prec;
   delete m;
   delete a;
}

double T0Func(const Vector &x, double t)
{
   int dim = x.Size();

   double r2 = x * x;

   double d = 4.0 * kappa_ * t + pow(TWidth_, 2) / M_LN2;

   double e = exp(-r2 / d);

   double s = pow(1.0 + 4.0 * M_LN2 * kappa_ * t / pow(TWidth_, 2), 0.5 * dim);

   return (TMax_ - TInf_) * e / s;
}

double Temperature(const Vector &x, double t)
{
   int dim = x.Size();

   double tol = 1e-12;

   double d = 4.0 * kappa_ * t + pow(TWidth_, 2) / M_LN2;
   double spread = sqrt(fabs(d * log(tol)));

   double xt[3];
   Vector xtVec(xt, dim);

   double T = TInf_;

   int si = (int)ceil(0.5 * spread);
   switch (dim)
   {
      case 1:
      {
         for (int i=-si; i<=si; i++)
         {
            xtVec = x;
            xtVec.Add(i, aVec_[0]);
            T += T0Func(xtVec, t);
         }
      }
      break;
      case 2:
      {
         for (int i=-si; i<=si; i++)
         {
            for (int j=-si; j<=si; j++)
            {
               xtVec = x;
               xtVec.Add(i, aVec_[0]);
               xtVec.Add(j, aVec_[1]);
               T += T0Func(xtVec, t);
            }
         }
      }
      break;
      case 3:
      {
         for (int i=-si; i<=si; i++)
         {
            for (int j=-si; j<=si; j++)
            {
               for (int k=-si; k<=si; k++)
               {
                  xtVec = x;
                  xtVec.Add(i, aVec_[0]);
                  xtVec.Add(j, aVec_[1]);
                  xtVec.Add(k, aVec_[2]);
                  T += T0Func(xtVec, t);
               }
            }
         }
      }
      break;
   }

   return T;
}
