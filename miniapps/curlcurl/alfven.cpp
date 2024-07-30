//                       MFEM Changed from Example 10 - Parallel Version
//
// Compile with: make alfven
//
// Sample runs:
//    mpirun -np 4 alfven -m ../../data/beam-quad.mesh -s 3 -rs 2 -dt 3
//
// TODO
// define a nonlinear system for Newton (Newton call Jacobian iteration)

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

void B_exact(const Vector &x, Vector &B)
{
   const double R = sqrt(x(0)*x(0)+x(1)*x(1)), Z = x(2);
   const double q = q0 + q2*((R-R0)*(R-R0)+Z*Z)/a_i/a_i;
   double B_R, B_Z, B_phi, cosphi, sinphi;

   B_R = -Z/q/R*B0;
   B_Z = (R-R0)/q/R*B0;
   B_phi = R0/R*B0;

   cosphi = x(0)/R;
   sinphi = x(1)/R;

   B(0) = B_R*cosphi-B_phi*sinphi;
   B(1) = B_R*sinphi+B_phi*cosphi;
   B(2) = B_Z;
};

// Class for returning the B0 x (curl B0) of the Linear form
class BxCurlBCoefficient : public VectorCoefficient
{
private:
   GridFunction *B0;
   double epsilon;
public:
   BxCurlBCoefficient(GridFunction *b_gb, double epsilon_)
      : VectorCoefficient(3){B0=b_gb; epsilon=epsilon_;}

   using VectorCoefficient::Eval;

   virtual void Eval(Vector &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector curlB;
      T.SetIntPoint(&ip);
      B0->GetVectorValue(T, ip, K);
      B0->GetCurl(T, curlB)
      cross3D(curlB,K);
      K*=(1.0-epsilon);
   }
};

class NonlinearSystemOperator;

/** After spatial discretization, the linear Alfven model can be written as a
 *  system of ODEs:
 *     dV/dt = M1^{-1}*(-G*B + S)
 *     dB/dt = M2^{-1}*(GT*V + L*B),
 *     where V in H(div) and B in H(curl)
 *     M1 is the mass matrix in H(div) with bc eliminated
 *     M2 is the mass matrix in H(curl) without bc elimited
 *     K is the curl-curl operator
 *     G is B0x(curl) with trial in H(curl) and test in H(div)
 *     GT is transpose of G
 **/
class AlfvenOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &Bfespace, &Vfespace;
   Array<int> V_ess_tdof_list;

   ParBilinearForm MV, MB, L;
   ParMixedBilinearForm G;
   ParLinearForm S;   //source term = (1-epsilon) B0x(curl B0)
   double resi, epsi;
   ParGridFunction &b_bg;

   HypreParMatrix *Mmat1, *Mmat2; 
   CGSolver M1_solver, M2_solver;
   HypreSmoother M1_prec, M2_prec; // Preconditioner for the mass matrix M

   NewtonSolver newton_solver;

   Solver *block_solver;
   Solver *block_prec;

   Vector source;

   mutable Vector zv,zb; // auxiliary vector

public:
   AlfvenOperator(ParFiniteElementSpace &Vf, ParFiniteElementSpace &Bf,
                  ParGridFunction &b_bg_,
                  Array<int> &V_ess_bdr, double resi_);

   virtual void Mult(const Vector &vb, Vector &dvb_dt) const;
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);
   virtual ~AlfvenOperator();
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-quad.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 2;
   int ode_solver_type = 3;
   double t_final = 10.0;
   double dt = 0.1;
   double resi = 1e-4;
   double epsilon = 0.5;
   bool adaptive_lin_rtol = true;
   bool visualization = true;
   int vis_steps = 1;

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
                  "            11 - Forward Euler, 12 - RK2,\n\t"
                  "            13 - RK3 SSP, 14 - RK4."
                  "            22 - Implicit Midpoint Method,\n\t"
                  "            23 - SDIRK23 (A-stable), 24 - SDIRK34");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&resi, "-resi", "--resistivity",
                  "Resistivity coefficient.");
   args.AddOption(&epsilon, "-ep", "--epsilon",
                  "Epsilon in the source term.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
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
   if (dim!=3){
      cout << "wrong dimensions in mesh!"<<endl;
      MPI_Finalize();
      delete mesh;
      return 1;
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
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete mesh;
         return 3;
   }

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

   RT_FECollection V_fe_coll(order, dim); //H(div)
   ND_FECollection B_fe_coll(order, dim); //H(curl)
   ParFiniteElementSpace Vfespace(pmesh, &V_fe_coll),
                         Bfespace(pmesh, &B_fe_coll);

   HYPRE_BigInt glob_size_V = Vfespace.GlobalTrueVSize(),
                glob_size_B = Bfespace.GlobalTrueVSize();
                
   if (myid == 0)
   {
      cout << "Number of V unknowns: " << glob_size_V << endl;
      cout << "Number of B unknowns: " << glob_size_B << endl;
   }
   int V_true_size=Vfespace.TrueVSize(),
       B_true_size=Bfespace.TrueVSize(); 

   Array<int> true_offset(3);
   true_offset[0] = 0;
   true_offset[1] = V_true_size;
   true_offset[2] = B_true_size+V_true_size;

   // Define the block system
   BlockVector vb(true_offset);
   ParGridFunction v_gf, b_gf;
   v_gf.MakeTRef(&Vfespace, vb, true_offset[0]);
   b_gf.MakeTRef(&Bfespace, vb, true_offset[1]);
   v_gf=0.0;
   b_gf=0.0;

   // Define background B
   VectorFunctionCoefficient VecCoeff(dim, B_exact);
   ParGridFunction b_bg(Bfespace);
   b_bg.ProjectCoefficient(VecCoeff);

   Array<int> V_ess_bdr(Vfespace.GetMesh()->bdr_attributes.Max());
   V_ess_bdr = 0;
   V_ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed

   // Initialize the operator
   AlfvenOperator oper(Vfespace, Bfespace, b_bg, V_ess_bdr, resi);

   double t = 0.0;
   oper.SetTime(t);
   ode_solver->Init(oper);

   // Perform time-integration
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(vx, t, dt_real);

      last_step = (t >= t_final - 1e-8*dt);
   }

   // 12. Free the used memory.
   delete ode_solver;
   delete pmesh;

   return 0;
}

AlfvenOperator::AlfvenOperator(ParFiniteElementSpace &Vf, ParFiniteElementSpace &Bf,
                  ParGridFunction &b_bg_, Array<int> &V_ess_bdr, double resi_, double epsi_)
   : TimeDependentOperator(Vf.TrueVSize()+Bf.TrueVSize(), 0.0), 
     Vfespace(Vf), Bfespace(Bf),
     M1(&Vfespace), M2(&Vfespace), L(&Bfespace), S(&Vfespace), G(&Bfespace, &Vfespace),
     b_bg(b_bg), resi(resi_), epsi(epsi_),
     M1_solver(Vf.GetComm()), M2_solver(Bf.GetComm()), newton_solver(Vf.GetComm()),
     source(Vf.TrueVSize()), zv(Vf.TrueVSize()), zb(Bf.TrueVSize())
{
   const double rel_tol = 1e-8;

   Vfespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Note M1 does not eliminate the bc but Mmat1 eliminate the bc
   ConstantCoefficient one(1.0);
   M1.AddDomainIntegrator(new VectorMassIntegrator(one));
   M1.Assemble();
   M1.Finalize();
   Mmat1 = M1.ParallelAssemble();
   HypreParMatrix *Me = Mmat1->EliminateRowsCols(ess_tdof_list);
   delete Me;

   M2.AddDomainIntegrator(new VectorMassIntegrator(one));
   M2.Assemble();
   M2.Finalize();
   Mmat2 = M2.ParallelAssemble();

   M1_solver.iterative_mode = false;
   M1_solver.SetRelTol(rel_tol);
   M1_solver.SetAbsTol(0.0);
   M1_solver.SetMaxIter(30);
   M1_solver.SetPrintLevel(0);
   M1_prec.SetType(HypreSmoother::Jacobi);
   M1_solver.SetPreconditioner(M1_prec);
   M1_solver.SetOperator(*Mmat1);

   M2_solver.iterative_mode = false;
   M2_solver.SetRelTol(rel_tol);
   M2_solver.SetAbsTol(0.0);
   M2_solver.SetMaxIter(30);
   M2_solver.SetPrintLevel(0);
   M2_prec.SetType(HypreSmoother::Jacobi);
   M2_solver.SetPreconditioner(M2_prec);
   M2_solver.SetOperator(*Mmat2);

   //assemble curl curl 
   ConstantCoefficient resi_coeff(resi);
   L.AddDomainIntegrator(new CurlCurlIntegrator(resi_coeff));
   L.Assemble();
   L.Finalize();

   //assemble G = (Bxcurl(trial), test)
   VectorGridFunctionCoefficient b_gf_coeff(&b_bg);
   G.AddDomainIntegrator(new MixedCrossCurlIntegrator(resi_coeff));
   G.Assemble();
   G.Finalize();

   //assemble GV = (Bxcurl(trial), test)^T
   VectorGridFunctionCoefficient b_gf_coeff(&b_bg);
   GV.AddDomainIntegrator(new TransposeIntegrator(new MixedCrossCurlIntegrator(resi_coeff)));
   GV.Assemble();
   GV.Finalize();

   //assemble source
   BxCurlBCoefficient b_source_coeff(&b_bg, epsi);
   S.AddDomainIntegrator(new VectorDomainLFIntegrator(b_source_coeff));
   S.Assemble(); 
   S.ParallelAssemble(source);

   HypreSmoother *J_hypreSmoother = new HypreSmoother;
   J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
   J_hypreSmoother->SetPositiveDiagonal(true);
   J_prec = J_hypreSmoother;

   MINRESSolver *J_minres = new MINRESSolver(f.GetComm());
   J_minres->SetRelTol(rel_tol);
   J_minres->SetAbsTol(0.0);
   J_minres->SetMaxIter(300);
   J_minres->SetPrintLevel(-1);
   J_minres->SetPreconditioner(*J_prec);
}

/*
 *     dV/dt = M1^{-1}*(-G*B + S)
 *     dB/dt = M2^{-1}*(GT*V + L*B),
 */
void AlfvenOperator::Mult(const Vector &vb, Vector &dvb_dt) const
{
   int V_true_size=Vfespace.TrueVSize(),
       B_true_size=Bfespace.TrueVSize(); 

   Vector v(vb.GetData() + 0,           V_true_size);
   Vector b(vb.GetData() + V_true_size, B_true_size);
   Vector dv_dt(dvx_dt.GetData() +  0,          V_true_size);
   Vector db_dt(dvx_dt.GetData() + V_true_size, B_true_size);

   zv = source;
   zv.Neg();
   G.TrueAddMult(v, zv);
   zv.SetSubVector(ess_tdof_list, 0.0);
   zv.Net();
   M1_solver.Mult(zv, dv_dt);

   zb = 0.0;
   L.TrueAddMult(b, zb);
   GV.TrueAddMult(v, zb);
   M2_solver.Mult(zb, db_dt);
}

void AlfvenOperator::ImplicitSolve(const double dt,
                                         const Vector &vx, Vector &dvx_dt)
{
   int V_true_size=Vfespace.TrueVSize(),
       B_true_size=Bfespace.TrueVSize(); 

   nonlinear_oper->SetParameters(dt, &v, &x);
   Vector zero; 
   newton_solver.Mult(zero, dv_dt);
   MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");
   add(v, dt, dv_dt, dx_dt);
}

AlfvenOperator::~AlfvenOperator()
{
   delete J_solver;
   delete J_prec;
   delete nonlinear_oper;
   delete model;
   delete Mmat;
}


