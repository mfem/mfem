// This file is part of the RBVMS application. For more information and source
// code availability visit https://idoakkerman.github.io/
//
//   _____  ______      ____  __  _____
//   |  __ \|  _ \ \    / /  \/  |/ ____|
//   | |__) | |_) \ \  / /| \  / | (___
//   |  _  /|  _ < \ \/ / | |\/| |\___ \
//   | | \ \| |_) | \  /  | |  | |____) |
//   |_|  \_\____/   \/   |_|  |_|_____/
//
//
// RBVMS is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.
//------------------------------------------------------------------------------
#include <sys/stat.h>
#include "mfem.hpp"
//#include "../util/coefficients.hpp"
//#include "../util/solver.hpp"

//#include "formulation.hpp"
#include "integrator.hpp"

using namespace std;
using namespace mfem;

void line(int len)
{
   for (int b=0; b<len; ++b) { cout<<"-"; }
   cout<<"\n";
}

int problem = 1;

void CheckBoundaries(Array<bool> &bnd_flags,
                     Array<int> &markers,
                     Array<int> &bc_bnds,
                     int mark = 1)
{
   int amax = bnd_flags.Size();
   markers.SetSize(amax);
   markers = 0;
   // Check strong boundaries
   for (int b = 0; b < bc_bnds.Size(); b++)
   {
      int bnd = bc_bnds[b]-1;
      if ( bnd < 0 || bnd >= amax )
      {
         mfem_error("Boundary out of range.");
      }
      if (bnd_flags[bnd])
      {
         mfem_error("Boundary specified more then once.");
      }
      bnd_flags[bnd] = true;
      markers[bnd] = mark;
   }
}

// Custom block preconditioner for the Jacobian
class JacobianPreconditioner : public
   BlockLowerTriangularPreconditioner
{
protected:
   Array<Solver *> prec;

public:
   /// Constructor
   JacobianPreconditioner(Array<int> &offsets)
      : BlockLowerTriangularPreconditioner (offsets), prec(offsets.Size()-1)
   { prec = nullptr;};

   /// SetPreconditioners
   void SetPreconditioner(int i, Solver *pc)
   { prec[i] = pc; };

   /// Set the diagonal and off-diagonal operators
   virtual void SetOperator(const Operator &op)
   {
      BlockOperator *jacobian = (BlockOperator *) &op;

      for (int i = 0; i < prec.Size(); ++i)
      {
         if (prec[i])
         {
            prec[i]->SetOperator(jacobian->GetBlock(i,i));
            SetDiagonalBlock(i, prec[i]);
         }
         for (int j = i+1; j < prec.Size(); ++j)
         {
            SetBlock(j,i, &jacobian->GetBlock(j,i));
         }
      }
   };

   // Destructor
   virtual ~JacobianPreconditioner()
   {
      for (int i = 0; i < prec.Size(); ++i)
      {
         if (prec[i]) { delete prec[i]; }
      }
   };
};

/// This class help monitor the convergence of the linear Krylov solve.
class GeneralResidualMonitor : public IterativeSolverMonitor
{
private:
   const std::string prefix;
   int interval;
   mutable real_t norm0;

public:
   /// Constructor
   GeneralResidualMonitor( const std::string& prefix_,
                           int print_iv)
      : prefix(prefix_), interval(-1)
   {
      interval = print_iv;
   }

   /// Print residual
   virtual void MonitorResidual(int it,
                                real_t norm,
                                const Vector &r,
                                bool final)
   {
      if (interval < 0) { return; }
      if (it == 0) { norm0 = norm; }

      if ( ( it%interval == 0) || final )
      {
         mfem::out<<prefix<<" iteration "<<std::setw(3)<<it
                  <<": ||r|| = "
                  <<std::setw(8)<<std::defaultfloat<<std::setprecision(3)
                  <<norm
                  <<", ||r||/||r_0|| = "
                  <<std::setw(6)<<std::fixed<<std::setprecision(2)
                  <<100*norm/norm0<<" %\n";
      }
   };

};

void sol_u_fun(const Vector &coord,  real_t t, Vector &u)
{
   double x = coord[0];
   double y = coord[1];
   if (problem == 1)
   {
      double r = sqrt(x*x + y*y);
      double r0 = 0.5;

      // Base flow + BC
      u[0] = fmin(1.0, r - r0);
      u[1] = 0.0;

      // Disturbance 1
      x = coord[0];
      y = coord[1] - 3.0;
      r = sqrt(x*x + y*y);
      r0 = 1.0;
      u[0] += 2.0*fmax(0.0, r0 - r);
      u[1] -= 0.5*fmax(0.0, r0 - r);

      // Disturbance 2
      x = coord[0] + 3.0;
      y = coord[1];
      r = sqrt(x*x + y*y);
      r0 = 1.0;
      u[0] -= 0.5*fmax(0.0, r0 - r);
      u[1] += 2.0*fmax(0.0, r0 - r);
   }
   else
   {
      const double eps = 1e-6;

      u[0] = pow(4.0*x*(1.0-x),0.01)*y*y*y;
      u[1] = 0.0;

      if (y     < eps) { u[0] = 0.0; }
      if (x     < eps) { u[0] = 0.0; }
      if (1.0-x < eps) { u[0] = 0.0; }
      if (1.0-y < eps)
      {
         u[0] = 1.0;
         u[1] = 0.0;
         if (x     < eps) { u[0] = 0.5; }
         if (1.0-x < eps) { u[0] = 0.5; }
      }
   }
}

real_t sol_p_fun(const Vector &coord,  real_t t)
{
   return 0;
}


int main(int argc, char *argv[])
{

   // 1. Initialize MPI and HYPRE and print info
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   OptionsParser args(argc, argv);

   // Mesh and discretization parameters
   const char *mesh_file = "../../../data/inline-quad.mesh";
   const char *ref_file  = "";
   int order = 1;
   int ref_levels = 0;
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_file, "-rf", "--ref-file",
                  "File with refinement data");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order isoparametric space.");

   args.AddOption(&problem, "-p", "--problem",
                  "Problem");

   // Problem parameters
   Array<int> strong_bdr;
   Array<int> weak_bdr;
   Array<int> outflow_bdr;
   Array<int> suction_bdr;
   Array<int> blowing_bdr;
   Array<int> master_bdr;
   Array<int> slave_bdr;

   real_t rho_param = 1.204;
   real_t mu_param = 1.825e-5;

   args.AddOption(&strong_bdr, "-sbc", "--strong-bdr",
                  "Boundaries where Dirichelet BCs are enforced strongly.");
   args.AddOption(&weak_bdr, "-wbc", "--weak-bdr",
                  "Boundaries where Dirichelet BCs are enforced weakly.");
   args.AddOption(&outflow_bdr, "-out", "--outflow-bdr",
                  "Outflow boundaries.");
   args.AddOption(&suction_bdr, "-suc", "--suction-bdr",
                  "Suction boundaries.");
   args.AddOption(&blowing_bdr, "-blow", "--blowing-bdr",
                  "Blowing boundaries.");
   args.AddOption(&master_bdr, "-mbc", "--master-bdr",
                  "Periodic master boundaries.");
   args.AddOption(&slave_bdr, "-sbc", "--slave-bdr",
                  "Periodic slave boundaries.");

   args.AddOption(&mu_param, "-mu", "--dyn-visc",
                  "Sets the dynamic diffusion parameters, should be positive.");
   args.AddOption(&rho_param, "-rho", "--density",
                  "Sets the density parameters, should be positive.");

   // Time stepping params
   int ode_solver_type = 21;
   real_t t_final = 10.0;
   real_t dt = 0.01;
   real_t dt_max = 1.0;
   real_t dt_min = 0.0001;

   real_t cfl_target = 2.0;
   real_t dt_gain = -1.0;

   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::ImplicitTypes.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--dt",
                  "Time step.");
   args.AddOption(&dt_min, "-dtmn", "--dt-min",
                  "Minimum time step size.");
   args.AddOption(&dt_max, "-dtmx", "--dt-max",
                  "Maximum time step size.");
   args.AddOption(&cfl_target, "-cfl", "--cfl-target",
                  "CFL target.");
   args.AddOption(&dt_gain, "-dtg", "--dt-gain",
                  "Gain coefficient for time step adjustment.");

   // Solver parameters
   double GMRES_RelTol = 1e-3;
   int    GMRES_MaxIter = 250;
   double Newton_RelTol = 1e-3;
   int    Newton_MaxIter = 10;

   args.AddOption(&GMRES_RelTol, "-lt", "--linear-tolerance",
                  "Relative tolerance for the GMRES solver.");
   args.AddOption(&GMRES_MaxIter, "-li", "--linear-itermax",
                  "Maximum iteration count for the GMRES solver.");
   args.AddOption(&Newton_RelTol, "-nt", "--newton-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&Newton_MaxIter, "-ni", "--newton-itermax",
                  "Maximum iteration count for the Newton solver.");

   // Solution input/output params
   bool restart = false;
   int restart_interval = -1;
   real_t dt_vis = 10*dt;
   const char *vis_dir = "solution";
   args.AddOption(&restart, "-rs", "--restart", "-f", "--fresh",
                  "Restart from solution.");
   args.AddOption(&restart_interval, "-ri", "--restart-interval",
                  "Interval between archieved time steps.\n\t"
                  "For negative values output is skipped.");
   args.AddOption(&dt_vis, "-dtv", "--dt_vis",
                  "Time interval between visualization points.");
   args.AddOption(&vis_dir, "-vd", "--vis-dir",
                  "Directory for visualization files.\n\t");


   // Parse parameters
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) {  args.PrintOptions(cout);}

   // 3. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Refine mesh
   {
      if (mesh.NURBSext && (strlen(ref_file) != 0))
      {
         mesh.RefineNURBSFromFile(ref_file);
      }

      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
      if (Mpi::Root()) { mesh.PrintInfo();}
   }
   // Partition mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Boundary conditions
   if (Mpi::Root())
   {
      if (strong_bdr.Size()>0) {cout<<"Strong  = "; strong_bdr.Print();}
      if (weak_bdr.Size()>0) {cout<<"Weak    = "; weak_bdr.Print();}
      if (outflow_bdr.Size()>0) { cout<<"Outflow = "; outflow_bdr.Print() ;}
      if (suction_bdr.Size()>0) { cout<<"Suction = "; suction_bdr.Print() ;}
      if (blowing_bdr.Size()>0) { cout<<"Blowing = "; blowing_bdr.Print() ;}
      if (master_bdr.Size()>0) {cout<<"Periodic (master) = "; master_bdr.Print();}
   }

   Array<bool> bnd_flag(pmesh.bdr_attributes.Max());
   bnd_flag = true;
   for (int b = 0; b < pmesh.bdr_attributes.Size(); b++)
   {
      bnd_flag[pmesh.bdr_attributes[b]-1] = false;
   }
   // Provide more usefull error!!
   Array<int> strong_marker, weak_marker,outflow_marker,
         suction_marker, blowing_marker, master_marker, slave_marker;

   CheckBoundaries(bnd_flag, strong_marker, strong_bdr);
   CheckBoundaries(bnd_flag, weak_marker, weak_bdr, 2);
   CheckBoundaries(bnd_flag, outflow_marker, outflow_bdr);
   CheckBoundaries(bnd_flag, suction_marker, suction_bdr);
   CheckBoundaries(bnd_flag, blowing_marker, blowing_bdr);
   CheckBoundaries(bnd_flag, master_marker, master_bdr);
   CheckBoundaries(bnd_flag, slave_marker, slave_bdr);

   MFEM_VERIFY(master_bdr.Size() == slave_bdr.Size(),
               "Master-slave count do not match.");
   bool allSet = true;
   for (int b = 0; b < bnd_flag.Size(); b++)
   {
      //   allSet = allSet || bnd_flag[b];
      MFEM_VERIFY(bnd_flag[b], "Not all boundaries have a boundary condition set.");
   }

   // Select the time integrator
   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
   ODESolverWithStates*  ode_solver_ws = dynamic_cast<ODESolverWithStates*>
                                         (ode_solver.get());
   int nstate = ode_solver->GetStateSize();

   if (nstate > 1 && ( restart || restart_interval > 0 ))
   {
      mfem_error("RBVMS restart not available for this time integrator \n"
                 "Time integrator can have a maximum of one statevector.");
   }

   // 4. Define a finite element space on the mesh.
   Array<FiniteElementCollection *> fecs(2);
   if (pmesh.NURBSext)
   {
      fecs[0] = new NURBSFECollection(order);
      fecs[1] = new NURBSFECollection(order);
   }
   else
   {
      fecs[0] = new H1_FECollection(order, dim);
      fecs[1] = new H1_FECollection(order, dim);
   }

   Array<ParFiniteElementSpace *> spaces(2);
   spaces[0] = new ParFiniteElementSpace(&pmesh, fecs[0], dim,
                                         Ordering::byNODES);  //, Ordering::byVDIM);
   //,master_bdr, slave_bdr);
   spaces[1] = new ParFiniteElementSpace(&pmesh, fecs[1], 1, Ordering::byNODES);
   //   ,master_bdr, slave_bdr);

   // Report the degree of freedoms used
   {
      Array<int> tdof(num_procs),udof(num_procs),pdof(num_procs);
      tdof = 0;
      tdof[myid] = spaces[0]->GetTrueVSize();
      MPI_Reduce(tdof.GetData(), udof.GetData(), num_procs,
                 MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

      tdof = 0;
      tdof[myid] = spaces[1]->GetTrueVSize();
      MPI_Reduce(tdof.GetData(), pdof.GetData(), num_procs,
                 MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

      int udof_t = spaces[0]->GlobalTrueVSize();
      int pdof_t = spaces[1]->GlobalTrueVSize();
      if (Mpi::Root())
      {
         mfem::out << "Number of finite element unknowns:\n";
         mfem::out << "\tVelocity = "<< udof_t << endl;
         mfem::out << "\tPressure = "<< pdof_t << endl;
         mfem::out << "Number of finite element unknowns per partition:\n";
         mfem::out <<  "\tVelocity = "; udof.Print(mfem::out, num_procs);
         mfem::out <<  "\tPressure = "; pdof.Print(mfem::out, num_procs);
         for (int s = 0; s < num_procs; s++) { pdof[s] += udof[s]; }
         mfem::out <<  "\tTotal    = "; pdof.Print(mfem::out, num_procs);
      }
      tdof = 0;
      tdof[myid] = spaces[0]->GetVSize();
      MPI_Reduce(tdof.GetData(), udof.GetData(), num_procs,
                 MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

      tdof = 0;
      tdof[myid] = spaces[1]->GetVSize();
      MPI_Reduce(tdof.GetData(), pdof.GetData(), num_procs,
                 MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

      if (Mpi::Root())
      {
         mfem::out << "Number of finite element unknowns per partition:\n";
         mfem::out <<  "\tVelocity = "; udof.Print(mfem::out, num_procs);
         mfem::out <<  "\tPressure = "; pdof.Print(mfem::out, num_procs);
         for (int s = 0; s < num_procs; s++) { pdof[s] += udof[s]; }
         mfem::out <<  "\tTotal    = "; pdof.Print(mfem::out, num_procs);
      }
   }

   // Get vector offsets
   Array<int> bOffsets(3);
   bOffsets[0] = 0;
   bOffsets[1] = spaces[0]->GetTrueVSize();
   bOffsets[2] = spaces[1]->GetTrueVSize();
   bOffsets.PartialSum();

   // 5. Define the time stepping algorithm

   // Set up the preconditioner
   JacobianPreconditioner jac_prec(bOffsets);

   HypreILU* pc_mom = new HypreILU();
   //HypreILU* pc_cont = new HypreILU();

   //HypreSmoother* pc_mom = new HypreSmoother();
   HypreILU* pc_cont = new HypreILU();


   jac_prec.SetPreconditioner(0, pc_mom);
   jac_prec.SetPreconditioner(1, pc_cont);

   // Set up the Jacobian solver
   int printInterval = Mpi::Root() ?  100 : -1;
   GeneralResidualMonitor j_monitor("\t\tFGMRES", printInterval);
   FGMRESSolver j_gmres(MPI_COMM_WORLD);
   j_gmres.iterative_mode = false;
   j_gmres.SetRelTol(GMRES_RelTol);
   j_gmres.SetAbsTol(1e-12);
   j_gmres.SetMaxIter(GMRES_MaxIter);
   j_gmres.SetPrintLevel(-1);
   j_gmres.SetMonitor(j_monitor);
   j_gmres.SetPreconditioner(jac_prec);

   // Set up the Newton solver
   NewtonSystemSolver newton_solver(MPI_COMM_WORLD,bOffsets);
   // NewtonSolver newton_solver(MPI_COMM_WORLD);
   newton_solver.iterative_mode = true;
   newton_solver.SetPrintLevel(1);
   newton_solver.SetRelTol(Newton_RelTol);
   newton_solver.SetAbsTol(1e-12);
   newton_solver.SetMaxIter(Newton_MaxIter );
   newton_solver.SetSolver(j_gmres);

   // Define the physical parameters
   ConstantCoefficient rho(1.0);
   ConstantCoefficient mu(mu_param);
   VectorFunctionCoefficient sol_u(dim, sol_u_fun);
   FunctionCoefficient sol_p(sol_p_fun);

   Vector force_param(dim);
   force_param = 0.0;
   VectorConstantCoefficient force(force_param);

   // Define weak form and evolution
   ParBlockTimeDepNonlinearForm form(spaces);
   form.AddTimeDepDomainIntegrator(new RBVMSIntegrator(rho, mu, force));

   form.AddTimeDepBdrFaceIntegrator(new OutflowIntegrator(rho),
                                    outflow_marker);

   form.AddTimeDepBdrFaceIntegrator(new WeakBCIntegrator(rho, mu, sol_u),
                                    weak_marker);

   // {
   Array<int>  pressure_marker (pmesh.bdr_attributes.Max());
   pressure_marker = 0;
   Array<Array<int>*> marker(2);
   marker[0] = &strong_marker;
   marker[1] = &pressure_marker;

   Array<Vector*> rhs(2);
   rhs = nullptr;

   form.SetEssentialBC(marker,rhs);
   //}

   BlockEvolution evo(form, newton_solver);

   real_t t = 0.0;
   evo.SetTime(t);
   ode_solver->Init(evo);

   // 6. Define the solution vector, grid function and output
   BlockVector xp(bOffsets);
   BlockVector dxp(bOffsets);
   BlockVector xp0(bOffsets);
   BlockVector xpi(bOffsets);

   // Define the gridfunctions
   ParGridFunction x_u(spaces[0]);
   ParGridFunction x_p(spaces[1]);

   Array<ParGridFunction*> dx_u(nstate);
   Array<ParGridFunction*> dx_p(nstate);

   for (int i = 0; i < nstate; i++)
   {
      dx_u[i] = new ParGridFunction(spaces[0]);
      dx_p[i] = new ParGridFunction(spaces[1]);
   }

   // Define the visualisation output
   VisItDataCollection vdc("step", &pmesh);
   vdc.SetPrefixPath(vis_dir);
   vdc.RegisterField("u", &x_u);
   vdc.RegisterField("p", &x_p);

   // Get the start vector(s) from file -- or from function
   // real_t t;
   int si, ri, vi;
   struct stat info;
   if (restart && stat("restart/step.dat", &info) == 0)
   {
      // Read
      if (Mpi::Root())
      {
         real_t dtr;
         std::ifstream in("restart/step.dat", std::ifstream::in);
         in>>t>>si>>ri>>vi;
         in>>dtr;
         in.close();
         cout<<"Restarting from step "<<ri-1<<endl;
         if (dt_gain > 0) { dt = dtr; }
      }
      // Synchronize
      MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&si, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&ri, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&vi, 1, MPI_INT, 0, MPI_COMM_WORLD);

      // Open data files
      VisItDataCollection rdc("step", &pmesh);
      rdc.SetPrefixPath("restart");
      rdc.SetPrecision(18);
      rdc.Load(ri-1);

      x_u = *rdc.GetField("u");
      x_p = *rdc.GetField("p");

      x_u.GetTrueDofs(xp.GetBlock(0));
      x_p.GetTrueDofs(xp.GetBlock(1));

      if (nstate == 1 && rdc.GetField("du") && rdc.GetField("dp"))
      {
         *dx_u[0] = *rdc.GetField("du");
         *dx_p[0] = *rdc.GetField("dp");

         dx_u[0]->GetTrueDofs(dxp.GetBlock(0));
         dx_p[0]->GetTrueDofs(dxp.GetBlock(1));

         ode_solver_ws->GetState().Append(dxp);
      }
   }
   else
   {
      // Define initial condition from file
      t = 0.0; si = 0; ri = 1; vi = 1;
      sol_u.SetTime(-1.0);
      sol_p.SetTime(-1.0);
      x_u.ProjectCoefficient(sol_u, ProjectType::ELEMENT);
      x_p.ProjectCoefficient(sol_p, ProjectType::ELEMENT);

      x_u.GetTrueDofs(xp.GetBlock(0));
      x_p.GetTrueDofs(xp.GetBlock(1));

      if (nstate == 1)
      {
         // Print header
         if (Mpi::Root())
         {
            line(80);
            cout<<" Solve for initial rate. " << endl;
            cout<<std::defaultfloat<<std::setprecision(4);
            cout<<"   dt = " << dt << endl;
            line(80);
         }
         //evo.ImplicitSolve(dt,xp,dxp);
         dxp = 0.0;
         ode_solver_ws->GetState().Append(dxp);
      }

      // Visualize initial condition
      vdc.SetCycle(0);
      vdc.SetTime(0.0);
      vdc.Save();
   }

   // Define the restart output
   VisItDataCollection rdc("step", &pmesh);
   rdc.SetPrefixPath("restart");
   rdc.SetPrecision(18);

   // Define the restart writer
   rdc.RegisterField("u", &x_u);
   rdc.RegisterField("p", &x_p);
   if (nstate == 1)
   {
      rdc.RegisterField("du", dx_u[0]);
      rdc.RegisterField("dp", dx_p[0]);
   }

   // 7. Actual time integration

   // Open output file
   std::ofstream os;
   if (Mpi::Root())
   {
      std::ostringstream filename;
      filename << "output_"<<std::setw(6)<<setfill('0')<<si<< ".dat";
      os.open(filename.str().c_str());

      // Header
      char dimName[] = "xyz";
      int i = 6;
      os <<"# 1: step"<<"\t"<<"2: time"<<"\t"<<"3: dt"<<"\t"
         <<"4: cfl"<<"\t"<<"5: outflow"<<"\t";

      for (int b=0; b<pmesh.bdr_attributes.Size(); ++b)
      {
         int bnd = pmesh.bdr_attributes[b];
         for (int v=0; v<dim; ++v)
         {
            std::ostringstream forcename;
            forcename <<i++<<": F"<<dimName[v]<<"_"<<bnd;
            os<<forcename.str()<<"\t";
         }
      }
      os<<endl;
   }

   // Loop till final time reached
   while (t < t_final)
   {
      // Print header
      if (Mpi::Root())
      {
         line(80);
         cout<<std::defaultfloat<<std::setprecision(4);
         cout<<" step = " << si << endl;
         cout<<"   dt = " << dt << endl;
         cout<<std::defaultfloat<<std::setprecision(6);;
         cout<<" time = [" << t << ", " << t+dt <<"]"<< endl;
         cout<<std::defaultfloat<<std::setprecision(4);
         line(80);
      }
      // Actual time step
      xp0 = xp;
      ode_solver->Step(xp, t, dt);
      si++;

      // Postprocess solution
      real_t cfl = 0;//form.GetCFL();
      real_t outflow = 0.0;//form.GetOutflow();
      DenseMatrix bdrForce;// = form.GetForce();
      /*      if (Mpi::Root())
      {
            // Print to file
            int nbdr = pmesh.bdr_attributes.Size();
            os << std::setw(10);
            os << si<<"\t"<<t<<"\t"<<dt<<"\t"<<cfl<<"\t"<<outflow<<"\t";
            for (int b=0; b<nbdr; ++b)
            {
               int bnd = pmesh.bdr_attributes[b];
               for (int v=0; v<dim; ++v)
               {
                  os<<bdrForce(bnd-1,v)<<"\t";
               }
            }
            os<<"\n"<< std::flush;

            // Print line lambda function
            auto pline = [](int len)
            {
               cout<<" +";
               for (int b=0; b<len; ++b) { cout<<"-"; }
               cout<<"+\n";
            };

            // Print boundary header
            if (nbdr >= 10)
            {
               cout<<"\n";
               pline(10+13*nbdr);
               cout<<" | Boundary | ";
               for (int b=0; b<nbdr; ++b)
               {
                  cout<<std::setw(10)<<pmesh.bdr_attributes[b]<<" | ";
               }
               cout<<"\n";
               pline(10+13*nbdr);

               // Print actual forces
               char dimName[] = "xyz";
               for (int v=0; v<dim; ++v)
               {
                  cout<<" | Force "<<dimName[v]<<"  | ";
                  for (int b=0; b<nbdr; ++b)
                  {
                     int bnd = pmesh.bdr_attributes[b];
                     cout<<std::defaultfloat<<std::setprecision(4)<<std::setw(10);
                     cout<<bdrForce(bnd-1,v)<<" | ";
                  }
                  cout<<"\n";
               }
               pline(10+13*nbdr);
               cout<<"\n"<<std::flush;
            }
            }
      */

      // Write visualization files
      while (t >= dt_vis*vi)
      {
         // Interpolate solution
         real_t fac = (t-dt_vis*vi)/dt;

         // Report to screen
         if (Mpi::Root())
         {
            line(80);
            cout << "Visit output: " <<vi << endl;
            cout << "        Time: " <<t-dt<<" "<<t-fac*dt<<" "<<t<<endl;
            line(80);
         }
         // Copy solution in grid functions
         add (fac, xp0,(1.0-fac), xp, xpi);

         x_u.Distribute(xpi.GetBlock(0));
         x_p.Distribute(xpi.GetBlock(1));

         // Actually write to file
         vdc.SetCycle(vi);
         vdc.SetTime(dt_vis*vi);
         vdc.Save();
         vi++;
      }

      // Change time step
      real_t dt0 = dt;
      if ((dt_gain > 0))
      {
         dt *= pow(cfl_target/cfl, dt_gain);
         dt = min(dt, dt_max);
         dt = max(dt, dt_min);
      }

      // Print cfl and dt to screen
      if (Mpi::Root())
      {
         line(80);
         cout<<" outflow = "<<outflow<<endl;
         cout<<" cfl = "<<cfl<<endl;
         cout<<" dt  = "<<dt0<<" --> "<<dt<<endl;
         line(80);
      }

      // Write restart files
      if (restart_interval > 0 && si%restart_interval == 0)
      {
         // Report to screen
         if (Mpi::Root())
         {
            line(80);
            cout << "Restart output:" << ri << endl;
            line(80);
         }
         if (nstate == 1)
         {
            ode_solver_ws->GetState().Get(0,dxp);
         }

         // Actually write to file
         rdc.SetCycle(ri);
         rdc.SetTime(t);
         rdc.Save();
         ri++;

         // print meta file
         if (Mpi::Root())
         {
            std::ofstream step("restart/step.dat", std::ifstream::out);
            step<<t<<"\t"<<si<<"\t"<<ri<<"\t"<<vi<<endl;
            step<<dt<<endl;
            step.close();
         }

      }
      if (Mpi::Root())
      {
         cout<<endl<<endl;
         cout<<"\n"<<std::flush;
      }
   }
   os.close();

   // 8. Free the used memory.
   for (int i = 0; i < fecs.Size(); ++i)
   {
      delete fecs[i];
   }
   for (int i = 0; i < spaces.Size(); ++i)
   {
      delete spaces[i];
   }

   return 0;
}

