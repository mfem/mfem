//                  MFEM Example 9 Continuous - Parallel Version
//
// Author: Madison Sheridan
// Created: 05/25/2022
//
// Based on Example 9p
//
// Compile with: make ex9p-continuous
//
// Sample runs:
//    mpirun -np 4 ex9p-continuous -m ../data/periodic-segment.mesh -p 0 -dt 0.005
//    mpirun -np 4 ex9p-continuous -m ../data/periodic-square.mesh -p 0 -dt 0.01
//    mpirun -np 4 ex9p-continuous -m ../data/periodic-hexagon.mesh -p 0 -dt 0.01
//    mpirun -np 4 ex9p-continuous -m ../data/periodic-square.mesh -p 1 -dt 0.005 -tf 9
//    mpirun -np 4 ex9p-continuous -m ../data/periodic-hexagon.mesh -p 1 -dt 0.005 -tf 9
//    mpirun -np 4 ex9p-continuous -m ../data/amr-quad.mesh -p 1 -rp 1 -dt 0.002 -tf 9
//    mpirun -np 4 ex9p-continuous -m ../data/amr-quad.mesh -p 1 -rp 1 -dt 0.02 -s 13 -tf 9
//    mpirun -np 4 ex9p-continuous -m ../data/star-q3.mesh -p 1 -rp 1 -dt 0.004 -tf 9
//    mpirun -np 4 ex9p-continuous -m ../data/star-mixed.mesh -p 1 -rp 1 -dt 0.004 -tf 9
//    mpirun -np 4 ex9p-continuous -m ../data/disc-nurbs.mesh -p 1 -rp 1 -dt 0.005 -tf 9
//    mpirun -np 4 ex9p-continuous -m ../data/disc-nurbs.mesh -p 2 -rp 1 -dt 0.005 -tf 9
//    mpirun -np 4 ex9p-continuous -m ../data/periodic-square.mesh -p 3 -rp 2 -dt 0.0025 -tf 9 -vs 20
//    mpirun -np 4 ex9p-continuous -m ../data/periodic-cube.mesh -p 0 -o 2 -rp 1 -dt 0.01 -tf 8
//    mpirun -np 4 ex9p-continuous -m ../data/periodic-square.msh -p 0 -rs 2 -dt 0.005 -tf 2
//    mpirun -np 4 ex9p-continuous -m ../data/periodic-cube.msh -p 0 -rs 1 -o 2 -tf 2
//    mpirun -np 3 ex9p-continuous -m ../data/amr-hex.mesh -p 1 -rs 1 -rp 0 -dt 0.005 -tf 0.5
//
// Device sample runs:
//    mpirun -np 4 ex9p-continuous -pa
//    mpirun -np 4 ex9p-continuous -ea
//    mpirun -np 4 ex9p-continuous -fa
//    mpirun -np 4 ex9p-continuous -pa -m ../data/periodic-cube.mesh
//    mpirun -np 4 ex9p-continuous -pa -m ../data/periodic-cube.mesh -d cuda
//    mpirun -np 4 ex9p-continuous -ea -m ../data/periodic-cube.mesh -d cuda
//    mpirun -np 4 ex9p-continuous -fa -m ../data/periodic-cube.mesh -d cuda
//    mpirun -np 4 ex9p-continuous -pa -m ../data/amr-quad.mesh -p 1 -rp 1 -dt 0.002 -tf 9 -d cuda
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Continuous Galerkin
//               bilinear forms in MFEM (face integrators), the use of explicit
//               ODE time integrators, the definition of periodic
//               boundary conditions through periodic meshes, as well as the use
//               of GLVis for persistent visualization of a time-evolving
//               solution. Saving of time-dependent data files for visualization
//               with VisIt (visit.llnl.gov) and ParaView (paraview.org), as
//               well as the optional saving with ADIOS2 (adios2.readthedocs.io)
//               are also illustrated.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <chrono> // For timer

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Exact solution
double u0_function(const Vector &x);
double exact_sol(const Vector &x, double t);
double zero_sol(const Vector &x, double t) { return 0;}

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

/** A time-dependent operator for the right-hand side of the ODE. The CG weak
    form of du/dt = -v.grad(u) is M du/dt = (D + K) u, where M and K are the mass
    and advection matrices, and D is the artificial viscosity matrix as
    described in Guermond/Popov 2016. This can be written as a general ODE,
    du/dt = M^{-1} K u, and this class is used to evaluate the right-hand side.
*/
class FE_Evolution : public TimeDependentOperator
{
private:
   ParFiniteElementSpace &pfes;
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   HypreParMatrix Mmat, Kmat;
   HypreParVector lumpedM;
   HYPRE_BigInt * row_starts, *col_starts;
   const SparseMatrix K_spmat;
   ParBilinearForm *D_form;
   HypreSolver *M_prec;
   HyprePCG M_solver; // Symmetric system, can use CG

   mutable SparseMatrix dij_matrix;
   HypreParMatrix *D;

   Array<int> K_smap;
   mutable Vector z;

public:
   FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_);

   /** FE_Evolution::build_dij_matrix
   Builds dij_matrix used in the low order approximation, which is based on
   Guermond/Popov 2016.
   */
   void build_dij_matrix(const Vector &U,
                         const VectorFunctionCoefficient &velocity) ;

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   virtual ~FE_Evolution();
};


int main(int argc, char *argv[])
{
   // 0. Start Timer
   clock_t t_start, t_end;
   t_start = clock();

   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   problem = 4;
   const char *mesh_file = "../data/periodic-square.mesh";
   int ser_ref_levels = 3;
   int par_ref_levels = 0;
   int order = 1;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 1;
   double t_final = 10.0;
   bool one_time_step = false;
   double dt = 0.01;
   bool match_dt_to_h = false;
   bool visualization = true;
   bool visit = false;
   bool paraview = false;
   bool adios2 = false;
   bool binary = false;
   int vis_steps = 2;
   int precision = 12;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&ea, "-ea", "--element-assembly", "-no-ea",
                  "--no-element-assembly", "Enable Element Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            11 - Backward Euler,\n\t"
                  "            12 - SDIRK23 (L-stable), 13 - SDIRK33,\n\t"
                  "            22 - Implicit Midpoint Method,\n\t"
                  "            23 - SDIRK23 (A-stable), 24 - SDIRK34");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&one_time_step, "-ots", "--one-time-step",
                  "-no-ots", "--no-one-time-step",
                  "Set end time to one time step for convergence testing.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&match_dt_to_h, "-ct", "--conv-test",
                  "-no-ct", "--no-conv-test",
                  "Enable convergence testing by matching dt to h.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&adios2, "-adios2", "--adios2-streams", "-no-adios2",
                  "--no-adios2-streams",
                  "Save data using adios2 streams.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle geometrically periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      // Explicit methods
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      // Implicit (L-stable) methods
      case 11: ode_solver = new BackwardEulerSolver; break;
      case 12: ode_solver = new SDIRK23Solver(2); break;
      case 13: ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         if (Mpi::Root())
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete mesh;
         return 3;
   }

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter. If the mesh is of NURBS type, we convert it
   //    to a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 6. Define the parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }
   double hmin, hmax, kmin, kmax;
   pmesh->GetCharacteristics(hmin, hmax, kmin, kmax);
   if (match_dt_to_h) { dt = hmin / sqrt(2); }
   if (one_time_step) {t_final = dt; }

   // 7. Define the parallel H1 finite element space on the
   //    parallel refined mesh of the given polynomial order.
   // H1_FECollection fec(order, dim, BasisType::Positive);
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_BigInt global_vSize = fes->GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   // 8. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the H1 discretization.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(exact_sol);

   ParBilinearForm *m = new ParBilinearForm(fes);
   ParBilinearForm *k = new ParBilinearForm(fes);
   if (pa)
   {
      m->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      k->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   else if (ea)
   {
      m->SetAssemblyLevel(AssemblyLevel::ELEMENT);
      k->SetAssemblyLevel(AssemblyLevel::ELEMENT);
   }
   else if (fa)
   {
      m->SetAssemblyLevel(AssemblyLevel::FULL);
      k->SetAssemblyLevel(AssemblyLevel::FULL);
   }

   m->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   constexpr double alpha = -1.0;
   k->AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));

   // ParLinearForm *b = new ParLinearForm(fes);
   // b->AddBdrFaceIntegrator(
   //    new BoundaryFlowIntegrator(inflow, velocity, alpha));

   int skip_zeros = 0;
   m->Assemble();
   k->Assemble(skip_zeros);
   m->Finalize();
   k->Finalize(skip_zeros);
   //

   HypreParMatrix *M = m->ParallelAssemble();
   HypreParMatrix *K = k->ParallelAssemble();

   // 9. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   ParGridFunction *u = new ParGridFunction(fes);
   u->ProjectCoefficient(u0);
   HypreParVector *U = u->GetTrueDofs();

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "ex9p-continuous-mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex9p-continuous-init." << setfill('0') << setw(6) << myid;
      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(precision);
      pmesh->Print(omesh);
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example9-Parallel-Continuous", pmesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example9-Parallel-Continuous", pmesh);
         dc->SetPrecision(precision);
         // To save the mesh using MFEM's parallel mesh format:
         // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
      }
      dc->RegisterField("solution", u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      pd = new ParaViewDataCollection("Example9P-Continuous", pmesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("solution", u);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }

   // Optionally output a BP (binary pack) file using ADIOS2. This can be
   // visualized with the ParaView VTX reader.
#ifdef MFEM_USE_ADIOS2
   ADIOS2DataCollection *adios2_dc = NULL;
   if (adios2)
   {
      std::string postfix(mesh_file);
      postfix.erase(0, std::string("../data/").size() );
      postfix += "_o" + std::to_string(order);
      const std::string collection_name = "ex9p-continuous-" + postfix + ".bp";

      adios2_dc = new ADIOS2DataCollection(MPI_COMM_WORLD, collection_name, pmesh);
      // output data substreams are half the number of mpi processes
      adios2_dc->SetParameter("SubStreams", std::to_string(num_procs/2) );
      // adios2_dc->SetLevelsOfDetail(2);
      adios2_dc->RegisterField("solution", u);
      adios2_dc->SetCycle(0);
      adios2_dc->SetTime(0.0);
      adios2_dc->Save();
   }
#endif

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
   }

   // 10. Define the time-dependent evolution operator describing the ODE
   //     right-hand side, and perform time-integration (looping over the time
   //     iterations, ti, with a time-step dt).
   FE_Evolution adv(*m, *k);

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      // Build new D matrix since this matrix is time dependent
      adv.build_dij_matrix(*U, velocity);
      double dt_real = min(dt, t_final - t);
      ode_solver->Step(*U, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         if (Mpi::Root())
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }

         // 11. Extract the parallel grid function corresponding to the finite
         //     element approximation U (the local solution on each processor).
         *u = *U; // Synchronizes MPI processes.

         if (visualization)
         {
            sout << "parallel " << num_procs << " " << myid << "\n";
            sout << "solution\n" << *pmesh << *u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }

         if (paraview)
         {
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
         }

#ifdef MFEM_USE_ADIOS2
         // transient solutions can be visualized with ParaView
         if (adios2)
         {
            adios2_dc->SetCycle(ti);
            adios2_dc->SetTime(t);
            adios2_dc->Save();
         }
#endif
      }
   }

   // Output final computation time
   double computation_time;
   if (Mpi::Root())
   {
      t_end = clock();
      computation_time = ((float)(t_end - t_start))/CLOCKS_PER_SEC;
      cout << "Single processor final computation time: "
           << computation_time << " seconds.\n";
   }

   // 12. Save the final solution in parallel. This output can be viewed later
   //     using GLVis: "glvis -np <np> -m ex9p-continuous-mesh -g ex9p-continuous-final".
   {
      *u = *U;
      ostringstream sol_name;
      sol_name << "ex9p-continuous-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   // 13. Save the exact solution and compute the error. This can be viewed
   //     later using: "glvis -np <np> -m ex9p-continuous-mesh -g ex9p-continuous-final-exact"
   {
      FunctionCoefficient u_exact(exact_sol);
      u_exact.SetTime(t);
      ParGridFunction *u_ex = new ParGridFunction(fes);
      u_ex->ProjectCoefficient(u_exact);
      ostringstream e_sol_name;
      e_sol_name << "ex9p-continuous-final-exact." << setfill('0') << setw(6) << myid;
      ofstream e_osol(e_sol_name.str().c_str());
      e_osol.precision(precision);
      u_ex->Save(e_osol);

      double L1_error = u->ComputeL1Error(u_exact);
      double L2_error = u->ComputeL2Error(u_exact);
      double Linf_error = u->ComputeMaxError(u_exact);

      if (myid == 0)
      {
         cout << "\n|| E_h - E ||_{L^1} = " << L1_error << '\n' << endl;
         cout << "\n|| E_h - E ||_{L^2} = " << L2_error << '\n' << endl;
         cout << "\n|| E_h - E ||_{L^inf} = " << Linf_error << '\n' << endl;
      }
      if (Mpi::Root())
      {
         ostringstream convergence_filename;
         convergence_filename << "ex9p-analysis/ex9pc_np" << num_procs;// num_procs
         if (ser_ref_levels != 0) {
            convergence_filename << "_s" << setfill('0') << setw(2) << ser_ref_levels;
         }
         if (par_ref_levels != 0) {
            convergence_filename << "_p" << setfill('0') << setw(2) << par_ref_levels;
         }
         convergence_filename << "_refinement_"
                              << setfill('0') << setw(2)
                              << to_string(par_ref_levels + ser_ref_levels)
                              << ".out";
         ofstream convergence_file(convergence_filename.str().c_str());
         convergence_file.precision(precision);
         convergence_file << "Processor_Runtime " << computation_time << "\n"
                          << "n_processes " << num_procs << "\n"
                          << "n_refinements "
                          << to_string(par_ref_levels + ser_ref_levels) << "\n"
                          << "n_Dofs " << global_vSize << "\n"
                          << "h " << hmin << "\n"
                          << "L1_Error " << L1_error << "\n"
                          << "L2_Error " << L2_error << "\n"
                          << "Linf_Error " << Linf_error << "\n"
                          << "dt " << dt << "\n"
                          << "Endtime " << t << "\n";
         convergence_file.close();
      }

      delete u_ex;
   }

   // 15. Free the used memory.
   delete U;
   delete u;
   delete k;
   delete m;
   delete fes;
   delete pmesh;
   delete ode_solver;
   delete pd;
#ifdef MFEM_USE_ADIOS2
   if (adios2)
   {
      delete adios2_dc;
   }
#endif
   delete dc;

   return 0;
}


/*
Implementation of class FE_Evolution
*/
FE_Evolution::FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_)
   : TimeDependentOperator(K_.Height()),
     pfes(*(K_.ParFESpace())),
     M_solver(pfes.GetComm()),
     z(K_.Height()),
     dij_matrix(K_.SpMat()),
     D_form(&K_),
     K_spmat(K_.SpMat()),
     K_smap(),
     lumpedM()
{
   M_.FormSystemMatrix(ess_tdof_list, Mmat);
   Mmat.GetDiag(lumpedM);
   K_.FormSystemMatrix(ess_tdof_list, Kmat);
   row_starts = Kmat.GetRowStarts();
   col_starts = Kmat.GetColStarts();

   // M_solver.iterative_mode = false;
   // M_solver.SetRelTol(1e-9);
   // M_solver.SetAbsTol(0.0);
   // M_solver.SetTol(1e-9);
   // M_solver.SetMaxIter(100);
   // M_solver.SetPrintLevel(0);
   // M_prec = new HypreBoomerAMG;
   // M_solver.SetPreconditioner(*M_prec);
   // M_solver.SetOperator(Mmat);

   // Assuming K is finalized. (From extrapolator.)
   // K_smap is used later to symmetrically form dij_matrix
   const int *I = K_spmat.GetI(), *J = K_spmat.GetJ(), n = K_spmat.Size();
   K_smap.SetSize(I[n]);
   for (int row = 0, j = 0; row < n; row++)
   {
      for (int end = I[row+1]; j < end; j++)
      {
         int col = J[j];
         // Find the offset, _j, of the (col,row) entry and store it in smap[j].
         for (int _j = I[col], _end = I[col+1]; true; _j++)
         {
            MFEM_VERIFY(_j != _end, "Can't find the symmetric entry!");

            if (J[_j] == row) { K_smap[j] = _j; break; }
         }
      }
   }
}

// TODO: Implement an ImplicitSolve function
// Solve the equation:
//    u_t = M^{-1}(Ku),
// by solving associated linear system
//    (M - dt*K) d = K*u
void FE_Evolution::ImplicitSolve(const double dt, const Vector &x, Vector &k)
{
   Kmat.Mult(x, z);
}

void FE_Evolution::build_dij_matrix(const Vector &U,
                                    const VectorFunctionCoefficient &velocity)
{
   const int *I = K_spmat.HostReadI(), *J = K_spmat.HostReadJ(), n = K_spmat.Size();

   const double *K_data = K_spmat.HostReadData();

   double *D_data = dij_matrix.HostReadWriteData();
   dij_matrix.HostReadWriteI(); dij_matrix.HostReadWriteJ();

   for (int i = 0, k = 0; i < n; i++)
   {
      double rowsum = 0.;
      for (int end = I[i+1]; k < end; k++)
      {
         int j = J[k];
         if (i != j)
         {
            double kij = K_data[k];
            double kji = K_data[K_smap[k]];
            double dij = fmax(abs(kij),abs(kji));

            D_data[k] = dij;
            D_data[K_smap[k]] = dij;
            rowsum += dij;
         }
      }
      dij_matrix(i,i) = -rowsum;
   }
   D = D_form->ParallelAssemble(&dij_matrix);
   // D = new HypreParMatrix(MPI_COMM_WORLD, row_starts, col_starts, &dij_matrix);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = Ml^{-1} ((D + K) x)
   D->Mult(x, z);

   Vector rhs(y.Size());
   Kmat.Mult(x, rhs);
   z+= rhs;

   // M_solver.Mult(z, y);
   const int s = y.Size();
   for (int i = 0; i < s; i++)
   {
      y(i) = z(i) / lumpedM(i);
   }
}

FE_Evolution::~FE_Evolution()
{
   delete D;
   delete M_prec;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      {
         // Clockwise rotation in 2D around the origin
         const double w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
            case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 3:
      {
         // Clockwise twisting rotation in 2D around the origin
         const double w = M_PI/2;
         double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 4:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(1./2.); v(1) = sqrt(1./2.); break;
            case 3: v(0) = sqrt(1./3.); v(1) = sqrt(1./3.); v(2) = sqrt(1./3.);
               break;
         }
         break;
      }
   }
}

// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 4: // Putting case 4 at the beginning to default to latter cases
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                        erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const double f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}

// Exact Solution u(x,t) (so far only implemented for case 0)
// Note that u(x,0) = u_0(x)
double exact_sol(const Vector &x, const double t)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   // Get velocity
   Vector v = X;
   velocity_function(X, v);
   // cout << "velocity: " << v[0] << " " << v[1] << endl;
   // cout << "time: " << t << endl;
   // cout << "problem: " << problem << endl;
   // cout << "dim: " << dim << endl;

   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
            {
               double val = exp(-40.*pow(X(0)-0.5-v[0]*t,2));
               return val;
            }
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               double val = ( erfc(w*(X(0)-v[0]*t - cx-rx))*erfc(-w*(X(0)-v[0]*t-cx+rx)) *
                        erfc(w*(X(1)-v[1]*t - cy-ry))*erfc(-w*(X(1)-v[1]*t-cy+ry)) )/16;
               cout << "ret val: " << val << endl;
               return val;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const double f = M_PI;
         return sin(f*(X(0) - v[0]*t))*sin(f*(X(1)-v[1]*t));
      }
      case 4:
      {
         switch (dim)
         {
            case 1:
            {
               double coeff = M_PI /(bb_max[0] - bb_min[0]);
               double val = sin(coeff*(X[0] - v[0]*t));
               return val;
            }
            case 2:
            {
               Vector coeff = v;
               for (int i = 0; i < dim; i++)
               {
                  coeff[i] = 2 * M_PI / (bb_max[i] - bb_min[i]);
               }
               double val = sin(coeff[0]*(X[0]-v[0]*t))*sin(coeff[1]*(X[1]-v[1]*t));
               return val;
            }
            case 3:
            {
               return 0;
            }
         }
      }
   }
   return 100.;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: return 0.0;
   }
   return 0.0;
}
