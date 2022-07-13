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
void test_sparse_matrices(const SparseMatrix &A, const SparseMatrix &AT);
void create_global_expansion_matrix(ParFiniteElementSpace &pfes, SparseMatrix & spm);
void test_hpm_pgf(HypreParVector U, HypreParMatrix k);

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
    du/dt = M^{-1} (D + K) u, and this class is used to evaluate the right-hand side.
*/
class FE_Evolution : public TimeDependentOperator
{
private:
   ParFiniteElementSpace &pfes;
   ParMesh &pmesh;
   Array<int> ess_tdof_list; 
   Array<HYPRE_BigInt> gi;

   HypreParMatrix M_hpm, *K_hpm, *KT_hpm;
   HypreParVector lumpedM;
   ParBilinearForm *D_form;
   ParBilinearForm &K_pbf, &KT_pbf;
   HypreSolver *M_prec;
   HyprePCG M_solver; // Symmetric system, can use CG

   SparseMatrix dij_matrix;
   SparseMatrix dij_sparse;
   SparseMatrix K_spmat, KT_spmat;
   SparseMatrix K_spmat_wide;
   HypreParMatrix *D;
   HypreParMatrix *W;

   Array<int> K_smap;

   double timestep;

public:
   FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, ParBilinearForm &KT_);

   /** FE_Evolution::build_dij_matrix
   Builds dij_matrix used in the low order approximation, which is based on
   Guermond/Popov 2016.
   */
   void build_dij_matrix(const Vector &U,
                         const VectorFunctionCoefficient &velocity) ;
   void apply_dij(const Vector &u, Vector &du) const;
   void calculate_timestep();
   double get_timestep();

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
   double t_final = 2.84; // Period of the exact solution for p.4
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
   if (match_dt_to_h) { dt = hmin/2.; }
   if (one_time_step) { t_final = dt; }

   // 7. Define the parallel H1 finite element space on the
   //    parallel refined mesh of the given polynomial order.
   // H1_FECollection fec(order, dim, BasisType::Positive);
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_BigInt global_vSize = fes->GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << global_vSize << endl;
      cout << "Dim: " << dim << endl;
   }

   // 8. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the H1 discretization.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(exact_sol);

   ParBilinearForm *m = new ParBilinearForm(fes);
   ParBilinearForm *k = new ParBilinearForm(fes);
   ParBilinearForm *kT = new ParBilinearForm(fes);

   if (pa)
   {
      m->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      k->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      kT->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   else if (ea)
   {
      m->SetAssemblyLevel(AssemblyLevel::ELEMENT);
      k->SetAssemblyLevel(AssemblyLevel::ELEMENT);
      kT->SetAssemblyLevel(AssemblyLevel::ELEMENT);
   }
   else if (fa)
   {
      m->SetAssemblyLevel(AssemblyLevel::FULL);
      k->SetAssemblyLevel(AssemblyLevel::FULL);
      kT->SetAssemblyLevel(AssemblyLevel::FULL);
   }

   m->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   constexpr double alpha = -1.0;
   k->AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
   kT->AddDomainIntegrator(new ConservativeConvectionIntegrator(velocity, -alpha));

   int skip_zeros = 0;
   m->Assemble();
   k->Assemble(skip_zeros);
   kT->Assemble(skip_zeros);
   m->Finalize();
   k->Finalize(skip_zeros);
   kT->Finalize(skip_zeros);

   // 9. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   ParGridFunction *u = new ParGridFunction(fes);
   u->ProjectCoefficient(u0);
   HypreParVector *U = u->GetTrueDofs();
   *U = 1.; // Set U to ones for testing row sums and mat-vec multiplication

   HypreParMatrix * K = k->ParallelAssemble(); 
   // HypreParMatrix *K_e = k->ParallelAssembleElim();
   // cout << "K size: (" << K->Height() << "," << K->Width() << ")" << endl;
   // cout << "K_e size: (" << K_e->Height() << "," << K_e->Width() << ")" << endl;
   cout << "pfes.GetVSize(): " << fes->GetVSize() << endl;
   cout << "pfes.GetTrueVSize(): " << fes->GetTrueVSize() << endl;
   cout << "pfes.GetFaceNbrVSize(): " << fes->GetFaceNbrVSize() << endl;

   test_hpm_pgf(*U, *K);
   assert(false);

   
   // SparseMatrix K_diag;
   // K->GetDiag(K_diag);  

   // if (Mpi::Root())
   // {
   //    cout << "Root:\n"
   //         << "k size: " << k->SpMat().Height() << "," << k->SpMat().Width() << endl
   //         << "u size: " << u->Size() << " U size: " << U->Size() << endl
   //         << "K_diag size: " << K_diag.Height() << "," << K_diag.Width() << endl;
   // }

   ConstantCoefficient zero(0.0);

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
   FE_Evolution adv(*m, *k, *kT);

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   // dij_matrix has no time dependence
   adv.build_dij_matrix(*u, velocity);
   // adv.calculate_timestep();

   // Verify our timestamp satisfies the cfl condition
   // assert (dt <= adv.get_timestep());

   bool done = false;
   for (int ti = 0; !done; )
   {
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
         // *u = *U; // Synchronizes MPI processes.

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
      double zero_L2_error = u->ComputeL2Error(zero);

      if (myid == 0)
      {
         cout << "\n|| E_h - E ||_{L^1} = " << L1_error << '\n' << endl;
         cout << "\n|| E_h - E ||_{L^2} = " << L2_error << '\n' << endl;
         cout << "\n|| E_h - E ||_{L^inf} = " << Linf_error << '\n' << endl;
         cout << "\n|| E ||_{L^2} = " << zero_L2_error << '\n' << endl;
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

   cout << "Freeing memory\n";
   // 15. Free the used memory.
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

   cout << "Done Freeing memory!\n";

   return 0;
}


/*
Implementation of class FE_Evolution
*/
FE_Evolution::FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, ParBilinearForm &KT_)
   : TimeDependentOperator(K_.Height()),
     pfes(*(K_.ParFESpace())),
     pmesh(*(pfes.GetParMesh())),
     M_solver(pfes.GetComm()),
     D_form(&K_),
     K_smap(),
     lumpedM(&pfes),
     timestep(0.),
     K_pbf(K_),
     KT_pbf(KT_)
   //   ,
   //   dij_matrix(K_pbf.Height())
{
   if (pfes.GetParMesh()->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   pmesh.GetGlobalVertexIndices(gi);

   cout << "Initialize FE_Evolution class.\n";
   M_.FormSystemMatrix(ess_tdof_list, M_hpm);
   M_hpm.GetDiag(lumpedM);

   K_hpm = K_pbf.ParallelAssemble();
   K_hpm->MergeDiagAndOffd(K_spmat);
   // create_global_expansion_matrix(pfes, K_spmat);
   dij_matrix = K_spmat;

   DenseMatrix * den = K_spmat.ToDenseMatrix();
   DenseMatrix denT;
   den->Transpose(denT);
   Vector sums_;
   denT.GetRowSums(sums_);
   // sums_.Print(cout);
   // assert(false);

   KT_hpm = KT_pbf.ParallelAssemble();
   KT_hpm->MergeDiagAndOffd(KT_spmat);

   // cout << "K_spmat size: (" << K_spmat.Height() << ',' << K_spmat.Width() << ")\n";
   // cout << "KT_spmat size: (" << KT_spmat.Height() << ',' << KT_spmat.Width() << ")\n";

   M_solver.iterative_mode = false;
   // M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetTol(1e-9);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
   M_prec = new HypreBoomerAMG;
   M_solver.SetPreconditioner(*M_prec);
   M_solver.SetOperator(M_hpm);

   // Assuming K is finalized. (From extrapolator.)
   // K_smap is used later to symmetrically form dij_matrix
//    const int *I = K_spmat.GetI(), *J = K_spmat.GetJ(), n = K_spmat.Size();
//    K_smap.SetSize(I[n]);
//    for (int row = 0, j = 0; row < n; row++)
//    {
//       for (int end = I[row+1]; j < end; j++)
//       {
//          int col = J[j];
//          // Find the offset, _j, of the (col,row) entry and store it in smap[j].
//          for (int _j = I[col], _end = I[col+1]; true; _j++)
//          {
//             MFEM_VERIFY(_j != _end, "Can't find the symmetric entry!");

//             if (J[_j] == row) { K_smap[j] = _j; break; }
//          }
//       }
//    }
//    cout << "End FEEvolution constructor.\n";
}

// TODO: Implement an ImplicitSolve function
// Solve the equation:
//    u_t = M^{-1}(Ku),
// by solving associated linear system
//    (M - dt*K) d = K*u
void FE_Evolution::ImplicitSolve(const double dt, const Vector &x, Vector &k)
{
   // K_hpm.Mult(x, z);
}

/******************************************************************************
 * FE_Evolution::build_dij_matrix(const Vector &U,
                                    const VectorFunctionCoefficient &velocity)
 * Purpose:
 * ***************************************************************************/
void FE_Evolution::build_dij_matrix(const Vector &U,
                                    const VectorFunctionCoefficient &velocity)
{
   cout << "Build dij\n";
   cout << "dij size: " << dij_matrix.Height() << "," << dij_matrix.Width() << endl;

   const int *I = K_spmat.HostReadI(), *J = K_spmat.HostReadJ();
   const int n = K_spmat.Size(), n_local = 0;

   const double *K_data = K_spmat.HostReadData();
   const double *KT_data = KT_spmat.HostReadData();

   for (int i = 0, k = 0; i < n; i++)
   {
      for (int end = I[i+1]; k < end; k++)
      {
         int j = J[k];
         if (j > i) // We only need to look at the upper diagonal since we have access to the transpose.
         {
            double kij = K_data[k];
            double kji = KT_data[k];
            double dij = fmax(abs(kij), abs(kji));
            
            dij_matrix(i,j) = dij;
            if (j < n) { dij_matrix(j,i) = dij; }
         }
      }
   }

   // TODO: better way to set row sums?
   Vector row_sums(n);
   dij_matrix.GetRowSums(row_sums);
   row_sums *= -1;
   
   for (int i = 0; i < n; i++)
   {
      dij_matrix(i,i) = row_sums[i];
   }

   cout << "dij_matrix size: (" << dij_matrix.Height() << ',' << dij_matrix.Width() << ")\n";

   // Check that our matrix dij is symmetric.
   // if (dij_matrix.IsSymmetric()) {
   //    cout << "ERROR: dij matrix must be symmetric.\n";
   //    cout << "Val: " << dij_matrix.IsSymmetric() << endl;
   //    return;
   // }
   
   // D = D_form->ParallelAssemble(&dij_matrix);
   // D = new HypreParMatrix(MPI_COMM_WORLD, pfes.GlobalVSize(), pfes.GetDofOffsets(), &dij_matrix);  
   // W = RAP(D, pfes.Dof_TrueDof_Matrix());
   cout << "Finished dij matrix.\n";
}

/******************************************************************************
 * FE_Evolution::apply_dij(ParGridFunction &u, Vector &du) const
 * Purpose:
 * ***************************************************************************/
// void FE_Evolution::apply_dij(const Vector &u, Vector &du) const
// {
//    const int s = u.Size();
//    const int *I = dij_matrix.HostReadI(), *J = dij_matrix.HostReadJ();
//    const double *D_data = dij_matrix.HostReadData();
   
//    for (int i = 0; i < s; i++)
//    {
//       du(i) = 0.;
//       for (int k = I[i]; k < I[i + 1]; k++)
//       {
//          int j = J[k];

//          du(i) += D_data[k] * u(j);
//       }
//    }
// }

/******************************************************************************
 * FE_Evolution::calculate_timestep()
 * Purpose:
 *    Compute maximum timestep according to global CFL condition outline in
 *    Corollary 4.2 in Guermond 2016.
 * ***************************************************************************/
void FE_Evolution::calculate_timestep()
{
   cout << "Calculating timestep.\n"; 
   int n = lumpedM.Size();
   double t_min = 0;
   double t_temp = 0;
   Vector dii;
   dij_matrix.GetDiag(dii);

   for (int i = 0; i < n; i++) 
   {
      t_temp = lumpedM(i) / (2. * abs(dii[i]));
      if (t_temp > t_min) { t_min = t_temp; }
   }

   this->timestep = t_min;
   cout << "Finished calculating timestep.\n";
}

/******************************************************************************
 * FE_Evolution::get_timestep()
 * Purpose:
 *    Retrieve timestep.
 * ***************************************************************************/
double FE_Evolution::get_timestep()
{
   return timestep;
}

/******************************************************************************
 * FE_Evolution::Mult()
 * Purpose:
 * ***************************************************************************/
void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   cout << "FE_Evolution::Mult()\n";
   ParGridFunction u(&pfes), z(&pfes), rhs(&pfes), temp(&pfes);
   u = x;
   K_pbf.Mult(u, z);

   const int s = u.Size();

   // dij_matrix.Mult(u, rhs);
   // z += rhs;

   HypreParVector *U = z.GetTrueDofs();
   int s_true = U->Size();

   assert(lumpedM.Size() == s_true);

   // for (int i = 0; i < s_true; i++)
   // {
   //    U[i] = U[i] / lumpedM(i);
   // }
   M_solver.Mult(z, temp);
   u.SetFromTrueDofs(*temp.GetTrueDofs());
   y = u;

   assert (y.Size() == x.Size());
   cout << "FE_Evolution::Mult() end.\n";
}

/******************************************************************************
 * FE_Evolution::~FE_Evolution()
 * Purpose:
 * ***************************************************************************/
FE_Evolution::~FE_Evolution()
{
   // delete D;
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

   // Reset the velocity vector
   v = 0.0;

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
      case 5:
      {
         switch (dim)
         {
            case 1: v(0) = 1; break;
            case 2: v(0) = sqrt(1./2.); v(1) = sqrt(1./2.); break;
            case 3: v(0) = sqrt(1./3.); v(1) = sqrt(1./3.); v(2) = sqrt(1./3.);
               break;
         }
         break;
      }
   }
}

// Exact Solution u(x,t) 
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
               Vector coeff = v;
               for (int i = 0; i < dim; i++)
               {
                  coeff[i] = 2 * M_PI / (bb_max[i] - bb_min[i]);
               }
               double val = sin(coeff[0]*(X[0]-v[0]*t))*sin(coeff[1]*(X[1]-v[1]*t))*sin(coeff[2]*(X[2]-v[2]*t));
               return val;
            }
         }
      }
      case 5: // step function
      {
         switch (dim)
         {
            case 1:
            case 2:
            {
               if (pow(X[0],2) + pow(X[1], 2) < 0.25)
               {
                  return 1.;
               }
               else{
                  return 0.;
               }
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
      case 3:
      case 4: return 0.0;
   }
   return 0.0;
}

// Function to test the matrix generated by the conservative convection integrator is in fact
// the right matrix we need
void test_sparse_matrices(const SparseMatrix &A, const SparseMatrix &AT)
{
   const int *I = A.HostReadI(), *J = A.HostReadJ();
   const int n = A.Size();
   const double *K_data = A.HostReadData();

   const int *IT = AT.HostReadI(), *JT = AT.HostReadJ();
   const int nT = AT.Size();
   const double *KT_data = AT.HostReadData();

   if (!Mpi::Root())
   {
      cout << "n: " << n << " nT: " << nT << endl;

      for (int i = 0, k = 0; i < n; i++)
      {
         // int end = I[i+1], endT = IT[i+1];
         // cout << "row i, end: " << end << " endT: " << endT << endl;
         for (int end = I[i+1]; k < end; k++)
         {
            int j = J[k], jT = JT[k];
            if (j != jT) { 
               cout << "row i, j: " << j << " jT: " << jT << endl;
            }
            // if (i != j)
            // {
            //    double kij = A[k];
            //    double kji = AT[k];
            // }
            
         }
      }
   }
}

void create_global_expansion_matrix(ParFiniteElementSpace &pfes, SparseMatrix & spm)
{
   const int *I = spm.HostReadI(), *J = spm.HostReadJ();
   const int n = spm.Height();
   const double *K_data = spm.HostReadData();

   for (int i = 0, k = 0, i_new = 0; i < n; i++)
   {
      for (int end = I[i+1]; k < end; k++)
         {
            int j = J[k];
            cout << "row i: " << i << " j: " << j << " local j: " << pfes.GetLocalTDofNumber(j) << endl;
         }
   }
   cout << "Num of face neighbord dofs: " << pfes.num_face_nbr_dofs << endl;
}

// void test_hpm_pgf(HypreParVector U, HypreParMatrix K_hpm)
// {
//    if (!Mpi::Root()) // Just output results on one processor.
//    {
//       // U = 1; // tdof
//       cout << "Testing row sums\n";
//       SparseMatrix K_spm, K_diag;
//       K_hpm.GetDiag(K_spm); // tdof x tdof
//       K_hpm.MergeDiagAndOffd(K_diag); // tdof x gdof
//       cout << "K_hpm size: (" << K_hpm.Height() << "," << K_hpm.Width() << ")" << endl;
//       cout << "K_spm size: (" << K_spm.Height() << "," << K_spm.Width() << ")" << endl;
//       cout << "K_diag size: (" << K_diag.Height() << "," << K_diag.Width() << ")" << endl;
//       cout << "U size: " << U.Size() << endl;

//       // Vector z(U.Size());
//       // K_hpm.Mult(U, z);
//       // cout << "Resulting vector z values: \n";
//       // z.Print(cout);
//       Vector y(U.Size());
//       K_spm.Mult(U, y); // Test to see how HypreParVector handles multiplication with mismatched sizes.
//       cout << "Resulting vector y values: \n";
//       y.Print(cout);
//       Vector DiagRowSums(K_diag.Height()), SpmRowSums(K_spm.Height());
//       K_diag.GetRowSums(DiagRowSums);
//       K_spm.GetRowSums(SpmRowSums);
//       cout << "(tdof x gdof) Row Sums: \n";
//       DiagRowSums.Print(cout);
//       cout << "K_diag spmat.Print()\n";
//       K_diag.Print(cout);

//       cout << "(tdof x tdof) Row Sums: \n";
//       SpmRowSums.Print(cout);
//       cout << "K_spm spmat.Print()\n";
//       K_spm.Print(cout);

//       assert(false); // terminate the program early
//    }
   
// }

void test_hpm_pgf(HypreParVector U, HypreParMatrix K_hpm)
{
   // *U = 1.; // Set U to ones for testing row sums and mat-vec multiplication
   Vector y(U.Size());

   if (!Mpi::Root()) // Just output results on one processor.
   {
      // U = 1; // tdof
      cout << "Testing row sums\n";
      SparseMatrix K_spm_diag, K_spm_merged;
      K_hpm.GetDiag(K_spm_diag); // tdof x tdof
      K_hpm.MergeDiagAndOffd(K_spm_merged); // tdof x gdof
      cout << "K_spm_diag size: ("
           << K_spm_diag.Height() << "," << K_spm_diag.Width() << ")\n";
      cout << "K_spm_merged size: ("
           << K_spm_merged.Height() << "," << K_spm_merged.Width() << ")\n";
      cout << "U size: " << U.Size() << endl;

      Vector U_full(K_spm_merged.Width()); U_full = 1.0;

      // See how HypreParVector handles multiplication with mismatched sizes.
      K_spm_merged.Mult(U_full, y);
      cout << "Resulting vector y values: \n";
      y.Print();
      cout << endl;

      Vector DiagRowSums(K_spm_diag.Height()),
             MergedRowSums(K_spm_merged.Height());
      K_spm_merged.GetRowSums(MergedRowSums);
      K_spm_diag.GetRowSums(DiagRowSums);
      cout << "Diag (tdof x tdof) Row Sums: \n"; DiagRowSums.Print();
      cout << "Merged (tdof x gdof) Row Sums: \n"; MergedRowSums.Print();
   }

   y = 0.0;
   K_hpm.Mult(U, y);
   if (!Mpi::Root()) { cout << endl; y.Print(); }
}
