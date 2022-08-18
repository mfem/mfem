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
//    mpirun -np 4 ex9p-continuous -m ../data/ref-segment.mesh -ot -rs 1 -rp 0 -p 6 -f 2 -tf 0.5
//    mpirun -np 4 ex9p-continuous -m ../data/ref-square.mesh -ot -rs 1 -rp 0 -p 6 -f 2 -tf 0.5
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

// Choice for the flux. The form K and Dij is determined based on this value.
int flux;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);
void test_sparse_matrices(const SparseMatrix &A, const SparseMatrix &AT);
void create_global_expansion_matrix(ParFiniteElementSpace &pfes, SparseMatrix & spm);
void test_hpm_pgf(HypreParVector U, HypreParMatrix k);
SparseMatrix build_sparse_from_dense(DenseMatrix & dm);

// Exact solution
double exact_sol(const Vector &x, double t);
double zero_sol(const Vector &x, double t) { return 0;}

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

class MyCoefficient : public Coefficient
{
private:
   const double timestep;
   const GridFunction *GridF;
   const GridFunction *GridF_galerkin;

public:
   MyCoefficient() : GridF(NULL), GridF_galerkin(NULL), timestep(0.) { }
   // Construct GridFunctionCoefficient from multiple GridFunctions
   MyCoefficient(const GridFunction *gf1, const GridFunction *gf2, const int tau)
      : timestep(tau)
   {
      GridF = gf1;
      GridF_galerkin = gf2;
   }

   // Evaluate the coeffficient at a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

double MyCoefficient::Eval (ElementTransformation &T,
                            const IntegrationPoint &ip)
{
   Mesh *gf_mesh = GridF->FESpace()->GetMesh();
   if (T.mesh == gf_mesh)
   {
      Vector grad;
      GridF->GetGradient(T, grad);
      double val = (GridF_galerkin->GetValue(T, ip) - GridF->GetValue(T, ip)) / timestep;
      val += GridF->GetValue(T, ip) * grad[0]; // hardcoded for 1D
      val *= GridF->GetValue(T, ip);
      return val;
   }
   // else
   // {
   //    IntegrationPoint coarse_ip;
   //    ElementTransformation *coarse_T = RefinedToCoarse(*gf_mesh, T, ip, coarse_ip);
   //    return GridF->GetValue(*coarse_T, coarse_ip, Component);
   // }
   assert (false);
   return 0;
}

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

   HypreParMatrix *M_hpm, *K_hpm, *KT_hpm;
   HypreParVector lumpedM;
   ParBilinearForm &K_pbf, &KT_pbf;
   SparseMatrix dij_sparse, K_spmat, KT_spmat, entropy_viscosity;
   Vector dii, entropy_viscosity_dii;

   double timestep;
   bool high_order;

public:
   FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, ParBilinearForm &KT_);

   /** FE_Evolution::build_dij_matrix
   Builds dij_matrix used in the low order approximation, which is based on
   Guermond/Popov 2016.
   */
   void set_high_order(bool order);
   void build_dij_matrix(const Vector &U);
   void calculate_timestep();
   double get_timestep();

   Vector compute_entropy_min(const Vector &U);
   Vector compute_entropy_max(const Vector &U);
   Vector compute_entropy_res(const Vector &U, const GridFunction *gf1, const GridFunction *gf2);

   void Update(const Vector &x);
   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   virtual ~FE_Evolution();

private:
   void build_entropy_viscosity(const Vector R);
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
   flux = 1;
   const char *mesh_file = "../data/periodic-square.mesh";
   int ser_ref_levels = 3;
   int par_ref_levels = 0;
   int order = 1;
   bool high = false;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 1;
   double t_final = 2.84; // Period of the exact solution for p.4
   bool one_time_step = false;
   bool optimize_timestep = false;
   double dt = 0.01;
   bool match_dt_to_h = false;
   bool gif = false;
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
   args.AddOption(&flux, "-f", "--flux",
                  "Flux type: 1 - Linear Transport,\n\t"
                  "           2 - Burgers");
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
   args.AddOption(&high, "-ho", "--high-order", "-lo", "--low-order",
                  "Set order of FEM approximation.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&one_time_step, "-ots", "--one-time-step",
                  "-no-ots", "--no-one-time-step",
                  "Set end time to one time step for convergence testing.");
   args.AddOption(&optimize_timestep, "-ot", "--optimize-timestep",
                  "-no-ot", "--no-optimize-timestep",
                  "Set timestep according to CFL.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&match_dt_to_h, "-ct", "--conv-test",
                  "-no-ct", "--no-conv-test",
                  "Enable convergence testing by matching dt to h.");
   args.AddOption(&gif, "-gif", "--save-files-for-gif", "-no-gif", 
                  "--dont-save-files-for-gif",
                  "Enable or disable file output for gif creation.");
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
   if (match_dt_to_h) { dt = hmin; }
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

   // 8. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(exact_sol);
   ParGridFunction *u = new ParGridFunction(fes);
   u->ProjectCoefficient(u0);
   HypreParVector *U = u->GetTrueDofs();

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

   // 9. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the H1 discretization.
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

   switch (flux) 
   {
      case 1:
      {
         VectorFunctionCoefficient velocity(dim, velocity_function);
         k->AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
         kT->AddDomainIntegrator(new ConservativeConvectionIntegrator(velocity, -alpha));
         break;
      }
      case 2:
      {
         VectorGridFunctionCoefficient velocity(u);
         k->AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
         kT->AddDomainIntegrator(new ConservativeConvectionIntegrator(velocity, -alpha));
         break;
      }
   }

   int skip_zeros = 0;
   m->Assemble();
   k->Assemble(skip_zeros);
   kT->Assemble(skip_zeros);
   m->Finalize();
   k->Finalize(skip_zeros);
   kT->Finalize(skip_zeros);

   // 10. Define the time-dependent evolution operator describing the ODE
   //     right-hand side, and perform time-integration (looping over the time
   //     iterations, ti, with a time-step dt).
   FE_Evolution adv(*m, *k, *kT);

   double t = 0.0;
   adv.SetTime(t);
   adv.set_high_order(high);
   ode_solver->Init(adv);

   // dij_matrix has no time dependence
   adv.build_dij_matrix(*u);
   adv.calculate_timestep();

   // Optimize time step, unless running convergence analysis
   if (optimize_timestep && !match_dt_to_h)
   {
      dt = adv.get_timestep(); // According to CFL, take min across processors.
   }

   // assert(dt <= adv.get_timestep()); // In either case, we must satisfy the CFL.
   if (dt > adv.get_timestep()) {
      cout << "CFL condition not satisfied.\n";
   }

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);
      adv.Update(*U);
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

         if (gif)
         {
            ostringstream sol_name;
            sol_name << "ex9p-continuous:" << to_string(double(t/t_final)) << ":." << setfill('0') << setw(6) << myid;
            ofstream osol(sol_name.str().c_str());
            osol.precision(precision);
            u->Save(osol);
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


   // 15. Free the used memory.
   cout << "Freeing memory\n";
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
     lumpedM(&pfes),
     timestep(0.),
     K_pbf(K_),
     KT_pbf(KT_),
     dii(pfes.GetTrueVSize()),
     entropy_viscosity_dii(pfes.GetTrueVSize())
{
   if (pfes.GetParMesh()->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   cout << "Initialize FE_Evolution class.\n";
   M_hpm = M_.ParallelAssemble();
   M_hpm->GetDiag(lumpedM);

   K_hpm = K_pbf.ParallelAssemble();
   K_hpm->MergeDiagAndOffd(K_spmat);
   dij_sparse = K_spmat;
   entropy_viscosity = K_spmat;

   KT_hpm = KT_pbf.ParallelAssemble();
   KT_hpm->MergeDiagAndOffd(KT_spmat);

   // Set the method to 1st order by default
   high_order = false;

   cout << "End FEEvolution constructor.\n";
}

/******************************************************************************
 * FE_Evolution::set_high_order()
 * Purpose: 
 *    Set whether to use the low order artificial viscosity method (2016 paper)
 *    or to use the high order viscosity method (2017 paper).
 * ***************************************************************************/
void FE_Evolution::set_high_order(bool order) { this->high_order = order; }

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
 *                                const VectorFunctionCoefficient &velocity)
 * Purpose:
 *    Build the artificial viscosity as described in (2016 paper).
 * ***************************************************************************/
void FE_Evolution::build_dij_matrix(const Vector &U)
{
   cout << "Build dij\n";

   const int m = dij_sparse.Height();
   const int *I = dij_sparse.HostReadI(), *J = dij_sparse.HostReadJ();

   double *D_data = dij_sparse.HostReadWriteData();
   const double *K_data = K_spmat.HostReadData();
   const double *KT_data = KT_spmat.HostReadData();

   for (int i = 0, k = 0; i < m; i++)
   {
      double rowsum = 0;
      for (int end = I[i+1]; k < end; k++)
      {
         int j = J[k]; // global index

         if (i != j) {
            double kij = K_spmat(i,j);
            double kji = KT_spmat(i,j);
            double dij = fmax(abs(kij), abs(kji));

            D_data[k] = dij;

            rowsum += dij;
         }
         else {
            D_data[k] = 0; // Need to clear entry before Mult()
         }
      }
      dii(i) = -1 *  rowsum; // To be used in Mult() in place of diagonal entries
   }

   cout << "Finished dij matrix.\n";
}

/******************************************************************************
 * FE_Evolution::build_entropy_viscosity()
 * Purpose: 
 *    Build entropy viscosity coefficient as described in (2017 paper).
 * ***************************************************************************/
void FE_Evolution::build_entropy_viscosity(const Vector R)
{
   // Assert we only run this function when needed.
   assert(high_order);

   cout << "Build entropy viscosity.\n";

   const int m = entropy_viscosity.Height();
   const int *I = entropy_viscosity.HostReadI(), *J = entropy_viscosity.HostReadJ();

   double *ev_data = entropy_viscosity.HostReadWriteData();
   const double *D_data = dij_sparse.HostReadData();

   for (int i = 0, k = 0; i < m; i++)
   {
      double rowsum = 0;
      for (int end = I[i+1]; k < end; k++)
      {
         int j = J[k]; // global index

         if (i != j) {
            double val = min(D_data[k], min(abs(R[i]), abs(R[j])));
            ev_data[k] = val; // TODO: min of three objects?
            rowsum += val;
         }
         else {
            ev_data[k] = 0; // Need to consider dii, not dij_sparse
         }
      }
      entropy_viscosity_dii(i) = -1 * rowsum;
   }

   cout << "Finished entropy viscosity.\n";
}

/******************************************************************************
 * FE_Evolution::calculate_timestep()
 * Purpose:
 *    Compute maximum timestep according to global CFL condition outline in
 *    Corollary 4.2 in Guermond 2016.
 * ***************************************************************************/
void FE_Evolution::calculate_timestep()
{
   int n = lumpedM.Size();
   double t_min = 1.;
   double t_temp = 0;

   for (int i = 0; i < n; i++)
   {
      assert(lumpedM(i) > 0); // Assumption, equation (3.6)
      assert(dii[i] < 0);

      t_temp = lumpedM(i) / (2. * abs(dii[i]));
      
      if (t_temp < t_min && t_temp > 1e-12) { t_min = t_temp; }
   }

   this->timestep = t_min;
}


/******************************************************************************
 * FE_Evolution::get_timestep()
 * Purpose:
 *    Retrieve timestep.
 * ***************************************************************************/
double FE_Evolution::get_timestep()
{
   MPI_Allreduce(MPI_IN_PLACE,
                 &this->timestep,
                 1,
                 MPI_DOUBLE,
                 MPI_MIN,
                 pmesh.GetComm());
   return timestep;
}

/******************************************************************************
 * FE_Evolution::compute_entropy_min()
 * Purpose: 
 *    Compute minimum entropy on support of shape function. Currently, this
 *    function assumes the entropy is u^2/2.
 * ***************************************************************************/
Vector FE_Evolution::compute_entropy_min(const Vector &U)
{
   // Assert we only run this function when needed.
   assert(high_order);

   const int m = dij_sparse.Height();
   const int *I = dij_sparse.HostReadI(), *J = dij_sparse.HostReadJ();
   Vector entropy_min(m);

   for (int i = 0, k = 0; i < m; i++)
   {
      double _min = 0;
      for (int end = I[i+1]; k < end; k++)
      {
         int j = J[k]; // global column index

         double val = pow(U[j],2)/2.;
         if (k == I[i])
         {
            _min = val; // Set the first entry.
         }
         else if (val < _min)
         {
            _min = val;
         }
      }
      entropy_min[i] = _min;
   }

   return entropy_min;
}

/******************************************************************************
 * FE_Evolution::compute_entropy_max()
 * Purpose: 
 *    Compute maximum entropy on support of shape function. Currently, this
 *    function assumes the entropy is u^2/2.
 * ***************************************************************************/
Vector FE_Evolution::compute_entropy_max(const Vector &U)
{   
   // Assert we only run this function when needed.
   assert(high_order);

   const int m = dij_sparse.Height();
   const int *I = dij_sparse.HostReadI(), *J = dij_sparse.HostReadJ();
   Vector entropy_max(m);

   for (int i = 0, k = 0; i < m; i++)
   {
      double _max = 0;
      for (int end = I[i+1]; k < end; k++)
      {
         int j = J[k]; // global column index

         double val = pow(U[j], 2) / 2.;
         if (val > _max)
         {
            _max = val;
         }
      }
      entropy_max[i] = _max;
   }

   return entropy_max;
}

/******************************************************************************
 * FE_Evolution::compute_entropy_res
 * Purpose: 
 *    Calculate the entropy residual as described in equation (43) in (2017 paper).
 * ***************************************************************************/
Vector FE_Evolution::compute_entropy_res(const Vector &U, const GridFunction *gf1, const GridFunction *gf2)
{
   // Assert we only run this function when needed.
   assert(high_order);

   Vector entropy_min = compute_entropy_min(U);
   Vector entropy_max = compute_entropy_max(U);

   ParLinearForm *R = new ParLinearForm(&pfes);
   MyCoefficient coeff(gf1, gf2, timestep);
   R->AddDomainIntegrator(new DomainLFIntegrator(coeff));
   Vector res;
   R->ParallelAssemble(res);

   for (int i = 0; i < res.Size(); i++)
   {
      double denom = entropy_max[i] - entropy_min[i];
      if (denom != 0)
      {
         res[i] = (res[i] * 2) / denom;
      }
      else
      {
         cout << "Divide by 0 avoided in compute_entropy_res().\n";
      }      
   }

   return res;
}

/******************************************************************************
 * FE_Evolution::Update()
 * Purpose: 
 *    Prepare the FE_Evolution class before the next mult call.  It is 
 *    necessary to make a call to the Update() function before calling
 *    Mult() if the method requires an update to dij_sparse or the 
 *    entropy viscosity is updated at each time step, as is the case
 *    in the high order method described in the (2017 paper).
 * ***************************************************************************/
void FE_Evolution::Update(const Vector &x)
{
   if (!high_order)
   {
      // Todo: optionally update dij_matrix if needed
      return;
   }
   else
   {
      // Update entropy viscosity
      const HypreParVector * U = dynamic_cast<const HypreParVector*>(&x);
      Vector * x_global = U->GlobalVector();
      int n = x.Size();

      // Preliminary assertions to ensure proper sizes
      assert(lumpedM.Size() == n);
      assert(dii.Size() == n);

      // 1. Solve Galerkin problem (equation 42)
      ParGridFunction * u_g = new ParGridFunction(&pfes);
      Vector z_g(n), y_g(n);
      K_hpm->Mult(*U, z_g);
      
      for (int i = 0; i < n; i++)
      {
         y_g[i] = z_g[i] / lumpedM(i);
      }

      // 2. Update the corresponding ParGridFunction
      *u_g = y_g;

      // 3. Compute entropy residual
      ParGridFunction *u = new ParGridFunction(&pfes);
      *u = *U;
      Vector R = compute_entropy_res(*U, u, u_g);

      // 5. Compute etropy viscosity
      this->build_entropy_viscosity(R);
   }
}

/******************************************************************************
 * FE_Evolution::Mult()
 * Purpose: 
 *    Low-order solution from (2016 paper) or high-order solution from
 *    (2017 paper).
 * ***************************************************************************/
void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   const HypreParVector * U = dynamic_cast<const HypreParVector*>(&x);
   Vector * x_global = U->GlobalVector();
   int n = x.Size();

   // Preliminary assertions to ensure proper sizes
   assert(lumpedM.Size() == n);
   assert(dii.Size() == n);
   y.SetSize(n); // TODO: Somehow y is of size local at this point. Resizing is a bandaid.

   // Apply K
   Vector z(n), rhs(n);
   K_hpm->Mult(*U, z);

   if (!high_order) // low-order viscosity
   {
      dij_sparse.Mult(*x_global, rhs);
      z += rhs;

      for (int i = 0; i < n; i++)
      {
         double diag_comp = dii(i) * x(i); // Leftover piece from rhs due to sparsity issue in dij.
         y[i] = ( z[i] + diag_comp ) / lumpedM(i);
      }
   }
   else // high-order viscosity
   {
      entropy_viscosity.Mult(*x_global, rhs);
      z += rhs;

      for (int i = 0; i < n; i++)
      {
         double diag_comp = entropy_viscosity_dii(i) * x(i); // Leftover piece from rhs due to sparsity issue in dij.
         y[i] = ( z[i] + diag_comp ) / lumpedM(i);
      }
   }

   assert (y.Size() == n);
}

/******************************************************************************
 * FE_Evolution::~FE_Evolution()
 * Purpose:
 * ***************************************************************************/
FE_Evolution::~FE_Evolution()
{
}

/******************************************************************************
 * velocity_function()
 * Purpose:
 *    Set the velocity for the various problems we are trying to solve.
 * ***************************************************************************/
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

/******************************************************************************
 * exact_sol()
 * Purpose:
 *    Set up the problem we are trying to solve.  At t=0, this function
 *    provices the initial conditions for the problem. If there exists
 *    a known exact solution, then that can be called using this function
 *    as well for the desired time.
 * ***************************************************************************/
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
         double coeff = M_PI;
         switch (dim)
         {
            case 1:
            {
               double val = sin(coeff*(X[0]-v[0]*t));
               return val;
            }
            case 2:
            {
               double val = sin(coeff*(X[0]-v[0]*t))*sin(coeff*(X[1]-v[1]*t));
               return val;
            }
            case 3:
            {
               double val = cos(coeff*(X[0]-v[0]*t)) * cos(coeff*(X[1]-v[1]*t)) * cos(coeff*(X[2]-v[2]*t));
               return val;
            }
         }
      }
      case 5: // step function
      {
         switch (dim)
         {
            case 1:
            {
               if (pow(fmod(x[0] - v[0]*t, 2.), 2) < 0.25)
               { 
                  return 1.;
               }
               else {
                  return 0.;
               }
            }
            case 2:
            {
               if (pow(X[0]-v[0]*t,2) + pow(X[1]-v[1]*t, 2) < 0.25)
               {
                  return 1.;
               }
               else{
                  return 0.;
               }
            }
            case 3:
            {
               if (pow(X[0]-v[0]*t,2) + pow(X[1]-v[1]*t,2) + pow(X[2]-v[2]*t,2) < 0.25)
               {
                  return 1.;
               }
               else{
                  return 0.;
               }
            }
         }

      }
      case 6: // max(0.5 - abs(x), 0.) wave for Burgers' eq
      {
         switch (dim)
         {
            case 1:
            {
               return max(0.1, 0.5 - abs(X[0]));
               break;
            }
            case 2:
            {
               // return max(0.5 - sqrt(pow(X[0],2) + pow(X[1],2)), 0.);
               if (x[0] < 1./2. - 3.*t/5.) {
                  if (x[1] > 1/2 + 3*t/20) {
                     return -0.2;
                  } else {
                     return 0.5;
                  }

               } else if (x[0] < 1./2. - t/4.) {
                  if (x[1] > -8*x[0]/7 + 15/14 - 15*t/28) {
                     return -1;
                  } else {
                     return 0.5;
                  }

               } else if (x[0] < 1./2. + t/2.) {
                  if (x[1] > x[0]/6 + 5/12 - 5*t/24) {
                     return -1.;
                  } else {
                     return 0.5;
                  }

               } else if (x[0] < 1./2. + 4.*t/5.) {
                  if (x[1] > x[0] - (5/(18*t)) * pow((x[0] + t - 1/2),2) ) {
                     return -1;
                  } else {
                     return (2 * x[0] - 1) / (2 * t);
                  }

               } else {
                  assert(x[0] >= 1./2. + 4.*t/5.);
                  if (x[1] > 1./2. - t/10) {
                     return -1;
                  } else {
                     return 0.8;
                  }
               }

               break;
            }
         }
      }
      case 7:
      {
         switch (dim) 
         {
            case 1:
            {
               return (.5 + sin(x[0]));
            }
         }
      }
   }
   return 100.;
}

/******************************************************************************
 * inflow_function()
 * Purpose:
 *    Currently unused.
 * ***************************************************************************/
double inflow_function(const Vector &x)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4: 
      case 5: return 0.0;
   }
   return 0.0;
}

/******************************************************************************
 * Testing functions
 * The below functions have no functionality for the method itself, but have
 * provided useful in development.
 * ***************************************************************************/

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
   cout << "Num of face neighbor dofs: " << pfes.num_face_nbr_dofs << endl;
}

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
      y -= MergedRowSums;
      cout << "Should be 0s: \n"; y.Print();
   }

   // This will be the way we should implement the mulitplication. By using the hpm.
   y = 0.0;
   K_hpm.Mult(U, y);
   if (!Mpi::Root()) { cout << endl; y.Print(); }
}

SparseMatrix build_sparse_from_dense(DenseMatrix & dm)
{
   double a;
   const int m = dm.Height(), n = dm.Width();

   Array<int> I, J;
   Array<double> data;
   int counter = 0;

   for (int i = 0; i < m; i++)
   {
      I.Append(counter);

      for (int j = 0; j < n; j++)
      {
         a = dm(i,j);
         if (a != 0)
         {
            data.Append(a);
            J.Append(j);
            // Increment counter for I array
            counter++;
         }
      }
   }

   SparseMatrix spm(I, J, data, m, n);
   // spm->PrintInfo(cout);
   // assert(false);
   return spm;
}
