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

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

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
   OperatorHandle M, K;
   ParBilinearForm *D_form;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;

   const SparseMatrix &K_mat;
   mutable SparseMatrix *dij_matrix;
   HypreParMatrix *D;

   Array<int> K_smap;
   mutable Vector z;
   Vector *M_lumped;

public:
   FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, const Vector &b_);

   /** FE_Evolution::build_dij_matrix
   Builds dij_matrix used in the low order approximation, which is based on
   Guermond/Popov 2016.
   */
   void build_dij_matrix(const Vector &U,
                         const VectorFunctionCoefficient &velocity) ;
   // Essentially a Mult function for the dij_matrix operator, however this
   // has been replaced by a ParallelAssemble function as this function had issues.
   void ApplyDijMatrix(ParGridFunction &u, Vector &du) const;
   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   virtual ~FE_Evolution();
};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ser_ref_levels = 3;
   int par_ref_levels = 0;
   int order = 1;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 2;
   double t_final = 10.0;
   double dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool paraview = false;
   bool adios2 = false;
   bool binary = false;
   int vis_steps = 2;
   int precision = 8;
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
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
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

   // 7. Define the parallel H1 finite element space on the
   //    parallel refined mesh of the given polynomial order.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_BigInt global_vSize = fes->GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   // Array<int> ess_tdof_list;
   // if (pmesh->bdr_attributes.Size())
   // {
   //    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   //    ess_bdr = 1;
   //    fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   // }

   // 8. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the H1 discretization.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

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

   // m->AddDomainIntegrator(new MassIntegrator);
   m->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   constexpr double alpha = -1.0;
   k->AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));

   ParLinearForm *b = new ParLinearForm(fes);
   b->AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, alpha));

   int skip_zeros = 0;
   m->Assemble();
   k->Assemble(skip_zeros);
   b->Assemble();
   m->Finalize();
   k->Finalize(skip_zeros);

   HypreParMatrix *M = m->ParallelAssemble();
   HypreParMatrix *K = k->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();

   // 9. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   ParGridFunction *u = new ParGridFunction(fes);
   u->ProjectCoefficient(u0);
   HypreParVector *U = u->GetTrueDofs();

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "ex9-mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex9-init." << setfill('0') << setw(6) << myid;
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
   FE_Evolution adv(*m, *k, *B);

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

   // 12. Save the final solution in parallel. This output can be viewed later
   //     using GLVis: "glvis -np <np> -m ex9-mesh -g ex9-final".
   {
      *u = *U;
      ostringstream sol_name;
      sol_name << "ex9p-continuous-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   // 13. Free the used memory.
   delete U;
   delete u;
   delete k;
   delete m;
   delete b;
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
FE_Evolution::FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_,
                           const Vector &b_)
   : TimeDependentOperator(M_.Height()),
     b(b_),
     pfes(*(M_.ParFESpace())),
     M_solver(pfes.GetComm()),
     z(M_.Height()),
     dij_matrix(&K_.SpMat()),
     D_form(&K_),
     K_mat(K_.SpMat()),
     K_smap()
{
   M_lumped = new Vector;
   M_.SpMat().GetDiag(*M_lumped);
   if (M_.GetAssemblyLevel()==AssemblyLevel::LEGACY)
   {
      M.Reset(M_.ParallelAssemble(), true);
      K.Reset(K_.ParallelAssemble(), true);
   }
   else
   {
      M.Reset(&M_, false);
      K.Reset(&K_, false);
   }

   M_solver.SetOperator(*M);

   Array<int> ess_tdof_list;
   if (M_.GetAssemblyLevel()==AssemblyLevel::LEGACY)
   {
      HypreParMatrix &M_mat = *M.As<HypreParMatrix>();
      HypreParMatrix &K_mat = *K.As<HypreParMatrix>();
      HypreSmoother *hypre_prec = new HypreSmoother(M_mat, HypreSmoother::Jacobi);
      M_prec = hypre_prec;
   }
   else
   {
      M_prec = new OperatorJacobiSmoother(M_, ess_tdof_list);
   }

   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);

   // Assuming K is finalized. (From extrapolator.)
   // K_smap is used later to symmetrically form dij_matrix
   const int *I = K_mat.GetI(), *J = K_mat.GetJ(), n = K_mat.Size();
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
   K->Mult(x, z);
}

void FE_Evolution::build_dij_matrix(const Vector &U,
                                    const VectorFunctionCoefficient &velocity)
{
   const int *I = K_mat.HostReadI(), *J = K_mat.HostReadJ(), n = K_mat.Size();

   const double *K_data = K_mat.HostReadData();

   double *D_data = dij_matrix->HostReadWriteData();
   dij_matrix->HostReadWriteI(); dij_matrix->HostReadWriteJ();

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
      (*dij_matrix)(i,i) = -rowsum;
   }
   D = D_form->ParallelAssemble(dij_matrix);
   /* Check that columns sum to 0. */
   // Vector test_v(U.Size());
   // test_v = 1.;
   // cout << "Pre Result: " << test_v[2] << endl;
   // D.MultTranspose(test_v, test_v);
   // cout << "Post Result: " << test_v[2] << endl;

}

/* Deprecated now that ParallelAssemble() is implemented. */
// void FE_Evolution::ApplyDijMatrix(ParGridFunction &u, Vector &du) const
// {
//    const int s = u.Size();
//    const int *I = D.HostReadI(), *J = D.HostReadJ();
//    const double *D_data = D.HostReadData();
//
//    u.ExchangeFaceNbrData();
//    const Vector &u_np = u.FaceNbrData();
//
//    for (int i = 0; i < s; i++)
//    {
//       du(i) = 0.0;
//       for (int k = I[i]; k < I[i + 1]; k++)
//       {
//          int j = J[k];
//          double u_j  = (j < s) ? u(j) : u_np[j - s];
//          double d_ij = D_data[k];
//          du(i) += d_ij * u_j;
//       }
//    }
// }

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = Ml^{-1} ((D + K) x)

   /* Deprecated now that ParallelAssemble() is implemented. */
   // ParGridFunction u_gf(&pfes);
   // u_gf = x;
   // ApplyDijMatrix(u_gf, z);

   D->Mult(x, z);
   // z += b; // Neumann BCs

   Vector rhs(y.Size());
   K->Mult(x, rhs);
   z+= rhs;

   // M_solver.Mult(z, y);
   const int s = y.Size();
   for (int i = 0; i < s; i++)
   {
      y(i) = z(i) / (*M_lumped)(i);
   }
}

FE_Evolution::~FE_Evolution()
{
   delete M_prec;
   delete D;
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
