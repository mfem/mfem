//               MFEM Maxwell Mini App
//               Simple Full-Wave Electromagnetic Simulation Code
//
// Compile with: make maxwell
//
// Sample runs:
//
//   By default the sources and fields are all zero
//     mpirun -np 4 maxwell
//     mpirun -np 4 maxwell -m ../../data/ball-nurbs.mesh -rs 2
//                          -abcs '-1'
//                          -dp '-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5'
//                          -cs '0.0 0.0 -0.5 .2 10'
//                          -ds '0.0 0.0 0.5 .2 10'
//
//     mpirun -np 4 maxwell -m ../../data/fichera.mesh -rs 3
//                          -ts 0.25 -tf 10 -dbcs '-1'
//                          -dp '-0.5 -0.5 0.0 -0.5 -0.5 1.0 0.1 1 .5 1'
//
//     mpirun -np 4 maxwell -m ../../data/fichera.mesh -rs 3
//                          -ts 0.25 -tf 10
//                           -dp '-0.5 -0.5 0.0 -0.5 -0.5 1.0 0.1 1 .5 1'
//                          -dbcs '4 8 19 21' -abcs '5 18'
//
// Description:
//               This mini app solves a simple 3D full-wave
//               electromagnetic problem.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
//#include "pfem_extras.hpp"
//#include "volta_solver.hpp"
#include "maxwell_solver.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;
using namespace mfem::electromagnetics;

// Permittivity Function
static Vector ds_params_(0);  // Center, Radius, and Permittivity
//                               of dielectric sphere
double dielectric_sphere(const Vector &);
double epsilon(const Vector &x) { return dielectric_sphere(x); }

// Permeability Function
static Vector ms_params_(0);  // Center, Inner and Outer Radii, and
//                               Permeability of magnetic shell
double magnetic_shell(const Vector &);
double muInv(const Vector & x) { return 1.0/magnetic_shell(x); }

// Conductivity Function
static Vector cs_params_(0);  // Center, Radius, and Conductivity
//                               of conductive sphere
double conductive_sphere(const Vector &);
double sigma(const Vector &x) { return conductive_sphere(x); }

// Polarization
//static Vector vp_params_(0);  // Axis Start, Axis End, Cylinder Radius,
//                               Polarization Magnitude, and Frequency
//void voltaic_pile(const Vector &, double t, Vector &);

// Current Density Function
//static Vector cr_params_(0);  // Axis Start, Axis End, Inner Ring Radius,
//                               Outer Ring Radius, Total Current of
//                               current ring (annulus), and Frequency
//void current_ring(const Vector &, double t, Vector &);

// Current Density Function
static Vector dp_params_(0);  // Axis Start, Axis End, Rod Radius,
//                               Total Current of Rod, and Frequency
void dipole_pulse(const Vector &x, double t, Vector &j);
/*
void current_src(const Vector &x, double t, Vector &j)
{
   if ( vp_params_.Size()*cr_params_.Size() == 0 )
   {
      if ( vp_params_.Size() > 0 )
      {
         voltaic_pile(x, t, j);
      }
      if ( cr_params_.Size() > 0 )
      {
         current_ring(x, t, j);
      }
   }
   else
   {
      Vector j_cr(x.Size());
      voltaic_pile(x, t, j);
      current_ring(x, t, j_cr);
      j += j_cr;
   }
}
*/
/*
// E Boundary Condition
static Vector e_uniform_(0);
double phi_bc_uniform(const Vector &);
*/
void dEdtBCFunc(const Vector &x, double t, Vector &E);

static double tScale_ = 1e-9;
static double freq_ = 750.0e6;
static int prob_ = -1;
/*
double permittivity_func(const Vector &);
double permeability_func(const Vector &);
double inv_permeability_func(const Vector &x)
{ return 1.0/permeability_func(x); }
*/
void EFieldFunc(const Vector &, Vector&);
void BFieldFunc(const Vector &, Vector&);

int SnapTimeStep(double tmax, double dtmax, double & dt);

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   if ( mpi.Root() ) { display_banner(cout); }

   // Parse command-line options.
   const char *mesh_file = "./butterfly_3d.mesh";
   int sOrder = 1;
   int tOrder = 1;
   int sr = 0, pr = 0;
   int max_its = 100;
   bool visualization = true;
   bool visit = true;
   double dt = 1.0e-12;
   double t0 = 0.0;
   double ts = 1.0;
   double tf = 40.0;

   Array<int> abcs;
   Array<int> dbcs;
   /*
   Array<int> nbcs;

   Vector dbcv;
   Vector nbcv;

   bool dbcg = false;
   */
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&sOrder, "-so", "--spatial-order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&tOrder, "-to", "--temporal-order",
                  "Time integration order.");
   args.AddOption(&prob_, "-p", "--problem",
                  "Problem Setup.");
   args.AddOption(&sr, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&pr, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&max_its, "-n", "--number-of-steps",
                  "Number of time steps.");
   /*
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step size.");
   */
   args.AddOption(&t0, "-t0", "--initial-time",
                  "Beginning of time interval to simulate (ns).");
   args.AddOption(&tf, "-tf", "--final-time",
                  "End of time interval to simulate (ns).");
   args.AddOption(&ts, "-ts", "--snapshot-time",
                  "Time between snapshots (ns).");
   args.AddOption(&freq_, "-f", "--frequency",
                  "Frequency.");

   /*
   args.AddOption(&e_uniform_, "-uebc", "--uniform-e-bc",
                  "Specify if the three components of the constant electric field");
   */
   args.AddOption(&ds_params_, "-ds", "--dielectric-sphere-params",
                  "Center, Radius, and Permittivity of Dielectric Sphere");
   args.AddOption(&ms_params_, "-ms", "--magnetic-shell-params",
                  "Center, Inner Radius, Outer Radius, and Permeability of Magnetic Shell");
   args.AddOption(&cs_params_, "-cs", "--conductive-sphere-params",
                  "Center, Radius, and Conductivity of Conductive Sphere");
   //   args.AddOption(&vp_params_, "-vp", "--voltaic-pile-params",
   //               "Axis End Points, Radius, and Polarization of Cylindrical Voltaic Pile");
   // args.AddOption(&cr_params_, "-cr", "--current-ring-params",
   //               "Axis End Points, Inner Radius, Outer Radius, Total Current of Annulus, and the Frequency of Oscillation");
   args.AddOption(&dp_params_, "-dp", "--dipole-pulse-params",
                  "Axis End Points, Radius, Total Current, and the Frequency of Oscillation");
   /*
   args.AddOption(&cs_params_, "-cs", "--charged-sphere-params",
                  "Center, Radius, and Total Charge of Charged Sphere");
   */
   args.AddOption(&abcs, "-abcs", "--absorbing-bc-surf",
                  "Absorbing Boundary Condition Surfaces");
   args.AddOption(&dbcs, "-dbcs", "--dirichlet-bc-surf",
                  "Dirichlet Boundary Condition Surfaces");
   /*
   args.AddOption(&dbcv, "-dbcv", "--dirichlet-bc-vals",
                  "Dirichlet Boundary Condition Values");
   args.AddOption(&dbcg, "-dbcg", "--dirichlet-bc-gradient",
                  "-no-dbcg", "--no-dirichlet-bc-gradient",
                  "Dirichlet Boundary Condition Gradient (phi = -z)");
   args.AddOption(&nbcs, "-nbcs", "--neumann-bc-surf",
                  "Neumann Boundary Condition Surfaces");
   args.AddOption(&nbcv, "-nbcv", "--neumann-bc-vals",
                  "Neumann Boundary Condition Values");
   */
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visualization",
                  "Enable or disable VisIt visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(cout);
   }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (mpi.Root())
      {
         cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      }
      MPI_Finalize();
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   // int sdim = mesh->SpaceDimension();
   // int dim = mesh->Dimension();
   mesh->SetCurvature(2);

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement.
   {
      int ref_levels = sr;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   // mesh->EnsureNCMesh();

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = pr;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   /*
   // If the gradient bc was selected but the E field was not specified
   // set a default vector value.
   if ( dbcg && e_uniform_.Size() != sdim )
   {
      e_uniform_.SetSize(sdim);
      e_uniform_ = 0.0;
      e_uniform_(sdim-1) = 1.0;
   }

   // If values for Dirichlet BCs were not set assume they are zero
   if (dbcv.Size() < dbcs.Size() && !dbcg )
   {
      dbcv.SetSize(dbcs.Size());
      dbcv = 0.0;
   }

   // If values for Neumann BCs were not set assume they are zero
   if (nbcv.Size() < nbcs.Size() )
   {
      nbcv.SetSize(nbcs.Size());
      nbcv = 0.0;
   }

   // Create the Electrostatic solver
   VoltaSolver Volta(pmesh, order, dbcs, dbcv, nbcs, nbcv,
                     ( ds_params_.Size() > 0 ) ? dielectric_sphere : NULL,
                     ( e_uniform_.Size() > 0 ) ? phi_bc_uniform    : NULL,
                     ( cs_params_.Size() > 0 ) ? conductive_sphere : NULL,
                     ( vp_params_.Size() > 0 ) ? voltaic_pile      : NULL);

   // Initialize GLVis visualization
   if (visualization)
   {
      Volta.InitializeGLVis();
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("Maxwell-AMR-Parallel", &pmesh);

   if ( visit )
   {
      Volta.RegisterVisItFields(visit_dc);
   }

   // The initial AMR loop. In each iteration we solve an electrostatic
   // problem on the current mesh, visualize the solution, estimate the
   // error on all elements, refine the worst elements and update all
   // objects to work with the new mesh.  Once a sufficiently resolved
   // mesh has been produced the time evolution loop will begin.
   const int max_dofs = 100000;
   for (int it = 1; it <= 100; it++)
   {
      // Display the current number of DoFs in each finite element space
      Volta.PrintSizes(it);

      // Solve the system and compute any auxiliary fields
      Volta.Solve();

      // Determine the current size of the linear system
      int prob_size = Volta.GetProblemSize();

      // Write fields to disk for VisIt
      if ( visit )
      {
         Volta.WriteVisItFields(it);
      }

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         Volta.DisplayToGLVis();
      }

      // Check stopping criteria
      if (prob_size > max_dofs)
      {
         break;
      }

      // Estimate element errors using the Zienkiewicz-Zhu error estimator.
      Vector errors(pmesh.GetNE());
      {
         Volta.GetErrorEstimates(errors);
      }
      double local_max_err = errors.Max();
      double global_max_err;
      MPI_Allreduce(&local_max_err, &global_max_err, 1,
                    MPI_DOUBLE, MPI_MAX, pmesh.GetComm());

      // Make a list of elements whose error is larger than a fraction
      // of the maximum element error. These elements will be refined.
      Array<int> ref_list;
      const double frac = 0.7;
      double threshold = frac * global_max_err;
      for (int i = 0; i < errors.Size(); i++)
      {
         if (errors[i] >= threshold) { ref_list.Append(i); }
      }

      // Refine the selected elements. Since we are going to transfer the
      // grid function x from the coarse mesh to the new fine mesh in the
      // next step, we need to request the "two-level state" of the mesh.
      pmesh.GeneralRefinement(ref_list);

      // Update the electrostatic solver to reflect the new state of the mesh.
      Volta.Update();

      // Wait for user input
      char c;
      if (myid == 0)
      {
         cout << "press (q)uit or (c)ontinue --> " << flush;
         cin >> c;
      }
      MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

      if (c != 'c')
      {
         break;
      }
   }
   */
   // lossy = abcs.Size() > 0 || cs_params_.Size() > 0;

   // Create the Electromagnetic solver
   MaxwellSolver Maxwell(pmesh, sOrder,
                         ( ds_params_.Size() > 0 ) ? epsilon     : NULL,
                         ( ms_params_.Size() > 0 ) ? muInv       : NULL,
                         ( cs_params_.Size() > 0 ) ? sigma       : NULL,
                         ( dp_params_.Size() > 0 ) ? dipole_pulse : NULL,
                         abcs, dbcs,
                         (       dbcs.Size() > 0 ) ? dEdtBCFunc  : NULL
                        );

   // Display the current number of DoFs in each finite element space
   Maxwell.PrintSizes();

   VectorFunctionCoefficient EFieldCoef(3,EFieldFunc);
   VectorFunctionCoefficient BFieldCoef(3,BFieldFunc);

   Maxwell.SetInitialEField(EFieldCoef);
   Maxwell.SetInitialBField(BFieldCoef);

   double energy = Maxwell.GetEnergy();
   if ( mpi.Root() )
   {
      cout << "Energy(" << t0 << "ns):  " << energy << "J" << endl;
   }

   double dtmax = Maxwell.GetMaximumTimeStep();

   // Convert times from nanoseconds to seconds
   t0 *= tScale_;
   tf *= tScale_;
   ts *= tScale_;

   if ( mpi.Root() )
   {
      cout << "Maximum Time Step:  " << dtmax << endl;
   }

   int nsteps = SnapTimeStep(tf-t0, dtmax, dt);
   /*
   if ( nsteps > max_its )
   {
      if ( mpi.Root() )
      {
         cout << "Computed number of time steps is too large." << endl;
      }
      nsteps = max_its;
   }
   */
   if ( mpi.Root() )
   {
      cout << "Number of Time Steps:  " << nsteps << endl;
      cout << "Time Step Size (ns):        " << dt/tScale_ << endl;
   }

   SIAVSolver siaSolver(tOrder);
   siaSolver.Init(Maxwell.GetNegCurl(), Maxwell);


   // Initialize GLVis visualization
   if (visualization)
   {
      Maxwell.InitializeGLVis();
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("Maxwell-Parallel", &pmesh);

   double t = t0;
   Maxwell.SetTime(t);

   if ( visit )
   {
      Maxwell.RegisterVisItFields(visit_dc);
   }

   // Write initial fields to disk for VisIt
   if ( visit )
   {
      Maxwell.WriteVisItFields(0);
   }

   // Send the initial condition by socket to a GLVis server.
   if (visualization)
   {
      Maxwell.DisplayToGLVis();
   }

   // The main time evolution loop.
   while (t < tf)
   {
      // Compute the next time step.
      // Maxwell.Solve();
      // cout << "siSolver.Step" << endl;
      siaSolver.Run(Maxwell.GetBField(), Maxwell.GetEField(), t, dt, t + ts);

      // Maxwell.SetTime(t);

      energy = Maxwell.GetEnergy();
      if ( mpi.Root() )
      {
         cout << "Energy(" << t/tScale_ << "ns):  " << energy << "J" << endl;
      }

      // cout << "Maxwell.Sync" << endl;
      Maxwell.SyncGridFuncs();

      // Write fields to disk for VisIt
      if ( visit )
      {
         // cout << "Maxwell.WriteVisIt" << endl;
         Maxwell.WriteVisItFields((int)( (t - t0) / ts ) );
      }

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         // cout << "Maxwell.DisplayGLVis" << endl;
         Maxwell.DisplayToGLVis();
      }
   }

   return 0;
}

// Print the Maxwell ascii logo to the given ostream
void display_banner(ostream & os)
{
   os << "     ___    ____                                      " << endl
      << "    /   |  /   /                           __   __    " << endl
      << "   /    |_/ _ /__  ___  _____  _  __ ____ |  | |  |   " << endl
      << "  /         \\__  \\ \\  \\/  /\\ \\/ \\/ // __ \\|  | |  |   "
      << endl
      << " /   /|_/   // __ \\_>    <  \\     /\\  ___/|  |_|  |__ " << endl
      << "/___/  /_  /(____  /__/\\_ \\  \\/\\_/  \\___  >____/____/ " << endl
      << "         \\/       \\/      \\/             \\/           " << endl
      << flush;
}

// A sphere with constant permittivity.  The sphere has a radius,
// center, and permittivity specified on the command line and stored
// in ds_params_.
double dielectric_sphere(const Vector &x)
{
   double r2 = 0.0;

   for (int i=0; i<x.Size(); i++)
   {
      r2 += (x(i)-ds_params_(i))*(x(i)-ds_params_(i));
   }

   if ( sqrt(r2) <= ds_params_(x.Size()) )
   {
      return ds_params_(x.Size()+1) * epsilon0_;
   }
   return epsilon0_;
}

// A spherical shell with constant permeability.  The sphere has inner
// and outer radii, center, and relative permeability specified on the
// command line and stored in ms_params_.
double magnetic_shell(const Vector &x)
{
   double r2 = 0.0;

   for (int i=0; i<x.Size(); i++)
   {
      r2 += (x(i)-ms_params_(i))*(x(i)-ms_params_(i));
   }

   if ( sqrt(r2) >= ms_params_(x.Size()) &&
        sqrt(r2) <= ms_params_(x.Size()+1) )
   {
      return mu0_*ms_params_(x.Size()+2);
   }
   return mu0_;
}

// A sphere with constant conductivity.  The sphere has a radius,
// center, and conductivity specified on the command line and stored
// in ls_params_.
double conductive_sphere(const Vector &x)
{
   double r2 = 0.0;

   for (int i=0; i<x.Size(); i++)
   {
      r2 += (x(i)-cs_params_(i))*(x(i)-cs_params_(i));
   }

   if ( sqrt(r2) <= cs_params_(x.Size()) )
   {
      return cs_params_(x.Size()+1);
   }
   return 0.0;
}

// A sphere with constant charge density.  The sphere has a radius,
// center, and total charge specified on the command line and stored
// in cs_params_.
double charged_sphere(const Vector &x)
{
   double r2 = 0.0;
   double rho = 0.0;

   if ( cs_params_(x.Size()) > 0.0 )
   {
      switch ( x.Size() )
      {
         case 2:
            rho = cs_params_(x.Size()+1)/(M_PI*pow(cs_params_(x.Size()),2));
            break;
         case 3:
            rho = 0.75*cs_params_(x.Size()+1)/(M_PI*pow(cs_params_(x.Size()),3));
            break;
         default:
            rho = 0.0;
      }
   }

   for (int i=0; i<x.Size(); i++)
   {
      r2 += (x(i)-cs_params_(i))*(x(i)-cs_params_(i));
   }

   if ( sqrt(r2) <= cs_params_(x.Size()) )
   {
      return rho;
   }
   return 0.0;
}

// A cylindrical rod of current density.  The rod has two axis end
// points, a radus, a current amplitude in Amperes, and a frequency.
void dipole_pulse(const Vector &x, double t, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   Vector  v(x.Size());  // Normalized Axis vector
   Vector xu(x.Size());  // x vector relative to the axis end-point

   xu = x;

   for (int i=0; i<x.Size(); i++)
   {
      xu[i] -= dp_params_[i];
      v[i]   = dp_params_[x.Size()+i] - dp_params_[i];
   }

   double h = v.Norml2();

   if ( h == 0.0 )
   {
      return;
   }
   v /= h;

   double r = dp_params_[2*x.Size()+0];
   double a = dp_params_[2*x.Size()+1] * tScale_;
   double b = dp_params_[2*x.Size()+2] * tScale_;
   double c = dp_params_[2*x.Size()+3] * tScale_;

   double xv = xu * v;

   // Compute perpendicular vector from axis to x
   xu.Add(-xv, v);

   double xp = xu.Norml2();

   if ( xv >= 0.0 && xv <= h && xp <= r )
   {
      j = v;
   }

   j *= a * (t - b) * exp(-0.5 * pow((t-b)/c, 2)) / (c * c);
}
/*
// To produce a uniform electric field the potential can be set
// to (- Ex x - Ey y - Ez z).
double phi_bc_uniform(const Vector &x)
{
   double phi = 0.0;

   for (int i=0; i<x.Size(); i++)
   {
      phi -= x(i) * e_uniform_(i);
   }

   return phi;
}
*/
/*
double
permittivity_func(const Vector &x)
{
   double eps = epsilon0_;

   switch (prob_)
   {
      case 0:
         eps = epsilon0_;
         break;
      default:
         eps = epsilon0_;
   }
   return eps;
}

double
permeability_func(const Vector &x)
{
   double mu = mu0_;

   switch (prob_)
   {
      case 0:
         mu = mu0_;
         break;
      default:
         mu = mu0_;
   }
   return mu;
}
*/
void
EFieldFunc(const Vector &x, Vector &E)
{
   E.SetSize(3);
   E = 0.0;
   // E(2) = cos(M_PI*x(0));
   /*
   switch (prob_)
   {
      case 0:
         E = 0.0;
         E(2) = cos(M_PI*x(0));
         break;
      default:
         E = 0.0;
   }
   */
}

void
BFieldFunc(const Vector &x, Vector &B)
{
   B.SetSize(3);
   B = 0.0;
   // B(1) = -sqrt(epsilon0_*mu0_)*cos(M_PI*x(0));
   /*
   switch (prob_)
   {
      case 0:
         B = 0.0;
         B(1) = sqrt(epsilon0_*mu0_)*cos(M_PI*x(0));
         break;
      default:
         B = 0.0;
   }
   */
}

void
dEdtBCFunc(const Vector &x, double t, Vector &dE)
{
   dE.SetSize(3);
   dE = 0.0;
   switch (prob_)
   {
      case 0:
         dE = 0.0;
         dE(2) = 2.0 * M_PI * freq_ *
                 cos(2.0 * M_PI * freq_ * (t - x(0) * sqrt(epsilon0_ * mu0_)));
         break;
      case 1:
      {
         double arg = 2.0 * M_PI * freq_ * (t - x(0) * sqrt(epsilon0_ * mu0_));
         dE = 0.0;
         dE(2) = 2.0 * M_PI * freq_ * exp(-0.25 * pow(arg,2)) *
                 (cos(arg) + 0.25 * arg * sin(arg) );
      }
      break;
      default:
         dE = 0.0;
   }
}

int
SnapTimeStep(double tmax, double dtmax, double & dt)
{
   double dsteps = tmax/dtmax;

   int nsteps = pow(10,(int)ceil(log10(dsteps)));

   for (int i=1; i<=5; i++)
   {
      int a = (int)ceil(log10(dsteps/pow(5.0,i)));
      int nstepsi = (int)pow(5,i)*max(1,(int)pow(10,a));

      nsteps = min(nsteps,nstepsi);
   }

   dt = tmax / nsteps;

   return nsteps;
}
