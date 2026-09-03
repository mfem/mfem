//                       MFEM Example 20 - Parallel Version
//
// Compile with: make ex20p
//
// Sample runs:  mpirun -np 4 ex20p
//               mpirun -np 4 ex20p -p 1 -o 1 -n 120 -dt 0.1
//               mpirun -np 4 ex20p -p 1 -o 2 -n 60 -dt 0.2
//               mpirun -np 4 ex20p -p 1 -o 3 -n 40 -dt 0.3
//               mpirun -np 4 ex20p -p 1 -o 4 -n 30 -dt 0.4
//
// Description: This example demonstrates the use of the variable order,
//              symplectic ODE integration algorithm.  Symplectic integration
//              algorithms are designed to conserve energy when integrating, in
//              time, systems of ODEs which are derived from Hamiltonian
//              systems.
//
//              Hamiltonian systems define the energy of a system as a function
//              of time (t), a set of generalized coordinates (q), and their
//              corresponding generalized momenta (p).
//
//                 H(q,p,t) = T(p) + V(q,t)
//
//              Hamilton's equations then specify how q and p evolve in time:
//
//                 dq/dt =  dH/dp
//                 dp/dt = -dH/dq
//
//              To use the symplectic integration classes we need to define an
//              mfem::Operator P which evaluates the action of dH/dp, and an
//              mfem::TimeDependentOperator F which computes -dH/dq.
//
//              This example offers five simple 1D Hamiltonians:
//              0) Simple Harmonic Oscillator (mass on a spring)
//                 H = ( p^2 / m + q^2 / k ) / 2
//              1) Pendulum
//                 H = ( p^2 / m - k ( 1 - cos(q) ) ) / 2
//              2) Gaussian Potential Well
//                 H = ( p^2 / m ) / 2 - k exp(-q^2 / 2)
//              3) Quartic Potential
//                 H = ( p^2 / m + k ( 1 + q^2 ) q^2 ) / 2
//              4) Negative Quartic Potential
//                 H = ( p^2 / m + k ( 1 - q^2 /8 ) q^2 ) / 2
//
//              In all cases these Hamiltonians are shifted by constant values
//              so that the energy will remain positive. The mean and standard
//              deviation of the computed energies at each time step are
//              displayed upon completion. When run in parallel the same
//              Hamiltonian system is evolved on each processor but starting
//              from different initial conditions.
//
//              We then use GLVis to visualize the results in a non-standard way
//              by defining the axes to be q, p, and t rather than x, y, and z.
//              In this space we build a ribbon-like mesh on each processor with
//              nodes at (0,0,t) and (q,p,t).  When these ribbons are bonded
//              together on the t-axis they resemble a Rotini pasta.  Finally we
//              plot the energy as a function of time as a scalar field on this
//              Rotini-like mesh.
//
//              For a more traditional plot of the results, including q, p, and
//              H from each processor, can be obtained by selecting the "-gp"
//              option. This creates a collection of data files and an input
//              deck for the GnuPlot application (not included with MFEM). To
//              visualize these results on most linux systems type the command
//              "gnuplot gnuplot_ex20p.inp". The data files, named
//              "ex20p_?????.dat", should be simple enough to display with other
//              plotting programs as well.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Constants used in the Hamiltonian
static int prob_ = 0;
static real_t m_ = 1.0;
static real_t k_ = 1.0;

// Hamiltonian functional, see below for implementation
real_t hamiltonian(real_t q, real_t p, real_t t);

class GradT : public Operator
{
public:
   GradT() : Operator(1) {}
   void Mult(const Vector &x, Vector &y) const override { y.Set(1.0/m_, x); }
};

class NegGradV : public TimeDependentOperator
{
public:
   NegGradV() : TimeDependentOperator(1) {}
   void Mult(const Vector &x, Vector &y) const override;
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   MPI_Comm comm = MPI_COMM_WORLD;

   // 2. Parse command-line options.
   int order  = 1;
   int nsteps = 100;
   real_t dt  = 0.1;
   bool visualization = true;
   bool gnuplot = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Time integration order.");
   args.AddOption(&prob_, "-p", "--problem-type",
                  "Problem Type:\n"
                  "\t  0 - Simple Harmonic Oscillator\n"
                  "\t  1 - Pendulum\n"
                  "\t  2 - Gaussian Potential Well\n"
                  "\t  3 - Quartic Potential\n"
                  "\t  4 - Negative Quartic Potential");
   args.AddOption(&nsteps, "-n", "--number-of-steps",
                  "Number of time steps.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step size.");
   args.AddOption(&m_, "-m", "--mass",
                  "Mass.");
   args.AddOption(&k_, "-k", "--spring-const",
                  "Spring constant.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&gnuplot, "-gp", "--gnuplot", "-no-gp", "--no-gnuplot",
                  "Enable or disable GnuPlot visualization.");
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

   // 3. Create and Initialize the Symplectic Integration Solver
   SIAVSolver siaSolver(order);
   GradT    P;
   NegGradV F;
   siaSolver.Init(P,F);

   // 4. Set the initial conditions
   real_t t = 0.0;
   Vector q(1), p(1);
   Vector e(nsteps+1);
   q(0) = sin(2.0*M_PI*(real_t)myid/num_procs);
   p(0) = cos(2.0*M_PI*(real_t)myid/num_procs);

   // 5. Prepare GnuPlot output file if needed
   ostringstream oss;
   ofstream ofs;
   if (gnuplot)
   {
      oss << "ex20p_" << setfill('0') << setw(5) << myid << ".dat";
      ofs.open(oss.str().c_str());
      ofs << t << "\t" << q(0) << "\t" << p(0) << endl;
   }

   // 6. Create a Mesh for visualization in phase space
   int nverts = (visualization) ? 2*num_procs*(nsteps+1) : 0;
   int nelems = (visualization) ? (nsteps * num_procs) : 0;
   Mesh mesh(2, nverts, nelems, 0, 3);

   int   *part = (visualization) ? (new int[nelems]) : NULL;
   int    v[4];
   Vector x0(3); x0 = 0.0;
   Vector x1(3); x1 = 0.0;

   // 7. Perform time-stepping
   real_t e_mean = 0.0;

   for (int i = 0; i < nsteps; i++)
   {
      // 7a. Record initial state
      if (i == 0)
      {
         e[0] = hamiltonian(q(0),p(0),t);
         e_mean += e[0];

         if (visualization)
         {
            for (int j = 0; j < num_procs; j++)
            {
               mesh.AddVertex(x0);
               x1[0] = q(0);
               x1[1] = p(0);
               x1[2] = 0.0;
               mesh.AddVertex(x1);
            }
         }
      }

      // 7b. Advance the state of the system
      siaSolver.Step(q,p,t,dt);
      e[i+1] = hamiltonian(q(0),p(0),t);
      e_mean += e[i+1];

      // 7c. Record the state of the system
      if (gnuplot)
      {
         ofs << t << "\t" << q(0) << "\t" << p(0) << "\t" << e[i+1] << endl;
      }

      // 7d. Add results to GLVis visualization
      if (visualization)
      {
         x0[2] = t;
         for (int j = 0; j < num_procs; j++)
         {
            mesh.AddVertex(x0);
            x1[0] = q(0);
            x1[1] = p(0);
            x1[2] = t;
            mesh.AddVertex(x1);
            v[0] = 2 * num_procs * i + 2 * j;
            v[1] = 2 * num_procs * (i + 1) + 2 * j;
            v[2] = 2 * num_procs * (i + 1) + 2 * j + 1;
            v[3] = 2 * num_procs * i + 2 * j + 1;
            mesh.AddQuad(v);
            part[num_procs * i + j] = j;
         }
      }
   }

   // 8. Compute and display mean and standard deviation of the energy
   e_mean /= (nsteps + 1);
   real_t e_var = 0.0;
   for (int i = 0; i <= nsteps; i++)
   {
      e_var += pow(e[i] - e_mean, 2);
   }
   e_var /= (nsteps + 1);
   real_t e_sd = sqrt(e_var);

   real_t e_loc_stats[2];
   real_t *e_stats = (myid == 0) ? new real_t[2 * num_procs] : (real_t*)NULL;

   e_loc_stats[0] = e_mean;
   e_loc_stats[1] = e_sd;
   MPI_Gather(e_loc_stats, 2, MPITypeMap<real_t>::mpi_type, e_stats, 2,
              MPITypeMap<real_t>::mpi_type, 0, comm);

   if (myid == 0)
   {
      cout << endl << "Mean and standard deviation of the energy "
           << "for different initial conditions" << endl;
      for (int i = 0; i < num_procs; i++)
      {
         cout << i << ": " << e_stats[2 * i + 0]
              << "\t" << e_stats[2 * i + 1] << endl;
      }
      delete [] e_stats;
   }

   // 9. Finalize the GnuPlot output
   if (gnuplot)
   {
      ofs.close();
      if (myid == 0)
      {
         ofs.open("gnuplot_ex20p.inp");
         for (int i = 0; i < num_procs; i++)
         {
            ostringstream ossi;
            ossi << "ex20p_" << setfill('0') << setw(5) << i << ".dat";
            if (i == 0)
            {
               ofs << "plot";
            }
            ofs << " '" << ossi.str() << "' using 1:2 w l t 'q" << i << "',"
                << " '" << ossi.str() << "' using 1:3 w l t 'p" << i << "',"
                << " '" << ossi.str() << "' using 1:4 w l t 'H" << i << "'";
            if (i < num_procs-1)
            {
               ofs << ",";
            }
            else
            {
               ofs << ";" << endl;
            }
         }
         ofs.close();
      }
   }

   // 10. Finalize the GLVis output
   if (visualization)
   {
      mesh.FinalizeQuadMesh(1);
      ParMesh pmesh(comm, mesh, part);
      delete [] part;

      H1_FECollection fec(order = 1, 2);
      ParFiniteElementSpace fespace(&pmesh, &fec);
      ParGridFunction energy(&fespace);
      energy = 0.0;
      for (int i = 0; i <= nsteps; i++)
      {
         energy[2*i+0] = e[i];
         energy[2*i+1] = e[i];
      }

      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sock(vishost, visport);
      sock.precision(8);
      sock << "parallel " << num_procs << " " << myid << "\n"
           << "solution\n" << pmesh << energy
           << "window_title 'Energy in Phase Space'\n"
           << "keys\n maac\n" << "axis_labels 'q' 'p' 't'\n"<< flush;
   }
}

real_t hamiltonian(real_t q, real_t p, real_t t)
{
   real_t h = 1.0 - 0.5 / m_ + 0.5 * p * p / m_;
   switch (prob_)
   {
      case 1:
         h += k_ * (1.0 - cos(q));
         break;
      case 2:
         h += k_ * (1.0 - exp(-0.5 * q * q));
         break;
      case 3:
         h += 0.5 * k_ * (1.0 + q * q) * q * q;
         break;
      case 4:
         h += 0.5 * k_ * (1.0 - 0.125 * q * q) * q * q;
         break;
      default:
         h += 0.5 * k_ * q * q;
         break;
   }
   return h;
}

void NegGradV::Mult(const Vector &x, Vector &y) const
{
   switch (prob_)
   {
      case 1:
         y(0) = - k_* sin(x(0));
         break;
      case 2:
         y(0) = - k_ * x(0) * exp(-0.5 * x(0) * x(0));
         break;
      case 3:
         y(0) = - k_ * (1.0 + 2.0 * x(0) * x(0)) * x(0);
         break;
      case 4:
         y(0) = - k_ * (1.0 - 0.25 * x(0) * x(0)) * x(0);
         break;
      default:
         y(0) = - k_ * x(0);
         break;
   };
}
