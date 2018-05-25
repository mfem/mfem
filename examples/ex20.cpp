//                       MFEM Example 18 - Version
//
// Compile with: make ex18
//
// Sample runs:  ex18
//
// Description: This example demonstrates the use of the variable
//              order, symplectic ODE integration algorithm.
//              Symplectic integration algorithms are designed to
//              conserve energy when integrating, in time, systems of
//              ODEs which are derived from Hamiltonain systems.
//
//              Hamiltonian systems define the energy of a system as a
//              function of time (t), a set of generalized coordinates
//              (q), and their corresponding generalized momenta (p).
//                 H(q,p,t) = T(p) + V(q,t)
//              Hamilton's equations then specify how q and p evolve
//              in time:
//                 dq/dt =  dH/dp
//                 dp/dt = -dH/dq
//              To use the symplectic integration classes we need to
//              define an mfem::Operator P which evaluates the action
//              of dH/dp, and an mfem::TimeDependentOperator F which
//              computes -dH/dq.
//
//              This example offers five simple 1D Hamiltonians:
//                 0) Simple Harmonic Oscillator (mass on a spring)
//                    H = ( p^2 / m + q^2 / k ) / 2
//                 1) Pendulum
//                    H = ( p^2 / m - k ( 1 - cos(q) ) ) / 2
//                 2) Gaussian Potential Well
//                    H = ( p^2 / m ) / 2 - k exp(-q^2 / 2)
//                 3) Quartic Potential
//                    H = ( p^2 / m + k ( 1 + q^2 ) q^2 ) / 2
//                 4) Negative Quartic Potential
//                    H = ( p^2 / m + k ( 1 - q^2 /8 ) q^2 ) / 2
//              In all cases these Hamiltonians are shifted by constant
//              values so that the energy will remain positive.
//
//              We then use GLVis to visualize the results in a
//              non-standard way by defining the axes to be q, p, and
//              t rather than x, y, and z.  In this space we build a
//              ribbon-like mesh with nodes at (0,0,t) and (q,p,t).
//              Finally we plot the energy as a function of time as a
//              scalar field on this ribbon-like mesh.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static int prob_ = 0;
static double m_ = 1.0;
static double k_ = 1.0;

double hamiltonian(double q, double p, double t);

class GradT : public Operator
{
public:
   GradT() : Operator(1) {}

   void Mult(const Vector &x, Vector &y) const { y.Set(1.0/m_, x); }

private:
};

class NegGradV : public TimeDependentOperator
{
public:
   NegGradV() : TimeDependentOperator(1) {}

   void Mult(const Vector &x, Vector &y) const;

private:
};

int main(int argc, char *argv[])
{
   // Parse command-line options.
   int order  = 1;
   int nsteps = 100;
   double dt  = 0.1;
   bool visualization = true;
   bool gnuplot = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Time integration order.");
   args.AddOption(&prob_, "-p", "--problem-type",
                  "Problem Type: \n"
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
                  "Spring Constant.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&gnuplot, "-gp", "--gnuplot", "-no-gp",
                  "--no-gnuplot",
                  "Enable or disable GnuPlot visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   SIAVSolver siaSolver(order);

   GradT    P;
   NegGradV F;

   siaSolver.Init(P,F);

   double t = 0.0;

   Vector q(1), p(1);
   q(0) = 0.0;
   p(0) = 1.0;

   ofstream ofs;
   if ( gnuplot )
   {
      ofs.open("ex18.dat");
      ofs << t << "\t" << q(0) << "\t" << p(0) << endl;
   }

   Vector e(nsteps+1);

   int nverts = (visualization)?2*(nsteps+1):0;
   int nelems = (visualization)?nsteps:0;
   Mesh mesh(2, nverts, nelems, 0, 3);

   int    v[4];
   Vector x0(3); x0 = 0.0; //x0(0) = M_PI;
   Vector x1(3); x1 = 0.0;

   double e_mean = 0.0;

   for (int i=0; i<nsteps; i++)
   {
      if ( i == 0 )
      {
         e[0] = hamiltonian(q(0),p(0),t);
         e_mean += e[0];

         if ( visualization )
         {
            x1[0] = q(0);
            x1[1] = p(0);
            x1[2] = 0.0;
            mesh.AddVertex(x0);
            mesh.AddVertex(x1);
         }
      }

      siaSolver.Step(q,p,t,dt);

      e[i+1] = hamiltonian(q(0),p(0),t);
      e_mean += e[i+1];

      if ( gnuplot )
      {
         ofs << t << "\t" << q(0) << "\t" << p(0) << "\t" << e[i+1] << endl;
      }

      if ( visualization )
      {
         x0[2] = t;
         x1[0] = q(0);
         x1[1] = p(0);
         x1[2] = t;
         mesh.AddVertex(x0);
         mesh.AddVertex(x1);
         v[0] = 2*i;
         v[1] = 2*(i+1);
         v[2] = 2*(i+1)+1;
         v[3] = 2*i+1;
         mesh.AddQuad(v);
      }
   }

   e_mean /= (nsteps + 1);
   double e_var = 0.0;
   for (int i=0; i<=nsteps; i++)
   {
      e_var += pow(e[i] - e_mean, 2);
   }
   e_var /= (nsteps + 1);
   double e_sd = sqrt(e_var);
   cout << endl << "Mean and standard deviation of the energy" << endl;
   cout << e_mean << "\t" << e_sd << endl;

   if ( gnuplot )
   {
      ofs.close();

      ofs.open("gnuplot_ex18.inp");
      ofs << "plot 'ex18.dat' using 1:2 w l t 'q', "
          << "'ex18.dat' using 1:3 w l t 'p', "
          << "'ex18.dat' using 1:4 w l t 'H'" << endl;
      ofs.close();
   }

   if ( visualization )
   {
      H1_FECollection fec(order = 1, 2);
      FiniteElementSpace fespace(&mesh, &fec);
      GridFunction energy(&fespace);
      energy = 0.0;
      for (int i=0; i<=nsteps; i++)
      {
         energy[2*i+0] = e[i];
         energy[2*i+1] = e[i];
      }

      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sock(vishost, visport);
      sock.precision(8);
      sock << "solution\n" << mesh << energy
           << "window_title 'Energy in Phase Space'\n"
           << "keys\n maac\n" << "axis_labels 'q' 'p' 't'\n"<< flush;
   }
}

double hamiltonian(double q, double p, double t)
{
   double h = 1.0 - 0.5 / m_ + 0.5 * p * p / m_;
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

void
NegGradV::Mult(const Vector &x, Vector &y) const
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
