#include "mfem.hpp"

#include <cmath>
#include <iomanip>

using namespace mfem;

// Scalar logistic ODE:
//   du/dt = alpha*u*(1-u)
// Explicit Euler with variable dt(i):
//   u_{i+1} = u_i + dt_i*alpha*u_i*(1-u_i)
// Objective:
//   J = 0.5*(u_m - target)^2
// Discrete adjoint (scalar):
//   lambda_m = (u_m - target)
//   lambda_i = (dF/du at u_i)^T * lambda_{i+1}
//   dF/du = 1 + dt_i*alpha*(1 - 2*u_i)

int main(int argc, char *argv[])
{
   int s = 3;
   double alpha = 2.0;
   double dt0 = 0.001;
   double omega = 0.2;
   double Tfinal = 1.0;
   double u0 = 0.2;
   double target = 0.7;
   double eps = 1e-7;
   double err = 1e-4;

   OptionsParser args(argc, argv);
   args.AddOption(&s,      "-s",   "--checkpoints", "Checkpoint budget s (real checkpoints).");
   args.AddOption(&alpha,  "-a",   "--alpha",       "Logistic growth alpha.");
   args.AddOption(&dt0,    "-dt0", "--dt0",         "Base for the time step dt.");
   args.AddOption(&Tfinal, "-T",   "--tfinal",      "Terminate when accumulated time reaches Tfinal.");
   args.AddOption(&u0,     "-u0",  "--u0",          "Initial scalar state u0.");
   args.AddOption(&target, "-ut",  "--target",      "Target value in J=0.5*(u_m-target)^2.");
   args.AddOption(&eps,    "-eps", "--fd-eps",      "Finite-difference epsilon.");
   args.AddOption(&err,    "-err", "--time_err",     "Allowed time integration error per time step.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   MFEM_VERIFY(s > 0, "Need s > 0.");
   MFEM_VERIFY(dt0 > 0.0, "Need dt0 > 0.");
   MFEM_VERIFY(Tfinal > 0.0, "Need Tfinal > 0.");

   using Step = mfem::DynamicCheckpointing<double>::Step;

   auto dt_func = [&](Step i)
   {
      const double dt = dt0 * (1.0 + 0.5 * std::sin(omega * double(i)));
      MFEM_VERIFY(dt > 0.0, "dt_func produced non-positive dt.");
      return dt;
   };

   struct my_state
   {
      my_state(double a, double b, double c, double d=0.0, double tp_=0.0){
         t=a; dt=b; u=c; up=d; tp=tp_;
      }

      //t,dt,u
      double t;
      double dt;
      double u;
      double tp;
      double up;
   }; 

   // Checkpoint manager:
   //   State   = double
   //   Snapshot= double
   mfem::DynamicCheckpointing<my_state> ckpt(s);

   auto make_snapshot = [](const my_state &u) -> my_state { return u; };
   auto restore_snapshot = [](const my_state &snap, my_state &out) { out = snap; };

   double tmax=Tfinal;

   auto primal_step = [&](my_state &su, Step i)
   {
      su.up=su.u;
      su.tp=su.t;

      double t=su.t;
      double dt = su.dt;
      bool flag=true;

      if((tmax-t)<su.dt)
      {
         dt=tmax-t;
         flag=false;
      }

      double u=su.u;

      double s0=alpha * u * (1.0 - u);
      double ue=u+dt*s0;
      double s1=alpha * ue * (1.0 - ue);
      double uh=u + 0.5 * dt * (s0 + s1);
      double ee = std::abs(uh-ue);

      if((ee < 0.5*err) && (flag))
      {
         dt=1.25*dt;
         ue=u+dt*s0;
         s1=alpha * ue * (1.0 - ue);
         uh=u + 0.5 * dt * (s0 + s1);
         ee = std::abs(uh-ue);
      } 

      while(ee > err)
      {
         dt=0.5*dt;
         ue=u+dt*s0;
         s1=alpha * ue * (1.0 - ue);
         uh=u + 0.5 * dt * (s0 + s1);
         ee = std::abs(uh-ue);
      } 

      //std::cout<<" t="<<t+dt<<" dt="<<dt<<" err="<<ee<<std::endl;

      su.t = t+dt;
      su.dt = dt;
      su.u = uh;
   };

   auto adjoint_step = [&](double &lambda, const my_state &u_i, Step i)
   {
      const double dt = u_i.dt;
      // const double dF_du = 1.0 + dt * alpha * (1.0 - 2.0 * u_i.u);
      // lambda = dF_du * lambda;

      const double s0=alpha * (1.0 -2.0 * u_i.u) *lambda;
      const double le= lambda+dt*s0;
      const double s1= alpha * (1.0 -2.0 * u_i.up) * le;
      lambda = lambda + 0.5 *dt *(s0+s1);
   };

   // ---------------- Forward sweep (unknown m) ----------------
   my_state u(0.0, dt0, u0);
   double t_phys = 0.0;

   Step i = 0;
   while (t_phys < Tfinal)
   {
      ckpt.ForwardStep(i, u, primal_step, make_snapshot);
      t_phys = u.t;
      ++i;
   }

   const Step m = i;
   const double u_m = u.u;
   const double J = 0.5 * (u_m - target) * (u_m - target);

   mfem::out << std::setprecision(15);
   mfem::out << "\n[Scalar] Forward finished:\n";
   mfem::out << "  m (steps) = " << m << "\n";
   mfem::out << "  t_phys    = " << t_phys << "\n";
   mfem::out << "  u_m       = " << u_m << "\n";
   mfem::out << "  J         = " << J << "\n\n";

   mfem::out << "[Scalar] Checkpoint set after forward sweep (step, level, stored):\n";
   for (const auto &cp : ckpt.GetCheckpointInfo())
   {
      mfem::out << "  step=" << cp.step
                << ", level=" << cp.level
                << ", stored=" << (cp.stored ? "yes" : "no")
                << (cp.stored ? "" : " (placeholder)")
                << "\n";
   }
   mfem::out << "\n";

   MFEM_VERIFY(m > 0, "Forward produced m=0 steps; nothing to do.");

   // ---------------- Backward sweep (adjoint) ----------------
   double lambda = (u_m - target); // terminal adjoint = dJ/du_m
   my_state u_work (0.0,0.0,0.0);            // scratch primal state u_i

   for (Step j = m - 1; j >= 0; --j)
   {
      ckpt.BackwardStep(j, lambda, u_work,
                        primal_step, adjoint_step,
                        make_snapshot, restore_snapshot);
      if (j == 0) { break; } // avoid signed underflow
   }

   const double dJ_du0_adjoint = lambda;

   // analytic solution
   const double sol=u0*exp(alpha*Tfinal)/(1.0-u0+u0*exp(alpha*Tfinal));
   const double grd=exp(alpha*Tfinal)/std::pow(1.0-u0+u0*exp(alpha*Tfinal),2.0);

   mfem::out << "true sol = "<<sol<<" \n";


   const double dJ_du0_fd = (sol - target) * grd;

   const double abs_err = std::abs(dJ_du0_adjoint - dJ_du0_fd);
   const double rel_err = abs_err / (std::abs(dJ_du0_fd) + 1e-30);

   mfem::out << "[Scalar] Gradient check (dJ/du0):\n";
   mfem::out << "  adjoint = " << dJ_du0_adjoint << "\n";
   mfem::out << "  true grad = " << dJ_du0_fd << "\n";
   mfem::out << "  abs err = " << abs_err << "\n";
   mfem::out << "  rel err = " << rel_err << "\n\n";

   return 0;
}

