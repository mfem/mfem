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
   double dt0 = 0.02;
   double omega = 0.2;
   double Tfinal = 1.0;
   double u0 = 0.2;
   double target = 0.7;
   double eps = 1e-7;

   OptionsParser args(argc, argv);
   args.AddOption(&s,      "-s",   "--checkpoints", "Checkpoint budget s (real checkpoints).");
   args.AddOption(&alpha,  "-a",   "--alpha",       "Logistic growth alpha.");
   args.AddOption(&dt0,    "-dt0", "--dt0",         "Base dt for dt(i)=dt0*(1+0.5*sin(omega*i)).");
   args.AddOption(&omega,  "-om",  "--omega",       "Omega in dt(i)=dt0*(1+0.5*sin(omega*i)).");
   args.AddOption(&Tfinal, "-T",   "--tfinal",      "Terminate when accumulated time reaches Tfinal.");
   args.AddOption(&u0,     "-u0",  "--u0",          "Initial scalar state u0.");
   args.AddOption(&target, "-ut",  "--target",      "Target value in J=0.5*(u_m-target)^2.");
   args.AddOption(&eps,    "-eps", "--fd-eps",      "Finite-difference epsilon.");
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

   // Checkpoint manager:
   //   State   = double
   //   Snapshot= double
   mfem::DynamicCheckpointing<double> ckpt(s);

   auto make_snapshot = [](const double &u) -> double { return u; };
   auto restore_snapshot = [](const double &snap, double &out) { out = snap; };

   auto primal_step = [&](double &u, Step i)
   {
      const double dt = dt_func(i);
      u = u + dt * alpha * u * (1.0 - u);
   };

   auto adjoint_step = [&](double &lambda, const double &u_i, Step i)
   {
      const double dt = dt_func(i);
      const double dF_du = 1.0 + dt * alpha * (1.0 - 2.0 * u_i);
      lambda = dF_du * lambda;
   };

   // ---------------- Forward sweep (unknown m) ----------------
   double u = u0;
   double t_phys = 0.0;

   Step i = 0;
   while (t_phys < Tfinal)
   {
      ckpt.ForwardStep(i, u, primal_step, make_snapshot);
      t_phys += dt_func(i);
      ++i;
   }

   const Step m = i;
   const double u_m = u;
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
   double u_work = 0.0;            // scratch primal state u_i

   for (Step j = m - 1; j >= 0; --j)
   {
      ckpt.BackwardStep(j, lambda, u_work,
                        primal_step, adjoint_step,
                        make_snapshot, restore_snapshot);
      if (j == 0) { break; } // avoid signed underflow
   }

   const double dJ_du0_adjoint = lambda;

   // ---------------- Finite-difference gradient check ----------------
   auto forward_only_J = [&](double u_init)
   {
      double uu = u_init;
      double tt = 0.0;
      Step k = 0;
      while (tt < Tfinal)
      {
         primal_step(uu, k);
         tt += dt_func(k);
         ++k;
      }
      const double r = (uu - target);
      return 0.5 * r * r;
   };

   const double Jp = forward_only_J(u0 + eps);
   const double Jm = forward_only_J(u0 - eps);
   const double dJ_du0_fd = (Jp - Jm) / (2.0 * eps);

   const double abs_err = std::abs(dJ_du0_adjoint - dJ_du0_fd);
   const double rel_err = abs_err / (std::abs(dJ_du0_fd) + 1e-30);

   mfem::out << "[Scalar] Gradient check (dJ/du0):\n";
   mfem::out << "  adjoint = " << dJ_du0_adjoint << "\n";
   mfem::out << "  FD      = " << dJ_du0_fd << "\n";
   mfem::out << "  abs err = " << abs_err << "\n";
   mfem::out << "  rel err = " << rel_err << "\n\n";

   return 0;
}

