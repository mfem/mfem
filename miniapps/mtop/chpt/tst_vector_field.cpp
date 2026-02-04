#include "mfem.hpp"

#include <cmath>
#include <iomanip>

using namespace mfem;

// Vector logistic ODE (componentwise):
//   du/dt = alpha*u*(1-u)   (applied to each component)
// Explicit Euler:
//   u_{i+1}[j] = u_i[j] + dt_i*alpha*u_i[j]*(1-u_i[j])
// Objective:
//   J = 0.5*||u_m - u_target||^2
// Terminal adjoint:
//   lambda_m = u_m - u_target
// Adjoint step (componentwise):
//   lambda_i[j] = (1 + dt_i*alpha*(1 - 2*u_i[j])) * lambda_{i+1}[j]

int main(int argc, char *argv[])
{
   int s = 3;
   int n = 16;
   double alpha = 2.0;
   double dt0 = 0.02;
   double omega = 0.2;
   double Tfinal = 1.0;
   double target_val = 0.7;
   double eps = 1e-7;

   OptionsParser args(argc, argv);
   args.AddOption(&s,         "-s",   "--checkpoints", "Checkpoint budget s (real checkpoints).");
   args.AddOption(&n,         "-n",   "--size",        "Vector dimension n.");
   args.AddOption(&alpha,     "-a",   "--alpha",       "Logistic growth alpha.");
   args.AddOption(&dt0,       "-dt0", "--dt0",         "Base dt for dt(i)=dt0*(1+0.5*sin(omega*i)).");
   args.AddOption(&omega,     "-om",  "--omega",       "Omega in dt(i)=dt0*(1+0.5*sin(omega*i)).");
   args.AddOption(&Tfinal,    "-T",   "--tfinal",      "Terminate when accumulated time reaches Tfinal.");
   args.AddOption(&target_val,"-tv",  "--target",      "Target value for each component.");
   args.AddOption(&eps,       "-eps", "--fd-eps",      "Finite-difference epsilon (directional).");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   MFEM_VERIFY(s > 0, "Need s > 0.");
   MFEM_VERIFY(n > 0, "Need n > 0.");
   MFEM_VERIFY(dt0 > 0.0, "Need dt0 > 0.");
   MFEM_VERIFY(Tfinal > 0.0, "Need Tfinal > 0.");

   using Step = mfem::DynamicCheckpointing<mfem::Vector>::Step;

   auto dt_func = [&](Step i)
   {
      const double dt = dt0 * (1.0 + 0.5 * std::sin(omega * double(i)));
      MFEM_VERIFY(dt > 0.0, "dt_func produced non-positive dt.");
      return dt;
   };

   // Checkpoint manager:
   //   State   = mfem::Vector
   //   Snapshot= mfem::Vector
   mfem::DynamicCheckpointing<mfem::Vector> ckpt(s);

   auto make_snapshot = [](const mfem::Vector &u) -> mfem::Vector { return u; };
   auto restore_snapshot = [](const mfem::Vector &snap, mfem::Vector &out) { out = snap; };

   auto primal_step = [&](mfem::Vector &u, Step i)
   {
      const double dt = dt_func(i);
      for (int j = 0; j < u.Size(); ++j)
      {
         const double uj = u[j];
         u[j] = uj + dt * alpha * uj * (1.0 - uj);
      }
   };

   auto adjoint_step = [&](mfem::Vector &lambda, const mfem::Vector &u_i, Step i)
   {
      const double dt = dt_func(i);
      MFEM_ASSERT(lambda.Size() == u_i.Size(), "lambda and u_i size mismatch");
      for (int j = 0; j < lambda.Size(); ++j)
      {
         const double dF_du = 1.0 + dt * alpha * (1.0 - 2.0 * u_i[j]);
         lambda[j] *= dF_du;
      }
   };

   // Initial condition and target
   mfem::Vector u0(n), u_target(n);
   for (int j = 0; j < n; ++j)
   {
      u0[j] = 0.2 + 0.05 * std::cos(0.7 * (j + 1));
   }
   u_target = target_val;

   // ---------------- Forward sweep (unknown m) ----------------
   mfem::Vector u = u0;
   double t_phys = 0.0;

   Step i = 0;
   while (t_phys < Tfinal)
   {
      ckpt.ForwardStep(i, u, primal_step, make_snapshot);
      t_phys += dt_func(i);
      ++i;
   }

   const Step m = i;
   const mfem::Vector u_m = u;

   mfem::Vector diff(u_m);
   diff -= u_target;
   const double J = 0.5 * mfem::InnerProduct(diff, diff);

   mfem::out << std::setprecision(15);
   mfem::out << "\n[Vector] Forward finished:\n";
   mfem::out << "  m (steps) = " << m << "\n";
   mfem::out << "  t_phys    = " << t_phys << "\n";
   mfem::out << "  J         = " << J << "\n";
   mfem::out << "  ||u_m||   = " << u_m.Norml2() << "\n";
   mfem::out << "  ||u_m-ut||= " << diff.Norml2() << "\n\n";

   mfem::out << "[Vector] Checkpoint set after forward sweep (step, level, stored):\n";
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
   mfem::Vector lambda = diff; // terminal = dJ/du_m
   mfem::Vector u_work(n);     // scratch primal u_i

   for (Step j = m - 1; j >= 0; --j)
   {
      ckpt.BackwardStep(j, lambda, u_work,
                        primal_step, adjoint_step,
                        make_snapshot, restore_snapshot);
      if (j == 0) { break; }
   }
   const mfem::Vector &grad_u0 = lambda;

   // ---------------- Directional FD check ----------------
   mfem::Vector v(n);
   for (int j = 0; j < n; ++j) { v[j] = std::sin(0.3 * (j + 1)) + 0.1; }

   auto forward_only_J = [&](const mfem::Vector &u_init)
   {
      mfem::Vector uu = u_init;
      double tt = 0.0;
      Step k = 0;
      while (tt < Tfinal)
      {
         primal_step(uu, k);
         tt += dt_func(k);
         ++k;
      }
      mfem::Vector dd(uu);
      dd -= u_target;
      return 0.5 * mfem::InnerProduct(dd, dd);
   };

   mfem::Vector u_plus(u0), u_minus(u0);
   u_plus.Add(eps, v);
   u_minus.Add(-eps, v);

   const double Jp = forward_only_J(u_plus);
   const double Jm = forward_only_J(u_minus);
   const double dJ_dir_fd = (Jp - Jm) / (2.0 * eps);

   const double dJ_dir_adj = mfem::InnerProduct(grad_u0, v);

   const double abs_err = std::abs(dJ_dir_adj - dJ_dir_fd);
   const double rel_err = abs_err / (std::abs(dJ_dir_fd) + 1e-30);

   mfem::out << "[Vector] Directional derivative check:\n";
   mfem::out << "  v·grad adjoint = " << dJ_dir_adj << "\n";
   mfem::out << "  FD directional = " << dJ_dir_fd << "\n";
   mfem::out << "  abs err        = " << abs_err << "\n";
   mfem::out << "  rel err        = " << rel_err << "\n\n";

   return 0;
}

