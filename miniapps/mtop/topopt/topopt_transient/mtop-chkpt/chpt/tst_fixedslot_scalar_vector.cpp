#include "mfem.hpp"

#include "dynamic_checkpointing.hpp"
#include "fixed_slot_checkpoint_storage.hpp"

#include <cmath>
#include <iomanip>
#include <string>

using namespace mfem;

// Logistic ODE (scalar or componentwise):
//   du/dt = alpha*u*(1-u)
// Explicit Euler:
//   u_{i+1} = u_i + dt_i * alpha*u_i*(1-u_i)
//
// Objective:
//   Scalar: J = 0.5*(u_m - target)^2
//   Vector: J = 0.5*||u_m - u_target||^2
//
// Discrete adjoint:
//   dF/du = 1 + dt_i*alpha*(1 - 2*u_i)
//   lambda_i = (dF/du at u_i) * lambda_{i+1}

static inline double LogisticStep(const double u, const double alpha, const double dt)
{
   return u + dt * alpha * u * (1.0 - u);
}

static inline double LogisticJac(const double u, const double alpha, const double dt)
{
   return 1.0 + dt * alpha * (1.0 - 2.0*u);
}

// --------------------------
// Scalar run (double)
// --------------------------
template <typename Storage>
static void RunScalarFixedSlot(int s,
                               Storage &storage,
                               double alpha,
                               double dt0,
                               double omega,
                               double Tfinal,
                               double u0,
                               double target,
                               double eps)
{
   using CKPT = mfem::DynamicCheckpointing<double, Storage>;
   using Step = typename CKPT::Step;

   CKPT ckpt(s, storage);

   auto dt_func = [&](Step i)
   {
      const double dt = dt0 * (1.0 + 0.5 * std::sin(omega * double(i)));
      MFEM_VERIFY(dt > 0.0, "dt_func produced non-positive dt.");
      return dt;
   };

   auto make_snapshot = [](const double &u) { return u; };
   auto restore_snapshot = [](const double &snap, double &out) { out = snap; };

   auto primal_step = [&](double &u, Step i)
   {
      const double dt = dt_func(i);
      u = LogisticStep(u, alpha, dt);
   };

   auto adjoint_step = [&](double &lambda, const double &u_i, Step i)
   {
      const double dt = dt_func(i);
      lambda *= LogisticJac(u_i, alpha, dt);
   };

   // Forward sweep (unknown m; stop on accumulated physical time)
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

   // Backward sweep
   MFEM_VERIFY(m > 0, "Scalar run produced m=0 steps.");

   double lambda = (u_m - target); // terminal condition dJ/du_m
   double u_work = 0.0;

   for (Step j = m - 1; j >= 0; --j)
   {
      ckpt.BackwardStep(j, lambda, u_work,
                        primal_step, adjoint_step,
                        make_snapshot, restore_snapshot);
      if (j == 0) { break; }
   }
   const double dJ_du0_adj = lambda;

   // FD check
   auto forward_only_J = [&](double u_init)
   {
      double uu = u_init;
      double tt = 0.0;
      Step k = 0;
      while (tt < Tfinal)
      {
         uu = LogisticStep(uu, alpha, dt_func(k));
         tt += dt_func(k);
         ++k;
      }
      const double r = uu - target;
      return 0.5 * r * r;
   };

   const double Jp = forward_only_J(u0 + eps);
   const double Jm = forward_only_J(u0 - eps);
   const double dJ_du0_fd = (Jp - Jm) / (2.0 * eps);

   mfem::out << "\n[FixedSlot][Scalar]\n";
   mfem::out << "  m steps        = " << m << "\n";
   mfem::out << "  t_phys         = " << t_phys << "\n";
   mfem::out << "  u_m            = " << u_m << "\n";
   mfem::out << "  J              = " << J << "\n";
   mfem::out << "  dJ/du0 adjoint  = " << dJ_du0_adj << "\n";
   mfem::out << "  dJ/du0 FD       = " << dJ_du0_fd << "\n";
   mfem::out << "  abs err         = " << std::abs(dJ_du0_adj - dJ_du0_fd) << "\n";
}

// --------------------------
// Vector run (mfem::Vector)
// --------------------------
template <typename Storage>
static void RunVectorFixedSlot(int s,
                               Storage &storage,
                               int n,
                               double alpha,
                               double dt0,
                               double omega,
                               double Tfinal,
                               double target_val,
                               double eps)
{
   using CKPT = mfem::DynamicCheckpointing<mfem::Vector, Storage>;
   using Step = typename CKPT::Step;

   CKPT ckpt(s, storage);

   auto dt_func = [&](Step i)
   {
      const double dt = dt0 * (1.0 + 0.5 * std::sin(omega * double(i)));
      MFEM_VERIFY(dt > 0.0, "dt_func produced non-positive dt.");
      return dt;
   };

   auto make_snapshot = [](const mfem::Vector &u) { return u; };
   auto restore_snapshot = [](const mfem::Vector &snap, mfem::Vector &out) { out = snap; };

   auto primal_step = [&](mfem::Vector &u, Step i)
   {
      const double dt = dt_func(i);
      for (int j = 0; j < u.Size(); ++j)
      {
         u[j] = LogisticStep(u[j], alpha, dt);
      }
   };

   auto adjoint_step = [&](mfem::Vector &lambda, const mfem::Vector &u_i, Step i)
   {

      const double dt = dt_func(i);
      MFEM_ASSERT(lambda.Size() == u_i.Size(), "lambda and u_i size mismatch.");
      for (int j = 0; j < lambda.Size(); ++j)
      {
         lambda[j] *= LogisticJac(u_i[j], alpha, dt);
      }
   };

   // Initial condition and target
   mfem::Vector u0(n), u_target(n);
   for (int j = 0; j < n; ++j)
   {
      u0[j] = 0.2 + 0.05 * std::cos(0.7 * (j + 1));
   }
   u_target = target_val;

   // Forward sweep
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

   // Backward sweep
   MFEM_VERIFY(m > 0, "Vector run produced m=0 steps.");

   mfem::Vector lambda = diff; // terminal adjoint = dJ/du_m
   mfem::Vector u_work(n);

   for (Step j = m - 1; j >= 0; --j)
   {
      ckpt.BackwardStep(j, lambda, u_work,
                        primal_step, adjoint_step,
                        make_snapshot, restore_snapshot);
      if (j == 0) { break; }
   }
   const mfem::Vector &grad_u0 = lambda;

   // Directional FD check
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

   mfem::out << "\n[FixedSlot][Vector]\n";
   mfem::out << "  n              = " << n << "\n";
   mfem::out << "  m steps         = " << m << "\n";
   mfem::out << "  t_phys          = " << t_phys << "\n";
   mfem::out << "  J               = " << J << "\n";
   mfem::out << "  ||u_m||_2        = " << u_m.Norml2() << "\n";
   mfem::out << "  ||u_m-u_target|| = " << diff.Norml2() << "\n";
   mfem::out << "  v·grad adjoint   = " << dJ_dir_adj << "\n";
   mfem::out << "  FD directional   = " << dJ_dir_fd << "\n";
   mfem::out << "  abs err          = " << std::abs(dJ_dir_adj - dJ_dir_fd) << "\n";
}

int main(int argc, char *argv[])
{
   // Backend selection:
   //   0 = fixed-slot memory (single RAM block)
   //   1 = fixed-slot file   (single file with fixed offsets)
   int backend = 0;

   // Common parameters
   int s = 8;                 // number of REAL stored checkpoints
   double alpha = 2.0;
   double dt0 = 0.02;
   double omega = 0.2;
   double Tfinal = 1.0;
   double eps = 1e-7;

   // Scalar parameters
   double u0 = 0.2;
   double target_s = 0.7;

   // Vector parameters
   int n = 64;                // must stay fixed for fixed-slot vector packing
   double target_v = 0.7;

   // File backend parameters
   std::string scalar_file = "scalar_fixedslots.bin";
   std::string vector_file = "vector_fixedslots.bin";
   bool truncate_files = true;
   bool flush_on_store = true; // safer for demo correctness with iostreams

   OptionsParser args(argc, argv);
   args.AddOption(&backend, "-b", "--backend",
                  "Backend: 0=memory fixed-slots, 1=file fixed-slots (single file).");
   args.AddOption(&s, "-s", "--checkpoints", "Checkpoint budget s (real checkpoints).");

   args.AddOption(&alpha, "-a", "--alpha", "Logistic growth alpha.");
   args.AddOption(&dt0, "-dt0", "--dt0", "Base dt for dt(i)=dt0*(1+0.5*sin(omega*i)).");
   args.AddOption(&omega, "-om", "--omega", "Omega for dt(i)=dt0*(1+0.5*sin(omega*i)).");
   args.AddOption(&Tfinal, "-T", "--tfinal", "Stop when accumulated time reaches Tfinal.");
   args.AddOption(&eps, "-eps", "--fd-eps", "FD epsilon.");

   args.AddOption(&u0, "-u0", "--u0", "Scalar initial condition u0.");
   args.AddOption(&target_s, "-ts", "--target-scalar", "Scalar target.");

   args.AddOption(&n, "-n", "--size", "Vector dimension (fixed).");
   args.AddOption(&target_v, "-tv", "--target-vector", "Vector target value per component.");

   args.AddOption(&scalar_file, "-sf", "--scalar-file", "File for scalar fixed-slot storage.");
   args.AddOption(&vector_file, "-vf", "--vector-file", "File for vector fixed-slot storage.");
   args.AddOption(&truncate_files, "-tr", "--truncate", "-ntr", "--no-truncate",
                  "Truncate checkpoint files on startup.");
   args.AddOption(&flush_on_store, "-fl", "--flush", "-nfl", "--no-flush",
                  "Flush on each Store() (demo-safety; slower).");

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

   mfem::out << std::setprecision(15);

   if (backend == 0)
   {
      mfem::out << "\nUsing fixed-slot MEMORY backend (single RAM block)\n";

      // Scalar: fixed-size POD => trivial packer
      mfem::FixedSlotMemoryCheckpointStorage<double> stor_s(s);
      RunScalarFixedSlot(s, stor_s, alpha, dt0, omega, Tfinal, u0, target_s, eps);

      // Vector: fixed-size packing (n must remain constant)
      mfem::FixedVectorPacker packer(n);
      mfem::FixedSlotMemoryCheckpointStorage<mfem::Vector, mfem::FixedVectorPacker> stor_v(s, packer);
      RunVectorFixedSlot(s, stor_v, n, alpha, dt0, omega, Tfinal, target_v, eps);
   }
   else if (backend == 1)
   {
      mfem::out << "\nUsing fixed-slot FILE backend (single file with fixed offsets)\n";

      // Scalar file
      mfem::FixedSlotFileCheckpointStorage<double> stor_s(scalar_file, s,
                                                         mfem::TrivialFixedPacker<double>(),
                                                         truncate_files,
                                                         flush_on_store);
      RunScalarFixedSlot(s, stor_s, alpha, dt0, omega, Tfinal, u0, target_s, eps);

      // Vector file (fixed-size packing with n)
      mfem::FixedVectorPacker packer(n);
      mfem::FixedSlotFileCheckpointStorage<mfem::Vector, mfem::FixedVectorPacker>
         stor_v(vector_file, s, packer, truncate_files, flush_on_store);
      RunVectorFixedSlot(s, stor_v, n, alpha, dt0, omega, Tfinal, target_v, eps);
   }
   else
   {
      MFEM_ABORT("Unknown backend. Use -b 0 (memory) or -b 1 (file).");
   }

   mfem::out << "\nDone.\n";
   return 0;
}

