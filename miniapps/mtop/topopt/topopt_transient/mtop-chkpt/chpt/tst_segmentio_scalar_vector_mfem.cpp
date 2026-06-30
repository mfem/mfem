#include "mfem.hpp"
#include "dynamic_checkpointing.hpp"
#include "segment_checkpoint_storage.hpp"

#include <cmath>
#include <iomanip>

using namespace mfem;

static double LogisticStep(double u, double alpha, double dt)
{
   return u + dt * alpha * u * (1.0 - u);
}

static double LogisticJac(double u, double alpha, double dt)
{
   return 1.0 + dt * alpha * (1.0 - 2.0*u);
}

#if MFEM_HAVE_FILESYSTEM
static void PurgeDir(const std::string &dir)
{
   std::error_code ec;
   mfem_fs::remove_all(mfem_fs::path(dir), ec);
   mfem_fs::create_directories(mfem_fs::path(dir), ec);
   MFEM_VERIFY(!ec, "Failed to purge/create directory.");
}
#endif

int main(int argc, char *argv[])
{
   int s = 3;
   double dt0 = 0.02;
   double omega = 0.2;
   double Tfinal = 1.0;

   // scalar params
   double alpha = 2.0;
   double u0 = 0.2;
   double target_s = 0.7;
   double eps = 1e-7;

   // vector params
   int n = 16;
   double target_v = 0.7;

   // segmented storage params
   std::string dir_scalar = "chk_scalar_segments";
   std::string dir_vector = "chk_vector_segments";
   long long records_per_file = 4096;
   bool keep_files = false;
   bool purge_dirs = true;

   OptionsParser args(argc, argv);
   args.AddOption(&s, "-s", "--checkpoints", "Checkpoint budget s (real checkpoints).");
   args.AddOption(&dt0, "-dt0", "--dt0", "Base dt for dt(i)=dt0*(1+0.5*sin(omega*i)).");
   args.AddOption(&omega, "-om", "--omega", "Omega for dt(i)=dt0*(1+0.5*sin(omega*i)).");
   args.AddOption(&Tfinal, "-T", "--tfinal", "Stop when accumulated time reaches Tfinal.");

   args.AddOption(&alpha, "-a", "--alpha", "Logistic growth alpha.");
   args.AddOption(&u0, "-u0", "--u0", "Scalar initial u0.");
   args.AddOption(&target_s, "-ts", "--target-scalar", "Scalar target.");
   args.AddOption(&eps, "-eps", "--fd-eps", "FD epsilon (scalar and vector directional).");

   args.AddOption(&n, "-n", "--size", "Vector dimension.");
   args.AddOption(&target_v, "-tv", "--target-vector", "Vector target value per component.");

   args.AddOption(&dir_scalar, "-ds", "--dir-scalar", "Directory for scalar segment files.");
   args.AddOption(&dir_vector, "-dv", "--dir-vector", "Directory for vector segment files.");
   args.AddOption(&records_per_file, "-rpf", "--records-per-file",
                  "How many handles belong to one segment file (range size).");

   args.AddOption(&keep_files, "-k", "--keep-files", "-nk", "--no-keep-files",
                  "Keep segment files (debug).");
   args.AddOption(&purge_dirs, "-p", "--purge-dirs", "-np", "--no-purge-dirs",
                  "Purge checkpoint directories at start (recommended).");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   MFEM_VERIFY(s > 0, "Need s > 0.");
   MFEM_VERIFY(records_per_file > 0, "Need records_per_file > 0.");

#if MFEM_HAVE_FILESYSTEM
   if (purge_dirs)
   {
      PurgeDir(dir_scalar);
      PurgeDir(dir_vector);
   }
#else
   MFEM_VERIFY(!purge_dirs, "purge_dirs requires <filesystem> support.");
#endif

   using StepS = mfem::DynamicCheckpointing<
      double,
      mfem::SegmentedFileCheckpointStorage<double>>::Step;

   auto dt_func = [&](StepS i)
   {
      const double dt = dt0 * (1.0 + 0.5 * std::sin(omega * double(i)));
      MFEM_VERIFY(dt > 0.0, "dt_func produced non-positive dt.");
      return dt;
   };

   mfem::out << std::setprecision(15);

   // ============================================================
   // A) Scalar with segmented storage
   // ============================================================
   mfem::SegmentedFileCheckpointStorage<double> stor_s(dir_scalar, records_per_file,
                                                       "seg_", ".bin", true, keep_files);
   {
      mfem::DynamicCheckpointing<double, mfem::SegmentedFileCheckpointStorage<double>> ckpt(s, stor_s);

      auto make_snapshot = [](const double &u) { return u; };
      auto restore_snapshot = [](const double &snap, double &out) { out = snap; };

      auto primal_step = [&](double &u, StepS i)
      {
         u = LogisticStep(u, alpha, dt_func(i));
      };

      auto adjoint_step = [&](double &lambda, const double &u_i, StepS i)
      {
         lambda *= LogisticJac(u_i, alpha, dt_func(i));
      };

      double u = u0;
      double t_phys = 0.0;
      StepS i = 0;
      while (t_phys < Tfinal)
      {
         ckpt.ForwardStep(i, u, primal_step, make_snapshot);
         t_phys += dt_func(i);
         ++i;
      }
      const StepS m = i;
      const double u_m = u;
      const double J = 0.5 * (u_m - target_s) * (u_m - target_s);

      double lambda = (u_m - target_s);
      double u_work = 0.0;

      for (StepS j = m - 1; j >= 0; --j)
      {
         ckpt.BackwardStep(j, lambda, u_work,
                           primal_step, adjoint_step,
                           make_snapshot, restore_snapshot);
         if (j == 0) { break; }
      }
      const double dJ_du0_adj = lambda;

      auto forward_only_J = [&](double u_init)
      {
         double uu = u_init;
         double tt = 0.0;
         StepS k = 0;
         while (tt < Tfinal)
         {
            uu = LogisticStep(uu, alpha, dt_func(k));
            tt += dt_func(k);
            ++k;
         }
         const double r = uu - target_s;
         return 0.5 * r * r;
      };

      const double Jp = forward_only_J(u0 + eps);
      const double Jm = forward_only_J(u0 - eps);
      const double dJ_du0_fd = (Jp - Jm) / (2.0 * eps);

      mfem::out << "\n[SegmentIO][Scalar]\n";
      mfem::out << "  m steps        = " << m << "\n";
      mfem::out << "  J              = " << J << "\n";
      mfem::out << "  dJ/du0 adjoint  = " << dJ_du0_adj << "\n";
      mfem::out << "  dJ/du0 FD       = " << dJ_du0_fd << "\n";
      mfem::out << "  abs err         = " << std::abs(dJ_du0_adj - dJ_du0_fd) << "\n";
   }

   // ============================================================
   // B) Vector with segmented storage
   // ============================================================
   mfem::SegmentedFileCheckpointStorage<mfem::Vector> stor_v(dir_vector, records_per_file,
                                                             "seg_", ".bin", true, keep_files);
   {
      mfem::DynamicCheckpointing<mfem::Vector, mfem::SegmentedFileCheckpointStorage<mfem::Vector>>
         ckpt(s, stor_v);

      using StepV = mfem::DynamicCheckpointing<
         mfem::Vector,
         mfem::SegmentedFileCheckpointStorage<mfem::Vector>>::Step;

      auto dt_func_v = [&](StepV i)
      {
         const double dt = dt0 * (1.0 + 0.5 * std::sin(omega * double(i)));
         MFEM_VERIFY(dt > 0.0, "dt_func produced non-positive dt.");
         return dt;
      };

      auto make_snapshot = [](const mfem::Vector &u) { return u; };
      auto restore_snapshot = [](const mfem::Vector &snap, mfem::Vector &out) { out = snap; };

      auto primal_step = [&](mfem::Vector &u, StepV i)
      {
         const double dt = dt_func_v(i);
         for (int j = 0; j < u.Size(); ++j)
         {
            u[j] = LogisticStep(u[j], alpha, dt);
         }
      };

      auto adjoint_step = [&](mfem::Vector &lambda, const mfem::Vector &u_i, StepV i)
      {
         const double dt = dt_func_v(i);
         MFEM_ASSERT(lambda.Size() == u_i.Size(), "Size mismatch.");
         for (int j = 0; j < lambda.Size(); ++j)
         {
            lambda[j] *= LogisticJac(u_i[j], alpha, dt);
         }
      };

      mfem::Vector u0v(n), ut(n);
      for (int j = 0; j < n; ++j) { u0v[j] = 0.2 + 0.05 * std::cos(0.7*(j+1)); }
      ut = target_v;

      mfem::Vector u = u0v;
      double t_phys = 0.0;
      StepV i = 0;

      while (t_phys < Tfinal)
      {
         ckpt.ForwardStep(i, u, primal_step, make_snapshot);
         t_phys += dt_func_v(i);
         ++i;
      }
      const StepV m = i;

      mfem::Vector diff(u);
      diff -= ut;
      const double J = 0.5 * mfem::InnerProduct(diff, diff);

      mfem::Vector lambda = diff;
      mfem::Vector u_work(n);

      for (StepV j = m - 1; j >= 0; --j)
      {
         ckpt.BackwardStep(j, lambda, u_work,
                           primal_step, adjoint_step,
                           make_snapshot, restore_snapshot);
         if (j == 0) { break; }
      }
      const mfem::Vector &grad_u0 = lambda;

      mfem::Vector v(n);
      for (int j = 0; j < n; ++j) { v[j] = std::sin(0.3*(j+1)) + 0.1; }

      auto forward_only_J = [&](const mfem::Vector &u_init)
      {
         mfem::Vector uu = u_init;
         double tt = 0.0;
         StepV k = 0;
         while (tt < Tfinal)
         {
            primal_step(uu, k);
            tt += dt_func_v(k);
            ++k;
         }
         mfem::Vector dd(uu);
         dd -= ut;
         return 0.5 * mfem::InnerProduct(dd, dd);
      };

      mfem::Vector u_plus(u0v), u_minus(u0v);
      u_plus.Add(eps, v);
      u_minus.Add(-eps, v);

      const double Jp = forward_only_J(u_plus);
      const double Jm = forward_only_J(u_minus);
      const double dJ_dir_fd = (Jp - Jm) / (2.0 * eps);
      const double dJ_dir_adj = mfem::InnerProduct(grad_u0, v);

      mfem::out << "\n[SegmentIO][Vector]\n";
      mfem::out << "  m steps             = " << m << "\n";
      mfem::out << "  J                   = " << J << "\n";
      mfem::out << "  v·grad adjoint      = " << dJ_dir_adj << "\n";
      mfem::out << "  FD directional      = " << dJ_dir_fd << "\n";
      mfem::out << "  abs err             = " << std::abs(dJ_dir_adj - dJ_dir_fd) << "\n";
      mfem::out << "  ||u - u_target||2   = " << diff.Norml2() << "\n";
   }

   // Optional cleanup: delete segment files this run touched
   if (!keep_files)
   {
      stor_s.PurgeAllFiles();
      stor_v.PurgeAllFiles();
   }

   mfem::out << "\nDone.\n";
   return 0;
}

