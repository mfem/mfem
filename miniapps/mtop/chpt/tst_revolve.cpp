#include "mfem.hpp"


// ============================================================
// Mini example 1: scalar state (double)
// u_{n+1} = factor * u_n
// J = 0.5 (u_N - target)^2  => lambda_N = (u_N - target)
// lambda_n = factor * lambda_{n+1}
// ============================================================
static void RunScalarExample(int Nsteps, int Ncheck, double factor)
{
   out << "\n--- Scalar REVOLVE example ---\n";
   out << "Nsteps=" << Nsteps << ", Ncheckpoints=" << Ncheck << ", factor=" << factor << "\n";

   const size_t snap_bytes = sizeof(double);
   FixedSlotMemoryStorage storage(Ncheck, snap_bytes);
   FixedStepRevolveCheckpointing<FixedSlotMemoryStorage> ckpt(Nsteps, Ncheck, snap_bytes, storage);

   auto make_snapshot = [](const double &u, uint8_t *outb, size_t bytes)
   {
      MFEM_VERIFY(bytes == sizeof(double), "scalar snapshot size mismatch");
      std::memcpy(outb, &u, sizeof(double));
   };
   auto restore_snapshot = [](double &u, const uint8_t *inb, size_t bytes)
   {
      MFEM_VERIFY(bytes == sizeof(double), "scalar snapshot size mismatch");
      std::memcpy(&u, inb, sizeof(double));
   };

   auto primal_step = [factor](int /*step*/, double &u)
   {
      u *= factor;
   };
   auto adjoint_step = [factor](int /*step*/, const double & /*u_step*/, double &lambda)
   {
      lambda *= factor;
   };

   const double target = 2.0;

   // Forward
   double u = 1.0;
   for (int i = 0; i < Nsteps; ++i)
   {
      ckpt.ForwardStep(i, u, primal_step, make_snapshot);
   }
   const double uN = u;

   // Init adjoint at final state
   double lambda = (uN - target);

   // Reverse
   double u_work = 0.0; // will be overwritten by restore_snapshot
   for (int i = Nsteps - 1; i >= 0; --i)
   {
      ckpt.BackwardStep(i, lambda, u_work,
                        primal_step, adjoint_step,
                        make_snapshot, restore_snapshot);
   }

   // Analytic lambda_0 = factor^Nsteps * (uN - target)
   double factorN = 1.0;
   for (int k = 0; k < Nsteps; ++k) { factorN *= factor; }
   const double lambda0_exact = factorN * (uN - target);

   out << "uN = " << uN << "\n";
   out << "lambda0 (computed) = " << lambda << "\n";
   out << "lambda0 (exact)    = " << lambda0_exact << "\n";
}

// ============================================================
// Mini example 2: mfem::Vector state
// u_{n+1} = factor * u_n (elementwise scalar multiply)
// J = 0.5 ||u_N - target||^2  => lambda_N = (u_N - target)
// lambda_n = factor * lambda_{n+1}
// ============================================================
static void RunVectorExample(int Nsteps, int Ncheck, int dim, double factor)
{
   out << "\n--- mfem::Vector REVOLVE example ---\n";
   out << "Nsteps=" << Nsteps << ", Ncheckpoints=" << Ncheck
       << ", dim=" << dim << ", factor=" << factor << "\n";

   MFEM_VERIFY(dim > 0, "dim must be > 0");

   const size_t snap_bytes = sizeof(double) * size_t(dim);
   FixedSlotMemoryStorage storage(Ncheck, snap_bytes);
   FixedStepRevolveCheckpointing<FixedSlotMemoryStorage> ckpt(Nsteps, Ncheck, snap_bytes, storage);

   auto make_snapshot = [](const mfem::Vector &u, uint8_t *outb, size_t bytes)
   {
      MFEM_VERIFY(bytes == sizeof(double) * size_t(u.Size()), "Vector snapshot size mismatch");
      std::memcpy(outb, u.GetData(), bytes);
   };
   auto restore_snapshot = [](mfem::Vector &u, const uint8_t *inb, size_t bytes)
   {
      MFEM_VERIFY(bytes == sizeof(double) * size_t(u.Size()), "Vector snapshot size mismatch");
      std::memcpy(u.GetData(), inb, bytes);
   };

   auto primal_step = [factor](int /*step*/, mfem::Vector &u)
   {
      u *= factor;
   };
   auto adjoint_step = [factor](int /*step*/, const mfem::Vector & /*u_step*/, mfem::Vector &lambda)
   {
      lambda *= factor;
   };

   mfem::Vector u(dim);
   for (int k = 0; k < dim; ++k) { u[k] = 1.0 + 0.1*k; }

   mfem::Vector target(dim);
   target = 2.0;

   // Forward
   for (int i = 0; i < Nsteps; ++i)
   {
      ckpt.ForwardStep(i, u, primal_step, make_snapshot);
   }
   mfem::Vector uN(u); // copy for reporting

   // Init adjoint: lambda_N = uN - target
   mfem::Vector lambda(dim);
   lambda = u;
   lambda -= target;

   // Reverse
   mfem::Vector u_work(dim);
   u_work = 0.0;
   for (int i = Nsteps - 1; i >= 0; --i)
   {
      ckpt.BackwardStep(i, lambda, u_work,
                        primal_step, adjoint_step,
                        make_snapshot, restore_snapshot);
   }

   out << "||uN||_2      = " << uN.Norml2() << "\n";
   out << "||lambda0||_2 = " << lambda.Norml2() << "\n";
}

int main(int argc, char *argv[])
{
   mfem::Device device("cpu");
   device.Print();

   int Nsteps = 20;
   int Ncheck = 3;
   int dim = 5;
   double factor = 1.05;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&Nsteps, "-n", "--num-steps", "Number of primal steps.");
   args.AddOption(&Ncheck, "-s", "--num-checkpoints", "Number of checkpoints (snaps).");
   args.AddOption(&dim, "-d", "--dim", "Vector dimension for the mfem::Vector example.");
   args.AddOption(&factor, "-f", "--factor", "Scalar factor in the toy update u_{n+1}=f*u_n.");
   args.Parse();

   if (!args.Good())
   {
      args.PrintUsage(out);
      return 1;
   }
   args.PrintOptions(out);

   RunScalarExample(Nsteps, Ncheck, factor);
   RunVectorExample(Nsteps, Ncheck, dim, factor);

   return 0;
}

