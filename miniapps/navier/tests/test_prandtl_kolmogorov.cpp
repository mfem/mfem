#include "navier_solver.hpp"
#include "rans/prandtl_kolmogorov.hpp"
#include <algorithm>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   double mu_calibration_const = 0.55;
} ctx;

double turb_length_scale_mms(const Vector &c)
{
   double x = c(0);
   double y = c(1);

   return 1 + x + y;
}

Vector dturb_length_scale_mmsdx(const Vector &c)
{
   Vector dldx(2);

   dldx(0) = 1.0;
   dldx(1) = 1.0;

   return dldx;
}

double kinvis_mms(const Vector &c)
{
   double x = c(0);
   double y = c(1);

   return 0.025*(2.0+cos(x)*sin(y));
}

Vector dkinvis_mmsdx(const Vector &c)
{
   double x = c(0);
   double y = c(1);
   Vector dkvdx(2);

   dkvdx(0) = -(sin(x)*sin(y))*0.025;
   dkvdx(1) = cos(x)*cos(y)*0.025;

   return dkvdx;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int polynomial_order = 2;

   Mesh mesh = Mesh::MakeCartesian2D(64, 64, Element::QUADRILATERAL, false,
                                     2.0*M_PI,
                                     2.0*M_PI);

   // mesh.EnsureNodes();
   // mesh.Transform([](const Vector &c, Vector &u)
   // {
   //    u(0) = c(0);
   //    u(1) = c(1);

   //    if (c(0) == M_PI && c(1) == M_PI)
   //    {
   //       u(0) = c(0) + 0.25;
   //       u(1) = c(1) + 0.125;
   //    }
   // });


   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   H1_FECollection fec(polynomial_order);
   ParFiniteElementSpace velocity_fes(pmesh, &fec, pmesh->Dimension());
   ParFiniteElementSpace k_fes(pmesh, &fec);

   VectorFunctionCoefficient vel_coeff(pmesh->Dimension(), [&](const Vector &c,
                                                               double t,
                                                               Vector &u)
   {
      // const double x = c(0);
      const double y = c(1);
      u(0) = y*(1/M_PI - (-M_PI + y)/pow(M_PI,2));
      u(1) = 0.0;
   });

   FunctionCoefficient k_ex_coeff([&](const Vector &c, double t)
   {
      const double x = c(0);
      const double y = c(1);
      return 2 + cos(x)*sin(y);
   });

   FunctionCoefficient f_coeff([&](const Vector &c, double t)
   {
      const double x = c(0);
      const double y = c(1);
      const double l = turb_length_scale_mms(c);
      const Vector dldx = dturb_length_scale_mmsdx(c);
      const double mu = ctx.mu_calibration_const;

      const double nu = kinvis_mms(c);
      const Vector dnudx = dkinvis_mmsdx(c);

      return (y*(-2*M_PI + y)*sin(x)*sin(y))/pow(M_PI,2) - (8*mu*pow(M_PI - y,
                                                                     2)*l*sqrt(2 + cos(x)*sin(y)))/pow(M_PI,4) + pow(2 + cos(x)*sin(y),
                                                                           1.5)/l + 2*cos(x)*sin(y)*(nu + mu*l*sqrt(2 + cos(x)*sin(y))) - cos(x)*cos(y)*((
                                                                                    mu*cos(x)*cos(y)*l + 2*mu*
                                                                                    (2 + cos(x)*sin(y))*dldx(1))/(2.*sqrt(2 + cos(x)*sin(y))) + dnudx(1)) + sin(
                x)*sin(y)*(-(mu*l*sin(x)*sin(
                                y))/(2.*sqrt(2 + cos(x)*sin(y))) + mu*sqrt(2 + cos(x)*sin(y))*dldx(1) + dnudx(
                              0));
   });

   FunctionCoefficient kv_coeff(kinvis_mms);
   FunctionCoefficient tls_coeff(turb_length_scale_mms);

   ParGridFunction vel_gf(&velocity_fes);
   vel_gf.ProjectCoefficient(vel_coeff);

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   ParGridFunction k_gf(&k_fes), kex_gf(&k_fes);

   Vector k_tdof(k_fes.GetTrueVSize());
   k_tdof = 0.0;
   Vector rk_tdof(k_tdof.Size()), y_tdof(k_tdof.Size());

   k_gf.ProjectCoefficient(k_ex_coeff);
   k_gf.ParallelProject(k_tdof);

   PrandtlKolmogorov pk(k_fes, vel_coeff, kv_coeff, f_coeff, tls_coeff,
                        k_ex_coeff, ctx.mu_calibration_const, ess_bdr);

   // FD check
   if (0)
   {
      pk.SetTime(0.0);
      pk.SetEvalMode(TimeDependentOperator::EvalMode::ADDITIVE_TERM_2);
      Vector x(k_tdof), fx(pk.Height()), dfdx(pk.Height()), fxme(pk.Height()),
             fxpe(pk.Height());

      fx = 0.0;
      fxpe = 0.0;
      fxme = 0.0;

      pk.Mult(x, fx);
      out << "sum(fx) - 4*pi^2 = " << fx.Sum() - 4*M_PI*M_PI << std::endl;

      const double eps = 1e-8;

      std::ofstream Afdmat_out("afdmat.dat");
      Afdmat_out << std::setprecision(12);

      for (int col = 0; col < pk.Height(); col++)
      {
         x(col) += eps;
         pk.Mult(x, fxpe);
         x(col) -= 2*eps;
         pk.Mult(x, fxme);
         x(col) += eps;

         for (int i = 0; i < pk.Height(); i++)
         {
            dfdx(i) = (fxpe(i) - fxme(i)) / (2*eps);
         }

         dfdx.Print(Afdmat_out, pk.Height());
      }

      int jcur = 0;
      pk.SUNImplicitSetup(x, fx, 1, &jcur, 0.0);
      std::ofstream Amat_out("amat.dat");
      Amat_out << std::setprecision(12);
      pk.Amat->PrintMatlab(Amat_out);

      Afdmat_out.flush();
      Amat_out.flush();

      exit(0);
   }

   double t_final = 1e-3;
   double t = 0.0;
   double dt = t_final;

   ARKStepSolver pk_ode(MPI_COMM_WORLD, ARKStepSolver::IMEX);
   pk_ode.Init(pk);
   pk_ode.SetSStolerances(1e-5, 1e-5);
   pk_ode.SetMaxStep(dt);

   int flag = 0;
   // FILE* fout = fopen("arkode.log", "w");
   // int flag = ARKStepSetDiagnostics(pk_ode.GetMem(), fout);
   flag = ARKStepSetPostprocessStepFn(pk_ode.GetMem(),
                                      PrandtlKolmogorov::PostProcessCallback);
   MFEM_VERIFY(flag >= 0, "error in ARKStepSetPostprocessStepFn()");

   flag = ARKStepSetPostprocessStageFn(pk_ode.GetMem(),
                                       PrandtlKolmogorov::PostProcessCallback);
   MFEM_VERIFY(flag >= 0, "error in ARKStepSetPostprocessStageFn()");

   flag = ARKStepSetStagePredictFn(pk_ode.GetMem(),
                                   PrandtlKolmogorov::PostProcessCallback);
   MFEM_VERIFY(flag >= 0, "error in ARKStepSetStagePredictFn()");

   // flag = ARKStepSetLSetupFrequency(pk_ode.GetMem(), 1);
   // MFEM_VERIFY(flag >= 0, "error in ARKStepSetLSetupFrequency()");

   // pk_ode.UseSundialsLinearSolver();
   pk_ode.SetIMEXTableNum(ARK324L2SA_ERK_4_2_3, ARK324L2SA_DIRK_4_2_3);

   bool done = false;
   int ti = 0;
   int vis_steps = 10;

   char vishost[] = "128.15.198.77";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   if (Mpi::Root())
   {
      std::cout << "time step: " << ti << ", time: " << t << std::endl;
   }
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << *pmesh << k_gf << "\n" << "pause\n" << std::flush;

   while (!done)
   {
      pk_ode.Step(k_tdof, t, dt);

      ti++;
      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         k_ex_coeff.SetTime(t);
         k_gf.SetFromTrueDofs(k_tdof);
         double l2error = k_gf.ComputeL2Error(k_ex_coeff);
         kex_gf.ProjectCoefficient(k_ex_coeff);
         for (int i = 0; i < kex_gf.Size(); i++)
         {
            kex_gf(i) = abs(kex_gf(i) - k_gf(i));
         }
         if (Mpi::Root())
         {
            printf("time step: %d, time: %.3E, l2error = %.8E\n", ti, t, l2error);
            pk_ode.PrintInfo();
         }

         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << k_gf << std::flush;
      }
   }

   delete pmesh;
}