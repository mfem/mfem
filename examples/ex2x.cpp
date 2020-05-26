//                                MFEM Example 2x
//
// Compile with: make ex2x
//
// Sample runs:  ex2x
//               ex2x -p 1 -tf 2 -tol 1e-2 -dt 3e-3
//               ex2x -gp -p 6 -tf 4 -tol 1e-2
// Description:
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/*
class InitialCondition : public VectorCoefficient
{
private:
  int problem_;
  double beta_;

public:
  InitialCondition(int problem);

  void Eval(Vector &V, ElementTransformation &T,
       const IntegrationPoint &ip);
};
*/
void InitialCondition(int problem, Vector &y);

class ExampleTDO : public TimeDependentOperator
{
private:
   int problem_;
   double beta_;

public:
   ExampleTDO(int problem);
   void Mult(const Vector &y, Vector &dydt) const;
};

int main(int argc, char *argv[])
{
   int prob = 1;
   int ode_solver_type = 6;
   int ode_msr_type = 1;
   int ode_acc_type = 3;
   int ode_rej_type = 2;
   int ode_lim_type = 1;

   double t_init  = 0.0;
   double t_final = 1.0;;
   double dt = -1.0;
   double tol = 1e-2;
   double rho = 1.2;

   double diff_eta = 1.0;

   double gamma_acc = 0.9;
   double kI_acc = 1.0 / 15.0;
   double kP_acc = 0.13;
   double kD_acc = 0.2;

   double gamma_rej = 0.9;
   double kI_rej = 0.2;
   double kP_rej = 0.0;
   double kD_rej = 0.2;

   double lim_lo  = 1.0;
   double lim_hi  = 1.2;
   double lim_max = 2.0;

   bool epus = true;

   bool gnuplot = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&prob, "-p", "--problem-type",
                  "Problem Type From Gustafsson 1988:  1 - 7");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Heun-Euler, 2 - RKF12, "
                  "3 - BogackiShampine, 4 - RKF45, 5 - Cash-Karp, "
                  "6 - Dormand-Prince.");
   args.AddOption(&ode_msr_type, "-err", "-error-measure",
                  "Error measure:\n"
                  "\t   1 - Absolute/Relative Error with Infinity-Norm\n"
                  "\t   2 - Absolute/Relative Error with 2-Norm\n");
   args.AddOption(&ode_acc_type, "-acc", "-accept-factor",
                  "Adjustment factor after accepted steps:\n"
                  "\t   1 - Standard error (integrated and scaled)\n"
                  "\t   2 - Integrated error\n"
                  "\t   3 - Proportional and Integrated errors\n"
                  "\t   4 - Proportional, Integrated, and Derivative errors\n");
   args.AddOption(&ode_rej_type, "-rej", "-reject-factor",
                  "Adjustment factor after rejected steps:\n"
                  "\t   1 - Standard error (integrated and scaled)\n"
                  "\t   2 - Integrated error\n"
                  "\t   3 - Proportional and Integrated errors\n"
                  "\t   4 - Proportional, Integrated, and Derivative errors\n");
   args.AddOption(&ode_lim_type, "-lim", "-limiter",
                  "Adjustment limiter:\n"
                  "\t   1 - Dead zone limiter\n"
                  "\t   2 - Maximum limiter");
   args.AddOption(&kP_acc, "-kPa", "--k-P-acc",
                  "Proportional gain for accepted steps.");
   args.AddOption(&kI_acc, "-kIa", "--k-I-acc",
                  "Integral gain for accepted steps.");
   args.AddOption(&kD_acc, "-kDa", "--k-D-acc",
                  "Derivative gain for accepted steps.");
   args.AddOption(&kP_rej, "-kPr", "--k-P-rej",
                  "Proportional gain for rejected steps.");
   args.AddOption(&kI_rej, "-kIr", "--k-I-rej",
                  "Integral gain for rejected steps.");
   args.AddOption(&kD_rej, "-kDr", "--k-D-rej",
                  "Derivative gain for rejected steps.");
   args.AddOption(&lim_lo, "-lo", "--lower-limit",
                  "Lower limit of dead zone.");
   args.AddOption(&lim_hi, "-hi", "--upper-limit",
                  "Upper limit of dead zone.");
   args.AddOption(&lim_max, "-max", "--max-limit",
                  "Limiter maximum.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Initial time step size.");
   args.AddOption(&tol, "-tol", "--tolerance",
                  "Tolerance.");
   args.AddOption(&rho, "-rho", "--rejection",
                  "Rejection tolerance.");
   args.AddOption(&epus, "-epus", "--error-per-unit-step",
                  "-eps", "--error-per-step",
                  "Select Error per step or error per unit step.");
   args.AddOption(&gnuplot, "-gp", "--gnuplot", "-no-gp", "--no-gnuplot",
                  "Enable or disable GnuPlot visualization.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (dt <= 0.0) { dt = 0.1 * (t_final - t_init); }

   // 4. Prepare GnuPlot output file if needed
   ofstream ofs;
   if (gnuplot || visualization)
   {
      ofs.open("ex2x.dat");
   }

   ODEController ode_controller;

   ODEEmbeddedSolver        * ode_solver   = NULL;
   ODERelativeErrorMeasure  * ode_err_msr  = NULL;
   ODEStepAdjustmentFactor  * ode_step_acc = NULL;
   ODEStepAdjustmentFactor  * ode_step_rej = NULL;
   ODEStepAdjustmentLimiter * ode_step_lim = NULL;

   switch (ode_solver_type)
   {
      case 1: ode_solver = new HeunEulerSolver; break;
      case 2: ode_solver = new FehlbergRK12Solver; break;
      case 3: ode_solver = new BogackiShampineSolver; break;
      case 4: ode_solver = new FehlbergRK45Solver; break;
      case 5: ode_solver = new CashKarpSolver; break;
      case 6: ode_solver = new DormandPrinceSolver; break;
   }
   switch (ode_msr_type)
   {
      case 1:
         ode_err_msr = new MaxAbsRelDiffMeasure(diff_eta);
         break;
      case 2:
         ode_err_msr = new L2AbsRelDiffMeasure(diff_eta);
         break;
      default:
         cout << "Unknown error measure type: " << ode_msr_type << '\n';
         return 3;
   }
   switch (ode_acc_type)
   {
      case 1:
         ode_step_acc = new StdAdjFactor(gamma_acc, kI_acc);
         break;
      case 2:
         ode_step_acc = new IAdjFactor(kI_acc);
         break;
      case 3:
         ode_step_acc = new PIAdjFactor(kP_acc, kI_acc);
         break;
      case 4:
         ode_step_acc = new PIDAdjFactor(kP_acc, kI_acc, kD_acc);
         break;
      default:
         cout << "Unknown adjustment factor type: " << ode_acc_type << '\n';
         return 3;
   }
   switch (ode_rej_type)
   {
      case 1:
         ode_step_rej = new StdAdjFactor(gamma_rej, kI_rej);
         break;
      case 2:
         ode_step_rej = new IAdjFactor(kI_rej);
         break;
      case 3:
         ode_step_rej = new PIAdjFactor(kP_rej, kI_rej);
         break;
      case 4:
         ode_step_rej = new PIDAdjFactor(kP_rej, kI_rej, kD_rej);
         break;
      default:
         cout << "Unknown adjustment factor type: " << ode_rej_type << '\n';
         return 3;
   }
   switch (ode_lim_type)
   {
      case 1:
         ode_step_lim = new DeadZoneLimiter(lim_lo, lim_hi, lim_max);
         break;
      case 2:
         ode_step_lim = new MaxLimiter(lim_max);
         break;
      default:
         cout << "Unknown adjustment limiter type: " << ode_lim_type << '\n';
         return 3;
   }

   ExampleTDO tdo(prob);
   ode_solver->Init(tdo);

   ode_controller.Init(*ode_solver, *ode_err_msr,
                       *ode_step_acc, *ode_step_rej, *ode_step_lim);

   ode_controller.SetTimeStep(dt);
   ode_controller.SetTolerance(tol);
   ode_controller.SetRejectionLimit(rho);
   if (epus)
   {
      ode_controller.SetErrorPerUnitStep();
   }
   else
   {
      ode_controller.SetErrorPerStep();
   }
   if (gnuplot || visualization) { ode_controller.SetOutput(ofs, true); }

   Vector y;
   InitialCondition(prob, y);

   if (gnuplot || visualization)
   {
      ofs << t_init << '\t' << 0 << '\t' << 0.0 << '\t' << dt;
      for (int i=0; i<y.Size(); i++)
      {
         ofs << "\t" << y(i);
      }
      ofs << endl;

   }
   if (gnuplot)
   {
      ofstream ofs_inp("gnuplot_ex2x.inp");
      ofs_inp << "plot 'ex2x.dat' using 1:2 w l t 'dt';\npause -1;\n";
      ofs_inp << "plot ";
      for (int i=0; i<y.Size(); i++)
      {
         ofs_inp << "'ex2x.dat' using 1:" << i + 3
                 << " w l t 'y" << i + 1 << "'";
         if (i < y.Size() - 1)
         {
            ofs_inp << ", ";
         }
      }
      ofs_inp << ";\npause -1;\n" << endl;
      ofs_inp.close();
   }

   ode_controller.Run(y, t_init, t_final);

   ofs.close();

   if (visualization)
   {
      ifstream ifs("ex2x.dat");

      int nsteps = ode_controller.GetNSteps();
      int neqns  = tdo.Width();

      cout << "Num Steps: " << nsteps << endl;

      H1_FECollection fec(1, 1);

      Mesh mesh(nsteps);
      FiniteElementSpace fespace(&mesh, &fec);
      GridFunction dt_gf(&fespace);
      GridFunction r_gf(&fespace);
      GridFunction n_gf(&fespace);

      Mesh y_mesh(1, (nsteps+1) * neqns, nsteps * neqns);
      for (int j=0; j<neqns; j++)
      {
         for (int i=0; i<nsteps; i++)
         {
            int v[2];
            v[0] = (nsteps + 1) * j + i;
            v[1] = (nsteps + 1) * j + i + 1;
            y_mesh.AddSegment(v);
         }
         for (int i=0; i<=nsteps; i++)
         {
            double v = 0.0;
            y_mesh.AddVertex(&v);
         }
      }
      FiniteElementSpace y_fespace(&y_mesh, &fec);
      GridFunction y_gf(&y_fespace);

      for (int i=0; i<=nsteps; i++)
      {
         double t, ns, r, dt, y;
         //Vector y(neqns);
         ifs >> t >> ns >> r >> dt;

         mesh.GetVertex(i)[0] = t;
         dt_gf[i] = dt;
         r_gf[i] = r / tol;
         n_gf[i] = (double)ns;

         for (int j=0; j<neqns; j++)
         {
            ifs >> y;

            y_mesh.GetVertex((nsteps+1) * j + i)[0] = t;
            y_gf[(nsteps + 1) * j + i] = y;
         }
      }

      ifs.close();

      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream y_sock(vishost, visport);
      y_sock.precision(8);
      y_sock << "solution\n" << y_mesh << y_gf
             << " window_title 'Solution'"
             << " window_geometry 0 0 400 350" << " keys mac" << flush;

      socketstream n_sock(vishost, visport);
      n_sock.precision(8);
      n_sock << "solution\n" << mesh << n_gf
             << " window_title 'Number of Steps'"
             << " window_geometry 400 0 400 350" << " keys mac" << flush;

      socketstream r_sock(vishost, visport);
      r_sock.precision(8);
      r_sock << "solution\n" << mesh << r_gf
             << " window_title 'Normalized Error'"
             << " window_geometry 800 0 400 350" << " keys mac" << flush;

      socketstream dt_sock(vishost, visport);
      dt_sock.precision(8);
      dt_sock << "solution\n" << mesh << dt_gf
              << " window_title 'Time Step'"
              << " window_geometry 1200 0 400 350" << " keys mac" << flush;
   }

   delete ode_solver;
   delete ode_err_msr;
   delete ode_step_acc;
   delete ode_step_rej;
   delete ode_step_lim;
}

/*
InitialCondition::InitialCondition(int problem)
  : VectorCoefficient(0),
    problem_(problem),
    beta_(0.0)
{
  switch(problem_)
  {
  case 1:
  case 2:
    vdim = 4;
    break;
  case 3:
  case 6:
    vdim = 3;
    break;
  case 4:
  case 5:
  case 7:
    vdim = 2;
    beta_ = 3.0;
    break;
  case 8:
    vdim = 2;
    beta_ = 8.533;
    break;
  default:
    vdim = 0;
    break;
  }
}

void InitialCondition::Eval(Vector &v, ElementTransformation &T,
             const IntegrationPoint &ip)
{
  v.SetSize(vdim);

  switch(problem_)
  {
  case 1:
    v(0) = v(2) = 1.0;
    v(1) = v(3) = 0.0;
    break;
  case 2:
    v = 1.0;
    break;
  case 3:
    v(0) = 1.0; v(1) = v(2) = 0.0;
    break;
  case 4:
  case 5:
    v(0) = 2.0; v(1) = 0.0;
    break;
  case 6:
    v(0) = v(1) = 1.0; v(2) = 0.0;
    break;
  case 7:
  case 8:
    v(0) = 1.3; v(1) = beta_;
    break;
  default:
    v = 0.0;
    break;
  }
}
*/

void InitialCondition(int problem, Vector &y)
{
   switch (problem)
   {
      case 1:
         y.SetSize(4);
         y(0) = y(2) = 1.0;
         y(1) = y(3) = 0.0;
         break;
      case 2:
         y.SetSize(4);
         y = 1.0;
         break;
      case 3:
         y.SetSize(3);
         y(0) = 1.0; y(1) = y(2) = 0.0;
         break;
      case 4:
      case 5:
         y.SetSize(2);
         y(0) = 2.0; y(1) = 0.0;
         break;
      case 6:
         y.SetSize(3);
         y(0) = y(1) = 1.0; y(2) = 0.0;
         break;
      case 7:
         y.SetSize(2);
         y(0) = 1.3; y(1) = 3.0;
         break;
      case 8:
         y.SetSize(2);
         y(0) = 1.3; y(1) = 8.533;
         break;
      default:
         y = 0.0;
         break;
   }
}

ExampleTDO::ExampleTDO(int problem)
   : problem_(problem)
{
   switch (problem_)
   {
      case 1:
         height = width = 4;
         break;
      case 2:
         height = width = 4;
         beta_ = 0.1;
         break;
      case 3:
         height = width = 3;
         break;
      case 4:
         height = width = 2;
         break;
      case 5:
         height = width = 2;
         break;
      case 6:
         height = width = 3;
         break;
      case 7:
         height = width = 2;
         beta_ = 3.0;
         break;
      case 8:
         height = width = 2;
         beta_ = 8.533;
         break;
      default:
         beta_ = 0.0;
   }
}

void ExampleTDO::Mult(const Vector &y, Vector & dydt) const
{
   dydt.SetSize(y.Size());

   switch (problem_)
   {
      case 1:
         dydt(0) =     -1.0 * y(0) +         y(1);
         dydt(1) =   -100.0 * y(0) -         y(1);
         dydt(2) =   -100.0 * y(2) +         y(3);
         dydt(3) = -10000.0 * y(2) - 100.0 * y(3);
         break;
      case 2:
         dydt(0) =   -1.0 * y(0) + 2.0;
         dydt(1) =  -10.0 * y(1) + beta_ * y(0) * y(0);
         dydt(2) =  -40.0 * y(2) + 4.0 * beta_ * (y(0) * y(0) + y(1) * y(1));
         dydt(3) = -100.0 * y(3) +
                   10.0 * beta_ * (y(0) * y(0) + y(1) * y(1) + y(2) * y(2));
         break;
      case 3:
         dydt(0) =  -0.04 * y(0) + 0.01 * y(1) * y(2);
         dydt(1) = 400.0  * y(0) - 100.0 * y(1) * (y(2) + 30.0 * y(1));
         dydt(2) =  30.0  * y(1) * y(1);
         break;
      case 4:
         dydt(0) = y(1);
         dydt(1) = (1.0 - y(0) * y(0)) * y(1) - y(0);
         break;
      case 5:
         dydt(0) = y(1);
         dydt(1) = 50.0 * (1.0 - y(0) * y(0)) * y(1) - 10.0 * y(0);
         break;
      case 6:
         dydt(0) = -(55.0 + y(2)) * y(0) + 65.0 * y(1);
         dydt(1) = 0.0785 * (y(0) - y(1));
         dydt(2) =    0.1 * y(0);
         break;
      case 7:
      case 8:
         dydt(0) = 1.0 + y(0) * y(0) * y(1) - (beta_ + 1.0) * y(0);
         dydt(1) = y(0) * (beta_ - y(0) * y(1));
         break;
      default:
         dydt = 0.0;
         break;
   }
}

