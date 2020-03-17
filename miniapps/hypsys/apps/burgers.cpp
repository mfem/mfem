#include "burgers.hpp"

Configuration ConfigBurgers;

double AnalyticalSolutionBurgers(const Vector &x, double t);
double InitialConditionBurgers(const Vector &x);
void InflowFunctionBurgers(const Vector &x, double t, Vector &u);

Burgers::Burgers(FiniteElementSpace *fes_, BlockVector &u_block,
                 Configuration &config_)
   : HyperbolicSystem(fes_, u_block, 1, config_,
   VectorFunctionCoefficient (1, InflowFunctionBurgers))
{
   ConfigBurgers = config_;

   FunctionCoefficient ic(InitialConditionBurgers);

   if (ConfigBurgers.ConfigNum == 0)
   {
      ProblemName = "Burgers Equation - Smooth Solution";
      MFEM_ABORT("No such test case implemented.");
   }
   else
   {
      SolutionKnown = true;
      SteadyState = false;
      TimeDepBC = true;
      ProjType = 1;
      u0.ProjectCoefficient(ic);
      ProblemName = "Burgers Equation - Riemann Problem";
   }
}

void Burgers::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                           int e, int k, int i) const
{
   FluxEval = 0.5 * u(0) * u(0);
}

double Burgers::GetWaveSpeed(const Vector &u, const Vector n, int e, int k, int i) const
{
   return abs(u(0) * n.Sum());
}

double Burgers::EvaluateBdrCond(const Vector &inflow, const Vector &x, const Vector &normal,
                                int n, int e, int i, int attr, int DofInd) const
{
   return inflow(DofInd);
}

void Burgers::ComputeErrors(Array<double> &errors, const GridFunction &u,
                            double DomainSize, double t) const
{
   errors.SetSize(3);
   FunctionCoefficient uAnalytic(AnalyticalSolutionBurgers);
   uAnalytic.SetTime(t);
   errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
   errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
}


double AnalyticalSolutionBurgers(const Vector &x, double t)
{
   const int dim = x.Size();

   // Map to the reference domain [-1,1].
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (ConfigBurgers.bbMin(i) + ConfigBurgers.bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (ConfigBurgers.bbMax(i) - ConfigBurgers.bbMin(i));
   }

   switch (ConfigBurgers.ConfigNum)
   {
      case 1:
      {
         if (dim != 2) { MFEM_ABORT("Test case only implemented in 2D."); }

         X(0) += 1.;
         X(1) += 1.;
         X *= 0.5; // Map to test case specific domain [0,1].

         if (X(0) <= 0.5 - 0.6 * t)
         {
            return X(1) >= 0.5 + 0.15 * t ? -0.2 : 0.5;
         }
         if (0.5 - 0.6 * t < X(0) && X(0) < 0.5 - 0.25 * t)
         {
            return X(1) > -8. / 7. * X(0) + 15. / 14. - 15. / 28. * t ? -1. : 0.5;
         }
         if (0.5 - 0.25 * t < X(0) && X(0) < 0.5 + 0.5 * t)
         {
            return X(1) > X(0) / 6. + 5. / 12. - 5. / 24. * t ? -1. : 0.5;
         }
         if (0.5 + 0.5 * t < X(0) && X(0) < 0.5 + 0.8 * t)
         {
            return X(1) > X(0) - 5. / (18. * t) * (X(0) + t - 0.5)
                   * (X(0) + t - 0.5) ? -1. : (2. * X(0) - 1.) / (2 * t);
         }
         if (0.5 + 0.8 * t <= X(0))
         {
            return X(1) >= 0.5 - 0.1 * t ? -1 : 0.8;
         }

         break;
      }
      default:
         MFEM_ABORT("No such test case implemented.");
   }
}

double InitialConditionBurgers(const Vector &x)
{
   return AnalyticalSolutionBurgers(x, 0.);
}

void InflowFunctionBurgers(const Vector &x, double t, Vector &u)
{
   u(0) = AnalyticalSolutionBurgers(x, t);
}
