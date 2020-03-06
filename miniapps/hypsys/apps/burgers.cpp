#include "burgers.hpp"

Configuration ConfigBurgers;

double AnalyticalSolutionBurgers(const Vector &x, double t);
double InitialConditionBurgers(const Vector &x);
void InflowFunctionBurgers(const Vector &x, double t, Vector &u);

Burgers::Burgers(FiniteElementSpace *fes_, BlockVector &u_block,
                 Configuration &config_)
   : HyperbolicSystem(fes_, u_block, 1, config_, VectorFunctionCoefficient (1,
                                                                            InflowFunctionBurgers))
{
   ConfigBurgers = config_;
   SteadyState = false;
   SolutionKnown = true;

   Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();

   TimeDepBC = true;
   FunctionCoefficient ic(InitialConditionBurgers);

   if (ConfigBurgers.ConfigNum == 0)
   {
      ProjType = 0;
      L2_Projection(ic, u0);
   }
   else
   {
      ProjType = 1;
      u0.ProjectCoefficient(ic);
   }
}

void Burgers::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                           int e, int k, int i) const
{
   FluxEval = 0.5 * u(0) * u(0);
}

double Burgers::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                             int i) const
{
   return abs(u(0) * n.Sum());
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

   // Map to the reference domain [-1,1] x [-1,1].
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (ConfigBurgers.bbMin(i) + ConfigBurgers.bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (ConfigBurgers.bbMax(i) - ConfigBurgers.bbMin(i));
   }

   // TODO default test case to work in all dimensions
   switch (dim)
   {
      case 2: // TODO use transformed coordinates X instead of x
      {
         if (x(0) <= 0.5 - 0.6 * t)
         {
            return x(1) >= 0.5 + 0.15 * t ? -0.2 : 0.5;
         }
         if (0.5 - 0.6 * t < x(0) && x(0) < 0.5 - 0.25 * t)
         {
            return x(1) > -8. / 7. * x(0) + 15. / 14. - 15. / 28. * t ? -1. : 0.5;
         }
         if (0.5 - 0.25 * t < x(0) && x(0) < 0.5 + 0.5 * t)
         {
            return x(1) > x(0) / 6. + 5. / 12. - 5. / 24. * t ? -1. : 0.5;
         }
         if (0.5 + 0.5 * t < x(0) && x(0) < 0.5 + 0.8 * t)
         {
            return x(1) > x(0) - 5. / (18. * t) * (x(0) + t - 0.5)
                   * (x(0) + t - 0.5) ? -1. : (2. * x(0) - 1.) / (2 * t);
         }
         if (0.5 + 0.8 * t <= x(0))
         {
            return x(1) >= 0.5 - 0.1 * t ? -1 : 0.8;
         }

         break;
      }
      default:
         MFEM_ABORT("Invalid space dimension.");
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
