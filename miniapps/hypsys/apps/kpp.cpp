#include "kpp.hpp"

Configuration ConfigKPP;

double AnalyticalSolutionKPP(const Vector &x, double t);
double InitialConditionKPP(const Vector &x);
void InflowFunctionKPP(const Vector &x, double t, Vector &u);

KPP::KPP(FiniteElementSpace *fes_, BlockVector &u_block,
         Configuration &config_)
   : HyperbolicSystem(fes_, u_block, 1, config_, VectorFunctionCoefficient(1, InflowFunctionKPP))
{
   ConfigKPP = config_;
   SteadyState = false;
   SolutionKnown = false;

   Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();

   // Initialize the state.
   FunctionCoefficient ic(InitialConditionKPP);

   if (ConfigKPP.ConfigNum == 0)
   {
      // Use L2 projection to achieve optimal convergence order.
      L2_FECollection l2_fec(fes->GetFE(0)->GetOrder(), dim);
      FiniteElementSpace l2_fes(mesh, &l2_fec, NumEq, Ordering::byNODES);
      GridFunction l2_proj(&l2_fes);
      l2_proj.ProjectCoefficient(ic);
      u0.ProjectGridFunction(l2_proj);
   }
   else // Bound preserving projection.
   {
      u0.ProjectCoefficient(ic);
   }
}

void KPP::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                       int e, int k, int i) const
{
   FluxEval(0,0) = sin(u(0));
   FluxEval(0,1) = cos(u(0));
}

double KPP::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                              int i) const
{
   return 1.;
}

void KPP::ComputeErrors(Array<double> &errors, const GridFunction &u,
                             double DomainSize, double t) const
{
   errors.SetSize(3);
   FunctionCoefficient uAnalytic(AnalyticalSolutionKPP);
   uAnalytic.SetTime(t);
   errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
   errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
}


double AnalyticalSolutionKPP(const Vector &x, double t)
{
   const int dim = x.Size();

   // Map to the reference domain [-1,1].
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (ConfigKPP.bbMin(i) + ConfigKPP.bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (ConfigKPP.bbMax(i) - ConfigKPP.bbMin(i));
   }

   X(0) = 2 * X(0);
   X(1) = 2 * X(1) - 0.5;

   return X.Norml2() <= 1 ? 3.5 * M_PI : 0.25 * M_PI; // TODO this is just the initial condition.
}

double InitialConditionKPP(const Vector &x)
{
   return AnalyticalSolutionKPP(x, 0.);
}

void InflowFunctionKPP(const Vector &x, double t, Vector &u)
{
   u(0) = AnalyticalSolutionKPP(x, t);
}
