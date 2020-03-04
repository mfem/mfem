#include "template.hpp"

Configuration ConfigTEMPLATE;

void AnalyticalSolutionTEMPLATE(const Vector &x, double t, Vector &u);
void InitialConditionTEMPLATE(const Vector &x, Vector &u);
void InflowFunctionTEMPLATE(const Vector &x, double t, Vector &u);

TEMPLATE::TEMPLATE(FiniteElementSpace *fes_, BlockVector &u_block,
                   Configuration &config_)
   : HyperbolicSystem(fes_, u_block, NUMEQ, config_,
                      VectorFunctionCoefficient(NUMEQ, InflowFunctionTEMPLATE))
{
   ConfigTEMPLATE = config_;
   SteadyState = false;
   SolutionKnown = false;

   Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();

   VectorFunctionCoefficient ic(NumEq, InitialConditionTEMPLATE);

   if (ConfigTEMPLATE.ConfigNum == 0)
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

void TEMPLATE::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                            int e, int k, int i) const
{
   // TODO
}

double TEMPLATE::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                              int i) const
{
   //TODO
}

void TEMPLATE::ComputeErrors(Array<double> &errors, const GridFunction &u,
                             double DomainSize, double t) const
{
   errors.SetSize(3);
   VectorFunctionCoefficient uAnalytic(NumEq, AnalyticalSolutionTEMPLATE);
   uAnalytic.SetTime(t);
   errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
   errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
}


void AnalyticalSolutionTEMPLATE(const Vector &x, double t, Vector &u)
{
   // TODO
}

void InitialConditionTEMPLATE(const Vector &x, Vector &u)
{
   AnalyticalSolutionTEMPLATE(x, 0., u);
}

void InflowFunctionTEMPLATE(const Vector &x, double t, Vector &u)
{
   AnalyticalSolutionTEMPLATE(x, t, u);
}
