#include "template.hpp"

Configuration ConfigTEMPLATE;

void AnalyticalSolutionTEMPLATE(const Vector &x, double t, Vector &u);
void InitialConditionTEMPLATE(const Vector &x, Vector &u);
void InflowFunctionTEMPLATE(const Vector &x, double t, Vector &u);

TEMPLATE::TEMPLATE(FiniteElementSpace *fes_, BlockVector &u_block,
                   Configuration &config_)
   : HyperbolicSystem(fes_, u_block, fes_->GetMesh()->Dimension() + 1, config_)
{
   ConfigTEMPLATE = config_;
   SteadyState = false;
   SolutionKnown = false;

   Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();

   // Initialize the state.
   VectorFunctionCoefficient ic(NumEq, InitialConditionTEMPLATE);
   VectorFunctionCoefficient bc(NumEq, InflowFunctionTEMPLATE);

   if (ConfigTEMPLATE.ConfigNum == 0)
   {
      // Use L2 projection to achieve optimal convergence order.
      L2_FECollection l2_fec(fes->GetFE(0)->GetOrder(), dim);
      FiniteElementSpace l2_fes(mesh, &l2_fec, NumEq, Ordering::byNODES);
      GridFunction l2_proj(&l2_fes);
      l2_proj.ProjectCoefficient(bc);
      inflow.ProjectGridFunction(l2_proj);
      l2_proj.ProjectCoefficient(ic);
      u0.ProjectGridFunction(l2_proj);
   }
   else
   {
      // Bound preserving projection.
      u0.ProjectCoefficient(ic);
      inflow.ProjectCoefficient(bc);
   }
}

void TEMPLATE::EvaluateFlux(const Vector &u, DenseMatrix &f,
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
   uAnalytic.SetTime(0); // Right now we use initial condition = solution.
   errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
   errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
}

void TEMPLATE::WriteErrors(const Array<double> &errors) const
{
   ofstream file("errors.txt", ios_base::app);

   if (!file)
   {
      MFEM_ABORT("Error opening file.");
   }
   else
   {
      ostringstream strs;
      strs << errors[0] << " " << errors[1] << " " << errors[2] << "\n";
      string str = strs.str();
      file << str;
      file.close();
   }
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
