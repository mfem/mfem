#include "shallowwater.hpp"

Configuration ConfigShallowWater;

const double GravConst = 1.; // TODO

void AnalyticalSolutionSWE(const Vector &x, double t, Vector &u);
void InitialConditionSWE(const Vector &x, Vector &u);
void InflowFunctionSWE(const Vector &x, double t, Vector &u);

ShallowWater::ShallowWater(FiniteElementSpace *fes_, BlockVector &u_block,
                           Configuration &config_)
   : HyperbolicSystem(fes_, u_block, fes_->GetMesh()->Dimension() + 1, config_,
                      VectorFunctionCoefficient(fes_->GetMesh()->Dimension() + 1, InflowFunctionSWE))
{
   ConfigShallowWater = config_;
   SteadyState = false;
   SolutionKnown = true;

   Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();

   VectorFunctionCoefficient ic(NumEq, InitialConditionSWE);

   if (ConfigShallowWater.ConfigNum == 0)
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

void ShallowWater::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                                int e, int k, int i) const
{
   const int dim = u.Size() - 1;
   double H0 = 0.001;

   if (u.Size() != NumEq)
   {
      MFEM_ABORT("Invalid solution vector.");
   }
   if (u(0) < H0)
   {
      MFEM_ABORT("Water height too small.");
   }

   switch (dim)
   {
      case 1:
      {
         FluxEval(0,0) = u(1);
         FluxEval(1,0) = u(1)*u(1)/u(0) + GravConst / 2. * u(0)*u(0);
         break;
      }
      case 2:
      {
         FluxEval(0,0) = u(1);
         FluxEval(0,1) = u(2);

         FluxEval(1,0) = u(1)*u(1)/u(0) + 0.5 * GravConst * u(0)*u(0);
         FluxEval(1,1) = u(1)*u(2)/u(0);

         FluxEval(2,0) = u(2)*u(1)/u(0);
         FluxEval(2,1) = u(2)*u(2)/u(0) + 0.5 * GravConst * u(0)*u(0);
         break;
      }
      default: MFEM_ABORT("Invalid space dimension.");
   }
}

/* void ShallowWater::EvaluateFluxDerivative(const Vector &u, Vector &df, int n)
{
   switch (dim)
   {
   case 1:
   {
      df(0) = 1;
      df(1) = pow(-u(1) / u(0), 2.) + GravConst * u(0);
      df * = normal(0);
      break;
   }
   case 2:
   {
      double velx = u(1) / u(0);
      double vely = u(2) / u(0);

      break;
   }
   default:
      MFEM_ABORT("Invalid space dimension.");
   }
} */

double ShallowWater::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                                  int i) const
{
   switch (u.Size())
   {
      case 2:
         return abs(u(1)*n(0)) / u(0) + sqrt(GravConst * u(0));
      case 3:
         return abs(u(1)*n(0) + u(2)*n(1)) / u(0) + sqrt(GravConst * u(0));
      default: MFEM_ABORT("Invalid solution vector.");
   }
}

void ShallowWater::ComputeErrors(Array<double> &errors, const GridFunction &u,
                                 double DomainSize, double t) const
{
   errors.SetSize(3);
   VectorFunctionCoefficient uAnalytic(NumEq, AnalyticalSolutionSWE);
   // Right now we use initial condition = solution due to periodic mesh.
   uAnalytic.SetTime(0);
   errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
   errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
}


void AnalyticalSolutionSWE(const Vector &x, double t, Vector &u)
{
   const int dim = x.Size();
   u.SetSize(dim+1);

   // Map to the reference domain [-1,1] x [-1,1].
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (ConfigShallowWater.bbMin(i) + ConfigShallowWater.bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (ConfigShallowWater.bbMax(i) - ConfigShallowWater.bbMin(i));
   }

   switch (ConfigShallowWater.ConfigNum)
   {
      case 0: // Vorticity advection
      {
         X *= 50.; // Map to test case specific domain [-50,50] x [-50,50].

         if (dim == 1)
         {
            MFEM_ABORT("Test case only implemented in 2D.");
         }

         // Parameters
         double M = .5;
         double c1 = -.04;
         double c2 = .02;
         double a = M_PI / 4.;
         double x0 = 0.;
         double y0 = 0.;

         double f = -c2 * ( pow(X(0) - x0 - M*t*cos(a), 2.)
                            + pow(X(1) - y0 - M*t*sin(a), 2.) );

         u(0) = 1.;
         u(1) = M*cos(a) + c1 * (X(1) - y0 - M*t*sin(a)) * exp(f);
         u(2) = M*sin(a) - c1 * (X(0) - x0 - M*t*cos(a)) * exp(f);
         u *= 1. - c1*c1 / (4.*c2*GravConst) * exp(2.*f);

         break;
      }
      case 1:
      {
         u = 0.;
         u(0) = X.Norml2() < 0.5 ? 1. : .125;
         break;
      }
      default:
      {
         MFEM_ABORT("No such test case implemented.");
      }
   }
}

void InitialConditionSWE(const Vector &x, Vector &u)
{
   AnalyticalSolutionSWE(x, 0., u);
}

void InflowFunctionSWE(const Vector &x, double t, Vector &u)
{
   AnalyticalSolutionSWE(x, t, u);
}
