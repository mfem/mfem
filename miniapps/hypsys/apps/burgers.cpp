#include "burgers.hpp"

Configuration ConfigBurgers;

void AnalyticalSolutionBurgers(const Vector &x, double t, Vector &u);
void InitialConditionBurgers(const Vector &x, Vector &u);
void InflowFunctionBurgers(const Vector &x, double t, Vector &u);

Burgers::Burgers(FiniteElementSpace *fes_, BlockVector &u_block,
                 Configuration &config_)
   : HyperbolicSystem(fes_, u_block, 1, config_,
                      VectorFunctionCoefficient (1, InflowFunctionBurgers))
{
   ConfigBurgers = config_;
   VectorFunctionCoefficient ic(NumEq, InitialConditionBurgers);

   switch (ConfigBurgers.ConfigNum)
   {
      case 0:
      {
         ProblemName = "Burgers Equation - 1D";
         glvis_scale = "on";
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         L2_Projection(ic, u0);
         break;
      }
      case 1:
      {
         ProblemName = "Burgers Equation - Riemann Problem";
         glvis_scale = "on";
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = true;
         ProjType = 1;
         L2_Projection(ic, u0);
         break;
      }
      case 2:
      {
         ProblemName = "Burgers Equation - MoST Gimmick";
         glvis_scale = "on";
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;

         Mesh *mesh = fes->GetMesh();
         const int nd = fes->GetFE(0)->GetDof();
         const int ne = fes->GetNE();
         if (mesh->Dimension() != 2) { MFEM_ABORT("Test case is 2D."); }
         u0 = 0.;

         for (int e = 0; e < ne; e++)
         {
            int id = mesh->GetElement(e)->GetAttribute();
            for (int j = 0; j < nd; j++)
            {
               switch (id)
               {
                  case 1:
                  {
                     u0(e*nd+j) = 1.;
                     break;
                  }
                  case 2:
                  case 3:
                  case 4:
                  {
                     u0(e*nd+j) = 0.125;
                     break;
                  }
                  default:
                     MFEM_ABORT("Too many element IDs.");
               }
            }
         }

         break;
      }
      default:
         MFEM_ABORT("No such test case implemented.");
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
   VectorFunctionCoefficient uAnalytic(NumEq, AnalyticalSolutionBurgers);
   uAnalytic.SetTime(t);
   errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
   errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
}


void AnalyticalSolutionBurgers(const Vector &x, double t, Vector &u)
{
   const int dim = x.Size();
   Vector X(dim);

   // Map to the reference domain [0,1]^d.
   for (int i = 0; i < dim; i++)
   {
      double factor = 1.0 / ( ConfigBurgers.bbMax(i) - ConfigBurgers.bbMin(i));
      X(i) = factor * (x(i) - ConfigBurgers.bbMin(i));
      t *= pow(factor, 1.0 / (double(dim)));
   }

   switch (ConfigBurgers.ConfigNum)
   {
      case 0:
      {
         if (dim != 1) { MFEM_ABORT("Test case only implemented in 1D."); }

         double un = sin(2.0*M_PI*X(0));
         double fn, fpn;
         double tol = 1.E-15;
         double error = 1.0;
         int iter = 0, maxiter = 100;

         while(error > tol)
         {
            // Do not trust this solution at a time later than t = 0.1.
            if (iter == maxiter) { break; }

            fn = sin(2.0*M_PI*(X(0)-un*t))-un;
            fpn = -2.0*M_PI*t*cos(2.*M_PI*(X(0)-un*t))-1.0;
            un -= fn/fpn;
            error = abs(sin(2.*M_PI*(X(0)-un*t))-un);
            iter++;
         }

         u(0) = un;
         break;
      }
      case 1:
      case 2:
      {
         if (dim != 2) { MFEM_ABORT("Test case only implemented in 2D."); }

         if (X(0) <= 0.5 - 0.6 * t)
         {
            u(0) = X(1) >= 0.5 + 0.15 * t ? -0.2 : 0.5;
         }
         else if (X(0) < 0.5 - 0.25 * t)
         {
            u(0) = X(1) > -8. / 7. * X(0) + 15. / 14. - 15. / 28. * t ? -1. : 0.5;
         }
         else if (X(0) < 0.5 + 0.5 * t)
         {
            u(0) = X(1) > X(0) / 6. + 5. / 12. - 5. / 24. * t ? -1. : 0.5;
         }
         else if (X(0) < 0.5 + 0.8 * t)
         {
            u(0) = X(1) > X(0) - 5. / (18. * t) * (X(0) + t - 0.5)
                   * (X(0) + t - 0.5) ? -1. : (2. * X(0) - 1.) / (2 * t);
         }
         else
         {
            u(0) = X(1) >= 0.5 - 0.1 * t ? -1 : 0.8;
         }

         break;
      }
   }
}

void InitialConditionBurgers(const Vector &x,Vector &u)
{
   switch (ConfigBurgers.ConfigNum)
   {
      case 0:
      case 1: { AnalyticalSolutionBurgers(x, 0.0, u); break; }
      case 2: { u(0) = 0.0; break; }
   }
}

void InflowFunctionBurgers(const Vector &x, double t, Vector &u)
{
   switch (ConfigBurgers.ConfigNum)
   {
      case 0:
      case 1: { AnalyticalSolutionBurgers(x, t, u); break; }
      case 2: { u(0) = 0.0; break; }
   }
}
