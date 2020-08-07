#include "euler.hpp"

Configuration ConfigEuler;

double SpHeatRatio;

void AnalyticalSolutionEuler(const Vector &x, double t, Vector &u);
void InitialConditionEuler(const Vector &x, Vector &u);
void InflowFunctionEuler(const Vector &x, double t, Vector &u);

Euler::Euler(FiniteElementSpace *fes_, BlockVector &u_block,
             Configuration &config_)
   : HyperbolicSystem(fes_, u_block, fes_->GetMesh()->Dimension() + 2, config_,
                      VectorFunctionCoefficient(fes_->GetMesh()->Dimension() + 2,
                                                InflowFunctionEuler))
{
   ConfigEuler = config_;
   VectorFunctionCoefficient ic(NumEq, InitialConditionEuler);

   switch (ConfigEuler.ConfigNum)
   {
      case 0:
      {
         // Periodic meshes must be used for this problem.
         ProblemName = "Euler Equations of Gas dynamics - Smooth Vortex";
         glvis_scale = "on";
         SpHeatRatio = 1.4;
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 0;
         L2_Projection(ic, u0);
         break;
      }
      case 1:
      {
         ProblemName = "Euler Equations of Gas dynamics - SOD Shock Tube";
         glvis_scale = "on";
         SpHeatRatio = 1.4;
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         L2_Projection(ic, u0);
         break;
      }
      case 2:
      {
         ProblemName = "Euler Equations of Gas dynamics - Woodward Colella";
         glvis_scale = "on";
         SpHeatRatio = 1.4;
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 3:
      {
         ProblemName = "Euler Equations of Gas dynamics - Double Mach Reflection";
         glvis_scale = "on";
         SpHeatRatio = 1.4;
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = true;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 4:
      {
         ProblemName = "Euler Equations of Gas dynamics - Sedov Blast";
         glvis_scale = "on";
         SpHeatRatio = 5.0 / 3.0;
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = true;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 5:
      {
         ProblemName = "Euler Equations of Gas dynamics - Noh Problem";
         glvis_scale = "on";
         SpHeatRatio = 5.0 / 3.0;
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = true;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 6:
      {
         ProblemName = "Euler Equations of Gas dynamics - MoST Gimmick";
         glvis_scale = "on";
         SpHeatRatio = 1.4;
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;

         Mesh *mesh = fes->GetMesh();
         const int nd = fes->GetFE(0)->GetDof();
         const int ne = fes->GetNE();
         if (mesh->Dimension() != 2) { MFEM_ABORT("Test case works only in 2D."); }
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
                     u0(3*ne*nd + e*nd+j) = 1. / SpHeatRatio;
                     break;
                  }
                  case 2:
                  case 3:
                  case 4:
                  {
                     u0(e*nd+j) = 0.125;
                     u0(3*ne*nd + e*nd+j) = 0.1 / SpHeatRatio;
                     break;
                  }
                  default:
                     MFEM_ABORT("Too many element IDs.");
               }
            }
         }

         break;
      }
      case 7:
      {
         ProblemName = "Euler Equations of Gas dynamics - Constricted Channel";
         glvis_scale = "on";
         SpHeatRatio = 1.4;
         SolutionKnown = false;
         SteadyState = true;
         TimeDepBC = false;
         ProjType = 0;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 8:
      {
         ProblemName = "Euler Equations of Gas dynamics - Gresho Vortex";
         glvis_scale = "on";
         SpHeatRatio = 1.4;
         SolutionKnown = true;
         SteadyState = true;
         TimeDepBC = false;
         ProjType = 0;
         u0.ProjectCoefficient(ic);
         break;
      }
      default:
         MFEM_ABORT("No such test case implemented.");
   }
}

double Euler::EvaluatePressure(const Vector &u) const
{
   double aux = 0.0;
   for (int l = 0; l < dim; l++)
   {
      aux += u(1+l) * u(1+l);
   }
   double pressure = (SpHeatRatio - 1.0) * (u(dim+1) - 0.5 * aux / u(0));
   if (pressure < 0.)
   {
      ostringstream press_str;
      press_str << pressure;
      string err_msg = "Negative pressure p = ";
      MFEM_ABORT(err_msg << press_str.str());
   }
   return pressure;
}

void Euler::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                         int e, int k, int i) const
{
   double pressure = EvaluatePressure(u);
   CheckAdmissibility(u);

   switch (dim)
   {
      case 1:
      {
         double vx = u(1) / u(0);
         FluxEval(0,0) = u(1);
         FluxEval(1,0) = u(1) * vx + pressure;
         FluxEval(2,0) = (u(2) + pressure) * vx;
         break;
      }
      case 2:
      {
         double vx = u(1) / u(0);
         double vy = u(2) / u(0);
         double energy = u(3) + pressure;

         FluxEval(0,0) = u(1);
         FluxEval(0,1) = u(2);

         FluxEval(1,0) = u(1) * vx + pressure;
         FluxEval(1,1) = u(1) * vy;

         FluxEval(2,0) = u(2) * vx;
         FluxEval(2,1) = u(2) * vy + pressure;

         FluxEval(3,0) = energy * vx;
         FluxEval(3,1) = energy * vy;
         break;
      }
      case 3:
      {
         double vx = u(1) / u(0);
         double vy = u(2) / u(0);
         double vz = u(3) / u(0);
         double energy = u(4) + pressure;

         FluxEval(0,0) = u(1);
         FluxEval(0,1) = u(2);
         FluxEval(0,2) = u(3);

         FluxEval(1,0) = u(1) * vx + pressure;
         FluxEval(1,1) = u(1) * vy;
         FluxEval(1,2) = u(1) * vz;

         FluxEval(2,0) = u(2) * vx;
         FluxEval(2,1) = u(2) * vy + pressure;
         FluxEval(2,2) = u(2) * vz;

         FluxEval(3,0) = u(3) * vx;
         FluxEval(3,1) = u(3) * vy;
         FluxEval(3,2) = u(3) * vz + pressure;

         FluxEval(4,0) = energy * vx;
         FluxEval(4,1) = energy * vy;
         FluxEval(4,2) = energy * vz;
         break;
      }
      default:
         MFEM_ABORT("Invalid space dimension.");
   }
}

double Euler::GetGMS(const Vector &uL, const Vector &uR, const Vector &normal) const
{
   CheckAdmissibility(uL);
   CheckAdmissibility(uR);
   double pL = EvaluatePressure(uL);
   double pR = EvaluatePressure(uR);
   double aL = sqrt(SpHeatRatio * pL / uL(0));
   double aR = sqrt(SpHeatRatio * pR / uR(0));
   double vL = uL(1)/uL(0) * normal(0);
   double vR = uR(1)/uR(0) * normal(0);

   double p = pow( (aL+aR-0.5*(SpHeatRatio-1.)*(vR-vL)) / (aL*pow(pL, (1.-SpHeatRatio)/(2.*SpHeatRatio)) + aR*pow(pR, (1.-SpHeatRatio)/(2.*SpHeatRatio)) ) , 2.*SpHeatRatio/(SpHeatRatio-1.) );

   double lambda1 = vL - aL * sqrt( 1. + (SpHeatRatio+1.)/(2.*SpHeatRatio) * max(0., (p-pL)/pL) );
   double lambda3 = vR + aR * sqrt( 1. + (SpHeatRatio+1.)/(2.*SpHeatRatio) * max(0., (p-pR)/pR) );
   return max(abs(lambda1), abs(lambda3));
}

double Euler::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                           int i) const
{
   CheckAdmissibility(u);
   switch (u.Size())
   {
      case 3:
         return abs( u(1)*n(0) / u(0) ) + sqrt(SpHeatRatio * EvaluatePressure(u) / u(0));
      case 4:
         return abs( (u(1)*n(0) + u(2)*n(1)) / u(0) )
                + sqrt(SpHeatRatio * EvaluatePressure(u) / u(0));
      case 5:
         return abs( (u(1)*n(0) + u(2)*n(1) + u(3)*n(2)) / u(0) )
                + sqrt(SpHeatRatio * EvaluatePressure(u) / u(0));
   }
}

void Euler::CheckAdmissibility(const Vector &u) const
{
   double RhoMin = 1.e-12;

   if (u.Size() != NumEq) { MFEM_ABORT("Invalid solution vector."); }

   if (u(0) < RhoMin)
   {
      ostringstream rho_str;
      rho_str << u(0);
      string err_msg = "Density too small rho = ";
      MFEM_ABORT(err_msg << rho_str.str());
   }
}

void Euler::SetBdrCond(const Vector &y1, Vector &y2, const Vector &normal,
                       int attr) const
{
   switch (attr)
   {
      case -1: // wall boundary
      {
         if (dim == 1)
         {
            y2(0) = y1(0);
            y2(1) = -y1(1);
            y2(2) = y1(2);
         }
         else if (dim == 2)
         {
            double MomTimesNorm = y1(1) * normal(0) + y1(2) * normal(1);
            y2(0) = y1(0);
            y2(1) = y1(1) - 2. * MomTimesNorm * normal(0);
            y2(2) = y1(2) - 2. * MomTimesNorm * normal(1);
            y2(3) = y1(3);
         }
         else
         {
            double MomTimesNorm = y1(1) * normal(0) + y1(2) * normal(1) + y1(3) * normal(2);
            y2(0) = y1(0);
            y2(1) = y1(1) - 2. * MomTimesNorm * normal(0);
            y2(2) = y1(2) - 2. * MomTimesNorm * normal(1);
            y2(3) = y1(3) - 2. * MomTimesNorm * normal(2);
            y2(4) = y1(4);
         }
         break;
      }
      case -2: // supersonic outlet
      {
         y2 = y1;
         break;
      }
      case -3: // supersonic inlet
      {
         break;
      }
      // TODO subsonic in- and outlet
      default:
         MFEM_ABORT("Invalid boundary attribute.");
   }
}

void Euler::ComputeDerivedQuantities(const GridFunction &u, GridFunction &d1, GridFunction &d2) const
{
   double density, momentum;
   const IntegrationRule ir = u.FESpace()->GetFE(0)->GetNodes();

   for (int e = 0; e < ne; e++)
   {
      for (int i = 0; i < nd; i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         density = u.GetValue(e, ip, 1);
         momentum = u.GetValue(e, ip, 2);
         d1(e*nd + i) = pow(momentum / density, 2.0);

         if (dim > 1)
         {
            momentum = u.GetValue(e, ip, 3);
            d1(e*nd + i) += pow(momentum / density, 2.0);
         }
         if (dim > 2)
         {
            momentum = u.GetValue(e, ip, 4);
            d1(e*nd + i) += pow(momentum / density, 2.0);
         }

         d2(e*nd + i) = (SpHeatRatio - 1.0) * (u.GetValue(e, ip, dim+2) - 0.5 * density * d1(e*nd + i));
         d1(e*nd + i) = sqrt(d1(e*nd + i));
      }
   }
}

void Euler::ComputeErrors(Array<double> & errors, const GridFunction &u,
                          double DomainSize, double t) const
{
   errors.SetSize(NumEq*3);
   Vector component(dim+2);
   VectorFunctionCoefficient uAnalytic(NumEq, AnalyticalSolutionEuler);

   if (ConfigEuler.ConfigNum == 0) { uAnalytic.SetTime(0);  }
   else { uAnalytic.SetTime(t); }

   component = 0.0;
   component(0) = 1.0;
   VectorConstantCoefficient weight1(component);
   errors[0] = u.ComputeLpError(1.0, uAnalytic, NULL, &weight1) / DomainSize;
   errors[1] = u.ComputeLpError(2.0, uAnalytic, NULL, &weight1) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight1);

   component = 0.0;
   component(1) = 1.0;
   VectorConstantCoefficient weight2(component);
   errors[3] = u.ComputeLpError(1.0, uAnalytic, NULL, &weight2) / DomainSize;
   errors[4] = u.ComputeLpError(2.0, uAnalytic, NULL, &weight2) / DomainSize;
   errors[5] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight2);

   component = 0.0;
   component(2) = 1.0;
   VectorConstantCoefficient weight3(component);
   errors[6] = u.ComputeLpError(1.0, uAnalytic, NULL, &weight3) / DomainSize;
   errors[7] = u.ComputeLpError(2.0, uAnalytic, NULL, &weight3) / DomainSize;
   errors[8] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight3);

   if (dim > 1)
   {
      component = 0.0;
      component(3) = 1.0;
      VectorConstantCoefficient weight4(component);
      errors[9] = u.ComputeLpError(1.0, uAnalytic, NULL, &weight4) / DomainSize;
      errors[10] = u.ComputeLpError(2.0, uAnalytic, NULL, &weight4) / DomainSize;
      errors[11] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight4);
   }

   if (dim > 2)
   {
      component = 0.0;
      component(4) = 1.0;
      VectorConstantCoefficient weight5(component);
      errors[12] = u.ComputeLpError(1.0, uAnalytic, NULL, &weight5) / DomainSize;
      errors[13] = u.ComputeLpError(2.0, uAnalytic, NULL, &weight5) / DomainSize;
      errors[14] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight5);
   }
}

void EvaluateEnergy(Vector &u, const double &pressure)
{
   const int dim = u.Size() - 2;
   double aux = 0.0;
   for (int l = 0; l < dim; l++)
   {
      aux += u(1+l)*u(1+l);
   }
   u(dim+1) = pressure / (SpHeatRatio - 1.0) + 0.5 * aux / u(0);
}

void AnalyticalSolutionEuler(const Vector &x, double t, Vector &u)
{
   const int dim = x.Size();
   Vector X(dim);

   for (int i = 0; i < dim; i++)
   {
      switch (ConfigEuler.ConfigNum)
      {
         case 0:
         case 5:
         case 8: // Map to the reference domain [-1,1]^d.
         {
            double center = 0.5 * (ConfigEuler.bbMin(i) + ConfigEuler.bbMax(i));
            double factor = 2.0 / (ConfigEuler.bbMax(i) - ConfigEuler.bbMin(i));
            X(i) = factor * (x(i) - center);
            t *= pow(factor, 1.0 / (double(dim)));
            break;
         }
         case 3: // Map to the reference domain [0,1]^d.
         {
            double factor = 1.0 / (ConfigEuler.bbMax(i) - ConfigEuler.bbMin(i));
            X(i) = factor * (x(i) - ConfigEuler.bbMin(i));
            t *= pow(factor, 1.0 / (double(dim)));
            break;
         }
      }
   }

   switch (ConfigEuler.ConfigNum)
   {
      case 0:
      {
         if (dim != 2) { MFEM_ABORT("Test case works only in 2D."); }

         // Map to test case specific domain [-5,5]^d.
         X *= 5.0;
         t *= 5.0;

         double beta = 5.0;
         double r = X.Norml2();
         double T0 = 1.0 - (SpHeatRatio - 1.0) * beta * beta
                     / (8.0 * SpHeatRatio * M_PI * M_PI) * exp(1.0 - r*r);

         u(0) = pow(T0, 1.0 / (SpHeatRatio - 1.0));
         u(1) = (1.0 - beta / (2.0 * M_PI) * exp(0.5 * (1.0 - r*r)) * X(1)) * u(0);
         u(2) = (1.0 + beta / (2.0 * M_PI) * exp(0.5 * (1.0 - r*r)) * X(0)) * u(0);
         EvaluateEnergy(u, u(0) * T0);
         break;
      }
      case 3:
      {
         if (dim != 2) { MFEM_ABORT("Test case works only in 2D."); }

         // Map to test case specific domain [0,4] x [0,1].
         X(0) = 4.0 * X(0);
         t *= 2.0;

         bool PostShock = X(0) < 1.0/6.0 + (X(1) + 20.0*t) / sqrt(3.0);

         if (PostShock)
         {
            u(0) = 8.0;
            u(1) =  66.0 * cos(M_PI / 6.0);
            u(2) = -66.0 * sin(M_PI / 6.0);
            EvaluateEnergy(u, 116.5);
         }
         else
         {
            u = 0.0;
            u(0) = 1.4;
            EvaluateEnergy(u, 1.0);
         }

         break;
      }
      case 5:
      {
         double r = X.Norml2();

         if (r > t / 3.)
         {
            u(0) = 1.0 + t / r;
            for (int l = 0; l < dim; l++) { u(l+1) = -X(l) / r * u(0); }
            EvaluateEnergy(u, 1.0E-6);
         }
         else
         {
            u(0) = 1.0;
            for (int l = 0; l < dim; l++) { u(l+1) = 0.0; }
            EvaluateEnergy(u, 16.0 / 3.0);
         }

         break;
      }
      case 6:
      {
         if (dim != 2) { MFEM_ABORT("Test case works only in 2D."); }

         bool left = X(0) < 0. && X(0)*X(0) + 0.24 * (X(1)+1.)*(X(1)+1.) > 1.05;
         X(0) = 0.5 * (X(0) + 1.);
         X(1) = 0.25 * (X(1) + 1.);

         if (left)
         {
            u(0) = 8.;
            u(1) = 66. * cos(M_PI / 6.);
            u(2) = -66. * sin(M_PI / 6.);
            EvaluateEnergy(u, 116.5);
         }
         else
         {
            u = 0.;
            u(0) = 1.4;
            EvaluateEnergy(u, 1.);
         }

         break;
      }
      case 8:
      {
         if (dim != 2) { MFEM_ABORT("Test case works only in 2D."); }

         double pressure = 3.0 + 4.0*log(2.0);
         double r = X.Norml2();

         u = 0.0;
         u(0) = 1.0;

         if (r < 0.2)
         {
            u(1) = -5.0 * X(1);
            u(2) =  5.0 * X(0);
            pressure = 5.0 + 12.5*r*r;
         }
         else if (r < 0.4)
         {
            u(1) = -(2.0 / r - 5.0) * X(1);
            u(2) =  (2.0 / r - 5.0) * X(0);
            pressure = 9.0 + 4.0 * (log(r) - log(0.2)) + 12.5*r*r - 20.0*r;
         }

         EvaluateEnergy(u, pressure);
         break;
      }
      default:
         MFEM_ABORT("Analytical solution not known.");
   }
}

void InitialConditionEuler(const Vector &x, Vector &u)
{
   const int dim = x.Size();
   Vector X(dim);

   for (int i = 0; i < dim; i++)
   {
      switch (ConfigEuler.ConfigNum)
      {
         case 4: // Map to the reference domain [-1,1]^d.
         {
            double center = 0.5 * (ConfigEuler.bbMin(i) + ConfigEuler.bbMax(i));
            double factor = 2.0 / (ConfigEuler.bbMax(i) - ConfigEuler.bbMin(i));
            X(i) = factor * (x(i) - center);
            break;
         }
         case 1:
         case 2: // Map to the reference domain [0,1]^d.
         {
            double factor = 1.0 / (ConfigEuler.bbMax(i) - ConfigEuler.bbMin(i));
            X(i) = factor * (x(i) - ConfigEuler.bbMin(i));
            break;
         }
      }
   }

   switch (ConfigEuler.ConfigNum)
   {
      case 0:
      case 3:
      case 5:
      case 6:
      case 8:
      {
         AnalyticalSolutionEuler(x, 0.0, u);
         break;
      }
      case 1:
      {
         if (dim != 1) { MFEM_ABORT("Test case works only in 1D."); }

         u = 0.0;
         u(0) = X.Norml2() < 0.5 ? 1.0 : 0.125;
         EvaluateEnergy(u, X.Norml2() < 0.5 ? 1.0 : 0.1);
         break;
      }
      case 2:
      {
         if (dim != 1) { MFEM_ABORT("Test case works only in 1D."); }

         u = 0.0;
         u(0) = 1.0;
         if (X(0) < 0.1)
         {
            EvaluateEnergy(u, 1000.);
         }
         else if (X(0) < 0.9)
         {
            EvaluateEnergy(u, 0.01);
         }
         else
         {
            EvaluateEnergy(u, 100.);
         }
         break;
      }
      case 4:
      {
         // Map to test case specific domain [-5,5]^d.
         X *= 5.0;

         u = 0.0;
         u(0) = 1.0;
         // TODO make sure that energy is essentially a delta distribution.
         u(dim+1) = X.Norml2() < 1.0E-1 ? 1000.0 : 1.0E-8;
         break;
      }
      case 7:
      {
         u = 0.0;
         u(0) = 1.0;
         u(1) = 1.0;
         EvaluateEnergy(u, 0.1);
         break;
      }
   }
}

void InflowFunctionEuler(const Vector &x, double t, Vector &u)
{
   switch (ConfigEuler.ConfigNum)
   {
      case 0:
      case 3:
      case 5:
      case 6:
      case 8:
      {
         AnalyticalSolutionEuler(x, t, u);
         break;
      }
      case 7:
      {
         InitialConditionEuler(x, u);
         break;
      }
      case 1:
      case 2:
      case 4: break; // No boundary conditions needed.
   }
}
