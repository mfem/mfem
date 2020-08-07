#include "advection.hpp"

Configuration ConfigAdv;

void AnalyticalSolutionAdv(const Vector &x, double t, Vector &u);
void InitialConditionAdv(const Vector &x, Vector &u);
void InflowFunctionAdv(const Vector &x, double t, Vector &u);
void VelocityFunctionAdv(const Vector &x, Vector &v);

Advection::Advection(FiniteElementSpace *fes_, BlockVector &u_block,
                     Configuration &config_, bool NodalQuadRule)
   : HyperbolicSystem(fes_, u_block, 1, config_,
                      VectorFunctionCoefficient (1, InflowFunctionAdv))
{
   ConfigAdv = config_;
   VectorFunctionCoefficient ic(NumEq, InitialConditionAdv);

   switch (ConfigAdv.ConfigNum)
   {
      case 0:
      {
         ProblemName = "Advection - Smooth Circular Convection";
         glvis_scale = "on";
         SolutionKnown = true;
         SteadyState = true;
         TimeDepBC = false;
         ProjType = 0;
         L2_Projection(ic, u0);
         break;
      }
      case 1:
      {
         ProblemName = "Advection - Solid Body Rotation";
         glvis_scale = "on";
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 2:
      {
         ProblemName = "Advection - Step function";
         glvis_scale = "on";
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 3:
      {
         ProblemName = "Advection - Smooth profile";
         glvis_scale = "on";
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 0;
         L2_Projection(ic, u0);;
         break;
      }
      case 4:
      {
         ProblemName = "Advection - Discontinuous and Smooth profile";
         glvis_scale = "on";
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 5:
      {
         ProblemName = "Advection - C1 curve";
         glvis_scale = "on";
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      default:
         MFEM_ABORT("No such test case implemented.");
   }

   // The following computes and stores all necessary evaluations of the time-independent velocity.
   Mesh *mesh = fes->GetMesh();
   DofInfo dofs(fes);
   const int ne = fes->GetNE();
   const IntegrationRule *IntRuleElem = GetElementIntegrationRule(fes,
                                                                  NodalQuadRule);
   const IntegrationRule *IntRuleFace = GetFaceIntegrationRule(fes); // TODO removed NodalQuadRule!
   const int nqe = IntRuleElem->GetNPoints();
   nqf = IntRuleFace->GetNPoints();
   Vector vec, vval;
   VelocityVector.SetSize(dim);
   DenseMatrix VelEval, mat(dim, nqe);

   VelElem.SetSize(dim, nqe, ne);
   VelFace.SetSize(dim, dofs.NumBdrs, ne*nqf);
   VectorFunctionCoefficient velocity(dim, VelocityFunctionAdv);

   Array<int> bdrs, orientation;
   Array<IntegrationPoint> eip(nqf*dofs.NumBdrs);

   if (dim==1)      { mesh->GetElementVertices(0, bdrs); }
   else if (dim==2) { mesh->GetElementEdges(0, bdrs, orientation); }
   else if (dim==3) { mesh->GetElementFaces(0, bdrs, orientation); }

   for (int i = 0; i < dofs.NumBdrs; i++)
   {
      FaceElementTransformations *help
         = mesh->GetFaceElementTransformations(bdrs[i]);

      if (help->Elem1No != 0)
      {
         // NOTE: If this error ever occurs, use neighbor element to
         // obtain the correct quadrature points and weight.
         MFEM_ABORT("First element has inward pointing normal.");
      }
      for (int k = 0; k < nqf; k++)
      {
         const IntegrationPoint &ip = IntRuleFace->IntPoint(k);
         help->Loc1.Transform(ip, eip[i*nqf + k]);
      }
   }

   for (int e = 0; e < ne; e++)
   {
      ElementTransformation *eltrans = fes->GetElementTransformation(e);
      velocity.Eval(VelEval, *eltrans, *IntRuleElem);

      for (int k = 0; k < nqe; k++)
      {
         VelEval.GetColumnReference(k, vec);
         mat.SetCol(k, vec);
      }

      VelElem(e) = mat;

      if (dim==1)      { mesh->GetElementVertices(e, bdrs); }
      else if (dim==2) { mesh->GetElementEdges(e, bdrs, orientation); }
      else if (dim==3) { mesh->GetElementFaces(e, bdrs, orientation); }

      for (int i = 0; i < dofs.NumBdrs; i++)
      {
         FaceElementTransformations *facetrans
            = mesh->GetFaceElementTransformations(bdrs[i]);

         for (int k = 0; k < nqf; k++)
         {
            if (facetrans->Elem1No != e)
            {
               velocity.Eval(vval, *facetrans->Elem2, eip[i * nqf + k]);
            }
            else
            {
               velocity.Eval(vval, *facetrans->Elem1, eip[i * nqf + k]);
            }

            for (int l = 0; l < dim; l++)
            {
               VelFace(l, i, e * nqf + k) = vval(l);
            }
         }
      }
   }
}

void Advection::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                             int e, int k, int i) const
{
   Vector x(dim), v(dim);
   VelocityFunctionAdv(x, v);
   v *= u(0);
   FluxEval.SetRow(0, v);
   // if (i == -1) // Element terms.
   // {
   //    VelocityVector = VelElem(e).GetColumn(k);
   //    VelocityVector *= u(0);
   //    FluxEval.SetRow(0, VelocityVector);
   // }
   // else
   // {
   //    VelocityVector = VelFace(e*nqf+k).GetColumn(i);
   //    VelocityVector *= u(0);
   //    FluxEval.SetRow(0, VelocityVector);
   // }
}

double Advection::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                               int i) const
{
   if (i == -1) // Element terms.
   {
      VelocityVector = VelElem(e).GetColumn(k);
   }
   else
   {
      VelocityVector = VelFace(e*nqf+k).GetColumn(i);
   }

   return abs(VelocityVector * n);
}

void Advection::ComputeErrors(Array<double> &errors, const GridFunction &u,
                              double DomainSize, double t) const
{
   errors.SetSize(3);
   VectorFunctionCoefficient uAnalytic(NumEq, AnalyticalSolutionAdv);
   uAnalytic.SetTime(t);
   errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
   errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
}


void VelocityFunctionAdv(const Vector &x, Vector &v)
{
   const int dim = x.Size();
   Vector X(dim);
   double s = 1.0;

   for (int i = 0; i < dim; i++)
   {
      switch (ConfigAdv.ConfigNum)
      {
         case 0:
         case 1:
         case 4:
         case 5: // Map to the reference domain [0,1]^d.
         {
            X(i) = (x(i) - ConfigAdv.bbMin(i)) / (ConfigAdv.bbMax(i) - ConfigAdv.bbMin(i));
            s *= ConfigAdv.bbMax(i) - ConfigAdv.bbMin(i);
            break;
         }
         case 2:
         case 3: // Map to the reference domain [-1,1]^d.
         {
            double center = 0.5 * (ConfigAdv.bbMin(i) + ConfigAdv.bbMax(i));
            X(i) = 2. * (x(i) - center) / (ConfigAdv.bbMax(i) - ConfigAdv.bbMin(i));
            s *= ConfigAdv.bbMax(i) - ConfigAdv.bbMin(i);
            break;
         }
      }
   }

   // Scale to be normed to a full revolution.
   s = 2.0 * pow(s, 1./dim) * M_PI;

   switch (ConfigAdv.ConfigNum)
   {
      case 0: // Rotation around corner.
      {
         switch (dim)
         {
            case 1: v(0) = s; break;
            case 2: v(0) = s*X(1); v(1) = -s*X(0); break;
            case 3: v(0) = s*X(1); v(1) = -s*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 1: // Rotation around center.
      {
         switch (dim)
         {
            case 1: v(0) = s; break;
            case 2: v(0) = s * (0.5-X(1)); v(1) = s*(X(0)-0.5); break;
            case 3: v(0) = s * (0.5-X(1)); v(1) = s*(X(0)-0.5); v(2) = 0.0; break;
         }
         break;
      }
      case 2:
      case 3:
      case 4:
      case 5:
      {
         switch (dim)
         {
            case 1: v(0) = s; break;
            case 2: v(0) = s; v(1) = -0.5*s; break;
            case 3: v(0) = s; v(1) = -0.5*s; v(2) = 0.25*s; break;
         }
         break;
      }
   }
}

void AnalyticalSolutionAdv(const Vector &x, double t, Vector &u)
{
   const int dim = x.Size();
   Vector X(dim);

   for (int i = 0; i < dim; i++)
   {
      switch (ConfigAdv.ConfigNum)
      {
         case 0:
         case 1:
         case 4:
         case 5: // Map to the reference domain [0,1]^d.
         {
            X(i) = (x(i) - ConfigAdv.bbMin(i)) / (ConfigAdv.bbMax(i) - ConfigAdv.bbMin(i));
            break;
         }
         case 2:
         case 3: // Map to the reference domain [-1,1]^d.
         {
            double center = 0.5 * (ConfigAdv.bbMin(i) + ConfigAdv.bbMax(i));
            X(i) = 2.0 * (x(i) - center) / (ConfigAdv.bbMax(i) - ConfigAdv.bbMin(i));
            break;
         }
      }
   }

   double r = X.Norml2();

   switch (ConfigAdv.ConfigNum)
   {
      case 0:
      {
         double a = 0.5, b = 0.03, c = 0.1;
         u(0) = 0.25 * (1. + tanh((r+c-a)/b)) * (1. - tanh((r-c-a)/b));
         break;
      }
      case 1:
      {
         if (dim==1) { MFEM_ABORT("Test case not implemented in 1D."); }

         double s = 0.15;
         double cone = sqrt(pow(X(0)-0.5, 2.) + pow(X(1)-0.25, 2.));
         double hump = sqrt(pow(X(0)-0.25, 2.) + pow(X(1)-0.5, 2.));

         u(0) = (1. - cone / s) * (cone <= s) +
                0.25 * (1. + cos(M_PI*hump / s)) * (hump <= s) +
                ( ( sqrt(pow(X(0)-0.5, 2.) + pow(X(1)-0.75, 2.)) <= s ) &&
                  ( abs(X(0)-0.5) >= 0.025 || (X(1) >= 0.85) ) ? 1. : 0. );
         break;
      }
      case 2: { u(0) = r < 0.2 ? 1. : 0.; break; }
      case 3: { u(0) = exp(-25. * r*r); break; }
      case 4: { u(0) = abs(r - 0.3) < 0.1 ? 1. : ( (abs(r-0.7) < 0.2) ? (exp(10.)*exp(-1./(r-0.5))*exp(1./(r-0.9))) : 0. ); }
      case 5: { u(0) = abs(r-0.25) <= 0.15 ? 0.5*(1.+cos(M_PI*(r-0.25)/0.15)) : 0.; }
   }
}

void InitialConditionAdv(const Vector &x, Vector &u)
{
   AnalyticalSolutionAdv(x, 0.0, u);
}

void InflowFunctionAdv(const Vector &x, double t, Vector &u)
{
   AnalyticalSolutionAdv(x, t, u);
}
