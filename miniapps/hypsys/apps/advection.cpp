#include "advection.hpp"

Configuration ConfigAdvection;

double AnalyticalSolutionAdv(const Vector &x, double t);
double InitialConditionAdv(const Vector &x);
void InflowFunctionAdv(const Vector &x, double t, Vector &u);
void VelocityFunctionAdv(const Vector &x, Vector &v);

Advection::Advection(FiniteElementSpace *fes_, BlockVector &u_block,
                     Configuration &config_, bool NodalQuadRule, DofInfo &dofs)
   : HyperbolicSystem(fes_, u_block, 1, config_,
                      VectorFunctionCoefficient (1, InflowFunctionAdv))
{
   ConfigAdvection = config_;

   FunctionCoefficient ic(InitialConditionAdv);

   switch (ConfigAdvection.ConfigNum)
   {
      case 0:
      {
         ProblemName = "Advection - Smooth Circular Convection";
         valuerange = "0 1";
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
         valuerange = "0 1";
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 2: // For debugging and having a real conservation law.
      {
         ProblemName = "Advection - Translation";
         valuerange = "0 1";
         SolutionKnown = false;
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
   const int dim = mesh->Dimension();
   const int ne = fes->GetNE();
   const IntegrationRule *IntRuleElem = GetElementIntegrationRule(fes, NodalQuadRule);
   const IntegrationRule *IntRuleFace = GetFaceIntegrationRule(fes, NodalQuadRule);
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

      if (NodalQuadRule)
      {
         VelFace(e) = 0.;
         for (int i = 0; i < dofs.NumBdrs; i++)
         {
            for (int j = 0; j < dofs.NumFaceDofs; j++)
            {
               for (int l = 0; l < dim; l++)
               {
                  VelFace(l,i,e) += VelElem(l,dofs.BdrDofs(j,i),e);
               }
            }
         }
      }
      else
      {
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
                  velocity.Eval(vval, *facetrans->Elem2, eip[i*nqf+k]);
               }
               else
               {
                  velocity.Eval(vval, *facetrans->Elem1, eip[i*nqf+k]);
               }

               for (int l = 0; l < dim; l++)
               {
                  VelFace(l,i,e*nqf+k) = vval(l);
               }
            }
         }
      }
   }
}

void Advection::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                             int e, int k, int i) const
{
   if (i == -1) // Element terms.
   {
      VelocityVector = VelElem(e).GetColumn(k);
      VelocityVector *= u(0);
      FluxEval.SetRow(0, VelocityVector);
   }
   else
   {
      VelocityVector = VelFace(e*nqf+k).GetColumn(i);
      VelocityVector *= u(0);
      FluxEval.SetRow(0, VelocityVector);
   }
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

void Advection::SetBdrCond(const Vector &y1, Vector &y2, const Vector &normal,
                           int attr) const
{
   return;
}

void Advection::ComputeErrors(Array<double> &errors, const GridFunction &u,
                              double DomainSize, double t) const
{
   errors.SetSize(3);
   FunctionCoefficient uAnalytic(AnalyticalSolutionAdv);
   uAnalytic.SetTime(t);
   errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
   errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
}


void VelocityFunctionAdv(const Vector &x, Vector &v)
{
   double s = 1.;
   const int dim = x.Size();

   // Map to the reference [-1,1] domain.
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (ConfigAdvection.bbMin(i) + ConfigAdvection.bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center)
             / (ConfigAdvection.bbMax(i) - ConfigAdvection.bbMin(i));
      s *= ConfigAdvection.bbMax(i) - ConfigAdvection.bbMin(i);
   }

   // Scale to be normed to a full revolution.
   s = pow(s, 1./dim) * M_PI;

   switch (ConfigAdvection.ConfigNum)
   {
      case 0: // Rotation around corner.
      {
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = s*(X(1)+1.); v(1) = -s*(X(0)+1.); break;
            case 3: v(0) = s*(X(1)+1.); v(1) = -s*(X(0)+1.); v(2) = 0.0; break;
         }
         break;
      }
      case 1: // Rotation around center.
      {
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = -s*X(1); v(1) = s*X(0); break;
            case 3: v(0) = -s*X(1); v(1) = s*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 2: // Constant velocity.
      {
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = 1.0; v(1) = -0.5; break;
            case 3: v(0) = 1.0; v(1) = -0.5; v(2) = 0.25; break;
         }
         break;
      }
      default:
         MFEM_ABORT("No such test case implemented.");
   }
}

double AnalyticalSolutionAdv(const Vector &x, double t)
{
   const int dim = x.Size();

   // Map to the reference [-1,1] domain.
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (ConfigAdvection.bbMin(i) + ConfigAdvection.bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center)
             / (ConfigAdvection.bbMax(i) - ConfigAdvection.bbMin(i));
   }

   switch (ConfigAdvection.ConfigNum)
   {
      case 0:
      {
         Vector Y(dim); Y = 1.;
         X += Y;
         X *= 0.5; // Map to test case specific domain [0,1].

         double r = X.Norml2();
         double a = 0.5, b = 0.03, c = 0.1;
         return 0.25 * (1. + tanh((r+c-a)/b)) * (1. - tanh((r-c-a)/b));
      }
      case 1:
      {
         if (dim==1) { return abs(X(0) + 0.7) <= 0.15; }

         double s = 0.0225;
         double coef = (0.5/sqrt(s));
         double slit = (X(0) <= -0.05) || (X(0) >= 0.05) || (X(1) >= 0.7);
         double cone = coef * sqrt(pow(X(0), 2.) + pow(X(1) + 0.5, 2.));
         double hump = coef * sqrt(pow(X(0) + 0.5, 2.) + pow(X(1), 2.));

         return (slit && ((pow(X(0),2.) + pow(X(1)-.5,2.))<=4.*s)) ? 1. : 0.
                + (1. - cone) * (pow(X(0), 2.) + pow(X(1)+.5, 2.) <= 4.*s)
                + .25 * (1. + cos(M_PI*hump))
                * ((pow(X(0)+.5, 2.) + pow(X(1), 2.)) <= 4.*s);
      }
      case 2:
      {
         return abs(X.Norml2()) < 0.2 ? 1. : 0.;
      }
      default:
         MFEM_ABORT("No such test case implemented.");
   }
   return 0.;
}

double InitialConditionAdv(const Vector &x)
{
   return AnalyticalSolutionAdv(x, 0.);
}

void InflowFunctionAdv(const Vector &x, double t, Vector &u)
{
   u(0) = AnalyticalSolutionAdv(x, t);
}
