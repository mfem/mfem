#include "advection.hpp"

int ConfigNum;
int dim; // TODO put at better spot
Vector bbMin, bbMax;

void Velocity(const Vector &x, Vector &v);
double InitialCondition(const Vector &x);
double Inflow(const Vector &x);

Advection::Advection(FiniteElementSpace *fes_, Configuration &config)
   : HyperbolicSystem(fes_, config), fes(fes_)
{
   ConfigNum = config.ConfigNum;
   dim = fes->GetMesh()->Dimension();
	bbMin = config.bbMin;
   bbMax = config.bbMax;
}

void Advection::EvaluateFlux(const Vector &u, DenseMatrix &f) const
{
	MFEM_ABORT("For Advection this routine must not be called.");
}

void Advection::PreprocessProblem(FiniteElementSpace *fes, GridFunction &u)
{
	Mesh *mesh = fes->GetMesh();
	const IntegrationRule *IntRuleElem = GetElementIntegrationRule(fes);
	const int ne = fes->GetNE();
	const int nd = fes->GetFE(0)->GetDof();
	const int nq = IntRuleElem->GetNPoints();
	
	DenseMatrix adjJ(dim);
	DenseMatrix dshape(nd,dim);
	Vector vec1, vec2(dim);
	
	VectorFunctionCoefficient velocity(fes->GetMesh()->Dimension(), Velocity);
	ElemInt.SetSize(dim, nq, ne);
	// BdrInt.SetSize(nd, nq, ne);
	DenseMatrix VelEval, mat(dim, nq);
	
	for (int e = 0; e < ne; e++)
	{
		const FiniteElement *el = fes->GetFE(e);
		ElementTransformation *eltrans = fes->GetElementTransformation(e);
		velocity.Eval(VelEval, *eltrans, *IntRuleElem);
		
		for (int k = 0; k < IntRuleElem->GetNPoints(); k++)
		{
			const IntegrationPoint &ip = IntRuleElem->IntPoint(k);
			el->CalcDShape(ip, dshape);
			CalcAdjugate(eltrans->Jacobian(), adjJ);
			VelEval.GetColumnReference(k, vec1);
			vec1 *= ip.weight;
			adjJ.Mult(vec1, vec2);
			mat.SetCol(k, vec2);
		}
		
		ElemInt(e) = mat;
	}

	//////////////////////////////////////////////////////

   // Model parameters.
   FunctionCoefficient u0(InitialCondition);
   FunctionCoefficient inflow(Inflow);

   // Convective matrix.
   BilinearForm conv(fes);
   conv.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   conv.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   conv.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   conv.Assemble();
   conv.Finalize();
   K = conv.SpMat();

   // Inflow boundary.
   LinearForm infl(fes);
   infl.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5));
   infl.Assemble();
   b = infl;

   // Initialize solution vector.
   u.ProjectCoefficient(u0);
}

// void Advection::PostprocessProblem(const GridFunction &u, Array<double> &errors)
// {
//    if (SolutionKnown)
//    {
//       switch (ConfigNum)
//       {
//          case 0:
//          {
//             FunctionCoefficient uAnalytic(Inflow);
//             errors[0] = u.ComputeLpError(1., uAnalytic);
//             errors[1] = u.ComputeLpError(2., uAnalytic);
//             errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
//             break;
//          }
//          case 1:
//          {
//             FunctionCoefficient uAnalytic(InitialCondition);
//             errors[0] = u.ComputeLpError(1., uAnalytic);
//             errors[1] = u.ComputeLpError(2., uAnalytic);
//             errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
//             break;
//          }
//          default: MFEM_ABORT("No such test case implemented.");
//       }
//    }
// }

void Velocity(const Vector &x, Vector &v)
{
   double scale = 1.;

   // Map to the reference [-1,1] domain.
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bbMin(i) + bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (bbMax(i) - bbMin(i));
      scale *= bbMax(i) - bbMin(i);
   }

   scale = pow(scale, 1./dim) * M_PI; // Scale to be normed to half a revolution.

   switch (ConfigNum)
   {
      case 0: // Rotation around corner.
      {
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = scale*(X(1)+1.); v(1) = -scale*(X(0)+1.); break;
            case 3: v(0) = scale*(X(1)+1.); v(1) = -scale*(X(0)+1.); v(2) = 0.0; break;
         }
         break;
      }
      case 1: // Rotation around center.
      {
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = -scale*X(1); v(1) = scale*X(0); break;
            case 3: v(0) = -scale*X(1); v(1) = scale*X(0); v(2) = 0.0; break;
         }
         break;
      }
      default: { MFEM_ABORT("No such test case implemented."); }
   }
}

double InitialCondition(const Vector &x)
{
   // Map to the reference [-1,1] domain.
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bbMin(i) + bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (bbMax(i) - bbMin(i));
   }

   switch (ConfigNum)
   {
      case 0: // Smooth solution used for grid convergence studies.
      {
         Vector Y(dim); Y = 1.;
         X.Add(1., Y);
         X *= 0.5;
         double r = X.Norml2();
         double a = 0.5, b = 0.03, c = 0.1;
         return 0.25 * (1. + tanh((r+c-a)/b)) * (1. - tanh((r-c-a)/b));
      }
      case 1: // Solid body rotation.
      {
         double scale = 0.0225;
         double coef = (0.5/sqrt(scale));
         double slit = (X(0) <= -0.05) || (X(0) >= 0.05) || (X(1) >= 0.7);
         double cone = coef * sqrt(pow(X(0), 2.) + pow(X(1) + 0.5, 2.));
         double hump = coef * sqrt(pow(X(0) + 0.5, 2.) + pow(X(1), 2.));

         return (slit && ((pow(X(0),2.) + pow(X(1)-.5,2.))<=4.*scale)) ? 1. : 0.
                + (1. - cone) * (pow(X(0), 2.) + pow(X(1)+.5, 2.) <= 4.*scale)
                + .25 * (1. + cos(M_PI*hump))
                * ((pow(X(0)+.5, 2.) + pow(X(1), 2.)) <= 4.*scale);
      }
      default: { MFEM_ABORT("No such test case implemented."); }
   }
   return 0.;
}

double Inflow(const Vector &x)
{
   switch (ConfigNum)
   {
      case 0:
      {
         double r = x.Norml2();
         double a = 0.5, b = 0.03, c = 0.1;
         return 0.25 * (1. + tanh((r+c-a)/b)) * (1. - tanh((r-c-a)/b));
      }
      default: { return 0.; }
   }
}
