#include "advection.hpp"

Configuration config;

void VelocityFunction(const Vector &x, Vector &v);
double InitialCondition(const Vector &x);
double InflowFunction(const Vector &x);

Advection::Advection(FiniteElementSpace *fes_, Configuration &config_)
							: HyperbolicSystem(fes_, config_)
{	
	config = config_;
	
	if (config.ConfigNum == 0)
	{
		SteadyState = true;
	}
	else
	{
		SteadyState = false;
	}
	
	Mesh *mesh = fes->GetMesh();
	const int dim = mesh->Dimension();
	const int ne = fes->GetNE();
	const IntegrationRule *IntRuleElem = GetElementIntegrationRule(fes);
	const IntegrationRule *IntRuleFace = GetFaceIntegrationRule(fes);
	const int nqe = IntRuleElem->GetNPoints();
	const int nqf = IntRuleFace->GetNPoints();
	Vector vec, vval;
	DenseMatrix VelEval, mat(dim, nqe);
	
	int NumBdrs;
	const FiniteElement *el = fes->GetFE(0);
	switch (el->GetGeomType())
	{
		case Geometry::SEGMENT: NumBdrs = 2; break;
		case Geometry::TRIANGLE: NumBdrs = 3; break;
		case Geometry::SQUARE:
		case Geometry::TETRAHEDRON: NumBdrs = 4; break;
		case Geometry::CUBE: NumBdrs = 6; break;
		default:
			MFEM_ABORT("Invalid Geometry type.");
	}
	
	VectorFunctionCoefficient velocity(dim, VelocityFunction);
	VelElem.SetSize(dim, nqe, ne);
	VelFace.SetSize(dim, NumBdrs, ne*nqf);
	
	Array<int> bdrs, orientation;
	Array<IntegrationPoint> eip(nqf*NumBdrs);
	
	if (dim==1)      { mesh->GetElementVertices(0, bdrs); }
   else if (dim==2) { mesh->GetElementEdges(0, bdrs, orientation); }
   else if (dim==3) { mesh->GetElementFaces(0, bdrs, orientation); }
	
	for (int i = 0; i < NumBdrs; i++)
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
		
		for (int i = 0; i < NumBdrs; i++)
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
	
	// Obtain the correct inflow function.
   FunctionCoefficient bc(InflowFunction);
	
	if (config.ConfigNum == 0) // Convergence test: use high order projection.
   {
      L2_FECollection l2_fec(fes->GetFE(0)->GetOrder(), dim);
      FiniteElementSpace l2_fes(mesh, &l2_fec);
      GridFunction l2_inflow(&l2_fes);
      l2_inflow.ProjectCoefficient(bc);
      inflow.ProjectGridFunction(l2_inflow);
   }
   else
	{
		inflow.ProjectCoefficient(bc);
	}
	
	// Initialize solution vector.
   FunctionCoefficient ic(InitialCondition);
	u0.ProjectCoefficient(ic);
}

void Advection::EvaluateFlux(const Vector &u, DenseMatrix &f) const
{
	// Due to possibly non-constant velocity a different approach is used.
	// Preprocessing for Advection is done in the Advcetion constructor.
	MFEM_ABORT("Do not call this routine for objects of type Advection.");
}

void Advection::ComputeErrors(Array<double> &errors, double DomainSize, const GridFunction &u) const
{
	errors.SetSize(3);
	switch (config.ConfigNum)
	{
		case 0:
		{
			FunctionCoefficient uAnalytic(InflowFunction);
			errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
			errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
			errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic) / DomainSize;
			break;
		}
		case 1:
		{
			FunctionCoefficient uAnalytic(InitialCondition);
			errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
			errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
			errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic) / DomainSize;
			break;
		}
		default: MFEM_ABORT("Solution is not known analytically for this testcase.");
	}
}
	
void Advection::WriteErrors(const Array<double> &errors) const
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


void VelocityFunction(const Vector &x, Vector &v)
{
   double s = 1.;
	const int dim = x.Size();
	
   // Map to the reference [-1,1] domain.
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (config.bbMin(i) + config.bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (config.bbMax(i) - config.bbMin(i));
      s *= config.bbMax(i) - config.bbMin(i);
   }

   // Scale to be normed to a full revolution.
   s = pow(s, 1./dim) * M_PI;

   switch (config.ConfigNum)
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
      default: { MFEM_ABORT("No such test case implemented."); }
   }
}

double InitialCondition(const Vector &x)
{
	const int dim = x.Size();
	
   // Map to the reference [-1,1] domain.
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (config.bbMin(i) + config.bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (config.bbMax(i) - config.bbMin(i));
   }

   switch (config.ConfigNum)
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
			if (dim==1) return abs(X(0) + 0.7) <= 0.15;
			
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
      default: { MFEM_ABORT("No such test case implemented."); }
   }
   return 0.;
}

double InflowFunction(const Vector &x)
{
   switch (config.ConfigNum)
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
