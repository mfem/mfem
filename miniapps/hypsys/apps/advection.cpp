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
		w.SetSize(fes->GetVSize());
		w = 0.;
	}
	else
	{
		SteadyState = false;
	}
	
	Mesh *mesh = fes->GetMesh();
	DenseMatrix adjJ(dim);
	Vector vec;
	VectorFunctionCoefficient velocity(dim, VelocityFunction);
	DenseMatrix VelEval, mat(dim, nqe);
	Array <int> bdrs, orientation;
	ElemInt.SetSize(dim, nqe, ne);
	BdrInt.SetSize(dofs->NumBdrs, nqf, ne);
	
	Array<IntegrationPoint> eip(nqf*dofs->NumBdrs);
	
	if (dim==1)      { mesh->GetElementVertices(0, bdrs); }
   else if (dim==2) { mesh->GetElementEdges(0, bdrs, orientation); }
   else if (dim==3) { mesh->GetElementFaces(0, bdrs, orientation); }
	
	for (int i = 0; i < dofs->NumBdrs; i++)
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
		const FiniteElement *el = fes->GetFE(e);
		ElementTransformation *eltrans = fes->GetElementTransformation(e);
		velocity.Eval(VelEval, *eltrans, *IntRuleElem);
		
		for (int k = 0; k < nqe; k++)
		{
			const IntegrationPoint &ip = IntRuleElem->IntPoint(k);
			eltrans->SetIntPoint(&ip);
			CalcAdjugate(eltrans->Jacobian(), adjJ);
			VelEval.GetColumnReference(k, vec);
			vec *= ip.weight;
			adjJ.Mult(vec, vec1);
			mat.SetCol(k, vec1);
		}
		
		ElemInt(e) = mat;
		
		if (dim==1)      { mesh->GetElementVertices(e, bdrs); }
      else if (dim==2) { mesh->GetElementEdges(e, bdrs, orientation); }
      else if (dim==3) { mesh->GetElementFaces(e, bdrs, orientation); }
		
		for (int i = 0; i < dofs->NumBdrs; i++)
		{
			Vector vval, nor(dim);
			FaceElementTransformations *facetrans
				= mesh->GetFaceElementTransformations(bdrs[i]);
			
			for (int k = 0; k < nqf; k++)
			{
				const IntegrationPoint &ip = IntRuleFace->IntPoint(k);
				facetrans->Face->SetIntPoint(&ip);
				
				if (dim == 1)
				{
					IntegrationPoint aux;
					facetrans->Loc1.Transform(ip, aux);
					nor(0) = 2.*aux.x - 1.0;
				}
				else
				{
					CalcOrtho(facetrans->Face->Jacobian(), nor);
				}
				
				if (facetrans->Elem1No != e)
				{
					facetrans->Elem2->SetIntPoint(&eip[i*nqf+k]);
					velocity.Eval(vval, *facetrans->Elem2, eip[i*nqf+k]);
					nor *= -1.;
				}
				else
				{
					facetrans->Loc1.Transform(ip, eip[i*nqf+k]);
					velocity.Eval(vval, *facetrans->Elem1, eip[i*nqf+k]);
				}
				
				nor /= nor.Norml2();
				BdrInt(i,k,e) = facetrans->Face->Weight() * (vval * nor);
			}
		}
	}
	
	// Obtain the correct inflow function.
   FunctionCoefficient Inflow(InflowFunction);
	
	if (config.ConfigNum == 0) // Convergence test: use high order projection.
   {
      L2_FECollection l2_fec(fes->GetFE(0)->GetOrder(), dim);
      FiniteElementSpace l2_fes(mesh, &l2_fec);
      GridFunction l2_inflow(&l2_fes);
      l2_inflow.ProjectCoefficient(Inflow);
      inflow.ProjectGridFunction(l2_inflow);
   }
   else
	{
		inflow.ProjectCoefficient(Inflow);
	}
	
	// Initialize solution vector.
   FunctionCoefficient u0(InitialCondition);
	u.ProjectCoefficient(u0);
	
	InitialMass = LumpedMassMat * u;
	
	// Visualization with GLVis, VisIt is currently not supported.
	{
      ofstream omesh("grid.mesh");
      omesh.precision(config.precision);
      fes->GetMesh()->Print(omesh);
      ofstream osol("initial.gf");
      osol.precision(config.precision);
      u.Save(osol);
   }
}

Advection::~Advection()
{
	double DomainSize = LumpedMassMat.Sum();
	
	cout << "Difference in solution mass: "
		  << abs(InitialMass - LumpedMassMat * u) / DomainSize << endl;
	
   if (SolutionKnown)
   {
		Array<double> errors(3);
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
         default: MFEM_ABORT("No such test case implemented.");
      }
      
      // write output
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
   
   {
      ofstream osol("final.gf");
      osol.precision(config.precision);
      u.Save(osol);
   }
}

void Advection::EvaluateFlux(const Vector &u, DenseMatrix &f) const
{
	// Due to possibly non-constant velocity a different approach is used.
	// Preprocessing for Advection is done in the Advcetion constructor.
	MFEM_ABORT("Do not call this routine for objects of type Advection.");
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
