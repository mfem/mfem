#include "shallowwater.hpp"

Configuration ConfigSWE;

const double GravConst = 1.; // TODO

void AnalyticalSolutionSWE(const Vector &x, double t, Vector &u);
void InitialConditionSWE(const Vector &x, Vector &u);
void InflowFunctionSWE(const Vector &x, double t, Vector &u);

ShallowWater::ShallowWater(FiniteElementSpace *fes_, BlockVector &u_block,
									Configuration &config_)
   : HyperbolicSystem(fes_, u_block, fes_->GetMesh()->Dimension()+1, config_)
{
   ConfigSWE = config_;
	SteadyState = false;
	SolutionKnown = false; // TODO this is set temporarily until error computation works
	
	Mesh *mesh = fes->GetMesh();
	const int dim = mesh->Dimension();

   // Initialize the state.
   VectorFunctionCoefficient ic(NumEq, InitialConditionSWE);
   VectorFunctionCoefficient bc(NumEq, InflowFunctionSWE);

	if (ConfigSWE.ConfigNum == 0)
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
   }
}

void ShallowWater::EvaluateFlux(const Vector &u, DenseMatrix &f,
									     int e, int k, int i) const
{
	const int dim = u.Size() - 1;
   double H0 = 0.001;

   if (u.Size() != NumEq) { MFEM_ABORT("Invalid solution vector."); }
   if (u(0) < H0) { MFEM_ABORT("Water height too small."); }

   switch (dim)
   {
      case 1:
      {
         f(0,0) = u(1);
         f(1,0) = u(1)*u(1)/u(0) + GravConst / 2. * u(0)*u(0);
			break;
      }
      case 2:
      {
         f(0,0) = u(1);
         f(0,1) = u(2);
         f(1,0) = u(1)*u(1)/u(0) + GravConst / 2. * u(0)*u(0);
         f(1,1) = u(1)*u(2)/u(0);
         f(2,0) = u(2)*u(1)/u(0);
         f(2,1) = u(2)*u(2)/u(0) + GravConst / 2. * u(0)*u(0);
			break;
      }
      default: MFEM_ABORT("Invalid space dimensions.");
   }
}

double ShallowWater::GetWaveSpeed(const Vector &u, const Vector n, int e, int k, int i) const
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

void ShallowWater::ComputeErrors(Array<double> &errors, double DomainSize,
                              const GridFunction &u) const
{
	//TODO
}

void ShallowWater::WriteErrors(const Array<double> &errors) const
{
	//TODO
}

void AnalyticalSolutionSWE(const Vector &x, double t, Vector &u)
{
	const int dim = x.Size();
	u.SetSize(dim+1);

   // Map to the reference domain [-1,1] x [-1,1].
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (ConfigSWE.bbMin(i) + ConfigSWE.bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (ConfigSWE.bbMax(i) - ConfigSWE.bbMin(i));
   }
   
   switch (ConfigSWE.ConfigNum)
   {
      case 0: // Vorticity advection
      {
			X *= 50.; // Map to test case specific domain [-50,50] x [-50,50].
			
			if (dim == 1) 
				MFEM_ABORT("Test case only implemented in 2D.");
			
			// Parameters
			double M = .5;
			double c1 = -.04;
			double c2 = .02;
			double a = M_PI / 6.;
			double x0 = -20.;
			double y0 = -10.;
			
			double f = -c2 * ( pow(X(0) - x0 - M*t*cos(a), 2.)
								  + pow(X(1) - y0 - M*t*sin(a), 2.) );
			
			u(0) = 1.;
			u(1) = M*cos(a) + c1 * (X(1) - y0 - M*t*sin(a)) * exp(f);
			u(2) = M*sin(a) - c1 * (X(0) - x0 - M*t*cos(a)) * exp(f);
			u *= 1. - c1*c1 / (4.*c2*GravConst) * exp(2.*f);
			
			break;
		}
		default: { u = 0.; u(0) = X.Norml2() < 0.5 ? 1. : .1; }
	}
}


void InitialConditionSWE(const Vector &x, Vector &u)
{
	AnalyticalSolutionSWE(x, 0., u);
}

void InflowFunctionSWE(const Vector &x, double t, Vector &u)
{
	u.SetSize(x.Size()+1);
	AnalyticalSolutionSWE(x, t, u);
}

// void VolumeTerms::AssembleElementVolumeTerms(const int e,
// 															const DenseMatrix &uEl, DenseMatrix &VolTerms)
// {
// 	cout << "H" << endl;
// 	const FiniteElement *el = fes->GetFE(e);
// 	const int nd = el->GetDof();
// 	const int nq = ir->GetNPoints();
// 	int i, k;
// 	
// 	VolTerms.SetSize(nd, NumEq); VolTerms = 0.;
// 	DenseMatrix uQ(nq, NumEq), FluxEval, tmp4(nd, NumEq);
// 	Vector uQuad(nq), tmp(NumEq), tmp2(dim), tmp3(nd);
// 	
// 	ElementTransformation *trans = fes->GetElementTransformation(e);
// 	DenseMatrix adjJ = trans->AdjugateJacobian();
// 	
// 	for (i = 0; i < NumEq; i++)
// 	{
// 		shape.Mult(uEl.GetColumn(i), uQuad);
// 		uQ.SetCol(i, uQuad);
// 	}
// 	
// 	for (k = 0; k < nq; k++)
// 	{
// 		const IntegrationPoint &ip = ir->IntPoint(k);
// 		uQ.GetRow(k, tmp);
// 		EvalFluxFunction(tmp, FluxEval);
// 		for (i = 0; i < NumEq; i++)
// 		{
// 			FluxEval.GetRow(i, tmp);
// 			adjJ.Mult(tmp, tmp2);
// 			dShape(k).Mult(tmp2, tmp3);
// 			tmp4.SetCol(i, tmp3);
// 		}
// 		VolTerms += tmp4;
// 	}
// }

