HyperbolicSystem::HyperbolicSystem(FiniteElementSpace *fes_, Configuration &config)
											: TimeDependentOperator(fes_->GetVSize()), fes(fes_),
											  scheme(config.scheme), inflow(fes_), u(fes_),
											  z(fes_->GetVSize())
{
	const char* fecol = fes->FEColl()->Name();
	if (strncmp(fecol, "L2", 2))
		MFEM_ABORT("FiniteElementSpace must be L2 conforming (DG).");
	if (strncmp(fecol, "L2_T2", 5))
		MFEM_ABORT("Shape functions must be represented in Bernstein basis.");
	
	t = 0.;
	mesh = fes->GetMesh();
	IntRuleElem = GetElementIntegrationRule(fes);
	IntRuleFace = GetFaceIntegrationRule(fes);
	const FiniteElement *el = fes->GetFE(0);
	dim = mesh->Dimension();
	ne = mesh->GetNE();
	nd = el->GetDof();
	nqe = IntRuleElem->GetNPoints();
	nqf = IntRuleFace->GetNPoints();
	Array <int> bdrs, orientation;
	vec1.SetSize(dim);
	vec2.SetSize(nd);
	vec3.SetSize(nd);
	uElem.SetSize(nd);
	uEval.SetSize(1); // TODO Vector valued soultion.
	QuadWeightFace.SetSize(nqf);
	
	// The min/max bounds are represented as H1 functions of the same order
	// as the solution, thus having 1:1 dof correspondence inside each element.
	H1_FECollection fecBounds(max(fes->GetFE(0)->GetOrder(), 1), dim,
									  BasisType::GaussLobatto);
	FiniteElementSpace fesBounds(mesh, &fecBounds);
	dofs =  new DofInfo(fes, &fesBounds);
	
	ShapeEval.SetSize(nd,nqe);
	DShapeEval.SetSize(nd,dim,nqe);
	ShapeEvalFace.SetSize(dofs->NumBdrs, dofs->NumFaceDofs, nqf);
	Vector shape(nd);
	DenseMatrix dshape(nd,dim);
	
	for (int k = 0; k < nqe; k++)
	{
		const IntegrationPoint &ip = IntRuleElem->IntPoint(k);
		el->CalcShape(ip, shape);
		el->CalcDShape(ip, dshape);
		ShapeEval.SetCol(k, shape);
		DShapeEval(k) = dshape;
	}
	
	for (int k = 0; k < nqf; k++)
	{
		const IntegrationPoint &ip = IntRuleFace->IntPoint(k);
		QuadWeightFace(k) = ip.weight;
		
		if (dim==1)      { mesh->GetElementVertices(0, bdrs); }
		else if (dim==2) { mesh->GetElementEdges(0, bdrs, orientation); }
		else if (dim==3) { mesh->GetElementFaces(0, bdrs, orientation); }
		
		for (int i = 0; i < dofs->NumBdrs; i++)
		{
			FaceElementTransformations *facetrans = 
			mesh->GetFaceElementTransformations(bdrs[i]);
			
			IntegrationPoint eip;
			facetrans->Face->SetIntPoint(&ip);
			facetrans->Loc1.Transform(ip, eip);
			el->CalcShape(eip, shape);
			
			for (int j = 0; j < dofs->NumFaceDofs; j++)
			{
				ShapeEvalFace(i,j,k) = shape(dofs->BdrDofs(j,i));
			}
		}
	}
	
	// Compute the lumped mass matrix.
	BilinearForm ml(fes);
	ml.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
	ml.Assemble();
	ml.Finalize();
	ml.SpMat().GetDiag(LumpedMassMat);
	
	MassMat = new MassMatrixDG(fes);
	InvMassMat = new InverseMassMatrixDG(MassMat);
}

HyperbolicSystem::~HyperbolicSystem()
{
	delete dofs;
	delete MassMat;
	delete InvMassMat;
}


void HyperbolicSystem::Mult(const Vector &x, Vector &y) const
{
	switch (scheme)
	{
		case 0: // Standard Finite Element Approximation.
		{
			EvolveStandard(x, y);
			break;
		}
		case 1: // Monolithic Convex Limiting.
		{
			EvolveMCL(x, y);
			break;
		}
		default:
		{
			MFEM_ABORT("Unknown Evolution Scheme.");
		}
	}
}

double HyperbolicSystem::ConvergenceCheck(double dt, double tol)
{
	z = u;
	z -= w;
	
	double res = 0.;
	if (scheme == 0) // Standard, i.e. use consistent mass matrix.
	{
		MassMat->Mult(z, w);
		res = w.Norml2() / dt;
	}
	else // use lumped mass matrix.
	{
		for (int i = 0; i < u.Size(); i++)
		{
			res += pow(LumpedMassMat(i) * z(i), 2.);
		}
		res = sqrt(res) / dt;
	}
	
	if (res > tol)
		w = u;
	
	return res;
}

const IntegrationRule* GetElementIntegrationRule(FiniteElementSpace *fes)
{
	const FiniteElement *el = fes->GetFE(0);
	ElementTransformation *eltrans = fes->GetElementTransformation(0);
	int order = eltrans->OrderGrad(el) + eltrans->Order() + el->GetOrder();
   return &IntRules.Get(el->GetGeomType(), order);
}

// Appropriate quadrature rule for faces according to DGTraceIntegrator.
const IntegrationRule *GetFaceIntegrationRule(FiniteElementSpace *fes)
{
   int i, order;
   // Use the first mesh face and element as indicator.
   const FaceElementTransformations *Trans =
      fes->GetMesh()->GetFaceElementTransformations(0);
   const FiniteElement *el = fes->GetFE(0);

   if (Trans->Elem2No >= 0)
   {
      order = min(Trans->Elem1->OrderW(), Trans->Elem2->OrderW())
					+ 2*el->GetOrder();
   }
   else
   {
      order = Trans->Elem1->OrderW() + 2*el->GetOrder();
   }
   if (el->Space() == FunctionSpace::Pk)
   {
      order++;
   }
   return &IntRules.Get(Trans->FaceGeom, order);
}
