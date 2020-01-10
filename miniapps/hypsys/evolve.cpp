void HyperbolicSystem::EvolveStandard(const Vector &x, Vector &y) const
{
	z = 0.;
	const int dim = fes->GetMesh()->Dimension();
	const int nd = fes->GetFE(0)->GetDof();
	const int nqe = IntRuleElem->GetNPoints();
	const int nqf = IntRuleFace->GetNPoints();
	Array<int> vdofs;
	Vector vec1(dim), vec2(nd), vec3(nd);
	Vector uElem, uEval(1); // TODO Vector valued soultion.
	
	for (int e = 0; e < fes->GetNE(); e++)
	{
		fes->GetElementVDofs(e, vdofs);
		x.GetSubVector(vdofs, uElem);
		vec3 = 0.;
		
		for (int k = 0; k < nqe; k++)
		{
			EvaluateSolution(uElem, uEval, k); // TODO optimize
			ElemInt(e).GetColumn(k, vec1);
			DShapeEval(k).Mult(vec1, vec2);
			vec2 *= uEval(0);
			vec3 += vec2;
		}
		
		z.AddElementVector(vdofs, vec3); // TODO Vector valued soultion.

		// Here, the use of Bernstein elements is essential.
		for (int k = 0; k < nqf; k++)
		{
			const IntegrationPoint &ip = IntRuleFace->IntPoint(k);
			for (int i = 0; i < dofs->NumBdrs; i++)
			{
				if (BdrInt(i,k,e) >= 0)
				{
					EvaluateSolution(uElem, uEval, k, i);
				}
				else
				{
					uEval = 0.;
					for (int j = 0; j < dofs->NumFaceDofs; j++)
					{
						// TODO vector valued
						double uFace = dofs->NbrDofs(i,j,e) < 0 ?
							inflow(e*nd+dofs->BdrDofs(j,i)) :
							x(dofs->NbrDofs(i,j,e));
						uEval(0) += uFace * ShapeEvalFace(i,j,k);
					}
				}
				
				for (int j = 0; j < dofs->NumFaceDofs; j++)
				{
					z(vdofs[dofs->BdrDofs(j,i)]) -= ip.weight
						* ShapeEvalFace(i,j,k) * BdrInt(i,k,e) * uEval(0);
				}
			}
		}
	}

	InvMassMat->Mult(z, y);
}

void HyperbolicSystem::EvolveMCL(const Vector &x, Vector &y) const
{
	MFEM_ABORT("TODO.");
}
