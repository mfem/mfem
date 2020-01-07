void HyperbolicSystem::EvolveStandard(const Vector &x, Vector &y) const
{
	z = 0.;
	const int nd = fes->GetFE(0)->GetDof();
	const int nq = IntRuleElem->GetNPoints();
	Array<int> vdofs;
	Vector vec1(nd), vec2(nd);
	Vector uEl, uEval(1); // TODO Vector valued soultion.
	
	for (int e = 0; e < fes->GetNE(); e++)
	{
		fes->GetElementVDofs(e, vdofs);
		x.GetSubVector(vdofs, uEl);
		vec2 = 0.;
		for (int k = 0; k < nq; k++)
		{
			EvaluateSolution(uEl, uEval, k); // TODO optimize
			vec1 = ElemInt(e).GetColumn(k);
			DShapeEval(k).AddMult(vec1, vec2);
		}

		vec2 *= uEval(0);
		z.AddElementVector(vdofs, vec2); // TODO Vector valued soultion.
		
		// Here, the use of Bernstein elements is essential.
		// TODO call FluxLumping
	}

	InvMassMat->Mult(z, y);
}

void HyperbolicSystem::EvolveMCL(const Vector &x, Vector &y) const
{
	MFEM_ABORT("TODO.");
}
