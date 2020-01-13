void HyperbolicSystem::EvolveStandard(const Vector &x, Vector &y) const
{
	z = 0.;
	
	for (int e = 0; e < fes->GetNE(); e++)
	{
		fes->GetElementVDofs(e, vdofs);
		x.GetSubVector(vdofs, uElem);
		vec3 = 0.;
		
		for (int k = 0; k < nqe; k++)
		{
			ShapeEval.GetColumn(k, vec2);
			uEval(0) = uElem * vec2;
			
			ElemInt(e).GetColumn(k, vec1);
			DShapeEval(k).Mult(vec1, vec2);
			vec2 *= uEval(0);
			vec3 += vec2;
		}
		
		z.AddElementVector(vdofs, vec3); // TODO Vector valued soultion.

		// Here, the use of Bernstein elements is essential.
		for (int i = 0; i < dofs->NumBdrs; i++)
		{
			for (int k = 0; k < nqf; k++)
			{
				uEval = 0.;
				if (BdrInt(i,k,e) >= 0)
				{
					for (int j = 0; j < dofs->NumFaceDofs; j++)
					{
						uEval(0) += uElem(dofs->BdrDofs(j,i)) * ShapeEvalFace(i,j,k);
					}
				}
				else
				{
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
					z(vdofs[dofs->BdrDofs(j,i)]) -= QuadWeightFace(k)
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
