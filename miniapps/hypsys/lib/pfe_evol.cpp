#include "pfe_evol.hpp"

ParFE_Evolution::ParFE_Evolution(ParFiniteElementSpace *pfes_,
                                 HyperbolicSystem *hyp_,
                                 DofInfo &dofs_, EvolutionScheme scheme_,
                                 const Vector &LumpedMassMat_)
   : FE_Evolution(pfes_, hyp_, dofs_,scheme_,
                  LumpedMassMat_), pfes(pfes_),x_gf_MPI(pfes_),
     xSizeMPI(pfes_->GetTrueVSize()), vec3(nd) { }

void ParFE_Evolution::EvaluateSolution(const Vector &x, Vector &y1, Vector &y2,
												   int e, int i, int k) const
{
// 	y1 = y2 = 0.;
// 	for (int n = 0; n < hyp->NumEq; n++)
// 	{
// 		for (int j = 0; j < dofs.NumFaceDofs; j++)
// 		{
// 			nbr = dofs.NbrDofs(i,j,e);
// 			DofInd = e*nd+dofs.BdrDofs(j,i);
// 			if (nbr < 0)
// 			{
// 				uNbr = hyp->inflow(DofInd); // TODO vector valued
// 			}
// 			else
// 			{
// 				// nbr in different MPI task? // TODO vector valued
//             uNbr = (nbr < xSizeMPI) ? x(nbr) : xMPI(nbr-xSizeMPI);
// 			}
// 			
// 			uEval(n) += x(DofInd) * ShapeEvalFace(i,j,k);
//          uNbrEval(n) += uNbr * ShapeEvalFace(i,j,k);
// 		}
// 	}
}

double ParFE_Evolution::ConvergenceCheck(double dt, double tol,
                                         const Vector &u) const
{
   z = u;
   z -= uOld;

   double res, resMPI = 0.;
   if (scheme == 0) // Standard, i.e. use consistent mass matrix.
   {
      MassMat->Mult(z, uOld);
      for (int i = 0; i < u.Size(); i++)
      {
         resMPI += uOld(i)*uOld(i);
      }
      MPI_Allreduce(&resMPI, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      res = sqrt(res) / dt;
   }
   else // use lumped mass matrix.
   {
      for (int i = 0; i < u.Size(); i++)
      {
         resMPI += pow(LumpedMassMat(i) * z(i), 2.);
      }
      MPI_Allreduce(&resMPI, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      res = sqrt(res) / dt;
   }

   uOld = u;
   return res;
}

void ParFE_Evolution::EvolveStandard(const Vector &x, Vector &y) const
{
   z = 0.;
   x_gf_MPI = x;
   x_gf_MPI.ExchangeFaceNbrData();
   Vector &xMPI = x_gf_MPI.FaceNbrData();

   for (int e = 0; e < fes->GetNE(); e++)
   {
      fes->GetElementVDofs(e, vdofs);
      x.GetSubVector(vdofs, uElem);
      vec3 = 0.;
      DenseMatrix vel = hyp->VelElem(e);

      for (int k = 0; k < nqe; k++)
      {
         ShapeEval.GetColumn(k, vec2);
         uEval(0) = uElem * vec2;

         ElemInt(nqe*e+k).Mult(vel.GetColumn(k), vec1);
         DShapeEval(k).Mult(vec1, vec2);
         vec2 *= uEval(0);
         vec3 += vec2;
      }

      z.AddElementVector(vdofs, vec3); // TODO Vector valued soultion.
		
		DenseMatrix ElemNormals = ElemNor(e); // TODO allocate

      // Here, the use of nodal basis functions is essential, i.e. shape
      // functions must vanish on faces that their node is not associated with.
      for (int i = 0; i < dofs.NumBdrs; i++)
      {
			Vector FaceNor(dim); // TODO allocate
			ElemNormals.GetColumn(i, FaceNor);
         for (int k = 0; k < nqf; k++)
         {
            Vector NumFlux(hyp->NumEq); // TODO allocate
				EvaluateSolution(x, uEval, uNbrEval, e, i, k);
				
            // ADVECTION
            for (int l = 0; l < dim; l++)
            {
               NumFlux(0) += FaceNor(l) * hyp->VelFace(l,i,e*nqf+k);
            }

            uEval = uNbrEval = 0.;

            for (int j = 0; j < dofs.NumFaceDofs; j++)
            {
               nbr = dofs.NbrDofs(i,j,e);
               if (nbr < 0)
               {
                  DofInd = e*nd+dofs.BdrDofs(j,i);
                  uNbr = hyp->inflow(DofInd);
               }
               else
               {
                  // nbr in different MPI task?
                  uNbr = (nbr < xSizeMPI) ? x(nbr) : xMPI(nbr-xSizeMPI);
               }

               uEval(0) += uElem(dofs.BdrDofs(j,i)) * ShapeEvalFace(i,j,k);
               uNbrEval(0) += uNbr * ShapeEvalFace(i,j,k);
            }

            // Lax-Friedrichs flux (equals full upwinding for Advection).
            NumFlux(0) = 0.5 * ( NumFlux(0) * (uEval(0) + uNbrEval(0))
								 + abs(NumFlux(0)) * (uEval(0) - uNbrEval(0)) );
				
				NumFlux *= BdrInt(i,k,e) * QuadWeightFace(k);

            for (int n = 0; n < hyp->NumEq; n++)
				{
					for (int j = 0; j < dofs.NumFaceDofs; j++)
					{
               	z(vdofs[n*nd+dofs.BdrDofs(j,i)]) -= ShapeEvalFace(i,j,k)
							* NumFlux(n);
					}
				}
         }
      }
   }

   InvMassMat->Mult(z, y);
}

void ParFE_Evolution::EvolveMCL(const Vector &x, Vector &y) const
{
   MFEM_ABORT("TODO.");
}
