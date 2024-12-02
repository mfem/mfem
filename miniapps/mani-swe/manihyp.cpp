#include "manihyp.hpp"

namespace mfem
{

void CalcOrtho(const DenseMatrix &faceJ, const DenseMatrix &elemJ, Vector &n)
{
   const int sdim = faceJ.Height();
   const int dim = elemJ.Width();
   MFEM_ASSERT(sdim == 3 && dim == 2, "Only supports 2D manifold in 3D");

   Vector tangent(faceJ.GetData(), sdim);
   Vector normal1(elemJ.GetData(), sdim);
   Vector normal2(elemJ.GetData()+sdim, sdim);

   Vector surfaceNormal(sdim);
   normal1.cross3D(normal2, surfaceNormal);

   tangent.cross3D(surfaceNormal, n);
   n *= std::sqrt((tangent*tangent)/(n*n));
}

void ManifoldCoord::convertElemState(ElementTransformation &el,
                                     const int nrScalar, const int nrVector,
                                     const Vector &state, Vector &phys_state)
{
   for (int i=0; i<nrScalar; i++)
   {
      phys_state[i] = state[i];
   }
   const DenseMatrix &J = el.Jacobian();
   mani_vec_state.UseExternalData(state.GetData()+nrScalar, dim, nrVector);
   phys_vec_state.UseExternalData(phys_state.GetData()+nrScalar, sdim, nrVector);
   Mult(J, mani_vec_state, phys_vec_state);
}

void ManifoldCoord::convertFaceState(FaceElementTransformations &el,
                                     const int nrScalar, const int nrVector,
                                     const Vector &stateL, const Vector &stateR,
                                     Vector &normalL, Vector &normalR,
                                     Vector &stateL_L, Vector &stateR_L,
                                     Vector &stateL_R, Vector &stateR_R)
{
   // face Jacobian
   const DenseMatrix fJ = el.Jacobian();

   // element Jacobians
   const DenseMatrix &J1 = el.Elem1->Jacobian();
   const DenseMatrix &J2 = el.Elem2->Jacobian();

   normal_comp.SetSize(nrVector);

   // Compute interface normal vectors at each element
   CalcOrtho(fJ, J1, normalL);
   CalcOrtho(fJ, J2, normalR);
   real_t tangent_norm = std::sqrt(normalL*normalL);

   // copy scalar states
   for (int i=0; i<nrScalar; i++)
   {
      stateL_R[i] = stateL[i];
      stateR_L[i] = stateR[i];
   }

   // Convert Left element vector states to physical states
   mani_vec_state.UseExternalData(stateL.GetData() + nrScalar, dim, nrVector);
   phys_vec_state.UseExternalData(stateL_R.GetData() + nrScalar, sdim, nrVector);
   Mult(J1, mani_vec_state, phys_vec_state);
   stateL_L = stateL_R; // Left to Left done
   for (int i=0; i<nrVector; i++)
   {
      phys_vec_state.GetColumnReference(i, phys_vec);
      const real_t normal_comp = phys_vec*normalL/tangent_norm;
      phys_vec.Add(-normal_comp, normalL).Add(normal_comp, normalR);
   }

   // Convert Right element vector states to physical states
   mani_vec_state.UseExternalData(stateR.GetData() + nrScalar, dim, nrVector);
   phys_vec_state.UseExternalData(stateR_L.GetData() + nrScalar, sdim, nrVector);
   Mult(J2, mani_vec_state, phys_vec_state);
   stateR_R = stateR_L;
   for (int i=0; i<nrVector; i++)
   {
      phys_vec_state.GetColumnReference(i, phys_vec);
      const real_t normal_comp = phys_vec*normalR;
      phys_vec.Add(-normal_comp, normalR).Add(normal_comp, normalL);
   }
}

} // end of namespace mfem
