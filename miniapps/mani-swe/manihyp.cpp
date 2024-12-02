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

void ManifoldCoord::convertElemState(ElementTransformation &Tr,
                                     const int nrScalar, const int nrVector,
                                     const Vector &state, Vector &phys_state) const
{
   for (int i=0; i<nrScalar; i++)
   {
      phys_state[i] = state[i];
   }
   const DenseMatrix &J = Tr.Jacobian();
   mani_vec_state.UseExternalData(state.GetData()+nrScalar, dim, nrVector);
   phys_vec_state.UseExternalData(phys_state.GetData()+nrScalar, sdim, nrVector);
   Mult(J, mani_vec_state, phys_vec_state);
}

void ManifoldCoord::convertFaceState(FaceElementTransformations &Tr,
                                     const int nrScalar, const int nrVector,
                                     const Vector &stateL, const Vector &stateR,
                                     Vector &normalL, Vector &normalR,
                                     Vector &stateL_L, Vector &stateR_L,
                                     Vector &stateL_R, Vector &stateR_R) const
{
   // face Jacobian
   const DenseMatrix fJ = Tr.Jacobian();

   // element Jacobians
   // TODO: Handle boundary (Tr.Elem2==nullptr)
   const DenseMatrix &J1 = Tr.Elem1->Jacobian();
   const DenseMatrix &J2 = Tr.Elem2->Jacobian();

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

real_t ManifoldFlux::ComputeFlux(const Vector &state, ElementTransformation &Tr,
                                 DenseMatrix &flux) const
{
   phys_state.SetSize(nrScalar + coord.sdim*nrVector);
   coord.convertElemState(Tr, nrScalar, nrVector, state, phys_state);
   return org_flux.ComputeFlux(phys_state, Tr, flux);
}

real_t ManifoldFlux::ComputeNormalFluxes(const Vector &stateL,
                                         const Vector &stateR,
                                         FaceElementTransformations &Tr,
                                         Vector &normalL, Vector &normalR,
                                         Vector &stateL_L, Vector &stateR_L,
                                         Vector &fluxL_L, Vector &fluxR_L,
                                         Vector &stateL_R, Vector &stateR_R,
                                         Vector &fluxL_R, Vector &fluxR_R) const
{
   coord.convertFaceState(Tr, nrScalar, nrVector,
                          stateL, stateR,
                          normalL, normalR,
                          stateL_L, stateR_L,
                          stateL_R, stateR_R);
   real_t mcs = -mfem::infinity();
   ElementTransformation *Tr1 = Tr.Elem1;
   ElementTransformation *Tr2 = Tr.Elem2 ? Tr.Elem2 : Tr.Elem1;
   mcs = std::max(mcs, org_flux.ComputeFluxDotN(
                     stateL_L, normalL, *Tr1, fluxL_L));
   mcs = std::max(mcs, org_flux.ComputeFluxDotN(
                     stateR_L, normalL, *Tr1, fluxR_L));
   mcs = std::max(mcs, org_flux.ComputeFluxDotN(
                     stateL_R, normalR, *Tr2, fluxL_R));
   mcs = std::max(mcs, org_flux.ComputeFluxDotN(
                     stateR_R, normalR, *Tr2, fluxR_R));
   return mcs;
}

} // end of namespace mfem
