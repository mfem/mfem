#include "manihyp.hpp"

namespace mfem
{

void sphere(const Vector &x, Vector &y, const real_t r)
{
   y = x; y *= r/std::sqrt(y*y);
}

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
   const DenseMatrix &J1 = Tr.Elem1->Jacobian();
   const DenseMatrix &J2 = Tr.Elem2 ? Tr.Elem2->Jacobian() : Tr.Elem1->Jacobian();

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

ManifoldHyperbolicFormIntegrator::ManifoldHyperbolicFormIntegrator(
   const ManifoldNumericalFlux &flux, const IntegrationRule *ir)
   :numFlux(flux), maniFlux(flux.GetManifoldFluxFunction()),
    coord(maniFlux.GetCoordinate()), intrule(ir)
{
   switch (maniFlux.GetCoordinate().dim)
   {
      case 2:
      {
         hess_map.SetSize(4);
         hess_map[0] = 0;
         hess_map[1] = 1;
         hess_map[2] = 1;
         hess_map[3] = 2;
         break;
      }
      default:
      {
         MFEM_ABORT("Only support 2D manifold");
      }
   }
}

void ManifoldHyperbolicFormIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr,
   const Vector &elfun, Vector &elvect)
{
   // element info
   const int dof = el.GetDof();
   const int dim = el.GetDim();
   const int sdim = Tr.GetSpaceDim();
   const int nrScalar = maniFlux.GetNumScalars();
   const int nrVector = (maniFlux.num_equations - nrScalar)/sdim;
   const int vdim = elfun.Size();
   MFEM_ASSERT((vdim - nrScalar) / dim == nrVector,
               "The number of equations and vector dimension disagree");

   //TODO: Consider ordering of basis..
   // J1 (phi1, phi2, ...), J2 (phi1, phi2, ...) or
   // (J1, J2) phi1, (J1, J2) phi2, ... ?
   MFEM_ASSERT(nrVector == 1, "More than one vector is not yet supported.")


   // Local basis function info

   // TODO: handle MFEM_THREAD_SAFE.
   // Since we are only using MPI for parallel,
   // we do not have to worry about this for now
   shape.SetSize(dof);
   dshape.SetSize(dof, dim);
   gshape.SetSize(dof, sdim);
   vector_gshape.SetSize(dof*dim, sdim);
   hess_shape.SetSize(dof, dim*(dim+1)/2);
   Hess.SetSize(dim*(dim+1)/2, sdim);
   HessMat.SetSize(sdim, dim, dim);
   gradJ.SetSize(sdim, sdim);
   Vector gshape_i;

   elvect.SetSize(vdim * dof); elvect=0.0;
   state.SetSize(vdim);
   phys_state.SetSize(maniFlux.num_equations);
   phys_flux.SetSize(maniFlux.num_equations, sdim);
   Vector phys_flux_vector_vectorview(phys_flux.GetData(), sdim*sdim);
   const IntegrationRule &ir = intrule ? *intrule : GetRule(el, el, Tr);


   const int num_equations = elfun.Size() / dof;
   const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);
   DenseMatrix elvec_mat(elvect.GetData(), dof, num_equations);

   const FiniteElementSpace * nodal_fes = Tr.mesh->GetNodalFESpace();
   const FiniteElement &nodal_el = *nodal_fes->GetFE(Tr.ElementNo);
   const GridFunction * nodes = Tr.mesh->GetNodes();
   nodes->GetElementDofValues(Tr.ElementNo, x_nodes);
   DenseMatrix x_nodes_mat(x_nodes.GetData(), dof, sdim);

   const DenseMatrix elfun_scalars(elfun.GetData(), dof, nrScalar);
   const DenseMatrix elfun_vectors(elfun.GetData() + dof*nrScalar, dof*dim,
                                   nrVector);
   DenseMatrix elmat_scalars(elvect.GetData(), dof, nrScalar);
   DenseTensor elmat_vectors(elvect.GetData() + dof*nrScalar, dof, dim, nrVector);
   Vector mani_scalars(state.GetData(), nrScalar);
   Vector phys_scalars(phys_state.GetData(), nrScalar);
   DenseMatrix mani_vec(state.GetData() + nrScalar, dim, nrVector);
   DenseMatrix phys_vec(phys_state.GetData() + nrScalar, sdim, nrVector);

   for (int i=0; i<ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);

      // local mapping information
      const DenseMatrix &J = Tr.Jacobian();
      const DenseMatrix &adjJ = Tr.AdjugateJacobian();
      nodal_el.CalcHessian(ip, hess_shape);
      MultAtB(hess_shape, x_nodes_mat, Hess);
      for (int d1=0; d1<dim; d1++)
         for (int d2=0; d2<dim; d2++)
            for (int sd=0; sd<sdim; sd++)
            {
               HessMat(sd, d1, d2) = Hess(hess_map[d1*dim + d2], sd);
            }

      // basis information
      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);
      Mult(dshape, adjJ, gshape);

      // state at current quadrature points
      elfun_mat.MultTranspose(shape, state);
      // Convert manifold state to a physical state
      // NOTE: Not sure whether this is actually needed
      // because ComputeFlux will handle this to evaluate the flux.
      phys_scalars = mani_scalars;
      Mult(J, mani_vec, phys_vec);

      // compute flux and update max_char_speed
      max_char_speed = std::max(max_char_speed,
                                maniFlux.ComputeFlux(state, Tr, phys_flux));

      // Integrate
      phys_flux.GetSubMatrix(0, nrScalar, 0, sdim, phys_flux_scalars);
      AddMult_a_ABt(ip.weight, gshape, phys_flux_scalars, elmat_scalars);

      phys_flux.GetSubMatrix(nrScalar, maniFlux.num_equations, 0, sdim,
                             phys_flux_vectors);
      phys_flux_vectors.Transpose();
      for (int d=0; d<dim; d++)
      {
         Mult(HessMat(d), adjJ, gradJ);
      }
   }
}

void ManifoldHyperbolicFormIntegrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
}

const IntegrationRule &ManifoldHyperbolicFormIntegrator::GetRule(
   const FiniteElement &trial_fe,
   const FiniteElement &test_fe,
   ElementTransformation &Trans)
{
   int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
   return IntRules.Get(trial_fe.GetGeomType(), order);
}

} // end of namespace mfem
