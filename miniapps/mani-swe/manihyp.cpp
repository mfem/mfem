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
   MFEM_ASSERT(faceJ.Width() == 1, "FaceJ is not a vector");

   Vector tangent(faceJ.GetData(), sdim);
   Vector normal1(elemJ.GetData(), sdim);
   Vector normal2(elemJ.GetData() + sdim, sdim);

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
   const real_t tangent_norm = Tr.Weight();

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
      const real_t normal_comp = phys_vec*normalL/(tangent_norm*tangent_norm);
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
      const real_t normal_comp = phys_vec*normalR/(tangent_norm*tangent_norm);
      phys_vec.Add(-normal_comp, normalR).Add(normal_comp, normalL);
   }
}

const IntegrationRule &ManifoldVectorMassIntegrator::GetRule(
   const FiniteElement &trial_fe,
   const FiniteElement &test_fe,
   ElementTransformation &Trans)
{
   return IntRules.Get(Trans.GetGeometryType(),
                       trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderJ()*2 + Trans.OrderW());
}

void ManifoldVectorMassIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans,
   DenseMatrix &elmat )
{
   int dof = el.GetDof();
   dim = el.GetDim();
   sdim = Trans.GetSpaceDim();
   real_t w;

#ifdef MFEM_THREAD_SAFE
   Vector shape;
#endif
   elmat.SetSize(dof*dim);
   elmat_comp.SetSize(dof);
   elmat_comp_weighted.SetSize(dof);
   JtJ.SetSize(dim);
   shape.SetSize(dof);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, Trans);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint (&ip);

      el.CalcPhysShape(Trans, shape);
      const DenseMatrix &J = Trans.Jacobian();
      MultAtB(J, J, JtJ);

      w = Trans.Weight() * ip.weight;
      MultVVt(shape, elmat_comp);
      for (int col=0; col<dim; col++)
      {
         for (int row=0; row<dim; row++)
         {
            elmat_comp_weighted = elmat_comp;
            elmat_comp_weighted *= w*JtJ(row, col);
            elmat.AddSubMatrix(dof*row, dof*col, elmat_comp_weighted);
         }
      }
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
   ElementTransformation *Tr1 = Tr.Elem1;
   ElementTransformation *Tr2 = Tr.Elem2 ? Tr.Elem2 : Tr.Elem1;

   real_t mcs = org_flux.ComputeFluxDotN(
                   stateL_L, normalL, *Tr1, fluxL_L);
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
    coord(maniFlux.GetCoordinate()), intrule(ir), dg_fec(0, coord.dim)
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
   const int vdim = elfun.Size()/dof;
   MFEM_ASSERT((vdim - nrScalar) / dim == nrVector,
               "The number of equations and vector dimension disagree");

   elvect.SetSize(vdim*dof);
   elvect = 0.0;

   shape.SetSize(dof);
   dshape.SetSize(dof, dim);
   gshape.SetSize(dof, sdim);
   vector_gshape.SetSize(dof*dim, sdim*sdim);
   vector_gshape_comp.SetSize(dof, sdim*sdim);
   state.SetSize(vdim);
   phys_flux.SetSize(maniFlux.num_equations, sdim);
   phys_flux_scalars.SetSize(nrScalar, sdim);
   phys_flux_vectors.SetSize(nrVector*sdim, sdim);
   gradJ.SetSize(sdim, sdim);
   Vector gradJ_vectorview(gradJ.GetData(), sdim*sdim);
   HessMat.SetSize(sdim, dim, dim);
   Vector gshape_vectorview(gshape.GetData(), dof*sdim);
   Vector J_comp;

   const DenseMatrix u(elfun.GetData(), dof, vdim);
   const DenseMatrix u_scalars(elfun.GetData(), dof, nrScalar);
   const DenseMatrix u_vectors(elfun.GetData() + nrScalar*dof, dim*dof, nrVector);

   DenseMatrix divflux_scalars(elvect.GetData(), dof, nrScalar);
   DenseMatrix divflux_vectors(elvect.GetData() + nrScalar*dof, dim*dof, nrVector);

   const IntegrationRule &ir = intrule ? *intrule : GetRule(el, el, Tr);
   for (int i=0; i<ir.GetNPoints(); i++)
   {
      const IntegrationPoint & ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);

      const DenseMatrix &J = Tr.Jacobian();
      const DenseMatrix &invJ = Tr.InverseJacobian();
      const DenseMatrix &Hess = Tr.Hessian();
      for (int d1 = 0; d1 < dim; d1++)
         for (int d2 = 0; d2 < dim; d2++)
            for (int sd = 0; sd < sdim; sd++)
            {
               HessMat(sd, d1, d2) = Hess(sd, hess_map[d1*dim + d2]);
            }

      el.CalcShape(ip, shape);
      el.CalcPhysDShape(Tr, gshape);

      u.MultTranspose(shape, state);

      max_char_speed = std::max(max_char_speed, maniFlux.ComputeFlux(state, Tr,
                                                                     phys_flux));
      phys_flux.GetSubMatrix(0, nrScalar, 0, sdim, phys_flux_scalars);
      AddMult_a_ABt(ip.weight*Tr.Weight(), gshape, phys_flux_scalars,
                    divflux_scalars);

      // prepare physical vector flux for integration
      phys_flux.GetSubMatrix(nrScalar, maniFlux.num_equations, 0, sdim,
                             phys_flux_vectors);
      phys_flux_vectors.Transpose();
      phys_flux_vectors.Resize(sdim*sdim, nrVector);

      for (int d = 0; d < dim; d++)
      {
         Mult(HessMat(d), invJ, gradJ);
         MultVWt(shape, gradJ_vectorview, vector_gshape_comp);
         J_comp.SetDataAndSize(J.GetData() + sdim*d, sdim);
         vector_gshape_comp.Resize(sdim*dof, sdim);
         AddMult_a_VWt(1.0, gshape_vectorview, J_comp, vector_gshape_comp);
         vector_gshape_comp.Resize(dof, sdim*sdim);
         vector_gshape.SetSubMatrix(dof*d, 0, vector_gshape_comp);
      }
      AddMult_a(ip.weight*Tr.Weight(), vector_gshape, phys_flux_vectors,
                divflux_vectors);
      phys_flux_vectors.Resize(sdim*nrVector, sdim);
   }
}

void ManifoldHyperbolicFormIntegrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   // element info
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();
   const int dim = el1.GetDim();
   const int sdim = Tr.GetSpaceDim();
   const int nrScalar = maniFlux.GetNumScalars();
   const int nrVector = (maniFlux.num_equations - nrScalar)/sdim;
   const int vdim = elfun.Size()/(dof1 + dof2);

   elvect.SetSize((dof1+dof2)*vdim);
   elvect = 0.0;
   shape1.SetSize(dof1);
   shape2.SetSize(dof2);
   stateL.SetSize(vdim);
   stateR.SetSize(vdim);
   phys_hatFL.SetSize(maniFlux.num_equations);
   phys_hatFR.SetSize(maniFlux.num_equations);
   hatFL.SetSize(vdim);
   hatFR.SetSize(vdim);
   const DenseMatrix phys_hatFL_vectors(phys_hatFL.GetData() + nrScalar, sdim,
                                        nrVector);
   const DenseMatrix phys_hatFR_vectors(phys_hatFR.GetData() + nrScalar, sdim,
                                        nrVector);
   DenseMatrix hatFL_vectors(hatFL.GetData() + nrScalar, dim, nrVector);
   DenseMatrix hatFR_vectors(hatFR.GetData() + nrScalar, dim, nrVector);
   const DenseMatrix u1(elfun.GetData(), dof1, vdim);
   const DenseMatrix u2(elfun.GetData() + dof1*vdim, dof2, vdim);
   DenseMatrix jumpflux1(elvect.GetData(), dof1, vdim);
   DenseMatrix jumpflux2(elvect.GetData()+dof1*vdim, dof2, vdim);
   const IntegrationRule &ir = intrule ? *intrule : GetRule(el1, el2, Tr);
   for (int i=0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint & ip = ir.IntPoint(i);
      Tr.SetAllIntPoints(&ip);
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
      el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

      u1.MultTranspose(shape1, stateL);
      u2.MultTranspose(shape2, stateR);

      const DenseMatrix &J1 = Tr.Elem1->Jacobian();
      const DenseMatrix &J2 = Tr.Elem2->Jacobian();
      max_char_speed = std::max(max_char_speed, numFlux.Eval(stateL, stateR, Tr,
                                                             phys_hatFL, phys_hatFR));
      for (int j=0; j<nrScalar; j++) { hatFL[j] = phys_hatFL[j]; hatFR[j] = phys_hatFR[j]; }
      MultAtB(J1, phys_hatFL_vectors, hatFL_vectors);
      MultAtB(J2, phys_hatFR_vectors, hatFR_vectors);
      AddMult_a_VWt(-ip.weight, shape1, hatFL, jumpflux1);
      AddMult_a_VWt(+ip.weight, shape2, hatFR, jumpflux2);
   }
}

ManifoldDGHyperbolicConservationLaws::ManifoldDGHyperbolicConservationLaws(
   FiniteElementSpace &vfes,
   ManifoldHyperbolicFormIntegrator &formIntegrator,
   const int nrScalar)
   : TimeDependentOperator(vfes.GetTrueVSize()),
     vfes(vfes),
     dim(vfes.GetMesh()->Dimension()),
     sdim(vfes.GetMesh()->SpaceDimension()),
     nrScalar(nrScalar),
     nrVector((vfes.GetVDim()-nrScalar)/dim),
     formIntegrator(formIntegrator),
     z(vfes.GetTrueVSize())
{
   ComputeInvMass();
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pvfes = dynamic_cast<ParFiniteElementSpace *>(&vfes);
   if (pvfes)
   {
      parallel = true;
      comm = pvfes->GetComm();
   }
#endif

   if (parallel)
   {
#ifdef MFEM_USE_MPI
      nonlinearForm.reset(new ParNonlinearForm(pvfes));
#endif
   }
   else
   {
      nonlinearForm.reset(new NonlinearForm(&vfes));
   }
   nonlinearForm->AddDomainIntegrator(&formIntegrator);
   nonlinearForm->AddInteriorFaceIntegrator(&formIntegrator);
   nonlinearForm->UseExternalIntegrators();
}

void ManifoldDGHyperbolicConservationLaws::ComputeInvMass()
{
   InverseIntegrator inv_mass(new MassIntegrator());
   InverseIntegrator inv_vec_mass(new ManifoldVectorMassIntegrator());

   invmass.resize(vfes.GetNE());
   invmass_vec.resize(vfes.GetNE());
   for (int i=0; i<vfes.GetNE(); i++)
   {
      int dof = vfes.GetFE(i)->GetDof();
      invmass[i].SetSize(dof);
      inv_mass.AssembleElementMatrix(*vfes.GetFE(i),
                                     *vfes.GetElementTransformation(i),
                                     invmass[i]);
      invmass_vec[i].SetSize(dim*dof);
      inv_vec_mass.AssembleElementMatrix(*vfes.GetFE(i),
                                         *vfes.GetElementTransformation(i),
                                         invmass_vec[i]);
   }
}

void ManifoldDGHyperbolicConservationLaws::Mult(const Vector &x,
                                                Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   formIntegrator.ResetMaxCharSpeed();
   // 1. Apply Nonlinear form to obtain an auxiliary result
   //         z = - <F̂(u_h,n), [[v]]>_e
   //    If weak-divergence is not preassembled, we also have weak-divergence
   //         z = - <F̂(u_h,n), [[v]]>_e + (F(u_h), ∇v)
   if (force)
   {
      force->Assemble();
      nonlinearForm->AddMult(x,z);
   }
   else
   {
      nonlinearForm->Mult(x, z);
   }
   // Apply block inverse mass
   Vector zval; // z_loc, dof*num_eq

   DenseMatrix current_zmat; // view of element auxiliary result, dof x num_eq
   DenseMatrix current_ymat; // view of element result, dof x num_eq
   Array<int> vdofs;
   Array<int> vdofs_scalars;
   Array<int> vdofs_vectors;
   for (int i=0; i<vfes.GetNE(); i++)
   {
      int dof = vfes.GetFE(i)->GetDof();
      vfes.GetElementVDofs(i, vdofs);

      // Scalar mass inversion
      vdofs_scalars.MakeRef(vdofs.GetData(), nrScalar*dof, false);
      z.GetSubVector(vdofs_scalars, zval);
      current_zmat.UseExternalData(zval.GetData(), dof, nrScalar);
      current_ymat.SetSize(dof, nrScalar);
      mfem::Mult(invmass[i], current_zmat, current_ymat);
      y.SetSubVector(vdofs_scalars, current_ymat.GetData());

      // Vector mass inversion
      vdofs_vectors.MakeRef(vdofs.GetData() + nrScalar*dof, nrVector*dof*dim, false);
      z.GetSubVector(vdofs_vectors, zval);
      current_zmat.UseExternalData(zval.GetData(), dof*dim, nrVector);
      current_ymat.SetSize(dof*dim, nrVector);
      mfem::Mult(invmass_vec[i], current_zmat, current_ymat);
      y.SetSubVector(vdofs_vectors, current_ymat.GetData());
   }
   max_char_speed = formIntegrator.GetMaxCharSpeed();
   if (parallel)
   {
#ifdef MFEM_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);
#endif
   }
}

} // end of namespace mfem
