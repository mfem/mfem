// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "compute_curl.hpp"
#include "../../general/forall.hpp"

using namespace mfem;
using namespace navier;

CurlEvaluator::CurlEvaluator(ParFiniteElementSpace &fes_) : fes(fes_)
{
   dim = fes_.GetMesh()->Dimension();
   if (dim == 2)
   {
      scalar_fes = new ParFiniteElementSpace(fes.GetParMesh(), fes.FEColl());
   }
   else
   {
      scalar_fes = nullptr;
   }
   u_gf.SetSpace(&fes);
   curl_u_gf.SetSpace(&GetCurlSpace());

   CountElementsPerDof();
}

void CurlEvaluator::CountElementsPerDof()
{
   // We count the number of elements containing each DOF

   // First, we fill an E-vector with ones, and then use the element restriction
   // transpose to obtain an L-vector with the number of *local* elements
   // containing each DOF.
   ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *el_restr = fes.GetElementRestriction(ordering);

   Vector evec(el_restr->Height());
   Vector lvec(el_restr->Width());
   evec.UseDevice(true);
   lvec.UseDevice(true);
   evec = 1.0;
   el_restr->MultTranspose(evec, lvec);

   // Use the group communicator reduction to obtain an L-vector with the
   // *total* number of elements (across all MPI ranks) containing each DOF.
   const GroupCommunicator &gcomm = fes.GroupComm();
   gcomm.Reduce(lvec.HostReadWrite(), GroupCommunicator::Sum);
   gcomm.Bcast(lvec.HostReadWrite());

   // Place the resulting L-vector into an array of dimension (ndof, ne),
   // where ndof is the number of DOFs per element, and ne is the number of
   // elements. *Note:* it is important that we place the DOFs in
   // *lexicographic* order in this array.
   const int ne = fes.GetNE();
   const int ndof = fes.GetFE(0)->GetDof();

   els_per_dof.SetSize(ne*ndof);

   const Table &e2dTable = fes.GetElementToDofTable();
   auto nodal_fe = dynamic_cast<const NodalFiniteElement*>(fes.GetFE(0));
   MFEM_VERIFY(nodal_fe != nullptr, "NodalFiniteElement is required.")

   // Make a copy, because otherwise we get "invalid host pointer access" when
   // the table is used later (device ptr cannot be valid for operator[]).
   Array<int> J(e2dTable.Size_of_connections());
   J.CopyFrom(e2dTable.GetJ());

   const int *d_element_map = J.Read();
   const int *d_lex = nodal_fe->GetLexicographicOrdering().Read();
   const double *d_lvec = lvec.Read();
   auto d_els_per_dof = Reshape(els_per_dof.Write(), ndof, ne);

   MFEM_FORALL(e, ne,
   {
      for (int j = 0; j < ndof; ++j)
      {
         int d = d_lex[j];
         const int sgid = d_element_map[ndof*e + d];
         const int gid = (sgid >= 0) ? sgid : -1 - sgid;
         d_els_per_dof(j, e) = d_lvec[gid];
      }
   });
}

void CurlEvaluator::ComputeCurlLegacy_(
   const Vector &u, Vector &curl_u, bool perp_grad) const
{
   // If perp_grad is true, we are computing the perpendicular gradient of a
   // scalar field (in 2D). If perp_grad is false, we are computing the curl of
   // a vector field.
   const ParFiniteElementSpace &dom_fes = perp_grad ? GetCurlSpace() : fes;
   const ParFiniteElementSpace &ran_fes = perp_grad ? fes : GetCurlSpace();

   ParGridFunction &dom_gf = perp_grad ? curl_u_gf : u_gf;
   ParGridFunction &ran_gf = perp_grad ? u_gf : curl_u_gf;

   // AccumulateAndCountZones.
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(ran_fes.GetVSize());
   zones_per_vdof = 0;

   dom_gf.Distribute(u);

   ran_gf = 0.0;
   ran_gf.HostReadWrite();

   // Local interpolation.
   int elndofs;
   Array<int> dom_dofs, ran_dofs;
   Vector vals;
   Vector loc_data;
   int dom_vdim = dom_fes.GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;
   Vector curl;

   for (int e = 0; e < dom_fes.GetNE(); ++e)
   {
      dom_fes.GetElementVDofs(e, dom_dofs);
      ran_fes.GetElementVDofs(e, ran_dofs);

      dom_gf.GetSubVector(dom_dofs, loc_data);
      vals.SetSize(ran_dofs.Size());
      ElementTransformation *tr = dom_fes.GetElementTransformation(e);
      const FiniteElement *el = dom_fes.GetFE(e);
      elndofs = el->GetDof();
      int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval and GetVectorGradientHat.
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(dom_vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, dom_vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         if (dim == 2)
         {
            if (dom_vdim == 1)
            {
               curl.SetSize(2);
               curl(0) = grad(0, 1);
               curl(1) = -grad(0, 0);
            }
            else
            {
               curl.SetSize(1);
               curl(0) = grad(1, 0) - grad(0, 1);
            }
         }
         else if (dim == 3)
         {
            curl.SetSize(3);
            curl(0) = grad(2, 1) - grad(1, 2);
            curl(1) = grad(0, 2) - grad(2, 0);
            curl(2) = grad(1, 0) - grad(0, 1);
         }

         for (int j = 0; j < curl.Size(); ++j)
         {
            vals(elndofs * j + dof) = curl(j);
         }
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < ran_dofs.Size(); j++)
      {
         int ldof = ran_dofs[j];
         ran_gf(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication.

   // Count the zones globally.
   const GroupCommunicator &gcomm = ran_fes.GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<double>(ran_gf.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(ran_gf.GetData());

   // Compute means.
   for (int i = 0; i < ran_gf.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         ran_gf(i) /= nz;
      }
   }

   ran_gf.ParallelProject(curl_u);
}

void CurlEvaluator::ComputeCurlPA_(
   const Vector &u, Vector &curl_u, bool perp_grad) const
{
   MFEM_ASSERT(!perp_grad || dim == 2, "Cannot compute perp grad in 3D");
   // If perp_grad is true, we are computing the perpendicular gradient of a
   // scalar field (in 2D). If perp_grad is false, we are computing the curl of
   // a vector field.
   const FiniteElementSpace &dom_fes = perp_grad ? GetCurlSpace() : fes;
   const FiniteElementSpace &ran_fes = perp_grad ? fes : GetCurlSpace();

   ParGridFunction &dom_gf = perp_grad ? curl_u_gf : u_gf;
   ParGridFunction &ran_gf = perp_grad ? u_gf : curl_u_gf;

   ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *dom_el_restr = dom_fes.GetElementRestriction(ordering);
   const ElementRestriction *ran_el_restr =
      dynamic_cast<const ElementRestriction*>(
         ran_fes.GetElementRestriction(ordering));
   MFEM_ASSERT(ran_el_restr != NULL, "Bad element restriction.")

   const int dom_vdim = dom_fes.GetVDim();
   const int ran_vdim = ran_fes.GetVDim();

   // Make sure internal E-vectors have correct size
   u_evec.SetSize(dom_el_restr->Height());
   du_evec.SetSize(dom_el_restr->Height()*dim);
   curl_u_evec.SetSize(ran_el_restr->Height());

   // Convert from T-vector to L-vector
   dom_gf.Distribute(u);
   // Convert from L-vector to E-vector
   dom_el_restr->Mult(dom_gf, u_evec);

   const FiniteElement &fe = *dom_fes.GetFE(0);

   // If the QuadratureInterpolator is not already constructed, create it.
   auto quad_interp = perp_grad ? scalar_quad_interp : vector_quad_interp;
   if (!quad_interp)
   {
      if (ir_lex.Size() == 0)
      {
         const IntegrationRule &ir = fe.GetNodes();
         ir_lex.SetSize(ir.Size());
         auto nodal_fe = dynamic_cast<const NodalFiniteElement*>(&fe);
         MFEM_VERIFY(nodal_fe != nullptr, "NodalFiniteElement is required.")
         const Array<int> &lex = nodal_fe->GetLexicographicOrdering();
         lex.HostRead();
         for (int i = 0; i < ir_lex.Size(); ++i)
         {
            ir_lex[i] = ir[lex[i]];
         }
      }
      quad_interp = new QuadratureInterpolator(dom_fes, ir_lex);
      // This is the default layout, setting here explicitly for clarity.
      quad_interp->SetOutputLayout(QVectorLayout::byNODES);
   }

   // Compute physical derivatives element-by-element
   quad_interp->PhysDerivatives(u_evec, du_evec);

   const int ne = fes.GetNE();
   const int ndof = fe.GetDof();

   const auto d_u = Reshape(du_evec.Read(), ndof, dom_vdim, dim, ne);
   const auto d_curl = Reshape(curl_u_evec.Write(), ndof, ran_vdim, ne);
   const auto d_els_per_dof = Reshape(els_per_dof.Read(), ndof, ne);

   // Compute the curl, scaling shared DOFs by one over the number of elements
   // containing that DOF. When we assemble (sum) from E-vector to L-vector to
   // T-vector, this will give the mean value.
   if (dim == 2)
   {
      if (perp_grad)
      {
         MFEM_FORALL(i, ne,
         {
            for (int j = 0; j < ndof; ++ j)
            {
               double a = 1.0/double(d_els_per_dof(j, i));
               d_curl(j, 0, i) = d_u(j, 0, 1, i)*a;
               d_curl(j, 1, i) = -d_u(j, 0, 0, i)*a;
            }
         });
      }
      else
      {
         MFEM_FORALL(i, ne,
         {
            for (int j = 0; j < ndof; ++ j)
            {
               double a = 1.0/double(d_els_per_dof(j, i));
               d_curl(j, 0, i) = (d_u(j, 1, 0, i) - d_u(j, 0, 1, i))*a;
            }
         });
      }
   }
   else if (dim == 3)
   {
      MFEM_FORALL(i, ne,
      {
         for (int j = 0; j < ndof; ++ j)
         {
            double a = 1.0/double(d_els_per_dof(j, i));
            d_curl(j, 0, i) = (d_u(j, 2, 1, i) - d_u(j, 1, 2, i))*a;
            d_curl(j, 1, i) = (d_u(j, 0, 2, i) - d_u(j, 2, 0, i))*a;
            d_curl(j, 2, i) = (d_u(j, 1, 0, i) - d_u(j, 0, 1, i))*a;
         }
      });
   }

   ran_el_restr->MultTranspose(curl_u_evec, ran_gf);
   ran_gf.ParallelAssemble(curl_u);
}

ParFiniteElementSpace &CurlEvaluator::GetCurlSpace()
{
   if (scalar_fes) { return *scalar_fes; }
   else { return fes; }
}

const ParFiniteElementSpace &CurlEvaluator::GetCurlSpace() const
{
   if (scalar_fes) { return *scalar_fes; }
   else { return fes; }
}

void CurlEvaluator::ComputeCurl(const Vector &u, Vector &curl_u) const
{
   if (partial_assembly) { ComputeCurlPA_(u, curl_u, false); }
   else { ComputeCurlLegacy_(u, curl_u, false); }
}

void CurlEvaluator::ComputePerpGrad(const Vector &u, Vector &perp_grad_u) const
{
   if (partial_assembly) { ComputeCurlPA_(u, perp_grad_u, true); }
   else { ComputeCurlLegacy_(u, perp_grad_u, true); }
}

void CurlEvaluator::ComputeCurlCurl(
   const Vector &u, Vector &curl_curl_u) const
{
   u_curl_tmp.SetSize(GetCurlSpace().GetTrueVSize());
   if (dim == 2)
   {
      ComputeCurl(u, u_curl_tmp);
      ComputePerpGrad(u_curl_tmp, curl_curl_u);
   }
   else if (dim == 3)
   {
      ComputeCurl(u, u_curl_tmp);
      ComputeCurl(u_curl_tmp, curl_curl_u);
   }
}

CurlEvaluator::~CurlEvaluator()
{
   delete scalar_quad_interp;
   delete vector_quad_interp;
   delete scalar_fes;
}
