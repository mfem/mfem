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
      curl_u.SetSpace(scalar_fes);
   }
   else
   {
      scalar_fes = nullptr;
      curl_u.SetSpace(&fes);
   }
}

ParFiniteElementSpace &CurlEvaluator::GetCurlSpace()
{
   if (scalar_fes) { return *scalar_fes; }
   else { return fes; }
}

void CurlEvaluator::ComputeCurl(
   const ParGridFunction &u, ParGridFunction &curl_u) const
{
   // For now, on host. TODO: do this computation on device
   u.HostRead();

   const FiniteElementSpace *dom_fes = u.FESpace();
   const FiniteElementSpace *ran_fes = curl_u.FESpace();

   // AccumulateAndCountZones.
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(ran_fes->GetVSize());
   zones_per_vdof = 0;

   curl_u = 0.0;
   curl_u.HostReadWrite();

   // Local interpolation.
   int elndofs;
   Array<int> dom_dofs, ran_dofs;
   Vector vals;
   Vector loc_data;
   int dom_vdim = dom_fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;
   Vector curl;

   for (int e = 0; e < dom_fes->GetNE(); ++e)
   {
      dom_fes->GetElementVDofs(e, dom_dofs);
      ran_fes->GetElementVDofs(e, ran_dofs);

      u.GetSubVector(dom_dofs, loc_data);
      vals.SetSize(ran_dofs.Size());
      ElementTransformation *tr = dom_fes->GetElementTransformation(e);
      const FiniteElement *el = dom_fes->GetFE(e);
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
         curl_u(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication.

   // Count the zones globally.
   GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<double>(curl_u.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(curl_u.GetData());

   // Compute means.
   for (int i = 0; i < curl_u.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         curl_u(i) /= nz;
      }
   }
}

void CurlEvaluator::ComputeCurlCurl(
   const ParGridFunction &u, ParGridFunction &curl_curl_u) const
{
   ComputeCurl(u, curl_u);
   ComputeCurl(curl_u, curl_curl_u);
}

CurlEvaluator::~CurlEvaluator()
{
   delete scalar_fes;
}
