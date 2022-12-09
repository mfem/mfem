// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "marking.hpp"

namespace mfem
{

void ShiftedFaceMarker::MarkElements(const ParGridFunction &ls_func)
{
   elemStatus.SetSize(pmesh.GetNE() + pmesh.GetNSharedFaces());
   ess_inactive.SetSize(pfes_sltn->GetVSize());
   ess_inactive = -1;
   const int max_elem_attr = (pmesh.attributes).Max();
   int activeCount = 0;
   int inactiveCount = 0;
   int cutCount = 0;

   if (!initial_marking_done) { elemStatus = SBElementType::INSIDE; }
   else { level_set_index += 1; }

   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

   // This tolerance is relevant for points that are exactly on the zero LS.
   const double eps = 1e-10;
   auto outside_of_domain = [&](double value)
   {
      if (include_cut_cell)
      {
         // Points on the zero LS are considered outside the domain.
         return (value - eps < 0.0);
      }
      else
      {
         // Points on the zero LS are considered inside the domain.
         return (value + eps < 0.0);
      }
   };

   Vector vals;
   // Check elements on the current MPI rank
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      const IntegrationRule &ir = pfes_sltn->GetFE(i)->GetNodes();
      ls_func.GetValues(i, ir, vals);
    
      int count = 0;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         if (outside_of_domain(vals(j))) { count++; }
      }

      if (count == ir.GetNPoints()) // completely outside
      {
	inactiveCount++;
	elemStatus[i] = SBElementType::OUTSIDE;
	pmesh.SetAttribute(i, max_elem_attr+1);
      }
      else if ((count > 0) && (count < ir.GetNPoints())) // partially outside
      {
	cutCount++;
	/*MFEM_VERIFY(elemStatus[i] <= SBElementType::OUTSIDE,
      	    " One element cut by multiple level-sets.");*/
	elemStatus[i] = SBElementType::CUT + level_set_index;
	if (include_cut_cell){
	  Array<int> dofs;
	  pfes_sltn->GetElementVDofs(i, dofs);
	  for (int k = 0; k < dofs.Size(); k++)
	    {
	      ess_inactive[dofs[k]] = 0;
	    }
	}
	else{
	  pmesh.SetAttribute(i, max_elem_attr+1);
	}
      }
      else // inside
      {
	activeCount++;
	Array<int> dofs;
	pfes_sltn->GetElementVDofs(i, dofs);
	for (int k = 0; k < dofs.Size(); k++)
	  {
	    ess_inactive[dofs[k]] = 0;	       
	  }
      }
   }

   pmesh.ExchangeFaceNbrNodes();

   // Check neighbors on the adjacent MPI rank
   for (int i = pmesh.GetNE(); i < pmesh.GetNE()+pmesh.GetNSharedFaces(); i++)
   {
      int shared_fnum = i-pmesh.GetNE();
      FaceElementTransformations *tr =
         pmesh.GetSharedFaceTransformations(shared_fnum);
      int Elem2NbrNo = tr->Elem2No - pmesh.GetNE();

      ElementTransformation *eltr =
         pmesh.GetFaceNbrElementTransformation(Elem2NbrNo);
      const IntegrationRule &ir =
         IntRulesLo.Get(pmesh.GetElementBaseGeometry(0), 4*eltr->OrderJ());

      const int nip = ir.GetNPoints();
      vals.SetSize(nip);
      int count = 0;
      for (int j = 0; j < nip; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         vals(j) = ls_func.GetValue(tr->Elem2No, ip);
         if (outside_of_domain(vals(j))) { count++; }
      }

      if (count == ir.GetNPoints()) // completely outside
      {
	/*         MFEM_VERIFY(elemStatus[i] != SBElementType::OUTSIDE,
		   "An element cannot be excluded by more than 1 level-set.");*/
         elemStatus[i] = SBElementType::OUTSIDE;
      }
      else if (count > 0) // partially outside
      {
        /* MFEM_VERIFY(elemStatus[i] <= SBElementType::OUTSIDE,
	   "An element cannot be cut by multiple level-sets.");*/
         elemStatus[i] = SBElementType::CUT + level_set_index;
      }
   }
   initial_marking_done = true;
   std::cout << " active elemSta " << activeCount << " cut " << cutCount << " inacive " << inactiveCount <<  std::endl;
   // Synchronize
   for (int i = 0; i < ess_inactive.Size() ; i++) { ess_inactive[i] += 1; }
   pfes_sltn->Synchronize(ess_inactive);
   for (int i = 0; i < ess_inactive.Size() ; i++) { ess_inactive[i] -= 1; }
   pmesh.SetAttributes();
}

  Array<int>& ShiftedFaceMarker::GetEss_Vdofs(){
    return ess_inactive;
  }
  Array<int>& ShiftedFaceMarker::GetElement_Status(){
    return elemStatus;
  }

}
