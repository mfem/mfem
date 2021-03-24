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

#include "marking.hpp"

namespace mfem
{

void ShiftedFaceMarker::MarkElements(Array<int> &elem_marker) const
{
   elem_marker.SetSize(pmesh.GetNE() + pmesh.GetNSharedFaces());
   elem_marker = SBElementType::INSIDE;

   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

   Vector vals;
   // Check elements on the current MPI rank
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      ElementTransformation *Tr = pmesh.GetElementTransformation(i);
      const IntegrationRule &ir =
         IntRulesLo.Get(pmesh.GetElementBaseGeometry(i), 4*Tr->OrderJ());
      ls_func.GetValues(i, ir, vals);

      int count = 0;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         if (vals(j) <= 0.) { count++; }
      }

      if (count == ir.GetNPoints()) // completely outside
      {
         elem_marker[i] = SBElementType::OUTSIDE;
      }
      else if (count > 0) // partially outside
      {
         elem_marker[i] = SBElementType::CUT;
      }
   }

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
         IntRulesLo.Get(pmesh.GetElementBaseGeometry(0),
                        4*eltr->OrderJ());

      const int nip = ir.GetNPoints();
      vals.SetSize(nip);
      int count = 0;
      for (int j = 0; j < nip; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         vals[j] = ls_func.GetValue(tr->Elem2No, ip);
         if (vals[j] <= 0.) { count++; }
      }

      if (count == ir.GetNPoints()) // completely outside
      {
         elem_marker[i] = SBElementType::OUTSIDE;
      }
      else if (count > 0) // partially outside
      {
         elem_marker[i] = SBElementType::CUT;
      }
   }
}

}
