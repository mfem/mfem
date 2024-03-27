// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem/eltrans.hpp"
#include "mfem.hpp"
#include "general/forall.hpp"

namespace mfem
{
namespace navier
{

using qoi_func_t = std::function<void(ElementTransformation &, int, int)>;

void ContactQoiEvaluator(
   ParMesh &primary_mesh,
   ParMesh &secondary_mesh,
   Array<int> &contact_bdr_marker,
   IntegrationRule &ir_face,
   qoi_func_t qoi_func,
   Vector &qoi_mem,
   int qoi_size_on_pt)
{
   const int dim = primary_mesh.Dimension();

   // Find quadrature point locations on solid side (primary, the locations where we need the QoI).
   std::vector<int> primary_element_idx;
   for (int be = 0; be < primary_mesh.GetNBE(); be++)
   {
      const int bdr_el_attr = primary_mesh.GetBdrAttribute(be);
      if (contact_bdr_marker[bdr_el_attr-1] == 0)
      {
         continue;
      }
      primary_element_idx.push_back(be);
   }

   Vector primary_element_coords(primary_element_idx.size() *
                                 ir_face.GetNPoints() * dim);
   primary_element_coords = 0.0;

   auto pec = Reshape(primary_element_coords.ReadWrite(), dim,
                      ir_face.GetNPoints(), primary_element_idx.size());

   for (int be = 0; be < primary_element_idx.size(); be++)
   {
      auto Tr = primary_mesh.GetBdrElementTransformation(primary_element_idx[be]);
      for (int qp = 0; qp < ir_face.GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir_face.IntPoint(qp);

         Vector x(dim);
         Tr->Transform(ip, x);

         for (int d = 0; d < dim; d++)
         {
            pec(d, qp, be) = x(d);
         }
      }
   }

   // Now find the elements on the secondary mesh where we can interpolate
   // those coordinates.
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(secondary_mesh);
   finder.FindPoints(primary_element_coords, Ordering::byVDIM);
   auto finder_codes = finder.GetCode();
   for (auto &code : finder_codes)
   {
      if (code != 1) {MFEM_ABORT("couldn't locate all points on boundaries")};
   }

   auto secondary_element_idx = finder.GetElem();
   const int num_requested_pts = secondary_element_idx.Size();

   qoi_mem.SetSize(num_requested_pts * qoi_size_on_pt);

   auto ref_coords = Reshape(finder.GetReferencePosition().Read(),
                             dim,
                             num_requested_pts);

   for (int i = 0; i < num_requested_pts; i++)
   {
      IntegrationPoint ip;
      ip.Set2(ref_coords(0, i), ref_coords(1, i));
      auto tr = secondary_mesh.GetElementTransformation(secondary_element_idx[i]);
      tr->SetIntPoint(&ip);
      qoi_func(*tr, i, num_requested_pts);
   }
}

}
}
