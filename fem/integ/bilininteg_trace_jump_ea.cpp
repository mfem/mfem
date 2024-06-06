// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../general/forall.hpp"
#include "../fe/face_map_utils.hpp"
#include "../bilininteg.hpp"
#include "bilininteg_mass_kernels.hpp"

namespace mfem
{

void NormalTraceJumpIntegrator::AssembleEAInteriorFaces(
   const FiniteElementSpace &trial_fes,
   const FiniteElementSpace &test_fes,
   Vector &emat,
   const bool add)
{
   Mesh &mesh = *trial_fes.GetMesh();
   const int dim = mesh.Dimension();
   const FaceType ftype = FaceType::Interior;
   const int nf = mesh.GetNFbyType(ftype);

   const Geometry::Type geom = mesh.GetFaceGeometry(0);
   const int trial_order = trial_fes.GetMaxElementOrder();
   const int test_order = test_fes.GetMaxElementOrder();
   const int qorder = test_order + trial_order - 1;
   const IntegrationRule &ir = IntRule ? *IntRule : IntRules.Get(geom, qorder);
   const int nquad = ir.Size();

   // const auto face_geom =
   //    mesh.GetFaceGeometricFactors(ir, FaceGeometricFactors::DETERMINANTS, ftype);
   Vector pa_data(nquad * nf);
   for (int f = 0; f < nf; ++f)
   {
      for (int q = 0; q < nquad; ++q)
      {
         pa_data[q + f*nquad] = ir[q].weight;
      }
   }

   const FiniteElement &trial_face_el = *trial_fes.GetFaceElement(0);
   const auto maps = &trial_face_el.GetDofToQuad(ir, DofToQuad::TENSOR);
   const int ndof_face = trial_face_el.GetDof();

   const Array<real_t> &B = maps->B;
   const int d1d = maps->ndof;
   const int q1d = maps->nqpt;

   Vector mass_emat(ndof_face*ndof_face*nf);

   // Note: dim is the element dimension, and we integrate over the faces (one
   // dimension less)
   if (dim == 2)
   {
      internal::EAMassAssemble1D(nf, B, pa_data, mass_emat, false, d1d, q1d);
   }
   else if (dim == 3)
   {
      internal::EAMassAssemble2D(nf, B, pa_data, mass_emat, false, d1d, q1d);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }

   const FiniteElement &test_el = *test_fes.GetFE(0);
   const int ndof_vol = test_el.GetDof();
   const auto *tbe = dynamic_cast<const TensorBasisElement*>(&test_el);
   MFEM_VERIFY(tbe, "");
   const Array<int> &dof_map = tbe->GetDofMap();

   const auto face_mats = Reshape(mass_emat.Read(), ndof_face, ndof_face, nf);
   auto el_mats = Reshape(emat.ReadWrite(), ndof_vol, ndof_face, 2, nf);

   if (!add)
   {
      emat = 0.0;
   }

   int fidx = 0;
   for (int f = 0; f < mesh.GetNumFaces(); ++f)
   {
      Mesh::FaceInformation finfo = mesh.GetFaceInformation(f);
      if (!finfo.IsInterior()) { continue; }
      for (int el_i = 0; el_i < 2; ++el_i)
      {
         const int orient = finfo.element[el_i].orientation;
         const int lf_i = finfo.element[el_i].local_face_id;

         Array<int> face_map(ndof_face);
         test_el.GetFaceMap(lf_i, face_map);

         for (int i_nat = 0; i_nat < ndof_face; ++i_nat)
         {
            const int i_lex = internal::ToLexOrdering2D(lf_i, ndof_face, i_nat);
            const int i_face = internal::PermuteFace2D(lf_i, 0, orient, ndof_face, i_lex);
            // Lexicographic volume DOF
            const int vol_lex = face_map[i_lex];
            // Convert to MFEM ordering
            const int i_s = dof_map[vol_lex];
            const int i = (i_s >= 0) ? i_s : -1 - i_s;
            for (int j = 0; j < ndof_face; ++j)
            {
               el_mats(i, j, el_i, fidx) += face_mats(i_face, j, fidx);
            }
         }
      }
      fidx++;
   }
}

}
