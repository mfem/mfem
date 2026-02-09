// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

   const Geometry::Type geom = mesh.GetTypicalFaceGeometry();
   const int trial_order = trial_fes.GetMaxElementOrder();
   const int test_order = test_fes.GetMaxElementOrder();
   const int qorder = test_order + trial_order - 1;
   const IntegrationRule &ir = IntRule ? *IntRule : IntRules.Get(geom, qorder);
   const int nquad = ir.Size();

   Vector pa_data(nquad * nf);
   {
      const auto d_w = ir.GetWeights().Read();
      auto d_pa_data = Reshape(pa_data.Write(), nquad, nf);
      mfem::forall(nquad * nf, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int q = idx % nquad;
         const int f = idx / nquad;
         d_pa_data(q, f) = d_w[q];
      });
   }

   const FiniteElement &trial_face_el = *trial_fes.GetTypicalTraceElement();
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

   const FiniteElement &test_el = *test_fes.GetTypicalFE();
   const int n_faces_per_el = 2*dim; // assuming tensor product
   // Get all the local face maps (mapping from lexicographic face index to
   // lexicographic volume index, depending on the local face index).
   Array<int> face_maps(ndof_face * n_faces_per_el);
   for (int lf_i = 0; lf_i < n_faces_per_el; ++lf_i)
   {
      Array<int> face_map(ndof_face);
      test_el.GetFaceMap(lf_i, face_map);
      for (int i = 0; i < ndof_face; ++i)
      {
         face_maps[i + lf_i*ndof_face] = face_map[i];
      }
   }

   Array<int> face_info(nf * 4);
   {
      int fidx = 0;
      for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
      {
         Mesh::FaceInformation finfo = mesh.GetFaceInformation(f);
         if (!finfo.IsInterior() || finfo.IsNonconformingCoarse()) { continue; }
         face_info[0 + fidx*4] = finfo.element[0].local_face_id;
         face_info[1 + fidx*4] = finfo.element[0].orientation;
         face_info[2 + fidx*4] = finfo.element[1].local_face_id;
         face_info[3 + fidx*4] = finfo.element[1].orientation;
         fidx++;
      }
   }

   const int ndof_vol = test_el.GetDof();
   const auto d_face_maps = Reshape(face_maps.Read(), ndof_face, n_faces_per_el);
   const auto d_face_info = Reshape(face_info.Read(), 2, 2, nf);

   real_t *d_emat;
   if (add)
   {
      d_emat = emat.ReadWrite();
   }
   else
   {
      d_emat = emat.Write();
      emat = 0.0; // Will execute on device, since Write() sets the device flag
   }

   const auto face_mats = Reshape(mass_emat.Read(), ndof_face, ndof_face, nf);
   auto el_mats = Reshape(d_emat, ndof_vol, ndof_face, 2, nf);

   auto permute_face = [=] MFEM_HOST_DEVICE(int local_face_id, int orient,
                                            int size1d, int index)
   {
      if (dim == 2)
      {
         return internal::PermuteFace2D(local_face_id, orient, size1d, index);
      }
      else // dim == 3
      {
         return internal::PermuteFace3D(local_face_id, orient, size1d, index);
      }
   };

   auto permute_face_2 = [=] MFEM_HOST_DEVICE(int local_face_1, int local_face_2,
                                              int orient, int size1d, int index)
   {
      if (dim == 2)
      {
         return internal::PermuteFace2D(local_face_1, local_face_2, orient,
                                        size1d, index);
      }
      else // dim == 3
      {
         return internal::PermuteFace3D(local_face_1, local_face_2, orient,
                                        size1d, index);
      }
   };

   if (mesh.Conforming())
   {
      mfem::forall_3D(nf, ndof_face, ndof_face, 2, [=] MFEM_HOST_DEVICE (int f)
      {
         MFEM_FOREACH_THREAD(el_i, z, 2)
         {
            const int lf_i = d_face_info(0, el_i, f);
            const int orient = d_face_info(1, el_i, f);
            // Loop over face indices in "native ordering"
            MFEM_FOREACH_THREAD(i_lex, x, ndof_face)
            {
               // Convert to lexicographic relative to the face itself
               const int i_face = permute_face(lf_i, orient, d1d, i_lex);
               // Convert from lexicographic face DOF to volume DOF
               const int i = d_face_maps(i_lex, lf_i);
               MFEM_FOREACH_THREAD(j, y, ndof_face)
               {
                  el_mats(i, j, el_i, f) += face_mats(i_face, j, f);
               }
            }
         }
      });
   }
   else
   {
      const InterpolationManager &interp =
         test_fes.GetInterpolationManager(ElementDofOrdering::LEXICOGRAPHIC, ftype);

      auto interp_configs = interp.GetFaceInterpConfig().Read();
      const int nc_size = interp.GetNumInterpolators();
      auto d_interp = Reshape(interp.GetInterpolators().Read(),
                              ndof_face, ndof_face, nc_size);

      mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
      {
         const InterpConfig conf = interp_configs[f];
         const int master_side = conf.master_side;
         const int interp_index = conf.index;

         const int lf_0 = d_face_info(0, 0, f);

         for (int el_i = 0; el_i < 2; ++el_i)
         {
            const int lf_i = d_face_info(0, el_i, f);
            const int orient = d_face_info(1, el_i, f);

            for (int j = 0; j < ndof_face; j++)
            {
               for (int i_lex = 0; i_lex < ndof_face; i_lex++)
               {
                  real_t val = 0.0;
                  if (conf.is_non_conforming && el_i == master_side)
                  {
                     // Interpolate from el_i (coarse element) to the fine face.
                     // The mapping is given by d_interp, which uses indices
                     // relative to element 0.

                     // i0 is lexicographic relative to element 0
                     const int i0 = permute_face_2(lf_i, lf_0, orient, d1d, i_lex);

                     // k0 is lexicographic relative to element 0
                     for (int k0 = 0; k0 < ndof_face; k0++)
                     {
                        // k is relative to the face itself
                        const int k = permute_face(lf_0, orient, d1d, k0);
                        val += d_interp(k0, i0, interp_index)
                               * face_mats(k, j, f);
                     }
                  }
                  else
                  {
                     // Convert to lexicographic relative to the face itself
                     const int i_face = permute_face(lf_i, orient, d1d, i_lex);
                     val = face_mats(i_face, j, f);
                  }
                  // Convert from lexicographic face DOF to volume DOF
                  const int i = d_face_maps(i_lex, lf_i);
                  el_mats(i, j, el_i, f) += val;
               }
            }
         }
      });
   }
}

}
