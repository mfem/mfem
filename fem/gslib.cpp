// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "gslib.hpp"

#ifdef MFEM_USE_GSLIB

// Ignore warnings from the gslib header (GCC version)
#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#include "gslib.h"

#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic pop
#endif

namespace mfem
{

FindPointsGSLIB::FindPointsGSLIB()
   : mesh(NULL), gsl_mesh(), fdata2D(NULL), fdata3D(NULL), dim(-1)
{
   gsl_comm = new comm;
#ifdef MFEM_USE_MPI
   MPI_Init(NULL, NULL);
   MPI_Comm comm = MPI_COMM_WORLD;;
   comm_init(gsl_comm, comm);
#else
   comm_init(gsl_comm, 0);
#endif
}

FindPointsGSLIB::~FindPointsGSLIB()
{
   delete gsl_comm;
}

#ifdef MFEM_USE_MPI
FindPointsGSLIB::FindPointsGSLIB(MPI_Comm _comm)
   : mesh(NULL), gsl_mesh(), fdata2D(NULL), fdata3D(NULL), dim(-1)
{
   gsl_comm = new comm;
   comm_init(gsl_comm, _comm);
}
#endif

void FindPointsGSLIB::Setup(Mesh &m, double bb_t, double newt_tol, int npt_max)
{
   MFEM_VERIFY(m.GetNodes() != NULL, "Mesh nodes are required.");

   mesh = &m;
   const GridFunction *nodes = mesh->GetNodes();
   const FiniteElementSpace *fes = nodes->FESpace();

   dim = mesh->Dimension();
   const int NE = mesh->GetNE(),
             dof_cnt = fes->GetFE(0)->GetDof(),
             pts_cnt = NE * dof_cnt;
   gsl_mesh.SetSize(dim * pts_cnt);

   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement *>(fes->GetFE(0));
   const Array<int> &dof_map = tbe->GetDofMap();

   DenseMatrix pos(dof_cnt, dim);
   Vector posV(pos.Data(), dof_cnt * dim);
   Array<int> xdofs(dof_cnt * dim);

   int pt_id = 0;
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, xdofs);
      nodes->GetSubVector(xdofs, posV);
      for (int j = 0; j < dof_cnt; j++)
      {
         for (int d = 0; d < dim; d++)
         {
            gsl_mesh(pts_cnt * d + pt_id) = pos(dof_map[j], d);
         }
         pt_id++;
      }
   }

   const unsigned dof1D = fes->GetFE(0)->GetOrder() + 1;
   if (dim == 2)
   {
      unsigned nr[2] = {dof1D, dof1D};
      unsigned mr[2] = {2*dof1D, 2*dof1D};
      double * const elx[2] = { &gsl_mesh(0), &gsl_mesh(pts_cnt) };
      fdata2D = findpts_setup_2(gsl_comm, elx, nr, NE, mr, bb_t,
                                pts_cnt, pts_cnt, npt_max, newt_tol);
   }
   else
   {
      unsigned nr[3] = {dof1D, dof1D, dof1D};
      unsigned mr[3] = {2*dof1D, 2*dof1D, 2*dof1D};
      double * const elx[3] =
      { &gsl_mesh(0), &gsl_mesh(pts_cnt), &gsl_mesh(2*pts_cnt) };
      fdata3D = findpts_setup_3(gsl_comm, elx, nr, NE, mr, bb_t,
                                pts_cnt, pts_cnt, npt_max, newt_tol);
   }
}

void FindPointsGSLIB::FindPoints(Vector &point_pos, Array<unsigned int> &codes,
                                 Array<unsigned int> &proc_ids,
                                 Array<unsigned int> &elem_ids,
                                 Vector &ref_pos, Vector &dist)
{
   const int points_cnt = point_pos.Size() / dim;
   if (dim == 2)
   {
      const double *xv_base[2];
      xv_base[0] = point_pos.GetData();
      xv_base[1] = point_pos.GetData() + points_cnt;
      unsigned xv_stride[2];
      xv_stride[0] = sizeof(double);
      xv_stride[1] = sizeof(double);
      findpts_2(codes.GetData(), sizeof(unsigned int),
                proc_ids.GetData(), sizeof(unsigned int),
                elem_ids.GetData(), sizeof(unsigned int),
                ref_pos.GetData(), sizeof(double) * dim,
                dist.GetData(), sizeof(double),
                xv_base, xv_stride, points_cnt, fdata2D);
   }
   else
   {
      const double *xv_base[3];
      xv_base[0] = point_pos.GetData();
      xv_base[1] = point_pos.GetData() + points_cnt;
      xv_base[2] = point_pos.GetData() + 2*points_cnt;
      unsigned xv_stride[3];
      xv_stride[0] = sizeof(double);
      xv_stride[1] = sizeof(double);
      xv_stride[2] = sizeof(double);
      findpts_3(codes.GetData(), sizeof(unsigned int),
                proc_ids.GetData(), sizeof(unsigned int),
                elem_ids.GetData(), sizeof(unsigned int),
                ref_pos.GetData(), sizeof(double) * dim,
                dist.GetData(), sizeof(double),
                xv_base, xv_stride, points_cnt, fdata3D);
   }
}

void FindPointsGSLIB::Interpolate(Array<unsigned int> &codes,
                                  Array<unsigned int> &proc_ids,
                                  Array<unsigned int> &elem_ids,
                                  Vector &ref_pos, const GridFunction &field_in,
                                  Vector &field_out)
{
   Vector node_vals;
   GetNodeValues(field_in, node_vals);

   const int points_cnt = ref_pos.Size() / dim;
   if (dim==2)
   {
      findpts_eval_2(field_out.GetData(), sizeof(double),
                     codes.GetData(), sizeof(unsigned int),
                     proc_ids.GetData(), sizeof(unsigned int),
                     elem_ids.GetData(), sizeof(unsigned int),
                     ref_pos.GetData(), sizeof(double) * dim,
                     points_cnt, node_vals.GetData(), fdata2D);
   }
   else
   {
      findpts_eval_3(field_out.GetData(), sizeof(double),
                     codes.GetData(), sizeof(unsigned int),
                     proc_ids.GetData(), sizeof(unsigned int),
                     elem_ids.GetData(), sizeof(unsigned int),
                     ref_pos.GetData(), sizeof(double) * dim,
                     points_cnt, node_vals.GetData(), fdata3D);
   }
}

void FindPointsGSLIB::FreeData()
{
   (dim == 2) ? findpts_free_2(fdata2D) : findpts_free_3(fdata3D);
}

void FindPointsGSLIB::GetNodeValues(const GridFunction &gf_in,
                                    Vector &node_vals)
{
   MFEM_ASSERT(gf_in.FESpace()->GetVDim() == 1, "Scalar function expected.");

   const GridFunction *nodes = mesh->GetNodes();
   const FiniteElementSpace *fes = nodes->FESpace();
   const IntegrationRule &ir = fes->GetFE(0)->GetNodes();

   const int NE = mesh->GetNE(), dof_cnt = ir.GetNPoints();
   node_vals.SetSize(NE * dof_cnt);

   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement *>(fes->GetFE(0));
   const Array<int> &dof_map = tbe->GetDofMap();

   int pt_id = 0;
   Vector vals_el;
   for (int i = 0; i < NE; i++)
   {
      gf_in.GetValues(i, ir, vals_el);
      for (int j = 0; j < dof_cnt; j++)
      {
         node_vals(pt_id++) = vals_el(dof_map[j]);
      }
   }
}

} // namespace mfem

#endif // MFEM_USE_GSLIB
