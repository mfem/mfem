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

namespace mfem
{

findpts_gslib::findpts_gslib()
   : ir(), gllmesh(), fda(NULL), fdb(NULL), dim(-1), nel(-1), qo(-1), msz(-1)
{
   comm_init(&cc, 0);
}

#ifdef MFEM_USE_MPI
findpts_gslib::findpts_gslib(MPI_Comm _comm)
   : ir(), gllmesh(), fda(NULL), fdb(NULL), dim(-1), nel(-1), qo(-1), msz(-1)
{
   comm_init(&cc, _comm);
}
#endif

void findpts_gslib::gslib_findpts_setup(Mesh &mesh, double bb_t,
                                        double newt_tol, int npt_max)
{
   MFEM_VERIFY(mesh.GetNodes() != NULL, "Mesh nodes are required.");

   const GridFunction *nodes = mesh.GetNodes();
   const FiniteElementSpace *fes = nodes->FESpace();

   ir = fes->GetFE(0)->GetNodes();
   dim = mesh.Dimension();
   nel = mesh.GetNE();
   qo = fes->GetFE(0)->GetOrder() + 1;
   int nsp = ir.GetNPoints();
   msz = nel*nsp;
   gllmesh.SetSize(dim*msz);

   int npt = nel*nsp;

   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement *>(fes->GetFE(0));
   const Array<int> &dof_map = tbe->GetDofMap();

   const int dof = fes->GetFE(0)->GetDof();
   DenseMatrix pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);
   Array<int> xdofs(dof * dim);

   int pt_id = 0;
   for (int i = 0; i < nel; i++)
   {
      fes->GetElementVDofs(i, xdofs);
      nodes->GetSubVector(xdofs, posV);
      for (int j = 0; j < nsp; j++)
      {
         for (int d = 0; d < dim; d++)
         {
            gllmesh(npt*d + pt_id) = pos(dof_map[j], d);
         }
         pt_id++;
      }
   }

   const int NE = nel, NR = qo;
   int ntot = pow(NR,dim)*NE;
   if (dim==2)
   {
      unsigned nr[2] = {NR,NR};
      unsigned mr[2] = {2*NR,2*NR};
      double *const elx[2] = {&gllmesh[0], &gllmesh[ntot]};
      this->fda=findpts_setup_2(&this->cc,elx,nr,NE,mr,bb_t,ntot,ntot,npt_max,newt_tol);
   }
   else
   {
      unsigned nr[3] = {NR,NR,NR};
      unsigned mr[3] = {2*NR,2*NR,2*NR};
      double *const elx[3] = {&gllmesh[0], &gllmesh[ntot], &gllmesh[2*ntot]};
      this->fdb=findpts_setup_3(&this->cc,elx,nr,NE,mr,bb_t,ntot,ntot,npt_max,newt_tol);
   }
}

void findpts_gslib::gslib_findpts(Vector &point_pos, Array<uint> &codes,
                                  Array<uint> &proc_ids, Array<uint> &elem_ids,
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
      findpts_2(codes.GetData(),sizeof(uint),
                proc_ids.GetData(),sizeof(uint),
                elem_ids.GetData(),sizeof(uint),
                ref_pos.GetData(),sizeof(double) * dim,
                dist.GetData(),sizeof(double),
                xv_base, xv_stride,
                points_cnt, fda);
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
      findpts_3(codes.GetData(), sizeof(uint),
                proc_ids.GetData(), sizeof(uint),
                elem_ids.GetData(), sizeof(uint),
                ref_pos.GetData(), sizeof(double) * dim,
                dist.GetData(), sizeof(double),
                xv_base, xv_stride,
                points_cnt, fdb);
   }
}

void findpts_gslib::gslib_findpts_eval(Array<uint> &codes, Array<uint> &proc_ids,
                                       Array<uint> &elem_ids, Vector &ref_pos,
                                       const GridFunction &field_in,
                                       Vector &field_out)
{
   const int points_cnt = ref_pos.Size() / dim;
   Vector node_vals(nel * ir.GetNPoints());
   GetNodeValues(field_in, node_vals);

   if (dim==2)
   {
      findpts_eval_2(field_out.GetData(), sizeof(double),
                     codes.GetData(), sizeof(uint),
                     proc_ids.GetData(), sizeof(uint),
                     elem_ids.GetData(), sizeof(uint),
                     ref_pos.GetData(), sizeof(double) * dim,
                     points_cnt, field_in.GetData(), fda);
   }
   else
   {
      findpts_eval_3(field_out.GetData(), sizeof(double),
                     codes.GetData(), sizeof(uint),
                     proc_ids.GetData(), sizeof(uint),
                     elem_ids.GetData(), sizeof(uint),
                     ref_pos.GetData(), sizeof(double) * dim,
                     points_cnt, field_in.GetData(), fdb);
   }
}

void findpts_gslib::gslib_findpts_free()
{
   (dim == 2) ? findpts_free_2(this->fda) : findpts_free_3(this->fdb);
}

void findpts_gslib::GetNodeValues(const GridFunction &gf_in, Vector &node_vals)
{
   MFEM_ASSERT(gf_in.FESpace()->GetVDim() == 1, "Scalar function expected.");

   const int nsp = ir.GetNPoints();
   Vector vals_el;

   int pt_id = 0;
   for (int i = 0; i < nel; i++)
   {
      gf_in.GetValues(i, ir, vals_el);
      for (int j = 0; j < nsp; j++) { node_vals(pt_id++) = vals_el(j); }
   }
}

} // namespace mfem

#endif // MFEM_USE_GSLIB
