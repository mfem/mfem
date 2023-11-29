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

#include "gslib.hpp"
#include "../general/forall.hpp"
#include "geom.hpp"

#ifdef MFEM_USE_GSLIB

// Ignore warnings from the gslib header (GCC version)
#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2
#define dlong int
#define dfloat double

// External GSLIB header (the MFEM header is gslib.hpp)
namespace gslib
{
#include "gslib.h"
extern "C" {
   struct hash_data_3
   {
      ulong hash_n;
      struct dbl_range bnd[3];
      double fac[3];
      uint *offset;
   };

   struct findpts_dummy_ms_data
   {
      unsigned int *nsid;
      double       *distfint;
   };

   struct findpts_data_3
   {
      struct crystal cr;
      struct findpts_local_data_3 local;
      struct hash_data_3 hash;
      struct array savpt;
      struct findpts_dummy_ms_data fdms;
      uint   fevsetup;
   };

   uint hash_opt_size_3(struct findpts_local_hash_data_3 *p,
                        const struct obbox_3 *const obb,
                        const uint nel,
                        const uint max_size);
} //extern C

} //namespace gslib

#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic pop
#endif

namespace mfem
{
dlong getHashSize(const struct gslib::findpts_data_3 *fd,
                  dlong nel, dlong max_hash_size)
{
   const gslib::findpts_local_data_3 *fd_local = &fd->local;
   auto hash_data_copy = fd_local->hd;
   return hash_opt_size_3(&hash_data_copy, fd_local->obb, nel, max_hash_size);
}

FindPointsGSLIB::FindPointsGSLIB()
   : mesh(NULL),
     fec_map_lin(NULL),
     fdata2D(NULL), fdataD(NULL), cr(NULL), gsl_comm(NULL),
     dim(-1), points_cnt(0), setupflag(false), default_interp_value(0),
     avgtype(AvgType::ARITHMETIC), bdr_tol(1e-8)
{
   mesh_split.SetSize(4);
   ir_split.SetSize(4);
   fes_rst_map.SetSize(4);
   gf_rst_map.SetSize(4);
   for (int i = 0; i < mesh_split.Size(); i++)
   {
      mesh_split[i] = NULL;
      ir_split[i] = NULL;
      fes_rst_map[i] = NULL;
      gf_rst_map[i] = NULL;
   }

   gsl_comm = new gslib::comm;
   cr       = new gslib::crystal;
#ifdef MFEM_USE_MPI
   int initialized;
   MPI_Initialized(&initialized);
   if (!initialized) { MPI_Init(NULL, NULL); }
   MPI_Comm comm = MPI_COMM_WORLD;
   comm_init(gsl_comm, comm);
#else
   comm_init(gsl_comm, 0);
#endif
}

FindPointsGSLIB::~FindPointsGSLIB()
{
   delete gsl_comm;
   delete cr;
   for (int i = 0; i < 4; i++)
   {
      if (mesh_split[i]) { delete mesh_split[i]; mesh_split[i] = NULL; }
      if (ir_split[i]) { delete ir_split[i]; ir_split[i] = NULL; }
      if (fes_rst_map[i]) { delete fes_rst_map[i]; fes_rst_map[i] = NULL; }
      if (gf_rst_map[i]) { delete gf_rst_map[i]; gf_rst_map[i] = NULL; }
   }
   if (fec_map_lin) { delete fec_map_lin; fec_map_lin = NULL; }
}

#ifdef MFEM_USE_MPI
FindPointsGSLIB::FindPointsGSLIB(MPI_Comm comm_)
   : mesh(NULL),
     fec_map_lin(NULL),
     fdata2D(NULL), fdataD(NULL), cr(NULL), gsl_comm(NULL),
     dim(-1), points_cnt(0), setupflag(false), default_interp_value(0),
     avgtype(AvgType::ARITHMETIC), bdr_tol(1e-8)
{
   mesh_split.SetSize(4);
   ir_split.SetSize(4);
   fes_rst_map.SetSize(4);
   gf_rst_map.SetSize(4);
   for (int i = 0; i < mesh_split.Size(); i++)
   {
      mesh_split[i] = NULL;
      ir_split[i] = NULL;
      fes_rst_map[i] = NULL;
      gf_rst_map[i] = NULL;
   }

   gsl_comm = new gslib::comm;
   cr      = new gslib::crystal;
   comm_init(gsl_comm, comm_);
}
#endif

void FindPointsGSLIB::Setup(Mesh &m, const double bb_t, const double newt_tol,
                            const int npt_max)
{
   MFEM_VERIFY(m.GetNodes() != NULL, "Mesh nodes are required.");
   const int meshOrder = m.GetNodes()->FESpace()->GetMaxElementOrder();

   // call FreeData if FindPointsGSLIB::Setup has been called already
   if (setupflag) { FreeData(); }

   crystal_init(cr, gsl_comm);
   mesh = &m;
   dim  = mesh->Dimension();
   unsigned dof1D = meshOrder + 1;

   setupSW.Clear();
   setupSW.Start();
   SetupSplitMeshes();
   if (dim == 2)
   {
      if (ir_split[0]) { delete ir_split[0]; ir_split[0] = NULL; }
      ir_split[0] = new IntegrationRule(3*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[0], ir_split[0], meshOrder);

      if (ir_split[1]) { delete ir_split[1]; ir_split[1] = NULL; }
      ir_split[1] = new IntegrationRule(pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[1], ir_split[1], meshOrder);
   }
   else if (dim == 3)
   {
      if (ir_split[0]) { delete ir_split[0]; ir_split[0] = NULL; }
      ir_split[0] = new IntegrationRule(pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[0], ir_split[0], meshOrder);

      if (ir_split[1]) { delete ir_split[1]; ir_split[1] = NULL; }
      ir_split[1] = new IntegrationRule(4*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[1], ir_split[1], meshOrder);

      if (ir_split[2]) { delete ir_split[2]; ir_split[2] = NULL; }
      ir_split[2] = new IntegrationRule(3*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[2], ir_split[2], meshOrder);

      if (ir_split[3]) { delete ir_split[3]; ir_split[3] = NULL; }
      ir_split[3] = new IntegrationRule(8*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[3], ir_split[3], meshOrder);
   }
   setupSW.Stop();
   setup_split_time = setupSW.RealTime();

   setupSW.Clear();
   setupSW.Start();
   GetNodalValues(mesh->GetNodes(), gsl_mesh);
   setupSW.Stop();
   setup_nodalmapping_time = setupSW.RealTime();

   mesh_points_cnt = gsl_mesh.Size()/dim;

   DEV.local_hash_size = mesh_points_cnt;
   DEV.dof1d = (int)dof1D;

   setupSW.Clear();
   setupSW.Start();
   if (dim == 2)
   {
      unsigned nr[2] = { dof1D, dof1D };
      unsigned mr[2] = { 2*dof1D, 2*dof1D };
      double * const elx[2] = { &gsl_mesh(0), &gsl_mesh(mesh_points_cnt) };
      fdata2D = findpts_setup_2(gsl_comm, elx, nr, NE_split_total, mr, bb_t,
                                DEV.local_hash_size,
                                mesh_points_cnt, npt_max, newt_tol);
   }
   else
   {
      unsigned nr[3] = { dof1D, dof1D, dof1D };
      unsigned mr[3] = { 2*dof1D, 2*dof1D, 2*dof1D };
      double * const elx[3] =
      { &gsl_mesh(0), &gsl_mesh(mesh_points_cnt), &gsl_mesh(2*mesh_points_cnt) };
      fdataD = findpts_setup_3(gsl_comm, elx, nr, NE_split_total, mr, bb_t,
                               DEV.local_hash_size,
                               mesh_points_cnt, npt_max, newt_tol);
   }
   setupSW.Stop();
   setup_findpts_setup_time = setupSW.RealTime();
   setupflag = true;
}

void FindPointsGSLIB::FindPoints(const Vector &point_pos,
                                 int point_pos_ordering)
{
   MFEM_VERIFY(setupflag, "Use FindPointsGSLIB::Setup before finding points.");
   auto *findptsData = (gslib::findpts_data_3 *)this->fdataD;

   bool dev_mode = point_pos.UseDevice();
   points_cnt = point_pos.Size() / dim;
   gsl_code.SetSize(points_cnt);
   gsl_proc.SetSize(points_cnt);
   gsl_elem.SetSize(points_cnt);
   gsl_ref.SetSize(points_cnt * dim);
   gsl_dist.SetSize(points_cnt);

   setupSW.Clear();
   setupSW.Start();
   if (dev_mode)
   {
      FindPointsOnDevice(point_pos, point_pos_ordering);
   }
   else
   {
      auto xvFill = [&](const double *xv_base[], unsigned xv_stride[])
      {
         for (int d = 0; d < dim; d++)
         {
            if (point_pos_ordering == Ordering::byNODES)
            {
               xv_base[d] = point_pos.GetData() + d*points_cnt;
               xv_stride[d] = sizeof(double);
            }
            else
            {
               xv_base[d] = point_pos.GetData() + d;
               xv_stride[d] = dim*sizeof(double);
            }
         }
      };

      if (dim == 2)
      {
         const double *xv_base[2];
         unsigned xv_stride[2];
         xvFill(xv_base, xv_stride);
         findpts_2(gsl_code.GetData(), sizeof(unsigned int),
                   gsl_proc.GetData(), sizeof(unsigned int),
                   gsl_elem.GetData(), sizeof(unsigned int),
                   gsl_ref.GetData(),  sizeof(double) * dim,
                   gsl_dist.GetData(), sizeof(double),
                   xv_base, xv_stride, points_cnt, fdata2D);
      }
      else  // dim == 3
      {
         const double *xv_base[3];
         unsigned xv_stride[3];
         xvFill(xv_base, xv_stride);
         findpts_3(gsl_code.GetData(), sizeof(unsigned int),
                   gsl_proc.GetData(), sizeof(unsigned int),
                   gsl_elem.GetData(), sizeof(unsigned int),
                   gsl_ref.GetData(),  sizeof(double) * dim,
                   gsl_dist.GetData(), sizeof(double),
                   xv_base, xv_stride, points_cnt,
                   findptsData);
      }
   } //device_mode
   setupSW.Stop();
   findpts_findpts_time = setupSW.RealTime();

   setupSW.Clear();
   setupSW.Start();
   // Set the element number and reference position to 0 for points not found
   for (int i = 0; i < points_cnt; i++)
   {
      if (gsl_code[i] == 2 ||
          (gsl_code[i] == 1 && gsl_dist(i) > bdr_tol))
      {
         gsl_elem[i] = 0;
         for (int d = 0; d < dim; d++) { gsl_ref(i*dim + d) = -1.; }
         gsl_code[i] = 2;
      }
   }

   // Map element number for simplices, and ref_pos from [-1,1] to [0,1] for
   // both simplices and quads. Also sets code to 1 for points found on element
   // faces/edges.
   MapRefPosAndElemIndices();
   setupSW.Stop();
   findpts_mapelemrst_time = setupSW.RealTime();
}

namespace internal
{
// mfem::forall-based copy kernel -- used by protected methods below.
// Needed as a workaround for the nvcc restriction that methods with mfem::forall
// in them must to be public.
template <typename T>
static inline void device_copy(T *d_dest, const T *d_src, int size)
{
   mfem::forall(size, [=] MFEM_HOST_DEVICE (int i) { d_dest[i] = d_src[i]; });
}

template <typename T1, typename T2>
static inline void device_copy(T1 *d_dest, const T2 *d_src, int size)
{
   mfem::forall(size, [=] MFEM_HOST_DEVICE (int i) { d_dest[i] = (T1)d_src[i]; });
}
} // namespace internal

static slong lfloor(dfloat x) { return floor(x); }

static ulong hash_index_aux(dfloat low, dfloat fac, ulong n, dfloat x)
{
   const slong i = lfloor((x - low) * fac);
   return i < 0 ? 0 : (n - 1 < (ulong)i ? n - 1 : (ulong)i);
}

static ulong hash_index_3(const gslib::hash_data_3 *p, const dfloat x[3])
{
   const ulong n = p->hash_n;
   return (hash_index_aux(p->bnd[2].min, p->fac[2], n, x[2]) * n +
           hash_index_aux(p->bnd[1].min, p->fac[1], n, x[1])) *
          n +
          hash_index_aux(p->bnd[0].min, p->fac[0], n, x[0]);
}

void FindPointsGSLIB::FindPointsOnDevice(const Vector &point_pos,
                                         int point_pos_ordering)
{
   MemoryType mt = point_pos.GetMemory().GetMemoryType();
   setupSW.Clear();
   setupSW.Start();
   SetupDevice(mt);
   setupSW.Stop();
   findpts_setup_device_arrays_time = setupSW.RealTime();

   gsl_code_dev.SetSize(points_cnt);
   gsl_elem_dev.SetSize(points_cnt);

   gsl_ref.UseDevice(true);
   gsl_dist.UseDevice(true);

   FindPointsLocal(point_pos,
                   point_pos_ordering,
                   gsl_code_dev,
                   gsl_elem_dev,
                   gsl_ref,
                   gsl_dist,
                   points_cnt);

   // TODO: Only transfer information for points found
   // Transfer information from DEVICE
   internal::device_copy(gsl_code.GetData(), gsl_code_dev.Read(), points_cnt);
   internal::device_copy(gsl_elem.GetData(), gsl_elem_dev.Read(), points_cnt);
   gsl_proc = gsl_comm->id;

   gsl_ref.HostReadWrite();
   gsl_dist.HostReadWrite();
   point_pos.HostRead();
   MPI_Barrier(gsl_comm->c);

   /* send unfound and border points to global hash cells */
   struct gslib::array hash_pt, src_pt, out_pt;

   struct srcPt_t
   {
      dfloat x[3];
      unsigned int index, proc;
   };

   struct outPt_t
   {
      dfloat r[3], dist2;
      unsigned int index, code, el, proc;
   };

   const int id = gsl_comm->id,
             np = gsl_comm->np;

   if (np == 1)
   {
      return;
   }

   int find_elsewhere = 0;
   int found_local = 0;
   int hashptn = 0;
   {
      int index;
      unsigned int *code = gsl_code.GetData(), *proc = gsl_proc.GetData();
      struct srcPt_t *pt;

      array_init(struct srcPt_t, &hash_pt, points_cnt);
      pt = (struct srcPt_t *)hash_pt.ptr;

      dfloat x[dim];
      for (index = 0; index < points_cnt; ++index)
      {
         for (int d = 0; d < dim; ++d)
         {
            int idx = point_pos_ordering == 0 ?
                      index + d*points_cnt :
                      index*dim + d;
            x[d] = point_pos(idx);
         }
         *proc = id;
         if (*code != 0)
         {
            find_elsewhere++;
            const auto hi = hash_index_3(DEV.hash, x);
            for (int d = 0; d < dim; ++d)
            {
               pt->x[d] = x[d];
            }
            pt->index = index;
            pt->proc = hi % np;
            ++pt;
         }
         else { found_local++; }
         code++;
         proc++;
      }
      hash_pt.n = pt - (struct srcPt_t *)hash_pt.ptr;
      sarray_transfer(struct srcPt_t, &hash_pt, proc, 1, DEV.cr);
      hashptn = hash_pt.n;
   }
   MPI_Barrier(gsl_comm->c);

   /* look up points in hash cells, route to possible procs */
   {
      const unsigned int *const hash_offset = DEV.hash->offset;
      int count = 0, *proc, *proc_p;
      const struct srcPt_t *p = (struct srcPt_t *)hash_pt.ptr,
                            *const pe = p + hash_pt.n;
      struct srcPt_t *q;

      for (; p != pe; ++p)
      {
         const int hi = hash_index_3(DEV.hash, p->x) / np;
         const int i = hash_offset[hi], ie = hash_offset[hi + 1];
         count += ie - i;
      }

      proc = (int *) gslib::smalloc(count*sizeof(int), __FILE__,__LINE__);
      proc_p = proc;
      array_init(struct srcPt_t, &src_pt, count);
      q = (struct srcPt_t *)src_pt.ptr;

      p = (struct srcPt_t *)hash_pt.ptr;
      for (; p != pe; ++p)
      {
         const int hi = hash_index_3(DEV.hash, p->x) / np;
         int i = hash_offset[hi];
         const int ie = hash_offset[hi + 1];
         for (; i != ie; ++i)
         {
            const int pp = hash_offset[i];
            if (pp == p->proc)
            {
               continue;   /* don't send back to source proc */
            }
            *proc_p++ = pp;
            *q++ = *p;
         }
      }

      array_free(&hash_pt);
      src_pt.n = proc_p - proc;

      sarray_transfer_ext(struct srcPt_t, &src_pt,
                          reinterpret_cast<unsigned int *>(proc), sizeof(int), DEV.cr);

      free(proc);
   }
   MPI_Barrier(gsl_comm->c);

   /* look for other procs' points, send back */
   {
      int n = src_pt.n;
      const struct srcPt_t *spt;
      struct outPt_t *opt;
      array_init(struct outPt_t, &out_pt, n);
      out_pt.n = n;
      spt = (struct srcPt_t *)src_pt.ptr;
      opt = (struct outPt_t *)out_pt.ptr;
      for (; n; --n, ++spt, ++opt)
      {
         opt->index = spt->index;
         opt->proc = spt->proc;
      }
      spt = (struct srcPt_t *)src_pt.ptr;
      opt = (struct outPt_t *)out_pt.ptr;

      n = out_pt.n;
      Vector gsl_ref_l(n*dim);
      Vector gsl_dist_l(n);
      gsl_ref_l.UseDevice(true);
      gsl_dist_l.UseDevice(true);
      gsl_ref_l = 0.0;
      gsl_dist_l = 0.0;
      gsl_code_dev.SetSize(n);
      gsl_elem_dev.SetSize(n);
      Vector point_pos_l(n*dim);
      point_pos_l.UseDevice(true);
      point_pos_l.HostReadWrite();

      for (int point = 0; point < n; ++point)
      {
         for (int d = 0; d < dim; d++)
         {
            int idx = point_pos_ordering == 0 ? point + d*n : point*dim + d;
            point_pos_l(idx) = spt[point].x[d];
         }
      }

      FindPointsLocal(point_pos_l,
                      point_pos_ordering,
                      gsl_code_dev,
                      gsl_elem_dev,
                      gsl_ref_l,
                      gsl_dist_l,
                      n);

      gsl_ref_l.HostRead();
      gsl_dist_l.HostRead();
      gsl_code_dev.HostRead();
      gsl_elem_dev.HostRead();

      // unpack arrays into opt
      for (int point = 0; point < n; point++)
      {
         //  opt[point].code = gsl_code_l2[point];
         opt[point].code = gsl_code_dev[point];
         opt[point].el = gsl_elem_dev[point];
         opt[point].dist2 = gsl_dist_l(point);
         for (int d = 0; d < dim; ++d)
         {
            opt[point].r[d] = gsl_ref_l(dim * point + d);
         }
      }

      array_free(&src_pt);

      /* group by code to eliminate unfound points */
      sarray_sort(struct outPt_t, opt, out_pt.n, code, 0, &DEV.cr->data);

      n = out_pt.n;
      while (n && opt[n - 1].code == CODE_NOT_FOUND)
      {
         --n;
      }
      out_pt.n = n;

      sarray_transfer(struct outPt_t, &out_pt, proc, 1, DEV.cr);
   }
   MPI_Barrier(gsl_comm->c);

   /* merge remote results with user data */
   int npt_found_on_other_proc = 0;
   {
      int n = out_pt.n;
      struct outPt_t *opt = (struct outPt_t *)out_pt.ptr;
      for (; n; --n, ++opt)
      {
         const int index = opt->index;
         if (gsl_code[index] == CODE_INTERNAL)
         {
            continue;
         }
         if (gsl_code[index] == CODE_NOT_FOUND || opt->code == CODE_INTERNAL ||
             opt->dist2 < gsl_dist(index))
         {
            npt_found_on_other_proc++;
            for (int d = 0; d < dim; ++d)
            {
               gsl_ref(dim * index + d) = opt->r[d];
            }
            gsl_dist(index) = opt->dist2;
            gsl_proc[index] = opt->proc;
            gsl_elem[index] = opt->el;
            gsl_code[index] = opt->code;
         }
      }
      array_free(&out_pt);
   }
   MPI_Barrier(gsl_comm->c);
}

void FindPointsGSLIB::SetupDevice(MemoryType mt)
{
   MFEM_VERIFY(dim == 3,"GPU stuff only supported for dim=3");
   auto *findptsData = (gslib::findpts_data_3 *)this->fdataD;
   DEV.tol = findptsData->local.tol;
   DEV.hash = &findptsData->hash;
   DEV.cr = &findptsData->cr;

   const int mesh_pts_cnt = gsl_mesh.Size()/dim;

   //TO DO: make DEV.O_x/y/z point to gsl_mesh if it is already on device?
   // will gsl_mesh ever be on device since we do setup on cpu
   DEV.o_x.SetSize(mesh_pts_cnt);
   DEV.o_x.UseDevice(true);
   DEV.o_y.SetSize(mesh_pts_cnt);
   DEV.o_y.UseDevice(true);
   DEV.o_z.SetSize(mesh_pts_cnt);
   DEV.o_z.UseDevice(true);

   internal::device_copy(DEV.o_x.ReadWrite(), gsl_mesh.HostRead(), mesh_pts_cnt);
   internal::device_copy(DEV.o_y.ReadWrite(), gsl_mesh.HostRead()+mesh_pts_cnt,
                         mesh_pts_cnt);
   internal::device_copy(DEV.o_z.ReadWrite(), gsl_mesh.HostRead()+2*mesh_pts_cnt,
                         mesh_pts_cnt);

   DEV.o_c.SetSize(dim*NE_split_total);
   DEV.o_c.UseDevice(true);
   auto p_o_c = DEV.o_c.HostReadWrite();

   DEV.o_A.SetSize(dim*dim*NE_split_total);
   DEV.o_A.UseDevice(true);
   auto p_o_A = DEV.o_A.HostReadWrite();

   DEV.o_min.SetSize(dim*NE_split_total);
   DEV.o_min.UseDevice(true);
   auto p_o_min = DEV.o_min.HostReadWrite();

   DEV.o_max.SetSize(dim*NE_split_total);
   DEV.o_max.UseDevice(true);
   auto p_o_max = DEV.o_max.HostReadWrite();

   const int dim2 = dim*dim;

   for (int e = 0; e < NE_split_total; ++e)
   {
      auto box = findptsData->local.obb[e];

      for (int d = 0; d < dim; d++)
      {
         //         DEV.o_c(dim * e + d) = box.c0[d];
         p_o_c[dim*e+d] = box.c0[d];
         p_o_min[dim*e+d] = box.x[d].min;
         p_o_max[dim*e+d] = box.x[d].max;
      }

      for (int i = 0; i < dim2; ++i)
      {
         p_o_A[dim2*e+i] = box.A[i];
      }
   }

   DEV.o_hashMin.SetSize(dim); DEV.o_hashMin.UseDevice(true);
   DEV.o_hashFac.SetSize(dim); DEV.o_hashFac.UseDevice(true);
   auto hash = findptsData->local.hd;

   auto p_o_hashMin = DEV.o_hashMin.HostReadWrite();
   auto p_o_hashFac = DEV.o_hashFac.HostReadWrite();
   for (int d = 0; d < dim; ++d)
   {
      p_o_hashMin[d] = hash.bnd[d].min;
      p_o_hashFac[d] = hash.fac[d];
   }

   DEV.hash_n = hash.hash_n;

   const auto hd_d_size = getHashSize(findptsData,
                                      NE_split_total,
                                      DEV.local_hash_size);

   DEV.o_offset.SetSize(hd_d_size, mt);
   auto p_o_offset = DEV.o_offset.ReadWrite();
   mfem::forall(hd_d_size, [=] MFEM_HOST_DEVICE (int i) { p_o_offset[i] = findptsData->local.hd.offset[i]; });

   DEV.o_wtend_x.SetSize(6*DEV.dof1d);
   DEV.o_wtend_x.UseDevice(true);
   DEV.o_wtend_y.SetSize(6*DEV.dof1d);
   DEV.o_wtend_y.UseDevice(true);

   internal::device_copy(DEV.o_wtend_x.ReadWrite(),
                         findptsData->local.fed.wtend[0],
                         6*DEV.dof1d);
   internal::device_copy(DEV.o_wtend_y.ReadWrite(),
                         findptsData->local.fed.wtend[1],
                         6*DEV.dof1d);
   if (dim == 3)
   {
      DEV.o_wtend_z.SetSize(6*DEV.dof1d);
      DEV.o_wtend_z.UseDevice(true);
      internal::device_copy(DEV.o_wtend_z.ReadWrite(),
                            findptsData->local.fed.wtend[2],
                            6*DEV.dof1d);
   }

   // Get gll points
   DEV.gll1d.SetSize(DEV.dof1d);
   DEV.gll1d.UseDevice(true);
   internal::device_copy(DEV.gll1d.ReadWrite(),
                         findptsData->local.fed.z[0],
                         DEV.dof1d);

   DEV.lagcoeff.SetSize(DEV.dof1d);
   DEV.lagcoeff.UseDevice(true);
   internal::device_copy(DEV.lagcoeff.ReadWrite(),
                         findptsData->local.fed.lag_data[0],
                         DEV.dof1d);

   DEV.info.SetSize(50*points_cnt);
   DEV.info.UseDevice(true);
   DEV.info = -1.0;

   MFEM_DEVICE_SYNC;
}

void FindPointsGSLIB::FindPoints(Mesh &m, const Vector &point_pos,
                                 int point_pos_ordering, const double bb_t,
                                 const double newt_tol, const int npt_max)
{
   if (!setupflag || (mesh != &m) )
   {
      Setup(m, bb_t, newt_tol, npt_max);
   }
   FindPoints(point_pos, point_pos_ordering);
}

void FindPointsGSLIB::Interpolate(const Vector &point_pos,
                                  const GridFunction &field_in, Vector &field_out,
                                  int point_pos_ordering)
{
   FindPoints(point_pos, point_pos_ordering);
   Interpolate(field_in, field_out);
}

void FindPointsGSLIB::Interpolate(Mesh &m, const Vector &point_pos,
                                  const GridFunction &field_in, Vector &field_out,
                                  int point_pos_ordering)
{
   FindPoints(m, point_pos, point_pos_ordering);
   Interpolate(field_in, field_out);
}

void FindPointsGSLIB::FreeData()
{
   if (!setupflag) { return; }
   crystal_free(cr);
   if (dim == 2)
   {
      findpts_free_2(fdata2D);
   }
   else
   {
      findpts_free_3((gslib::findpts_data_3 *)this->fdataD);
   }
   gsl_code.DeleteAll();
   gsl_proc.DeleteAll();
   gsl_elem.DeleteAll();
   gsl_mesh.Destroy();
   gsl_ref.Destroy();
   gsl_dist.Destroy();
   for (int i = 0; i < 4; i++)
   {
      if (mesh_split[i]) { delete mesh_split[i]; mesh_split[i] = NULL; }
      if (ir_split[i]) { delete ir_split[i]; ir_split[i] = NULL; }
      if (fes_rst_map[i]) { delete fes_rst_map[i]; fes_rst_map[i] = NULL; }
      if (gf_rst_map[i]) { delete gf_rst_map[i]; gf_rst_map[i] = NULL; }
   }
   if (fec_map_lin) { delete fec_map_lin; fec_map_lin = NULL; }
   setupflag = false;
}

void FindPointsGSLIB::SetupSplitMeshes()
{
   fec_map_lin = new H1_FECollection(1, dim);
   if (mesh->Dimension() == 2)
   {
      int Nvert = 7;
      int NEsplit = 3;
      mesh_split[0] = new Mesh(2, Nvert, NEsplit, 0, 2);

      const double quad_v[7][2] =
      {
         {0, 0}, {0.5, 0}, {1, 0}, {0, 0.5},
         {1./3., 1./3.}, {0.5, 0.5}, {0, 1}
      };
      const int quad_e[3][4] =
      {
         {0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 5, 6}
      };

      for (int j = 0; j < Nvert; j++)
      {
         mesh_split[0]->AddVertex(quad_v[j]);
      }
      for (int j = 0; j < NEsplit; j++)
      {
         int attribute = j + 1;
         mesh_split[0]->AddQuad(quad_e[j], attribute);
      }
      mesh_split[0]->FinalizeQuadMesh(1, 1, true);

      fes_rst_map[0] = new FiniteElementSpace(mesh_split[0], fec_map_lin, dim);
      gf_rst_map[0] = new GridFunction(fes_rst_map[0]);
      gf_rst_map[0]->UseDevice(false);
      const int npt = gf_rst_map[0]->Size()/dim;
      for (int k = 0; k < dim; k++)
      {
         for (int j = 0; j < npt; j++)
         {
            (*gf_rst_map[0])(j+k*npt) = quad_v[j][k];
         }
      }

      mesh_split[1] = new Mesh(Mesh::MakeCartesian2D(1, 1,
                                                     Element::QUADRILATERAL));
   }
   else if (mesh->Dimension() == 3)
   {
      mesh_split[0] = new Mesh(Mesh::MakeCartesian3D(1, 1, 1,
                                                     Element::HEXAHEDRON));
      // Tetrahedron
      {
         int Nvert = 15;
         int NEsplit = 4;
         mesh_split[1] = new Mesh(3, Nvert, NEsplit, 0, 3);

         const double hex_v[15][3] =
         {
            {0, 0, 0.}, {1, 0., 0.}, {0., 1., 0.}, {0, 0., 1.},
            {0.5, 0., 0.}, {0.5, 0.5, 0.}, {0., 0.5, 0.},
            {0., 0., 0.5}, {0.5, 0., 0.5}, {0., 0.5, 0.5},
            {1./3., 0., 1./3.}, {1./3., 1./3., 1./3.}, {0, 1./3., 1./3.},
            {1./3., 1./3., 0}, {0.25, 0.25, 0.25}
         };
         const int hex_e[4][8] =
         {
            {7, 10, 4, 0, 12, 14, 13, 6},
            {10, 8, 1, 4, 14, 11, 5, 13},
            {14, 11, 5, 13, 12, 9, 2, 6},
            {7, 3, 8, 10, 12, 9, 11, 14}
         };

         for (int j = 0; j < Nvert; j++)
         {
            mesh_split[1]->AddVertex(hex_v[j]);
         }
         for (int j = 0; j < NEsplit; j++)
         {
            int attribute = j + 1;
            mesh_split[1]->AddHex(hex_e[j], attribute);
         }
         mesh_split[1]->FinalizeHexMesh(1, 1, true);

         fes_rst_map[1] = new FiniteElementSpace(mesh_split[1], fec_map_lin, dim);
         gf_rst_map[1] = new GridFunction(fes_rst_map[1]);
         gf_rst_map[1]->UseDevice(false);
         const int npt = gf_rst_map[1]->Size()/dim;
         for (int k = 0; k < dim; k++)
         {
            for (int j = 0; j < npt; j++)
            {
               (*gf_rst_map[1])(j+k*npt) = hex_v[j][k];
            }
         }
      }
      // Prism
      {
         int Nvert = 14;
         int NEsplit = 3;
         mesh_split[2] = new Mesh(3, Nvert, NEsplit, 0, 3);

         const double hex_v[14][3] =
         {
            {0, 0, 0}, {0.5, 0, 0}, {1, 0, 0}, {0, 0.5, 0},
            {1./3., 1./3., 0}, {0.5, 0.5, 0}, {0, 1, 0},
            {0, 0, 1}, {0.5, 0, 1}, {1, 0, 1}, {0, 0.5, 1},
            {1./3., 1./3., 1}, {0.5, 0.5, 1}, {0, 1, 1}
         };
         const int hex_e[3][8] =
         {
            {0, 1, 4, 3, 7, 8, 11, 10},
            {1, 2, 5, 4, 8, 9, 12, 11},
            {3, 4, 5, 6, 10, 11, 12, 13}
         };

         for (int j = 0; j < Nvert; j++)
         {
            mesh_split[2]->AddVertex(hex_v[j]);
         }
         for (int j = 0; j < NEsplit; j++)
         {
            int attribute = j + 1;
            mesh_split[2]->AddHex(hex_e[j], attribute);
         }
         mesh_split[2]->FinalizeHexMesh(1, 1, true);

         fes_rst_map[2] = new FiniteElementSpace(mesh_split[2], fec_map_lin, dim);
         gf_rst_map[2] = new GridFunction(fes_rst_map[2]);
         gf_rst_map[2]->UseDevice(false);
         const int npt = gf_rst_map[2]->Size()/dim;
         for (int k = 0; k < dim; k++)
         {
            for (int j = 0; j < npt; j++)
            {
               (*gf_rst_map[2])(j+k*npt) = hex_v[j][k];
            }
         }
      }
      // Pyramid
      {
         int Nvert = 23;
         int NEsplit = 8;
         mesh_split[3] = new Mesh(3, Nvert, NEsplit, 0, 3);

         const double hex_v[23][3] =
         {
            {0.0000, 0.0000, 0.0000}, {0.5000, 0.0000, 0.0000},
            {0.0000, 0.0000, 0.5000}, {0.3333, 0.0000, 0.3333},
            {0.0000, 0.5000, 0.0000}, {0.3333, 0.3333, 0.0000},
            {0.0000, 0.3333, 0.3333}, {0.2500, 0.2500, 0.2500},
            {1.0000, 0.0000, 0.0000}, {0.5000, 0.0000, 0.5000},
            {0.5000, 0.5000, 0.0000}, {0.3333, 0.3333, 0.3333},
            {0.0000, 1.0000, 0.0000}, {0.0000, 0.5000, 0.5000},
            {0.0000, 0.0000, 1.0000}, {1.0000, 0.5000, 0.0000},
            {0.6667, 0.3333, 0.3333}, {0.6667, 0.6667, 0.0000},
            {0.5000, 0.5000, 0.2500}, {1.0000, 1.0000, 0.0000},
            {0.5000, 0.5000, 0.5000}, {0.5000, 1.0000, 0.0000},
            {0.3333, 0.6667, 0.3333}
         };
         const int hex_e[8][8] =
         {
            {2, 3, 1, 0, 6, 7, 5, 4}, {3, 9, 8, 1, 7, 11, 10, 5},
            {7, 11, 10, 5, 6, 13, 12, 4}, {2, 14, 9, 3, 6, 13, 11, 7},
            {9, 16, 15, 8, 11, 18, 17, 10}, {16, 20, 19, 15, 18, 22, 21, 17},
            {18, 22, 21, 17, 11, 13, 12, 10}, {9, 14, 20, 16, 11, 13, 22, 18}
         };

         for (int j = 0; j < Nvert; j++)
         {
            mesh_split[3]->AddVertex(hex_v[j]);
         }
         for (int j = 0; j < NEsplit; j++)
         {
            int attribute = j + 1;
            mesh_split[3]->AddHex(hex_e[j], attribute);
         }
         mesh_split[3]->FinalizeHexMesh(1, 1, true);

         fes_rst_map[3] = new FiniteElementSpace(mesh_split[3], fec_map_lin, dim);
         gf_rst_map[3] = new GridFunction(fes_rst_map[3]);
         gf_rst_map[3]->UseDevice(false);
         const int npt = gf_rst_map[3]->Size()/dim;
         for (int k = 0; k < dim; k++)
         {
            for (int j = 0; j < npt; j++)
            {
               (*gf_rst_map[3])(j+k*npt) = hex_v[j][k];
            }
         }
      }
   }

   NE_split_total = 0;
   split_element_map.SetSize(0);
   split_element_index.SetSize(0);
   int NEsplit = 0;
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      const Geometry::Type gt   = mesh->GetElement(e)->GetGeometryType();
      if (gt == Geometry::TRIANGLE || gt == Geometry::PRISM)
      {
         NEsplit = 3;
      }
      else if (gt == Geometry::TETRAHEDRON)
      {
         NEsplit = 4;
      }
      else if (gt == Geometry::PYRAMID)
      {
         NEsplit = 8;
      }
      else if (gt == Geometry::SQUARE || gt == Geometry::CUBE)
      {
         NEsplit = 1;
      }
      else
      {
         MFEM_ABORT("Unsupported geometry type.");
      }
      NE_split_total += NEsplit;
      for (int i = 0; i < NEsplit; i++)
      {
         split_element_map.Append(e);
         split_element_index.Append(i);
      }
   }
}

void FindPointsGSLIB::SetupIntegrationRuleForSplitMesh(Mesh *meshin,
                                                       IntegrationRule *irule,
                                                       int order)
{
   H1_FECollection fec(order, dim);
   FiniteElementSpace nodal_fes(meshin, &fec, dim);
   meshin->SetNodalFESpace(&nodal_fes);
   const int NEsplit = meshin->GetNE();

   const int dof_cnt = nodal_fes.GetFE(0)->GetDof(),
             pts_cnt = NEsplit * dof_cnt;
   Vector irlist(dim * pts_cnt);

   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement *>(nodal_fes.GetFE(0));
   MFEM_VERIFY(tbe != NULL, "TensorBasis FiniteElement expected.");
   const Array<int> &dof_map = tbe->GetDofMap();

   DenseMatrix pos(dof_cnt, dim);
   Vector posV(pos.Data(), dof_cnt * dim);
   Array<int> xdofs(dof_cnt * dim);

   // Create an IntegrationRule on the nodes of the reference submesh.
   MFEM_ASSERT(irule->GetNPoints() == pts_cnt, "IntegrationRule does not have"
               "the correct number of points.");
   GridFunction *nodesplit = meshin->GetNodes();
   int pt_id = 0;
   for (int i = 0; i < NEsplit; i++)
   {
      nodal_fes.GetElementVDofs(i, xdofs);
      nodesplit->GetSubVector(xdofs, posV);
      for (int j = 0; j < dof_cnt; j++)
      {
         for (int d = 0; d < dim; d++)
         {
            irlist(pts_cnt * d + pt_id) = pos(dof_map[j], d);
         }
         irule->IntPoint(pt_id).x = irlist(pt_id);
         irule->IntPoint(pt_id).y = irlist(pts_cnt + pt_id);
         if (dim == 3)
         {
            irule->IntPoint(pt_id).z = irlist(2*pts_cnt + pt_id);
         }
         pt_id++;
      }
   }
}

void FindPointsGSLIB::GetNodalValues(const GridFunction *gf_in,
                                     Vector &node_vals)
{
   const GridFunction *nodes     = gf_in;
   const FiniteElementSpace *fes = nodes->FESpace();
   const int NE                  = mesh->GetNE();
   const int vdim                = fes->GetVDim();

   IntegrationRule *ir_split_temp = NULL;

   const int maxOrder = fes->GetMaxElementOrder();
   const int dof_1D =  maxOrder+1;
   const int pts_el = std::pow(dof_1D, dim);
   const int pts_cnt = NE_split_total * pts_el;
   node_vals.SetSize(vdim * pts_cnt);
   node_vals = 0.0;

   int gsl_mesh_pt_index = 0;

   for (int e = 0; e < NE; e++)
   {
      const FiniteElement *fe   = fes->GetFE(e);
      const Geometry::Type gt   = fe->GetGeomType();
      bool el_to_split = true;
      if (gt == Geometry::TRIANGLE)
      {
         ir_split_temp = ir_split[0];
      }
      else if (gt == Geometry::TETRAHEDRON)
      {
         ir_split_temp = ir_split[1];
      }
      else if (gt == Geometry::PRISM)
      {
         ir_split_temp = ir_split[2];
      }
      else if (gt == Geometry::PYRAMID)
      {
         ir_split_temp = ir_split[3];
      }
      else if (gt == Geometry::SQUARE)
      {
         ir_split_temp = ir_split[1];
         el_to_split = gf_in->FESpace()->IsVariableOrder();
      }
      else if (gt == Geometry::CUBE)
      {
         ir_split_temp = ir_split[0];
         el_to_split = gf_in->FESpace()->IsVariableOrder();
      }
      else
      {
         MFEM_ABORT("Unsupported geometry type.");
      }

      if (el_to_split) // Triangle/Tet/Prism or Quads/Hex but variable order
      {
         // Fill gsl_mesh with location of split points.
         Vector locval(vdim);
         for (int i = 0; i < ir_split_temp->GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir_split_temp->IntPoint(i);
            nodes->GetVectorValue(e, ip, locval);
            for (int d = 0; d < vdim; d++)
            {
               node_vals(pts_cnt*d + gsl_mesh_pt_index) = locval(d);
            }
            gsl_mesh_pt_index++;
         }
      }
      else // Quad/Hex and constant polynomial order
      {
         const int dof_cnt_split = fe->GetDof();

         const TensorBasisElement *tbe =
            dynamic_cast<const TensorBasisElement *>(fes->GetFE(e));
         MFEM_VERIFY(tbe != NULL, "TensorBasis FiniteElement expected.");
         Array<int> dof_map(dof_cnt_split);
         const Array<int> &dm = tbe->GetDofMap();
         if (dm.Size() > 0) { dof_map = dm; }
         else { for (int i = 0; i < dof_cnt_split; i++) { dof_map[i] = i; } }

         DenseMatrix pos(dof_cnt_split, vdim);
         Vector posV(pos.Data(), dof_cnt_split * vdim);
         Array<int> xdofs(dof_cnt_split * vdim);

         fes->GetElementVDofs(e, xdofs);
         nodes->GetSubVector(xdofs, posV);
         for (int j = 0; j < dof_cnt_split; j++)
         {
            for (int d = 0; d < vdim; d++)
            {
               node_vals(pts_cnt * d + gsl_mesh_pt_index) = pos(dof_map[j], d);
            }
            gsl_mesh_pt_index++;
         }
      }
   }
}


void FindPointsGSLIB::MapRefPosAndElemIndices()
{
   gsl_mfem_ref.SetSize(points_cnt*dim);
   gsl_mfem_elem.SetSize(points_cnt);
   internal::device_copy(gsl_mfem_ref.GetData(), gsl_ref.HostRead(),
                         points_cnt*dim);
   internal::device_copy(gsl_mfem_elem.GetData(), gsl_elem.GetData(), points_cnt);

   gsl_mfem_ref += 1.;  // map  [-1, 1] to [0, 2] to [0, 1]
   gsl_mfem_ref *= 0.5;

   int nptorig = points_cnt,
       npt = points_cnt;

   // tolerance for point to be marked as on element edge/face
   double btol = 1e-12;

   GridFunction *gf_rst_map_temp = NULL;
   int nptsend = 0;

   for (int index = 0; index < npt; index++)
   {
      if (gsl_code[index] != 2 && gsl_proc[index] != gsl_comm->id)
      {
         nptsend +=1;
      }
   }

   // Pack data to send via crystal router
   struct gslib::array *outpt = new gslib::array;
   struct out_pt { double r[3]; uint index, el, proc, code; };
   struct out_pt *pt;
   array_init(struct out_pt, outpt, nptsend);
   outpt->n=nptsend;
   pt = (struct out_pt *)outpt->ptr;
   for (int index = 0; index < npt; index++)
   {
      if (gsl_code[index] == 2 || gsl_proc[index] == gsl_comm->id)
      {
         continue;
      }
      for (int d = 0; d < dim; ++d)
      {
         pt->r[d]= gsl_mfem_ref(index*dim + d);
      }
      pt->index = index;
      pt->proc  = gsl_proc[index];
      pt->el    = gsl_elem[index];
      pt->code  = gsl_code[index];
      ++pt;
   }

   // Transfer data to target MPI ranks
   sarray_transfer(struct out_pt, outpt, proc, 1, cr);

   // Map received points
   npt = outpt->n;
   pt = (struct out_pt *)outpt->ptr;
   for (int index = 0; index < npt; index++)
   {
      IntegrationPoint ip;
      ip.Set3(&pt->r[0]);
      const int elem = pt->el;
      const int mesh_elem = split_element_map[elem];
      const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(mesh_elem);

      const Geometry::Type gt = fe->GetGeomType();
      pt->el = mesh_elem;

      if (gt == Geometry::SQUARE || gt == Geometry::CUBE)
      {
         // check if it is on element boundary
         pt->code = Geometry::CheckPoint(gt, ip, -btol) ? 0 : 1;
         ++pt;
         continue;
      }
      else if (gt == Geometry::TRIANGLE)
      {
         gf_rst_map_temp = gf_rst_map[0];
      }
      else if (gt == Geometry::TETRAHEDRON)
      {
         gf_rst_map_temp = gf_rst_map[1];
      }
      else if (gt == Geometry::PRISM)
      {
         gf_rst_map_temp = gf_rst_map[2];
      }
      else if (gt == Geometry::PYRAMID)
      {
         gf_rst_map_temp = gf_rst_map[3];
      }

      int local_elem = split_element_index[elem];
      Vector mfem_ref(dim);
      // map to rst of macro element
      gf_rst_map_temp->GetVectorValue(local_elem, ip, mfem_ref);

      for (int d = 0; d < dim; d++)
      {
         pt->r[d] = mfem_ref(d);
      }

      // check if point is on element boundary
      ip.Set3(&pt->r[0]);
      pt->code = Geometry::CheckPoint(gt, ip, -btol) ? 0 : 1;
      ++pt;
   }

   // Transfer data back to source MPI rank
   sarray_transfer(struct out_pt, outpt, proc, 1, cr);
   npt = outpt->n;

   // First copy mapped information for points on other procs
   pt = (struct out_pt *)outpt->ptr;
   for (int index = 0; index < npt; index++)
   {
      gsl_mfem_elem[pt->index] = pt->el;
      for (int d = 0; d < dim; d++)
      {
         gsl_mfem_ref(d + pt->index*dim) = pt->r[d];
      }
      gsl_code[pt->index] = pt->code;
      ++pt;
   }
   array_free(outpt);
   delete outpt;

   // Now map information for points on the same proc
   for (int index = 0; index < nptorig; index++)
   {
      if (gsl_code[index] != 2 && gsl_proc[index] == gsl_comm->id)
      {

         IntegrationPoint ip;
         Vector mfem_ref(gsl_mfem_ref.GetData()+index*dim, dim);
         ip.Set2(mfem_ref.GetData());
         if (dim == 3) { ip.z = mfem_ref(2); }

         const int elem = gsl_elem[index];
         const int mesh_elem = split_element_map[elem];
         const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(mesh_elem);
         const Geometry::Type gt = fe->GetGeomType();
         gsl_mfem_elem[index] = mesh_elem;
         if (gt == Geometry::SQUARE || gt == Geometry::CUBE)
         {
            gsl_code[index] = Geometry::CheckPoint(gt, ip, -btol) ? 0 : 1;
            continue;
         }
         else if (gt == Geometry::TRIANGLE)
         {
            gf_rst_map_temp = gf_rst_map[0];
         }
         else if (gt == Geometry::TETRAHEDRON)
         {
            gf_rst_map_temp = gf_rst_map[1];
         }
         else if (gt == Geometry::PRISM)
         {
            gf_rst_map_temp = gf_rst_map[2];
         }
         else if (gt == Geometry::PYRAMID)
         {
            gf_rst_map_temp = gf_rst_map[3];
         }

         int local_elem = split_element_index[elem];
         gf_rst_map_temp->GetVectorValue(local_elem, ip, mfem_ref);

         // Check if the point is on element boundary
         ip.Set2(mfem_ref.GetData());
         if (dim == 3) { ip.z = mfem_ref(2); }
         gsl_code[index]  = Geometry::CheckPoint(gt, ip, -btol) ? 0 : 1;
      }
   }
}

struct evalSrcPt_t
{
   double r[3];
   unsigned int index, proc, el;
};

struct evalOutPt_t
{
   double out;
   unsigned int index, proc;
};

void FindPointsGSLIB::InterpolateOnDevice(const Vector &field_in,
                                          Vector &field_out,
                                          const int nel,
                                          const int ncomp,
                                          const int dof1Dsol,
                                          const int gf_ordering,
                                          MemoryType mt)
{
   struct gslib::array src, outpt;
   int nlocal = 0;
   /* weed out unfound points, send out */
   Array<int> gsl_elem_temp;
   Vector gsl_ref_temp;
   Array<int> index_temp;
   {
      int index;
      const unsigned int *code = gsl_code.GetData(), *proc = gsl_proc.GetData(),
                          *el   = gsl_elem.GetData();
      const dfloat *r = gsl_ref.GetData();

      int numSend = 0;

      for (index = 0; index < points_cnt; ++index)
      {
         numSend += (gsl_code[index] != CODE_NOT_FOUND &&
                     gsl_proc[index] != gsl_comm->id);
         nlocal += (gsl_code[index] != CODE_NOT_FOUND &&
                    gsl_proc[index] == gsl_comm->id);
      }

      gsl_elem_temp.SetSize(nlocal);

      gsl_ref_temp.SetSize(nlocal*dim);
      gsl_ref_temp.UseDevice(true);
      gsl_ref_temp.HostReadWrite();

      index_temp.SetSize(nlocal);

      evalSrcPt_t *pt;
      array_init(evalSrcPt_t, &src, numSend);
      pt = (evalSrcPt_t *)src.ptr;

      int ctr = 0;
      for (index = 0; index < points_cnt; ++index)
      {
         if (*code != CODE_NOT_FOUND && *proc != gsl_comm->id)
         {
            for (int d = 0; d < dim; ++d)
            {
               pt->r[d] = r[d];
            }
            pt->index = index;
            pt->proc = *proc;
            pt->el = *el;
            ++pt;
         }
         else if (*code != CODE_NOT_FOUND && *proc == gsl_comm->id)
         {
            gsl_elem_temp[ctr] = *el;
            for (int d = 0; d < dim; ++d)
            {
               gsl_ref_temp(dim*ctr+d) = r[d];
            }
            index_temp[ctr] = index;

            ctr++;
         }
         r += dim;
         code++;
         proc++;
         el++;
      }

      src.n = pt - (evalSrcPt_t *)src.ptr;
      sarray_transfer(evalSrcPt_t, &src, proc, 1, cr);
   }

   //evaluate points that are already local
   {
      Vector interp_vals(nlocal*ncomp);
      interp_vals.UseDevice(true);

      InterpolateLocal(field_in,
                       gsl_elem_temp,
                       gsl_ref_temp,
                       interp_vals,
                       nlocal, ncomp,
                       nel, dof1Dsol);
      MPI_Barrier(gsl_comm->c);

      interp_vals.HostReadWrite();

      // now put these in correct positions
      int interp_Offset = interp_vals.Size()/ncomp;
      for (int i = 0; i < ncomp; i++)
      {
         for (int j = 0; j < nlocal; j++)
         {
            int pt_index = index_temp[j];
            int idx = gf_ordering == Ordering::byNODES ?
                      pt_index + i*points_cnt :
                      pt_index*ncomp + i;
            field_out(idx) = interp_vals(j + interp_Offset*i);
         }
      }
   }
   MPI_Barrier(gsl_comm->c);

   if (gsl_comm->np == 1)
   {
      array_free(&src);
      return;
   }

   // evaluate points locally
   {
      int n = src.n;
      const evalSrcPt_t *spt;
      evalOutPt_t *opt;
      spt = (evalSrcPt_t *)src.ptr;

      // Copy to host vector
      Array<int> gsl_elem_temp(n);
      gsl_ref_temp.SetSize(n*dim);
      gsl_ref_temp.HostReadWrite();

      spt = (evalSrcPt_t *)src.ptr;
      //   opt = (evalOutPt_t *)outpt.ptr;
      for (int i = 0; i < n; i++, ++spt)
      {
         gsl_elem_temp[i] = spt->el;
         for (int d = 0; d < dim; d++)
         {
            gsl_ref_temp(i*dim + d) = spt->r[d];
         }
      }

      Vector interp_vals(n*ncomp);
      interp_vals.UseDevice(true);
      InterpolateLocal(field_in,
                       gsl_elem_temp,
                       gsl_ref_temp,
                       interp_vals, n, ncomp,
                       nel, dof1Dsol);
      MPI_Barrier(gsl_comm->c);
      interp_vals.HostReadWrite();

      field_out.HostWrite();
      // Now the interpolated values need to be sent back component wise
      int Offset = interp_vals.Size()/ncomp;
      for (int i = 0; i < ncomp; i++)
      {
         spt = (evalSrcPt_t *)src.ptr;
         array_init(evalOutPt_t, &outpt, n);
         outpt.n = n;
         opt = (evalOutPt_t *)outpt.ptr;

         for (int j = 0; j < n; j++)
         {
            opt->index = spt->index;
            opt->proc = spt->proc;
            opt->out = interp_vals(j + Offset*i);
            spt++;
            opt++;
         }

         sarray_transfer(struct evalOutPt_t, &outpt, proc, 1, cr);

         opt = (evalOutPt_t *)outpt.ptr;
         for (int index = 0; index < outpt.n; index++)
         {
            int idx = gf_ordering == Ordering::byNODES ?
                      opt->index + i*points_cnt :
                      opt->index*ncomp + i;
            field_out(idx) = opt->out;
            ++opt;
         }
         array_free(&outpt);
      }
      array_free(&src);
   }
   //finished evaluating points received from other processors.
}

void FindPointsGSLIB::Interpolate(const GridFunction &field_in,
                                  Vector &field_out)
{
   const int  gf_order   = field_in.FESpace()->GetMaxElementOrder(),
              mesh_order = mesh->GetNodalFESpace()->GetMaxElementOrder();

   const FiniteElementCollection *fec_in =  field_in.FESpace()->FEColl();
   const H1_FECollection *fec_h1 = dynamic_cast<const H1_FECollection *>(fec_in);
   const L2_FECollection *fec_l2 = dynamic_cast<const L2_FECollection *>(fec_in);

   setupSW.Clear();
   setupSW.Start();
   if (Device::IsEnabled() && field_in.UseDevice())
   {
      MFEM_VERIFY(fec_h1,"Only h1 functions supported on device right now.");
      MFEM_VERIFY(fec_h1->GetBasisType() == BasisType::GaussLobatto,
                  "basis not supported");
      Vector node_vals;
      GetNodalValues(&field_in, node_vals);

      Vector node_vals_dev;
      node_vals_dev.UseDevice(true);
      node_vals_dev.SetSize(node_vals.Size());
      internal::device_copy(node_vals_dev.Write(), node_vals.GetData(),
                            node_vals.Size());

      const int ncomp  = field_in.FESpace()->GetVDim();
      const int maxOrder = field_in.FESpace()->GetMaxElementOrder();
      DEV.dof1dsol =  maxOrder+1;

      DEV.gll1dsol.UseDevice();
      DEV.gll1dsol.SetSize(DEV.dof1dsol);
      DEV.lagcoeffsol.UseDevice();
      DEV.lagcoeffsol.SetSize(DEV.dof1dsol);
      if (DEV.dof1dsol != DEV.dof1d)
      {
         Vector temp(DEV.dof1dsol);

         gslib::lobatto_nodes(temp.GetData(), DEV.dof1dsol);
         internal::device_copy(DEV.gll1dsol.Write(), temp.GetData(), DEV.dof1dsol);

         gslib::gll_lag_setup(temp.GetData(), DEV.dof1dsol);
         internal::device_copy(DEV.lagcoeffsol.Write(), temp.GetData(), DEV.dof1dsol);
      }
      else
      {
         internal::device_copy(DEV.gll1dsol.ReadWrite(), DEV.gll1d.Read(), DEV.dof1dsol);
         internal::device_copy(DEV.lagcoeffsol.ReadWrite(), DEV.lagcoeff.Read(),
                               DEV.dof1dsol);
      }

      field_out.SetSize(points_cnt*ncomp);

      // At this point make sure FindPoints was called with device mode?
      // Otherwise copy necessary data?
      InterpolateOnDevice(node_vals, field_out, NE_split_total, ncomp,
                          DEV.dof1dsol,
                          field_in.FESpace()->GetOrdering(),
                          field_in.GetMemory().GetMemoryType());

      setupSW.Stop();
      interpolate_h1_time = setupSW.RealTime();
      return;
   }


   if (fec_h1 && gf_order == mesh_order &&
       fec_h1->GetBasisType() == BasisType::GaussLobatto &&
       field_in.FESpace()->IsVariableOrder() ==
       mesh->GetNodalFESpace()->IsVariableOrder())
   {
      InterpolateH1(field_in, field_out);
      setupSW.Stop();
      interpolate_h1_time = setupSW.RealTime();
      return;
   }
   else
   {
      InterpolateGeneral(field_in, field_out);
      if (!fec_l2 || avgtype == AvgType::NONE) { return; }
   }

   // For points on element borders, project the L2 GridFunction to H1 and
   // re-interpolate.
   if (fec_l2)
   {
      Array<int> indl2;
      for (int i = 0; i < points_cnt; i++)
      {
         if (gsl_code[i] == 1) { indl2.Append(i); }
      }
      int borderPts = indl2.Size();
#ifdef MFEM_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &borderPts, 1, MPI_INT, MPI_SUM, gsl_comm->c);
#endif
      if (borderPts == 0) { return; } // no points on element borders

      Vector field_out_l2(field_out.Size());
      VectorGridFunctionCoefficient field_in_dg(&field_in);
      int gf_order_h1 = std::max(gf_order, 1); // H1 should be at least order 1
      H1_FECollection fec(gf_order_h1, dim);
      const int ncomp = field_in.FESpace()->GetVDim();
      FiniteElementSpace fes(mesh, &fec, ncomp);
      GridFunction field_in_h1(&fes);

      if (avgtype == AvgType::ARITHMETIC)
      {
         field_in_h1.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);
      }
      else if (avgtype == AvgType::HARMONIC)
      {
         field_in_h1.ProjectDiscCoefficient(field_in_dg, GridFunction::HARMONIC);
      }
      else
      {
         MFEM_ABORT("Invalid averaging type.");
      }

      if (gf_order_h1 == mesh_order) // basis is GaussLobatto by default
      {
         InterpolateH1(field_in_h1, field_out_l2);
      }
      else
      {
         InterpolateGeneral(field_in_h1, field_out_l2);
      }

      // Copy interpolated values for the points on element border
      for (int j = 0; j < ncomp; j++)
      {
         for (int i = 0; i < indl2.Size(); i++)
         {
            int idx = field_in.FESpace()->GetOrdering() == Ordering::byNODES ?
                      indl2[i] + j*points_cnt:
                      indl2[i]*ncomp + j;
            field_out(idx) = field_out_l2(idx);
         }
      }
   }
}

void FindPointsGSLIB::InterpolateH1(const GridFunction &field_in,
                                    Vector &field_out)
{
   FiniteElementSpace ind_fes(mesh, field_in.FESpace()->FEColl());
   if (field_in.FESpace()->IsVariableOrder())
   {
      for (int e = 0; e < ind_fes.GetMesh()->GetNE(); e++)
      {
         ind_fes.SetElementOrder(e, field_in.FESpace()->GetElementOrder(e));
      }
      ind_fes.Update(false);
   }
   GridFunction field_in_scalar(&ind_fes);
   Vector node_vals;

   const int ncomp      = field_in.FESpace()->GetVDim(),
             points_fld = field_in.Size() / ncomp;
   MFEM_VERIFY(points_cnt == gsl_code.Size(),
               "FindPointsGSLIB::InterpolateH1: Inconsistent size of gsl_code");

   field_out.SetSize(points_cnt*ncomp);
   field_out = default_interp_value;

   for (int i = 0; i < ncomp; i++)
   {
      const int dataptrin  = i*points_fld,
                dataptrout = i*points_cnt;
      if (field_in.FESpace()->GetOrdering() == Ordering::byNODES)
      {
         field_in_scalar.NewDataAndSize(field_in.GetData()+dataptrin, points_fld);
      }
      else
      {
         for (int j = 0; j < points_fld; j++)
         {
            field_in_scalar(j) = field_in(i + j*ncomp);
         }
      }
      GetNodalValues(&field_in_scalar, node_vals);

      if (dim==2)
      {
         findpts_eval_2(field_out.GetData()+dataptrout, sizeof(double),
                        gsl_code.GetData(),       sizeof(unsigned int),
                        gsl_proc.GetData(),    sizeof(unsigned int),
                        gsl_elem.GetData(),    sizeof(unsigned int),
                        gsl_ref.GetData(),     sizeof(double) * dim,
                        points_cnt, node_vals.GetData(), fdata2D);
      }
      else
      {
         findpts_eval_3(field_out.GetData()+dataptrout, sizeof(double),
                        gsl_code.GetData(),       sizeof(unsigned int),
                        gsl_proc.GetData(),    sizeof(unsigned int),
                        gsl_elem.GetData(),    sizeof(unsigned int),
                        gsl_ref.GetData(),     sizeof(double) * dim,
                        points_cnt, node_vals.GetData(), (gslib::findpts_data_3 *)this->fdataD);
      }
   }
   if (field_in.FESpace()->GetOrdering() == Ordering::byVDIM)
   {
      Vector field_out_temp = field_out;
      for (int i = 0; i < ncomp; i++)
      {
         for (int j = 0; j < points_cnt; j++)
         {
            field_out(i + j*ncomp) = field_out_temp(j + i*points_cnt);
         }
      }
   }
}

void FindPointsGSLIB::InterpolateGeneral(const GridFunction &field_in,
                                         Vector &field_out)
{
   int ncomp   = field_in.VectorDim(),
       nptorig = points_cnt,
       npt     = points_cnt;

   field_out.SetSize(points_cnt*ncomp);
   field_out = default_interp_value;

   if (gsl_comm->np == 1) // serial
   {
      for (int index = 0; index < npt; index++)
      {
         if (gsl_code[index] == 2) { continue; }
         IntegrationPoint ip;
         ip.Set2(gsl_mfem_ref.GetData()+index*dim);
         if (dim == 3) { ip.z = gsl_mfem_ref(index*dim + 2); }
         Vector localval(ncomp);
         field_in.GetVectorValue(gsl_mfem_elem[index], ip, localval);
         if (field_in.FESpace()->GetOrdering() == Ordering::byNODES)
         {
            for (int i = 0; i < ncomp; i++)
            {
               field_out(index + i*npt) = localval(i);
            }
         }
         else //byVDIM
         {
            for (int i = 0; i < ncomp; i++)
            {
               field_out(index*ncomp + i) = localval(i);
            }
         }
      }
   }
   else // parallel
   {
      // Determine number of points to be sent
      int nptsend = 0;
      for (int index = 0; index < npt; index++)
      {
         if (gsl_code[index] != 2) { nptsend +=1; }
      }

      // Pack data to send via crystal router
      struct gslib::array *outpt = new gslib::array;
      struct out_pt { double r[3], ival; uint index, el, proc; };
      struct out_pt *pt;
      array_init(struct out_pt, outpt, nptsend);
      outpt->n=nptsend;
      pt = (struct out_pt *)outpt->ptr;
      for (int index = 0; index < npt; index++)
      {
         if (gsl_code[index] == 2) { continue; }
         for (int d = 0; d < dim; ++d) { pt->r[d]= gsl_mfem_ref(index*dim + d); }
         pt->index = index;
         pt->proc  = gsl_proc[index];
         pt->el    = gsl_mfem_elem[index];
         ++pt;
      }

      // Transfer data to target MPI ranks
      sarray_transfer(struct out_pt, outpt, proc, 1, cr);

      if (ncomp == 1)
      {
         // Interpolate the grid function
         npt = outpt->n;
         pt = (struct out_pt *)outpt->ptr;
         for (int index = 0; index < npt; index++)
         {
            IntegrationPoint ip;
            ip.Set3(&pt->r[0]);
            pt->ival = field_in.GetValue(pt->el, ip, 1);
            ++pt;
         }

         // Transfer data back to source MPI rank
         sarray_transfer(struct out_pt, outpt, proc, 1, cr);
         npt = outpt->n;
         pt = (struct out_pt *)outpt->ptr;
         for (int index = 0; index < npt; index++)
         {
            field_out(pt->index) = pt->ival;
            ++pt;
         }
         array_free(outpt);
         delete outpt;
      }
      else // ncomp > 1
      {
         // Interpolate data and store in a Vector
         npt = outpt->n;
         pt = (struct out_pt *)outpt->ptr;
         Vector vec_int_vals(npt*ncomp);
         for (int index = 0; index < npt; index++)
         {
            IntegrationPoint ip;
            ip.Set3(&pt->r[0]);
            Vector localval(vec_int_vals.GetData()+index*ncomp, ncomp);
            field_in.GetVectorValue(pt->el, ip, localval);
            ++pt;
         }

         // Save index and proc data in a struct
         struct gslib::array *savpt = new gslib::array;
         struct sav_pt { uint index, proc; };
         struct sav_pt *spt;
         array_init(struct sav_pt, savpt, npt);
         savpt->n=npt;
         spt = (struct sav_pt *)savpt->ptr;
         pt  = (struct out_pt *)outpt->ptr;
         for (int index = 0; index < npt; index++)
         {
            spt->index = pt->index;
            spt->proc  = pt->proc;
            ++pt; ++spt;
         }

         array_free(outpt);
         delete outpt;

         // Copy data from save struct to send struct and send component wise
         struct gslib::array *sendpt = new gslib::array;
         struct send_pt { double ival; uint index, proc; };
         struct send_pt *sdpt;
         for (int j = 0; j < ncomp; j++)
         {
            array_init(struct send_pt, sendpt, npt);
            sendpt->n=npt;
            spt  = (struct sav_pt *)savpt->ptr;
            sdpt = (struct send_pt *)sendpt->ptr;
            for (int index = 0; index < npt; index++)
            {
               sdpt->index = spt->index;
               sdpt->proc  = spt->proc;
               sdpt->ival  = vec_int_vals(j + index*ncomp);
               ++sdpt; ++spt;
            }

            sarray_transfer(struct send_pt, sendpt, proc, 1, cr);
            sdpt = (struct send_pt *)sendpt->ptr;
            for (int index = 0; index < static_cast<int>(sendpt->n); index++)
            {
               int idx = field_in.FESpace()->GetOrdering() == Ordering::byNODES ?
                         sdpt->index + j*nptorig :
                         sdpt->index*ncomp + j;
               field_out(idx) = sdpt->ival;
               ++sdpt;
            }
            array_free(sendpt);
         }
         array_free(savpt);
         delete sendpt;
         delete savpt;
      } // ncomp > 1
   } // parallel
}

void OversetFindPointsGSLIB::Setup(Mesh &m, const int meshid,
                                   GridFunction *gfmax,
                                   const double bb_t, const double newt_tol,
                                   const int npt_max)
{
   MFEM_VERIFY(m.GetNodes() != NULL, "Mesh nodes are required.");
   const int meshOrder = m.GetNodes()->FESpace()->GetMaxElementOrder();

   // FreeData if OversetFindPointsGSLIB::Setup has been called already
   if (setupflag) { FreeData(); }

   crystal_init(cr, gsl_comm);
   mesh = &m;
   dim  = mesh->Dimension();
   const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(0);
   unsigned dof1D = fe->GetOrder() + 1;

   SetupSplitMeshes();
   if (dim == 2)
   {
      if (ir_split[0]) { delete ir_split[0]; ir_split[0] = NULL; }
      ir_split[0] = new IntegrationRule(3*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[0], ir_split[0], meshOrder);

      if (ir_split[1]) { delete ir_split[1]; ir_split[1] = NULL; }
      ir_split[1] = new IntegrationRule(pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[1], ir_split[1], meshOrder);
   }
   else if (dim == 3)
   {
      if (ir_split[0]) { delete ir_split[0]; ir_split[0] = NULL; }
      ir_split[0] = new IntegrationRule(pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[0], ir_split[0], meshOrder);

      if (ir_split[1]) { delete ir_split[1]; ir_split[1] = NULL; }
      ir_split[1] = new IntegrationRule(4*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[1], ir_split[1], meshOrder);

      if (ir_split[2]) { delete ir_split[2]; ir_split[2] = NULL; }
      ir_split[2] = new IntegrationRule(3*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[2], ir_split[2], meshOrder);

      if (ir_split[3]) { delete ir_split[3]; ir_split[3] = NULL; }
      ir_split[3] = new IntegrationRule(8*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[3], ir_split[3], meshOrder);
   }

   GetNodalValues(mesh->GetNodes(), gsl_mesh);

   MFEM_ASSERT(meshid>=0, " The ID should be greater than or equal to 0.");

   const int pts_cnt = gsl_mesh.Size()/dim,
             NEtot = pts_cnt/(int)pow(dof1D, dim);

   distfint.SetSize(pts_cnt);
   if (!gfmax)
   {
      distfint = 0.0;
   }
   else
   {
      GetNodalValues(gfmax, distfint);
   }
   u_meshid = (unsigned int)meshid;

   if (dim == 2)
   {
      unsigned nr[2] = { dof1D, dof1D };
      unsigned mr[2] = { 2*dof1D, 2*dof1D };
      double * const elx[2] = { &gsl_mesh(0), &gsl_mesh(pts_cnt) };
      fdata2D = findptsms_setup_2(gsl_comm, elx, nr, NEtot, mr, bb_t,
                                  pts_cnt, pts_cnt, npt_max, newt_tol,
                                  &u_meshid, &distfint(0));
   }
   else
   {
      unsigned nr[3] = { dof1D, dof1D, dof1D };
      unsigned mr[3] = { 2*dof1D, 2*dof1D, 2*dof1D };
      double * const elx[3] =
      { &gsl_mesh(0), &gsl_mesh(pts_cnt), &gsl_mesh(2*pts_cnt) };
      fdataD = findptsms_setup_3(gsl_comm, elx, nr, NEtot, mr, bb_t,
                                 pts_cnt, pts_cnt, npt_max, newt_tol,
                                 &u_meshid, &distfint(0));
   }
   setupflag = true;
   overset   = true;
}

void OversetFindPointsGSLIB::FindPoints(const Vector &point_pos,
                                        Array<unsigned int> &point_id,
                                        int point_pos_ordering)
{
   MFEM_VERIFY(setupflag, "Use OversetFindPointsGSLIB::Setup before "
               "finding points.");
   MFEM_VERIFY(overset, " Please setup FindPoints for overlapping grids.");
   points_cnt = point_pos.Size() / dim;
   unsigned int match = 0; // Don't find points in the mesh if point_id = mesh_id

   gsl_code.SetSize(points_cnt);
   gsl_proc.SetSize(points_cnt);
   gsl_elem.SetSize(points_cnt);
   gsl_ref.SetSize(points_cnt * dim);
   gsl_dist.SetSize(points_cnt);

   auto xvFill = [&](const double *xv_base[], unsigned xv_stride[])
   {
      for (int d = 0; d < dim; d++)
      {
         if (point_pos_ordering == Ordering::byNODES)
         {
            xv_base[d] = point_pos.GetData() + d*points_cnt;
            xv_stride[d] = sizeof(double);
         }
         else
         {
            xv_base[d] = point_pos.GetData() + d;
            xv_stride[d] = dim*sizeof(double);
         }
      }
   };
   if (dim == 2)
   {
      const double *xv_base[2];
      unsigned xv_stride[2];
      xvFill(xv_base, xv_stride);
      findptsms_2(gsl_code.GetData(), sizeof(unsigned int),
                  gsl_proc.GetData(), sizeof(unsigned int),
                  gsl_elem.GetData(), sizeof(unsigned int),
                  gsl_ref.GetData(),  sizeof(double) * dim,
                  gsl_dist.GetData(), sizeof(double),
                  xv_base,            xv_stride,
                  point_id.GetData(), sizeof(unsigned int), &match,
                  points_cnt, fdata2D);
   }
   else  // dim == 3
   {
      const double *xv_base[3];
      unsigned xv_stride[3];
      xvFill(xv_base, xv_stride);
      findptsms_3(gsl_code.GetData(), sizeof(unsigned int),
                  gsl_proc.GetData(), sizeof(unsigned int),
                  gsl_elem.GetData(), sizeof(unsigned int),
                  gsl_ref.GetData(),  sizeof(double) * dim,
                  gsl_dist.GetData(), sizeof(double),
                  xv_base,            xv_stride,
                  point_id.GetData(), sizeof(unsigned int), &match,
                  points_cnt, (gslib::findpts_data_3 *)this->fdataD);
   }

   // Set the element number and reference position to 0 for points not found
   for (int i = 0; i < points_cnt; i++)
   {
      if (gsl_code[i] == 2 ||
          (gsl_code[i] == 1 && gsl_dist(i) > bdr_tol))
      {
         gsl_elem[i] = 0;
         for (int d = 0; d < dim; d++) { gsl_ref(i*dim + d) = -1.; }
         gsl_code[i] = 2;
      }
   }

   // Map element number for simplices, and ref_pos from [-1,1] to [0,1] for both
   // simplices and quads.
   MapRefPosAndElemIndices();
}

void OversetFindPointsGSLIB::Interpolate(const Vector &point_pos,
                                         Array<unsigned int> &point_id,
                                         const GridFunction &field_in,
                                         Vector &field_out,
                                         int point_pos_ordering)
{
   FindPoints(point_pos, point_id, point_pos_ordering);
   Interpolate(field_in, field_out);
}


} // namespace mfem
#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND
#undef dlong
#undef dfloat

#endif // MFEM_USE_GSLIB
