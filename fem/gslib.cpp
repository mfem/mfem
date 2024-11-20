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

   struct hash_data_2
   {
      ulong hash_n;
      struct dbl_range bnd[2];
      double fac[2];
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

   struct findpts_data_2
   {
      struct crystal cr;
      struct findpts_local_data_2 local;
      struct hash_data_2 hash;
      struct array savpt;
      struct findpts_dummy_ms_data fdms;
      uint   fevsetup;
   };
} //extern C

} //namespace gslib

#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic pop
#endif

namespace mfem
{

FindPointsGSLIB::FindPointsGSLIB()
   : mesh(NULL),
     fec_map_lin(NULL),
     fdataD(NULL), cr(NULL), gsl_comm(NULL),
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
   comm_free(gsl_comm);
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
     fdataD(NULL), cr(NULL), gsl_comm(NULL),
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
   spacedim = mesh->SpaceDimension();
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
      double * const elx[2] =
      {
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(0),
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(mesh_points_cnt)
      };
      fdataD = findpts_setup_2(gsl_comm, elx, nr, NE_split_total, mr, bb_t,
                               DEV.local_hash_size,
                               mesh_points_cnt, npt_max, newt_tol);
   }
   else
   {
      unsigned nr[3] = { dof1D, dof1D, dof1D };
      unsigned mr[3] = { 2*dof1D, 2*dof1D, 2*dof1D };
      double * const elx[3] =
      {
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(0),
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(mesh_points_cnt),
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(2*mesh_points_cnt)
      };
      fdataD = findpts_setup_3(gsl_comm, elx, nr, NE_split_total, mr, bb_t,
                               DEV.local_hash_size,
                               mesh_points_cnt, npt_max, newt_tol);
   }
   setupSW.Stop();
   setup_findpts_setup_time = setupSW.RealTime();
   setupflag = true;
}

void FindPointsGSLIB::SetupSurf(Mesh &m, const double bb_t,
                                const double newt_tol,
                                const int npt_max)
{
   const int num_procs = Mpi::WorldSize();
   const int myid      = Mpi::WorldRank();

   // EnsureNodes call could be useful if the mesh is 1st order and has no gridfunction defined
   MFEM_VERIFY(m.GetNodes() != NULL, "Mesh nodes are required.");

   crystal_init(cr, gsl_comm);     // Crystal Router
   mesh     = &m;
   dim      = mesh->Dimension();       // This is reference dimension
   spacedim = mesh->SpaceDimension();  // This is physical dimension

   // Max element order since we will ultimately have all elements of the same order
   const int meshOrder = mesh->GetNodes()->FESpace()->GetMaxElementOrder();
   unsigned dof1D      = meshOrder +
                         1;         // dof in one dimension based on max mesh order (since that's the order we we will have ultimately)
   unsigned pts_el     = std::pow(dof1D, dim);  // Number of points in an element

   // call FreeData if FindPointsGSLIB::Setup has been called already
   if (setupflag) { FreeData(); }

   setupSW.Clear();
   setupSW.Start();

   SetupSplitMeshesSurf();  // A call to set NE_split_total, _index, _map arrays, NOT PRODUCTION READY

   setupSW.Stop();
   setup_split_time = setupSW.RealTime();

   setupSW.Clear();
   setupSW.Start();

   GetNodalValuesSurf(mesh->GetNodes(), gsl_mesh);

   setupSW.Stop();
   setup_nodalmapping_time = setupSW.RealTime();

   mesh_points_cnt     = gsl_mesh.Size()/spacedim;
   DEV.local_hash_size = mesh_points_cnt;
   DEV.dof1d           = (int)dof1D;

   setupSW.Clear();
   setupSW.Start();

   if (spacedim==2)
   {
      unsigned nr[1] = { dof1D };
      unsigned mr[1] = { 2*dof1D };
      double * const elx[2] =
      {
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(0),
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(mesh_points_cnt)
      };
      fdataD = findptssurf_setup_2( gsl_comm,
                                    elx,
                                    nr,
                                    NE_split_total,
                                    mr,
                                    bb_t,
                                    DEV.local_hash_size,
                                    mesh_points_cnt,
                                    npt_max,
                                    newt_tol );
      setupflag = true;
   }
   else if (spacedim==3)
   {
      unsigned nr[2] = { dof1D, dof1D };
      unsigned mr[2] = { 2*dof1D, 2*dof1D };
      double * const elx[3] =
      {
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(0),
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(mesh_points_cnt),
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(2*mesh_points_cnt)
      };
      fdataD = findptssurf_setup_3( gsl_comm,
                                    elx,
                                    nr,
                                    NE_split_total,
                                    mr,
                                    bb_t,
                                    DEV.local_hash_size,
                                    mesh_points_cnt,
                                    npt_max,
                                    newt_tol );
      setupflag = true;
   }
   setupSW.Stop();
   setup_findpts_setup_time = setupSW.RealTime();
}

void FindPointsGSLIB::FindPoints(const Vector &point_pos,
                                 int point_pos_ordering)
{
   MFEM_VERIFY(setupflag, "Use FindPointsGSLIB::Setup before finding points.");
   bool dev_mode = (point_pos.UseDevice() && Device::IsEnabled());
   points_cnt = point_pos.Size() / dim;
   gsl_code.SetSize(points_cnt);
   gsl_proc.SetSize(points_cnt);
   gsl_elem.SetSize(points_cnt);
   gsl_ref.SetSize(points_cnt*dim);
   gsl_dist.SetSize(points_cnt);
   gsl_newton.SetSize(points_cnt);

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
         auto *findptsData = (gslib::findpts_data_2 *)this->fdataD;
         const double *xv_base[2];
         unsigned xv_stride[2];
         xvFill(xv_base, xv_stride);
         findpts_2(gsl_code.GetData(), sizeof(unsigned int),
                   gsl_proc.GetData(), sizeof(unsigned int),
                   gsl_elem.GetData(), sizeof(unsigned int),
                   gsl_ref.GetData(),  sizeof(double) * dim,
                   gsl_dist.GetData(), sizeof(double),
                   xv_base, xv_stride, points_cnt, findptsData);
      }
      else  // dim == 3
      {
         auto *findptsData = (gslib::findpts_data_3 *)this->fdataD;
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

void FindPointsGSLIB::FindPointsSurf( const Vector &point_pos,
                                      int point_pos_ordering )
{
   MFEM_VERIFY(setupflag, "Use FindPointsGSLIB::Setup before finding points.");
   // bool dev_mode = point_pos.UseDevice();
   points_cnt = point_pos.Size()/spacedim;

   gsl_code.SetSize(points_cnt);
   gsl_proc.SetSize(points_cnt);
   gsl_elem.SetSize(points_cnt);
   gsl_ref.SetSize(points_cnt*dim);  // stores best guess ref pos. for each point
   gsl_dist.SetSize(points_cnt);
   gsl_newton.SetSize(points_cnt);

   setupSW.Clear();
   setupSW.Start();

   FindPointsSurfOnDevice(point_pos, point_pos_ordering);

   setupSW.Stop();
   findpts_findpts_time = setupSW.RealTime();

   setupSW.Clear();
   setupSW.Start();

   // Set the element number and reference position to 0 for points not found
   for (int i=0; i<points_cnt; i++)
   {
      if ( gsl_code[i]==2 || (gsl_code[i]==1 && gsl_dist(i)>bdr_tol) )
      {
         gsl_elem[i] = 0;
         for (int d=0; d<dim; d++)
         {
            gsl_ref(i*dim + d) = -1.;
         }
         gsl_code[i] = 2;
      }
   }

   // Map element number for simplices, and ref_pos from [-1,1] to [0,1] for
   // both simplices and quads. Also sets code to 1 for points found on element
   // faces/edges.
   MapRefPosAndElemIndicesSurf();

   setupSW.Stop();
   findpts_mapelemrst_time = setupSW.RealTime();
}

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
           hash_index_aux(p->bnd[1].min, p->fac[1], n, x[1])) * n +
          hash_index_aux(p->bnd[0].min, p->fac[0], n, x[0]);
}

static ulong hash_index_2(const gslib::hash_data_2 *p, const dfloat x[2])
{
   const ulong n = p->hash_n;
   return (hash_index_aux(p->bnd[1].min, p->fac[1], n, x[1])) * n
          + hash_index_aux(p->bnd[0].min, p->fac[0], n, x[0]);
}

void FindPointsGSLIB::FindPointsOnDevice(const Vector &point_pos,
                                         int point_pos_ordering)
{
   point_pos.HostRead();
   Vector point_pos_copy = point_pos;
   point_pos_copy.UseDevice(true);
   point_pos_copy.HostReadWrite();
   MemoryType mt = point_pos.GetMemory().GetMemoryType();
   SW2.Clear();
   SW2.Start();
   SetupDevice(mt);
   SW2.Stop();
   findpts_setup_device_arrays_time = SW2.RealTime();

   gsl_ref.UseDevice(true);
   gsl_dist.UseDevice(true);

   if (dim == 2)
   {
      FindPointsLocal2(point_pos,
                       point_pos_ordering,
                       gsl_code,
                       gsl_elem,
                       gsl_ref,
                       gsl_dist,
                       gsl_newton,
                       points_cnt);
   }
   else
   {
      FindPointsLocal32(point_pos_copy,
                        point_pos_ordering,
                        gsl_code,
                        gsl_elem,
                        gsl_ref,
                        gsl_dist,
                        gsl_newton,
                        points_cnt);
   }

   // Sync from device to host
   gsl_ref.HostReadWrite();
   gsl_dist.HostReadWrite();
   point_pos.HostRead();
   DEV.info.HostReadWrite();
   gsl_newton.HostReadWrite();

   gsl_code.HostReadWrite();
   gsl_elem.HostReadWrite();
   gsl_proc.HostReadWrite();


   // TODO: Only transfer information for points found
   // Transfer information from DEVICE
   const int myid = gsl_comm->id;
   for (int i = 0; i < points_cnt; i++)
   {
      gsl_proc[i] = myid;
   }

   const int id = gsl_comm->id,
             np = gsl_comm->np;

   if (np == 1)
   {
      return;
   }

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
      int newton;
   };

   int find_elsewhere = 0;
   int found_local = 0;
   int hashptn = 0;
   {
      int index;
      auto *code = gsl_code.HostReadWrite();
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
         if (*code != CODE_INTERNAL)
         {
            find_elsewhere++;
            const auto hi = dim == 2 ? hash_index_2(DEV.hash2, x) :
                            hash_index_3(DEV.hash3, x);
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
      }
      hash_pt.n = pt - (struct srcPt_t *)hash_pt.ptr;
      sarray_transfer(struct srcPt_t, &hash_pt, proc, 1, DEV.cr);
      hashptn = hash_pt.n;
   }
   MPI_Barrier(gsl_comm->c);

   /* look up points in hash cells, route to possible procs */
   {
      const unsigned int *const hash_offset = dim == 2 ? DEV.hash2->offset :
                                              DEV.hash3->offset;
      int count = 0, *proc, *proc_p;
      const struct srcPt_t *p = (struct srcPt_t *)hash_pt.ptr,
                            *const pe = p + hash_pt.n;
      struct srcPt_t *q;

      for (; p != pe; ++p)
      {
         const int hi = dim == 2 ? hash_index_2(DEV.hash2, p->x)/np :
                        hash_index_3(DEV.hash3, p->x)/np;
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
         const int hi = dim == 2 ? hash_index_2(DEV.hash2, p->x)/np :
                        hash_index_3(DEV.hash3, p->x)/np;
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
      Vector gsl_ref_l, gsl_dist_l;
      gsl_ref_l.UseDevice(true);
      gsl_dist_l.UseDevice(true);
      gsl_ref_l.SetSize(n*dim);
      gsl_dist_l.SetSize(n);
      Vector point_pos_l;
      point_pos_l.UseDevice(true);
      point_pos_l.SetSize(n*dim);
      auto pointl = point_pos_l.HostWrite();
      Array<int> gsl_newton_l;
      gsl_newton_l.SetSize(n);

      Array<unsigned int> gsl_code_l, gsl_elem_l;
      gsl_code_l.SetSize(n);
      gsl_elem_l.SetSize(n);


      for (int point = 0; point < n; ++point)
      {
         for (int d = 0; d < dim; d++)
         {
            int idx = point_pos_ordering == 0 ? point + d*n : point*dim + d;
            pointl[idx] = spt[point].x[d];
         }
      }

      if (dim == 2)
      {
         FindPointsLocal2(point_pos_l,
                          point_pos_ordering,
                          gsl_code_l,
                          gsl_elem_l,
                          gsl_ref_l,
                          gsl_dist_l,
                          gsl_newton_l,
                          n);
      }
      else
      {
         if (gpu_code == 1)
         {
            FindPointsLocal32(point_pos_l,
                              point_pos_ordering,
                              gsl_code_l,
                              gsl_elem_l,
                              gsl_ref_l,
                              gsl_dist_l,
                              gsl_newton_l,
                              n);
         }
         else
         {
            FindPointsLocal3(point_pos_l,
                             point_pos_ordering,
                             gsl_code_l,
                             gsl_elem_l,
                             gsl_ref_l,
                             gsl_dist_l,
                             gsl_newton_l,
                             n);
         }
      }

      auto refl = gsl_ref_l.HostRead();
      auto distl = gsl_dist_l.HostRead();
      auto codel = gsl_code_l.HostRead();
      auto eleml = gsl_elem_l.HostRead();
      auto newtl = gsl_newton_l.HostRead();
      // DEV.info.HostRead();

      // unpack arrays into opt
      for (int point = 0; point < n; point++)
      {
         opt[point].code = codel[point];
         opt[point].el   = eleml[point];
         opt[point].dist2 = distl[point];
         for (int d = 0; d < dim; ++d)
         {
            opt[point].r[d] = refl[dim * point + d];
         }
         opt->newton = newtl[point];
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

   auto code = gsl_code.HostReadWrite();
   auto elem = gsl_elem.HostReadWrite();
   auto proc = gsl_proc.HostReadWrite();
   auto dist = gsl_dist.HostReadWrite();
   auto newton = gsl_newton.HostReadWrite();

   /* merge remote results with user data */
   int npt_found_on_other_proc = 0;
   {
      int n = out_pt.n;
      struct outPt_t *opt = (struct outPt_t *)out_pt.ptr;
      for (; n; --n, ++opt)
      {
         const int index = opt->index;
         if (code[index] == CODE_INTERNAL)
         {
            continue;
         }
         if (code[index] == CODE_NOT_FOUND || opt->code == CODE_INTERNAL ||
             opt->dist2 < dist[index])
         {
            npt_found_on_other_proc++;
            for (int d = 0; d < dim; ++d)
            {
               gsl_ref(dim * index + d) = opt->r[d];
            }
            dist[index] = opt->dist2;
            proc[index] = opt->proc;
            elem[index] = opt->el;
            code[index] = opt->code;
            newton[index] = opt->newton;
         }
      }
      array_free(&out_pt);
   }
   MPI_Barrier(gsl_comm->c);
}

void FindPointsGSLIB::SetupDevice(MemoryType mt)
{
   auto *findptsData3 = (gslib::findpts_data_3 *)this->fdataD;
   auto *findptsData2 = (gslib::findpts_data_2 *)this->fdataD;

   DEV.tol = dim == 2 ? findptsData2->local.tol : findptsData3->local.tol;
   if (dim == 3)
   {
      DEV.hash3 = &findptsData3->hash;
   }
   else
   {
      DEV.hash2 = &findptsData2->hash;
   }
   DEV.cr = dim == 2 ? &findptsData2->cr : &findptsData3->cr;

   gsl_mesh.UseDevice(true);

   DEV.o_c.UseDevice(true);
   DEV.o_c.SetSize(dim*NE_split_total);
   auto p_o_c = DEV.o_c.HostWrite();

   DEV.o_A.UseDevice(true);
   DEV.o_A.SetSize(dim*dim*NE_split_total);
   auto p_o_A = DEV.o_A.HostWrite();

   DEV.o_min.UseDevice(true);
   DEV.o_min.SetSize(dim*NE_split_total);
   auto p_o_min = DEV.o_min.HostWrite();

   DEV.o_max.UseDevice(true);
   DEV.o_max.SetSize(dim*NE_split_total);
   auto p_o_max = DEV.o_max.HostWrite();

   int n_box_ents = 3*dim + dim*dim;
   DEV.o_box.UseDevice(true);
   DEV.o_box.SetSize(n_box_ents*NE_split_total);
   auto p_o_box = DEV.o_box.HostWrite();

   const int dim2 = dim*dim;
   if (dim == 3)
   {
      for (int e = 0; e < NE_split_total; e++)
      {
         auto box = findptsData3->local.obb[e];
         for (int d = 0; d < dim; d++)
         {
            p_o_c[dim*e+d] = box.c0[d];
            p_o_min[dim*e+d] = box.x[d].min;
            p_o_max[dim*e+d] = box.x[d].max;
            p_o_box[n_box_ents*e + d] = box.c0[d];
            p_o_box[n_box_ents*e + dim + d] = box.x[d].min;
            p_o_box[n_box_ents*e + 2*dim + d] = box.x[d].max;
         }

         for (int d = 0; d < dim2; ++d)
         {
            p_o_A[dim2*e+d] = box.A[d];
            p_o_box[n_box_ents*e + 3*dim + d] = box.A[d];
         }
      }
   }
   else
   {
      for (int e = 0; e < NE_split_total; e++)
      {
         auto box = findptsData2->local.obb[e];
         for (int d = 0; d < dim; d++)
         {
            p_o_c[dim*e+d] = box.c0[d];
            p_o_min[dim*e+d] = box.x[d].min;
            p_o_max[dim*e+d] = box.x[d].max;
            p_o_box[n_box_ents*e + d] = box.c0[d];
            p_o_box[n_box_ents*e + dim + d] = box.x[d].min;
            p_o_box[n_box_ents*e + 2*dim + d] = box.x[d].max;
         }

         for (int d = 0; d < dim2; ++d)
         {
            p_o_A[dim2*e+d] = box.A[d];
            p_o_box[n_box_ents*e + 3*dim + d] = box.A[d];
         }
      }
   }

   DEV.o_hashMax.UseDevice(true);
   DEV.o_hashMin.UseDevice(true);
   DEV.o_hashFac.UseDevice(true);
   DEV.o_hashMax.SetSize(dim);
   DEV.o_hashMin.SetSize(dim);
   DEV.o_hashFac.SetSize(dim);
   if (dim == 2)
   {
      auto hash = findptsData2->local.hd;

      auto p_o_hashMax = DEV.o_hashMax.HostWrite();
      auto p_o_hashMin = DEV.o_hashMin.HostWrite();
      auto p_o_hashFac = DEV.o_hashFac.HostWrite();
      for (int d = 0; d < dim; d++)
      {
         p_o_hashMax[d] = hash.bnd[d].max;
         p_o_hashMin[d] = hash.bnd[d].min;
         p_o_hashFac[d] = hash.fac[d];
      }
      DEV.hash_n = hash.hash_n;
   }
   else
   {
      auto hash = findptsData3->local.hd;

      auto p_o_hashMax = DEV.o_hashMax.HostWrite();
      auto p_o_hashMin = DEV.o_hashMin.HostWrite();
      auto p_o_hashFac = DEV.o_hashFac.HostWrite();
      for (int d = 0; d < dim; d++)
      {
         p_o_hashMax[d] = hash.bnd[d].max;
         p_o_hashMin[d] = hash.bnd[d].min;
         p_o_hashFac[d] = hash.fac[d];
      }
      DEV.hash_n = hash.hash_n;
   }

   DEV.hd_d_size = dim == 2 ?
                   findptsData2->local.hd.offset[(int)std::pow(DEV.hash_n, dim)] :
                   findptsData3->local.hd.offset[(int)std::pow(DEV.hash_n, dim)];

   DEV.ou_offset.SetSize(DEV.hd_d_size);
   auto p_ou_offset = DEV.ou_offset.HostWrite();
   for (int i = 0; i < DEV.hd_d_size; i++)
   {
      p_ou_offset[i] = dim == 2 ? findptsData2->local.hd.offset[i] :
                       findptsData3->local.hd.offset[i];
   }

   int maxelementperbox = 0;
   for (int i = 0; i < DEV.hash_n*DEV.hash_n; i++)
   {
      maxelementperbox = std::fmax(maxelementperbox,
                                   (int)DEV.ou_offset[i+1]-(int)DEV.ou_offset[i]);
   }

   MPI_Allreduce(MPI_IN_PLACE, &maxelementperbox, 1, MPI_INT, MPI_MAX, gsl_comm->c);
   if (gsl_comm->id == 0)
   {
      mfem::out << "Max AABB intersections: " << maxelementperbox << std::endl;
   }

   DEV.o_wtend.UseDevice(true);
   DEV.o_wtend.SetSize(6*DEV.dof1d);
   DEV.o_wtend.HostWrite();
   DEV.o_wtend = dim == 2 ? findptsData2->local.fed.wtend[0] :
                 findptsData3->local.fed.wtend[0];

   // Get gll points
   DEV.gll1d.UseDevice(true);
   DEV.gll1d.SetSize(DEV.dof1d);
   DEV.gll1d.HostWrite();
   DEV.gll1d = dim == 2 ? findptsData2->local.fed.z[0] :
               findptsData3->local.fed.z[0];

   DEV.lagcoeff.UseDevice(true);
   DEV.lagcoeff.SetSize(DEV.dof1d);
   DEV.lagcoeff.HostWrite();
   DEV.lagcoeff = dim == 2 ? findptsData2->local.fed.lag_data[0] :
                  findptsData3->local.fed.lag_data[0];

   DEV.info.UseDevice(true);
   DEV.info.SetSize(0*points_cnt);
   DEV.info.HostWrite();
   //   DEV.info = -1;

   MFEM_DEVICE_SYNC;
}

Mesh* FindPointsGSLIB::GetBoundingBoxMesh(int type)
{
   MFEM_VERIFY(setupflag, "Call FindPointsGSLIB::Setup method first");

   auto *findptsData3 = (gslib::findpts_data_3 *)this->fdataD;
   auto *findptsData2 = (gslib::findpts_data_2 *)this->fdataD;

   int myid = gsl_comm->id;
   int save_rank = 0;
   int nve   = dim == 2 ? 4 : 8;
   int nel = 0;
   int hash_l_n = dim == 2 ? findptsData2->local.hd.hash_n :
                  findptsData3->local.hd.hash_n;
   int hash_g_n = dim == 2 ? findptsData2->hash.hash_n :
                  findptsData3->hash.hash_n;

   if (type == 0 || type == 1)
   {
      nel = NE_split_total;
   }
   else if (type == 2)
   {
      nel = std::pow(hash_l_n, dim);
   }
   else if (type == 3)
   {
      nel = std::pow(hash_g_n, dim);
   }

   Vector hashlmin(dim), hashlmax(dim), dxyzl(dim);
   for (int d = 0; d < dim; d++)
   {
      hashlmin[d] = dim == 2 ? findptsData2->local.hd.bnd[d].min :
                    findptsData3->local.hd.bnd[d].min;
      hashlmax[d] =  dim == 2 ? findptsData2->local.hd.bnd[d].max :
                     findptsData3->local.hd.bnd[d].max;
      dxyzl[d] = (hashlmax[d]-hashlmin[d])/hash_l_n;
   }

   Vector hashgmin(dim), hashgmax(dim), dxyzg(dim);
   for (int d = 0; d < dim; d++)
   {
      hashgmin[d] = dim == 2 ? findptsData2->hash.bnd[d].min :
                    findptsData3->hash.bnd[d].min;
      hashgmax[d] =  dim == 2 ? findptsData2->hash.bnd[d].max :
                     findptsData3->hash.bnd[d].max;
      dxyzg[d] = (hashgmax[d]-hashgmin[d])/hash_g_n;
   }

   long long nel_l = nel;
   long long nel_glob_l = nel_l;
   int ne_glob;

   if (type == 0 || type == 1 || type == 2)
   {
#ifdef MFEM_USE_MPI
      MPI_Reduce(&nel_l, &nel_glob_l, 1, MPI_LONG_LONG, MPI_SUM, save_rank,
                 gsl_comm->c);
#endif
      ne_glob = int(nel_glob_l);
   }
   else
   {
      ne_glob = nel;
   }

   int nverts = nve*ne_glob;
   Vector o_xyz(dim*nel*nve);

   Mesh *meshbb = NULL;
   if (gsl_comm->id == save_rank)
   {
      meshbb = new Mesh(dim, nverts, ne_glob, 0, dim);
   }

   Array<int> hash_el_count(gsl_comm->np);
   hash_el_count[0] = nel;

   if (dim == 3)
   {
      for (int e = 0; e < nel; e++)
      {
         auto box = findptsData3->local.obb[e];
         if (type == 0)
         {
            Vector minn(dim), maxx(dim);
            for (int d = 0; d < dim; d++)
            {
               minn[d] = box.x[d].min;
               maxx[d] = box.x[d].max;
            }
            int c = 0;
            o_xyz(e*nve*dim + c++) = minn[0];
            o_xyz(e*nve*dim + c++) = minn[1];
            o_xyz(e*nve*dim + c++) = minn[2];

            o_xyz(e*nve*dim + c++) = maxx[0];
            o_xyz(e*nve*dim + c++) = minn[1];
            o_xyz(e*nve*dim + c++) = minn[2];

            o_xyz(e*nve*dim + c++) = maxx[0];
            o_xyz(e*nve*dim + c++) = maxx[1];
            o_xyz(e*nve*dim + c++) = minn[2];

            o_xyz(e*nve*dim + c++) = minn[0];
            o_xyz(e*nve*dim + c++) = maxx[1];
            o_xyz(e*nve*dim + c++) = minn[2];

            o_xyz(e*nve*dim + c++) = minn[0];
            o_xyz(e*nve*dim + c++) = minn[1];
            o_xyz(e*nve*dim + c++) = maxx[2];

            o_xyz(e*nve*dim + c++) = maxx[0];
            o_xyz(e*nve*dim + c++) = minn[1];
            o_xyz(e*nve*dim + c++) = maxx[2];

            o_xyz(e*nve*dim + c++) = maxx[0];
            o_xyz(e*nve*dim + c++) = maxx[1];
            o_xyz(e*nve*dim + c++) = maxx[2];

            o_xyz(e*nve*dim + c++) = minn[0];
            o_xyz(e*nve*dim + c++) = maxx[1];
            o_xyz(e*nve*dim + c++) = maxx[2];
         }
         else if (type == 1)
         {
            Vector center(dim), A(dim*dim);
            for (int d = 0; d < dim; d++)
            {
               center[d] = box.c0[d];
            }
            for (int d = 0; d < dim*dim; d++)
            {
               A[d] = box.A[d];
            }

            DenseMatrix Amat(A.GetData(), dim, dim);
            Amat.Transpose();
            Amat.Invert();

            Vector v1(dim);
            Vector temp;

            v1(0) = -1.0; v1(1) = -1.0; v1(2) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 0, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = -1.0; v1(2) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 3, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = 1.0; v1(2) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 6, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = -1.0; v1(1) = 1.0; v1(2) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 9, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = -1.0; v1(1) = -1.0; v1(2) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 12, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = -1.0; v1(2) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 15, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = 1.0; v1(2) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 18, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = -1.0; v1(1) = 1.0; v1(2) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 21, dim);
            Amat.Mult(v1, temp);
            temp += center;
         } //type == 1
      } // e < nel
      if (type == 2)
      {
         int ec = 0;
         for (int l = 0; l < hash_l_n; l++)
         {
            for (int k = 0; k < hash_l_n; k++)
            {
               for (int j = 0; j < hash_l_n; j++)
               {
                  double x0 = hashlmin[0] + j*dxyzl[0];
                  double y0 = hashlmin[1] + k*dxyzl[1];
                  double z0 = hashlmin[2] + l*dxyzl[2];
                  o_xyz(ec*nve*dim + 0) = x0;
                  o_xyz(ec*nve*dim + 1) = y0;
                  o_xyz(ec*nve*dim + 2) = z0;

                  o_xyz(ec*nve*dim + 3) = x0+dxyzl[0];
                  o_xyz(ec*nve*dim + 4) = y0;
                  o_xyz(ec*nve*dim + 5) = z0;

                  o_xyz(ec*nve*dim + 6) = x0+dxyzl[0];
                  o_xyz(ec*nve*dim + 7) = y0+dxyzl[1];
                  o_xyz(ec*nve*dim + 8) = z0;

                  o_xyz(ec*nve*dim + 9) = x0;
                  o_xyz(ec*nve*dim + 10) = y0+dxyzl[1];
                  o_xyz(ec*nve*dim + 11) = z0;

                  o_xyz(ec*nve*dim + 12) = x0;
                  o_xyz(ec*nve*dim + 13) = y0;
                  o_xyz(ec*nve*dim + 14) = z0+dxyzl[2];

                  o_xyz(ec*nve*dim + 15) = x0+dxyzl[0];
                  o_xyz(ec*nve*dim + 16) = y0;
                  o_xyz(ec*nve*dim + 17) = z0+dxyzl[2];

                  o_xyz(ec*nve*dim + 18) = x0+dxyzl[0];
                  o_xyz(ec*nve*dim + 19) = y0+dxyzl[1];
                  o_xyz(ec*nve*dim + 20) = z0+dxyzl[2];

                  o_xyz(ec*nve*dim + 21) = x0;
                  o_xyz(ec*nve*dim + 22) = y0+dxyzl[1];
                  o_xyz(ec*nve*dim + 23) = z0+dxyzl[2];

                  ec++;
               }
            }
         }
      } //type == 2
      else if (type == 3)
      {
         int ec = 0;
         for (int l = 0; l < hash_g_n; l++)
         {
            for (int k = 0; k < hash_g_n; k++)
            {
               for (int j = 0; j < hash_g_n; j++)
               {
                  double x0 = hashgmin[0] + j*dxyzg[0];
                  double y0 = hashgmin[1] + k*dxyzg[1];
                  double z0 = hashgmin[2] + l*dxyzg[2];
                  o_xyz(ec*nve*dim + 0) = x0;
                  o_xyz(ec*nve*dim + 1) = y0;
                  o_xyz(ec*nve*dim + 2) = z0;

                  o_xyz(ec*nve*dim + 3) = x0+dxyzg[0];
                  o_xyz(ec*nve*dim + 4) = y0;
                  o_xyz(ec*nve*dim + 5) = z0;

                  o_xyz(ec*nve*dim + 6) = x0+dxyzg[0];
                  o_xyz(ec*nve*dim + 7) = y0+dxyzg[1];
                  o_xyz(ec*nve*dim + 8) = z0;

                  o_xyz(ec*nve*dim + 9) = x0;
                  o_xyz(ec*nve*dim + 10) = y0+dxyzg[1];
                  o_xyz(ec*nve*dim + 11) = z0;

                  o_xyz(ec*nve*dim + 12) = x0;
                  o_xyz(ec*nve*dim + 13) = y0;
                  o_xyz(ec*nve*dim + 14) = z0+dxyzg[2];

                  o_xyz(ec*nve*dim + 15) = x0+dxyzg[0];
                  o_xyz(ec*nve*dim + 16) = y0;
                  o_xyz(ec*nve*dim + 17) = z0+dxyzg[2];

                  o_xyz(ec*nve*dim + 18) = x0+dxyzg[0];
                  o_xyz(ec*nve*dim + 19) = y0+dxyzg[1];
                  o_xyz(ec*nve*dim + 20) = z0+dxyzg[2];

                  o_xyz(ec*nve*dim + 21) = x0;
                  o_xyz(ec*nve*dim + 22) = y0+dxyzg[1];
                  o_xyz(ec*nve*dim + 23) = z0+dxyzg[2];

                  ec++;
               }
            }
         }
      }
   }
   else
   {
      for (int e = 0; e < nel; e++)
      {
         auto box = findptsData2->local.obb[e];
         if (type == 0)
         {
            Vector minn(dim), maxx(dim);
            for (int d = 0; d < dim; d++)
            {
               minn[d] = box.x[d].min;
               maxx[d] = box.x[d].max;
            }
            o_xyz(e*nve*dim + 0) = minn[0];
            o_xyz(e*nve*dim + 1) = minn[1];

            o_xyz(e*nve*dim + 2) = maxx[0];
            o_xyz(e*nve*dim + 3) = minn[1];

            o_xyz(e*nve*dim + 4) = maxx[0];
            o_xyz(e*nve*dim + 5) = maxx[1];

            o_xyz(e*nve*dim + 6) = minn[0];
            o_xyz(e*nve*dim + 7) = maxx[1];
         }
         else if (type == 1)
         {
            Vector center(dim), A(dim*dim);
            for (int d = 0; d < dim; d++)
            {
               center[d] = box.c0[d];
            }
            for (int d = 0; d < dim*dim; d++)
            {
               A[d] = box.A[d];
            }

            DenseMatrix Amat(A.GetData(), dim, dim);
            Amat.Transpose();
            Amat.Invert();

            Vector v1(dim);
            Vector temp;

            v1(0) = -1.0; v1(1) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 0, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 2, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 4, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = -1.0; v1(1) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 6, dim);
            Amat.Mult(v1, temp);
            temp += center;
         } //type == 1
      } // e < nel
      if (type == 2)
      {
         int ec = 0;
         for (int k = 0; k < hash_l_n; k++)
         {
            for (int j = 0; j < hash_l_n; j++)
            {
               double x0 = hashlmin[0] + j*dxyzl[0];
               double y0 = hashlmin[1] + k*dxyzl[1];
               o_xyz(ec*nve*dim + 0) = x0;
               o_xyz(ec*nve*dim + 1) = y0;

               o_xyz(ec*nve*dim + 2) = x0+dxyzl[0];
               o_xyz(ec*nve*dim + 3) = y0;

               o_xyz(ec*nve*dim + 4) = x0+dxyzl[0];
               o_xyz(ec*nve*dim + 5) = y0+dxyzl[1];

               o_xyz(ec*nve*dim + 6) = x0;
               o_xyz(ec*nve*dim + 7) = y0+dxyzl[1];

               ec++;
            }
         }
      }
      else if (type == 3)
      {
         int ec = 0;
         for (int k = 0; k < hash_g_n; k++)
         {
            for (int j = 0; j < hash_g_n; j++)
            {
               double x0 = hashgmin[0] + j*dxyzg[0];
               double y0 = hashgmin[1] + k*dxyzg[1];
               o_xyz(ec*nve*dim + 0) = x0;
               o_xyz(ec*nve*dim + 1) = y0;

               o_xyz(ec*nve*dim + 2) = x0+dxyzg[0];
               o_xyz(ec*nve*dim + 3) = y0;

               o_xyz(ec*nve*dim + 4) = x0+dxyzg[0];
               o_xyz(ec*nve*dim + 5) = y0+dxyzg[1];

               o_xyz(ec*nve*dim + 6) = x0;
               o_xyz(ec*nve*dim + 7) = y0+dxyzg[1];

               ec++;
            }
         }
      }
   }

   int nsend = type == 3 ? 0 : nel*nve*dim;
   int nrecv = 0;
   MPI_Status status;
   int vidx = 0;
   int eidx = 0;
   int proc_hash_loc_idx = 0;
   if (myid == save_rank)
   {
      for (int p = 0; p < gsl_comm->np; p++)
      {
         if (p != save_rank)
         {
            MPI_Recv(&nrecv, 1, MPI_INT, p, 444, gsl_comm->c, &status);
            o_xyz.SetSize(nrecv);
            if (nrecv)
            {
               MPI_Recv(o_xyz.GetData(), nrecv, MPI_DOUBLE, p, 445, gsl_comm->c, &status);
            }
         }
         else
         {
            nrecv = type == 3 ? nel*nve*dim : nsend;
         }
         int nel_recv = nrecv/(dim*nve);

         // we keep track of how many hash cells are coming from each rank
         if (p != save_rank)
         {
            hash_el_count[p] = hash_el_count[p-1] + nel_recv;
         }
         for (int e = 0; e < nel_recv; e++)
         {
            for (int j = 0; j < nve; j++)
            {
               Vector ver(o_xyz.GetData() + e*nve*dim + j*dim, dim);
               meshbb->AddVertex(ver);
            }

            if (dim == 2)
            {
               const int inds[4] = {vidx++, vidx++, vidx++, vidx++};
               int attr = eidx+1;
               // for type == 2, we set element attribute based on the
               // proc from which the element must have come.
               if (type == 2)
               {
                  if (eidx >= hash_el_count[proc_hash_loc_idx])
                  {
                     proc_hash_loc_idx++;
                  }
                  attr = proc_hash_loc_idx+1;
               }
               else if (type == 3)
               {
                  attr = eidx % gsl_comm->np;
                  attr += 1;
               }
               meshbb->AddQuad(inds, attr);
               eidx++;
            }
            else
            {
               const int inds[8] = {vidx++, vidx++, vidx++, vidx++,
                                    vidx++, vidx++, vidx++, vidx++
                                   };
               meshbb->AddHex(inds, (eidx++)+1);
            }
         }
      }
      if (dim == 2)
      {
         meshbb->FinalizeQuadMesh(1, 1, true);
      }
      else
      {
         meshbb->FinalizeHexMesh(1, 1, true);
      }
   }
   else
   {
      MPI_Send(&nsend, 1, MPI_INT, save_rank, 444, gsl_comm->c);
      if (nsend)
      {
         MPI_Send(o_xyz.GetData(), nsend, MPI_DOUBLE, save_rank, 445, gsl_comm->c);
      }
   }

   return meshbb;
}

Mesh* FindPointsGSLIB::GetGSLIBMesh()
{
   MFEM_VERIFY(setupflag, "Call FindPointsGSLIB::Setup method first");
   int myid = gsl_comm->id;
   int num_ranks = gsl_comm->np;
   int nelem = NE_split_total;
   int ne_glob = nelem;
   int mesh_order = mesh->GetNodalFESpace()->GetMaxElementOrder();
   int dim = mesh->Dimension();
   int save_rank = 0;

   MPI_Allreduce(MPI_IN_PLACE, &ne_glob, 1, MPI_INT, MPI_SUM, gsl_comm->c);
   MPI_Allreduce(MPI_IN_PLACE, &mesh_order, 1, MPI_INT, MPI_MAX, gsl_comm->c);
   MPI_Allreduce(MPI_IN_PLACE, &dim, 1, MPI_INT, MPI_MAX, gsl_comm->c);

   int nodes_tot = ne_glob*std::pow(mesh_order+1, dim);
   int nodes_el  = std::pow(mesh_order+1, dim);
   int nve   = dim == 2 ? 4 : 8;
   int nverts = nve*ne_glob;

   Mesh *meshbb = NULL;
   GridFunction *bbnodes = NULL;
   FiniteElementSpace *bbspace = NULL;
   if (gsl_comm->id == save_rank)
   {
      meshbb = new Mesh(dim, nverts, ne_glob, 0, dim);

      Vector xyz(dim);
      xyz = 0.0;
      int vidx = 0;
      for (int e = 0; e < ne_glob; e++)
      {
         for (int v = 0; v < nve; v++)
         {
            meshbb->AddVertex(xyz);
         }
         int attr = 1;

         if (dim == 2)
         {
            const int inds[4] = {vidx++, vidx++, vidx++, vidx++};
            meshbb->AddQuad(inds, attr);
         }
         else if (dim == 3)
         {
            const int inds[8] = {vidx++, vidx++, vidx++, vidx++,
                                       vidx++, vidx++, vidx++, vidx++
                                    };
            meshbb->AddHex(inds, attr);
         }
      }
      if (dim == 2)
      {
         meshbb->FinalizeQuadMesh(1, 1, true);
      }
      else
      {
         meshbb->FinalizeHexMesh(1, 1, true);
      }
      meshbb->SetCurvature(mesh_order, true, -1, 0);
      bbnodes = meshbb->GetNodes();
      bbspace = bbnodes->FESpace();
      bbnodes->UseDevice(false);
   }
   MPI_Status status;

   Vector o_xyz;

   int ecount = 0;
   int nsend = gsl_mesh.Size();
   int nrecv;
   if (myid == save_rank)
   {
      for (int p = 0; p < gsl_comm->np; p++)
      {
         if (p != save_rank)
         {
            MPI_Recv(&nrecv, 1, MPI_INT, p, 444, gsl_comm->c, &status);
            o_xyz.SetSize(nrecv);
            if (nrecv)
            {
               MPI_Recv(o_xyz.GetData(), nrecv, MPI_DOUBLE, p, 445, gsl_comm->c, &status);
            }
         }
         else
         {
            nrecv = nsend;
            o_xyz.SetSize(nsend);
            o_xyz = gsl_mesh.HostRead();
         }
         int nel_recv = nrecv/(nodes_el*dim);

         // Need to format data from gsl_mesh (which was by nodes for all elements)
         // to element-by-element by nodes

         for (int e = 0; e < nel_recv; e++)
         {
            // Data for that element
            Vector eldata(nodes_el*dim);
            for (int d = 0; d < dim; d++)
            {
               Vector ddata(nodes_el);
               ddata = o_xyz.GetData() + e*nodes_el + d*nodes_el*nel_recv;
               for (int i = 0; i < nodes_el; i++)
               {
                  eldata(i+d*nodes_el) = ddata(i);
               }
            }
            Array<int> vdofs;
            Vector gfdata;
            bbspace->GetElementVDofs(ecount, vdofs);
            bbnodes->SetSubVector(vdofs, eldata);
            ecount++;
         }
      }
   }
   else
   {
      MPI_Send(&nsend, 1, MPI_INT, save_rank, 444, gsl_comm->c);
      if (nsend)
      {
         MPI_Send(gsl_mesh.HostRead(), nsend, MPI_DOUBLE, save_rank, 445, gsl_comm->c);
      }
   }


   return meshbb;
}

void FindPointsGSLIB::FindPointsSurfOnDevice( const Vector &point_pos,
                                              int point_pos_ordering )
{
   point_pos.HostRead();
   Vector point_pos_copy = point_pos;  // true copy of point_pos
   point_pos_copy.UseDevice(true);
   point_pos_copy.HostReadWrite();
   MemoryType mt = point_pos.GetMemory().GetMemoryType();

   SW2.Clear();
   SW2.Start();

   SurfSetupDevice(mt);

   SW2.Stop();
   findpts_setup_device_arrays_time = SW2.RealTime();

   gsl_ref.UseDevice(true);
   gsl_dist.UseDevice(true);

   if (spacedim==2)
   {
      FindPointsSurfLocal2(point_pos,
                           point_pos_ordering,
                           gsl_code,
                           gsl_elem,
                           gsl_ref,
                           gsl_dist,
                           gsl_newton,
                           points_cnt );
   }
   else
   {
      FindPointsSurfLocal32(point_pos_copy,
                            point_pos_ordering,
                            gsl_code,
                            gsl_elem,
                            gsl_ref,
                            gsl_dist,
                            gsl_newton,
                            points_cnt);
   }

   gsl_ref.HostReadWrite();
   gsl_dist.HostReadWrite();
   point_pos.HostRead();
   DEV.info.HostReadWrite();
   gsl_newton.HostReadWrite();

   gsl_code.HostReadWrite();
   gsl_proc.HostReadWrite();
   gsl_elem.HostReadWrite();

   const int myid = gsl_comm->id;
   for (int i=0; i<points_cnt; i++)
   {
      gsl_proc[i] = myid;
   }

   const int id = gsl_comm->id,
             np = gsl_comm->np;

   if (np==1)
   {
      return;
   }

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
      dfloat r[2], dist2;
      unsigned int index, code, el, proc;
      int newton;
   };

   int find_elsewhere = 0;
   int found_local = 0;
   int hashptn = 0;
   {
      int index;
      auto *code = gsl_code.HostReadWrite();
      struct srcPt_t *pt;

      array_init(struct srcPt_t, &hash_pt, points_cnt);
      pt = (struct srcPt_t *)hash_pt.ptr;

      dfloat x[spacedim];
      for (index=0; index<points_cnt; ++index)
      {
         for (int d=0; d<spacedim; ++d)
         {
            int idx = point_pos_ordering == 0 ? index + d*points_cnt :
                      index*spacedim + d;
            x[d] = point_pos(idx);
         }
         if (*code != CODE_INTERNAL)
         {
            find_elsewhere++;
            const auto hi = spacedim==2 ? hash_index_2(DEV.hash2, x) :
                            hash_index_3(DEV.hash3, x);
            for (int d=0; d<spacedim; ++d)
            {
               pt->x[d] = x[d];
            }
            pt->index = index;
            pt->proc = hi % np;
            ++pt;
         }
         else
         {
            found_local++;
         }
         code++;
      }
      hash_pt.n = pt - (struct srcPt_t *)hash_pt.ptr;
      sarray_transfer(struct srcPt_t, &hash_pt, proc, 1, DEV.cr);
      hashptn = hash_pt.n;
   }
   MPI_Barrier(gsl_comm->c);

   /* look up points in hash cells, route to possible procs */
   {
      const unsigned int *const hash_offset = spacedim==2 ? DEV.hash2->offset :
                                              DEV.hash3->offset;
      int count = 0, *proc, *proc_p;
      const struct srcPt_t *p = (struct srcPt_t *)hash_pt.ptr,
                            *const pe = p + hash_pt.n;
      struct srcPt_t *q;

      for (; p!=pe; ++p)
      {
         const int hi = spacedim==2 ? hash_index_2(DEV.hash2, p->x)/np :
                        hash_index_3(DEV.hash3, p->x)/np;
         const int i = hash_offset[hi], ie = hash_offset[hi + 1];
         count += ie - i;
      }

      proc = (int *) gslib::smalloc(count*sizeof(int), __FILE__,__LINE__);
      proc_p = proc;
      array_init(struct srcPt_t, &src_pt, count);

      q = (struct srcPt_t *)src_pt.ptr;
      p = (struct srcPt_t *)hash_pt.ptr;
      for (; p!=pe; ++p)
      {
         const int hi = spacedim==2 ? hash_index_2(DEV.hash2, p->x)/np :
                        hash_index_3(DEV.hash3, p->x)/np;
         int i        = hash_offset[hi];
         const int ie = hash_offset[hi + 1];
         for (; i!=ie; ++i)
         {
            const int pp = hash_offset[i];
            if (pp==p->proc)
            {
               continue;   /* don't send back to source proc */
            }
            *proc_p++ = pp;
            *q++      = *p;
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
         opt->proc  = spt->proc;
      }
      spt = (struct srcPt_t *)src_pt.ptr;
      opt = (struct outPt_t *)out_pt.ptr;

      n = out_pt.n;

      Vector gsl_ref_l;
      gsl_ref_l.UseDevice(true);
      gsl_ref_l.SetSize(n*dim);

      Vector gsl_dist_l;
      gsl_dist_l.UseDevice(true);
      gsl_dist_l.SetSize(n);

      Vector point_pos_l;
      point_pos_l.UseDevice(true);
      point_pos_l.SetSize(n*spacedim);
      point_pos_l.HostWrite();

      Array<int> gsl_newton_l;
      gsl_newton_l.SetSize(n);

      Array<unsigned int> gsl_code_l;
      gsl_code_l.SetSize(n);
      Array<unsigned int> gsl_elem_l;
      gsl_elem_l.SetSize(n);

      for (int point=0; point<n; ++point)
      {
         for (int d=0; d<spacedim; d++)
         {
            int idx = point_pos_ordering==0 ? point + d*n :
                      point*spacedim + d;
            point_pos_l(idx) = spt[point].x[d];
         }
      }

      if (spacedim==2)
      {
         FindPointsSurfLocal2(point_pos_l,
                              point_pos_ordering,
                              gsl_code_l,
                              gsl_elem_l,
                              gsl_ref_l,
                              gsl_dist_l,
                              gsl_newton_l,
                              n );
      }
      else
      {
         FindPointsSurfLocal32(point_pos_l,
                               point_pos_ordering,
                               gsl_code_l,
                               gsl_elem_l,
                               gsl_ref_l,
                               gsl_dist_l,
                               gsl_newton_l,
                               n);
      }

      gsl_ref_l   .HostRead();
      gsl_dist_l  .HostRead();
      gsl_code_l  .HostRead();
      gsl_elem_l  .HostRead();
      gsl_newton_l.HostRead();
      DEV.info    .HostRead();

      // unpack arrays into opt
      for (int point=0; point<n; ++point)
      {
         opt[point].code  = gsl_code_l[point];
         opt[point].el    = gsl_elem_l[point];
         opt[point].dist2 = gsl_dist_l(point);
         for (int d=0; d<dim; ++d)
         {
            opt[point].r[d] = gsl_ref_l(dim*point + d);
         }
         opt->newton = gsl_newton_l[point];
      }
      array_free(&src_pt);

      /* group by code to eliminate unfound points */
      sarray_sort(struct outPt_t, opt, out_pt.n, code, 0, &DEV.cr->data);

      n = out_pt.n;
      while (n && opt[n-1].code == CODE_NOT_FOUND)
      {
         --n;
      }
      out_pt.n = n;

      sarray_transfer(struct outPt_t, &out_pt, proc, 1, DEV.cr);
   }
   MPI_Barrier(gsl_comm->c);

   gsl_code.HostReadWrite();
   gsl_elem.HostReadWrite();
   gsl_proc.HostReadWrite();

   // /* merge remote results with user data */
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
         if ( gsl_code[index]==CODE_NOT_FOUND
              || opt->code==CODE_INTERNAL
              || opt->dist2<gsl_dist(index) )
         {
            npt_found_on_other_proc++;
            for (int d=0; d<dim; ++d)
            {
               gsl_ref(dim*index + d) = opt->r[d];
            }
            gsl_dist(index) = opt->dist2;
            gsl_proc[index] = opt->proc;
            gsl_elem[index] = opt->el;
            gsl_code[index] = opt->code;
            gsl_newton[index] = opt->newton;
         }
      }
      array_free(&out_pt);
   }
   MPI_Barrier(gsl_comm->c);
}

void lagrange_eval_second_derivative( double
                                      *p0,                  // 0 to pN-1: p0, pN to 2*pN-1: p1, 2*pN to 3*pN-1: p2
                                      double x,                    // ref. coords of the point of interest
                                      int i,                       // index of the Lagrange polynomial
                                      const double *z,             // GLL points
                                      const double *lagrangeCoeff, // Lagrange polynomial denominator term
                                      int pN)                      // number of GLL points
{
   double u0 = 1, u1 = 0, u2 = 0;
   for (int j=0; j<pN; ++j)
   {
      if (i!=j)
      {
         double d_j = 2 * (x-z[j]);
         u2 = d_j * u2 + u1;
         u1 = d_j * u1 + u0;
         u0 = d_j * u0;
      }
   }
   double *p1 = p0 + pN, *p2 = p0 + 2 * pN;
   p0[i] = lagrangeCoeff[i] * u0;
   p1[i] = 2.0 * lagrangeCoeff[i] * u1;
   p2[i] = 8.0 * lagrangeCoeff[i] * u2;
}

void FindPointsGSLIB::SurfSetupDevice(MemoryType mt)
{
   auto *findptsData3 = (gslib::findpts_data_3 *)this->fdataD;
   auto *findptsData2 = (gslib::findpts_data_2 *)this->fdataD;

   const int spacedim2 = spacedim*spacedim;

   if (spacedim==3)
   {
      DEV.tol   = findptsData3->local.tol;
      DEV.hash3 = &findptsData3->hash;
      DEV.cr    = &findptsData3->cr;
   }
   else
   {
      DEV.tol   = findptsData2->local.tol;
      DEV.hash2 = &findptsData2->hash;
      DEV.cr    = &findptsData2->cr;
   }

   gsl_mesh .UseDevice(true);

   DEV.o_c  .UseDevice(true);
   DEV.o_A  .UseDevice(true);
   DEV.o_min.UseDevice(true);
   DEV.o_max.UseDevice(true);
   DEV.o_c  .SetSize(spacedim*NE_split_total);
   DEV.o_A  .SetSize(spacedim2*NE_split_total);
   DEV.o_min.SetSize(spacedim*NE_split_total);
   DEV.o_max.SetSize(spacedim*NE_split_total);

   auto p_o_c   = DEV.o_c.HostWrite();
   auto p_o_A   = DEV.o_A.HostWrite();
   auto p_o_min = DEV.o_min.HostWrite();
   auto p_o_max = DEV.o_max.HostWrite();
   if (spacedim==3)
   {
      for (int e=0; e<NE_split_total; e++)
      {
         auto box = findptsData3->local.obb[e];
         for (int d=0; d<spacedim; d++)
         {
            p_o_c[spacedim*e + d]   = box.c0[d];
            p_o_min[spacedim*e + d] = box.x[d].min;
            p_o_max[spacedim*e + d] = box.x[d].max;
         }
         for (int d=0; d<spacedim2; ++d)
         {
            p_o_A[spacedim2*e+d] = box.A[d];
         }
      }
   }
   else   // spacedim==2
   {
      for (int e=0; e<NE_split_total; e++)
      {
         auto box = findptsData2->local.obb[e];
         for (int d=0; d<spacedim; d++)
         {
            p_o_c[spacedim*e + d]   = box.c0[d];
            p_o_min[spacedim*e + d] = box.x[d].min;
            p_o_max[spacedim*e + d] = box.x[d].max;
         }
         for (int d=0; d<spacedim2; ++d)
         {
            p_o_A[spacedim2*e+d] = box.A[d];
         }
      }
   }

   DEV.o_hashMax.UseDevice(true);
   DEV.o_hashMin.UseDevice(true);
   DEV.o_hashFac.UseDevice(true);
   DEV.o_hashMax.SetSize(spacedim);
   DEV.o_hashMin.SetSize(spacedim);
   DEV.o_hashFac.SetSize(spacedim);
   if (spacedim==2)
   {
      auto hash = findptsData2->local.hd;

      auto p_o_hashMax = DEV.o_hashMax.HostWrite();
      auto p_o_hashMin = DEV.o_hashMin.HostWrite();
      auto p_o_hashFac = DEV.o_hashFac.HostWrite();
      for (int d=0; d<spacedim; d++)
      {
         p_o_hashMax[d] = hash.bnd[d].max;
         p_o_hashMin[d] = hash.bnd[d].min;
         p_o_hashFac[d] = hash.fac[d];
      }
      DEV.hash_n = hash.hash_n;
   }
   else
   {
      auto hash = findptsData3->local.hd;

      auto p_o_hashMax = DEV.o_hashMax.HostWrite();
      auto p_o_hashMin = DEV.o_hashMin.HostWrite();
      auto p_o_hashFac = DEV.o_hashFac.HostWrite();
      for (int d=0; d<spacedim; d++)
      {
         p_o_hashMax[d] = hash.bnd[d].max;
         p_o_hashMin[d] = hash.bnd[d].min;
         p_o_hashFac[d] = hash.fac[d];
      }
      DEV.hash_n = hash.hash_n;
   }

   DEV.hd_d_size = spacedim==2 ?
                   findptsData2->local.hd.offset[(int)std::pow(DEV.hash_n,spacedim)] :
                   findptsData3->local.hd.offset[(int)std::pow(DEV.hash_n,spacedim)];

   DEV.ou_offset.SetSize(DEV.hd_d_size);
   auto p_ou_offset = DEV.ou_offset.HostWrite();
   for (int i=0; i<DEV.hd_d_size; i++)
   {
      p_ou_offset[i] = spacedim==2 ? findptsData2->local.hd.offset[i] :
                       findptsData3->local.hd.offset[i];
   }

   Vector gll1dtemp(DEV.dof1d),
          lagcoefftemp(DEV.dof1d),
          wtendtemp(6*DEV.dof1d);
   gslib::lobatto_nodes(gll1dtemp.GetData(),
                        DEV.dof1d);    // Get gll points [-1,1] for the given dof1d
   gslib::gll_lag_setup(lagcoefftemp.GetData(),
                        DEV.dof1d); // Get lagrange coefficients at the gll points
   for (int i=0; i<DEV.dof1d; i++)   // loop through all lagrange polynomials
   {
      lagrange_eval_second_derivative(wtendtemp.GetData(),
                                      -1.0,  // evaluate at -1
                                      i,     // for the ith lagrange polynomial
                                      gll1dtemp.GetData(),
                                      lagcoefftemp.GetData(),
                                      DEV.dof1d);
      lagrange_eval_second_derivative(wtendtemp.GetData()+3*DEV.dof1d,
                                      1.0,  // evaluate at 1
                                      i,    // for the ith lagrange polynomial
                                      gll1dtemp.GetData(),
                                      lagcoefftemp.GetData(),
                                      DEV.dof1d);
   }

   DEV.o_wtend.UseDevice(true);
   DEV.o_wtend.SetSize(6*DEV.dof1d);
   DEV.o_wtend.HostWrite();
   DEV.o_wtend = wtendtemp.GetData();

   DEV.gll1d.UseDevice(true);
   DEV.gll1d.SetSize(DEV.dof1d);
   DEV.gll1d.HostWrite();
   DEV.gll1d = gll1dtemp.GetData();

   DEV.lagcoeff.UseDevice(true);
   DEV.lagcoeff.SetSize(DEV.dof1d);
   DEV.lagcoeff.HostWrite();
   DEV.lagcoeff = lagcoefftemp.GetData();

   DEV.info.UseDevice(true);
   DEV.info.SetSize(0*points_cnt);
   DEV.info.HostWrite();
   //   DEV.info = -1;

   MFEM_DEVICE_SYNC;
}

Mesh* FindPointsGSLIB::GetBoundingBoxMeshSurf(int type)
{
   MFEM_VERIFY(setupflag, "Call FindPointsGSLIB::SetupSurf method first");

   int myid          = gsl_comm->id;
   auto *findptsData3 = (gslib::findpts_data_3 *)this->fdataD;
   auto *findptsData2 = (gslib::findpts_data_2 *)this->fdataD;
   int hash_l_n      = spacedim==2 ? findptsData2->local.hd.hash_n :
                       findptsData3->local.hd.hash_n;
   int hash_g_n      = spacedim==2 ? findptsData2->hash.hash_n     :
                       findptsData3->hash.hash_n;

   int nel = 0;
   if (type==0 || type==1) { nel = NE_split_total; }
   if (type==2) { nel = std::pow(hash_l_n, spacedim); }
   if (type==3) { nel = std::pow(hash_g_n, spacedim); }

   long long nel_l      = nel;
   long long nel_glob_l = nel_l;
   int ne_glob;
   int save_rank = 0;
   if (type==0 || type==1 || type==2)
   {
#ifdef MFEM_USE_MPI
      MPI_Reduce(&nel_l, &nel_glob_l, 1, MPI_LONG_LONG, MPI_SUM,
                 save_rank,  // adi: what is save_rank?
                 gsl_comm->c);
#endif
      ne_glob = int(nel_glob_l);
   }
   else
   {
      ne_glob = nel;
   }

   Vector hashlmin(spacedim), hashlmax(spacedim), dxyzl(spacedim);
   for (int d=0; d<spacedim; d++)
   {
      hashlmin[d] = spacedim==2 ? findptsData2->local.hd.bnd[d].min :
                    findptsData3->local.hd.bnd[d].min;
      hashlmax[d] = spacedim==2 ? findptsData2->local.hd.bnd[d].max :
                    findptsData3->local.hd.bnd[d].max;
      dxyzl[d]    = (hashlmax[d]-hashlmin[d])/hash_l_n;
   }

   Vector hashgmin(spacedim), hashgmax(spacedim), dxyzg(spacedim);
   for (int d=0; d<spacedim; d++)
   {
      hashgmin[d] = spacedim==2 ? findptsData2->hash.bnd[d].min :
                    findptsData3->hash.bnd[d].min;
      hashgmax[d] = spacedim==2 ? findptsData2->hash.bnd[d].max :
                    findptsData3->hash.bnd[d].max;
      dxyzg[d]    = (hashgmax[d]-hashgmin[d])/hash_g_n;
   }

   int nve    = spacedim == 2 ? 4 : 8;  // adi: number of vertices per bb element?
   int nverts = nve*ne_glob;
   Vector o_xyz(spacedim*nel*nve);

   Mesh *meshbb = NULL;
   if (gsl_comm->id==save_rank)
   {
      meshbb = new Mesh(spacedim, nverts, ne_glob, 0, spacedim);
   }

   Array<int> hash_el_count(gsl_comm->np);
   hash_el_count[0] = nel;

   if (spacedim==3)
   {
      for (int e=0; e<nel; e++)
      {
         auto box = findptsData3->local.obb[e];
         if (type==0)
         {
            Vector minn(spacedim), maxx(spacedim);
            for (int d=0; d<spacedim; d++)
            {
               minn[d] = box.x[d].min;
               maxx[d] = box.x[d].max;
            }
            int c = 0;
            // All 8 vertices of the bounding box are assigned positions here
            // in anti-clockwise order starting from the z=0 plane.
            o_xyz(e*nve*spacedim + c++) = minn[0];
            o_xyz(e*nve*spacedim + c++) = minn[1];
            o_xyz(e*nve*spacedim + c++) = minn[2];

            o_xyz(e*nve*spacedim + c++) = maxx[0];
            o_xyz(e*nve*spacedim + c++) = minn[1];
            o_xyz(e*nve*spacedim + c++) = minn[2];

            o_xyz(e*nve*spacedim + c++) = maxx[0];
            o_xyz(e*nve*spacedim + c++) = maxx[1];
            o_xyz(e*nve*spacedim + c++) = minn[2];

            o_xyz(e*nve*spacedim + c++) = minn[0];
            o_xyz(e*nve*spacedim + c++) = maxx[1];
            o_xyz(e*nve*spacedim + c++) = minn[2];

            o_xyz(e*nve*spacedim + c++) = minn[0];
            o_xyz(e*nve*spacedim + c++) = minn[1];
            o_xyz(e*nve*spacedim + c++) = maxx[2];

            o_xyz(e*nve*spacedim + c++) = maxx[0];
            o_xyz(e*nve*spacedim + c++) = minn[1];
            o_xyz(e*nve*spacedim + c++) = maxx[2];

            o_xyz(e*nve*spacedim + c++) = maxx[0];
            o_xyz(e*nve*spacedim + c++) = maxx[1];
            o_xyz(e*nve*spacedim + c++) = maxx[2];

            o_xyz(e*nve*spacedim + c++) = minn[0];
            o_xyz(e*nve*spacedim + c++) = maxx[1];
            o_xyz(e*nve*spacedim + c++) = maxx[2];
         } // type == 0
         else if (type==1)
         {
            Vector center(spacedim), A(spacedim*spacedim);
            for (int d=0; d<spacedim; d++)
            {
               center[d] = box.c0[d];
            }
            for (int d=0; d<spacedim*spacedim; d++)
            {
               A[d] = box.A[d];
            }
            DenseMatrix Amat(A.GetData(), spacedim, spacedim);
            Amat.Transpose();
            Amat.Invert();

            Vector v1(spacedim);
            Vector temp;

            v1(0) = -1.0; v1(1) = -1.0; v1(2) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + (e*nve+0)*spacedim, spacedim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = -1.0; v1(2) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + (e*nve+1)*spacedim, spacedim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = 1.0; v1(2) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + (e*nve+2)*spacedim, spacedim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = -1.0; v1(1) = 1.0; v1(2) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + (e*nve+3)*spacedim, spacedim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = -1.0; v1(1) = -1.0; v1(2) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + (e*nve+4)*spacedim, spacedim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = -1.0; v1(2) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + (e*nve+5)*spacedim, spacedim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = 1.0; v1(2) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + (e*nve+6)*spacedim, spacedim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = -1.0; v1(1) = 1.0; v1(2) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + (e*nve+7)*spacedim, spacedim);
            Amat.Mult(v1, temp);
            temp += center;
         } //type == 1
      }
      if (type==2)    // local hash mesh
      {
         int ec = 0;
         for (int l=0; l<hash_l_n; l++)
         {
            for (int k=0; k<hash_l_n; k++)
            {
               for (int j=0; j<hash_l_n; j++)
               {
                  double x0 = hashlmin[0] + j*dxyzl[0];
                  double y0 = hashlmin[1] + k*dxyzl[1];
                  double z0 = hashlmin[2] + l*dxyzl[2];

                  o_xyz(ec*nve*spacedim + 0) = x0;
                  o_xyz(ec*nve*spacedim + 1) = y0;
                  o_xyz(ec*nve*spacedim + 2) = z0;

                  o_xyz(ec*nve*spacedim + 3) = x0+dxyzl[0];
                  o_xyz(ec*nve*spacedim + 4) = y0;
                  o_xyz(ec*nve*spacedim + 5) = z0;

                  o_xyz(ec*nve*spacedim + 6) = x0+dxyzl[0];
                  o_xyz(ec*nve*spacedim + 7) = y0+dxyzl[1];
                  o_xyz(ec*nve*spacedim + 8) = z0;

                  o_xyz(ec*nve*spacedim + 9) = x0;
                  o_xyz(ec*nve*spacedim + 10) = y0+dxyzl[1];
                  o_xyz(ec*nve*spacedim + 11) = z0;

                  o_xyz(ec*nve*spacedim + 12) = x0;
                  o_xyz(ec*nve*spacedim + 13) = y0;
                  o_xyz(ec*nve*spacedim + 14) = z0+dxyzl[2];

                  o_xyz(ec*nve*spacedim + 15) = x0+dxyzl[0];
                  o_xyz(ec*nve*spacedim + 16) = y0;
                  o_xyz(ec*nve*spacedim + 17) = z0+dxyzl[2];

                  o_xyz(ec*nve*spacedim + 18) = x0+dxyzl[0];
                  o_xyz(ec*nve*spacedim + 19) = y0+dxyzl[1];
                  o_xyz(ec*nve*spacedim + 20) = z0+dxyzl[2];

                  o_xyz(ec*nve*spacedim + 21) = x0;
                  o_xyz(ec*nve*spacedim + 22) = y0+dxyzl[1];
                  o_xyz(ec*nve*spacedim + 23) = z0+dxyzl[2];

                  ec++;
               }
            }
         }
      } //type == 2
      else if (type==3)
      {
         int ec = 0;
         for (int l=0; l<hash_g_n; l++)
         {
            for (int k=0; k<hash_g_n; k++)
            {
               for (int j=0; j<hash_g_n; j++)
               {
                  double x0 = hashgmin[0] + j*dxyzg[0];
                  double y0 = hashgmin[1] + k*dxyzg[1];
                  double z0 = hashgmin[2] + l*dxyzg[2];

                  o_xyz(ec*nve*spacedim + 0) = x0;
                  o_xyz(ec*nve*spacedim + 1) = y0;
                  o_xyz(ec*nve*spacedim + 2) = z0;

                  o_xyz(ec*nve*spacedim + 3) = x0+dxyzg[0];
                  o_xyz(ec*nve*spacedim + 4) = y0;
                  o_xyz(ec*nve*spacedim + 5) = z0;

                  o_xyz(ec*nve*spacedim + 6) = x0+dxyzg[0];
                  o_xyz(ec*nve*spacedim + 7) = y0+dxyzg[1];
                  o_xyz(ec*nve*spacedim + 8) = z0;

                  o_xyz(ec*nve*spacedim + 9) = x0;
                  o_xyz(ec*nve*spacedim + 10) = y0+dxyzg[1];
                  o_xyz(ec*nve*spacedim + 11) = z0;

                  o_xyz(ec*nve*spacedim + 12) = x0;
                  o_xyz(ec*nve*spacedim + 13) = y0;
                  o_xyz(ec*nve*spacedim + 14) = z0+dxyzg[2];

                  o_xyz(ec*nve*spacedim + 15) = x0+dxyzg[0];
                  o_xyz(ec*nve*spacedim + 16) = y0;
                  o_xyz(ec*nve*spacedim + 17) = z0+dxyzg[2];

                  o_xyz(ec*nve*spacedim + 18) = x0+dxyzg[0];
                  o_xyz(ec*nve*spacedim + 19) = y0+dxyzg[1];
                  o_xyz(ec*nve*spacedim + 20) = z0+dxyzg[2];

                  o_xyz(ec*nve*spacedim + 21) = x0;
                  o_xyz(ec*nve*spacedim + 22) = y0+dxyzg[1];
                  o_xyz(ec*nve*spacedim + 23) = z0+dxyzg[2];

                  ec++;
               }
            }
         }
      }
   }    // spacedim == 3 if-block end
   else   // spacedim == 2
   {
      for (int e=0; e<nel; e++)
      {
         auto box = findptsData2->local.obb[e];
         if (type==0)
         {
            Vector minn(spacedim), maxx(spacedim);
            for (int d=0; d<spacedim; d++)
            {
               minn[d] = box.x[d].min;
               maxx[d] = box.x[d].max;
            }
            o_xyz(e*nve*spacedim + 0) = minn[0];
            o_xyz(e*nve*spacedim + 1) = minn[1];

            o_xyz(e*nve*spacedim + 2) = maxx[0];
            o_xyz(e*nve*spacedim + 3) = minn[1];

            o_xyz(e*nve*spacedim + 4) = maxx[0];
            o_xyz(e*nve*spacedim + 5) = maxx[1];

            o_xyz(e*nve*spacedim + 6) = minn[0];
            o_xyz(e*nve*spacedim + 7) = maxx[1];
         } // type == 0
         else if (type==1)
         {
            Vector center(spacedim), A(spacedim*spacedim);
            for (int d=0; d<spacedim; d++)
            {
               center[d] = box.c0[d];
            }
            for (int d=0; d<spacedim*spacedim; d++)
            {
               A[d] = box.A[d];
            }
            DenseMatrix Amat(A.GetData(), spacedim, spacedim);
            Amat.Transpose();
            Amat.Invert();

            Vector v1(spacedim);
            Vector temp;

            v1(0) = -1.0; v1(1) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + (e*nve+0)*spacedim, spacedim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + (e*nve+1)*spacedim, spacedim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + (e*nve+2)*spacedim, spacedim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = -1.0; v1(1) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + (e*nve+3)*spacedim, spacedim);
            Amat.Mult(v1, temp);
            temp += center;
         } //type == 1
      }
      if (type==2)
      {
         int ec = 0;
         for (int k=0; k<hash_l_n; k++)
         {
            for (int j=0; j < hash_l_n; j++)
            {
               double x0 = hashlmin[0] + j*dxyzl[0];
               double y0 = hashlmin[1] + k*dxyzl[1];

               o_xyz(ec*nve*spacedim + 0) = x0;
               o_xyz(ec*nve*spacedim + 1) = y0;

               o_xyz(ec*nve*spacedim + 2) = x0+dxyzl[0];
               o_xyz(ec*nve*spacedim + 3) = y0;

               o_xyz(ec*nve*spacedim + 4) = x0+dxyzl[0];
               o_xyz(ec*nve*spacedim + 5) = y0+dxyzl[1];

               o_xyz(ec*nve*spacedim + 6) = x0;
               o_xyz(ec*nve*spacedim + 7) = y0+dxyzl[1];

               ec++;
            }
         }
      }
      else if (type==3)
      {
         int ec = 0;
         for (int k=0; k<hash_g_n; k++)
         {
            for (int j=0; j<hash_g_n; j++)
            {
               double x0 = hashgmin[0] + j*dxyzg[0];
               double y0 = hashgmin[1] + k*dxyzg[1];

               o_xyz(ec*nve*spacedim + 0) = x0;
               o_xyz(ec*nve*spacedim + 1) = y0;

               o_xyz(ec*nve*spacedim + 2) = x0+dxyzg[0];
               o_xyz(ec*nve*spacedim + 3) = y0;

               o_xyz(ec*nve*spacedim + 4) = x0+dxyzg[0];
               o_xyz(ec*nve*spacedim + 5) = y0+dxyzg[1];

               o_xyz(ec*nve*spacedim + 6) = x0;
               o_xyz(ec*nve*spacedim + 7) = y0+dxyzg[1];

               ec++;
            }
         }
      }
   } // spacedim == 2 if-block end

   int nsend = type==3 ? 0 : nel*nve*spacedim;
   int nrecv = 0;
   MPI_Status status;
   int vidx = 0;
   int eidx = 0;
   int proc_hash_loc_idx = 0;
   if (myid==save_rank)
   {
      for (int p=0; p<gsl_comm->np; p++)
      {
         if (p!=save_rank)
         {
            MPI_Recv(&nrecv, 1, MPI_INT, p, 444, gsl_comm->c, &status);
            o_xyz.SetSize(nrecv);
            if (nrecv)
            {
               MPI_Recv(o_xyz.GetData(), nrecv, MPI_DOUBLE, p, 445, gsl_comm->c, &status);
            }
         }
         else
         {
            nrecv = type==3 ? nel*nve*spacedim : nsend;
         }
         int nel_recv = nrecv/(spacedim*nve);

         // we keep track of how many hash cells are coming from each rank
         if (p!=save_rank)
         {
            hash_el_count[p] = hash_el_count[p-1] + nel_recv;
         }
         for (int e=0; e<nel_recv; e++)
         {
            for (int j=0; j<nve; j++)
            {
               Vector ver(o_xyz.GetData() + e*nve*spacedim + j*spacedim, spacedim);
               meshbb->AddVertex(ver);
            }

            if (spacedim==2)
            {
               const int inds[4] = {vidx++, vidx++, vidx++, vidx++};
               int attr = eidx+1;
               // for type == 2, we set element attribute based on the
               // proc from which the element must have come.
               if (type==2)
               {
                  if (eidx >= hash_el_count[proc_hash_loc_idx])
                  {
                     proc_hash_loc_idx++;
                  }
                  attr = proc_hash_loc_idx+1;
               }
               else if (type==3)
               {
                  attr = eidx % gsl_comm->np;
                  attr += 1;
               }
               meshbb->AddQuad(inds, attr);
               eidx++;
            }
            else
            {
               const int inds[8] = { vidx++, vidx++, vidx++, vidx++, vidx++, vidx++, vidx++, vidx++ };
               meshbb->AddHex(inds, (eidx++)+1);
            }
         }
      }
      if (spacedim==2)
      {
         meshbb->FinalizeQuadMesh(1, 1, true);
      }
      else
      {
         meshbb->FinalizeHexMesh(1, 1, true);
      }
   }
   else
   {
      MPI_Send(&nsend, 1, MPI_INT, save_rank, 444, gsl_comm->c);
      if (nsend)
      {
         MPI_Send(o_xyz.GetData(), nsend, MPI_DOUBLE, save_rank, 445, gsl_comm->c);
      }
   }

   return meshbb;
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
   if (spacedim==2)
   {
      findpts_free_2((gslib::findpts_data_2 *)this->fdataD);
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
   tensor_product = true;
   int NEsplit = 0;
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      const Geometry::Type gt   = mesh->GetElement(e)->GetGeometryType();
      if (gt == Geometry::TRIANGLE || gt == Geometry::PRISM)
      {
         NEsplit = 3;
         tensor_product = false;
      }
      else if (gt == Geometry::TETRAHEDRON)
      {
         NEsplit = 4;
         tensor_product = false;
      }
      else if (gt == Geometry::PYRAMID)
      {
         NEsplit = 8;
         tensor_product = false;
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

// FIXME: only works for meshes that do not need splitting, just used to set necessary variables
void FindPointsGSLIB::SetupSplitMeshesSurf()
{
   NE_split_total = 0;
   split_element_map.SetSize(0);
   split_element_index.SetSize(0);
   int NEsplit = 0;
   for (int e=0; e<mesh->GetNE(); e++)
   {
      const Geometry::Type gt   = mesh->GetElement(e)->GetGeometryType();
      if (gt==Geometry::SEGMENT || gt == Geometry::SQUARE || gt == Geometry::CUBE)
      {
         NEsplit = 1;
      }
      else
      {
         MFEM_ABORT("Unsupported geometry type.");
      }
      NE_split_total += NEsplit;
      for (int i=0; i<NEsplit; i++)
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
   if (node_vals.UseDevice())
   {
      node_vals.HostWrite();
   }

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

void FindPointsGSLIB::GetNodalValuesSurf(const GridFunction *gf_in,
                                         Vector &node_vals)
{
   const GridFunction *nodes     = gf_in;
   const FiniteElementSpace *fes = nodes->FESpace();

   const int NE       = mesh->GetNE();
   const int vdim     = fes->GetVDim();
   const int maxOrder =
      fes->GetMaxElementOrder(); // for now, equal to GetDof() for any element
   const int dof_1D   = maxOrder+1;
   const int pts_el   = std::pow(dof_1D, dim);     // points nos. in one element
   const int pts_cnt  = NE *
                        pts_el;               // total points nos. in all elements

   // nodes are vdim ordered, i.e., all dim 0 dofs, then all dim 1 dofs, etc.
   node_vals.SetSize(
      vdim*pts_cnt);   // node_vals need to store all vdofs in mesh object
   node_vals = 0.0;
   if (node_vals.UseDevice()) { node_vals.HostWrite(); }

   int gsl_mesh_pt_index =
      0; // gsl_mesh_pt_index indexes the point (dof) under consideration
   for (int ie = 0; ie<NE; ie++)
   {
      const FiniteElement *fe   = fes->GetFE(ie);
      const Geometry::Type gt   = fe->GetGeomType();
      const int dof_cnt_split   = fe->GetDof();
      const IntegrationRule &ir = fe->GetNodes();
      Array<int> dof_map(dof_cnt_split);

      const TensorBasisElement *tbe = dynamic_cast<const TensorBasisElement *>
                                      (fes->GetFE(ie));  // could we use *fe here?
      MFEM_VERIFY(tbe != NULL, "TensorBasis FiniteElement expected.");
      const Array<int> &dm =
         tbe->GetDofMap(); // maps current dof IDs to their lexicographic order

      // GetDofMap() returns an empty array if nodes are already lexicographically ordered
      if (dm.Size()>0)
      {
         dof_map = dm;
      }
      else
      {
         for (int i = 0; i < dof_cnt_split; i++)
         {
            dof_map[i] = i;
         }
      }

      DenseMatrix pos(dof_cnt_split, vdim);
      Vector posV(pos.Data(), dof_cnt_split*vdim);
      Array<int> vdofs(dof_cnt_split * vdim);
      fes->GetElementVDofs(ie, vdofs);    // get non-lexi dof IDs
      nodes->GetSubVector(vdofs, posV);   // posV is used to assign data to pos
      /* At this stage, we have the node coordinates stored in pos DenseMatrix */

      // We also get the reference element positions for debugging reasons
      DenseMatrix pos_ref(ir.GetNPoints(), dim);
      for (int i=0; i<ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         pos_ref(i,0) = ip.x;
         if (dim == 2) { pos_ref(i,1) = ip.y; }
         if (dim == 3) { pos_ref(i,2) = ip.z; }
      }

      for (int j=0; j<dof_cnt_split; j++)   // lexicographic dof ID j
      {
         for (int d=0; d<vdim; d++)     // dof j's dimension d
         {
            node_vals(pts_cnt*d + gsl_mesh_pt_index) = pos(dof_map[j], d);
         }
         gsl_mesh_pt_index++;
      }
   }
}

void FindPointsGSLIB::MapRefPosAndElemIndices()
{
   gsl_mfem_ref.SetSize(points_cnt*dim);
   gsl_mfem_elem.SetSize(points_cnt);
   gsl_mfem_ref = gsl_ref.HostRead();
   gsl_mfem_elem = gsl_elem;

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

void FindPointsGSLIB::MapRefPosAndElemIndicesSurf()
{
   gsl_mfem_ref.SetSize(points_cnt*dim);  // ref dim = dim
   gsl_mfem_elem.SetSize(points_cnt);
   gsl_mfem_ref  = gsl_ref.HostRead();
   gsl_mfem_elem = gsl_elem;

   gsl_mfem_ref += 1.;  // map  [-1, 1] to [0, 2] to [0, 1]
   gsl_mfem_ref *= 0.5;

   int nptorig = points_cnt,
       npt     = points_cnt;

   // tolerance for point to be marked as on element edge/face
   double btol = 1e-12;

   GridFunction *gf_rst_map_temp = NULL;
   int nptsend = 0;

   // Count number of points to send to other procs in "nptsend"
   for (int index=0; index<npt; index++)
   {
      if (gsl_code[index]!=2 && gsl_proc[index]!=gsl_comm->id)
      {
         nptsend +=1;
      }
   }

   // Pack data to send via crystal router
   struct gslib::array *outpt = new gslib::array;
   // NOTE: r has a max size of 2, which is the ref dimension of a 3D physical surface
   struct out_pt { double r[2]; uint index, el, proc, code; };
   array_init(struct out_pt, outpt, nptsend); // see gslib/src/mem.h for array_init
   outpt->n=nptsend;                          // outpt true size

   struct out_pt *pt;
   pt = (struct out_pt *)
        outpt->ptr;          // pointer to the the double array in the first struct

   for (int index=0; index<npt; index++)
   {
      if (gsl_code[index]==2 || gsl_proc[index]==gsl_comm->id)
      {
         continue;
      }
      for (int d=0; d<dim; ++d)   // NOTE: pt->r[1] holds garbage for dim==1!!
      {
         pt->r[d]= gsl_mfem_ref(index*dim + d);
      }
      pt->index = index;
      pt->proc  = gsl_proc[index];
      pt->el    = gsl_elem[index];
      pt->code  = gsl_code[index];
      ++pt;
   }

   // Transfer data to target MPI ranks (stored in proc variable in the struct)
   // see gslib/src/sarray_transfer.cpp
   sarray_transfer(struct out_pt, outpt, proc, 1, cr);

   // Map received points
   npt = outpt->n;
   pt = (struct out_pt *)outpt->ptr;
   for (int index=0; index<npt; index++)
   {
      IntegrationPoint ip;
      ip.x = pt->r[0];
      if (dim==2)
      {
         ip.y = pt->r[1];
      }
      const int elem = pt->el;
      const int mesh_elem = split_element_map[elem];
      const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(mesh_elem);

      // check if the point is on element boundary
      // We always go in currently. So all code after this is if block is redundant.
      const Geometry::Type gt = fe->GetGeomType();
      pt->code = Geometry::CheckPoint(gt, ip, -btol) ? 0 : 1;
      ++pt;
   }

   // Transfer data back to source MPI rank
   sarray_transfer(struct out_pt, outpt, proc, 1, cr);
   npt = outpt->n;

   // First copy mapped information for points on other procs
   pt = (struct out_pt *)outpt->ptr;
   for (int index=0; index<npt; index++)
   {
      gsl_mfem_elem[pt->index] = pt->el;
      for (int d=0; d<dim; d++)
      {
         gsl_mfem_ref(d + pt->index*dim) = pt->r[d];
      }
      gsl_code[pt->index] = pt->code;
      ++pt;
   }
   array_free(outpt);
   delete outpt;

   // Now map information for points on the same proc
   for (int index=0; index<nptorig; index++)
   {
      if (gsl_code[index]!=2 && gsl_proc[index]==gsl_comm->id)
      {
         IntegrationPoint ip;
         Vector mfem_ref(gsl_mfem_ref.GetData()+index*dim, dim);
         real_t *ptemp = mfem_ref.GetData();
         ip.x = ptemp[0];
         if (dim==2)
         {
            ip.y = ptemp[1];
         }

         const int elem = gsl_elem[index];
         const int mesh_elem = split_element_map[elem];
         const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(mesh_elem);
         const Geometry::Type gt = fe->GetGeomType();
         gsl_mfem_elem[index] = mesh_elem;
         gsl_code[index] = Geometry::CheckPoint(gt, ip, -btol) ? 0 : 1;
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
   field_out.HostWrite();
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
      gsl_elem_temp.HostWrite();

      gsl_ref_temp.SetSize(nlocal*dim);
      gsl_ref_temp.UseDevice(true);
      gsl_ref_temp.HostWrite();

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

      if (dim == 2)
      {
         InterpolateLocal2(field_in,
                           gsl_elem_temp,
                           gsl_ref_temp,
                           interp_vals,
                           nlocal, ncomp,
                           nel, dof1Dsol);
      }
      else
      {
         InterpolateLocal3(field_in,
                           gsl_elem_temp,
                           gsl_ref_temp,
                           interp_vals,
                           nlocal, ncomp,
                           nel, dof1Dsol);

      }
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
      gsl_elem_temp.SetSize(n);
      gsl_elem_temp.HostWrite();

      gsl_ref_temp.SetSize(n*dim);
      gsl_ref_temp.HostWrite();

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
      if (dim == 3)
      {
         InterpolateLocal3(field_in,
                           gsl_elem_temp,
                           gsl_ref_temp,
                           interp_vals, n, ncomp,
                           nel, dof1Dsol);
      }
      else
      {
         InterpolateLocal2(field_in,
                           gsl_elem_temp,
                           gsl_ref_temp,
                           interp_vals, n, ncomp,
                           nel, dof1Dsol);
      }
      MPI_Barrier(gsl_comm->c);
      interp_vals.HostReadWrite();

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

void FindPointsGSLIB::InterpolateSurfOnDevice(const Vector &field_in,
                                              Vector &field_out,
                                              const int nel,
                                              const int ncomp,
                                              const int dof1Dsol,
                                              const int gf_ordering,
                                              MemoryType mt)
{
   field_out.HostWrite();
   struct gslib::array src, outpt;
   int nlocal = 0;
   /* weed out unfound points, send out */
   Array<int> gsl_elem_temp;
   Vector gsl_ref_temp;
   Array<int> index_temp;
   {
      int index;
      const unsigned int *code = gsl_code.GetData(),
                          *proc = gsl_proc.GetData(),
                           *el  = gsl_elem.GetData();
      const dfloat *r = gsl_ref.GetData();

      int numSend = 0;

      for (index=0; index<points_cnt; ++index)
      {
         numSend += (gsl_code[index] != CODE_NOT_FOUND &&
                     gsl_proc[index] != gsl_comm->id);
         nlocal += (gsl_code[index] != CODE_NOT_FOUND &&
                    gsl_proc[index] == gsl_comm->id);
      }

      gsl_elem_temp.SetSize(nlocal);
      gsl_elem_temp.HostWrite();

      gsl_ref_temp.SetSize(nlocal*dim);
      gsl_ref_temp.UseDevice(true);
      gsl_ref_temp.HostWrite();

      index_temp.SetSize(nlocal);

      evalSrcPt_t *pt;
      array_init(evalSrcPt_t, &src, numSend);
      pt = (evalSrcPt_t *)src.ptr;

      int ctr = 0;
      for (index=0; index<points_cnt; ++index)
      {
         if (*code!=CODE_NOT_FOUND && *proc!=gsl_comm->id)
         {
            for (int d=0; d<dim; ++d)
            {
               pt->r[d] = r[d];
            }
            pt->index = index;
            pt->proc = *proc;
            pt->el = *el;
            ++pt;
         }
         else if (*code!=CODE_NOT_FOUND && *proc==gsl_comm->id)
         {
            gsl_elem_temp[ctr] = *el;
            for (int d=0; d<dim; ++d)
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

      if (spacedim==2)
      {
         InterpolateSurfLocal2(field_in,
                               gsl_elem_temp,
                               gsl_ref_temp,
                               interp_vals,
                               nlocal,
                               ncomp,
                               nel,
                               dof1Dsol);
      }
      else
      {
         InterpolateSurfLocal3(field_in,
                               gsl_elem_temp,
                               gsl_ref_temp,
                               interp_vals,
                               nlocal,
                               ncomp,
                               nel,
                               dof1Dsol);

      }
      MPI_Barrier(gsl_comm->c);

      interp_vals.HostReadWrite();

      // now put these in correct positions
      int interp_Offset = interp_vals.Size()/ncomp;
      for (int i=0; i<ncomp; i++)
      {
         for (int j=0; j<nlocal; j++)
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
      gsl_elem_temp.SetSize(n);
      gsl_elem_temp.HostWrite();

      gsl_ref_temp.SetSize(n*dim);
      gsl_ref_temp.HostWrite();

      spt = (evalSrcPt_t *)src.ptr;
      //   opt = (evalOutPt_t *)outpt.ptr;
      for (int i=0; i<n; i++, ++spt)
      {
         gsl_elem_temp[i] = spt->el;
         for (int d=0; d<dim; d++)
         {
            gsl_ref_temp(i*dim + d) = spt->r[d];
         }
      }

      Vector interp_vals(n*ncomp);
      interp_vals.UseDevice(true);
      if (spacedim==3)
      {
         InterpolateSurfLocal3( field_in,
                                gsl_elem_temp,
                                gsl_ref_temp,
                                interp_vals,
                                n,
                                ncomp,
                                nel,
                                dof1Dsol );
      }
      else
      {
         InterpolateSurfLocal2( field_in,
                                gsl_elem_temp,
                                gsl_ref_temp,
                                interp_vals,
                                n,
                                ncomp,
                                nel,
                                dof1Dsol );
      }
      MPI_Barrier(gsl_comm->c);
      interp_vals.HostReadWrite();

      // Now the interpolated values need to be sent back component wise
      int Offset = interp_vals.Size()/ncomp;
      for (int i=0; i<ncomp; i++)
      {
         spt = (evalSrcPt_t *)src.ptr;
         array_init(evalOutPt_t, &outpt, n);
         outpt.n = n;
         opt = (evalOutPt_t *)outpt.ptr;

         for (int j=0; j<n; j++)
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
   if (Device::IsEnabled() && field_in.UseDevice() &&
       mesh->GetNumGeometries(dim) == 1 && mesh->GetNE() > 0 && tensor_product)
   {
      MFEM_VERIFY(fec_h1,"Only h1 functions supported on device right now.");
      MFEM_VERIFY(fec_h1->GetBasisType() == BasisType::GaussLobatto,
                  "basis not supported");

      Vector node_vals;
      const ElementDofOrdering ord = ElementDofOrdering::LEXICOGRAPHIC;
      const Operator *R = field_in.FESpace()->GetElementRestriction(ord);
      node_vals.UseDevice(true);
      node_vals.SetSize(R->Height(), Device::GetDeviceMemoryType());
      R->Mult(field_in, node_vals); //orders fields (N^D x VDIM x NEL)
      // GetNodalValues(&field_in, node_vals); // orders (N^D x NEL x VDIM)

      const int ncomp  = field_in.FESpace()->GetVDim();
      const int maxOrder = field_in.FESpace()->GetMaxElementOrder();
      DEV.dof1dsol =  maxOrder+1;

      DEV.gll1dsol.UseDevice(true);
      DEV.gll1dsol.SetSize(DEV.dof1dsol);
      DEV.lagcoeffsol.UseDevice(true);
      DEV.lagcoeffsol.SetSize(DEV.dof1dsol);
      DEV.gll1dsol.HostWrite();
      DEV.lagcoeffsol.HostWrite();
      if (DEV.dof1dsol != DEV.dof1d)
      {
         gslib::lobatto_nodes(DEV.gll1dsol.HostWrite(), DEV.dof1dsol);
         gslib::gll_lag_setup(DEV.lagcoeffsol.HostWrite(), DEV.dof1dsol);
      }
      else
      {
         DEV.gll1dsol = DEV.gll1d.HostRead();
         DEV.lagcoeffsol = DEV.lagcoeff.HostRead();
      }

      field_out.SetSize(points_cnt*ncomp);
      field_out.UseDevice(true);

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
      setupSW.Stop();
      interpolate_general_time = setupSW.RealTime();
      if (!fec_l2 || avgtype == AvgType::NONE) { return; }
   }

   // For points on element borders, project the L2 GridFunction to H1 and
   // re-interpolate.
   setupSW.Clear();
   setupSW.Start();
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
   setupSW.Stop();
   interpolate_l2_pass2_time = setupSW.RealTime();
}

void FindPointsGSLIB::InterpolateSurf(const GridFunction &field_in,
                                      Vector &field_out)
{
   const int  gf_order   = field_in.FESpace()->GetMaxElementOrder(),
              mesh_order = mesh->GetNodalFESpace()->GetMaxElementOrder();

   const FiniteElementCollection *fec_in =  field_in.FESpace()->FEColl();
   const H1_FECollection         *fec_h1 = dynamic_cast<const H1_FECollection *>
                                           (fec_in);
   // const L2_FECollection         *fec_l2 = dynamic_cast<const L2_FECollection *>(fec_in);

   setupSW.Clear();
   setupSW.Start();

   if ( //Device::IsEnabled() &&
      field_in.UseDevice()
      && mesh->GetNumGeometries(dim)==1
      && mesh->GetNE()>0
      && ( mesh->GetElementType(0) == Element::SEGMENT
           || mesh->GetElementType(0) == Element::QUADRILATERAL ) )
   {
      MFEM_VERIFY(fec_h1,"Only h1 functions supported on device right now.");
      MFEM_VERIFY(fec_h1->GetBasisType() == BasisType::GaussLobatto,
                  "basis not supported");
      Vector node_vals;
      node_vals.UseDevice(true);

      const ElementDofOrdering ord = ElementDofOrdering::LEXICOGRAPHIC;
      const Operator *R            = field_in.FESpace()->GetElementRestriction(ord);
      node_vals.SetSize(R->Height());
      R->Mult(field_in, node_vals); //orders fields (N^D x VDIM x NEL)

      const int ncomp    = field_in.FESpace()->GetVDim();
      const int maxOrder = field_in.FESpace()->GetMaxElementOrder();
      DEV.dof1dsol       =  maxOrder+1;

      DEV.gll1dsol.UseDevice(true);
      DEV.gll1dsol.SetSize(DEV.dof1dsol);
      DEV.lagcoeffsol.UseDevice(true);
      DEV.lagcoeffsol.SetSize(DEV.dof1dsol);
      DEV.gll1dsol.HostWrite();
      DEV.lagcoeffsol.HostWrite();
      if (DEV.dof1dsol != DEV.dof1d)
      {
         Vector temp(DEV.dof1dsol);
         gslib::lobatto_nodes(temp.GetData(), DEV.dof1dsol);
         DEV.gll1dsol = temp.GetData();
         MFEM_DEVICE_SYNC;

         gslib::gll_lag_setup(temp.GetData(), DEV.dof1dsol);
         DEV.lagcoeffsol = temp.GetData();
         MFEM_DEVICE_SYNC;
      }
      else
      {
         DEV.gll1dsol    = DEV.gll1d.HostRead();
         DEV.lagcoeffsol = DEV.lagcoeff.HostRead();
      }
      MFEM_DEVICE_SYNC;

      field_out.SetSize(points_cnt*ncomp);
      field_out.UseDevice(true);

      // At this point make sure FindPoints was called with device mode?
      // Otherwise copy necessary data?
      InterpolateSurfOnDevice( node_vals,
                               field_out,
                               NE_split_total,
                               ncomp,
                               DEV.dof1dsol,
                               field_in.FESpace()->GetOrdering(),
                               field_in.GetMemory().GetMemoryType() );

      setupSW.Stop();
      interpolate_h1_time = setupSW.RealTime();
      return;
   }
   else
   {
      MFEM_ABORT("In else branch of GSLIB::InterpolateSurf!");
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
                        points_cnt, node_vals.GetData(),
                        (gslib::findpts_data_2 *)this->fdataD);
      }
      else
      {
         findpts_eval_3(field_out.GetData()+dataptrout, sizeof(double),
                        gsl_code.GetData(),       sizeof(unsigned int),
                        gsl_proc.GetData(),    sizeof(unsigned int),
                        gsl_elem.GetData(),    sizeof(unsigned int),
                        gsl_ref.GetData(),     sizeof(double) * dim,
                        points_cnt, node_vals.GetData(),
                        (gslib::findpts_data_3 *)this->fdataD);
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
     double * const elx[2] =
     {
        pts_cnt == 0 ? nullptr : &gsl_mesh(0),
        pts_cnt == 0 ? nullptr : &gsl_mesh(pts_cnt)
     };
     fdataD = findptsms_setup_2(gsl_comm, elx, nr, NEtot, mr, bb_t,
                                 pts_cnt, pts_cnt, npt_max, newt_tol,
                                 &u_meshid, &distfint(0));
  }
  else
  {
     unsigned nr[3] = { dof1D, dof1D, dof1D };
     unsigned mr[3] = { 2*dof1D, 2*dof1D, 2*dof1D };
     double * const elx[3] =
     {
        pts_cnt == 0 ? nullptr : &gsl_mesh(0),
        pts_cnt == 0 ? nullptr : &gsl_mesh(pts_cnt),
        pts_cnt == 0 ? nullptr : &gsl_mesh(2*pts_cnt)
     };
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
                 points_cnt, (gslib::findpts_data_2 *)this->fdataD);
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
