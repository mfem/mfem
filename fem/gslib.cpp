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

#include "gslib.hpp"
#include "geom.hpp"
#include "../mesh/bb_grid_map.hpp"

#ifdef MFEM_USE_GSLIB

// Ignore warnings from the gslib header (GCC version)
#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2

// External GSLIB header (the MFEM header is gslib.hpp)
namespace gslib
{
#include "gslib.h"
#ifndef GSLIB_RELEASE_VERSION //gslib v1.0.7
#define GSLIB_RELEASE_VERSION 10007
#endif
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
     dim(-1), spacedim(-1), points_cnt(-1), setupflag(false),
     default_interp_value(0),
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
   int initialized = 0;
   MPI_Initialized(&initialized);
   if (!initialized) { MPI_Init(NULL, NULL); }
   MPI_Comm comm = MPI_COMM_WORLD;
   comm_init(gsl_comm, comm);
#else
   comm_init(gsl_comm, 0);
#endif
   crystal_init(cr, gsl_comm);
}

FindPointsGSLIB::~FindPointsGSLIB()
{
   crystal_free(cr);
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
     dim(-1), spacedim(-1), points_cnt(-1), setupflag(false),
     default_interp_value(0),
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
   crystal_init(cr, gsl_comm);
}
#endif

void FindPointsGSLIB::Setup(Mesh &m, const double bb_t, const double newt_tol,
                            const int npt_max)
{
   MFEM_VERIFY(m.GetNodes() != NULL, "Mesh nodes are required.");
   const int meshOrder = m.GetNodes()->FESpace()->GetMaxElementOrder();

   // call FreeData if FindPointsGSLIB::Setup has been called already
   if (m.Dimension() != m.SpaceDimension())
   {
      SetupSurf(m, bb_t, newt_tol, npt_max);
      return;
   }
   if (setupflag) { FreeData(); }

   mesh = &m;
   dim  = mesh->Dimension();
   spacedim = dim;
   unsigned dof1D = meshOrder + 1;

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

   mesh_points_cnt = gsl_mesh.Size()/dim;

   DEV.local_hash_size = mesh_points_cnt;
   DEV.dof1d = (int)dof1D;
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
   setupflag = true;
}


#define gslib_tmalloc(type, count) \
  ((type*) gslib::smalloc((count)*sizeof(type),__FILE__,__LINE__))
#define DO_MAX(a,b) do { unsigned temp = b; if(temp>a) a=temp; } while(0)

/* Calculates the diagonal length of the bounding box and expands its bounds by
 * 0.5*len*tol at both its min and max values.
 * Returns the length of the diagonal (could be used for expanding obboxes).
 */
double dbl_range_diag_expand_2(struct gslib::dbl_range *b, double tol)
{
   double l[2] = { b[0].max-b[0].min, b[1].max-b[1].min };
   double len = sqrt(l[0]*l[0] + l[1]*l[1])*0.5*tol;
   for (int i=0; i<2; i++)
   {
      b[i].min = b[i].min - len;
      b[i].max = b[i].max + len;
   }
   return len;
}

double dbl_range_diag_expand_3(struct gslib::dbl_range *b, double tol)
{
   double l[3] = { b[0].max-b[0].min, b[1].max-b[1].min, b[2].max-b[2].min };
   double len = sqrt(l[0]*l[0] + l[1]*l[1] + l[2]*l[2])*0.5*tol;
   for (int i=0; i<3; i++)
   {
      b[i].min = b[i].min - len;
      b[i].max = b[i].max + len;
   }
   return len;
}

static void bbox_2_tfm(double *out, const double x0[2], const double Ji[4],
                       const double *x, const double *y, unsigned n)
{
   unsigned i;
   for (i=0; i<n; ++i)
   {
      const double dx = x[i]-x0[0], dy = y[i]-x0[1];
      out[  i] = Ji[0]*dx + Ji[1]*dy;
      out[n+i] = Ji[2]*dx + Ji[3]*dy;
   }
}

static void bbox_3_tfm(double *out, const double x0[3], const double Ji[9],
                       const double *x, const double *y, const double *z,
                       unsigned n)
{
   unsigned i;
   for (i=0; i<n; ++i)
   {
      const double dx = x[i]-x0[0], dy = y[i]-x0[1], dz = z[i]-x0[2];
      out[    i] = Ji[0]*dx + Ji[1]*dy + Ji[2]*dz;
      out[  n+i] = Ji[3]*dx + Ji[4]*dy + Ji[5]*dz;
      out[2*n+i] = Ji[6]*dx + Ji[7]*dy + Ji[8]*dz;
   }
}

static struct gslib::dbl_range dbl_range_expand(struct gslib::dbl_range b,
                                                double tol)
{
   double a = (b.min+b.max)/2, l = (b.max-b.min)*(1+tol)/2;
   struct gslib::dbl_range m;
   m.min = a-l, m.max = a+l;
   return m;
}

void obboxsurf_calc_3(Vector &bb,
                      const double *const elx[3],
                      const unsigned n,
                      uint nel,
                      const unsigned m,
                      const double tol)
{
   const double *x = elx[0], *y = elx[1], *z = elx[2];
   const unsigned nr = n, ns = n;
   const unsigned mr = m, ms = m;
   const int n_el_ents = 18; // 3(c0) + 3(aabb_min) + 3(aabb_max) + 9(A)

   const unsigned nrs = nr*ns;
   const unsigned lbsize0 = gslib::lob_bnd_size(nr,mr),
                  lbsize1 = gslib::lob_bnd_size(ns,ms);

   unsigned wsize = 3*nr*ns+2*mr*(ns+ms+1);
   DO_MAX(wsize,2*nr*ns+3*nr);
   DO_MAX(wsize,gslib::gll_lag_size(nr));
   DO_MAX(wsize,gslib::gll_lag_size(ns));

   Vector datavec(2*(nr+ns)+lbsize0+lbsize1+wsize);
   double *data = datavec.GetData();

   {
      double *const I0r            = data;
      double *const I0s            = I0r  + 2*nr;
      double *const lob_bnd_data_r = data + 2*(nr+ns);
      double *const lob_bnd_data_s = data + 2*(nr+ns) + lbsize0;
      double *const work           = data + 2*(nr+ns) + lbsize0 + lbsize1;

#define SETUP_DIR(r) do { \
      gslib::lagrange_fun *const lag = gslib::gll_lag_setup(work, n##r); \
      lag(I0##r, work,n##r,1, 0); \
      gslib::lob_bnd_setup(lob_bnd_data_##r, n##r,m##r); \
    } while(0)
      SETUP_DIR(r);
      SETUP_DIR(s);
#undef SETUP_DIR

      for (int ie = 0; ie < nel; ie++,x+=nrs,y+=nrs,z+=nrs)
      {
         struct gslib::dbl_range ab[3];
         struct gslib::dbl_range tb[3];
         double x0[3], tv[9], A[9];

         /*
          * Find the center of the element (r=0 ref. coord.) in physical space
          * and store in x0.
          * tv[0], tv[1], tv[2]: kept empty at this point for convenience.
          * tv[3], tv[4]: dx/dr, dx/ds
          * tv[5], tv[6]: dy/dr, dy/ds
          * tv[7], tv[8]: dz/dr, dz/ds
          */
         x0[0] = gslib::tensor_ig2(tv+3, I0r,nr, I0s,ns, x, work);
         x0[1] = gslib::tensor_ig2(tv+5, I0r,nr, I0s,ns, y, work);
         x0[2] = gslib::tensor_ig2(tv+7, I0r,nr, I0s,ns, z, work);

         // tangent vector 1 moved to tv[0], tv[1], tv[2]
         tv[0] = tv[3], tv[1] = tv[5], tv[2] = tv[7];
         // tangent vector 2 moved to tv[3], tv[4], tv[5]
         tv[3] = tv[4], tv[4] = tv[6], tv[5] = tv[8];
         // normal vector to the plane formed by t1 and t2 (cross product)
         // is stored in tv[6], tv[7], tv[8]
         tv[6] = tv[1]*tv[5] - tv[2]*tv[4];
         tv[7] = tv[2]*tv[3] - tv[0]*tv[5];
         tv[8] = tv[0]*tv[4] - tv[1]*tv[3];
         // normalize the normal vector
         const double nmag  = sqrt(tv[6]*tv[6] + tv[7]*tv[7] + tv[8]*tv[8]);
         tv[6] = tv[6]/nmag;
         tv[7] = tv[7]/nmag;
         tv[8] = tv[8]/nmag;

         // Rodrigues formula to compute the rotation matrix
         // Axis of rotation is n x [0,0,1] = [n_2, -n_1, 0], and we must
         // normalize it
         double nmag2 = tv[6]*tv[6] + tv[7]*tv[7];
         if (nmag2 > 0)
         {
            nmag2 = sqrt(nmag2);
            tv[7] = tv[7]/nmag2;
            tv[6] = tv[6]/nmag2;
         }
         double kx = tv[7];
         double ky = -tv[6];
         double kz = 0.0;

         double ct = tv[8];
         double st = nmag2; //1.0 - ct*ct;

         // row-major
         A[0] = 1.0 + st*0.0 + (1.0-ct)*(-ky*ky-kz*kz);
         A[1] = 0.0 + st*(0.0) + (1.0-ct)*(kx*ky);
         A[2] = 0.0 + st*(ky) + (1.0-ct)*(kx*kz);

         A[3] = 0.0 + st*(0.0) + (1.0-ct)*(kx*ky);
         A[4] = 1.0 + st*(0.0) + (1.0-ct)*(-kx*kx-kz*kz);
         A[5] = 0.0 + st*(-kx) + (1.0-ct)*(ky*kz);

         A[6] = 0.0 + st*(-ky) + (1.0-ct)*(kx*kz);
         A[7] = 0.0 + st*(kx) + (1.0-ct)*(ky*kz);
         A[8] = 1.0 + st*(0.0) + (1.0-ct)*(-kx*kx-ky*ky);

         /* double work[2*m##r*(n##s+m##s+1)] */
#define DO_BOUND(bnd,r,s,x,work) do { \
        bnd = gslib::lob_bnd_2(lob_bnd_data_##r,n##r,m##r, \
                        lob_bnd_data_##s,n##s,m##s, \
                        x, work); \
      } while(0)

         DO_BOUND(ab[0],r,s,x,work);
         DO_BOUND(ab[1],r,s,y,work);
         DO_BOUND(ab[2],r,s,z,work);
         // expand bounding box based on (tol*diagonal_length) in each direction
         // to avoid 0 extent in 1 direction.
         double aabb_diag_len = dbl_range_diag_expand_3(ab, tol);
         bb[n_el_ents*ie + 3] = ab[0].min;
         bb[n_el_ents*ie + 4] = ab[1].min;
         bb[n_el_ents*ie + 5] = ab[2].min;
         bb[n_el_ents*ie + 6] = ab[0].max;
         bb[n_el_ents*ie + 7] = ab[1].max;
         bb[n_el_ents*ie + 8] = ab[2].max;

         Array<double> xtfm(3*nrs);
         bbox_3_tfm(xtfm.GetData(), x0,A, x,y,z,nrs);
         // The rotated z-coords are used to calculate z-bounds.
         DO_BOUND(tb[2],r,s,xtfm.GetData()+2*nrs,work);

         tb[2].min -= aabb_diag_len;
         tb[2].max += aabb_diag_len;

         // Also apply A to the tangent vectors, which allows us to
         // calculate the Jacobian matrix at the rotated element center.
         // NOTE that the z components of the rotated tangent vectors will
         // become zero, since the normal vector is aligned with the z-axis.
         double J[4], Ji[4];
         J[0] = A[0]*tv[0] + A[1]*tv[1] + A[2]*tv[2]; // rotated dx/dr
         J[1] = A[0]*tv[3] + A[1]*tv[4] + A[2]*tv[5]; // rotated dx/ds
         J[2] = A[3]*tv[0] + A[4]*tv[1] + A[5]*tv[2]; // rotated dy/dr
         J[3] = A[3]*tv[3] + A[4]*tv[4] + A[5]*tv[5]; // rotated dy/ds
         DenseMatrix JM(J, 2, 2);
         DenseMatrix JiM(Ji, 2, 2);
         CalcInverse(JM, JiM);

         // Now transform the already rotated x,y coordinates according to Ji to
         // their reference space.
         // Important to note that the nodes used here already have the element
         // center at (0,0). Hence, Ji can be directly applied to them.
         for (unsigned i=0; i<nrs; ++i)
         {
            const double xt = xtfm[i], yt = xtfm[nrs+i];
            xtfm[    i] = Ji[0]*xt + Ji[1]*yt;
            xtfm[nrs+i] = Ji[2]*xt + Ji[3]*yt;
         }
         // Bound these reference space xy coordinates
         DO_BOUND(tb[0],r,s,xtfm,work);
         DO_BOUND(tb[1],r,s,xtfm+nrs,work);
         // Expand the bounds based on the tol
         tb[0] = dbl_range_expand(tb[0],tol);
         tb[1] = dbl_range_expand(tb[1],tol);
#undef DO_BOUND

         /* We now have a BB whose bounds represent bounds of a OBB around the
          * original element.
          *
          * We calculate the center of the OBB in physical space by calculating
          * the center of this BB, which is the same as the displacement needed
          * to move from the element center in the transformed space to the BB center. This displacement is then untransformed by applying (Ji.A)^-1
          * to it, and added to known physical element center.
          *
          * This BB does not necessarily have known fixed size like [-1,1].
          * So, we premultiply a length scaling matrix, say L, to Ji.A to
          * L.Ji.A. This is the total transformation needed to move a physical
          * location that is inside the physical OBB to a location within [-1,1]^3. Any transformed point not in [-1,1]^3 is outside the OBB.
          *
          * It must be noted: this transformation matrix is only applied to
          * points that have been translated by the physical OBB center.
          */
         {
            // The center of the BB in the transformed space
            const double av0 = (tb[0].min+tb[0].max)/2,
                         av1 = (tb[1].min+tb[1].max)/2,
                         av2 = (tb[2].min+tb[2].max)/2;
            // First untransform the x,y coordinates by J to obtain all
            // components in the rotated space
            const double Jav0 = J[0]*av0 + J[1]*av1,
                         Jav1 = J[2]*av0 + J[3]*av1;
            // The physical displacement needed to move from the element center
            // to the OBB center is calculated by "un"rotating {Jav0,Jav1,av2}
            // by applying inverse of A.
            // The physical untransformed OBB center can then be obtained.

            bb[n_el_ents*ie + 0] = x0[0] + A[0]*Jav0 + A[3]*Jav1 + A[6]*av2;
            bb[n_el_ents*ie + 1] = x0[1] + A[1]*Jav0 + A[4]*Jav1 + A[7]*av2;
            bb[n_el_ents*ie + 2] = x0[2] + A[2]*Jav0 + A[5]*Jav1 + A[8]*av2;
         }

         // Finally, obtain (L.Ji.A) and store it in out->A
         {
            // The scaling matrix L's diagonal terms, needed to scale the
            // transformation to [-1,1]^3.
            const double di0 = 2/(tb[0].max-tb[0].min),
                         di1 = 2/(tb[1].max-tb[1].min),
                         di2 = 2/(tb[2].max-tb[2].min);

            // We finally construct the final transformation matrix A=L.Ji.A.
            // This maps a position relative to OBB center to a position in
            // [-1,1]^3, if the position is inside the OBB.
            bb[n_el_ents*ie + 9 ]=di0*(Ji[0]*A[0] + Ji[1]*A[3]);
            bb[n_el_ents*ie + 10]=di0*(Ji[0]*A[1] + Ji[1]*A[4]);
            bb[n_el_ents*ie + 11]=di0*(Ji[0]*A[2] + Ji[1]*A[5]);
            bb[n_el_ents*ie + 12]=di1*(Ji[2]*A[0] + Ji[3]*A[3]);
            bb[n_el_ents*ie + 13]=di1*(Ji[2]*A[1] + Ji[3]*A[4]);
            bb[n_el_ents*ie + 14]=di1*(Ji[2]*A[2] + Ji[3]*A[5]);
            bb[n_el_ents*ie + 15]=di2*A[6],
            bb[n_el_ents*ie + 16]=di2*A[7],
            bb[n_el_ents*ie + 17]=di2*A[8];
         }
      }
   }
}

void obboxedge_calc_2(Vector &bb,
                       const double *const elx[2],
                       const unsigned nr,
                       uint nel,
                       const unsigned mr,
                       const double tol)
{
   const double *x   = elx[0];
   const double *y   = elx[1];
   const int n_el_ents = 10; // 2(c0) + 2(aabb_min) + 2(aabb_max) + 4(A)

   const unsigned lbsize0 = gslib::lob_bnd_size(nr,mr);

   unsigned wsize = 2*nr+2*mr;
   DO_MAX(wsize,gslib::gll_lag_size(nr));

   Vector datavec(2*nr + lbsize0 + wsize);
   double *data = datavec.GetData();

   double *const I0r = data;
   double *const lob_bnd_data_r = data + 2*nr;
   double *const work = data + 2*nr + lbsize0;

#define SETUP_DIR(r) do { \
      gslib::lagrange_fun *const lag = gslib::gll_lag_setup(work, n##r); \
      lag(I0##r, work,n##r,1, 0); \
      gslib::lob_bnd_setup(lob_bnd_data_##r, n##r,m##r); \
    } while(0)

   SETUP_DIR(r);
#undef SETUP_DIR

   for (int ie = 0; ie < nel; ie++,x+=nr,y+=nr)
   {
      double x0[2], A[4];
      struct gslib::dbl_range ab[2], tb[2];

      x0[0] = gslib::tensor_ig1(A,I0r,nr,x);
      x0[1] = gslib::tensor_ig1(A+1,I0r,nr,y);
      //
      A[2] = sqrt(A[0]*A[0] + A[1]*A[1]);
      A[0] = A[0]/A[2];
      A[1] = A[1]/A[2];
      A[2] = -A[1];
      A[3] =  A[0];

      /* double work[2*m##r]
       * Find the bounds along a specific physical dimension.
       */
#define DO_BOUND(bnd,r,x,work) do { \
        bnd = gslib::lob_bnd_1(lob_bnd_data_##r,n##r,m##r, x, work); \
      } while(0)

      /* double work[2*n##r + 2*m##r] */
#define DO_EDGE(r,x,y,work) do { \
        DO_BOUND(ab[0],r,x,work); \
        DO_BOUND(ab[1],r,y,work); \
        bbox_2_tfm(work, x0,A, x,y,n##r); \
        DO_BOUND(tb[0],r,(work),(work)+2*n##r); \
        DO_BOUND(tb[1],r,(work)+n##r,(work)+2*n##r); \
      } while(0)
      DO_EDGE(r,x,y,work);
#undef DO_EDGE
#undef DO_BOUND

      double aabb_diag_len = dbl_range_diag_expand_2(ab, tol);

      const double av0 = (tb[0].min+tb[0].max)/2,
                   av1 = (tb[1].min+tb[1].max)/2;
      const double dx0 =  A[0]*av0 - A[1]*av1,
                   dx1 = -A[2]*av0 + A[3]*av1;
      bb[n_el_ents*ie + 0] = x0[0] + dx0;
      bb[n_el_ents*ie + 1] = x0[1] + dx1;
      bb[n_el_ents*ie + 2] = ab[0].min;
      bb[n_el_ents*ie + 3] = ab[1].min;
      bb[n_el_ents*ie + 4] = ab[0].max;
      bb[n_el_ents*ie + 5] = ab[1].max;

      // Expand by aabb_diag_len
      tb[0].min -= aabb_diag_len;
      tb[0].max += aabb_diag_len;
      tb[1].min -= aabb_diag_len;
      tb[1].max += aabb_diag_len;
      const double di0 = 2/(tb[0].max-tb[0].min),
                   di1 = 2/(tb[1].max-tb[1].min);
      bb[n_el_ents*ie + 6]=di0*A[0];
      bb[n_el_ents*ie + 7]=di0*A[1];
      bb[n_el_ents*ie + 8]=di1*A[2];
      bb[n_el_ents*ie + 9]=di1*A[3];
   }
}

void obboxedge_calc_3(Vector &bb,
                       const double *const elx[3],
                       const unsigned nr,
                       uint nel,
                       const unsigned mr,
                       const double tol)
{
   const double *x   = elx[0];
   const double *y   = elx[1];
   const double *z   = elx[2];
   const int n_el_ents = 18; // 3(c0) + 3(aabb_min) + 3(aabb_max) + 9(A)

   const unsigned lbsize0 = gslib::lob_bnd_size(nr,mr);
   unsigned wsize = 4*nr+2*mr;
   DO_MAX(wsize,gslib::gll_lag_size(nr));

   Vector datavec(2*nr + lbsize0 + wsize);
   double *data = datavec.GetData();

   {
      double *const I0r = data;                         // 2*nr doubles
      double *const lob_bnd_data_r = data + 2*nr;        // lbsize0 doubles
      double *const work = data + 2*nr + lbsize0;        // wsize doubles

#define SETUP_DIR(r) do { \
      gslib::lagrange_fun *const lag = gslib::gll_lag_setup(work, n##r); \
      lag(I0##r, work,n##r,1, 0); \
      gslib::lob_bnd_setup(lob_bnd_data_##r, n##r,m##r); \
    } while(0)

      SETUP_DIR(r);
#undef SETUP_DIR

      for (int ie = 0; ie < nel; ie++,x+=nr,y+=nr,z+=nr)
      {
         double x0[3], A[9], Ai[9];
         struct gslib::dbl_range ab[3], tb[3];

         x0[0] = gslib::tensor_ig1(A,I0r,nr,x);
         x0[1] = gslib::tensor_ig1(A+1,I0r,nr,y);
         x0[2] = gslib::tensor_ig1(A+2,I0r,nr,z);

         // normalize the normal vector
         double nmag  = A[0]*A[0] + A[1]*A[1] + A[2]*A[2];
         if (nmag > 0)
         {
            nmag = sqrt(nmag);
            A[0] = A[0]/nmag;
            A[1] = A[1]/nmag;
            A[2] = A[2]/nmag;
         }

         double nmag2 = A[0]*A[0] + A[1]*A[1];
         if (nmag2 > 0)
         {
            nmag2 = sqrt(nmag2);
            A[1] = A[1]/nmag2;
            A[0] = A[0]/nmag2;
         }
         double kx = A[1];
         double ky = -A[0];
         double kz = 0.0;

         double ct = A[2];
         double st = nmag2; //1.0 - ct*ct;

         // row-major rotation matrix to align the tangent with x-axis
         A[0] = 1.0 + st*0.0 + (1.0-ct)*(-ky*ky-kz*kz);
         A[1] = 0.0 + st*(0.0) + (1.0-ct)*(kx*ky);
         A[2] = 0.0 + st*(ky) + (1.0-ct)*(kx*kz);

         A[3] = 0.0 + st*(0.0) + (1.0-ct)*(kx*ky);
         A[4] = 1.0 + st*(0.0) + (1.0-ct)*(-kx*kx-kz*kz);
         A[5] = 0.0 + st*(-kx) + (1.0-ct)*(ky*kz);

         A[6] = 0.0 + st*(-ky) + (1.0-ct)*(kx*kz);
         A[7] = 0.0 + st*(kx) + (1.0-ct)*(ky*kz);
         A[8] = 1.0 + st*(0.0) + (1.0-ct)*(-kx*kx-ky*ky);

         DenseMatrix AM(A, 3, 3);
         DenseMatrix AiM(Ai, 3, 3);
         CalcInverse(AM, AiM);

#define DO_BOUND(bnd,r,x,work) do { \
        bnd = gslib::lob_bnd_1(lob_bnd_data_##r,n##r,m##r, x, work); \
      } while(0)

         /* double work[2*n##r + 2*m##r] */
#define DO_EDGE(r,x,y,z,work) do { \
        DO_BOUND(ab[0],r,x,work); \
        DO_BOUND(ab[1],r,y,work); \
        DO_BOUND(ab[2],r,z,work); \
        bbox_3_tfm(work, x0, A, x,y,z,n##r); \
        DO_BOUND(tb[0],r,(work),(work)+3*n##r); \
        DO_BOUND(tb[1],r,(work)+n##r,(work)+3*n##r); \
        DO_BOUND(tb[2],r,(work)+2*n##r,(work)+3*n##r); \
      } while(0)
         DO_EDGE(r,x,y,z,work);
#undef DO_EDGE
#undef DO_BOUND

         double aabb_diag_len = dbl_range_diag_expand_3(ab, tol);

         {
            const double av0 = (tb[0].min+tb[0].max)/2,
                         av1 = (tb[1].min+tb[1].max)/2,
                         av2 = (tb[2].min+tb[2].max)/2;
            bb[n_el_ents*ie + 0] = x0[0] + Ai[0]*av0 + Ai[1]*av1 + Ai[2]*av2;
            bb[n_el_ents*ie + 1] = x0[1] + Ai[3]*av0 + Ai[4]*av1 + Ai[5]*av2;
            bb[n_el_ents*ie + 2] = x0[2] + Ai[6]*av0 + Ai[7]*av1 + Ai[8]*av2;
            bb[n_el_ents*ie + 3] = ab[0].min;
            bb[n_el_ents*ie + 4] = ab[1].min;
            bb[n_el_ents*ie + 5] = ab[2].min;
            bb[n_el_ents*ie + 6] = ab[0].max;
            bb[n_el_ents*ie + 7] = ab[1].max;
            bb[n_el_ents*ie + 8] = ab[2].max;
         }

         tb[0].min -= aabb_diag_len;
         tb[0].max += aabb_diag_len;
         tb[1].min -= aabb_diag_len;
         tb[1].max += aabb_diag_len;
         tb[2].min -= aabb_diag_len;
         tb[2].max += aabb_diag_len;
         {
            const double di0 = 2/((1+tol)*(tb[0].max-tb[0].min)),
                         di1 = 2/((1+tol)*(tb[1].max-tb[1].min)),
                         di2 = 2/((1+tol)*(tb[2].max-tb[2].min));
            bb[n_el_ents*ie + 9 ]=di0*A[0];
            bb[n_el_ents*ie + 10]=di0*A[1];
            bb[n_el_ents*ie + 11]=di0*A[2];
            bb[n_el_ents*ie + 12]=di1*A[3];
            bb[n_el_ents*ie + 13]=di1*A[4];
            bb[n_el_ents*ie + 14]=di1*A[5];
            bb[n_el_ents*ie + 15]=di2*A[6];
            bb[n_el_ents*ie + 16]=di2*A[7];
            bb[n_el_ents*ie + 17]=di2*A[8];
         }
      }
   }
}

#undef DO_MAX
#undef gsl_tmalloc

void FindPointsGSLIB::findptssurf_setup_3(DEV_STRUCT &devs,
                                          const double *const elx[3],
                                          const unsigned n,
                                          const uint nel,
                                          const unsigned m,
                                          const double bbox_tol,
                                          const uint local_hash_size,
                                          const uint global_hash_size,
                                          const int rD)
{
   // compute element bounding boxes.
   int n_box_ents = 18;
   devs.bb.SetSize(nel*n_box_ents);
   if (rD == 1)
   {
      obboxedge_calc_3(devs.bb,elx,n,nel,m,bbox_tol);
   }
   else if (rD == 2)
   {
      obboxsurf_calc_3(devs.bb,elx,n,nel,m,bbox_tol);
   }
   else
   {
      MFEM_ABORT("FindPointsGSLIB::findptssurf_setup_3: rD must be 1 or 2");
   }

   Vector elmin(3*nel), elmax(3*nel);
   for (uint i = 0; i < nel; i++)
   {
      elmin(i) = devs.bb(n_box_ents*i + 3); // xmin
      elmin(i + nel) = devs.bb(n_box_ents*i + 4); // ymin
      elmin(i + 2*nel) = devs.bb(n_box_ents*i + 5); // zmin
      elmax(i) = devs.bb(n_box_ents*i + 6); // xmax
      elmax(i + nel) = devs.bb(n_box_ents*i + 7); // ymax
      elmax(i + 2*nel) = devs.bb(n_box_ents*i + 8); // zmax
   }
   MPI_Barrier(MPI_COMM_WORLD);

   // build local map
   BoundingBoxTensorGridMap bbmap(elmin, elmax, local_hash_size, nel, true);
   devs.lh_min = bbmap.GetHashMin();
   devs.lh_fac = bbmap.GetHashFac();
   devs.lh_offset = bbmap.GetHashMap();
   devs.lh_nx = bbmap.GetHashN()[0];

   // build global map
   GlobalBoundingBoxTensorGridMap gbbmap(gsl_comm->c, elmin, elmax,
                                         global_hash_size, nel, true);
   devs.gh_min = gbbmap.GetHashMin();
   devs.gh_fac = gbbmap.GetHashFac();
   devs.gh_offset = gbbmap.GetHashMap();
   devs.gh_nx = gbbmap.GetHashN()[0];
}

void FindPointsGSLIB::findptsedge_setup_2(DEV_STRUCT &devs,
                                          const double *const elx[2],
                                          const unsigned n,
                                          const uint nel,
                                          const unsigned m,
                                          const double bbox_tol,
                                          const uint local_hash_size,
                                          const uint global_hash_size)
{
   // compute element bounding boxes.
   int n_box_ents = 10;
   devs.bb.SetSize(nel*n_box_ents);
   obboxedge_calc_2(devs.bb,elx,n,nel,m,bbox_tol);
   Vector elmin(2*nel), elmax(2*nel);
   for (uint i = 0; i < nel; i++)
   {
      elmin(i) = devs.bb(n_box_ents*i + 2); // xmin
      elmin(i + nel) = devs.bb(n_box_ents*i + 3); // ymin
      elmax(i) = devs.bb(n_box_ents*i + 4); // xmax
      elmax(i + nel) = devs.bb(n_box_ents*i + 5); // ymax
   }

   // build local map
   BoundingBoxTensorGridMap bbmap(elmin, elmax, local_hash_size, nel, true);
   devs.lh_min = bbmap.GetHashMin();
   devs.lh_fac = bbmap.GetHashFac();
   devs.lh_offset = bbmap.GetHashMap();
   devs.lh_nx = bbmap.GetHashN()[0];

   // build global map
   GlobalBoundingBoxTensorGridMap gbbmap(gsl_comm->c, elmin, elmax,
                                         global_hash_size, nel, true);
   devs.gh_min = gbbmap.GetHashMin();
   devs.gh_fac = gbbmap.GetHashFac();
   devs.gh_offset = gbbmap.GetHashMap();
   devs.gh_nx = gbbmap.GetHashN()[0];
}

void FindPointsGSLIB::SetupSurf(Mesh &m, const double bb_t,
                                const double newt_tol,
                                const int npt_max)
{
   // EnsureNodes call could be useful if the mesh is 1st order and has no gridfunction defined
   MFEM_VERIFY(m.GetNodes() != NULL, "Mesh nodes are required.");

   // call FreeData if FindPointsGSLIB::Setup has been called already
   if (setupflag) { FreeData(); }

   mesh     = &m;
   dim      = mesh->Dimension();       // This is reference dimension
   spacedim = mesh->SpaceDimension();  // This is physical dimension

   bool tensor_product_only = mesh->GetNE() == 0 ||
         (mesh->GetNumGeometries(dim) == 1 &&
         (mesh->GetElementType(0)== Element::SEGMENT ||
          mesh->GetElementType(0) == Element::QUADRILATERAL));
   MFEM_VERIFY(tensor_product_only,
                "FindPoints only supports tensor-product elements for "
                "surface meshes.");
   MFEM_VERIFY(dim < spacedim,
                "FindPointsGSLIB::SetupSurf only supports surface meshes.");

   const int meshOrder = m.GetNodes()->FESpace()->GetMaxElementOrder();
   unsigned dof1D      = meshOrder + 1;

   SetupSplitMeshes();
   if (dim == 1)
   {
      if (ir_split[0]) { delete ir_split[0]; ir_split[0] = NULL; }
      ir_split[0] = new IntegrationRule(pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[0], ir_split[0], meshOrder);
   }
   else if (dim == 2)
   {
      if (ir_split[0]) { delete ir_split[0]; ir_split[0] = NULL; }
      ir_split[0] = new IntegrationRule(3*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[0], ir_split[0], meshOrder);

      if (ir_split[1]) { delete ir_split[1]; ir_split[1] = NULL; }
      ir_split[1] = new IntegrationRule(pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[1], ir_split[1], meshOrder);
   }
   else
   {
      MFEM_ABORT("3D surface meshes not supported");
   }

   GetNodalValues(mesh->GetNodes(), gsl_mesh);

   mesh_points_cnt     = gsl_mesh.Size()/spacedim;
   DEV.local_hash_size = mesh_points_cnt;
   DEV.dof1d           = (int)dof1D;
   DEV.tol             = newt_tol;

   unsigned nr = dof1D;
   unsigned mr = 2*dof1D;
   if (spacedim==2)
   {
      double * const elx[2] =
      {
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(0),
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(mesh_points_cnt)
      };
      findptsedge_setup_2(DEV,
                           elx,
                           nr,
                           NE_split_total,
                           mr,
                           bb_t,
                           mesh_points_cnt,
                           mesh_points_cnt);
   }
   else if (spacedim==3)
   {
         double * const elx[3] =
         {
            mesh_points_cnt == 0 ? nullptr : &gsl_mesh(0),
            mesh_points_cnt == 0 ? nullptr : &gsl_mesh(mesh_points_cnt),
            mesh_points_cnt == 0 ? nullptr : &gsl_mesh(2*mesh_points_cnt)
         };
         findptssurf_setup_3(DEV,
                           elx,
                           nr,
                           NE_split_total,
                           mr,
                           bb_t,
                           mesh_points_cnt,
                           mesh_points_cnt, dim);
   }

   // Compute avg element size in the mesh to set surface distance tolerance.
   DEV.surf_dist_tol = 0.0;
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      DEV.surf_dist_tol += mesh->GetElementVolume(e);
   }
   int nelem = NE_split_total;
#ifdef MFEM_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &DEV.surf_dist_tol, 1, MPI_DOUBLE, MPI_SUM, gsl_comm->c);
   MPI_Allreduce(MPI_IN_PLACE, &nelem, 1, MPI_INT, MPI_SUM, gsl_comm->c);
#endif
   DEV.surf_dist_tol /= nelem;
   DEV.surf_dist_tol *= 1e-10;

   setupflag = true;
}

void FindPointsGSLIB::FindPoints(const Vector &point_pos,
                                 int point_pos_ordering)
{
   MFEM_VERIFY(setupflag, "Use FindPointsGSLIB::Setup before finding points.");
   if (dim != spacedim)
   {
      FindPointsSurf(point_pos, point_pos_ordering);
      return;
   }
   bool dev_mode = (point_pos.UseDevice() && Device::IsEnabled());
   points_cnt = point_pos.Size() / dim;
   gsl_code.SetSize(points_cnt);
   gsl_proc.SetSize(points_cnt);
   gsl_elem.SetSize(points_cnt);
   gsl_ref.SetSize(points_cnt * dim);
   gsl_dist.SetSize(points_cnt);

   bool tensor_product_only = mesh->GetNE() == 0 ||
                              (mesh->GetNumGeometries(dim) == 1 &&
                               (mesh->GetElementType(0)==Element::QUADRILATERAL ||
                                mesh->GetElementType(0) == Element::HEXAHEDRON));
#ifdef MFEM_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &tensor_product_only, 1, MFEM_MPI_CXX_BOOL,
                 MPI_LAND, gsl_comm->c);
#endif

   // std::cout << dev_mode << " " << tensor_product_only << std::endl;
   if (dev_mode && tensor_product_only)
   {
#if GSLIB_RELEASE_VERSION == 10007
      if (!gpu_to_cpu_fallback)
      {
         MFEM_ABORT("Either update to gslib v1.0.9 for GPU support "
                    "or use SetGPUtoCPUFallback to use host-functions. See "
                    "INSTALL for instructions to update GSLIB.");
      }
#else
      FindPointsOnDevice(point_pos, point_pos_ordering);
      return;
#endif
   }

   auto pp = point_pos.HostRead();
   auto xvFill = [&](const double *xv_base[], unsigned xv_stride[])
   {
      for (int d = 0; d < dim; d++)
      {
         if (point_pos_ordering == Ordering::byNODES)
         {
            xv_base[d] = pp + d*points_cnt;
            xv_stride[d] = sizeof(double);
         }
         else
         {
            xv_base[d] = pp + d;
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
}

void FindPointsGSLIB::FindPointsSurf(const Vector &point_pos,
                                      int point_pos_ordering)
{
   MFEM_VERIFY(setupflag, "Use FindPointsGSLIB::Setup before finding points.");
   MFEM_VERIFY(dim < spacedim, "FindPointsSurf is only for surface meshes.");
   bool tensor_product_only = mesh->GetNE() == 0 ||
                           (mesh->GetNumGeometries(dim) == 1 &&
                              (mesh->GetElementType(0)==Element::SEGMENT ||
                              mesh->GetElementType(0) == Element::QUADRILATERAL));
   MFEM_VERIFY(tensor_product_only,
                "FindPointsSurf currently only supports tensor-product meshes.");

   points_cnt = point_pos.Size()/spacedim;

   gsl_code.SetSize(points_cnt);
   gsl_proc.SetSize(points_cnt);
   gsl_elem.SetSize(points_cnt);
   gsl_ref.SetSize(points_cnt*dim);  // stores best guess ref pos. for each point
   gsl_dist.SetSize(points_cnt);
   gsl_newton.SetSize(points_cnt);

   FindPointsSurfOnDevice(point_pos, point_pos_ordering);

   // Set the element number and reference position to 0 for points not found
   for (int i=0; i<points_cnt; i++)
   {
      if ( gsl_code[i]==2 || (gsl_code[i]==1 && gsl_dist(i)>bdr_tol) )
      {
         gsl_elem[i] = 0;
         for (int d=0; d<dim; d++)
         {
            gsl_ref(i*dim + d) = -1.;
            gsl_mfem_ref(i*dim + d) = -1.;
         }
         gsl_code[i] = 2;
      }
      gsl_mfem_elem[i] = gsl_elem[i];
   }

   // Map element number for simplices, and ref_pos from [-1,1] to [0,1] for
   // both simplices and quads. Also sets code to 1 for points found on element
   // faces/edges.
   // MapRefPosAndElemIndicesSurf();
}

slong lfloor(double x) { return floor(x); }

// Local hash mesh index in 1D for a given point
ulong hash_index_aux(double low, double fac, ulong n, double x)
{
   const slong i = lfloor((x - low) * fac);
   return i < 0 ? 0 : (n - 1 < (ulong)i ? n - 1 : (ulong)i);
}

// Hash mesh index in 3D for a given point
ulong hash_index_nd(const Vector &hash_min, const Vector &hash_fac,
                   const int n,  const Vector &x)
{
   const int dim = x.Size();
   if (dim == 2)
   {
   return (hash_index_aux(hash_min(1), hash_fac(1), n, x[1])) * n
          + hash_index_aux(hash_min(0), hash_fac(0), n, x[0]);
   }
   else if (dim == 3)
   {
      return (hash_index_aux(hash_min(2), hash_fac(2), n, x[2]) * n +
           hash_index_aux(hash_min(1), hash_fac(1), n, x[1])) * n +
           hash_index_aux(hash_min(0), hash_fac(0), n, x[0]);
   }
   else
   {
      MFEM_ABORT("hash_index_nd only supports 2D and 3D cases.");
   }
}

void FindPointsGSLIB::FindPointsSurfOnDevice(const Vector &point_pos,
                                              int point_pos_ordering)
{
   if (!DEV.setup_device)
   {
      SetupSurfDevice();
   }

   gsl_mfem_ref.SetSize(points_cnt * dim);
   gsl_mfem_elem.SetSize(points_cnt);
   gsl_mfem_ref = 0.0;
   gsl_mfem_elem = 0;

   gsl_ref.UseDevice(true);
   gsl_dist.UseDevice(true);


   if (spacedim==2)
   {
      FindPointsEdgeLocal2(point_pos,
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
      if (dim == 1)
      {
         FindPointsEdgeLocal3(point_pos,
                               point_pos_ordering,
                               gsl_code,
                               gsl_elem,
                               gsl_ref,
                               gsl_dist,
                               gsl_newton,
                               points_cnt);
      }
      else if (dim == 2)
      {
         FindPointsSurfLocal3(point_pos,
                               point_pos_ordering,
                               gsl_code,
                               gsl_elem,
                               gsl_ref,
                               gsl_dist,
                               gsl_newton,
                               points_cnt);

      }
   }
   gsl_ref.HostReadWrite();
   gsl_dist.HostReadWrite();
   gsl_code.HostReadWrite();
   gsl_elem.HostReadWrite();
   point_pos.HostRead();

   gsl_proc.HostWrite();
   gsl_newton.HostReadWrite();

   // tolerance for point to be marked as on element edge/face
   const int id = gsl_comm->id,
             np = gsl_comm->np;
   for (int i=0; i<points_cnt; i++)
   {
      gsl_proc[i] = id;
   }
      // Tolerance for point to be marked as on element edge/face based on the
   // obtained reference-space coordinates.
   double rbtol = 1e-12; // must match MapRefPosAndElemIndices for consistency

   for (int index = 0; index < points_cnt; index++)
   {
      gsl_mfem_elem[index] = gsl_elem[index];
      for (int d = 0; d < dim; d++)
      {
         gsl_mfem_ref(index * dim + d) = 0.5 * (gsl_ref(index * dim + d) + 1.0);
      }
      // Note: we check if the point is on element border and mark it as
      // such. We do not mark points as CODE_INTERNAL because the found
      // solution could be interior to the element even when the point
      // is not on the edge/surface. This case is handled in the kernels.
      if (dim == 1)
      {
         real_t ipx = gsl_mfem_ref(index);
         if (ipx < rbtol || ipx > 1.0 - rbtol)
         {
            gsl_code[index] = CODE_BORDER;
         }
      }
      else if (dim == 2)
      {
         real_t ipx = gsl_mfem_ref(index * dim + 0);
         real_t ipy = gsl_mfem_ref(index * dim + 1);
         if (ipx < rbtol || ipx > 1.0 - rbtol ||
               ipy < rbtol || ipy > 1.0 - rbtol)
         {
            gsl_code[index] = CODE_BORDER;
         }
      }
   }
   if (np == 1) { return; }

#ifdef MFEM_USE_MPI
   MPI_Barrier(gsl_comm->c);
#endif


   /* send unfound and border points to global hash cells */
   struct gslib::array hash_pt, src_pt, out_pt;

   struct srcPt_t
   {
      double x[3];
      unsigned int index, proc;
   };

   struct outPt_t
   {
      double r[2], dist2;
      unsigned int index, code, el, proc;
      int newton;
   };

   {
      int index;
      struct srcPt_t *pt;

      array_init(struct srcPt_t, &hash_pt, points_cnt);
      pt = (struct srcPt_t *)hash_pt.ptr;

      Vector x(spacedim);
      for (index=0; index<points_cnt; ++index)
      {
         if (gsl_code[index] != CODE_INTERNAL)
         {
            for (int d=0; d<spacedim; ++d)
            {
               int idx = point_pos_ordering == 0 ?
                        index + d*points_cnt :
                        index*spacedim + d;
               x[d] = point_pos(idx);
            }
            const auto hi = hash_index_nd(DEV.gh_min, DEV.gh_fac, DEV.gh_nx, x);
            for (int d=0; d<spacedim; ++d)
            {
               pt->x[d] = x[d];
            }
            pt->index = index;
            pt->proc = hi % np;
            ++pt;
         }
      }
      hash_pt.n = pt - (struct srcPt_t *)hash_pt.ptr;
      sarray_transfer(struct srcPt_t, &hash_pt, proc, 1, DEV.cr);
   }
#ifdef MFEM_USE_MPI
   MPI_Barrier(gsl_comm->c);
#endif

   /* look up points in hash cells, route to possible procs */
   {
      const unsigned int *const hash_offset = DEV.gh_offset.GetData();
      int count = 0;
      unsigned int *proc, *proc_p;
      const struct srcPt_t *p = (struct srcPt_t *)hash_pt.ptr,
                            *const pe = p + hash_pt.n;
      struct srcPt_t *q;
      Vector x(spacedim);

      for (; p!=pe; ++p)
      {
         for (int d = 0; d < spacedim; d++) { x[d] = p->x[d]; }
         const int hi = hash_index_nd(DEV.gh_min, DEV.gh_fac, DEV.gh_nx, x)/np;
         const int i = hash_offset[hi], ie = hash_offset[hi + 1];
         count += ie - i;
      }

      Array<unsigned int> proc_array(count);
      proc = proc_array.GetData();
      proc_p = proc;
      array_init(struct srcPt_t, &src_pt, count);
      q = (struct srcPt_t *)src_pt.ptr;
      p = (struct srcPt_t *)hash_pt.ptr;
      for (; p!=pe; ++p)
      {
         for (int d = 0; d < spacedim; d++) { x[d] = p->x[d]; }
         const int hi = hash_index_nd(DEV.gh_min, DEV.gh_fac, DEV.gh_nx, x)/np;
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

      sarray_transfer_ext(struct srcPt_t, &src_pt, proc, sizeof(uint), DEV.cr);
   }
#ifdef MFEM_USE_MPI
   MPI_Barrier(gsl_comm->c);
#endif

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
      Vector gsl_ref_l(n*dim), gsl_dist_l(n);
      gsl_ref_l.UseDevice(true);
      gsl_dist_l.UseDevice(true);

      Vector point_pos_l(n*spacedim);
      point_pos_l.UseDevice(true);
      auto pointl = point_pos_l.HostWrite();

      Array<unsigned int> gsl_code_l(n), gsl_elem_l(n);

      Array<int> gsl_newton_l(n);

      for (int point=0; point<n; ++point)
      {
         for (int d=0; d<spacedim; d++)
         {
            int idx = point_pos_ordering==0 ? point + d*n :
                      point*spacedim + d;
            pointl[idx] = spt[point].x[d];
         }
      }

      if (spacedim==2)
      {
         FindPointsEdgeLocal2(point_pos_l,
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
         if (dim == 1)
         {
            FindPointsEdgeLocal3(point_pos_l,
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
            FindPointsSurfLocal3(point_pos_l,
                                  point_pos_ordering,
                                  gsl_code_l,
                                  gsl_elem_l,
                                  gsl_ref_l,
                                  gsl_dist_l,
                                  gsl_newton_l,
                                  n);
         }
      }

      gsl_ref_l.HostRead();
      gsl_dist_l.HostRead();
      gsl_code_l.HostRead();
      gsl_elem_l.HostRead();
      gsl_newton_l.HostRead();

      // unpack arrays into opt
      for (int point=0; point<n; ++point)
      {
         opt[point].code = AsConst(gsl_code_l)[point];
         if (opt[point].code == CODE_NOT_FOUND)
         {
            continue;
         }
         opt[point].el   = AsConst(gsl_elem_l)[point];
         opt[point].dist2 = AsConst(gsl_dist_l)[point];
         for (int d = 0; d < dim; ++d)
         {
            opt[point].r[d] = AsConst(gsl_ref_l)[dim * point + d];
         }
         // Note: we check if the point is on element border and mark it as
         // such. We do not mark points as CODE_INTERNAL because the found
         // solution could be interior to the element even when the point
         // is not on the edge/surface. This case is handled in the kernels.
         if (dim == 1)
         {
            real_t ipx = 0.5*opt[point].r[0]+0.5;
            if (ipx < rbtol || ipx > 1.0 - rbtol)
            {
               opt[point].code = CODE_BORDER;
            }
         }
         else if (dim == 2)
         {
            real_t ipx = 0.5*opt[point].r[0]+0.5;
            real_t ipy = 0.5*opt[point].r[1]+0.5;
            if (ipx < rbtol || ipx > 1.0 - rbtol ||
                ipy < rbtol || ipy > 1.0 - rbtol)
            {
               opt[point].code = CODE_BORDER;
            }
         }
         opt[point].newton = gsl_newton_l[point];
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
#ifdef MFEM_USE_MPI
   MPI_Barrier(gsl_comm->c);
#endif

   // /* merge remote results with user data */
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
            for (int d=0; d<dim; ++d)
            {
               gsl_ref(dim*index + d) = opt->r[d];
               gsl_mfem_ref(dim*index + d) = 0.5*(opt->r[d] + 1.);
            }
            gsl_dist(index) = opt->dist2;
            gsl_proc[index] = opt->proc;
            gsl_elem[index] = opt->el;
            gsl_code[index] = opt->code;
            gsl_mfem_elem[index] = opt->el;
         }
      }
      array_free(&out_pt);
   }

   MPI_Barrier(gsl_comm->c);
}

void lagrange_eval_second_derivative(double *p0, double x, int i,
                                      const double *z,
                                      const double *lagrangeCoeff,
                                      int pN)
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

void FindPointsGSLIB::SetupSurfDevice()
{
   gsl_mesh.UseDevice(true);

   DEV.bb.UseDevice(true);
   DEV.lh_min.UseDevice(true);
   DEV.lh_fac.UseDevice(true);
   DEV.gh_min.UseDevice(true);
   DEV.gh_fac.UseDevice(true);
   DEV.cr = cr;

   Vector gll1dtemp(DEV.dof1d),
          lagcoefftemp(DEV.dof1d),
          wtendtemp(6*DEV.dof1d);
   gslib::lobatto_nodes(gll1dtemp.GetData(),
                        DEV.dof1d); // Get gll points [-1,1] for the given dof1d
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

   DEV.wtend.UseDevice(true);
   DEV.wtend.SetSize(6*DEV.dof1d);
   DEV.wtend.HostWrite();
   DEV.wtend = wtendtemp.GetData();

   DEV.gll1d.UseDevice(true);
   DEV.gll1d.SetSize(DEV.dof1d);
   DEV.gll1d.HostWrite();
   DEV.gll1d = gll1dtemp.GetData();

   DEV.lagcoeff.UseDevice(true);
   DEV.lagcoeff.SetSize(DEV.dof1d);
   DEV.lagcoeff.HostWrite();
   DEV.lagcoeff = lagcoefftemp.GetData();

   MFEM_DEVICE_SYNC;
}

#if GSLIB_RELEASE_VERSION >= 10009
// Local hash mesh index in 3D for a given point
ulong hash_index_3(const gslib::hash_data_3 *p, const double x[3])
{
   const ulong n = p->hash_n;
   return (hash_index_aux(p->bnd[2].min, p->fac[2], n, x[2]) * n +
           hash_index_aux(p->bnd[1].min, p->fac[1], n, x[1])) *
          n +
          hash_index_aux(p->bnd[0].min, p->fac[0], n, x[0]);
}

// Local hash mesh index in 2D for a given point
ulong hash_index_2(const gslib::hash_data_2 *p, const double x[2])
{
   const ulong n = p->hash_n;
   return (hash_index_aux(p->bnd[1].min, p->fac[1], n, x[1])) * n
          + hash_index_aux(p->bnd[0].min, p->fac[0], n, x[0]);
}

void FindPointsGSLIB::SetupDevice()
{
   auto *findptsData3 = (gslib::findpts_data_3 *)this->fdataD;
   auto *findptsData2 = (gslib::findpts_data_2 *)this->fdataD;

   DEV.newt_tol = dim == 2 ? findptsData2->local.tol : findptsData3->local.tol;
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

   int n_box_ents = 3*dim + dim*dim;
   DEV.bb.UseDevice(true); DEV.bb.SetSize(n_box_ents*NE_split_total);
   auto p_bb = DEV.bb.HostWrite();
   Vector elmin(dim*NE_split_total), elmax(dim*NE_split_total);

   const int dim2 = dim*dim;
   if (dim == 3)
   {
      for (int e = 0; e < NE_split_total; e++)
      {
         auto box = findptsData3->local.obb[e];
         for (int d = 0; d < dim; d++)
         {
            p_bb[n_box_ents*e + d] = box.c0[d];
            p_bb[n_box_ents*e + dim + d] = box.x[d].min;
            p_bb[n_box_ents*e + 2*dim + d] = box.x[d].max;
            elmin(d*NE_split_total + e) = box.x[d].min;
            elmax(d*NE_split_total + e) = box.x[d].max;
         }
         for (int d = 0; d < dim2; ++d)
         {
            p_bb[n_box_ents*e + 3*dim + d] = box.A[d];
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
            p_bb[n_box_ents*e + d] = box.c0[d];
            p_bb[n_box_ents*e + dim + d] = box.x[d].min;
            p_bb[n_box_ents*e + 2*dim + d] = box.x[d].max;
            elmin(d*NE_split_total + e) = box.x[d].min;
            elmax(d*NE_split_total + e) = box.x[d].max;
         }
         for (int d = 0; d < dim2; ++d)
         {
            p_bb[n_box_ents*e + 3*dim + d] = box.A[d];
         }
      }
   }

   DEV.lh_min.UseDevice(true); DEV.lh_min.SetSize(dim);
   DEV.lh_fac.UseDevice(true); DEV.lh_fac.SetSize(dim);
   if (dim == 2)
   {
      auto hash = findptsData2->local.hd;
      auto p_loc_hash_min = DEV.lh_min.HostWrite();
      auto p_loc_hash_fac = DEV.lh_fac.HostWrite();
      for (int d = 0; d < dim; d++)
      {
         p_loc_hash_min[d] = hash.bnd[d].min;
         p_loc_hash_fac[d] = hash.fac[d];
      }
      DEV.lh_nx = hash.hash_n;
   }
   else
   {
      auto hash = findptsData3->local.hd;
      auto p_loc_hash_min = DEV.lh_min.HostWrite();
      auto p_loc_hash_fac = DEV.lh_fac.HostWrite();
      for (int d = 0; d < dim; d++)
      {
         p_loc_hash_min[d] = hash.bnd[d].min;
         p_loc_hash_fac[d] = hash.fac[d];
      }
      DEV.lh_nx = hash.hash_n;
   }

   int h_o_size = dim == 2 ?
                  findptsData2->local.hd.offset[(int)std::pow(DEV.lh_nx, dim)] :
                  findptsData3->local.hd.offset[(int)std::pow(DEV.lh_nx, dim)];

   DEV.lh_offset.SetSize(h_o_size);
   auto p_ou_offset = DEV.lh_offset.HostWrite();
   for (int i = 0; i < h_o_size; i++)
   {
      p_ou_offset[i] = dim == 2 ? findptsData2->local.hd.offset[i] :
                       findptsData3->local.hd.offset[i];
   }
   DEV.lh_offset.HostReadWrite();

   BoundingBoxTensorGridMap lh(elmin, elmax, DEV.lh_nx, NE_split_total);
   Array<int> hash_map = lh.GetHashMap();
   Vector hash_fac = lh.GetHashFac();
   Vector hash_min = lh.GetHashMin();
   Array<int> hash_n = lh.GetHashN();
   int nents = hash_n[0];
   for (int d = 1; d < dim; d++)
   {
      nents *= hash_n[d];
   }
   int maxdiff = 0;
   for (int i = 0; i < nents; i++)
   {
      int diff = hash_map[i] - DEV.lh_offset[i];
      maxdiff = std::max(maxdiff, std::abs(diff));
   }
   std::cout << "Max diff in local hash offset: " << maxdiff << std::endl;
   MPI_Barrier(gsl_comm->c);

   // test global hash
   int h_n = 0,
       h_ng = 0;

   if (dim == 2)
   {
      h_n = findptsData2->hash.hash_n;
      h_ng = h_n*h_n;
   }
   else
   {
      h_n = findptsData3->hash.hash_n;
      h_ng = h_n*h_n*h_n;
   }
   unsigned int ncell = (h_ng-1)/(gsl_comm->np)+1;
   int gh_size = dim == 2 ?
                 findptsData2->hash.offset[ncell] :
                 findptsData3->hash.offset[ncell];
   Array<int> gslib_gh(gh_size);
   for (int i = 0; i < gh_size; i++)
   {
      gslib_gh[i] = dim == 2 ? findptsData2->hash.offset[i] :
                    findptsData3->hash.offset[i];
   }

   GlobalBoundingBoxTensorGridMap gh(gsl_comm->c, elmin, elmax, h_n,
                                     NE_split_total, false);
   Array<int> global_hash_map = gh.GetHashMap();
   int gh_maxdiff = 0;
   for (int i = 0; i < gh_size; i++)
   {
      int diff = global_hash_map[i] - gslib_gh[i];
      gh_maxdiff = std::max(gh_maxdiff, std::abs(diff));
   }
   std::cout << "Max diff in global hash offset: " << gsl_comm->id << " " <<
             gh_maxdiff << std::endl;
   MPI_Barrier(gsl_comm->c);
   MFEM_ABORT("k10-got-here");

   DEV.wtend.UseDevice(true);
   DEV.wtend.SetSize(6*DEV.dof1d);
   DEV.wtend.HostWrite();
   DEV.wtend = dim == 2 ? findptsData2->local.fed.wtend[0] :
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

   DEV.setup_device = true;
}

void FindPointsGSLIB::FindPointsOnDevice(const Vector &point_pos,
                                         int point_pos_ordering)
{
   if (!DEV.setup_device)
   {
      SetupDevice();
   }
   DEV.find_device = true;

   const int id = gsl_comm->id, np = gsl_comm->np;

   gsl_mfem_ref.SetSize(points_cnt * dim);
   gsl_mfem_elem.SetSize(points_cnt);
   gsl_ref.UseDevice(true);
   gsl_dist.UseDevice(true);
   // Initialize arrays for all points (gsl_code is set to not found on device)
   gsl_ref = -1.0;
   gsl_mfem_ref = 0.0;
   gsl_elem = 0;
   gsl_mfem_elem = 0;
   gsl_proc = id;

   if (dim == 2)
   {
      FindPointsLocal2(point_pos, point_pos_ordering, gsl_code, gsl_elem, gsl_ref,
                       gsl_dist, points_cnt);
   }
   else
   {
      FindPointsLocal3(point_pos, point_pos_ordering, gsl_code, gsl_elem, gsl_ref,
                       gsl_dist, points_cnt);
   }

   // Sync from device to host
   gsl_ref.HostReadWrite();
   gsl_dist.HostReadWrite();
   gsl_code.HostReadWrite();
   gsl_elem.HostReadWrite();
   point_pos.HostRead();

   // Tolerance for point to be marked as on element edge/face based on the
   // obtained reference-space coordinates.
   double rbtol = 1e-12; // must match MapRefPosAndElemIndices for consistency

   if (np == 1)
   {
      // Set gsl_mfem_elem using gsl_elem, gsl_mfem_ref using gsl_ref,
      // and gsl_code using element type, gsl_mfem_ref, and gsl_dist.
      for (int index = 0; index < points_cnt; index++)
      {
         if (gsl_code[index] == CODE_NOT_FOUND)
         {
            continue;
         }
         gsl_mfem_elem[index] = gsl_elem[index];
         for (int d = 0; d < dim; d++)
         {
            gsl_mfem_ref(index * dim + d) = 0.5 * (gsl_ref(index * dim + d) + 1.0);
         }
         IntegrationPoint ip;
         if (dim == 2)
         {
            ip.Set2(gsl_mfem_ref.GetData() + index * dim);
         }
         else if (dim == 3)
         {
            ip.Set3(gsl_mfem_ref.GetData() + index * dim);
         }
         // Note: This works for meshes with tensor product element only as
         // otherwise the element index might be invalid.
         const int elem = gsl_elem[index];
         const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(elem);
         const Geometry::Type gt = fe->GetGeomType(); // assumes quad/hex
         int setcode =
            Geometry::CheckPoint(gt, ip, -rbtol) ? CODE_INTERNAL : CODE_BORDER;
         gsl_code[index] = setcode == CODE_BORDER && gsl_dist(index) > bdr_tol
                           ? CODE_NOT_FOUND
                           : setcode;
      }
      return;
   }

#ifdef MFEM_USE_MPI
   MPI_Barrier(gsl_comm->c);
#endif
   /* send unfound and border points to global hash cells */
   struct gslib::array hash_pt, src_pt, out_pt;

   struct srcPt_t
   {
      double x[3];
      unsigned int index, proc;
   };

   struct outPt_t
   {
      double r[3], dist2;
      unsigned int index, code, el, proc;
   };

   {
      int index;
      struct srcPt_t *pt;

      array_init(struct srcPt_t, &hash_pt, points_cnt);
      pt = (struct srcPt_t *)hash_pt.ptr;

      auto x = new double[dim];
      for (index = 0; index < points_cnt; ++index)
      {
         if (gsl_code[index] != CODE_INTERNAL)
         {
            for (int d = 0; d < dim; ++d)
            {
               int idx = point_pos_ordering == 0 ?
                         index + d*points_cnt :
                         index*dim + d;
               x[d] = point_pos(idx);
            }
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
      }
      delete[] x;
      hash_pt.n = pt - (struct srcPt_t *)hash_pt.ptr;
      sarray_transfer(struct srcPt_t, &hash_pt, proc, 1, DEV.cr);
   }
#ifdef MFEM_USE_MPI
   MPI_Barrier(gsl_comm->c);
#endif

   /* look up points in hash cells, route to possible procs */
   {
      const unsigned int *const hash_offset = dim == 2 ? DEV.hash2->offset :
                                              DEV.hash3->offset;
      int count = 0;
      unsigned int *proc, *proc_p;
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

      Array<unsigned int> proc_array(count);
      proc = proc_array.GetData();
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
            /* don't send back to where it just came from */
            if (pp == p->proc)
            {
               continue;
            }
            *proc_p++ = pp;
            *q++ = *p;
         }
      }

      array_free(&hash_pt);
      src_pt.n = proc_p - proc;

      sarray_transfer_ext(struct srcPt_t, &src_pt, proc, sizeof(uint), DEV.cr);
   }
#ifdef MFEM_USE_MPI
   MPI_Barrier(gsl_comm->c);
#endif

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
      gsl_ref_l.UseDevice(true); gsl_ref_l.SetSize(n*dim);
      gsl_dist_l.UseDevice(true); gsl_dist_l.SetSize(n);

      Vector point_pos_l;
      point_pos_l.UseDevice(true); point_pos_l.SetSize(n*dim);
      auto pointl = point_pos_l.HostWrite();

      Array<unsigned int> gsl_code_l(n), gsl_elem_l(n);

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
         FindPointsLocal2(point_pos_l, point_pos_ordering, gsl_code_l,
                          gsl_elem_l, gsl_ref_l, gsl_dist_l, n);
      }
      else
      {
         FindPointsLocal3(point_pos_l, point_pos_ordering, gsl_code_l,
                          gsl_elem_l, gsl_ref_l, gsl_dist_l, n);
      }

      gsl_ref_l.HostRead();
      gsl_dist_l.HostRead();
      gsl_code_l.HostRead();
      gsl_elem_l.HostRead();

      // unpack arrays into opt
      for (int point = 0; point < n; point++)
      {
         opt[point].code = AsConst(gsl_code_l)[point];
         if (opt[point].code == CODE_NOT_FOUND)
         {
            continue;
         }
         opt[point].el   = AsConst(gsl_elem_l)[point];
         opt[point].dist2 = AsConst(gsl_dist_l)[point];
         for (int d = 0; d < dim; ++d)
         {
            opt[point].r[d] = AsConst(gsl_ref_l)[dim * point + d];
         }
         // for found points set gsl_code using reference space coords.
         IntegrationPoint ip;
         if (dim == 2)
         {
            ip.Set2(0.5*opt[point].r[0]+0.5, 0.5*opt[point].r[1]+0.5);
         }
         else
         {
            ip.Set3(0.5*opt[point].r[0]+0.5, 0.5*opt[point].r[1]+0.5,
                    0.5*opt[point].r[2]+0.5);
         }
         const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(opt[point].el);
         const Geometry::Type gt = fe->GetGeomType();
         int setcode = Geometry::CheckPoint(gt, ip, -rbtol) ?
                       CODE_INTERNAL : CODE_BORDER;
         opt[point].code = setcode==CODE_BORDER && opt[point].dist2>bdr_tol ?
                           CODE_NOT_FOUND : setcode;
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
#ifdef MFEM_USE_MPI
   MPI_Barrier(gsl_comm->c);
#endif

   /* merge remote results with user data */
   // For points found on other procs, we set gsl_mfem_elem, gsl_mfem_ref,
   // and gsl_code now.
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
             opt->dist2 < gsl_dist[index])
         {
            for (int d = 0; d < dim; ++d)
            {
               gsl_ref(dim * index + d) = opt->r[d];
               gsl_mfem_ref(dim*index + d) = 0.5*(opt->r[d] + 1.);
            }
            gsl_dist[index] = opt->dist2;
            gsl_proc[index] = opt->proc;
            gsl_elem[index] = opt->el;
            gsl_mfem_elem[index]   = opt->el;
            gsl_code[index] = opt->code;
         }
      }
      array_free(&out_pt);
   }

   // For points found locally, we set gsl_mfem_elem, gsl_mfem_ref, and gsl_code.
   for (int index = 0; index < points_cnt; index++)
   {
      if (gsl_code[index] == CODE_NOT_FOUND || gsl_proc[index] != id)
      {
         continue;
      }
      gsl_mfem_elem[index] = gsl_elem[index];
      for (int d = 0; d < dim; d++)
      {
         gsl_mfem_ref(index*dim + d) = 0.5*(gsl_ref(index*dim + d)+1.0);
      }
      IntegrationPoint ip;
      if (dim == 2)
      {
         ip.Set2(gsl_mfem_ref.GetData() + index*dim);
      }
      else if (dim == 3)
      {
         ip.Set3(gsl_mfem_ref.GetData() + index*dim);
      }
      const int elem = gsl_elem[index];
      const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(elem);
      const Geometry::Type gt = fe->GetGeomType(); // assumes quad/hex
      int setcode = Geometry::CheckPoint(gt, ip, -rbtol) ?
                    CODE_INTERNAL : CODE_BORDER;
      gsl_code[index] = setcode==CODE_BORDER && gsl_dist(index)>bdr_tol ?
                        CODE_NOT_FOUND : setcode;
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

void FindPointsGSLIB::InterpolateOnDevice(const Vector &field_in_evec,
                                          Vector &field_out,
                                          const int nel,
                                          const int ncomp,
                                          const int dof1Dsol,
                                          const int ordering)
{
   field_out.UseDevice(true);
   field_out.SetSize(points_cnt*ncomp);
   field_out = default_interp_value;

   DEV.dof1d_sol =  dof1Dsol;
   DEV.gll1d_sol.UseDevice(true);  DEV.gll1d_sol.SetSize(dof1Dsol);
   DEV.lagcoeff_sol.UseDevice(true); DEV.lagcoeff_sol.SetSize(dof1Dsol);
   if (DEV.dof1d_sol != DEV.dof1d || !DEV.find_device)
   {
      gslib::lobatto_nodes(DEV.gll1d_sol.HostWrite(), dof1Dsol);
      gslib::gll_lag_setup(DEV.lagcoeff_sol.HostWrite(), dof1Dsol);
   }
   else
   {
      DEV.gll1d_sol = DEV.gll1d.HostRead();
      DEV.lagcoeff_sol = DEV.lagcoeff.HostRead();
   }

   field_out.HostReadWrite(); //Reads in default value from device

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
      const double *r = gsl_ref.GetData();

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
         InterpolateLocal2(field_in_evec,
                           gsl_elem_temp,
                           gsl_ref_temp,
                           interp_vals,
                           nlocal, ncomp,
                           nel, dof1Dsol);
      }
      else
      {
         InterpolateLocal3(field_in_evec,
                           gsl_elem_temp,
                           gsl_ref_temp,
                           interp_vals,
                           nlocal, ncomp,
                           nel, dof1Dsol);

      }
#ifdef MFEM_USE_MPI
      MPI_Barrier(gsl_comm->c);
#endif

      interp_vals.HostRead();

      // now put these in correct positions
      int interp_Offset = interp_vals.Size()/ncomp;
      for (int i = 0; i < ncomp; i++)
      {
         for (int j = 0; j < nlocal; j++)
         {
            int pt_index = index_temp[j];
            int idx = ordering == Ordering::byNODES ?
                      pt_index + i*points_cnt :
                      pt_index*ncomp + i;
            field_out(idx) = AsConst(interp_vals)(j + interp_Offset*i);
         }
      }
   }
#ifdef MFEM_USE_MPI
   MPI_Barrier(gsl_comm->c);
#endif

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
      if (dim == 2)
      {
         InterpolateLocal2(field_in_evec,
                           gsl_elem_temp,
                           gsl_ref_temp,
                           interp_vals, n, ncomp,
                           nel, dof1Dsol);
      }
      else
      {
         InterpolateLocal3(field_in_evec,
                           gsl_elem_temp,
                           gsl_ref_temp,
                           interp_vals, n, ncomp,
                           nel, dof1Dsol);
      }
#ifdef MFEM_USE_MPI
      MPI_Barrier(gsl_comm->c);
#endif
      interp_vals.HostRead();

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
            opt->out = AsConst(interp_vals)(j + Offset*i);
            spt++;
            opt++;
         }

         sarray_transfer(struct evalOutPt_t, &outpt, proc, 1, cr);

         opt = (evalOutPt_t *)outpt.ptr;
         for (int index = 0; index < outpt.n; index++)
         {
            int idx = ordering == Ordering::byNODES ?
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

void FindPointsGSLIB::InterpolateSurfBase(const Vector &field_in,
                                          Vector &field_out,
                                          const int nel,
                                          const int ncomp,
                                          const int dof1Dsol,
                                          const int gf_ordering)
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
      const real_t *r = gsl_ref.GetData();

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
         InterpolateEdgeLocal2(field_in,
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
         if (dim == 1)
         {
            InterpolateEdgeLocal3(field_in,
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
         if (dim == 1)
         {
            InterpolateEdgeLocal3(field_in,
                                   gsl_elem_temp,
                                   gsl_ref_temp,
                                   interp_vals,
                                   n,
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
                                   n,
                                   ncomp,
                                   nel,
                                   dof1Dsol);
         }
      }
      else
      {
         InterpolateEdgeLocal2(field_in,
                                gsl_elem_temp,
                                gsl_ref_temp,
                                interp_vals,
                                n,
                                ncomp,
                                nel,
                                dof1Dsol);
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

#else
void FindPointsGSLIB::SetupDevice() {};
void FindPointsGSLIB::FindPointsOnDevice(const Vector &point_pos,
                                         int point_pos_ordering) {};
void FindPointsGSLIB::InterpolateOnDevice(const Vector &field_in_evec,
                                          Vector &field_out,
                                          const int nel, const int ncomp,
                                          const int dof1dsol,
                                          const int ordering) {};
#endif

void FindPointsGSLIB::FindPoints(Mesh &m, const Vector &point_pos,
                                 int point_pos_ordering, const double bb_t,
                                 const double newt_tol, const int npt_max)
{
   if (!setupflag || (mesh != &m))
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
   if (dim == spacedim)
   {
      if (dim == 2)
      {
         findpts_free_2((gslib::findpts_data_2 *)this->fdataD);
      }
      else
      {
         findpts_free_3((gslib::findpts_data_3 *)this->fdataD);
      }
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
   DEV.setup_device = false;
   DEV.find_device  = false;
   points_cnt = -1;
}

void FindPointsGSLIB::SetupSplitMeshes()
{
   fec_map_lin = new H1_FECollection(1, dim);
   if (mesh->Dimension() == 1)
   {
      mesh_split[0] = new Mesh(Mesh::MakeCartesian1D(1, Element::SEGMENT));
   }
   else if (mesh->Dimension() == 2)
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
      else if (gt == Geometry::SEGMENT || gt == Geometry::SQUARE ||
               gt == Geometry::CUBE)
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

   const int dof_cnt = nodal_fes.GetTypicalFE()->GetDof(),
             pts_cnt = NEsplit * dof_cnt;
   Vector irlist(dim * pts_cnt);

   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement *>(nodal_fes.GetTypicalFE());
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
         if (dim >= 2)
         {
            irule->IntPoint(pt_id).y = irlist(pts_cnt + pt_id);
         }
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
         // "split" if input mesh is not a tensor basis or has mixed order
         el_to_split =
            gf_in->FESpace()->IsVariableOrder() ||
            dynamic_cast<const TensorBasisElement *>(fes->GetFE(e)) == nullptr;
      }
      else if (gt == Geometry::SEGMENT || gt == Geometry::CUBE)
      {
         ir_split_temp = ir_split[0];
         // "split" if input mesh is not a tensor basis or has mixed order
         el_to_split =
            gf_in->FESpace()->IsVariableOrder() ||
            dynamic_cast<const TensorBasisElement *>(fes->GetFE(e)) == nullptr;
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
   gsl_mfem_ref = gsl_ref.HostRead();
   gsl_mfem_elem = gsl_elem;

   gsl_mfem_ref += 1.;  // map  [-1, 1] to [0, 2] to [0, 1]
   gsl_mfem_ref *= 0.5;

   int nptorig = points_cnt,
       npt = points_cnt;

   // Tolerance for point to be marked as on element edge/face based on the
   // obtained reference-space coordinates.
   double rbtol = 1e-12;

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
         pt->code = Geometry::CheckPoint(gt, ip, -rbtol) ? 0 : 1;
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
      pt->code = Geometry::CheckPoint(gt, ip, -rbtol) ? 0 : 1;
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
            gsl_code[index] = Geometry::CheckPoint(gt, ip, -rbtol) ? 0 : 1;
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
         gsl_code[index]  = Geometry::CheckPoint(gt, ip, -rbtol) ? 0 : 1;
      }
   }
}

void FindPointsGSLIB::Interpolate(const GridFunction &field_in,
                                  Vector &field_out)
{
   MFEM_VERIFY(setupflag, "FindPointsGSLIB::Setup must be called first.");
   if (dim != spacedim)
   {
      InterpolateSurf(field_in, field_out);
      return;
   }
   const int  gf_order   = field_in.FESpace()->GetMaxElementOrder(),
              mesh_order = mesh->GetNodalFESpace()->GetMaxElementOrder();

   const FiniteElementCollection *fec_in =  field_in.FESpace()->FEColl();
   const H1_FECollection *fec_h1 = dynamic_cast<const H1_FECollection *>(fec_in);
   const L2_FECollection *fec_l2 = dynamic_cast<const L2_FECollection *>(fec_in);

   bool tensor_product_only = mesh->GetNE() == 0 ||
                              (mesh->GetNumGeometries(dim) == 1 &&
                               (mesh->GetElementType(0)==Element::QUADRILATERAL ||
                                mesh->GetElementType(0) == Element::HEXAHEDRON));
#ifdef MFEM_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &tensor_product_only, 1, MFEM_MPI_CXX_BOOL,
                 MPI_LAND, gsl_comm->c);
#endif

   if (Device::IsEnabled() && field_in.UseDevice() && fec_h1 &&
       !field_in.FESpace()->IsVariableOrder() && tensor_product_only)
   {
#if GSLIB_RELEASE_VERSION == 10007
      if (!gpu_to_cpu_fallback)
      {
         MFEM_ABORT("Either update to gslib v1.0.9 for GPU support "
                    "or use SetGPUtoCPUFallback to use host-functions. See "
                    "INSTALL for instructions to update GSLIB");
      }
#else
      MFEM_VERIFY(fec_h1->GetBasisType() == BasisType::GaussLobatto,
                  "basis not supported");
      Vector node_vals;
      const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
      const Operator *R = field_in.FESpace()->GetElementRestriction(ordering);
      node_vals.UseDevice(true);
      node_vals.SetSize(R->Height(), Device::GetMemoryType());
      R->Mult(field_in, node_vals);
      // GetNodalValues(&field_in, node_vals);

      const int ncomp  = field_in.FESpace()->GetVDim();
      const int maxOrder = field_in.FESpace()->GetMaxElementOrder();

      InterpolateOnDevice(node_vals, field_out, NE_split_total, ncomp,
                          maxOrder+1, field_in.FESpace()->GetOrdering());
      return;
#endif
   }
   field_in.HostRead();
   field_out.HostWrite();

   if (fec_h1 && gf_order == mesh_order &&
       fec_h1->GetBasisType() == BasisType::GaussLobatto &&
       field_in.FESpace()->IsVariableOrder() ==
       mesh->GetNodalFESpace()->IsVariableOrder())
   {
      InterpolateH1(field_in, field_out);
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
      FiniteElementSpace fes(mesh, &fec, ncomp,
                             field_in.FESpace()->GetOrdering());
      GridFunction field_in_h1(&fes);
      field_in_h1.UseDevice(false);

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
            int idx = field_in_h1.FESpace()->GetOrdering() == Ordering::byNODES?
                      indl2[i] + j*points_cnt:
                      indl2[i]*ncomp + j;
            field_out(idx) = field_out_l2(idx);
         }
      }
   }
}

void FindPointsGSLIB::InterpolateSurf(const GridFunction &field_in,
                                      Vector &field_out)
{
   const int  gf_order   = field_in.FESpace()->GetMaxElementOrder(),
              mesh_order = mesh->GetNodalFESpace()->GetMaxElementOrder();

   const FiniteElementCollection *fec_in =  field_in.FESpace()->FEColl();
   const H1_FECollection *fec_h1 = dynamic_cast<const H1_FECollection *>
   (fec_in);
   // const L2_FECollection *fec_l2 = dynamic_cast<const L2_FECollection *
   // (fec_in);
   MFEM_VERIFY(fec_h1,"Only h1 functions supported on device right now.");
   MFEM_VERIFY(fec_h1->GetBasisType() == BasisType::GaussLobatto,
               "basis not supported");

   bool tensor_product_only = mesh->GetNE() == 0 ||
         (mesh->GetNumGeometries(dim) == 1 &&
         (mesh->GetElementType(0)== Element::SEGMENT ||
          mesh->GetElementType(0) == Element::QUADRILATERAL));
#ifdef MFEM_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &tensor_product_only, 1, MFEM_MPI_CXX_BOOL,
                 MPI_LAND, gsl_comm->c);
#endif
   MFEM_VERIFY(tensor_product_only,
                "FindPoints only supports tensor-product elements for "
                "surface meshes.");
   MFEM_VERIFY(dim < spacedim,
                "FindPointsGSLIB::SetupSurf only supports surface meshes.");

   bool device_mode = field_in.UseDevice();

   if (!field_in.FESpace()->IsVariableOrder())
   {
      Vector node_vals;
      node_vals.UseDevice(device_mode);

      const ElementDofOrdering ord = ElementDofOrdering::LEXICOGRAPHIC;
      const Operator *R            = field_in.FESpace()->GetElementRestriction(ord);
      node_vals.SetSize(R->Height());
      R->Mult(field_in, node_vals); //orders fields (N^D x VDIM x NEL)

      const int ncomp    = field_in.FESpace()->GetVDim();
      const int maxOrder = field_in.FESpace()->GetMaxElementOrder();
      DEV.dof1d_sol      =  maxOrder+1;
      DEV.gll1d_sol.UseDevice(device_mode);
      DEV.gll1d_sol.SetSize(DEV.dof1d_sol);
      DEV.lagcoeff_sol.UseDevice(device_mode);
      DEV.lagcoeff_sol.SetSize(DEV.dof1d_sol);
      DEV.gll1d_sol.HostWrite();
      DEV.lagcoeff_sol.HostWrite();
      if (DEV.dof1d_sol != DEV.dof1d)
      {
         Vector temp(DEV.dof1d_sol);
         gslib::lobatto_nodes(temp.GetData(), DEV.dof1d_sol);
         DEV.gll1d_sol = temp.GetData();
         MFEM_DEVICE_SYNC;

         gslib::gll_lag_setup(temp.GetData(), DEV.dof1d_sol);
         DEV.lagcoeff_sol = temp.GetData();
         MFEM_DEVICE_SYNC;
      }
      else
      {
         DEV.gll1d_sol    = DEV.gll1d.HostRead();
         DEV.lagcoeff_sol = DEV.lagcoeff.HostRead();
      }
      MFEM_DEVICE_SYNC;

      field_out.SetSize(points_cnt*ncomp);
      field_out.UseDevice(device_mode);

      InterpolateSurfBase(node_vals, field_out, NE_split_total, ncomp,
                          DEV.dof1d_sol, field_in.FESpace()->GetOrdering());
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
   field_in_scalar.UseDevice(false);
   Vector node_vals;

   const int ncomp      = field_in.FESpace()->GetVDim(),
             points_fld = field_in.Size() / ncomp;
   MFEM_VERIFY(points_cnt == gsl_code.Size(),
               "FindPointsGSLIB::InterpolateH1: Inconsistent size of gsl_code");

   field_out.SetSize(points_cnt*ncomp);
   field_out = default_interp_value;
   field_out.HostReadWrite();

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
   field_out.HostReadWrite();

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

void FindPointsGSLIB::DistributePointInfoToOwningMPIRanks(
   Array<unsigned int> &recv_elem, Vector &recv_ref,
   Array<unsigned int> &recv_code)
{
   MFEM_VERIFY(points_cnt >= 0,
               "Invalid size. Please make sure to call FindPoints method "
               "before calling this function.");

   // Pack data to send via crystal router
   struct gslib::array *outpt = new gslib::array;

   struct out_pt { double rst[3]; uint index, elem, proc, code; };
   struct out_pt *pt;
   array_init(struct out_pt, outpt, points_cnt);
   outpt->n=points_cnt;
   pt = (struct out_pt *)outpt->ptr;

   for (int index = 0; index < points_cnt; index++)
   {
      pt->index = index;
      pt->elem = gsl_mfem_elem[index];
      pt->proc  = gsl_proc[index];
      pt->code = gsl_code[index];
      for (int d = 0; d < dim; ++d)
      {
         pt->rst[d]= gsl_mfem_ref(index*dim + d);
      }
      ++pt;
   }

   // Transfer data to target MPI ranks
   sarray_transfer(struct out_pt, outpt, proc, 1, cr);

   // Store received data
   const int points_recv = outpt->n;
   recv_proc.SetSize(points_recv);
   recv_elem.SetSize(points_recv);
   recv_index.SetSize(points_recv);
   recv_code.SetSize(points_recv);
   recv_ref.SetSize(points_recv*dim);

   pt = (struct out_pt *)outpt->ptr;
   for (int index = 0; index < points_recv; index++)
   {
      recv_index[index] = pt->index;
      recv_elem[index] = pt->elem;
      recv_proc[index] = pt->proc;
      recv_code[index] = pt->code;
      for (int d = 0; d < dim; ++d)
      {
         recv_ref(index*dim + d)= pt->rst[d];
      }
      ++pt;
   }

   array_free(outpt);
   delete outpt;
}

void FindPointsGSLIB::DistributeInterpolatedValues(const Vector &int_vals,
                                                   const int vdim,
                                                   const int ordering,
                                                   Vector &field_out) const
{
   const int points_recv = recv_index.Size();;
   MFEM_VERIFY(points_recv == 0 ||
               int_vals.Size() % points_recv == 0,
               "Incompatible size. Please return interpolated values"
               "corresponding to points received using"
               "SendCoordinatesToOwningProcessors.");
   field_out.SetSize(points_cnt*vdim);

   for (int v = 0; v < vdim; v++)
   {
      // Pack data to send via crystal router
      struct gslib::array *outpt = new gslib::array;
      struct out_pt { double val; uint index, proc; };
      struct out_pt *pt;
      array_init(struct out_pt, outpt, points_recv);
      outpt->n=points_recv;
      pt = (struct out_pt *)outpt->ptr;
      for (int index = 0; index < points_recv; index++)
      {
         pt->index = recv_index[index];
         pt->proc  = recv_proc[index];
         pt->val = ordering == Ordering::byNODES ?
                   int_vals(index + v*points_recv) :
                   int_vals(index*vdim + v);
         ++pt;
      }

      // Transfer data to target MPI ranks
      sarray_transfer(struct out_pt, outpt, proc, 1, cr);

      // Store received data
      MFEM_VERIFY(outpt->n == points_cnt, "Incompatible size. Number of points "
                  "received does not match the number of points originally "
                  "found using FindPoints.");

      pt = (struct out_pt *)outpt->ptr;
      for (int index = 0; index < points_cnt; index++)
      {
         int idx = ordering == Ordering::byNODES ?
                   pt->index + v*points_cnt :
                   pt->index*vdim + v;
         field_out(idx) = pt->val;
         ++pt;
      }

      array_free(outpt);
      delete outpt;
   }
}

void FindPointsGSLIB::GetAxisAlignedBoundingBoxes(Vector &aabb)
{
   MFEM_VERIFY(setupflag, "Call FindPointsGSLIB::Setup method first");
   auto *findptsData3 = (gslib::findpts_data_3 *)this->fdataD;
   auto *findptsData2 = (gslib::findpts_data_2 *)this->fdataD;
   int nve   = spacedim == 2 ? 4 : 8;
   int nel = NE_split_total;
   aabb.SetSize(spacedim*nve*nel);

   if (spacedim == 3)
   {
      for (int e = 0; e < nel; e++)
      {
         Vector minn(spacedim), maxx(spacedim);
         if (dim == spacedim)
         {
            auto box = findptsData3->local.obb[e];
            for (int d = 0; d < spacedim; d++)
            {
               minn[d] = box.x[d].min;
               maxx[d] = box.x[d].max;
            }
         }
         else
         {
            int n_el_ents = 18;
            for (int d = 0; d < spacedim; d++)
            {
               minn[d] = DEV.bb(e*n_el_ents + spacedim + d);
               maxx[d] = DEV.bb(e*n_el_ents + 2*spacedim + d);
            }
         }
         int c = 0;
         aabb(e*nve*spacedim + c++) = minn[0]; /* first vertex - x */
         aabb(e*nve*spacedim + c++) = minn[1]; /* y */
         aabb(e*nve*spacedim + c++) = minn[2]; /* z */
         aabb(e*nve*spacedim + c++) = maxx[0]; /* second vertex - x */
         aabb(e*nve*spacedim + c++) = minn[1]; /* . */
         aabb(e*nve*spacedim + c++) = minn[2]; /* . */
         aabb(e*nve*spacedim + c++) = maxx[0];
         aabb(e*nve*spacedim + c++) = maxx[1];
         aabb(e*nve*spacedim + c++) = minn[2];
         aabb(e*nve*spacedim + c++) = minn[0];
         aabb(e*nve*spacedim + c++) = maxx[1];
         aabb(e*nve*spacedim + c++) = minn[2];
         aabb(e*nve*spacedim + c++) = minn[0];
         aabb(e*nve*spacedim + c++) = minn[1];
         aabb(e*nve*spacedim + c++) = maxx[2];
         aabb(e*nve*spacedim + c++) = maxx[0];
         aabb(e*nve*spacedim + c++) = minn[1];
         aabb(e*nve*spacedim + c++) = maxx[2];
         aabb(e*nve*spacedim + c++) = maxx[0];
         aabb(e*nve*spacedim + c++) = maxx[1];
         aabb(e*nve*spacedim + c++) = maxx[2];
         aabb(e*nve*spacedim + c++) = minn[0];
         aabb(e*nve*spacedim + c++) = maxx[1];
         aabb(e*nve*spacedim + c++) = maxx[2];
      }
   }
   else // spacedim = 2
   {
      for (int e = 0; e < nel; e++)
      {
         Vector minn(spacedim), maxx(spacedim);
         if (dim == spacedim)
         {
            auto box = findptsData2->local.obb[e];
            for (int d = 0; d < spacedim; d++)
            {
               minn[d] = box.x[d].min;
               maxx[d] = box.x[d].max;
            }
         }
         else
         {
            int n_el_ents = 10;
            for (int d = 0; d < spacedim; d++)
            {
               minn[d] = DEV.bb(e*n_el_ents + spacedim + d);
               maxx[d] = DEV.bb(e*n_el_ents + 2*spacedim + d);
            }
         }
         aabb(e*nve*spacedim + 0) = minn[0]; /* first vertex - x */
         aabb(e*nve*spacedim + 1) = minn[1]; /* y */
         aabb(e*nve*spacedim + 2) = maxx[0]; /* second vertex - x */
         aabb(e*nve*spacedim + 3) = minn[1]; /* . */
         aabb(e*nve*spacedim + 4) = maxx[0]; /* . */
         aabb(e*nve*spacedim + 5) = maxx[1];
         aabb(e*nve*spacedim + 6) = minn[0];
         aabb(e*nve*spacedim + 7) = maxx[1];
      }
   }
}

Mesh* FindPointsGSLIB::GetBoundingBoxMesh(int type)
{
   MFEM_VERIFY(setupflag, "Call FindPointsGSLIB::Setup method first");
   int save_rank = 0;
   int myid = gsl_comm->id;
   Vector bbvert;
   if (type == 0)
   {
      GetAxisAlignedBoundingBoxes(bbvert);
   }
   else
   {
      DenseTensor obbA;
      Vector obbC;
      GetOrientedBoundingBoxes(obbA, obbC, bbvert);
   }
   int nve   = spacedim == 2 ? 4 : 8;
   int nel = NE_split_total;
   int ne_glob = nel;
#ifdef MFEM_USE_MPI
   MPI_Allreduce(&nel, &ne_glob, 1, MPI_INT, MPI_SUM, gsl_comm->c);
#endif


   int nverts = nve*ne_glob;
   Mesh *meshbb = NULL;
   if (gsl_comm->id == save_rank)
   {
      meshbb = new Mesh(spacedim, nverts, ne_glob, 0, spacedim);
   }

   int nsend = nel*nve*spacedim;
   MFEM_VERIFY(nsend == bbvert.Size(),
               "Inconsistent size of bounding box vertices");
   int nrecv = 0;
   MPI_Status status;
   int vidx = 0;
   int eidx = 0;
   if (myid == save_rank)
   {
      for (int p = 0; p < gsl_comm->np; p++)
      {
         if (p != save_rank)
         {
            MPI_Recv(&nrecv, 1, MPI_INT, p, 444, gsl_comm->c, &status);
            bbvert.SetSize(nrecv);
            if (nrecv)
            {
               MPI_Recv(bbvert.GetData(), nrecv, MPI_DOUBLE, p, 445, gsl_comm->c, &status);
            }
         }
         else
         {
            nrecv = nsend;
         }
         int nel_recv = nrecv/(spacedim*nve);
         for (int e = 0; e < nel_recv; e++)
         {
            for (int j = 0; j < nve; j++)
            {
               Vector ver(bbvert.GetData() + e*nve*spacedim + j*spacedim, spacedim);
               meshbb->AddVertex(ver);
            }

            if (spacedim == 2)
            {
               const int inds[4] = {vidx++, vidx++, vidx++, vidx++};
               int attr = eidx+1;
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
      if (spacedim == 2)
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
         MPI_Send(bbvert.GetData(), nsend, MPI_DOUBLE, save_rank, 445, gsl_comm->c);
      }
   }
   MPI_Barrier(gsl_comm->c);

   return meshbb;
}

void FindPointsGSLIB::GetOrientedBoundingBoxes(DenseTensor &obbA, Vector &obbC,
                                               Vector &obbV)
{
   MFEM_VERIFY(setupflag, "Call FindPointsGSLIB::Setup method first");
   auto *findptsData3 = (gslib::findpts_data_3 *)this->fdataD;
   auto *findptsData2 = (gslib::findpts_data_2 *)this->fdataD;
   int nve   = spacedim == 2 ? 4 : 8;
   int nel = NE_split_total;

   obbA.SetSize(spacedim, spacedim, nel);
   obbC.SetSize(spacedim*nel);
   obbV.SetSize(spacedim*nve*nel);
   if (spacedim == 3)
   {
      for (int e = 0; e < nel; e++)
      {
         double *Ad = obbA.GetData(e);
         if (dim == spacedim)
         {
            auto box = findptsData3->local.obb[e];
            for (int d = 0; d < spacedim; d++)
            {
               obbC(e*spacedim + d) = box.c0[d];
            }
            for (int i = 0; i < spacedim; i++)
            {
               for (int j = 0; j < spacedim; j++)
               {
                  Ad[i*spacedim + j] = box.A[i + j*spacedim]; // GSLIB uses row-major storage
               }
            }
         }
         else
         {
            int n_el_ents = 18;
            for (int d = 0; d < spacedim; d++)
            {
               obbC(e*spacedim + d) = DEV.bb(n_el_ents*e + d);
            }
            for (int i = 0; i < spacedim; i++)
            {
               for (int j = 0; j < spacedim; j++)
               {
                  Ad[i*spacedim + j] = DEV.bb[n_el_ents*e + 9 + j*spacedim+i];
               }
            }
         }

         DenseMatrix Amat = obbA(e);
         Amat.Invert();
         Vector center(obbC.GetData() + e*spacedim, spacedim);

         Vector v1(spacedim);
         Vector temp;
         v1(0) = -1.0; v1(1) = -1.0; v1(2) = -1.0;
         temp.SetDataAndSize(obbV.GetData() + e*nve*spacedim + 0, spacedim);
         Amat.Mult(v1, temp);
         temp += center;
         v1(0) = 1.0; v1(1) = -1.0; v1(2) = -1.0;
         temp.SetDataAndSize(obbV.GetData() + e*nve*spacedim + 3, spacedim);
         Amat.Mult(v1, temp);
         temp += center;
         v1(0) = 1.0; v1(1) = 1.0; v1(2) = -1.0;
         temp.SetDataAndSize(obbV.GetData() + e*nve*spacedim + 6, spacedim);
         Amat.Mult(v1, temp);
         temp += center;
         v1(0) = -1.0; v1(1) = 1.0; v1(2) = -1.0;
         temp.SetDataAndSize(obbV.GetData() + e*nve*spacedim + 9, spacedim);
         Amat.Mult(v1, temp);
         temp += center;
         v1(0) = -1.0; v1(1) = -1.0; v1(2) = 1.0;
         temp.SetDataAndSize(obbV.GetData() + e*nve*spacedim + 12, spacedim);
         Amat.Mult(v1, temp);
         temp += center;
         v1(0) = 1.0; v1(1) = -1.0; v1(2) = 1.0;
         temp.SetDataAndSize(obbV.GetData() + e*nve*spacedim + 15, spacedim);
         Amat.Mult(v1, temp);
         temp += center;
         v1(0) = 1.0; v1(1) = 1.0; v1(2) = 1.0;
         temp.SetDataAndSize(obbV.GetData() + e*nve*spacedim + 18, spacedim);
         Amat.Mult(v1, temp);
         temp += center;
         v1(0) = -1.0; v1(1) = 1.0; v1(2) = 1.0;
         temp.SetDataAndSize(obbV.GetData() + e*nve*spacedim + 21, spacedim);
         Amat.Mult(v1, temp);
         temp += center;
      }
   }
   else // spacedim = 2
   {
      for (int e = 0; e < nel; e++)
      {
         double *Ad = obbA.GetData(e);
         if (dim == spacedim)
         {
            auto box = findptsData2->local.obb[e];
            for (int d = 0; d < spacedim; d++)
            {
               obbC(e*spacedim + d) = box.c0[d];
            }
            for (int i = 0; i < spacedim; i++)
            {
               for (int j = 0; j < spacedim; j++)
               {
                  Ad[i*spacedim + j] = box.A[i + j*spacedim]; // GSLIB uses row-major storage
               }
            }
         }
         else
         {
            int n_el_ents = 10;
            for (int d = 0; d < spacedim; d++)
            {
               obbC(e*spacedim + d) = DEV.bb(n_el_ents*e + d);
            }
            for (int i = 0; i < spacedim; i++)
            {
               for (int j = 0; j < spacedim; j++)
               {
                  Ad[i*spacedim + j] = DEV.bb[n_el_ents*e + 6 + j*spacedim+i];
               }
            }
         }


         DenseMatrix Amat = obbA(e);
         Amat.Invert();
         Vector center(obbC.GetData() + e*spacedim, spacedim);

         Vector v1(spacedim);
         Vector temp;
         v1(0) = -1.0; v1(1) = -1.0;
         temp.SetDataAndSize(obbV.GetData() + e*nve*spacedim + 0, spacedim);
         Amat.Mult(v1, temp);
         temp += center;
         v1(0) = 1.0; v1(1) = -1.0;
         temp.SetDataAndSize(obbV.GetData() + e*nve*spacedim + 2, spacedim);
         Amat.Mult(v1, temp);
         temp += center;
         v1(0) = 1.0; v1(1) = 1.0;
         temp.SetDataAndSize(obbV.GetData() + e*nve*spacedim + 4, spacedim);
         Amat.Mult(v1, temp);
         temp += center;
         v1(0) = -1.0; v1(1) = 1.0;
         temp.SetDataAndSize(obbV.GetData() + e*nve*spacedim + 6, spacedim);
         Amat.Mult(v1, temp);
         temp += center;
      }
   }
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

   mesh = &m;
   dim  = mesh->Dimension();
   const FiniteElement *fe = mesh->GetNodalFESpace()->GetTypicalFE();
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
   MFEM_VERIFY(overset, "Please setup FindPoints for overlapping grids.");
   points_cnt = point_pos.Size() / dim;
   unsigned int match = 0; // Don't find points in the mesh if point_id=mesh_id

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
      auto *findptsData = (gslib::findpts_data_2 *)this->fdataD;
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
                  points_cnt, findptsData);
   }
   else  // dim == 3
   {
      auto *findptsData = (gslib::findpts_data_3 *)this->fdataD;
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
                  points_cnt, findptsData);
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

GSOPGSLIB::GSOPGSLIB(Array<long long> &ids)
{
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
   crystal_init(cr, gsl_comm);
   UpdateIdentifiers(ids);
}

#ifdef MFEM_USE_MPI
GSOPGSLIB::GSOPGSLIB(MPI_Comm comm_, Array<long long> &ids)
   : cr(NULL), gsl_comm(NULL)
{
   gsl_comm = new gslib::comm;
   cr      = new gslib::crystal;
   comm_init(gsl_comm, comm_);
   crystal_init(cr, gsl_comm);
   UpdateIdentifiers(ids);
}
#endif

GSOPGSLIB::~GSOPGSLIB()
{
   crystal_free(cr);
   gslib_gs_free(gsl_data);
   comm_free(gsl_comm);
   delete gsl_comm;
   delete cr;
}

void GSOPGSLIB::UpdateIdentifiers(const Array<long long> &ids)
{
   long long minval = ids.Min();
#ifdef MFEM_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &minval, 1, MPI_LONG_LONG_INT,
                 MPI_MIN, gsl_comm->c);
#endif
   MFEM_VERIFY(minval >= 0, "Unique identifier cannot be negative.");
   if (gsl_data != NULL) { gslib_gs_free(gsl_data); }
   num_ids = ids.Size();
   gsl_data = gslib_gs_setup(ids.GetData(),
                             ids.Size(),
                             gsl_comm, 0,
                             gslib::gs_crystal_router, 0);
}

void GSOPGSLIB::GS(Vector &senddata, GSOp op)
{
   MFEM_VERIFY(senddata.Size() == num_ids,
               "Incompatible setup and GOP operation.");
   if (op == GSOp::ADD)
   {
      gslib_gs(senddata.GetData(),gslib::gs_double,gslib::gs_add,0,gsl_data,0);
   }
   else if (op == GSOp::MUL)
   {
      gslib_gs(senddata.GetData(),gslib::gs_double,gslib::gs_mul,0,gsl_data,0);
   }
   else if (op == GSOp::MAX)
   {
      gslib_gs(senddata.GetData(),gslib::gs_double,gslib::gs_max,0,gsl_data,0);
   }
   else if (op == GSOp::MIN)
   {
      gslib_gs(senddata.GetData(),gslib::gs_double,gslib::gs_min,0,gsl_data,0);
   }
   else
   {
      MFEM_ABORT("Invalid GSOp operation.");
   }
}

// Get Hash Range
static void GetHashRange(const int d, const Array<int> &lh_n,
                         const Vector &lh_fac,
                         const Vector &lh_bnd_min, const Vector &lh_bnd_max,
                         const double &xmin, const double &xmax,
                         int &imin, int &imax)
{
   int i0 = floor( (xmin - lh_bnd_min[d]) * lh_fac[d] );
   int i1 = ceil ( (xmax - lh_bnd_min[d]) * lh_fac[d] );
   imin = i0 < 0 ? 0 : i0;
   imax = i1 < lh_n[d] ? i1 : lh_n[d];
   if (imax == imin) { ++imax; }
}


static void SetHashFac(Vector &lh_fac, const Array<int> &nx,
                       const Vector &lh_bnd_min, const Vector &lh_bnd_max)
{
   int dim = lh_bnd_min.Size();
   for (int d = 0; d < dim; d++)
   {
      lh_fac[d] = nx[d] / (lh_bnd_max[d] - lh_bnd_min[d]);
   }
}


// Get HashCount
static int GetHashCount(const Array<int> &lh_n, const Vector &lh_fac,
                        const Vector &lh_bnd_min, const Vector &lh_bnd_max,
                        const Vector &elmin, const Vector &elmax,
                        Array<int> &elmin_h, Array<int> &elmax_h)
{
   int count = 0;
   const int dim = lh_bnd_min.Size();
   const int nel = elmin.Size()/dim;
   elmin_h.SetSize(dim * nel);
   elmax_h.SetSize(dim * nel);
   for (int i = 0; i < nel; i++)
   {
      int count_el = 1;
      for (int d = 0; d < dim; d++)
      {
         GetHashRange(d, lh_n, lh_fac, lh_bnd_min, lh_bnd_max,
                      elmin[d*nel + i], elmax[d*nel + i],
                      elmin_h[d*nel + i], elmax_h[d*nel + i]);
         int imax = elmax_h[d*nel + i];
         int imin = elmin_h[d*nel + i];
         count_el *= (imax - imin);
      }
      count += count_el;
   }
   return count;
}

#if defined(MFEM_USE_MPI)
void GlobalBoundingBoxTensorGridMap::SetupCrystal(const MPI_Comm &comm_)
{
   if (cr != nullptr)
   {
      crystal_free(cr);
   }
   if (gsl_comm != nullptr)
   {
      comm_free(gsl_comm);
   }

   gsl_comm = new gslib::comm;
   cr      = new gslib::crystal;
   comm_init(gsl_comm, comm_);
   gslib::crystal_init(cr, gsl_comm);
}

GlobalBoundingBoxTensorGridMap::GlobalBoundingBoxTensorGridMap(ParMesh &pmesh,
                                                               int nx)
{
   GridFunction *nodes = pmesh.GetNodes();
   MFEM_VERIFY(nodes != nullptr,
               "BoundingBoxTensorGridMap requires a mesh with nodes defined.");
   const int nel = pmesh.GetNE();
   Vector elmin, elmax;
   int nref = 3;
   nodes->GetElementBounds(elmin, elmax, nref);
   dim = elmin.Size() / nel;
   Array<int> nx_arr(dim);
   nx_arr = nx;
   Setup(pmesh.GetComm(), elmin, elmax, nx_arr, nel);
}

GlobalBoundingBoxTensorGridMap::GlobalBoundingBoxTensorGridMap(
   const MPI_Comm &comm,
   Vector &elmin,
   Vector &elmax,
   int n, int nel,
   bool by_max_size)
{
   dim = elmin.Size() / nel;
   Array<int> nx_arr(dim);
   if (!by_max_size)
   {
      nx_arr = n;
   }
   else
   {
      long long int nx = n;
      MPI_Allreduce(MPI_IN_PLACE, &nx, 1, MPI_LONG_LONG, MPI_SUM, comm);
      nx = ceil(pow((double)nx,1./dim));
      nx_arr = nx;
   }
   Setup(comm, elmin, elmax, nx_arr, nel);
}

GlobalBoundingBoxTensorGridMap::GlobalBoundingBoxTensorGridMap(
   const MPI_Comm &comm,
   Vector &elmin, Vector &elmax,
   Array<int> &nx, int nel)
{
   Setup(comm, elmin, elmax, nx, nel);
}

void GlobalBoundingBoxTensorGridMap::Setup(const MPI_Comm &comm,
                                           Vector &elmin, Vector &elmax,
                                           Array<int> &nx, int nel)
{
   SetupCrystal(comm);
   dim = elmin.Size() / nel;
   gh_bnd_min.SetSize(dim);
   gh_bnd_max.SetSize(dim);
   gh_fac.SetSize(dim);
   gh_n.SetSize(dim);
   gh_n = nx;

   MFEM_VERIFY(nx.Size() == dim,
               "BoundingBoxTensorGridMap requires nx to have the same size as the number of dimensions.");
   for (int d = 0; d < nx.Size(); d++)
   {
      MFEM_VERIFY(nx[d] > 0,
                  "BoundingBoxTensorGridMap requires positive number of divisions in each dimension.");
   }
   for (int d = 0; d < dim; d++)
   {
      Vector elmind(elmin.GetData() + d*nel, nel);
      Vector elmaxd(elmax.GetData() + d*nel, nel);
      gh_bnd_min[d] = elmind.Min();
      gh_bnd_max[d] = elmaxd.Max();
   }

   Vector gh_bnd_min_loc = gh_bnd_min;
   Vector gh_bnd_max_loc = gh_bnd_max;

   MPI_Allreduce(MPI_IN_PLACE, gh_bnd_min.GetData(), dim,
                 MFEM_MPI_REAL_T, MPI_MIN, comm);
   MPI_Allreduce(MPI_IN_PLACE, gh_bnd_max.GetData(), dim,
                 MFEM_MPI_REAL_T, MPI_MAX, comm);

   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get rank of the current process
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   SetHashFac(gh_fac, gh_n, gh_bnd_min, gh_bnd_max);

   int gh_nd = gh_n[0];
   for (int d = 1; d < dim; d++)
   {
      gh_nd *= gh_n[d];
   }

   // Hash cell ranges for each element in each dimension
   Array<int> elmin_h, elmax_h;
   int store_size = GetHashCount(gh_n, gh_fac,
                                 gh_bnd_min, gh_bnd_max,
                                 elmin, elmax,
                                 elmin_h, elmax_h);

   Array<int> loc_idx_min(dim), loc_idx_max(dim), lh_n(dim);
   int loc_idx_tot = 1;

   for (int d = 0; d < dim; d++)
   {
      GetHashRange(d, gh_n, gh_fac, gh_bnd_min, gh_bnd_max,
                   gh_bnd_min_loc[d], gh_bnd_max_loc[d],
                   loc_idx_min[d], loc_idx_max[d]);
      lh_n[d] = loc_idx_max[d] - loc_idx_min[d];
      loc_idx_tot *= lh_n[d];
   }

   struct hashInfo_s
   {
      unsigned int index, proc;
   };
   struct gslib::array hashInfo_pt;
   array_init(struct hashInfo_s, &hashInfo_pt, store_size);
   hashInfo_pt.n=store_size;

   struct hashInfo_s *pt = (struct hashInfo_s *)hashInfo_pt.ptr;
   Array<int> marker(loc_idx_tot);
   marker = 0;

   for (int e = 0; e < nel; e++)
   {
      int klim = dim < 3 ? 1 : (elmax_h[2*nel+e]-elmin_h[2*nel+e]);
      int jlim = dim < 2 ? 1 : (elmax_h[1*nel+e]-elmin_h[1*nel+e]);
      int ilim = (elmax_h[0*nel+e]-elmin_h[0*nel+e]);
      for (int k = 0; k < klim; k++)
      {
         int koff = dim < 3 ? 0 : (elmin_h[2*nel + e] + k)* gh_n[0] * gh_n[1];
         int koff_loc = dim < 3 ? 0 :
                        (elmin_h[2*nel+e]+k - loc_idx_min[2])*lh_n[0] * lh_n[1];
         for (int j = 0; j < jlim; j++)
         {
            int joff = dim < 2 ? 0 : (elmin_h[1*nel + e] + j) * gh_n[0];
            int joff_loc = dim < 2 ? 0 :
                           (elmin_h[1*nel+e]+j - loc_idx_min[1])*lh_n[0];
            for (int i = 0; i < ilim; i++)
            {
               int ioff = elmin_h[0*nel + e] + i;
               int ioff_loc = elmin_h[0*nel+e]+i - loc_idx_min[0];
               int idx = ioff + joff + koff;
               int idx_loc = ioff_loc + joff_loc + koff_loc;
               if (marker[idx_loc] == 1) { continue; }
               pt->proc = idx % num_procs;
               pt->index = idx / num_procs;
               marker[idx_loc] = 1;
               ++pt;
            }
         }
      }
   }
   MPI_Barrier(comm);
   int npts = marker.Sum();
   hashInfo_pt.n = npts;

   // transfer info to other ranks and sort by index
   sarray_transfer(struct hashInfo_s, &hashInfo_pt, proc, 1, cr);
   sarray_sort(struct hashInfo_s, hashInfo_pt.ptr, hashInfo_pt.n,
               index, 0, &(cr->data));

   int nrecv = hashInfo_pt.n;

   n_local_cells = (gh_nd-1)/num_procs+1;
   gh_offset.SetSize(n_local_cells + 1 + nrecv);
   gh_offset = 0;
   Array<int> hash_el_count(n_local_cells);
   hash_el_count = 0;

   pt = (struct hashInfo_s *)hashInfo_pt.ptr;
   for (int i = 0; i < nrecv; i++)
   {
      int idx = pt->index;
      hash_el_count[idx]++;
      ++pt;
   }

   gh_offset[0] = n_local_cells + 1;
   for (int e = 0; e < n_local_cells; e++)
   {
      gh_offset[e + 1] = gh_offset[e] + hash_el_count[e];
   }

   pt = (struct hashInfo_s *)hashInfo_pt.ptr;
   for (int i = 0; i < nrecv; i++)
   {
      int idx = pt->index;
      int proc = pt->proc;
      gh_offset[gh_offset[idx+1]-hash_el_count[idx]]=proc;
      hash_el_count[idx]--;
      ++pt;
   }

   array_free(&hashInfo_pt);
   MPI_Barrier(comm);
}

int GlobalBoundingBoxTensorGridMap::GetGlobalHashCellFromPoint(
   Vector &xyz) const
{
   MFEM_ASSERT(xyz.Size() == dim,
               "Point must have the same dimension as the hash.");
   int sum = 0;
   for (int d = dim-1; d >= 0; --d)
   {
      if (xyz(d) < gh_bnd_min(d) || xyz(d) > gh_bnd_max(d))
      {
         return -1; // Point is outside the bounds of the hash
      }
      sum *= gh_n[d];
      int i = (int)floor((xyz(d) - gh_bnd_min(d)) * gh_fac[d]);
      sum += i < 0 ? 0 : (gh_n[d] - 1 < i ? gh_n[d] - 1 : i);
   }
   return sum;
}

void GlobalBoundingBoxTensorGridMap::GlobalHashCellToProcAndLocalIndex(int i,
                                                                       int &proc, int &idx) const
{
   proc = i % num_procs;
   idx = i / num_procs;
}

void GlobalBoundingBoxTensorGridMap::GetProcAndLocalIndexFromPoint(Vector &xyz,
                                                                   int &proc, int &idx) const
{
   int cell = GetGlobalHashCellFromPoint(xyz);
   if (cell < 0)
   {
      proc = -1;
      idx = -1;
      return; // Point is outside the bounds of the hash
   }
   GlobalHashCellToProcAndLocalIndex(cell, proc, idx);
}

void GlobalBoundingBoxTensorGridMap::MapPointsToProcs(Vector &xyz, int ordering,
                                                      std::map<int, std::vector<int>> &pt_idx_to_procs) const
{
   MFEM_ASSERT(xyz.Size() % dim == 0,
               "Point array size must be a multiple of the hash dimension.");
   int npts = xyz.Size() / dim;
   pt_idx_to_procs.clear();

   // struct to hold point info:
   // info: holds local hash cell index when first sent.
   //       holds proc when returned back.
   // proc: holds the proc that the hash cell info is on.
   // loc_index: local index of the point in the input vector.
   struct ptInfo_s
   {
      unsigned int info, proc, loc_index;
   };
   struct gslib::array ptInfo_pt;
   array_init(struct ptInfo_s, &ptInfo_pt, npts);
   ptInfo_pt.n=npts;
   struct ptInfo_s *pt = (struct ptInfo_s *)ptInfo_pt.ptr;

   for (int i = 0; i < npts; i++)
   {
      Vector pt_xyz(dim);
      for (int d = 0; d < dim; d++)
      {
         pt_xyz(d) = ordering == 0 ? xyz(d*npts + i) : xyz(i*dim + d);
      }

      int cell = GetGlobalHashCellFromPoint(pt_xyz);
      if (cell < 0)
      {
         ++pt;
         continue; // Point is outside the bounds of the hash
      }
      int proc, idx;
      GlobalHashCellToProcAndLocalIndex(cell, proc, idx);
      pt->info = idx;
      pt->proc = proc;
      pt->loc_index = i;
      ++pt;
   }
   MPI_Barrier(MPI_COMM_WORLD);

   // transfer info to ranks that hold each hash cell's info
   sarray_transfer(struct ptInfo_s, &ptInfo_pt, proc, 1, cr);

   int nrecv = ptInfo_pt.n;
   pt = (struct ptInfo_s *)ptInfo_pt.ptr;
   int ncount = 0;
   for (int i = 0; i < nrecv; i++)
   {
      int idx = pt->info;
      int loc_count = gh_offset[idx+1]-gh_offset[idx];
      ncount += loc_count;
      ++pt;
   }

   struct gslib::array sendptInfo_pt;
   array_init(struct ptInfo_s, &sendptInfo_pt, ncount);
   sendptInfo_pt.n=ncount;
   struct ptInfo_s *spt = (struct ptInfo_s *)sendptInfo_pt.ptr;
   pt = (struct ptInfo_s *)ptInfo_pt.ptr;

   for (int i = 0; i < nrecv; i++)
   {
      Array<int> procs = MapCellToProcs(pt->info);
      for (int k = 0; k < procs.Size(); k++)
      {
         spt->info = procs[k];
         spt->proc = pt->proc;
         spt->loc_index = pt->loc_index;
         ++spt;
      }
      ++pt;
   }

   array_free(&ptInfo_pt);
   sarray_transfer(struct ptInfo_s, &sendptInfo_pt, proc, 1, cr);

   nrecv = sendptInfo_pt.n;
   spt = (struct ptInfo_s *)sendptInfo_pt.ptr;

   for (int i =0; i < nrecv; i++)
   {
      int pt_idx = spt->loc_index;
      int proc = spt->info;
      pt_idx_to_procs[pt_idx].push_back(proc);
      ++spt;
   }

   array_free(&sendptInfo_pt);
   MPI_Barrier(MPI_COMM_WORLD);
}

Array<int> GlobalBoundingBoxTensorGridMap::MapCellToProcs(int l_idx) const
{
   MFEM_ASSERT(l_idx >= 0 && l_idx < n_local_cells-1,
               "Access element " << l_idx << " of local hash with cells = "
               << n_local_cells - 1);
   int start = gh_offset[l_idx];
   int end = gh_offset[l_idx + 1];
   Array<int> elements(end - start);
   for (int j = start; j < end; j++)
   {
      elements[j - start] = gh_offset[j];
   }
   return elements;
}

GlobalBoundingBoxTensorGridMap::~GlobalBoundingBoxTensorGridMap()
{
   if (cr != nullptr)
   {
      crystal_free(cr);
      cr = nullptr;
   }
   if (gsl_comm != nullptr)
   {
      comm_free(gsl_comm);
      gsl_comm = nullptr;
   }
}

#endif // defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

} // namespace mfem
#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND

#endif // MFEM_USE_GSLIB
