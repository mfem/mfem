// Copyright (c) 2017, Lawrence LivermoreA National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "AnalyticalSurface.hpp"

namespace mfem
{

AnalyticSurface::AnalyticSurface(const Array<bool> &marker)
    : dof_marker(marker), dists(0)
{
   //geometry->ComputeDistances(coord, pmesh, pfes_mesh);
}

AnalyticCompositeSurface::AnalyticCompositeSurface
    (const Array<AnalyticSurface *> &surf)
    : AnalyticSurface(), surfaces(surf), dof_to_surface(0)
{
   UpdateDofToSurface();
}

// For each DOF, the Array dof_to_surface has its corresponding surface index.
// -1: the DOF is not associated with any surface.
// -2: the DOF is associated with more than one surface.
void AnalyticCompositeSurface::UpdateDofToSurface()
{
   if (surfaces.Size() == 0)
   {
      dof_to_surface.DeleteAll();
      return;
   }

   dof_to_surface.SetSize(surfaces[0]->dof_marker.Size());
   dof_to_surface = -1;
   for (int s = 0; s < surfaces.Size(); s++)
   {
      for (int i = 0; i < dof_to_surface.Size(); i++)
      {
         if (surfaces[s]->dof_marker[i] == true)
         {
            if (dof_to_surface[i] == -1)    { dof_to_surface[i] =  s; }
            else if (dof_to_surface[i] >= 0) { dof_to_surface[i] = -2; }
         }
      }
   }
}

const AnalyticSurface *AnalyticCompositeSurface::GetSurface(int dof_id) const
{
   if (surfaces.Size() == 0)       { return nullptr; }
   if (dof_to_surface[dof_id] < 0) { return nullptr; }

   return surfaces[dof_to_surface[dof_id]];
}


void AnalyticCompositeSurface::ConvertPhysCoordToParam(const Vector &coord_x,
                                                       Vector &coord_t)
{
   coord_t = coord_x;
   for (int s = 0; s < surfaces.Size(); s++)
   {
      surfaces[s]->ConvertPhysCoordToParam(coord_x, coord_t);
   }
}

void AnalyticCompositeSurface::ConvertParamCoordToPhys(const Vector &coord_t,
                                                       Vector &coord_x) const
{
   coord_x = coord_t;
   for (int s = 0; s < surfaces.Size(); s++)
   {
      surfaces[s]->ConvertParamCoordToPhys(coord_t, coord_x);
   }
}

void AnalyticCompositeSurface::ConvertParamToPhys(const Array<int> &vdofs,
                                                  const Vector &coord_t,
                                                  Vector &coord_x) const
{
   coord_x = coord_t;
   for (int s = 0; s < surfaces.Size(); s++)
   {
      surfaces[s]->ConvertParamToPhys(vdofs, coord_t, coord_x);
   }
}

void Analytic2DCurve::ConvertPhysCoordToParam(const Vector &coord_x,
                                              Vector &coord_t)
{
   const int ndof = coord_x.Size() / 2;
   double t;
   dists.SetSize(ndof);
   dists = 0.0;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_marker[i])
      {
         t_of_xy(coord_x(i), coord_x(ndof + i), dists(i), t);
         coord_t(i)        = t;
         coord_t(ndof + i) = 0.0;
      }
   }
}

void Analytic2DCurve::ConvertParamCoordToPhys(const Vector &coord_t,
                                              Vector &coord_x) const
{
   const int ndof = coord_x.Size() / 2;
   double x, y;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_marker[i])
      {
         xy_of_t(coord_t(i), dists(i), x, y);
         coord_x(i)        = x;
         coord_x(ndof + i) = y;
      }
   }
}

void Analytic2DCurve::ConvertParamToPhys(const Array<int> &vdofs,
                                         const Vector &coord_t,
                                         Vector &coord_x) const
{
   const int ndof = vdofs.Size() / 2;
   double x, y;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_marker[vdofs[i]])
      {
         xy_of_t(coord_t(i), dists(vdofs[i]), x, y);
         coord_x(i)        = x;
         coord_x(ndof + i) = y;
      }
   }
}

void Analytic2DCurve::Deriv_1(const double *param, DenseMatrix &deriv) const
{
   deriv.SetSize(2, 1);
   deriv(0, 0) = dx_dt(param[0]);
   deriv(1, 0) = dy_dt(param[0]);
}

void Analytic2DCurve::Deriv_2(const double *param, DenseTensor &deriv) const
{
   deriv.SetSize(2, 1, 1);
   deriv(0, 0, 0) = dx_dtdt(param[0]);
   deriv(1, 0, 0) = dy_dtdt(param[0]);
}

void Analytic3DCurve::ConvertPhysCoordToParam(const Vector &coord_x,
                                              Vector &coord_t)
{
   const int ndof = coord_x.Size() / 3;
   double t;
   dists.SetSize(2 * ndof);
   dists = 0.0;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_marker[i])
      {
         t_of_xyz(coord_x(i), coord_x(ndof + i), coord_x(2 * ndof + i),
                  dists(i), dists(ndof + i), t);
         coord_t(i)            = t;
         coord_t(ndof + i)     = 0.0;
         coord_t(2 * ndof + i) = 0.0;
      }
   }
}

void Analytic3DCurve::ConvertParamCoordToPhys(const Vector &coord_t,
                                              Vector &coord_x) const
{
   const int ndof = coord_x.Size() / 3;
   double x, y, z;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_marker[i])
      {
         xyz_of_t(coord_t(i), dists(i), dists(ndof + i),
                  x, y, z);
         coord_x(i)            = x;
         coord_x(ndof + i)     = y;
         coord_x(2 * ndof + i) = z;
      }
   }
}

void Analytic3DCurve::ConvertParamToPhys(const Array<int> &vdofs,
                                         const Vector &coord_t,
                                         Vector &coord_x) const
{
   const int ndof = vdofs.Size() / 3;
   double x, y, z;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_marker[vdofs[i]])
      {
         xyz_of_t(coord_t(i), dists(vdofs[i]),
                  dists(dof_marker.Size() + vdofs[i]), x, y, z);
         coord_x(i)            = x;
         coord_x(ndof + i)     = y;
         coord_x(2 * ndof + i) = z;
      }
   }
}

void Analytic3DCurve::Deriv_1(const double *param, DenseMatrix &deriv) const
{
   deriv.SetSize(3, 1);
   deriv(0, 0) = dx_dt(param[0]);
   deriv(1, 0) = dy_dt(param[0]);
   deriv(2, 0) = dz_dt(param[0]);
}

void Analytic3DCurve::Deriv_2(const double *param, DenseTensor &deriv) const
{
   deriv.SetSize(3, 1, 1);
   deriv(0, 0, 0) = dx_dtdt(param[0]);
   deriv(1, 0, 0) = dy_dtdt(param[0]);
   deriv(2, 0, 0) = dz_dtdt(param[0]);
}

void Analytic3DSurface::ConvertPhysCoordToParam(const Vector &coord_x,
                                                Vector &coord_t)
{
   const int ndof = coord_x.Size() / 3;
   double u, v;
   dists.SetSize(ndof);
   dists = 0.0;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_marker[i])
      {
         uv_of_xyz(coord_x(i), coord_x(ndof + i), coord_x(2 * ndof + i),
                   dists(i), u, v);
         coord_t(i)            = u;
         coord_t(ndof + i)     = v;
         coord_t(2 * ndof + i) = 0.0;
      }
   }
}

void Analytic3DSurface::ConvertParamCoordToPhys(const Vector &coord_t,
                                                Vector &coord_x) const
{
   const int ndof = coord_x.Size() / 3;
   double x, y, z;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_marker[i])
      {
         xyz_of_uv(coord_t(i), coord_t(ndof + i), dists(i), x, y, z);
         coord_x(i)            = x;
         coord_x(ndof + i)     = y;
         coord_x(2 * ndof + i) = z;
      }
   }
}

void Analytic3DSurface::ConvertParamToPhys(const Array<int> &vdofs,
                                           const Vector &coord_t,
                                           Vector &coord_x) const
{
   const int ndof = vdofs.Size() / 3;
   double x, y, z;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_marker[vdofs[i]])
      {
         xyz_of_uv(coord_t(i), coord_t(ndof + i), dists(vdofs[i]),
                  x, y, z);
         coord_x(i)            = x;
         coord_x(ndof + i)     = y;
         coord_x(2 * ndof + i) = z;
      }
   }
}

void Analytic3DSurface::Deriv_1(const double *param, DenseMatrix &deriv) const
{
   deriv.SetSize(3, 2);
   deriv(0, 0) = dx_du(param[0], param[1]);
   deriv(1, 0) = dy_du(param[0], param[1]);
   deriv(2, 0) = dz_du(param[0], param[1]);
   deriv(0, 1) = dx_dv(param[0], param[1]);
   deriv(1, 1) = dy_dv(param[0], param[1]);
   deriv(2, 1) = dz_dv(param[0], param[1]);
}

void Analytic3DSurface::Deriv_2(const double *param, DenseTensor &deriv) const
{
   deriv.SetSize(3, 2, 2);
   deriv(0, 0, 0) = dx_dudu(param[0], param[1]);
   deriv(1, 0, 0) = dy_dudu(param[0], param[1]);
   deriv(2, 0, 0) = dz_dudu(param[0], param[1]);
   deriv(0, 0, 1) = dx_dudv(param[0], param[1]);
   deriv(1, 0, 1) = dy_dudv(param[0], param[1]);
   deriv(2, 0, 1) = dz_dudv(param[0], param[1]);
   deriv(0, 1, 0) = deriv(0, 0, 1);
   deriv(1, 1, 0) = deriv(1, 0, 1);
   deriv(2, 1, 0) = deriv(2, 0, 1);
   deriv(0, 1, 1) = dx_dvdv(param[0], param[1]);
   deriv(1, 1, 1) = dy_dvdv(param[0], param[1]);
   deriv(2, 1, 1) = dz_dvdv(param[0], param[1]);
}

} // namespace mfem
