// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
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

#include "AnalyticSurface.hpp"

#include <utility>

namespace mfem
{

AnalyticCompositeSurface::AnalyticCompositeSurface
    (const Array<AnalyticSurface *> &surf, Array<int> dof_surface)
    : surfaces(surf), dof_to_surface(std::move(dof_surface))
{
   for (int i = 0; i < dof_to_surface.Size(); i++)
   {
      MFEM_VERIFY(dof_to_surface[i] >= -1 &&
                  dof_to_surface[i] < surfaces.Size(),
                  "Invalid analytic surface index.");
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
      surfaces[s]->ConvertPhysCoordToParam(coord_x, coord_t,
                                           dof_to_surface, s);
   }
}

void AnalyticCompositeSurface::ConvertParamCoordToPhys(const Vector &coord_t,
                                                       Vector &coord_x) const
{
   coord_x = coord_t;
   for (int s = 0; s < surfaces.Size(); s++)
   {
      surfaces[s]->ConvertParamCoordToPhys(coord_t, coord_x,
                                           dof_to_surface, s);
   }
}

void AnalyticCompositeSurface::ConvertParamToPhys(const Array<int> &vdofs,
                                                  const Vector &coord_t,
                                                  Vector &coord_x) const
{
   coord_x = coord_t;
   for (int s = 0; s < surfaces.Size(); s++)
   {
      surfaces[s]->ConvertParamToPhys(vdofs, coord_t, coord_x,
                                      dof_to_surface, s);
   }
}

void AnalyticCompositeSurface::PushForwardTangent(const Vector &t,
                                                   const Vector &dt,
                                                   Vector &dx) const
{
   MFEM_VERIFY(t.Size() == dt.Size(), "incompatible state and direction");
   MFEM_VERIFY(dof_to_surface.Size() > 0 &&
               t.Size() % dof_to_surface.Size() == 0,
               "invalid parametric state size");

   const int ndof = dof_to_surface.Size();
   const int dim = t.Size() / ndof;
   dx.SetSize(t.Size());
   dx = 0.0;

   for (int i = 0; i < ndof; i++)
   {
      const AnalyticSurface *surface = GetSurface(i);
      if (!surface)
      {
         for (int d = 0; d < dim; d++)
         {
            dx(i + d * ndof) = dt(i + d * ndof);
         }
         continue;
      }

      double param[3] = { 0.0, 0.0, 0.0 };
      for (int d = 0; d < dim; d++) { param[d] = t(i + d * ndof); }

      DenseMatrix deriv;
      surface->Deriv_1(param, deriv);
      MFEM_VERIFY(deriv.Height() == dim &&
                  surface->NumParams() <= dim,
                  "invalid analytic surface derivative");
      for (int c = 0; c < dim; c++)
      {
         for (int a = 0; a < surface->NumParams(); a++)
         {
            dx(i + c * ndof) +=
               deriv(c, a) * dt(i + a * ndof);
         }
      }
   }
}

void AnalyticCompositeSurface::PullBackDual(const Vector &t,
                                            const Vector &rx,
                                            Vector &rt) const
{
   MFEM_VERIFY(t.Size() == rx.Size(), "incompatible state and dual vector");
   MFEM_VERIFY(dof_to_surface.Size() > 0 &&
               t.Size() % dof_to_surface.Size() == 0,
               "invalid parametric state size");

   const int ndof = dof_to_surface.Size();
   const int dim = t.Size() / ndof;
   rt.SetSize(t.Size());
   rt = 0.0;

   for (int i = 0; i < ndof; i++)
   {
      const AnalyticSurface *surface = GetSurface(i);
      if (!surface)
      {
         for (int d = 0; d < dim; d++)
         {
            rt(i + d * ndof) = rx(i + d * ndof);
         }
         continue;
      }

      double param[3] = { 0.0, 0.0, 0.0 };
      for (int d = 0; d < dim; d++) { param[d] = t(i + d * ndof); }

      DenseMatrix deriv;
      surface->Deriv_1(param, deriv);
      MFEM_VERIFY(deriv.Height() == dim &&
                  surface->NumParams() <= dim,
                  "invalid analytic surface derivative");
      for (int a = 0; a < surface->NumParams(); a++)
      {
         for (int c = 0; c < dim; c++)
         {
            rt(i + a * ndof) +=
               deriv(c, a) * rx(i + c * ndof);
         }
      }
   }
}

void AnalyticCompositeSurface::PullBackGradientAction(
   const Vector &t, const Vector &rx, const Vector &dt,
   const Vector &hx_dx, Vector &rt) const
{
   MFEM_VERIFY(t.Size() == rx.Size() && t.Size() == dt.Size() &&
               t.Size() == hx_dx.Size(),
               "incompatible gradient action vectors");
   MFEM_VERIFY(dof_to_surface.Size() > 0 &&
               t.Size() % dof_to_surface.Size() == 0,
               "invalid parametric state size");

   const int ndof = dof_to_surface.Size();
   const int dim = t.Size() / ndof;
   rt.SetSize(t.Size());
   rt = 0.0;

   for (int i = 0; i < ndof; i++)
   {
      const AnalyticSurface *surface = GetSurface(i);
      if (!surface)
      {
         for (int d = 0; d < dim; d++)
         {
            rt(i + d * ndof) = hx_dx(i + d * ndof);
         }
         continue;
      }

      double param[3] = { 0.0, 0.0, 0.0 };
      for (int d = 0; d < dim; d++) { param[d] = t(i + d * ndof); }

      DenseMatrix deriv_1;
      DenseTensor deriv_2;
      surface->Deriv_1(param, deriv_1);
      surface->Deriv_2(param, deriv_2);
      const int nparam = surface->NumParams();
      MFEM_VERIFY(deriv_1.Height() == dim && nparam <= dim,
                  "invalid analytic surface derivative");

      for (int a = 0; a < nparam; a++)
      {
         for (int c = 0; c < dim; c++)
         {
            rt(i + a * ndof) +=
               deriv_1(c, a) * hx_dx(i + c * ndof);
            for (int b = 0; b < nparam; b++)
            {
               rt(i + a * ndof) +=
                  rx(i + c * ndof) * deriv_2(c, a, b) *
                  dt(i + b * ndof);
            }
         }
      }
   }
}

void Analytic2DCurve::ConvertPhysCoordToParam(const Vector &coord_x,
                                              Vector &coord_t,
                                              const Array<int> &dof_to_surface,
                                              int surface_id)
{
   const int ndof = coord_x.Size() / 2;
   double t;
   dists.SetSize(ndof);
   dists = 0.0;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_to_surface[i] == surface_id)
      {
         t_of_xy(coord_x(i), coord_x(ndof + i), dists(i), t);
         coord_t(i)        = t;
         coord_t(ndof + i) = 0.0;
      }
   }
}

void Analytic2DCurve::ConvertParamCoordToPhys(const Vector &coord_t,
                                              Vector &coord_x,
                                              const Array<int> &dof_to_surface,
                                              int surface_id) const
{
   const int ndof = coord_x.Size() / 2;
   double x, y;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_to_surface[i] == surface_id)
      {
         xy_of_t(coord_t(i), dists(i), x, y);
         coord_x(i)        = x;
         coord_x(ndof + i) = y;
      }
   }
}

void Analytic2DCurve::ConvertParamToPhys(const Array<int> &vdofs,
                                         const Vector &coord_t,
                                         Vector &coord_x,
                                         const Array<int> &dof_to_surface,
                                         int surface_id) const
{
   const int ndof = vdofs.Size() / 2;
   double x, y;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_to_surface[vdofs[i]] == surface_id)
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
                                              Vector &coord_t,
                                              const Array<int> &dof_to_surface,
                                              int surface_id)
{
   const int ndof = coord_x.Size() / 3;
   double t;
   dists.SetSize(2 * ndof);
   dists = 0.0;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_to_surface[i] == surface_id)
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
                                              Vector &coord_x,
                                              const Array<int> &dof_to_surface,
                                              int surface_id) const
{
   const int ndof = coord_x.Size() / 3;
   double x, y, z;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_to_surface[i] == surface_id)
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
                                         Vector &coord_x,
                                         const Array<int> &dof_to_surface,
                                         int surface_id) const
{
   const int ndof = vdofs.Size() / 3;
   double x, y, z;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_to_surface[vdofs[i]] == surface_id)
      {
         xyz_of_t(coord_t(i), dists(vdofs[i]),
                  dists(dof_to_surface.Size() + vdofs[i]), x, y, z);
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
                                                Vector &coord_t,
                                                const Array<int> &dof_to_surface,
                                                int surface_id)
{
   const int ndof = coord_x.Size() / 3;
   double u, v;
   dists.SetSize(ndof);
   dists = 0.0;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_to_surface[i] == surface_id)
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
                                                Vector &coord_x,
                                                const Array<int> &dof_to_surface,
                                                int surface_id) const
{
   const int ndof = coord_x.Size() / 3;
   double x, y, z;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_to_surface[i] == surface_id)
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
                                           Vector &coord_x,
                                           const Array<int> &dof_to_surface,
                                           int surface_id) const
{
   const int ndof = vdofs.Size() / 3;
   double x, y, z;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_to_surface[vdofs[i]] == surface_id)
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
