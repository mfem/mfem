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

AnalyticSurface::AnalyticSurface(const Array<int> &marker)
   : dof_to_surface(marker),
     distance_gf()
{
   //geometry->ComputeDistances(coord, pmesh, pfes_mesh);
}

void AnalyticCompositeSurface::ConvertPhysCoordToParam(const Vector &coord_x,
                                                       Vector &coord_t) const
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
                                              Vector &coord_t) const
{
   const int ndof = coord_x.Size() / 2;
   double t;
   for (int i = 0; i < ndof; i++)
   {
      if (dof_to_surface[i] == surface_id)
      {
         t_of_xy(coord_x(i), coord_x(ndof + i), distance_gf, t);
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
      if (dof_to_surface[i] == surface_id)
      {
         xy_of_t(coord_t(i), distance_gf, x, y);
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
      if (dof_to_surface[vdofs[i]] == surface_id)
      {
         xy_of_t(coord_t(i), distance_gf, x, y);
         coord_x(i)        = x;
         coord_x(ndof + i) = y;
      }
   }
}

void Analytic2DCurve::Deriv_1(const double *param, double *deriv) const
{
   deriv[0] = dx_dt(param[0]);
   deriv[1] = dy_dt(param[0]);
}

void Analytic2DCurve::Deriv_2(const double *param, double *deriv) const
{
   deriv[0] = dx_dtdt(param[0]);
   deriv[1] = dy_dtdt(param[0]);
}

} // namespace mfem
