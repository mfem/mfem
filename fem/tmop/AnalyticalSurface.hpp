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

#ifndef MFEM_ANALYTICAL_SURFACE
#define MFEM_ANALYTICAL_SURFACE

#include "../pgridfunc.hpp"

namespace mfem
{

class AnalyticSurface
{
protected:
   friend class AnalyticCompositeSurface;

   const Array<int> &dof_to_surface;

public:
   ParGridFunction distance_gf;

   AnalyticSurface(const Array<int> &marker);

   // Go from physical to parametric coordinates on the whole mesh.
   // 2D: (x, y)    -> t.
   // 3D: (x, y, z) -> (u, v).
   virtual void ConvertPhysCoordToParam(const Vector &coord_x,
                                        Vector &coord_t) const = 0;

   // Go from parametric to physical coordinates on the whole mesh:
   // 2D: t      -> (x, y).
   // 3D: (u, v) -> (x, y, z).
   virtual void ConvertParamCoordToPhys(const Vector &coord_t,
                                        Vector &coord_x) const = 0;

   // Go from parametric to physical coordinates on a single element.
   virtual void ConvertParamToPhys(const Array<int> &vdofs,
                                   const Vector &coord_t,
                                   Vector &coord_x) const = 0;

   // First derivatives:
   // 2D curve:  t     -> (dx/dt, dy/dt).
   // 3D curve:  t     -> (dx/dt, dy/dt, dz/dt).
   // 3D surf:  (u, v) -> (dx/du, dy/du, dz/du, dx/dv, dy/dv, dz/dv).
   virtual void Deriv_1(const double *param, double *deriv) const = 0;

   // Second derivatives:
   // 2D curve:  t     -> (dx_dtdt, dy_dtdt).
   // 3D curve:  t     -> (dx_dtdt, dy_dtdt, dz_dtdt).
   virtual void Deriv_2(const double *param, double *deriv) const = 0;
};

class AnalyticCompositeSurface : public AnalyticSurface
{
protected:
   const Array<const AnalyticSurface *> &surfaces;
   Array<int> d_t_s;

public:
   AnalyticCompositeSurface(const Array<int> &dof_surf,
                            const Array<const AnalyticSurface *> &surf);

   /// Must be called after the Array of surfaces is changed.
   void UpdateDofToSurface();

   const AnalyticSurface *GetSurface(int surf_id) const
   { return surfaces[surf_id]; }

   void ConvertPhysCoordToParam(const Vector &coord_x,
                                Vector &coord_t) const override;

   void ConvertParamCoordToPhys(const Vector &coord_t,
                                Vector &coord_x) const override;

   void ConvertParamToPhys(const Array<int> &vdofs,
                           const Vector &coord_t,
                           Vector &coord_x) const override;

   void Deriv_1(const double *param, double *deriv) const override
   { MFEM_ABORT("Use GetSurface(surface_id)->Deriv_1(...);") }

   void Deriv_2(const double *param, double *deriv) const override
   { MFEM_ABORT("Use GetSurface(surface_id)->Deriv_2(...);") }
};

class Analytic2DCurve : public AnalyticSurface
{
protected:
   const int surface_id;

public:
   Analytic2DCurve(const Array<int> &marker, int surf_id)
    : AnalyticSurface(marker), surface_id(surf_id) { }

   // (x, y) -> t on the whole mesh.
   void ConvertPhysCoordToParam(const Vector &coord_x,
                                Vector &coord_t) const override;

   // t -> (x, y) on the whole mesh.
   void ConvertParamCoordToPhys(const Vector &coord_t,
                                Vector &coord_x) const override;

   // t -> (x, y) on a single element.
   void ConvertParamToPhys(const Array<int> &vdofs,
                           const Vector &coord_t,
                           Vector &coord_x) const override;

   // t -> (dx_dt, dy_dt).
   void Deriv_1(const double *param, double *deriv) const override;

   // t -> (dx_dtdt, dy_dtdt).
   void Deriv_2(const double *param, double *deriv) const override;

   virtual void xy_of_t(double t, const Vector &dist,
                        double &x, double &y) const = 0;
   virtual void t_of_xy(double x, double y, const Vector &dist,
                        double &t) const = 0;

   virtual double dx_dt(double t) const = 0;
   virtual double dy_dt(double t) const = 0;

   virtual double dx_dtdt(double t) const = 0;
   virtual double dy_dtdt(double t) const = 0;
};

}
#endif
