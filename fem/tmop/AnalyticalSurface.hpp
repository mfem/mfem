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

class Square
{
   protected:

   ParFiniteElementSpace &pfes_mesh;
   ParGridFunction &distance_gf;
   const ParMesh &pmesh;
   const ParGridFunction &coord;

   // attr = 1 ==> top
   // attr = 2 ==> right
   // attr = 3 ==> bottom
   // attr = 4 ==> left
   double width;
   double depth;
   Vector center;
   Array<int> attr_marker;
   std::vector<double> cornersX;
   std::vector<double> cornersY;

   public:
   Square(ParFiniteElementSpace &pfes_mesh, ParGridFunction &distance_gf,
          const ParMesh & pmesh, const ParGridFunction & coord);

   virtual void GetTFromX(Vector &coordsT, const Vector &coordsX, const int & j_x, const Vector &dist);
   virtual void GetXFromT(Vector &coordsX, const Vector &coordsT, const int &j_x, const Vector &dist);
   virtual void ComputeDistances(const ParGridFunction &coord, const ParMesh & pmesh, const ParFiniteElementSpace &pfes_mesh);
   virtual void SetScaleMatrix(const Array<int> &vdofs, int i, int a, DenseMatrix &Pmat_scale);
   virtual void SetScaleMatrixFourthOrder(const Vector &elfun, const Array<int> & vdofs, DenseMatrix &Pmat_scale);
   virtual void SetHessianScaleMatrix(const Vector &elfun, const Array<int> & vdofs, int i, int idim, int j, int jdim, DenseMatrix &Pmat_hessian);
   virtual void convertToPhysical(const Array<int> & vdofs,const Vector &elfun, Vector &convertedX);
};

class AnalyticSurface
{
protected:
   const Array<bool> &surface_dof_marker;

public:
   Square *geometry;
   ParFiniteElementSpace &pfes_mesh;
   ParGridFunction distance_gf;
   const ParGridFunction &coord;
   const ParMesh &pmesh;

   AnalyticSurface(const Array<bool> &marker, ParFiniteElementSpace &pfes_mesh,
                   const ParGridFunction &coord, const ParMesh &pmesh);

   // Go from physical to parametric coordinates on the whole mesh.
   // 2D: (x, y)    -> t.
   // 3D: (x, y, z) -> (u, v).
   virtual void ConvertPhysCoordToParam(GridFunction &coord) const = 0;

   // Go from parametric to physical coordinates on the whole mesh:
   // 2D: t      -> (x, y).
   // 3D: (u, v) -> (x, y, z).
   virtual void ConvertParamCoordToPhys(Vector &coord) const = 0;

   // Go from parametric to physical coordinates on a single element.
   virtual void ConvertParamToPhys(const Array<int> &vdofs,
                                   const Vector &coord_t,
                                   Vector &coord_x) const = 0;

   // First derivatives:
   // 2D: t      -> (dx/dt, dy/dt).
   // 3D: (u, v) -> (dx/du, dy/du, dz/du, dx/dv, dy/dv, dz/dv).
   virtual void Deriv_1(const double *param, double *deriv) const = 0;

   // Derivative d(x_ai) / dt.
   // Fills just one entry of the Pmat_scale.
   void SetScaleMatrix(const Array<int> & vdofs, int i, int a, DenseMatrix & Pmat_scale);

   // First derivatives as a 4th order tensor.
   void SetScaleMatrixFourthOrder(const Vector &elfun, const Array<int> &vdofs, DenseMatrix & Pmat_scale);

   // Second derivatives.
   void SetHessianScaleMatrix(const Vector &elfun, const Array<int> & vdofs, int i, int idim, int j, int jdim, DenseMatrix &Pmat_hessian);

   ~AnalyticSurface();
};

class Analytic2DCurve : public AnalyticSurface
{
public:
   Analytic2DCurve(const Array<bool> &marker, ParFiniteElementSpace &pfes_mesh,
                   const ParGridFunction &coord, const ParMesh &pmesh)
    : AnalyticSurface(marker, pfes_mesh, coord, pmesh) { }

   void ConvertPhysCoordToParam(GridFunction &coord) const override;

   void ConvertParamCoordToPhys(Vector &coord) const override;

   void ConvertParamToPhys(const Array<int> &vdofs,
                           const Vector &coord_t,
                           Vector &coord_x) const override;

   void Deriv_1(const double *param, double *deriv) const override;

   virtual void t_of_xy(double x, double y, const Vector &dist,
                        double &t) const = 0;
   virtual void xy_of_t(double t, const Vector &dist,
                        double &x, double &y) const = 0;

   virtual double dx_dt(double t) const = 0;
   virtual double dy_dt(double t) const = 0;
};

}
#endif
