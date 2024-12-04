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

#ifndef MFEM_FE_H1
#define MFEM_FE_H1

#include "fe_base.hpp"
#include "fe_pyramid.hpp"

namespace mfem
{

/// Arbitrary order H1 elements in 1D
class H1_SegmentElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x, d2shape_x;
#endif

public:
   /// Construct the H1_SegmentElement of order @a p and BasisType @a btype
   H1_SegmentElement(const int p, const int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &Hessian) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order H1 elements in 2D on a square
class H1_QuadrilateralElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y, d2shape_x, d2shape_y;
#endif

public:
   /// Construct the H1_QuadrilateralElement of order @a p and BasisType @a btype
   H1_QuadrilateralElement(const int p,
                           const int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &Hessian) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order H1 elements in 3D on a cube
class H1_HexahedronElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z,
           d2shape_x, d2shape_y, d2shape_z;
#endif

public:
   /// Construct the H1_HexahedronElement of order @a p and BasisType @a btype
   H1_HexahedronElement(const int p, const int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &Hessian) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order H1 elements in 2D on a triangle
class H1_TriangleElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
   mutable Vector ddshape_x, ddshape_y, ddshape_l;
   mutable DenseMatrix du, ddu;
#endif
   DenseMatrixInverse Ti;

public:
   /// Construct the H1_TriangleElement of order @a p and BasisType @a btype
   H1_TriangleElement(const int p, const int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &ddshape) const override;
};


/// Arbitrary order H1 elements in 3D  on a tetrahedron
class H1_TetrahedronElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l, u;
   mutable Vector ddshape_x, ddshape_y, ddshape_z, ddshape_l;
   mutable DenseMatrix du, ddu;
#endif
   DenseMatrixInverse Ti;

public:
   /// Construct the H1_TetrahedronElement of order @a p and BasisType @a btype
   H1_TetrahedronElement(const int p,
                         const int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &ddshape) const override;
};



/// Arbitrary order H1 elements in 3D on a wedge
class H1_WedgeElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   H1_TriangleElement TriangleFE;
   H1_SegmentElement  SegmentFE;

public:
   /// Construct the H1_WedgeElement of order @a p and BasisType @a btype
   H1_WedgeElement(const int p,
                   const int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

class H1_FuentesPyramidElement
   : public NodalFiniteElement, public FuentesPyramid
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector tmp_i, u;
   mutable DenseMatrix tmp1_ij, tmp2_ij, du;
   mutable DenseTensor tmp_ijk;
#endif
   DenseMatrixInverse Ti;
   /*
    static void calcBasis(const int p, const IntegrationPoint &ip,
                          Vector *tmp_x, double *tmp_y, double *tmp_z,
                          double *u);
   */
   void calcBasis(const int p, const IntegrationPoint &ip,
                  Vector &phi_i, DenseMatrix &phi_ij, Vector &u) const;
   void calcGradBasis(const int p, const IntegrationPoint &ip,
                      Vector &phi_i, DenseMatrix &dphi_i,
                      DenseMatrix &phi_ij, DenseTensor &dphi_ij,
                      DenseMatrix &du) const;
   /*
   {
       calcBasis(p, ip, tmp_x, tmp_y.GetData(), tmp_z.GetData(),
                 u.GetData());
    }
   */
   // static void calcDBasis(const int p, const IntegrationPoint &ip,
   //                      DenseMatrix &du);

public:
   H1_FuentesPyramidElement(const int p,
                            const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   void CalcRawShape(const IntegrationPoint &ip, Vector &shape) const;
   void CalcRawDShape(const IntegrationPoint &ip,
                      DenseMatrix &dshape) const;
};

class H1_BergotPyramidElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_0, shape_1, shape_2;
   mutable Vector dshape_0_0, dshape_1_0, dshape_2_0;
   mutable Vector dshape_0_1, dshape_1_1, dshape_2_1;
   // mutable Vector u;
   // mutable DenseMatrix du;
   mutable Vector shape_x, shape_y, shape_z;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_z_dt, u;
   mutable Vector ddshape_x, ddshape_y, ddshape_z;
   mutable DenseMatrix du, ddu;
#endif
   DenseMatrixInverse Ti;
   /*
    static void calcBasis(const int p, const IntegrationPoint &ip,
                          double *tmp_x, double *tmp_y, double *tmp_z,
                          double *u);
    // static void calcDBasis(const int p, const IntegrationPoint &ip,
    //                      DenseMatrix &du);

    static inline double lam0(const double x, const double y, const double z)
    { return (z < 1.0) ? (1.0 - x - z) * (1.0 - y - z) / (1.0 - z): 0.0; }
    static inline double lam1(const double x, const double y, const double z)
    { return (z < 1.0) ? x * (1.0 - y - z) / (1.0 - z): 0.0; }
    static inline double lam2(const double x, const double y, const double z)
    { return (z < 1.0) ? x * y / (1.0 - z): 0.0; }
    static inline double lam3(const double x, const double y, const double z)
    { return (z < 1.0) ? (1.0 - x - z) * y / (1.0 - z): 0.0; }
    static inline double lam4(const double x, const double y, const double z)
    { return z; }

    static void grad_lam0(const double x, const double y, const double z,
                          double du[]);
    static void grad_lam1(const double x, const double y, const double z,
                          double du[]);
    static void grad_lam2(const double x, const double y, const double z,
                          double du[]);
    static void grad_lam3(const double x, const double y, const double z,
                          double du[]);
    static void grad_lam4(const double x, const double y, const double z,
                          double du[]);

    static inline double mu0(const double x) { return 1.0 - x; }
    static inline double mu1(const double x) { return x; }

    static inline double dmu0(const double x) { return -1.0; }
    static inline double dmu1(const double x) { return 1.0; }


    static inline double nu0(const double x, const double y)
    { return 1.0 - x - y; }
    static inline double nu1(const double x, const double y) { return x; }
    static inline double nu2(const double x, const double y) { return y; }

    static inline void grad_nu0(const double x, const double y, double dnu[])
    { dnu[0] = -1.0; dnu[1] = -1.0;}
    static inline void grad_nu1(const double x, const double y, double dnu[])
    { dnu[0] = 1.0; dnu[1] = 0.0;}
    static inline void grad_nu2(const double x, const double y, double dnu[])
    { dnu[0] = 0.0; dnu[1] = 1.0;}

    static void phi_E(const int p, const double s0, double s1, double *u);
    static void phi_E(const int p, const double s0, double s1, double *u,
                      double *duds0, double *duds1);

    static void calcScaledLegendre(const int p, const double x, const double t,
                                   double *u);
    static void calcScaledLegendre(const int p, const double x, const double t,
                                   double *u, double *dudx, double *dudt);

    static void calcIntegratedLegendre(const int p, const double x,
                                       const double t, double *u);
    static void calcIntegratedLegendre(const int p, const double x,
                                       const double t, double *u,
                                       double *dudx, double *dudt);

    static void calcScaledJacobi(const int p, const double alpha,
                                 const double x, const double t,
                                 double *u);
    static void calcScaledJacobi(const int p, const double alpha,
                                 const double x, const double t,
                                 double *u, double *dudx, double *dudt);

    static void calcIntegratedJacobi(const int p, const double alpha,
                                     const double x, const double t,
                                     double *u);
    static void calcIntegratedJacobi(const int p, const double alpha,
                                     const double x, const double t,
                                     double *u, double *dudx, double *dudt);
   */
public:
   H1_BergotPyramidElement(const int p,
                           const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

} // namespace mfem

#endif
