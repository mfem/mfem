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

#ifndef __RHS_HPP__
#define __RHS_HPP__

#include "mfem.hpp"
#include "general/forall.hpp"

// 0 - Solution described in the CEED MS 36 report
// 1 - Solution from the "ecp_special_2023" paper (option with cosine):
//       w(n,x) = \sum_{k=0}^n a^k \cos(b^k \pi (x - 1/2)),   x \in [0,1]
//     with a = 1/2, b = 3.
// 2 - Solution from the "ecp_special_2023" paper (option with sine):
//       w(n,x) = \sum_{k=0}^n a^k \sin(b^k \pi x),   x \in [0,1]
//     with a = 1/2, b = 3.
#define CEED_SOLVER_BP_SOLUTION_OPTION 1

namespace mfem
{

#if (CEED_SOLVER_BP_SOLUTION_OPTION == 0)

MFEM_HOST_DEVICE inline
double s(int k, double x)
{
   return sin(2*M_PI*k*x);
}

MFEM_HOST_DEVICE inline
double u(int k, double x)
{
   double skx = s(k,x);
   double sgn = skx < 0 ? -1.0 : 1.0;
   return exp(-1/skx/skx)*sgn;
}

MFEM_HOST_DEVICE inline
double u_xx(int k, double x)
{
   double kpix = k*M_PI*x;
   double csc_2kpix = 1.0/sin(2*kpix);
   double sgn = sin(2*kpix) < 0 ? -1.0 : 1.0;
   return 2*exp(-csc_2kpix*csc_2kpix)*k*k*M_PI*M_PI
          *(1 + 6*cos(4*kpix) + cos(8*kpix))
          *pow(csc_2kpix,6)
          *sgn;
}

MFEM_HOST_DEVICE inline
double w(int n, double x)
{
   double wkx = 0.0;
   double xx = 2*x - 1; // transform from [0,1] to [-1,1]
   for (int j=0; j<n; ++j)
   {
      int k = pow(3, j);
      wkx += u(k, xx);
   }
   return wkx;
}

MFEM_HOST_DEVICE inline
double w_xx(int n, double x)
{
   double wkx = 0.0;
   double xx = 2*x - 1; // transform from [0,1] to [-1,1]
   if (xx == 0.0) { return 0.0; }
   for (int j=0; j<n; ++j)
   {
      int k = pow(3, j);
      wkx += 4*u_xx(k, xx); // factor of four from reference interval transf.
   }
   return wkx;
}

#elif (CEED_SOLVER_BP_SOLUTION_OPTION == 1)

MFEM_HOST_DEVICE inline
double w(int n, double x)
{
   //  w(n,x) = \sum_{k=0}^n a^k \cos(b^k \pi (x - 1/2))
   const double a = 0.5, b = 3.;
   double ak = 1.0;
   double xk = M_PI * (x - 0.5);
   double w_ = ak * cos(xk);
   for (int k = 1; k <= n; k++)
   {
      ak *= a;
      xk *= b;
      w_ += ak * cos(xk);
   }
   return w_;
}

MFEM_HOST_DEVICE inline
double w_x(int n, double x)
{
   //  w'(n,x) = -\pi \sum_{k=0}^n a^k b^k \sin(b^k \pi (x - 1/2))
   const double a = 0.5, b = 3.;
   double ck = -M_PI;
   double xk = M_PI * (x - 0.5);
   double w_x_ = ck * sin(xk);
   for (int k = 1; k <= n; k++)
   {
      ck *= a * b;
      xk *= b;
      w_x_ += ck * sin(xk);
   }
   return w_x_;
}

MFEM_HOST_DEVICE inline
double w_xx(int n, double x)
{
   // w''(n,x) = -\pi^2 \sum_{k=0}^n a^k b^{2 k} \cos(b^k \pi (x - 1/2))
   const double a = 0.5, b = 3.;
   double ck = -(M_PI * M_PI);
   double xk = M_PI * (x - 0.5);
   double w_xx_ = ck * cos(xk);
   for (int k = 1; k <= n; k++)
   {
      ck *= a * b*b;
      xk *= b;
      w_xx_ += ck * cos(xk);
   }
   return w_xx_;
}

#elif (CEED_SOLVER_BP_SOLUTION_OPTION == 2)

MFEM_HOST_DEVICE inline
double w(int n, double x)
{
   //  w(n,x) = \sum_{k=0}^n a^k \sin(b^k \pi x)
   const double a = 0.5, b = 3.;
   double ak = 1.0;
   double xk = M_PI * x;
   double w_ = ak * sin(xk);
   for (int k = 1; k <= n; k++)
   {
      ak *= a;
      xk *= b;
      w_ += ak * sin(xk);
   }
   return w_;
}

MFEM_HOST_DEVICE inline
double w_xx(int n, double x)
{
   // w''(n,x) = -\pi^2 \sum_{k=0}^n a^k b^{2 k} \sin(b^k \pi x)
   const double a = 0.5, b = 3.;
   double ck = -(M_PI * M_PI);
   double xk = M_PI * x;
   double w_xx_ = ck * sin(xk);
   for (int k = 1; k <= n; k++)
   {
      ck *= a * b*b;
      xk *= b;
      w_xx_ += ck * sin(xk);
   }
   return w_xx_;
}

#endif // CEED_SOLVER_BP_SOLUTION_OPTION

struct ExactSolution : Coefficient
{
   int dim, n;
   ExactSolution(int dim_, int n_=0) : dim(dim_), n(n_) { }
   using Coefficient::Eval;
   double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      double xyz[3];
      Vector transip(xyz, 3);
      T.Transform(ip, transip);
      if (dim == 1)
      {
         return w(n, xyz[0]);
      }
      if (dim == 2)
      {
         return w(n, xyz[0])*w(n, xyz[1]);
      }
      else // dim == 3
      {
         return w(n, xyz[0])*w(n, xyz[1])*w(n, xyz[2]);
      }
   }
};

struct ExactGrad : VectorCoefficient
{
   int dim, n;
   ExactGrad(int dim_, int n_)
      : VectorCoefficient(dim_), dim(dim_), n(n_) { }
   using VectorCoefficient::Eval;
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      double xyz[3];
      Vector transip(xyz, 3);
      T.Transform(ip, transip);
      V.SetSize(dim);
      if (dim == 1)
      {
         V(0) = w_x(n, xyz[0]);
      }
      if (dim == 2)
      {
         V(0) = w_x(n, xyz[0])*  w(n, xyz[1]);
         V(1) =   w(n, xyz[0])*w_x(n, xyz[1]);
      }
      else // dim == 3
      {
         const double wnx = w(n, xyz[0]);
         const double wny = w(n, xyz[1]);
         const double wnz = w(n, xyz[2]);
         V(0) = w_x(n, xyz[0])*wny           *wnz;
         V(1) = wnx           *w_x(n, xyz[1])*wnz;
         V(2) = wnx           *wny           *w_x(n, xyz[2]);
      }
   }
};

MFEM_HOST_DEVICE inline
double rhs_1d(const int n, const double *xyz)
{
   return -w_xx(n, xyz[0]);
}

MFEM_HOST_DEVICE inline
double rhs_2d(const int n, const double *xyz)
{
   return -w_xx(n, xyz[0])*w(n, xyz[1]) - w(n, xyz[0])*w_xx(n, xyz[1]);
}

MFEM_HOST_DEVICE inline
double rhs_3d(const int n, const double *xyz)
{
   return -w_xx(n, xyz[0])*w(n, xyz[1])*w(n, xyz[2])
          - w(n, xyz[0])*w_xx(n, xyz[1])*w(n, xyz[2])
          - w(n, xyz[0])*w(n, xyz[1])*w_xx(n, xyz[2]);
}

using RHSFunctionType = double(*)(int dim, const double *xyz);

template <RHSFunctionType F>
void ProjectRHS_(int n, QuadratureFunction &qf)
{
   QuadratureSpaceBase &qs = *qf.GetSpace();
   Mesh &mesh = *qs.GetMesh();
   const IntegrationRule &ir = qs.GetIntRule(0);

   auto *geom = mesh.GetGeometricFactors(ir, GeometricFactors::COORDINATES);

   const int dim = qs.GetMesh()->Dimension();
   const int nq = ir.Size();
   const int N = qf.Size();

   const double *d_x = geom->X.Read();
   double *d_q = qf.Write();

   mfem::forall(N, [=] MFEM_HOST_DEVICE (int ii)
   {
      const int i = ii / nq;
      const int j = ii % nq;
      double xvec[3];
      for (int d = 0; d < dim; ++d)
      {
         xvec[d] = d_x[j + d*nq + i*dim*nq];
      }
      d_q[ii] = F(n, xvec);
   });
}

void ProjectRHS(int n, QuadratureFunction &qf)
{
   const int dim = qf.GetSpace()->GetMesh()->Dimension();
   switch (dim)
   {
      case 1: ProjectRHS_<rhs_1d>(n, qf); break;
      case 2: ProjectRHS_<rhs_2d>(n, qf); break;
      case 3: ProjectRHS_<rhs_3d>(n, qf); break;
      default: MFEM_ABORT("Unsupported dimension.");
   }
}

struct RHS : Coefficient
{
   int dim, n;
   RHS(int dim_, int n_=0) : dim(dim_), n(n_) { }
   using Coefficient::Eval;
   double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      double xyz[3];
      Vector transip(xyz, 3);
      T.Transform(ip, transip);
      if (dim == 1)
      {
         return -w_xx(n, xyz[0]);
      }
      if (dim == 2)
      {
         return -w_xx(n, xyz[0])*w(n, xyz[1]) - w(n, xyz[0])*w_xx(n, xyz[1]);
      }
      else // dim == 3
      {
         return -w_xx(n, xyz[0])*w(n, xyz[1])*w(n, xyz[2])
                - w(n, xyz[0])*w_xx(n, xyz[1])*w(n, xyz[2])
                - w(n, xyz[0])*w(n, xyz[1])*w_xx(n, xyz[2]);
      }
   }

   void Project(QuadratureFunction &qf) override
   {
      ProjectRHS(n,qf);
   }
};

}

#endif
