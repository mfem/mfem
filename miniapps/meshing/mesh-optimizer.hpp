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

// MFEM Mesh Optimizer Miniapp - Serial/Parallel Shared Code

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

real_t size_indicator(const Vector &x)
{
   // semi-circle
   const real_t xc = x(0) - 0.0, yc = x(1) - 0.5,
                zc = (x.Size() == 3) ? x(2) - 0.5 : 0.0;
   const real_t r = sqrt(xc*xc + yc*yc + zc*zc);
   real_t r1 = 0.45; real_t r2 = 0.55; real_t sf=30.0;
   real_t val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));

   val = std::max((real_t) 0.,val);
   val = std::min((real_t) 1.,val);
   return val;
}

real_t size_indicator_periodic(const Vector &x)
{
   // top right
   real_t xc = x(0) - 0.75, yc = x(1) - 0.75,
          zc = (x.Size() == 3) ? x(2) - 0.0 : 0.0;
   real_t r = sqrt(xc*xc + yc*yc + zc*zc);
   real_t r1 = 0.45; real_t r2 = 0.55; real_t sf=30.0;
   real_t val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));
   val = std::max((real_t) 0.,val);
   val = std::min((real_t) 1.,val);

   // bottom right
   xc = x(0) - 0.75; yc = x(1) + 1.25;
   zc = (x.Size() == 3) ? x(2) - 0.0 : 0.0;
   r = sqrt(xc*xc + yc*yc + zc*zc);
   r1 = 0.45; r2 = 0.55; sf=30.0;
   real_t val1 = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));
   val = std::max(val, val1);

   // top left
   xc = x(0) + 1.25; yc = x(1) - 0.75;
   zc = (x.Size() == 3) ? x(2) - 0.0 : 0.0;
   r = sqrt(xc*xc + yc*yc + zc*zc);
   r1 = 0.45; r2 = 0.55; sf=30.0;
   real_t val2 = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));
   val = std::max(val, val2);

   // bottom left
   xc = x(0) + 1.25; yc = x(1) + 1.25;
   zc = (x.Size() == 3) ? x(2) - 0.0 : 0.0;
   r = sqrt(xc*xc + yc*yc + zc*zc);
   r1 = 0.45; r2 = 0.55; sf=30.0;
   real_t val3 = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));
   val = std::max(val, val3);

   return val;
}

void calc_mass_volume(const GridFunction &g, real_t &mass, real_t &vol)
{
   Mesh &mesh = *g.FESpace()->GetMesh();
   const int NE = mesh.GetNE();
   Vector g_vals;
   mass = 0.0, vol = 0.0;
   for (int e = 0; e < NE; e++)
   {
      ElementTransformation &Tr = *mesh.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(mesh.GetElementBaseGeometry(e),
                                               Tr.OrderJ());
      g.GetValues(Tr, ir, g_vals);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         Tr.SetIntPoint(&ip);
         mass   += g_vals(j) * ip.weight * Tr.Weight();
         vol    += ip.weight * Tr.Weight();
      }
   }

#ifdef MFEM_USE_MPI
   auto gp = dynamic_cast<const ParGridFunction *>(&g);
   if (gp)
   {
      MPI_Comm comm = gp->ParFESpace()->GetComm();
      MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, comm);
      MPI_Allreduce(MPI_IN_PLACE, &vol,  1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, comm);
   }
#endif
}

void ConstructSizeGF(GridFunction &size)
{
   const bool per = size.FESpace()->GetMesh()->GetNodalFESpace()->IsDGSpace();

   // Indicator for small (value -> 1) or big (value -> 0) elements.
   if (per)
   {
      FunctionCoefficient size_ind_coeff(size_indicator_periodic);
      size.ProjectCoefficient(size_ind_coeff);
   }
   else
   {
      FunctionCoefficient size_ind_coeff(size_indicator);
      size.ProjectCoefficient(size_ind_coeff);
   }

   // Determine small/big target sizes based on the total number of
   // elements and the volume occupied by small elements.
   real_t volume_ind, volume;
   calc_mass_volume(size, volume_ind, volume);
   Mesh &mesh = *size.FESpace()->GetMesh();
   int NE = mesh.GetNE();
#ifdef MFEM_USE_MPI
   auto size_p = dynamic_cast<const ParGridFunction *>(&size);
   if (size_p) { NE = size_p->ParFESpace()->GetParMesh()->GetGlobalNE(); }
#endif
   NCMesh *ncmesh = mesh.ncmesh;
   // For parallel NC meshes, all tasks have all root elements.
   NE = (ncmesh) ? ncmesh->GetNumRootElements() : NE;
   // Keep the 2D linear-size ratio at 3 and make the 3D target moderately
   // stronger by increasing its linear-size ratio from 3 to 4.
   const real_t size_ratio = (mesh.Dimension() == 2) ? 9 : 64;
   const real_t small_el_size = volume_ind / NE +
                                (volume - volume_ind) / (size_ratio * NE);
   const real_t big_el_size   = size_ratio * small_el_size;
   for (int i = 0; i < size.Size(); i++)
   {
      size(i) = size(i) * small_el_size + (1.0 - size(i)) * big_el_size;
   }
}

real_t material_indicator_2d(const Vector &x)
{
   real_t xc = x(0)-0.5, yc = x(1)-0.5;
   real_t th = 22.5*M_PI/180.;
   real_t xn =  cos(th)*xc + sin(th)*yc;
   real_t yn = -sin(th)*xc + cos(th)*yc;
   real_t th2 = (th > 45.*M_PI/180) ? M_PI/2 - th : th;
   real_t stretch = 1/cos(th2);
   xc = xn/stretch; yc = yn/stretch;
   real_t tfac = 20;
   real_t s1 = 3;
   real_t s2 = 3;
   real_t wgt = std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) + 1);
   if (wgt > 1) { wgt = 1; }
   if (wgt < 0) { wgt = 0; }
   return wgt;
}

real_t discrete_ori_2d(const Vector &x)
{
   return M_PI * x(1) * (1.0 - x(1)) * cos(2 * M_PI * x(0));
}

real_t size_and_orientation_scale_2d(const Vector &x)
{
   // Concentrate target resolution smoothly near the center of the unit square.
   return 0.14 - 0.08 * sin(M_PI * x(0)) * sin(M_PI * x(1));
}

real_t discrete_size_2d(const Vector &x)
{
   const real_t scale = size_and_orientation_scale_2d(x);
   // Target 8 stores area, so a linear scale of 0.1 is stored as 0.1^2 = 0.01.
   return scale * scale;
}

void discrete_aspr_3d(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v.SetSize(dim);
   real_t l1, l2, l3;
   l1 = 1.;
   l2 = 1. + 5*x(1);
   l3 = 1. + 10*x(2);
   v[0] = l1/pow(l2*l3,0.5);
   v[1] = l2/pow(l1*l3,0.5);
   v[2] = l3/pow(l2*l1,0.5);
}

class HessianCoefficient : public TMOPMatrixCoefficient
{
private:
   int metric;

public:
   HessianCoefficient(int dim, int metric_id)
      : TMOPMatrixCoefficient(dim), metric(metric_id) { }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector pos(3);
      T.Transform(ip, pos);
      if (metric != 36 && metric != 85)
      {
         const real_t xc = pos(0) - 0.5, yc = pos(1) - 0.5;
         const real_t r = sqrt(xc*xc + yc*yc);
         real_t r1 = 0.15; real_t r2 = 0.35; real_t sf=30.0;
         const real_t eps = 0.5;

         const real_t tan1 = std::tanh(sf*(r-r1)),
                      tan2 = std::tanh(sf*(r-r2));

         K(0, 0) = eps + 1.0 * (tan1 - tan2);
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
      }
      else if (metric == 36) // Size + Alignment
      {
         const real_t xc = pos(0), yc = pos(1);
         real_t theta = M_PI * yc * (1.0 - yc) * cos(2 * M_PI * xc);
         const real_t alpha_bar = size_and_orientation_scale_2d(pos);

         K(0, 0) =  cos(theta);
         K(1, 0) =  sin(theta);
         K(0, 1) = -sin(theta);
         K(1, 1) =  cos(theta);

         K *= alpha_bar;
      }
      else if (metric == 85) // Shape + Alignment
      {
         Vector x = pos;
         real_t xc = x(0)-0.5, yc = x(1)-0.5;
         real_t th = 22.5*M_PI/180.;
         real_t xn =  cos(th)*xc + sin(th)*yc;
         real_t yn = -sin(th)*xc + cos(th)*yc;
         xc = xn; yc=yn;

         real_t tfac = 20;
         real_t s1 = 3;
         real_t s2 = 2;
         real_t wgt = std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) + 1)
                      - std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) - 1);
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         xc = pos(0), yc = pos(1);
         real_t theta = M_PI * (yc) * (1.0 - yc) * cos(2 * M_PI * xc);

         K(0, 0) =  cos(theta);
         K(1, 0) =  sin(theta);
         K(0, 1) = -sin(theta);
         K(1, 1) =  cos(theta);

         real_t asp_ratio_tar = 0.1 + 1*(1-wgt)*(1-wgt);

         K(0, 0) *=  1/pow(asp_ratio_tar,0.5);
         K(1, 0) *=  1/pow(asp_ratio_tar,0.5);
         K(0, 1) *=  pow(asp_ratio_tar,0.5);
         K(1, 1) *=  pow(asp_ratio_tar,0.5);
      }
   }

   void EvalGrad(DenseMatrix &K, ElementTransformation &T,
                 const IntegrationPoint &ip, int comp) override
   {
      Vector pos(3);
      T.Transform(ip, pos);
      K = 0.;
      if (metric != 36 && metric != 85)
      {
         const real_t xc = pos(0) - 0.5, yc = pos(1) - 0.5;
         const real_t r = sqrt(xc*xc + yc*yc);
         real_t r1 = 0.15; real_t r2 = 0.35; real_t sf=30.0;

         const real_t tan1 = std::tanh(sf*(r-r1)),
                      tan2 = std::tanh(sf*(r-r2));
         real_t tan1d = 0., tan2d = 0.;
         if (r > 0.001)
         {
            tan1d = (1.-tan1*tan1)*(sf)/r,
            tan2d = (1.-tan2*tan2)*(sf)/r;
         }

         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         if (comp == 0) { K(0, 0) = tan1d*xc - tan2d*xc; }
         else if (comp == 1) { K(0, 0) = tan1d*yc - tan2d*yc; }
      }
      else if (metric == 36)
      {
         const real_t xc = pos(0), yc = pos(1);
         const real_t theta = M_PI * yc * (1.0 - yc) * cos(2 * M_PI * xc);
         const real_t alpha_bar = size_and_orientation_scale_2d(pos);
         const real_t dalpha =
            (comp == 0) ?
            -0.08 * M_PI * cos(M_PI * xc) * sin(M_PI * yc) :
            -0.08 * M_PI * sin(M_PI * xc) * cos(M_PI * yc);
         const real_t dtheta =
            (comp == 0) ?
            -2 * M_PI * M_PI * yc * (1.0 - yc) * sin(2 * M_PI * xc) :
            M_PI * (1.0 - 2.0 * yc) * cos(2 * M_PI * xc);
         const real_t c = cos(theta), s = sin(theta);

         K(0, 0) =  dalpha * c - alpha_bar * s * dtheta;
         K(1, 0) =  dalpha * s + alpha_bar * c * dtheta;
         K(0, 1) = -dalpha * s - alpha_bar * c * dtheta;
         K(1, 1) =  dalpha * c - alpha_bar * s * dtheta;
      }
      else if (metric == 85)
      {
         const real_t x = pos(0), y = pos(1);
         const real_t th = 22.5 * M_PI / 180.0;
         const real_t ct = cos(th), st = sin(th);
         const real_t xc = x - 0.5, yc = y - 0.5;
         const real_t xn = ct * xc + st * yc;
         const real_t yn = -st * xc + ct * yc;
         const real_t dxn = (comp == 0) ? ct : st;
         const real_t dyn = (comp == 0) ? -st : ct;
         const real_t tfac = 20.0, s1 = 3.0, s2 = 2.0;
         const real_t u = tfac * yn + s2 * sin(s1 * M_PI * xn);
         const real_t du = tfac * dyn +
                           s2 * s1 * M_PI * cos(s1 * M_PI * xn) * dxn;
         const real_t tanh_plus = std::tanh(u + 1.0);
         const real_t tanh_minus = std::tanh(u - 1.0);
         const real_t raw_wgt = tanh_plus - tanh_minus;
         const real_t wgt = std::max(0.0, std::min(1.0, raw_wgt));
         const real_t dwgt = (raw_wgt > 0.0 && raw_wgt < 1.0) ?
            ((1.0 - tanh_plus * tanh_plus) -
             (1.0 - tanh_minus * tanh_minus)) * du : 0.0;

         const real_t aspect = 0.1 + (1.0 - wgt) * (1.0 - wgt);
         const real_t daspect = -2.0 * (1.0 - wgt) * dwgt;
         const real_t sqrt_aspect = sqrt(aspect);
         const real_t inv_sqrt_aspect = 1.0 / sqrt_aspect;
         const real_t dsqrt_aspect = 0.5 * daspect * inv_sqrt_aspect;
         const real_t dinv_sqrt_aspect =
            -0.5 * daspect * inv_sqrt_aspect / aspect;

         const real_t theta = M_PI * y * (1.0 - y) * cos(2 * M_PI * x);
         const real_t dtheta =
            (comp == 0) ?
            -2 * M_PI * M_PI * y * (1.0 - y) * sin(2 * M_PI * x) :
            M_PI * (1.0 - 2.0 * y) * cos(2 * M_PI * x);
         const real_t c = cos(theta), s = sin(theta);
         const real_t dc = -s * dtheta, ds = c * dtheta;

         K(0, 0) = dc * inv_sqrt_aspect + c * dinv_sqrt_aspect;
         K(1, 0) = ds * inv_sqrt_aspect + s * dinv_sqrt_aspect;
         K(0, 1) = -ds * sqrt_aspect - s * dsqrt_aspect;
         K(1, 1) = dc * sqrt_aspect + c * dsqrt_aspect;
      }
   }
};

class HRHessianCoefficient : public TMOPMatrixCoefficient
{
private:
   int dim;
   // 0 - size target in an annular region,
   // 1 - size+aspect-ratio in an annular region,
   // 2 - size+aspect-ratio target for a rotate sine wave.
   int hr_target_type;

public:
   HRHessianCoefficient(int dim_, int hr_target_type_ = 0)
      : TMOPMatrixCoefficient(dim_), dim(dim_),
        hr_target_type(hr_target_type_) { }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector pos(3);
      T.Transform(ip, pos);
      if (hr_target_type == 0) // size only circle
      {
         real_t small = 0.001, big = 0.01;
         if (dim == 3) { small = 0.005, big = 0.1; }
         const real_t xc = pos(0) - 0.5, yc = pos(1) - 0.5;
         real_t r;
         if (dim == 2)
         {
            r = sqrt(xc*xc + yc*yc);
         }
         else
         {
            const real_t zc = pos(2) - 0.5;
            r = sqrt(xc*xc + yc*yc + zc*zc);
         }
         const real_t r1 = 0.15, r2 = 0.35;
         const real_t sf = (dim == 3) ? 10.0 : 30.0;

         const real_t tan1 = std::tanh(sf*(r-r1)),
                      tan2 = std::tanh(sf*(r-r2));

         real_t ind = tan1 - tan2;
         if (dim == 3)
         {
            ind *= 0.5 / std::tanh(0.5 * sf * (r2 - r1));
         }
         else
         {
            if (ind > 1.0) { ind = 1.0; }
            if (ind < 0.0) { ind = 0.0; }
         }
         real_t val = ind * small + (1.0 - ind) * big;
         // val is det(W), so the isotropic length scale is val^(1/dim).
         const real_t scale = std::pow(val, 1.0 / dim);
         K = 0.0;
         K(0, 0) = 1.0;
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
         K(0, 0) *= scale;
         K(1, 1) *= scale;
         if (dim == 3) { K(2, 2) = scale; }
      }
      else if (hr_target_type == 1) // circle with size and AR
      {
         const real_t small = 0.001, big = 0.01;
         const real_t xc = pos(0)-0.5, yc = pos(1)-0.5;
         const real_t rv = xc*xc + yc*yc;
         real_t r = 0;
         if (rv>0.) {r = sqrt(rv);}

         real_t r1 = 0.2; real_t r2 = 0.3; real_t sf=30.0;
         const real_t szfac = 1;
         const real_t asfac = 4;
         const real_t eps2 = szfac/asfac;
         const real_t eps1 = szfac;

         real_t tan1 = std::tanh(sf*(r-r1)+1),
                tan2 = std::tanh(sf*(r-r2)-1);
         real_t wgt = 0.5*(tan1-tan2);

         tan1 = std::tanh(sf*(r-r1)),
         tan2 = std::tanh(sf*(r-r2));

         real_t ind = (tan1 - tan2);
         if (ind > 1.0) {ind = 1.;}
         if (ind < 0.0) {ind = 0.;}
         real_t szval = ind * small + (1.0 - ind) * big;

         real_t th = std::atan2(yc,xc)*180./M_PI;
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         real_t maxval = eps2 + eps1*(1-wgt)*(1-wgt);
         real_t minval = eps1;
         real_t avgval = 0.5*(maxval+minval);
         real_t ampval = 0.5*(maxval-minval);
         real_t val1 = avgval + ampval*sin(2.*th*M_PI/180.+90*M_PI/180.);
         real_t val2 = avgval + ampval*sin(2.*th*M_PI/180.-90*M_PI/180.);

         K(0,1) = 0.0;
         K(1,0) = 0.0;
         K(0,0) = val1;
         K(1,1) = val2;

         K(0,0) *= pow(szval,0.5);
         K(1,1) *= pow(szval,0.5);
      }
      else if (hr_target_type == 2) // sharp rotated sine wave
      {
         real_t xc = pos(0)-0.5, yc = pos(1)-0.5;
         real_t th = 15.5*M_PI/180.;
         real_t xn =  cos(th)*xc + sin(th)*yc;
         real_t yn = -sin(th)*xc + cos(th)*yc;
         real_t th2 = (th > 45.*M_PI/180) ? M_PI/2 - th : th;
         real_t stretch = 1/cos(th2);
         xc = xn/stretch;
         yc = yn;
         real_t tfac = 20;
         real_t s1 = 3;
         real_t s2 = 2;
         real_t yl1 = -0.025;
         real_t yl2 =  0.025;
         real_t wgt = std::tanh((tfac*(yc-yl1) + s2*std::sin(s1*M_PI*xc)) + 1) -
                      std::tanh((tfac*(yc-yl2) + s2*std::sin(s1*M_PI*xc)) - 1);
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         const real_t eps2 = 25;
         const real_t eps1 = 1;
         K(1,1) = eps1/eps2 + eps1*(1-wgt)*(1-wgt);
         K(0,0) = eps1;
         K(0,1) = 0.0;
         K(1,0) = 0.0;
      }
      else { MFEM_ABORT("Unsupported option / wrong input."); }
   }

   void EvalGrad(DenseMatrix &K, ElementTransformation &T,
                 const IntegrationPoint &ip, int comp) override
   {
      K = 0.;
   }
};

// Additional IntegrationRules that can be used with the --quad-type option.
IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);

// Defined with respect to the icf mesh.
real_t weight_fun(const Vector &x)
{
   const real_t r = sqrt(x(0)*x(0) + x(1)*x(1) + 1e-12);
   const real_t den = 0.002;
   real_t l2 = 0.2 + 0.5*std::tanh((r-0.16)/den) - 0.5*std::tanh((r-0.17)/den)
               + 0.5*std::tanh((r-0.23)/den) - 0.5*std::tanh((r-0.24)/den);
   return l2;
}

// Used for the adaptive limiting examples.
real_t adapt_lim_fun(const Vector &x)
{
   // Bump between these rad values, with sf sharpness.
   real_t r1 = 0.25, r2 = 0.35, sf = 30.0, r;
   if (x.Size() == 2)
   {
      const real_t xc = x(0) - 0.1, yc = x(1) - 0.2;
      r = sqrt(xc*xc + yc*yc);
   }
   else
   {
      const real_t xc = x(0) - 0.9, yc = x(1) - 0.2, zc = x(2) - 0.0;
      r = sqrt(xc*xc + yc*yc + zc*zc);
   }

   real_t val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));
   val = std::max((real_t) 0.,val);
   val = std::min((real_t) 1.,val);
   return val;
}

// Second field for adaptive limiting examples: uses different xc, yc, zc.
real_t adapt_lim_fun2(const Vector &x)
{
   // Bump between these rad values, with sf sharpness.
   real_t r1 = 0.25, r2 = 0.35, sf = 30.0, r;
   if (x.Size() == 2)
   {
      const real_t xc = x(0) - 0.9, yc = x(1) - 0.2;
      r = sqrt(xc*xc + yc*yc);
   }
   else
   {
      const real_t xc = x(0) - 0.1, yc = x(1) - 0.2, zc = x(2) - 0.0;
      r = sqrt(xc*xc + yc*yc + zc*zc);
   }

   real_t val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));
   val = std::max((real_t) 0.,val);
   val = std::min((real_t) 1.,val);
   return val;
}

// Used for exact surface alignment
real_t surface_level_set(const Vector &x)
{
   const int type = 1;

   const int dim = x.Size();
   if (type == 0)
   {
      const real_t sine = 0.25 * std::sin(4 * M_PI * x(0));
      return (x(1) >= sine + 0.5) ? 1.0 : -1.0;
   }
   else
   {
      if (dim == 2)
      {
         const real_t xc = x(0) - 0.5, yc = x(1) - 0.5;
         const real_t r = sqrt(xc*xc + yc*yc);
         return r-0.3;
      }
      else
      {
         const real_t xc = x(0) - 0.5, yc = x(1) - 0.5, zc = x(2) - 0.5;
         const real_t r = sqrt(xc*xc + yc*yc + zc*zc);
         return r-0.3;
      }
   }
}

int material_id(int el_id, const GridFunction &g)
{
   const FiniteElementSpace *fes = g.FESpace();
   const FiniteElement *fe = fes->GetFE(el_id);
   Vector g_vals;
   const IntegrationRule &ir =
      IntRules.Get(fe->GetGeomType(), fes->GetOrder(el_id) + 2);

   real_t integral = 0.0;
   g.GetValues(el_id, ir, g_vals);
   ElementTransformation *Tr = fes->GetMesh()->GetElementTransformation(el_id);
   int approach = 1;
   if (approach == 0)   // integral based
   {
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);
         integral += ip.weight * g_vals(q) * Tr->Weight();
      }
      return (integral > 0.0) ? 1.0 : 0.0;
   }
   else if (approach == 1)   // minimum value based
   {
      real_t minval = g_vals.Min();
      return minval > 0.0 ? 1.0 : 0.0;
   }
   return 0.0;
}

void DiffuseField(GridFunction &field, int smooth_steps)
{
   // Setup the Laplacian operator
   BilinearForm *Lap = new BilinearForm(field.FESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();

   // Setup the smoothing operator
   DSmoother *S = new DSmoother(0,1.0,smooth_steps);
   S->iterative_mode = true;
   S->SetOperator(Lap->SpMat());

   Vector tmp(field.Size());
   tmp = 0.0;
   S->Mult(tmp, field);

   delete S;
   delete Lap;
}

#ifdef MFEM_USE_MPI
void DiffuseField(ParGridFunction &field, int smooth_steps)
{
   // Setup the Laplacian operator
   ParBilinearForm *Lap = new ParBilinearForm(field.ParFESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();
   HypreParMatrix *A = Lap->ParallelAssemble();

   HypreSmoother *S = new HypreSmoother(*A,0,smooth_steps);
   S->iterative_mode = true;

   Vector tmp(A->Width());
   field.SetTrueVector();
   Vector fieldtrue = field.GetTrueVector();
   tmp = 0.0;
   S->Mult(tmp, fieldtrue);

   field.SetFromTrueDofs(fieldtrue);

   delete S;
   delete A;
   delete Lap;
}
#endif
