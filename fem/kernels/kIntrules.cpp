// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../general/okina.hpp"
#include "../../fem/fem.hpp"
#include "kIntrules.hpp"

// *****************************************************************************
MFEM_NAMESPACE

// *****************************************************************************
void kIPPts(const IntegrationPoint *ip, const size_t N, double *pts){
   GET_CUDA;
   GET_ADRS(pts);
   GET_ADRS_T(ip,IntegrationPoint);
   if (cuda){ assert(false); }
   forall(i, N, d_pts[i] = d_ip[i].x; );
}

// *****************************************************************************
double kIPGetX(const IntegrationPoint *ip, const size_t offset){
   GET_ADRS_T(ip,IntegrationPoint);
   return d_ip[offset].x;
}

// *****************************************************************************
double kIPGetY(const IntegrationPoint *ip, const size_t offset){
   GET_ADRS_T(ip,IntegrationPoint);
   return d_ip[offset].y;
}

// *****************************************************************************
double kIPGetZ(const IntegrationPoint *ip, const size_t offset){
   GET_ADRS_T(ip,IntegrationPoint);
   return d_ip[offset].z;
}

// *****************************************************************************
void kIPSetX(const IntegrationPoint *ip, const double x, const size_t offset){
   GET_ADRS_T(ip,IntegrationPoint);
   forall(i, 1, d_ip[offset].x = x; );
}

// *****************************************************************************

void kIPSetX(const IntegrationPoint *ip, const double *x, const int i, const size_t offset){
   GET_CONST_ADRS(x);
   GET_ADRS_T(ip, IntegrationPoint);
   forall(i, 1, d_ip[offset].x = d_x[i]; );
}

// *****************************************************************************
void kIPSetY(const IntegrationPoint *ip, const double y, const size_t offset){
   GET_ADRS_T(ip,IntegrationPoint);
   forall(i, 1, d_ip[offset].y = y; );
}

// *****************************************************************************
void kIPSetZ(const IntegrationPoint *ip, const double z, const size_t offset){
   GET_ADRS_T(ip,IntegrationPoint);
   forall(i, 1, d_ip[offset].z = z; );
}

// *****************************************************************************
void kIPSetXY(const IntegrationPoint *ip,
              const double *x, const int i,
              const double *y, const int j, const size_t offset){
   GET_ADRS_T(ip,IntegrationPoint);
   GET_CONST_ADRS(x);
   GET_CONST_ADRS(y);
   forall(k, 1, {
         d_ip[offset].x = d_x[i];
         d_ip[offset].y = d_y[j];
      });
}

// *****************************************************************************
void kIPSetIPXY(const int nx,
                const IntegrationPoint *ip,
                const IntegrationPoint *ipx,
                const IntegrationPoint *ipy,
                const size_t j, const size_t i){
   GET_ADRS_T(ip,IntegrationPoint);
   GET_CONST_ADRS_T(ipx,IntegrationPoint);
   GET_CONST_ADRS_T(ipy,IntegrationPoint);
   forall(k, 1, {
         d_ip[j*nx+i].x = d_ipx[i].x;
         d_ip[j*nx+i].y = d_ipy[j].x;
         d_ip[j*nx+i].weight = d_ipx[i].weight * d_ipy[j].weight;
      });
}

// *****************************************************************************
void kIPSetW(const IntegrationPoint *ip, const double w, const size_t offset){
   GET_ADRS_T(ip,IntegrationPoint);
   forall(i, 1, d_ip[offset].weight = w; );
}

// *****************************************************************************
void kIPSet1W(const IntegrationPoint *ip,
              const double x, const double w, const size_t offset){
   GET_ADRS_T(ip,IntegrationPoint);
   forall(i, 1, {
         d_ip[offset].x = x;
         d_ip[offset].weight = w;
      });
}

// *****************************************************************************
void kIntRulesInit(const size_t N, IntegrationPoint *ip){
   GET_ADRS_T(ip,IntegrationPoint);
   forall(i, N, {
         d_ip[i].x = d_ip[i].y = d_ip[i].z = d_ip[i].weight = 0.0; 
      });
}

// *****************************************************************************
void kIntRulesPointIni(IntegrationPoint *ip){
   GET_ADRS_T(ip,IntegrationPoint);
   forall(i, 1, d_ip[0].x = 0.0; );
}

// *****************************************************************************
void kIntRulesLinear1DIni(IntegrationPoint *ip){
   GET_ADRS_T(ip,IntegrationPoint);
   forall(i, 1, {
         d_ip[0].x = 0.0;
         d_ip[1].x = 1.0;
      });
}

// *****************************************************************************
void kIntRulesLinear2DIni(IntegrationPoint *ip){
   GET_ADRS_T(ip,IntegrationPoint);
   forall(i, 1, {
         d_ip[0].x = 0.0;
         d_ip[0].y = 0.0;
         d_ip[1].x = 1.0;
         d_ip[1].y = 0.0;
         d_ip[2].x = 0.0;
         d_ip[2].y = 1.0;
      });
}

// *****************************************************************************
void kIntRulesLinear3DIni(IntegrationPoint *ip){
   GET_ADRS_T(ip,IntegrationPoint);
   forall(i, 1, {
         d_ip[0].x = 0.0;
         d_ip[0].y = 0.0;
         d_ip[0].z = 0.0;
         d_ip[1].x = 1.0;
         d_ip[1].y = 0.0;
         d_ip[1].z = 0.0;
         d_ip[2].x = 0.0;
         d_ip[2].y = 1.0;
         d_ip[2].z = 0.0;
         d_ip[3].x = 0.0;
         d_ip[3].y = 0.0;
         d_ip[3].z = 1.0;
      });
}

// *****************************************************************************
void kIntRulesBiLinear2DIni(IntegrationPoint *ip){
   GET_ADRS_T(ip,IntegrationPoint);
   forall(i, 1, {
         d_ip[0].x = 0.0;
         d_ip[0].y = 0.0;
         d_ip[1].x = 1.0;
         d_ip[1].y = 0.0;
         d_ip[2].x = 1.0;
         d_ip[2].y = 1.0;
         d_ip[3].x = 0.0;
         d_ip[3].y = 1.0;
      });
}

// *****************************************************************************
void kIntRulesTriLinear3DIni(IntegrationPoint *ip){
   GET_ADRS_T(ip,IntegrationPoint);
   forall(i, 1, {
         d_ip[0].x = 0.0;
         d_ip[0].y = 0.0;
         d_ip[0].z = 0.0;

         d_ip[1].x = 1.0;
         d_ip[1].y = 0.0;
         d_ip[1].z = 0.0;

         d_ip[2].x = 1.0;
         d_ip[2].y = 1.0;
         d_ip[2].z = 0.0;

         d_ip[3].x = 0.0;
         d_ip[3].y = 1.0;
         d_ip[3].z = 0.0;

         d_ip[4].x = 0.0;
         d_ip[4].y = 0.0;
         d_ip[4].z = 1.0;

         d_ip[5].x = 1.0;
         d_ip[5].y = 0.0;
         d_ip[5].z = 1.0;

         d_ip[6].x = 1.0;
         d_ip[6].y = 1.0;
         d_ip[6].z = 1.0;

         d_ip[7].x = 0.0;
         d_ip[7].y = 1.0;
         d_ip[7].z = 1.0;
      });
}

// *****************************************************************************
void kCalcChebyshev(const int p, const double x, double *u){
   GET_ADRS(u);

   GET_CUDA;
   if (cuda){ assert(false); }
   
   // recursive definition, z in [-1,1]
   // T_0(z) = 1,  T_1(z) = z
   // T_{n+1}(z) = 2*z*T_n(z) - T_{n-1}(z)
   double z;
   d_u[0] = 1.;
   if (p == 0) { return; }
   d_u[1] = z = 2.*x - 1.;
   for (int n = 1; n < p; n++)
   {
      d_u[n+1] = 2*z*d_u[n] - d_u[n-1];
   }
}

// *****************************************************************************
void kCalcChebyshev(const int p, const double x, double *u, double *d){
   GET_ADRS(u);
   GET_ADRS(d);
      
   GET_CUDA;
   if (cuda){ assert(false); }
   
   double z;
   d_u[0] = 1.;
   d_d[0] = 0.;
   if (p == 0) { return; }
   d_u[1] = z = 2.*x - 1.;
   d_d[1] = 2.;
   for (int n = 1; n < p; n++)
   {
      d_u[n+1] = 2*z*d_u[n] - d_u[n-1];
      d_d[n+1] = (n + 1)*(z*d_d[n]/n + 2*d_u[n]);
   }
}

      
// *****************************************************************************
MFEM_NAMESPACE_END
