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

#include "stab_tau.hpp"

using namespace mfem;

real_t FFH92Tau::GetElementSize(ElementTransformation &T)
{
   const DenseMatrix &dxdxi = T.Jacobian();
   row.SetSize(dim);
   h.SetSize(dim);

   for (int i = 0; i < dim; i++)
   {
      dxdxi.GetRow(i, row);
      h[i] = row.Norml2();
   }
   switch (dim)
   {
      case 1:
         return h[0];
      case 2:
         return h[0]*h[1]*sqrt(2.0/(h[0]*h[0] + h[1]*h[1]));
      case 3:
         return h[0]*h[1]*h[2]*(3.0/(h[0]*h[0] + h[1]*h[1] + h[2]*h[2]));
   }
   mfem_error("Wrong dim!");
   return -1.0;
}

real_t FFH92Tau::GetInverseEstimate(ElementTransformation &T,
                                    const IntegrationPoint &ip, real_t scale)
{
   if (Ci>0.0)
   {
      return 1.0/Ci;
   }
   else
   {
      return 1.0/(invEst_cf->Eval(T,ip)*scale);
   }
}

real_t FFH92Tau::Eval(ElementTransformation &T,
                      const IntegrationPoint &ip)
{
   real_t k = kappa->Eval(T, ip);
   adv->Eval(a, T, ip);
   real_t hk = GetElementSize(T);
   real_t ci = GetInverseEstimate(T, ip, hk*hk);
   real_t mk = std::min(1.0/3.0, 2*ci);
   real_t ap = a.Normlp(p);
   // Prevent division by zero
   ap = std::max(ap,std::numeric_limits<real_t>::min());
   real_t pe = mk*ap*hk/(k_fac*k); // k_fac = 2 for CD and k_fac = 4 for NS
   real_t xi = std::min(pe,1.0);
   real_t tau = hk*xi/(2*ap);

   if (print)
   {
      std::cout<<"\n==========================\n";
      std::cout<<"  kappa = "<<k  <<std::endl;
      std::cout<<"  adv   = "; a.Print(std::cout);
      std::cout<<"  h     = "<<hk <<std::endl;
      std::cout<<"  Ci    = "<<ci<<" "
               <<( (Ci<0) ? "(Computed)" :"(Specified)")<<std::endl;
      std::cout<<"  1/Ci  = "<<1.0/ci<<std::endl;
      std::cout<<"  mk    = "<<mk <<std::endl;
      std::cout<<"  |a|_p = "<<ap <<std::endl;
      std::cout<<"  Pe    = "<<pe <<std::endl;
      std::cout<<"  xi    = "<<xi <<std::endl;
      std::cout<<"  tau   = "<<tau<<std::endl;
      std::cout<<"  tau   = "<<mk*hk*hk/(8*k)<<std::endl;
      std::cout<<"==========================\n\n";
      print = false;
   }

   return tau;
}

real_t FF91Delta::Eval(ElementTransformation &T,
                      const IntegrationPoint &ip)
{
   real_t k = kappa->Eval(T, ip);
   adv->Eval(a, T, ip);
   real_t hk = GetElementSize(T);
   real_t ci = GetInverseEstimate(T, ip, hk*hk);
   real_t mk = std::min(1.0/3.0, 2*ci);
   real_t ap = a.Normlp(p); // Preventing division by zero not necessary
   real_t pe = mk*ap*hk/(k_fac*k);
   real_t xi = std::min(pe,1.0);
   real_t delta = lambda*ap*hk*xi;

   if (print)
   {
      std::cout<<"\n==========================\n";
      std::cout<<"  kappa  = "<<k  <<std::endl;
      std::cout<<"  adv    = "; a.Print(std::cout);
      std::cout<<"  h      = "<<hk <<std::endl;
      std::cout<<"  Ci     = "<<ci<<" "
               <<( (Ci<0) ? "(Computed)" :"(Specified)")<<std::endl;
      std::cout<<"  1/Ci   = "<<1.0/ci<<std::endl;
      std::cout<<"  mk     = "<<mk <<std::endl;
      std::cout<<"  |a|_p  = "<<ap <<std::endl;
      std::cout<<"  Pe     = "<<pe <<std::endl;
      std::cout<<"  xi     = "<<xi <<std::endl;
      std::cout<<"  lambda = "<<lambda <<std::endl;
      std::cout<<"  delta  = "<<delta<<std::endl;
      std::cout<<"==========================\n\n";
      print = false;
   }

   return delta;
}

