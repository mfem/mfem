// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
using namespace mfem;

#include "unit_tests.hpp"

#ifdef MFEM_USE_SYCL

#include <cassert>

#include "general/forall.hpp"

#include "general/debug.hpp"

void sycl_kernels()
{
   auto Q = Sycl::Queue();

   const int NE = 2;
   const int TX = 3, TY = 4, BZ = 2;
   int TZ = 1;

   Vector d(NE*TX*TY*TZ);
   auto hD_w = Reshape(d.HostWrite(),NE,TX,TY,TZ);
   auto init = [&](int e, int x, int y, int z) { hD_w(e, x, y, z) = 0.0; };
   auto host_forall = [&](auto &q_func)
   {
      for (int e = 0; e < NE; ++e)
         for (int x = 0; x < TX; ++x)
            for (int y = 0; y < TY; ++y)
               for (int z = 0; z < TZ; ++z) { q_func(e, x, y, z); }
   };

   host_forall(init);
   auto dD_w = Reshape(d.ReadWrite(),NE,TX,TY,TZ);
   mfem::forall_2D(NE, TX, TY, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(x,x,TX)
      {
         MFEM_FOREACH_THREAD(y,y,TY)
         {
            dD_w(e,x,y,0) += 1;
         }
      }
   });
   // read back on host for the requirements
   auto hD_r = Reshape(d.HostRead(),NE,TX,TY,TZ);
   auto req_d = [&](int e, int x, int y, int z)
   {
      const int d = hD_r(e,x,y,z);
      // dbg("e:%d thread:%d%d = %d",e,x,y,d);
      REQUIRE(d == 1);
   };
   host_forall(req_d);

   /*auto dbg_d = [&](int e, int x, int y, int z)
   {
      dbg("e:%d thread:%d%d = %d",e,x,y,z,hD_r(e,x,y,z));
   };
   host_forall(dbg_d);*/

   d.HostReadWrite();
   host_forall(init);
   //host_forall(dbg_d);
   d.ReadWrite();
   mfem::forall_2D_batch(NE, TX, TY, BZ, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(x,x,TX)
      {
         MFEM_FOREACH_THREAD(y,y,TY)
         {
            //std::cout << e << x << y << std::endl;
            dD_w(e,x,y,0) += 1;
         }
      }
   });
   Q.wait();
   d.HostRead();
   host_forall(req_d);
   //host_forall(dbg_d);

   TZ = 3;
   d.SetSize(NE*TX*TY*TZ);
   hD_w = Reshape(d.HostWrite(),NE,TX,TY,TZ);
   host_forall(init);
   dD_w = Reshape(d.ReadWrite(),NE,TX,TY,TZ);
   mfem::forall_3D(NE, TX, TY, TZ, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(x,x,TX)
      {
         MFEM_FOREACH_THREAD(y,y,TY)
         {
            MFEM_FOREACH_THREAD(z,z,TZ)
            {
               dD_w(e,x,y,z) += 1;
            }
         }
      }
   });
   hD_r = Reshape(d.HostRead(),NE,TX,TY,TZ);
   host_forall(req_d);

   d.HostReadWrite();
   host_forall(init);
   d.ReadWrite();
   mfem::forall_3D_grid(NE, TX, TY, TZ, 4, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(x,x,TX)
      {
         MFEM_FOREACH_THREAD(y,y,TY)
         {
            MFEM_FOREACH_THREAD(z,z,TZ)
            {
               dD_w(e,x,y,z) += 1;
            }
         }
      }
   });
   d.HostRead();
   host_forall(req_d);
}

#endif // MFEM_USE_SYCL
