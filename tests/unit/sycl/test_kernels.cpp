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

#include "general/forall.hpp"

void sycl_kernels()
{
   const int NE = 2, TX = 3, TY = 4, BZ = 2;
   int TZ = 1; // starting with 2D tests

   Vector d(NE*TX*TY*TZ);
   d.UseDevice(true);

   auto hD = Reshape(d.HostReadWrite(),TX,TY,TZ,NE);

   auto host_forall = [&](auto q_func)
   {
      for (int e = 0; e < NE; ++e)
         for (int x = 0; x < TX; ++x)
            for (int y = 0; y < TY; ++y)
               for (int z = 0; z < TZ; ++z) { q_func(x, y, z, e); }
   };
   auto ini = [&](int x, int y, int z, int e) { hD(x,y,z,e) = 0.0; };
   auto req_ini = [&](int x, int y, int z, int e) { REQUIRE(hD(x,y,z,e) == 0.0); };
   auto req_one = [&](int x, int y, int z, int e) { REQUIRE(hD(x,y,z,e) == 1.0); };
   auto req_one_ini = [&](int x, int y, int z, int e) {req_one(x,y,z,e), ini(x,y,z,e); };

   host_forall(ini), host_forall(req_ini);

   // forall_2D
   auto dD = Reshape(d.ReadWrite(),TX,TY,TZ,NE);
   mfem::forall_2D(NE, TX, TY, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(x,x,TX)
      {
         MFEM_FOREACH_THREAD(y,y,TY)
         {
            dD(x,y,0,e) += 1.0;
         }
      }
   });
   d.HostReadWrite();
   host_forall(req_one_ini);

   // forall_2D_batch
   d.ReadWrite();
   mfem::forall_2D_batch(NE, TX, TY, BZ, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(x,x,TX)
      {
         MFEM_FOREACH_THREAD(y,y,TY)
         {
            dD(x,y,0,e) += 1.0;
         }
      }
   });
   d.HostReadWrite(), host_forall(req_one);

   TZ = 3; // switching to 3D
   d.SetSize(NE*TX*TY*TZ);
   hD = Reshape(d.HostReadWrite(),TX,TY,TZ,NE);
   host_forall(ini), host_forall(req_ini);

   // forall_3D
   dD = Reshape(d.ReadWrite(),TX,TY,TZ,NE);
   mfem::forall_3D(NE, TX, TY, TZ, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(x,x,TX)
      {
         MFEM_FOREACH_THREAD(y,y,TY)
         {
            MFEM_FOREACH_THREAD(z,z,TZ)
            {
               dD(x,y,z,e) += 1.0;
            }
         }
      }
   });
   d.HostReadWrite(), host_forall(req_one_ini);

   // forall_3D_grid
   d.ReadWrite();
   mfem::forall_3D_grid(NE, TX, TY, TZ, 5, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(x,x,TX)
      {
         MFEM_FOREACH_THREAD(y,y,TY)
         {
            MFEM_FOREACH_THREAD(z,z,TZ)
            {
               dD(x,y,z,e) += 1.0;
            }
         }
      }
   });
   d.HostReadWrite(), host_forall(req_one);
}

#endif // MFEM_USE_SYCL
