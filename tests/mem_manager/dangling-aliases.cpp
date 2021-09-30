// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include <mfem.hpp>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   MPI_Session mpi;

   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (mpi.Root()) { args.PrintOptions(cout); }

   Device device(device_config);
   if (mpi.Root()) { device.Print(); }

   //-----------------------------------------------------------
   int width = 1000;
   Vector v_r, v_i; // this initialization causes failure below
   {
      // Vector v_r, v_i; // this initialization is fine
      Vector v(width);
      v.UseDevice(true);
      v = 0.0;
      cout << mpi.WorldRank() << ": v.HostRead() = " << v.HostRead() << endl;

      v_r.MakeRef(v, 0, width/2);
      v_i.MakeRef(v, width/2, width/2);
   }
   Vector w(width);
   w.UseDevice(true);
   w = 0.0;
   cout << mpi.WorldRank() << ": w.HostRead() = " << w.HostRead() << endl;
   Vector w_r, w_i;

   MPI_Barrier(MPI_COMM_WORLD);
   cout << mpi.WorldRank() << ": == # START # ==" << endl;
   MPI_Barrier(MPI_COMM_WORLD);
   w_r.MakeRef(w, 0, width/2);        // fails
   w_i.MakeRef(w, width/2, width/2);  // fails
   // with one of the following two messages:
   // 1) Verification failed: (h_mt == maps->aliases.at(h_ptr).mem->h_mt) is false:
   //    -->
   //    ... in function: static void mfem::MemoryManager::CheckHostMemoryType_(mfem::MemoryType, void *)
   //    ... in file: general/mem_manager.cpp:1377
   // 2) alias already exists with different base/offset!
   MPI_Barrier(MPI_COMM_WORLD);
   cout << mpi.WorldRank() << ": == # END # ==" << endl;

   return 0;
}
