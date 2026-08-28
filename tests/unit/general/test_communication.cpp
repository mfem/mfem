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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

#ifdef MFEM_USE_MPI

TEST_CASE("DeviceGroupCommunicator", "[Parallel][GPU]")
{
   const int nx = 6, ny = 6;
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   pmesh.UniformRefinement();

   const int p = 2;
   H1_FECollection fec(p, pmesh.Dimension());
   ParFiniteElementSpace pfes(&pmesh, &fec);
   GroupCommunicator &gc = pfes.GroupComm();

   ParGridFunction x(&pfes), y_h(&pfes), y_d(&pfes);

   x = Mpi::WorldRank();

   SECTION("Bcast")
   {
      y_h = x;
      gc.Bcast(y_h.HostReadWrite());
      y_d = x;
      if (!Device::Allows(Backend::DEVICE_MASK))
      {
         // When not running on device, using 'gc' will not use the
         // DeviceGroupCommunicator, so to test it, we explicitly create it.
         gc.GetDeviceComm().BcastBeginLDofs(*y_d.GetArrayView());
         gc.GetDeviceComm().BcastEndLDofs(*y_d.GetArrayView());
      }
      else
      {
         gc.Bcast(*y_d.GetArrayView());
         CHECK(y_d.GetMemory().DeviceIsValid());
      }
      y_d -= y_h;
      const real_t error = ParNormlp(y_d, infinity(), pmesh.GetComm());
      CHECK(error == 0_r); // numbers should match exactly!
   }

   auto reduce_check = [&](void (*Op)(GroupCommunicator::OpData<real_t>),
                           DeviceGroupCommunicator::Op op) -> void
   {
      y_h = x;
      gc.Reduce(y_h.HostReadWrite(), Op);
      y_d = x;
      if (!Device::Allows(Backend::DEVICE_MASK))
      {
         // When not running on device, using 'gc' will not use the
         // DeviceGroupCommunicator, so to test it, we explicitly create it.
         gc.GetDeviceComm().ReduceBeginLDofs(*y_d.GetArrayView());
         gc.GetDeviceComm().ReduceEndLDofs(*y_d.GetArrayView(), op);
      }
      else
      {
         gc.Reduce(*y_d.GetArrayView(), Op);
         CHECK(y_d.GetMemory().DeviceIsValid());
      }
      y_d -= y_h;
      const real_t error = ParNormlp(y_d, infinity(), pmesh.GetComm());
      CHECK(error == 0_r); // numbers should match exactly! (reducing integers)
   };

   SECTION("Reduce Sum")
   {
      reduce_check(GroupCommunicator::Sum, DeviceGroupCommunicator::Op::Sum);
   }

   SECTION("Reduce Min")
   {
      reduce_check(GroupCommunicator::Min, DeviceGroupCommunicator::Op::Min);
   }

   SECTION("Reduce Max")
   {
      reduce_check(GroupCommunicator::Max, DeviceGroupCommunicator::Op::Max);
   }
}

#endif
