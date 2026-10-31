// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_PETSC)

#include <petscvec.h>

namespace
{
struct PetscSession
{
   PetscSession() { MFEMInitializePetsc(); }
   ~PetscSession() { MFEMFinalizePetsc(); }
};

void EnsurePetscSession()
{
   static PetscSession session;
}

class IdentityGradientOperator : public IdentityOperator
{
public:
   IdentityGradientOperator() : IdentityOperator(1), _jac(1)
   {
      _jac.Add(0, 0, 1.0);
      _jac.Finalize();
   }

   Operator &GetGradient(const Vector &) const override
   {
      return const_cast<SparseMatrix &>(_jac);
   }

private:
   SparseMatrix _jac;
};
}

TEST_CASE("PetscNonlinearSolver accepts non-empty rhs", "[Parallel][PETSc]")
{
   EnsurePetscSession();

   IdentityGradientOperator oper;
   PetscNonlinearSolver solver(MPI_COMM_WORLD, "nl_");
   solver.SetRelTol(1.0e-12);
   solver.SetAbsTol(1.0e-12);
   solver.SetMaxIter(5);
   solver.SetPrintLevel(0);
   solver.SetJacobianType(Operator::PETSC_MATAIJ);
   solver.SetOperator(oper);

   Vector x(1);

   Vector empty_rhs;
   x = 0.0;
   solver.Mult(empty_rhs, x);
   REQUIRE(x(0) == MFEM_Approx(0.0));

   Vector nonempty_rhs(1);
   nonempty_rhs(0) = 2.5;
   x = 0.0;
   solver.Mult(nonempty_rhs, x);
   REQUIRE(x.Size() == 1);
   REQUIRE(x(0) == MFEM_Approx(nonempty_rhs(0)));
}

TEST_CASE("PETSc vector operations synchronize MFEM data",
          "[Parallel][PETSc][GPU]")
{
   EnsurePetscSession();
   PetscParVector x(MPI_COMM_WORLD, PetscInt(4 * Mpi::WorldSize()));
   auto values = x.Write(true);
   forall(4, [=] MFEM_HOST_DEVICE(int i) { values[i] = i + 1.0; });
   auto host = x.HostReadWrite();
   for (int i = 0; i < 4; ++i) { host[i] += 10.0; }
   REQUIRE(x.HostRead()[0] == MFEM_Approx(11.0));

   x *= 2.0;
   REQUIRE(x.HostRead()[0] == MFEM_Approx(22.0));
   x += 1.0;
   REQUIRE(x.HostRead()[0] == MFEM_Approx(23.0));

   Vector source(4);
   auto src = source.Write(true);
   forall(4, [=] MFEM_HOST_DEVICE(int i) { src[i] = i + 3.0; });
   PetscParVector y(MPI_COMM_WORLD, source, true);
   REQUIRE(y.HostRead()[0] == MFEM_Approx(3.0));
   x += y;
   REQUIRE(x.HostRead()[0] == MFEM_Approx(26.0));
   x -= y;
   REQUIRE(x.HostRead()[0] == MFEM_Approx(23.0));
   x = y;
   REQUIRE(x.HostRead()[0] == MFEM_Approx(3.0));

   x.HostReadWrite()[0] = 7.0;
   x.UpdateVecFromFlags();
   REQUIRE(VecScale(x, 2.0) == 0);
   x.UpdateMemoryFromVec();
   REQUIRE(x.HostRead()[0] == MFEM_Approx(14.0));
}

TEST_CASE("PETSc zero assignment overwrites MFEM writes",
          "[Parallel][PETSc][GPU]")
{
   EnsurePetscSession();
   PetscParVector x(MPI_COMM_WORLD, PetscInt(4 * Mpi::WorldSize()));
   x = 0.0;

   SECTION("Host writes")
   {
      x.HostReadWrite()[0] = 99.0;
   }
   SECTION("Device writes")
   {
      auto ptr = x.Write(true);
      forall(4, [=] MFEM_HOST_DEVICE(int i) { ptr[i] = 99.0; });
   }

   x = 0.0;
   const auto result = x.HostRead();
   for (int i = 0; i < 4; ++i) { REQUIRE(result[i] == MFEM_Approx(0.0)); }
}

#if PETSC_VERSION_GE(3,11,0)
TEST_CASE("PETSc read-locked vectors remain usable as sources",
          "[Parallel][PETSc][GPU]")
{
   EnsurePetscSession();
   PetscParVector source(MPI_COMM_WORLD, PetscInt(4 * Mpi::WorldSize()));
   PetscParVector destination(MPI_COMM_WORLD, PetscInt(4 * Mpi::WorldSize()));
   auto values = source.Write(true);
   forall(4, [=] MFEM_HOST_DEVICE(int i) { values[i] = i + 1.0; });
   source.UpdateVecFromFlags();

   REQUIRE(VecLockReadPush(source) == 0);
   // Reading through MFEM may fill a cache distinct from PETSc's buffer.
   REQUIRE(source.HostRead()[0] == MFEM_Approx(1.0));
   source.UpdateVecFromFlags();
   destination = source;
   REQUIRE(destination.HostRead()[0] == MFEM_Approx(1.0));
   destination += source;
   REQUIRE(destination.HostRead()[0] == MFEM_Approx(2.0));
   destination -= source;
   REQUIRE(destination.HostRead()[0] == MFEM_Approx(1.0));
   Vector *global = source.GlobalVector();
   REQUIRE(global->HostRead()[0] == MFEM_Approx(1.0));
   delete global;
   REQUIRE(VecLockReadPop(source) == 0);

   source.HostReadWrite()[0] = 10.0;
   source.UpdateVecFromFlags();
   destination = source;
   REQUIRE(destination.HostRead()[0] == MFEM_Approx(10.0));
}
#endif

#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
TEST_CASE("PETSc callbacks preserve one-sided placements",
          "[Parallel][PETSc][GPU]")
{
   if (!Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK)) { return; }
   EnsurePetscSession();

   class CopyOperator : public Operator
   {
   public:
      bool on_device = false;
      CopyOperator() : Operator(4) {}
      void Mult(const Vector &x, Vector &y) const override
      {
         const auto src = x.Read(on_device);
         auto dst = y.Write(on_device);
         forall_switch(on_device, 4, [=] MFEM_HOST_DEVICE(int i)
         { dst[i] = 2.0 * src[i]; });

         // Repeated access must preserve writes in an MFEM alternate buffer.
         const auto result = y.HostRead();
         const auto input = x.HostRead();
         for (int i = 0; i < 4; ++i)
         {
            REQUIRE(result[i] == MFEM_Approx(2.0 * input[i]));
         }
         auto host = y.HostReadWrite();
         for (int i = 0; i < 4; ++i) { host[i] += 1.0; }
         auto updated = y.ReadWrite(on_device);
         forall_switch(on_device, 4, [=] MFEM_HOST_DEVICE(int i)
         { updated[i] -= 1.0; });
      }
   } op;

   Vector x(4), y(4);
   auto input = x.HostWrite();
   for (int i = 0; i < 4; ++i) { input[i] = i + 1.0; }
   y.HostWrite();
   y = 0.0;

   SECTION("Host placement with a device callback")
   {
      op.on_device = true;
   }
   SECTION("Device placement with a host callback")
   {
      x.ReadWrite(true);
      y.ReadWrite(true);
   }

   PetscParMatrix matrix(MPI_COMM_WORLD, &op, Operator::PETSC_MATSHELL);
   matrix.Mult(x, y);
   const auto result = y.HostRead();
   const auto original = x.HostRead();
   for (int i = 0; i < 4; ++i)
   {
      REQUIRE(result[i] == MFEM_Approx(2.0 * (i + 1.0)));
      REQUIRE(original[i] == MFEM_Approx(i + 1.0));
   }
}

TEST_CASE("PETSc placement preserves fragmented data and unrelated ranges",
          "[Parallel][PETSc][GPU]")
{
   if (!Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK)) { return; }
   EnsurePetscSession();

   IdentityOperator op(4);
   PetscParMatrix matrix(MPI_COMM_WORLD, &op, Operator::PETSC_MATSHELL);
   Vector storage(8), output(4);
   auto host = storage.HostWrite();
   for (int i = 0; i < 8; ++i) { host[i] = i + 1.0; }
   auto device = storage.GetMemory().Write(true, 0, 2);
   forall(2, [=] MFEM_HOST_DEVICE(int i) { device[i] = 10.0 * (i + 1); });
   auto tail = storage.GetMemory().Write(true, 6, 2);
   forall(2, [=] MFEM_HOST_DEVICE(int i) { tail[i] = 70.0 + i; });
   Vector input(storage, 0, 4);
   REQUIRE_FALSE(input.GetMemory().HostIsValid());
   REQUIRE_FALSE(input.GetMemory().DeviceIsValid());
   matrix.Mult(input, output);
   auto result = output.HostRead();
   REQUIRE(result[0] == MFEM_Approx(10.0));
   REQUIRE(result[1] == MFEM_Approx(20.0));
   REQUIRE(result[2] == MFEM_Approx(3.0));
   REQUIRE(result[3] == MFEM_Approx(4.0));
   auto untouched = storage.GetMemory().CreateAlias(6, 2);
   REQUIRE_FALSE(untouched.HostIsValid());
   REQUIRE(untouched.DeviceIsValid());
   REQUIRE(untouched.HostRead()[0] == MFEM_Approx(70.0));
   untouched.Reset();

   SECTION("Read-write placement")
   {
      auto out_host = output.HostWrite();
      for (int i = 0; i < 4; ++i) { out_host[i] = 1.0; }
      auto out_device = output.GetMemory().Write(true, 0, 2);
      forall(2, [=] MFEM_HOST_DEVICE(int i) { out_device[i] = 5.0; });
      matrix.Mult(1.0, input, 1.0, output);
      result = output.HostRead();
      REQUIRE(result[0] == MFEM_Approx(15.0));
      REQUIRE(result[1] == MFEM_Approx(25.0));
      REQUIRE(result[2] == MFEM_Approx(4.0));
      REQUIRE(result[3] == MFEM_Approx(5.0));
   }
}
#endif

#endif
