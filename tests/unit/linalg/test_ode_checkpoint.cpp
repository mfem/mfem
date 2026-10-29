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

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

using namespace mfem;

namespace
{

namespace fs = std::filesystem;

using StorageType = Checkpointer::StorageType;

const StorageType all_storages[] =
{
   StorageType::MemoryBlock, StorageType::FileBlock,
   StorageType::MemorySnapshots, StorageType::FileSnapshots
};

bool IsFile(StorageType storage)
{
   return storage == StorageType::FileBlock ||
          storage == StorageType::FileSnapshots;
}

/// du_i/dt = t - (i+1) u_i, with a closed-form implicit solve.
class ForcedDecay : public TimeDependentOperator
{
public:
   explicit ForcedDecay(int n) : TimeDependentOperator(n) { }

   void Mult(const Vector &u, Vector &k) const override
   {
      k.SetSize(u.Size());
      for (int i = 0; i < u.Size(); i++) { k(i) = GetTime() - (i + 1)*u(i); }
   }

   void ImplicitSolve(const real_t gamma, const Vector &u, Vector &k) override
   {
      k.SetSize(u.Size());
      for (int i = 0; i < u.Size(); i++)
      {
         const real_t rate = i + 1;
         k(i) = (GetTime() - rate*u(i))/(1.0 + gamma*rate);
      }
   }
};

/// Solver kinds: 0 forward Euler, 1 RK4, 2 backward Euler.
std::unique_ptr<ODESolver> MakeSolver(int kind)
{
   if (kind == 1) { return std::make_unique<RK4Solver>(); }
   if (kind == 2) { return std::make_unique<BackwardEulerSolver>(); }
   return std::make_unique<ForwardEulerSolver>();
}

/// Live state of an independent time loop.
struct ODELive
{
   ForcedDecay oper;
   std::unique_ptr<ODESolver> solver;
   Vector x;
   StateId step = 0;
   real_t t = 0.0;
   real_t dt = 0.05;

   explicit ODELive(int kind) : oper(3), solver(MakeSolver(kind)), x(3)
   {
      x(0) = 1.0; x(1) = -0.5; x(2) = 0.25;
      solver->Init(oper);
   }

   ODECheckpointer::Fields All() { return {&step, &t, &dt}; }

   void Run(StateId to)
   {
      while (step < to) { solver->Step(x, t, dt); ++step; }
   }
};

bool Same(const ODELive &left, const ODELive &right)
{
   if (left.step != right.step || left.t != right.t || left.dt != right.dt ||
       left.x.Size() != right.x.Size()) { return false; }
   for (int i = 0; i < left.x.Size(); i++)
   {
      if (left.x(i) != right.x(i)) { return false; }
   }
   return true;
}

/// Return an unused explicit store path under the temporary directory.
std::string NewStorePath(const std::string &name)
{
   const auto stamp = std::chrono::steady_clock::now()
                      .time_since_epoch().count();
   const fs::path path = fs::temp_directory_path() /
                         ("mfem-ode-checkpoint-test-" + name + "-" +
                          std::to_string(stamp));
   fs::remove_all(path);
   return path.string();
}

/// Select MFEM_ERROR_THROW for the lifetime of the guard.
class ThrowOnError
{
private:
   ErrorAction previous;

public:
   ThrowOnError() : previous(get_error_action())
   { set_error_action(MFEM_ERROR_THROW); }
   ~ThrowOnError() { set_error_action(previous); }
};

} // namespace

TEST_CASE("ODE checkpointers restore exact single-step continuation",
          "[ODECheckpoint]")
{
   for (int kind = 0; kind < 3; kind++)
   {
      for (const StorageType storage : all_storages)
      {
         INFO("solver " << kind << ", storage " << static_cast<int>(storage));
         ODELive live(kind), mid(kind), ref(kind);
         auto checkpoints = MakeODECheckpointer(storage, live.x, *live.solver,
                                                live.All(), 4);
         live.Run(3);
         checkpoints->Store(7);
         REQUIRE(checkpoints->GetInfo(7).state == 3);
         live.Run(8);
         checkpoints->Restore(7);
         mid.Run(3);
         REQUIRE(Same(live, mid));
         live.Run(8);
         ref.Run(8);
         REQUIRE(Same(live, ref));
      }
   }
}

TEST_CASE("ODE checkpointers bind optional fields and an internal position",
          "[ODECheckpoint]")
{
   for (const StorageType storage : all_storages)
   {
      INFO("storage " << static_cast<int>(storage));
      {
         ODELive live(0);
         auto checkpoints = MakeODECheckpointer(storage, live.x, *live.solver,
                                                {}, 2);
         REQUIRE(checkpoints->RecordBytes() ==
                 3*sizeof(std::uint64_t) + 5*sizeof(real_t));
         REQUIRE(checkpoints->CurrentState() == 0);
         checkpoints->SetCurrentState(2);
         checkpoints->Store(1);
         REQUIRE(checkpoints->GetInfo(1).state == 2);
         live.x = 9.0;
         live.t = 4.0;
         checkpoints->SetCurrentState(5);
         checkpoints->Restore(1);
         REQUIRE(checkpoints->CurrentState() == 2);
         REQUIRE(live.x(0) == 1.0);
         REQUIRE(live.x(2) == 0.25);
         REQUIRE(live.t == 4.0);
         REQUIRE(live.step == 0);
      }
      {
         ODELive live(0);
         const ODECheckpointer::Fields fields{&live.step, &live.t, nullptr};
         auto checkpoints = MakeODECheckpointer(storage, live.x, *live.solver,
                                                fields, 2);
         checkpoints->Store(1);
         live.Run(2);
         live.dt = 0.3;
         checkpoints->Restore(1);
         REQUIRE(live.step == 0);
         REQUIRE(live.t == 0.0);
         REQUIRE(live.dt == 0.3);
      }
   }
}

TEST_CASE("ODE checkpointers follow the Checkpointer record contract",
          "[ODECheckpoint]")
{
   for (const StorageType storage : all_storages)
   {
      INFO("storage " << static_cast<int>(storage));
      ODELive live(0);
      auto checkpoints = MakeODECheckpointer(storage, live.x, *live.solver,
                                             live.All(), 2);
      REQUIRE(checkpoints->Capacity() == std::optional<std::size_t>(2));
      checkpoints->Store(5);
      live.Run(1);
      checkpoints->Store(5);
      REQUIRE(checkpoints->Size() == 1);
      REQUIRE(checkpoints->GetInfo(5).state == 1);
      checkpoints->Store(3);
      REQUIRE(checkpoints->Size() == 2);

      CheckpointInfo info;
      REQUIRE(checkpoints->FindAtOrBefore(4, info));
      REQUIRE(info.checkpoint == 3);
      REQUIRE(info.state == 1);
      REQUIRE_FALSE(checkpoints->FindAtOrBefore(0, info));
      REQUIRE(info.checkpoint == 3);

      checkpoints->Erase(99);
      REQUIRE(checkpoints->Size() == 2);
      checkpoints->Erase(5);
      REQUIRE_FALSE(checkpoints->Contains(5));
      REQUIRE(checkpoints->Size() == 1);

      // The erased record is reused at full capacity.
      live.Run(2);
      checkpoints->Store(8);
      REQUIRE(checkpoints->Size() == 2);
      checkpoints->Restore(3);
      REQUIRE(live.step == 1);
      checkpoints->Restore(8);
      REQUIRE(live.step == 2);
   }

   ODELive live(0);
   ODEMemorySnapshotCheckpointer growing(live.x, *live.solver, live.All());
   REQUIRE_FALSE(growing.Capacity().has_value());
   for (CheckpointId id = 1; id <= 10; id++)
   {
      growing.Store(id);
      live.Run(static_cast<StateId>(id));
   }
   REQUIRE(growing.Size() == 10);
   growing.Restore(4);
   REQUIRE(live.step == 3);
}

TEST_CASE("File ODE checkpointers close and reopen into fresh objects",
          "[ODECheckpoint]")
{
   for (const StorageType storage : all_storages)
   {
      if (!IsFile(storage)) { continue; }
      INFO("storage " << static_cast<int>(storage));
      const std::string path =
         NewStorePath(std::to_string(static_cast<int>(storage)));
      ODELive ref(1);
      ref.Run(6);
      {
         ODELive live(1);
         auto checkpoints = MakeODECheckpointer(storage, live.x, *live.solver,
                                                live.All(), 3, path);
         live.Run(2);
         checkpoints->Store(1);
         live.Run(4);
         checkpoints->Store(2);
         checkpoints->Erase(1);
         live.Run(5);
         checkpoints->Store(3);
         checkpoints->Close();
         checkpoints->Close();
         REQUIRE(checkpoints->CurrentState() == 5);
         REQUIRE(checkpoints->Capacity() == std::optional<std::size_t>(3));
      }
      REQUIRE(fs::exists(path));
      {
         ODELive live(1);
         auto checkpoints = MakeODECheckpointer(storage, live.x, *live.solver,
                                                live.All(), 3, path, true);
         REQUIRE(checkpoints->Size() == 2);
         REQUIRE_FALSE(checkpoints->Contains(1));
         REQUIRE(checkpoints->GetInfo(2).state == 4);
         checkpoints->Restore(2);
         live.Run(6);
         REQUIRE(Same(live, ref));
      }
      fs::remove_all(path);
   }

   std::string automatic;
   {
      ODELive live(0);
      ODEFileSnapshotCheckpointer checkpoints(live.x, *live.solver,
                                              live.All());
      automatic = checkpoints.Path();
      checkpoints.Store(1);
      REQUIRE(fs::exists(automatic));
   }
   REQUIRE_FALSE(fs::exists(automatic));
}

TEST_CASE("ODE checkpointers drive controller replay",
          "[ODECheckpoint]")
{
   SECTION("Interval schedule without a window")
   {
      ODELive live(1), ref(1);
      ODEMemoryBlockCheckpointer checkpoints(live.x, *live.solver, live.All(),
                                             3);
      ODEStatePropagator propagator(checkpoints, *live.solver);
      CheckpointController controller(checkpoints, propagator);
      IntervalSchedule schedule(8, 3);
      controller.Run(schedule, 8);
      REQUIRE(live.step == 8);
      REQUIRE(checkpoints.Size() == 3);
      controller.RestoreState(5);
      ref.Run(5);
      REQUIRE(Same(live, ref));
   }

   SECTION("Store-everything schedule with a memory window")
   {
      ODELive live(2), ref(2);
      ODEFileSnapshotCheckpointer checkpoints(live.x, *live.solver,
                                              live.All());
      ODEMemorySnapshotCheckpointer memory(live.x, *live.solver, live.All());
      ExactCheckpointWindow window(memory, 2);
      ODEStatePropagator propagator(checkpoints, *live.solver);
      CheckpointController controller(checkpoints, propagator, &window);
      StoreEverythingSchedule schedule;
      schedule.Configure(4, 5);
      controller.Run(schedule, 4);
      REQUIRE(checkpoints.Size() == 5);
      REQUIRE(window.Size() == 2);
      controller.RestoreState(2);
      ref.Run(2);
      REQUIRE(Same(live, ref));
   }
}

TEST_CASE("ODE checkpoint factory selects the requested layout",
          "[ODECheckpoint]")
{
   ODELive live(0);
   const auto make = [&](StorageType storage)
   {
      return MakeODECheckpointer(storage, live.x, *live.solver, live.All(), 1);
   };
   const auto memory_block = make(StorageType::MemoryBlock);
   const auto file_block = make(StorageType::FileBlock);
   const auto memory_snapshots = make(StorageType::MemorySnapshots);
   const auto file_snapshots = make(StorageType::FileSnapshots);
   REQUIRE(dynamic_cast<ODEMemoryBlockCheckpointer *>(memory_block.get()));
   REQUIRE(dynamic_cast<ODEFileBlockCheckpointer *>(file_block.get()));
   REQUIRE(dynamic_cast<ODEMemorySnapshotCheckpointer *>(
              memory_snapshots.get()));
   REQUIRE(dynamic_cast<ODEFileSnapshotCheckpointer *>(file_snapshots.get()));
}

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("ODE checkpointers reject invalid use", "[ODECheckpoint]")
{
   ThrowOnError guard;
   ODELive live(0);

   AB2Solver multistep;
   multistep.Init(live.oper);
   REQUIRE_THROWS_AS(ODEMemorySnapshotCheckpointer(live.x, multistep,
                                                   live.All()),
                     ErrorException);
   REQUIRE_THROWS_AS(MakeODECheckpointer(StorageType::MemoryBlock, live.x,
                                         *live.solver, live.All(),
                                         std::nullopt),
                     ErrorException);
   REQUIRE_THROWS_AS(MakeODECheckpointer(StorageType::MemorySnapshots, live.x,
                                         *live.solver, live.All(),
                                         std::nullopt, "unused"),
                     ErrorException);

   for (const StorageType storage : all_storages)
   {
      INFO("storage " << static_cast<int>(storage));
      auto checkpoints = MakeODECheckpointer(storage, live.x, *live.solver,
                                             live.All(), 1);
      REQUIRE_THROWS_AS(checkpoints->Restore(1), ErrorException);
      REQUIRE_THROWS_AS(checkpoints->GetInfo(1), ErrorException);
      checkpoints->Store(1);
      REQUIRE_THROWS_AS(checkpoints->Store(2), ErrorException);
      live.step = -1;
      REQUIRE_THROWS_AS(checkpoints->Store(1), ErrorException);
      live.step = 0;
      if (IsFile(storage))
      {
         checkpoints->Close();
         REQUIRE_THROWS_AS(checkpoints->Contains(1), ErrorException);
         REQUIRE(checkpoints->CurrentState() == 0);
      }
   }

   const std::string path = NewStorePath("layout");
   {
      ODEFileSnapshotCheckpointer checkpoints(live.x, *live.solver,
                                              live.All(), {}, path);
      checkpoints.Store(1);
   }
   REQUIRE_THROWS_AS(ODEFileSnapshotCheckpointer(live.x, *live.solver, {}, {},
                                                 path, true),
                     ErrorException);
   REQUIRE_THROWS_AS(ODEFileSnapshotCheckpointer(live.x, *live.solver,
                                                 live.All(), {}, path),
                     ErrorException);
   fs::remove_all(path);
}
#endif // MFEM_USE_EXCEPTIONS
