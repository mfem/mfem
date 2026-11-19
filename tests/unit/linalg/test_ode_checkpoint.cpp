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

#if defined(MFEM_USE_EXCEPTIONS) && defined(MFEM_USE_MPI) && \
    defined(MFEM_USE_PETSC)
#include <petscts.h>
#endif

#include <chrono>
#include <filesystem>
#include <limits>
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
   StateId step{0};
   real_t t = 0.0;
   real_t dt = 0.05;

   explicit ODELive(int kind) : oper(3), solver(MakeSolver(kind)), x(3)
   {
      x(0) = 1.0; x(1) = -0.5; x(2) = 0.25;
      solver->Init(oper);
   }

   ODECheckpointer::StateParameters All() { return {&step, &t, &dt}; }

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

#if defined(MFEM_USE_EXCEPTIONS) && defined(MFEM_USE_MPI) && \
    defined(MFEM_USE_PETSC)
class CheckpointPetscSession
{
private:
   bool owned;

public:
   CheckpointPetscSession()
   {
      PetscBool initialized;
      MFEM_VERIFY(PetscInitialized(&initialized) == 0,
                  "Cannot query PETSc initialization.");
      owned = !initialized;
      if (owned) { MFEMInitializePetsc(); }
   }

   ~CheckpointPetscSession()
   {
      if (!owned) { return; }
      PetscBool finalized;
      MFEM_VERIFY(PetscFinalized(&finalized) == 0,
                  "Cannot query PETSc finalization.");
      if (!finalized) { MFEMFinalizePetsc(); }
   }
};
#endif

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
         live.Run(StateId{3});
         checkpoints->Store(CheckpointId{7});
         REQUIRE(checkpoints->GetInfo(CheckpointId{7}).state == StateId{3});
         live.Run(StateId{8});
         checkpoints->Restore(CheckpointId{7});
         mid.Run(StateId{3});
         REQUIRE(Same(live, mid));
         live.Run(StateId{8});
         ref.Run(StateId{8});
         REQUIRE(Same(live, ref));
      }
   }
}

TEST_CASE("ODE checkpointers bind optional state parameters and an internal "
          "position", "[ODECheckpoint]")
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
         REQUIRE(checkpoints->CurrentState() == StateId{0});
         checkpoints->SetCurrentState(StateId{2});
         checkpoints->Store(CheckpointId{1});
         REQUIRE(checkpoints->GetInfo(CheckpointId{1}).state == StateId{2});
         live.x = 9.0;
         live.t = 4.0;
         checkpoints->SetCurrentState(StateId{5});
         checkpoints->Restore(CheckpointId{1});
         REQUIRE(checkpoints->CurrentState() == StateId{2});
         REQUIRE(live.x(0) == 1.0);
         REQUIRE(live.x(2) == 0.25);
         REQUIRE(live.t == 4.0);
         REQUIRE(live.step == StateId{0});
      }
      {
         ODELive live(0);
         const ODECheckpointer::StateParameters state_params =
            {&live.step, &live.t, nullptr};
         auto checkpoints = MakeODECheckpointer(storage, live.x, *live.solver,
                                                state_params, 2);
         checkpoints->Store(CheckpointId{1});
         live.Run(StateId{2});
         live.dt = 0.3;
         checkpoints->Restore(CheckpointId{1});
         REQUIRE(live.step == StateId{0});
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
      checkpoints->Store(CheckpointId{5});
      live.Run(StateId{1});
      checkpoints->Store(CheckpointId{5});
      REQUIRE(checkpoints->Size() == 1);
      REQUIRE(checkpoints->GetInfo(CheckpointId{5}).state == StateId{1});
      checkpoints->Store(CheckpointId{3});
      REQUIRE(checkpoints->Size() == 2);

      CheckpointInfo info;
      REQUIRE(checkpoints->FindAtOrBefore(StateId{4}, info));
      REQUIRE(info.checkpoint == CheckpointId{3});
      REQUIRE(info.state == StateId{1});
      REQUIRE_FALSE(checkpoints->FindAtOrBefore(StateId{0}, info));
      REQUIRE(info.checkpoint == CheckpointId{3});

      checkpoints->Erase(CheckpointId{99});
      REQUIRE(checkpoints->Size() == 2);
      checkpoints->Erase(CheckpointId{5});
      REQUIRE_FALSE(checkpoints->Contains(CheckpointId{5}));
      REQUIRE(checkpoints->Size() == 1);

      // The erased record is reused at full capacity.
      live.Run(StateId{2});
      checkpoints->Store(CheckpointId{8});
      REQUIRE(checkpoints->Size() == 2);
      checkpoints->Restore(CheckpointId{3});
      REQUIRE(live.step == StateId{1});
      checkpoints->Restore(CheckpointId{8});
      REQUIRE(live.step == StateId{2});
   }

   ODELive live(0);
   ODEMemorySnapshotCheckpointer growing(live.x, *live.solver, live.All());
   REQUIRE_FALSE(growing.Capacity().has_value());
   for (std::int64_t position = 1; position <= 10; position++)
   {
      growing.Store(CheckpointId{static_cast<std::uint64_t>(position)});
      live.Run(StateId{position});
   }
   REQUIRE(growing.Size() == 10);
   growing.Restore(CheckpointId{4});
   REQUIRE(live.step == StateId{3});
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
      ref.Run(StateId{6});
      {
         ODELive live(1);
         auto checkpoints = MakeODECheckpointer(storage, live.x, *live.solver,
                                                live.All(), 3, path);
         live.Run(StateId{2});
         checkpoints->Store(CheckpointId{1});
         live.Run(StateId{4});
         checkpoints->Store(CheckpointId{2});
         checkpoints->Erase(CheckpointId{1});
         live.Run(StateId{5});
         checkpoints->Store(CheckpointId{3});
         checkpoints->Close();
         checkpoints->Close();
         REQUIRE(checkpoints->CurrentState() == StateId{5});
         REQUIRE(checkpoints->Capacity() == std::optional<std::size_t>(3));
      }
      REQUIRE(fs::exists(path));
      {
         ODELive live(1);
         auto checkpoints = MakeODECheckpointer(storage, live.x, *live.solver,
                                                live.All(), 3, path, true);
         REQUIRE(checkpoints->Size() == 2);
         REQUIRE_FALSE(checkpoints->Contains(CheckpointId{1}));
         REQUIRE(checkpoints->GetInfo(CheckpointId{2}).state == StateId{4});
         checkpoints->Restore(CheckpointId{2});
         live.Run(StateId{6});
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
      checkpoints.Store(CheckpointId{1});
      REQUIRE(fs::exists(automatic));
   }
   REQUIRE_FALSE(fs::exists(automatic));
}

TEST_CASE("ODE checkpoint identifiers retain their full integer ranges",
          "[ODECheckpoint]")
{
   const StateId last_state{std::numeric_limits<std::int64_t>::max()};
   const CheckpointId last_checkpoint{
      std::numeric_limits<std::uint64_t>::max()};
   for (const StorageType storage : all_storages)
   {
      INFO("storage " << static_cast<int>(storage));
      const std::string path = IsFile(storage) ? NewStorePath("id-range") : "";
      {
         ODELive live(0);
         auto checkpoints = MakeODECheckpointer(storage, live.x, *live.solver,
                                                live.All(), 1, path);
         live.step = last_state;
         checkpoints->Store(last_checkpoint);
         live.step = StateId{-1};
         checkpoints->Restore(last_checkpoint);
         REQUIRE(live.step == last_state);
         CheckpointInfo nearest;
         REQUIRE(checkpoints->FindAtOrBefore(last_state, nearest));
         REQUIRE(nearest.checkpoint == last_checkpoint);
         REQUIRE(nearest.state == last_state);
         checkpoints->Close();
      }
      if (IsFile(storage))
      {
         {
            ODELive live(0);
            auto reopened = MakeODECheckpointer(storage, live.x, *live.solver,
                                                live.All(), 1, path, true);
            REQUIRE(reopened->Contains(last_checkpoint));
            reopened->Restore(last_checkpoint);
            REQUIRE(live.step == last_state);
            if (storage == StorageType::FileSnapshots)
            {
               REQUIRE(fs::exists(fs::path(path) /
                                  "record-18446744073709551615.bin"));
            }
         }
         fs::remove_all(path);
      }
   }
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
      IntervalSchedule schedule(StateId{8}, StateId{3});
      controller.Run(schedule, StateId{8});
      REQUIRE(live.step == StateId{8});
      REQUIRE(checkpoints.Size() == 3);
      controller.RestoreState(StateId{5});
      ref.Run(StateId{5});
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
      schedule.Configure(StateId{4}, 5);
      controller.Run(schedule, StateId{4});
      REQUIRE(checkpoints.Size() == 5);
      REQUIRE(window.Size() == 2);
      controller.RestoreState(StateId{2});
      ref.Run(StateId{2});
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
      REQUIRE_THROWS_AS(checkpoints->Restore(CheckpointId{1}), ErrorException);
      REQUIRE_THROWS_AS(checkpoints->GetInfo(CheckpointId{1}), ErrorException);
      checkpoints->Store(CheckpointId{1});
      REQUIRE_THROWS_AS(checkpoints->Store(CheckpointId{2}), ErrorException);
      live.step = StateId{-1};
      REQUIRE_THROWS_AS(checkpoints->Store(CheckpointId{1}), ErrorException);
      live.step = StateId{0};
      if (IsFile(storage))
      {
         checkpoints->Close();
         REQUIRE_THROWS_AS(checkpoints->Contains(CheckpointId{1}),
                           ErrorException);
         REQUIRE(checkpoints->CurrentState() == StateId{0});
      }
   }

   const std::string path = NewStorePath("layout");
   {
      ODEFileSnapshotCheckpointer checkpoints(live.x, *live.solver,
                                              live.All(), {}, path);
      checkpoints.Store(CheckpointId{1});
   }
   REQUIRE_THROWS_AS(ODEFileSnapshotCheckpointer(live.x, *live.solver, {}, {},
                                                 path, true),
                     ErrorException);
   REQUIRE_THROWS_AS(ODEFileSnapshotCheckpointer(live.x, *live.solver,
                                                 live.All(), {}, path),
                     ErrorException);
   fs::remove_all(path);
}

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_PETSC)
TEST_CASE("ODE checkpointers reject PETSc BDF internal history",
          "[Parallel][PETSc][ODECheckpoint]")
{
   static CheckpointPetscSession petsc_session;
   ThrowOnError guard;
   ODELive live(0);
   PetscODESolver solver(MPI_COMM_SELF);
   solver.Init(live.oper);
   REQUIRE(TSSetType(solver, TSBDF) == 0);
   REQUIRE(solver.GetStateSize() == 0);
   for (const StorageType storage : all_storages)
   {
      INFO("storage " << static_cast<int>(storage));
      REQUIRE_THROWS_AS(MakeODECheckpointer(storage, live.x, solver,
                                            live.All(), 4), ErrorException);
   }
}
#endif

#ifdef MFEM_USE_SUNDIALS
TEST_CASE("ODE checkpointers reject SUNDIALS internal state",
          "[ODECheckpoint]")
{
   Sundials::Init();
   ThrowOnError guard;
   ODELive live(0);
   std::unique_ptr<ODESolver> solvers[] =
   {
      std::make_unique<CVODESolver>(CV_BDF),
      std::make_unique<CVODESSolver>(CV_BDF),
      std::make_unique<ARKStepSolver>()
   };
   for (const auto &solver : solvers)
   {
      REQUIRE(solver->GetStateSize() == 0);
      for (const StorageType storage : all_storages)
      {
         INFO("storage " << static_cast<int>(storage));
         REQUIRE_THROWS_AS(MakeODECheckpointer(storage, live.x, *solver,
                                               live.All(), 4), ErrorException);
      }
   }
}
#endif
#endif // MFEM_USE_EXCEPTIONS
