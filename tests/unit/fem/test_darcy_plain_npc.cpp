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
#include "fem/darcy_plain_npc.hpp"

using namespace mfem;

/** The NO-DEVICE half of the pair. Its twin, over the same helper, is
    "The plain NPC path solves an exact polynomial under a device" in
    tests/unit/miniapps/test_debug_device.cpp, whose binary configures
    Device("debug"). The two differ in the Device and in nothing else, which
    is what makes a disagreement between them attributable.

    See fem/darcy_plain_npc.hpp for why the answer is exact and why the
    configuration is meq's. */
TEST_CASE("The plain NPC path solves an exact polynomial",
          "[DarcyHybridization][NPC][GPU]")
{
   const int order = GENERATE(1, 2);
   const int n = GENERATE(2, 4);
   // Triangles because that is what meq runs; quadrilaterals because that is
   // what every other Darcy device case in this tree runs.
   const Element::Type geom =
      GENERATE(Element::QUADRILATERAL, Element::TRIANGLE);
   // With mass = 0 the domain term on the potential-mass form is
   // arithmetically inert, and an inert term cannot distinguish a device that
   // evaluates it from one that drops it -- which is the fault this case
   // exists to catch. mass = 3 puts a non-zero source on the same integrator
   // with a load to match, so BOTH halves move err_p off round-off if either
   // is lost. See the header.
   const real_t mass = GENERATE(0.0, 3.0);
   CAPTURE(order, n, geom == Element::TRIANGLE, mass);

   const darcy_plain_npc::Result r = darcy_plain_npc::Solve(order, n, geom, mass);
   CAPTURE(r.err_u, r.err_p, r.r0_fields, r.r0_trace, r.step, r.datum,
           r.nonfinite, r.trace_size, r.load, r.batched_residual);


   // The route, pinned. This case earns its keep by running what meq runs,
   // and PotNL is what makes that true; see the header. A silent move onto
   // the batched linear residual would leave every assertion below passing
   // while the case stopped testing meq's configuration.
   REQUIRE_FALSE(r.batched_residual);
   // FIRST, because every norm below is meaningless otherwise: Norml2()
   // cannot see a NaN, and an all-NaN vector reports a norm of zero.
   REQUIRE(r.nonfinite == 0);

   // The problem is driven by the datum alone, so a zero datum would make
   // u = p = 0 the answer and every error below would pass on nothing.
   REQUIRE(r.datum > 0.1);

   // There was something to solve. This is the assertion that meq's symptom
   // trips: a Newton that takes zero steps has been told its residual is
   // already converged.
   REQUIRE(r.r0_fields > 1e-3);
   REQUIRE(r.r0_trace > 1e-3);
   REQUIRE(r.step > 1e-3);

   // And the answer, which is arithmetic rather than a second run.

   // The non-zero arm's source is live. Without this the arm could silently
   // degrade to the inert one and every assertion below would still pass.
   // 1e-3 and not something larger: the load is an ELEMENT integral, so it
   // scales with the element measure -- 5.6e-02 at order 2 on 4x4 triangles,
   // and smaller again on any finer mesh. The question it settles is
   // live-or-zero, not how big.
   if (mass != 0.0) { REQUIRE(r.load > 1e-3); }
   else { REQUIRE(r.load == 0.0); }
   REQUIRE(r.err_u < 1e-10);
   REQUIRE(r.err_p < 1e-10);
}

/** @brief The same problem driven through DarcyNPCOperator and
    DarcyNPCSolver under a NewtonSolver -- the layer meq runs.

    **This exists because the case above could not see a defect that was
    there.** It calls NPCResidual() directly, and meq's M-80 names
    DarcyNPCOperator::Mult as the `residual` leg their Newton evaluates. That
    routine writes its answer into three BLOCK ALIASES of y and told neither
    the BlockVector nor y, so under a Device the caller read a stale y. A
    NewtonSolver handed a zero residual reports convergence at iteration zero
    and leaves the fields at the initial iterate -- whose potential block is
    zero under NPC.

    So the two assertions that matter are r0_operator, taken through the
    operator before any solve, and its >= 1. Both are meq's M-79 symptom
    written down as a requirement. */
TEST_CASE("The NPC operator drives a Newton to the exact polynomial",
          "[DarcyHybridization][NPC][GPU]")
{
   const int order = GENERATE(1, 2);
   const int n = GENERATE(2, 4);
   const Element::Type geom =
      GENERATE(Element::QUADRILATERAL, Element::TRIANGLE);
   const real_t mass = GENERATE(0.0, 3.0);
   CAPTURE(order, n, geom == Element::TRIANGLE, mass);

   const darcy_plain_npc::Result r =
      darcy_plain_npc::Solve(order, n, geom, mass, true);
   CAPTURE(r.err_u, r.err_p, r.r0_operator, r.its, r.converged, r.datum,
           r.nonfinite, r.r_dev_valid);

   REQUIRE(r.nonfinite == 0);
   REQUIRE(r.datum > 0.1);

   // meq's symptom, both halves. A residual that reads zero through the
   // operator is what makes the Newton below take no steps at all.
   REQUIRE(r.r0_operator > 1e-3);
   // **`its` is the assertion that catches it, and r0_operator is NOT** --
   // which is worth knowing because the reverse is what you would guess.
   // Measured with the sync gated off: r0_operator stays > 1e-3 and `its`
   // comes back 0. The reason is that the measurement above ends in
   // R.HostRead(), and host-reading the residual IS the missing sync, so a
   // probe that reads it correctly cannot see the fault. Newton's own r is a
   // UseDevice(true) work vector that nothing host-reads, so Newton sees the
   // stale zero and reports convergence at iteration 0. A correct measurement
   // of a quantity can hide the defect in how everyone else obtains it.
   REQUIRE(r.its >= 1);
   REQUIRE(r.converged);

   REQUIRE(r.err_u < 1e-10);
   REQUIRE(r.err_p < 1e-10);
   // **Where the data IS, not what it says.** A check that ends in a host
   // read cannot distinguish flags that were repaired from data that was
   // migrated to the host, because SyncAlias() will bring device-valid blocks
   // down and unprotect them. This samples residency directly instead.
   //
   // What it pins is that the residual leaves the operator device-resident,
   // with no d2h round trip per Newton evaluation -- a property the device
   // offload plan needs and nothing else asserts. Measured: it does NOT
   // separate the one-hop sync from the two-hop one, which stay
   // device-resident alike.
   if (Device::Allows(Backend::DEVICE_MASK)) { REQUIRE(r.r_dev_valid); }
}
