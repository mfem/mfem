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
using namespace mfem;

#define CATCH_CONFIG_RUNNER
#include "run_unit_tests.hpp"

#include "test_assembly.hpp"

#if defined(MFEM_USE_MPI) && !defined(_WIN32)

#if defined(__has_include) && __has_include("general/nvtx.hpp") && !defined(_WIN32)
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kFuchsia
#include "general/nvtx.hpp"
#else
#define dbg(...)
#endif

namespace dfem_derivative_assembly
{

TEST_CASE("dfem/debug_device", "[Parallel][DFEM][DebugDevice]")
{
   dbg();
   const auto all_tests = launch_all_non_regression_tests;
   const int p = !all_tests ? 2 : GENERATE(1, 2, 3);
   static_assert(std::tuple_size<results_t>::value == 3);

   SECTION("2D p=" + std::to_string(p))
   {
      auto [filename, expected_fnorms] =
         GENERATE(table<std::string, results_t>(
      {
         {
            "../../data/star.mesh",
            {24.987721237546996, 146.52903344877006, 539.651561004031}
         },
         {
            "../../data/star-q3.mesh",
            {24.95956386897864, 150.755043841781, 736.2018345106584}
         },
         {
            "../../data/inline-quad.mesh",
            {15.270705433752719, 110.0016890554948, 419.94593064913283}
         },
         {
            "../../data/periodic-square.mesh",
            {6.733003290855952, 70.85200869436, 6656.219250863574}
         }
      }));
      DFemDerivativeAssembly<2>(filename.c_str(), p, expected_fnorms);
   }
}

} // namespace dfem_derivative_assembly

#endif // MFEM_USE_MPI && !_WIN32

int main(int argc, char *argv[])
{
   mfem::Mpi::Init();
   mfem::Hypre::Init();

   dbg();
   Device device("debug");
   device.Print();
   return RunCatchSession(argc, argv, {"[Parallel][DFEM][DebugDevice]"}, Root());
}
