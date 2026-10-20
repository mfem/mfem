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

#ifdef MFEM_USE_GLVIS

TEST_CASE("glvis_stream gather", "[General]")
{
   glvis_stream glvis;
   glvis.SetVisualize(false);
   glvis << "mesh\n" << std::endl;
   REQUIRE(glvis.size() > 0);
   glvis << std::flush;
   REQUIRE(glvis.size() == 0);
}

#ifdef MFEM_USE_MPI
TEST_CASE("glvis_stream gather parallel", "[Parallel]")
{
   glvis_stream glvis(MPI_COMM_WORLD);
   glvis.SetVisualize(false);
   glvis << "mesh\n" << std::flush;
   REQUIRE(glvis.size() == 0);
}
#endif

#endif
