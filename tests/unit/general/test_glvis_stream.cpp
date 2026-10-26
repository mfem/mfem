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

#ifndef MFEM_USE_GLVIS
TEST_CASE("glvis stream is not built", "[GLVis]")
{
   // GLVis is off. The case keeps the tag run from matching nothing.
}
#endif

#ifdef MFEM_USE_GLVIS

TEST_CASE("glvis_stream gather", "[GLVis]")
{
   glvis_stream glvis;
   glvis.SetVisualize(false);
   glvis << "mesh\n" << std::endl;
   REQUIRE(glvis.size() > 0);
   glvis << std::flush;
   REQUIRE(glvis.size() == 0);
}

// Reading to EOF sets the inherited failbit. reset() must clear it.
TEST_CASE("glvis_stream reset after EOF", "[GLVis]")
{
   glvis_stream glvis;
   glvis.SetVisualize(false);
   glvis << "mesh\n";
   REQUIRE(glvis.size() > 0);

   glvis.seekg(0);
   std::string token;
   glvis >> token;
   REQUIRE(token == "mesh");
   glvis >> token;
   REQUIRE(glvis.fail());

   glvis.reset();
   const std::string payload = "solution\n";
   glvis << payload;
   REQUIRE(glvis.good());
   REQUIRE(glvis.size() == payload.size());
   glvis << std::flush;
   REQUIRE(glvis.size() == 0);
}

#ifdef MFEM_USE_MPI
TEST_CASE("glvis_stream gather parallel", "[GLVis][Parallel]")
{
   glvis_stream glvis(MPI_COMM_WORLD);
   glvis.SetVisualize(false);
   glvis << "mesh\n" << std::flush;
   REQUIRE(glvis.size() == 0);
}

TEST_CASE("glvis_stream parallel header", "[GLVis][Parallel]")
{
   glvis_stream glvis(MPI_COMM_WORLD);
   glvis.SetVisualize(false);
   glvis << "parallel " << Mpi::WorldSize() << ' '
         << Mpi::WorldRank() << '\n' << std::flush;
   REQUIRE(glvis.size() == 0);
}
#endif

#endif
