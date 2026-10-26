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

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

using namespace mfem;

#ifdef MFEM_USE_GLVIS

extern int GLVisStreamSession(
   bool fix_elem_orient,
   bool save_coloring,
   bool keep_attr,
   bool headless,
   const std::string &plot_caption,
   const std::string &data_type,
   std::vector<std::unique_ptr<std::istream>> &&streams);

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

// keys "q" closes the window, so GLVisStreamSession can return. The watchdog
// exits the process if the SDL loop stays up.
TEST_CASE("glvis_stream session returns", "[GLVis]")
{
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction x(&fes);
   x = 1.0;

   std::stringstream payload;
   mesh.Print(payload);
   x.Save(payload);
   payload << "q\n";

   std::vector<std::unique_ptr<std::istream>> streams;
   streams.push_back(std::make_unique<std::stringstream>(payload.str()));

   std::atomic<bool> done {false};
   std::thread watchdog([&done]()
   {
      for (int i = 0; i < 60 && !done.load(); ++i)
      {
         std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      if (!done.load()) { std::_Exit(2); }
   });

   const int status = GLVisStreamSession(
                         true, true, false, false, "",
                         "fem2d_gf_data_keys", std::move(streams));
   done.store(true);
   watchdog.join();
   REQUIRE(status == 0);
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
