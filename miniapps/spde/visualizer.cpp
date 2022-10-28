// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

#include "visualizer.hpp"

namespace mfem {
namespace spde {

void Visualizer::ExportToParaView() {
  ParaViewDataCollection paraview_dc("SurrogateMaterial", mesh_);
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(order_);
  paraview_dc.SetCycle(0);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);
  paraview_dc.SetTime(0.0);  // set the time
  paraview_dc.RegisterField("random_field", &g1_);
  if (is_3D_) {
    paraview_dc.RegisterField("topological_support", &g2_);
    paraview_dc.RegisterField("imperfect_topology", &g3_);
    paraview_dc.RegisterField("level_set", &g4_);
  }
  paraview_dc.Save();
}

void Visualizer::SendToGLVis() {
  char vishost[] = "localhost";
  int visport = 19916;
  int num_procs = Mpi::WorldSize();
  int process_rank = Mpi::WorldRank();
  socketstream uout;
  std::ostringstream oss_u;
  uout.open(vishost, visport);
  uout.precision(8);
  oss_u.str("");
  oss_u.clear();
  oss_u << "Random Field";
  uout << "parallel " << num_procs << " " << process_rank << "\n"
       << "solution\n"
       << *mesh_ << g1_ << "window_title '" << oss_u.str() << "'" << std::flush;
  uout.close();

  if (!is_3D_) {
    return;
  }

  socketstream vout, wout, lout;
  std::ostringstream oss_v, oss_w, oss_l;
  vout.open(vishost, visport);
  vout.precision(8);
  oss_v.str("");
  oss_v.clear();
  oss_v << "Topological Support";
  vout << "parallel " << num_procs << " " << process_rank << "\n"
       << "solution\n"
       << *mesh_ << g2_ << "window_title '" << oss_v.str() << "'" << std::flush;
  vout.close();

  wout.open(vishost, visport);
  wout.precision(8);
  oss_w.str("");
  oss_w.clear();
  oss_w << "Imperfect Topology";
  wout << "parallel " << num_procs << " " << process_rank << "\n"
       << "solution\n"
       << *mesh_ << g3_ << "window_title '" << oss_w.str() << "'" << std::flush;
  wout.close();

  lout.open(vishost, visport);
  lout.precision(8);
  oss_l.str("");
  oss_l.clear();
  oss_l << "Level Set";
  lout << "parallel " << num_procs << " " << process_rank << "\n"
       << "solution\n"
       << *mesh_ << g4_ << "window_title '" << oss_l.str() << "'" << std::flush;
  lout.close();
}

}  // namespace spde
}  // namespace mfem
