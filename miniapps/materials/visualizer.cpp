#include "visualizer.hpp"

namespace mfem {
namespace materials {

void Visualizer::ExportToParaView(){
  ParaViewDataCollection paraview_dc("SurrogateMaterial", mesh_);
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(order_);
  paraview_dc.SetCycle(0);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);
  paraview_dc.SetTime(0.0); // set the time
  paraview_dc.RegisterField("random_field",&g1_);
  paraview_dc.RegisterField("topological_support",&g2_);
  paraview_dc.RegisterField("imperfect_topology",&g3_);
  paraview_dc.Save();
}

void Visualizer::SendToGLVis(){
  char vishost[] = "localhost";
  int  visport   = 19916;
  int num_procs = Mpi::WorldSize();
  int process_rank = Mpi::WorldRank();
  socketstream uout, vout, wout;
  std::ostringstream oss_u, oss_v, oss_w;
  uout.open(vishost, visport);
  uout.precision(8);
  oss_u.str(""); oss_u.clear();
  oss_u << "Random Field";
  uout << "parallel " << num_procs << " " << process_rank << "\n"
        << "solution\n" << *mesh_ << g1_
        << "window_title '" << oss_u.str() << "'" << std::flush;
  uout.close();


  vout.open(vishost, visport);
  vout.precision(8);
  oss_v.str(""); oss_v.clear();
  oss_v << "Topological Support";
  vout << "parallel " << num_procs << " " << process_rank << "\n"
        << "solution\n" << *mesh_ << g2_
        << "window_title '" << oss_v.str() << "'" << std::flush;
  vout.close();

  wout.open(vishost, visport);
  wout.precision(8);
  oss_w.str(""); oss_w.clear();
  oss_w << "Imperfect Topology";
  wout << "parallel " << num_procs << " " << process_rank << "\n"
        << "solution\n" << *mesh_ << g3_
        << "window_title '" << oss_w.str() << "'" << std::flush;
  wout.close();
}

} // namespace materials
} // namespace mfem
