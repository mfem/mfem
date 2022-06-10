#include "mfem.hpp"

namespace mfem {
namespace materials {

// ===========================================================================
// Header interface
// ===========================================================================

class Visualizer {
 public:
  Visualizer(ParMesh& mesh, int order, GridFunction& g1, GridFunction& g2, GridFunction& g3)
    : mesh_(&mesh), order_(order), g1_(g1), g2_(g2), g3_(g3) {}
 
  void ExportToParaView();
  void SendToGLVis();
 private:
  ParMesh* mesh_;
  int order_;
  GridFunction& g1_;
  GridFunction& g2_;
  GridFunction& g3_;
};

// ===========================================================================
// Implementation details
// ===========================================================================

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
  int myid = Mpi::WorldRank();
  socketstream uout, vout, wout;
  ostringstream oss_u, oss_v, oss_w;
  uout.open(vishost, visport);
  uout.precision(8);
  vout.open(vishost, visport);
  vout.precision(8);
  wout.open(vishost, visport);
  wout.precision(8);
  oss_u.str(""); oss_u.clear();
  oss_v.str(""); oss_v.clear();
  oss_w.str(""); oss_w.clear();
  oss_u << "Random Field";
  oss_v << "Topological Support";
  oss_w << "Imperfect Topology";
  uout << "parallel " << num_procs << " " << myid << "\n"
        << "solution\n" << *mesh_ << g1_
        << "window_title '" << oss_u.str() << "'" << flush;
  vout << "parallel " << num_procs << " " << myid << "\n"
        << "solution\n" << *mesh_ << g2_
        << "window_title '" << oss_v.str() << "'" << flush;
  wout << "parallel " << num_procs << " " << myid << "\n"
        << "solution\n" << *mesh_ << g3_
        << "window_title '" << oss_w.str() << "'" << flush;
}

} // namespace materials
} // namespace mfem
