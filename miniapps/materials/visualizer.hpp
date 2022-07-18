#include "mfem.hpp"

namespace mfem {
namespace materials {

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

} // namespace materials
} // namespace mfem
