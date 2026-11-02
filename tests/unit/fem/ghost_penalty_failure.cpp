#include "mfem.hpp"
#include <limits>
#include <memory>
#include <string>

using namespace mfem;

int main(int argc, char *argv[])
{
   if (argc != 2) { return 2; }
   const std::string scenario(argv[1]);
#ifdef MFEM_USE_MPI
   if (scenario == "parallel-rank")
   {
      Mpi::Init(argc, argv);
      if (Mpi::WorldRank() != 0)
      {
         MPI_Barrier(MPI_COMM_WORLD);
         return 0;
      }
   }
#endif
   if (scenario == "rank" || scenario == "near-rank" ||
       scenario == "dimensions" || scenario == "parallel-rank" ||
       scenario == "zero-rank" || scenario == "nonfinite-projection")
   {
      DenseMatrix polynomials(scenario == "dimensions" ? 1 : 3, 2);
      DenseMatrix shapes(polynomials.Height(), 1), coefficients;
      polynomials = 1.0;
      shapes = 1.0;
      if (scenario == "zero-rank") { polynomials = 0.0; }
      if (scenario == "nonfinite-projection")
      {
         polynomials(0, 0) = std::numeric_limits<real_t>::infinity();
      }
      if (scenario == "near-rank")
      {
         polynomials(0, 0) += std::numeric_limits<real_t>::epsilon();
      }
      ProjectionGhostPenaltyDiagnostics diagnostics;
      ProjectionGhostPenaltyKernel::ComputeProjection(
         polynomials, shapes, 0.0, coefficients, diagnostics,
         "failure test face=(0,1) p1=1 p2=1");
      return 0;
   }
   Mesh mesh = Mesh::MakeCartesian2D(
      1, 1, scenario == "geometry" ? Element::QUADRILATERAL : Element::TRIANGLE,
      true);
   if (scenario == "curved") { mesh.SetCurvature(2); }
   if (scenario == "nonconforming") { mesh.EnsureNCMesh(true); }
   H1_FECollection collection(1, 2);
   FiniteElementSpace space(&mesh, &collection);
   Array<int> active(mesh.GetNE()), cut(mesh.GetNE());
   active = 1;
   cut = 1;
   if (scenario == "flags") { cut.SetSize(0); }
   if (scenario == "variable-order") { space.SetElementOrder(0, 2); }
   auto selection = std::make_shared<GhostFaceSelection>(space, active, cut);
   if (scenario == "stale")
   {
      mesh.UniformRefinement();
      selection->Validate();
      return 0;
   }
   if (scenario == "vdim")
   {
      VectorProjectionGhostPenaltyIntegrator integrator(selection);
      return 0;
   }
   ProjectionGhostPenaltyOptions options;
   if (scenario == "gamma") { options.gamma = -1.0; }
   for (int face = 0; face < mesh.GetNumFaces(); ++face)
   {
      auto *trans = mesh.GetInteriorFaceTransformations(face);
      if (!trans) { continue; }
      IntegrationRule rule(IntRules.Get(Geometry::TRIANGLE, 4));
      if (scenario == "weight") { rule.IntPoint(0).weight = -1.0; }
      if (scenario == "nonfinite-weight")
      {
         rule.IntPoint(0).weight = std::numeric_limits<real_t>::infinity();
      }
      DenseMatrix matrix;
      ProjectionGhostPenaltyKernel::Assemble(
         *space.GetFE(trans->Elem1No), *space.GetFE(trans->Elem2No),
         *trans->Elem1, *trans->Elem2, options, matrix, nullptr, &rule, &rule);
      break;
   }
   return 0;
}
