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
   if (scenario.find("tensor-") == 0)
   {
      const int dimension = scenario == "tensor-nonaffine-hex" ? 3 : 2;
      Mesh mesh = dimension == 3
         ? Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON)
         : Mesh::MakeCartesian2D(2, 1, scenario == "tensor-q-simplex"
                                 ? Element::TRIANGLE : Element::QUADRILATERAL,
                                 true);
      if (scenario == "tensor-mixed")
      {
         mesh = Mesh(2, 6, 3, 0, 2);
         const real_t coordinates[6][2] = {{0, 0}, {1, 0}, {1, 1},
                                           {0, 1}, {2, 0}, {2, 1}};
         for (const auto &position : coordinates) { mesh.AddVertex(position); }
         mesh.AddQuad(0, 1, 2, 3);
         mesh.AddTriangle(1, 4, 2);
         mesh.AddTriangle(4, 5, 2);
         mesh.FinalizeMesh();
      }
      if (scenario == "tensor-nonaffine-quad" ||
          scenario == "tensor-nonaffine-hex")
      { mesh.GetVertex(mesh.GetNV() - 1)[0] += 0.1; }
      if (scenario == "tensor-curved") { mesh.SetCurvature(2); }
      for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
      {
         auto *position = mesh.GetVertex(vertex);
         if ((scenario == "tensor-frame-auto" ||
              scenario == "tensor-frame-q") &&
             position[0] > 0.5)
         { position[1] += 0.3 * (position[0] - 0.5); }
         if (scenario == "tensor-degenerate") { position[1] = 0.0; }
      }
      H1_FECollection collection(1, dimension);
      FiniteElementSpace space(&mesh, &collection);
      Array<int> active(mesh.GetNE()), cut(mesh.GetNE());
      active = 1;
      cut = 1;
      auto selection = std::make_shared<GhostFaceSelection>(space, active, cut);
      ProjectionGhostPenaltyOptions options;
      if (scenario == "tensor-q-simplex" || scenario == "tensor-frame-q")
      {
         options.projection_space = ProjectionGhostPenaltySpace::TENSOR_PRODUCT;
      }
      if (scenario == "tensor-projection-space")
      {
         options.projection_space = static_cast<ProjectionGhostPenaltySpace>(-1);
      }
      for (int face = 0; face < mesh.GetNumFaces(); ++face)
      {
         auto *trans = mesh.GetInteriorFaceTransformations(face);
         if (!trans) { continue; }
         IntegrationRule rule(IntRules.Get(trans->Elem1->GetGeometryType(), 4));
         if (scenario == "tensor-order") { rule.SetOrder(0); }
         if (scenario == "tensor-weight") { rule.IntPoint(0).weight = 0.0; }
         DenseMatrix matrix;
         ProjectionGhostPenaltyKernel::Assemble(*space.GetFE(trans->Elem1No),
            *space.GetFE(trans->Elem2No), *trans->Elem1, *trans->Elem2,
            options, matrix, nullptr, &rule, &rule);
      }
      return 0;
   }
   if (scenario.find("curved-") == 0)
   {
      Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, true);
      mesh.SetCurvature(scenario == "curved-fold" ? 3 : 2);
      mesh.Transform([&](const Vector &position, Vector &value)
      {
         value = position;
         value(0) += 0.1 * position(1) * (1.0 - position(1));
         if (scenario == "curved-inverted") { value(1) *= -1.0; }
         if (scenario == "curved-singular") { value(1) = 0.0; }
         if (scenario == "curved-near-singular") { value(1) *= 1.e-16; }
         if (scenario == "curved-fold")
         {
            value(1) *= 1.0 - 5.0 * (position(0) - 0.5) * (position(0) - 0.5);
         }
      });
      if (scenario == "curved-nonfinite")
      { (*mesh.GetNodes())(0) = std::numeric_limits<real_t>::infinity(); }
      H1_FECollection collection(2, 2);
      FiniteElementSpace space(&mesh, &collection);
      ProjectionGhostPenaltyOptions options;
      options.projection_space = scenario == "curved-q" ?
         ProjectionGhostPenaltySpace::TENSOR_PRODUCT :
         ProjectionGhostPenaltySpace::TOTAL_DEGREE;
      if (scenario == "curved-overflow")
      { options.projection_degree = std::numeric_limits<int>::max(); }
      if (scenario == "curved-zero-gamma")
      {
         options.gamma = 0.0;
         options.projection_space = ProjectionGhostPenaltySpace::AUTO;
      }
      for (int face = 0; face < mesh.GetNumFaces(); ++face)
      {
         auto *trans = mesh.GetInteriorFaceTransformations(face);
         if (!trans) { continue; }
         IntegrationRule rule(IntRules.Get(Geometry::SQUARE, 14));
         if (scenario == "curved-order") { rule.SetOrder(4); }
         if (scenario == "curved-weight") { rule.IntPoint(0).weight = -1.0; }
         DenseMatrix matrix;
         ProjectionGhostPenaltyKernel::Assemble(*space.GetFE(trans->Elem1No),
            *space.GetFE(trans->Elem2No), *trans->Elem1, *trans->Elem2, options,
            matrix, nullptr, &rule, &rule);
      }
      return 0;
   }
   if (scenario == "rational-geometry" || scenario == "embedded-geometry")
   {
      H1_QuadrilateralElement element(1);
      NURBS2DFiniteElement rational(1);
      IsoparametricTransformation trans;
      trans.SetFE(scenario == "rational-geometry" ?
                  static_cast<const FiniteElement *>(&rational) : &element);
      trans.GetPointMat().SetSize(scenario == "embedded-geometry" ? 3 : 2, 4);
      trans.GetPointMat() = 0.0;
      ProjectionGhostPenaltyOptions options;
      options.projection_space = ProjectionGhostPenaltySpace::TOTAL_DEGREE;
      DenseMatrix matrix;
      ProjectionGhostPenaltyKernel::Assemble(element, element, trans, trans,
                                             options, matrix);
      return 0;
   }
   Mesh mesh = scenario == "geometry" ? Mesh::MakeCartesian1D(2)
      : Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE, true);
   if (scenario == "curved") { mesh.SetCurvature(2); }
   if (scenario == "nonconforming") { mesh.EnsureNCMesh(true); }
   H1_FECollection collection(1, mesh.Dimension());
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
