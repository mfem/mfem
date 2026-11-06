#include "mfem.hpp"

#ifdef MFEM_USE_LAPACK

#include "unit_tests.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <vector>

using namespace mfem;

namespace
{

real_t Roundoff(real_t scale, int size = 1)
{
   return 512.0 * std::numeric_limits<real_t>::epsilon() * size *
          std::max(scale, std::numeric_limits<real_t>::min());
}

Mesh PatchMesh(int dimension, real_t scale = 1.0)
{
   Mesh mesh(dimension, dimension + 2, 2, 0, dimension);
   const real_t vertices[5][3] = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -2}};
   for (int vertex = 0; vertex < dimension + 2; ++vertex)
   {
      real_t position[3];
      for (int axis = 0; axis < 3; ++axis)
      {
         position[axis] = scale * vertices[vertex][axis];
      }
      if (dimension == 2 && vertex == 3) { position[1] = -2.0 * scale; }
      mesh.AddVertex(position);
   }
   if (dimension == 2)
   {
      mesh.AddTriangle(0, 1, 2);
      mesh.AddTriangle(1, 0, 3);
      mesh.FinalizeTriMesh(1, 0, true);
   }
   else
   {
      mesh.AddTet(0, 1, 2, 3);
      mesh.AddTet(0, 2, 1, 4);
      mesh.FinalizeTetMesh(1, 0, true);
   }
   return mesh;
}

void TransformTensorMesh(Mesh &mesh, int mapping, real_t scale = 1.0)
{
   for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
   {
      real_t *position = mesh.GetVertex(vertex);
      const real_t first = position[0], second = position[1];
      const real_t third = mesh.Dimension() == 3 ? position[2] : 0.0;
      if (mapping == 1)
      {
         position[0] = 0.8 * first - 0.6 * second + 2.3;
         position[1] = 0.6 * first + 0.8 * second - 0.7;
      }
      if (mapping == 2)
      {
         position[0] = first + 0.3 * second + 0.2 * third + 2.3;
         position[1] = second + 0.25 * third - 0.7;
      }
      for (int axis = 0; axis < mesh.Dimension(); ++axis)
      { position[axis] *= scale; }
   }
}

Mesh TensorPatchMesh(int dimension, int mapping = 0, real_t scale = 1.0,
                     bool permute = false)
{
   Mesh mesh = dimension == 2
      ? Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, true)
      : Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
   for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
   {
      real_t &coordinate = mesh.GetVertex(vertex)[0];
      coordinate = coordinate <= 0.5 ? 2 * coordinate : 4 * coordinate - 1;
   }
   if (permute)
   {
      Array<int> vertices;
      mesh.GetElementVertices(1, vertices);
      int *reordered = mesh.GetElement(1)->GetVertices();
      for (int vertex = 0; vertex < vertices.Size(); ++vertex)
      { reordered[vertex] = vertices[4 * (vertex / 4) + (vertex + 1) % 4]; }
      mesh.FinalizeTopology();
      mesh.Finalize();
   }
   TransformTensorMesh(mesh, mapping, scale);
   return mesh;
}

FaceElementTransformations &InteriorFace(Mesh &mesh)
{
   for (int face = 0; face < mesh.GetNumFaces(); ++face)
   {
      auto *trans = mesh.GetInteriorFaceTransformations(face);
      if (trans) { return *trans; }
   }
   MFEM_ABORT("Missing test interior face.");
   return *mesh.GetFaceElementTransformations(0);
}

void CurveMesh(Mesh &mesh, int geometry_order, real_t amplitude = 0.12)
{
   mesh.SetCurvature(geometry_order);
   mesh.Transform([=](const Vector &position, Vector &value)
   {
      value = position;
      const real_t first = position(0), second = position(1);
      value(0) += amplitude * first * second;
      value(1) += amplitude * first * second;
      if (geometry_order >= 2)
      {
         value(0) += amplitude * second * (1.0 - second);
         value(1) += 0.3 * amplitude * first * first;
      }
      if (geometry_order >= 3)
      { value(0) += 0.1 * amplitude * second * second * second; }
      if (position.Size() == 3)
      {
         value(2) += amplitude * second * position(2);
         if (geometry_order >= 2)
         { value(2) += 0.2 * amplitude * first * first; }
      }
   });
}

template <typename Form>
void AddPenalty(Form &form, std::shared_ptr<const GhostFaceSelection> selection,
                const ProjectionGhostPenaltyOptions &options = {})
{
   form.SetAssemblyLevel(AssemblyLevel::LEGACY);
   if (selection->GetVDim() == 1)
   {
      form.AddInteriorFaceIntegrator(
         new ScalarProjectionGhostPenaltyIntegrator(selection, options));
   }
   else
   {
      form.AddInteriorFaceIntegrator(
         new VectorProjectionGhostPenaltyIntegrator(selection, options));
   }
   form.Assemble();
   form.Finalize();
}

void SamplePolynomial(const FiniteElement &element,
                      ElementTransformation &trans,
                      const std::vector<int> &powers, Vector &values,
                      int offset, int mapping = 0)
{
   Vector position(element.GetDim());
   for (int node = 0; node < element.GetDof(); ++node)
   {
      trans.Transform(element.GetNodes().IntPoint(node), position);
      if (mapping != 0)
      {
         position(0) -= 2.3;
         position(1) += 0.7;
         const real_t first = position(0), second = position(1);
         if (mapping == 1)
         {
            position(0) = 0.8 * first + 0.6 * second;
            position(1) = -0.6 * first + 0.8 * second;
         }
         else
         {
            const real_t third = element.GetDim() == 3 ? position(2) : 0.0;
            position(1) -= 0.25 * third;
            position(0) -= 0.3 * position(1) + 0.2 * third;
         }
      }
      real_t value = 1.0;
      for (int axis = 0; axis < element.GetDim(); ++axis)
      {
         value *= std::pow(position(axis), powers[axis]);
      }
      values(offset + node) = value;
   }
}

}

TEST_CASE("Ghost penalty rectangular SVD projection", "[GhostPenalty]")
{
   DenseMatrix polynomials(5, 2), shapes(5, 2), coefficients;
   for (int row = 0; row < 5; ++row)
   {
      const real_t coordinate = row - 2;
      polynomials(row, 0) = 1.0;
      polynomials(row, 1) = coordinate;
      shapes(row, 0) = coordinate * coordinate + 2 * coordinate + 1;
      shapes(row, 1) = coordinate * coordinate * coordinate - coordinate;
   }
   ProjectionGhostPenaltyDiagnostics diagnostics;
   ProjectionGhostPenaltyKernel::ComputeProjection(polynomials, shapes, 0.0,
                                                   coefficients, diagnostics);
   REQUIRE(diagnostics.rank == 2);
   REQUIRE(std::abs(coefficients(0, 0) - 3.0L) < Roundoff(3.0));
   REQUIRE(std::abs(coefficients(1, 0) - 2.0L) < Roundoff(2.0));
   REQUIRE(std::abs(coefficients(0, 1)) < Roundoff(1.0));
   REQUIRE(std::abs(coefficients(1, 1) - 12.0L / 5.0L) < Roundoff(3.0));
   DenseMatrix residual(5, 2), orthogonality(2, 2);
   Mult(polynomials, coefficients, residual);
   residual -= shapes;
   MultAtB(polynomials, residual, orthogonality);
   REQUIRE(orthogonality.FNorm() <
           Roundoff(polynomials.FNorm() * shapes.FNorm(), 5));
   DenseMatrixSVD decomposition(polynomials, 'S', 'S');
   decomposition.Eval(polynomials);
   DenseMatrix left(decomposition.LeftSingularvectors()), reconstruction(5, 2);
   for (int column = 0; column < 2; ++column)
   {
      for (int row = 0; row < 5; ++row)
      {
         left(row, column) *= decomposition.Singularvalue(column);
      }
   }
   Mult(left, decomposition.RightSingularvectors(), reconstruction);
   reconstruction -= polynomials;
   REQUIRE(reconstruction.FNorm() < Roundoff(polynomials.FNorm(), 5));
}

TEST_CASE("Ghost penalty mixed-order simplex kernels", "[GhostPenalty]")
{
   for (int dimension : {2, 3})
   {
      for (int order = 1; order <= (dimension == 2 ? 5 : 3); ++order)
      {
         CAPTURE(dimension, order);
         Mesh mesh = PatchMesh(dimension);
         auto &trans = InteriorFace(mesh);
         H1_FECollection first_collection(order, dimension);
         H1_FECollection second_collection(order + 1, dimension);
         const auto &first = *first_collection.FiniteElementForGeometry(
            trans.Elem1->GetGeometryType());
         const auto &second = *second_collection.FiniteElementForGeometry(
            trans.Elem2->GetGeometryType());
         const int dofs = first.GetDof() + second.GetDof();
         ProjectionGhostPenaltyOptions options;
         ProjectionGhostPenaltyDiagnostics diagnostics;
         DenseMatrix matrix, swapped;
         ProjectionGhostPenaltyKernel::Assemble(first, second, *trans.Elem1,
                                                *trans.Elem2, options, matrix,
                                                &diagnostics);
         ProjectionGhostPenaltyKernel::Assemble(second, first, *trans.Elem2,
                                                *trans.Elem1, options, swapped);
         INFO("rank=" << diagnostics.rank
                      << " condition=" << diagnostics.condition_number);
         std::cout << "ghost kernel dim=" << dimension << " p=" << order << '/'
                   << order + 1 << " rank=" << diagnostics.rank
                   << " condition=" << diagnostics.condition_number << '\n';
         const real_t tolerance = Roundoff(matrix.FNorm(), dofs);
         REQUIRE(diagnostics.rank ==
                 (dimension == 2
                     ? (order + 1) * (order + 2) / 2
                     : (order + 1) * (order + 2) * (order + 3) / 6));
         REQUIRE(std::abs(diagnostics.face_length - std::sqrt(5.0)) <
                 Roundoff(3.0));
         for (int row = 0; row < dofs; ++row)
         {
            const int swapped_row = row < first.GetDof() ? second.GetDof() + row
                                                         : row - first.GetDof();
            for (int column = 0; column < dofs; ++column)
            {
               const int swapped_column = column < first.GetDof()
                                             ? second.GetDof() + column
                                             : column - first.GetDof();
               REQUIRE(matrix(row, column) == matrix(column, row));
               REQUIRE(std::abs(matrix(row, column) -
                                swapped(swapped_row, swapped_column)) <=
                       tolerance);
            }
         }
         Vector values(dofs), action(dofs);
         for (int first_power = 0; first_power <= order; ++first_power)
         {
            for (int second_power = 0; second_power <= order - first_power;
                 ++second_power)
            {
               const int max_third =
                  dimension == 3 ? order - first_power - second_power : 0;
               for (int third_power = 0; third_power <= max_third;
                    ++third_power)
               {
                  const std::vector<int> powers{first_power, second_power,
                                                third_power};
                  SamplePolynomial(first, *trans.Elem1, powers, values, 0);
                  SamplePolynomial(second, *trans.Elem2, powers, values,
                                   first.GetDof());
                  matrix.Mult(values, action);
                  REQUIRE(action.Norml2() <
                          Roundoff(matrix.FNorm() * values.Norml2(), dofs));
               }
            }
         }
         Vector eigenvalues;
         matrix.Eigenvalues(eigenvalues);
         REQUIRE(eigenvalues.Min() >= -tolerance);
         Mesh scaled_mesh = PatchMesh(dimension, 2.5);
         auto &scaled_trans = InteriorFace(scaled_mesh);
         DenseMatrix scaled;
         ProjectionGhostPenaltyKernel::Assemble(
            first, second, *scaled_trans.Elem1, *scaled_trans.Elem2, options,
            scaled);
         scaled *= std::pow(2.5, 2 - dimension);
         scaled -= matrix;
         REQUIRE(scaled.FNorm() < tolerance);
         options.gamma = 0.0;
         ProjectionGhostPenaltyKernel::Assemble(first, second, *trans.Elem1,
                                                *trans.Elem2, options, scaled);
         REQUIRE(scaled.FNorm() == 0.0);
      }
   }
}

TEST_CASE("Ghost penalty independent quadrature energy", "[GhostPenalty]")
{
   for (int family : {0, 1, 2, 3, 4})
   for (int dimension : {2, 3})
   {
      Mesh mesh = family == 0 || family == 3 ? PatchMesh(dimension)
                  : TensorPatchMesh(dimension);
      if (family >= 3) { CurveMesh(mesh, 2); }
      auto &trans = InteriorFace(mesh);
      H1_FECollection first_collection(2, dimension),
         second_collection(3, dimension);
      const auto &first = *first_collection.FiniteElementForGeometry(
         trans.Elem1->GetGeometryType());
      const auto &second = *second_collection.FiniteElementForGeometry(
         trans.Elem2->GetGeometryType());
      const IntegrationRule &first_rule =
         IntRules.Get(first.GetGeomType(), family >= 3 ? 12 : 4);
      const IntegrationRule &second_rule =
         IntRules.Get(second.GetGeomType(), family >= 3 ? 14 : 6);
      ProjectionGhostPenaltyOptions options;
      if (family == 1 || family >= 3)
      { options.projection_space = ProjectionGhostPenaltySpace::TOTAL_DEGREE; }
      options.projection_degree = 0;
      options.gamma = 1.7;
      ProjectionGhostPenaltyDiagnostics diagnostics;
      DenseMatrix matrix;
      ProjectionGhostPenaltyKernel::Assemble(
         first, second, *trans.Elem1, *trans.Elem2, options, matrix,
         &diagnostics, &first_rule, &second_rule);
      Vector values(first.GetDof() + second.GetDof()), action(values.Size());
      for (int index = 0; index < values.Size(); ++index)
      {
         values(index) = std::sin(1.7 * index + 0.2);
      }
      const FiniteElement *elements[] = {&first, &second};
      ElementTransformation *transforms[] = {trans.Elem1, trans.Elem2};
      const IntegrationRule *rules[] = {&first_rule, &second_rule};
      long double integral = 0.0L, volume = 0.0L;
      std::vector<long double> samples, weights;
      int offset = 0;
      for (int side = 0; side < 2; ++side)
      {
         Vector shape(elements[side]->GetDof());
         for (int point = 0; point < rules[side]->GetNPoints(); ++point)
         {
            const auto &integration_point = rules[side]->IntPoint(point);
            transforms[side]->SetIntPoint(&integration_point);
            const long double weight =
               integration_point.weight * transforms[side]->Weight();
            elements[side]->CalcShape(integration_point, shape);
            long double value = 0.0L;
            for (int basis = 0; basis < shape.Size(); ++basis)
            {
               value += static_cast<long double>(shape(basis)) *
                        values(offset + basis);
            }
            integral += weight * value;
            volume += weight;
            weights.push_back(weight);
            samples.push_back(value);
         }
         offset += shape.Size();
      }
      const long double mean = integral / volume;
      long double reference = 0.0L;
      for (size_t point = 0; point < samples.size(); ++point)
      {
         reference +=
            weights[point] * (samples[point] - mean) * (samples[point] - mean);
      }
      reference *=
         options.gamma / (diagnostics.face_length * diagnostics.face_length);
      matrix.Mult(values, action);
      REQUIRE(std::abs(values * action - reference) <
              Roundoff(reference, values.Size()));
   }
}

TEST_CASE("Ghost penalty well-conditioned mass-matrix reference",
          "[GhostPenalty]")
{
   for (int family : {0, 1, 2, 3, 4})
   for (int dimension : {2, 3})
   {
      Mesh mesh = family == 0 || family == 3 ? PatchMesh(dimension)
                  : TensorPatchMesh(dimension);
      if (family >= 3) { CurveMesh(mesh, 2); }
      auto &trans = InteriorFace(mesh);
      H1_FECollection collection(2, dimension);
      const auto &element =
         *collection.FiniteElementForGeometry(trans.Elem1->GetGeometryType());
      const int dofs = 2 * element.GetDof();
      const int polynomials = family == 2 ? 1 << dimension : dimension + 1;
      DenseMatrix mass(polynomials), coupling(polynomials, dofs),
         diagonal(dofs);
      mass = 0.0;
      coupling = 0.0;
      diagonal = 0.0;
      ElementTransformation *transforms[] = {trans.Elem1, trans.Elem2};
      const auto &rule = IntRules.Get(element.GetGeomType(), family >= 3 ? 14 : 6);
      Vector shape(element.GetDof()), position(dimension), basis(polynomials);
      for (int side = 0; side < 2; ++side)
      {
         const int offset = side * element.GetDof();
         for (int point = 0; point < rule.GetNPoints(); ++point)
         {
            const auto &integration_point = rule.IntPoint(point);
            transforms[side]->SetIntPoint(&integration_point);
            transforms[side]->Transform(integration_point, position);
            const real_t weight =
               integration_point.weight * transforms[side]->Weight();
            element.CalcShape(integration_point, shape);
            basis = 1.0;
            for (int polynomial = 1; polynomial < polynomials; ++polynomial)
            {
               if (family != 2)
               { basis(polynomial) = position(polynomial - 1); }
               else for (int axis = 0; axis < dimension; ++axis)
               {
                  if (polynomial & (1 << axis))
                  { basis(polynomial) *= position(axis); }
               }
            }
            for (int row = 0; row < polynomials; ++row)
            {
               for (int column = 0; column < polynomials; ++column)
               {
                  mass(row, column) += weight * basis(row) * basis(column);
               }
               for (int column = 0; column < shape.Size(); ++column)
               {
                  coupling(row, offset + column) +=
                     weight * basis(row) * shape(column);
               }
            }
            for (int row = 0; row < shape.Size(); ++row)
            {
               for (int column = 0; column < shape.Size(); ++column)
               {
                  diagonal(offset + row, offset + column) +=
                     weight * shape(row) * shape(column);
               }
            }
         }
      }
      DenseMatrixInverse inverse(mass);
      DenseMatrix coefficients(polynomials, dofs), projected(dofs);
      inverse.Mult(coupling, coefficients);
      MultAtB(coupling, coefficients, projected);
      diagonal -= projected;
      ProjectionGhostPenaltyOptions options;
      if (family == 1 || family >= 3)
      { options.projection_space = ProjectionGhostPenaltySpace::TOTAL_DEGREE; }
      options.projection_degree = 1;
      ProjectionGhostPenaltyDiagnostics diagnostics;
      DenseMatrix matrix;
      ProjectionGhostPenaltyKernel::Assemble(element, element, *trans.Elem1,
                                             *trans.Elem2, options, matrix,
                                             &diagnostics);
      diagonal *=
         options.gamma / (diagnostics.face_length * diagnostics.face_length);
      diagonal -= matrix;
      REQUIRE(diagonal.FNorm() < Roundoff(matrix.FNorm(), dofs));
   }
}

TEST_CASE("Ghost penalty serial vector assembly and selection",
          "[GhostPenalty]")
{
   for (int family : {0, 1, 2, 3, 4})
   for (int dimension : {2, 3})
   {
      Mesh mesh = family == 0 || family == 3 ? PatchMesh(dimension)
                  : TensorPatchMesh(dimension, 2);
      if (family >= 3) { CurveMesh(mesh, 2, 0.04); }
      ProjectionGhostPenaltyOptions options;
      if (family == 1 || family >= 3)
      { options.projection_space = ProjectionGhostPenaltySpace::TOTAL_DEGREE; }
      for (int order : {1, 2, 3})
      {
         H1_FECollection collection(order, dimension);
         FiniteElementSpace scalar_space(&mesh, &collection);
         Array<int> active(mesh.GetNE()), cut(mesh.GetNE());
         active = 1;
         cut = 0;
         cut[1] = 1;
         auto scalar_selection =
            std::make_shared<GhostFaceSelection>(scalar_space, active, cut);
         REQUIRE(scalar_selection->GetNumLocalFaces() == 1);
         BilinearForm scalar_form(&scalar_space);
         scalar_form.UsePrecomputedSparsity(order == 2);
         AddPenalty(scalar_form, scalar_selection, options);
         DenseMatrix scalar_matrix;
         scalar_form.SpMat().ToDenseMatrix(scalar_matrix);
         REQUIRE(scalar_matrix.FNorm() > 0.0);
         for (auto ordering : {Ordering::byNODES, Ordering::byVDIM})
         {
            for (int components : {2, 3, 4})
            {
               CAPTURE(dimension, order, ordering, components);
               FiniteElementSpace vector_space(&mesh, &collection, components,
                                               ordering);
               auto selection = std::make_shared<GhostFaceSelection>(
                  vector_space, active, cut);
               BilinearForm form(&vector_space);
               AddPenalty(form, selection, options);
               DenseMatrix matrix;
               form.SpMat().ToDenseMatrix(matrix);
               const real_t tolerance =
                  Roundoff(scalar_matrix.FNorm(), scalar_matrix.Height());
               for (int first_component = 0; first_component < components;
                    ++first_component)
               {
                  for (int second_component = 0; second_component < components;
                       ++second_component)
                  {
                     for (int row = 0; row < scalar_space.GetVSize(); ++row)
                     {
                        for (int column = 0; column < scalar_space.GetVSize();
                             ++column)
                        {
                           const real_t expected =
                              first_component == second_component
                                 ? scalar_matrix(row, column)
                                 : 0.0;
                           REQUIRE(
                              std::abs(matrix(vector_space.DofToVDof(
                                                 row, first_component),
                                              vector_space.DofToVDof(
                                                 column, second_component)) -
                                       expected) <= tolerance);
                        }
                     }
                  }
               }
               Array<int> empty_cut(mesh.GetNE());
               empty_cut = 0;
               auto empty_selection = std::make_shared<GhostFaceSelection>(
                  vector_space, active, empty_cut);
               VectorProjectionGhostPenaltyIntegrator integrator(
                  empty_selection);
               auto &trans = InteriorFace(mesh);
               integrator.AssembleFaceMatrix(*vector_space.GetFE(trans.Elem1No),
                                             *vector_space.GetFE(trans.Elem2No),
                                             trans, matrix);
               REQUIRE(matrix.Height() ==
                       components * (vector_space.GetFE(0)->GetDof() +
                                     vector_space.GetFE(1)->GetDof()));
               REQUIRE(matrix.FNorm() == 0.0);
            }
         }
      }
   }
}

TEST_CASE("Ghost penalty affine tensor kernels", "[GhostPenalty]")
{
   for (int dimension : {2, 3})
   for (int mapping : {0, 1, 2})
   for (auto mode : {ProjectionGhostPenaltySpace::AUTO,
                     ProjectionGhostPenaltySpace::TOTAL_DEGREE,
                     ProjectionGhostPenaltySpace::TENSOR_PRODUCT})
   for (int order = 1; order <= (dimension == 2 ? 5 : 3); ++order)
   {
      CAPTURE(dimension, mapping, order, static_cast<int>(mode));
      Mesh mesh = TensorPatchMesh(dimension, mapping);
      auto &trans = InteriorFace(mesh);
      H1_FECollection first_collection(order, dimension),
         second_collection(order + 1, dimension);
      const auto geometry = trans.Elem1->GetGeometryType();
      const auto &first = *first_collection.FiniteElementForGeometry(geometry);
      const auto &second =
         *second_collection.FiniteElementForGeometry(geometry);
      const int dofs = first.GetDof() + second.GetDof();
      ProjectionGhostPenaltyOptions options;
      options.projection_space = mode;
      ProjectionGhostPenaltyDiagnostics diagnostics;
      DenseMatrix matrix, swapped;
      ProjectionGhostPenaltyKernel::Assemble(first, second, *trans.Elem1,
         *trans.Elem2, options, matrix, &diagnostics);
      const bool tensor = mode != ProjectionGhostPenaltySpace::TOTAL_DEGREE;
      const int expected_rank = tensor
         ? static_cast<int>(std::pow(order + 1, dimension))
         : dimension == 2 ? (order + 1) * (order + 2) / 2
         : (order + 1) * (order + 2) * (order + 3) / 6;
      REQUIRE(diagnostics.rank == expected_rank);
      REQUIRE(diagnostics.projection_space == (tensor
         ? ProjectionGhostPenaltySpace::TENSOR_PRODUCT
         : ProjectionGhostPenaltySpace::TOTAL_DEGREE));
      REQUIRE(std::isfinite(diagnostics.condition_number));
      const real_t tolerance = Roundoff(matrix.FNorm(), dofs);
      ProjectionGhostPenaltyKernel::Assemble(second, first, *trans.Elem2,
         *trans.Elem1, options, swapped);
      for (int row = 0; row < dofs; ++row)
      for (int column = 0; column < dofs; ++column)
      {
         const int swapped_row = row < first.GetDof() ? second.GetDof() + row
                                                     : row - first.GetDof();
         const int swapped_column = column < first.GetDof()
            ? second.GetDof() + column : column - first.GetDof();
         REQUIRE(matrix(row, column) == matrix(column, row));
         REQUIRE(std::abs(matrix(row, column) -
                          swapped(swapped_row, swapped_column)) <= tolerance);
      }
      Vector values(dofs), action(dofs);
      for (int first_power = 0; first_power <= order; ++first_power)
      for (int second_power = 0; second_power <= order; ++second_power)
      for (int third_power = 0; third_power <= (dimension == 3 ? order : 0);
           ++third_power)
      {
         if (!tensor && first_power + second_power + third_power > order)
         { continue; }
         const std::vector<int> powers{first_power, second_power, third_power};
         SamplePolynomial(first, *trans.Elem1, powers, values, 0, mapping);
         SamplePolynomial(second, *trans.Elem2, powers, values,
                          first.GetDof(), mapping);
         matrix.Mult(values, action);
         REQUIRE(action.Norml2() <=
                 Roundoff(matrix.FNorm() * values.Norml2(), dofs));
      }
      Vector eigenvalues;
      matrix.Eigenvalues(eigenvalues);
      REQUIRE(eigenvalues.Min() >= -tolerance);
      Mesh scaled_mesh = TensorPatchMesh(dimension, mapping, 2.5);
      auto &scaled_trans = InteriorFace(scaled_mesh);
      DenseMatrix scaled;
      ProjectionGhostPenaltyKernel::Assemble(first, second, *scaled_trans.Elem1,
         *scaled_trans.Elem2, options, scaled);
      scaled *= std::pow(2.5, 2 - dimension);
      scaled -= matrix;
      REQUIRE(scaled.FNorm() <= tolerance);
      if (order <= 2)
      {
         Mesh permuted_mesh = TensorPatchMesh(dimension, mapping, 1.0, true);
         auto &permuted_trans = InteriorFace(permuted_mesh);
         DenseMatrix permuted;
         ProjectionGhostPenaltyKernel::Assemble(
            first, second, *permuted_trans.Elem1, *permuted_trans.Elem2,
            options, permuted);
         const FiniteElement *elements[] = {&first, &second};
         ElementTransformation *original[] = {trans.Elem1, trans.Elem2};
         ElementTransformation *reordered[] =
            {permuted_trans.Elem1, permuted_trans.Elem2};
         Array<int> permutation(dofs);
         permutation = -1;
         int offset = 0;
         for (int side = 0; side < 2; ++side)
         {
            Vector position(dimension), other(dimension);
            for (int node = 0; node < elements[side]->GetDof(); ++node)
            {
               original[side]->Transform(
                  elements[side]->GetNodes().IntPoint(node), position);
               for (int candidate = 0; candidate < elements[side]->GetDof();
                    ++candidate)
               {
                  reordered[side]->Transform(
                     elements[side]->GetNodes().IntPoint(candidate), other);
                  other -= position;
                  if (other.Normlinf() < Roundoff(10.0))
                  { permutation[offset + node] = offset + candidate; break; }
               }
               REQUIRE(permutation[offset + node] >= 0);
            }
            offset += elements[side]->GetDof();
         }
         for (int row = 0; row < dofs; ++row)
         for (int column = 0; column < dofs; ++column)
         {
            REQUIRE(std::abs(matrix(row, column) -
               permuted(permutation[row], permutation[column])) <= tolerance);
         }
      }
      options.gamma = 0.0;
      ProjectionGhostPenaltyKernel::Assemble(first, second, *trans.Elem1,
         *trans.Elem2, options, scaled);
      REQUIRE(scaled.FNorm() == 0.0);
   }
}

TEST_CASE("Ghost penalty P and Q reproduction policies", "[GhostPenalty]")
{
   for (int dimension : {2, 3})
   {
      Mesh mesh = TensorPatchMesh(dimension);
      auto &trans = InteriorFace(mesh);
      H1_FECollection collection(1, dimension);
      const auto &element =
         *collection.FiniteElementForGeometry(trans.Elem1->GetGeometryType());
      Vector values(2 * element.GetDof()), action(values.Size());
      const std::vector<int> powers{1, 1, dimension == 3 ? 1 : 0};
      SamplePolynomial(element, *trans.Elem1, powers, values, 0);
      SamplePolynomial(element, *trans.Elem2, powers, values, element.GetDof());
      ProjectionGhostPenaltyOptions options;
      DenseMatrix matrix;
      ProjectionGhostPenaltyKernel::Assemble(element, element, *trans.Elem1,
         *trans.Elem2, options, matrix);
      matrix.Mult(values, action);
      REQUIRE(action.Norml2() <
              Roundoff(matrix.FNorm() * values.Norml2(), values.Size()));
      options.projection_space = ProjectionGhostPenaltySpace::TOTAL_DEGREE;
      ProjectionGhostPenaltyKernel::Assemble(element, element, *trans.Elem1,
         *trans.Elem2, options, matrix);
      matrix.Mult(values, action);
      REQUIRE(values * action >
              Roundoff(matrix.FNorm() * values.Norml2() * values.Norml2()));
      for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
      {
         auto *position = mesh.GetVertex(vertex);
         if (position[0] > 1.0) { position[1] += 0.3 * (position[0] - 1.0); }
      }
      auto &mismatched = InteriorFace(mesh);
      ProjectionGhostPenaltyKernel::Assemble(
         element, element, *mismatched.Elem1, *mismatched.Elem2, options, matrix);
      REQUIRE(matrix.FNorm() > 0.0);
      SamplePolynomial(element, *mismatched.Elem1, {0, 1, 0}, values, 0);
      SamplePolynomial(element, *mismatched.Elem2, {0, 1, 0}, values,
                       element.GetDof());
      matrix.Mult(values, action);
      REQUIRE(action.Norml2() <
              Roundoff(matrix.FNorm() * values.Norml2(), values.Size()));
   }
}

TEST_CASE("Ghost penalty curved physical P kernels", "[GhostPenalty][Curved]")
{
   for (int dimension : {2, 3})
   for (bool tensor : {false, true})
   for (int geometry_order : {1, 2, 3})
   for (int degree : {1, 2})
   {
      if ((!tensor && geometry_order == 1) ||
          (degree == 2 && geometry_order != 2)) { continue; }
      CAPTURE(dimension, tensor, geometry_order, degree);
      Mesh mesh = tensor ? TensorPatchMesh(dimension) : PatchMesh(dimension);
      CurveMesh(mesh, geometry_order);
      auto &trans = InteriorFace(mesh);
      const int order = geometry_order * degree;
      H1_FECollection first_collection(order, dimension),
         second_collection(order + 1, dimension);
      const auto geometry = trans.Elem1->GetGeometryType();
      const auto &first = *first_collection.FiniteElementForGeometry(geometry);
      const auto &second = *second_collection.FiniteElementForGeometry(geometry);
      ProjectionGhostPenaltyOptions options;
      options.projection_space = ProjectionGhostPenaltySpace::TOTAL_DEGREE;
      options.projection_degree = degree;
      options.quadrature_extra = 0;
      ProjectionGhostPenaltyDiagnostics diagnostics, reversed;
      DenseMatrix matrix, swapped, refined;
      ProjectionGhostPenaltyKernel::Assemble(first, second, *trans.Elem1,
         *trans.Elem2, options, matrix, &diagnostics);
      ProjectionGhostPenaltyKernel::Assemble(second, first, *trans.Elem2,
         *trans.Elem1, options, swapped, &reversed);
      const int dofs = matrix.Height();
      const real_t tolerance = Roundoff(matrix.FNorm(), dofs);
      for (int row = 0; row < dofs; ++row)
      {
         const int reversed_row = row < first.GetDof() ? row + second.GetDof()
                                  : row - first.GetDof();
         for (int column = 0; column < dofs; ++column)
         {
            const int reversed_column = column < first.GetDof() ?
                                        column + second.GetDof() :
                                        column - first.GetDof();
            REQUIRE(matrix(row, column) == matrix(column, row));
            REQUIRE(std::abs(matrix(row, column) -
                             swapped(reversed_row, reversed_column)) < tolerance);
         }
      }
      Vector eigenvalues;
      matrix.Eigenvalues(eigenvalues);
      REQUIRE(eigenvalues.Min() >= -tolerance);
      Vector values(dofs), action(dofs);
      for (int first_power = 0; first_power <= degree; ++first_power)
      for (int second_power = 0; second_power <= degree - first_power;
           ++second_power)
      for (int third_power = 0; third_power <= (dimension == 3 ?
           degree - first_power - second_power : 0); ++third_power)
      {
         const std::vector<int> powers{first_power, second_power, third_power};
         SamplePolynomial(first, *trans.Elem1, powers, values, 0);
         SamplePolynomial(second, *trans.Elem2, powers, values, first.GetDof());
         matrix.Mult(values, action);
         REQUIRE(action.Norml2() < tolerance * values.Norml2());
      }
      ElementTransformation *transforms[] = {trans.Elem1, trans.Elem2};
      const auto &reference_rule = IntRules.Get(geometry, 18);
      for (int side = 0; side < 2; ++side)
      {
         REQUIRE(diagnostics.geometry_orders[side] == geometry_order);
         const int weight_order = tensor ? dimension * geometry_order - 1 :
                                  dimension * (geometry_order - 1);
         REQUIRE(diagnostics.required_orders[side] ==
                 2 * (order + side) + weight_order);
         REQUIRE(diagnostics.quadrature_orders[side] >=
                 diagnostics.required_orders[side]);
         REQUIRE(diagnostics.minimum_jacobians[side] > 0.0);
         REQUIRE(diagnostics.volumes[side] == reversed.volumes[1 - side]);
         real_t volume = 0.0;
         Vector centroid(dimension), position(dimension);
         centroid = 0.0;
         for (int point = 0; point < reference_rule.GetNPoints(); ++point)
         {
            const auto &integration_point = reference_rule.IntPoint(point);
            transforms[side]->SetIntPoint(&integration_point);
            const real_t weight = integration_point.weight *
                                  transforms[side]->Weight();
            transforms[side]->Transform(integration_point, position);
            volume += weight;
            centroid.Add(weight, position);
         }
         centroid /= volume;
         centroid -= diagnostics.centroids[side];
         REQUIRE(centroid.Norml2() < Roundoff(diagnostics.face_length, 10));
         REQUIRE(std::abs(volume - diagnostics.volumes[side]) <
                 Roundoff(volume, 10));
      }
      options.quadrature_extra = 6;
      ProjectionGhostPenaltyKernel::Assemble(first, second, *trans.Elem1,
         *trans.Elem2, options, refined);
      refined -= matrix;
      REQUIRE(refined.FNorm() < tolerance);
      mesh.Transform([](const Vector &position, Vector &value)
      {
         value = position;
         value(0) = 0.8 * position(0) - 0.6 * position(1);
         value(1) = 0.6 * position(0) + 0.8 * position(1);
         value *= 2.5;
         value(0) += 3.1;
      });
      auto &scaled_trans = InteriorFace(mesh);
      ProjectionGhostPenaltyKernel::Assemble(first, second, *scaled_trans.Elem1,
         *scaled_trans.Elem2, options, refined);
      refined *= std::pow(2.5, 2 - dimension);
      refined -= matrix;
      REQUIRE(refined.FNorm() < tolerance);
      options.gamma = 0.0;
      ProjectionGhostPenaltyKernel::Assemble(first, second, *scaled_trans.Elem1,
         *scaled_trans.Elem2, options, refined);
      REQUIRE(refined.FNorm() == 0.0);
   }
}

TEST_CASE("Ghost penalty large coordinate offsets", "[GhostPenalty][Curved]")
{
   for (int dimension : {2, 3})
   for (int mapping : {0, 1, 2, 3})
   {
      CAPTURE(dimension, mapping);
      Mesh mesh = dimension == 2 ?
         Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, true) :
         Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
      if (mapping == 2) { mesh.SetCurvature(2); }
      mesh.Transform([=](const Vector &position, Vector &value)
      {
         value = position;
         if (mapping == 0) { value(0) += 0.25 * position(1); }
         if (mapping == 1 || mapping == 2)
         { value(0) += 0.125 * position(0) * position(1); }
         if (mapping == 2)
         { value(0) += 0.25 * position(1) * (1.0 - position(1)); }
         if (mapping == 3)
         { value(1) += 0.125 * std::max(real_t(0.0), position(0) - 0.5); }
      });
      H1_FECollection collection(2, dimension);
      auto &trans = InteriorFace(mesh);
      const auto &element = *collection.FiniteElementForGeometry(
         trans.Elem1->GetGeometryType());
      ProjectionGhostPenaltyOptions options;
      options.projection_space = mapping == 0 ?
         ProjectionGhostPenaltySpace::AUTO :
         ProjectionGhostPenaltySpace::TOTAL_DEGREE;
      options.projection_degree = 2;
      options.quadrature_extra = 0;
      ProjectionGhostPenaltyDiagnostics original, translated;
      DenseMatrix matrix, shifted, refined, swapped;
      ProjectionGhostPenaltyKernel::Assemble(element, element, *trans.Elem1,
         *trans.Elem2, options, matrix, &original);
      const real_t offset = MFEM_PRECISION_SWITCH(1.e12, 1.e5);
      mesh.Transform([=](const Vector &position, Vector &value)
      {
         value = position;
         value(0) += offset;
         value(1) -= 0.5 * offset;
         if (dimension == 3) { value(2) += 0.25 * offset; }
      });
      auto &shifted_trans = InteriorFace(mesh);
      ProjectionGhostPenaltyKernel::Assemble(element, element,
         *shifted_trans.Elem1, *shifted_trans.Elem2, options, shifted, &translated);
      const real_t tolerance = Roundoff(matrix.FNorm(), matrix.Height());
      REQUIRE(original.projection_space == translated.projection_space);
      REQUIRE(std::abs(original.face_length - translated.face_length) <
              Roundoff(original.face_length));
      for (int side = 0; side < 2; ++side)
      {
         REQUIRE(original.required_orders[side] == translated.required_orders[side]);
         REQUIRE(std::abs(original.volumes[side] - translated.volumes[side]) <
                 Roundoff(original.volumes[side]));
      }
      shifted -= matrix;
      REQUIRE(shifted.FNorm() < tolerance);
      ProjectionGhostPenaltyKernel::Assemble(element, element,
         *shifted_trans.Elem2, *shifted_trans.Elem1, options, swapped);
      for (int row = 0; row < matrix.Height(); ++row)
      for (int column = 0; column < matrix.Width(); ++column)
      {
         const int swapped_row = (row + element.GetDof()) % matrix.Height();
         const int swapped_column = (column + element.GetDof()) % matrix.Width();
         REQUIRE(std::abs(matrix(row, column) -
                          swapped(swapped_row, swapped_column)) < tolerance);
      }
      options.quadrature_extra = 6;
      ProjectionGhostPenaltyKernel::Assemble(element, element,
         *shifted_trans.Elem1, *shifted_trans.Elem2, options, refined);
      refined -= matrix;
      REQUIRE(refined.FNorm() < tolerance);
   }
}

TEST_CASE("Ghost penalty curved geometry metrics", "[GhostPenalty][Curved]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, true);
   mesh.SetCurvature(2);
   mesh.Transform([](const Vector &position, Vector &value)
   {
      value = position;
      value(0) += 4.0 * position(1) * (1.0 - position(1));
   });
   auto &trans = InteriorFace(mesh);
   H1_QuadrilateralElement element(2);
   ProjectionGhostPenaltyOptions options;
   options.projection_space = ProjectionGhostPenaltySpace::TOTAL_DEGREE;
   options.projection_degree = 1;
   ProjectionGhostPenaltyDiagnostics diagnostics;
   DenseMatrix matrix;
   ProjectionGhostPenaltyKernel::Assemble(element, element, *trans.Elem1,
      *trans.Elem2, options, matrix, &diagnostics);
   for (int side = 0; side < 2; ++side)
   {
      REQUIRE(std::abs(diagnostics.volumes[side] - 0.5) < Roundoff(1.0));
      REQUIRE(std::abs(diagnostics.centroids[side](0) -
                       (0.25 + 0.5 * side + 4.0 / 6.0)) < Roundoff(2.0));
      REQUIRE(std::abs(diagnostics.centroids[side](1) - 0.5) < Roundoff(1.0));
   }
   REQUIRE(diagnostics.face_length > std::sqrt(1.25));
   GeometryRefiner refiner;
   const auto &points = refiner.Refine(Geometry::SQUARE, 16)->RefPts;
   DenseMatrix physical;
   trans.Elem1->Transform(points, physical);
   real_t fine_diameter = 0.0;
   for (int first = 0; first < points.GetNPoints(); ++first)
   for (int second = 0; second < first; ++second)
   {
      fine_diameter = std::max(fine_diameter,
                              std::hypot(physical(0, first) - physical(0, second),
                                         physical(1, first) - physical(1, second)));
   }
   REQUIRE(fine_diameter >= diagnostics.face_length - Roundoff(fine_diameter));
   REQUIRE(fine_diameter < 1.05 * diagnostics.face_length);
   Mesh variable = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, true);
   variable.SetCurvature(2);
   variable.Transform([](const Vector &position, Vector &value)
   {
      value = position;
      value(0) *= 1.0 + 0.2 * position(1);
      value(1) *= 1.0 + 0.2 * position(1);
   });
   auto &variable_trans = InteriorFace(variable);
   ProjectionGhostPenaltyKernel::Assemble(element, element,
      *variable_trans.Elem1, *variable_trans.Elem2, options, matrix, &diagnostics);
   const real_t exact_volume = 0.5 * (1.0 + 0.3 + 0.08 / 3.0);
   REQUIRE(std::abs(diagnostics.volumes[0] - exact_volume) < Roundoff(1.0));
   variable_trans.Elem1->SetIntPoint(&Geometries.GetCenter(Geometry::SQUARE));
   REQUIRE(std::abs(variable_trans.Elem1->Weight() - exact_volume) > 1.e-3);
}

TEST_CASE("Ghost penalty mixed geometry orders", "[GhostPenalty][Curved]")
{
   for (int dimension : {2, 3})
   for (bool tensor : {false, true})
   {
      Mesh affine = tensor ? TensorPatchMesh(dimension) : PatchMesh(dimension);
      Mesh curved(affine);
      curved.SetCurvature(2);
      curved.Transform([=](const Vector &position, Vector &value)
      {
         value = position;
         if (tensor)
         {
            value(0) += 0.04 * (position(0) - 1.0) *
                        position(1) * (1.0 - position(1));
         }
         else { value(dimension - 1) += 0.04 * std::pow(position(dimension - 1), 2); }
      });
      auto &first_trans = InteriorFace(affine);
      auto &second_trans = InteriorFace(curved);
      H1_FECollection collection(2, dimension);
      const auto &element = *collection.FiniteElementForGeometry(
         first_trans.Elem1->GetGeometryType());
      ProjectionGhostPenaltyOptions options;
      options.projection_space = ProjectionGhostPenaltySpace::TOTAL_DEGREE;
      options.projection_degree = 1;
      ProjectionGhostPenaltyDiagnostics diagnostics;
      DenseMatrix matrix;
      ProjectionGhostPenaltyKernel::Assemble(element, element, *first_trans.Elem1,
         *second_trans.Elem2, options, matrix, &diagnostics);
      REQUIRE(diagnostics.geometry_orders[0] == 1);
      REQUIRE(diagnostics.geometry_orders[1] == 2);
      REQUIRE(diagnostics.required_orders[0] < diagnostics.required_orders[1]);
      Vector values(matrix.Height()), action(matrix.Height());
      for (int axis = 0; axis < dimension; ++axis)
      {
         std::vector<int> powers(dimension, 0);
         powers[axis] = 1;
         SamplePolynomial(element, *first_trans.Elem1, powers, values, 0);
         SamplePolynomial(element, *second_trans.Elem2, powers, values,
                          element.GetDof());
         matrix.Mult(values, action);
         REQUIRE(action.Norml2() < Roundoff(matrix.FNorm() * values.Norml2(),
                                           values.Size()));
      }
   }
}

TEST_CASE("Ghost penalty curved tensor reference orientation",
          "[GhostPenalty][Curved]")
{
   for (int dimension : {2, 3})
   {
      Mesh original = TensorPatchMesh(dimension);
      Mesh permuted = TensorPatchMesh(dimension, 0, 1.0, true);
      CurveMesh(original, 2);
      CurveMesh(permuted, 2);
      auto &first_trans = InteriorFace(original);
      auto &second_trans = InteriorFace(permuted);
      H1_FECollection collection(2, dimension);
      const auto &element = *collection.FiniteElementForGeometry(
         first_trans.Elem1->GetGeometryType());
      ProjectionGhostPenaltyOptions options;
      options.projection_space = ProjectionGhostPenaltySpace::TOTAL_DEGREE;
      options.projection_degree = 1;
      DenseMatrix first, second;
      ProjectionGhostPenaltyKernel::Assemble(element, element, *first_trans.Elem1,
         *first_trans.Elem2, options, first);
      ProjectionGhostPenaltyKernel::Assemble(element, element, *second_trans.Elem1,
         *second_trans.Elem2, options, second);
      Array<int> permutation(first.Height());
      ElementTransformation *original_sides[] = {first_trans.Elem1, first_trans.Elem2};
      ElementTransformation *permuted_sides[] = {second_trans.Elem1, second_trans.Elem2};
      for (int side = 0; side < 2; ++side)
      {
         DenseMatrix original_points, permuted_points;
         original_sides[side]->Transform(element.GetNodes(), original_points);
         permuted_sides[side]->Transform(element.GetNodes(), permuted_points);
         for (int node = 0; node < element.GetDof(); ++node)
         {
            int match = -1;
            for (int candidate = 0; candidate < element.GetDof(); ++candidate)
            {
               real_t distance = 0.0;
               for (int axis = 0; axis < dimension; ++axis)
               {
                  distance = std::hypot(distance, original_points(axis, node) -
                                        permuted_points(axis, candidate));
               }
               if (distance < Roundoff(4.0)) { match = candidate; }
            }
            REQUIRE(match >= 0);
            permutation[side * element.GetDof() + node] =
               side * element.GetDof() + match;
         }
      }
      const real_t tolerance = Roundoff(first.FNorm(), first.Height());
      for (int row = 0; row < first.Height(); ++row)
      for (int column = 0; column < first.Width(); ++column)
      {
         REQUIRE(std::abs(first(row, column) -
                          second(permutation[row], permutation[column])) < tolerance);
      }
   }
}

TEST_CASE("Ghost penalty curved representability and straight geometry",
          "[GhostPenalty][Curved]")
{
   for (int dimension : {2, 3})
   for (bool tensor : {false, true})
   {
      Mesh mesh = tensor ? TensorPatchMesh(dimension) : PatchMesh(dimension);
      H1_FECollection collection(1, dimension);
      auto &trans = InteriorFace(mesh);
      const auto &element = *collection.FiniteElementForGeometry(
         trans.Elem1->GetGeometryType());
      ProjectionGhostPenaltyOptions options;
      options.projection_space = ProjectionGhostPenaltySpace::TOTAL_DEGREE;
      DenseMatrix affine, straight, curved;
      ProjectionGhostPenaltyKernel::Assemble(element, element, *trans.Elem1,
         *trans.Elem2, options, affine);
      mesh.SetCurvature(2);
      auto &straight_trans = InteriorFace(mesh);
      ProjectionGhostPenaltyKernel::Assemble(element, element,
         *straight_trans.Elem1, *straight_trans.Elem2, options, straight);
      straight -= affine;
      REQUIRE(straight.FNorm() < Roundoff(affine.FNorm(), affine.Height()));
      CurveMesh(mesh, 2);
      auto &curved_trans = InteriorFace(mesh);
      ProjectionGhostPenaltyKernel::Assemble(element, element,
         *curved_trans.Elem1, *curved_trans.Elem2, options, curved);
      Vector values(curved.Height()), action(curved.Height());
      const std::vector<int> powers{1, 0, 0};
      SamplePolynomial(element, *curved_trans.Elem1, powers, values, 0);
      SamplePolynomial(element, *curved_trans.Elem2, powers, values,
                       element.GetDof());
      curved.Mult(values, action);
      REQUIRE(values * action > Roundoff(curved.FNorm() * (values * values),
                                         values.Size()));
      values = 1.0;
      curved.Mult(values, action);
      REQUIRE(action.Norml2() < Roundoff(curved.FNorm() * values.Norml2(),
                                        values.Size()));
   }
}

#ifdef MFEM_USE_MPI
namespace
{

void MarkMesh(Mesh &mesh, int scenario, Array<int> &active, Array<int> &cut)
{
   active.SetSize(mesh.GetNE());
   cut.SetSize(mesh.GetNE());
   Vector center(mesh.Dimension());
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      mesh.GetElementCenter(element, center);
      active[element] = scenario != 1 || center(0) < 0.75;
      cut[element] = scenario == 0 || (scenario == 1 && center(0) < 0.27);
   }
}

std::vector<std::vector<long long>> CoordinateKeys(FiniteElementSpace &space)
{
   auto *parallel_space = dynamic_cast<ParFiniteElementSpace *>(&space);
   std::unique_ptr<GridFunction> field;
   if (parallel_space) { field.reset(new ParGridFunction(parallel_space)); }
   else
   {
      field.reset(new GridFunction(&space));
   }
   std::vector<std::vector<long long>> keys(space.GetTrueVSize());
   for (int axis = 0; axis <= space.GetMesh()->Dimension(); ++axis)
   {
      VectorFunctionCoefficient coefficient(
         space.GetVDim(),
         [&](const Vector &position, Vector &value)
         {
            for (int component = 0; component < value.Size(); ++component)
            {
               value(component) =
                  axis < position.Size() ? position(axis) : component;
            }
         });
      field->ProjectCoefficient(coefficient);
      Vector values;
      if (parallel_space)
      {
         static_cast<ParGridFunction *>(field.get())->GetTrueDofs(values);
      }
      else
      {
         field->GetTrueDofs(values);
      }
      for (int row = 0; row < values.Size(); ++row)
      {
         keys[row].push_back(std::llround(1.e8 * values(row)));
      }
   }
   return keys;
}

void ParallelComparison(int dimension, int order, int components,
                        Ordering::Type ordering, int scenario, int layers,
                        MPI_Comm communicator, bool tensor = false,
                        const ProjectionGhostPenaltyOptions &options = {},
                        int mapping = 0, int geometry_order = 0)
{
   int ranks;
   MPI_Comm_size(communicator, &ranks);
   Mesh mesh = dimension == 2
                  ? Mesh::MakeCartesian2D(4, 1, tensor ? Element::QUADRILATERAL
                                          : Element::TRIANGLE, true)
                  : Mesh::MakeCartesian3D(4, 1, 1, tensor ? Element::HEXAHEDRON
                                          : Element::TETRAHEDRON);
   Array<int> partition(mesh.GetNE());
   Vector center(dimension);
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      mesh.GetElementCenter(element, center);
      partition[element] =
         std::min(ranks - 1, static_cast<int>(center(0) * ranks));
   }
   TransformTensorMesh(mesh, mapping);
   if (geometry_order) { CurveMesh(mesh, geometry_order); }
   H1_FECollection collection(order, dimension);
   FiniteElementSpace serial_space(&mesh, &collection, components, ordering);
   Array<int> active, cut;
   MarkMesh(mesh, scenario, active, cut);
   auto serial_selection =
      std::make_shared<GhostFaceSelection>(serial_space, active, cut, layers);
   BilinearForm serial_form(&serial_space);
   AddPenalty(serial_form, serial_selection, options);
   DenseMatrix serial_matrix;
   serial_form.SpMat().ToDenseMatrix(serial_matrix);
   const auto serial_keys = CoordinateKeys(serial_space);
   std::map<std::vector<long long>, int> serial_indices;
   for (int row = 0; row < serial_space.GetTrueVSize(); ++row)
   {
      REQUIRE(serial_indices.emplace(serial_keys[row], row).second);
   }
   ParMesh parallel_mesh(communicator, mesh, partition);
   ParFiniteElementSpace parallel_space(&parallel_mesh, &collection, components,
                                        ordering);
   MarkMesh(parallel_mesh, scenario, active, cut);
   auto parallel_selection =
      std::make_shared<GhostFaceSelection>(parallel_space, active, cut, layers);
   const real_t local_count = parallel_selection->GetNumLocalFaces() +
                              0.5 * parallel_selection->GetNumSharedFaces();
   real_t global_count;
   MPI_Allreduce(&local_count, &global_count, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, communicator);
   REQUIRE(global_count == serial_selection->GetNumLocalFaces());
   ParBilinearForm form(&parallel_space);
   form.KeepNbrBlock(false);
   form.UsePrecomputedSparsity(order == 2);
   AddPenalty(form, parallel_selection, options);
   std::unique_ptr<HypreParMatrix> matrix(form.ParallelAssemble());
   const auto parallel_keys = CoordinateKeys(parallel_space);
   Array<int> indices(parallel_space.GetTrueVSize());
   for (int row = 0; row < indices.Size(); ++row)
   {
      auto found = serial_indices.find(parallel_keys[row]);
      REQUIRE(found != serial_indices.end());
      indices[row] = found->second;
   }
   real_t local_error = 0.0;
   Vector values(indices.Size()), action(indices.Size()),
      transpose(indices.Size());
   for (int column = 0; column < serial_matrix.Width(); ++column)
   {
      for (int row = 0; row < values.Size(); ++row)
      {
         values(row) = indices[row] == column ? 1.0 : 0.0;
      }
      matrix->Mult(values, action);
      for (int row = 0; row < values.Size(); ++row)
      {
         local_error = std::max(
            local_error,
            std::abs(action(row) - serial_matrix(indices[row], column)));
      }
   }
   real_t global_error;
   MPI_Allreduce(&local_error, &global_error, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, communicator);
   REQUIRE(global_error <=
           Roundoff(serial_matrix.FNorm(), serial_matrix.Height()));
   Vector serial_values(serial_matrix.Height()),
      serial_action(serial_matrix.Height());
   for (int row = 0; row < serial_values.Size(); ++row)
   {
      serial_values(row) = std::sin(0.713 * row + 0.31);
   }
   for (int row = 0; row < values.Size(); ++row)
   {
      values(row) = serial_values(indices[row]);
   }
   serial_matrix.Mult(serial_values, serial_action);
   matrix->Mult(values, action);
   const real_t local_energy = values * action;
   real_t energy;
   MPI_Allreduce(&local_energy, &energy, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, communicator);
   REQUIRE(std::abs(energy - serial_values * serial_action) <=
           Roundoff(serial_matrix.FNorm() * serial_values.Norml2() *
                       serial_values.Norml2(),
                    serial_values.Size()));
   REQUIRE(energy >= -Roundoff(serial_matrix.FNorm() * serial_values.Norml2() *
                                  serial_values.Norml2(),
                               serial_values.Size()));
   matrix->MultTranspose(values, transpose);
   transpose -= action;
   real_t local_asymmetry = transpose.Normlinf(), asymmetry;
   MPI_Allreduce(&local_asymmetry, &asymmetry, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, communicator);
   REQUIRE(asymmetry <= Roundoff(serial_matrix.FNorm() * serial_values.Norml2(),
                                 serial_values.Size()));
   values = 1.0;
   matrix->Mult(values, action);
   real_t local_null = action.Normlinf(), null_error;
   MPI_Allreduce(&local_null, &null_error, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, communicator);
   REQUIRE(null_error <= Roundoff(serial_matrix.FNorm(), serial_values.Size()));
   if (scenario == 0 && ranks > 1)
   {
      SparseMatrix off_diagonal;
      HYPRE_BigInt *column_map;
      matrix->GetOffd(off_diagonal, column_map);
      real_t local_maximum = 0.0, maximum;
      for (int entry = 0; entry < off_diagonal.NumNonZeroElems(); ++entry)
      {
         local_maximum =
            std::max(local_maximum, std::abs(off_diagonal.GetData()[entry]));
      }
      MPI_Allreduce(&local_maximum, &maximum, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, communicator);
      REQUIRE(maximum > Roundoff(serial_matrix.FNorm()));
   }
   if (scenario == 1 && ranks == 4 && layers == 0)
   {
      const int empty = parallel_selection->GetNumLocalFaces() +
                           parallel_selection->GetNumSharedFaces() ==
                        0;
      int empty_ranks;
      MPI_Allreduce(&empty, &empty_ranks, 1, MPI_INT, MPI_SUM, communicator);
      REQUIRE(empty_ranks > 0);
      int local_active = 0;
      for (int element = 0; element < active.Size(); ++element)
      { local_active += active[element]; }
      const int inactive = local_active == 0;
      int inactive_ranks;
      MPI_Allreduce(&inactive, &inactive_ranks, 1, MPI_INT, MPI_SUM, communicator);
      REQUIRE(inactive_ranks > 0);
   }
}

}

TEST_CASE("Ghost penalty distributed true-DOF matrices",
          "[GhostPenalty][Parallel]")
{
   for (int dimension : {2, 3})
   {
      for (int order : {1, 2})
      {
         for (auto ordering : {Ordering::byNODES, Ordering::byVDIM})
         {
            for (int components : {1, 3})
            {
               CAPTURE(dimension, order, ordering, components);
               ParallelComparison(dimension, order, components, ordering, 0, 0,
                                  MPI_COMM_WORLD);
            }
         }
      }
   }
   for (int scenario : {1, 2})
   {
      ParallelComparison(2, 2, 2, Ordering::byVDIM, scenario, 0,
                         MPI_COMM_WORLD);
      ParallelComparison(3, 1, 1, Ordering::byNODES, scenario, 1,
                         MPI_COMM_WORLD);
   }
   ParallelComparison(2, 1, 1, Ordering::byNODES, 0, 0, MPI_COMM_SELF);
}

TEST_CASE("Ghost penalty distributed tensor true-DOF matrices",
          "[GhostPenalty][Parallel]")
{
   for (int dimension : {2, 3})
   for (int order : {1, 2})
   for (auto ordering : {Ordering::byNODES, Ordering::byVDIM})
   for (int components : {1, 3})
   for (auto mode : {ProjectionGhostPenaltySpace::AUTO,
                     ProjectionGhostPenaltySpace::TOTAL_DEGREE,
                     ProjectionGhostPenaltySpace::TENSOR_PRODUCT})
   {
      CAPTURE(dimension, order, ordering, components, static_cast<int>(mode));
      ProjectionGhostPenaltyOptions options;
      options.projection_space = mode;
      ParallelComparison(dimension, order, components, ordering, 0, 0,
                         MPI_COMM_WORLD, true, options, 2);
   }
   for (int dimension : {2, 3})
   for (int scenario : {1, 2})
   for (int layers : {0, 1})
   {
      ParallelComparison(dimension, 2, 2, Ordering::byVDIM, scenario, layers,
                         MPI_COMM_WORLD, true);
   }
   ParallelComparison(2, 1, 1, Ordering::byNODES, 0, 0, MPI_COMM_SELF, true);
}

TEST_CASE("Ghost penalty distributed curved true-DOF matrices",
          "[GhostPenalty][Curved][Parallel]")
{
   ProjectionGhostPenaltyOptions options;
   options.projection_space = ProjectionGhostPenaltySpace::TOTAL_DEGREE;
   options.projection_degree = 1;
   for (int dimension : {2, 3})
   for (bool tensor : {false, true})
   for (int geometry_order : {1, 2, 3})
   {
      if (!tensor && geometry_order == 1) { continue; }
      for (auto ordering : {Ordering::byNODES, Ordering::byVDIM})
      for (int components : {1, 2})
      {
         CAPTURE(dimension, tensor, geometry_order, ordering, components);
         ParallelComparison(dimension, 2, components, ordering, 0, 0,
                            MPI_COMM_WORLD, tensor, options, 0, geometry_order);
      }
      for (int scenario : {1, 2})
      {
         ParallelComparison(dimension, 1, 2, Ordering::byVDIM, scenario, 1,
                            MPI_COMM_WORLD, tensor, options, 0, geometry_order);
      }
   }
   ParallelComparison(2, 2, 2, Ordering::byNODES, 0, 0, MPI_COMM_SELF,
                      true, options, 0, 2);
}
#endif

#endif
