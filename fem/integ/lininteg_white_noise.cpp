#include "../fem.hpp"
#include "../../general/forall.hpp"
#include <cmath>
#include <limits>

namespace mfem
{
namespace
{

int WhiteNoiseSize(std::int64_t size)
{
   MFEM_VERIFY(size >= 0 && size <= std::numeric_limits<int>::max(),
               "White-noise element storage exceeds the supported size.");
   return int(size);
}

int WhiteNoiseTensorSize(int size_1d, int dimension)
{
   MFEM_VERIFY(size_1d > 0, "Empty tensor basis or quadrature rule.");
   int size = 1;
   for (int axis = 0; axis < dimension; ++axis)
   {
      size = WhiteNoiseSize(std::int64_t(size) * size_1d);
   }
   return size;
}

void WhiteNoiseTensorFirst(const DofToQuad &maps, int elements, int components,
                          const Array<int> &markers, const Vector &weights,
                          const Vector &normal_q, Vector &work)
{
   const int dofs_1d = maps.ndof;
   const int points_1d = maps.nqpt;
   const int points = weights.Size() / elements;
   const int remaining_points = points / points_1d;
   const auto basis = Reshape(maps.B.Read(), points_1d, dofs_1d);
   const auto scale = Reshape(weights.Read(), points, elements);
   const auto normals = Reshape(normal_q.Read(), components, points, elements);
   const int *active = markers.Read();
   real_t *result = work.Write();
   mfem::forall(work.Size(), [=] MFEM_HOST_DEVICE(int entry)
   {
      const int dof = entry % dofs_1d;
      const int remaining = (entry / dofs_1d) % remaining_points;
      const int component = (entry / dofs_1d / remaining_points) % components;
      const int element = entry / dofs_1d / remaining_points / components;
      if (!active[element]) { return; }
      real_t value = 0.0;
      for (int point = 0; point < points_1d; ++point)
      {
         const int quad = point + points_1d * remaining;
         value += basis(point, dof) * scale(quad, element) *
                  normals(component, quad, element);
      }
      result[entry] = value;
   });
}

void WhiteNoiseTensorAxis(const DofToQuad &maps, int elements, int left_size,
                         const Array<int> &markers, const Vector &input,
                         Vector &output, bool add)
{
   const int dofs_1d = maps.ndof;
   const int points_1d = maps.nqpt;
   const int element_size = output.Size() / elements;
   const auto basis = Reshape(maps.B.Read(), points_1d, dofs_1d);
   const real_t *source = input.Read();
   const int *active = markers.Read();
   real_t *result = add ? output.ReadWrite() : output.Write();
   mfem::forall(output.Size(), [=] MFEM_HOST_DEVICE(int entry)
   {
      if (!active[entry / element_size]) { return; }
      const int left = entry % left_size;
      const int dof = (entry / left_size) % dofs_1d;
      const int right = entry / left_size / dofs_1d;
      real_t value = 0.0;
      for (int point = 0; point < points_1d; ++point)
      {
         value += basis(point, dof) *
                  source[left + left_size * (point + points_1d * right)];
      }
      if (add) { result[entry] += value; }
      else { result[entry] = value; }
   });
}

void WhiteNoiseFull(const DofToQuad &maps, int elements, int components,
                   const Array<int> &markers, const Vector &weights,
                   const Vector &normal_q, Vector &element_rhs)
{
   const int dofs = maps.ndof;
   const int points = maps.nqpt;
   const auto basis = Reshape(maps.B.Read(), points, dofs);
   const auto scale = Reshape(weights.Read(), points, elements);
   const auto normals = Reshape(normal_q.Read(), components, points, elements);
   const int *active = markers.Read();
   real_t *result = element_rhs.ReadWrite();
   mfem::forall(element_rhs.Size(), [=] MFEM_HOST_DEVICE(int entry)
   {
      const int dof = entry % dofs;
      const int component = (entry / dofs) % components;
      const int element = entry / dofs / components;
      if (!active[element]) { return; }
      real_t value = 0.0;
      for (int point = 0; point < points; ++point)
      {
         value += basis(point, dof) * scale(point, element) *
                  normals(component, point, element);
      }
      result[entry] += value;
   });
}

}

QuadratureWhiteNoiseLFIntegrator::QuadratureWhiteNoiseLFIntegrator(
   const FiniteElementSpace &fes, const QuadratureFunction &normal_q_)
   : space(fes), normal_q(normal_q_),
     quadrature_space(dynamic_cast<const QuadratureSpace*>(normal_q.GetSpace())),
     mesh_sequence(fes.GetMesh()->GetSequence()),
     space_sequence(fes.GetSequence()), vector_dimension(fes.GetVDim()),
     quadrature_size(quadrature_space ? quadrature_space->GetSize() : 0)
{
   VerifyBinding();
   space.GetMesh()->EnsureNodes();
   square_root_weights.UseDevice(true);
   first_work.UseDevice(true);
   second_work.UseDevice(true);
}

bool QuadratureWhiteNoiseLFIntegrator::SupportsDevice(
   const FiniteElementSpace &fes) const
{
   if (&fes != &space || !quadrature_space ||
       normal_q.GetSpace() != quadrature_space) { return false; }
   const Mesh &mesh = *fes.GetMesh();
   if (quadrature_space->GetMesh() != &mesh ||
       quadrature_space->GetNE() != fes.GetNE() ||
       quadrature_space->GetSize() != quadrature_size ||
       mesh.GetSequence() != mesh_sequence ||
       fes.GetSequence() != space_sequence) { return false; }
   if (vector_dimension <= 0 || fes.GetVDim() != vector_dimension ||
       normal_q.GetVDim() != vector_dimension ||
       std::int64_t(normal_q.Size()) !=
       std::int64_t(quadrature_size) * vector_dimension) { return false; }
   const int dimension = mesh.Dimension();
   if ((dimension != 2 && dimension != 3) ||
       mesh.SpaceDimension() != dimension || mesh.NURBSext ||
       fes.IsVariableOrder() || mesh.GetNumGeometries(dimension) > 1)
   {
      return false;
   }
   if (!dynamic_cast<const H1_FECollection*>(fes.FEColl()) &&
       !dynamic_cast<const L2_FECollection*>(fes.FEColl())) { return false; }
   const FiniteElement &element = *fes.GetTypicalFE();
   const Geometry::Type geometry = element.GetGeomType();
   const bool simplex = geometry == Geometry::TRIANGLE ||
                        geometry == Geometry::TETRAHEDRON;
   const bool tensor = geometry == Geometry::SQUARE ||
                       geometry == Geometry::CUBE;
   return (simplex || (tensor && UsesTensorBasis(fes))) &&
          element.GetRangeType() == FiniteElement::SCALAR &&
          element.GetMapType() == FiniteElement::VALUE;
}

void QuadratureWhiteNoiseLFIntegrator::VerifyBinding() const
{
   MFEM_VERIFY(SupportsDevice(space),
               "White-noise assembly requires a matching, unchanged scalar or "
               "vector H1/value-mapped L2 space and volume quadrature function "
               "on a uniform-order triangle, quad, tet, or hex mesh. "
               "Recreate the integrator after updating the space or layout.");
}

void QuadratureWhiteNoiseLFIntegrator::SetIntRule(const IntegrationRule *rule)
{
   MFEM_VERIFY(!rule,
               "Set the integration rule through the QuadratureSpace instead.");
}

void QuadratureWhiteNoiseLFIntegrator::Prepare()
{
   VerifyBinding();
   if (space.GetNE() == 0) { return; }
   Mesh &mesh = *space.GetMesh();
   if (!maps)
   {
      quadrature_rule = &quadrature_space->GetIntRule(0);
      MFEM_VERIFY(quadrature_rule->GetNPoints() > 0,
                  "A nonempty quadrature rule is required.");
      for (int point = 0; point < quadrature_rule->GetNPoints(); ++point)
      {
         const IntegrationPoint &location = quadrature_rule->IntPoint(point);
         const real_t weight = location.weight;
         MFEM_VERIFY(std::isfinite(weight) && weight > 0.0,
                     "White-noise quadrature weights must be finite "
                     "and positive.");
         MFEM_VERIFY(std::isfinite(location.x) && std::isfinite(location.y) &&
                     (mesh.Dimension() == 2 || std::isfinite(location.z)),
                     "White-noise quadrature coordinates must be finite.");
      }
      const FiniteElement &element = *space.GetTypicalFE();
      const bool tensor = UsesTensorBasis(space);
      maps = &element.GetDofToQuad(*quadrature_rule,
                                  tensor ? DofToQuad::TENSOR : DofToQuad::FULL);
      const int points = quadrature_rule->GetNPoints();
      MFEM_VERIFY(std::int64_t(points) * space.GetNE() == quadrature_size,
                  "A uniform element quadrature layout is required.");
      WhiteNoiseSize(std::int64_t(element.GetDof()) * vector_dimension *
                     space.GetNE());
      if (tensor)
      {
         MFEM_VERIFY(WhiteNoiseTensorSize(maps->nqpt, mesh.Dimension()) ==
                     points &&
                     WhiteNoiseTensorSize(maps->ndof, mesh.Dimension()) ==
                     element.GetDof(), "Incompatible tensor quadrature layout.");
         for (int point = 0; point < points; ++point)
         {
            const IntegrationPoint &location = quadrature_rule->IntPoint(point);
            const int point_x = point % maps->nqpt;
            const int point_y = (point / maps->nqpt) % maps->nqpt;
            const int point_z = point / maps->nqpt / maps->nqpt;
            MFEM_VERIFY(location.x == quadrature_rule->IntPoint(point_x).x &&
                        location.y == quadrature_rule->IntPoint(point_y).x &&
                        (mesh.Dimension() == 2 ||
                         location.z == quadrature_rule->IntPoint(point_z).x),
                        "Tensor quadrature must use identical 1D point sets "
                        "in lexicographic order.");
         }
         const std::int64_t first_size =
            std::int64_t(normal_q.Size() / maps->nqpt) * maps->ndof;
         first_work.SetSize(WhiteNoiseSize(first_size), Device::GetMemoryType());
         if (mesh.Dimension() == 3)
         {
            const std::int64_t second_size =
               (first_size / maps->nqpt) * maps->ndof;
            second_work.SetSize(WhiteNoiseSize(second_size),
                                Device::GetMemoryType());
         }
      }
      else
      {
         MFEM_VERIFY(maps->nqpt == points && maps->ndof == element.GetDof(),
                     "Incompatible simplex basis evaluation layout.");
      }
   }
   if (nodes_sequence == mesh.GetNodesSequence()) { return; }
   const Vector &weights = quadrature_space->GetWeights();
   MFEM_VERIFY(weights.Size() == quadrature_size, "Invalid quadrature weights.");
   const real_t *host_weights = weights.HostRead();
   for (int point = 0; point < weights.Size(); ++point)
   {
      MFEM_VERIFY(std::isfinite(host_weights[point]) && host_weights[point] > 0.0,
                  "White-noise geometric weights must be finite and positive.");
   }
   square_root_weights.SetSize(weights.Size(), Device::GetMemoryType());
   const real_t *input = weights.Read();
   real_t *output = square_root_weights.Write();
   mfem::forall(weights.Size(), [=] MFEM_HOST_DEVICE(int point)
   {
      output[point] = sqrt(input[point]);
   });
   nodes_sequence = mesh.GetNodesSequence();
}

void QuadratureWhiteNoiseLFIntegrator::AssembleDevice(
   const FiniteElementSpace &fes, const Array<int> &markers, Vector &element_rhs)
{
   MFEM_VERIFY(&fes == &space, "The integrator is bound to another FE space.");
   Prepare();
   const int elements = space.GetNE();
   MFEM_VERIFY(markers.Size() == elements, "Invalid element marker size.");
   if (elements == 0)
   {
      MFEM_VERIFY(element_rhs.Size() == 0, "Invalid empty element vector.");
      return;
   }
   MFEM_VERIFY(element_rhs.Size() == WhiteNoiseSize(
                  std::int64_t(space.GetTypicalFE()->GetDof()) *
                  vector_dimension * elements), "Invalid element vector size.");
   if (!UsesTensorBasis(space))
   {
      WhiteNoiseFull(*maps, elements, vector_dimension, markers,
                    square_root_weights, normal_q, element_rhs);
      return;
   }
   WhiteNoiseTensorFirst(*maps, elements, vector_dimension, markers,
                         square_root_weights, normal_q, first_work);
   if (space.GetMesh()->Dimension() == 2)
   {
      WhiteNoiseTensorAxis(*maps, elements, maps->ndof, markers,
                          first_work, element_rhs, true);
   }
   else
   {
      WhiteNoiseTensorAxis(*maps, elements, maps->ndof, markers,
                          first_work, second_work, false);
      WhiteNoiseTensorAxis(*maps, elements, maps->ndof * maps->ndof, markers,
                          second_work, element_rhs, true);
   }
}

void QuadratureWhiteNoiseLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &element, ElementTransformation &transformation,
   Vector &element_rhs)
{
   Prepare();
   const int element_index = transformation.ElementNo;
   MFEM_VERIFY(transformation.mesh == space.GetMesh() && element_index >= 0 &&
               element_index < space.GetNE() &&
               &element == space.GetFE(element_index),
               "Element transformation does not match the bound FE space.");
   const int dofs = element.GetDof();
   const int points = quadrature_rule->GetNPoints();
   if (use_basis_cache && !ordinary_maps)
   {
      const int basis_size = WhiteNoiseSize(std::int64_t(dofs) * points);
      WhiteNoiseSize(std::int64_t(basis_size) *
                     space.GetMesh()->Dimension());
      ordinary_maps = &element.GetDofToQuad(*quadrature_rule, DofToQuad::FULL);
   }
   const real_t *basis = use_basis_cache ? ordinary_maps->Bt.HostRead() : nullptr;
   if (!use_basis_cache) { shape.SetSize(dofs); }
   const int offset = quadrature_space->Offset(element_index);
   const real_t *normals = normal_q.HostRead();
   const real_t *weights = square_root_weights.HostRead();
   element_rhs.SetSize(dofs * vector_dimension);
   real_t *result = element_rhs.HostWrite();
   for (int entry = 0; entry < element_rhs.Size(); ++entry) { result[entry] = 0.0; }
   for (int point = 0; point < points; ++point)
   {
      if (!use_basis_cache)
      {
         element.CalcShape(quadrature_rule->IntPoint(point), shape);
      }
      const real_t *point_basis = basis ? basis + dofs * point : shape.GetData();
      const int quad = offset + point;
      for (int component = 0; component < vector_dimension; ++component)
      {
         const real_t value = weights[quad] *
                              normals[component + vector_dimension * quad];
         for (int dof = 0; dof < dofs; ++dof)
         {
            result[dof + dofs * component] += point_basis[dof] * value;
         }
      }
   }
}

}
