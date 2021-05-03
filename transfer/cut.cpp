#include "cut.hpp"

#include "transferutils.hpp"

namespace mfem {

void TransformToReference(ElementTransformation &Trans, int type,
                          const Vector &physical_p, const double &w,
                          IntegrationPoint &ref_p) {

  int dim = physical_p.Size();
  Trans.TransformBack(physical_p, ref_p);

  assert(ref_p.x >= -1e-8);
  assert(ref_p.y >= -1e-8);
  assert(ref_p.z >= -1e-8);

  assert(ref_p.x <= 1 + 1e-8);
  assert(ref_p.y <= 1 + 1e-8);
  assert(ref_p.z <= 1 + 1e-8);

  ref_p.weight = w;

  if (type != Geometry::TRIANGLE && dim == 2) {
    ref_p.weight *= 0.5;
  } else if (type != Geometry::TETRAHEDRON && dim == 3) {
    ref_p.weight *= 6;
  }
}

bool Cut2D::BuildQuadrature(const FiniteElementSpace &from_space,
                            const int from_elem_idx,
                            const FiniteElementSpace &to_space,
                            const int to_elem_idx,
                            IntegrationRule &from_quadrature,
                            IntegrationRule &to_quadrature) {

  MakePolygon(*from_space.GetMesh(), from_elem_idx, from_);
  MakePolygon(*to_space.GetMesh(), to_elem_idx, to_);

  if (!builder_.apply(q_rule_, from_, to_, physical_quadrature_)) {
    return false;
  }

  assert(false && "SCALE WITH FROM AND TO VOLUMES");

  int from_type = from_space.GetFE(from_elem_idx)->GetGeomType();
  int to_type = to_space.GetFE(to_elem_idx)->GetGeomType();

  const int n_qp = physical_quadrature_.n_points();

  from_quadrature.SetSize(n_qp);
  to_quadrature.SetSize(n_qp);

  ElementTransformation &from_trans =
      *from_space.GetElementTransformation(from_elem_idx);
  ElementTransformation &to_trans =
      *to_space.GetElementTransformation(to_elem_idx);

  Vector p(2);
  for (int qp = 0; qp < n_qp; ++qp) {

    for (int d = 0; d < 2; ++d) {
      p(d) = physical_quadrature_.points[qp][d];
    }

    double w = physical_quadrature_.weights[qp];

    TransformToReference(from_trans, from_type, p, w, from_quadrature[qp]);
    TransformToReference(to_trans, to_type, p, w, to_quadrature[qp]);
  }

  return true;
}

void Cut2D::MakePolygon(Mesh &mesh, const int elem_idx, Polygon_t &polygon) {

  mesh.GetPointMatrix(elem_idx, buffer_pts);

  const int n_points = buffer_pts.Width();
  polygon.resize(n_points);

  for (int k = 0; k < n_points; ++k) {
    for (int d = 0; d < 2; ++d) {
      polygon.points[k][d] = buffer_pts(d, k);
    }
  }

  assert(::moonolith::measure(polygon) > 0.0);
}

void Cut2D::SetQuadratureRule(const IntegrationRule &ir) {
  const int size = ir.Size();
  q_rule_.resize(size);

  double rule_w = 0.0;

  for (int k = 0; k < size; ++k) {
    auto &qp = ir[k];
    q_rule_.points[k][0] = qp.x;
    q_rule_.points[k][1] = qp.y;
    q_rule_.weights[k] = qp.weight;
    rule_w += qp.weight;
  }

  q_rule_.normalize();
  order_ = ir.GetOrder();
  mfem::out << "Rule weight: " << rule_w << '\n';
}

void Cut2D::SetIntegrationOrder(const int order) {
  if (order_ != order) {
    const IntegrationRule &ir = IntRules.Get(Geometry::TRIANGLE, order);
    assert(ir.GetOrder() == order);
    SetQuadratureRule(ir);
  }
}

bool Cut3D::BuildQuadrature(const FiniteElementSpace &from_space,
                            const int from_elem_idx,
                            const FiniteElementSpace &to_space,
                            const int to_elem_idx,
                            IntegrationRule &from_quadrature,
                            IntegrationRule &to_quadrature) {

  MakePolyhedron(*from_space.GetMesh(), from_elem_idx, from_);
  MakePolyhedron(*to_space.GetMesh(), to_elem_idx, to_);

  if (!builder_.apply(q_rule_, from_, to_, physical_quadrature_)) {
    return false;
  }

  int from_type = from_space.GetFE(from_elem_idx)->GetGeomType();
  int to_type = to_space.GetFE(to_elem_idx)->GetGeomType();

  const int n_qp = physical_quadrature_.n_points();

  from_quadrature.SetSize(n_qp);
  to_quadrature.SetSize(n_qp);

  ElementTransformation &from_trans =
      *from_space.GetElementTransformation(from_elem_idx);
  ElementTransformation &to_trans =
      *to_space.GetElementTransformation(to_elem_idx);

  Vector p(3);
  for (int qp = 0; qp < n_qp; ++qp) {

    for (int d = 0; d < 3; ++d) {
      p(d) = physical_quadrature_.points[qp][d];
    }

    double w = physical_quadrature_.weights[qp];

    TransformToReference(from_trans, from_type, p, w, from_quadrature[qp]);
    TransformToReference(to_trans, to_type, p, w, to_quadrature[qp]);
  }

  return true;
}

void Cut3D::SetIntegrationOrder(const int order) {
  if (order_ != order) {
    const IntegrationRule &ir = IntRules.Get(Geometry::TETRAHEDRON, order);
    assert(ir.GetOrder() == order);
    SetQuadratureRule(ir);
  }
}

void Cut3D::MakePolyhedron(Mesh &mesh, const int elem_idx,
                           Polyhedron_t &polyhedron) {
  // TODO check counter-clockwise faces
  using namespace std;

  const int dim = mesh.Dimension();

  assert(mesh.GetElement(elem_idx));

  const Element &e = *mesh.GetElement(elem_idx);
  const int e_type = e.GetType();

  mesh.GetElementFaces(elem_idx, buffer_faces, buffer_cor);
  mesh.GetPointMatrix(elem_idx, buffer_pts);
  mesh.GetElementVertices(elem_idx, buffer_vertices);

  const int n_faces = buffer_faces.Size();
  polyhedron.clear();
  polyhedron.el_ptr.resize(n_faces + 1);
  polyhedron.points.resize(buffer_vertices.Size());
  polyhedron.el_index.resize(MaxVertsXFace(e_type) * n_faces);
  polyhedron.el_ptr[0] = 0;

  for (int i = 0; i < buffer_vertices.Size(); ++i) {
    const int offset = i * dim;

    for (int j = 0; j < dim; ++j) {
      polyhedron.points[offset + j] = buffer_pts(j, i);
    }
  }

  Array<int> f2v;
  for (int i = 0; i < buffer_faces.Size(); ++i) {
    mesh.GetFaceVertices(buffer_faces[i], f2v);
    const int eptr = polyhedron.el_ptr[i];

    for (int j = 0; j < f2v.Size(); ++j) {
      const int v_offset = buffer_vertices.Find(f2v[j]);
      polyhedron.el_index[eptr + j] = v_offset;
    }

    polyhedron.el_ptr[i + 1] = polyhedron.el_ptr[i] + f2v.Size();
  }
}

void Cut3D::SetQuadratureRule(const IntegrationRule &ir) {
  const int size = ir.Size();
  q_rule_.resize(size);

  for (int k = 0; k < size; ++k) {
    auto &qp = ir[k];
    q_rule_.points[k][0] = qp.x;
    q_rule_.points[k][1] = qp.y;
    q_rule_.points[k][2] = qp.z;
    q_rule_.weights[k] = qp.weight;
  }

  q_rule_.normalize();
}

} // namespace mfem
