#ifndef MFEM_TRANSFER_CUT_HPP
#define MFEM_TRANSFER_CUT_HPP

#include "../fem/fem.hpp"

#include "moonolith_build_quadrature.hpp"

namespace mfem {

class CutBase {
public:
  virtual ~CutBase() = default;
  virtual bool BuildQuadrature(const FiniteElementSpace &from_space,
                               const int from_elem_idx,
                               const FiniteElementSpace &to_space,
                               const int to_elem_idx,
                               IntegrationRule &from_quadrature,
                               IntegrationRule &to_quadrature) = 0;

  virtual void SetQuadratureRule(const IntegrationRule &ir) = 0;
  virtual void SetIntegrationOrder(const int order) = 0;
  virtual void describe() const {}
};

class Cut2D : public CutBase {
public:
  using Polygon_t = moonolith::Polygon<double, 2>;
  using Quadrature_t = moonolith::Quadrature<double, 2>;
  using BuildQuadrature_t = moonolith::BuildQuadrature<Polygon_t>;

  bool BuildQuadrature(const FiniteElementSpace &from_space,
                       const int from_elem_idx,
                       const FiniteElementSpace &to_space,
                       const int to_elem_idx, IntegrationRule &from_quadrature,
                       IntegrationRule &to_quadrature) override;

  void SetQuadratureRule(const IntegrationRule &ir) override;
  void SetIntegrationOrder(const int order) override;

  void MakePolygon(Mesh &mesh, const int elem_idx, Polygon_t &polygon);
  void describe() const override;

private:
  Polygon_t from_, to_;
  Quadrature_t q_rule_;
  Quadrature_t physical_quadrature_;
  BuildQuadrature_t builder_;
  DenseMatrix buffer_pts;
  int order_{-1};
  double intersection_measure_{0};
};

class Cut3D : public CutBase {
public:
  using Polyhedron_t = moonolith::Polyhedron<double>;
  using Quadrature_t = moonolith::Quadrature<double, 3>;
  using BuildQuadrature_t = moonolith::BuildQuadrature<Polyhedron_t>;

  bool BuildQuadrature(const FiniteElementSpace &from_space,
                       const int from_elem_idx,
                       const FiniteElementSpace &to_space,
                       const int to_elem_idx, IntegrationRule &from_quadrature,
                       IntegrationRule &to_quadrature) override;

  void SetQuadratureRule(const IntegrationRule &ir) override;
  void SetIntegrationOrder(const int order) override;

  void MakePolyhedron(Mesh &mesh, const int elem_idx, Polyhedron_t &polyhedron);
  void describe() const override;

private:
  Polyhedron_t from_, to_;
  Quadrature_t q_rule_;
  Quadrature_t physical_quadrature_;
  BuildQuadrature_t builder_;
  DenseMatrix buffer_pts;
  Array<int> buffer_vertices;
  Array<int> buffer_faces, buffer_cor;
  int order_{-1};
  double intersection_measure_{0};
};

} // namespace mfem

#endif // MFEM_TRANSFER_CUT_HPP
