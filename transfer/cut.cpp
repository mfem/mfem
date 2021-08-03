// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "cut.hpp"

#include "transferutils.hpp"

#include "moonolith_build_quadrature.hpp"

using namespace mfem::private_;

namespace mfem
{

template <class Polytope> class CutGeneric : public Cut
{
public:
   using Point = typename Polytope::Point;
   static const int Dim = Point::n_dims;
   using Quadrature_t = moonolith::Quadrature<double, Dim>;
   using BuildQuadrature_t = moonolith::BuildQuadrature<Polytope>;

   bool BuildQuadrature(const FiniteElementSpace &from_space,
                        const int from_elem_idx,
                        const FiniteElementSpace &to_space,
                        const int to_elem_idx, IntegrationRule &from_quadrature,
                        IntegrationRule &to_quadrature) override;

   void Describe() const override;

protected:
   inline Quadrature_t &GetQRule() { return q_rule_; }
   inline int GetOrder() const { return order_; }

   inline void SetOrder(const int order) { order_ = order; }

   virtual void MakePolytope(Mesh &mesh, const int elem_idx,
                             Polytope &polygon) = 0;

private:
   Polytope from_, to_;
   Quadrature_t q_rule_;
   Quadrature_t physical_quadrature_;
   BuildQuadrature_t builder_;
   int order_{-1};
   double intersection_measure_{0};
};

class Cut2D : public CutGeneric<moonolith::Polygon<double, 2>>
{
public:
   using Polygon_t = moonolith::Polygon<double, 2>;
   using Quadrature_t = moonolith::Quadrature<double, 2>;
   using BuildQuadrature_t = moonolith::BuildQuadrature<Polygon_t>;

   void SetIntegrationOrder(const int order) override;

protected:
   void SetQuadratureRule(const IntegrationRule &ir) override;
   void MakePolytope(Mesh &mesh, const int elem_idx,
                     Polygon_t &polygon) override;

private:
   DenseMatrix buffer_pts;
};

class Cut3D : public CutGeneric<moonolith::Polyhedron<double>>
{
public:
   using Polyhedron_t = moonolith::Polyhedron<double>;
   using Quadrature_t = moonolith::Quadrature<double, 3>;
   using BuildQuadrature_t = moonolith::BuildQuadrature<Polyhedron_t>;

   void SetIntegrationOrder(const int order) override;

protected:
   void SetQuadratureRule(const IntegrationRule &ir) override;
   void MakePolytope(Mesh &mesh, const int elem_idx,
                     Polyhedron_t &polyhedron) override;

private:
   DenseMatrix buffer_pts;
   Array<int> buffer_vertices;
   Array<int> buffer_faces, buffer_cor;
};

void TransformToReference(ElementTransformation &Trans, int type,
                          const Vector &physical_p, const double &w,
                          IntegrationPoint &ref_p)
{

   int dim = physical_p.Size();
   Trans.TransformBack(physical_p, ref_p);

   assert(ref_p.x >= -1e-8);
   assert(ref_p.y >= -1e-8);
   assert(ref_p.z >= -1e-8);

   assert(ref_p.x <= 1 + 1e-8);
   assert(ref_p.y <= 1 + 1e-8);
   assert(ref_p.z <= 1 + 1e-8);

   ref_p.weight = w;
}

template <class Polytope>
bool CutGeneric<Polytope>::BuildQuadrature(const FiniteElementSpace &from_space,
                                           const int from_elem_idx,
                                           const FiniteElementSpace &to_space,
                                           const int to_elem_idx,
                                           IntegrationRule &from_quadrature,
                                           IntegrationRule &to_quadrature)
{

   MakePolytope(*from_space.GetMesh(), from_elem_idx, from_);
   MakePolytope(*to_space.GetMesh(), to_elem_idx, to_);

   if (!builder_.apply(q_rule_, from_, to_, physical_quadrature_))
   {
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

   double from_measure = moonolith::measure(from_);
   double to_measure = moonolith::measure(to_);

   Vector p(Dim);
   for (int qp = 0; qp < n_qp; ++qp)
   {

      for (int d = 0; d < Dim; ++d)
      {
         p(d) = physical_quadrature_.points[qp][d];
      }

      double w = physical_quadrature_.weights[qp];
      intersection_measure_ += w;

      TransformToReference(from_trans, from_type, p, w / from_measure,
                           from_quadrature[qp]);
      TransformToReference(to_trans, to_type, p, w / to_measure,
                           to_quadrature[qp]);
   }

   return true;
}

template <class Polytope> void CutGeneric<Polytope>::Describe() const
{
   mfem::out << "Cut measure " << intersection_measure_ << '\n';
}

template class CutGeneric<::moonolith::Polygon<double, 2>>;
template class CutGeneric<::moonolith::Polyhedron<double>>;

void Cut2D::MakePolytope(Mesh &mesh, const int elem_idx, Polygon_t &polygon)
{

   mesh.GetPointMatrix(elem_idx, buffer_pts);

   const int n_points = buffer_pts.Width();
   polygon.resize(n_points);

   for (int k = 0; k < n_points; ++k)
   {
      for (int d = 0; d < 2; ++d)
      {
         polygon.points[k][d] = buffer_pts(d, k);
      }
   }

   assert(::moonolith::measure(polygon) > 0.0);
}

void Cut2D::SetQuadratureRule(const IntegrationRule &ir)
{
   const int size = ir.Size();
   auto &q_rule = this->GetQRule();

   q_rule.resize(size);

   double rule_w = 0.0;
   for (int k = 0; k < size; ++k)
   {
      auto &qp = ir[k];
      q_rule.points[k][0] = qp.x;
      q_rule.points[k][1] = qp.y;
      q_rule.weights[k] = qp.weight;
      rule_w += qp.weight;
   }

   this->SetOrder(ir.GetOrder());
   q_rule.normalize();
}

void Cut2D::SetIntegrationOrder(const int order)
{
   if (this->GetOrder() != order)
   {
      const IntegrationRule &ir = IntRules.Get(Geometry::TRIANGLE, order);
      assert(ir.GetOrder() == order);
      SetQuadratureRule(ir);
   }
}

void Cut3D::SetIntegrationOrder(const int order)
{
   if (this->GetOrder() != order)
   {
      const IntegrationRule &ir = IntRules.Get(Geometry::TETRAHEDRON, order);
      assert(ir.GetOrder() == order);
      SetQuadratureRule(ir);
   }
}

void Cut3D::MakePolytope(Mesh &mesh, const int elem_idx,
                         Polyhedron_t &polyhedron)
{
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

   for (int i = 0; i < buffer_vertices.Size(); ++i)
   {
      for (int j = 0; j < dim; ++j)
      {
         polyhedron.points[i][j] = buffer_pts(j, i);
      }
   }

   Array<int> f2v;
   for (int i = 0; i < buffer_faces.Size(); ++i)
   {
      mesh.GetFaceVertices(buffer_faces[i], f2v);
      const int eptr = polyhedron.el_ptr[i];

      for (int j = 0; j < f2v.Size(); ++j)
      {
         const int v_offset = buffer_vertices.Find(f2v[j]);
         polyhedron.el_index[eptr + j] = v_offset;
      }

      polyhedron.el_ptr[i + 1] = polyhedron.el_ptr[i] + f2v.Size();
   }

   polyhedron.fix_ordering();
}

void Cut3D::SetQuadratureRule(const IntegrationRule &ir)
{
   const int size = ir.Size();

   auto &q_rule = this->GetQRule();

   q_rule.resize(size);

   double rule_w = 0.0;

   for (int k = 0; k < size; ++k)
   {
      auto &qp = ir[k];
      q_rule.points[k][0] = qp.x;
      q_rule.points[k][1] = qp.y;
      q_rule.points[k][2] = qp.z;
      q_rule.weights[k] = qp.weight;
      rule_w += qp.weight;
   }

   this->SetOrder(ir.GetOrder());
   q_rule.normalize();
}

std::shared_ptr<Cut> NewCut(const int dim)
{
   if (dim == 2)
   {
      return std::make_shared<Cut2D>();
   }
   else if (dim == 3)
   {
      return std::make_shared<Cut3D>();
   }
   else
   {
      assert(false);
      return nullptr;
   }
}

} // namespace mfem
