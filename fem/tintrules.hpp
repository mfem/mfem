// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TEMPLATE_INTEGRATION_RULES
#define MFEM_TEMPLATE_INTEGRATION_RULES

#include "../config/tconfig.hpp"
#include "../linalg/ttensor.hpp"
#include "geom.hpp"

namespace mfem
{

// Templated integration rules, cf. intrules.?pp

template <Geometry::Type G, int Q, int Order, typename real_t>
class GenericIntegrationRule
{
public:
   static const Geometry::Type geom = G;
   static const int dim = Geometry::Constants<geom>::Dimension;
   static const int qpts = Q;
   static const int order = Order;

   static const bool tensor_prod = false;

   typedef real_t real_type;

protected:
   TVector<qpts,real_t> weights;

public:
   GenericIntegrationRule()
   {
      const IntegrationRule &ir = GetIntRule();
      MFEM_ASSERT(ir.GetNPoints() == qpts, "quadrature rule mismatch");
      for (int j = 0; j < qpts; j++)
      {
         weights[j] = ir.IntPoint(j).weight;
      }
   }

   // Default copy constructor

   static const IntegrationRule &GetIntRule()
   {
      return IntRules.Get(geom, order);
   }

   // Multi-component weight assignment. qpt_layout_t must be (qpts x n1 x ...)
   template <AssignOp::Type Op, typename qpt_layout_t, typename qpt_data_t>
   void AssignWeights(const qpt_layout_t &qpt_layout,
                      qpt_data_t &qpt_data) const
   {
      MFEM_STATIC_ASSERT(qpt_layout_t::rank > 1, "invalid rank");
      MFEM_STATIC_ASSERT(qpt_layout_t::dim_1 == qpts, "invalid size");
      for (int j = 0; j < qpts; j++)
      {
         TAssign<Op>(qpt_layout.ind1(j), qpt_data, weights.data[j]);
      }
   }

   template <typename qpt_data_t>
   void ApplyWeights(qpt_data_t &qpt_data) const
   {
      AssignWeights<AssignOp::Mult>(ColumnMajorLayout2D<qpts,1>(), qpt_data);
   }
};

template <int Dim, int Q, typename real_t>
class TProductIntegrationRule_base;

template <int Q, typename real_t>
class TProductIntegrationRule_base<1,Q,real_t>
{
protected:
   TVector<Q,real_t> weights_1d;

public:
   // Multi-component weight assignment. qpt_layout_t must be (qpts x n1 x ...)
   template <AssignOp::Type Op, typename qpt_layout_t, typename qpt_data_t>
   void AssignWeights(const qpt_layout_t &qpt_layout,
                      qpt_data_t &qpt_data) const
   {
      MFEM_STATIC_ASSERT(qpt_layout_t::rank > 1, "invalid rank");
      MFEM_STATIC_ASSERT(qpt_layout_t::dim_1 == Q, "invalid size");
      for (int j = 0; j < Q; j++)
      {
         TAssign<Op>(qpt_layout.ind1(j), qpt_data, weights_1d.data[j]);
      }
   }

   template <typename qpt_data_t>
   void ApplyWeights(qpt_data_t &qpt_data) const
   {
      AssignWeights<AssignOp::Mult>(ColumnMajorLayout2D<Q,1>(), qpt_data);
   }
};

template <int Q, typename real_t>
class TProductIntegrationRule_base<2,Q,real_t>
{
protected:
   TVector<Q,real_t> weights_1d;

public:
   // Multi-component weight assignment. qpt_layout_t must be (qpts x n1 x ...)
   template <AssignOp::Type Op, typename qpt_layout_t, typename qpt_data_t>
   void AssignWeights(const qpt_layout_t &qpt_layout,
                      qpt_data_t &qpt_data) const
   {
      MFEM_STATIC_ASSERT(qpt_layout_t::rank > 1, "invalid rank");
      MFEM_STATIC_ASSERT(qpt_layout_t::dim_1 == Q*Q, "invalid size");
      MFEM_FLOPS_ADD(Q*Q);
      for (int j2 = 0; j2 < Q; j2++)
      {
         for (int j1 = 0; j1 < Q; j1++)
         {
            TAssign<Op>(
               qpt_layout.ind1(TMatrix<Q,Q>::layout.ind(j1,j2)), qpt_data,
               weights_1d.data[j1]*weights_1d.data[j2]);
         }
      }
   }

   template <typename qpt_data_t>
   void ApplyWeights(qpt_data_t &qpt_data) const
   {
      AssignWeights<AssignOp::Mult>(ColumnMajorLayout2D<Q*Q,1>(), qpt_data);
   }
};

template <int Q, typename real_t>
class TProductIntegrationRule_base<3,Q,real_t>
{
protected:
   TVector<Q,real_t> weights_1d;

public:
   // Multi-component weight assignment. qpt_layout_t must be (qpts x n1 x ...)
   template <AssignOp::Type Op, typename qpt_layout_t, typename qpt_data_t>
   void AssignWeights(const qpt_layout_t &qpt_layout,
                      qpt_data_t &qpt_data) const
   {
      MFEM_STATIC_ASSERT(qpt_layout_t::rank > 1, "invalid rank");
      MFEM_STATIC_ASSERT(qpt_layout_t::dim_1 == Q*Q*Q, "invalid size");
      MFEM_FLOPS_ADD(2*Q*Q*Q);
      for (int j3 = 0; j3 < Q; j3++)
      {
         for (int j2 = 0; j2 < Q; j2++)
         {
            for (int j1 = 0; j1 < Q; j1++)
            {
               TAssign<Op>(
                  qpt_layout.ind1(TTensor3<Q,Q,Q>::layout.ind(j1,j2,j3)),
                  qpt_data,
                  weights_1d.data[j1]*weights_1d.data[j2]*weights_1d.data[j3]);
            }
         }
      }
   }

   template <typename qpt_data_t>
   void ApplyWeights(qpt_data_t &qpt_data) const
   {
      AssignWeights<AssignOp::Mult>(ColumnMajorLayout2D<Q*Q*Q,1>(), qpt_data);
   }
};

template <int Dim, int Q, int Order, typename real_t>
class TProductIntegrationRule
   : public TProductIntegrationRule_base<Dim,Q,real_t>
{
public:
   static const Geometry::Type geom =
      ((Dim == 1) ? Geometry::SEGMENT :
       ((Dim == 2) ? Geometry::SQUARE : Geometry::CUBE));
   static const int dim = Dim;
   static const int qpts_1d = Q;
   static const int qpts = (Dim == 1) ? Q : ((Dim == 2) ? (Q*Q) : (Q*Q*Q));
   static const int order = Order;

   static const bool tensor_prod = true;

   typedef real_t real_type;

protected:
   using TProductIntegrationRule_base<Dim,Q,real_t>::weights_1d;

public:
   // default constructor, default copy constructor
};

template <int Dim, int Q, typename real_t>
class GaussIntegrationRule
   : public TProductIntegrationRule<Dim, Q, 2*Q-1, real_t>
{
public:
   typedef TProductIntegrationRule<Dim,Q,2*Q-1,real_t> base_class;

   using base_class::geom;
   using base_class::order;
   using base_class::qpts_1d;

protected:
   using base_class::weights_1d;

public:
   GaussIntegrationRule()
   {
      const IntegrationRule &ir_1d = Get1DIntRule();
      MFEM_ASSERT(ir_1d.GetNPoints() == qpts_1d, "quadrature rule mismatch");
      for (int j = 0; j < qpts_1d; j++)
      {
         weights_1d.data[j] = ir_1d.IntPoint(j).weight;
      }
   }

   static const IntegrationRule &Get1DIntRule()
   {
      return IntRules.Get(Geometry::SEGMENT, order);
   }
   static const IntegrationRule &GetIntRule()
   {
      return IntRules.Get(geom, order);
   }
};

template <Geometry::Type G, int Order, typename real_t = real_t>
class TIntegrationRule;

template <int Order, typename real_t>
class TIntegrationRule<Geometry::SEGMENT, Order, real_t>
   : public GaussIntegrationRule<1, Order/2+1, real_t> { };

template <int Order, typename real_t>
class TIntegrationRule<Geometry::SQUARE, Order, real_t>
   : public GaussIntegrationRule<2, Order/2+1, real_t> { };

template <int Order, typename real_t>
class TIntegrationRule<Geometry::CUBE, Order, real_t>
   : public GaussIntegrationRule<3, Order/2+1, real_t> { };

// Triangle integration rules (based on intrules.cpp)
// These specializations define the number of quadrature points for each rule as
// a compile-time constant.
// TODO: add higher order rules
template <typename real_t>
class TIntegrationRule<Geometry::TRIANGLE, 0, real_t>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 1, 0, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TRIANGLE, 1, real_t>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 1, 1, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TRIANGLE, 2, real_t>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 3, 2, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TRIANGLE, 3, real_t>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 4, 3, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TRIANGLE, 4, real_t>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 6, 4, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TRIANGLE, 5, real_t>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 7, 5, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TRIANGLE, 6, real_t>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 12, 6, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TRIANGLE, 7, real_t>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 12, 7, real_t> { };

// Tetrahedron integration rules (based on intrules.cpp)
// These specializations define the number of quadrature points for each rule as
// a compile-time constant.
// TODO: add higher order rules
template <typename real_t>
class TIntegrationRule<Geometry::TETRAHEDRON, 0, real_t>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 1, 0, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TETRAHEDRON, 1, real_t>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 1, 1, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TETRAHEDRON, 2, real_t>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 4, 2, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TETRAHEDRON, 3, real_t>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 5, 3, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TETRAHEDRON, 4, real_t>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 11, 4, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TETRAHEDRON, 5, real_t>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 14, 5, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TETRAHEDRON, 6, real_t>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 24, 6, real_t> { };
template <typename real_t>
class TIntegrationRule<Geometry::TETRAHEDRON, 7, real_t>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 31, 7, real_t> { };

} // namespace mfem

#endif // MFEM_TEMPLATE_INTEGRATION_RULES
