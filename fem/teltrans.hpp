// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TEMPLATE_ELEMENT_TRANSFORMATION
#define MFEM_TEMPLATE_ELEMENT_TRANSFORMATION

#include "../config/tconfig.hpp"
#include "tevaluator.hpp"
#include "../mesh/element.hpp"

namespace mfem
{

// Templated element transformation classes, cf. eltrans.?pp

// Element transformation class, templated on a mesh type and an integration
// rule. It is constructed from a mesh (e.g. class TMesh) and shape evaluator
// (e.g. class ShapeEvaluator) objects. Allows computation of physical
// coordinates and Jacobian matrices corresponding to the reference integration
// points. The desired result is specified through the template subclass Result
// and stored in an object of the same type.
template <typename Mesh_t, typename IR, typename real_t = double>
class TElementTransformation
{
public:
   typedef real_t                            real_type;
   typedef typename Mesh_t::FE_type          FE_type;
   typedef typename Mesh_t::FESpace_type     FESpace_type;
   typedef typename Mesh_t::nodeLayout_type  nodeLayout_type;
   typedef ShapeEvaluator<FE_type,IR,real_t> ShapeEval;

   typedef TElementTransformation<Mesh_t,IR,real_t> T_type;

   // Enumeration for the result type of the TElementTransformation::Eval()
   // method. The types can obtained by summing constants from this enumeration
   // and used as a template parameter in struct Result.
   enum EvalOperations
   {
      EvalNone        = 0,
      EvalCoordinates = 1,
      EvalJacobians   = 2,
      LoadAttributes  = 4,
      LoadElementIdxs = 8
   };

   template <typename coeff_t, typename kernel_t> struct Get
   {
      static const int EvalOps =
         (EvalCoordinates * coeff_t::uses_coordinates +
          EvalJacobians   * coeff_t::uses_Jacobians +
          LoadAttributes  * coeff_t::uses_attributes  +
          LoadElementIdxs * coeff_t::uses_element_idxs) |
         (EvalJacobians   * kernel_t::uses_Jacobians);
   };

   // Templated struct Result, used to specify the type result that is computed
   // by the TElementTransformation::Eval() method and stored in this structure.
   // The template parameter EvalOps is a sum (bitwise or) of constants from
   // the enum EvalOperations. The parameter NE is the number of elements to be
   // processed in the Eval() method.
   template<int EvalOps, int NE> struct Result;

   static const int dim  = Mesh_t::dim;
   static const int sdim = Mesh_t::space_dim;
   static const int dofs = FE_type::dofs;
   static const int qpts = IR::qpts;

protected:
#ifdef MFEM_TEMPLATE_ELTRANS_HAS_NODE_DOFS
   TTensor3<dofs,sdim,1,real_t> nodes_dof;
#endif

   ShapeEval        evaluator;
   FESpace_type     fes;
   nodeLayout_type  node_layout;
   const real_t    *nodes;

   const Element* const *elements;

   template <int NE>
   inline MFEM_ALWAYS_INLINE
   void SetAttributes(int el, int (&attrib)[NE]) const
   {
      for (int i = 0; i < NE; i++)
      {
         attrib[i] = elements[el+i]->GetAttribute();
      }
   }

public:
   // Constructor.
   TElementTransformation(const Mesh_t &mesh, const ShapeEval &eval)
      : evaluator(eval),
        fes(mesh.t_fes),
        node_layout(mesh.node_layout),
        nodes(mesh.Nodes.GetData()),               // real_t = double
        elements(mesh.m_mesh.GetElementsArray())
   { }

   // Evaluate coordinates and/or Jacobian matrices at quadrature points.
   template<int EvalOps, int NE>
   inline MFEM_ALWAYS_INLINE
   void Eval(int el, Result<EvalOps,NE> &F)
   {
      F.Eval(el, *this);
   }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
   template<int EvalOps, int NE>
   inline MFEM_ALWAYS_INLINE
   void EvalSerialized(int el, const real_t *nodeData, Result<EvalOps,NE> &F)
   {
      F.EvalSerialized(el, *this, nodeData);
   }
#endif

   template <int NE> struct Result<0,NE> // 0 = EvalNone
   {
      static const int ne = NE;
      // x_type x;
      // Jt_type Jt;
      // int attrib[NE];
      // int first_elem_idx;
      inline MFEM_ALWAYS_INLINE
      void Eval(int el, T_type &T)
      {
         // T.SetAttributes(el, attrib);
         // first_elem_idx = el;
      }
#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
      inline MFEM_ALWAYS_INLINE
      void EvalSerialized(int el, T_type &T, const real_t *nodeData) { }
#endif
   };
   template <int NE> struct Result<1,NE> // 1 = EvalCoordinates
   {
      static const int ne = NE;
#ifdef MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES
      typedef TTensor3<qpts,sdim,NE,real_t,true> x_type;
#else
      typedef TTensor3<qpts,sdim,NE,real_t/*,true*/> x_type;
#endif
      x_type x;

      typedef TTensor3<dofs,sdim,NE,real_t> nodes_dof_t;
#ifdef MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES
      nodes_dof_t nodes_dof;
#endif

      inline MFEM_ALWAYS_INLINE
      void Eval(int el, T_type &T)
      {
#ifdef MFEM_TEMPLATE_ELTRANS_HAS_NODE_DOFS
         MFEM_STATIC_ASSERT(NE == 1, "only NE == 1 is supported");
         TTensor3<dofs,sdim,1,real_t> &nodes_dof = T.nodes_dof;
#elif !defined(MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES)
         nodes_dof_t nodes_dof;
#endif
         T.fes.SetElement(el);
         T.fes.VectorExtract(T.node_layout, T.nodes,
                             nodes_dof.layout, nodes_dof);
         T.evaluator.Calc(nodes_dof.layout.merge_23(), nodes_dof,
                          x.layout.merge_23(), x);
      }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
      inline MFEM_ALWAYS_INLINE
      void EvalSerialized(int el, T_type &T, const real_t *nodeData)
      {
         T.evaluator.Calc(nodes_dof_t::layout.merge_23(),
                          &nodeData[el*nodes_dof_t::size],
                          x.layout.merge_23(), x);
      }
#endif
   };
   template <int NE> struct Result<2,NE> // 2 = EvalJacobians
   {
      static const int ne = NE;
#ifdef MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES
      typedef TTensor4<qpts,dim,sdim,NE,real_t,true> Jt_type;
#else
      typedef TTensor4<qpts,dim,sdim,NE,real_t/*,true*/> Jt_type;
#endif
      Jt_type Jt;

      typedef TTensor3<dofs,sdim,NE,real_t> nodes_dof_t;
#ifdef MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES
      nodes_dof_t nodes_dof;
#endif

      inline MFEM_ALWAYS_INLINE
      void Eval(int el, T_type &T)
      {
#ifdef MFEM_TEMPLATE_ELTRANS_HAS_NODE_DOFS
         MFEM_STATIC_ASSERT(NE == 1, "only NE == 1 is supported");
         TTensor3<dofs,sdim,1,real_t> &nodes_dof = T.nodes_dof;
#elif !defined(MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES)
         nodes_dof_t nodes_dof;
#endif
         T.fes.SetElement(el);
         T.fes.VectorExtract(T.node_layout, T.nodes,
                             nodes_dof.layout, nodes_dof);
         T.evaluator.CalcGrad(nodes_dof.layout.merge_23(), nodes_dof,
                              Jt.layout.merge_34(), Jt);
      }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
      inline MFEM_ALWAYS_INLINE
      void EvalSerialized(int el, T_type &T, const real_t *nodeData)
      {
         T.evaluator.CalcGrad(nodes_dof_t::layout.merge_23(),
                              &nodeData[el*nodes_dof_t::size],
                              Jt.layout.merge_34(), Jt);
      }
#endif
   };
   template <int NE> struct Result<3,NE> // 3 = EvalCoordinates|EvalJacobians
   {
      static const int ne = NE;
      typedef TTensor3<qpts,sdim,NE,real_t,true> x_type;
      x_type x;
#ifdef MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES
      typedef TTensor4<qpts,dim,sdim,NE,real_t,true> Jt_type;
#else
      typedef TTensor4<qpts,dim,sdim,NE,real_t/*,true*/> Jt_type;
#endif
      Jt_type Jt;

      typedef TTensor3<dofs,sdim,NE,real_t> nodes_dof_t;
#ifdef MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES
      nodes_dof_t nodes_dof;
#endif

      inline MFEM_ALWAYS_INLINE
      void Eval(int el, T_type &T)
      {
#ifdef MFEM_TEMPLATE_ELTRANS_HAS_NODE_DOFS
         MFEM_STATIC_ASSERT(NE == 1, "only NE == 1 is supported");
         TTensor3<dofs,sdim,1,real_t> &nodes_dof = T.nodes_dof;
#elif !defined(MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES)
         nodes_dof_t nodes_dof;
#endif
         T.fes.SetElement(el);
         T.fes.VectorExtract(T.node_layout, T.nodes,
                             nodes_dof.layout, nodes_dof);
         T.evaluator.Calc(nodes_dof.layout.merge_23(), nodes_dof,
                          x.layout.merge_23(), x);
         T.evaluator.CalcGrad(nodes_dof.layout.merge_23(), nodes_dof,
                              Jt.layout.merge_34(), Jt);
      }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
      inline MFEM_ALWAYS_INLINE
      void EvalSerialized(int el, T_type &T, const real_t *nodeData)
      {
         T.evaluator.Calc(nodes_dof_t::layout.merge_23(),
                          &nodeData[el*nodes_dof_t::size],
                          x.layout.merge_23(), x);
         T.evaluator.CalcGrad(nodes_dof_t::layout.merge_23(),
                              &nodeData[el*nodes_dof_t::size],
                              Jt.layout.merge_34(), Jt);
      }
#endif
   };
   template <int NE> struct Result<6,NE> // 6 = EvalJacobians|LoadAttributes
   {
      static const int ne = NE;
#ifdef MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES
      typedef TTensor4<qpts,dim,sdim,NE,real_t,true> Jt_type;
#else
      typedef TTensor4<qpts,dim,sdim,NE,real_t/*,true*/> Jt_type;
#endif
      Jt_type Jt;

      typedef TTensor3<dofs,sdim,NE,real_t> nodes_dof_t;
#ifdef MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES
      nodes_dof_t nodes_dof;
#endif
      int attrib[NE];

      inline MFEM_ALWAYS_INLINE
      void Eval(int el, T_type &T)
      {
#ifdef MFEM_TEMPLATE_ELTRANS_HAS_NODE_DOFS
         MFEM_STATIC_ASSERT(NE == 1, "only NE == 1 is supported");
         TTensor3<dofs,sdim,1,real_t> &nodes_dof = T.nodes_dof;
#elif !defined(MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES)
         nodes_dof_t nodes_dof;
#endif
         T.fes.SetElement(el);
         T.fes.VectorExtract(T.node_layout, T.nodes,
                             nodes_dof.layout, nodes_dof);
         T.evaluator.CalcGrad(nodes_dof.layout.merge_23(), nodes_dof,
                              Jt.layout.merge_34(), Jt);
         T.SetAttributes(el, attrib);
      }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
      inline MFEM_ALWAYS_INLINE
      void EvalSerialized(int el, T_type &T, const real_t *nodeData)
      {
         T.evaluator.CalcGrad(nodes_dof_t::layout.merge_23(),
                              &nodeData[el*nodes_dof_t::size],
                              Jt.layout.merge_34(), Jt);
         T.SetAttributes(el, attrib);
      }
#endif
   };
   template <int NE> struct Result<10,NE> // 10 = EvalJacobians|LoadElementIdxs
   {
      static const int ne = NE;
#ifdef MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES
      typedef TTensor4<qpts,dim,sdim,NE,real_t,true> Jt_type;
#else
      typedef TTensor4<qpts,dim,sdim,NE,real_t/*,true*/> Jt_type;
#endif
      Jt_type Jt;

      typedef TTensor3<dofs,sdim,NE,real_t> nodes_dof_t;
#ifdef MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES
      nodes_dof_t nodes_dof;
#endif
      int first_elem_idx;

      inline MFEM_ALWAYS_INLINE
      void Eval(int el, T_type &T)
      {
#ifdef MFEM_TEMPLATE_ELTRANS_HAS_NODE_DOFS
         MFEM_STATIC_ASSERT(NE == 1, "only NE == 1 is supported");
         TTensor3<dofs,sdim,1,real_t> &nodes_dof = T.nodes_dof;
#elif !defined(MFEM_TEMPLATE_ELTRANS_RESULT_HAS_NODES)
         nodes_dof_t nodes_dof;
#endif
         T.fes.SetElement(el);
         T.fes.VectorExtract(T.node_layout, T.nodes,
                             nodes_dof.layout, nodes_dof);
         T.evaluator.CalcGrad(nodes_dof.layout.merge_23(), nodes_dof,
                              Jt.layout.merge_34(), Jt);
         first_elem_idx = el;
      }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
      inline MFEM_ALWAYS_INLINE
      void EvalSerialized(int el, T_type &T, const real_t *nodeData)
      {
         T.evaluator.CalcGrad(nodes_dof_t::layout.merge_23(),
                              &nodeData[el*nodes_dof_t::size],
                              Jt.layout.merge_34(), Jt);
         first_elem_idx = el;
      }
#endif
   };
};

} // namespace mfem

#endif // MFEM_TEMPLATE_ELEMENT_TRANSFORMATION
