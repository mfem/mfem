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

#ifndef MFEM_TEMPLATE_EVALUATOR
#define MFEM_TEMPLATE_EVALUATOR

#include "../config/tconfig.hpp"
#include "../linalg/ttensor.hpp"
#include "../general/error.hpp"
#include "fespace.hpp"

namespace mfem
{

// Templated classes for transitioning between degrees of freedom and quadrature
// points values.

/** @brief Shape evaluators -- values of basis functions on the reference element
    @tparam FE some form of TFiniteElement, probably got from TMesh::FE_type
    @tparam IR some form of TIntegrationRule
    @tparam TP tensor product or not
    @tparam real_t data type for mesh nodes, solution basis, mesh basis
*/
template <class FE, class IR, bool TP, typename real_t>
class ShapeEvaluator_base;

/// ShapeEvaluator without tensor-product structure
template <class FE, class IR, typename real_t>
class ShapeEvaluator_base<FE, IR, false, real_t>
{
public:
   static const int DOF = FE::dofs;
   static const int NIP = IR::qpts;
   static const int DIM = FE::dim;

protected:
   TMatrix<NIP,DOF,real_t,true> B;
   TMatrix<DOF,NIP,real_t,true> Bt;
   TTensor3<NIP,DIM,DOF,real_t,true> G;
   TTensor3<DOF,NIP,DIM,real_t> Gt;

public:
   ShapeEvaluator_base(const FE &fe)
   {
      fe.CalcShapes(IR::GetIntRule(), B.data, G.data);
      TAssign<AssignOp::Set>(Bt.layout, Bt, B.layout.transpose_12(), B);
      TAssign<AssignOp::Set>(Gt.layout.merge_23(), Gt,
                             G.layout.merge_12().transpose_12(), G);
   }

   // default copy constructor

   /** @brief Multi-component shape evaluation from DOFs to quadrature points.
       dof_layout is (DOF x NumComp) and qpt_layout is (NIP x NumComp). */
   template <typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
   inline MFEM_ALWAYS_INLINE
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      MFEM_STATIC_ASSERT(dof_layout_t::rank  == 2 &&
                         dof_layout_t::dim_1 == DOF,
                         "invalid dof_layout_t.");
      MFEM_STATIC_ASSERT(qpt_layout_t::rank  == 2 &&
                         qpt_layout_t::dim_1 == NIP,
                         "invalid qpt_layout_t.");
      MFEM_STATIC_ASSERT(dof_layout_t::dim_2 == qpt_layout_t::dim_2,
                         "incompatible dof- and qpt- layouts.");

      Mult_AB<false>(B.layout, B,
                     dof_layout, dof_data,
                     qpt_layout, qpt_data);
   }

   /** @brief Multi-component shape evaluation transpose from quadrature points to
       DOFs.  qpt_layout is (NIP x NumComp) and dof_layout is (DOF x NumComp). */
   template <bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      MFEM_STATIC_ASSERT(dof_layout_t::rank  == 2 &&
                         dof_layout_t::dim_1 == DOF,
                         "invalid dof_layout_t.");
      MFEM_STATIC_ASSERT(qpt_layout_t::rank  == 2 &&
                         qpt_layout_t::dim_1 == NIP,
                         "invalid qpt_layout_t.");
      MFEM_STATIC_ASSERT(dof_layout_t::dim_2 == qpt_layout_t::dim_2,
                         "incompatible dof- and qpt- layouts.");

      Mult_AB<Add>(Bt.layout, Bt,
                   qpt_layout, qpt_data,
                   dof_layout, dof_data);
   }

   /** @brief Multi-component gradient evaluation from DOFs to quadrature points.
      dof_layout is (DOF x NumComp) and grad_layout is (NIP x DIM x NumComp). */
   template <typename dof_layout_t, typename dof_data_t,
             typename grad_layout_t, typename grad_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcGrad(const dof_layout_t  &dof_layout,
                 const dof_data_t    &dof_data,
                 const grad_layout_t &grad_layout,
                 grad_data_t         &grad_data) const
   {
      MFEM_STATIC_ASSERT(dof_layout_t::rank  == 2 &&
                         dof_layout_t::dim_1 == DOF,
                         "invalid dof_layout_t.");
      MFEM_STATIC_ASSERT(grad_layout_t::rank  == 3 &&
                         grad_layout_t::dim_1 == NIP &&
                         grad_layout_t::dim_2 == DIM,
                         "invalid grad_layout_t.");
      MFEM_STATIC_ASSERT(dof_layout_t::dim_2 == grad_layout_t::dim_3,
                         "incompatible dof- and grad- layouts.");

      Mult_AB<false>(G.layout.merge_12(), G,
                     dof_layout, dof_data,
                     grad_layout.merge_12(), grad_data);
   }

   /** @brief Multi-component gradient evaluation transpose from quadrature points to
      DOFs. grad_layout is (NIP x DIM x NumComp), dof_layout is (DOF x NumComp). */
   template <bool Add,
             typename grad_layout_t, typename grad_data_t,
             typename dof_layout_t, typename dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcGradT(const grad_layout_t &grad_layout,
                  const grad_data_t   &grad_data,
                  const dof_layout_t  &dof_layout,
                  dof_data_t          &dof_data) const
   {
      MFEM_STATIC_ASSERT(dof_layout_t::rank  == 2 &&
                         dof_layout_t::dim_1 == DOF,
                         "invalid dof_layout_t.");
      MFEM_STATIC_ASSERT(grad_layout_t::rank  == 3 &&
                         grad_layout_t::dim_1 == NIP &&
                         grad_layout_t::dim_2 == DIM,
                         "invalid grad_layout_t.");
      MFEM_STATIC_ASSERT(dof_layout_t::dim_2 == grad_layout_t::dim_3,
                         "incompatible dof- and grad- layouts.");

      Mult_AB<Add>(Gt.layout.merge_23(), Gt,
                   grad_layout.merge_12(), grad_data,
                   dof_layout, dof_data);
   }

   /** @brief Multi-component assemble.
       qpt_layout is (NIP x NumComp),
       M_layout is (DOF x DOF x NumComp) */
   template <typename qpt_layout_t, typename qpt_data_t,
             typename M_layout_t, typename M_data_t>
   inline MFEM_ALWAYS_INLINE
   void Assemble(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
                 const M_layout_t &M_layout, M_data_t &M_data) const
   {
      // M_{i,j,k} = \sum_{s} B_{s,i} B_{s,j} qpt_data_{s,k}
      // Using TensorAssemble: <1,NIP,NC> --> <DOF,1,DOF,NC>
#if 0
      TensorAssemble<false>(
         B.layout, B,
         qpt_layout.template split_1<1,NIP>(), qpt_data,
         M_layout.template split_1<DOF,1>(), M_data);
#else
      TensorAssemble<false>(
         Bt.layout, Bt, B.layout, B,
         qpt_layout.template split_1<1,NIP>(), qpt_data,
         M_layout.template split_1<DOF,1>(), M_data);
#endif
   }

   /** @brief Multi-component assemble of grad-grad element matrices.
       qpt_layout is (NIP x DIM x DIM x NumComp), and
       D_layout is (DOF x DOF x NumComp). */
   template <typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   inline MFEM_ALWAYS_INLINE
   void AssembleGradGrad(const qpt_layout_t &qpt_layout,
                         const qpt_data_t   &qpt_data,
                         const D_layout_t   &D_layout,
                         D_data_t           &D_data) const
   {
      const int NC = qpt_layout_t::dim_4;
      typedef typename qpt_data_t::data_type entry_type;
      TTensor4<NIP,DIM,DOF,NC,entry_type> F;
      for (int k = 0; k < NC; k++)
      {
         // Next loop performs a batch of matrix-matrix products of size
         // (DIM x DIM) x (DIM x DOF) --> (DIM x DOF)
         for (int j = 0; j < NIP; j++)
         {
            Mult_AB<false>(qpt_layout.ind14(j,k), qpt_data,
                           G.layout.ind1(j), G,
                           F.layout.ind14(j,k), F);
         }
      }
      // (DOF x (NIP x DIM)) x ((NIP x DIM) x DOF x NC) --> (DOF x DOF x NC)
      Mult_2_1<false>(Gt.layout.merge_23(), Gt,
                      F.layout.merge_12(), F,
                      D_layout, D_data);
   }
};

template <int Dim, int DOF, int NIP, typename real_t>
class TProductShapeEvaluator;

/// ShapeEvaluator with 1D tensor-product structure
template <int DOF, int NIP, typename real_t>
class TProductShapeEvaluator<1, DOF, NIP, real_t>
{
protected:
   static const int TDOF = DOF; // total dofs

   TMatrix<NIP,DOF,real_t,true> B_1d, G_1d;
   TMatrix<DOF,NIP,real_t,true> Bt_1d, Gt_1d;

public:
   TProductShapeEvaluator() { }

   /** @brief Multi-component shape evaluation from DOFs to quadrature points.
       dof_layout is (DOF x NumComp) and qpt_layout is (NIP x NumComp). */
   template <typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
   inline MFEM_ALWAYS_INLINE
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      Mult_AB<false>(B_1d.layout, B_1d,
                     dof_layout, dof_data,
                     qpt_layout, qpt_data);
   }

   /** @brief Multi-component shape evaluation transpose from quadrature points
       to DOFs.  qpt_layout is (NIP x NumComp) and dof_layout is (DOF x NumComp). */
   template <bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      Mult_AB<Add>(Bt_1d.layout, Bt_1d,
                   qpt_layout, qpt_data,
                   dof_layout, dof_data);
   }

   /** @brief Multi-component gradient evaluation from DOFs to quadrature points.
       dof_layout is (DOF x NumComp) and grad_layout is (NIP x DIM x NumComp). */
   template <typename dof_layout_t, typename dof_data_t,
             typename grad_layout_t, typename grad_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcGrad(const dof_layout_t  &dof_layout,
                 const dof_data_t    &dof_data,
                 const grad_layout_t &grad_layout,
                 grad_data_t         &grad_data) const
   {
      // grad_data(nip,dim,comp) = sum(dof) G(nip,dim,dof) * dof_data(dof,comp)
      Mult_AB<false>(G_1d.layout, G_1d,
                     dof_layout, dof_data,
                     grad_layout.merge_12(), grad_data);
   }

   /** @brief Multi-component gradient evaluation transpose from quadrature points to
       DOFs. grad_layout is (NIP x DIM x NumComp), dof_layout is (DOF x NumComp). */
   template <bool Add,
             typename grad_layout_t, typename grad_data_t,
             typename dof_layout_t, typename dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcGradT(const grad_layout_t &grad_layout,
                  const grad_data_t   &grad_data,
                  const dof_layout_t  &dof_layout,
                  dof_data_t          &dof_data) const
   {
      // dof_data(dof,comp) +=
      //    sum(nip,dim) G(nip,dim,dof) * grad_data(nip,dim,comp)
      Mult_AB<Add>(Gt_1d.layout, Gt_1d,
                   grad_layout.merge_12(), grad_data,
                   dof_layout, dof_data);
   }

   /** @brief Multi-component assemble.
       qpt_layout is (NIP x NumComp), M_layout is (DOF x DOF x NumComp) */
   template <typename qpt_layout_t, typename qpt_data_t,
             typename M_layout_t, typename M_data_t>
   inline MFEM_ALWAYS_INLINE
   void Assemble(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
                 const M_layout_t &M_layout, M_data_t &M_data) const
   {
      // M_{i,j,k} = \sum_{s} B_1d_{s,i} B_{s,j} qpt_data_{s,k}
      // Using TensorAssemble: <1,NIP,NC> --> <DOF,1,DOF,NC>
#if 0
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         qpt_layout.template split_1<1,NIP>(), qpt_data,
         M_layout.template split_1<DOF,1>(), M_data);
#else
      TensorAssemble<false>(
         Bt_1d.layout, Bt_1d, B_1d.layout, B_1d,
         qpt_layout.template split_1<1,NIP>(), qpt_data,
         M_layout.template split_1<DOF,1>(), M_data);
#endif
   }

   /** @brief Multi-component assemble of grad-grad element matrices.
       qpt_layout is (NIP x DIM x DIM x NumComp), and
       D_layout is (DOF x DOF x NumComp). */
   template <typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   inline MFEM_ALWAYS_INLINE
   void AssembleGradGrad(const qpt_layout_t &qpt_layout,
                         const qpt_data_t   &qpt_data,
                         const D_layout_t   &D_layout,
                         D_data_t           &D_data) const
   {
      // D_{i,j,k} = \sum_{s} G_1d_{s,i} G_{s,j} qpt_data_{s,k}
      // Using TensorAssemble: <1,NIP,NC> --> <DOF,1,DOF,NC>
#if 0
      TensorAssemble<false>(
         G_1d.layout, G_1d,
         qpt_layout.merge_12().merge_23().template split_1<1,NIP>(), qpt_data,
         D_layout.template split_1<DOF,1>(), D_data);
#else
      TensorAssemble<false>(
         Gt_1d.layout, Gt_1d, G_1d.layout, G_1d,
         qpt_layout.merge_12().merge_23().template split_1<1,NIP>(), qpt_data,
         D_layout.template split_1<DOF,1>(), D_data);
#endif
   }
};

/// ShapeEvaluator with 2D tensor-product structure
template <int DOF, int NIP, typename real_t>
class TProductShapeEvaluator<2, DOF, NIP, real_t>
{
protected:
   TMatrix<NIP,DOF,real_t,true> B_1d, G_1d;
   TMatrix<DOF,NIP,real_t,true> Bt_1d, Gt_1d;

public:
   static const int TDOF = DOF*DOF; // total dofs
   static const int TNIP = NIP*NIP; // total qpts

   TProductShapeEvaluator() { }

   template <bool Dx, bool Dy,
             typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
   inline MFEM_ALWAYS_INLINE
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      const int NC = dof_layout_t::dim_2;
      typedef typename qpt_data_t::data_type entry_type;
      // DOF x DOF x NC --> NIP x DOF x NC --> NIP x NIP x NC
      TTensor3<NIP,DOF,NC,entry_type> A;

      // (1) A_{i,j,k} = \sum_s B_1d_{i,s} dof_data_{s,j,k}
      Mult_2_1<false>(B_1d.layout, Dx ? G_1d : B_1d,
                      dof_layout. template split_1<DOF,DOF>(), dof_data,
                      A.layout, A);
      // (2) qpt_data_{i,j,k} = \sum_s B_1d_{j,s} A_{i,s,k}
      Mult_1_2<false>(Bt_1d.layout, Dy ? Gt_1d : Bt_1d,
                      A.layout, A,
                      qpt_layout.template split_1<NIP,NIP>(), qpt_data);
   }

   /** @brief Multi-component shape evaluation from DOFs to quadrature points.
       dof_layout is (TDOF x NumComp) and qpt_layout is (TNIP x NumComp). */
   template <typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
   inline MFEM_ALWAYS_INLINE
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      Calc<false,false>(dof_layout, dof_data, qpt_layout, qpt_data);
   }

   template <bool Dx, bool Dy, bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      const int NC = dof_layout_t::dim_2;
      typedef typename qpt_data_t::data_type entry_type;
      // NIP x NIP X NC --> NIP x DOF x NC --> DOF x DOF x NC
      TTensor3<NIP,DOF,NC,entry_type> A;

      // (1) A_{i,j,k} = \sum_s B_1d_{s,j} qpt_data_{i,s,k}
      Mult_1_2<false>(B_1d.layout, Dy ? G_1d : B_1d,
                      qpt_layout.template split_1<NIP,NIP>(), qpt_data,
                      A.layout, A);
      // (2) dof_data_{i,j,k} = \sum_s B_1d_{s,i} A_{s,j,k}
      Mult_2_1<Add>(Bt_1d.layout, Dx ? Gt_1d : Bt_1d,
                    A.layout, A,
                    dof_layout.template split_1<DOF,DOF>(), dof_data);
   }

   /** @brief Multi-component shape evaluation transpose from quadrature points to DOFs.
       qpt_layout is (TNIP x NumComp) and dof_layout is (TDOF x NumComp). */
   template <bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      CalcT<false,false,Add>(qpt_layout, qpt_data, dof_layout, dof_data);
   }

   /** @brief Multi-component gradient evaluation from DOFs to quadrature points.
       dof_layout is (TDOF x NumComp) and grad_layout is (TNIP x DIM x NumComp). */
   template <typename dof_layout_t, typename dof_data_t,
             typename grad_layout_t, typename grad_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcGrad(const dof_layout_t  &dof_layout,
                 const dof_data_t    &dof_data,
                 const grad_layout_t &grad_layout,
                 grad_data_t &grad_data) const
   {
      Calc<true,false>(dof_layout, dof_data,
                       grad_layout.ind2(0), grad_data);
      Calc<false,true>(dof_layout, dof_data,
                       grad_layout.ind2(1), grad_data);
   }

   /** @brief Multi-component gradient evaluation transpose from quadrature points to
       DOFs. grad_layout is (TNIP x DIM x NumComp), dof_layout is
       (TDOF x NumComp). */
   template <bool Add,
             typename grad_layout_t, typename grad_data_t,
             typename dof_layout_t, typename dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcGradT(const grad_layout_t &grad_layout,
                  const grad_data_t   &grad_data,
                  const dof_layout_t  &dof_layout,
                  dof_data_t          &dof_data) const
   {
      CalcT<true,false, Add>(grad_layout.ind2(0), grad_data,
                             dof_layout, dof_data);
      CalcT<false,true,true>(grad_layout.ind2(1), grad_data,
                             dof_layout, dof_data);
   }

   /** @brief Multi-component assemble.
       qpt_layout is (TNIP x NumComp), M_layout is (TDOF x TDOF x NumComp) */
   template <typename qpt_layout_t, typename qpt_data_t,
             typename M_layout_t, typename M_data_t>
   inline MFEM_ALWAYS_INLINE
   void Assemble(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
                 const M_layout_t &M_layout, M_data_t &M_data) const
   {
      const int NC = qpt_layout_t::dim_2;
      typedef typename qpt_data_t::data_type entry_type;

      // Using TensorAssemble: <I,NIP,J> --> <DOF,I,DOF,J>

#if 0
      TTensor4<DOF,NIP,DOF,NC> A;
      // qpt_data<NIP1,NIP2,NC> --> A<DOF2,NIP1,DOF2,NC>
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         qpt_layout.template split_1<NIP,NIP>(), qpt_data,
         A.layout, A);
      // A<DOF2,NIP1,DOF2*NC> --> M<DOF1,DOF2,DOF1,DOF2*NC>
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         TTensor3<DOF,NIP,DOF*NC>::layout, A,
         M_layout.merge_23().template split_12<DOF,DOF,DOF,DOF*NC>(), M_data);
#elif 1
      TTensor4<DOF,NIP,DOF,NC,entry_type> A;
      // qpt_data<NIP1,NIP2,NC> --> A<DOF2,NIP1,DOF2,NC>
      TensorAssemble<false>(
         Bt_1d.layout, Bt_1d, B_1d.layout, B_1d,
         qpt_layout.template split_1<NIP,NIP>(), qpt_data,
         A.layout, A);
      // A<DOF2,NIP1,DOF2*NC> --> M<DOF1,DOF2,DOF1,DOF2*NC>
      TensorAssemble<false>(
         Bt_1d.layout, Bt_1d, B_1d.layout, B_1d,
         A.layout.merge_34(), A,
         M_layout.merge_23().template split_12<DOF,DOF,DOF,DOF*NC>(), M_data);
#else
      TTensor3<NIP,NIP,DOF> F3;
      TTensor4<NIP,NIP,DOF,DOF> F4;
      TTensor3<NIP,DOF,DOF*DOF> H3;
      for (int k = 0; k < NC; k++)
      {
         // <1,NIP1,NIP2> --> <1,NIP1,NIP2,DOF1>
         TensorProduct<AssignOp::Set>(
            qpt_layout.ind2(k).template split_1<NIP,NIP>().
            template split_1<1,NIP>(), qpt_data,
            B_1d.layout, B_1d, F3.layout.template split_1<1,NIP>(), F3);
         // <NIP1,NIP2,DOF1> --> <NIP1,NIP2,DOF1,DOF2>
         TensorProduct<AssignOp::Set>(
            F3.layout, F3, B_1d.layout, B_1d, F4.layout, F4);
         // <NIP1,NIP2,DOF1,DOF2> --> <NIP1,DOF2,DOF1,DOF2>
         Mult_1_2<false>(B_1d.layout, B_1d,
                         F4.layout.merge_34(), F4,
                         H3.layout, H3);
         // <NIP1,DOF2,DOF1,DOF2> --> <DOF1,DOF2,DOF1,DOF2>
         Mult_2_1<false>(Bt_1d.layout, Bt_1d,
                         H3.layout, H3,
                         M_layout.ind3(k).template split_1<DOF,DOF>(),
                         M_data);
      }
#endif
   }

   template <int D1, int D2, bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   inline MFEM_ALWAYS_INLINE
   void Assemble(const qpt_layout_t &qpt_layout,
                 const qpt_data_t   &qpt_data,
                 const D_layout_t   &D_layout,
                 D_data_t           &D_data) const
   {
      const int NC = qpt_layout_t::dim_2;
      typedef typename qpt_data_t::data_type entry_type;
      TTensor4<DOF,NIP,DOF,NC,entry_type> A;

      // Using TensorAssemble: <I,NIP,J> --> <DOF,I,DOF,J>

      // qpt_data<NIP1,NIP2,NC> --> A<DOF2,NIP1,DOF2,NC>
      TensorAssemble<false>(
         Bt_1d.layout, D1 == 0 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 == 0 ? B_1d : G_1d,
         qpt_layout.template split_1<NIP,NIP>(), qpt_data,
         A.layout, A);
      // A<DOF2,NIP1,DOF2*NC> --> M<DOF1,DOF2,DOF1,DOF2*NC>
      TensorAssemble<Add>(
         Bt_1d.layout, D1 == 1 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 == 1 ? B_1d : G_1d,
         A.layout.merge_34(), A,
         D_layout.merge_23().template split_12<DOF,DOF,DOF,DOF*NC>(), D_data);
   }

   /** @brief Multi-component assemble of grad-grad element matrices.
      qpt_layout is (TNIP x DIM x DIM x NumComp), and
      D_layout is (TDOF x TDOF x NumComp). */
   template <typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   inline MFEM_ALWAYS_INLINE
   void AssembleGradGrad(const qpt_layout_t &qpt_layout,
                         const qpt_data_t   &qpt_data,
                         const D_layout_t   &D_layout,
                         D_data_t           &D_data) const
   {
#if 1
      Assemble<0,0,false>(qpt_layout.ind23(0,0), qpt_data, D_layout, D_data);
      Assemble<1,0,true >(qpt_layout.ind23(1,0), qpt_data, D_layout, D_data);
      Assemble<0,1,true >(qpt_layout.ind23(0,1), qpt_data, D_layout, D_data);
      Assemble<1,1,true >(qpt_layout.ind23(1,1), qpt_data, D_layout, D_data);
#else
      const int NC = qpt_layout_t::dim_4;
      TTensor3<NIP,NIP,DOF> F3;
      TTensor4<NIP,NIP,DOF,DOF> F4;
      TTensor3<NIP,DOF,DOF*DOF> H3;

      for (int k = 0; k < NC; k++)
      {
         for (int d1 = 0; d1 < 2; d1++)
         {
            const AssignOp::Type Set = AssignOp::Set;
            const AssignOp::Type Add = AssignOp::Add;
            // <1,NIP1,NIP2> --> <1,NIP1,NIP2,DOF1>
            TensorProduct<Set>(qpt_layout.ind23(d1,0).ind2(k).
                               template split_1<NIP,NIP>().
                               template split_1<1,NIP>(), qpt_data,
                               G_1d.layout, G_1d,
                               F3.layout.template split_1<1,NIP>(), F3);
            // <NIP1,NIP2,DOF1> --> <NIP1,NIP2,DOF1,DOF2>
            TensorProduct<Set>(F3.layout, F3,
                               B_1d.layout, B_1d,
                               F4.layout, F4);
            // <1,NIP1,NIP2> --> <1,NIP1,NIP2,DOF1>
            TensorProduct<Set>(qpt_layout.ind23(d1,1).ind2(k).
                               template split_1<NIP,NIP>().
                               template split_1<1,NIP>(), qpt_data,
                               B_1d.layout, B_1d,
                               F3.layout.template split_1<1,NIP>(), F3);
            // <NIP1,NIP2,DOF1> --> <NIP1,NIP2,DOF1,DOF2>
            TensorProduct<Add>(F3.layout, F3,
                               G_1d.layout, G_1d,
                               F4.layout, F4);

            Mult_1_2<false>(B_1d.layout, d1 == 0 ? B_1d : G_1d,
                            F4.layout.merge_34(), F4,
                            H3.layout, H3);
            if (d1 == 0)
            {
               Mult_2_1<false>(Bt_1d.layout, Gt_1d,
                               H3.layout, H3,
                               D_layout.ind3(k).template split_1<DOF,DOF>(),
                               D_data);
            }
            else
            {
               Mult_2_1<true>(Bt_1d.layout, Bt_1d,
                              H3.layout, H3,
                              D_layout.ind3(k).template split_1<DOF,DOF>(),
                              D_data);
            }
         }
      }
#endif
   }
};

/// ShapeEvaluator with 3D tensor-product structure
template <int DOF, int NIP, typename real_t>
class TProductShapeEvaluator<3, DOF, NIP, real_t>
{
protected:
   TMatrix<NIP,DOF,real_t,true> B_1d, G_1d;
   TMatrix<DOF,NIP,real_t,true> Bt_1d, Gt_1d;

public:
   static const int TDOF = DOF*DOF*DOF; // total dofs
   static const int TNIP = NIP*NIP*NIP; // total qpts

   TProductShapeEvaluator() { }

   template <bool Dx, bool Dy, bool Dz,
             typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
   inline MFEM_ALWAYS_INLINE
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      const int NC = dof_layout_t::dim_2;
      typedef typename qpt_data_t::data_type entry_type;
      TVector<NIP*DOF*DOF*NC,entry_type> QDD;
      TVector<NIP*NIP*DOF*NC,entry_type> QQD;

      // QDD_{i,jj,k} = \sum_s B_1d_{i,s} dof_data_{s,jj,k}
      Mult_2_1<false>(B_1d.layout, Dx ? G_1d : B_1d,
                      dof_layout.template split_1<DOF,DOF*DOF>(), dof_data,
                      TTensor3<NIP,DOF*DOF,NC>::layout, QDD);
      // QQD_{i,j,kk} = \sum_s B_1d_{j,s} QDD_{i,s,kk}
      Mult_1_2<false>(Bt_1d.layout, Dy ? Gt_1d : Bt_1d,
                      TTensor3<NIP,DOF,DOF*NC>::layout, QDD,
                      TTensor3<NIP,NIP,DOF*NC>::layout, QQD);
      // qpt_data_{ii,j,k} = \sum_s B_1d_{j,s} QQD_{ii,s,k}
      Mult_1_2<false>(Bt_1d.layout, Dz ? Gt_1d : Bt_1d,
                      TTensor3<NIP*NIP,DOF,NC>::layout, QQD,
                      qpt_layout.template split_1<NIP*NIP,NIP>(), qpt_data);
   }

   /** @brief Multi-component shape evaluation from DOFs to quadrature points.
       dof_layout is (TDOF x NumComp) and qpt_layout is (TNIP x NumComp). */
   template <typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
   inline MFEM_ALWAYS_INLINE
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      Calc<false,false,false>(dof_layout, dof_data, qpt_layout, qpt_data);
   }

   template <bool Dx, bool Dy, bool Dz, bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      const int NC = dof_layout_t::dim_2;
      typedef typename qpt_data_t::data_type entry_type;
      TVector<NIP*DOF*DOF*NC,entry_type> QDD;
      TVector<NIP*NIP*DOF*NC,entry_type> QQD;

      // QQD_{ii,j,k} = \sum_s B_1d_{s,j} qpt_data_{ii,s,k}
      Mult_1_2<false>(B_1d.layout, Dz ? G_1d : B_1d,
                      qpt_layout.template split_1<NIP*NIP,NIP>(), qpt_data,
                      TTensor3<NIP*NIP,DOF,NC>::layout, QQD);
      // QDD_{i,j,kk} = \sum_s B_1d_{s,j} QQD_{i,s,kk}
      Mult_1_2<false>(B_1d.layout, Dy ? G_1d : B_1d,
                      TTensor3<NIP,NIP,DOF*NC>::layout, QQD,
                      TTensor3<NIP,DOF,DOF*NC>::layout, QDD);
      // dof_data_{i,jj,k} = \sum_s B_1d_{s,i} QDD_{s,jj,k}
      Mult_2_1<Add>(Bt_1d.layout, Dx ? Gt_1d : Bt_1d,
                    TTensor3<NIP,DOF*DOF,NC>::layout, QDD,
                    dof_layout.template split_1<DOF,DOF*DOF>(), dof_data);
   }

   /** @brief Multi-component shape evaluation transpose from quadrature points to DOFs.
       qpt_layout is (TNIP x NumComp) and dof_layout is (TDOF x NumComp). */
   template <bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      CalcT<false,false,false,Add>(qpt_layout, qpt_data, dof_layout, dof_data);
   }

   /** @brief Multi-component gradient evaluation from DOFs to quadrature points.
       dof_layout is (TDOF x NumComp) and grad_layout is (TNIP x DIM x NumComp). */
   template <typename dof_layout_t, typename dof_data_t,
             typename grad_layout_t, typename grad_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcGrad(const dof_layout_t  &dof_layout,
                 const dof_data_t    &dof_data,
                 const grad_layout_t &grad_layout,
                 grad_data_t         &grad_data) const
   {
      Calc<true,false,false>(dof_layout, dof_data,
                             grad_layout.ind2(0), grad_data);
      Calc<false,true,false>(dof_layout, dof_data,
                             grad_layout.ind2(1), grad_data);
      Calc<false,false,true>(dof_layout, dof_data,
                             grad_layout.ind2(2), grad_data);
      // optimization: the x-transition (dof->nip) is done twice -- once for the
      // y-derivatives and second time for the z-derivatives.
   }

   /** @brief Multi-component gradient evaluation transpose from quadrature points to
       DOFs. grad_layout is (TNIP x DIM x NumComp), dof_layout is
       (TDOF x NumComp). */
   template <bool Add,
             typename grad_layout_t, typename grad_data_t,
             typename dof_layout_t, typename dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void CalcGradT(const grad_layout_t &grad_layout,
                  const grad_data_t   &grad_data,
                  const dof_layout_t  &dof_layout,
                  dof_data_t          &dof_data) const
   {
      CalcT<true,false,false, Add>(grad_layout.ind2(0), grad_data,
                                   dof_layout, dof_data);
      CalcT<false,true,false,true>(grad_layout.ind2(1), grad_data,
                                   dof_layout, dof_data);
      CalcT<false,false,true,true>(grad_layout.ind2(2), grad_data,
                                   dof_layout, dof_data);
   }

   /** @brief Multi-component assemble.
       qpt_layout is (TNIP x NumComp), M_layout is (TDOF x TDOF x NumComp) */
   template <typename qpt_layout_t, typename qpt_data_t,
             typename M_layout_t, typename M_data_t>
   inline MFEM_ALWAYS_INLINE
   void Assemble(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
                 const M_layout_t &M_layout, M_data_t &M_data) const
   {
      const int NC = qpt_layout_t::dim_2;
      typedef typename qpt_data_t::data_type entry_type;
      TTensor4<DOF,NIP*NIP,DOF,NC,entry_type> A1;
      TTensor4<DOF,DOF*NIP,DOF,DOF*NC,entry_type> A2;

      // Using TensorAssemble: <I,NIP,J> --> <DOF,I,DOF,J>

#if 0
      // qpt_data<NIP1*NIP2,NIP3,NC> --> A1<DOF3,NIP1*NIP2,DOF3,NC>
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         qpt_layout.template split_1<NIP*NIP,NIP>(), qpt_data,
         A1.layout, A1);
      // A1<DOF3*NIP1,NIP2,DOF3*NC> --> A2<DOF2,DOF3*NIP1,DOF2,DOF3*NC>
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         TTensor3<DOF*NIP,NIP,DOF*NC>::layout, A1,
         A2.layout, A2);
      // A2<DOF2*DOF3,NIP1,DOF2*DOF3*NC> --> M<DOF1,DOF2*DOF3,DOF1,DOF2*DOF3*NC>
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         TTensor3<DOF*DOF,NIP,DOF*DOF*NC>::layout, A2,
         M_layout.merge_23().template split_12<DOF,DOF*DOF,DOF,DOF*DOF*NC>(),
         M_data);
#else
      // qpt_data<NIP1*NIP2,NIP3,NC> --> A1<DOF3,NIP1*NIP2,DOF3,NC>
      TensorAssemble<false>(
         Bt_1d.layout, Bt_1d, B_1d.layout, B_1d,
         qpt_layout.template split_1<NIP*NIP,NIP>(), qpt_data,
         A1.layout, A1);
      // A1<DOF3*NIP1,NIP2,DOF3*NC> --> A2<DOF2,DOF3*NIP1,DOF2,DOF3*NC>
      TensorAssemble<false>(
         Bt_1d.layout, Bt_1d, B_1d.layout, B_1d,
         TTensor3<DOF*NIP,NIP,DOF*NC>::layout, A1,
         A2.layout, A2);
      // A2<DOF2*DOF3,NIP1,DOF2*DOF3*NC> --> M<DOF1,DOF2*DOF3,DOF1,DOF2*DOF3*NC>
      TensorAssemble<false>(
         Bt_1d.layout, Bt_1d, B_1d.layout, B_1d,
         TTensor3<DOF*DOF,NIP,DOF*DOF*NC>::layout, A2,
         M_layout.merge_23().template split_12<DOF,DOF*DOF,DOF,DOF*DOF*NC>(),
         M_data);
#endif
   }

   template <int D1, int D2, bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   inline MFEM_ALWAYS_INLINE
   void Assemble(const qpt_layout_t &qpt_layout,
                 const qpt_data_t   &qpt_data,
                 const D_layout_t   &D_layout,
                 D_data_t           &D_data) const
   {
      const int NC = qpt_layout_t::dim_2;
      typedef typename qpt_data_t::data_type entry_type;
      TTensor4<DOF,NIP*NIP,DOF,NC,entry_type> A1;
      TTensor4<DOF,DOF*NIP,DOF,DOF*NC,entry_type> A2;

      // Using TensorAssemble: <I,NIP,J> --> <DOF,I,DOF,J>

      // qpt_data<NIP1*NIP2,NIP3,NC> --> A1<DOF3,NIP1*NIP2,DOF3,NC>
      TensorAssemble<false>(
         Bt_1d.layout, D1 != 2 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 != 2 ? B_1d : G_1d,
         qpt_layout.template split_1<NIP*NIP,NIP>(), qpt_data,
         A1.layout, A1);
      // A1<DOF3*NIP1,NIP2,DOF3*NC> --> A2<DOF2,DOF3*NIP1,DOF2,DOF3*NC>
      TensorAssemble<false>(
         Bt_1d.layout, D1 != 1 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 != 1 ? B_1d : G_1d,
         TTensor3<DOF*NIP,NIP,DOF*NC>::layout, A1,
         A2.layout, A2);
      // A2<DOF2*DOF3,NIP1,DOF2*DOF3*NC> --> M<DOF1,DOF2*DOF3,DOF1,DOF2*DOF3*NC>
      TensorAssemble<Add>(
         Bt_1d.layout, D1 != 0 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 != 0 ? B_1d : G_1d,
         TTensor3<DOF*DOF,NIP,DOF*DOF*NC>::layout, A2,
         D_layout.merge_23().template split_12<DOF,DOF*DOF,DOF,DOF*DOF*NC>(),
         D_data);
   }

#if 0
   template <typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   inline MFEM_ALWAYS_INLINE
   void Assemble(int D1, int D2,
                 const qpt_layout_t &qpt_layout,
                 const qpt_data_t   &qpt_data,
                 const D_layout_t   &D_layout,
                 D_data_t           &D_data) const
   {
      const int NC = qpt_layout_t::dim_2;
      TTensor4<DOF,NIP*NIP,DOF,NC> A1;
      TTensor4<DOF,DOF*NIP,DOF,DOF*NC> A2;

      // Using TensorAssemble: <I,NIP,J> --> <DOF,I,DOF,J>

      // qpt_data<NIP1*NIP2,NIP3,NC> --> A1<DOF3,NIP1*NIP2,DOF3,NC>
      TensorAssemble<false>(
         Bt_1d.layout, D1 != 2 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 != 2 ? B_1d : G_1d,
         qpt_layout.template split_1<NIP*NIP,NIP>(), qpt_data,
         A1.layout, A1);
      // A1<DOF3*NIP1,NIP2,DOF3*NC> --> A2<DOF2,DOF3*NIP1,DOF2,DOF3*NC>
      TensorAssemble<false>(
         Bt_1d.layout, D1 != 1 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 != 1 ? B_1d : G_1d,
         TTensor3<DOF*NIP,NIP,DOF*NC>::layout, A1,
         A2.layout, A2);
      // A2<DOF2*DOF3,NIP1,DOF2*DOF3*NC> --> M<DOF1,DOF2*DOF3,DOF1,DOF2*DOF3*NC>
      TensorAssemble<true>(
         Bt_1d.layout, D1 != 0 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 != 0 ? B_1d : G_1d,
         TTensor3<DOF*DOF,NIP,DOF*DOF*NC>::layout, A2,
         D_layout.merge_23().template split_12<DOF,DOF*DOF,DOF,DOF*DOF*NC>(),
         D_data);
   }
#endif

   /** @brief Multi-component assemble of grad-grad element matrices.
       qpt_layout is (TNIP x DIM x DIM x NumComp), and
       D_layout is (TDOF x TDOF x NumComp). */
   template <typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   inline MFEM_ALWAYS_INLINE
   void AssembleGradGrad(const qpt_layout_t &qpt_layout,
                         const qpt_data_t   &qpt_data,
                         const D_layout_t   &D_layout,
                         D_data_t           &D_data) const
   {
#if 1
      // NOTE: This function compiles into a large chunk of machine code
      Assemble<0,0,false>(qpt_layout.ind23(0,0), qpt_data, D_layout, D_data);
      Assemble<1,0,true >(qpt_layout.ind23(1,0), qpt_data, D_layout, D_data);
      Assemble<2,0,true >(qpt_layout.ind23(2,0), qpt_data, D_layout, D_data);
      Assemble<0,1,true >(qpt_layout.ind23(0,1), qpt_data, D_layout, D_data);
      Assemble<1,1,true >(qpt_layout.ind23(1,1), qpt_data, D_layout, D_data);
      Assemble<2,1,true >(qpt_layout.ind23(2,1), qpt_data, D_layout, D_data);
      Assemble<0,2,true >(qpt_layout.ind23(0,2), qpt_data, D_layout, D_data);
      Assemble<1,2,true >(qpt_layout.ind23(1,2), qpt_data, D_layout, D_data);
      Assemble<2,2,true >(qpt_layout.ind23(2,2), qpt_data, D_layout, D_data);
#else
      TAssign<AssignOp::Set>(D_layout, D_data, 0.0);
      for (int d2 = 0; d2 < 3; d2++)
      {
         for (int d1 = 0; d1 < 3; d1++)
         {
            Assemble(d1, d2, qpt_layout.ind23(d1,d2), qpt_data,
                     D_layout, D_data);
         }
      }
#endif
   }
};

/// ShapeEvaluator with tensor-product structure in any dimension
template <class FE, class IR, typename real_t>
class ShapeEvaluator_base<FE, IR, true, real_t>
   : public TProductShapeEvaluator<FE::dim, FE::dofs_1d, IR::qpts_1d, real_t>
{
protected:
   typedef TProductShapeEvaluator<FE::dim,FE::dofs_1d,
           IR::qpts_1d,real_t> base_class;
   using base_class::B_1d;
   using base_class::Bt_1d;
   using base_class::G_1d;
   using base_class::Gt_1d;

public:
   ShapeEvaluator_base(const FE &fe)
   {
      fe.Calc1DShapes(IR::Get1DIntRule(), B_1d.data, G_1d.data);
      TAssign<AssignOp::Set>(Bt_1d.layout, Bt_1d,
                             B_1d.layout.transpose_12(), B_1d);
      TAssign<AssignOp::Set>(Gt_1d.layout, Gt_1d,
                             G_1d.layout.transpose_12(), G_1d);
   }

   // default copy constructor
};

/// General ShapeEvaluator for any scalar FE type (L2 or H1)
template <class FE, class IR, typename real_t>
class ShapeEvaluator
   : public ShapeEvaluator_base<FE,IR,FE::tensor_prod && IR::tensor_prod,real_t>
{
public:
   typedef real_t real_type;
   static const int dim  = FE::dim;
   static const int qpts = IR::qpts;
   static const bool tensor_prod = FE::tensor_prod && IR::tensor_prod;
   typedef FE FE_type;
   typedef IR IR_type;
   typedef ShapeEvaluator_base<FE,IR,tensor_prod,real_t> base_class;

   using base_class::Calc;
   using base_class::CalcT;
   using base_class::CalcGrad;
   using base_class::CalcGradT;

   ShapeEvaluator(const FE &fe) : base_class(fe) { }

   // default copy constructor
};


/** @brief Field evaluators -- values of a given global FE grid function
    This is roughly speaking a templated version of GridFunction
*/
template <typename FESpace_t, typename VecLayout_t, typename IR,
          typename complex_t, typename real_t>
class FieldEvaluator_base
{
protected:
   typedef typename FESpace_t::FE_type       FE_type;
   typedef ShapeEvaluator<FE_type,IR,real_t> ShapeEval_type;

   FESpace_t       fespace;
   ShapeEval_type  shapeEval;
   VecLayout_t     vec_layout;

   /// With this constructor, fespace is a shallow copy.
   inline MFEM_ALWAYS_INLINE
   FieldEvaluator_base(const FESpace_t &tfes, const ShapeEval_type &shape_eval,
                       const VecLayout_t &vec_layout)
      : fespace(tfes),
        shapeEval(shape_eval),
        vec_layout(vec_layout)
   { }

   /// This constructor creates new fespace, not a shallow copy.
   inline MFEM_ALWAYS_INLINE
   FieldEvaluator_base(const FE_type &fe, const FiniteElementSpace &fes)
      : fespace(fe, fes), shapeEval(fe), vec_layout(fes)
   { }
};

/// complex_t - dof/qpt data type, real_t - ShapeEvaluator (FE basis) data type
template <typename FESpace_t, typename VecLayout_t, typename IR,
          typename complex_t = double, typename real_t = double>
class FieldEvaluator
   : public FieldEvaluator_base<FESpace_t,VecLayout_t,IR,complex_t,real_t>
{
public:
   typedef complex_t                         complex_type;
   typedef FESpace_t                         FESpace_type;
   typedef typename FESpace_t::FE_type       FE_type;
   typedef ShapeEvaluator<FE_type,IR,real_t> ShapeEval_type;
   typedef VecLayout_t                       VecLayout_type;

   // this type
   typedef FieldEvaluator<FESpace_t,VecLayout_t,IR,complex_t,real_t> T_type;

   static const int dofs = FE_type::dofs;
   static const int dim  = FE_type::dim;
   static const int qpts = IR::qpts;
   static const int vdim = VecLayout_t::vec_dim;

protected:

   typedef FieldEvaluator_base<FESpace_t,VecLayout_t,IR,complex_t,real_t>
   base_class;

   using base_class::fespace;
   using base_class::shapeEval;
   using base_class::vec_layout;
   const complex_t *data_in;
   complex_t       *data_out;

public:
   /// With this constructor, fespace is a shallow copy of tfes.
   inline MFEM_ALWAYS_INLINE
   FieldEvaluator(const FESpace_t &tfes, const ShapeEval_type &shape_eval,
                  const VecLayout_type &vec_layout,
                  const complex_t *global_data_in, complex_t *global_data_out)
      : base_class(tfes, shape_eval, vec_layout),
        data_in(global_data_in),
        data_out(global_data_out)
   { }

   /// With this constructor, fespace is a shallow copy of f.fespace.
   inline MFEM_ALWAYS_INLINE
   FieldEvaluator(const FieldEvaluator &f,
                  const complex_t *global_data_in, complex_t *global_data_out)
      : base_class(f.fespace, f.shapeEval, f.vec_layout),
        data_in(global_data_in),
        data_out(global_data_out)
   { }

   /// This constructor creates a new fespace, not a shallow copy.
   inline MFEM_ALWAYS_INLINE
   FieldEvaluator(const FiniteElementSpace &fes,
                  const complex_t *global_data_in, complex_t *global_data_out)
      : base_class(FE_type(*fes.FEColl()), fes),
        data_in(global_data_in),
        data_out(global_data_out)
   { }

   // Default copy constructor

   inline MFEM_ALWAYS_INLINE FESpace_type &FESpace() { return fespace; }
   inline MFEM_ALWAYS_INLINE ShapeEval_type &ShapeEval() { return shapeEval; }
   inline MFEM_ALWAYS_INLINE VecLayout_type &VecLayout() { return vec_layout; }

   inline MFEM_ALWAYS_INLINE
   void SetElement(int el)
   {
      fespace.SetElement(el);
   }

   /// val_layout_t is (qpts x vdim x NE)
   template <typename val_layout_t, typename val_data_t>
   inline MFEM_ALWAYS_INLINE
   void GetValues(int el, const val_layout_t &l, val_data_t &vals)
   {
      const int ne = val_layout_t::dim_3;
      TTensor3<dofs,vdim,ne,typename val_data_t::data_type> val_dofs;
      SetElement(el);
      fespace.VectorExtract(vec_layout, data_in, val_dofs.layout, val_dofs);
      shapeEval.Calc(val_dofs.layout.merge_23(), val_dofs, l.merge_23(), vals);
   }

   /// grad_layout_t is (qpts x dim x vdim x NE)
   template <typename grad_layout_t, typename grad_data_t>
   inline MFEM_ALWAYS_INLINE
   void GetGradients(int el, const grad_layout_t &l, grad_data_t &grad)
   {
      const int ne = grad_layout_t::dim_4;
      TTensor3<dofs,vdim,ne,typename grad_data_t::data_type> val_dofs;
      SetElement(el);
      fespace.VectorExtract(vec_layout, data_in, val_dofs.layout, val_dofs);
      shapeEval.CalcGrad(val_dofs.layout.merge_23(), val_dofs,
                         l.merge_34(), grad);
   }

   // TODO: add method GetValuesAndGradients()

   template <typename DataType>
   inline MFEM_ALWAYS_INLINE
   void Eval(DataType &F)
   {
      // T.SetElement() must be called outside
      Action<DataType::InData,true>::Eval(vec_layout, *this, F);
   }

   template <typename DataType>
   inline MFEM_ALWAYS_INLINE
   void Eval(int el, DataType &F)
   {
      SetElement(el);
      Eval(F);
   }

   template <bool Add, typename DataType>
   inline MFEM_ALWAYS_INLINE
   void Assemble(DataType &F)
   {
      // T.SetElement() must be called outside
      Action<DataType::OutData,true>::
      template Assemble<Add>(vec_layout, *this, F);
   }

   template <bool Add, typename DataType>
   inline MFEM_ALWAYS_INLINE
   void Assemble(int el, DataType &F)
   {
      SetElement(el);
      Assemble<Add>(F);
   }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
   template <typename DataType>
   inline MFEM_ALWAYS_INLINE
   void EvalSerialized(const typename DataType::vcomplex_t *loc_dofs,
                       DataType &F)
   {
      Action<DataType::InData,true>::EvalSerialized(*this, loc_dofs, F);
   }

   template <bool Add, typename DataType>
   inline MFEM_ALWAYS_INLINE
   void AssembleSerialized(const DataType &F,
                           typename DataType::vcomplex_t *loc_dofs)
   {
      Action<DataType::OutData,true>::
      template AssembleSerialized<Add>(*this, F, loc_dofs);
   }
#endif

   /** @brief Enumeration for the data type used by the Eval() and Assemble() methods.
       The types can be obtained by summing constants from this enumeration and used
       as a template parameter in struct Data. */
   enum InOutData
   {
      None      = 0,
      Values    = 1,
      Gradients = 2
   };

   /** @brief  Auxiliary templated struct AData, used by the Eval() and Assemble()
       methods.

       The template parameter IOData is "bitwise or" of constants from
       the enum InOutData. The parameter NE is the number of elements to be
       processed in the Eval() and Assemble() methods. */
   template<int IOData, typename impl_traits_t> struct AData;

   template <typename it_t> struct AData<0,it_t> // 0 = None
   {
      // Do we need this?
   };

   template <typename it_t> struct AData<1,it_t> // 1 = Values
   {
      static const int ne = it_t::batch_size;
      typedef typename it_t::vcomplex_t vcomplex_t;
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
      typedef TTensor3<dofs,vdim,ne,vcomplex_t,true> val_dofs_t;
      val_dofs_t val_dofs;
#else
      typedef TTensor3<dofs,vdim,ne,vcomplex_t> val_dofs_t;
#endif
      TTensor3<qpts,vdim,ne,vcomplex_t>      val_qpts;
   };

   template <typename it_t> struct AData<2,it_t> // 2 = Gradients
   {
      static const int ne = it_t::batch_size;
      typedef typename it_t::vcomplex_t vcomplex_t;
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
      typedef TTensor3<dofs,vdim,ne,vcomplex_t,true> val_dofs_t;
      val_dofs_t val_dofs;
#else
      typedef TTensor3<dofs,vdim,ne,vcomplex_t> val_dofs_t;
#endif
      TTensor4<qpts,dim,vdim,ne,vcomplex_t>      grad_qpts;
   };

   template <typename it_t> struct AData<3,it_t> // 3 = Values+Gradients
   {
      static const int ne = it_t::batch_size;
      typedef typename it_t::vcomplex_t vcomplex_t;
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
      typedef TTensor3<dofs,vdim,ne,vcomplex_t,true> val_dofs_t;
      val_dofs_t val_dofs;
#else
      typedef TTensor3<dofs,vdim,ne,vcomplex_t> val_dofs_t;
#endif
      TTensor3<qpts,    vdim,ne,vcomplex_t,true>  val_qpts;
      TTensor4<qpts,dim,vdim,ne,vcomplex_t>      grad_qpts;
   };

   /** @brief This struct is similar to struct AData, adding separate static data
       members for the input (InData) and output (OutData) data types. */
   template <int IData, int OData, typename it_t>
   struct BData : public AData<IData|OData,it_t>
   {
      typedef T_type eval_type;
      static const int InData = IData;
      static const int OutData = OData;
   };

   /** @brief This struct implements the input (Eval, EvalSerialized) and output
       (Assemble, AssembleSerialized) operations for the given Ops.
       Ops is "bitwise or" of constants from the enum InOutData. */
   template <int Ops, bool dummy> struct Action;

   template <bool dummy> struct Action<0,dummy> // 0 = None
   {
      // Do we need this?
   };

   template <bool dummy> struct Action<1,dummy> // 1 = Values
   {
      template <typename vec_layout_t, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void Eval(const vec_layout_t &l, T_type &T, AData_t &D)
      {
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
         typename AData_t::val_dofs_t &val_dofs = D.val_dofs;
#else
         typename AData_t::val_dofs_t val_dofs;
#endif
         T.fespace.VectorExtract(l, T.data_in, val_dofs.layout, val_dofs);
         T.shapeEval.Calc(val_dofs.layout.merge_23(), val_dofs,
                          D.val_qpts.layout.merge_23(), D.val_qpts);
      }

      template <bool Add, typename vec_layout_t, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void Assemble(const vec_layout_t &l, T_type &T, AData_t &D)
      {
         const AssignOp::Type Op = Add ? AssignOp::Add : AssignOp::Set;
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
         typename AData_t::val_dofs_t &val_dofs = D.val_dofs;
#else
         typename AData_t::val_dofs_t val_dofs;
#endif
         T.shapeEval.template CalcT<false>(
            D.val_qpts.layout.merge_23(), D.val_qpts,
            val_dofs.layout.merge_23(), val_dofs);
         T.fespace.template VectorAssemble<Op>(
            val_dofs.layout, val_dofs, l, T.data_out);
      }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
      template <typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void EvalSerialized(T_type &T,
                          const typename AData_t::vcomplex_t *loc_dofs,
                          AData_t &D)
      {
         T.shapeEval.Calc(AData_t::val_dofs_t::layout.merge_23(), loc_dofs,
                          D.val_qpts.layout.merge_23(), D.val_qpts);
      }

      template <bool Add, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void AssembleSerialized(T_type &T, const AData_t &D,
                              typename AData_t::vcomplex_t *loc_dofs)
      {
         T.shapeEval.template CalcT<Add>(
            D.val_qpts.layout.merge_23(), D.val_qpts,
            AData_t::val_dofs_t::layout.merge_23(), loc_dofs);
      }
#endif
   };

   template <bool dummy> struct Action<2,dummy> // 2 = Gradients
   {
      template <typename vec_layout_t, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void Eval(const vec_layout_t &l, T_type &T, AData_t &D)
      {
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
         typename AData_t::val_dofs_t &val_dofs = D.val_dofs;
#else
         typename AData_t::val_dofs_t val_dofs;
#endif
         T.fespace.VectorExtract(l, T.data_in, val_dofs.layout, val_dofs);
         T.shapeEval.CalcGrad(val_dofs.layout.merge_23(),  val_dofs,
                              D.grad_qpts.layout.merge_34(), D.grad_qpts);
      }

      template <bool Add, typename vec_layout_t, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void Assemble(const vec_layout_t &l, T_type &T, AData_t &D)
      {
         const AssignOp::Type Op = Add ? AssignOp::Add : AssignOp::Set;
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
         typename AData_t::val_dofs_t &val_dofs = D.val_dofs;
#else
         typename AData_t::val_dofs_t val_dofs;
#endif
         T.shapeEval.template CalcGradT<false>(
            D.grad_qpts.layout.merge_34(), D.grad_qpts,
            val_dofs.layout.merge_23(), val_dofs);
         T.fespace.template VectorAssemble<Op>(
            val_dofs.layout, val_dofs, l, T.data_out);
      }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
      template <typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void EvalSerialized(T_type &T,
                          const typename AData_t::vcomplex_t *loc_dofs,
                          AData_t &D)
      {
         T.shapeEval.CalcGrad(AData_t::val_dofs_t::layout.merge_23(), loc_dofs,
                              D.grad_qpts.layout.merge_34(), D.grad_qpts);
      }

      template <bool Add, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void AssembleSerialized(T_type &T, const AData_t &D,
                              typename AData_t::vcomplex_t *loc_dofs)
      {
         T.shapeEval.template CalcGradT<Add>(
            D.grad_qpts.layout.merge_34(), D.grad_qpts,
            AData_t::val_dofs_t::layout.merge_23(), loc_dofs);
      }
#endif
   };

   template <bool dummy> struct Action<3,dummy> // 3 = Values+Gradients
   {
      template <typename vec_layout_t, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void Eval(const vec_layout_t &l, T_type &T, AData_t &D)
      {
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
         typename AData_t::val_dofs_t &val_dofs = D.val_dofs;
#else
         typename AData_t::val_dofs_t val_dofs;
#endif
         T.fespace.VectorExtract(l, T.data_in, val_dofs.layout, val_dofs);
         T.shapeEval.Calc(val_dofs.layout.merge_23(), val_dofs,
                          D.val_qpts.layout.merge_23(), D.val_qpts);
         T.shapeEval.CalcGrad(val_dofs.layout.merge_23(),  val_dofs,
                              D.grad_qpts.layout.merge_34(), D.grad_qpts);
      }

      template <bool Add, typename vec_layout_t, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void Assemble(const vec_layout_t &l, T_type &T, AData_t &D)
      {
         const AssignOp::Type Op = Add ? AssignOp::Add : AssignOp::Set;
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
         typename AData_t::val_dofs_t &val_dofs = D.val_dofs;
#else
         typename AData_t::val_dofs_t val_dofs;
#endif
         T.shapeEval.template CalcT<false>(
            D.val_qpts.layout.merge_23(), D.val_qpts,
            val_dofs.layout.merge_23(), val_dofs);
         T.shapeEval.template CalcGradT<true>(
            D.grad_qpts.layout.merge_34(), D.grad_qpts,
            val_dofs.layout.merge_23(),  val_dofs);
         T.fespace.template VectorAssemble<Op>(
            val_dofs.layout, val_dofs, l, T.data_out);
      }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
      template <typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void EvalSerialized(T_type &T,
                          const typename AData_t::vcomplex_t *loc_dofs,
                          AData_t &D)
      {
         T.shapeEval.Calc(AData_t::val_dofs_t::layout.merge_23(), loc_dofs,
                          D.val_qpts.layout.merge_23(), D.val_qpts);
         T.shapeEval.CalcGrad(AData_t::val_dofs_t::layout.merge_23(), loc_dofs,
                              D.grad_qpts.layout.merge_34(), D.grad_qpts);
      }

      template <bool Add, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void AssembleSerialized(T_type &T, const AData_t &D,
                              typename AData_t::vcomplex_t *loc_dofs)
      {
         T.shapeEval.template CalcT<Add>(
            D.val_qpts.layout.merge_23(), D.val_qpts,
            AData_t::val_dofs_t::layout.merge_23(), loc_dofs);
         T.shapeEval.template CalcGradT<true>(
            D.grad_qpts.layout.merge_34(), D.grad_qpts,
            AData_t::val_dofs_t::layout.merge_23(), loc_dofs);
      }
#endif
   };

   /** @brief This struct implements element matrix computation for some combinations
       of input (InOps) and output (OutOps) operations. */
   template <int InOps, int OutOps, typename it_t> struct TElementMatrix;

   // Case 1,1 = Values,Values
   template <typename it_t> struct TElementMatrix<1,1,it_t>
   {
      // qpt_layout_t is (nip), M_layout_t is (dof x dof)
      // it_t::batch_size = 1 is assumed
      template <typename qpt_layout_t, typename qpt_data_t,
                typename M_layout_t, typename M_data_t>
      static inline MFEM_ALWAYS_INLINE
      void Compute(const qpt_layout_t &a, const qpt_data_t &A,
                   const M_layout_t &m, M_data_t &M, ShapeEval_type &ev)
      {
         ev.Assemble(a.template split_1<qpts,1>(), A,
                     m.template split_2<dofs,1>(), M);
      }
   };

   // Case 2,2 = Gradients,Gradients
   template <typename it_t> struct TElementMatrix<2,2,it_t>
   {
      /** @brief Assemble element mass matrix
          @param a the layout for the quadrature point data
          @param A given quadrature point data for element (incl. coefficient,
                 geometry)
          @param m the layout for the resulting element mass matrix
          @param M the resulting element mass matrix
          @param ev the shape evaluator
          qpt_layout_t is (nip), M_layout_t is (dof x dof)
          NE = 1 is assumed */
      template <typename qpt_layout_t, typename qpt_data_t,
                typename M_layout_t, typename M_data_t>
      static inline MFEM_ALWAYS_INLINE
      void Compute(const qpt_layout_t &a, const qpt_data_t &A,
                   const M_layout_t &m, M_data_t &M, ShapeEval_type &ev)
      {
         ev.AssembleGradGrad(a.template split_3<dim,1>(), A,
                             m.template split_2<dofs,1>(), M);
      }
   };

   template <typename kernel_t, typename impl_traits_t> struct Spec
   {
      static const int InData =
         Values*kernel_t::in_values + Gradients*kernel_t::in_gradients;
      static const int OutData =
         Values*kernel_t::out_values + Gradients*kernel_t::out_gradients;

      typedef BData<InData,OutData,impl_traits_t>          DataType;
      typedef TElementMatrix<InData,OutData,impl_traits_t> ElementMatrix;
   };
};

} // namespace mfem

#endif // MFEM_TEMPLATE_EVALUATOR
