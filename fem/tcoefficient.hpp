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

#ifndef MFEM_TEMPLATE_COEFFICIENT
#define MFEM_TEMPLATE_COEFFICIENT

#include "../config/tconfig.hpp"
#include "../linalg/ttensor.hpp"
#include "../linalg/tlayout.hpp"
#include "../linalg/vector.hpp"
#include "gridfunc.hpp"

namespace mfem
{

/// Templated coefficient classes, cf. coefficient.?pp

class TCoefficient
{
public:
   static const int rank = 0; // 0 - scalar, 1 - vector, 2 - matrix
   static const bool is_const          = false;
   static const bool uses_coordinates  = false;
   static const bool uses_Jacobians    = false;
   static const bool uses_attributes   = false;
   static const bool uses_element_idxs = false;
};

template <typename complex_t = real_t>
class TConstantCoefficient : public TCoefficient
{
public:
   static const bool is_const = true;
   typedef complex_t complex_type;

   complex_t value;

   TConstantCoefficient(complex_t val) : value(val) { }
   // default copy constructor

   // T_result_t is the transformation result type (not used here).
   template <typename T_result_t, typename c_layout_t, typename c_data_t>
   inline MFEM_ALWAYS_INLINE
   void Eval(const T_result_t &T, const c_layout_t &l, c_data_t &c) const
   {
      TAssign<AssignOp::Set>(l, c, value);
   }
};


/** @brief Function coefficient.
    @tparam Func has to implement at least one of the following methods,
    depending on the dimension that will be used:
    complex_t Eval1D(real_t);
    complex_t Eval2D(real_t,real_t);
    complex_t Eval3D(real_t,real_t,real_t);
    Use MFEM_FLOPS_ADD() to count flops inside Eval*D. */
template <typename Func, typename complex_t = real_t>
class TFunctionCoefficient : public TCoefficient
{
public:
   static const bool uses_coordinates = true;
   typedef complex_t complex_type;

protected:
   Func F;

   template <int dim, bool dummy> struct Dim;
   template <bool dummy> struct Dim<1,dummy>
   {
      template <typename T_result_t, typename c_layout_t, typename c_data_t>
      static inline MFEM_ALWAYS_INLINE
      void Eval(Func &F, const T_result_t &T, const c_layout_t &l, c_data_t &c)
      {
         const int qpts = T_result_t::x_type::layout_type::dim_1;
         const int ne   = T_result_t::x_type::layout_type::dim_3;
         const int vs   = sizeof(T.x[0])/sizeof(T.x[0][0]);
         for (int k = 0; k < ne; k++)
         {
            for (int i = 0; i < qpts; i++)
            {
               for (int s = 0; s < vs; s++)
               {
                  c[l.ind(i,k)][s] = F.Eval1D(T.x(i,0,k)[s]);
               }
            }
         }
      }
   };
   template <bool dummy> struct Dim<2,dummy>
   {
      template <typename T_result_t, typename c_layout_t, typename c_data_t>
      static inline MFEM_ALWAYS_INLINE
      void Eval(Func &F, const T_result_t &T, const c_layout_t &l, c_data_t &c)
      {
         const int qpts = T_result_t::x_type::layout_type::dim_1;
         const int ne   = T_result_t::x_type::layout_type::dim_3;
         const int vs   = sizeof(T.x[0])/sizeof(T.x[0][0]);
         for (int k = 0; k < ne; k++)
         {
            for (int i = 0; i < qpts; i++)
            {
               for (int s = 0; s < vs; s++)
               {
                  c[l.ind(i,k)][s] = F.Eval2D(T.x(i,0,k)[s], T.x(i,1,k)[s]);
               }
            }
         }
      }
   };
   template <bool dummy> struct Dim<3,dummy>
   {
      template <typename T_result_t, typename c_layout_t, typename c_data_t>
      static inline MFEM_ALWAYS_INLINE
      void Eval(Func &F, const T_result_t &T, const c_layout_t &l, c_data_t &c)
      {
         const int qpts = T_result_t::x_type::layout_type::dim_1;
         const int ne   = T_result_t::x_type::layout_type::dim_3;
         const int vs   = sizeof(T.x[0])/sizeof(T.x[0][0]);
         for (int k = 0; k < ne; k++)
         {
            for (int i = 0; i < qpts; i++)
            {
               for (int s = 0; s < vs; s++)
               {
                  c[l.ind(i,k)][s] =
                     F.Eval3D(T.x(i,0,k)[s], T.x(i,1,k)[s], T.x(i,2,k)[s]);
               }
            }
         }
      }
   };

public:
   /// Constructor for the case when Func has no data members.
   TFunctionCoefficient() : F() { }
   /// Constructor for the case when Func has data members.
   TFunctionCoefficient(Func &F_) : F(F_) { }
   // Default copy constructor, Func has to have copy constructor.

   template <typename T_result_t, typename c_layout_t, typename c_data_t>
   inline MFEM_ALWAYS_INLINE
   void Eval(const T_result_t &T, const c_layout_t &l, c_data_t &c)
   {
      const int qpts = T_result_t::x_type::layout_type::dim_1;
      const int sdim = T_result_t::x_type::layout_type::dim_2;
      const int ne   = T_result_t::x_type::layout_type::dim_3;
      MFEM_STATIC_ASSERT(c_layout_t::rank == 2 && c_layout_t::dim_1 == qpts &&
                         c_layout_t::dim_2 == ne, "invalid c_layout_t");

      Dim<sdim,true>::Eval(F, T, l, c);
   }
};


/// Piecewise constant coefficient class. The subdomains where the coefficient
/// is constant are given by the mesh attributes.
template <typename complex_t = real_t>
class TPiecewiseConstCoefficient : public TCoefficient
{
public:
   static const bool uses_attributes = true;
   typedef complex_t complex_type;

protected:
   Vector constants; // complex_t = double

public:
   /// Note: in the input array index i corresponds to mesh attribute i+1.
   TPiecewiseConstCoefficient(const Vector &constants)
      : constants(constants) { }
   // default copy constructor

   template <typename T_result_t, typename c_layout_t, typename c_data_t>
   inline MFEM_ALWAYS_INLINE
   void Eval(const T_result_t &T, const c_layout_t &l, c_data_t &c)
   {
      const int ne = T_result_t::ne;
      const int vs = sizeof(T.attrib[0])/sizeof(T.attrib[0][0]);
      MFEM_STATIC_ASSERT(vs == sizeof(c[0])/sizeof(c[0][0]), "");
      for (int i = 0; i < ne; i++)
      {
         typename c_data_t::data_type ci;
         for (int s = 0; s < vs; s++)
         {
            ci[s] = constants(T.attrib[i][s]-1);
         }
         TAssign<AssignOp::Set>(l.ind2(i), c, ci);
      }
   }
};

/// GridFunction coefficient class.
template <typename FieldEval>
class TGridFunctionCoefficient : public TCoefficient
{
public:
   static const bool uses_element_idxs = true;

   typedef typename FieldEval::FESpace_type   FESpace_type;
   typedef typename FieldEval::ShapeEval_type ShapeEval_type;
   typedef typename FieldEval::VecLayout_type VecLayout_type;
   typedef typename FieldEval::complex_type   complex_type;

protected:
   FieldEval fieldEval;

public:
   // This constructor uses a shallow copy of fE.fespace as part of fieldEval.
   inline MFEM_ALWAYS_INLINE
   TGridFunctionCoefficient(const FieldEval &fE,
                            const complex_type *data)
      : fieldEval(fE, data, NULL)
   { }

   // This constructor uses a shallow copy of tfes as part of fieldEval.
   inline MFEM_ALWAYS_INLINE
   TGridFunctionCoefficient(const FESpace_type &tfes,
                            const ShapeEval_type &shapeEval,
                            const VecLayout_type &vec_layout,
                            const complex_type *data)
      : fieldEval(tfes, shapeEval, vec_layout, data, NULL)
   { }

   // This constructor creates new FESpace_type as part of fieldEval.
   inline MFEM_ALWAYS_INLINE
   TGridFunctionCoefficient(const FiniteElementSpace &fes,
                            const complex_type *data)
      : fieldEval(fes, data, NULL)
   { }

   // This constructor creates new FESpace_type as part of fieldEval.
   inline MFEM_ALWAYS_INLINE
   TGridFunctionCoefficient(const GridFunction &func)
      : fieldEval(*func.FESpace(), func.GetData(), NULL)
   { }

   // default copy constructor

   template <typename T_result_t, typename c_layout_t, typename c_data_t>
   inline MFEM_ALWAYS_INLINE
   void Eval(const T_result_t &T, const c_layout_t &l, c_data_t &c)
   {
      const int ne = T_result_t::ne;
      const int vdim = FieldEval::vdim;
      const int qpts = FieldEval::qpts;
      MFEM_STATIC_ASSERT(c_layout_t::rank  == 2,    "tensor rank must be 2");
      MFEM_STATIC_ASSERT(c_layout_t::dim_1 == qpts, "incompatible quadrature");
      MFEM_STATIC_ASSERT(c_layout_t::dim_2 == ne,    "");
      MFEM_STATIC_ASSERT(vdim == 1, "vdim != 1 is not supported");

      fieldEval.GetValues(T.first_elem_idx, l.template split_2<1,ne>(), c);
   }
};


/// Auxiliary class that is used to simplify the evaluation of a coefficient and
/// scaling it by the weights of a quadrature rule.
template <typename IR, typename coeff_t, typename impl_traits_t>
struct IntRuleCoefficient
{
   static const int qpts = IR::qpts;
   static const int ne   = impl_traits_t::batch_size;
   typedef typename coeff_t::complex_type complex_type;
   typedef typename impl_traits_t::vcomplex_t vcomplex_t;

   template <bool is_const, bool dummy> struct Aux;

   // constant coefficient
   template <bool dummy> struct Aux<true,dummy>
   {
      typedef struct { } result_t;
      TMatrix<qpts,1,complex_type> cw;

      inline MFEM_ALWAYS_INLINE Aux(const IR &int_rule, const coeff_t &c)
      {
         c.Eval(true, cw.layout, cw);
         int_rule.ApplyWeights(cw);
      }

      template <typename T_result_t>
      inline MFEM_ALWAYS_INLINE
      void Eval(const T_result_t &F, result_t &res) { }

      inline MFEM_ALWAYS_INLINE
      const complex_type &get(const result_t &res, int i, int k) const
      {
         return cw(i,0);
      }
   };
   // non-constant coefficient
   template <bool dummy> struct Aux<false,dummy>
   {
      typedef TMatrix<qpts,ne,vcomplex_t> result_t;
#ifdef MFEM_TEMPLATE_INTRULE_COEFF_PRECOMP
      TMatrix<qpts,1,typename IR::real_type> w;
#else
      IR int_rule;
#endif
      coeff_t c;

#ifdef MFEM_TEMPLATE_INTRULE_COEFF_PRECOMP
      inline MFEM_ALWAYS_INLINE Aux(const IR &int_rule, const coeff_t &c)
         : c(c)
      {
         int_rule.template AssignWeights<AssignOp::Set>(w.layout, w);
      }
#else
      inline MFEM_ALWAYS_INLINE Aux(const IR &int_rule, const coeff_t &c)
         : int_rule(int_rule), c(c) { }
#endif
      template <typename T_result_t>
      inline MFEM_ALWAYS_INLINE
      void Eval(const T_result_t &F, result_t &res)
      {
         c.Eval(F, res.layout, res);
#ifdef MFEM_TEMPLATE_INTRULE_COEFF_PRECOMP
         for (int i = 0; i < ne; i++)
         {
            TAssign<AssignOp::Mult>(res.layout.ind2(i), res,
                                    w.layout.merge_12(), w);
         }
#else
         int_rule.template AssignWeights<AssignOp::Mult>(res.layout, res);
#endif
      }

      inline MFEM_ALWAYS_INLINE
      const vcomplex_t &get(const result_t &res, int i, int k) const
      {
         return res(i,k);
      }
   };

   typedef Aux<coeff_t::is_const,true> Type;
};

} // namespace mfem

#endif // MFEM_TEMPLATE_COEFFICIENT
