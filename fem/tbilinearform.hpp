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

#ifndef MFEM_TEMPLATE_BILINEAR_FORM
#define MFEM_TEMPLATE_BILINEAR_FORM

#include "../config/tconfig.hpp"
#include "../linalg/simd.hpp"
#include "../linalg/ttensor.hpp"
#include "bilinearform.hpp"
#include "tevaluator.hpp"
#include "teltrans.hpp"
#include "tcoefficient.hpp"
#include "fespace.hpp"

namespace mfem
{

/** @brief Templated bilinear form class, cf. bilinearform.?pp

// complex_t - sol dof data type
    @tparam meshType typically TMesh, which is templated on FE type
// real_t - mesh nodes, sol basis, mesh basis data type
    @tparam solFESpace eg. H1_FiniteElementSpace
    @tparam IR integration rule, typically TIntegrationRule, which is further
               templated on element geometry
    @tparam IntegratorType typically a TIntegrator, which is templated on a
                           kernel, eg. TDiffusionKernel or TMassKernel. This
                           describes what actual problem you solve.
    @tparam solVecLayout_t describes how degrees of freedom are laid out,
                           scalar or vector, column/row major, etc.
    @tparam complex_t data type for solution dofs
    @tparam real_t data type for mesh nodes, solution basis, and mesh basis
*/
template <typename meshType, typename solFESpace,
          typename IR, typename IntegratorType,
          typename solVecLayout_t = ScalarLayout,
          typename complex_t = double, typename real_t = double,
          typename impl_traits_t = AutoSIMDTraits<complex_t,real_t> >
class TBilinearForm : public Operator
{
public:
   typedef impl_traits_t impl_traits_type;

protected:
   typedef complex_t complex_type;
   typedef real_t    real_type;

   typedef typename meshType::FE_type            meshFE_type;
   typedef ShapeEvaluator<meshFE_type,IR,real_t> meshShapeEval;
   typedef typename solFESpace::FE_type          solFE_type;
   typedef ShapeEvaluator<solFE_type,IR,real_t>  solShapeEval;
   typedef solVecLayout_t                        solVecLayout_type;

   static const int dim  = meshType::dim;
   static const int sdim = meshType::space_dim;
   static const int dofs = solFE_type::dofs;
   static const int vdim = solVecLayout_t::vec_dim;
   static const int qpts = IR::qpts;
   static const int AB   = impl_traits_t::align_bytes;
   static const int SS   = impl_traits_t::simd_size;
   static const int BE   = impl_traits_t::batch_size;
   static const int TE   = SS*BE;

   typedef typename impl_traits_t::vcomplex_t vcomplex_t;
   typedef typename impl_traits_t::vreal_t    vreal_t;

   /// @name IntegratorType defines several internal types
   ///@{
   typedef IntegratorType integ_t;
   /// coeff_t might be TConstantCoefficient or TFunctionCoefficient, for example
   typedef typename integ_t::coefficient_type coeff_t;
   /// kernel_t may be TDiffusionKernel or TMassKernel
   typedef typename integ_t::template kernel<sdim,dim,vcomplex_t>::type kernel_t;
   /// p_assembled_t is something like a TTensor or TMatrix for partial assembly
   typedef typename kernel_t::template p_asm_data<qpts>::type p_assembled_t;
   /// f_assembled_t is something like a TTensor or TMatrix for full assembly
   typedef typename kernel_t::template f_asm_data<qpts>::type f_assembled_t;
   ///@}

   typedef typename kernel_t::template
   CoefficientEval<IR,coeff_t,impl_traits_t>::Type coeff_eval_t;


   typedef TElementTransformation<meshType,IR,real_t> Trans_t;
   struct T_result
   {
      static const int EvalOps =
         Trans_t::template Get<coeff_t,kernel_t>::EvalOps;
      typedef typename Trans_t::template Result<EvalOps,impl_traits_t> Type;
   };

   typedef FieldEvaluator<solFESpace,solVecLayout_t,IR,
           complex_t,real_t> solFieldEval;

   /** @brief Contains matrix sizes, type of kernel (ElementMatrix is templated
       on a kernel, e.g. ElementMatrix::Compute may be AssembleGradGrad()). */
   struct S_spec
   {
      typedef typename solFieldEval::template Spec<kernel_t,impl_traits_t> Spec;
      typedef typename Spec::DataType DataType;
      typedef typename Spec::ElementMatrix ElementMatrix;
   };

   // Data members

   meshType      mesh;
   meshShapeEval meshEval;

   solFE_type         sol_fe;
   solShapeEval       solEval;
   mutable solFESpace solFES;
   solVecLayout_t     solVecLayout;

   IR int_rule;

   coeff_t coeff;

   Memory<p_assembled_t> assembled_data;

   const FiniteElementSpace &in_fes;

public:
   TBilinearForm(const IntegratorType &integ, const FiniteElementSpace &sol_fes)
      : Operator(sol_fes.GetNDofs()*vdim),
        mesh(*sol_fes.GetMesh()),
        meshEval(mesh.fe),
        sol_fe(*sol_fes.FEColl()),
        solEval(sol_fe),
        solFES(sol_fe, sol_fes),
        solVecLayout(sol_fes),
        int_rule(),
        coeff(integ.coeff),
        assembled_data(),
        in_fes(sol_fes)
   {
      assembled_data.Reset(AB == 64 ? MemoryType::HOST_64 :
                           AB == 32 ? MemoryType::HOST_32 :
                           MemoryType::HOST);
   }

   virtual ~TBilinearForm()
   {
      assembled_data.Delete();
   }

   /// Get the input finite element space prolongation matrix
   virtual const Operator *GetProlongation() const
   { return ((FiniteElementSpace &)in_fes).GetProlongationMatrix(); }
   /// Get the input finite element space restriction matrix
   virtual const Operator *GetRestriction() const
   { return ((FiniteElementSpace &)in_fes).GetRestrictionMatrix(); }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      if (!assembled_data.Empty())
      {
         MultAssembled(x, y);
      }
      else
      {
         MultUnassembled(x, y);
      }
   }

   // complex_t = double
   void MultUnassembled(const Vector &x, Vector &y) const
   {
      y = 0.0;

      // For better performance, create stack copies of solFES, and solEval
      // inside 'solFEval'. The element-transformation 'T' also copies the
      // meshFES, meshEval, etc internally.
      // Is performance actually better with this implementation?
      Trans_t T(mesh, meshEval);
      solFieldEval solFEval(solFES, solEval, solVecLayout,
                            x.GetData(), y.GetData());
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el += TE)
      {
#if 0
         typename S_spec::DataType R;
         solFEval.Eval(el, R);

         typename T_result::Type F;
         T.Eval(el, F);
#else
         typename T_result::Type F;
         T.Eval(el, F);

         typename S_spec::DataType R;
         solFEval.Eval(el, R);
#endif

         typename coeff_eval_t::result_t res;
         wQ.Eval(F, res);

         for (int k = 0; k < BE; k++)
         {
            kernel_t::Action(k, F, wQ, res, R);
         }

         solFEval.template Assemble<true>(R);
      }
   }

   /// Partial assembly of quadrature point data
   void Assemble()
   {
      Trans_t T(mesh, meshEval);
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      if (assembled_data.Empty())
      {
         const int size = ((NE+TE-1)/TE)*BE;
         assembled_data.New(size, assembled_data.GetMemoryType());
      }
      for (int el = 0; el < NE; el += TE)
      {
         typename T_result::Type F;
         T.Eval(el, F);

         typename coeff_eval_t::result_t res;
         wQ.Eval(F, res);

         for (int k = 0; k < BE; k++)
         {
            kernel_t::Assemble(k, F, wQ, res, assembled_data[el/SS+k]);
         }
      }
   }

   inline MFEM_ALWAYS_INLINE
   void ElementAddMultAssembled(int el, solFieldEval &solFEval) const
   {
      typename S_spec::DataType R;
      solFEval.Eval(el, R);

      for (int k = 0; k < BE; k++)
      {
         kernel_t::MultAssembled(k, assembled_data[el/SS+k], R);
      }

      solFEval.template Assemble<true>(R);
   }

   // complex_t = double
   void MultAssembled(const Vector &x, Vector &y) const
   {
      y = 0.0;

      solFieldEval solFEval(solFES, solEval, solVecLayout,
                            x.GetData(), y.GetData());

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el += TE)
      {
         ElementAddMultAssembled(el, solFEval);
      }
   }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
   // complex_t = double
   void TestElementwiseExtractAssemble(const Vector &x, Vector &y) const
   {
      y = 0.0;

      solVecLayout_type solVecLayout(this->solVecLayout);
      solFESpace solFES(this->solFES);

      TTensor3<dofs,vdim,BE,vcomplex_t> xy_dof;

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el += TE)
      {
         solFES.SetElement(el);

         solFES.VectorExtract(solVecLayout, x, xy_dof.layout, xy_dof);
         solFES.VectorAssemble(xy_dof.layout, xy_dof, solVecLayout, y);
      }
   }

   // real_t = double
   void SerializeNodes(Vector &sNodes) const
   {
      typedef typename meshType::FESpace_type meshFESpace;
      meshFESpace meshFES(mesh.t_fes);
      typedef TTensor3<meshFE_type::dofs,sdim,BE,vreal_t> lnodes_t;

      const int NE = mesh.GetNE();
      // TODO: How do we make sure that this array is aligned properly, AND the
      //       compiler knows that it is aligned? => ALIGN_32|ALIGN_64 when ready
      const int NVE = (NE+TE-1)/TE;
      vreal_t *vsNodes = new vreal_t[lnodes_t::size*NVE];
      sNodes.NewDataAndSize(vsNodes[0].vec, (lnodes_t::size*SS)*NVE);
      sNodes.MakeDataOwner();
      for (int el = 0; el < NE; el += TE)
      {
         meshFES.SetElement(el);
         meshFES.VectorExtract(mesh.node_layout, mesh.Nodes,
                               lnodes_t::layout, vsNodes);
         vsNodes += lnodes_t::size;
      }
   }

   /// Partial assembly from "serialized" nodes
   // real_t = double
   void AssembleFromSerializedNodes(const Vector &sNodes)
   {
      Trans_t T(mesh, meshEval);
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      if (assembled_data.Empty())
      {
         const int size = ((NE+TE-1)/TE)*BE;
         assembled_data.New(size, assembled_data.GetMemoryType());
      }
      const vreal_t *vsNodes = (const vreal_t*)(sNodes.GetData());
      for (int el = 0; el < NE; el += TE)
      {
         typename T_result::Type F;
         T.EvalSerialized(el, vsNodes, F);

         typename coeff_eval_t::result_t res;
         wQ.Eval(F, res);

         for (int k = 0; k < BE; k++)
         {
            kernel_t::Assemble(k, F, wQ, res, assembled_data[el/SS+k]);
         }
      }
   }

   // complex_t = double
   void Serialize(const Vector &x, Vector &sx) const
   {
      typedef TTensor3<dofs,vdim,BE,vcomplex_t> vdof_data_t;

      solVecLayout_t solVecLayout(this->solVecLayout);
      solFESpace solFES(this->solFES);

      const int NE = mesh.GetNE();
      // TODO: How do we make sure that this array is aligned properly, AND
      //       the compiler knows that it is aligned? => ALIGN_32|ALIGN_64 when ready
      const int NVE = (NE+TE-1)/TE;
      vreal_t *vsx = new vreal_t[vdof_data_t::size*NVE];
      sx.NewDataAndSize(vsx[0].vec, (vdof_data_t::size*SS)*NVE);
      sx.MakeDataOwner();
      for (int el = 0; el < NE; el += TE)
      {
         solFES.SetElement(el);
         solFES.VectorExtract(solVecLayout, x, vdof_data_t::layout, vsx);
         vsx += vdof_data_t::size;
      }
   }

   /// serialized vector sx --> serialized vector 'sy'
   // complex_t = double
   void MultAssembledSerialized(const Vector &sx, Vector &sy) const
   {
      solFieldEval solFEval(solFES, solEval, solVecLayout, NULL, NULL);

      const int NE = mesh.GetNE();
      const vreal_t *vsx = (const vreal_t*)(sx.GetData());
      vreal_t *vsy = (vreal_t*)(sy.GetData());

      for (int el = 0; el < NE; el += TE)
      {
         typename S_spec::DataType R;
         solFEval.EvalSerialized(vsx, R);

         for (int k = 0; k < BE; k++)
         {
            kernel_t::MultAssembled(k, assembled_data[el/SS+k], R);
         }

         solFEval.template AssembleSerialized<false>(R, vsy);

         vsx += vdim*dofs*BE;
         vsy += vdim*dofs*BE;
      }
   }
#endif // MFEM_TEMPLATE_ENABLE_SERIALIZE

   /// Assemble the operator in a SparseMatrix.
   // complex_t = double
   void AssembleMatrix(SparseMatrix &M) const
   {
      Trans_t T(mesh, meshEval);
      solFESpace solFES(this->solFES);
      solShapeEval solEval(this->solEval);
      solVecLayout_t solVecLayout(this->solVecLayout);
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el += TE)
      {
         f_assembled_t asm_qpt_data[BE];
         {
            typename T_result::Type F;
            T.Eval(el, F);

            typename coeff_eval_t::result_t res;
            wQ.Eval(F, res);

            for (int k = 0; k < BE; k++)
            {
               kernel_t::Assemble(k, F, wQ, res, asm_qpt_data[k]);
            }
         }

         // For now, when vdim > 1, assume block-diagonal matrix with the same
         // diagonal block for all components.
         for (int k = 0; k < BE; k++)
         {
            const int el_k = el+SS*k;
            if (el_k >= NE) { break; }

            TMatrix<dofs,dofs,vcomplex_t> M_loc;
            S_spec::ElementMatrix::Compute(
               asm_qpt_data[k].layout, asm_qpt_data[k], M_loc.layout, M_loc,
               solEval);

            solFES.SetElement(el_k);
            for (int bi = 0; bi < vdim; bi++)
            {
               solFES.AssembleBlock(bi, bi, solVecLayout, M_loc, M);
            }
         }
      }
   }

   /// Assemble element matrices and store them as a DenseTensor object.
   // complex_t = double
   void AssembleMatrix(DenseTensor &M) const
   {
      Trans_t T(mesh, meshEval);
      solShapeEval solEval(this->solEval);
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el += TE)
      {
         f_assembled_t asm_qpt_data[BE];
         {
            typename T_result::Type F;
            T.Eval(el, F);

            typename coeff_eval_t::result_t res;
            wQ.Eval(F, res);

            for (int k = 0; k < BE; k++)
            {
               kernel_t::Assemble(k, F, wQ, res, asm_qpt_data[k]);
            }
         }

         // For now, when vdim > 1, assume block-diagonal matrix with the same
         // diagonal block for all components.
         // M is assumed to be (dof x dof x NE).
         for (int k = 0; k < BE; k++)
         {
            const int el_k = el+SS*k;
            if (el_k >= NE) { break; }

            TMatrix<dofs,dofs,vcomplex_t> M_loc;
            S_spec::ElementMatrix::Compute(
               asm_qpt_data[k].layout, asm_qpt_data[k], M_loc.layout, M_loc,
               solEval);

            for (int s = 0; s < SS && el_k+s < NE; s++)
            {
               complex_t *M_data = M.GetData(el_k+s);
               for (int j = 0; j < dofs; j++)
               {
                  for (int i = 0; i < dofs; i++)
                  {
                     M_data[j+dofs*i] = M_loc(i,j)[s];
                  }
               }
            }
         }
      }
   }

   /// Assemble element matrices and add them to the bilinear form
   // complex_t = double
   void AssembleBilinearForm(BilinearForm &a) const
   {
      Trans_t T(mesh, meshEval);
      solShapeEval solEval(this->solEval);
      coeff_eval_t wQ(int_rule, coeff);

      Array<int> vdofs;
      const Array<int> *dof_map = sol_fe.GetDofMap();
      const int *dof_map_ = dof_map->GetData();
      DenseMatrix M_loc_perm(dofs*vdim,dofs*vdim); // initialized with zeros

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el += TE)
      {
         f_assembled_t asm_qpt_data[BE];
         {
            typename T_result::Type F;
            T.Eval(el, F);

            typename coeff_eval_t::result_t res;
            wQ.Eval(F, res);

            for (int k = 0; k < BE; k++)
            {
               kernel_t::Assemble(k, F, wQ, res, asm_qpt_data[k]);
            }
         }

         // For now, when vdim > 1, assume block-diagonal matrix with the same
         // diagonal block for all components.
         for (int k = 0; k < BE; k++)
         {
            const int el_k = el+SS*k;
            if (el_k >= NE) { break; }

            TMatrix<dofs,dofs,vcomplex_t> M_loc;
            S_spec::ElementMatrix::Compute(
               asm_qpt_data[k].layout, asm_qpt_data[k], M_loc.layout, M_loc,
               solEval);

            if (dof_map) // switch from tensor-product ordering
            {
               for (int s = 0; s < SS && el_k+s < NE; s++)
               {
                  for (int i = 0; i < dofs; i++)
                  {
                     for (int j = 0; j < dofs; j++)
                     {
                        M_loc_perm(dof_map_[i],dof_map_[j]) = M_loc(i,j)[s];
                     }
                  }
                  for (int bi = 1; bi < vdim; bi++)
                  {
                     M_loc_perm.CopyMN(M_loc_perm, dofs, dofs, 0, 0,
                                       bi*dofs, bi*dofs);
                  }
                  a.AssembleElementMatrix(el_k+s, M_loc_perm, vdofs);
               }
            }
            else if (SS == 1)
            {
               DenseMatrix DM(M_loc.data[0].vec, dofs, dofs);
               if (vdim == 1)
               {
                  a.AssembleElementMatrix(el_k, DM, vdofs);
               }
               else
               {
                  for (int bi = 0; bi < vdim; bi++)
                  {
                     M_loc_perm.CopyMN(DM, dofs, dofs, 0, 0, bi*dofs, bi*dofs);
                  }
                  a.AssembleElementMatrix(el_k, M_loc_perm, vdofs);
               }
            }
            else
            {
               for (int s = 0; s < SS && el_k+s < NE; s++)
               {
                  for (int i = 0; i < dofs; i++)
                  {
                     for (int j = 0; j < dofs; j++)
                     {
                        M_loc_perm(i,j) = M_loc(i,j)[s];
                     }
                  }
                  for (int bi = 1; bi < vdim; bi++)
                  {
                     M_loc_perm.CopyMN(M_loc_perm, dofs, dofs, 0, 0,
                                       bi*dofs, bi*dofs);
                  }
                  a.AssembleElementMatrix(el_k+s, M_loc_perm, vdofs);
               }
            }
         }
      }
   }

   /// Multiplication using assembled element matrices stored as a DenseTensor.
   // complex_t = double
   void AddMult(DenseTensor &M, const Vector &x, Vector &y) const
   {
      // For now, when vdim > 1, assume block-diagonal matrix with the same
      // diagonal block for all components.
      // M is assumed to be (dof x dof x NE).
      solVecLayout_t solVecLayout(this->solVecLayout);
      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el++)
      {
         TTensor3<dofs,vdim,1,AutoSIMD<complex_t,1,1> > x_dof, y_dof;

         solFES.SetElement(el);
         solFES.VectorExtract(solVecLayout, x, x_dof.layout, x_dof);
         Mult_AB<false>(TMatrix<dofs,dofs>::layout,
                        M(el).Data(),
                        x_dof.layout.merge_23(), x_dof,
                        y_dof.layout.merge_23(), y_dof);
         solFES.VectorAssemble(y_dof.layout, y_dof, solVecLayout, y);
      }
   }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_BILINEAR_FORM
