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
#include "../linalg/ttensor.hpp"
#include "bilinearform.hpp"
#include "tevaluator.hpp"
#include "teltrans.hpp"
#include "tcoefficient.hpp"
#include "fespace.hpp"

namespace mfem
{

// Templated bilinear form class, cf. bilinearform.?pp

// complex_t - sol dof data type
// real_t - mesh nodes, sol basis, mesh basis data type
template <typename meshType, typename solFESpace,
          typename IR, typename IntegratorType,
          typename solVecLayout_t = ScalarLayout,
          typename complex_t = double, typename real_t = double>
class TBilinearForm : public Operator
{
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

   typedef IntegratorType integ_t;
   typedef typename integ_t::coefficient_type coeff_t;
   typedef typename integ_t::template kernel<sdim,dim,complex_t>::type kernel_t;
   typedef typename kernel_t::template p_asm_data<qpts>::type p_assembled_t;
   typedef typename kernel_t::template f_asm_data<qpts>::type f_assembled_t;

   typedef TElementTransformation<meshType,IR,real_t> Trans_t;
   template <int NE> struct T_result
   {
      static const int EvalOps =
         Trans_t::template Get<coeff_t,kernel_t>::EvalOps;
      typedef typename Trans_t::template Result<EvalOps,NE> Type;
   };

   typedef FieldEvaluator<solFESpace,solVecLayout_t,IR,
           complex_t,real_t> solFieldEval;
   template <int BE> struct S_spec
   {
      typedef typename solFieldEval::template Spec<kernel_t,BE> Spec;
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

   p_assembled_t *assembled_data;

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
        assembled_data(NULL),
        in_fes(sol_fes)
   { }

   virtual ~TBilinearForm()
   {
      delete [] assembled_data;
   }

   /// Get the input finite element space prolongation matrix
   virtual const Operator *GetProlongation() const
   { return ((FiniteElementSpace &)in_fes).GetProlongationMatrix(); }
   /// Get the input finite element space restriction matrix
   virtual const Operator *GetRestriction() const
   { return ((FiniteElementSpace &)in_fes).GetRestrictionMatrix(); }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      if (assembled_data)
      {
         const int num_elem = 1;
         MultAssembled<num_elem>(x, y);
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

      const int BE = 1; // batch-size of elements
      typedef typename kernel_t::template
      CoefficientEval<IR,coeff_t,BE>::Type coeff_eval_t;

      // For better performance, create stack copies of solFES, and solEval
      // inside 'solFEval'. The element-transformation 'T' also copies the
      // meshFES, meshEval, etc internally.
      // Is performance actually better with this implementation?
      Trans_t T(mesh, meshEval);
      solFieldEval solFEval(solFES, solEval, solVecLayout,
                            x.GetData(), y.GetData());
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el++)
      {
#if 0
         typename S_spec<BE>::DataType R;
         solFEval.Eval(el, R);

         typename T_result<BE>::Type F;
         T.Eval(el, F);
#else
         typename T_result<BE>::Type F;
         T.Eval(el, F);

         typename S_spec<BE>::DataType R;
         solFEval.Eval(el, R);
#endif

         typename coeff_eval_t::result_t res;
         wQ.Eval(F, res);

         kernel_t::Action(0, F, wQ, res, R);

         solFEval.template Assemble<true>(R);
      }
   }

   // Partial assembly of quadrature point data
   void Assemble()
   {
      const int BE = 1; // batch-size of elements
      typedef typename kernel_t::template
      CoefficientEval<IR,coeff_t,BE>::Type coeff_eval_t;

      Trans_t T(mesh, meshEval);
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      if (!assembled_data)
      {
         assembled_data = new p_assembled_t[NE];
      }
      for (int el = 0; el < NE; el++) // BE == 1
      {
         typename T_result<BE>::Type F;
         T.Eval(el, F);

         typename coeff_eval_t::result_t res;
         wQ.Eval(F, res);

         for (int k = 0; k < BE; k++)
         {
            kernel_t::Assemble(k, F, wQ, res, assembled_data[el+k]);
         }
      }
   }

   template <int num_elem>
   inline MFEM_ALWAYS_INLINE
   void ElementAddMultAssembled(int el, solFieldEval &solFEval) const
   {
      typename S_spec<num_elem>::DataType R;
      solFEval.Eval(el, R);

      for (int k = 0; k < num_elem; k++)
      {
         kernel_t::MultAssembled(k, assembled_data[el+k], R);
      }

      solFEval.template Assemble<true>(R);
   }

   // complex_t = double
   template <int num_elem>
   void MultAssembled(const Vector &x, Vector &y) const
   {
      y = 0.0;

      solFieldEval solFEval(solFES, solEval, solVecLayout,
                            x.GetData(), y.GetData());

      const int NE = mesh.GetNE();
      const int bNE = NE-NE%num_elem;
      for (int el = 0; el < bNE; el += num_elem)
      {
         ElementAddMultAssembled<num_elem>(el, solFEval);
      }
      for (int el = bNE; el < NE; el++)
      {
         ElementAddMultAssembled<1>(el, solFEval);
      }
   }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
   // complex_t = double
   void TestElementwiseExtractAssemble(const Vector &x, Vector &y) const
   {
      y = 0.0;

      solVecLayout_type solVecLayout(this->solVecLayout);
      solFESpace solFES(this->solFES);

      TTensor3<dofs,vdim,1,complex_t> xy_dof;

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el++)
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
      typedef TTensor3<meshFE_type::dofs,sdim,1,real_t> lnodes_t;

      const int NE = mesh.GetNE();
      sNodes.SetSize(lnodes_t::size*NE);
      real_t *lNodes = sNodes.GetData();
      for (int el = 0; el < NE; el++)
      {
         meshFES.SetElement(el);
         meshFES.VectorExtract(mesh.node_layout, mesh.Nodes,
                               lnodes_t::layout, lNodes);
         lNodes += lnodes_t::size;
      }
   }

   // partial assembly from "serialized" nodes
   // real_t = double
   void AssembleFromSerializedNodes(const Vector &sNodes)
   {
      const int  BE = 1; // batch-size of elements
      typedef typename kernel_t::template
      CoefficientEval<IR,coeff_t,BE>::Type coeff_eval_t;

      Trans_t T(this->mesh, this->meshEval);
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      if (!assembled_data)
      {
         assembled_data = new p_assembled_t[NE];
      }
      for (int el = 0; el < NE; el++)
      {
         typename T_result<BE>::Type F;
         T.EvalSerialized(el, sNodes.GetData(), F);

         typename coeff_eval_t::result_t res;
         wQ.Eval(F, res);

         kernel_t::Assemble(0, F, wQ, res, assembled_data[el]);
      }
   }

   // complex_t = double
   void Serialize(const Vector &x, Vector &sx) const
   {
      solVecLayout_t solVecLayout(this->solVecLayout);
      typedef TTensor3<dofs,vdim,1,complex_t> vdof_data_t;
      solFESpace solFES(this->solFES);

      const int NE = mesh.GetNE();
      sx.SetSize(vdim*dofs*NE);
      complex_t *loc_sx = sx.GetData();
      for (int el = 0; el < NE; el++)
      {
         solFES.SetElement(el);
         solFES.VectorExtract(solVecLayout, x, vdof_data_t::layout, loc_sx);
         loc_sx += vdim*dofs;
      }
   }

   // serialized vector sx --> serialized vector 'sy'
   // complex_t = double
   void MultAssembledSerialized(const Vector &sx, Vector &sy) const
   {
      solFieldEval solFEval(solFES, solEval, solVecLayout, NULL, NULL);

      const int NE = mesh.GetNE();
      const complex_t *loc_sx = sx.GetData();
      complex_t *loc_sy = sy.GetData();
      for (int el = 0; el < NE; el++)
      {
         typename S_spec<1>::DataType R;
         solFEval.EvalSerialized(loc_sx, R);

         kernel_t::MultAssembled(0, assembled_data[el], R);

         solFEval.template AssembleSerialized<false>(R, loc_sy);

         loc_sx += vdim*dofs;
         loc_sy += vdim*dofs;
      }
   }
#endif // MFEM_TEMPLATE_ENABLE_SERIALIZE

   // Assemble the operator in a SparseMatrix.
   // complex_t = double
   void AssembleMatrix(SparseMatrix &M) const
   {
      const int BE = 1; // batch-size of elements
      typedef typename kernel_t::template
      CoefficientEval<IR,coeff_t,BE>::Type coeff_eval_t;

      Trans_t T(mesh, meshEval);
      solFESpace solFES(this->solFES);
      solShapeEval solEval(this->solEval);
      solVecLayout_t solVecLayout(this->solVecLayout);
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el++)
      {
         f_assembled_t asm_qpt_data;
         {
            typename T_result<BE>::Type F;
            T.Eval(el, F);

            typename coeff_eval_t::result_t res;
            wQ.Eval(F, res);

            kernel_t::Assemble(0, F, wQ, res, asm_qpt_data);
         }

         // For now, when vdim > 1, assume block-diagonal matrix with the same
         // diagonal block for all components.
         TMatrix<dofs,dofs> M_loc;
         S_spec<BE>::ElementMatrix::Compute(
            asm_qpt_data.layout, asm_qpt_data, M_loc.layout, M_loc, solEval);

         solFES.SetElement(el);
         for (int bi = 0; bi < vdim; bi++)
         {
            solFES.AssembleBlock(bi, bi, solVecLayout, M_loc, M);
         }
      }
   }

   // Assemble element matrices and store them as a DenseTensor object.
   // complex_t = double
   void AssembleMatrix(DenseTensor &M) const
   {
      const int BE = 1; // batch-size of elements
      typedef typename kernel_t::template
      CoefficientEval<IR,coeff_t,BE>::Type coeff_eval_t;

      Trans_t T(mesh, meshEval);
      solShapeEval solEval(this->solEval);
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el++)
      {
         f_assembled_t asm_qpt_data;
         {
            typename T_result<BE>::Type F;
            T.Eval(el, F);

            typename coeff_eval_t::result_t res;
            wQ.Eval(F, res);

            kernel_t::Assemble(0, F, wQ, res, asm_qpt_data);
         }

         // For now, when vdim > 1, assume block-diagonal matrix with the same
         // diagonal block for all components.
         // M is assumed to be (dof x dof x NE).
         TMatrix<dofs,dofs> M_loc;
         S_spec<BE>::ElementMatrix::Compute(
            asm_qpt_data.layout, asm_qpt_data, M_loc.layout, M_loc, solEval);

         complex_t *M_data = M.GetData(el);
         M_loc.template AssignTo<AssignOp::Set>(M_data);
      }
   }

   // Assemble element matrices and add them to the bilinear form
   // complex_t = double
   void AssembleBilinearForm(BilinearForm &a) const
   {
      const int BE = 1; // batch-size of elements
      typedef typename kernel_t::template
      CoefficientEval<IR,coeff_t,BE>::Type coeff_eval_t;

      Trans_t T(mesh, meshEval);
      solShapeEval solEval(this->solEval);
      coeff_eval_t wQ(int_rule, coeff);

      Array<int> vdofs;
      const Array<int> *dof_map = sol_fe.GetDofMap();
      const int *dof_map_ = dof_map->GetData();
      DenseMatrix M_loc_perm(dofs*vdim,dofs*vdim); // initialized with zeros

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el++)
      {
         f_assembled_t asm_qpt_data;
         {
            typename T_result<BE>::Type F;
            T.Eval(el, F);

            typename coeff_eval_t::result_t res;
            wQ.Eval(F, res);

            kernel_t::Assemble(0, F, wQ, res, asm_qpt_data);
         }

         // For now, when vdim > 1, assume block-diagonal matrix with the same
         // diagonal block for all components.
         TMatrix<dofs,dofs> M_loc;
         S_spec<BE>::ElementMatrix::Compute(
            asm_qpt_data.layout, asm_qpt_data, M_loc.layout, M_loc, solEval);

         if (dof_map) // switch from tensor-product ordering
         {
            for (int i = 0; i < dofs; i++)
            {
               for (int j = 0; j < dofs; j++)
               {
                  M_loc_perm(dof_map_[i],dof_map_[j]) = M_loc(i,j);
               }
            }
            for (int bi = 1; bi < vdim; bi++)
            {
               M_loc_perm.CopyMN(M_loc_perm, dofs, dofs, 0, 0,
                                 bi*dofs, bi*dofs);
            }
            a.AssembleElementMatrix(el, M_loc_perm, vdofs);
         }
         else
         {
            DenseMatrix DM(M_loc.data, dofs, dofs);
            if (vdim == 1)
            {
               a.AssembleElementMatrix(el, DM, vdofs);
            }
            else
            {
               for (int bi = 0; bi < vdim; bi++)
               {
                  M_loc_perm.CopyMN(DM, dofs, dofs, 0, 0, bi*dofs, bi*dofs);
               }
               a.AssembleElementMatrix(el, M_loc_perm, vdofs);
            }
         }
      }
   }

   // Multiplication using assembled element matrices stored as a DenseTensor.
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
         TTensor3<dofs,vdim,1,complex_t> x_dof, y_dof;

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
