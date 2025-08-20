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

#include "mfem.hpp"

namespace mfem::reconstruction
{

/// Enumerator for different function examples
enum ExamplesType
{
   plane,
   sine,
   exp_sine,
   num_examples  // last
};

/// Enumerator for reconstruction types
enum ReconstructionType
{
   norm,
   average,
   face_average,
   weak_face_average,
   bounded_variation,
   num_reconstructions  // last
};

/// Enumerator for solver types
enum SolverType
{
   direct_inv,
   cg,
   bicgstab,
   minres,
   num_solvers  // last
};

/// Enumerator for regularization types
enum RegularizationType
{
   direct,
   l2,
   h1,
   num_regularization  // last
};

/// Iterative solver parameters
struct IterativeSolverParams
{
   int print_level;
   int max_iter;
   real_t rtol;
   real_t atol;
};

/// Visualization parameters
struct VisualizationParams
{
   bool save_to_files;
   bool visualization;
   int port;
};

/// Global configuration struct
struct GlobalConfiguration
{
   ReconstructionType rec;
   VisualizationParams vis;
   IterativeSolverParams newton;
   SolverType solver_type;
   IterativeSolverParams solver;
   RegularizationType reg_type;
   real_t reg;
   ExamplesType example;
   int ord_smooth;
   int ord_src;
   int ord_dst;
   int ser_ref;
   int par_ref;
   bool preserve_volumes;
};

/// @name Specialized integrators
///@{

/** Class for an asymmetric (out-of-element) mass integrator $a_K(u,v) := (E(u),v)_K$,
 *  where $E$ is an extension of $u$ (with original domain $\hat{K}$) to $K$.
 *
 *  @note Currently we don't need to derive from MassIntegrator, as GetRule is public.
 *  This could be it's own standalone function.
 *  @todo Kernel implementation: requires kernel_dispatch.hpp. MFEM_THREAD_SAFE...
 */
class AsymmetricMassIntegrator : public MassIntegrator
{
private:
   Vector neighbor_shape, e_shape;
   Vector phys_neighbor_ip;
   DenseMatrix phys_neighbor_pts;
   IntegrationPoint ngh_ip;
   int neighbor_ndof, e_ndof;

protected:
   const mfem::FiniteElementSpace *tr_fes, *te_fes;

public:

   AsymmetricMassIntegrator() {};

   /// Assembles element mass matrix, extending @a neighbor_fe to @a e_fe.
   void AsymmetricElementMatrix(const FiniteElement &neighbor_fe,
                                const FiniteElement &e_fe,
                                ElementTransformation &neighbor_trans,
                                ElementTransformation &e_trans,
                                DenseMatrix &el_mat,
                                IterativeSolverParams &newton);

   /// Assembles element mass matrix, computing @a neighbor_fe in @a e_trans.
   void AsymmetricElementMatrix(const FiniteElement &neighbor_fe,
                                ElementTransformation &neighbor_trans,
                                ElementTransformation &e_trans,
                                DenseMatrix &el_mat,
                                IterativeSolverParams &newton);
};

/// @brief Out-of-element mass matrix.
/// Trial and test functions come from different elements.
void AsymmetricMassIntegrator::AsymmetricElementMatrix(const FiniteElement
                                                       &neighbor_fe,
                                                       const FiniteElement &e_fe,
                                                       ElementTransformation &neighbor_trans,
                                                       ElementTransformation &e_trans,
                                                       DenseMatrix &elmat,
                                                       IterativeSolverParams &newton)
{
   neighbor_ndof = neighbor_fe.GetDof();
   e_ndof = e_fe.GetDof();

   e_shape.SetSize(e_ndof);
   neighbor_shape.SetSize(neighbor_ndof);
   elmat.SetSize(e_ndof, neighbor_ndof);
   elmat = 0.0;

   /* Notes:
    * Let T_K : ref_K -> K from reference to physical
    * Let phi_i_K a shape function on the element K
    * and phi_i a shape function on the reference element ref_K
    * Let N(K) be an adjacent element to K
    *
    * Then phi_i_K(x) = phi_i(inv_T_K(x)), x in Omega.
    * Moreover,
    * int_N(K) phi_i_K(x) v(x) dx
    * = int_N(K) phi_i(inv_T_K(x)) v(x) dx
    * = int_ref_K phi_i(inv_T_K( T_N(K)(y) )) v( T_N(K)(y) ) |Jac T_N(K)| dy.
    */

   const int int_order = e_fe.GetOrder() + neighbor_fe.GetOrder() +
                         e_trans.OrderW();
   const IntegrationRule &ir = IntRules.Get(e_fe.GetGeomType(), int_order);

   neighbor_trans.Transform(ir, phys_neighbor_pts);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      phys_neighbor_pts.GetColumn(i, phys_neighbor_ip);

      // Pullback physical neighboring integration points to
      // "extended" reference element under e_trans
      using InvTr = mfem::InverseElementTransformation;
      InvTr inv_tr(&e_trans);
      inv_tr.SetPrintLevel(newton.print_level);
      inv_tr.SetMaxIter(newton.max_iter);
      inv_tr.SetPhysicalRelTol(newton.rtol);
      inv_tr.SetSolverType(InvTr::SolverType::Newton);
      inv_tr.SetInitialGuessType(InvTr::InitGuessType::ClosestPhysNode);
      MFEM_VERIFY(inv_tr.Transform(phys_neighbor_ip,
                                   ngh_ip) != InvTr::TransformResult::Unknown, "");

      neighbor_trans.SetIntPoint(&ngh_ip);
      e_trans.SetIntPoint(&ip);

      // Compute shape functions on e_fe
      neighbor_fe.CalcPhysShape(neighbor_trans, neighbor_shape);
      e_fe.CalcPhysShape(e_trans, e_shape);

      e_shape *= e_trans.Weight() * ip.weight;
      AddMultVWt(e_shape, neighbor_shape, elmat);
   }
}

/// @brief Out-of-element symmetric mass matrix (misnomer).
/// Shape functions come from neighboring element.
void AsymmetricMassIntegrator::AsymmetricElementMatrix(const FiniteElement
                                                       &neighbor_fe,
                                                       ElementTransformation &neighbor_trans,
                                                       ElementTransformation &e_trans,
                                                       DenseMatrix &elmat,
                                                       IterativeSolverParams &newton)
{
   neighbor_ndof = neighbor_fe.GetDof();

   neighbor_shape.SetSize(neighbor_ndof);
   elmat.SetSize(neighbor_ndof);
   elmat = 0.0;

   const int int_order = 2*neighbor_fe.GetOrder() + e_trans.OrderW();
   const IntegrationRule &ir = IntRules.Get(neighbor_fe.GetGeomType(), int_order);

   neighbor_trans.Transform(ir, phys_neighbor_pts);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      phys_neighbor_pts.GetColumn(i, phys_neighbor_ip);

      // Pullback physical neighboring integration points to
      // "extended" reference element under e_trans
      using InvTr = mfem::InverseElementTransformation;
      InverseElementTransformation inv_tr(&e_trans);
      inv_tr.SetPrintLevel(newton.print_level);
      inv_tr.SetMaxIter(newton.max_iter);
      inv_tr.SetPhysicalRelTol(newton.rtol);
      inv_tr.SetSolverType(InvTr::SolverType::Newton);
      inv_tr.SetInitialGuessType(InvTr::InitGuessType::ClosestPhysNode);
      MFEM_VERIFY(inv_tr.Transform(phys_neighbor_ip,
                                   ngh_ip) != InvTr::TransformResult::Unknown, "");

      neighbor_trans.SetIntPoint(&ngh_ip);
      e_trans.SetIntPoint(&ip);

      neighbor_fe.CalcPhysShape(neighbor_trans, neighbor_shape);
      AddMult_a_VVt(e_trans.Weight()*ip.weight, neighbor_shape, elmat);
   }
}


/** Integrator of the DPG form $\langle v, \lbrace w \rbrace \rangle$ where
 *  $\lbrace \cdot \rbrace$ represents the average of the traces of the argument.
 */
class TraceAverageIntegrator : public BilinearFormIntegrator
{
private:
   Vector shape_self, shape_other, face_shape;

public:
   TraceAverageIntegrator() { };
   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement& trial_face_fe,
                           const FiniteElement& test_fe_self,
                           const FiniteElement& test_fe_other,
                           FaceElementTransformations& Trans,
                           DenseMatrix& elmat);
};

void TraceAverageIntegrator::AssembleFaceMatrix(const FiniteElement&
                                                trial_face_fe,
                                                const FiniteElement& test_fe_self,
                                                const FiniteElement& test_fe_other,
                                                FaceElementTransformations& Trans,
                                                DenseMatrix& elmat)
{
   const bool has_other = (Trans.Elem2No>=0);
   const bool is_map_type_value = (trial_face_fe.GetMapType() ==
                                   FiniteElement::VALUE);
   const int trial_face_ndofs = trial_face_fe.GetDof();
   const int test_self_ndofs = test_fe_self.GetDof();
   const int test_other_ndofs = has_other?test_fe_self.GetDof():0;

   face_shape.SetSize(trial_face_ndofs);
   shape_self.SetSize(test_self_ndofs);
   shape_other.SetSize(test_other_ndofs);

   elmat.SetSize(test_self_ndofs + test_other_ndofs, trial_face_ndofs);
   elmat = 0.0;

   const IntegrationRule* ir = IntRule;
   if (!ir)
   {
      const int order = has_other?std::max(test_fe_self.GetOrder(),
                                           test_fe_other.GetOrder()):test_fe_self.GetOrder() +
                        trial_face_fe.GetOrder() + is_map_type_value*Trans.OrderW();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const auto& ip = ir->IntPoint(p);
      Trans.SetAllIntPoints(&ip);

      trial_face_fe.CalcShape(ip, face_shape);
      test_fe_self.CalcPhysShape(*Trans.Elem1, shape_self);
      if (has_other) { test_fe_other.CalcPhysShape(*Trans.Elem2, shape_other); }

      face_shape *= is_map_type_value?Trans.Weight()*ip.weight:ip.weight;

      for (int i = 0; i < test_self_ndofs; i++)
      {
         for (int j = 0; j < trial_face_ndofs; j++)
         {
            elmat(i, j) += shape_self(i) * face_shape(j);
         }
      }
      if (has_other)
      {
         for (int i = 0; i < test_other_ndofs; i++)
         {
            for (int j = 0; j < trial_face_ndofs; j++)
            {
               elmat(test_self_ndofs + i, j) += shape_other(i) * face_shape(j);
            }
         }
         elmat *= 0.5;
      }
   }
}

///@}

/// @name Auxiliary functions
///@{

/// @brief Dense small least squares solver
void LSSolver(Solver& solver,
              const DenseMatrix& A,
              const Vector& b,
              Vector& x,
              real_t shift = 0.0,
              DenseMatrix* M = nullptr)
{
   x.SetSize(A.Width());
   x = 0.0;

   Vector Atb(A.Width());
   A.MultTranspose(b, Atb);

   auto AtA_reg = OperatorPtr(Operator::Type::ANY_TYPE);
   auto _M = OperatorPtr(Operator::Type::ANY_TYPE);
   if (dynamic_cast<IterativeSolver*>(&solver))
   {
      if (M) { _M.Reset(M, false); }
      else { _M.Reset(new IdentityOperator(A.Width())); }

      auto _At = new TransposeOperator(A);
      auto _AtA = new ProductOperator(_At, &A, true, false);
      auto _AtA_reg = new SumOperator(_AtA, 1.0, _M.As<Operator>(), shift, true,
                                      false);
      AtA_reg.Reset(_AtA_reg, true);
      solver.SetOperator(*_AtA_reg);
   }
   else if (dynamic_cast<DenseMatrixInverse*>(&solver))
   {
      auto _AtA_reg = new DenseMatrix(A.Width());
      Vector col_i(A.Height()), col_j(A.Height());
      for (int i = 0; i < A.Width(); i++)
      {
         A.GetColumn(i, col_i);
         for (int j = 0; j < A.Width(); j++)
         {
            A.GetColumn(j, col_j);
            (*_AtA_reg)(i,j) = col_i * col_j + (M?(*M)(i,j):(i==j))*shift;
         }
      }
      AtA_reg.Reset(_AtA_reg, true);
      solver.SetOperator(*_AtA_reg);
   }
   else { mfem_error("Solver not implemented for this wrapper!"); }

   solver.Mult(Atb, x);
}

/// @brief Dense small least squares solver, with constrains @a C with value @a c
void LSSolver(Solver& solver,
              const DenseMatrix& A,
              const DenseMatrix& C,
              const Vector& b,
              const Vector& c,
              Vector& x,
              Vector& y,
              real_t shift = 0.0,
              DenseMatrix* M = nullptr)
{
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = A.Width();
   offsets[2] = A.Width() + C.Height();

   // Block vectors
   BlockVector rhs(offsets), z(offsets);
   rhs = 0.0;
   z = 0.0;

   Vector Atb(A.Width());
   A.MultTranspose(b, Atb);

   rhs.SetVector(Atb, offsets[0]);
   rhs.SetVector(c, offsets[1]);

   auto AtA_reg = OperatorPtr(Operator::Type::ANY_TYPE);
   auto _M = OperatorPtr(Operator::Type::ANY_TYPE);
   auto block_solver = OperatorPtr(Operator::Type::ANY_TYPE);

   if (auto it_solver = dynamic_cast<IterativeSolver*>(&solver))
   {
      if (M) { _M.Reset(M, false); }
      else { _M.Reset(new IdentityOperator(A.Width())); }

      auto _At = new TransposeOperator(A);
      auto _AtA = new ProductOperator(_At, &A, true, false);
      auto _AtA_reg = new SumOperator(_AtA, 1.0, _M.As<Operator>(), shift, true,
                                      false);
      AtA_reg.Reset(_AtA_reg, true);

      // Block operator
      if (it_solver->GetComm() != MPI_COMM_NULL)
      {
         block_solver.Reset(new SchurConstrainedSolver(it_solver->GetComm(), *_AtA_reg,
                                                       *const_cast<DenseMatrix*>(&C), solver), true);
      }
      else
      {
         block_solver.Reset(new SchurConstrainedSolver(*_AtA_reg,
                                                       *const_cast<DenseMatrix*>(&C), solver), true);
      }
   }
   else if (dynamic_cast<DenseMatrixInverse*>(&solver))
   {
      auto _AtA_reg = new DenseMatrix(A.Width());
      Vector col_i(A.Height()), col_j(A.Height());
      for (int i = 0; i < A.Width(); i++)
      {
         A.GetColumn(i, col_i);
         for (int j = 0; j < A.Width(); j++)
         {
            A.GetColumn(j, col_j);
            (*_AtA_reg)(i,j) = col_i * col_j + (M?(*M)(i,j):(i==j))*shift;
         }
      }
      AtA_reg.Reset(_AtA_reg, true);

      block_solver.Reset(new SchurConstrainedSolver(*_AtA_reg,
                                                    *const_cast<DenseMatrix*>(&C), solver), true);
   }
   else { mfem_error("Solver not implemented for this wrapper!"); }

   block_solver.As<SchurConstrainedSolver>()->LagrangeSystemMult(rhs, z);

   x.SetSize(A.Width());
   y.SetSize(C.Width());
   x = 0.0;
   y = 0.0;

   x += z.GetBlock(0);
   y += z.GetBlock(1);
}

/// @brief Dense small least squares solver, with constrains @a C with value @a c
/// but the constraint is one-dimensional, and passed as a vector. Does not return
/// the multiplier.
void LSSolver(Solver& solver,
              const DenseMatrix& A,
              const Vector& C,
              const Vector& b,
              const real_t& c,
              Vector& x,
              real_t shift = 0.0,
              DenseMatrix* M = nullptr)
{
   x.SetSize(A.Width());
   x = 0.0;

   Vector Atb(A.Width());
   A.MultTranspose(b, Atb);

   Vector aux_den(A.Width()), aux_num(A.Width());

   auto AtA_reg = OperatorPtr(Operator::Type::ANY_TYPE);
   auto _M = OperatorPtr(Operator::Type::ANY_TYPE);
   if (dynamic_cast<IterativeSolver*>(&solver))
   {
      if (M) { _M.Reset(M, false); }
      else { _M.Reset(new IdentityOperator(A.Width())); }

      auto _At = new TransposeOperator(A);
      auto _AtA = new ProductOperator(_At, &A, true, false);
      auto _AtA_reg = new SumOperator(_AtA, 1.0, _M.As<Operator>(), shift, true,
                                      false);
      AtA_reg.Reset(_AtA_reg, true);
      solver.SetOperator(*_AtA_reg);
   }
   else if (dynamic_cast<DenseMatrixInverse*>(&solver))
   {
      auto _AtA_reg = new DenseMatrix(A.Width());
      Vector col_i(A.Height()), col_j(A.Height());
      for (int i = 0; i < A.Width(); i++)
      {
         A.GetColumn(i, col_i);
         for (int j = 0; j < A.Width(); j++)
         {
            A.GetColumn(j, col_j);
            (*_AtA_reg)(i,j) = col_i * col_j + (M?(*M)(i,j):(i==j))*shift;
         }
      }
      AtA_reg.Reset(_AtA_reg, true);
      solver.SetOperator(*_AtA_reg);
   }
   else { mfem_error("Solver not implemented for this wrapper!"); }

   solver.Mult(C, aux_den);
   solver.Mult(Atb, aux_num);

   real_t mult_num = c - (C * aux_num);
   real_t mult_den = (C * aux_den);

   if (std::abs(mult_den) < 0.001) { mfem_warning("Denominator of lagrange multiplier close to zero!"); }
   if (std::abs(mult_num) < 0.001) { mfem_warning("Numerator of lagrange multiplier close to zero!"); }
   real_t mult = mult_num/mult_den;

   // Pertube RHS and solve
   add(Atb, mult, C, Atb);
   solver.Mult(Atb, x);
}

/// @brief Check both residuals: of the minimization and the least squares problem
void CheckLSSolver(const DenseMatrix& A, const Vector& b, const Vector& x)
{
   Vector res(b), sol(A.Width());
   A.AddMult(x,res,-1.0);
   A.MultTranspose(res, sol);
   mfem::out << "\nNorm of the residual of LS problem: " << res.Norml2()
             << "\nNorm of the solution of LS problem: " << sol.Norml2()
             << std::endl;
}

/// @brief Get the L1-Jacobi row-subs for the @b normal matrix associated with @a A
void GetNormalL1Diag(const DenseMatrix& A, Vector& l1_diag)
{
   l1_diag.SetSize(A.Height());
   Vector col_j(A.Height()), col_k(A.Height());
   real_t sums;
   for (int i = 0; i < A.Height(); i++)
   {
      sums = 0.0;
      for (int j = 0; j < A.Width(); j++)
      {
         A.GetColumn(j, col_j);
         for (int k = 0; k < A.Width(); k++) { A.GetColumn(k, col_k); }
         sums += std::abs(col_j * col_k);
      }
      l1_diag(i) = sums;
   }
}

/// @brief Saturates a neighborhood with hopes to well-pose the least squares problem.
/// @todo Make friends with @a MCMesh
void SaturateNeighborhood(NCMesh& mesh, const int element_idx,
                          const int target_ndofs, const int contributed_ndofs,
                          Array<int>& neighbors)
{
   Array<int> temp;
   mesh.FindNeighbors(element_idx, neighbors);
   neighbors.Append(element_idx);
   MFEM_VERIFY(mesh.GetNumElements() * contributed_ndofs >= target_ndofs,
               "Mesh too small!");
   while (neighbors.Size() * contributed_ndofs < target_ndofs)
   {
      mesh.NeighborExpand(neighbors, temp);
      neighbors = temp;
   }
   neighbors.Unique();
}

/// @brief Get face averages at quadrature points, associated to
/// the face associated to @a face_trans
void ComputeFaceAverage(const FiniteElementSpace& fes,
                        FaceElementTransformations& face_trans,
                        const IntegrationRule& ir,
                        const GridFunction& global_function,
                        Vector& face_values)
{
   const bool has_other = (face_trans.Elem2No >= 0);
   const FiniteElement* fe_self = fes.GetFE(face_trans.Elem1No);
   const FiniteElement* fe_other = has_other?fes.GetFE(face_trans.Elem2No):
                                   fe_self;

   const int ndof_self = fe_self->GetDof();
   const int ndof_other = has_other?fe_other->GetDof():0;

   Array<int> dofs_self, dofs_other;
   fes.GetElementDofs(face_trans.Elem1No, dofs_self);
   if (has_other) { fes.GetElementDofs(face_trans.Elem2No, dofs_other); }

   Vector shape_self(ndof_self), shape_other(ndof_other);
   Vector dofs_at_self, dofs_at_other;
   global_function.GetSubVector(dofs_self, dofs_at_self);
   if (has_other) { global_function.GetSubVector(dofs_other, dofs_at_other); }

   face_values.SetSize(ir.GetNPoints());
   face_values = 0.0;

   for (int p = 0; p < ir.GetNPoints(); p++)
   {
      const IntegrationPoint& ip = ir.IntPoint(p);
      face_trans.SetAllIntPoints(&ip);

      fe_self->CalcPhysShape(*face_trans.Elem1, shape_self);
      if (has_other) { fe_other->CalcPhysShape(*face_trans.Elem2, shape_other); }

      face_values(p) = shape_self*dofs_at_self;
      if (has_other) { face_values(p) += shape_other*dofs_at_other; }
   }
   if (has_other) { face_values *= 0.5; }
}

/// @brief Get matrix associated to traces of shape functions on
/// the face associated to @a face_trans
void ComputeFaceMatrix(int e_idx,
                       const FiniteElementSpace& fes,
                       FaceElementTransformations& face_trans,
                       const IntegrationRule& ir,
                       DenseMatrix& e_shape_to_q_face)
{
   MFEM_VERIFY(face_trans.Elem1No == e_idx || face_trans.Elem2No == e_idx,
               "Selected element does not conform face in face_trans!");

   const auto& fe_self = *fes.GetFE(e_idx);
   const int ndof_self = fe_self.GetDof();
   auto& e_trans = (face_trans.Elem1No == e_idx)?*face_trans.Elem1:
                   *face_trans.Elem2;

   Vector shape_self(ndof_self);
   e_shape_to_q_face.SetSize(ir.GetNPoints(), ndof_self);
   for (int p = 0; p < ir.GetNPoints(); p++)
   {
      const IntegrationPoint& ip = ir.IntPoint(p);
      face_trans.SetAllIntPoints(&ip);
      fe_self.CalcPhysShape(e_trans, shape_self);
      e_shape_to_q_face.SetRow(p, shape_self);
   }
}

/// @brief Boilerplate code for getting the orders a priori
const IntegrationRule& GetCommonIntegrationRule(const FiniteElementSpace&
                                                fes_src,
                                                const FiniteElementSpace& fes_dst,
                                                const FaceElementTransformations& face_trans)
{
   const bool has_other = (face_trans.Elem2No >= 0);
   const FiniteElement* fe_src_self = fes_src.GetFE(face_trans.Elem1No);
   const FiniteElement* fe_src_other = has_other?fes_src.GetFE(face_trans.Elem2No):
                                       fe_src_self;
   const FiniteElement* fe_dst_self = fes_dst.GetFE(face_trans.Elem1No);
   const FiniteElement* fe_dst_other = has_other?fes_dst.GetFE(face_trans.Elem2No):
                                       fe_dst_self;

   const int order = face_trans.OrderW() +
                     std::max(fe_src_self->GetOrder(), fe_dst_self->GetOrder()) +
                     std::max(fe_src_other->GetOrder(), fe_dst_other->GetOrder());
   return IntRules.Get(face_trans.GetGeometryType(), order);
}

///@}

/// @name Reconstruction functions
///@{

/// @brief Local-averages-based reconstruction
void AverageReconstruction(Solver& solver,
                           ParGridFunction& src,
                           ParGridFunction& dst,
                           ParMesh& mesh,
                           IterativeSolverParams& newton,
                           RegularizationType reg_type = direct,
                           real_t reg = 0.0,
                           int print_level = -1,
                           bool preserve_volumes = false)
{
   AsymmetricMassIntegrator mass;
   const auto fes_src = src.ParFESpace();
   const auto fes_dst = dst.ParFESpace();
   // GetElementTransformation requires non-null ptr
   auto neighbor_trans = std::make_unique<IsoparametricTransformation>();
   auto e_trans = std::make_unique<IsoparametricTransformation>();

   Array<int> neighbors_e, e_dst_dofs;

   Vector src_e, dst_e;
   Vector src_neighbor, dst_neighbor;
   Vector punity_src_e, punity_dst_e;
   Vector punity_src_neighbor, punity_dst_neighbor;

   DenseMatrix e_to_e_mat, neighbor_to_neighbor_mat;
   DenseMatrix e_to_neighbor_mat;

   // Auxiliary constant coefficients and partition of unity
   ConstantCoefficient ccf_ones(1.0);
   ParGridFunction punity_src(fes_src), punity_dst(fes_dst);
   punity_src.ProjectCoefficient(ccf_ones);
   punity_dst.ProjectCoefficient(ccf_ones);

   for (int e_idx = 0; e_idx < mesh.GetNE(); e_idx++ )
   {
      auto& fe_src_e = *fes_src->GetFE(e_idx);
      auto& fe_dst_e = *fes_dst->GetFE(e_idx);
      const int fe_src_e_ndof = fe_src_e.GetDof();
      const int fe_dst_e_ndof = fe_dst_e.GetDof();

      fes_dst->GetElementDofs(e_idx, e_dst_dofs);

      fes_src->GetElementTransformation(e_idx, e_trans.get());

      SaturateNeighborhood(*mesh.pncmesh, e_idx, fe_dst_e_ndof, fe_src_e_ndof,
                           neighbors_e);
      if (preserve_volumes) { neighbors_e.DeleteFirst(e_idx); }
      const int num_neighbors = neighbors_e.Size();

      DenseMatrix fe_dst_to_neighbors_mat(num_neighbors, fe_dst_e_ndof);
      Vector neighbors_volume(num_neighbors);
      Vector src_neighbors_avg(num_neighbors);
      Vector e_to_e_avg(fe_dst_e_ndof);
      for (int i = 0; i < num_neighbors; i++)
      {
         const int neighbor_idx = neighbors_e[i];
         auto& fe_src_neighbor = *fes_src->GetFE(neighbor_idx);
         auto& fe_dst_neighbor = *fes_dst->GetFE(neighbor_idx);
         const int fe_dst_neighbor_ndof = fe_dst_neighbor.GetDof();

         fes_dst->GetElementTransformation(neighbor_idx, neighbor_trans.get());

         Vector e_to_neighbor_avg(fe_dst_neighbor_ndof);

         neighbors_volume(i) = mesh.GetElementVolume(neighbor_idx);

         // Get subvectors
         src.GetElementDofValues(neighbor_idx, src_neighbor);
         punity_src.GetElementDofValues(neighbor_idx, punity_src_neighbor);
         punity_dst.GetElementDofValues(neighbor_idx, punity_dst_neighbor);

         // neighbor-DOFs to neighbor-DOFs (on neighbor) mass matrix
         mass.AssembleElementMatrix2(fe_src_neighbor,
                                     fe_dst_neighbor,
                                     *neighbor_trans.get(),
                                     neighbor_to_neighbor_mat);
         src_neighbors_avg(i) = neighbor_to_neighbor_mat.InnerProduct(src_neighbor,
                                                                      punity_dst_neighbor);

         // e-DOFs to neighbor-DOFs (on neighbor) mass matrix
         mass.AsymmetricElementMatrix(fe_dst_neighbor,
                                      fe_src_e,
                                      *neighbor_trans.get(),
                                      *e_trans.get(),
                                      e_to_neighbor_mat,
                                      newton);

         e_to_neighbor_mat.MultTranspose(punity_src_neighbor, e_to_neighbor_avg);
         fe_dst_to_neighbors_mat.SetRow(i, e_to_neighbor_avg);
      }

      fe_dst_to_neighbors_mat.InvLeftScaling(neighbors_volume);
      src_neighbors_avg /=neighbors_volume;

      auto reg_mat = std::make_unique<DenseMatrix>();
      auto _diff_int = std::make_unique<DiffusionIntegrator>();
      auto _h1_int = std::make_unique<SumIntegrator>(0);
      switch (reg_type)
      {
         case l2:
            mass.AssembleElementMatrix(fe_dst_e, *e_trans.get(), *reg_mat.get());
            break;
         case h1:
            _h1_int->AddIntegrator(&mass);
            _h1_int->AddIntegrator(_diff_int.get());
            _h1_int->AssembleElementMatrix(fe_dst_e, *e_trans.get(), *reg_mat.get());
            break;
         case direct:
         default:
            reg_mat.reset(nullptr);
            break;
      }

      if (preserve_volumes)
      {
         // Average at e
         real_t e_volume = mesh.GetElementVolume(e_idx);

         src.GetElementDofValues(e_idx, src_e);
         punity_src.GetElementDofValues(e_idx, punity_src_e);
         punity_dst.GetElementDofValues(e_idx, punity_dst_e);

         mass.AssembleElementMatrix(fe_src_e,
                                    *e_trans.get(),
                                    e_to_e_mat);

         real_t src_e_avg = e_to_e_mat.InnerProduct(src_e, punity_src_e);
         src_e_avg /= e_volume;

         // (Average) projector
         mass.AssembleElementMatrix(fe_dst_e,
                                    *e_trans.get(),
                                    e_to_e_mat);

         e_to_e_mat.MultTranspose(punity_dst_e, e_to_e_avg);
         e_to_e_avg /= e_volume;

         LSSolver(solver,
                  fe_dst_to_neighbors_mat, e_to_e_avg,
                  src_neighbors_avg, src_e_avg,
                  dst_e, reg, reg_mat.get());
      }
      else
      {
         LSSolver(solver,
                  fe_dst_to_neighbors_mat,
                  src_neighbors_avg,
                  dst_e, reg, reg_mat.get());
      }

      if (auto iter_solver = dynamic_cast<IterativeSolver*>(&solver))
      {
         if (!iter_solver->GetConverged())
         {
            mfem_error("\n\tIterative solver failed to converge!");
         }
      }
      if (print_level >= 0) { CheckLSSolver(fe_dst_to_neighbors_mat, src_neighbor, dst_e); }

      dst.SetSubVector(e_dst_dofs, dst_e);

      e_dst_dofs.DeleteAll();
      neighbors_e.DeleteAll();
   }
}

/// @brief An out-of-element, L2-based reconstruction
void L2Reconstruction(Solver& solver,
                      ParGridFunction& src,
                      ParGridFunction& dst,
                      ParMesh& mesh,
                      IterativeSolverParams& newton,
                      RegularizationType reg_type = direct,
                      real_t reg = 0.0,
                      bool preserve_volumes = false)
{
   AsymmetricMassIntegrator mass;
   const auto fes_src = src.ParFESpace();
   const auto fes_dst = dst.ParFESpace();
   // GetElementTransformation requires non-null ptr
   auto neighbor_trans = std::make_unique<IsoparametricTransformation>();
   auto e_trans = std::make_unique<IsoparametricTransformation>();

   Array<int> neighbors_e, e_dst_dofs;

   Vector src_e, dst_e, rhs_e;
   Vector src_neighbor, dst_neighbor;
   Vector punity_src_e, punity_dst_e;
   Vector punity_src_neighbor, punity_dst_neighbor;

   DenseMatrix out_of_elem_shape;

   ConstantCoefficient ccf_ones(1.0);
   ParGridFunction punity_src(fes_src), punity_dst(fes_dst);
   punity_src.ProjectCoefficient(ccf_ones);
   punity_dst.ProjectCoefficient(ccf_ones);

   for (int e_idx = 0; e_idx < mesh.GetNE(); e_idx++ )
   {
      auto& fe_src_e = *fes_src->GetFE(e_idx);
      auto& fe_dst_e = *fes_dst->GetFE(e_idx);
      const int fe_src_e_ndof = fe_src_e.GetDof();
      const int fe_dst_e_ndof = fe_dst_e.GetDof();

      fes_dst->GetElementDofs(e_idx, e_dst_dofs);

      fes_src->GetElementTransformation(e_idx, e_trans.get());

      SaturateNeighborhood(*mesh.pncmesh, e_idx, fe_dst_e_ndof, fe_src_e_ndof,
                           neighbors_e);
      if (preserve_volumes) { neighbors_e.DeleteFirst(e_idx); }
      const int num_neighbors = neighbors_e.Size();

      out_of_elem_shape.SetSize(fe_dst_e_ndof);
      rhs_e.SetSize(fe_dst_e_ndof);
      dst_e.SetSize(fe_dst_e_ndof);
      punity_dst_e.SetSize(fe_dst_e_ndof);

      out_of_elem_shape = 0.0;
      rhs_e = 0.0;
      for (int i = 0; i < num_neighbors; i++)
      {
         const int neighbor_idx = neighbors_e[i];
         auto& fe_dst_neighbor = *fes_dst->GetFE(neighbor_idx);

         fes_dst->GetElementTransformation(neighbor_idx, neighbor_trans.get());

         // Matrix accumulates aditively
         DenseMatrix neighbor_mass;
         mass.AsymmetricElementMatrix(fe_dst_neighbor,
                                      *neighbor_trans.get(),
                                      *e_trans.get(),
                                      neighbor_mass,
                                      newton);

         out_of_elem_shape.AddMatrix(neighbor_mass, 0, 0);

         // RHS accumulates aditively
         DenseMatrix rhs_neighbor_mat;
         mass.AsymmetricElementMatrix(fe_dst_neighbor,
                                      fe_src_e,
                                      *neighbor_trans.get(),
                                      *e_trans.get(),
                                      rhs_neighbor_mat,
                                      newton);

         src.GetElementDofValues(neighbor_idx, src_neighbor);
         rhs_neighbor_mat.AddMultTranspose(src_neighbor, rhs_e);
      }

      auto reg_mat = std::make_unique<DenseMatrix>();
      auto _diff_int = std::make_unique<DiffusionIntegrator>();
      auto _h1_int = std::make_unique<SumIntegrator>(0);
      switch (reg_type)
      {
         case l2:
            mass.AssembleElementMatrix(fe_dst_e, *e_trans.get(), *reg_mat.get());
            break;
         case h1:
            _h1_int->AddIntegrator(&mass);
            _h1_int->AddIntegrator(_diff_int.get());
            _h1_int->AssembleElementMatrix(fe_dst_e, *e_trans.get(), *reg_mat.get());
            break;
         case direct:
         default:
            reg_mat.reset(nullptr);
            break;
      }

      // These systems come from a least squares formulation,
      // so no need to use LSSolver
      if (preserve_volumes)
      {
         DenseMatrix e_to_e_mat;
         Vector e_to_e_avg(fe_dst_e_ndof);
         // Average at e
         real_t e_volume = mesh.GetElementVolume(e_idx);

         src.GetElementDofValues(e_idx, src_e);
         punity_src.GetElementDofValues(e_idx, punity_src_e);
         punity_dst.GetElementDofValues(e_idx, punity_dst_e);

         mass.AssembleElementMatrix(fe_src_e,
                                    *e_trans.get(),
                                    e_to_e_mat);

         real_t src_e_avg = e_to_e_mat.InnerProduct(src_e, punity_src_e);
         src_e_avg /= e_volume;

         // (Average) projector
         mass.AssembleElementMatrix(fe_dst_e,
                                    *e_trans.get(),
                                    e_to_e_mat);

         e_to_e_mat.MultTranspose(punity_dst_e, e_to_e_avg);
         e_to_e_avg /= e_volume;

         LSSolver(solver,
                  out_of_elem_shape, e_to_e_avg,
                  rhs_e, src_e_avg,
                  dst_e, reg, reg_mat.get());
      }
      else
      {
         solver.SetOperator(out_of_elem_shape);
         solver.Mult(rhs_e, dst_e);
      }

      dst.SetSubVector(e_dst_dofs, dst_e);

      neighbors_e.DeleteAll();
      e_dst_dofs.DeleteAll();
   }
}

/// @brief A element-based, face-average reconstruction
void FaceReconstruction(Solver& solver,
                        ParGridFunction& src,
                        ParGridFunction& dst,
                        ParMesh& mesh,
                        RegularizationType reg_type = direct,
                        real_t reg = 0.0,
                        int print_level = -1,
                        bool preserve_volumes = false)
{
   MassIntegrator mass;
   const auto fes_src = src.ParFESpace();
   const auto fes_dst = dst.ParFESpace();

   FaceElementTransformations* face_trans = nullptr;
   auto e_trans = std::make_unique<IsoparametricTransformation>();

   // Auxiliary constant coefficients and partition of unity
   ConstantCoefficient ccf_ones(1.0);
   ParGridFunction punity_src(fes_src), punity_dst(fes_dst);
   punity_src.ProjectCoefficient(ccf_ones);
   punity_dst.ProjectCoefficient(ccf_ones);

   Array<int> faces_e, orientation_e, face_dofs, e_dst_dofs;
   Vector avg_at_face, src_avg_at_face_dofs;

   for (int e_idx = 0; e_idx < mesh.GetNE(); e_idx++)
   {
      auto& fe_dst_e = *fes_dst->GetFE(e_idx);
      const int ndof_e = fe_dst_e.GetDof();

      mesh.GetElementEdges(e_idx, faces_e, orientation_e);

      fes_src->GetElementTransformation(e_idx, e_trans.get());

      // Assumes all faces are equal
      face_trans = mesh.GetFaceElementTransformations(faces_e[0]);
      auto ir = GetCommonIntegrationRule(*fes_src, *fes_dst, *face_trans);

      // Setup RHS and Matrix
      Array<int> offsets(1 + faces_e.Size());
      offsets = ir.GetNPoints();
      offsets[0] = 0;
      offsets.PartialSum();

      DenseMatrix e_to_faces(offsets.Last(), ndof_e);
      BlockVector src_face_values(offsets);
      src_face_values = 0.0;

      for (int i = 0; i < faces_e.Size(); i++)
      {
         const int f_idx = faces_e[i];
         face_trans = mesh.GetFaceElementTransformations(f_idx);

         // RHS
         fes_src->GetFaceDofs(f_idx, face_dofs);
         src.GetSubVector(face_dofs, src_avg_at_face_dofs);
         ComputeFaceAverage(*fes_src, *face_trans, ir, src, avg_at_face);
         src_face_values.AddSubVector(avg_at_face, offsets[i]);

         // Matrix
         DenseMatrix e_to_f;
         ComputeFaceMatrix(e_idx, *fes_dst, *face_trans, ir, e_to_f);
         e_to_faces.SetSubMatrix(offsets[i], 0, e_to_f);
      }

      Vector dst_e;
      fes_dst->GetElementDofs(e_idx, e_dst_dofs);

      auto reg_mat = std::make_unique<DenseMatrix>();
      auto _diff_int = std::make_unique<DiffusionIntegrator>();
      auto _h1_int = std::make_unique<SumIntegrator>(0);
      switch (reg_type)
      {
         case l2:
            mass.AssembleElementMatrix(fe_dst_e, *e_trans.get(), *reg_mat.get());
            break;
         case h1:
            _h1_int->AddIntegrator(&mass);
            _h1_int->AddIntegrator(_diff_int.get());
            _h1_int->AssembleElementMatrix(fe_dst_e, *e_trans.get(), *reg_mat.get());
            break;
         case direct:
         default:
            reg_mat.reset(nullptr);
            break;
      }

      if (preserve_volumes)
      {
         // Average at e
         real_t e_volume = mesh.GetElementVolume(e_idx);

         fes_dst->GetElementTransformation(e_idx, e_trans.get());
         auto& fe_dst_e = *fes_dst->GetFE(e_idx);
         auto& fe_src_e = *fes_src->GetFE(e_idx);
         const int fe_dst_e_ndof = fe_dst_e.GetDof();

         Vector src_e, punity_src_e, punity_dst_e;
         Vector e_to_e_avg(fe_dst_e_ndof);

         src.GetElementDofValues(e_idx, src_e);
         punity_src.GetElementDofValues(e_idx, punity_src_e);
         punity_dst.GetElementDofValues(e_idx, punity_dst_e);

         DenseMatrix e_to_e_mat;
         mass.AssembleElementMatrix(fe_src_e,
                                    *e_trans.get(),
                                    e_to_e_mat);

         real_t src_e_avg = e_to_e_mat.InnerProduct(src_e, punity_src_e);
         src_e_avg /= e_volume;

         // Average projector
         mass.AssembleElementMatrix(fe_dst_e,
                                    *e_trans.get(),
                                    e_to_e_mat);

         e_to_e_mat.MultTranspose(punity_dst_e, e_to_e_avg);
         e_to_e_avg /= e_volume;

         LSSolver(solver,
                  e_to_faces, e_to_e_avg,
                  src_face_values, src_e_avg,
                  dst_e, reg, reg_mat.get());
      }
      else
      {
         LSSolver(solver, e_to_faces, src_face_values, dst_e, reg, reg_mat.get());
      }

      if (auto iter_solver = dynamic_cast<IterativeSolver*>(&solver))
      {
         if (!iter_solver->GetConverged())
         {
            mfem_error("\n\tIterative solver failed to converge!");
         }
      }
      if (print_level >= 0) { CheckLSSolver(e_to_faces, src_face_values, dst_e); }
      dst.SetSubVector(e_dst_dofs, dst_e);

      e_dst_dofs.DeleteAll();
      faces_e.DeleteAll();
      orientation_e.DeleteAll();
      face_dofs.DeleteAll();
   }
}

/// @brief A element-based, face-average reconstruction using weak jumps
void WeakFaceReconstruction(Solver& solver,
                            ParGridFunction& src,
                            ParGridFunction& dst,
                            ParMesh& mesh,
                            RegularizationType reg_type = direct,
                            real_t reg = 0.0,
                            int print_level = -1,
                            bool preserve_volumes = false)
{
   auto& fes_src = *src.ParFESpace();
   auto& fes_dst = *dst.ParFESpace();

   Array<int> e_dst_dofs;
   Array<int> faces_e, orientation_e;

   Vector src_e, dst_e;
   Vector punity_src_e, punity_dst_e;

   // Auxiliary constant coefficients and partition of unity
   ConstantCoefficient ccf_ones(1.0);
   ParGridFunction punity_src(&fes_src), punity_dst(&fes_dst);
   punity_src.ProjectCoefficient(ccf_ones);
   punity_dst.ProjectCoefficient(ccf_ones);

   MassIntegrator mass;
   TraceIntegrator trace_int;
   TraceAverageIntegrator average_int;
   TraceJumpIntegrator jump_int;

   for (int e_idx=0; e_idx < mesh.GetNE(); e_idx++)
   {
      const auto& fe_src_e = *fes_src.GetFE(e_idx);
      const auto& fe_dst_e = *fes_dst.GetFE(e_idx);

      const int dst_e_ndofs = fe_dst_e.GetDof();

      fes_dst.GetElementDofs(e_idx, e_dst_dofs);

      mesh.GetElementEdges(e_idx, faces_e, orientation_e);

      auto e_trans = fes_src.GetElementTransformation(e_idx);

      Vector e_rhs(dst_e_ndofs);
      DenseMatrix e_mat(dst_e_ndofs);
      e_rhs = 0.0;
      e_mat = 0.0;
      for (int i = 0; i < faces_e.Size(); i++)
      {
         const int f_idx = faces_e[i];

         auto& face_trans = *mesh.GetFaceElementTransformations(f_idx);
         const int elem1_idx = face_trans.Elem1No;
         const int elem2_idx = face_trans.Elem2No;

         const auto& fe_elem1 = *fes_src.GetFE(elem1_idx);
         const auto& fe_elem2 = (elem2_idx>=0)?*fes_src.GetFE(e_idx):fe_elem1;

         DenseMatrix local_face_mat;

         Array<int> offsets(3);
         offsets[0] = 0;
         offsets[1] = fes_src.GetFE(elem1_idx)->GetDof();
         offsets[2] = (elem2_idx>=0)?fes_src.GetFE(elem2_idx)->GetDof():0;
         offsets.PartialSum();
         BlockVector face_avg(offsets);
         face_avg = 0.0;

         Vector src_elem;
         src.GetElementDofValues(elem1_idx, src_elem);
         face_avg.AddSubVector(src_elem, 0);
         if (elem2_idx >=0)
         {
            src.GetElementDofValues(elem2_idx, src_elem);
            face_avg.AddSubVector(src_elem, offsets[1]);
         }

         Vector punity_dst_e, face_trace(dst_e_ndofs);
         punity_dst.GetElementDofValues(e_idx, punity_dst_e);

         trace_int.AssembleTraceFaceMatrix(e_idx, fe_dst_e, fe_dst_e, face_trans,
                                           local_face_mat);
         local_face_mat.Mult(punity_dst_e, face_trace);

         average_int.AssembleFaceMatrix(fe_dst_e, fe_elem1, fe_elem2, face_trans,
                                        local_face_mat);

         add(e_rhs, local_face_mat.InnerProduct(punity_dst_e,face_avg), face_trace,
             e_rhs);
         AddMultVVt(face_trace, e_mat);
      }

      // Regularize
      auto reg_mat = std::make_unique<DenseMatrix>();
      auto _diff_int = std::make_unique<DiffusionIntegrator>();
      auto _h1_int = std::make_unique<SumIntegrator>(0);
      switch (reg_type)
      {
         case l2:
            mass.AssembleElementMatrix(fe_dst_e, *e_trans, *reg_mat.get());
            break;
         case h1:
            _h1_int->AddIntegrator(&mass);
            _h1_int->AddIntegrator(_diff_int.get());
            _h1_int->AssembleElementMatrix(fe_dst_e, *e_trans, *reg_mat.get());
            break;
         case direct:
         default:
            reg_mat.reset(nullptr);
            break;
      }

      // Solve
      dst_e.SetSize(dst_e_ndofs);
      if (preserve_volumes)
      {
         DenseMatrix e_to_e_mat;
         Vector e_to_e_avg;
         // Average at e
         real_t e_volume = mesh.GetElementVolume(e_idx);

         src.GetElementDofValues(e_idx, src_e);
         punity_src.GetElementDofValues(e_idx, punity_src_e);
         punity_dst.GetElementDofValues(e_idx, punity_dst_e);

         mass.AssembleElementMatrix(fe_src_e, *e_trans, e_to_e_mat);

         real_t src_e_avg = e_to_e_mat.InnerProduct(src_e, punity_src_e);
         src_e_avg /= e_volume;

         // (Average) projector
         mass.AssembleElementMatrix(fe_dst_e, *e_trans, e_to_e_mat);

         e_to_e_mat.MultTranspose(punity_dst_e, e_to_e_avg);
         e_to_e_avg /= e_volume;

         LSSolver(solver, e_mat, e_to_e_avg, e_rhs, src_e_avg, dst_e, reg,
                  reg_mat.get());
      }
      else
      {
         LSSolver(solver, e_mat, e_rhs, dst_e, reg, reg_mat.get());
      }

      if (auto iter_solver = dynamic_cast<IterativeSolver*>(&solver))
      {
         if (!iter_solver->GetConverged())
         {
            mfem_error("\n\tIterative solver failed to converge!");
         }
      }
      if (print_level >= 0) { CheckLSSolver(e_mat, e_rhs, dst_e); }

      dst.SetSubVector(e_dst_dofs, dst_e);

      e_dst_dofs.DeleteAll();
      faces_e.DeleteAll();
      orientation_e.DeleteAll();
   }
}

/// @brief A minimum-variation, face-constrained reconstruction average
void BoundedVariationReconstruction(Solver& solver,
                                    ParGridFunction& src,
                                    ParGridFunction& dst,
                                    ParMesh& mesh)
{
   mfem_error("Not fully implemented yet!");
   DiffusionIntegrator diffusion_int;

   const auto fes_src = src.ParFESpace();
   const auto fes_dst = dst.ParFESpace();

   auto e_trans = std::make_unique<IsoparametricTransformation>();
   auto neighbor_trans = std::make_unique<IsoparametricTransformation>();

   Array<int> e_dst_dofs, neighbors_e;

   for (int e_idx=0; e_idx < mesh.GetNE(); e_idx++)
   {
      auto& fe_src_e = *fes_src->GetFE(e_idx);
      auto& fe_dst_e = *fes_dst->GetFE(e_idx);
      const int fe_src_e_ndof = fe_src_e.GetDof();
      const int fe_dst_e_ndof = fe_dst_e.GetDof();

      fes_dst->GetElementDofs(e_idx, e_dst_dofs);

      fes_dst->GetElementTransformation(e_idx, e_trans.get());

      SaturateNeighborhood(*mesh.pncmesh, e_idx, fe_dst_e_ndof, fe_src_e_ndof,
                           neighbors_e);
      // if (preserve_volumes) { neighbors_e.DeleteFirst(e_idx); }
      const int num_neighbors = neighbors_e.Size();

      // Minimizer
      DenseMatrix diffusion_mat(fe_dst_e_ndof);
      diffusion_mat = 0.0;

      for (int i = 0; i < num_neighbors; i++)
      {
         const int neighbor_idx = neighbors_e[i];
         auto& fe_dst_neighbor = *fes_dst->GetFE(neighbor_idx);

         fes_dst->GetElementTransformation(neighbor_idx, neighbor_trans.get());

         DenseMatrix neighbor_diffusion_mat;
         diffusion_int.AssembleElementMatrix(fe_dst_neighbor,
                                             *neighbor_trans.get(),
                                             diffusion_mat);
         diffusion_mat.AddMatrix(neighbor_diffusion_mat, 0, 0);
      }


      // Constaint
      //
      // RHS
      //
      // RHS constaint
      /*
       * // compute <{u_hat} (xhat dot n), psi_hat>_E
       * VectorConstantCoefficient xhat(Vector({1.0,0.0}));
       * Vector b_xhat(mesh.GetNE());
       * ParBilinearForm B_xhat(src.ParFESpace());
       * B_xhat.AddInteriorFaceIntegrator(new DGTraceIntegrator(xhat, 1.0, 0.0));
       * B_xhat.AddBdrFaceIntegrator(new DGTraceIntegrator(xhat, 2.0,
       *                                                   0.0)); // note the 2 enforces du/dx=0 at the x boundaries
       * B_xhat.Assemble();
       * B_xhat.Finalize();
       * matrix = std::unique_ptr<HypreParMatrix>(B_xhat.ParallelAssemble());
       * matrix->Mult(*src.GetTrueDofs(), b_xhat);

       * // compute <{u_hat} (yhat dot n), psi_hat>_E
       * VectorConstantCoefficient yhat(Vector({0.0,1.0}));
       * Vector b_yhat(mesh.GetNE());
       * ParBilinearForm B_yhat(src.ParFESpace());
       * B_yhat.AddInteriorFaceIntegrator(new DGTraceIntegrator(yhat, 1.0, 0.0));
       * B_yhat.AddBdrFaceIntegrator(new DGTraceIntegrator(yhat, 2.0,
       *                                                   0.0)); // note the 2 enforces du/dy=0 at the y boundaries
       * B_yhat.Assemble();
       * B_yhat.Finalize();
       * matrix = std::unique_ptr<HypreParMatrix>(B_yhat.ParallelAssemble());
       * matrix->Mult(*src.GetTrueDofs(), b_yhat); */

      e_dst_dofs.DeleteAll();
      neighbors_e.DeleteAll();
   }
}

///@}

namespace example
{

using std::exp, std::sin, std::hypot, std::function;
real_t k_x;
real_t k_y;

function<real_t(const Vector &)> plane = [](const Vector& x)
{
   return 1.0 + k_x * x(0) + k_y * x(1);
};

function<real_t(const Vector &)> sine = [](const Vector& x)
{
   return sin(2.0 * M_PI * k_x * x(0)) * sin(2.0 * M_PI * k_y * x(1));
};

function<real_t(const Vector &)> exp_sine = [](const Vector& x)
{
   return exp(hypot(x(0),x(1))) *
          sin(2.0 * M_PI * k_x * x(0)) *
          sin(2.0 * M_PI * k_y * x(1));
};

} // end namespace example

} // end namespace mfem::reconstruction

using namespace mfem;
using namespace reconstruction;

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);

   // Default command-line options
   GlobalConfiguration params
   {
      .rec = norm,
      .vis = {
         .save_to_files = false,
         .visualization = false,
         .port = 19916,
      },
      .newton = {
         .print_level = 0,
         .max_iter = 1000,
         .rtol = 1.0e-15,
         .atol = 0.0,
      },
      .solver_type = direct_inv,
      .solver = {
         .print_level = -1,
         .max_iter = 1000,
         .rtol = 1.0e-30,
         .atol = 0.0,
      },
      .reg_type = direct,
      .reg = 0.0,
      .example = plane,
      .ord_smooth = 3,
      .ord_src = 0,
      .ord_dst = 1,
      .ser_ref = 0,
      .par_ref = 0,
      .preserve_volumes = false,
   };
   example::k_x = 2.0;
   example::k_y = 4.0;

   // Parse options
   OptionsParser args(argc, argv);

   args.AddOption((int*)&params.rec, "-rec", "--reconstruction",
                  "Reconstruction methods to be considered:"
                  "\n\t0: L2-norm least squares"
                  "\n\t1: Local-averages least square"
                  "\n\t2: Face-averages (DFR-like) method"
                  "\n\t3: Weak face-averages (DFR-like) method");

   args.AddOption(&params.vis.save_to_files, "-s", "--save", "-no-s",
                  "--no-save", "Show or not show approximation error.");
   args.AddOption(&params.vis.visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.AddOption(&params.vis.port, "-p", "--send-port", "Socket for GLVis.");

   args.AddOption((int*)&params.solver_type, "-S", "--solver",
                  "Solvers to be considered:"
                  "\n\t0: Direct solver"
                  "\n\t1: CG - Conjugate Gradient"
                  "\n\t2: BICGSTAB - BIConjugate Gradient STABilized"
                  "\n\t3: MINRES - MINimal RESidual");

   args.AddOption(&params.solver.print_level, "-Spl", "--solver-print",
                  "Print level for the iterative solver:"
                  "\n\t0: All"
                  "\n\t1: First and last with warnings"
                  "\n\t2: Errors"
                  "\n\t3: None"
                  "\n\tNOTE: A negative value deactivates LS check");
   args.AddOption(&params.solver.max_iter, "-Smi", "--solver-max-iter",
                  "Maximum number of iterations for the solver");
   args.AddOption(&params.solver.rtol, "-Srtol", "--solver-rtol",
                  "Relative tolerance for the iterative solver");
   args.AddOption(&params.solver.atol, "-Satol", "--solver-atol",
                  "Absolute tolerance for the iterative solver");

   args.AddOption(&params.newton.print_level, "-Npl", "--newton-print-level",
                  "Print level for the Newton solver (q-points)");
   args.AddOption(&params.newton.max_iter, "-Nmi", "--newton-max-iter",
                  "Maximum number of iterations for the Newton solver (q-points)");
   args.AddOption(&params.newton.rtol, "-Nrtol", "--newton-rtol",
                  "Relative tolerance for the Newton solver (q-points)");
   args.AddOption(&params.newton.atol, "-Natol", "--newton-atol",
                  "Absolute tolerance for the Newton solver (q-points)");

   args.AddOption((int*)&params.reg_type, "-reg", "--regularization",
                  "Regularization types to be considered:"
                  "\n\t0: Direct ('little l2') regularization"
                  "\n\t1: L2 regularization"
                  "\n\t2: H1 regularization");
   args.AddOption(&params.reg, "-reg-f", "--regularization-factor",
                  "Regularization factor for the least squares problem");

   args.AddOption(&params.ord_smooth, "-O", "--order-smooth",
                  "Order original broken space.");
   // TODO(Gabriel): Not implemented yet
   // args.AddOption(&params.ord_src, "-A", "--order-source",
   //                "Order source broken space.");
   args.AddOption(&params.ord_dst, "-R", "--order-reconstruction",
                  "Order of reconstruction broken space.");

   args.AddOption(&params.ser_ref, "-rs", "--refine-serial",
                  "Number of serial refinement steps.");
   args.AddOption(&params.par_ref, "-rp", "--refine-parallel",
                  "Number of parallel refinement steps.");
   args.AddOption(&params.preserve_volumes, "-V", "--preserve-volumes", "-no-V",
                  "--no-preserve-volumes", "Preserve averages (volumes) by"
                  " solving a constrained least squares problem");

   args.AddOption((int*)&params.example, "-E", "--example",
                  "Function to be reconstructed:"
                  "\n\t0: 1 + k_x x + k_y y"
                  "\n\t1: sin(k_x x) sin(k_y y)"
                  "\n\t2: exp(r) cos(k_x x) sin(k_y y)");
   args.AddOption(&example::k_x, "-Ex", "--example-Kx", "Value k_x");
   args.AddOption(&example::k_y, "-Ey", "--example-Ky", "Value k_y");

   args.ParseCheck();
   MFEM_VERIFY(params.ser_ref >= 0, "Invalid number of serial refinements!");
   MFEM_VERIFY(params.par_ref >= 0, "Invalid number of parallel refinements!");
   MFEM_VERIFY(params.reg >= 0.0,   "Invalid regularization term!");
   MFEM_VERIFY(params.ord_smooth > params.ord_src,
               "Smooth space must be more regular!");
   MFEM_VERIFY((0 <= params.solver_type) && (params.solver_type < num_solvers),
               "Invalid solver type: " << params.solver_type);
   MFEM_VERIFY((0 <= params.reg_type) && (params.reg_type < num_regularization),
               "Invalid regularization type: " << params.reg_type);

   if (Mpi::Root())
   {
      mfem::out << "Number of serial refs.:  " << params.ser_ref << "\n"
                << "Number of parallel refs: " << params.par_ref << "\n\n"
                << "Order smooth FES:        " << params.ord_smooth << "\n"
                << "Order source FES:        " << params.ord_src << "\n"
                << "Order destination FES:   " << params.ord_dst << "\n\n"
                << "Newton relative tol:     " << params.newton.rtol << "\n"
                // << "Newton absolute tol:     " << params.newton.atol << "\n"
                << "Newton max. num. iter.:  " << params.newton.max_iter << "\n\n"
                << std::endl;
   }

   // Mesh
   const int num_x = 2;
   const int num_y = 2;
   Mesh serial_mesh = Mesh::MakeCartesian2D(num_x, num_y, Element::QUADRILATERAL);
   for (int i = 0; i < params.ser_ref; ++i) { serial_mesh.UniformRefinement(); }
   serial_mesh.EnsureNCMesh();
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   for (int i = 0; i < params.par_ref; ++i) { mesh.UniformRefinement(); }
   mesh.EnsureNCMesh();

   // target function u(x,y)
   std::function<real_t(const Vector &)> u_function;
   switch (params.example)
   {
      default:
      case plane:
         u_function = example::plane;
         break;
      case sine:
         u_function = example::sine;
         break;
      case exp_sine:
         u_function = example::exp_sine;
         break;
   }
   FunctionCoefficient u_coefficient(u_function);

   // Broken spaces
   L2_FECollection fec_smooth(params.ord_smooth, mesh.Dimension());
   L2_FECollection fec_src(params.ord_src, mesh.Dimension());
   L2_FECollection fec_dst(params.ord_dst, mesh.Dimension());

   ParFiniteElementSpace fes_smooth(&mesh, &fec_smooth);
   ParFiniteElementSpace fes_src(&mesh, &fec_src);
   ParFiniteElementSpace fes_dst(&mesh, &fec_dst);

   // Grid Functions
   ParGridFunction u_smooth(&fes_smooth);
   ParGridFunction u_src(&fes_src);
   ParGridFunction u_dst_avg(&fes_src);
   ParGridFunction diff(&fes_src);
   ParGridFunction u_dst(&fes_dst);

   u_smooth.ProjectCoefficient(u_coefficient);
   // TODO(Gabriel): What if fes_src is H.O.?
   u_smooth.GetElementAverages(u_src);

   // Solver choice
   // TODO(Gabriel): Support more PCs?
   /* Notes:
    * OperatorJacobiSmoother will call AssembleDiagonal on
    * Operator, even if the Setup function will be called
    * later on. DenseMatrix does not have this method
    * implemented.
    */
   std::shared_ptr<Solver> solver(nullptr);
   switch (params.solver_type)
   {
      case bicgstab:
         solver.reset(new BiCGSTABSolver(MPI_COMM_WORLD));
         break;
      case cg:
         solver.reset(new CGSolver(MPI_COMM_WORLD));
         break;
      case minres:
         solver.reset(new MINRESSolver(MPI_COMM_WORLD));
         break;
      case direct_inv:
      default:
         solver.reset(new DenseMatrixInverse());
   }

   // Solver setting
   solver->iterative_mode = false;
   if (params.solver_type != direct_inv)
   {
      auto it_solver = std::static_pointer_cast<IterativeSolver>(solver);
      auto& isp = params.solver;
      IterativeSolver::PrintLevel print_level;
      switch (isp.print_level)
      {
         case 0:
            print_level.All();
            break;
         case 1:
            print_level.FirstAndLast();
            print_level.Warnings();
         case 2:
            print_level.Errors();
            break;
         default:
            print_level.None();
      }

      it_solver->SetPrintLevel(print_level);
      it_solver->SetMaxIter(isp.max_iter);
      it_solver->SetRelTol(isp.rtol);
      it_solver->SetAbsTol(isp.atol);
   }

   // Reconstruction function
   switch (params.rec)
   {
      case norm:
         L2Reconstruction(*solver.get(), u_src, u_dst, mesh, params.newton,
                          params.reg_type, params.reg, params.preserve_volumes);
         break;
      case face_average:
         FaceReconstruction(*solver.get(), u_src, u_dst, mesh, params.reg_type,
                            params.reg, params.solver.print_level, params.preserve_volumes);
         break;
      case weak_face_average:
         WeakFaceReconstruction(*solver.get(), u_src, u_dst, mesh);
         break;
      case bounded_variation:
         BoundedVariationReconstruction(*solver.get(), u_src, u_dst, mesh);
         break;
      case average:
      default:
         AverageReconstruction(*solver.get(), u_src, u_dst, mesh, params.newton,
                               params.reg_type, params.reg, params.solver.print_level,
                               params.preserve_volumes);
   }
   u_dst.GetElementAverages(u_dst_avg);

   // Visualization
   {
      char vishost[] = "localhost";
      socketstream glvis_smooth(vishost, params.vis.port);
      socketstream glvis_src(vishost, params.vis.port);
      socketstream glvis_dst_avg(vishost, params.vis.port);
      socketstream glvis_dst(vishost, params.vis.port);

      if (glvis_smooth && glvis_src && glvis_dst_avg && glvis_dst &&
          params.vis.visualization)
      {
         //glvis_smooth.precision(8);
         glvis_smooth << "parallel " << mesh.GetNRanks()
                      << " " << mesh.GetMyRank() << "\n"
                      << "solution\n" << mesh << u_smooth
                      << "window_title 'original'\n" << std::flush;
         MPI_Barrier(mesh.GetComm());
         //glvis_src.precision(8);
         glvis_src << "parallel " << mesh.GetNRanks()
                   << " " << mesh.GetMyRank() << "\n"
                   << "solution\n" << mesh << u_src
                   << "window_title 'averages'\n" << std::flush;
         MPI_Barrier(mesh.GetComm());
         //glvis_dst.precision(8);
         glvis_dst << "parallel " << mesh.GetNRanks()
                   << " " << mesh.GetMyRank() << "\n"
                   << "solution\n" << mesh << u_dst
                   << "window_title 'reconstruction'\n" << std::flush;
         MPI_Barrier(mesh.GetComm());
         //glvis_dst.precision(8);
         glvis_dst_avg << "parallel " << mesh.GetNRanks()
                       << " " << mesh.GetMyRank() << "\n"
                       << "solution\n" << mesh << u_dst_avg
                       << "window_title 'rec average'\n" << std::flush;
      }
      else if (params.vis.visualization)
      {
         MFEM_WARNING("Cannot connect to glvis server, disabling visualization.")
      }
   }

   // Error studies
   real_t error = u_dst.ComputeL2Error(u_coefficient);

   ConstantCoefficient ccf_zeros(0.0);
   subtract(u_dst_avg, u_src, diff);
   real_t error_avg = diff.ComputeL2Error(ccf_zeros);

   if (Mpi::Root())
   {
      mfem::out << "\n|| Rec(u_h) - u ||_{L^2} = " << error << "\n" << std::endl;
      mfem::out << "\n|| Avg(Rec(u_h)) - u_h ||_{L^2} = " << error_avg << "\n" <<
                std::endl;
   }

   if (params.vis.save_to_files && Mpi::Root())
   {
      Vector el_error(mesh.GetNE());
      ParGridFunction punity_src(&fes_src);
      ConstantCoefficient ccf_ones(1.0);
      punity_src.ProjectCoefficient(ccf_ones);

      punity_src.ComputeElementLpErrors(2.0, ccf_zeros, el_error);
      real_t hmax = el_error.Max();

      std::ofstream file;
      file.open("convergence.csv", std::ios::out | std::ios::app);
      if (!file.is_open()) { mfem_error("Failed to open file"); }
      file << std::scientific << std::setprecision(16);
      file << error
           << "," << fes_src.GetNDofs()
           << "," << fes_dst.GetNDofs()
           << "," << hmax
           << "," << mesh.GetNE() << std::endl;
      file.close();
   }

   Mpi::Finalize();
   return 0;
}
