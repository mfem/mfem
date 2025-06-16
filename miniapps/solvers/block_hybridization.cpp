// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "block_hybridization.hpp"

using namespace std;
using namespace mfem;
using namespace blocksolvers;

namespace mfem
{
namespace blocksolvers
{
void BlockHybridizationSolver::Init(const int ne)
{
   hat_offsets.SetSize(ne + 1);
   hat_offsets[0] = 0;
   for (int i = 0; i < ne; ++i)
   {
      hat_offsets[i + 1] = trial_space.GetFE(i)->GetDof();
   }
   hat_offsets.PartialSum();

   data_offsets.SetSize(ne + 1);
   data_offsets[0] = 0;

   ipiv_offsets.SetSize(ne + 1);
   ipiv_offsets[0] = 0;

   test_offsets.SetSize(ne + 1);
   test_offsets[0] = 0;

   for (int i = 0; i < ne; ++i)
   {
      const int trial_size = trial_space.GetFE(i)->GetDof();
      test_offsets[i + 1] = test_space.GetFE(i)->GetDof();
      const int matrix_size = trial_size + test_offsets[i + 1];

      data_offsets[i + 1] = data_offsets[i] + matrix_size*matrix_size;
      ipiv_offsets[i + 1] = ipiv_offsets[i] + matrix_size;
   }
   test_offsets.PartialSum();

   data = new double[data_offsets.Last()]();
   ipiv = new int[ipiv_offsets.Last()];

   mixed_dofs.Reserve(ipiv_offsets.Last());
}

void BlockHybridizationSolver::ConstructCt(const ParFiniteElementSpace &c_space)
{
   ParMesh &pmesh(*trial_space.GetParMesh());
   Ct = new SparseMatrix(hat_offsets.Last(), c_space.GetNDofs());
   Array<int> dofs, c_dofs;
   const double eps = 1e-12;
   DenseMatrix elmat;
   FaceElementTransformations *FTr;
   NormalTraceJumpIntegrator c_int;
   const int num_faces = pmesh.GetNumFaces();

   for (int i = 0; i < num_faces; ++i)
   {
      FTr = pmesh.GetInteriorFaceTransformations(i);
      if (!FTr)
      {
         continue;
      }

      int o1 = hat_offsets[FTr->Elem1No];
      int s1 = hat_offsets[FTr->Elem1No + 1] - o1;
      int o2 = hat_offsets[FTr->Elem2No];
      int s2 = hat_offsets[FTr->Elem2No + 1] - o2;

      dofs.SetSize(s1 + s2);
      for (int j = 0; j < s1; ++j)
      {
         dofs[j] = o1 + j;
      }
      for (int j = 0; j < s2; ++j)
      {
         dofs[s1 + j] = o2 + j;
      }
      c_space.GetFaceDofs(i, c_dofs);
      c_int.AssembleFaceMatrix(*c_space.GetFaceElement(i),
                               *trial_space.GetFE(FTr->Elem1No),
                               *trial_space.GetFE(FTr->Elem2No),
                               *FTr,
                               elmat);
      elmat.Threshold(eps * elmat.MaxMaxNorm());
      Ct->AddSubMatrix(dofs, c_dofs, elmat);
   }

   const int num_shared_faces = pmesh.GetNSharedFaces();
   for (int i = 0; i < num_shared_faces; ++i)
   {
      const int face_no = pmesh.GetSharedFace(i);
      FTr = pmesh.GetFaceElementTransformations(face_no);
      c_space.GetFaceDofs(face_no, c_dofs);
      const FiniteElement *face_fe(c_space.GetFaceElement(face_no));
      const FiniteElement *fe(trial_space.GetFE(FTr->Elem1No));

      int o1 = hat_offsets[FTr->Elem1No];
      int s1 = hat_offsets[FTr->Elem1No + 1] - o1;

      dofs.SetSize(s1);
      for (int j = 0; j < s1; ++j)
      {
         dofs[j] = o1 + j;
      }
      c_int.AssembleFaceMatrix(*face_fe, *fe, *fe, *FTr, elmat);
      elmat.Threshold(eps * elmat.MaxMaxNorm());
      Ct->AddSubMatrix(dofs, c_dofs, elmat);
   }
   Ct->Finalize();
}

void BlockHybridizationSolver::ConstructH(const shared_ptr<ParBilinearForm> &a,
                                          const shared_ptr<ParMixedBilinearForm> &b,
                                          const Array<int> &marker,
                                          const ParFiniteElementSpace &c_space)
{
   ParMesh &pmesh(*trial_space.GetParMesh());
   const int ne = pmesh.GetNE();
   const double eps = 1e-12;
   SparseMatrix H(Ct->Width());
   DenseMatrix Ct_local, Minv_Ct_local, H_local;

   Array<int> dofs, c_dofs;
   Array<int> c_dof_marker(Ct->Width());
   c_dof_marker = -1;
   int c_mark_start = 0;

   for (int i = 0; i < ne; ++i)
   {
      trial_space.GetElementDofs(i, dofs);
      const int trial_size = dofs.Size();

      DenseMatrix A(trial_size);
      a->ComputeElementMatrix(i, A);
      A.Threshold(eps * A.MaxMaxNorm());

      const int test_offset = test_offsets[i];
      const int test_size = test_offsets[i + 1] - test_offset;

      DenseMatrix B(test_size, trial_size);
      b->ComputeElementMatrix(i, B);
      B.Neg();
      B.Threshold(eps * B.MaxMaxNorm());

      const int matrix_size = trial_size + test_size;
      DenseMatrix M(data + data_offsets[i], matrix_size, matrix_size);

      M.CopyMN(A, 0, 0);
      M.CopyMN(B, trial_size, 0);
      M.CopyMNt(B, 0, trial_size);

      if (elimination_)
      {
         FiniteElementSpace::AdjustVDofs(dofs);
         for (int j = 0; j < trial_size; ++j)
         {
            if (marker[dofs[j]])
            {
               for (int k = 0; k < matrix_size; ++k)
               {
                  if (k == j)
                  {
                     M(k, k) = 1.0;
                  }
                  else
                  {
                     M(k, j) = 0.0;
                     M(j, k) = 0.0;
                  }
               }
            }
         }
      }

      c_dofs.SetSize(0);
      dofs.SetSize(trial_size);
      const int hat_offset = hat_offsets[i];

      for (int j = 0; j < trial_size; ++j)
      {
         const int row = hat_offset + j;
         const int ncols = Ct->RowSize(row);
         const int *cols = Ct->GetRowColumns(row);
         for (int l = 0; l < ncols; ++l)
         {
            const int c_dof = cols[l];
            if (c_dof_marker[c_dof] < c_mark_start)
            {
               c_dof_marker[c_dof] = c_mark_start + c_dofs.Size();
               c_dofs.Append(c_dof);
            }
         }
         dofs[j] = row;
      }

      mixed_dofs.Append(dofs);
      for (int j = 0; j < test_size; ++j)
      {
         mixed_dofs.Append(hat_offsets.Last() + test_offset + j);
      }

      Ct_local.SetSize(M.Height(), c_dofs.Size()); // Ct_local = [C 0]^T
      Ct_local = 0.0;
      for (int j = 0; j < trial_size; ++j)
      {
         const int row = dofs[j];
         const int ncols = Ct->RowSize(row);
         const int *cols = Ct->GetRowColumns(row);
         const double *vals = Ct->GetRowEntries(row);
         for (int l = 0; l < ncols; ++l)
         {
            const int loc = c_dof_marker[cols[l]] - c_mark_start;
            Ct_local(j, loc) = vals[l];
         }
      }

      LUFactors Minv(data + data_offsets[i], ipiv + ipiv_offsets[i]);
      Minv.Factor(matrix_size);
      Minv_Ct_local = Ct_local;
      Minv.Solve(Ct_local.Height(), Ct_local.Width(), Minv_Ct_local.Data());

      H_local.SetSize(Ct_local.Width());

      MultAtB(Ct_local, Minv_Ct_local, H_local);
      H.AddSubMatrix(c_dofs, c_dofs, H_local);

      c_mark_start += c_dofs.Size();
      MFEM_VERIFY(c_mark_start >= 0, "overflow");
   }
   H.Finalize(1, true);  // skip_zeros = 1 (default), fix_empty_rows = true

   OperatorPtr pP(Operator::Hypre_ParCSR);
   pP.ConvertFrom(c_space.Dof_TrueDof_Matrix());
   OperatorPtr dH(pP.Type());
   dH.MakeSquareBlockDiag(c_space.GetComm(), c_space.GlobalVSize(),
                          c_space.GetDofOffsets(), &H);
   OperatorPtr AP(ParMult(dH.As<HypreParMatrix>(), pP.As<HypreParMatrix>()));
   OperatorPtr R(pP.As<HypreParMatrix>()->Transpose());
   pH = ParMult(R.As<HypreParMatrix>(), AP.As<HypreParMatrix>(), true);
}

BlockHybridizationSolver::BlockHybridizationSolver(const
                                                   shared_ptr<ParBilinearForm> &a,
                                                   const shared_ptr<ParMixedBilinearForm> &b,
                                                   const IterSolveParameters &param,
                                                   const Array<int> &ess_bdr_attr)
   : DarcySolver(a->ParFESpace()->GetTrueVSize(),
                 b->TestParFESpace()->GetTrueVSize()),
     trial_space(*a->ParFESpace()), test_space(*b->TestParFESpace()),
     elimination_(false),
     solver_(a->ParFESpace()->GetComm())
{
   ParMesh &pmesh(*trial_space.GetParMesh());
   const int ne = pmesh.GetNE();

   StopWatch chrono;

   chrono.Start();
   Init(ne);
   chrono.Stop();
   cout << "init time: " << chrono.RealTime() << endl;

   Array<int> ess_dof_marker;
   for (int attr : ess_bdr_attr)
   {
      if (attr)
      {
         elimination_ = true;
         trial_space.GetEssentialVDofs(ess_bdr_attr, ess_dof_marker);
         break;
      }
   }

   chrono.Restart();
   const int order = trial_space.FEColl()->GetOrder()-1;
   DG_Interface_FECollection fec(order, pmesh.Dimension());
   c_fes = new ParFiniteElementSpace(&pmesh, &fec);
   ParFiniteElementSpace &c_space(*c_fes);

   ConstructCt(c_space);
   chrono.Stop();
   cout << "ct time: " << chrono.RealTime() << endl;

   chrono.Restart();
   ConstructH(a, b, ess_dof_marker, c_space);
   chrono.Stop();
   cout << "h time: " << chrono.RealTime() << endl;

   M = new HypreBoomerAMG(*pH);
   M->SetPrintLevel(0);

   SetOptions(solver_, param);
   solver_.SetPreconditioner(*M);
   solver_.SetOperator(*pH);
}

BlockHybridizationSolver::~BlockHybridizationSolver()
{
   delete M;
   delete [] ipiv;
   delete [] data;
   delete Ct;
   delete c_fes;
}

void BlockHybridizationSolver::ReduceRHS(const Vector &b,
                                         const Vector &sol,
                                         BlockVector &rhs,
                                         Vector &b_r) const
{
   const SparseMatrix &R = *trial_space.GetRestrictionMatrix();
   Vector x_e(b);
   if (elimination_) { EliminateEssentialBC(sol, x_e);}
   Vector x0(R.Width());
   BlockVector block_x(x_e.GetData(), offsets_);
   R.MultTranspose(block_x.GetBlock(0), x0);

   ParMesh &pmesh(*trial_space.GetParMesh());
   const int ne = pmesh.GetNE();

   rhs.SetVector(block_x.GetBlock(1), hat_offsets.Last());
   Array<bool> dof_marker(x0.Size());
   dof_marker = false;
   Array<int> dofs;
   Vector Minv_sub_vec, g_i;

   for (int i = 0; i < ne; ++i)
   {
      trial_space.GetElementDofs(i, dofs);
      const int trial_size = dofs.Size();
      g_i.MakeRef(rhs, hat_offsets[i], trial_size);
      x0.GetSubVector(dofs, g_i);  // reverses the sign if dof < 0

      trial_space.AdjustVDofs(dofs);
      for (int j = 0; j < trial_size; ++j)
      {
         int dof = dofs[j];
         if (dof_marker[dof])
         {
            g_i(j) = 0.0;
         }
         else
         {
            dof_marker[dof] = true;
         }
      }

      const int ipiv_offset = ipiv_offsets[i];
      const int matrix_size = ipiv_offsets[i + 1] - ipiv_offset;

      mixed_dofs.GetSubArray(ipiv_offset, matrix_size, dofs);

      LUFactors Minv(data + data_offsets[i], ipiv + ipiv_offset);
      rhs.GetSubVector(dofs, Minv_sub_vec);
      Minv.Solve(Minv_sub_vec.Size(), 1, Minv_sub_vec.GetData());
      rhs.SetSubVector(dofs, Minv_sub_vec);
   }
   Ct->MultTranspose(rhs.GetBlock(0), b_r);
}

void BlockHybridizationSolver::ComputeSolution(Vector &y,
                                               BlockVector &rhs,
                                               const Vector &rhs_r,
                                               Array<int> &block_offsets) const
{
   BlockVector Ct_lambda(block_offsets);
   Ct->Mult(rhs_r, Ct_lambda.GetBlock(0));
   Ct_lambda.GetBlock(1) = 0.0;

   ParMesh &pmesh(*trial_space.GetParMesh());
   const int ne = pmesh.GetNE();
   Array<int> dofs;
   Vector Minv_sub_vec;

   for (int i = 0; i < ne; ++i)
   {
      const int ipiv_offset = ipiv_offsets[i];
      const int matrix_size = ipiv_offsets[i + 1] - ipiv_offset;

      mixed_dofs.GetSubArray(ipiv_offset, matrix_size, dofs);

      LUFactors Minv(data + data_offsets[i], ipiv + ipiv_offset);
      Ct_lambda.GetSubVector(dofs, Minv_sub_vec);
      Minv.Solve(Minv_sub_vec.Size(), 1, Minv_sub_vec.GetData());
      Minv_sub_vec.Neg();
      rhs.AddElementVector(dofs, Minv_sub_vec);
   }

   const SparseMatrix &R = *trial_space.GetRestrictionMatrix();
   Vector x0(R.Width());
   Vector sub_vec;
   for (int i = 0; i < ne; ++i)
   {
      trial_space.GetElementDofs(i, dofs);
      sub_vec.MakeRef(rhs.GetBlock(0), hat_offsets[i], dofs.Size());
      x0.SetSubVector(dofs, sub_vec);
   }
   BlockVector block_y(y, offsets_);
   R.Mult(x0, block_y.GetBlock(0));
   y.SetVector(rhs.GetBlock(1), offsets_[1]);
}

void BlockHybridizationSolver::Mult(const Vector &x, Vector &y) const
{
   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = hat_offsets.Last();
   block_offsets[2] = offsets_[2] - offsets_[1];
   block_offsets.PartialSum();

   BlockVector rhs(block_offsets);

   Vector rhs_r(Ct->Width());
   rhs_r = 0.0;
   ReduceRHS(x, y, rhs, rhs_r);

   Vector rhs_true(pH->Height());
   const Operator &P(*c_fes->GetProlongationMatrix());
   P.MultTranspose(rhs_r, rhs_true);

   Vector lambda_true(rhs_true.Size());
   lambda_true = 0.0;

   solver_.Mult(rhs_true, lambda_true);

   P.Mult(lambda_true, rhs_r);
   ComputeSolution(y, rhs, rhs_r, block_offsets);
}
} // namespace blocksolvers
} // namespace mfem
