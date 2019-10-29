// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "fem.hpp"
#include "../general/forall.hpp"

namespace mfem
{

ElementWiseSmoother::ElementWiseSmoother(const mfem::FiniteElementSpace& fespace)
   :
   mfem::Solver(fespace.GetVSize()),
   fespace_(fespace)
{
}

void ElementWiseSmoother::Mult(const Vector& b, Vector& x) const
{
   Vector local_residual;
   Vector local_correction;
   Vector b_local;
   Array<int> local_dofs;

   if (!iterative_mode)
   {
      x = 0.0;
   }

   for (int e = 0; e < fespace_.GetNE(); ++e)
   {
      fespace_.GetElementDofs(e, local_dofs);
      b.GetSubVector(local_dofs, b_local);
      local_correction.SetSize(b_local.Size());
      ElementResidual(e, b_local, x, local_residual);

      LocalSmoother(e, local_residual, local_correction);

      x.AddElementVector(local_dofs, local_correction);
   }
}

void ElementWiseSmoother::ElementResidual(int e, const Vector& b_local,
                                          const Vector& x,
                                          Vector& r_local) const
{
   r_local.SetSize(b_local.Size());
   r_local = 0.0;

   GetElementFromMatVec(e, x, r_local);
   r_local -= b_local;
   r_local *= -1.0;
}

AdditiveSchwarzLORSmoother::AdditiveSchwarzLORSmoother(const mfem::FiniteElementSpace& fespace,
                                                   const Array<int>& ess_tdof_list,
                                                   mfem::BilinearForm& aform,
                                                   const mfem::Vector& diag,
                                                   mfem::SparseMatrix* LORmat,
                                                   double scale)
   :
   mfem::Solver(fespace.GetVSize()),
   fespace_(fespace),
   ess_tdof_list_(ess_tdof_list),
   aform_(aform),
   diag_(diag),
   LORmat_(LORmat),
   scale_(scale)
{
   const Table& el_dof = fespace_.GetElementToDofTable();
   Table dof_el;
   mfem::Transpose(el_dof, dof_el);
   mfem::Mult(el_dof, dof_el, el_to_el_);

   countingVector.SetSize(fespace_.GetTrueVSize());
   countingVector = 0.0;

   Array<int> local_dofs;

   for (int e = 0; e < fespace_.GetNE(); ++e)
   {
      fespace_.GetElementDofs(e, local_dofs);
      for (int i = 0; i < local_dofs.Size(); ++i)
      {
         countingVector[local_dofs[i]] += 1.0;
      }
   }

   for (int i = 0; i < fespace_.GetTrueVSize(); ++i)
   {
      countingVector[i] = 1.0 / std::sqrt(countingVector[i]);
   }
}

void AdditiveSchwarzLORSmoother::Mult(const Vector& b, Vector& x) const
{
   x = 0.0;
   Array<int> local_dofs;
   Vector b_local;
   Vector x_local;

   for (int e = 0; e < fespace_.GetNE(); ++e)
   {
      fespace_.GetElementDofs(e, local_dofs);
      b.GetSubVector(local_dofs, b_local);
      x_local.SetSize(b_local.Size());

      for (int i = 0; i < local_dofs.Size(); ++i)
      {
         b_local[i] *= countingVector[local_dofs[i]];
      }

      LocalSmoother(e, b_local, x_local);

      for (int i = 0; i < local_dofs.Size(); ++i)
      {
         x_local[i] *= countingVector[local_dofs[i]];
      }

      x.AddElementVector(local_dofs, x_local);
   }

   x *= scale_;

   auto I = ess_tdof_list_.Read();
   auto B = b.Read();
   auto X = x.Write();
   MFEM_FORALL(i, ess_tdof_list_.Size(), X[I[i]] = B[I[i]]; );
}

void AdditiveSchwarzLORSmoother::LocalSmoother(int e, const Vector& in, Vector& out) const
{
   Array<int> local_dofs;
   fespace_.GetElementDofs(e, local_dofs);
   DenseMatrix elmat(local_dofs.Size(), local_dofs.Size());
   LORmat_->GetSubMatrix(local_dofs, local_dofs, elmat);
   DenseMatrixInverse inv(elmat);
   inv.Mult(in, out);
}

AdditiveSchwarzApproxLORSmoother::AdditiveSchwarzApproxLORSmoother(const mfem::FiniteElementSpace& fespace,
                                                   const Array<int>& ess_tdof_list,
                                                   mfem::BilinearForm& aform,
                                                   const mfem::Vector& diag,
                                                   const mfem::Vector& LORdiag,
                                                   mfem::SparseMatrix* LORmat,
                                                   double scale)
   :
   mfem::Solver(fespace.GetVSize()),
   fespace_(fespace),
   ess_tdof_list_(ess_tdof_list),
   aform_(aform),
   diag_(diag),
   LORdiag_(LORdiag),
   LORmat_(LORmat),
   scale_(scale)
{
   const Table& el_dof = fespace_.GetElementToDofTable();
   Table dof_el;
   mfem::Transpose(el_dof, dof_el);
   mfem::Mult(el_dof, dof_el, el_to_el_);

   countingVector.SetSize(fespace_.GetTrueVSize());
   countingVector = 0.0;

   Array<int> local_dofs;
   Array<int> local_dofs_lex;

   for (int e = 0; e < fespace_.GetNE(); ++e)
   {
      fespace_.GetElementDofs(e, local_dofs);
      for (int i = 0; i < local_dofs.Size(); ++i)
      {
         countingVector[local_dofs[i]] += 1.0;
      }

      Array<int> neighbors;
      el_to_el_.GetRow(e, neighbors);
   }

   for (int i = 0; i < fespace_.GetTrueVSize(); ++i)
   {
      countingVector[i] = 1.0 / std::sqrt(countingVector[i]);
   }

   A1.resize(fespace_.GetNE());
   A2.resize(fespace_.GetNE());
   B1.resize(fespace_.GetNE());
   B2.resize(fespace_.GetNE());
   invA2.resize(fespace_.GetNE());
   invB1.resize(fespace_.GetNE());
   schurA1.resize(fespace_.GetNE());
   schurB2.resize(fespace_.GetNE());
   syl.resize(fespace_.GetNE());
   inv.resize(fespace_.GetNE());

   for (int e = 0; e < fespace_.GetNE(); ++e)
   {
      const FiniteElement& el = *fespace_.GetFE(e);
      const TensorBasisElement* ltel =
         dynamic_cast<const TensorBasisElement*>(&el);
      MFEM_VERIFY(ltel, "FE space must be tensor product space");
      lexdofmap = ltel->GetDofMap();

      fespace_.GetElementDofs(e, local_dofs);
      elmat = DenseMatrix(local_dofs.Size(), local_dofs.Size());

      local_dofs_lex.SetSize(local_dofs.Size());
      for (int i = 0; i < local_dofs.Size(); ++i)
      {
         local_dofs_lex[i] = local_dofs[lexdofmap[i]];
      }

      LORmat_->GetSubMatrix(local_dofs_lex, local_dofs_lex, elmat);
      inv[e].SetOperator(elmat);

      int N1D = std::sqrt(local_dofs_lex.Size());
      DenseMatrix Atilde = elmat;

      for (int i = 0; i < N1D; ++i)
      {
         for (int j = 0; j < N1D; ++j)
         {
            DenseMatrix blockMat(N1D, N1D);
            for (int k = 0; k < N1D; ++k)
            {
               for (int l = 0; l < N1D; ++l)
               {
                  blockMat(k,l) = elmat(k + i*N1D, l + j*N1D);
               }
            }

            Atilde.SetRow(j + i * N1D, Vector(blockMat.Data(), N1D * N1D));
         }
      }

      A1[e].SetSize(N1D, N1D);
      A2[e].SetSize(N1D, N1D);
      B1[e].SetSize(N1D, N1D);
      B2[e].SetSize(N1D, N1D);

      DenseMatrixSVD svd(Atilde);
      svd.Eval(Atilde);

      // for (int i = 0; i < local_dofs.Size(); ++i)
      // {
      //    std::cout << svd.Singularvalue(i) << " ";
      // }
      // std::cout << std::endl;

      Vector A1vec(A1[e].Data(), N1D * N1D);
      svd.GetU().GetColumn(0, A1vec);
      A1[e] *= std::sqrt(svd.Singularvalue(0));

      Vector A2vec(A2[e].Data(), N1D * N1D);
      svd.GetU().GetColumn(1, A2vec);
      A2[e] *= std::sqrt(svd.Singularvalue(1));

      Vector B1vec(B1[e].Data(), N1D * N1D);
      svd.GetV_T().GetRow(0, B1vec);
      B1[e] *= std::sqrt(svd.Singularvalue(0));

      Vector B2vec(B2[e].Data(), N1D * N1D);
      svd.GetV_T().GetRow(1, B2vec);
      B2[e] *= std::sqrt(svd.Singularvalue(1));


      ////---- Test if "tensorization" worked

      // Vector xTest(local_dofs.Size());
      // Vector yTest(local_dofs.Size());
      // Vector yTest2(local_dofs.Size());
      // Vector yTest3(local_dofs.Size());

      // xTest.Randomize();

      // tic_toc.Clear();
      // tic_toc.Start();
      // elmat.Mult(xTest, yTest);
      // tic_toc.Stop();
      // std::cout << "elmat mult: " << tic_toc.RealTime() << std::endl;

      // tic_toc.Clear();
      // tic_toc.Start();
      // TensorProductMult2D(A1, B1, xTest, yTest2);
      // TensorProductMult2D(A2, B2, xTest, yTest3);
      // yTest2 += yTest3;
      // tic_toc.Stop();
      // std::cout << "tp mult:    " << tic_toc.RealTime() << std::endl;

      // yTest -= yTest2;

      // std::cout << "norm = " << yTest.Norml2() << std::endl;

      ////---- Test end

      invA2[e].SetOperator(A2[e]);
      invB1[e].SetOperator(B1[e]);

      invA2[e].Mult(A1[e]); // A1 = inv(A2) * A1
      invB1[e].Mult(B2[e]); // B2 = inv(B1) * B2

      schurA1[e] = new DenseMatrixSchurDecomposition(A1[e]);
      schurB2[e] = new DenseMatrixSchurDecomposition(B2[e]);

      // ////---- Test if Schur decomposition worked
      // DenseMatrix testMat(A1.Height(), A1.Width());
      // mfem::Mult(schurA1->GetT(), schurA1->GetQ_T(), testMat);
      // DenseMatrix tmp(testMat);
      // mfem::Mult(schurA1->GetQ(), tmp, testMat);

      // std::cout << "A1 = " << A1.MaxMaxNorm() << std::endl;
      // std::cout << "testMat = " << testMat.MaxMaxNorm() << std::endl;

      // testMat -= A1;
      // std::cout << "diff = " << testMat.MaxMaxNorm() << std::endl;
      // ////---- Test end

      syl[e] = new DenseMatrixSylvesterSolver(schurB2[e]->GetT(), schurA1[e]->GetT(), false, true);
   }
}

void AdditiveSchwarzApproxLORSmoother::Mult(const Vector& b, Vector& x) const
{
   x = 0.0;
   Array<int> local_dofs;
   Vector b_local, b_local_lex;
   Vector x_local, x_local_lex;

   for (int e = 0; e < fespace_.GetNE(); ++e)
   {
      fespace_.GetElementDofs(e, local_dofs);
      b.GetSubVector(local_dofs, b_local);
      x_local.SetSize(b_local.Size());
      b_local_lex.SetSize(b_local.Size());
      x_local_lex.SetSize(b_local.Size());

      double minval = 1e300;
      for (int i = 0; i < local_dofs.Size(); ++i)
      {
         minval = std::min(minval, diag_[local_dofs[i]]);
      }

      for (int i = 0; i < local_dofs.Size(); ++i)
      {
         b_local[i] *= countingVector[local_dofs[i]];
         // b_local[i] *= diag_[local_dofs[i]];
         // b_local[i] *= minval;
         // b_local[i] *= sqrt(elmat(i,i) / LORdiag_[local_dofs[i]]);
      }

      for (int i = 0; i < lexdofmap.Size(); ++i)
      {
         b_local_lex[i] = b_local[lexdofmap[i]];
      }

      LocalSmoother(e, b_local_lex, x_local_lex);

      for (int i = 0; i < lexdofmap.Size(); ++i)
      {
         x_local[lexdofmap[i]] = x_local_lex[i];
      }

      for (int i = 0; i < local_dofs.Size(); ++i)
      {
         // x_local[i] *= sqrt(elmat(i,i) / LORdiag_[local_dofs[i]]);
         // x_local[i] *= minval;
         // x_local[i] *= diag_[local_dofs[i]];
         x_local[i] *= countingVector[local_dofs[i]];
      }

      x.AddElementVector(local_dofs, x_local);
   }

   x *= scale_;

   auto I = ess_tdof_list_.Read();
   auto B = b.Read();
   auto X = x.Write();
   MFEM_FORALL(i, ess_tdof_list_.Size(), X[I[i]] = B[I[i]]; );
}

void AdditiveSchwarzApproxLORSmoother::LocalSmoother(int e, const Vector& in, Vector& out) const
{
   TensorProductMult2D(invA2[e], invB1[e], in, out);
   TensorProductMult2D(schurA1[e]->GetQ_T(), schurB2[e]->GetQ_T(), out, out);
   syl[e]->Mult(out, out);
   TensorProductMult2D(schurA1[e]->GetQ(), schurB2[e]->GetQ(), out, out);
   
   // inv[e].Mult(in, out);
}

void AdditiveSchwarzApproxLORSmoother::TensorProductMult2D(const DenseMatrix& A, const DenseMatrix& B, const Vector& in, Vector& out) const
{
   const int N1D = std::sqrt(in.Size());
   DenseMatrix IN(in.GetData(), N1D, N1D);
   DenseMatrix OUT(out.GetData(), N1D, N1D);
   DenseMatrix tmp(N1D, N1D);

   mfem::MultABt(IN, A, tmp);
   mfem::Mult(B, tmp, OUT);
}

// Computes out = (A (x) B) * in
void AdditiveSchwarzApproxLORSmoother::TensorProductMult2D(const Operator& A, const Operator& B, const Vector& in, Vector& out) const
{
   const int N1D = std::sqrt(in.Size());
   Vector x1D(N1D);
   Vector y1D(N1D);

   auto in_ = Reshape(in.Read(), N1D, N1D);
   auto out_ = Reshape(out.ReadWrite(), N1D, N1D);

   for (int i = 0; i < N1D; ++i)
   {
      for (int j = 0; j < N1D; ++j)
      {
         x1D(j) = in_(j,i);
      }
      
      B.Mult(x1D, y1D);

      for (int j = 0; j < N1D; ++j)
      {
         out_(j,i) = y1D(j);
      }
   }

   for (int i = 0; i < N1D; ++i)
   {
      for (int j = 0; j < N1D; ++j)
      {
         x1D(j) = out_(i,j);
      }
      
      A.Mult(x1D, y1D);

      for (int j = 0; j < N1D; ++j)
      {
         out_(i,j) = y1D(j);
      }
   }
}





ElementWiseJacobi::ElementWiseJacobi(const mfem::FiniteElementSpace& fespace,
                                     mfem::BilinearForm& aform,
                                     const mfem::Vector& global_diag,
                                     double scale)
   :
   ElementWiseSmoother(fespace),
   aform_(aform),
   global_diag_(global_diag),
   scale_(scale)
{
   const Table& el_dof = fespace_.GetElementToDofTable();
   Table dof_el;
   mfem::Transpose(el_dof, dof_el);
   mfem::Mult(el_dof, dof_el, el_to_el_);
}

// x is *global*, y is local to the element
void ElementWiseJacobi::GetElementFromMatVec(int element, const mfem::Vector& x,
                                             mfem::Vector& y_element) const
{
   Array<int> neighbors;
   el_to_el_.GetRow(element, neighbors);
   Array<int> row_dofs;
   Array<int> col_dofs;
   fespace_.GetElementDofs(element, row_dofs);
   Vector x_local;
   Vector z_element;
   DenseMatrix elmat;

   y_element.SetSize(row_dofs.Size());
   z_element.SetSize(row_dofs.Size());
   y_element = 0.0;

   for (int i = 0; i < neighbors.Size(); ++i)
   {
      int ne = neighbors[i];
      fespace_.GetElementDofs(ne, col_dofs);

      z_element = 0.0;
      aform_.ElementMatrixMult(ne, x, z_element);

      // okay, this next section seems very inefficient
      // (could probably store and precompute some kind of map?)
      for (int j = 0; j < row_dofs.Size(); ++j)
      {
         const int rd = row_dofs[j];
         for (int k = 0; k < col_dofs.Size(); ++k)
         {
            const int cd = col_dofs[k];
            if (rd == cd)
            {
               y_element[j] += z_element[k];
               break;
            }
         }
      }
   }
}

void ElementWiseJacobi::LocalSmoother(int e, const Vector& in, Vector& out) const
{
   DenseMatrix elmat;
   Array<int> local_dofs;
   fespace_.GetElementDofs(e, local_dofs);
   for (int i = 0; i < in.Size(); ++i)
   {
      out[i] = (scale_ / global_diag_(local_dofs[i])) * in[i];
   }
}

}
