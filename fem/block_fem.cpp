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

#include "block_fem.hpp"

using namespace std;

namespace mfem
{

BlockGridFunction::BlockGridFunction(FiniteElementSpace *fes)
   : Vector(2*(fes->GetVSize()))
{
   gfr_ = new GridFunction(fes, &data[0]);
   gfi_ = new GridFunction(fes, &data[fes->GetVSize()]);
}

void
BlockGridFunction::Update()
{
   FiniteElementSpace * fes = gfr_->FESpace();

   int vsize = fes->GetVSize();

   const Operator *T = fes->GetUpdateOperator();
   if (T)
   {
      // Update the individual GridFunction objects.  This will allocate
      // new data arrays for each GridFunction.
      gfr_->Update();
      gfi_->Update();

      // Our data array now contains old data as well as being the wrong size
      // so reallocate it.
      this->SetSize(2 * vsize);

      // Create temporary vectors which point to the new data array
      Vector gf_r(&data[0], vsize);
      Vector gf_i(&data[vsize], vsize);

      // Copy the updated GridFunctions into the new data array
      gf_r = *gfr_;
      gf_i = *gfi_;

      // Replace the individual data arrays with pointers into the new data array
      gfr_->NewDataAndSize(&data[0], vsize);
      gfi_->NewDataAndSize(&data[vsize], vsize);
   }
   else
   {
      // The existing data will not be transferred to the new GridFunctions
      // so delete it a allocate a new array
      this->SetSize(2 * vsize);

      // Point the individual GridFunctions to the new data array
      gfr_->NewDataAndSize(&data[0], vsize);
      gfi_->NewDataAndSize(&data[vsize], vsize);

      // These updates will only set the proper 'sequence' value within
      // the individual GridFunction objects because their sizes are
      // already correct
      gfr_->Update();
      gfi_->Update();
   }
}

void
BlockGridFunction::ProjectCoefficient(Coefficient &real_coeff,
                                        Coefficient &imag_coeff)
{
   gfr_->ProjectCoefficient(real_coeff);
   gfi_->ProjectCoefficient(imag_coeff);
}

void
BlockGridFunction::ProjectCoefficient(VectorCoefficient &real_vcoeff,
                                        VectorCoefficient &imag_vcoeff)
{
   gfr_->ProjectCoefficient(real_vcoeff);
   gfi_->ProjectCoefficient(imag_vcoeff);
}

void
BlockGridFunction::ProjectBdrCoefficient(Coefficient &real_coeff,
                                           Coefficient &imag_coeff,
                                           Array<int> &attr)
{
   gfr_->ProjectBdrCoefficient(real_coeff, attr);
   gfi_->ProjectBdrCoefficient(imag_coeff, attr);
}

void
BlockGridFunction::ProjectBdrCoefficientNormal(VectorCoefficient &real_vcoeff,
                                                 VectorCoefficient &imag_vcoeff,
                                                 Array<int> &attr)
{
   gfr_->ProjectBdrCoefficientNormal(real_vcoeff, attr);
   gfi_->ProjectBdrCoefficientNormal(imag_vcoeff, attr);
}

void
BlockGridFunction::ProjectBdrCoefficientTangent(VectorCoefficient
                                                  &real_vcoeff,
                                                  VectorCoefficient
                                                  &imag_vcoeff,
                                                  Array<int> &attr)
{
   gfr_->ProjectBdrCoefficientTangent(real_vcoeff, attr);
   gfi_->ProjectBdrCoefficientTangent(imag_vcoeff, attr);
}


BlockLinearForm::BlockLinearForm(FiniteElementSpace *f,
                                     BlockOperator::Convention convention)
   : Vector(2*(f->GetVSize())),
     conv_(convention)
{
   lfr_ = new LinearForm(f, &data[0]);
   lfi_ = new LinearForm(f, &data[f->GetVSize()]);
}

BlockLinearForm::~BlockLinearForm()
{
   delete lfr_;
   delete lfi_;
}

void
BlockLinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                                       LinearFormIntegrator *lfi_imag)
{
   if ( lfi_real ) { lfr_->AddDomainIntegrator(lfi_real); }
   if ( lfi_imag ) { lfi_->AddDomainIntegrator(lfi_imag); }
}

void
BlockLinearForm::Update()
{
   FiniteElementSpace *fes = lfr_->FESpace();

   this->Update(fes);
}

void
BlockLinearForm::Update(FiniteElementSpace *fes)
{
   int vsize = fes->GetVSize();
   SetSize(2 * vsize);

   Vector lfr(&data[0], vsize);
   Vector lfi(&data[vsize], vsize);

   lfr_->Update(fes, lfr, 0);
   lfi_->Update(fes, lfi, 0);
}

void
BlockLinearForm::Assemble()
{
   lfr_->Assemble();
   lfi_->Assemble();
   if (conv_ == BlockOperator::BLOCK_SYMMETRIC)
   {
      *lfi_ *= -1.0;
   }
}

complex<double>
BlockLinearForm::operator()(const BlockGridFunction &gf) const
{
   double s = (conv_ == BlockOperator::HERMITIAN)?1.0:-1.0;
   return complex<double>((*lfr_)(gf.real()) - s * (*lfi_)(gf.imag()),
                          (*lfr_)(gf.imag()) + s * (*lfi_)(gf.real()));
}


BlockBilinearForm::BlockBilinearForm(FiniteElementSpace *f,
                                   BlockOperator::Convention convention)
   : conv_(convention),
     blfr_(new BilinearForm(f)),
     blfi_(new BilinearForm(f))
{}

BlockBilinearForm::~BlockBilinearForm()
{
   delete blfr_;
   delete blfi_;
}

void BlockBilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                                           BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { blfr_->AddDomainIntegrator(bfi_real); }
   if (bfi_imag) { blfi_->AddDomainIntegrator(bfi_imag); }
}

void
BlockBilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                                        BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { blfr_->AddBoundaryIntegrator(bfi_real); }
   if (bfi_imag) { blfi_->AddBoundaryIntegrator(bfi_imag); }
}

void
BlockBilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                                        BilinearFormIntegrator *bfi_imag,
                                        Array<int> & bdr_marker)
{
   if (bfi_real) { blfr_->AddBoundaryIntegrator(bfi_real, bdr_marker); }
   if (bfi_imag) { blfi_->AddBoundaryIntegrator(bfi_imag, bdr_marker); }
}

void
BlockBilinearForm::Assemble(int skip_zeros)
{
   blfr_->Assemble(skip_zeros);
   blfi_->Assemble(skip_zeros);
}

void
BlockBilinearForm::Finalize(int skip_zeros)
{
   blfr_->Finalize(skip_zeros);
   blfi_->Finalize(skip_zeros);
}

BlockSparseMatrix *
BlockBilinearForm::AssembleCompSpMat()
{
   return new BlockSparseMatrix(&blfr_->SpMat(),
                                  &blfi_->SpMat(),
                                  false, false, conv_);

}

void
BlockBilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list,
                                   Vector &x, Vector &b,
                                   OperatorHandle &A,
                                   Vector &X, Vector &B,
                                   int ci)
{
   FiniteElementSpace * fes = blfr_->FESpace();

   int vsize  = fes->GetVSize();
   // int tvsize = pfes->GetTrueVSize();

   double s = (conv_ == BlockOperator::HERMITIAN)?1.0:-1.0;

   // Allocate temporary vectors
   Vector b_0(vsize);  b_0 = 0.0;
   // Vector B_0(tvsize); B_0 = 0.0;

   // Extract the real and imaginary parts of the input vectors
   MFEM_ASSERT(x.Size() == 2 * vsize, "Input GridFunction of incorrect size!");
   Vector x_r(x.GetData(), vsize);
   Vector x_i(&(x.GetData())[vsize], vsize);

   MFEM_ASSERT(b.Size() == 2 * vsize, "Input LinearForm of incorrect size!");
   Vector b_r(b.GetData(), vsize);
   Vector b_i(&(b.GetData())[vsize], vsize);
   b_i *= s;
   /*
   X.SetSize(2 * tvsize);
   Vector X_r(X.GetData(), tvsize);
   Vector X_i(&(X.GetData())[tvsize], tvsize);

   B.SetSize(2 * tvsize);
   Vector B_r(B.GetData(), tvsize);
   Vector B_i(&(B.GetData())[tvsize], tvsize);
   */
   SparseMatrix * A_r = new SparseMatrix;
   SparseMatrix * A_i = new SparseMatrix;
   Vector X_0, B_0;

   b_0 = b_r;
   blfr_->FormLinearSystem(ess_tdof_list, x_r, b_r, *A_r, X_0, B_0, ci);

   int tvsize = B_0.Size();
   X.SetSize(2 * tvsize);
   B.SetSize(2 * tvsize);
   Vector X_r(X.GetData(), tvsize);
   Vector X_i(&(X.GetData())[tvsize], tvsize);
   Vector B_r(B.GetData(), tvsize);
   Vector B_i(&(B.GetData())[tvsize], tvsize);
   X_r = X_0; B_r = B_0;

   b_0 = 0.0;
   blfi_->FormLinearSystem(ess_tdof_list, x_i, b_0, *A_i, X_0, B_0, false);
   B_r -= B_0;

   b_0 = b_i;
   blfr_->FormLinearSystem(ess_tdof_list, x_i, b_0, *A_r, X_0, B_0, ci);
   X_i = X_0; B_i = B_0;

   b_0 = 0.0;
   blfi_->FormLinearSystem(ess_tdof_list, x_r, b_0, *A_i, X_0, B_0, false);
   B_i += B_0;

   B_i *= s;
   b_i *= s;

   // A = A_r + i A_i
   A.Clear();
   BlockSparseMatrix * A_sp =
      new BlockSparseMatrix(A_r, A_i, true, true, conv_);
   A.Reset<BlockSparseMatrix>(A_sp, true);
}

void
BlockBilinearForm::RecoverFEMSolution(const Vector &X, const Vector &b,
                                     Vector &x)
{
   FiniteElementSpace * fes = blfr_->FESpace();

   const SparseMatrix *P = fes->GetConformingProlongation();

   int vsize  = fes->GetVSize();
   int tvsize = X.Size() / 2;

   Vector X_r(X.GetData(), tvsize);
   Vector X_i(&(X.GetData())[tvsize], tvsize);

   Vector x_r(x.GetData(), vsize);
   Vector x_i(&(x.GetData())[vsize], vsize);

   if (!P)
   {
      x = X;
   }
   else
   {
      // Apply conforming prolongation
      P->Mult(X_r, x_r);
      P->Mult(X_i, x_i);
   }
}

void
BlockBilinearForm::Update(FiniteElementSpace *nfes)
{
   if ( blfr_ ) { blfr_->Update(nfes); }
   if ( blfi_ ) { blfi_->Update(nfes); }
}


#ifdef MFEM_USE_MPI

ParBlockGridFunction::ParBlockGridFunction(ParFiniteElementSpace *pfes)
   : Vector(2*(pfes->GetVSize()))
{
   pgfr_ = new ParGridFunction(pfes, &data[0]);
   pgfi_ = new ParGridFunction(pfes, &data[pfes->GetVSize()]);
}

void
ParBlockGridFunction::Update()
{
   ParFiniteElementSpace * pfes = pgfr_->ParFESpace();

   int vsize = pfes->GetVSize();

   const Operator *T = pfes->GetUpdateOperator();
   if (T)
   {
      // Update the individual GridFunction objects.  This will allocate
      // new data arrays for each GridFunction.
      pgfr_->Update();
      pgfi_->Update();

      // Our data array now contains old data as well as being the wrong size
      // so reallocate it.
      this->SetSize(2 * vsize);

      // Create temporary vectors which point to the new data array
      Vector gf_r(&data[0], vsize);
      Vector gf_i(&data[vsize], vsize);

      // Copy the updated GridFunctions into the new data array
      gf_r = *pgfr_;
      gf_i = *pgfi_;

      // Replace the individual data arrays with pointers into the new data array
      pgfr_->NewDataAndSize(&data[0], vsize);
      pgfi_->NewDataAndSize(&data[vsize], vsize);
   }
   else
   {
      // The existing data will not be transferred to the new GridFunctions
      // so delete it a allocate a new array
      this->SetSize(2 * vsize);

      // Point the individual GridFunctions to the new data array
      pgfr_->NewDataAndSize(&data[0], vsize);
      pgfi_->NewDataAndSize(&data[vsize], vsize);

      // These updates will only set the proper 'sequence' value within
      // the individual GridFunction objects because their sizes are
      // already correct
      pgfr_->Update();
      pgfi_->Update();
   }
}

void
ParBlockGridFunction::ProjectCoefficient(Coefficient &real_coeff,
                                           Coefficient &imag_coeff)
{
   pgfr_->ProjectCoefficient(real_coeff);
   pgfi_->ProjectCoefficient(imag_coeff);
}

void
ParBlockGridFunction::ProjectCoefficient(VectorCoefficient &real_vcoeff,
                                           VectorCoefficient &imag_vcoeff)
{
   pgfr_->ProjectCoefficient(real_vcoeff);
   pgfi_->ProjectCoefficient(imag_vcoeff);
}

void
ParBlockGridFunction::ProjectBdrCoefficient(Coefficient &real_coeff,
                                              Coefficient &imag_coeff,
                                              Array<int> &attr)
{
   pgfr_->ProjectBdrCoefficient(real_coeff, attr);
   pgfi_->ProjectBdrCoefficient(imag_coeff, attr);
}

void
ParBlockGridFunction::ProjectBdrCoefficientNormal(VectorCoefficient
                                                    &real_vcoeff,
                                                    VectorCoefficient
                                                    &imag_vcoeff,
                                                    Array<int> &attr)
{
   pgfr_->ProjectBdrCoefficientNormal(real_vcoeff, attr);
   pgfi_->ProjectBdrCoefficientNormal(imag_vcoeff, attr);
}

void
ParBlockGridFunction::ProjectBdrCoefficientTangent(VectorCoefficient
                                                     &real_vcoeff,
                                                     VectorCoefficient
                                                     &imag_vcoeff,
                                                     Array<int> &attr)
{
   pgfr_->ProjectBdrCoefficientTangent(real_vcoeff, attr);
   pgfi_->ProjectBdrCoefficientTangent(imag_vcoeff, attr);
}

void
ParBlockGridFunction::Distribute(const Vector *tv)
{
   ParFiniteElementSpace * pfes = pgfr_->ParFESpace();
   HYPRE_Int size = pfes->GetTrueVSize();

   double * tvd = tv->GetData();
   Vector tvr(tvd, size);
   Vector tvi(&tvd[size], size);

   pgfr_->Distribute(tvr);
   pgfi_->Distribute(tvi);
}

void
ParBlockGridFunction::ParallelProject(Vector &tv) const
{
   ParFiniteElementSpace * pfes = pgfr_->ParFESpace();
   HYPRE_Int size = pfes->GetTrueVSize();

   double * tvd = tv.GetData();
   Vector tvr(tvd, size);
   Vector tvi(&tvd[size], size);

   pgfr_->ParallelProject(tvr);
   pgfi_->ParallelProject(tvi);
}


ParBlockLinearForm::ParBlockLinearForm(ParFiniteElementSpace *pfes,
                                           BlockOperator::Convention
                                           convention)
   : Vector(2*(pfes->GetVSize())),
     conv_(convention)
{
   plfr_ = new ParLinearForm(pfes, &data[0]);
   plfi_ = new ParLinearForm(pfes, &data[pfes->GetVSize()]);

   HYPRE_Int * tdof_offsets = pfes->GetTrueDofOffsets();

   int n = (HYPRE_AssumedPartitionCheck()) ? 2 : pfes->GetNRanks();
   tdof_offsets_ = new HYPRE_Int[n+1];

   for (int i=0; i<=n; i++)
   {
      tdof_offsets_[i] = 2 * tdof_offsets[i];
   }
}

ParBlockLinearForm::~ParBlockLinearForm()
{
   delete plfr_;
   delete plfi_;
   delete [] tdof_offsets_;
}

void
ParBlockLinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                                          LinearFormIntegrator *lfi_imag)
{
   if ( lfi_real ) { plfr_->AddDomainIntegrator(lfi_real); }
   if ( lfi_imag ) { plfi_->AddDomainIntegrator(lfi_imag); }
}

void
ParBlockLinearForm::Update(ParFiniteElementSpace *pf)
{
   ParFiniteElementSpace *pfes = (pf!=NULL)?pf:plfr_->ParFESpace();
   int vsize = pfes->GetVSize();
   SetSize(2 * vsize);

   Vector plfr(&data[0], vsize);
   Vector plfi(&data[vsize], vsize);

   plfr_->Update(pfes, plfr, 0);
   plfi_->Update(pfes, plfi, 0);
}

void
ParBlockLinearForm::Assemble()
{
   plfr_->Assemble();
   plfi_->Assemble();
   if (conv_ == BlockOperator::BLOCK_SYMMETRIC)
   {
      *plfi_ *= -1.0;
   }
}

void
ParBlockLinearForm::ParallelAssemble(Vector &tv)
{
   HYPRE_Int size = plfr_->ParFESpace()->GetTrueVSize();

   double * tvd = tv.GetData();
   Vector tvr(tvd, size);
   Vector tvi(&tvd[size], size);

   plfr_->ParallelAssemble(tvr);
   plfi_->ParallelAssemble(tvi);
}

HypreParVector *
ParBlockLinearForm::ParallelAssemble()
{
   const ParFiniteElementSpace * pfes = plfr_->ParFESpace();

   HypreParVector * tv = new HypreParVector(pfes->GetComm(),
                                            2*(pfes->GlobalTrueVSize()),
                                            tdof_offsets_);

   HYPRE_Int size = pfes->GetTrueVSize();

   double * tvd = tv->GetData();
   Vector tvr(tvd, size);
   Vector tvi(&tvd[size], size);

   plfr_->ParallelAssemble(tvr);
   plfi_->ParallelAssemble(tvi);

   return tv;
}

complex<double>
ParBlockLinearForm::operator()(const ParBlockGridFunction &gf) const
{
   double s = (conv_ == BlockOperator::HERMITIAN)?1.0:-1.0;
   return complex<double>((*plfr_)(gf.real()) - s * (*plfi_)(gf.imag()),
                          (*plfr_)(gf.imag()) + s * (*plfi_)(gf.real()));
}


ParBlockBilinearForm::ParBlockBilinearForm(ParFiniteElementSpace *pf,
                                         BlockOperator::Convention
                                         convention)
   : conv_(convention),
     pblfr_(new ParBilinearForm(pf)),
     pblfi_(new ParBilinearForm(pf))
{}

ParBlockBilinearForm::~ParBlockBilinearForm()
{
   delete pblfr_;
   delete pblfi_;
}

void ParBlockBilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                                              BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { pblfr_->AddDomainIntegrator(bfi_real); }
   if (bfi_imag) { pblfi_->AddDomainIntegrator(bfi_imag); }
}

void
ParBlockBilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                                           BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { pblfr_->AddBoundaryIntegrator(bfi_real); }
   if (bfi_imag) { pblfi_->AddBoundaryIntegrator(bfi_imag); }
}

void
ParBlockBilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                                           BilinearFormIntegrator *bfi_imag,
                                           Array<int> & bdr_marker)
{
   if (bfi_real) { pblfr_->AddBoundaryIntegrator(bfi_real, bdr_marker); }
   if (bfi_imag) { pblfi_->AddBoundaryIntegrator(bfi_imag, bdr_marker); }
}

void
ParBlockBilinearForm::Assemble(int skip_zeros)
{
   pblfr_->Assemble(skip_zeros);
   pblfi_->Assemble(skip_zeros);
}

void
ParBlockBilinearForm::Finalize(int skip_zeros)
{
   pblfr_->Finalize(skip_zeros);
   pblfi_->Finalize(skip_zeros);
}

BlockHypreParMatrix *
ParBlockBilinearForm::ParallelAssemble()
{
   return new BlockHypreParMatrix(pblfr_->ParallelAssemble(),
                                    pblfi_->ParallelAssemble(),
                                    true, true, conv_);

}

void
ParBlockBilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list,
                                      Vector &x, Vector &b,
                                      OperatorHandle &A,
                                      Vector &X, Vector &B,
                                      int ci)
{
   ParFiniteElementSpace * pfes = pblfr_->ParFESpace();

   int tvs = pfes->TrueVSize();
   cout << "TrueVSize returns " << tvs << endl;
   cout << "GetVSize returns " << pfes->GetVSize() << endl;

   int vsize = x.Size() / 2;
   // int vsize  = pfes->GetVSize();
   // int tvsize = pfes->GetTrueVSize();

   cout << "x.Size/2 returns " << vsize << endl;

   double s = (conv_ == BlockOperator::HERMITIAN)?1.0:-1.0;

   // Allocate temporary vectors
   Vector b_0(vsize);  b_0 = 0.0;
   // Vector B_0(tvsize); B_0 = 0.0;

   // Extract the real and imaginary parts of the input vectors
   // MFEM_ASSERT(x.Size() == 2 * vsize, "Input GridFunction of incorrect size!");
   Vector x_r(x.GetData(), vsize);
   Vector x_i(&(x.GetData())[vsize], vsize);

   MFEM_ASSERT(b.Size() == 2 * vsize, "Input LinearForm of incorrect size!");
   Vector b_r(b.GetData(), vsize);
   Vector b_i(&(b.GetData())[vsize], vsize);
   b_i *= s;
   /*
   X.SetSize(2 * tvsize);
   Vector X_r(X.GetData(), tvsize);
   Vector X_i(&(X.GetData())[tvsize], tvsize);

   B.SetSize(2 * tvsize);
   Vector B_r(B.GetData(), tvsize);
   Vector B_i(&(B.GetData())[tvsize], tvsize);
   */
   OperatorHandle A_r, A_i;
   Vector X_0, B_0;
   cout << "pblfr fls 1" << endl << flush;
   b_0 = b_r;
   pblfr_->FormLinearSystem(ess_tdof_list, x_r, b_0, A_r, X_0, B_0, ci);

   int tvsize = B_0.Size();
   X.SetSize(2 * tvsize);
   B.SetSize(2 * tvsize);
   Vector X_r(X.GetData(), tvsize);
   Vector X_i(&(X.GetData())[tvsize], tvsize);
   Vector B_r(B.GetData(), tvsize);
   Vector B_i(&(B.GetData())[tvsize], tvsize);
   X_r = X_0; B_r = B_0;
   cout << "pblfi fls 1" << endl << flush;
   b_0 = 0.0;
   pblfi_->FormLinearSystem(ess_tdof_list, x_i, b_0, A_i, X_0, B_0, false);
   B_r -= B_0;
   cout << "pblfr fls 2" << endl << flush;
   b_0 = b_i;
   pblfr_->FormLinearSystem(ess_tdof_list, x_i, b_0, A_r, X_0, B_0, ci);
   X_i = X_0; B_i = B_0;
   cout << "pblfi fls 2" << endl << flush;
   b_0 = 0.0;
   pblfi_->FormLinearSystem(ess_tdof_list, x_r, b_0, A_i, X_0, B_0, false);
   B_i += B_0;

   B_i *= s;
   b_i *= s;

   // A = A_r + i A_i
   A.Clear();
   if ( A_r.Type() == Operator::Hypre_ParCSR &&
        A_i.Type() == Operator::Hypre_ParCSR )
   {
      BlockHypreParMatrix * A_hyp =
         new BlockHypreParMatrix(A_r.As<HypreParMatrix>(),
                                   A_i.As<HypreParMatrix>(),
                                   A_r.OwnsOperator(),
                                   A_i.OwnsOperator(),
                                   conv_);
      A.Reset<BlockHypreParMatrix>(A_hyp, true);
   }
   else
   {
      BlockOperator * A_op =
         new BlockOperator(A_r.As<Operator>(),
                             A_i.As<Operator>(),
                             A_r.OwnsOperator(),
                             A_i.OwnsOperator(),
                             conv_);
      A.Reset<BlockOperator>(A_op, true);
   }
}

void
ParBlockBilinearForm::RecoverFEMSolution(const Vector &X, const Vector &b,
                                        Vector &x)
{
   ParFiniteElementSpace * pfes = pblfr_->ParFESpace();

   const Operator &P = *pfes->GetProlongationMatrix();

   int vsize  = pfes->GetVSize();
   int tvsize = X.Size() / 2;

   Vector X_r(X.GetData(), tvsize);
   Vector X_i(&(X.GetData())[tvsize], tvsize);

   Vector x_r(x.GetData(), vsize);
   Vector x_i(&(x.GetData())[vsize], vsize);

   // Apply conforming prolongation
   P.Mult(X_r, x_r);
   P.Mult(X_i, x_i);
}

void
ParBlockBilinearForm::Update(FiniteElementSpace *nfes)
{
   if ( pblfr_ ) { pblfr_->Update(nfes); }
   if ( pblfi_ ) { pblfi_->Update(nfes); }
}


#endif // MFEM_USE_MPI

}
