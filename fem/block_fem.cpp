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

BlockGridFunction::BlockGridFunction(BlockFiniteElementSpace *f)
   : BlockObject(f->GetNBlocks()), Vector(f->GetVSize()), fes(f)
{
   gf = new GridFunction*[nblocks];
   for (int i=0; i<nblocks; i++)
   {
      gf[i] = new GridFunction(&f->GetBlock(i), &data[f->GetVSizeOffset(i)]);
   }
}

BlockGridFunction::~BlockGridFunction()
{
   for (int i=0; i<nblocks; i++)
   {
      delete gf[i];
   }
   delete [] gf;
}

void
BlockGridFunction::Update()
{
   /*
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
   */
}
/*
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
*/

BlockLinearForm::BlockLinearForm(BlockFiniteElementSpace *f)
   : BlockObject(f->GetNBlocks()), Vector(f->GetVSize()), fes(f)
{
   lf = new LinearForm*[nblocks];
   for (int i=0; i<nblocks; i++)
   {
      lf[i] = new LinearForm(&f->GetBlock(i), &data[f->GetVSizeOffset(i)]);
   }
}

BlockLinearForm::~BlockLinearForm()
{
   for (int i=0; i<nblocks; i++)
   {
      delete lf[i];
   }
   delete [] lf;
}

void
BlockLinearForm::AddDomainIntegrator(int index, LinearFormIntegrator *lfi)
{
   CheckIndex(index);
   lf[index]->AddDomainIntegrator(lfi);
}

void
BlockLinearForm::Update()
{
   /*
    FiniteElementSpace *fes = lfr_->FESpace();

    this->Update(fes);
   */
}

void
BlockLinearForm::Update(BlockFiniteElementSpace *f)
{
   /*
    int vsize = fes->GetVSize();
    SetSize(2 * vsize);

    Vector lfr(&data[0], vsize);
    Vector lfi(&data[vsize], vsize);

    lfr_->Update(fes, lfr, 0);
    lfi_->Update(fes, lfi, 0);
   */
}

void
BlockLinearForm::Assemble()
{
   for (int i=0; i<nblocks; i++)
   {
      lf[i]->Assemble();
   }
}

double
BlockLinearForm::operator()(const BlockGridFunction &gf) const
{
   double v = 0.0;
   for (int i=0; i<nblocks; i++)
   {
      v += (*lf[i])(gf.GetBlock(i));
   }
   return v;
}


BlockBilinearForm::BlockBilinearForm(BlockFiniteElementSpace *f,
                                     bool symmetric)
   : TPBlockObject(f->GetNBlocks()),
     trial_fes(f), test_fes(f), sym(symmetric)
{
   blf = new Matrix*[nblocks];
   // blf = new BilinearForm*[nblocks];
   // mblf = new MixedBilinearForm*[nblocks];
   mixed = new bool[nblocks];
   for (int i=0; i<nblocks; i++)
   {
      blf[i] = NULL;
      mixed[i] = false;
   }
}

BlockBilinearForm::~BlockBilinearForm()
{
   if ( !sym )
   {
      for (int i=0; i<nblocks; i++)
      {
         delete blf[i];
      }
   }
   else
   {
      for (int r=0; r<nrows; r++)
      {
         for (int c=r; c<nrows; c++)
         {
            delete blf[r * nrows + c];
         }
      }
   }
   delete [] blf;
}

void BlockBilinearForm::initBilinearForm(int r, int c)
{
   int index = CheckIndex(r, c);
   if ( blf[index] != NULL ) { return; }

   if ( &trial_fes->GetBlock(r) == &test_fes->GetBlock(c) )
   {
      blf[index] = new BilinearForm(&trial_fes->GetBlock(r));
   }
   else
   {
      blf[index] = new MixedBilinearForm(&trial_fes->GetBlock(r),
					 &test_fes->GetBlock(c));
   }
   if ( sym && r != c )
   {
      int indexT = CheckIndex(c, r);
      blf[indexT] = blf[index];
   }
}

void BlockBilinearForm::AddDomainIntegrator(int r, int c,
                                            BilinearFormIntegrator *bfi)
{
   int index = CheckIndex(r, c);
   if ( blf[index] == NULL )
   {
      initBilinearForm(r, c);
   }
   // if (bfi_real) { blfr_->AddDomainIntegrator(bfi_real); }
   //  if (bfi_imag) { blfi_->AddDomainIntegrator(bfi_imag); }
}

void
BlockBilinearForm::AddBoundaryIntegrator(int r, int c,
                                         BilinearFormIntegrator *bfi)
{
   //   if (bfi_real) { blfr_->AddBoundaryIntegrator(bfi_real); }
   // if (bfi_imag) { blfi_->AddBoundaryIntegrator(bfi_imag); }
}

void
BlockBilinearForm::AddBoundaryIntegrator(int r, int c,
                                         BilinearFormIntegrator *bfi,
                                         Array<int> & bdr_marker)
{
   //  if (bfi_real) { blfr_->AddBoundaryIntegrator(bfi_real, bdr_marker); }
   // if (bfi_imag) { blfi_->AddBoundaryIntegrator(bfi_imag, bdr_marker); }
}

void
BlockBilinearForm::Assemble(int skip_zeros)
{
   // blfr_->Assemble(skip_zeros);
   //  blfi_->Assemble(skip_zeros);
}

void
BlockBilinearForm::Finalize(int skip_zeros)
{
   // blfr_->Finalize(skip_zeros);
   // blfi_->Finalize(skip_zeros);
}
/*
BlockSparseMatrix *
BlockBilinearForm::AssembleCompSpMat()
{
// return new BlockSparseMatrix(&blfr_->SpMat(),
//                              &blfi_->SpMat(),
//                              false, false, conv_);
}
*/
void
BlockBilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list,
                                    Vector &x, Vector &b,
                                    OperatorHandle &A,
                                    Vector &X, Vector &B,
                                    int ci)
{
   /*
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
   */
}

void
BlockBilinearForm::RecoverFEMSolution(const Vector &X, const Vector &b,
                                      Vector &x)
{
   /*
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
   */
}

void
BlockBilinearForm::Update(BlockFiniteElementSpace *nfes)
{
   // if ( blfr_ ) { blfr_->Update(nfes); }
   // if ( blfi_ ) { blfi_->Update(nfes); }
}


#ifdef MFEM_USE_MPI

ParBlockGridFunction::ParBlockGridFunction(ParBlockFiniteElementSpace *pf)
   : BlockObject(pf->GetNBlocks()), Vector(pf->GetVSize()), pfes(pf)
{
   pgf = new ParGridFunction*[nblocks];
   for (int i=0; i<nblocks; i++)
   {
      pgf[i] = new ParGridFunction(&pf->GetBlock(i),
                                   &data[pf->GetVSizeOffset(i)]);
   }
}

void
ParBlockGridFunction::Update()
{
   /*
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
   */
}
/*
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
*/
void
ParBlockGridFunction::Distribute(const Vector *tv)
{
   /*
    ParFiniteElementSpace * pfes = pgfr_->ParFESpace();
    HYPRE_Int size = pfes->GetTrueVSize();

    double * tvd = tv->GetData();
    Vector tvr(tvd, size);
    Vector tvi(&tvd[size], size);

    pgfr_->Distribute(tvr);
    pgfi_->Distribute(tvi);
   */
}

void
ParBlockGridFunction::ParallelProject(Vector &tv) const
{
   /*
    ParFiniteElementSpace * pfes = pgfr_->ParFESpace();
    HYPRE_Int size = pfes->GetTrueVSize();

    double * tvd = tv.GetData();
    Vector tvr(tvd, size);
    Vector tvi(&tvd[size], size);

    pgfr_->ParallelProject(tvr);
    pgfi_->ParallelProject(tvi);
   */
}


ParBlockLinearForm::ParBlockLinearForm(ParBlockFiniteElementSpace *pf)
   : BlockObject(pf->GetNBlocks()), Vector(pf->GetVSize()), pfes(pf)
{
   plf = new ParLinearForm*[nblocks];
   for (int i=0; i<nblocks; i++)
   {
      plf[i] = new ParLinearForm(&pf->GetBlock(i),
                                 &data[pf->GetVSizeOffset(i)]);
   }
}

ParBlockLinearForm::~ParBlockLinearForm()
{
   for (int i=0; i<nblocks; i++)
   {
      delete plf[i];
   }
   delete [] plf;
}

void
ParBlockLinearForm::AddDomainIntegrator(int index,
                                        LinearFormIntegrator *lfi)
{
   CheckIndex(index);
   plf[index]->AddDomainIntegrator(lfi);
}

void
ParBlockLinearForm::Update(ParBlockFiniteElementSpace *pf)
{
   /*
    ParFiniteElementSpace *pfes = (pf!=NULL)?pf:plfr_->ParFESpace();
    int vsize = pfes->GetVSize();
    SetSize(2 * vsize);

    Vector plfr(&data[0], vsize);
    Vector plfi(&data[vsize], vsize);

    plfr_->Update(pfes, plfr, 0);
    plfi_->Update(pfes, plfi, 0);
   */
}

void
ParBlockLinearForm::Assemble()
{
   /*
    plfr_->Assemble();
    plfi_->Assemble();
    if (conv_ == BlockOperator::BLOCK_SYMMETRIC)
    {
       *plfi_ *= -1.0;
    }
   */
}

void
ParBlockLinearForm::ParallelAssemble(Vector &tv)
{
   /*
    HYPRE_Int size = plfr_->ParFESpace()->GetTrueVSize();

    double * tvd = tv.GetData();
    Vector tvr(tvd, size);
    Vector tvi(&tvd[size], size);

    plfr_->ParallelAssemble(tvr);
    plfi_->ParallelAssemble(tvi);
   */
}

BlockVector *
ParBlockLinearForm::ParallelAssemble()
{
   /*
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
   */
}

double
ParBlockLinearForm::operator()(const ParBlockGridFunction &gf) const
{
   double v = 0.0;
   for (int i=0; i<nblocks; i++)
   {
      v += (*plf[i])(gf.GetBlock(i));
   }
   return v;
}


ParBlockBilinearForm::ParBlockBilinearForm(ParBlockFiniteElementSpace *pf,
                                           bool symmetric)
   : TPBlockObject(pf->GetNBlocks()),
     trial_pfes(pf), test_pfes(pf), sym(symmetric)
{
   pblf = new Matrix*[nblocks];
   mixed = new bool[nblocks];
   for (int i=0; i<nblocks; i++)
   {
      pblf[i] = NULL;
      mixed[i] = false;
   }
}

ParBlockBilinearForm::~ParBlockBilinearForm()
{
   if ( !sym )
   {
      for (int i=0; i<nblocks; i++)
      {
         delete pblf[i];
      }
   }
   else
   {
      for (int r=0; r<nrows; r++)
      {
         for (int c=r; c<nrows; c++)
         {
            delete pblf[r * nrows + c];
         }
      }
   }
   delete [] pblf;
}

void ParBlockBilinearForm::initParBilinearForm(int r, int c)
{
   int index = CheckIndex(r, c);
   if ( pblf[index] != NULL ) { return; }

   if ( &trial_pfes->GetBlock(r) == &test_pfes->GetBlock(c) )
   {
      pblf[index] = new ParBilinearForm(&trial_pfes->GetBlock(r));
   }
   else
   {
      pblf[index] = new ParMixedBilinearForm(&trial_pfes->GetBlock(r),
					     &test_pfes->GetBlock(c));
   }
   if ( sym && r != c )
   {
      int indexT = CheckIndex(c, r);
      pblf[indexT] = pblf[index];
   }
}

void ParBlockBilinearForm::AddDomainIntegrator(int r, int c,
                                               BilinearFormIntegrator *bfi)
{
   // if (bfi_real) { pblfr_->AddDomainIntegrator(bfi_real); }
   // if (bfi_imag) { pblfi_->AddDomainIntegrator(bfi_imag); }
}

void
ParBlockBilinearForm::AddBoundaryIntegrator(int r, int c,
                                            BilinearFormIntegrator *bfi)
{
   // if (bfi_real) { pblfr_->AddBoundaryIntegrator(bfi_real); }
   // if (bfi_imag) { pblfi_->AddBoundaryIntegrator(bfi_imag); }
}

void
ParBlockBilinearForm::AddBoundaryIntegrator(int r, int c,
                                            BilinearFormIntegrator *bfi,
                                            Array<int> & bdr_marker)
{
   // if (bfi_real) { pblfr_->AddBoundaryIntegrator(bfi_real, bdr_marker); }
   // if (bfi_imag) { pblfi_->AddBoundaryIntegrator(bfi_imag, bdr_marker); }
}

void
ParBlockBilinearForm::Assemble(int skip_zeros)
{
   // pblfr_->Assemble(skip_zeros);
   //  pblfi_->Assemble(skip_zeros);
}

void
ParBlockBilinearForm::Finalize(int skip_zeros)
{
   // pblfr_->Finalize(skip_zeros);
   // pblfi_->Finalize(skip_zeros);
}

BlockOperator *
ParBlockBilinearForm::ParallelAssemble()
{
   /*
   return new BlockHypreParMatrix(pblfr_->ParallelAssemble(),
                                     pblfi_->ParallelAssemble(),
                                     true, true, conv_);
   */
}

void
ParBlockBilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list,
                                       Vector &x, Vector &b,
                                       OperatorHandle &A,
                                       Vector &X, Vector &B,
                                       int ci)
{
   /*
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
   */
}

void
ParBlockBilinearForm::RecoverFEMSolution(const Vector &X, const Vector &b,
                                         Vector &x)
{
   /*
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
   */
}

void
ParBlockBilinearForm::Update(ParBlockFiniteElementSpace *nfes)
{
   /*
    if ( pblfr_ ) { pblfr_->Update(nfes); }
    if ( pblfi_ ) { pblfi_->Update(nfes); }
   */
}


#endif // MFEM_USE_MPI

}
