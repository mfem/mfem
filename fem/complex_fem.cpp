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

#include "complex_fem.hpp"

using namespace std;

namespace mfem
{
  
ComplexGridFunction::ComplexGridFunction(FiniteElementSpace *f)
   : Vector(2*(f->GetVSize()))
{
   gfr_ = new GridFunction(f, &data[0]);
   gfi_ = new GridFunction(f, &data[f->GetVSize()]);
}

void
ComplexGridFunction::ProjectCoefficient(Coefficient &real_coeff,
					Coefficient &imag_coeff)
{
   gfr_->ProjectCoefficient(real_coeff);
   gfi_->ProjectCoefficient(imag_coeff);
}

void
ComplexGridFunction::ProjectCoefficient(VectorCoefficient &real_vcoeff,
					VectorCoefficient &imag_vcoeff)
{
   gfr_->ProjectCoefficient(real_vcoeff);
   gfi_->ProjectCoefficient(imag_vcoeff);
}


ComplexLinearForm::ComplexLinearForm(FiniteElementSpace *f,
				     ComplexOperator::Convention convention)
   : Vector(2*(f->GetVSize())),
     conv_(convention)
{
   lfr_ = new LinearForm(f, &data[0]);
   lfi_ = new LinearForm(f, &data[f->GetVSize()]);
}

ComplexLinearForm::~ComplexLinearForm()
{
   delete lfr_;
   delete lfi_;
}

void
ComplexLinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi_real,
				       LinearFormIntegrator *lfi_imag)
{
   if ( lfi_real ) { lfr_->AddDomainIntegrator(lfi_real); }
   if ( lfi_imag ) { lfi_->AddDomainIntegrator(lfi_imag); }
}

void
ComplexLinearForm::Update(FiniteElementSpace *f)
{
   lfr_->Update(f);
   lfi_->Update(f);
}

void
ComplexLinearForm::Assemble()
{
   lfr_->Assemble();
   lfi_->Assemble();
   if (conv_ == ComplexOperator::BLOCK_SYMMETRIC)
   {
      *lfi_ *= -1.0;
   }
}

complex<double>
ComplexLinearForm::operator()(const ComplexGridFunction &gf) const
{
   double s = (conv_ == ComplexOperator::HERMITIAN)?1.0:-1.0;
   return complex<double>((*lfr_)(gf.real()) - s * (*lfi_)(gf.imag()),
                          (*lfr_)(gf.imag()) + s * (*lfi_)(gf.real()));
}


SesquilinearForm::SesquilinearForm(FiniteElementSpace *f,
				   ComplexOperator::Convention convention)
   : conv_(convention),
     blfr_(new BilinearForm(f)),
     blfi_(new BilinearForm(f))
{}

SesquilinearForm::~SesquilinearForm()
{
   delete blfr_;
   delete blfi_;
}

void SesquilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
					   BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { blfr_->AddDomainIntegrator(bfi_real); }
   if (bfi_imag) { blfi_->AddDomainIntegrator(bfi_imag); }
}

void
SesquilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
					BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { blfr_->AddBoundaryIntegrator(bfi_real); }
   if (bfi_imag) { blfi_->AddBoundaryIntegrator(bfi_imag); }
}

void
SesquilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
					BilinearFormIntegrator *bfi_imag,
					Array<int> & bdr_marker)
{
   if (bfi_real) { blfr_->AddBoundaryIntegrator(bfi_real, bdr_marker); }
   if (bfi_imag) { blfi_->AddBoundaryIntegrator(bfi_imag, bdr_marker); }
}

void
SesquilinearForm::Assemble(int skip_zeros)
{
   blfr_->Assemble(skip_zeros);
   blfi_->Assemble(skip_zeros);
}

void
SesquilinearForm::Finalize(int skip_zeros)
{
   blfr_->Finalize(skip_zeros);
   blfi_->Finalize(skip_zeros);
}

ComplexSparseMatrix *
SesquilinearForm::AssembleCompSpMat()
{
   return new ComplexSparseMatrix(&blfr_->SpMat(),
				  &blfi_->SpMat(),
				  false, false, conv_);

}

void
SesquilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list,
				   Vector &x, Vector &b,
				   OperatorHandle &A,
				   Vector &X, Vector &B,
				   int ci)
{
   FiniteElementSpace * fes = blfr_->FESpace();

   int vsize  = fes->GetVSize();
   // int tvsize = pfes->GetTrueVSize();

   double s = (conv_ == ComplexOperator::HERMITIAN)?1.0:-1.0;

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
   ComplexSparseMatrix * A_sp =
     new ComplexSparseMatrix(A_r, A_i, true, true, conv_);
   A.Reset<ComplexSparseMatrix>(A_sp, true);
}

void
SesquilinearForm::RecoverFEMSolution(const Vector &X, const Vector &b,
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
SesquilinearForm::Update(FiniteElementSpace *nfes)
{
   if ( blfr_ ) { blfr_->Update(nfes); }
   if ( blfi_ ) { blfi_->Update(nfes); }
}


#ifdef MFEM_USE_MPI

ParComplexGridFunction::ParComplexGridFunction(ParFiniteElementSpace *f)
   : Vector(2*(f->GetVSize()))
{
   pgfr_ = new ParGridFunction(f, &data[0]);
   pgfi_ = new ParGridFunction(f, &data[f->GetVSize()]);
}

void
ParComplexGridFunction::ProjectCoefficient(Coefficient &real_coeff,
                                           Coefficient &imag_coeff)
{
   pgfr_->ProjectCoefficient(real_coeff);
   pgfi_->ProjectCoefficient(imag_coeff);
}

void
ParComplexGridFunction::ProjectCoefficient(VectorCoefficient &real_vcoeff,
                                           VectorCoefficient &imag_vcoeff)
{
   pgfr_->ProjectCoefficient(real_vcoeff);
   pgfi_->ProjectCoefficient(imag_vcoeff);
}

void
ParComplexGridFunction::Distribute(const Vector *tv)
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
ParComplexGridFunction::ParallelProject(Vector &tv) const
{
   ParFiniteElementSpace * pfes = pgfr_->ParFESpace();
   HYPRE_Int size = pfes->GetTrueVSize();

   double * tvd = tv.GetData();
   Vector tvr(tvd, size);
   Vector tvi(&tvd[size], size);

   pgfr_->ParallelProject(tvr);
   pgfi_->ParallelProject(tvi);
}


ParComplexLinearForm::ParComplexLinearForm(ParFiniteElementSpace *pf,
                                           ComplexOperator::Convention
                                           convention)
   : Vector(2*(pf->GetVSize())),
     conv_(convention)
{
   plfr_ = new ParLinearForm(pf, &data[0]);
   plfi_ = new ParLinearForm(pf, &data[pf->GetVSize()]);

   HYPRE_Int * tdof_offsets = pf->GetTrueDofOffsets();

   int n = (HYPRE_AssumedPartitionCheck()) ? 2 : pf->GetNRanks();
   tdof_offsets_ = new HYPRE_Int[n+1];

   for (int i=0; i<=n; i++)
   {
      tdof_offsets_[i] = 2 * tdof_offsets[i];
   }
}

ParComplexLinearForm::~ParComplexLinearForm()
{
   delete plfr_;
   delete plfi_;
   delete [] tdof_offsets_;
}

void
ParComplexLinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                                          LinearFormIntegrator *lfi_imag)
{
   if ( lfi_real ) { plfr_->AddDomainIntegrator(lfi_real); }
   if ( lfi_imag ) { plfi_->AddDomainIntegrator(lfi_imag); }
}

void
ParComplexLinearForm::Update(ParFiniteElementSpace *pf)
{
   plfr_->Update(pf);
   plfi_->Update(pf);
}

void
ParComplexLinearForm::Assemble()
{
   plfr_->Assemble();
   plfi_->Assemble();
   if (conv_ == ComplexOperator::BLOCK_SYMMETRIC)
   {
      *plfi_ *= -1.0;
   }
}

void
ParComplexLinearForm::ParallelAssemble(Vector &tv)
{
   HYPRE_Int size = plfr_->ParFESpace()->GetTrueVSize();

   double * tvd = tv.GetData();
   Vector tvr(tvd, size);
   Vector tvi(&tvd[size], size);

   plfr_->ParallelAssemble(tvr);
   plfi_->ParallelAssemble(tvi);
}

HypreParVector *
ParComplexLinearForm::ParallelAssemble()
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
ParComplexLinearForm::operator()(const ParComplexGridFunction &gf) const
{
   double s = (conv_ == ComplexOperator::HERMITIAN)?1.0:-1.0;
   return complex<double>((*plfr_)(gf.real()) - s * (*plfi_)(gf.imag()),
                          (*plfr_)(gf.imag()) + s * (*plfi_)(gf.real()));
}


ParSesquilinearForm::ParSesquilinearForm(ParFiniteElementSpace *pf,
                                         ComplexOperator::Convention
                                         convention)
   : conv_(convention),
     pblfr_(new ParBilinearForm(pf)),
     pblfi_(new ParBilinearForm(pf))
{}

ParSesquilinearForm::~ParSesquilinearForm()
{
   delete pblfr_;
   delete pblfi_;
}

void ParSesquilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                                              BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { pblfr_->AddDomainIntegrator(bfi_real); }
   if (bfi_imag) { pblfi_->AddDomainIntegrator(bfi_imag); }
}

void
ParSesquilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                                           BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { pblfr_->AddBoundaryIntegrator(bfi_real); }
   if (bfi_imag) { pblfi_->AddBoundaryIntegrator(bfi_imag); }
}

void
ParSesquilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                                           BilinearFormIntegrator *bfi_imag,
                                           Array<int> & bdr_marker)
{
   if (bfi_real) { pblfr_->AddBoundaryIntegrator(bfi_real, bdr_marker); }
   if (bfi_imag) { pblfi_->AddBoundaryIntegrator(bfi_imag, bdr_marker); }
}

void
ParSesquilinearForm::Assemble(int skip_zeros)
{
   pblfr_->Assemble(skip_zeros);
   pblfi_->Assemble(skip_zeros);
}

void
ParSesquilinearForm::Finalize(int skip_zeros)
{
   pblfr_->Finalize(skip_zeros);
   pblfi_->Finalize(skip_zeros);
}

ComplexHypreParMatrix *
ParSesquilinearForm::ParallelAssemble()
{
   return new ComplexHypreParMatrix(pblfr_->ParallelAssemble(),
                                    pblfi_->ParallelAssemble(),
                                    true, true, conv_);

}

void
ParSesquilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list,
                                      Vector &x, Vector &b,
                                      OperatorHandle &A,
                                      Vector &X, Vector &B,
                                      int ci)
{
   ParFiniteElementSpace * pfes = pblfr_->ParFESpace();

   int vsize  = pfes->GetVSize();
   // int tvsize = pfes->GetTrueVSize();

   double s = (conv_ == ComplexOperator::HERMITIAN)?1.0:-1.0;

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
   OperatorHandle A_r, A_i;
   Vector X_0, B_0;

   b_0 = b_r;
   pblfr_->FormLinearSystem(ess_tdof_list, x_r, b_r, A_r, X_0, B_0, ci);

   int tvsize = B_0.Size();
   X.SetSize(2 * tvsize);
   B.SetSize(2 * tvsize);
   Vector X_r(X.GetData(), tvsize);
   Vector X_i(&(X.GetData())[tvsize], tvsize);
   Vector B_r(B.GetData(), tvsize);
   Vector B_i(&(B.GetData())[tvsize], tvsize);
   X_r = X_0; B_r = B_0;

   b_0 = 0.0;
   pblfi_->FormLinearSystem(ess_tdof_list, x_i, b_0, A_i, X_0, B_0, false);
   B_r -= B_0;

   b_0 = b_i;
   pblfr_->FormLinearSystem(ess_tdof_list, x_i, b_0, A_r, X_0, B_0, ci);
   X_i = X_0; B_i = B_0;

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
      ComplexHypreParMatrix * A_hyp =
         new ComplexHypreParMatrix(A_r.As<HypreParMatrix>(),
                                   A_i.As<HypreParMatrix>(),
                                   A_r.OwnsOperator(),
                                   A_i.OwnsOperator(),
                                   conv_);
      A.Reset<ComplexHypreParMatrix>(A_hyp, true);
   }
   else
   {
      ComplexOperator * A_op =
         new ComplexOperator(A_r.As<Operator>(),
                             A_i.As<Operator>(),
                             A_r.OwnsOperator(),
                             A_i.OwnsOperator(),
                             conv_);
      A.Reset<ComplexOperator>(A_op, true);
   }
}

void
ParSesquilinearForm::RecoverFEMSolution(const Vector &X, const Vector &b,
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
ParSesquilinearForm::Update(FiniteElementSpace *nfes)
{
   if ( pblfr_ ) { pblfr_->Update(nfes); }
   if ( pblfi_ ) { pblfi_->Update(nfes); }
}


#endif // MFEM_USE_MPI

}
