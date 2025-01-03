// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "complex_fem.hpp"
#include "../general/forall.hpp"

using namespace std;

namespace mfem
{

ComplexGridFunction::ComplexGridFunction(FiniteElementSpace *fes)
   : Vector(2*(fes->GetVSize()))
{
   UseDevice(true);
   this->Vector::operator=(0.0);

   gfr = new GridFunction();
   gfr->MakeRef(fes, *this, 0);

   gfi = new GridFunction();
   gfi->MakeRef(fes, *this, fes->GetVSize());
}

void
ComplexGridFunction::Update()
{
   FiniteElementSpace *fes = gfr->FESpace();
   const int vsize = fes->GetVSize();

   const Operator *T = fes->GetUpdateOperator();
   if (T)
   {
      // Update the individual GridFunction objects. This will allocate new data
      // arrays for each GridFunction.
      gfr->Update();
      gfi->Update();

      // Our data array now contains old data as well as being the wrong size so
      // reallocate it.
      UseDevice(true);
      this->SetSize(2 * vsize);
      this->Vector::operator=(0.0);

      // Create temporary vectors which point to the new data array
      Vector gf_r; gf_r.MakeRef(*this, 0, vsize);
      Vector gf_i; gf_i.MakeRef(*this, vsize, vsize);

      // Copy the updated GridFunctions into the new data array
      gf_r = *gfr;
      gf_i = *gfi;
      gf_r.SyncAliasMemory(*this);
      gf_i.SyncAliasMemory(*this);

      // Replace the individual data arrays with pointers into the new data
      // array
      gfr->MakeRef(*this, 0, vsize);
      gfi->MakeRef(*this, vsize, vsize);
   }
   else
   {
      // The existing data will not be transferred to the new GridFunctions so
      // delete it and allocate a new array
      UseDevice(true);
      this->SetSize(2 * vsize);
      this->Vector::operator=(0.0);

      // Point the individual GridFunctions to the new data array
      gfr->MakeRef(*this, 0, vsize);
      gfi->MakeRef(*this, vsize, vsize);

      // These updates will only set the proper 'sequence' value within the
      // individual GridFunction objects because their sizes are already correct
      gfr->Update();
      gfi->Update();
   }
}

void
ComplexGridFunction::ProjectCoefficient(Coefficient &real_coeff,
                                        Coefficient &imag_coeff)
{
   gfr->SyncMemory(*this);
   gfi->SyncMemory(*this);
   gfr->ProjectCoefficient(real_coeff);
   gfi->ProjectCoefficient(imag_coeff);
   gfr->SyncAliasMemory(*this);
   gfi->SyncAliasMemory(*this);
}

void
ComplexGridFunction::ProjectCoefficient(VectorCoefficient &real_vcoeff,
                                        VectorCoefficient &imag_vcoeff)
{
   gfr->SyncMemory(*this);
   gfi->SyncMemory(*this);
   gfr->ProjectCoefficient(real_vcoeff);
   gfi->ProjectCoefficient(imag_vcoeff);
   gfr->SyncAliasMemory(*this);
   gfi->SyncAliasMemory(*this);
}

void
ComplexGridFunction::ProjectBdrCoefficient(Coefficient &real_coeff,
                                           Coefficient &imag_coeff,
                                           Array<int> &attr)
{
   gfr->SyncMemory(*this);
   gfi->SyncMemory(*this);
   gfr->ProjectBdrCoefficient(real_coeff, attr);
   gfi->ProjectBdrCoefficient(imag_coeff, attr);
   gfr->SyncAliasMemory(*this);
   gfi->SyncAliasMemory(*this);
}

void
ComplexGridFunction::ProjectBdrCoefficientNormal(VectorCoefficient &real_vcoeff,
                                                 VectorCoefficient &imag_vcoeff,
                                                 Array<int> &attr)
{
   gfr->SyncMemory(*this);
   gfi->SyncMemory(*this);
   gfr->ProjectBdrCoefficientNormal(real_vcoeff, attr);
   gfi->ProjectBdrCoefficientNormal(imag_vcoeff, attr);
   gfr->SyncAliasMemory(*this);
   gfi->SyncAliasMemory(*this);
}

void
ComplexGridFunction::ProjectBdrCoefficientTangent(VectorCoefficient
                                                  &real_vcoeff,
                                                  VectorCoefficient
                                                  &imag_vcoeff,
                                                  Array<int> &attr)
{
   gfr->SyncMemory(*this);
   gfi->SyncMemory(*this);
   gfr->ProjectBdrCoefficientTangent(real_vcoeff, attr);
   gfi->ProjectBdrCoefficientTangent(imag_vcoeff, attr);
   gfr->SyncAliasMemory(*this);
   gfi->SyncAliasMemory(*this);
}


ComplexLinearForm::ComplexLinearForm(FiniteElementSpace *fes,
                                     ComplexOperator::Convention convention)
   : Vector(2*(fes->GetVSize())),
     conv(convention)
{
   UseDevice(true);
   this->Vector::operator=(0.0);

   lfr = new LinearForm();
   lfr->MakeRef(fes, *this, 0);

   lfi = new LinearForm();
   lfi->MakeRef(fes, *this, fes->GetVSize());
}

ComplexLinearForm::ComplexLinearForm(FiniteElementSpace *fes,
                                     LinearForm *lf_r, LinearForm *lf_i,
                                     ComplexOperator::Convention convention)
   : Vector(2*(fes->GetVSize())),
     conv(convention)
{
   UseDevice(true);
   this->Vector::operator=(0.0);

   lfr = new LinearForm(fes, lf_r);
   lfi = new LinearForm(fes, lf_i);

   lfr->MakeRef(fes, *this, 0);
   lfi->MakeRef(fes, *this, fes->GetVSize());
}

ComplexLinearForm::~ComplexLinearForm()
{
   delete lfr;
   delete lfi;
}

void
ComplexLinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                                       LinearFormIntegrator *lfi_imag)
{
   if ( lfi_real ) { lfr->AddDomainIntegrator(lfi_real); }
   if ( lfi_imag ) { lfi->AddDomainIntegrator(lfi_imag); }
}

void
ComplexLinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                                       LinearFormIntegrator *lfi_imag,
                                       Array<int> &elem_attr_marker)
{
   if ( lfi_real ) { lfr->AddDomainIntegrator(lfi_real, elem_attr_marker); }
   if ( lfi_imag ) { lfi->AddDomainIntegrator(lfi_imag, elem_attr_marker); }
}

void
ComplexLinearForm::AddBoundaryIntegrator(LinearFormIntegrator *lfi_real,
                                         LinearFormIntegrator *lfi_imag)
{
   if ( lfi_real ) { lfr->AddBoundaryIntegrator(lfi_real); }
   if ( lfi_imag ) { lfi->AddBoundaryIntegrator(lfi_imag); }
}

void
ComplexLinearForm::AddBoundaryIntegrator(LinearFormIntegrator *lfi_real,
                                         LinearFormIntegrator *lfi_imag,
                                         Array<int> &bdr_attr_marker)
{
   if ( lfi_real ) { lfr->AddBoundaryIntegrator(lfi_real, bdr_attr_marker); }
   if ( lfi_imag ) { lfi->AddBoundaryIntegrator(lfi_imag, bdr_attr_marker); }
}

void
ComplexLinearForm::AddBdrFaceIntegrator(LinearFormIntegrator *lfi_real,
                                        LinearFormIntegrator *lfi_imag)
{
   if ( lfi_real ) { lfr->AddBdrFaceIntegrator(lfi_real); }
   if ( lfi_imag ) { lfi->AddBdrFaceIntegrator(lfi_imag); }
}

void
ComplexLinearForm::AddBdrFaceIntegrator(LinearFormIntegrator *lfi_real,
                                        LinearFormIntegrator *lfi_imag,
                                        Array<int> &bdr_attr_marker)
{
   if ( lfi_real ) { lfr->AddBdrFaceIntegrator(lfi_real, bdr_attr_marker); }
   if ( lfi_imag ) { lfi->AddBdrFaceIntegrator(lfi_imag, bdr_attr_marker); }
}

void
ComplexLinearForm::Update()
{
   FiniteElementSpace *fes = lfr->FESpace();
   this->Update(fes);
}

void
ComplexLinearForm::Update(FiniteElementSpace *fes)
{
   UseDevice(true);
   SetSize(2 * fes->GetVSize());
   this->Vector::operator=(0.0);

   lfr->MakeRef(fes, *this, 0);
   lfi->MakeRef(fes, *this, fes->GetVSize());
}

void
ComplexLinearForm::Assemble()
{
   lfr->SyncMemory(*this);
   lfi->SyncMemory(*this);
   lfr->Assemble();
   lfi->Assemble();
   if (conv == ComplexOperator::BLOCK_SYMMETRIC) { *lfi *= -1.0; }
   lfr->SyncAliasMemory(*this);
   lfi->SyncAliasMemory(*this);
}

complex<real_t>
ComplexLinearForm::operator()(const ComplexGridFunction &gf) const
{
   real_t s = (conv == ComplexOperator::HERMITIAN) ? 1.0 : -1.0;
   lfr->SyncMemory(*this);
   lfi->SyncMemory(*this);
   return complex<real_t>((*lfr)(gf.real()) - s * (*lfi)(gf.imag()),
                          (*lfr)(gf.imag()) + s * (*lfi)(gf.real()));
}


bool SesquilinearForm::RealInteg()
{
   int nint = blfr->GetFBFI()->Size() + blfr->GetDBFI()->Size() +
              blfr->GetBBFI()->Size() + blfr->GetBFBFI()->Size();
   return (nint != 0);
}

bool SesquilinearForm::ImagInteg()
{
   int nint = blfi->GetFBFI()->Size() + blfi->GetDBFI()->Size() +
              blfi->GetBBFI()->Size() + blfi->GetBFBFI()->Size();
   return (nint != 0);
}

SesquilinearForm::SesquilinearForm(FiniteElementSpace *f,
                                   ComplexOperator::Convention convention)
   : conv(convention),
     blfr(new BilinearForm(f)),
     blfi(new BilinearForm(f))
{}

SesquilinearForm::SesquilinearForm(FiniteElementSpace *f,
                                   BilinearForm *bfr, BilinearForm *bfi,
                                   ComplexOperator::Convention convention)
   : conv(convention),
     blfr(new BilinearForm(f,bfr)),
     blfi(new BilinearForm(f,bfi))
{}

void SesquilinearForm::SetDiagonalPolicy(mfem::Matrix::DiagonalPolicy dpolicy)
{
   diag_policy = dpolicy;
}

SesquilinearForm::~SesquilinearForm()
{
   delete blfr;
   delete blfi;
}

void SesquilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                                           BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { blfr->AddDomainIntegrator(bfi_real); }
   if (bfi_imag) { blfi->AddDomainIntegrator(bfi_imag); }
}

void SesquilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                                           BilinearFormIntegrator *bfi_imag,
                                           Array<int> & elem_marker)
{
   if (bfi_real) { blfr->AddDomainIntegrator(bfi_real, elem_marker); }
   if (bfi_imag) { blfi->AddDomainIntegrator(bfi_imag, elem_marker); }
}

void
SesquilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                                        BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { blfr->AddBoundaryIntegrator(bfi_real); }
   if (bfi_imag) { blfi->AddBoundaryIntegrator(bfi_imag); }
}

void
SesquilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                                        BilinearFormIntegrator *bfi_imag,
                                        Array<int> & bdr_marker)
{
   if (bfi_real) { blfr->AddBoundaryIntegrator(bfi_real, bdr_marker); }
   if (bfi_imag) { blfi->AddBoundaryIntegrator(bfi_imag, bdr_marker); }
}

void
SesquilinearForm::AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi_real,
                                            BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { blfr->AddInteriorFaceIntegrator(bfi_real); }
   if (bfi_imag) { blfi->AddInteriorFaceIntegrator(bfi_imag); }
}

void SesquilinearForm::AddBdrFaceIntegrator(BilinearFormIntegrator *bfi_real,
                                            BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { blfr->AddBdrFaceIntegrator(bfi_real); }
   if (bfi_imag) { blfi->AddBdrFaceIntegrator(bfi_imag); }
}

void SesquilinearForm::AddBdrFaceIntegrator(BilinearFormIntegrator *bfi_real,
                                            BilinearFormIntegrator *bfi_imag,
                                            Array<int> &bdr_marker)
{
   if (bfi_real) { blfr->AddBdrFaceIntegrator(bfi_real, bdr_marker); }
   if (bfi_imag) { blfi->AddBdrFaceIntegrator(bfi_imag, bdr_marker); }
}

void
SesquilinearForm::Assemble(int skip_zeros)
{
   blfr->Assemble(skip_zeros);
   blfi->Assemble(skip_zeros);
}

void
SesquilinearForm::Finalize(int skip_zeros)
{
   blfr->Finalize(skip_zeros);
   blfi->Finalize(skip_zeros);
}

ComplexSparseMatrix *
SesquilinearForm::AssembleComplexSparseMatrix()
{
   return new ComplexSparseMatrix(&blfr->SpMat(),
                                  &blfi->SpMat(),
                                  false, false, conv);
}

void
SesquilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list,
                                   Vector &x, Vector &b,
                                   OperatorHandle &A,
                                   Vector &X, Vector &B,
                                   int ci)
{
   FiniteElementSpace *fes = blfr->FESpace();
   const int vsize = fes->GetVSize();

   // Allocate temporary vector
   Vector b_0;
   b_0.UseDevice(true);
   b_0.SetSize(vsize);
   b_0 = 0.0;

   // Extract the real and imaginary parts of the input vectors
   MFEM_ASSERT(x.Size() == 2 * vsize, "Input GridFunction of incorrect size!");
   x.Read();
   Vector x_r; x_r.MakeRef(x, 0, vsize);
   Vector x_i; x_i.MakeRef(x, vsize, vsize);

   MFEM_ASSERT(b.Size() == 2 * vsize, "Input LinearForm of incorrect size!");
   b.Read();
   Vector b_r; b_r.MakeRef(b, 0, vsize);
   Vector b_i; b_i.MakeRef(b, vsize, vsize);

   if (conv == ComplexOperator::BLOCK_SYMMETRIC) { b_i *= -1.0; }

   const int tvsize = fes->GetTrueVSize();
   OperatorHandle A_r, A_i;

   X.UseDevice(true);
   X.SetSize(2 * tvsize);
   X = 0.0;

   B.UseDevice(true);
   B.SetSize(2 * tvsize);
   B = 0.0;

   Vector X_r; X_r.MakeRef(X, 0, tvsize);
   Vector X_i; X_i.MakeRef(X, tvsize, tvsize);
   Vector B_r; B_r.MakeRef(B, 0, tvsize);
   Vector B_i; B_i.MakeRef(B, tvsize, tvsize);

   Vector X_0, B_0;

   if (RealInteg())
   {
      blfr->SetDiagonalPolicy(diag_policy);

      b_0 = b_r;
      blfr->FormLinearSystem(ess_tdof_list, x_r, b_0, A_r, X_0, B_0, ci);
      X_r = X_0; B_r = B_0;

      b_0 = b_i;
      blfr->FormLinearSystem(ess_tdof_list, x_i, b_0, A_r, X_0, B_0, ci);
      X_i = X_0; B_i = B_0;

      if (ImagInteg())
      {
         blfi->SetDiagonalPolicy(mfem::Matrix::DiagonalPolicy::DIAG_ZERO);

         b_0 = 0.0;
         blfi->FormLinearSystem(ess_tdof_list, x_i, b_0, A_i, X_0, B_0, false);
         B_r -= B_0;

         b_0 = 0.0;
         blfi->FormLinearSystem(ess_tdof_list, x_r, b_0, A_i, X_0, B_0, false);
         B_i += B_0;
      }
   }
   else if (ImagInteg())
   {
      blfi->SetDiagonalPolicy(diag_policy);

      b_0 = b_i;
      blfi->FormLinearSystem(ess_tdof_list, x_r, b_0, A_i, X_0, B_0, ci);
      X_r = X_0; B_i = B_0;

      b_0 = b_r; b_0 *= -1.0;
      blfi->FormLinearSystem(ess_tdof_list, x_i, b_0, A_i, X_0, B_0, ci);
      X_i = X_0; B_r = B_0; B_r *= -1.0;
   }
   else
   {
      MFEM_ABORT("Real and Imaginary part of the Sesquilinear form are empty");
   }

   if (RealInteg() && ImagInteg())
   {
      // Modify RHS and offdiagonal blocks (imaginary parts of the matrix) to
      // conform with standard essential BC treatment
      if (A_i.Is<ConstrainedOperator>())
      {
         const int n = ess_tdof_list.Size();
         auto d_B_r = B_r.Write();
         auto d_B_i = B_i.Write();
         auto d_X_r = X_r.Read();
         auto d_X_i = X_i.Read();
         auto d_idx = ess_tdof_list.Read();
         mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
         {
            const int j = d_idx[i];
            d_B_r[j] = d_X_r[j];
            d_B_i[j] = d_X_i[j];
         });
         A_i.As<ConstrainedOperator>()->SetDiagonalPolicy
         (mfem::Operator::DiagonalPolicy::DIAG_ZERO);
      }
   }

   if (conv == ComplexOperator::BLOCK_SYMMETRIC)
   {
      B_i *= -1.0;
      b_i *= -1.0;
   }

   x_r.SyncAliasMemory(x);
   x_i.SyncAliasMemory(x);
   b_r.SyncAliasMemory(b);
   b_i.SyncAliasMemory(b);

   X_r.SyncAliasMemory(X);
   X_i.SyncAliasMemory(X);
   B_r.SyncAliasMemory(B);
   B_i.SyncAliasMemory(B);

   // A = A_r + i A_i
   A.Clear();
   if ( A_r.Type() == Operator::MFEM_SPARSEMAT ||
        A_i.Type() == Operator::MFEM_SPARSEMAT )
   {
      ComplexSparseMatrix * A_sp =
         new ComplexSparseMatrix(A_r.As<SparseMatrix>(),
                                 A_i.As<SparseMatrix>(),
                                 A_r.OwnsOperator(),
                                 A_i.OwnsOperator(),
                                 conv);
      A.Reset<ComplexSparseMatrix>(A_sp, true);
   }
   else
   {
      ComplexOperator * A_op =
         new ComplexOperator(A_r.Ptr(),
                             A_i.Ptr(),
                             A_r.OwnsOperator(),
                             A_i.OwnsOperator(),
                             conv);
      A.Reset<ComplexOperator>(A_op, true);
   }
   A_r.SetOperatorOwner(false);
   A_i.SetOperatorOwner(false);
}

void
SesquilinearForm::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                   OperatorHandle &A)

{
   OperatorHandle A_r, A_i;
   if (RealInteg())
   {
      blfr->SetDiagonalPolicy(diag_policy);
      blfr->FormSystemMatrix(ess_tdof_list, A_r);
   }
   if (ImagInteg())
   {
      blfi->SetDiagonalPolicy(RealInteg() ?
                              mfem::Matrix::DiagonalPolicy::DIAG_ZERO :
                              diag_policy);
      blfi->FormSystemMatrix(ess_tdof_list, A_i);
   }
   if (!RealInteg() && !ImagInteg())
   {
      MFEM_ABORT("Both Real and Imaginary part of the Sesquilinear form are empty");
   }

   if (RealInteg() && ImagInteg())
   {
      // Modify offdiagonal blocks (imaginary parts of the matrix) to conform
      // with standard essential BC treatment
      if (A_i.Is<ConstrainedOperator>())
      {
         A_i.As<ConstrainedOperator>()->SetDiagonalPolicy
         (mfem::Operator::DiagonalPolicy::DIAG_ZERO);
      }
   }

   // A = A_r + i A_i
   A.Clear();
   if ( A_r.Type() == Operator::MFEM_SPARSEMAT ||
        A_i.Type() == Operator::MFEM_SPARSEMAT )
   {
      ComplexSparseMatrix * A_sp =
         new ComplexSparseMatrix(A_r.As<SparseMatrix>(),
                                 A_i.As<SparseMatrix>(),
                                 A_r.OwnsOperator(),
                                 A_i.OwnsOperator(),
                                 conv);
      A.Reset<ComplexSparseMatrix>(A_sp, true);
   }
   else
   {
      ComplexOperator * A_op =
         new ComplexOperator(A_r.Ptr(),
                             A_i.Ptr(),
                             A_r.OwnsOperator(),
                             A_i.OwnsOperator(),
                             conv);
      A.Reset<ComplexOperator>(A_op, true);
   }
   A_r.SetOperatorOwner(false);
   A_i.SetOperatorOwner(false);
}

void
SesquilinearForm::RecoverFEMSolution(const Vector &X, const Vector &b,
                                     Vector &x)
{
   FiniteElementSpace *fes = blfr->FESpace();

   const SparseMatrix *P = fes->GetConformingProlongation();
   if (!P)
   {
      x = X;
      return;
   }

   const int vsize  = fes->GetVSize();
   const int tvsize = X.Size() / 2;

   X.Read();
   Vector X_r; X_r.MakeRef(const_cast<Vector&>(X), 0, tvsize);
   Vector X_i; X_i.MakeRef(const_cast<Vector&>(X), tvsize, tvsize);

   x.Write();
   Vector x_r; x_r.MakeRef(x, 0, vsize);
   Vector x_i; x_i.MakeRef(x, vsize, vsize);

   // Apply conforming prolongation
   P->Mult(X_r, x_r);
   P->Mult(X_i, x_i);

   x_r.SyncAliasMemory(x);
   x_i.SyncAliasMemory(x);
}

void
SesquilinearForm::Update(FiniteElementSpace *nfes)
{
   if ( blfr ) { blfr->Update(nfes); }
   if ( blfi ) { blfi->Update(nfes); }
}


#ifdef MFEM_USE_MPI

ParComplexGridFunction::ParComplexGridFunction(ParFiniteElementSpace *pfes)
   : Vector(2*(pfes->GetVSize()))
{
   UseDevice(true);
   this->Vector::operator=(0.0);

   pgfr = new ParGridFunction();
   pgfr->MakeRef(pfes, *this, 0);

   pgfi = new ParGridFunction();
   pgfi->MakeRef(pfes, *this, pfes->GetVSize());
}

void
ParComplexGridFunction::Update()
{
   ParFiniteElementSpace *pfes = pgfr->ParFESpace();
   const int vsize = pfes->GetVSize();

   const Operator *T = pfes->GetUpdateOperator();
   if (T)
   {
      // Update the individual GridFunction objects. This will allocate new data
      // arrays for each GridFunction.
      pgfr->Update();
      pgfi->Update();

      // Our data array now contains old data as well as being the wrong size so
      // reallocate it.
      UseDevice(true);
      this->SetSize(2 * vsize);
      this->Vector::operator=(0.0);

      // Create temporary vectors which point to the new data array
      Vector gf_r; gf_r.MakeRef(*this, 0, vsize);
      Vector gf_i; gf_i.MakeRef(*this, vsize, vsize);

      // Copy the updated GridFunctions into the new data array
      gf_r = *pgfr; gf_r.SyncAliasMemory(*this);
      gf_i = *pgfi; gf_i.SyncAliasMemory(*this);

      // Replace the individual data arrays with pointers into the new data
      // array
      pgfr->MakeRef(*this, 0, vsize);
      pgfi->MakeRef(*this, vsize, vsize);
   }
   else
   {
      // The existing data will not be transferred to the new GridFunctions so
      // delete it and allocate a new array
      UseDevice(true);
      this->SetSize(2 * vsize);
      this->Vector::operator=(0.0);

      // Point the individual GridFunctions to the new data array
      pgfr->MakeRef(*this, 0, vsize);
      pgfi->MakeRef(*this, vsize, vsize);

      // These updates will only set the proper 'sequence' value within the
      // individual GridFunction objects because their sizes are already correct
      pgfr->Update();
      pgfi->Update();
   }
}

void
ParComplexGridFunction::ProjectCoefficient(Coefficient &real_coeff,
                                           Coefficient &imag_coeff)
{
   pgfr->SyncMemory(*this);
   pgfi->SyncMemory(*this);
   pgfr->ProjectCoefficient(real_coeff);
   pgfi->ProjectCoefficient(imag_coeff);
   pgfr->SyncAliasMemory(*this);
   pgfi->SyncAliasMemory(*this);
}

void
ParComplexGridFunction::ProjectCoefficient(VectorCoefficient &real_vcoeff,
                                           VectorCoefficient &imag_vcoeff)
{
   pgfr->SyncMemory(*this);
   pgfi->SyncMemory(*this);
   pgfr->ProjectCoefficient(real_vcoeff);
   pgfi->ProjectCoefficient(imag_vcoeff);
   pgfr->SyncAliasMemory(*this);
   pgfi->SyncAliasMemory(*this);
}

void
ParComplexGridFunction::ProjectBdrCoefficient(Coefficient &real_coeff,
                                              Coefficient &imag_coeff,
                                              Array<int> &attr)
{
   pgfr->SyncMemory(*this);
   pgfi->SyncMemory(*this);
   pgfr->ProjectBdrCoefficient(real_coeff, attr);
   pgfi->ProjectBdrCoefficient(imag_coeff, attr);
   pgfr->SyncAliasMemory(*this);
   pgfi->SyncAliasMemory(*this);
}

void
ParComplexGridFunction::ProjectBdrCoefficientNormal(VectorCoefficient
                                                    &real_vcoeff,
                                                    VectorCoefficient
                                                    &imag_vcoeff,
                                                    Array<int> &attr)
{
   pgfr->SyncMemory(*this);
   pgfi->SyncMemory(*this);
   pgfr->ProjectBdrCoefficientNormal(real_vcoeff, attr);
   pgfi->ProjectBdrCoefficientNormal(imag_vcoeff, attr);
   pgfr->SyncAliasMemory(*this);
   pgfi->SyncAliasMemory(*this);
}

void
ParComplexGridFunction::ProjectBdrCoefficientTangent(VectorCoefficient
                                                     &real_vcoeff,
                                                     VectorCoefficient
                                                     &imag_vcoeff,
                                                     Array<int> &attr)
{
   pgfr->SyncMemory(*this);
   pgfi->SyncMemory(*this);
   pgfr->ProjectBdrCoefficientTangent(real_vcoeff, attr);
   pgfi->ProjectBdrCoefficientTangent(imag_vcoeff, attr);
   pgfr->SyncAliasMemory(*this);
   pgfi->SyncAliasMemory(*this);
}

void
ParComplexGridFunction::Distribute(const Vector *tv)
{
   ParFiniteElementSpace *pfes = pgfr->ParFESpace();
   const int tvsize = pfes->GetTrueVSize();

   tv->Read();
   Vector tvr; tvr.MakeRef(const_cast<Vector&>(*tv), 0, tvsize);
   Vector tvi; tvi.MakeRef(const_cast<Vector&>(*tv), tvsize, tvsize);

   pgfr->SyncMemory(*this);
   pgfi->SyncMemory(*this);
   pgfr->Distribute(tvr);
   pgfi->Distribute(tvi);
   pgfr->SyncAliasMemory(*this);
   pgfi->SyncAliasMemory(*this);
}

void
ParComplexGridFunction::ParallelProject(Vector &tv) const
{
   ParFiniteElementSpace *pfes = pgfr->ParFESpace();
   const int tvsize = pfes->GetTrueVSize();

   tv.Write();
   Vector tvr; tvr.MakeRef(tv, 0, tvsize);
   Vector tvi; tvi.MakeRef(tv, tvsize, tvsize);

   pgfr->SyncMemory(*this);
   pgfi->SyncMemory(*this);
   pgfr->ParallelProject(tvr);
   pgfi->ParallelProject(tvi);
   pgfr->SyncAliasMemory(*this);
   pgfi->SyncAliasMemory(*this);

   tvr.SyncAliasMemory(tv);
   tvi.SyncAliasMemory(tv);
}


ParComplexLinearForm::ParComplexLinearForm(ParFiniteElementSpace *pfes,
                                           ComplexOperator::Convention
                                           convention)
   : Vector(2*(pfes->GetVSize())),
     conv(convention)
{
   UseDevice(true);
   this->Vector::operator=(0.0);

   plfr = new ParLinearForm();
   plfr->MakeRef(pfes, *this, 0);

   plfi = new ParLinearForm();
   plfi->MakeRef(pfes, *this, pfes->GetVSize());

   HYPRE_BigInt *tdof_offsets_fes = pfes->GetTrueDofOffsets();

   int n = (HYPRE_AssumedPartitionCheck()) ? 2 : pfes->GetNRanks();
   tdof_offsets = new HYPRE_BigInt[n+1];

   for (int i = 0; i <= n; i++)
   {
      tdof_offsets[i] = 2 * tdof_offsets_fes[i];
   }
}


ParComplexLinearForm::ParComplexLinearForm(ParFiniteElementSpace *pfes,
                                           ParLinearForm *plf_r,
                                           ParLinearForm *plf_i,
                                           ComplexOperator::Convention
                                           convention)
   : Vector(2*(pfes->GetVSize())),
     conv(convention)
{
   UseDevice(true);
   this->Vector::operator=(0.0);

   plfr = new ParLinearForm(pfes, plf_r);
   plfi = new ParLinearForm(pfes, plf_i);

   plfr->MakeRef(pfes, *this, 0);
   plfi->MakeRef(pfes, *this, pfes->GetVSize());

   HYPRE_BigInt *tdof_offsets_fes = pfes->GetTrueDofOffsets();

   int n = (HYPRE_AssumedPartitionCheck()) ? 2 : pfes->GetNRanks();
   tdof_offsets = new HYPRE_BigInt[n+1];

   for (int i = 0; i <= n; i++)
   {
      tdof_offsets[i] = 2 * tdof_offsets_fes[i];
   }
}

ParComplexLinearForm::~ParComplexLinearForm()
{
   delete plfr;
   delete plfi;
   delete [] tdof_offsets;
}

void
ParComplexLinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                                          LinearFormIntegrator *lfi_imag)
{
   if ( lfi_real ) { plfr->AddDomainIntegrator(lfi_real); }
   if ( lfi_imag ) { plfi->AddDomainIntegrator(lfi_imag); }
}

void
ParComplexLinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                                          LinearFormIntegrator *lfi_imag,
                                          Array<int> &elem_attr_marker)
{
   if ( lfi_real ) { plfr->AddDomainIntegrator(lfi_real, elem_attr_marker); }
   if ( lfi_imag ) { plfi->AddDomainIntegrator(lfi_imag, elem_attr_marker); }
}

void
ParComplexLinearForm::AddBoundaryIntegrator(LinearFormIntegrator *lfi_real,
                                            LinearFormIntegrator *lfi_imag)
{
   if ( lfi_real ) { plfr->AddBoundaryIntegrator(lfi_real); }
   if ( lfi_imag ) { plfi->AddBoundaryIntegrator(lfi_imag); }
}

void
ParComplexLinearForm::AddBoundaryIntegrator(LinearFormIntegrator *lfi_real,
                                            LinearFormIntegrator *lfi_imag,
                                            Array<int> &bdr_attr_marker)
{
   if ( lfi_real ) { plfr->AddBoundaryIntegrator(lfi_real, bdr_attr_marker); }
   if ( lfi_imag ) { plfi->AddBoundaryIntegrator(lfi_imag, bdr_attr_marker); }
}

void
ParComplexLinearForm::AddBdrFaceIntegrator(LinearFormIntegrator *lfi_real,
                                           LinearFormIntegrator *lfi_imag)
{
   if ( lfi_real ) { plfr->AddBdrFaceIntegrator(lfi_real); }
   if ( lfi_imag ) { plfi->AddBdrFaceIntegrator(lfi_imag); }
}

void
ParComplexLinearForm::AddBdrFaceIntegrator(LinearFormIntegrator *lfi_real,
                                           LinearFormIntegrator *lfi_imag,
                                           Array<int> &bdr_attr_marker)
{
   if ( lfi_real ) { plfr->AddBdrFaceIntegrator(lfi_real, bdr_attr_marker); }
   if ( lfi_imag ) { plfi->AddBdrFaceIntegrator(lfi_imag, bdr_attr_marker); }
}

void
ParComplexLinearForm::Update(ParFiniteElementSpace *pf)
{
   ParFiniteElementSpace *pfes = (pf != NULL) ? pf : plfr->ParFESpace();

   UseDevice(true);
   SetSize(2 * pfes->GetVSize());
   this->Vector::operator=(0.0);

   plfr->MakeRef(pfes, *this, 0);
   plfi->MakeRef(pfes, *this, pfes->GetVSize());
}

void
ParComplexLinearForm::Assemble()
{
   plfr->SyncMemory(*this);
   plfi->SyncMemory(*this);
   plfr->Assemble();
   plfi->Assemble();
   if (conv == ComplexOperator::BLOCK_SYMMETRIC) { *plfi *= -1.0; }
   plfr->SyncAliasMemory(*this);
   plfi->SyncAliasMemory(*this);
}

void
ParComplexLinearForm::ParallelAssemble(Vector &tv)
{
   const int tvsize = plfr->ParFESpace()->GetTrueVSize();

   tv.Write();
   Vector tvr; tvr.MakeRef(tv, 0, tvsize);
   Vector tvi; tvi.MakeRef(tv, tvsize, tvsize);

   plfr->SyncMemory(*this);
   plfi->SyncMemory(*this);
   plfr->ParallelAssemble(tvr);
   plfi->ParallelAssemble(tvi);
   plfr->SyncAliasMemory(*this);
   plfi->SyncAliasMemory(*this);

   tvr.SyncAliasMemory(tv);
   tvi.SyncAliasMemory(tv);
}

HypreParVector *
ParComplexLinearForm::ParallelAssemble()
{
   const ParFiniteElementSpace *pfes = plfr->ParFESpace();
   const int tvsize = pfes->GetTrueVSize();

   HypreParVector *tv = new HypreParVector(pfes->GetComm(),
                                           2*(pfes->GlobalTrueVSize()),
                                           tdof_offsets);

   tv->Write();
   Vector tvr; tvr.MakeRef(*tv, 0, tvsize);
   Vector tvi; tvi.MakeRef(*tv, tvsize, tvsize);

   plfr->SyncMemory(*this);
   plfi->SyncMemory(*this);
   plfr->ParallelAssemble(tvr);
   plfi->ParallelAssemble(tvi);
   plfr->SyncAliasMemory(*this);
   plfi->SyncAliasMemory(*this);

   tvr.SyncAliasMemory(*tv);
   tvi.SyncAliasMemory(*tv);

   return tv;
}

complex<real_t>
ParComplexLinearForm::operator()(const ParComplexGridFunction &gf) const
{
   plfr->SyncMemory(*this);
   plfi->SyncMemory(*this);
   real_t s = (conv == ComplexOperator::HERMITIAN) ? 1.0 : -1.0;
   return complex<real_t>((*plfr)(gf.real()) - s * (*plfi)(gf.imag()),
                          (*plfr)(gf.imag()) + s * (*plfi)(gf.real()));
}


bool ParSesquilinearForm::RealInteg()
{
   int nint = pblfr->GetFBFI()->Size() + pblfr->GetDBFI()->Size() +
              pblfr->GetBBFI()->Size() + pblfr->GetBFBFI()->Size();
   return (nint != 0);
}

bool ParSesquilinearForm::ImagInteg()
{
   int nint = pblfi->GetFBFI()->Size() + pblfi->GetDBFI()->Size() +
              pblfi->GetBBFI()->Size() + pblfi->GetBFBFI()->Size();
   return (nint != 0);
}

ParSesquilinearForm::ParSesquilinearForm(ParFiniteElementSpace *pf,
                                         ComplexOperator::Convention
                                         convention)
   : conv(convention),
     pblfr(new ParBilinearForm(pf)),
     pblfi(new ParBilinearForm(pf))
{}

ParSesquilinearForm::ParSesquilinearForm(ParFiniteElementSpace *pf,
                                         ParBilinearForm *pbfr,
                                         ParBilinearForm *pbfi,
                                         ComplexOperator::Convention convention)
   : conv(convention),
     pblfr(new ParBilinearForm(pf,pbfr)),
     pblfi(new ParBilinearForm(pf,pbfi))
{}

ParSesquilinearForm::~ParSesquilinearForm()
{
   delete pblfr;
   delete pblfi;
}

void ParSesquilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                                              BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { pblfr->AddDomainIntegrator(bfi_real); }
   if (bfi_imag) { pblfi->AddDomainIntegrator(bfi_imag); }
}

void ParSesquilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                                              BilinearFormIntegrator *bfi_imag,
                                              Array<int> & elem_marker)
{
   if (bfi_real) { pblfr->AddDomainIntegrator(bfi_real, elem_marker); }
   if (bfi_imag) { pblfi->AddDomainIntegrator(bfi_imag, elem_marker); }
}

void
ParSesquilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                                           BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { pblfr->AddBoundaryIntegrator(bfi_real); }
   if (bfi_imag) { pblfi->AddBoundaryIntegrator(bfi_imag); }
}

void
ParSesquilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                                           BilinearFormIntegrator *bfi_imag,
                                           Array<int> & bdr_marker)
{
   if (bfi_real) { pblfr->AddBoundaryIntegrator(bfi_real, bdr_marker); }
   if (bfi_imag) { pblfi->AddBoundaryIntegrator(bfi_imag, bdr_marker); }
}

void
ParSesquilinearForm::AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi_real,
                                               BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { pblfr->AddInteriorFaceIntegrator(bfi_real); }
   if (bfi_imag) { pblfi->AddInteriorFaceIntegrator(bfi_imag); }
}

void
ParSesquilinearForm::AddBdrFaceIntegrator(BilinearFormIntegrator *bfi_real,
                                          BilinearFormIntegrator *bfi_imag)
{
   if (bfi_real) { pblfr->AddBdrFaceIntegrator(bfi_real); }
   if (bfi_imag) { pblfi->AddBdrFaceIntegrator(bfi_imag); }
}

void
ParSesquilinearForm::AddBdrFaceIntegrator(BilinearFormIntegrator *bfi_real,
                                          BilinearFormIntegrator *bfi_imag,
                                          Array<int> &bdr_marker)
{
   if (bfi_real) { pblfr->AddBdrFaceIntegrator(bfi_real, bdr_marker); }
   if (bfi_imag) { pblfi->AddBdrFaceIntegrator(bfi_imag, bdr_marker); }
}

void
ParSesquilinearForm::Assemble(int skip_zeros)
{
   pblfr->Assemble(skip_zeros);
   pblfi->Assemble(skip_zeros);
}

void
ParSesquilinearForm::Finalize(int skip_zeros)
{
   pblfr->Finalize(skip_zeros);
   pblfi->Finalize(skip_zeros);
}

ComplexHypreParMatrix *
ParSesquilinearForm::ParallelAssemble()
{
   return new ComplexHypreParMatrix(pblfr->ParallelAssemble(),
                                    pblfi->ParallelAssemble(),
                                    true, true, conv);
}

void
ParSesquilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list,
                                      Vector &x, Vector &b,
                                      OperatorHandle &A,
                                      Vector &X, Vector &B,
                                      int ci)
{
   ParFiniteElementSpace *pfes = pblfr->ParFESpace();
   const int vsize = pfes->GetVSize();

   // Allocate temporary vector
   Vector b_0;
   b_0.UseDevice(true);
   b_0.SetSize(vsize);
   b_0 = 0.0;

   // Extract the real and imaginary parts of the input vectors
   MFEM_ASSERT(x.Size() == 2 * vsize, "Input GridFunction of incorrect size!");
   x.Read();
   Vector x_r; x_r.MakeRef(x, 0, vsize);
   Vector x_i; x_i.MakeRef(x, vsize, vsize);

   MFEM_ASSERT(b.Size() == 2 * vsize, "Input LinearForm of incorrect size!");
   b.Read();
   Vector b_r; b_r.MakeRef(b, 0, vsize);
   Vector b_i; b_i.MakeRef(b, vsize, vsize);

   if (conv == ComplexOperator::BLOCK_SYMMETRIC) { b_i *= -1.0; }

   const int tvsize = pfes->GetTrueVSize();
   OperatorHandle A_r, A_i;

   X.UseDevice(true);
   X.SetSize(2 * tvsize);
   X = 0.0;

   B.UseDevice(true);
   B.SetSize(2 * tvsize);
   B = 0.0;

   Vector X_r; X_r.MakeRef(X, 0, tvsize);
   Vector X_i; X_i.MakeRef(X, tvsize, tvsize);
   Vector B_r; B_r.MakeRef(B, 0, tvsize);
   Vector B_i; B_i.MakeRef(B, tvsize, tvsize);

   Vector X_0, B_0;

   if (RealInteg())
   {
      b_0 = b_r;
      pblfr->FormLinearSystem(ess_tdof_list, x_r, b_0, A_r, X_0, B_0, ci);
      X_r = X_0; B_r = B_0;

      b_0 = b_i;
      pblfr->FormLinearSystem(ess_tdof_list, x_i, b_0, A_r, X_0, B_0, ci);
      X_i = X_0; B_i = B_0;

      if (ImagInteg())
      {
         b_0 = 0.0;
         pblfi->FormLinearSystem(ess_tdof_list, x_i, b_0, A_i, X_0, B_0, false);
         B_r -= B_0;

         b_0 = 0.0;
         pblfi->FormLinearSystem(ess_tdof_list, x_r, b_0, A_i, X_0, B_0, false);
         B_i += B_0;
      }
   }
   else if (ImagInteg())
   {
      b_0 = b_i;
      pblfi->FormLinearSystem(ess_tdof_list, x_r, b_0, A_i, X_0, B_0, ci);
      X_r = X_0; B_i = B_0;

      b_0 = b_r; b_0 *= -1.0;
      pblfi->FormLinearSystem(ess_tdof_list, x_i, b_0, A_i, X_0, B_0, ci);
      X_i = X_0; B_r = B_0; B_r *= -1.0;
   }
   else
   {
      MFEM_ABORT("Real and Imaginary part of the Sesquilinear form are empty");
   }

   if (RealInteg() && ImagInteg())
   {
      // Modify RHS to conform with standard essential BC treatment
      const int n = ess_tdof_list.Size();
      auto d_B_r = B_r.Write();
      auto d_B_i = B_i.Write();
      auto d_X_r = X_r.Read();
      auto d_X_i = X_i.Read();
      auto d_idx = ess_tdof_list.Read();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         const int j = d_idx[i];
         d_B_r[j] = d_X_r[j];
         d_B_i[j] = d_X_i[j];
      });
      // Modify offdiagonal blocks (imaginary parts of the matrix) to conform
      // with standard essential BC treatment
      if (A_i.Type() == Operator::Hypre_ParCSR)
      {
         HypreParMatrix * Ah;
         A_i.Get(Ah);
         hypre_ParCSRMatrix *Aih = *Ah;
         Ah->HypreReadWrite();
         const int *d_ess_tdof_list =
            ess_tdof_list.GetMemory().Read(GetHypreForallMemoryClass(), n);
         HYPRE_Int *d_diag_i = Aih->diag->i;
         real_t *d_diag_data = Aih->diag->data;
         mfem::hypre_forall(n, [=] MFEM_HOST_DEVICE (int k)
         {
            const int j = d_ess_tdof_list[k];
            d_diag_data[d_diag_i[j]] = 0.0;
         });
      }
      else
      {
         A_i.As<ConstrainedOperator>()->SetDiagonalPolicy
         (mfem::Operator::DiagonalPolicy::DIAG_ZERO);
      }
   }

   if (conv == ComplexOperator::BLOCK_SYMMETRIC)
   {
      B_i *= -1.0;
      b_i *= -1.0;
   }

   x_r.SyncAliasMemory(x);
   x_i.SyncAliasMemory(x);
   b_r.SyncAliasMemory(b);
   b_i.SyncAliasMemory(b);

   X_r.SyncAliasMemory(X);
   X_i.SyncAliasMemory(X);
   B_r.SyncAliasMemory(B);
   B_i.SyncAliasMemory(B);

   // A = A_r + i A_i
   A.Clear();
   if ( A_r.Type() == Operator::Hypre_ParCSR ||
        A_i.Type() == Operator::Hypre_ParCSR )
   {
      ComplexHypreParMatrix * A_hyp =
         new ComplexHypreParMatrix(A_r.As<HypreParMatrix>(),
                                   A_i.As<HypreParMatrix>(),
                                   A_r.OwnsOperator(),
                                   A_i.OwnsOperator(),
                                   conv);
      A.Reset<ComplexHypreParMatrix>(A_hyp, true);
   }
   else
   {
      ComplexOperator * A_op =
         new ComplexOperator(A_r.As<Operator>(),
                             A_i.As<Operator>(),
                             A_r.OwnsOperator(),
                             A_i.OwnsOperator(),
                             conv);
      A.Reset<ComplexOperator>(A_op, true);
   }
   A_r.SetOperatorOwner(false);
   A_i.SetOperatorOwner(false);
}

void
ParSesquilinearForm::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                      OperatorHandle &A)
{
   OperatorHandle A_r, A_i;
   if (RealInteg())
   {
      pblfr->FormSystemMatrix(ess_tdof_list, A_r);
   }
   if (ImagInteg())
   {
      pblfi->FormSystemMatrix(ess_tdof_list, A_i);
   }
   if (!RealInteg() && !ImagInteg())
   {
      MFEM_ABORT("Both Real and Imaginary part of the Sesquilinear form are empty");
   }

   if (RealInteg() && ImagInteg())
   {
      // Modify offdiagonal blocks (imaginary parts of the matrix) to conform
      // with standard essential BC treatment
      if ( A_i.Type() == Operator::Hypre_ParCSR )
      {
         int n = ess_tdof_list.Size();
         HypreParMatrix * Ah;
         A_i.Get(Ah);
         hypre_ParCSRMatrix * Aih = *Ah;
         for (int k = 0; k < n; k++)
         {
            int j = ess_tdof_list[k];
            Aih->diag->data[Aih->diag->i[j]] = 0.0;
         }
      }
      else
      {
         A_i.As<ConstrainedOperator>()->SetDiagonalPolicy
         (mfem::Operator::DiagonalPolicy::DIAG_ZERO);
      }
   }

   // A = A_r + i A_i
   A.Clear();
   if ( A_r.Type() == Operator::Hypre_ParCSR ||
        A_i.Type() == Operator::Hypre_ParCSR )
   {
      ComplexHypreParMatrix * A_hyp =
         new ComplexHypreParMatrix(A_r.As<HypreParMatrix>(),
                                   A_i.As<HypreParMatrix>(),
                                   A_r.OwnsOperator(),
                                   A_i.OwnsOperator(),
                                   conv);
      A.Reset<ComplexHypreParMatrix>(A_hyp, true);
   }
   else
   {
      ComplexOperator * A_op =
         new ComplexOperator(A_r.As<Operator>(),
                             A_i.As<Operator>(),
                             A_r.OwnsOperator(),
                             A_i.OwnsOperator(),
                             conv);
      A.Reset<ComplexOperator>(A_op, true);
   }
   A_r.SetOperatorOwner(false);
   A_i.SetOperatorOwner(false);
}

void
ParSesquilinearForm::RecoverFEMSolution(const Vector &X, const Vector &b,
                                        Vector &x)
{
   ParFiniteElementSpace *pfes = pblfr->ParFESpace();

   const Operator &P = *pfes->GetProlongationMatrix();

   const int vsize  = pfes->GetVSize();
   const int tvsize = X.Size() / 2;

   X.Read();
   Vector X_r; X_r.MakeRef(const_cast<Vector&>(X), 0, tvsize);
   Vector X_i; X_i.MakeRef(const_cast<Vector&>(X), tvsize, tvsize);

   x.Write();
   Vector x_r; x_r.MakeRef(x, 0, vsize);
   Vector x_i; x_i.MakeRef(x, vsize, vsize);

   // Apply conforming prolongation
   P.Mult(X_r, x_r);
   P.Mult(X_i, x_i);

   x_r.SyncAliasMemory(x);
   x_i.SyncAliasMemory(x);
}

void
ParSesquilinearForm::Update(FiniteElementSpace *nfes)
{
   if ( pblfr ) { pblfr->Update(nfes); }
   if ( pblfi ) { pblfi->Update(nfes); }
}

#endif // MFEM_USE_MPI

}
