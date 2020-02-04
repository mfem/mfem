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

ComplexGridFunction::ComplexGridFunction(FiniteElementSpace *fes)
   : Vector(2*(fes->GetVSize()))
{
   gfr = new GridFunction(fes, data);
   gfi = new GridFunction(fes, &data[fes->GetVSize()]);
}

void
ComplexGridFunction::Update()
{
   FiniteElementSpace * fes = gfr->FESpace();

   int vsize = fes->GetVSize();

   const Operator *T = fes->GetUpdateOperator();
   if (T)
   {
      // Update the individual GridFunction objects. This will allocate new data
      // arrays for each GridFunction.
      gfr->Update();
      gfi->Update();

      // Our data array now contains old data as well as being the wrong size so
      // reallocate it.
      this->SetSize(2 * vsize);

      // Create temporary vectors which point to the new data array
      Vector gf_r(data, vsize);
      Vector gf_i((data) ? &data[vsize] : data, vsize);

      // Copy the updated GridFunctions into the new data array
      gf_r = *gfr;
      gf_i = *gfi;

      // Replace the individual data arrays with pointers into the new data
      // array
      gfr->NewDataAndSize(data, vsize);
      gfi->NewDataAndSize((data) ? &data[vsize] : data, vsize);
   }
   else
   {
      // The existing data will not be transferred to the new GridFunctions so
      // delete it a allocate a new array
      this->SetSize(2 * vsize);

      // Point the individual GridFunctions to the new data array
      gfr->NewDataAndSize(data, vsize);
      gfi->NewDataAndSize((data) ? &data[vsize] : data, vsize);

      // These updates will only set the proper 'sequence' value within
      // the individual GridFunction objects because their sizes are
      // already correct
      gfr->Update();
      gfi->Update();
   }
}

void
ComplexGridFunction::ProjectCoefficient(Coefficient &real_coeff,
                                        Coefficient &imag_coeff)
{
   gfr->ProjectCoefficient(real_coeff);
   gfi->ProjectCoefficient(imag_coeff);
}

void
ComplexGridFunction::ProjectCoefficient(VectorCoefficient &real_vcoeff,
                                        VectorCoefficient &imag_vcoeff)
{
   gfr->ProjectCoefficient(real_vcoeff);
   gfi->ProjectCoefficient(imag_vcoeff);
}

void
ComplexGridFunction::ProjectBdrCoefficient(Coefficient &real_coeff,
                                           Coefficient &imag_coeff,
                                           Array<int> &attr)
{
   gfr->ProjectBdrCoefficient(real_coeff, attr);
   gfi->ProjectBdrCoefficient(imag_coeff, attr);
}

void
ComplexGridFunction::ProjectBdrCoefficientNormal(VectorCoefficient &real_vcoeff,
                                                 VectorCoefficient &imag_vcoeff,
                                                 Array<int> &attr)
{
   gfr->ProjectBdrCoefficientNormal(real_vcoeff, attr);
   gfi->ProjectBdrCoefficientNormal(imag_vcoeff, attr);
}

void
ComplexGridFunction::ProjectBdrCoefficientTangent(VectorCoefficient
                                                  &real_vcoeff,
                                                  VectorCoefficient
                                                  &imag_vcoeff,
                                                  Array<int> &attr)
{
   gfr->ProjectBdrCoefficientTangent(real_vcoeff, attr);
   gfi->ProjectBdrCoefficientTangent(imag_vcoeff, attr);
}


ComplexLinearForm::ComplexLinearForm(FiniteElementSpace *f,
                                     ComplexOperator::Convention convention)
   : Vector(2*(f->GetVSize())),
     conv(convention)
{
   lfr = new LinearForm(f, data);
   lfi = new LinearForm(f, &data[f->GetVSize()]);
}

ComplexLinearForm::ComplexLinearForm(FiniteElementSpace *fes,
                                     LinearForm *lf_r, LinearForm *lf_i,
                                     ComplexOperator::Convention convention)
   : Vector(2*(fes->GetVSize())),
     conv(convention)
{
   lfr = new LinearForm(fes, lf_r);  lfr->SetData(data);
   lfi = new LinearForm(fes, lf_i);  lfi->SetData(&data[fes->GetVSize()]);
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
   int vsize = fes->GetVSize();
   SetSize(2 * vsize);

   Vector vlfr(data, vsize);
   Vector vlfi((data) ? &data[vsize] : data, vsize);

   lfr->Update(fes, vlfr, 0);
   lfi->Update(fes, vlfi, 0);
}

void
ComplexLinearForm::Assemble()
{
   lfr->Assemble();
   lfi->Assemble();
   if (conv == ComplexOperator::BLOCK_SYMMETRIC)
   {
      *lfi *= -1.0;
   }
}

complex<double>
ComplexLinearForm::operator()(const ComplexGridFunction &gf) const
{
   double s = (conv == ComplexOperator::HERMITIAN)?1.0:-1.0;
   return complex<double>((*lfr)(gf.real()) - s * (*lfi)(gf.imag()),
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
   FiniteElementSpace * fes = blfr->FESpace();

   int vsize  = fes->GetVSize();

   // Allocate temporary vectors
   Vector b_0(vsize);  b_0 = 0.0;

   // Extract the real and imaginary parts of the input vectors
   MFEM_ASSERT(x.Size() == 2 * vsize, "Input GridFunction of incorrect size!");
   Vector x_r(x.GetData(), vsize);
   Vector x_i(&(x.GetData())[vsize], vsize);

   MFEM_ASSERT(b.Size() == 2 * vsize, "Input LinearForm of incorrect size!");
   Vector b_r(b.GetData(), vsize);
   Vector b_i(&(b.GetData())[vsize], vsize);

   if (conv == ComplexOperator::BLOCK_SYMMETRIC) { b_i *= -1.0; }

   int tvsize = fes->GetTrueVSize();
   SparseMatrix * A_r = nullptr;
   SparseMatrix * A_i = nullptr;

   X.SetSize(2 * tvsize);
   B.SetSize(2 * tvsize);

   Vector X_0(tvsize), B_0(tvsize);
   Vector X_r(X.GetData(),tvsize);
   Vector X_i(&(X.GetData())[tvsize], tvsize);
   Vector B_r(B.GetData(), tvsize);
   Vector B_i(&(B.GetData())[tvsize], tvsize);

   if (RealInteg())
   {
      A_r = new SparseMatrix;
      blfr->SetDiagonalPolicy(diag_policy);

      b_0 = b_r;
      blfr->FormLinearSystem(ess_tdof_list, x_r, b_0, *A_r, X_0, B_0, ci);
      X_r = X_0; B_r = B_0;

      b_0 = b_i;
      blfr->FormLinearSystem(ess_tdof_list, x_i, b_0, *A_r, X_0, B_0, ci);
      X_i = X_0; B_i = B_0;

      if (ImagInteg())
      {
         A_i = new SparseMatrix;
         blfi->SetDiagonalPolicy(mfem::Matrix::DiagonalPolicy::DIAG_ZERO);

         b_0 = 0.0;
         blfi->FormLinearSystem(ess_tdof_list, x_i, b_0, *A_i, X_0, B_0, false);
         B_r -= B_0;

         b_0 = 0.0;
         blfi->FormLinearSystem(ess_tdof_list, x_r, b_0, *A_i, X_0, B_0, false);
         B_i += B_0;
      }
   }
   else if (ImagInteg())
   {
      A_i = new SparseMatrix;
      blfi->SetDiagonalPolicy(diag_policy);

      b_0 = b_i;
      blfi->FormLinearSystem(ess_tdof_list, x_r, b_0, *A_i, X_0, B_0, ci);
      X_r = X_0; B_i = B_0;

      b_0 = b_r; b_0 *= -1.0;
      blfi->FormLinearSystem(ess_tdof_list, x_i, b_0, *A_i, X_0, B_0, ci);
      X_i = X_0; B_r = B_0; B_r *= -1.0;
   }
   else
   {
      MFEM_ABORT("Real and Imaginary part of the Sesquilinear form are empty");
   }

   if (conv == ComplexOperator::BLOCK_SYMMETRIC)
   {
      B_i *= -1.0;
      b_i *= -1.0;
   }
   // A = A_r + i A_i
   A.Clear();
   ComplexSparseMatrix * A_sp;
   A_sp = new ComplexSparseMatrix(A_r, A_i, true, true, conv);
   A.Reset<ComplexSparseMatrix>(A_sp, true);
}

void
SesquilinearForm::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                   OperatorHandle &A)

{
   SparseMatrix * A_r = nullptr;
   SparseMatrix * A_i = nullptr;

   if (RealInteg())
   {
      A_r = new SparseMatrix;
      blfr->SetDiagonalPolicy(diag_policy);
      blfr->FormSystemMatrix(ess_tdof_list, *A_r);
   }
   if (ImagInteg())
   {
      A_i = new SparseMatrix;
      blfr->SetDiagonalPolicy(diag_policy);
      blfi->FormSystemMatrix(ess_tdof_list, *A_i);
   }
   if (!RealInteg() && !ImagInteg())
   {
      MFEM_ABORT("Both Real and Imaginary part of the Sesquilinear form are empty");
   }

   // A = A_r + i A_i
   A.Clear();
   ComplexSparseMatrix * A_sp =
      new ComplexSparseMatrix(A_r, A_i, true, true, conv);
   A.Reset<ComplexSparseMatrix>(A_sp, true);
}

void
SesquilinearForm::RecoverFEMSolution(const Vector &X, const Vector &b,
                                     Vector &x)
{
   FiniteElementSpace * fes = blfr->FESpace();

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
   if ( blfr ) { blfr->Update(nfes); }
   if ( blfi ) { blfi->Update(nfes); }
}


#ifdef MFEM_USE_MPI

ParComplexGridFunction::ParComplexGridFunction(ParFiniteElementSpace *pfes)
   : Vector(2*(pfes->GetVSize()))
{
   pgfr = new ParGridFunction(pfes, data);
   pgfi = new ParGridFunction(pfes, (data) ? &data[pfes->GetVSize()]:data);
}

void
ParComplexGridFunction::Update()
{
   ParFiniteElementSpace * pfes = pgfr->ParFESpace();

   int vsize = pfes->GetVSize();

   const Operator *T = pfes->GetUpdateOperator();
   if (T)
   {
      // Update the individual GridFunction objects. This will allocate new data
      // arrays for each GridFunction.
      pgfr->Update();
      pgfi->Update();

      // Our data array now contains old data as well as being the wrong size
      // so reallocate it.
      this->SetSize(2 * vsize);

      // Create temporary vectors which point to the new data array
      Vector gf_r(data, vsize);
      Vector gf_i((data) ? &data[vsize] : data, vsize);

      // Copy the updated GridFunctions into the new data array
      gf_r = *pgfr;
      gf_i = *pgfi;

      // Replace the individual data arrays with pointers into the new data
      // array
      pgfr->NewDataAndSize(data, vsize);
      pgfi->NewDataAndSize((data) ? &data[vsize] : data, vsize);
   }
   else
   {
      // The existing data will not be transferred to the new GridFunctions so
      // delete it a allocate a new array
      this->SetSize(2 * vsize);

      // Point the individual GridFunctions to the new data array
      pgfr->NewDataAndSize(data, vsize);
      pgfi->NewDataAndSize((data) ? &data[vsize] : data, vsize);

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
   pgfr->ProjectCoefficient(real_coeff);
   pgfi->ProjectCoefficient(imag_coeff);
}

void
ParComplexGridFunction::ProjectCoefficient(VectorCoefficient &real_vcoeff,
                                           VectorCoefficient &imag_vcoeff)
{
   pgfr->ProjectCoefficient(real_vcoeff);
   pgfi->ProjectCoefficient(imag_vcoeff);
}

void
ParComplexGridFunction::ProjectBdrCoefficient(Coefficient &real_coeff,
                                              Coefficient &imag_coeff,
                                              Array<int> &attr)
{
   pgfr->ProjectBdrCoefficient(real_coeff, attr);
   pgfi->ProjectBdrCoefficient(imag_coeff, attr);
}

void
ParComplexGridFunction::ProjectBdrCoefficientNormal(VectorCoefficient
                                                    &real_vcoeff,
                                                    VectorCoefficient
                                                    &imag_vcoeff,
                                                    Array<int> &attr)
{
   pgfr->ProjectBdrCoefficientNormal(real_vcoeff, attr);
   pgfi->ProjectBdrCoefficientNormal(imag_vcoeff, attr);
}

void
ParComplexGridFunction::ProjectBdrCoefficientTangent(VectorCoefficient
                                                     &real_vcoeff,
                                                     VectorCoefficient
                                                     &imag_vcoeff,
                                                     Array<int> &attr)
{
   pgfr->ProjectBdrCoefficientTangent(real_vcoeff, attr);
   pgfi->ProjectBdrCoefficientTangent(imag_vcoeff, attr);
}

void
ParComplexGridFunction::Distribute(const Vector *tv)
{
   ParFiniteElementSpace * pfes = pgfr->ParFESpace();
   HYPRE_Int size = pfes->GetTrueVSize();

   double * tvd = tv->GetData();
   Vector tvr(tvd, size);
   Vector tvi((tvd) ? &tvd[size] : tvd, size);

   pgfr->Distribute(tvr);
   pgfi->Distribute(tvi);
}

void
ParComplexGridFunction::ParallelProject(Vector &tv) const
{
   ParFiniteElementSpace * pfes = pgfr->ParFESpace();
   HYPRE_Int size = pfes->GetTrueVSize();

   double * tvd = tv.GetData();
   Vector tvr(tvd, size);
   Vector tvi((tvd) ? &tvd[size] : tvd, size);

   pgfr->ParallelProject(tvr);
   pgfi->ParallelProject(tvi);
}


ParComplexLinearForm::ParComplexLinearForm(ParFiniteElementSpace *pfes,
                                           ComplexOperator::Convention
                                           convention)
   : Vector(2*(pfes->GetVSize())),
     conv(convention)
{
   plfr = new ParLinearForm(pfes, data);
   plfi = new ParLinearForm(pfes, (data) ? &data[pfes->GetVSize()]:data);

   HYPRE_Int * tdof_offsets_fes = pfes->GetTrueDofOffsets();

   int n = (HYPRE_AssumedPartitionCheck()) ? 2 : pfes->GetNRanks();
   tdof_offsets = new HYPRE_Int[n+1];

   for (int i=0; i<=n; i++)
   {
      tdof_offsets[i] = 2 * tdof_offsets_fes[i];
   }
}


ParComplexLinearForm::ParComplexLinearForm(ParFiniteElementSpace *pfes,
                                           ParLinearForm *plf_r, ParLinearForm *plf_i,
                                           ComplexOperator::Convention
                                           convention)
   : Vector(2*(pfes->GetVSize())),
     conv(convention)
{
   plfr = new ParLinearForm(pfes, plf_r);
   plfr->SetData(data);
   plfi = new ParLinearForm(pfes, plf_i);
   plfi->SetData((data) ? &data[pfes->GetVSize()]:data);

   HYPRE_Int * tdof_offsets_fes = pfes->GetTrueDofOffsets();

   int n = (HYPRE_AssumedPartitionCheck()) ? 2 : pfes->GetNRanks();
   tdof_offsets = new HYPRE_Int[n+1];

   for (int i=0; i<=n; i++)
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
   ParFiniteElementSpace *pfes = (pf!=NULL)?pf:plfr->ParFESpace();
   int vsize = pfes->GetVSize();
   SetSize(2 * vsize);

   Vector vplfr(data, vsize);
   Vector vplfi((data) ? &data[vsize] : data, vsize);

   plfr->Update(pfes, vplfr, 0);
   plfi->Update(pfes, vplfi, 0);
}

void
ParComplexLinearForm::Assemble()
{
   plfr->Assemble();
   plfi->Assemble();
   if (conv == ComplexOperator::BLOCK_SYMMETRIC)
   {
      *plfi *= -1.0;
   }
}

void
ParComplexLinearForm::ParallelAssemble(Vector &tv)
{
   HYPRE_Int size = plfr->ParFESpace()->GetTrueVSize();

   double * tvd = tv.GetData();
   Vector tvr(tvd, size);
   Vector tvi((tvd) ? &tvd[size] : tvd, size);

   plfr->ParallelAssemble(tvr);
   plfi->ParallelAssemble(tvi);
}

HypreParVector *
ParComplexLinearForm::ParallelAssemble()
{
   const ParFiniteElementSpace * pfes = plfr->ParFESpace();

   HypreParVector * tv = new HypreParVector(pfes->GetComm(),
                                            2*(pfes->GlobalTrueVSize()),
                                            tdof_offsets);

   HYPRE_Int size = pfes->GetTrueVSize();

   double * tvd = tv->GetData();
   Vector tvr(tvd, size);
   Vector tvi((tvd) ? &tvd[size] : tvd, size);

   plfr->ParallelAssemble(tvr);
   plfi->ParallelAssemble(tvi);

   return tv;
}

complex<double>
ParComplexLinearForm::operator()(const ParComplexGridFunction &gf) const
{
   double s = (conv == ComplexOperator::HERMITIAN)?1.0:-1.0;
   return complex<double>((*plfr)(gf.real()) - s * (*plfi)(gf.imag()),
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
                                         ParBilinearForm *pbfr, ParBilinearForm *pbfi,
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
   ParFiniteElementSpace * pfes = pblfr->ParFESpace();
   int vsize = pfes->GetVSize();

   // Allocate temporary vectors
   Vector b_0(vsize);  b_0 = 0.0;

   // Extract the real and imaginary parts of the input vectors
   Vector x_r(x.GetData(), vsize);
   Vector x_i(&(x.GetData())[vsize], vsize);

   MFEM_ASSERT(b.Size() == 2 * vsize, "Input LinearForm of incorrect size!");
   Vector b_r(b.GetData(), vsize);
   Vector b_i(&(b.GetData())[vsize], vsize);

   if (conv == ComplexOperator::BLOCK_SYMMETRIC) { b_i *= -1.0; }

   int tvsize = pfes->GetTrueVSize();

   OperatorHandle A_r, A_i;

   X.SetSize(2 * tvsize);
   B.SetSize(2 * tvsize);

   Vector X_0(tvsize), B_0(tvsize);
   Vector X_r(X.GetData(),tvsize);
   Vector X_i(&(X.GetData())[tvsize], tvsize);
   Vector B_r(B.GetData(), tvsize);
   Vector B_i(&(B.GetData())[tvsize], tvsize);

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

   // Modify RHS and offdiagonal blocks (Imaginary parts of the matrix) to
   // conform with standard essential BC treatment i.e. zero out rows and
   // columns and place ones on the diagonal.
   if (RealInteg() && ImagInteg())
   {
      if ( A_i.Type() == Operator::Hypre_ParCSR )
      {
         HypreParMatrix * Ah;  A_i.Get(Ah);
         int n = ess_tdof_list.Size();
         hypre_ParCSRMatrix * Aih =
            (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*Ah);
         for (int k=0; k<n; k++)
         {
            int j=ess_tdof_list[k];
            Aih->diag->data[Aih->diag->i[j]] = 0.0;
            B_r(j) = X_r(j);
            B_i(j) = X_i(j);
         }
      }
   }

   if (conv == ComplexOperator::BLOCK_SYMMETRIC)
   {
      B_i *= -1.0;
      b_i *= -1.0;
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

   // Modify offdiagonal blocks (Imaginary parts of the matrix) to
   // conform with standard essential BC treatment i.e. zero out rows and
   // columns and place ones on the diagonal.
   if (RealInteg() && ImagInteg())
   {
      if ( A_i.Type() == Operator::Hypre_ParCSR )
      {
         int n = ess_tdof_list.Size();
         int j;

         HypreParMatrix * Ah;  A_i.Get(Ah);
         hypre_ParCSRMatrix * Aih =
            (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*Ah);
         for (int k=0; k<n; k++)
         {
            j=ess_tdof_list[k];
            Aih->diag->data[Aih->diag->i[j]] = 0.0;
         }
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
}

void
ParSesquilinearForm::RecoverFEMSolution(const Vector &X, const Vector &b,
                                        Vector &x)
{
   ParFiniteElementSpace * pfes = pblfr->ParFESpace();

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
   if ( pblfr ) { pblfr->Update(nfes); }
   if ( pblfi ) { pblfi->Update(nfes); }
}


#endif // MFEM_USE_MPI

}
