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

#include "braginskii_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;
namespace mfem
{
using namespace miniapps;

namespace plasma
{

DiffPerpCoefficient::DiffPerpCoefficient()
{}

double
DiffPerpCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
   return diff_i_perp();
}

DiffCrossCoefficient::DiffCrossCoefficient()
{}

double
DiffCrossCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
   return diff_i_cross();
}

DiffCoefficient::DiffCoefficient(int dim, ParGridFunction & B)
   : MatrixCoefficient(dim),
     BCoef_(&B),
     bHat_(dim)
{}
/*
void DiffCoefficient::SetT(ParGridFunction & T)
{
}
*/
void DiffCoefficient::SetB(ParGridFunction & B)
{
   BCoef_.SetGridFunction(&B);
}

void DiffCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                           const IntegrationPoint &ip)
{
   double diff_perp  = diff_i_perp();
   double diff_cross = diff_i_cross();

   BCoef_.Eval(bHat_, T, ip);
   bHat_ /= bHat_.Norml2();

   K.SetSize(bHat_.Size());

   if (width == 2)
   {
      K(0,0) = (1.0 - bHat_[0] * bHat_[0]) * diff_perp;
      K(0,1) = -bHat_[0] * bHat_[1] * diff_perp;
      K(1,0) = K(0,1);
      K(1,1) = (1.0 - bHat_[1] * bHat_[1]) * diff_perp;
   }
   else
   {
      K(0,0) = (1.0 - bHat_[0] * bHat_[0]) * diff_perp;
      K(0,1) = -bHat_[0] * bHat_[1] * diff_perp;
      K(0,2) = -bHat_[0] * bHat_[2] * diff_perp;
      K(1,0) = K(0,1);
      K(1,1) = (1.0 - bHat_[1] * bHat_[1]) * diff_perp;
      K(1,2) = -bHat_[1] * bHat_[2] * diff_perp;
      K(2,0) = K(0,2);
      K(2,1) = K(1,2);
      K(2,2) = (1.0 - bHat_[2] * bHat_[2]) * diff_perp;

      if (diff_cross != 0.0)
      {
         K(1,2) -= bHat_[0] * diff_cross;
         K(2,0) -= bHat_[1] * diff_cross;
         K(0,1) -= bHat_[2] * diff_cross;
         K(2,1) += bHat_[0] * diff_cross;
         K(0,2) += bHat_[1] * diff_cross;
         K(1,0) += bHat_[2] * diff_cross;
      }
   }
}

ChiParaCoefficient::ChiParaCoefficient(BlockVector & nBV, double zi)
   : nBV_(nBV),
     sfes_(NULL),
     nCoef_(&nGF_),
     ion_(false),
     zi_(zi),
     m_(me_u_),
     ni_(-1.0)
{}

ChiParaCoefficient::ChiParaCoefficient(BlockVector & nBV,
                                       double zi, double m)
   : nBV_(nBV),
     sfes_(NULL),
     nCoef_(&nGF_),
     ion_(true),
     zi_(zi),
     m_(m),
     ne_(-1.0),
     ni_(-1.0)
{}

void ChiParaCoefficient::SetT(ParGridFunction & T)
{
   sfes_ = T.ParFESpace();
   TCoef_.SetGridFunction(&T);
   nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
}

double
ChiParaCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
   double temp = TCoef_.Eval(T, ip);

   ni_ = nCoef_.Eval(T, ip);

   if (ion_)
   {
      return chi_i_para(m_, zi_, ni_, temp);
   }
   else
   {
      nGF_.MakeRef(sfes_, nBV_.GetBlock(0));
      ne_ = nCoef_.Eval(T, ip);
      nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
      return chi_e_para(ne_, temp, zi_, ni_);
   }
}

ChiPerpCoefficient::ChiPerpCoefficient(BlockVector & nBV, double zi)
   : nBV_(nBV),
     sfes_(NULL),
     nCoef_(&nGF_),
     ion_(false),
     mi_(-1.0),
     zi_(zi),
     ne_(-1.0),
     ni_(-1.0)
{}

ChiPerpCoefficient::ChiPerpCoefficient(BlockVector & nBV,
                                       double mi, double zi)
   : nBV_(nBV),
     sfes_(NULL),
     nCoef_(&nGF_),
     ion_(true),
     mi_(mi),
     zi_(zi),
     ne_(-1.0),
     ni_(-1.0)
{}

void ChiPerpCoefficient::SetT(ParGridFunction & T)
{
   sfes_ = T.ParFESpace();
   TCoef_.SetGridFunction(&T);
   nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
}

void ChiPerpCoefficient::SetB(ParGridFunction & B)
{
   BCoef_.SetGridFunction(&B);
}

double
ChiPerpCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
   BCoef_.Eval(B_, T, ip);
   double Bmag = B_.Norml2();
   double temp = TCoef_.Eval(T, ip);

   ni_ = nCoef_.Eval(T, ip);

   if (ion_)
   {
      return chi_i_perp(Bmag, mi_, zi_, ni_, temp);
   }
   else
   {
      nGF_.MakeRef(sfes_, nBV_.GetBlock(0));
      ne_ = nCoef_.Eval(T, ip);
      nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
      return chi_e_perp(Bmag, ne_, temp, zi_, ni_);
   }
}

ChiCrossCoefficient::ChiCrossCoefficient(BlockVector & nBV, double zi)
   : nBV_(nBV),
     sfes_(NULL),
     nCoef_(&nGF_),
     ion_(false),
     mi_(-1.0),
     zi_(zi),
     ne_(-1.0),
     ni_(-1.0)
{}

ChiCrossCoefficient::ChiCrossCoefficient(BlockVector & nBV,
                                         double mi, double zi)
   : nBV_(nBV),
     sfes_(NULL),
     nCoef_(&nGF_),
     ion_(true),
     mi_(mi),
     zi_(zi),
     ne_(-1.0),
     ni_(-1.0)
{}

void ChiCrossCoefficient::SetT(ParGridFunction & T)
{
   sfes_ = T.ParFESpace();
   TCoef_.SetGridFunction(&T);
   nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
}

void ChiCrossCoefficient::SetB(ParGridFunction & B)
{
   BCoef_.SetGridFunction(&B);
}

double
ChiCrossCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
   BCoef_.Eval(B_, T, ip);
   double Bmag = B_.Norml2();
   double temp = TCoef_.Eval(T, ip);

   ni_ = nCoef_.Eval(T, ip);

   if (ion_)
   {
      return chi_i_cross(Bmag, mi_, zi_, ni_, temp);
   }
   else
   {
      nGF_.MakeRef(sfes_, nBV_.GetBlock(0));
      ne_ = nCoef_.Eval(T, ip);
      nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
      return chi_e_cross(Bmag, ne_, temp, ni_, zi_);
   }
}

ChiCoefficient::ChiCoefficient(int dim, BlockVector & nBV, ParGridFunction & B,
                               double zi)
   : MatrixCoefficient(dim),
     nBV_(nBV),
     sfes_(NULL),
     nCoef_(&nGF_),
     BCoef_(&B),
     ion_(false),
     zi_(zi),
     mi_(-1.0),
     ne_(-1.0),
     ni_(-1.0),
     bHat_(dim)
{}

ChiCoefficient::ChiCoefficient(int dim, BlockVector & nBV, ParGridFunction & B,
                               double mi, double zi)
   : MatrixCoefficient(dim),
     nBV_(nBV),
     sfes_(NULL),
     nCoef_(&nGF_),
     BCoef_(&B),
     ion_(true),
     zi_(zi),
     mi_(mi),
     ne_(-1.0),
     ni_(-1.0),
     bHat_(dim)
{}

void ChiCoefficient::SetT(ParGridFunction & T)
{
   sfes_ = T.ParFESpace();
   TCoef_.SetGridFunction(&T);
   nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
}

void ChiCoefficient::SetB(ParGridFunction & B)
{
   BCoef_.SetGridFunction(&B);
}

void ChiCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                          const IntegrationPoint &ip)
{
   BCoef_.Eval(bHat_, T, ip);
   double bMag = bHat_.Norml2();;
   bHat_ /= bMag;

   double temp = TCoef_.Eval(T, ip);

   ni_ = nCoef_.Eval(T, ip);

   if (!ion_)
   {
      nGF_.MakeRef(sfes_, nBV_.GetBlock(0));
      ne_ = nCoef_.Eval(T, ip);
      nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
   }

   double chi_para = (ion_) ?
                     chi_i_para(mi_, zi_, ni_, temp) :
                     chi_e_para(ne_, temp, zi_, ni_);

   double chi_perp = (ion_) ?
                     chi_i_perp(bMag, mi_, zi_, ni_, temp) :
                     chi_e_perp(bMag, ne_, temp, zi_, ni_);

   double chi_cross = 0.0;

   K.SetSize(bHat_.Size());

   if (width == 2)
   {
      K(0,0) = bHat_[0] * bHat_[0] * (chi_para - chi_perp) + chi_perp;
      K(0,1) = bHat_[0] * bHat_[1] * (chi_para - chi_perp);
      K(1,0) = K(0,1);
      K(1,1) = bHat_[1] * bHat_[1] * (chi_para - chi_perp) + chi_perp;
   }
   else
   {
      chi_cross = (ion_) ?
                  chi_i_cross(bMag, mi_, zi_, ni_, temp) :
                  chi_e_cross(bMag, ne_, temp, zi_, ni_);

      K(0,0) = bHat_[0] * bHat_[0] * (chi_para - chi_perp) + chi_perp;
      K(0,1) = bHat_[0] * bHat_[1] * (chi_para - chi_perp);
      K(0,2) = bHat_[0] * bHat_[2] * (chi_para - chi_perp);
      K(1,0) = K(0,1);
      K(1,1) = bHat_[1] * bHat_[1] * (chi_para - chi_perp) + chi_perp;
      K(1,2) = bHat_[1] * bHat_[2] * (chi_para - chi_perp);
      K(2,0) = K(0,2);
      K(2,1) = K(1,2);
      K(2,2) = bHat_[2] * bHat_[2] * (chi_para - chi_perp) + chi_perp;

      if (chi_cross != 0.0)
      {
         K(1,2) -= bHat_[0] * chi_cross;
         K(2,0) -= bHat_[1] * chi_cross;
         K(0,1) -= bHat_[2] * chi_cross;
         K(2,1) += bHat_[0] * chi_cross;
         K(0,2) += bHat_[1] * chi_cross;
         K(1,0) += bHat_[2] * chi_cross;
      }
   }
}

Eta0Coefficient::Eta0Coefficient(BlockVector & nBV, double zi)
   : nBV_(nBV),
     nCoef_(&nGF_),
     ion_(false),
     zi_(zi),
     mi_(-1.0),
     ne_(-1.0),
     ni_(-1.0)
{}

Eta0Coefficient::Eta0Coefficient(BlockVector & nBV, double mi, double zi)
   : nBV_(nBV),
     nCoef_(&nGF_),
     ion_(true),
     zi_(zi),
     mi_(mi),
     ne_(-1.0),
     ni_(-1.0)
{}

void Eta0Coefficient::SetT(ParGridFunction & T)
{
   sfes_ = T.ParFESpace();
   TCoef_.SetGridFunction(&T);
   nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
}

double
Eta0Coefficient::Eval(ElementTransformation &T,
                      const IntegrationPoint &ip)
{
   double temp = TCoef_.Eval(T, ip);

   ni_ = nCoef_.Eval(T, ip);

   if (!ion_)
   {
      nGF_.MakeRef(sfes_, nBV_.GetBlock(0));
      ne_ = nCoef_.Eval(T, ip);
      nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
   }

   double eta0 = (ion_) ?
                 eta0_i(mi_, zi_, ni_, temp) :
                 eta0_e(ne_, temp, zi_, ni_);

   return eta0;
}

Eta1Coefficient::Eta1Coefficient(BlockVector & nBV, double zi)
   : nBV_(nBV),
     nCoef_(&nGF_),
     ion_(false),
     zi_(zi),
     mi_(-1.0),
     ne_(-1.0),
     ni_(-1.0)
{}

Eta1Coefficient::Eta1Coefficient(BlockVector & nBV, double mi, double zi)
   : nBV_(nBV),
     nCoef_(&nGF_),
     ion_(true),
     zi_(zi),
     mi_(mi),
     ne_(-1.0),
     ni_(-1.0)
{}

void Eta1Coefficient::SetT(ParGridFunction & T)
{
   sfes_ = T.ParFESpace();
   TCoef_.SetGridFunction(&T);
   nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
}

void Eta1Coefficient::SetB(ParGridFunction & B)
{
   BCoef_.SetGridFunction(&B);
}

double
Eta1Coefficient::Eval(ElementTransformation &T,
                      const IntegrationPoint &ip)
{
   BCoef_.Eval(bHat_, T, ip);
   double bMag = bHat_.Norml2();;
   double temp = TCoef_.Eval(T, ip);

   ni_ = nCoef_.Eval(T, ip);

   if (!ion_)
   {
      nGF_.MakeRef(sfes_, nBV_.GetBlock(0));
      ne_ = nCoef_.Eval(T, ip);
      nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
   }

   double eta1 = (ion_) ?
                 eta1_i(bMag, mi_, zi_, ni_, temp) :
                 eta1_e(bMag, ne_, temp, zi_, ni_);

   return eta1;
}

Eta3Coefficient::Eta3Coefficient(BlockVector & nBV, double zi)
   : nBV_(nBV),
     nCoef_(&nGF_),
     ion_(false),
     zi_(zi),
     mi_(-1.0),
     ne_(-1.0),
     ni_(-1.0)
{}

Eta3Coefficient::Eta3Coefficient(BlockVector & nBV, double mi, double zi)
   : nBV_(nBV),
     nCoef_(&nGF_),
     ion_(true),
     zi_(zi),
     mi_(mi),
     ne_(-1.0),
     ni_(-1.0)
{}

void Eta3Coefficient::SetT(ParGridFunction & T)
{
   sfes_ = T.ParFESpace();
   TCoef_.SetGridFunction(&T);
   nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
}

void Eta3Coefficient::SetB(ParGridFunction & B)
{
   BCoef_.SetGridFunction(&B);
}

double
Eta3Coefficient::Eval(ElementTransformation &T,
                      const IntegrationPoint &ip)
{
   BCoef_.Eval(bHat_, T, ip);
   double bMag = bHat_.Norml2();;
   double temp = TCoef_.Eval(T, ip);

   ni_ = nCoef_.Eval(T, ip);

   if (!ion_)
   {
      nGF_.MakeRef(sfes_, nBV_.GetBlock(0));
      ne_ = nCoef_.Eval(T, ip);
      nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
   }

   double eta3 = (ion_) ?
                 eta3_i(bMag, mi_, zi_, ni_, temp) :
                 eta3_e(bMag, ne_, temp);

   return eta3;
}

EtaCoefficient::EtaCoefficient(int dim, int bi, int bj,
                               BlockVector & nBV, ParGridFunction & B,
                               double zi)
   : MatrixCoefficient(dim),
     nBV_(nBV),
     nCoef_(&nGF_),
     BCoef_(&B),
     del_(dim),
     eps_(dim, dim, dim),
     bi_(bi),
     bj_(bj),
     ion_(false),
     zi_(zi),
     mi_(-1.0),
     ne_(-1.0),
     ni_(-1.0),
     bHat_(dim),
     bx_(dim)
{
   this->initSymbols();
}

EtaCoefficient::EtaCoefficient(int dim, int bi, int bj,
                               BlockVector & nBV, ParGridFunction & B,
                               double mi, double zi)
   : MatrixCoefficient(dim),
     nBV_(nBV),
     nCoef_(&nGF_),
     BCoef_(&B),
     del_(dim),
     eps_(dim, dim, dim),
     bi_(bi),
     bj_(bj),
     ion_(true),
     zi_(zi),
     mi_(mi),
     ne_(-1.0),
     ni_(-1.0),
     bHat_(dim),
     bx_(dim)
{
   this->initSymbols();
}

void EtaCoefficient::initSymbols()
{
   int dim = del_.Size();

   del_ = 0.0;
   for (int i=0; i<dim; i++) { del_(i,i) = 1.0; }

   eps_ = 0.0;
   if (dim == 3)
   {
      eps_(0,1,2) = 1.0;
      eps_(1,2,0) = 1.0;
      eps_(2,0,1) = 1.0;
      eps_(2,1,0) = -1.0;
      eps_(1,0,2) = -1.0;
      eps_(0,2,1) = -1.0;
   }
}

void EtaCoefficient::SetT(ParGridFunction & T)
{
   sfes_ = T.ParFESpace();
   TCoef_.SetGridFunction(&T);
   nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
}

void EtaCoefficient::SetB(ParGridFunction & B)
{
   BCoef_.SetGridFunction(&B);
}

void
EtaCoefficient::Eval(DenseMatrix & K, ElementTransformation &T,
                     const IntegrationPoint &ip)
{
   BCoef_.Eval(bHat_, T, ip);
   double bMag = bHat_.Norml2();
   bHat_ /= bMag;

   double temp = TCoef_.Eval(T, ip);

   ni_ = nCoef_.Eval(T, ip);

   if (!ion_)
   {
      nGF_.MakeRef(sfes_, nBV_.GetBlock(0));
      ne_ = nCoef_.Eval(T, ip);
      nGF_.MakeRef(sfes_, nBV_.GetBlock(1));
   }

   double eta0 = (ion_) ?
                 eta0_i(mi_, zi_, ni_, temp) :
                 eta0_e(ne_, temp, zi_, ni_);

   double eta1 = (ion_) ?
                 eta1_i(bMag, mi_, zi_, ni_, temp) :
                 eta1_e(bMag, ne_, temp, zi_, ni_);

   double eta2 = (ion_) ?
                 eta2_i(bMag, mi_, zi_, ni_, temp) :
                 eta2_e(bMag, ne_, temp, zi_, ni_);

   double eta3 = 0.0;

   double eta4 = 0.0;

   if (width == 3)
   {
      eta3 = (ion_) ?
             eta3_i(bMag, mi_, zi_, ni_, temp) :
             eta3_e(bMag, ne_, temp);

      eta4 = (ion_) ?
             eta4_i(bMag, mi_, zi_, ni_, temp) :
             eta4_e(bMag, ne_, temp);

      bx_ = 0.0;
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            for (int m=0; m<3; m++)
            {
               bx_(k, l) += eps_(k, l, m) * bHat_[m];
            }
         }
      }
   }

   K.SetSize(width);

   K = 0.0;

   if (width == 2)
   {
      for (int k=0; k<2; k++)
      {
         for (int l=0; l<2; l++)
         {
            K(k, l) += 2.0 * eta0 *
                       (0.5 * del_(bi_, bj_) - bHat_[bi_] * bHat_[bj_]) *
                       (0.5 * del_(k, l) - bHat_[k] * bHat_[l]);

            K(k, l) += eta1 * ((del_(bi_, k) - bHat_[bi_] * bHat_[k]) *
                               (del_(bj_, l) - bHat_[bj_] * bHat_[l]) +
                               (del_(bi_, l) - bHat_[bi_] * bHat_[l]) *
                               (del_(bj_, k) - bHat_[bj_] * bHat_[k]) -
                               (del_(bi_, bj_) - bHat_[bi_] * bHat_[bj_]) *
                               (del_(k, l) - bHat_[k] * bHat_[l]));

            K(k, l) += eta2 * ((del_(bi_, k) - bHat_[bi_] * bHat_[k]) *
                               bHat_[bj_] * bHat_[l] +
                               (del_(bi_, l) - bHat_[bi_] * bHat_[l]) *
                               bHat_[bj_] * bHat_[k] +
                               (del_(bj_, k) - bHat_[bj_] * bHat_[k]) *
                               bHat_[bi_] * bHat_[l] +
                               (del_(bj_, l) - bHat_[bj_] * bHat_[l]) *
                               bHat_[bi_] * bHat_[k]);
         }
      }
   }
   else
   {
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            K(k, l) += 3.0 * eta0 *
                       (del_(bi_, bj_) / 3.0 - bHat_[bi_] * bHat_[bj_]) *
                       (del_(k, l) / 3.0 - bHat_[k] * bHat_[l]);

            K(k, l) += eta1 * ((del_(bi_, k) - bHat_[bi_] * bHat_[k]) *
                               (del_(bj_, l) - bHat_[bj_] * bHat_[l]) +
                               (del_(bi_, l) - bHat_[bi_] * bHat_[l]) *
                               (del_(bj_, k) - bHat_[bj_] * bHat_[k]) -
                               (del_(bi_, bj_) - bHat_[bi_] * bHat_[bj_]) *
                               (del_(k, l) - bHat_[k] * bHat_[l]));

            K(k, l) += eta2 * ((del_(bi_, k) - bHat_[bi_] * bHat_[k]) *
                               bHat_[bj_] * bHat_[l] +
                               (del_(bi_, l) - bHat_[bi_] * bHat_[l]) *
                               bHat_[bj_] * bHat_[k] +
                               (del_(bj_, k) - bHat_[bj_] * bHat_[k]) *
                               bHat_[bi_] * bHat_[l] +
                               (del_(bj_, l) - bHat_[bj_] * bHat_[l]) *
                               bHat_[bi_] * bHat_[k]);

            K(k, l) += eta3 * ((del_(bi_, k) - bHat_[bi_] * bHat_[k]) *
                               bx_(bj_, l) +
                               (del_(bj_, k) - bHat_[bj_] * bHat_[k]) *
                               bx_(bi_, l) -
                               (del_(bi_, bj_) - bHat_[bi_] * bHat_[bj_]) *
                               bx_(k, l));

            K(k, l) += eta4 * (bx_(bi_, k) * bHat_[bj_] * bHat_[l] +
                               bx_(bi_, l) * bHat_[bj_] * bHat_[k] +
                               bx_(bj_, k) * bHat_[bi_] * bHat_[l] +
                               bx_(bj_, l) * bHat_[bi_] * bHat_[k]);
         }
      }
   }
}

dpdnCoefficient::dpdnCoefficient(int c,
                                 double m,
                                 VectorCoefficient & uCoef)
   : c_(c),
     m_(m),
     uCoef_(uCoef),
     u_(uCoef.GetVDim())
{
}

double dpdnCoefficient::Eval(ElementTransformation &T,
                             const IntegrationPoint &ip)
{
   uCoef_.Eval(u_, T, ip);

   return m_ * u_[c_];
}

dpduCoefficient::dpduCoefficient(double m,
                                 Coefficient & nCoef)
   : m_(m),
     nCoef_(nCoef)
{
}

double dpduCoefficient::Eval(ElementTransformation &T,
                             const IntegrationPoint &ip)
{
   return m_ * nCoef_.Eval(T, ip);
}

dEdnCoefficient::dEdnCoefficient(Coefficient & TCoef,
                                 double m,
                                 VectorCoefficient & uCoef)
   : TCoef_(TCoef),
     uCoef_(uCoef),
     m_(m),
     u_(uCoef.GetVDim())
{}

double
dEdnCoefficient::Eval(ElementTransformation &T,
                      const IntegrationPoint &ip)
{
   double temp = TCoef_.Eval(T, ip);
   uCoef_.Eval(u_, T, ip);

   return 1.5 * temp + 0.5 * m_ * (u_ * u_);
}

dEduCoefficient::dEduCoefficient(int c,
                                 double m,
                                 Coefficient & nCoef,
                                 VectorCoefficient & uCoef)
   : c_(c),
     m_(m),
     nCoef_(nCoef),
     uCoef_(uCoef),
     u_(uCoef.GetVDim())
{
}

double dEduCoefficient::Eval(ElementTransformation &T,
                             const IntegrationPoint &ip)
{
   double n = nCoef_.Eval(T, ip);
   uCoef_.Eval(u_, T, ip);

   return m_ * n * u_[c_];
}

TwoFluidTransportSolver::TwoFluidTransportSolver(ODESolver * implicitSolver,
                                                 ODESolver * explicitSolver,
                                                 DGParams & dg,
                                                 ParFiniteElementSpace & sfes,
                                                 ParFiniteElementSpace & vfes,
                                                 ParFiniteElementSpace & ffes,
                                                 Array<int> & offsets,
                                                 BlockVector & nBV,
                                                 BlockVector & uBV,
                                                 BlockVector & TBV,
                                                 ParGridFunction & B,
                                                 double ion_mass,
                                                 double ion_charge)
   : impSolver_(implicitSolver),
     expSolver_(explicitSolver),
     dg_(dg),
     sfes_(sfes),
     vfes_(vfes),
     ffes_(ffes),
     offsets_(offsets),
     nBV_(nBV),
     uBV_(uBV),
     TBV_(TBV),
     B_(B),
     ion_mass_(ion_mass),
     ion_charge_(ion_charge),
     tfDiff_(NULL)
{
   this->initDiffusion();
}

TwoFluidTransportSolver::~TwoFluidTransportSolver()
{
   delete tfDiff_;
}

void TwoFluidTransportSolver::initDiffusion()
{
   tfDiff_ = new TwoFluidDiffusion(dg_, sfes_, vfes_,
                                   offsets_, nBV_, uBV_, TBV_,
                                   B_, ion_mass_, ion_charge_);
   impSolver_->Init(*tfDiff_);
}

void TwoFluidTransportSolver::Update()
{
   tfDiff_->Update();
}

void TwoFluidTransportSolver::Step(Vector &x, double &t, double &dt)
{
   impSolver_->Step(x, t, dt);
}

TwoFluidDiffusion::TwoFluidDiffusion(DGParams & dg,
                                     ParFiniteElementSpace & sfes,
                                     ParFiniteElementSpace & vfes,
                                     Array<int> & offsets,
                                     BlockVector & nBV,
                                     BlockVector & uBV,
                                     BlockVector & TBV,
                                     ParGridFunction & B,
                                     double ion_mass,
                                     double ion_charge)
   : TimeDependentOperator(offsets.Last(), 0.0, IMPLICIT),
     dim_(sfes.GetParMesh()->SpaceDimension()),
     dg_(dg),
     sfes_(sfes),
     vfes_(vfes),
     offsets_(offsets),
     nBV_(nBV),
     uBV_(uBV),
     TBV_(TBV),
     B_(B),
     ion_mass_(ion_mass),
     ion_charge_(ion_charge),
     block_A_(offsets_),
     block_B_(offsets_),
     block_rhs_(offsets_),
     block_amg_(offsets_),
     gmres_(sfes.GetComm())
{
   this->initCoefficients();
   this->initBilinearForms();
}

TwoFluidDiffusion::~TwoFluidDiffusion()
{
   this->deleteBilinearForms();
   this->deleteCoefficients();
}

void TwoFluidDiffusion::deleteBilinearForms()
{
   for (unsigned int i=0; i<a_dndn_.size(); i++)
   {
      delete a_dndn_[i];
   }
   /*
   for (unsigned int i=0; i<stiff_D_.size(); i++)
   {
      delete stiff_D_[i];
   }
   */

   for (unsigned int i=0; i<a_dpdn_.size(); i++)
   {
      delete a_dpdn_[i];
   }
   for (unsigned int i=0; i<a_dpdu_.size(); i++)
   {
      delete a_dpdu_[i];
   }
   for (unsigned int i=0; i<stiff_eta_.size(); i++)
   {
      delete stiff_eta_[i];
   }

   for (unsigned int i=0; i<a_dEdn_.size(); i++)
   {
      delete a_dEdn_[i];
   }
   for (unsigned int i=0; i<a_dEdu_.size(); i++)
   {
      delete a_dEdu_[i];
   }
   for (unsigned int i=0; i<a_dEdT_.size(); i++)
   {
      delete a_dEdT_[i];
   }
   for (unsigned int i=0; i<stiff_chi_.size(); i++)
   {
      delete stiff_chi_[i];
   }
}

void TwoFluidDiffusion::deleteCoefficients()
{
   for (unsigned int i=0; i<dndnCoef_.size(); i++)
   {
      delete dndnCoef_[i];
   }

   for (unsigned int i=0; i<dpdnCoef_.size(); i++)
   {
      delete dpdnCoef_[i];
   }
   for (unsigned int i=0; i<dpduCoef_.size(); i++)
   {
      delete dpduCoef_[i];
   }

   for (unsigned int i=0; i<dEdnCoef_.size(); i++)
   {
      delete dEdnCoef_[i];
   }
   for (unsigned int i=0; i<dEduCoef_.size(); i++)
   {
      delete dEduCoef_[i];
   }
   for (unsigned int i=0; i<dEdTCoef_.size(); i++)
   {
      delete dEdTCoef_[i];
   }

   /*
   for (unsigned int i=0; i<dtDiffCoef_.size(); i++)
   {
      delete dtDiffCoef_[i];
   }
   */
   for (unsigned int i=0; i<dtChiCoef_.size(); i++)
   {
      delete dtChiCoef_[i];
   }
   for (unsigned int i=0; i<dtEtaCoef_.size(); i++)
   {
      delete dtEtaCoef_[i];
   }

   /*
   for (unsigned int i=0; i<diffCoef_.size(); i++)
   {
      delete diffCoef_[i];
   }
   */
   for (unsigned int i=0; i<chiCoef_.size(); i++)
   {
      delete chiCoef_[i];
   }
   for (unsigned int i=0; i<etaCoef_.size(); i++)
   {
      delete etaCoef_[i];
   }
}

void TwoFluidDiffusion::initCoefficients()
{
   int ns = 1;

   nGF_.resize(ns + 1);
   nCoef_.resize(ns + 1);
   for (int i=0; i<=ns; i++)
   {
      nGF_[i].MakeRef(&sfes_, nBV_.GetBlock(i));
      nCoef_[i].SetGridFunction(&nGF_[i]);
   }

   uGF_.resize(ns + 1);
   uCoef_.resize(ns + 1);
   for (int i=0; i<=ns; i++)
   {
      uGF_[i].MakeRef(&vfes_, uBV_.GetBlock(i));
      uCoef_[i].SetGridFunction(&uGF_[i]);
   }

   TGF_.resize(ns + 1);
   TCoef_.resize(ns + 1);
   for (int i=0; i<=ns; i++)
   {
      TGF_[i].MakeRef(&sfes_, TBV_.GetBlock(i));
      TCoef_[i].SetGridFunction(&TGF_[i]);
   }

   dndnCoef_.resize(ns + 1);
   dndnCoef_[0] = new ConstantCoefficient(1.0);
   dndnCoef_[1] = new ConstantCoefficient(1.0);

   dpdnCoef_.resize(dim_ * (ns + 1));
   for (int d=0; d<dim_; d++)
   {
      dpdnCoef_[d] = new dpdnCoefficient(d, me_u_, uCoef_[0]);
      dpdnCoef_[dim_ + d] = new dpdnCoefficient(d, ion_mass_, uCoef_[1]);
   }

   dpduCoef_.resize(dim_ * (ns + 1));
   for (int d=0; d<dim_; d++)
   {
      dpduCoef_[d] = new dpduCoefficient(me_u_, nCoef_[0]);
      dpduCoef_[dim_ + d] = new dpduCoefficient(ion_mass_, nCoef_[1]);
   }

   dEdnCoef_.resize(ns + 1);
   dEdnCoef_[0] = new dEdnCoefficient(TCoef_[0], me_u_, uCoef_[0]);
   dEdnCoef_[1] = new dEdnCoefficient(TCoef_[1], ion_mass_, uCoef_[1]);

   dEduCoef_.resize(dim_ * (ns + 1));
   for (int d=0; d<dim_; d++)
   {
      dEduCoef_[d] = new dEduCoefficient(d, me_u_,
                                         nCoef_[0], uCoef_[0]);
   }
   for (int d=0; d<dim_; d++)
   {
      dEduCoef_[dim_ + d] = new dEduCoefficient(d, ion_mass_,
                                                nCoef_[1], uCoef_[1]);
   }

   dEdTCoef_.resize(ns + 1);
   for (int i=0; i<=ns; i++)
   {
      dEdTCoef_[i] = new dEdTCoefficient(1.5, nCoef_[i]);
   }
   /*
   diffCoef_.resize(ns);
   dtDiffCoef_.resize(ns);
   diffCoef_[0] = new DiffCoefficient(dim_, B_);
   dtDiffCoef_[0] = new ScalarMatrixProductCoefficient(0.0, *diffCoef_[0]);
   */
   chiCoef_.resize(ns + 1);
   dtChiCoef_.resize(ns + 1);

   chiCoef_[0] = new ChiCoefficient(dim_, nBV_, B_, ion_charge_);
   chiCoef_[1] = new ChiCoefficient(dim_, nBV_, B_, ion_mass_, ion_charge_);

   chiCoef_[0]->SetT(TGF_[0]);
   chiCoef_[1]->SetT(TGF_[1]);

   dtChiCoef_[0] = new ScalarMatrixProductCoefficient(0.0, *chiCoef_[0]);
   dtChiCoef_[1] = new ScalarMatrixProductCoefficient(0.0, *chiCoef_[1]);

   etaCoef_.resize(dim_ * dim_ * (ns + 1));
   dtEtaCoef_.resize(dim_ * dim_ * (ns + 1));
   for (int i=0; i<dim_; i++)
   {
      for (int j=0; j<dim_; j++)
      {
         int k = dim_ * i + j;
         etaCoef_[k] = new EtaCoefficient(dim_, i, j, nBV_, B_, ion_charge_);
         etaCoef_[k]->SetT(TGF_[0]);
         dtEtaCoef_[k] = new ScalarMatrixProductCoefficient(0.0, *etaCoef_[k]);
      }
   }
   for (int i=0; i<dim_; i++)
   {
      for (int j=0; j<dim_; j++)
      {
         int k = dim_ * (dim_ + i) + j;
         etaCoef_[k] = new EtaCoefficient(dim_, i, j, nBV_, B_,
                                          ion_mass_, ion_charge_);
         etaCoef_[k]->SetT(TGF_[1]);
         dtEtaCoef_[k] = new ScalarMatrixProductCoefficient(0.0, *etaCoef_[k]);
      }
   }
}

void TwoFluidDiffusion::initBilinearForms()
{
   a_dndn_.resize(dndnCoef_.size());
   for (unsigned int i=0; i<dndnCoef_.size(); i++)
   {
      a_dndn_[i] = new ParBilinearForm(&sfes_);
      a_dndn_[i]->AddDomainIntegrator(new MassIntegrator(*dndnCoef_[i]));
   }
   /*
   stiff_D_.resize(diffCoef_.size());
   for (unsigned int i=0; i<diffCoef_.size(); i++)
   {
      stiff_D_[i] = new ParBilinearForm(&sfes_);
      stiff_D_[i]->AddDomainIntegrator(new DiffusionIntegrator(*diffCoef_[i]));
      stiff_D_[i]->AddInteriorFaceIntegrator(
         new DGDiffusionIntegrator(*diffCoef_[i], dg_.sigma, dg_.kappa));
      stiff_D_[i]->AddBdrFaceIntegrator(
         new DGDiffusionIntegrator(*diffCoef_[i], dg_.sigma, dg_.kappa));
   }
   */

   a_dpdn_.resize(dpdnCoef_.size());
   for (unsigned int i=0; i<dpdnCoef_.size(); i++)
   {
      a_dpdn_[i] = new ParBilinearForm(&sfes_);
      a_dpdn_[i]->AddDomainIntegrator(new MassIntegrator(*dpdnCoef_[i]));
   }

   a_dpdu_.resize(etaCoef_.size());
   for (unsigned int i=0; i<etaCoef_.size(); i++)
   {
      a_dpdu_[i] = new ParBilinearForm(&sfes_);
      if ((i % (dim_ * dim_)) % (dim_ + 1) == 0)
      {
         int spec = i / (dim_ * dim_);
         int row = dim_ * spec + (i %  (dim_ * dim_)) / dim_;
         a_dpdu_[i]->AddDomainIntegrator(new MassIntegrator(*dpduCoef_[row]));
      }
      a_dpdu_[i]->AddDomainIntegrator(new DiffusionIntegrator(*dtEtaCoef_[i]));
      a_dpdu_[i]->AddInteriorFaceIntegrator(
         new DGDiffusionIntegrator(*dtEtaCoef_[i], dg_.sigma, dg_.kappa));
      a_dpdu_[i]->AddBdrFaceIntegrator(
         new DGDiffusionIntegrator(*dtEtaCoef_[i], dg_.sigma, dg_.kappa));
   }

   stiff_eta_.resize(etaCoef_.size());
   for (unsigned int i=0; i<etaCoef_.size(); i++)
   {
      stiff_eta_[i] = new ParBilinearForm(&sfes_);
      stiff_eta_[i]->AddDomainIntegrator(new DiffusionIntegrator(*etaCoef_[i]));
      stiff_eta_[i]->AddInteriorFaceIntegrator(
         new DGDiffusionIntegrator(*etaCoef_[i], dg_.sigma, dg_.kappa));
      stiff_eta_[i]->AddBdrFaceIntegrator(
         new DGDiffusionIntegrator(*etaCoef_[i], dg_.sigma, dg_.kappa));
   }

   a_dEdn_.resize(dEdnCoef_.size());
   for (unsigned int i=0; i<dEdnCoef_.size(); i++)
   {
      a_dEdn_[i] = new ParBilinearForm(&sfes_);
      a_dEdn_[i]->AddDomainIntegrator(new MassIntegrator(*dEdnCoef_[i]));
   }

   a_dEdu_.resize(dEduCoef_.size());
   for (unsigned int i=0; i<dEduCoef_.size(); i++)
   {
      a_dEdu_[i] = new ParBilinearForm(&sfes_);
      a_dEdu_[i]->AddDomainIntegrator(new MassIntegrator(*dEduCoef_[i]));
   }

   a_dEdT_.resize(dEdTCoef_.size());
   for (unsigned int i=0; i<dEdTCoef_.size(); i++)
   {
      a_dEdT_[i] = new ParBilinearForm(&sfes_);
      a_dEdT_[i]->AddDomainIntegrator(new MassIntegrator(*dEdTCoef_[i]));
      a_dEdT_[i]->AddDomainIntegrator(new DiffusionIntegrator(*dtChiCoef_[i]));
      a_dEdT_[i]->AddInteriorFaceIntegrator(
         new DGDiffusionIntegrator(*dtChiCoef_[i], dg_.sigma, dg_.kappa));
      a_dEdT_[i]->AddBdrFaceIntegrator(
         new DGDiffusionIntegrator(*dtChiCoef_[i], dg_.sigma, dg_.kappa));
   }

   stiff_chi_.resize(chiCoef_.size());
   for (unsigned int i=0; i<chiCoef_.size(); i++)
   {
      stiff_chi_[i] = new ParBilinearForm(&sfes_);
      stiff_chi_[i]->AddDomainIntegrator(new DiffusionIntegrator(*chiCoef_[i]));
      stiff_chi_[i]->AddInteriorFaceIntegrator(
         new DGDiffusionIntegrator(*chiCoef_[i], dg_.sigma, dg_.kappa));
      stiff_chi_[i]->AddBdrFaceIntegrator(
         new DGDiffusionIntegrator(*chiCoef_[i], dg_.sigma, dg_.kappa));
   }
}

void TwoFluidDiffusion::setTimeStep(double dt)
{
   /*
   for (unsigned int i=0; i<dtDiffCoef_.size(); i++)
   {
     dtDiffCoef_[i]->SetAConst(dt);
   }
   */
   for (unsigned int i=0; i<dtChiCoef_.size(); i++)
   {
      dtChiCoef_[i]->SetAConst(dt);
   }
   for (unsigned int i=0; i<dtEtaCoef_.size(); i++)
   {
      dtEtaCoef_[i]->SetAConst(dt);
   }

}

void TwoFluidDiffusion::Assemble()
{
   for (unsigned int i=0; i<a_dndn_.size(); i++)
   {
      a_dndn_[i]->Assemble();
      a_dndn_[i]->Finalize();
   }

   for (unsigned int i=0; i<a_dpdn_.size(); i++)
   {
      a_dpdn_[i]->Assemble();
      a_dpdn_[i]->Finalize();
   }
   for (unsigned int i=0; i<a_dpdu_.size(); i++)
   {
      a_dpdu_[i]->Assemble();
      a_dpdu_[i]->Finalize();
   }

   for (unsigned int i=0; i<a_dEdn_.size(); i++)
   {
      a_dEdn_[i]->Assemble();
      a_dEdn_[i]->Finalize();
   }
   for (unsigned int i=0; i<a_dEdu_.size(); i++)
   {
      a_dEdu_[i]->Assemble();
      a_dEdu_[i]->Finalize();
   }
   for (unsigned int i=0; i<a_dEdT_.size(); i++)
   {
      a_dEdT_[i]->Assemble();
      a_dEdT_[i]->Finalize();
   }
   /*
   for (unsigned int i=0; i<stiff_D_.size(); i++)
   {
      delete stiff_D_[i]->Assemble();
      delete stiff_D_[i]->Finalize();
   }
   */
   for (unsigned int i=0; i<stiff_chi_.size(); i++)
   {
      stiff_chi_[i]->Assemble();
      stiff_chi_[i]->Finalize();
   }
   for (unsigned int i=0; i<stiff_eta_.size(); i++)
   {
      stiff_eta_[i]->Assemble();
      stiff_eta_[i]->Finalize();
   }
}

void TwoFluidDiffusion::initSolver()
{
   block_A_.SetBlock(0, 0, a_dndn_[0]->ParallelAssemble());
   block_A_.SetBlock(1, 1, a_dndn_[1]->ParallelAssemble());

   for (int d=0; d<dim_; d++)
   {
      block_A_.SetBlock(d + 2, 0, a_dpdn_[d]->ParallelAssemble());
   }
   for (int d=0; d<dim_; d++)
   {
      block_A_.SetBlock(dim_ + d + 2, 1, a_dpdn_[dim_ + d]->ParallelAssemble());
   }

   for (int di=0; di<dim_; di++)
   {
      for (int dj=0; dj<dim_; dj++)
      {
         block_A_.SetBlock(di + 2, dj + 2,
                           a_dpdu_[di * dim_ + dj]->ParallelAssemble());
         block_B_.SetBlock(di + 2, dj + 2, stiff_eta_[di * dim_ + dj]);
      }
   }
   for (int di=0; di<dim_; di++)
   {
      for (int dj=0; dj<dim_; dj++)
      {
         block_A_.SetBlock(dim_ + di + 2, dim_ + dj + 2,
                           a_dpdu_[dim_ * (dim_ + di) + dj]->ParallelAssemble());
         block_B_.SetBlock(dim_ + di + 2, dim_ + dj + 2,
                           stiff_eta_[dim_ * (dim_ + di) + dj]);
      }
   }

   block_A_.SetBlock(2 * (dim_ + 1), 0, a_dEdn_[0]->ParallelAssemble());
   block_A_.SetBlock(2 * (dim_ + 1) + 1, 1, a_dEdn_[1]->ParallelAssemble());

   for (int d=0; d<dim_; d++)
   {
      block_A_.SetBlock(2 * (dim_ + 1), d + 2, a_dEdu_[d]->ParallelAssemble());
   }
   for (int d=0; d<dim_; d++)
   {
      block_A_.SetBlock(2 * (dim_ + 1) + 1, d + dim_ + 2,
                        a_dEdu_[dim_ + d]->ParallelAssemble());
   }

   block_A_.SetDiagonalBlock(2 * (dim_ + 1), a_dEdT_[0]->ParallelAssemble());
   block_A_.SetDiagonalBlock(2 * (dim_ + 1) + 1,
                             a_dEdT_[1]->ParallelAssemble());

   block_B_.SetDiagonalBlock(2 * (dim_ + 1), stiff_chi_[0]);
   block_B_.SetDiagonalBlock(2 * (dim_ + 1) + 1, stiff_chi_[1]);

   block_A_.owns_blocks = 1;
   block_B_.owns_blocks = 0;
   /*
   {
      HypreParMatrix * hyp = NULL;
      for (int i=0; i<block_A_.NumRowBlocks(); i++)
      {
         for (int j=0; j<block_A_.NumRowBlocks(); j++)
         {
            if (!block_A_.IsZeroBlock(i,j))
            {
               hyp = dynamic_cast<HypreParMatrix*>(&block_A_.GetBlock(i,j));
               if (hyp != NULL)
               {
                  ostringstream oss; oss << "A_" << i << "_" << j << ".mat";
                  hyp->Print(oss.str().c_str());
               }
            }
         }
      }
   }
   */
   /*
   {
      HypreParMatrix * hyp = NULL;
      for (int i=0; i<block_B_.NumRowBlocks(); i++)
      {
         for (int j=0; j<block_B_.NumRowBlocks(); j++)
         {
            if (!block_B_.IsZeroBlock(i,j))
            {
               hyp = dynamic_cast<ParBilinearForm&>(block_B_.GetBlock(i,j)).ParallelAssemble();
               if (hyp != NULL)
               {
                  ostringstream oss; oss << "B_" << i << "_" << j << ".mat";
                  hyp->Print(oss.str().c_str());
               }
               delete hyp;
            }
         }
      }
   }
   */
   for (int i=0; i<block_A_.NumRowBlocks(); i++)
   {
      block_amg_.SetDiagonalBlock(i, &block_A_.GetBlock(i,i));
   }
   block_amg_.owns_blocks = 0;

   gmres_.SetAbsTol(0.0);
   gmres_.SetRelTol(1e-12);
   gmres_.SetMaxIter(200);
   gmres_.SetKDim(10);
   gmres_.SetPrintLevel(1);
   gmres_.SetOperator(block_A_);
   gmres_.SetPreconditioner(block_amg_);
}

void TwoFluidDiffusion::Update()
{}

void TwoFluidDiffusion::ImplicitSolve(const double dt,
                                      const Vector &x, Vector &y)
{
   this->setTimeStep(dt);
   this->Assemble();
   this->initSolver();

   block_B_.Mult(x, block_rhs_);
   block_rhs_ *= -1.0;

   y = 0.0;

   gmres_.Mult(block_rhs_, y);
}

DiffusionTDO::DiffusionTDO(ParFiniteElementSpace &fes,
                           ParFiniteElementSpace &dfes,
                           ParFiniteElementSpace &vfes,
                           MatrixCoefficient & nuCoef,
                           double dg_sigma,
                           double dg_kappa)
   : TimeDependentOperator(vfes.GetTrueVSize()),
     dim_(vfes.GetFE(0)->GetDim()),
     dt_(0.0),
     dg_sigma_(dg_sigma),
     dg_kappa_(dg_kappa),
     fes_(fes),
     dfes_(dfes),
     vfes_(vfes),
     m_(&fes_),
     d_(&fes_),
     rhs_(&fes_),
     x_(&vfes_),
     M_(NULL),
     D_(NULL),
     RHS_(fes_.GetTrueVSize()),
     X_(fes_.GetTrueVSize()),
     solver_(NULL),
     amg_(NULL),
     nuCoef_(nuCoef),
     dtNuCoef_(0.0, nuCoef_)
{
   m_.AddDomainIntegrator(new MassIntegrator);
   m_.AddDomainIntegrator(new DiffusionIntegrator(dtNuCoef_));
   m_.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dtNuCoef_,
                                                          dg_sigma_,
                                                          dg_kappa_));
   m_.AddBdrFaceIntegrator(new DGDiffusionIntegrator(dtNuCoef_,
                                                     dg_sigma_, dg_kappa_));
   d_.AddDomainIntegrator(new DiffusionIntegrator(nuCoef_));
   d_.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(nuCoef_,
                                                          dg_sigma_,
                                                          dg_kappa_));
   d_.AddBdrFaceIntegrator(new DGDiffusionIntegrator(nuCoef_,
                                                     dg_sigma_, dg_kappa_));
   d_.Assemble();
   d_.Finalize();
   D_ = d_.ParallelAssemble();
}

void DiffusionTDO::ImplicitSolve(const double dt, const Vector &x, Vector &y)
{
   y = 0.0;

   this->initSolver(dt);

   for (int d=0; d<dim_; d++)
   {
      ParGridFunction xd(&fes_, &(x.GetData()[(d+1) * fes_.GetVSize()]));
      ParGridFunction yd(&fes_, &(y.GetData()[(d+1) * fes_.GetVSize()]));

      D_->Mult(xd, rhs_);
      rhs_ *= -1.0;
      rhs_.ParallelAssemble(RHS_);

      X_ = 0.0;
      solver_->Mult(RHS_, X_);

      yd = X_;
   }
}

void DiffusionTDO::initSolver(double dt)
{
   bool newM = false;
   if (fabs(dt - dt_) > 1e-4 * dt)
   {
      dt_ = dt;
      dtNuCoef_.SetAConst(dt);
      m_.Assemble(0);
      m_.Finalize(0);
      if (M_ != NULL)
      {
         delete M_;
      }
      M_ = m_.ParallelAssemble();
      newM = true;
   }

   if (amg_ == NULL || newM)
   {
      if (amg_ != NULL) { delete amg_; }
      amg_ = new HypreBoomerAMG(*M_);
   }
   if (solver_ == NULL || newM)
   {
      if (solver_ != NULL) { delete solver_; }
      if (dg_sigma_ == -1.0)
      {
         HyprePCG * pcg = new HyprePCG(*M_);
         pcg->SetTol(1e-12);
         pcg->SetMaxIter(200);
         pcg->SetPrintLevel(0);
         pcg->SetPreconditioner(*amg_);
         solver_ = pcg;
      }
      else
      {
         HypreGMRES * gmres = new HypreGMRES(*M_);
         gmres->SetTol(1e-12);
         gmres->SetMaxIter(200);
         gmres->SetKDim(10);
         gmres->SetPrintLevel(0);
         gmres->SetPreconditioner(*amg_);
         solver_ = gmres;
      }

   }
}

// Implementation of class FE_Evolution
AdvectionTDO::AdvectionTDO(ParFiniteElementSpace &vfes,
                           Operator &A, SparseMatrix &Aflux, int num_equation,
                           double specific_heat_ratio)
   : TimeDependentOperator(A.Height()),
     dim_(vfes.GetFE(0)->GetDim()),
     num_equation_(num_equation),
     specific_heat_ratio_(specific_heat_ratio),
     vfes_(vfes),
     A_(A),
     Aflux_(Aflux),
     Me_inv_(vfes.GetFE(0)->GetDof(), vfes.GetFE(0)->GetDof(), vfes.GetNE()),
     state_(num_equation_),
     f_(num_equation_, dim_),
     flux_(vfes.GetNDofs(), dim_, num_equation_),
     z_(A.Height())
{
   // Standard local assembly and inversion for energy mass matrices.
   const int dof = vfes_.GetFE(0)->GetDof();
   DenseMatrix Me(dof);
   DenseMatrixInverse inv(&Me);
   MassIntegrator mi;
   for (int i = 0; i < vfes_.GetNE(); i++)
   {
      mi.AssembleElementMatrix(*vfes_.GetFE(i),
                               *vfes_.GetElementTransformation(i), Me);
      inv.Factor();
      inv.GetInverseMatrix(Me_inv_(i));
   }
}

void AdvectionTDO::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   max_char_speed_ = 0.;

   // 1. Create the vector z with the face terms -<F.n(u), [w]>.
   A_.Mult(x, z_);

   // 2. Add the element terms.
   // i.  computing the flux approximately as a grid function by interpolating
   //     at the solution nodes.
   // ii. multiplying this grid function by a (constant) mixed bilinear form for
   //     each of the num_equation, computing (F(u), grad(w)) for each equation.

   DenseMatrix xmat(x.GetData(), vfes_.GetNDofs(), num_equation_);
   GetFlux(xmat, flux_);

   for (int k = 0; k < num_equation_; k++)
   {
      Vector fk(flux_(k).GetData(), dim_ * vfes_.GetNDofs());
      Vector zk(z_.GetData() + k * vfes_.GetNDofs(), vfes_.GetNDofs());
      Aflux_.AddMult(fk, zk);
   }

   // 3. Multiply element-wise by the inverse mass matrices.
   Vector zval;
   Array<int> vdofs;
   const int dof = vfes_.GetFE(0)->GetDof();
   DenseMatrix zmat, ymat(dof, num_equation_);

   for (int i = 0; i < vfes_.GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      vfes_.GetElementVDofs(i, vdofs);
      z_.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), dof, num_equation_);
      mfem::Mult(Me_inv_(i), zmat, ymat);
      y.SetSubVector(vdofs, ymat.GetData());
   }
}

// Physicality check (at end)
bool StateIsPhysical(const Vector &state, const int dim,
                     const double specific_heat_ratio);

// Pressure (EOS) computation
inline double ComputePressure(const Vector &state, int dim,
                              double specific_heat_ratio)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   double den_vel2 = 0;
   for (int d = 0; d < dim; d++) { den_vel2 += den_vel(d) * den_vel(d); }
   den_vel2 /= den;

   return (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);
}

// Compute the vector flux F(u)
void ComputeFlux(const Vector &state, int dim, double specific_heat_ratio,
                 DenseMatrix &flux)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   MFEM_ASSERT(StateIsPhysical(state, dim, specific_heat_ratio), "");

   const double pres = ComputePressure(state, dim, specific_heat_ratio);

   for (int d = 0; d < dim; d++)
   {
      flux(0, d) = den_vel(d);
      for (int i = 0; i < dim; i++)
      {
         flux(1+i, d) = den_vel(i) * den_vel(d) / den;
      }
      flux(1+d, d) += pres;
   }

   const double H = (den_energy + pres) / den;
   for (int d = 0; d < dim; d++)
   {
      flux(1+dim, d) = den_vel(d) * H;
   }
}

// Compute the scalar F(u).n
void ComputeFluxDotN(const Vector &state, const Vector &nor,
                     double specific_heat_ratio,
                     Vector &fluxN)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   MFEM_ASSERT(StateIsPhysical(state, dim, specific_heat_ratio), "");

   const double pres = ComputePressure(state, dim, specific_heat_ratio);

   double den_velN = 0;
   for (int d = 0; d < dim; d++) { den_velN += den_vel(d) * nor(d); }

   fluxN(0) = den_velN;
   for (int d = 0; d < dim; d++)
   {
      fluxN(1+d) = den_velN * den_vel(d) / den + pres * nor(d);
   }

   const double H = (den_energy + pres) / den;
   fluxN(1 + dim) = den_velN * H;
}

// Compute the maximum characteristic speed.
inline double ComputeMaxCharSpeed(const Vector &state,
                                  int dim, double specific_heat_ratio)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);

   double den_vel2 = 0;
   for (int d = 0; d < dim; d++) { den_vel2 += den_vel(d) * den_vel(d); }
   den_vel2 /= den;

   const double pres = ComputePressure(state, dim, specific_heat_ratio);
   const double sound = sqrt(specific_heat_ratio * pres / den);
   const double vel = sqrt(den_vel2 / den);

   return vel + sound;
}

// Compute the flux at solution nodes.
void AdvectionTDO::GetFlux(const DenseMatrix &x, DenseTensor &flux) const
{
   const int dof = flux.SizeI();
   const int dim = flux.SizeJ();

   for (int i = 0; i < dof; i++)
   {
      for (int k = 0; k < num_equation_; k++) { state_(k) = x(i, k); }
      ComputeFlux(state_, dim, specific_heat_ratio_, f_);

      for (int d = 0; d < dim; d++)
      {
         for (int k = 0; k < num_equation_; k++)
         {
            flux(i, d, k) = f_(k, d);
         }
      }

      // Update max char speed
      const double mcs = ComputeMaxCharSpeed(state_, dim, specific_heat_ratio_);
      if (mcs > max_char_speed_) { max_char_speed_ = mcs; }
   }
}

// Implementation of class RiemannSolver
RiemannSolver::RiemannSolver(int num_equation, double specific_heat_ratio) :
   num_equation_(num_equation),
   specific_heat_ratio_(specific_heat_ratio),
   flux1_(num_equation),
   flux2_(num_equation) { }

double RiemannSolver::Eval(const Vector &state1, const Vector &state2,
                           const Vector &nor, Vector &flux)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();

   MFEM_ASSERT(StateIsPhysical(state1, dim, specific_heat_ratio_), "");
   MFEM_ASSERT(StateIsPhysical(state2, dim, specific_heat_ratio_), "");

   const double maxE1 = ComputeMaxCharSpeed(state1, dim, specific_heat_ratio_);
   const double maxE2 = ComputeMaxCharSpeed(state2, dim, specific_heat_ratio_);

   const double maxE = max(maxE1, maxE2);

   ComputeFluxDotN(state1, nor, specific_heat_ratio_, flux1_);
   ComputeFluxDotN(state2, nor, specific_heat_ratio_, flux2_);

   double normag = 0;
   for (int i = 0; i < dim; i++)
   {
      normag += nor(i) * nor(i);
   }
   normag = sqrt(normag);

   for (int i = 0; i < num_equation_; i++)
   {
      flux(i) = 0.5 * (flux1_(i) + flux2_(i))
                - 0.5 * maxE * (state2(i) - state1(i)) * normag;
   }

   return maxE;
}

// Implementation of class DomainIntegrator
DomainIntegrator::DomainIntegrator(const int dim, int num_equation)
   : flux_(num_equation, dim) { }

void DomainIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                              const FiniteElement &test_fe,
                                              ElementTransformation &Tr,
                                              DenseMatrix &elmat)
{
   // Assemble the form (vec(v), grad(w))

   // Trial space = vector L2 space (mesh dim)
   // Test space  = scalar L2 space

   const int dof_trial = trial_fe.GetDof();
   const int dof_test = test_fe.GetDof();
   const int dim = trial_fe.GetDim();

   shape_.SetSize(dof_trial);
   dshapedr_.SetSize(dof_test, dim);
   dshapedx_.SetSize(dof_test, dim);

   elmat.SetSize(dof_test, dof_trial * dim);
   elmat = 0.0;

   const int maxorder = max(trial_fe.GetOrder(), test_fe.GetOrder());
   const int intorder = 2 * maxorder;
   const IntegrationRule *ir = &IntRules.Get(trial_fe.GetGeomType(), intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Calculate the shape functions
      trial_fe.CalcShape(ip, shape_);
      shape_ *= ip.weight;

      // Compute the physical gradients of the test functions
      Tr.SetIntPoint(&ip);
      test_fe.CalcDShape(ip, dshapedr_);
      Mult(dshapedr_, Tr.AdjugateJacobian(), dshapedx_);

      for (int d = 0; d < dim; d++)
      {
         for (int j = 0; j < dof_test; j++)
         {
            for (int k = 0; k < dof_trial; k++)
            {
               elmat(j, k + d * dof_trial) += shape_(k) * dshapedx_(j, d);
            }
         }
      }
   }
}

// Implementation of class FaceIntegrator
FaceIntegrator::FaceIntegrator(RiemannSolver &rsolver, const int dim,
                               const int num_equation) :
   num_equation_(num_equation),
   max_char_speed_(0.0),
   rsolver_(rsolver),
   funval1_(num_equation_),
   funval2_(num_equation_),
   nor_(dim),
   fluxN_(num_equation_) { }

void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Tr,
                                        const Vector &elfun, Vector &elvect)
{
   // Compute the term <F.n(u),[w]> on the interior faces.
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();

   shape1_.SetSize(dof1);
   shape2_.SetSize(dof2);

   elvect.SetSize((dof1 + dof2) * num_equation_);
   elvect = 0.0;

   DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation_);
   DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equation_, dof2,
                          num_equation_);

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equation_);
   DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equation_, dof2,
                           num_equation_);

   // Integration order calculation from DGTraceIntegrator
   int intorder;
   if (Tr.Elem2No >= 0)
      intorder = (min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
   else
   {
      intorder = Tr.Elem1->OrderW() + 2*el1.GetOrder();
   }
   if (el1.Space() == FunctionSpace::Pk)
   {
      intorder++;
   }
   const IntegrationRule *ir = &IntRules.Get(Tr.FaceGeom, intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.Loc1.Transform(ip, eip1_);
      Tr.Loc2.Transform(ip, eip2_);

      // Calculate basis functions on both elements at the face
      el1.CalcShape(eip1_, shape1_);
      el2.CalcShape(eip2_, shape2_);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1_, funval1_);
      elfun2_mat.MultTranspose(shape2_, funval2_);

      Tr.Face->SetIntPoint(&ip);

      // Get the normal vector and the flux on the face
      CalcOrtho(Tr.Face->Jacobian(), nor_);
      const double mcs = rsolver_.Eval(funval1_, funval2_, nor_, fluxN_);

      // Update max char speed
      if (mcs > max_char_speed_) { max_char_speed_ = mcs; }

      fluxN_ *= ip.weight;
      for (int k = 0; k < num_equation_; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) -= fluxN_(k) * shape1_(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) += fluxN_(k) * shape2_(s);
         }
      }
   }
}

// Check that the state is physical - enabled in debug mode
bool StateIsPhysical(const Vector &state, int dim,
                     double specific_heat_ratio)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   if (den < 0)
   {
      cout << "Negative density: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }
   if (den_energy <= 0)
   {
      cout << "Negative energy: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }

   double den_vel2 = 0;
   for (int i = 0; i < dim; i++) { den_vel2 += den_vel(i) * den_vel(i); }
   den_vel2 /= den;

   const double pres = (specific_heat_ratio - 1.0) *
                       (den_energy - 0.5 * den_vel2);

   if (pres <= 0)
   {
      cout << "Negative pressure: " << pres << ", state: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }
   return true;
}

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI
