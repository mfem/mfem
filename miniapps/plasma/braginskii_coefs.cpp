// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "braginskii_coefs.hpp"

using namespace std;
namespace mfem
{

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

   K.SetSize(width);

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

   K.SetSize(width);

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
     eps2_(dim),
     eps3_(3, 3, 3),
     bi_(bi),
     bj_(bj),
     ion_(false),
     zi_(zi),
     mi_(-1.0),
     ne_(-1.0),
     ni_(-1.0),
     bPara_(3),
     bPerp_(3),
     bx_(3)
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
     eps2_(dim),
     eps3_(3, 3, 3),
     bi_(bi),
     bj_(bj),
     ion_(true),
     zi_(zi),
     mi_(mi),
     ne_(-1.0),
     ni_(-1.0),
     bPara_(3),
     bPerp_(3),
     bx_(3)
{
   this->initSymbols();
}

void EtaCoefficient::initSymbols()
{
   int dim = del_.Size();

   del_ = 0.0;
   for (int i=0; i<dim; i++) { del_(i,i) = 1.0; }
   /*
   eps2_ = 0.0;
   if (dim == 2)
   {
      eps2_(0,1) = 1.0;
      eps2_(1,0) = -1.0;
   }
   */
   eps3_ = 0.0;
   // if (dim == 3)
   {
      eps3_(0,1,2) = 1.0;
      eps3_(1,2,0) = 1.0;
      eps3_(2,0,1) = 1.0;
      eps3_(2,1,0) = -1.0;
      eps3_(1,0,2) = -1.0;
      eps3_(0,2,1) = -1.0;
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
   BCoef_.Eval(bPara_, T, ip);
   double bMag = bPara_.Norml2();
   bPara_ /= bMag;

   bx_ = 0.0;
   for (int k=0; k<3; k++)
   {
      for (int l=0; l<3; l++)
      {
         for (int m=0; m<3; m++)
         {
            bx_(k, l) += eps3_(k, l, m) * bPara_[m];
         }
      }
   }

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

   double eta3 = (ion_) ?
                 eta3_i(bMag, mi_, zi_, ni_, temp) :
                 eta3_e(bMag, ne_, temp);

   double eta4 = (ion_) ?
                 eta4_i(bMag, mi_, zi_, ni_, temp) :
                 eta4_e(bMag, ne_, temp);

   K.SetSize(width);

   K = 0.0;

   if (width == 2)
   {
      // eps2_.Mult(bPara_, bPerp_);

      for (int k=0; k<2; k++)
      {
         for (int l=0; l<2; l++)
         {
            /*
            K(k, l) += 2.0 * eta0 *
                             (0.5 * del_(bi_, bj_) - bPara_[bi_] * bPara_[bj_]) *
                             (0.5 * del_(k, l) - bPara_[k] * bPara_[l]);

             // K(k, l) += eta1 * ((del_(bi_, k) - bPara_[bi_] * bPara_[k]) *
                  //                        (del_(bj_, l) - bPara_[bj_] * bPara_[l]) +
             //                       (del_(bi_, l) - bPara_[bi_] * bPara_[l]) *
             //                       (del_(bj_, k) - bPara_[bj_] * bPara_[k]) -
             //                      2.0 * (del_(k, l) - bPara_[k] * bPara_[l]) *
                  //                    (del_(bi_, bj_) - bPara_[bi_] * bPara_[bj_]));

                  K(k, l) += 0.5 * eta1 * ((0.5 * del_(bi_, bj_) -
                                            bPara_[bi_]* bPara_[bj_]) *
                                           (bPerp_[k] * bPara_[l] +
                                            bPerp_[l] * bPara_[k]) +
                                           (0.5 * del_(k, l) -
                                            bPara_[k] * bPara_[l]) *
                                           (bPerp_[bi_] * bPara_[bj_] +
                                            bPerp_[bj_] * bPara_[bi_]));

                  K(k, l) += eta2 * ((del_(bi_, k) - bPara_[bi_] * bPara_[k]) *
                                     bPara_[bj_] * bPara_[l] +
                                     (del_(bi_, l) - bPara_[bi_] * bPara_[l]) *
                                     bPara_[bj_] * bPara_[k] +
                                     (del_(bj_, k) - bPara_[bj_] * bPara_[k]) *
                                     bPara_[bi_] * bPara_[l] +
                                     (del_(bj_, l) - bPara_[bj_] * bPara_[l]) *
                                     bPara_[bi_] * bPara_[k]);
            */
            K(k, l) += 3.0 * eta0 *
                       (del_(bi_, bj_) / 3.0 - bPara_[bi_] * bPara_[bj_]) *
                       (del_(k, l) / 3.0 - bPara_[k] * bPara_[l]);

            K(k, l) += eta1 * ((del_(bi_, k) - bPara_[bi_] * bPara_[k]) *
                               (del_(bj_, l) - bPara_[bj_] * bPara_[l]) +
                               (del_(bi_, l) - bPara_[bi_] * bPara_[l]) *
                               (del_(bj_, k) - bPara_[bj_] * bPara_[k]) -
                               (del_(bi_, bj_) - bPara_[bi_] * bPara_[bj_]) *
                               (del_(k, l) - bPara_[k] * bPara_[l]));

            K(k, l) += eta2 * ((del_(bi_, k) - bPara_[bi_] * bPara_[k]) *
                               bPara_[bj_] * bPara_[l] +
                               (del_(bi_, l) - bPara_[bi_] * bPara_[l]) *
                               bPara_[bj_] * bPara_[k] +
                               (del_(bj_, k) - bPara_[bj_] * bPara_[k]) *
                               bPara_[bi_] * bPara_[l] +
                               (del_(bj_, l) - bPara_[bj_] * bPara_[l]) *
                               bPara_[bi_] * bPara_[k]);

            K(k, l) += eta3 * ((del_(bi_, k) - bPara_[bi_] * bPara_[k]) *
                               bx_(bj_, l) +
                               (del_(bj_, k) - bPara_[bj_] * bPara_[k]) *
                               bx_(bi_, l) -
                               (del_(bi_, bj_) - bPara_[bi_] * bPara_[bj_]) *
                               bx_(k, l));

            K(k, l) += eta4 * (bx_(bi_, k) * bPara_[bj_] * bPara_[l] +
                               bx_(bi_, l) * bPara_[bj_] * bPara_[k] +
                               bx_(bj_, k) * bPara_[bi_] * bPara_[l] +
                               bx_(bj_, l) * bPara_[bi_] * bPara_[k]);
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
                       (del_(bi_, bj_) / 3.0 - bPara_[bi_] * bPara_[bj_]) *
                       (del_(k, l) / 3.0 - bPara_[k] * bPara_[l]);

            K(k, l) += eta1 * ((del_(bi_, k) - bPara_[bi_] * bPara_[k]) *
                               (del_(bj_, l) - bPara_[bj_] * bPara_[l]) +
                               (del_(bi_, l) - bPara_[bi_] * bPara_[l]) *
                               (del_(bj_, k) - bPara_[bj_] * bPara_[k]) -
                               (del_(bi_, bj_) - bPara_[bi_] * bPara_[bj_]) *
                               (del_(k, l) - bPara_[k] * bPara_[l]));

            K(k, l) += eta2 * ((del_(bi_, k) - bPara_[bi_] * bPara_[k]) *
                               bPara_[bj_] * bPara_[l] +
                               (del_(bi_, l) - bPara_[bi_] * bPara_[l]) *
                               bPara_[bj_] * bPara_[k] +
                               (del_(bj_, k) - bPara_[bj_] * bPara_[k]) *
                               bPara_[bi_] * bPara_[l] +
                               (del_(bj_, l) - bPara_[bj_] * bPara_[l]) *
                               bPara_[bi_] * bPara_[k]);

            K(k, l) += eta3 * ((del_(bi_, k) - bPara_[bi_] * bPara_[k]) *
                               bx_(bj_, l) +
                               (del_(bj_, k) - bPara_[bj_] * bPara_[k]) *
                               bx_(bi_, l) -
                               (del_(bi_, bj_) - bPara_[bi_] * bPara_[bj_]) *
                               bx_(k, l));

            K(k, l) += eta4 * (bx_(bi_, k) * bPara_[bj_] * bPara_[l] +
                               bx_(bi_, l) * bPara_[bj_] * bPara_[k] +
                               bx_(bj_, k) * bPara_[bi_] * bPara_[l] +
                               bx_(bj_, l) * bPara_[bi_] * bPara_[k]);
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

pAdvectionCoefficient::pAdvectionCoefficient(double m, Coefficient & nCoef,
                                             VectorCoefficient & uCoef)
   : VectorCoefficient(uCoef.GetVDim()),
     m_(m),
     nCoef_(nCoef),
     uCoef_(uCoef)
{}

void pAdvectionCoefficient::Eval(Vector & K,
                                 ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   double n = nCoef_.Eval(T, ip);
   uCoef_.Eval(K, T, ip);

   K *= n * m_;
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

   // The factor of amu_/q_ converts from amu * m^2/s to eV
   return 1.5 * temp + 0.5 * m_ * (u_ * u_) * amu_ / q_;
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

   // The factor of amu_/q_ converts from amu * m^2/s to eV
   return m_ * n * u_[c_] * amu_ / q_;
}

TAdvectionCoefficient::TAdvectionCoefficient(Coefficient & nCoef,
                                             VectorCoefficient & uCoef)
   : VectorCoefficient(uCoef.GetVDim()),
     nCoef_(nCoef),
     uCoef_(uCoef)
{}

void TAdvectionCoefficient::Eval(Vector & K,
                                 ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   double n = nCoef_.Eval(T, ip);
   uCoef_.Eval(K, T, ip);

   K *= 2.5 * n;
}

} // namespace plasma

} // namespace mfem
