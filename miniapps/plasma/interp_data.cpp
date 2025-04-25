// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "interp_data.hpp"

using namespace std;

namespace mfem
{

namespace plasma
{

Interp_Data::Interp_Data(istream &is)
   : init_flag_(0)
{
   double XDUM = 0.0;

   const int buflen = 1024;
   char buf[buflen];
   is.getline(buf, buflen);
   istringstream iss(buf);
   string word;
   iss >> std::ws;
   while (!iss.eof())
   {
      iss >> word;
      CASE_.push_back(word);
      iss >> std::ws;
   }

   NW_ = to_int(CASE_[CASE_.size()-2]);
   NH_ = to_int(CASE_[CASE_.size()-1]);

   is >> RDIM_ >> ZDIM_ >> RLEFT_ >> ZMID_;

   FIELD_.resize(NW_ * NH_);

   for (int j=0; j<NH_; j++)
   {
      for (int i=0; i<NW_; i++)
      {
         is >> FIELD_[NH_ * i + j];
      }
   }

   dr_ = RDIM_ / (NW_ - 1);
   dz_ = ZDIM_ / (NH_ - 1);
}

void Interp_Data::PrintInfo(ostream & out) const
{
   out << endl << "Outside Plasma Field Info:" << endl;
   out << "Size of grid: " << NW_ << " x " << NH_ << endl;
   out << "Range of R: " << RLEFT_ << " -> " << RLEFT_ + RDIM_ << endl;
   out << "Range of Z: " << ZMID_ - 0.5 * ZDIM_
       << " -> " << ZMID_ + 0.5 * ZDIM_ << endl;
}

double Interp_Data::InterpDataRZ(const Vector &rz)
{
   initInterpRZ(FIELD_, DATA_c_, DATA_d_, DATA_e_);
   return interpRZ(rz, FIELD_, DATA_c_, DATA_d_, DATA_e_);
}

void Interp_Data::initInterpRZ(const std::vector<double> &v,
                                ShiftedDenseMatrix &c,
                                ShiftedDenseMatrix &d,
                                ShiftedDenseMatrix &e)
{
   ExtendedDenseMatrix ve(&v[0], NW_, NH_);

   c.SetSize(NW_ + 3, NH_ + 2); c.SetShifts(2, 1); c = 0.0;
   d.SetSize(NW_ + 2, NH_ + 3); d.SetShifts(1, 2); d = 0.0;
   e.SetSize(NW_ + 1, NH_ + 1); e.SetShifts(1, 1); e = 0.0;

   // x-directed divided differences
   for (int i=-1; i<NW_; i++)
   {
      c(i,-1) = (ve(i+1,-1) - ve(i,-1)) / dr_;
   }
   for (int j=0; j<NH_; j++)
   {
      for (int i=-2; i<=NW_; i++)
      {
         c(i,j) = (ve(i+1,j) - ve(i,j)) / dr_;
      }
   }
   for (int i=-1; i<NW_; i++)
   {
      c(i,NH_) = (ve(i+1,NH_) - ve(i,NH_)) / dr_;
   }

   // y-directed divided differences
   for (int j=-1; j<NH_; j++)
   {
      d(-1,j) = (ve(-1,j+1) - ve(-1,j)) / dz_;
   }
   for (int i=0; i<NW_; i++)
   {
      for (int j=-2; j<=NH_; j++)
      {
         d(i,j) = (ve(i,j+1) - ve(i,j)) / dz_;
      }
   }
   for (int j=-1; j<NH_; j++)
   {
      d(NW_,j) = (ve(NW_,j+1) - ve(NW_,j)) / dz_;
   }

   // Second order divided differences
   for (int i=-1; i<NW_; i++)
   {
      for (int j=-1; j<NH_; j++)
      {
         e(i,j) = (c(i,j+1) - c(i,j)) / dz_;
      }
   }
}

double Interp_Data::interpRZ(const Vector &rz,
                              const std::vector<double> &v,
                              const ShiftedDenseMatrix &c,
                              const ShiftedDenseMatrix &d,
                              const ShiftedDenseMatrix &e)
{
   double r = rz[0];
   double z = rz[1];

   double rs = (r - RLEFT_) / RDIM_;
   double zs = (z - ZMID_ + 0.5 * ZDIM_) / ZDIM_;

   int i = std::max(0, std::min((int)floor(double(NW_-1) * rs), NW_-2));
   int j = std::max(0, std::min((int)floor(double(NH_-1) * zs), NH_-2));

   // Compute corners of local patch
   double r0 = RLEFT_ + RDIM_ * i / (NW_ - 1);
   double r1 = r0 + RDIM_ / (NW_ - 1);
   double z0 = ZMID_ - 0.5 * ZDIM_ + ZDIM_ * j / (NH_ - 1);
   double z1 = z0 + ZDIM_ / (NH_ - 1);

   // Prepare position dependent factors
   double wra = (r1 - r) / dr_;
   double wrb = (r - r0) / dr_;
   double wrc = (1.0 + 2.0 * wra);
   double wrd = (1.0 + 2.0 * wrb);
   double wra2 = wra * wra;
   double wrb2 = wrb * wrb;

   double wza = (z1 - z) / dz_;
   double wzb = (z - z0) / dz_;
   double wzc = (1.0 + 2.0 * wza);
   double wzd = (1.0 + 2.0 * wzb);
   double wza2 = wza * wza;
   double wzb2 = wzb * wzb;

   // Extract variable values at corners of local patch
   double p00 = v[NH_ * i + j];
   double p10 = v[NH_ * (i + 1) + j];
   double p01 = v[NH_ * i + j + 1];
   double p11 = v[NH_ * (i + 1) + j + 1];

   double var = p00 * wra2 * wrd * wza2 * wzd
                + p10 * wrb2 * wrc * wza2 * wzd
                + p01 * wra2 * wrd * wzb2 * wzc
                + p11 * wrb2 * wrc * wzb2 * wzc;

   // Compute dvar/dx at corners of local patch
   double wx00a = fabs(c(i-1,j) - c(i-2,j));
   double wx00b = fabs(c(i+1,j) - c(i,j));

   double wx10a = fabs(c(i,j) - c(i-1,j));
   double wx10b = fabs(c(i+2,j) - c(i+1,j));

   double wx01a = fabs(c(i-1,j+1) - c(i-2,j+1));
   double wx01b = fabs(c(i+1,j+1) - c(i,j+1));

   double wx11a = fabs(c(i,j+1) - c(i-1,j+1));
   double wx11b = fabs(c(i+2,j+1) - c(i+1,j+1));

   if (wx00a == 0.0 && wx00b == 0.0) { wx00a = 1.0; wx00b = 1.0; }
   if (wx10a == 0.0 && wx10b == 0.0) { wx10a = 1.0; wx10b = 1.0; }
   if (wx01a == 0.0 && wx01b == 0.0) { wx01a = 1.0; wx01b = 1.0; }
   if (wx11a == 0.0 && wx11b == 0.0) { wx11a = 1.0; wx11b = 1.0; }

   double px00 = (wx00b * c(i-1,j) + wx00a * c(i,j)) / (wx00b + wx00a);
   double px10 = (wx10b * c(i,j) + wx10a * c(i+1,j)) / (wx10b + wx10a);
   double px01 = (wx01b * c(i-1,j+1) + wx01a * c(i,j+1)) / (wx01b + wx01a);
   double px11 = (wx11b * c(i,j+1) + wx11a * c(i+1,j+1)) / (wx11b + wx11a);

   double varx = px00 * wra2 * wrb * wza2 * wzd
                 - px10 * wrb2 * wra * wza2 * wzd
                 + px01 * wrb * wra2 * wzb2 * wzc
                 - px11 * wra * wrb2 * wzb2 * wzc;
   var += varx * dr_;

   // Compute dvar/dy at corners of local patch
   double wy00a = fabs(d(i,j-1) - d(i,j-2));
   double wy00b = fabs(d(i,j+1) - d(i,j));

   double wy10a = fabs(d(i+1,j-1) - d(i+1,j-2));
   double wy10b = fabs(d(i+1,j+1) - d(i+1,j));

   double wy01a = fabs(d(i,j) - d(i,j-1));
   double wy01b = fabs(d(i,j+2) - d(i,j+1));

   double wy11a = fabs(d(i+1,j) - d(i+1,j-1));
   double wy11b = fabs(d(i+1,j+2) - d(i+1,j+1));

   if (wy00a == 0.0 && wy00b == 0.0) { wy00a = 1.0; wy00b = 1.0; }
   if (wy10a == 0.0 && wy10b == 0.0) { wy10a = 1.0; wy10b = 1.0; }
   if (wy01a == 0.0 && wy01b == 0.0) { wy01a = 1.0; wy01b = 1.0; }
   if (wy11a == 0.0 && wy11b == 0.0) { wy11a = 1.0; wy11b = 1.0; }

   double py00 = (wy00b * d(i,j-1) + wy00a * d(i,j)) / (wy00b + wy00a);
   double py10 = (wy10b * d(i+1,j-1) + wy10a * d(i+1,j)) / (wy10b + wy10a);
   double py01 = (wy01b * d(i,j) + wy01a * d(i,j+1)) / (wy01b + wy01a);
   double py11 = (wy11b * d(i+1,j) + wy11a * d(i+1,j)) / (wy11b + wy11a);

   double vary = py00 * wra2 * wrd * wza2 * wzb
                 + py10 * wrb2 * wrc * wza2 * wzb
                 - py01 * wra2 * wrd * wza * wzb2
                 - py11 * wrb2 * wrc * wza * wzb2;
   var += vary * dz_;

   // Compute d^2var/dxdy at corners of local patch
   double pxy00 = (wx00b * (wy00b * e(i-1,j-1) + wy00a * e(i-1,j)) +
                   wx00a * (wy00b * e(i,j-1) + wy00a * e(i,j))) /
                  ((wx00b + wx00a) * (wy00b + wy00a));
   double pxy10 = (wx10b * (wy10b * e(i,j-1) + wy10a * e(i,j)) +
                   wx10a * (wy10b * e(i+1,j-1) + wy10a * e(i+1,j))) /
                  ((wx10b + wx10a) * (wy10b + wy10a));
   double pxy01 = (wx01b * (wy01b * e(i-1,j) + wy01a * e(i-1,j+1)) +
                   wx01a * (wy01b * e(i,j) + wy01a * e(i,j+1))) /
                  ((wx01b + wx01a) * (wy01b + wy01a));
   double pxy11 = (wx11b * (wy11b * e(i,j) + wy11a * e(i,j+1)) +
                   wx11a * (wy11b * e(i+1,j) + wy11a * e(i+1,j+1))) /
                  ((wx11b + wx11a) * (wy11b + wy11a));

   double varxy = pxy00 * wra2 * wrb * wza2 * wzb
                  - pxy10 * wra * wrb2 * wza2 * wzb
                  - pxy01 * wra2 * wrb * wza * wzb2
                  + pxy11 * wra * wrb2 * wza * wzb2;

   var += dr_ * dz_ * varxy;

   return var;
}

void Interp_Data::ExtendedDenseMatrix::init()
{
   // Populate four corners
   SW_ = 3.0 * ((*this)(0,0) - (*this)(1,1)) + (*this)(2,2);
   SE_ = 3.0 * ((*this)(m_-1,0) - (*this)(m_-2,1)) + (*this)(m_-3,2);
   NW_ = 3.0 * ((*this)(0,n_-1) - (*this)(1,n_-2)) + (*this)(2,n_-3);
   NE_ = 3.0 * ((*this)(m_-1,n_-1) - (*this)(m_-2,n_-2))
         + (*this)(m_-3,n_-3);

   // Populate lowest rows
   for (int j=0; j<n_; j++)
   {
      S_(1,j) = 3.0 * ((*this)(0,j) - (*this)(1,j)) + (*this)(2,j);
      S_(0,j) = 3.0 * (2.0 * (*this)(0,j) + (*this)(2,j)) - 8.0 * (*this)(1,j);
   }

   // Populate highest rows
   for (int j=0; j<n_; j++)
   {
      N_(1,j) = 3.0 * (2.0 * (*this)(m_-1,j) + (*this)(m_-3,j))
                - 8.0 * (*this)(m_-2,j);
      N_(0,j) = 3.0 * ((*this)(m_-1,j) - (*this)(m_-2,j)) + (*this)(m_-3,j);
   }

   // Populate lowest columns
   for (int i=0; i<m_; i++)
   {
      W_(i,0) = 3.0 * (2.0 * (*this)(i,0) + (*this)(i,2)) - 8.0 * (*this)(i,1);
      W_(i,1) = 3.0 * ((*this)(i,0) - (*this)(i,1)) + (*this)(i,2);
   }

   // Populate highest columns
   for (int i=0; i<m_; i++)
   {
      E_(i,0) = 3.0 * ((*this)(i,n_-1) - (*this)(i,n_-2)) + (*this)(i,n_-3);
      E_(i,1) = 3.0 * (2.0 * (*this)(i,n_-1) + (*this)(i,n_-3))
                - 8.0 * (*this)(i,n_-2);
   }
}

} // namespace plasma

} // namespace mfem
