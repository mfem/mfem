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

#include "plasma.hpp"

using namespace std;

namespace mfem
{

namespace plasma
{

G_EQDSK_Data::G_EQDSK_Data(istream &is)
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

   is >> RDIM_ >> ZDIM_ >> RCENTR_ >> RLEFT_ >> ZMID_;
   is >> RMAXIS_ >> ZMAXIS_ >> SIMAG_ >> SIBRY_ >> BCENTR_;
   is >> CURRENT_ >> SIMAG_ >> XDUM >> RMAXIS_ >> XDUM;
   is >> ZMAXIS_ >> XDUM >> SIBRY_ >> XDUM >> XDUM;

   FPOL_.resize(NW_);
   PRES_.resize(NW_);
   FFPRIM_.resize(NW_);
   PPRIME_.resize(NW_);
   PSIRZ_.resize(NW_ * NH_);
   QPSI_.resize(NW_);

   for (int i=0; i<NW_; i++) { is >> FPOL_[i]; }
   for (int i=0; i<NW_; i++) { is >> PRES_[i]; }
   for (int i=0; i<NW_; i++) { is >> FFPRIM_[i]; }
   for (int i=0; i<NW_; i++) { is >> PPRIME_[i]; }
   for (int j=0; j<NH_; j++)
   {
      for (int i=0; i<NW_; i++)
      {
         is >> PSIRZ_[NH_ * i + j];
      }
   }
   //for (int i=0; i<NW_ * NH_; i++) ifs >> PSIRZ_[i];
   for (int i=0; i<NW_; i++) { is >> QPSI_[i]; }

   is >> NBBBS_ >> LIMITR_;

   RBBBS_.resize(NBBBS_);
   ZBBBS_.resize(NBBBS_);
   RLIM_.resize(LIMITR_);
   ZLIM_.resize(LIMITR_);

   for (int i=0; i<NBBBS_; i++) { is >> RBBBS_[i] >> ZBBBS_[i]; }
   for (int i=0; i<LIMITR_; i++)
   {
      is >> RLIM_[i] >> ZLIM_[i];
   }

   initInterpolation();
}

void G_EQDSK_Data::PrintInfo(ostream & out) const
{
   out << "Size of grid: " << NW_ << " x " << NH_ << endl;
   out << "Number of boundary points: " << NBBBS_ << endl;
   out << "Number of limiter points:  " << LIMITR_ << endl;
   out << endl;
   out << "Range of R: " << RLEFT_ << " -> " << RLEFT_ + RDIM_ << endl;
   out << "Range of Z: " << ZMID_ - 0.5 * ZDIM_
       << " -> " << ZMID_ + 0.5 * ZDIM_ << endl;
   out << "Location of magnetic axis: "
       << "(" << RMAXIS_ << "," << ZMAXIS_ << ")" << endl;
   out << "Poloidal flux at magnetic axis:   " << SIMAG_ << endl;
   out << "Poloidal flux at plasma boundary: " << SIBRY_ << endl;
   out << "R in meter of vacuum toroidal magnetic field BCENTR: "
       << RCENTR_ << endl;
   out << "Vacuum toroidal magnetic field in Tesla at RCENTR: "
       << BCENTR_ << endl;
   out << "Plasma current in Ampere: " << CURRENT_ << endl;
}

void G_EQDSK_Data::DumpGnuPlotData(const string &file) const
{
   ostringstream oss_dat, oss_inp;
   oss_inp << file << ".inp";
   oss_dat << file << ".dat";
   ofstream ofs_inp(oss_inp.str().c_str());
   ofstream ofs_dat(oss_dat.str().c_str());

   for (int i=0; i<NW_; i++)
   {
      ofs_dat << RLEFT_ + RDIM_ * i / (NW_ - 1)
              << '\t' << FPOL_[i]
              << '\t' << PRES_[i]
              << '\t' << FFPRIM_[i]
              << '\t' << PPRIME_[i]
              << '\t' << QPSI_[i]
              << '\n';
   }

   ofs_dat << "\n\n";
   for (int i=0; i<NW_; i++)
   {
      for (int j=0; j<NH_; j++)
      {
         ofs_dat << RLEFT_ + RDIM_ * i / (NW_ - 1)
                 << '\t' << ZMID_ - 0.5 * ZDIM_ + ZDIM_ * j / (NH_ - 1)
                 << '\t' << PSIRZ_[NH_ * i + j]
                 //      << '\t' << PSIRZ_[NH_ * i + j]
                 << '\n';
      }
      ofs_dat << '\n';
   }
   ofs_dat << "\n\n";
   for (int i=0; i<NBBBS_; i++)
   {
      ofs_dat << RBBBS_[i] << '\t' << ZBBBS_[i] << '\n';
   }
   ofs_dat << "\n\n";
   for (int i=0; i<LIMITR_; i++)
   {
      ofs_dat << RLIM_[i] << '\t' << ZLIM_[i] << '\n';
   }
   ofs_dat.close();

   ofs_inp << "plot '" << oss_dat.str()
           << "' index 0 using 1:2 w l t 'FPOL';\n";
   ofs_inp << "set size noratio 1,1;\n";
   ofs_inp << "pause -1;\n";
   ofs_inp << "plot '" << oss_dat.str()
           << "' index 0 using 1:3 w l t 'PRES';\n";
   ofs_inp << "pause -1;\n";
   ofs_inp << "plot '" << oss_dat.str()
           << "' index 0 using 1:4 w l t 'FFPRIME';\n";
   ofs_inp << "pause -1;\n";
   ofs_inp << "plot '" << oss_dat.str()
           << "' index 0 using 1:5 w l t 'PPRIME';\n";
   ofs_inp << "pause -1;\n";
   ofs_inp << "plot '" << oss_dat.str()
           << "' index 0 using 1:6 w l t 'QPSI';\n";
   ofs_inp << "pause -1;\n";
   ofs_inp << "set view map;\n";
   ofs_inp << "unset surface;\n";
   ofs_inp << "set contour base;\n";
   ofs_inp << "set cntrparam levels 20;\n";
   ofs_inp << "set size ratio -1;\n";
   ofs_inp << "set nokey;\n";
   ofs_inp << "splot '" << oss_dat.str()
           << "' index 1 with lines pal t 'PSIRZ';\n";
   ofs_inp << "set key;\n";
   ofs_inp << "pause -1;\n";
   ofs_inp << "set size ratio -1;\n";
   ofs_inp << "plot '" << oss_dat.str()
           << "' index 2 using 1:2 w l t 'BOUNDARY',";
   ofs_inp << " '" << oss_dat.str()
           << "' index 3 using 1:2 w l t 'LIMITER';\n";

   ofs_inp.close();
}

void G_EQDSK_Data::initInterpolation()
{
   c_.SetSize(NW_ - 1, NH_);
   d_.SetSize(NW_, NH_ - 1);
   e_.SetSize(NW_ - 1, NH_ - 1);

   dr_ = RDIM_ / (NW_ - 1);
   dz_ = ZDIM_ / (NH_ - 1);

   for (int i=0; i<NW_-1; i++)
      for (int j=0; j<NH_; j++)
      {
         c_(i,j) = (PSIRZ_[NH_ * (i + 1) + j] - PSIRZ_[NH_ * i + j]) / dr_;
      }

   for (int i=0; i<NW_; i++)
      for (int j=0; j<NH_ - 1; j++)
      {
         d_(i,j) = (PSIRZ_[NH_ * i + j + 1] - PSIRZ_[NH_ * i + j]) / dz_;
      }

   for (int i=0; i<NW_ - 1; i++)
      for (int j=0; j<NH_ - 1; j++)
      {
         e_(i,j) = (c_(i,j+1) - c_(i,j)) / dz_;
      }
}

double G_EQDSK_Data::InterpPsi(const Vector &pt)
{
   double x = pt[0];
   double y = (pt.Size() > 1) ? pt[1] : 0.0;

   int i = (int)floor(double(NW_) * (x - RLEFT_) / RDIM_);
   int j = (int)floor(double(NH_) * (y - ZMID_ + 0.5 * ZDIM_) / ZDIM_);

   if (i < 0 || i >= NW_)
   {
      cout << "x coordinate out of range: "
           << RLEFT_ << " <= " << x << " <= " << RLEFT_ + RDIM_ << endl;;
   }
   if (j < 0 || j >= NH_)
   {
      cout << "y coordinate out of range: "
           << ZMID_ - 0.5 * ZDIM_ << " <= " << y << " <= "
           << ZMID_ + 0.5 * ZDIM_ << endl;;
   }

   // Compute corners of local patch
   double r0 = RLEFT_ + RDIM_ * i / (NW_ - 1);
   double r1 = r0 + RDIM_ / (NW_ - 1);
   double z0 = ZMID_ - 0.5 * ZDIM_ + ZDIM_ * j / (NH_ - 1);
   double z1 = z0 + ZDIM_ / (NH_ - 1);

   // Prepare position dependent factors
   double wra = (r1 - x) / dr_;
   double wrb = (x - r0) / dr_;
   double wrc = (1.0 + 2.0 * wra);
   double wrd = (1.0 + 2.0 * wrb);
   double wra2 = wra * wra;
   double wrb2 = wrb * wrb;

   double wza = (z1 - y) / dz_;
   double wzb = (y - z0) / dz_;
   double wzc = (1.0 + 2.0 * wza);
   double wzd = (1.0 + 2.0 * wzb);
   double wza2 = wza * wza;
   double wzb2 = wzb * wzb;

   // Extract psi values at corners of local patch
   double p00 = PSIRZ_[NH_ * i + j];
   double p10 = PSIRZ_[NH_ * (i + 1) + j];
   double p01 = PSIRZ_[NH_ * i + j + 1];
   double p11 = PSIRZ_[NH_ * (i + 1) + j + 1];

   double psi = p00 * wra2 * wrd * wza2 * wzd
                + p10 * wrb2 * wrc * wza2 * wzd
                + p01 * wra2 * wrd * wzb2 * wzc
                + p11 * wrb2 * wrc * wzb2 * wzc;

   // Compute dpsi/dx at corners of local patch
   double wx00a = fabs(c_(i-1,j) - c_(i-2,j));
   double wx00b = fabs(c_(i+1,j) - c_(i,j));

   double wx10a = fabs(c_(i,j) - c_(i-1,j));
   double wx10b = fabs(c_(i+2,j) - c_(i+1,j));

   double wx01a = fabs(c_(i-1,j+1) - c_(i-2,j+1));
   double wx01b = fabs(c_(i+1,j+1) - c_(i,j+1));

   double wx11a = fabs(c_(i,j+1) - c_(i-1,j+1));
   double wx11b = fabs(c_(i+2,j+1) - c_(i+1,j+1));

   if (wx00a == 0.0 && wx00b == 0.0) { wx00a = 1.0; wx00b = 1.0; }
   if (wx10a == 0.0 && wx10b == 0.0) { wx10a = 1.0; wx10b = 1.0; }
   if (wx01a == 0.0 && wx01b == 0.0) { wx01a = 1.0; wx01b = 1.0; }
   if (wx11a == 0.0 && wx11b == 0.0) { wx11a = 1.0; wx11b = 1.0; }

   double px00 = (wx00b * c_(i-1,j) + wx00a * c_(i,j)) / (wx00b + wx00a);
   double px10 = (wx10b * c_(i,j) + wx10a * c_(i+1,j)) / (wx10b + wx10a);
   double px01 = (wx01b * c_(i-1,j+1) + wx01a * c_(i,j+1)) / (wx01b + wx01a);
   double px11 = (wx11b * c_(i,j+1) + wx11a * c_(i+1,j+1)) / (wx11b + wx11a);

   double psix = px00 * wra2 * wrb * wza2 * wzd
                 - px10 * wrb2 * wra * wza2 * wzd
                 + px01 * wrb * wra2 * wzb2 * wzc
                 - px11 * wra * wrb2 * wzb2 * wzc;
   psi += psix * dr_;

   // Compute dpsi/dy at corners of local patch
   double wy00a = fabs(d_(i,j-1) - d_(i,j-2));
   double wy00b = fabs(d_(i,j+1) - d_(i,j));

   double wy10a = fabs(d_(i+1,j-1) - d_(i+1,j-2));
   double wy10b = fabs(d_(i+1,j+1) - d_(i+1,j));

   double wy01a = fabs(d_(i,j) - d_(i,j-1));
   double wy01b = fabs(d_(i,j+2) - d_(i,j+1));

   double wy11a = fabs(d_(i+1,j) - d_(i+1,j-1));
   double wy11b = fabs(d_(i+1,j+2) - d_(i+1,j+1));

   if (wy00a == 0.0 && wy00b == 0.0) { wy00a = 1.0; wy00b = 1.0; }
   if (wy10a == 0.0 && wy10b == 0.0) { wy10a = 1.0; wy10b = 1.0; }
   if (wy01a == 0.0 && wy01b == 0.0) { wy01a = 1.0; wy01b = 1.0; }
   if (wy11a == 0.0 && wy11b == 0.0) { wy11a = 1.0; wy11b = 1.0; }

   double py00 = (wy00b * d_(i,j-1) + wy00a * d_(i,j)) / (wy00b + wy00a);
   double py10 = (wy10b * d_(i+1,j-1) + wy10a * d_(i+1,j)) / (wy10b + wy10a);
   double py01 = (wy01b * d_(i,j) + wy01a * d_(i,j+1)) / (wy01b + wy01a);
   double py11 = (wy11b * d_(i+1,j) + wy11a * d_(i+1,j)) / (wy11b + wy11a);

   double psiy = py00 * wra2 * wrd * wza2 * wzb + py10 * wrb2 * wrc * wza2 * wzb
                 - py01 * wra2 * wrd * wza * wzb2 - py11 * wrb2 * wrc * wza * wzb2;
   psi += psiy * dz_;

   // Compute d^2psi/dxdy at corners of local patch
   double pxy00 = (wx00b * (wy00b * e_(i-1,j-1) + wy00a * e_(i-1,j))) /
                  ((wx00b + wx00a) * (wy00b + wy00a));
   double pxy10 = (wx10b * (wy10b * e_(i,j-1) + wy10a * e_(i,j))) /
                  ((wx10b + wx10a) * (wy10b + wy10a));
   double pxy01 = (wx01b * (wy01b * e_(i-1,j) + wy01a * e_(i-1,j+1))) /
                  ((wx01b + wx01a) * (wy01b + wy01a));
   double pxy11 = (wx11b * (wy11b * e_(i,j) + wy11a * e_(i,j+1))) /
                  ((wx11b + wx11a) * (wy11b + wy11a));

   double psixy = pxy00 * wra2 * wrb * wza2 * wzb
                  - pxy10 * wra * wrb2 * wza2 * wzb
                  - pxy01 * wra2 * wrb * wza * wzb2
                  + pxy11 * wra * wrb2 * wza * wzb2;

   psi += dr_ * dz_ * psixy;

   return psi;
}

void G_EQDSK_Data::InterpNxGradPsi(const Vector &pt, Vector &b)
{
   b.SetSize(2);
   b = 0.0;

   double x = pt[0];
   double y = (pt.Size() > 1) ? pt[1] : 0.0;

   int i = (int)floor(double(NW_) * (x - RLEFT_) / RDIM_);
   int j = (int)floor(double(NH_) * (y - ZMID_ + 0.5 * ZDIM_) / ZDIM_);

   if (i < 0 || i >= NW_)
   {
      cout << "x coordinate out of range: "
           << RLEFT_ << " <= " << x << " <= " << RLEFT_ + RDIM_ << endl;;
   }
   if (j < 0 || j >= NH_)
   {
      cout << "y coordinate out of range: "
           << ZMID_ - 0.5 * ZDIM_ << " <= " << y << " <= "
           << ZMID_ + 0.5 * ZDIM_ << endl;;
   }

   // Compute corners of local patch
   double r0 = RLEFT_ + RDIM_ * i / (NW_ - 1);
   double r1 = r0 + RDIM_ / (NW_ - 1);
   double z0 = ZMID_ - 0.5 * ZDIM_ + ZDIM_ * j / (NH_ - 1);
   double z1 = z0 + ZDIM_ / (NH_ - 1);

   // Prepare position dependent factors
   double wra = (r1 - x) / dr_, dwra = -1.0 / dr_;
   double wrb = (x - r0) / dr_, dwrb = 1.0 / dr_;
   double wrc = (1.0 + 2.0 * wra), dwrc = 2.0 * dwra;
   double wrd = (1.0 + 2.0 * wrb), dwrd = 2.0 * dwrb;
   double wre = (3.0 * wra - 2.0);
   double wrf = (3.0 * wrb - 2.0);
   double wra2 = wra * wra, dwra2 = 2.0 * wra * dwra;
   double wrb2 = wrb * wrb, dwrb2 = 2.0 * wrb * dwrb;

   double wza = (z1 - y) / dz_, dwza = -1.0 / dz_;
   double wzb = (y - z0) / dz_, dwzb = 1.0 / dz_;
   double wzc = (1.0 + 2.0 * wza), dwzc = 2.0 * dwza;
   double wzd = (1.0 + 2.0 * wzb), dwzd = 2.0 * dwzb;
   double wze = (3.0 * wza - 2.0);
   double wzf = (3.0 * wzb - 2.0);
   double wza2 = wza * wza, dwza2 = 2.0 * wza * dwza;
   double wzb2 = wzb * wzb, dwzb2 = 2.0 * wzb * dwzb;

   // Extract psi values at corners of local patch
   double p00 = PSIRZ_[NH_ * i + j];
   double p10 = PSIRZ_[NH_ * (i + 1) + j];
   double p01 = PSIRZ_[NH_ * i + j + 1];
   double p11 = PSIRZ_[NH_ * (i + 1) + j + 1];

   b[0] -= (p00 * wra2 * wrd + p10 * wrb2 * wrc ) * (dwza2 * wzd + wza2 * dwzd)
           + (p01 * wra2 * wrd + p11 * wrb2 * wrc) * (dwzb2 * wzc + wzb2 * dwzc);
   b[1] += (p00 * wza2 * wzd + p01 * wzb2 * wzc) * (dwra2 * wrd + wra2 * dwrd)
           + (p10 * wza2 * wzd + p11 * wzb2 * wzc) * (dwrb2 * wrc + wrb2 * dwrc);

   // Compute dpsi/dx at corners of local patch
   double wx00a = fabs(c_(i-1,j) - c_(i-2,j));
   double wx00b = fabs(c_(i+1,j) - c_(i,j));

   double wx10a = fabs(c_(i,j) - c_(i-1,j));
   double wx10b = fabs(c_(i+2,j) - c_(i+1,j));

   double wx01a = fabs(c_(i-1,j+1) - c_(i-2,j+1));
   double wx01b = fabs(c_(i+1,j+1) - c_(i,j+1));

   double wx11a = fabs(c_(i,j+1) - c_(i-1,j+1));
   double wx11b = fabs(c_(i+2,j+1) - c_(i+1,j+1));

   if (wx00a == 0.0 && wx00b == 0.0) { wx00a = 1.0; wx00b = 1.0; }
   if (wx10a == 0.0 && wx10b == 0.0) { wx10a = 1.0; wx10b = 1.0; }
   if (wx01a == 0.0 && wx01b == 0.0) { wx01a = 1.0; wx01b = 1.0; }
   if (wx11a == 0.0 && wx11b == 0.0) { wx11a = 1.0; wx11b = 1.0; }

   double px00 = (wx00b * c_(i-1,j) + wx00a * c_(i,j)) / (wx00b + wx00a);
   double px10 = (wx10b * c_(i,j) + wx10a * c_(i+1,j)) / (wx10b + wx10a);
   double px01 = (wx01b * c_(i-1,j+1) + wx01a * c_(i,j+1)) / (wx01b + wx01a);
   double px11 = (wx11b * c_(i,j+1) + wx11a * c_(i+1,j+1)) / (wx11b + wx11a);

   b[0] -= dr_ * ((px00 * wra2 * wrb - px10 * wrb2 * wra)
                  * (dwza2 * wzd + wza2 * dwzd) +
                  (px01 * wrb * wra2 - px11 * wra * wrb2)
                  * (dwzb2 * wzc + wzb2 * dwzc));

   b[1] += dr_ * ((px00 * wza2 * wzd + px01 * wzb2 * wzc)
                  * (dwra2 * wrb + wra2 * dwrb ) -
                  (px10 * wza2 * wzd + px11 * wzb2 * wzc)
                  * (dwra * wrb2 + wra * dwrb2));

   // Compute dpsi/dy at corners of local patch
   double wy00a = fabs(d_(i,j-1) - d_(i,j-2));
   double wy00b = fabs(d_(i,j+1) - d_(i,j));

   double wy10a = fabs(d_(i+1,j-1) - d_(i+1,j-2));
   double wy10b = fabs(d_(i+1,j+1) - d_(i+1,j));

   double wy01a = fabs(d_(i,j) - d_(i,j-1));
   double wy01b = fabs(d_(i,j+2) - d_(i,j+1));

   double wy11a = fabs(d_(i+1,j) - d_(i+1,j-1));
   double wy11b = fabs(d_(i+1,j+2) - d_(i+1,j+1));

   if (wy00a == 0.0 && wy00b == 0.0) { wy00a = 1.0; wy00b = 1.0; }
   if (wy10a == 0.0 && wy10b == 0.0) { wy10a = 1.0; wy10b = 1.0; }
   if (wy01a == 0.0 && wy01b == 0.0) { wy01a = 1.0; wy01b = 1.0; }
   if (wy11a == 0.0 && wy11b == 0.0) { wy11a = 1.0; wy11b = 1.0; }

   double py00 = (wy00b * d_(i,j-1) + wy00a * d_(i,j)) / (wy00b + wy00a);
   double py10 = (wy10b * d_(i+1,j-1) + wy10a * d_(i+1,j)) / (wy10b + wy10a);
   double py01 = (wy01b * d_(i,j) + wy01a * d_(i,j+1)) / (wy01b + wy01a);
   double py11 = (wy11b * d_(i+1,j) + wy11a * d_(i+1,j)) / (wy11b + wy11a);

   b[0] -= dz_ * ((py00 * wra2 * wrd + py10 * wrb2 * wrc)
                  * (dwza2 * wzb + wza2 * dwzb) -
                  (py01 * wra2 * wrd + py11 * wrb2 * wrc)
                  * (dwza * wzb2 + wza * dwzb2));
   b[1] += dz_ * ((py00 * wza2 * wzb - py01 * wza * wzb2)
                  * (dwra2 * wrd + wra2 * dwrd) +
                  (py10 * wza2 * wzb - py11 * wza * wzb2)
                  * (dwrb2 * wrc + wrb2 * dwrc));

   // Compute d^2psi/dxdy at corners of local patch
   double pxy00 = (wx00b * (wy00b * e_(i-1,j-1) + wy00a * e_(i-1,j))) /
                  ((wx00b + wx00a) * (wy00b + wy00a));
   double pxy10 = (wx10b * (wy10b * e_(i,j-1) + wy10a * e_(i,j))) /
                  ((wx10b + wx10a) * (wy10b + wy10a));
   double pxy01 = (wx01b * (wy01b * e_(i-1,j) + wy01a * e_(i-1,j+1))) /
                  ((wx01b + wx01a) * (wy01b + wy01a));
   double pxy11 = (wx11b * (wy11b * e_(i,j) + wy11a * e_(i,j+1))) /
                  ((wx11b + wx11a) * (wy11b + wy11a));

   b[0] -= dr_ * dz_ * ((pxy00 * wra2 * wrb - pxy10 * wra * wrb2)
                        * (dwza2 * wzb + wza2 * dwzb) +
                        (pxy11 * wra * wrb2 - pxy01 * wra2 * wrb)
                        * (dwza * wzb2 + wza * dwzb2));
   b[1] += dr_ * dz_ * ((pxy00 * wza2 * wzb - pxy01 * wza * wzb2)
                        * (dwra2 * wrb + wra2 * dwrb) +
                        (pxy11 * wza * wzb2 - pxy10 * wza2 * wzb)
                        * (dwra * wrb2 + wra * dwrb2));
}

} // namespace plasma

} // namespace mfem
