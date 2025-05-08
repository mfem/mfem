// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "g_eqdsk_data.hpp"

using namespace std;

namespace mfem
{

namespace plasma
{

G_EQDSK_Data::G_EQDSK_Data(istream &is)
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
   for (int i=0; i<NW_; i++) { is >> QPSI_[i]; }

   is >> NBBBS_ >> LIMITR_;

   RBBBS_.resize(NBBBS_);
   ZBBBS_.resize(NBBBS_);
   RLIM_.resize(LIMITR_);
   ZLIM_.resize(LIMITR_);

   for (int i=0; i<NBBBS_; i++) { is >> RBBBS_[i] >> ZBBBS_[i]; }
   for (int i=0; i<LIMITR_; i++) { is >> RLIM_[i] >> ZLIM_[i]; }

   dr_ = RDIM_ / (NW_ - 1);
   dz_ = ZDIM_ / (NH_ - 1);

   double psi_bry = checkPsiBoundary();
   if ((SIBRY_ - SIMAG_) < 1e-2 * (psi_bry - SIMAG_)) { SIBRY_ = psi_bry; }

   dpsi_ = (SIBRY_ - SIMAG_) / (NW_ - 1);
}

void G_EQDSK_Data::PrintInfo(ostream & out) const
{
   out << endl << "G EQDSK File Info:" << endl;
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
   out << "Plasma current in Ampere: " << CURRENT_ << endl << endl;
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

double G_EQDSK_Data::checkPsiBoundary()
{
   double psi_mid = 0.0;
   double psi_min = DBL_MAX;
   double psi_max = DBL_MIN;

   Vector rz(2);
   double psi = 0.0;
   for (int i=0; i<NBBBS_; i++)
   {
      rz[0] = RBBBS_[i];
      rz[1] = ZBBBS_[i];
      psi = this->InterpPsiRZ(rz);

      psi_min = std::min(psi, psi_min);
      psi_max = std::max(psi, psi_max);

      if (NBBBS_ % 2 == 1)
      {
         if (i == (NBBBS_ - 1) / 2) { psi_mid = psi; }
      }
      else
      {
         if (i == NBBBS_ / 2 || i + 1 == NBBBS_ / 2) { psi_mid += 0.5 * psi; }
      }
   }
   //cout << psi_min << " <= psi <= " << psi_max << endl;
   //cout << "psi_mid = " << psi_mid << endl;

   return psi_mid;
}
/*
double G_EQDSK_Data::InterpFPol(double r)
{
   if (!checkFlag(FPOL))
   {
      initInterpR(FPOL_, FPOL_t_);
      setFlag(FPOL);
   }
   return interpR(r, FPOL_, FPOL_t_);
}
double G_EQDSK_Data::InterpPres(double r)
{
   if (!checkFlag(PRES))
   {
      initInterpR(PRES_, PRES_t_);
      setFlag(PRES);
   }
   return interpR(r, PRES_, PRES_t_);
}
double G_EQDSK_Data::InterpFFPrime(double r)
{
   if (!checkFlag(FFPRIM))
   {
      initInterpR(FFPRIM_, FFPRIM_t_);
      setFlag(FFPRIM);
   }
   return interpR(r, FFPRIM_, FFPRIM_t_);
}
double G_EQDSK_Data::InterpPPrime(double r)
{
   if (!checkFlag(PPRIME))
   {
      initInterpR(PPRIME_, PPRIME_t_);
      setFlag(PPRIME);
   }
   return interpR(r, PPRIME_, PPRIME_t_);
}
double G_EQDSK_Data::InterpQPsi(double r)
{
   if (!checkFlag(QPSI))
   {
      initInterpR(QPSI_, QPSI_t_);
      setFlag(QPSI);
   }
   return interpR(r, QPSI_, QPSI_t_);
}
double G_EQDSK_Data::InterpBTor(double r)
{
   if (!checkFlag(BTOR))
   {
      initInterpR(BTOR_, BTOR_t_);
      setFlag(BTOR);
   }
   return interpR(r, BTOR_, BTOR_t_);
}
*/
double G_EQDSK_Data::InterpFPolRZ(const Vector &rz)
{
   double psi = InterpPsiRZ(rz);

   if (!checkFlag(FPOL))
   {
      initInterpPsi(FPOL_, FPOL_t_);
      setFlag(FPOL);
   }

   return interpPsi(psi, FPOL_, FPOL_t_);
}

double G_EQDSK_Data::InterpPresRZ(const Vector &rz)
{
   double psi = InterpPsiRZ(rz);

   if (!checkFlag(PRES))
   {
      initInterpPsi(PRES_, PRES_t_);
      setFlag(PRES);
   }

   return interpPsi(psi, PRES_, PRES_t_);
}

double G_EQDSK_Data::InterpFFPrimeRZ(const Vector &rz)
{
   double psi = InterpPsiRZ(rz);

   if (!checkFlag(FFPRIM))
   {
      initInterpPsi(FFPRIM_, FFPRIM_t_);
      setFlag(FFPRIM);
   }
   return interpPsi(psi, FFPRIM_, FFPRIM_t_);
}

double G_EQDSK_Data::InterpPPrimeRZ(const Vector &rz)
{
   double psi = InterpPsiRZ(rz);

   if (!checkFlag(PPRIME))
   {
      initInterpPsi(PPRIME_, PPRIME_t_);
      setFlag(PPRIME);
   }
   return interpPsi(psi, PPRIME_, PPRIME_t_);
}

double G_EQDSK_Data::InterpPsiRZ(const Vector &rz)
{
   if (!checkFlag(PSIRZ))
   {
      initInterpRZ(PSIRZ_, PSIRZ_c_, PSIRZ_d_, PSIRZ_e_);
      setFlag(PSIRZ);
   }
   return interpRZ(rz, PSIRZ_, PSIRZ_c_, PSIRZ_d_, PSIRZ_e_);
}

double G_EQDSK_Data::InterpQRZ(const Vector &rz)
{
   double psi = InterpPsiRZ(rz);

   if (!checkFlag(QPSI))
   {
      initInterpPsi(QPSI_, QPSI_t_);
      setFlag(QPSI);
   }

   return interpPsi(psi, QPSI_, QPSI_t_);
}

void G_EQDSK_Data::InterpNxGradPsiRZ(const Vector &rz, Vector &nxdp)
{
   if (!checkFlag(PSIRZ))
   {
      initInterpRZ(PSIRZ_, PSIRZ_c_, PSIRZ_d_, PSIRZ_e_);
      setFlag(PSIRZ);
   }
   interpNxGradRZ(rz, PSIRZ_, PSIRZ_c_, PSIRZ_d_, PSIRZ_e_, nxdp);
}

void G_EQDSK_Data::InterpBPolRZ(const Vector &rz, Vector &bpol)
{
   InterpNxGradPsiRZ(rz, bpol);
   if (rz[0] > 1e-6 * RDIM_) { bpol /= rz[0]; }
}

double G_EQDSK_Data::InterpBTorRZ(const Vector &rz)
{
   if (rz[0] > 1e-6 * RDIM_)
   {
      return InterpFPolRZ(rz) / rz[0];
   }
   else
   {
      return 0.0;
   }
}

double G_EQDSK_Data::InterpJTorRZ(const Vector &rz)
{
   if (rz[0] > 1e-6 * RDIM_)
   {
      return InterpPPrimeRZ(rz) * rz[0] + InterpFFPrimeRZ(rz) / rz[0];
   }
   else
   {
      return 0.0;
   }
}

void G_EQDSK_Data::initInterpR(const std::vector<double> &v,
                               std::vector<double> &t)
{
   // Initialize the divided differences
   ShiftedVector m(NW_-1, 2); m = 0.0;

   m(-2) = -2.0 * v[2] + 5.0 * v[1] - 3.0 * v[0];
   m(-1) = -1.0 * v[2] + 3.0 * v[1] - 2.0 * v[0];
   for (int i=0; i<NW_-1; i++)
   {
      m(i) = v[i+1] - v[i];
   }
   m(NW_-1) = 2.0 * v[NW_-1] - 3.0 * v[NW_-2] + v[NW_-3];
   m(NW_)   = 3.0 * v[NW_-1] - 5.0 * v[NW_-2] + 2.0 * v[NW_-3];

   // Initialize the Slopes
   t.resize(NW_);

   for (int i=0; i<NW_; i++)
   {
      if (m(i+1) == m(i) && m(i-1) == m(i-2))
      {
         if (m(i) == m(i-1))
         {
            t[i] = m(i) * dr_;
         }
         else
         {
            t[i] = 0.5 * (m(i-1) + m(i)) * dr_;
         }
      }
      else
      {
         t[i] = (fabs(m(i+1) - m(i)) * m(i-1) +
                 fabs(m(i-1) - m(i-2)) * m(i)) * dr_ /
                (fabs(m(i+1) - m(i)) + fabs(m(i-1) - m(i-2)));
      }
   }
}

double G_EQDSK_Data::interpR(double r, const vector<double> &v,
                             const vector<double> &t)
{
   double rs = (r - RLEFT_) / RDIM_;

   int i = std::max(0, std::min((int)floor(double(NW_-1) * rs), NW_-2));

   // Compute ends of local patch
   double r0 = RLEFT_ + RDIM_ * i / (NW_ - 1);
   double r1 = r0 + RDIM_ / (NW_ - 1);

   // Prepare position dependent factors
   double wra = (r1 - r) / dr_;
   double wrb = (r - r0) / dr_;
   double wrc = (1.0 + 2.0 * wra);
   double wrd = (1.0 + 2.0 * wrb);
   double wra2 = wra * wra;
   double wrb2 = wrb * wrb;

   // Extract variable values at ends of local patch
   const double &p0 = v[i];
   const double &p1 = v[i+1];

   double var = p0 * wra2 * wrd + p1 * wrb2 * wrc;

   // Extract dvar/dx at ends of local patch
   const double &px0 = t[i];
   const double &px1 = t[i+1];

   double varx = px0 * wra2 * wrb - px1 * wrb2 * wra;

   var += varx * dr_;

   return var;
}

void G_EQDSK_Data::initInterpRZ(const std::vector<double> &v,
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

double G_EQDSK_Data::interpRZ(const Vector &rz,
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

void G_EQDSK_Data::interpNxGradRZ(const Vector &rz,
                                  const std::vector<double> &v,
                                  const ShiftedDenseMatrix &c,
                                  const ShiftedDenseMatrix &d,
                                  const ShiftedDenseMatrix &e,
                                  Vector &b)
{
   b.SetSize(2);
   b = 0.0;

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
   double wra = (r1 - r) / dr_, dwra = -1.0 / dr_;
   double wrb = (r - r0) / dr_, dwrb = 1.0 / dr_;
   double wrc = (1.0 + 2.0 * wra), dwrc = 2.0 * dwra;
   double wrd = (1.0 + 2.0 * wrb), dwrd = 2.0 * dwrb;
   double wra2 = wra * wra, dwra2 = 2.0 * wra * dwra;
   double wrb2 = wrb * wrb, dwrb2 = 2.0 * wrb * dwrb;

   double wza = (z1 - z) / dz_, dwza = -1.0 / dz_;
   double wzb = (z - z0) / dz_, dwzb = 1.0 / dz_;
   double wzc = (1.0 + 2.0 * wza), dwzc = 2.0 * dwza;
   double wzd = (1.0 + 2.0 * wzb), dwzd = 2.0 * dwzb;
   double wza2 = wza * wza, dwza2 = 2.0 * wza * dwza;
   double wzb2 = wzb * wzb, dwzb2 = 2.0 * wzb * dwzb;

   // Extract var values at corners of local patch
   double p00 = v[NH_ * i + j];
   double p10 = v[NH_ * (i + 1) + j];
   double p01 = v[NH_ * i + j + 1];
   double p11 = v[NH_ * (i + 1) + j + 1];

   b[0] -=
      (p00 * wra2 * wrd + p10 * wrb2 * wrc ) * (dwza2 * wzd + wza2 * dwzd)
      + (p01 * wra2 * wrd + p11 * wrb2 * wrc) * (dwzb2 * wzc + wzb2 * dwzc);
   b[1] +=
      (p00 * wza2 * wzd + p01 * wzb2 * wzc) * (dwra2 * wrd + wra2 * dwrd)
      + (p10 * wza2 * wzd + p11 * wzb2 * wzc) * (dwrb2 * wrc + wrb2 * dwrc);

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

   b[0] -= dr_ *
           ((px00 * wra2 * wrb - px10 * wrb2 * wra) *
            (dwza2 * wzd + wza2 * dwzd) +
            (px01 * wrb * wra2 - px11 * wra * wrb2) *
            (dwzb2 * wzc + wzb2 * dwzc));

   b[1] += dr_ *
           ((px00 * wza2 * wzd + px01 * wzb2 * wzc) *
            (dwra2 * wrb + wra2 * dwrb ) -
            (px10 * wza2 * wzd + px11 * wzb2 * wzc) *
            (dwra * wrb2 + wra * dwrb2));

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

   b[0] -= dz_ *
           ((py00 * wra2 * wrd + py10 * wrb2 * wrc) *
            (dwza2 * wzb + wza2 * dwzb) -
            (py01 * wra2 * wrd + py11 * wrb2 * wrc) *
            (dwza * wzb2 + wza * dwzb2));
   b[1] += dz_ *
           ((py00 * wza2 * wzb - py01 * wza * wzb2) *
            (dwra2 * wrd + wra2 * dwrd) +
            (py10 * wza2 * wzb - py11 * wza * wzb2) *
            (dwrb2 * wrc + wrb2 * dwrc));

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

   b[0] -= dr_ * dz_ * ((pxy00 * wra2 * wrb - pxy10 * wra * wrb2)
                        * (dwza2 * wzb + wza2 * dwzb) +
                        (pxy11 * wra * wrb2 - pxy01 * wra2 * wrb)
                        * (dwza * wzb2 + wza * dwzb2));
   b[1] += dr_ * dz_ * ((pxy00 * wza2 * wzb - pxy01 * wza * wzb2)
                        * (dwra2 * wrb + wra2 * dwrb) +
                        (pxy11 * wza * wzb2 - pxy10 * wza2 * wzb)
                        * (dwra * wrb2 + wra * dwrb2));
}

void G_EQDSK_Data::initInterpPsi(const std::vector<double> &v,
                                 std::vector<double> &t)
{
   // Initialize the divided differences
   ShiftedVector m(NW_-1, 2); m = 0.0;

   m(-2) = -2.0 * v[2] + 5.0 * v[1] - 3.0 * v[0];
   m(-1) = -1.0 * v[2] + 3.0 * v[1] - 2.0 * v[0];
   for (int i=0; i<NW_-1; i++)
   {
      m(i) = v[i+1] - v[i];
   }
   m(NW_-1) = 2.0 * v[NW_-1] - 3.0 * v[NW_-2] + v[NW_-3];
   m(NW_)   = 3.0 * v[NW_-1] - 5.0 * v[NW_-2] + 2.0 * v[NW_-3];

   // Initialize the Slopes
   t.resize(NW_);

   for (int i=0; i<NW_; i++)
   {
      if (m(i+1) == m(i) && m(i-1) == m(i-2))
      {
         if (m(i) == m(i-1))
         {
            t[i] = m(i) * dpsi_;
         }
         else
         {
            t[i] = 0.5 * (m(i-1) + m(i)) * dpsi_;
         }
      }
      else
      {
         t[i] = (fabs(m(i+1) - m(i)) * m(i-1) +
                 fabs(m(i-1) - m(i-2)) * m(i)) * dpsi_ /
                (fabs(m(i+1) - m(i)) + fabs(m(i-1) - m(i-2)));
      }
   }
}

double G_EQDSK_Data::interpPsi(double psi, const vector<double> &v,
                               const vector<double> &t)
{
   double psic = std::max(SIMAG_, std::min(psi, SIBRY_));

   double psis = (psic - SIMAG_) / (SIBRY_ - SIMAG_);

   int i = std::max(0, std::min((int)floor(double(NW_-1) * psis), NW_-2));

   // Compute ends of local patch
   double psi0 = SIMAG_ + (SIBRY_ - SIMAG_) * i / (NW_ - 1);
   double psi1 = psi0 + (SIBRY_ - SIMAG_) / (NW_ - 1);

   // Prepare position dependent factors
   double wra = (psi1 - psic) / dpsi_;
   double wrb = (psic - psi0) / dpsi_;
   double wrc = (1.0 + 2.0 * wra);
   double wrd = (1.0 + 2.0 * wrb);
   double wra2 = wra * wra;
   double wrb2 = wrb * wrb;

   // Extract variable values at ends of local patch
   const double &p0 = v[i];
   const double &p1 = v[i+1];

   double var = p0 * wra2 * wrd + p1 * wrb2 * wrc;

   // Extract dvar/dx at ends of local patch
   const double &px0 = t[i];
   const double &px1 = t[i+1];

   double varx = px0 * wra2 * wrb - px1 * wrb2 * wra;

   var += varx * dpsi_;

   return var;
}

void G_EQDSK_Data::ExtendedDenseMatrix::init()
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
