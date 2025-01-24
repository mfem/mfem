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

#include "cold_plasma_dielectric_coefs.hpp"

using namespace std;
namespace mfem
{
using namespace common;

namespace plasma
{

void StixCoefs_cold_plasma(Vector &V,
                           double omega,
                           double Bmag,
                           double nue,
                           double nui,
                           const Vector & number,
                           const Vector & charge,
                           const Vector & mass,
                           const Vector & temp,
                           int nuprof,
                           bool realPart)
{
   V.SetSize(5);
   complex<double> S(1.0, 0.0);
   complex<double> P(1.0, 0.0);
   complex<double> D(0.0, 0.0);
   complex<double> R(1.0, 0.0);
   complex<double> L(1.0, 0.0);
   double n = number[0];
   double q = charge[0];
   double m = mass[0];
   double Te = temp[0] * q_;
   double coul_log = CoulombLog(n, Te);
   double nuei = (nuprof == 0) ?
                 nu_ei(q, coul_log, m, Te, n) :
                 nue;
   complex<double> collision_correction(1.0, nuei/omega);
   double nui_res = 0.0;

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m;
      if (i == 0) { m_eff = m * collision_correction; }
      if (i == 1) { nui_res = nui; }
      complex<double> w_c =
         omega_c(Bmag, q, m_eff) - complex<double>(0.0, nui_res);
      complex<double> w_p = omega_p(n, q, m_eff);

      S -= w_p * w_p / (omega * omega - w_c * w_c);
      P -= w_p * w_p / (omega * omega);
      D += w_p * w_p * w_c / (omega * (omega * omega - w_c * w_c));
      R -= w_p * w_p / (omega * (omega + w_c));
      L -= w_p * w_p / (omega * (omega - w_c));
   }
   if (realPart)
   {
      V[0] = S.real();
      V[1] = P.real();
      V[2] = D.real();
      V[3] = R.real();
      V[4] = L.real();
   }
   else
   {
      V[0] = S.imag();
      V[1] = P.imag();
      V[2] = D.imag();
      V[3] = R.imag();
      V[4] = L.imag();
   }
}

complex<double> R_cold_plasma(double omega,
                              double Bmag,
                              double nue,
                              double nui,
                              const Vector & number,
                              const Vector & charge,
                              const Vector & mass,
                              const Vector & temp,
                              int nuprof)
{
   complex<double> val(1.0, 0.0);
   double n = number[0];
   double q = charge[0];
   double m = mass[0];
   double Te = temp[0] * q_;
   double coul_log = CoulombLog(n, Te);
   double nuei = (nuprof == 0) ?
                 nu_ei(q, coul_log, m, Te, n) :
                 nue;
   complex<double> collision_correction(1.0, nuei/omega);
   double nui_res = 0.0;

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m;
      if (i == 0) { m_eff = m * collision_correction; }
      if (i == 1) { nui_res = nui; }
      complex<double> w_c =
         omega_c(Bmag, q, m_eff) - complex<double>(0.0, nui_res);
      complex<double> w_p = omega_p(n, q, m_eff);
      val -= w_p * w_p / (omega * (omega + w_c));
   }
   return val;
}

complex<double> L_cold_plasma(double omega,
                              double Bmag,
                              double nue,
                              double nui,
                              const Vector & number,
                              const Vector & charge,
                              const Vector & mass,
                              const Vector & temp,
                              int nuprof,
                              double res_lim)
{
   complex<double> val(1.0, 0.0);
   double n = number[0];
   double q = charge[0];
   double m = mass[0];
   double Te = temp[0] * q_;
   double coul_log = CoulombLog(n, Te);
   double nuei = (nuprof == 0) ?
                 nu_ei(q, coul_log, m, Te, n) :
                 nue;
   complex<double> collision_correction(1.0, nuei/omega);
   double nui_res = 0.0;

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m;
      if (i == 0) { m_eff = m * collision_correction; }
      if (i == 1) { nui_res = nui; }
      complex<double> w_c =
         omega_c(Bmag, q, m_eff) - complex<double>(0.0, nui_res);
      if (res_lim != 0.0)
      {
         double expw_c = exp(-pow(1.0 - w_c.real() / omega, 2));
         w_c -= complex<double>(0.0, res_lim * omega * expw_c);
      }
      complex<double> w_p = omega_p(n, q, m_eff);
      val -= w_p * w_p / (omega * (omega - w_c));
   }
   return val;
}

complex<double> S_cold_plasma(double omega,
                              double Bmag,
                              double nue,
                              double nui,
                              const Vector & number,
                              const Vector & charge,
                              const Vector & mass,
                              const Vector & temp,
                              int nuprof,
                              double res_lim)
{
   complex<double> val(1.0, 0.0);
   double n = number[0];
   double q = charge[0];
   double m = mass[0];
   double Te = temp[0] * q_;
   double coul_log = CoulombLog(n, Te);
   double nuei = (nuprof == 0) ?
                 nu_ei(q, coul_log, m, Te, n) :
                 nue;
   complex<double> collision_correction(1.0, nuei/omega);
   double nui_res = 0.0;

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m;
      if (i == 0) { m_eff = m * collision_correction; }
      if (i == 1) { nui_res = nui; }
      complex<double> w_c =
         omega_c(Bmag, q, m_eff) - complex<double>(0.0, nui_res);
      complex<double> w_c_c = w_c;
      complex<double> num(1.0, 0.0);
      if (res_lim != 0.0)
      {
         double expw_c = exp(-pow(1.0 - w_c.real() / omega, 2));
         w_c_c -= complex<double>(0.0, res_lim * omega * expw_c);
         num += complex<double>(0.0, 0.5 * res_lim * expw_c);
      }
      complex<double> w_p = omega_p(n, q, m_eff);
      val -= w_p * w_p * num / ((omega + w_c) * (omega - w_c_c));

   }
   return val;
}

complex<double> D_cold_plasma(double omega,
                              double Bmag,
                              double nue,
                              double nui,
                              const Vector & number,
                              const Vector & charge,
                              const Vector & mass,
                              const Vector & temp,
                              int nuprof,
                              double res_lim)
{
   complex<double> val(0.0, 0.0);
   double n = number[0];
   double q = charge[0];
   double m = mass[0];
   double Te = temp[0] * q_;
   double coul_log = CoulombLog(n, Te);
   double nuei = (nuprof == 0) ?
                 nu_ei(q, coul_log, m, Te, n) :
                 nue;
   complex<double> collision_correction(1.0, nuei/omega);
   double nui_res = 0.0;

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m;
      if (i == 0) { m_eff = m*collision_correction; }
      if (i == 1) { nui_res = nui; }
      complex<double> w_c =
         omega_c(Bmag, q, m_eff) - complex<double>(0.0, nui_res);
      complex<double> w_c_c = w_c;
      complex<double> num = w_c;
      if (res_lim != 0.0)
      {
         double expw_c = exp(-pow(1.0 - w_c.real() / omega, 2));
         w_c_c -= complex<double>(0.0, res_lim * omega * expw_c);
         num -= complex<double>(0.0, 0.5 * omega * res_lim * expw_c);
      }
      complex<double> w_p = omega_p(n, q, m_eff);
      val += w_p * w_p * num / (omega * (omega + w_c) * (omega - w_c_c));
   }
   return val;
}

complex<double> P_cold_plasma(double omega,
                              double nue,
                              const Vector & number,
                              const Vector & charge,
                              const Vector & mass,
                              const Vector & temp,
                              int nuprof)
{
   complex<double> val(1.0, 0.0);
   double n = number[0];
   double q = charge[0];
   double m = mass[0];
   double Te = temp[0] * q_;
   double coul_log = CoulombLog(n, Te);
   double nuei = (nuprof == 0) ?
                 nu_ei(q, coul_log, m, Te, n) :
                 nue;
   complex<double> collision_correction(1.0, nuei/omega);

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m;
      if (i == 0) { m_eff = m*collision_correction; }
      complex<double> w_p = omega_p(n, q, m_eff);
      val -= w_p * w_p / (omega * omega);
   }
   if (fabs(val.real()) < 1e-4)
   {
      cout << "P near zero: " << val << endl;
   }
   return val;
}
// """""""""""""""""""""""""""""""""""""""""""""""""""""""
// Jim's old sheath impedance parameterization code for Kohno et al 2017
double gabsANY(double x)
{
   return 0.00013542350902761945 - 0.052148081768838075*x +
          0.2834385542799402*x*exp(-x) + 0.03053282857790852*pow(x,
                                                                 2) - 0.006477393187352886*pow(x, 3) + 0.0006099729221975197*pow(x,
                                                                       4) - 0.00002165822780075613*pow(x, 5);
}

double gargANY(double x)
{
   return -1.5282736631594822 + 0.7292398258852378*x - 0.17951815090296652*pow(x,
                                                                               2) + 0.01982701480205563*pow(x, 3) - 0.0008171081897105175*pow(x,
                                                                                     4) + 1.8339439276641656*tanh(0.91*x);
}

complex<double> ficmplxANY1(double x)
{
   complex<double> val(0,1);
   return gabsANY(x)*exp(val*gargANY(x));
}

double xafun(double x)
{
   return 1.0042042386309231 - 0.040348243785481734*x + 0.0015928747555414683*pow(
             x, 2) - 0.000028423592601970268*pow(x, 3)+ 0.06435835731325179*tanh(0.5*x);
}

double maxfun(double x)
{
   return 0.10853383673713796 - 0.006244170771533145*x +
          0.00024234826741913128*pow(x, 2) - 4.199121776132657e-6*pow(x,
                                                                      3)+ 0.008906384119401499*tanh(0.5*x);
}

complex<double> ficmplxANY(double omega, double vpp)
{return (maxfun(vpp)/maxfun(10))*ficmplxANY1(omega*xafun(10)/xafun(vpp));}

double vrectfun(double x)
{   return 3.069981715829813 + 0.06248514679413549*x + 0.04681196334146159*pow(x, 2) - 0.002436325220160285*pow(x, 3) + 0.00004692674475799567*pow(x, 4);}

complex<double> fdcmplxANY(double omega, double vpp)
{
   complex<double> val(0,1);
   double delta = pow(vrectfun(vpp),(3.0/4.0));
   complex<double> zinvd = -val*omega/delta;
   return zinvd;
}

complex<double> fecmplxANY(double vpp)
{   return 1.4912697017617276 - 0.39149171343072803*exp(-vpp) - 0.35029055741496556*vpp + 0.04370283762859965*pow(vpp, 2) - 0.0029806529060161235*pow(vpp, 3) + 0.00010448291016092317*pow(vpp, 4)- 1.4687221698127053e-6*pow(vpp, 5);}

complex<double> ftotcmplxANY(double omega, double vpp)
{ return ficmplxANY(omega, vpp) + fdcmplxANY(omega, vpp) + fecmplxANY(vpp);}
// """""""""""""""""""""""""""""""""""""""""""""""""""""""

double mu(double mass_e, double mass_i)
{
   return sqrt(mass_i / (2.0 * M_PI * mass_e));
}

double ff(double x)
{
   double a0 = 3.18553;
   double a1 = 3.70285;
   double a2 = 3.81991;
   double b1 = 1.13352;
   double b2 = 1.24171;
   double a3 = (2.0 * b2) / M_PI;
   double num = a0 + (a1 + (a2 + a3 * x) * x) * x;
   double den = 1.0 + (b1 + b2 * x) * x;

   return (num / den);
}

double gg(double w)
{
   double c0 = 0.966463;
   double c1 = 0.141639;

   return (c0+c1*tanh(w));
}

double phi0avg(double w, double xi)
{
   return (ff(gg(w)*xi));
}

double he(double x)
{
   double h1 = 0.607405;
   double h2 = 0.325497;
   double g1 = 0.624392;
   double g2 = 0.500595;
   double g3 = (M_PI * h2) / 4.0;
   double num = 1.0 + (h1 + h2 * x) * x;
   double den = 1.0 + (g1 + (g2 + g3 * x) * x) * x;

   return (num/den);
}

double phips(double bx, double wci, double mass_e, double mass_i)
{
   double mu_val = mu(mass_e, mass_i);
   double d3 = 0.995721;
   double arg = sqrt((mu_val * mu_val * bx * bx + 1.0)/(mu_val * mu_val + 1.0));
   double num = -log(arg);
   double den = 1.0 + d3 * wci * wci;

   return (num/den);
}

double niw(double wci, double bx, double phi, double mass_e,
           double mass_i)
{
   double d0 = 0.794443;
   double d1 = 0.803531;
   double d2 = 0.182378;
   double d4 = 0.0000901468;
   double nu1 = 1.455592;
   double abx = fabs(bx);
   double phid = phi - phips(abx,wci, mass_e, mass_i);
   double pre = d0 /(d2 + sqrt(phid));
   double wcip = wci * pow(phid, 0.25);
   double num = abx * abx + d4 + d1 * d1 * pow(wcip, (2.0*nu1));
   double den = 1.0 + d4 + d1 * d1 * pow(wcip, (2.0 * nu1));

   return (pre*sqrt(num/den));
}

double ye(double bx, double xi)
{
   double h0 = 1.161585;
   double abx = fabs(bx);

   return (h0*abx*he(xi));
}

double niww(double w, double wci, double bx, double xi,
            double mass_e, double mass_i)
{
   double k0 = 3.7616;
   double k1 = 0.22202;
   double phis = k0 + k1*(xi-k0);
   double phipr = phis + (phi0avg(w,xi)-phis)*tanh(w);

   return (niw(wci,bx,phipr,mass_e,mass_i));
}

complex<double> yd(double w, double wci, double bx, double xi,
                   double mass_e, double mass_i)
{
   double s0 = 1.12415;
   double delta = sqrt(phi0avg(w,xi)/niww(w,wci,bx,xi,mass_e,mass_i));
   complex<double> val(0.0, 1.0);

   return (-s0*val*(w/delta));
}

complex<double> yi(double w, double wci, double bx, double xi,
                   double mass_e, double mass_i)
{
   complex<double> val(0.0, 1.0);
   double p0 = 1.05554;
   complex<double> p1(0.797659, 0.0);
   double p2 = 1.47405;
   double p3 = 0.809615;
   double eps = 0.0001;
   complex<double> gfactornum(w * w - bx * bx * wci * wci, eps);
   complex<double> gfactorden(w * w - wci * wci, eps);
   complex<double> gfactor = gfactornum/gfactorden;
   double niwwa = niww(w,wci,bx,xi,mass_e,mass_i);
   double phi0avga = phi0avg(w,xi);
   complex<double> gamcup(fabs(bx)/(niwwa*sqrt(phi0avga)), 0.0);
   complex<double> wcup(p3*w/sqrt(niwwa), 0.0);
   complex<double> yicupden1 = wcup * wcup/gfactor - p1;
   complex<double> yicupden2 = p2*gamcup*wcup*val;
   complex<double> yicupden = yicupden1 + yicupden2;
   complex<double> yicup = val*p0*wcup/yicupden;

   return ((niwwa * yicup) / sqrt(phi0avga));
}

complex<double> ytot(double w, double wci, double bx, double xi,
                     double mass_e, double mass_i)
{
   complex<double> ytot = ye(bx,xi) +
                          yd(w,wci,bx,xi,mass_e,mass_i) + yi(w,wci,bx,xi,mass_e,mass_i);
   return (ytot);
}

double debye(double Te, double n0)
{
   //return (7.43e2*pow((Te/n0_cm),0.5));
   return sqrt((epsilon0_*Te*q_)/(n0*q_*q_));
}

SheathBase::SheathBase(const BlockVector & density,
                       const BlockVector & temp,
                       const ParFiniteElementSpace & L2FESpace,
                       const ParFiniteElementSpace & H1FESpace,
                       double omega,
                       const Vector & charges,
                       const Vector & masses,
                       bool realPart)
   : density_(density),
     temp_(temp),
     potential_(NULL),
     L2FESpace_(L2FESpace),
     H1FESpace_(H1FESpace),
     omega_(omega),
     realPart_(realPart),
     charges_(charges),
     masses_(masses)
{
}

SheathBase::SheathBase(const SheathBase &sb, bool realPart)
   : density_(sb.density_),
     temp_(sb.temp_),
     potential_(sb.potential_),
     L2FESpace_(sb.L2FESpace_),
     H1FESpace_(sb.H1FESpace_),
     omega_(sb.omega_),
     realPart_(realPart),
     charges_(sb.charges_),
     masses_(sb.masses_)
{}

double SheathBase::EvalIonDensity(ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   density_gf_.MakeRef(const_cast<ParFiniteElementSpace*>(&L2FESpace_),
                       density_.GetBlock(1).GetData());
   return density_gf_.GetValue(T, ip);
}

double SheathBase::EvalElectronTemp(ElementTransformation &T,
                                    const IntegrationPoint &ip)
{
   temperature_gf_.MakeRef(const_cast<ParFiniteElementSpace*>(&H1FESpace_),
                           temp_.GetBlock(0).GetData());
   return temperature_gf_.GetValue(T, ip);
}

complex<double> SheathBase::EvalSheathPotential(ElementTransformation &T,
                                                const IntegrationPoint &ip)
{
   double phir = (potential_) ? potential_->real().GetValue(T, ip) : 0.0 ;
   double phii = (potential_) ? potential_->imag().GetValue(T, ip) : 0.0 ;
   return complex<double>(phir, phii);
}

RectifiedSheathPotential::RectifiedSheathPotential(
   const BlockVector & density,
   const BlockVector & temp,
   const ParFiniteElementSpace & L2FESpace,
   const ParFiniteElementSpace & H1FESpace,
   double omega,
   const Vector & charges,
   const Vector & masses,
   bool realPart)
   : SheathBase(density, temp, L2FESpace, H1FESpace,
                omega, charges, masses, realPart)
{}

double RectifiedSheathPotential::Eval(ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   double density_val = EvalIonDensity(T, ip);
   double temp_val = EvalElectronTemp(T, ip);

   double wpi = omega_p(density_val, charges_[1], masses_[1]);
   double w_norm = omega_ / wpi;

   complex<double> phi = EvalSheathPotential(T, ip);
   double phi_mag = sqrt(pow(phi.real(), 2) + pow(phi.imag(), 2));
   double volt_norm = (phi_mag)/temp_val ; // V zero-to-peak
   if (isnan(volt_norm)) {volt_norm = 0.0;}

   // double phiRec = phi0avg(w_norm, volt_norm);

   return phi0avg(w_norm, volt_norm) * temp_val;
}

SheathImpedance::SheathImpedance(const ParGridFunction & B,
                                 const BlockVector & density,
                                 const BlockVector & temp,
                                 const ParFiniteElementSpace & L2FESpace,
                                 const ParFiniteElementSpace & H1FESpace,
                                 double omega,
                                 const Vector & charges,
                                 const Vector & masses,
                                 bool realPart)
   : SheathBase(density, temp, L2FESpace, H1FESpace,
                omega, charges, masses, realPart),
     B_(B)
{}

complex<double> SheathImpedance::z(const complex<double> &phi,
                                   ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   // Collect density, temperature, magnetic field, and potential field values
   Vector B(3);
   B_.GetVectorValue(T, ip, B);
   double Bmag = B.Norml2();                         // Units: T

   double density_val = EvalIonDensity(T, ip);       // Units: # / m^3
   double temp_val = EvalElectronTemp(T, ip);        // Units: eV

   double wci = omega_c(Bmag, charges_[1], masses_[1]);        // Units: s^{-1}
   double wpi = omega_p(density_val, charges_[1], masses_[1]); // Units: s^{-1}

   double w_norm = omega_ / wpi; // Unitless
   double wci_norm = wci / wpi;  // Unitless
   double phi_mag = sqrt(pow(phi.real(), 2) + pow(phi.imag(), 2));
   //double temp_val = 10.0; // 10 eV
   double debye_length = debye(temp_val, density_val); // Units: m
   Vector nor(3); nor = 0.0;
   nor.SetSize(T.GetSpaceDim());
   CalcOrtho(T.Jacobian(), nor);
   nor.SetSize(3);
   double normag = nor.Norml2();
   double bn = (B * nor)/(normag*Bmag); // Unitless
   // cout << "bn " << bn << endl;
   // cout << "wci " << wci << endl;
   // cout << "wpi " << wpi << endl;
   // cout << "dl " << debye_length << endl;
   // Setting up normalized V_RF:
   // Jim's newest parametrization (Myra et al 2017):
   double volt_norm = (phi_mag)/temp_val ; // Unitless: V zero-to-peak
   // cout << "vn " << volt_norm << endl;
   // Jim's old parametrization (Kohno et al 2017):
   //double volt_norm = (2*phi_mag)/temp_val ; // Unitless: V peak-to-peak

   //if ( volt_norm == 0){volt_norm = 190.5/temp_val;} // Initial Guess
   // This is only for old parameterization
   //if ( volt_norm > 20) {cout << "Warning: V_RF > Z Parameterization Limit!" << endl;}

   // Calculating Sheath Impedance:
   // Jim's newest parametrization (Myra et al 2017):
   complex<double> zsheath_norm = 1.0 / ytot(w_norm, wci_norm, bn, volt_norm,
                                             masses_[0], masses_[1]);
   // cout << "m0 " << masses_[0] << endl;
   // cout << "m1 " << masses_[1] << endl;
   // cout << "ytot " << 1.0 / zsheath_norm << endl;
   // Jim's old parametrization (Kohno et al 2017):
   //complex<double> zsheath_norm = 1.0 / ftotcmplxANY(w_norm, volt_norm);

   // Fixed sheath impedance:
   //complex<double> zsheath_norm(0.6, 0.4);

   /*
   cout << "Check 1:" << phi0avg(0.4, 6.) - 6.43176481712605 << endl;
   cout << "Check 2:" << niw(.2, .3, 13,masses_[0], masses_[1])- 0.07646452845544677 << endl;
   cout << "Check 3:" << niww(0.2, 0.3, 0.4, 13,masses_[0], masses_[1]) - 0.14077643642166277 << endl;
   cout << "Check 4:" << yd(0.2, 0.3, 0.4, 13,masses_[0], masses_[1]).imag()+0.025738204728120898 << endl;
   cout << "Check 5: " << ye(0.4, 3.6) - 0.1588274616204441 << endl;
   cout << "Check 6:" << yi(0.2, 0.3, 0.4, 13,masses_[0], masses_[1]).real() - 0.006543897148693344 << yi(0.2, 0.3, 0.4, 13,masses_[0], masses_[1]).imag()+0.013727440802110503 << endl;
   cout << "Check 7:" << ytot(0.2, 0.3, 0.4, 13,masses_[0], masses_[1]).real()-0.05185050837032144 << ytot(0.2, 0.3, 0.4, 13,masses_[0], masses_[1]).imag()+0.0394656455302314 << endl;
    */
   if (isnan(zsheath_norm.real()))
   {
      zsheath_norm = complex<double>(0.0, zsheath_norm.imag());
   }
   if (isnan(zsheath_norm.imag()))
   {
      zsheath_norm = complex<double>(zsheath_norm.real(), 0.0);
   }
   return zsheath_norm*debye_length/(epsilon0_*wpi); // Units: Ohm m^2
   /*
    if (realPart_)
    {
       return (zsheath_norm.real()*debye_length)/(epsilon0_*wpi); // Units: Ohm m^2

    }
    else
    {
       return (zsheath_norm.imag()*debye_length)/(epsilon0_*wpi); // Units: Ohm m^2
    }
   */
}

double SheathImpedance::Eval(ElementTransformation &T,
                             const IntegrationPoint &ip)
{
   complex<double> phi = EvalSheathPotential(T, ip); // Units: V

   complex<double> zsheath = z(phi, T, ip);

   if (realPart_)
   {
      return zsheath.real();
   }
   else
   {
      return zsheath.imag();
   }
}

StixCoefBase::StixCoefBase(const ParGridFunction & B,
                           const ParGridFunction & nue,
                           const ParGridFunction & nui,
                           const BlockVector & density,
                           const BlockVector & temp,
                           const ParFiniteElementSpace & L2FESpace,
                           const ParFiniteElementSpace & H1FESpace,
                           double omega,
                           const Vector & charges,
                           const Vector & masses,
                           int nuprof,
                           double res_lim,
                           bool realPart)
   : B_(B),
     nue_(nue),
     nui_(nui),
     density_(density),
     temp_(temp),
     L2FESpace_(L2FESpace),
     H1FESpace_(H1FESpace),
     omega_(omega),
     realPart_(realPart),
     nuprof_(nuprof),
     res_lim_(res_lim),
     BVec_(3),
     charges_(charges),
     masses_(masses)
{
   density_vals_.SetSize(charges_.Size());
   temp_vals_.SetSize(charges_.Size());
}

StixCoefBase::StixCoefBase(StixCoefBase & s)
   : B_(s.GetBField()),
     nue_(s.GetNue()),
     nui_(s.GetNui()),
     density_(s.GetDensityFields()),
     temp_(s.GetTemperatureFields()),
     L2FESpace_(s.GetDensityFESpace()),
     H1FESpace_(s.GetTemperatureFESpace()),
     omega_(s.GetOmega()),
     realPart_(s.GetRealPartFlag()),
     nuprof_(s.GetNuProf()),
     res_lim_(s.GetResonanceLimitorFactor()),
     BVec_(3),
     charges_(s.GetCharges()),
     masses_(s.GetMasses())
{
   density_vals_.SetSize(charges_.Size());
   temp_vals_.SetSize(charges_.Size());
}

double StixCoefBase::getBMagnitude(ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   B_.GetVectorValue(T.ElementNo, ip, BVec_);

   return BVec_.Norml2();
}

void StixCoefBase::fillDensityVals(ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   double min_den = 1e14;

   for (int i=0; i<density_vals_.Size(); i++)
   {
      density_gf_.MakeRef(const_cast<ParFiniteElementSpace*>(&L2FESpace_),
                          density_.GetBlock(i).GetData());
      density_vals_[i] = std::max(density_gf_.GetValue(T.ElementNo, ip),
                                  min_den);
   }
}

void StixCoefBase::fillTemperatureVals(ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   for (int i=0; i<temp_vals_.Size(); i++)
   {
      temperature_gf_.MakeRef(const_cast<ParFiniteElementSpace*>(&H1FESpace_),
                              temp_.GetBlock(i).GetData());
      temp_vals_[i] = temperature_gf_.GetValue(T.ElementNo, ip);
   }
}

StixLCoef::StixLCoef(const ParGridFunction & B,
                     const ParGridFunction & nue,
                     const ParGridFunction & nui,
                     const BlockVector & density,
                     const BlockVector & temp,
                     const ParFiniteElementSpace & L2FESpace,
                     const ParFiniteElementSpace & H1FESpace,
                     double omega,
                     const Vector & charges,
                     const Vector & masses,
                     int nuprof,
                     double res_lim,
                     bool realPart)
   : StixCoefBase(B, nue, nui, density, temp, L2FESpace, H1FESpace, omega,
                  charges, masses, nuprof, res_lim, realPart)
{}

double StixLCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density, temperature, and magnetic field values
   double Bmag = this->getBMagnitude(T, ip);
   nue_vals_ = nue_.GetValue(T, ip);
   nui_vals_ = nui_.GetValue(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex<double> L = L_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                                     density_vals_, charges_, masses_,
                                     temp_vals_, nuprof_, res_lim_);

   // Return the selected component
   if (realPart_)
   {
      return L.real();
   }
   else
   {
      return L.imag();
   }
}

StixRCoef::StixRCoef(const ParGridFunction & B,
                     const ParGridFunction & nue,
                     const ParGridFunction & nui,
                     const BlockVector & density,
                     const BlockVector & temp,
                     const ParFiniteElementSpace & L2FESpace,
                     const ParFiniteElementSpace & H1FESpace,
                     double omega,
                     const Vector & charges,
                     const Vector & masses,
                     int nuprof,
                     double res_lim,
                     bool realPart)
   : StixCoefBase(B, nue, nui, density, temp, L2FESpace, H1FESpace, omega,
                  charges, masses, nuprof, res_lim, realPart)
{}

double StixRCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density, temperature, and magnetic field values
   double Bmag = this->getBMagnitude(T, ip);
   nue_vals_ = nue_.GetValue(T, ip);
   nui_vals_ = nui_.GetValue(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex<double> R = R_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                                     density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);

   // Return the selected component
   if (realPart_)
   {
      return R.real();
   }
   else
   {
      return R.imag();
   }
}

StixSCoef::StixSCoef(const ParGridFunction & B,
                     const ParGridFunction & nue,
                     const ParGridFunction & nui,
                     const BlockVector & density,
                     const BlockVector & temp,
                     const ParFiniteElementSpace & L2FESpace,
                     const ParFiniteElementSpace & H1FESpace,
                     double omega,
                     const Vector & charges,
                     const Vector & masses,
                     int nuprof,
                     double res_lim,
                     bool realPart)
   : StixCoefBase(B, nue, nui, density, temp, L2FESpace, H1FESpace, omega,
                  charges, masses, nuprof, res_lim, realPart)
{}

double StixSCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density, temperature, and magnetic field values
   double Bmag = this->getBMagnitude(T, ip);
   nue_vals_ = nue_.GetValue(T, ip);
   nui_vals_ = nui_.GetValue(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex<double> S = S_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                                     density_vals_, charges_, masses_,
                                     temp_vals_, nuprof_, res_lim_);

   // Return the selected component
   if (realPart_)
   {
      return S.real();
   }
   else
   {
      return S.imag();
   }
}

StixDCoef::StixDCoef(const ParGridFunction & B,
                     const ParGridFunction & nue,
                     const ParGridFunction & nui,
                     const BlockVector & density,
                     const BlockVector & temp,
                     const ParFiniteElementSpace & L2FESpace,
                     const ParFiniteElementSpace & H1FESpace,
                     double omega,
                     const Vector & charges,
                     const Vector & masses,
                     int nuprof,
                     double res_lim,
                     bool realPart)
   : StixCoefBase(B, nue, nui, density, temp, L2FESpace, H1FESpace, omega,
                  charges, masses, nuprof, res_lim, realPart)
{}

double StixDCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density, temperature, and magnetic field values
   double Bmag = this->getBMagnitude(T, ip);
   nue_vals_ = nue_.GetValue(T, ip);
   nui_vals_ = nui_.GetValue(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex<double> D = D_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                                     density_vals_, charges_, masses_,
                                     temp_vals_, nuprof_, res_lim_);

   // Return the selected component
   if (realPart_)
   {
      return D.real();
   }
   else
   {
      return D.imag();
   }
}

StixPCoef::StixPCoef(const ParGridFunction & B,
                     const ParGridFunction & nue,
                     const ParGridFunction & nui,
                     const BlockVector & density,
                     const BlockVector & temp,
                     const ParFiniteElementSpace & L2FESpace,
                     const ParFiniteElementSpace & H1FESpace,
                     double omega,
                     const Vector & charges,
                     const Vector & masses,
                     int nuprof,
                     double res_lim,
                     bool realPart)
   : StixCoefBase(B, nue, nui, density, temp, L2FESpace, H1FESpace, omega,
                  charges, masses, nuprof, res_lim, realPart)
{}

double StixPCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density and temperature field values
   nue_vals_ = nue_.GetValue(T, ip);
   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex<double> P = P_cold_plasma(omega_, nue_vals_, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);

   // Return the selected component
   if (realPart_)
   {
      return P.real();
   }
   else
   {
      return P.imag();
   }
}

StixDensityCoef::StixDensityCoef(const ParGridFunction & B,
                                 const ParGridFunction & nue,
                                 const ParGridFunction & nui,
                                 const BlockVector & density,
                                 const BlockVector & temp,
                                 const ParFiniteElementSpace & L2FESpace,
                                 const ParFiniteElementSpace & H1FESpace,
                                 double omega,
                                 const Vector & charges,
                                 const Vector & masses,
                                 int nuprof,
                                 double res_lim,
                                 bool realPart,
                                 double a, int p)
   : StixCoefBase(B, nue, nui, density, temp, L2FESpace, H1FESpace, omega,
                  charges, masses, nuprof, res_lim, realPart), a_(a), p_(p)
{}

double StixDensityCoef::Eval(ElementTransformation &T,
                             const IntegrationPoint &ip)
{
   // Collect density and temperature field values
   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   return a_ * pow(density_vals_[0], p_);
}

StixTensorBase::StixTensorBase(const ParGridFunction & B,
                               const ParGridFunction & nue,
                               const ParGridFunction & nui,
                               const BlockVector & density,
                               const BlockVector & temp,
                               const ParFiniteElementSpace & L2FESpace,
                               const ParFiniteElementSpace & H1FESpace,
                               double omega,
                               const Vector & charges,
                               const Vector & masses,
                               int nuprof,
                               double res_lim,
                               bool realPart)
   : StixCoefBase(B, nue, nui, density, temp, L2FESpace, H1FESpace,
                  omega, charges, masses, nuprof, res_lim, realPart)
{}

void StixTensorBase::addParallelComp(double P, DenseMatrix & eps)
{
   // For b = B/|B|, add P * b b^T to epsilon
   for (int i=0; i<3; i++)
   {
      eps(i,i) += P * BVec_(i) * BVec_(i);
      for (int j = i+1; j<3; j++)
      {
         double eij = P * BVec_(i) * BVec_(j);
         eps(i,j) += eij;
         eps(j,i) += eij;
      }
   }
}

void StixTensorBase::addPerpDiagComp(double S, DenseMatrix & eps)
{
   // For b = B/|B|, add S * (I - b b^T) to epsilon
   for (int i=0; i<3; i++)
   {
      eps(i,i) += S * (1.0 - BVec_(i) * BVec_(i));
      for (int j = i+1; j<3; j++)
      {
         double eij = S * BVec_(i) * BVec_(j);
         eps(i,j) -= eij;
         eps(j,i) -= eij;
      }
   }
}

void StixTensorBase::addPerpSkewComp(double D, DenseMatrix & eps)
{
   // For b = B/|B|, add D * b\times to epsilon
   eps(1,2) -= D * BVec_[0];
   eps(2,1) += D * BVec_[0];

   eps(2,0) -= D * BVec_[1];
   eps(0,2) += D * BVec_[1];

   eps(0,1) -= D * BVec_[2];
   eps(1,0) += D * BVec_[2];
}

DielectricTensor::DielectricTensor(const ParGridFunction & B,
                                   const ParGridFunction & nue,
                                   const ParGridFunction & nui,
                                   const BlockVector & density,
                                   const BlockVector & temp,
                                   const ParFiniteElementSpace & L2FESpace,
                                   const ParFiniteElementSpace & H1FESpace,
                                   double omega,
                                   const Vector & charges,
                                   const Vector & masses,
                                   int nuprof,
                                   double res_lim,
                                   bool realPart)
   : MatrixCoefficient(3),
     StixTensorBase(B, nue, nui, density, temp, L2FESpace, H1FESpace, omega,
                    charges, masses, nuprof, res_lim, realPart)
{}

void DielectricTensor::Eval(DenseMatrix &epsilon, ElementTransformation &T,
                            const IntegrationPoint &ip)
{
   // Initialize dielectric tensor to appropriate size
   epsilon.SetSize(3); epsilon = 0.0;

   // Collect density, temperature, and magnetic field values
   double Bmag = this->getBMagnitude(T, ip);
   BVec_ /= Bmag;
   nue_vals_ = nue_.GetValue(T, ip);
   nui_vals_ = nui_.GetValue(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate the Stix Coefficients
   complex<double> S = S_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                                     density_vals_, charges_, masses_,
                                     temp_vals_, nuprof_, res_lim_);
   complex<double> P = P_cold_plasma(omega_, nue_vals_, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);
   complex<double> D = D_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                                     density_vals_, charges_, masses_,
                                     temp_vals_, nuprof_, res_lim_);

   this->addParallelComp(realPart_ ?  P.real() : P.imag(), epsilon);
   this->addPerpDiagComp(realPart_ ?  S.real() : S.imag(), epsilon);
   this->addPerpSkewComp(realPart_ ? -D.imag() : D.real(), epsilon);
   /*
   if (realPart_)
   {
      epsilon(0,0) =  (real(P) - real(S)) *
                      pow(sin(ph), 2) * pow(cos(th), 2) + real(S);
      epsilon(1,1) =  (real(P) - real(S)) * pow(cos(ph), 2) + real(S);
      epsilon(2,2) =  (real(P) - real(S)) *
                      pow(sin(ph), 2) * pow(sin(th), 2) + real(S);
      epsilon(0,1) = (real(P) - real(S)) * cos(ph) * cos(th) * sin(ph) -
                     imag(D) * sin(th) * sin(ph);
      epsilon(0,2) =  (real(P) - real(S)) *
                      pow(sin(ph), 2) * sin(th) * cos(th) + imag(D) * cos(ph);
      epsilon(1,2) = (real(P) - real(S)) * sin(th) * cos(ph) * sin(ph) -
                     imag(D) * cos(th) * sin(ph);
      epsilon(1,0) = (real(P) - real(S)) * cos(ph) * cos(th) * sin(ph) +
                     imag(D) * sin(th) * sin(ph);
      epsilon(2,1) = (real(P) - real(S)) * sin(th) * cos(ph) * sin(ph) +
                     imag(D) * cos(th) * sin(ph);
      epsilon(2,0) = (real(P) - real(S)) *
                     pow(sin(ph),2) * sin(th) * cos(th) - imag(D) * cos(ph);
   }
   else
   {
      epsilon(0,0) = (imag(P) - imag(S)) *
                     pow(sin(ph), 2) * pow(cos(th), 2) + imag(S);
      epsilon(1,1) = (imag(P) - imag(S)) * pow(cos(ph), 2) + imag(S);
      epsilon(2,2) = (imag(P) - imag(S)) *
                     pow(sin(ph), 2) * pow(sin(th), 2) + imag(S);
      epsilon(0,1) = (imag(P) - imag(S)) * cos(ph) * cos(th) * sin(ph) +
                     real(D) * sin(th) * sin(ph);
      epsilon(0,2) = (imag(P) - imag(S)) *
                     pow(sin(ph), 2) * sin(th) * cos(th) - real(D) * cos(ph);
      epsilon(1,2) = (imag(P) - imag(S)) * sin(th) * cos(ph) * sin(ph) +
                     real(D) * cos(th) * sin(ph);
      epsilon(1,0) = (imag(P) - imag(S)) * cos(ph) * cos(th) * sin(ph) -
                     real(D) * sin(th) * sin(ph);
      epsilon(2,1) = (imag(P) - imag(S)) * sin(th) * cos(ph) * sin(ph) -
                     real(D) * cos(th) * sin(ph);
      epsilon(2,0) = (imag(P) - imag(S)) *
                     pow(sin(ph), 2) * sin(th) * cos(th) + real(D) * cos(ph);
   }
   */
   epsilon *= epsilon0_;

   /*
   Vector lambda(3);
   epsilon.Eigenvalues(lambda);
   if (realPart_)
      cout << "Dielectric tensor eigenvalues: "
           << lambda[0] << " " << lambda[1] << " " << lambda[2]
      << " for B " << B[0] << " " << B[1] << " " << B[2]
      << " and rho " << density_vals_[0]
      << " " << density_vals_[1] << " " << density_vals_[2]
      << endl;
   else
      cout << "Conductivity tensor eigenvalues: "
           << lambda[0] << " " << lambda[1] << " " << lambda[2] << endl;
   */
}

InverseDielectricTensor::InverseDielectricTensor(
   const ParGridFunction & B,
   const ParGridFunction & nue,
   const ParGridFunction & nui,
   const BlockVector & density,
   const BlockVector & temp,
   const ParFiniteElementSpace & L2FESpace,
   const ParFiniteElementSpace & H1FESpace,
   double omega,
   const Vector & charges,
   const Vector & masses,
   int nuprof,
   double res_lim,
   bool realPart)
   : MatrixCoefficient(3),
     StixTensorBase(B, nue, nui, density, temp, L2FESpace, H1FESpace, omega,
                    charges, masses, nuprof, res_lim, realPart)
{}

void InverseDielectricTensor::Eval(DenseMatrix &epsilonInv,
                                   ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   // Initialize dielectric tensor to appropriate size
   epsilonInv.SetSize(3); epsilonInv = 0.0;

   // Collect density, temperature, and magnetic field values
   double Bmag = this->getBMagnitude(T, ip);
   BVec_ /= Bmag;
   nue_vals_ = nue_.GetValue(T, ip);
   nui_vals_ = nui_.GetValue(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate the Stix Coefficients
   complex<double> S = S_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                                     density_vals_, charges_, masses_,
                                     temp_vals_, nuprof_, res_lim_);
   complex<double> P = P_cold_plasma(omega_, nue_vals_, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);
   complex<double> D = D_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                                     density_vals_, charges_, masses_,
                                     temp_vals_, nuprof_, res_lim_);

   complex<double> Q = S * S - D * D;
   complex<double> QInv = 1.0 / Q;
   complex<double> SInv = S * QInv;
   complex<double> PInv = 1.0 / P;
   complex<double> DInv = D * QInv;

   this->addParallelComp(realPart_ ? PInv.real() :  PInv.imag(), epsilonInv);
   this->addPerpDiagComp(realPart_ ? SInv.real() :  SInv.imag(), epsilonInv);
   this->addPerpSkewComp(realPart_ ? DInv.imag() : -DInv.real(), epsilonInv);

   epsilonInv *= 1.0 / epsilon0_;
}

SPDDielectricTensor::SPDDielectricTensor(
   const ParGridFunction & B,
   const ParGridFunction & nue,
   const ParGridFunction & nui,
   const BlockVector & density,
   const BlockVector & temp,
   const ParFiniteElementSpace & L2FESpace,
   const ParFiniteElementSpace & H1FESpace,
   double omega,
   const Vector & charges,
   const Vector & masses,
   int nuprof,
   double res_lim)
   : MatrixCoefficient(3),
     StixCoefBase(B, nue, nui, density, temp, L2FESpace, H1FESpace, omega,
                  charges, masses, nuprof, res_lim, true)
{}

void SPDDielectricTensor::Eval(DenseMatrix &epsilon, ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   // Initialize dielectric tensor to appropriate size
   epsilon.SetSize(3);

   // Collect density, temperature, and magnetic field values
   double Bmag = this->getBMagnitude(T, ip);
   BVec_ /= Bmag;
   nue_vals_ = nue_.GetValue(T, ip);
   nui_vals_ = nui_.GetValue(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);
   /*
   Vector B(3);
   B_.GetVectorValue(T.ElementNo, ip, B);
   double Bmag = B.Norml2();
   double th = atan2(B(2), B(0));
   double ph = atan2(B(0) * cos(th) + B(2) * sin(th), -B(1));

   for (int i=0; i<density_vals_.Size(); i++)
   {
      density_gf_.MakeRef(const_cast<ParFiniteElementSpace*>(&L2FESpace_),
                          const_cast<Vector&>(density_.GetBlock(i)));
      density_vals_[i] = density_gf_.GetValue(T.ElementNo, ip);
   }

   for (int i=0; i<temp_vals_.Size(); i++)
   {
      temperature_gf_.MakeRef(const_cast<ParFiniteElementSpace*>(&H1FESpace_),
                              const_cast<Vector&>(temp_.GetBlock(i)));
      temp_vals_[i] = temperature_gf_.GetValue(T.ElementNo, ip);
   }
   */
   complex<double> S = S_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                                     density_vals_, charges_, masses_,
                                     temp_vals_, nuprof_, res_lim_);
   complex<double> P = P_cold_plasma(omega_, nue_vals_, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);
   complex<double> D = D_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                                     density_vals_, charges_, masses_,
                                     temp_vals_, nuprof_, res_lim_);

   epsilon(0,0) = abs(S + (P - S) * BVec_(0) * BVec_(0));
   epsilon(1,1) = abs(S + (P - S) * BVec_(1) * BVec_(1));
   epsilon(2,2) = abs(S + (P - S) * BVec_(2) * BVec_(2));
   epsilon(0,1) = abs((P - S) * BVec_(0) * BVec_(1) - D * BVec_(2));
   epsilon(1,0) = abs((P - S) * BVec_(1) * BVec_(0) + D * BVec_(2));
   epsilon(0,2) = abs((P - S) * BVec_(0) * BVec_(2) + D * BVec_(1));
   epsilon(2,0) = abs((P - S) * BVec_(2) * BVec_(0) - D * BVec_(1));
   epsilon(1,2) = abs((P - S) * BVec_(1) * BVec_(2) - D * BVec_(0));
   epsilon(2,1) = abs((P - S) * BVec_(2) * BVec_(1) + D * BVec_(0));
   /*
   epsilon(0,0) = abs((P - S) * pow(sin(ph), 2) * pow(cos(th), 2) + S);
   epsilon(1,1) = abs((P - S) * pow(cos(ph), 2) + S);
   epsilon(2,2) = abs((P - S) * pow(sin(ph), 2) * pow(sin(th), 2) + S);
   epsilon(0,1) = abs((P - S) * cos(ph) * cos(th) * sin(ph) -
                      D * sin(th) * sin(ph));
   epsilon(0,2) = abs((P - S) * pow(sin(ph), 2) * sin(th) * cos(th) -
                      D * cos(ph));
   epsilon(1,2) = abs((P - S) * sin(th) * cos(ph) * sin(ph) +
                      D * cos(th) * sin(ph));
   epsilon(1,0) = abs((P - S) * cos(ph) * cos(th) * sin(ph) +
                      D * sin(th) * sin(ph));
   epsilon(2,1) = abs((P - S) * sin(th) * cos(ph) * sin(ph) -
                      D * cos(th) * sin(ph));
   epsilon(2,0) = abs((P - S) * pow(sin(ph), 2) * sin(th) * cos(th) +
                      D * cos(ph));
   */
   /*
    double aP = fabs(P);
    double aSD = 0.5 * (fabs(S + D) + fabs(S - D));

   epsilon(0,0) =  (aP - aSD) * pow(sin(ph), 2) * pow(cos(th), 2) + aSD;
   epsilon(1,1) =  (aP - aSD) * pow(cos(ph), 2) + aSD;
   epsilon(2,2) =  (aP - aSD) * pow(sin(ph), 2) * pow(sin(th), 2) + aSD;
   epsilon(0,1) = -(aP - aSD) * cos(ph) * cos(th) * sin(ph);
   epsilon(0,2) =  (aP - aSD) * pow(sin(ph), 2) * sin(th) * cos(th);
   epsilon(1,2) = -(aP - aSD) * sin(th) * cos(ph) * sin(ph);
   epsilon(1,0) = epsilon(0,1);
   epsilon(2,1) = epsilon(1,2);
   epsilon(0,2) = epsilon(2,0);
    */

   epsilon *= epsilon0_;
}

PlasmaProfile::PlasmaProfile(Type type, const Vector & params,
                             G_EQDSK_Data *eqdsk)
   : type_(type), p_(params), eqdsk_(eqdsk), x_(3)
{
   MFEM_VERIFY(params.Size() == np_[type],
               "Incorrect number of parameters, " << params.Size()
               << ", for profile of type: " << type << ".");
}

double PlasmaProfile::Eval(ElementTransformation &T,
                           const IntegrationPoint &ip)
{
   if (type_ != CONSTANT)
   {
      T.Transform(ip, x_);
      x_.SetSize(3);
   }

   switch (type_)
   {
      case CONSTANT:
         return p_[0];
         break;
      case GRADIENT:
      {
         Vector x0(&p_[1], 3);
         Vector grad(&p_[4],3);

         x_ -= x0;

         return p_[0] + (grad * x_);
      }
      break;
      case TANH:
      {
         Vector x0(&p_[3], 3);
         Vector grad(&p_[6], 3);

         x_ -= x0;
         double a = 0.5 * log(3.0) * (grad * x_) / p_[2];

         if (fabs(a) < 10.0)
         {
            return p_[0] + (p_[1] - p_[0]) * tanh(a);
         }
         else
         {
            return p_[1];
         }
      }
      break;
      case ELLIPTIC_COS:
      {
         double pmin = p_[0];
         double pmax = p_[1];
         double a = p_[2];
         double b = p_[3];
         Vector x0(&p_[4], 3);

         x_ -= x0;
         double r = pow(x_[0] / a, 2) + pow(x_[1] / b, 2);
         return pmin + (pmax - pmin) * (0.5 - 0.5 * cos(M_PI * sqrt(r)));
      }
      break;
      case PARABOLIC:
      {
         double pmin = p_[0];
         double pmax = p_[1];
         double a = p_[2];
         double b = p_[3];
         Vector x0(&p_[4], 3);

         x_ -= x0;
         double r = pow(x_[0] / a, 2) + pow(x_[1] / b, 2);
         return pmax - (pmax - pmin) * r;
      }
      break;
      case PEDESTAL:
      {
         double pmin = p_[0];
         double pmax = p_[1];
         double lambda_n = p_[2]; // Damping length
         double nu = p_[3]; // Strength of decline
         Vector x0(&p_[4], 3);

         x_ -= x0;
         double rho = pow(pow(x_[0], 2) + pow(x_[1], 2), 0.5);
         return (pmax - pmin) * pow(cosh(pow((rho / lambda_n), nu)), -1.0) + pmin;
      }
      break;
      case NUABSORB:
      {
         double nu0 = p_[0];
         double decay = p_[1];
         double shift = p_[2];

         return (nu0*exp(-(x_[0]-shift)/decay));
      }
      break;
      case NUE:
      {
         double rad_res_loc = p_[0];
         double nu0 = p_[1];
         double width = 1e-5;
         double rho = pow(pow(x_[0], 2) + pow(x_[1], 2), 0.5);
         return nu0*exp(-pow(rho-rad_res_loc, 2)/width) + (4e11)*exp(-(x_[0]-0.6)/0.04);
      }
      break;
      case NUI:
      {
         double rad_res_loc = p_[0];
         double nu0 = p_[1];
         double width = p_[2];
         double rho = pow(pow(x_[0], 2) + pow(x_[1], 2), 0.5);
         return nu0*exp(-pow(rho-rad_res_loc, 2)/width);
      }
      break;
      case CMODDEN:
      {
         double rho = pow(pow(x_[0], 2) + pow(x_[1], 2), 0.5);

         double pmin1 = 1e11;
         double pmax1 = (2e20 - 3e19);
         double lam1 = 0.86253;
         double n1 = 60.0;
         double ne1 = (pmax1 - pmin1)* pow(cosh(pow((rho / lam1), n1)), -1.0) + pmin1;

         double pmin2 = 1e11;
         double pmax2 = 3e19;
         double lam2 = 0.915;
         double n2 = 46.5;
         double ne2 = (pmax2 - pmin2)* pow(cosh(pow((rho / lam2), n2)), -1.0) + pmin2;
         return ne1 + ne2;
      }
      break;
      case SPARC_RES:
      {
         double nu0 = p_[0];

         double A = 9.56300019e-02;
         double B = 1.27703065;
         double C = -1.47586242e-06;
         double D = 1.92995180;

         double E = 0.05125891;
         double F = 1.31119407;
         double G = -0.00925291;
         double H = 1.43560241;

         double r = x_[1];
         double z = x_[0];

         double val1 = B*z - C;
         double sincfunc1 = A*(sin(val1)/val1) + D;

         double val2 = F*z - G;
         double sincfunc2 = E*(sin(val2)/val2) + H;

         double res1 = nu0*exp(-pow(r-sincfunc1, 2)/0.002);
         double res2 = nu0*exp(-pow(r-sincfunc2, 2)/0.002);

         return res1+res2;
      }
      break;
      case SPARC_DEN:
      {
         /*
         double pos = pow(x_[0]-1.85, 2)/pow(0.53,2) + pow(x_[1],2)*(1 + 0.6*(x_[0]-1.85))/pow(0.85,2);

         double pmin1 = 1e19;
         double pmax1 = 3e20;
         double lam1 = 1.1;
         double n1 = 7.0;
         double ne1 = (pmax1 - pmin1)* pow(cosh(pow((pos / lam1), n1)), -1.0) + pmin1;

         return ne1;
          */
         double r = x_[1];
         double z = x_[0];

         double x_tok_data[2];
         Vector xTokVec(x_tok_data, 2);
         xTokVec[0] = r; xTokVec[1] = z;

         double psiRZ = 0.0;
         psiRZ = eqdsk_->InterpPsiRZ(xTokVec);

         double psiRZ_center = -2.74980762;
         double psiRZ_edge = -0.399621132;

         double val = fabs((psiRZ - psiRZ_center)/(psiRZ_center - psiRZ_edge));

         int bool_limits = 0;

         if (z >= -1.183 && z <= 1.19) {bool_limits = 1;}

         // double norm_sqrt_psi = 1.0;
         // if (val < 1 && bool_limits == 1) {norm_sqrt_psi = sqrt(val);}

         // FLOOR DENSITY:
         double ne = 1e12;

         // CORE DENSITY:
         double Coreden = 4.2e20;
         double LCFSden = 8.4e19;

         double nuee = 3.0;
         double nuei = 3.0;
         //ne = (pmax1 - pmin1)*pow(1 - pow(norm_sqrt_psi, nuei), nuee) + pmin1;
         if (val < 1 && bool_limits == 1)
         {
            ne = (Coreden - LCFSden)*pow(1 - pow(sqrt(val), nuei), nuee)
                 + LCFSden;
         }

         // DENSITY NEAR TOP/BOTTOM DIVERTOR:
         if (val < 1.0 && bool_limits == 0) {ne = 1e12;}

         // SOL DENSITY:
         if ( val >= 1.0)
         {
            // Scale lengths:
            double sl1 = 0.015;
            double sl2 = 0.006;
            double sl3 = 0.001;
            double Olim = LCFSden*exp(-(2.425 - 2.415)/sl1);
            double FRden = Olim*exp(-(2.445 - 2.425)/sl2);

            double temp_norm = (sqrt(val) - 1.0) / fabs(1.078669548034668 - 1.0);
            double tempr = temp_norm*(fabs(2.476101398468018-2.415)) + 2.415;

            if (tempr >= 2.415 && tempr <= 2.425 )
            {
               ne = LCFSden*exp(-(tempr-2.415)/sl1);
            }
            else if (tempr > 2.425 && tempr <= 2.445 )
            {
               ne = Olim*exp(-(tempr-2.425)/sl2);
            }
            else
            {
               ne = FRden*exp(-(tempr-2.445)/sl3);
               if (ne < 1e12) {ne = 1e12;}
            }
         }

         return ne;
      }
      break;
      case CUSTOM1:
      {
         double nu0 = p_[0];
         double decay = p_[1];

         return (nu0*exp(-x_[0]/decay));
      }
      break;
      case CUSTOM2:
      {
         double rad_res_loc = p_[0];
         double nu0 = p_[1];
         double width = 3e-5;
         double rho = pow(pow(x_[0], 2) + pow(x_[1], 2), 0.5);
         return nu0*exp(-pow(rho-rad_res_loc, 2)/width) + (1e14)*exp(-rho/0.1);
      }
      break;
      case POWER:
      {
         int comp = (int)rint(p_[0]);
         double a = p_[1];
         double b = p_[2];
         double p = p_[3];

         return a + (b - a) * pow(x_[comp], p);
      }
      break;
      case WHAM:
      {
         double a = p_[0];
         double b = p_[1];
         double c = p_[2];
         double d = p_[3];
         double p = p_[4];
         double ba = p_[5];
         double bb = p_[6];

         double bz = ba + bb * pow(x_[0], 4);
         double r_lim = sqrt(0.01 / bz);

         double rho_r = 0.5 * (1.0 + tanh(d * (r_lim - x_[1])));
         return a + ((b + (c - b) * pow(x_[0], p)) - a) * rho_r;
      }
      break;
      default:
         return 0.0;
   }
}

BFieldProfile::BFieldProfile(Type type, const Vector & params, bool unit,
                             G_EQDSK_Data *eqdsk)
   : VectorCoefficient(3), type_(type), p_(params), unit_(unit),
     eqdsk_(eqdsk), x3_(3), x_(x3_.GetData(), 3)
{
   MFEM_VERIFY(params.Size() == np_[type],
               "Incorrect number of parameters, " << params.Size()
               << ", for profile of type: " << type << ".");

   MFEM_VERIFY(type != B_EQDSK || eqdsk,
               "BFieldProfile: Profile type B_EQDSK was chosen "
               "but the G_EQDSK_Data object is NULL.");
}

void BFieldProfile::Eval(Vector &V, ElementTransformation &T,
                         const IntegrationPoint &ip)
{
   V.SetSize(3);
   if (type_ != CONSTANT)
   {
      x3_ = 0.0;
      T.Transform(ip, x_);
   }
   switch (type_)
   {
      case CONSTANT:
         if (unit_)
         {
            double bmag = pow( pow(p_[0], 2) + pow(p_[1], 2) + pow(p_[2], 2), 0.5);
            V[0] = p_[0] / bmag;
            V[1] = p_[1] / bmag;
            V[2] = p_[2] / bmag;
         }
         else
         {
            V[0] = p_[0];
            V[1] = p_[1];
            V[2] = p_[2];
         }
         break;
      case B_P:
      {
         double bp_abs = p_[0];
         double a = p_[1];
         double b = p_[2];
         Vector x0(&p_[3], 3);
         double bz = p_[6];

         x3_ -= x0;
         double r = pow(x_[0] / a, 2) + pow(x_[1] / b, 2);
         double bp = bp_abs * sin(3 * sqrt(r));
         bz *= 1.0/(0.68 + x_[0]);
         double theta = atan2(x_[1], x_[0]);

         if (unit_)
         {
            double bmag = pow( pow(bp, 2) + pow(bz, 2), 0.5);
            V[0] = -bp * sin(theta) / bmag;
            V[1] = bp * cos(theta) / bmag;
            V[2] = bz / bmag;
         }
         else
         {
            V[0] = -bp * sin(theta);
            V[1] = bp * cos(theta);
            V[2] = bz;
         }
      }
      break;
      case B_TOPDOWN:
      {
         double bp_val = p_[0];
         // double a = p_[1];
         // double b = p_[2];
         Vector x0(&p_[3], 3);

         x3_ -= x0;
         // double r = pow(x_[0] / a, 2) + pow(x_[1] / b, 2);
         double theta = atan2(x_[1], x_[0]);

         if (unit_)
         {
            double bmag = pow( pow(bp_val, 2), 0.5);
            V[0] = -bp_val * sin(theta) / bmag;
            V[1] = bp_val * cos(theta) / bmag;
            V[2] = 0 / bmag;
         }
         else
         {
            V[0] = -bp_val * sin(theta);
            V[1] = bp_val * cos(theta);
            V[2] = 0;
         }
      }
      break;
      case B_P_KOHNO:
      {
         double rmin = p_[0]; // Minor radius
         double rmaj = p_[1]; // Major radius
         double q0 = p_[2]; // Safety factor on magnetic axis
         double qa = p_[3]; // Edge safety factor
         Vector x0(&p_[4], 3); // Magnetic field axis
         double bz0 = p_[7]; // B toroidal

         x3_ -= x0;
         double rho = pow(pow(x_[0], 2) + pow(x_[1], 2), 0.5);
         double bp_coef = ((bz0 / rmaj) * pow(rmin, 2.0)) / (pow(rmin,
                                                                 2.0)*q0 + (qa - q0)*pow(rho, 2.0));

         if (unit_)
         {
            double bmag = pow( pow(bp_coef * x_[1], 2) + pow(-bp_coef * x_[0], 2) + pow(bz0,
                                                                                        2), 0.5);
            V[0] = bp_coef * x_[1] / bmag;
            V[1] = -bp_coef * x_[0] / bmag;
            V[2] = bz0 / bmag;
         }
         else
         {
            V[0] = bp_coef * x_[1];
            V[1] = -bp_coef * x_[0];
            V[2] = bz0;
         }
      }
      break;
      case B_EQDSK:
      {
         // Step 0: Extract parameters
         double u0 = p_[0];
         double v0 = p_[1];
         double z0 = p_[2];
         double theta = M_PI * p_[3] / 180.0;
         double st = sin(theta);
         double ct = cos(theta);

         // Step 1: Compute coordinates in 3D the Tokamak geometry
         double x_tok = x_[0] - u0;
         double y_tok = (x_[1] - v0) * st;
         double z_tok = (x_[1] - v0) * ct + z0;
         double r_tok = sqrt(x_tok * x_tok + y_tok * y_tok);

         // Step 2: Interpolate B field in poloidal cross section
         double x_tok_data[2];
         Vector xTokVec(x_tok_data, 2);
         xTokVec[0] = r_tok; xTokVec[1] = z_tok;

         double b_pol_data[2];
         Vector b_pol(b_pol_data, 2); b_pol = 0.0;
         double b_tor = 0.0;

         // eqdsk_->InterpNxGradPsiRZ(xTokVec, b_pol);
         eqdsk_->InterpBPolRZ(xTokVec, b_pol);
         b_tor = eqdsk_->InterpBTorRZ(xTokVec);

         // Step 3: Rotate B field from a poloidal cross section into
         //         the full Tokamak
         double b_tok_data[3];
         Vector b_tok(b_tok_data, 3);
         b_tok[0] = (b_pol[0] * x_tok - b_tor * y_tok) / r_tok;
         b_tok[1] = (b_pol[0] * y_tok + b_tor * x_tok) / r_tok;
         b_tok[2] = b_pol[1];

         // Step 4: Rotate B field into the slanted computational plane
         V[0] = b_tok[0];
         V[1] = b_tok[2] * ct + b_tok[1] * st;
         V[2] = b_tok[2] * st - b_tok[1] * ct;

         if (unit_)
         {
            double vmag = sqrt(V * V);
            V /= vmag;
         }
      }
      break;
      case B_SPARC:
      {
         //|B| = \sqrt(Fpol^2+d\Psi/dZ^2+d\Psi/dR^2)/R
         //where Fpol== R*Bphi , BR = - 1/R d\Psi/dZ, BZ = 1/R d\Psi/dR

         double b_pol_data[2];
         Vector b_pol(b_pol_data, 2); b_pol = 0.0;
         double b_tor = 0.0;

         double x_tok_data[2];
         Vector xTokVec(x_tok_data, 2);
         xTokVec[0] = x_[1]; xTokVec[1] = x_[0];

         eqdsk_->InterpBPolRZ(xTokVec, b_pol);
         b_tor = eqdsk_->InterpBTorRZ(xTokVec);

         V[0] = b_pol[1];
         V[1] = b_pol[0];
         V[2] = b_tor;

         if (unit_)
         {
            double vmag = sqrt(V * V);
            V /= vmag;
         }
      }
      break;
      case B_WHAM:
      {
         double a = p_[0];
         double b = p_[1];
         V[0] = a + b * pow(x_[0], 4);
         V[1] = -2.0 * b * x_[1] * pow(x_[0], 3);
         V[2] = 0.0;
      }
      break;
      default:
         if (unit_)
         {
            V[0] = 0.0;
            V[1] = 0.0;
            V[2] = 1.0;
         }
         else
         {
            V[0] = 0.0;
            V[1] = 0.0;
            V[2] = 5.4;
         }
   }
   /*
   {
      case CONSTANT:
         if (unit_)
         {
            double bmag = pow( pow(p_[0], 2) + pow(p_[1], 2) + pow(p_[2], 2), 0.5);
            V[0] = p_[0] / bmag;
            V[1] = p_[1] / bmag;
            V[2] = p_[2] / bmag;
         }
         else
         {
            V[0] = p_[0];
            V[1] = p_[1];
            V[2] = p_[2];
         }
         break;
      case B_P:
      {
         double bp_abs = p_[0];
         double a = p_[1];
         double b = p_[2];
         Vector x0(&p_[3], 3);
         double bz = p_[6];

         x_ -= x0;
         double r = pow(x_[0] / a, 2) + pow(x_[1] / b, 2);
         double bp = bp_abs * sin(3 * sqrt(r));
         bz *= 1.0/(0.68 + x_[0]);
         double theta = atan2(x_[1], x_[0]);

         if (unit_)
         {
            double bmag = pow( pow(bp, 2) + pow(bz, 2), 0.5);
            V[0] = -bp * sin(theta) / bmag;
            V[1] = bp * cos(theta) / bmag;
            V[2] = bz / bmag;
         }
         else
         {
            V[0] = -bp * sin(theta);
            V[1] = bp * cos(theta);
            V[2] = bz;
         }
      }
      break;
      default:
         if (unit_)
         {
            V[0] = 0.0;
            V[1] = 0.0;
            V[2] = 1.0;
         }
         else
         {
            V[0] = 0.0;
            V[1] = 0.0;
            V[2] = 5.4;
         }
   }
   */
}

ComplexPhaseCoefficient::ComplexPhaseCoefficient(
   VectorCoefficient *kRe,
   VectorCoefficient *kIm,
   Coefficient *vRe,
   Coefficient *vIm,
   bool realPart, bool inv_k)
   : kReCoef_(kRe), kImCoef_(kIm), vReCoef_(vRe), vImCoef_(vIm),
     realPart_(realPart), inv_k_(inv_k),
     kdim_(kRe ? kRe->GetVDim() :(kIm ? kIm->GetVDim() : 0)),
     xk_(kdim_), xs_(xk_.GetData(), kdim_),
     kRe_(kdim_), kIm_(kdim_)
{
   if ( kRe && kIm )
   {
      MFEM_ASSERT(kRe->GetVDim() == kIm->GetVDim(),
                  "Wave vector dimension mismatch in "
                  "ComplexPhaseCoefficient");
   }

   xk_  = 0.0;
   kRe_ = 0.0;
   kIm_ = 0.0;
}

double ComplexPhaseCoefficient::Eval(ElementTransformation &T,
                                     const IntegrationPoint &ip)
{
   T.Transform(ip, xs_);

   double sinkx = 0.0;
   double coskx = 1.0;

   if (kReCoef_)
   {
      kReCoef_->Eval(kRe_, T, ip);
      double phase = kRe_ * xk_;

      sinkx = sin(phase) * (inv_k_ ? -1.0 : 1.0);
      coskx = cos(phase);
   }
   if (kImCoef_)
   {
      kImCoef_->Eval(kIm_, T, ip);
      double phase = kIm_ * xk_;
      double expkx = exp((inv_k_ ? -1.0 : 1.0) * phase);

      sinkx *= expkx;
      coskx *= expkx;
   }

   double vRe = 0.0;
   double vIm = 0.0;

   if (vReCoef_)
   {
      vRe = vReCoef_->Eval(T, ip);
   }
   if (vImCoef_)
   {
      vIm = vImCoef_->Eval(T, ip);
   }

   double v = 0.0;
   if (realPart_)
   {
      v = coskx * vRe - sinkx * vIm;
   }
   else
   {
      v = sinkx * vRe + coskx * vIm;
   }
   return v;
}

ComplexPhaseVectorCoefficient::ComplexPhaseVectorCoefficient(
   VectorCoefficient *kRe,
   VectorCoefficient *kIm,
   VectorCoefficient *vRe,
   VectorCoefficient *vIm,
   bool realPart, bool inv_k)
   : VectorCoefficient(vRe ? vRe->GetVDim() :(vIm ? vIm->GetVDim() : 0)),
     kReCoef_(kRe), kImCoef_(kIm), vReCoef_(vRe), vImCoef_(vIm),
     realPart_(realPart), inv_k_(inv_k),
     kdim_(kRe ? kRe->GetVDim() :(kIm ? kIm->GetVDim() : 0)),
     xk_(kdim_), xs_(xk_.GetData(), kdim_),
     kRe_(kdim_), kIm_(kdim_),
     vRe_(vdim), vIm_(vdim)
{
   if ( vRe && vIm )
   {
      MFEM_ASSERT(vRe->GetVDim() == vIm->GetVDim(),
                  "Vector dimension mismatch in "
                  "ComplexPhaseVectorCoefficient");
   }
   if ( kRe && kIm )
   {
      MFEM_ASSERT(kRe->GetVDim() == kIm->GetVDim(),
                  "Wave vector dimension mismatch in "
                  "ComplexPhaseVectorCoefficient");
   }

   xk_  = 0.0;
   kRe_ = 0.0;
   kIm_ = 0.0;
   vRe_ = 0.0;
   vIm_ = 0.0;
}

void ComplexPhaseVectorCoefficient::Eval(Vector &V, ElementTransformation &T,
                                         const IntegrationPoint &ip)
{
   T.Transform(ip, xs_);

   double sinkx = 0.0;
   double coskx = 1.0;

   if (kReCoef_)
   {
      kReCoef_->Eval(kRe_, T, ip);
      double phase = kRe_ * xk_;

      sinkx = sin(phase) * (inv_k_ ? -1.0 : 1.0);
      coskx = cos(phase);
   }
   if (kImCoef_)
   {
      kImCoef_->Eval(kIm_, T, ip);
      double phase = kIm_ * xk_;
      double expkx = exp((inv_k_ ? -1.0 : 1.0) * phase);

      sinkx *= expkx;
      coskx *= expkx;
   }

   if (vReCoef_)
   {
      vReCoef_->Eval(vRe_, T, ip);
   }
   if (vImCoef_)
   {
      vImCoef_->Eval(vIm_, T, ip);
   }

   V.SetSize(vdim);
   if (realPart_)
   {
      add(coskx, vRe_, -sinkx, vIm_, V);
   }
   else
   {
      add(sinkx, vRe_, coskx, vIm_, V);
   }
}

} // namespace plasma

} // namespace mfem
