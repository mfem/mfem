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

#include "cold_plasma_dielectric_coefs.hpp"

using namespace std;
namespace mfem
{
using namespace common;

namespace plasma
{

complex<double> R_cold_plasma(double omega,
                              double Bmag,
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
                 nu_art(temp[0]);
   complex<double> collision_correction(1.0, nuei/omega);

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m;
      if ( i == 0) {m_eff = m*collision_correction;}
      complex<double> w_c = omega_c(Bmag, q, m_eff);
      complex<double> w_p = omega_p(n, q, m_eff);
      val -= w_p * w_p / (omega * (omega + w_c));
   }
   return val;
}

complex<double> L_cold_plasma(double omega,
                              double Bmag,
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
                 nu_art(temp[0]);
   complex<double> collision_correction(1.0, nuei/omega);

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m;
      if ( i == 0) {m_eff = m*collision_correction;}
      complex<double> w_c = omega_c(Bmag, q, m_eff);
      complex<double> w_p = omega_p(n, q, m_eff);
      val -= w_p * w_p / (omega * (omega - w_c));
   }
   return val;
}

complex<double> S_cold_plasma(double omega,
                              double Bmag,
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
                 nu_art(temp[0]);
   complex<double> collision_correction(1.0, nuei/omega);

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m;
      if (i == 0) { m_eff = m*collision_correction; }
      complex<double> w_c = omega_c(Bmag, q, m_eff);
      complex<double> w_p = omega_p(n, q, m_eff);
      val -= w_p * w_p / (omega * omega - w_c * w_c);

   }
   return val;
}

complex<double> D_cold_plasma(double omega,
                              double Bmag,
                              const Vector & number,
                              const Vector & charge,
                              const Vector & mass,
                              const Vector & temp,
                              int nuprof)
{
   complex<double> val(0.0, 0.0);
   double n = number[0];
   double q = charge[0];
   double m = mass[0];
   double Te = temp[0] * q_;
   double coul_log = CoulombLog(n, Te);
   double nuei = (nuprof == 0) ?
                 nu_ei(q, coul_log, m, Te, n) :
                 nu_art(temp[0]);
   complex<double> collision_correction(1.0, nuei/omega);

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m;
      if (i == 0) { m_eff = m*collision_correction; }
      complex<double> w_c = omega_c(Bmag, q, m_eff);
      complex<double> w_p = omega_p(n, q, m_eff);
      val += w_p * w_p * w_c / (omega * (omega * omega - w_c * w_c));
   }
   return val;
}

complex<double> P_cold_plasma(double omega,
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
                 nu_art(temp[0]);
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
                       const_cast<Vector&>(density_.GetBlock(1)));
   return density_gf_.GetValue(T, ip);
}

double SheathBase::EvalElectronTemp(ElementTransformation &T,
                                    const IntegrationPoint &ip)
{
   temperature_gf_.MakeRef(const_cast<ParFiniteElementSpace*>(&H1FESpace_),
                           const_cast<Vector&>(temp_.GetBlock(0)));
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
   // double temp_val = EvalElectronTemp(T, ip);

   // double Te = temp_val * q_; // Electron temperature, Units: J

   double wpi = omega_p(density_val, charges_[1], masses_[1]);

   // double vnorm = Te / (charges_[1] * q_);
   double w_norm = omega_ / wpi;

   complex<double> phi = EvalSheathPotential(T, ip);
   double phi_mag = sqrt(pow(phi.real(), 2) + pow(phi.imag(), 2));
   //double volt_norm = (phi_mag)/temp_val ; // V zero-to-peak
   double volt_norm = (2*phi_mag)/15.0 ; // V peak-to-peak

   double phiRec = phi0avg(w_norm, volt_norm);

   if (realPart_)
   {
      return phiRec; // * temp_val;
   }
   else
   {
      return phiRec; // * temp_val;
   }
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

double SheathImpedance::Eval(ElementTransformation &T,
                             const IntegrationPoint &ip)
{
   // Collect density, temperature, magnetic field, and potential field values
   Vector B(3);
   B_.GetVectorValue(T, ip, B);
   double Bmag = B.Norml2();                         // Units: T

   complex<double> phi = EvalSheathPotential(T, ip); // Units: V

   double density_val = EvalIonDensity(T, ip);       // Units: # / m^3
   // double temp_val = EvalElectronTemp(T, ip);        // Units: eV

   double wci = omega_c(Bmag, charges_[1], masses_[1]);        // Units: s^{-1}
   double wpi = omega_p(density_val, charges_[1], masses_[1]); // Units: s^{-1}

   double w_norm = omega_ / wpi; // Unitless
   double wci_norm = wci / wpi;  // Unitless
   double phi_mag = sqrt(pow(phi.real(), 2) + pow(phi.imag(), 2));
   double volt_norm = (phi_mag)/15.0 ; // Unitless: V zero-to-peak
   //double volt_norm = (2*phi_mag)/15.0 ; // Unitless: V peak-to-peak
   if ( volt_norm > 20) {cout << "Warning: V_RF > Z Parameterization Limit!" << endl;}

   double debye_length = debye(15.0, density_val); // Units: m
   Vector nor(3); nor = 0.0;
   nor.SetSize(T.GetSpaceDim());
   CalcOrtho(T.Jacobian(), nor);
   nor.SetSize(3);
   double normag = nor.Norml2();
   double bn = (B * nor)/(normag*Bmag); // Unitless

   // Jim's old parametrization (Kohno et al 2017):
   //complex<double> zsheath_norm = 1.0 / ftotcmplxANY(w_norm, volt_norm);

   // Jim's newest parametrization (Myra et al 2017):
   complex<double> zsheath_norm = 1.0 / ytot(w_norm, wci_norm, bn, volt_norm,
                                             masses_[0], masses_[1]);

   // Fixed sheath impedance:
   //complex<double> zsheath_norm(57.4699936705, 21.39395629068357);

   /*
   cout << "Check 1:" << phi0avg(0.4, 6.) - 6.43176481712605 << endl;
   cout << "Check 2:" << niw(.2, .3, 13,masses_[0], masses_[1])- 0.07646452845544677 << endl;
   cout << "Check 3:" << niww(0.2, 0.3, 0.4, 13,masses_[0], masses_[1]) - 0.14077643642166277 << endl;
   cout << "Check 4:" << yd(0.2, 0.3, 0.4, 13,masses_[0], masses_[1]).imag()+0.025738204728120898 << endl;
   cout << "Check 5: " << ye(0.4, 3.6) - 0.1588274616204441 << endl;
   cout << "Check 6:" << yi(0.2, 0.3, 0.4, 13,masses_[0], masses_[1]).real() - 0.006543897148693344 << yi(0.2, 0.3, 0.4, 13,masses_[0], masses_[1]).imag()+0.013727440802110503 << endl;
   cout << "Check 7:" << ytot(0.2, 0.3, 0.4, 13,masses_[0], masses_[1]).real()-0.05185050837032144 << ytot(0.2, 0.3, 0.4, 13,masses_[0], masses_[1]).imag()+0.0394656455302314 << endl;
    */

   if (realPart_)
   {
      return (zsheath_norm.real()*debye_length)/(epsilon0_*wpi); // Units: Ohm m^2

   }
   else
   {
      return (zsheath_norm.imag()*debye_length)/(epsilon0_*wpi); // Units: Ohm m^2
   }
}

StixCoefBase::StixCoefBase(const ParGridFunction & B,
                           const BlockVector & density,
                           const BlockVector & temp,
                           const ParFiniteElementSpace & L2FESpace,
                           const ParFiniteElementSpace & H1FESpace,
                           double omega,
                           const Vector & charges,
                           const Vector & masses,
                           int nuprof,
                           bool realPart)
   : B_(B),
     density_(density),
     temp_(temp),
     L2FESpace_(L2FESpace),
     H1FESpace_(H1FESpace),
     omega_(omega),
     realPart_(realPart),
     nuprof_(nuprof),
     BVec_(3),
     charges_(charges),
     masses_(masses)
{
   density_vals_.SetSize(charges_.Size());
   temp_vals_.SetSize(charges_.Size());
}

StixCoefBase::StixCoefBase(StixCoefBase & s)
   : B_(s.GetBField()),
     density_(s.GetDensityFields()),
     temp_(s.GetTemperatureFields()),
     L2FESpace_(s.GetDensityFESpace()),
     H1FESpace_(s.GetTemperatureFESpace()),
     omega_(s.GetOmega()),
     realPart_(s.GetRealPartFlag()),
     nuprof_(s.GetNu()),
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
   for (int i=0; i<density_vals_.Size(); i++)
   {
      density_gf_.MakeRef(const_cast<ParFiniteElementSpace*>(&L2FESpace_),
                          const_cast<Vector&>(density_.GetBlock(i)));
      density_vals_[i] = density_gf_.GetValue(T.ElementNo, ip);
   }
}

void StixCoefBase::fillTemperatureVals(ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   for (int i=0; i<temp_vals_.Size(); i++)
   {
      temperature_gf_.MakeRef(const_cast<ParFiniteElementSpace*>(&H1FESpace_),
                              const_cast<Vector&>(temp_.GetBlock(i)));
      temp_vals_[i] = temperature_gf_.GetValue(T.ElementNo, ip);
   }
}

StixSCoef::StixSCoef(const ParGridFunction & B,
                     const BlockVector & density,
                     const BlockVector & temp,
                     const ParFiniteElementSpace & L2FESpace,
                     const ParFiniteElementSpace & H1FESpace,
                     double omega,
                     const Vector & charges,
                     const Vector & masses,
                     int nuprof,
                     bool realPart)
   : StixCoefBase(B, density, temp, L2FESpace, H1FESpace, omega,
                  charges, masses, nuprof, realPart)
{}

double StixSCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density, temperature, and magnetic field values
   double Bmag = this->getBMagnitude(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex<double> S = S_cold_plasma(omega_, Bmag, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);

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
                     const BlockVector & density,
                     const BlockVector & temp,
                     const ParFiniteElementSpace & L2FESpace,
                     const ParFiniteElementSpace & H1FESpace,
                     double omega,
                     const Vector & charges,
                     const Vector & masses,
                     int nuprof,
                     bool realPart)
   : StixCoefBase(B, density, temp, L2FESpace, H1FESpace, omega,
                  charges, masses, nuprof, realPart)
{}

double StixDCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density, temperature, and magnetic field values
   double Bmag = this->getBMagnitude(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex<double> D = D_cold_plasma(omega_, Bmag, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);

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
                     const BlockVector & density,
                     const BlockVector & temp,
                     const ParFiniteElementSpace & L2FESpace,
                     const ParFiniteElementSpace & H1FESpace,
                     double omega,
                     const Vector & charges,
                     const Vector & masses,
                     int nuprof,
                     bool realPart)
   : StixCoefBase(B, density, temp, L2FESpace, H1FESpace, omega,
                  charges, masses, nuprof, realPart)
{}

double StixPCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density and temperature field values
   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex<double> P = P_cold_plasma(omega_, density_vals_,
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

StixTensorBase::StixTensorBase(const ParGridFunction & B,
                               const BlockVector & density,
                               const BlockVector & temp,
                               const ParFiniteElementSpace & L2FESpace,
                               const ParFiniteElementSpace & H1FESpace,
                               double omega,
                               const Vector & charges,
                               const Vector & masses,
                               int nuprof,
                               bool realPart)
   : StixCoefBase(B, density, temp, L2FESpace, H1FESpace,
                  omega, charges, masses, nuprof, realPart)
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
                                   const BlockVector & density,
                                   const BlockVector & temp,
                                   const ParFiniteElementSpace & L2FESpace,
                                   const ParFiniteElementSpace & H1FESpace,
                                   double omega,
                                   const Vector & charges,
                                   const Vector & masses,
                                   int nuprof,
                                   bool realPart)
   : MatrixCoefficient(3),
     StixTensorBase(B, density, temp, L2FESpace, H1FESpace, omega,
                    charges, masses, nuprof, realPart)
{}

void DielectricTensor::Eval(DenseMatrix &epsilon, ElementTransformation &T,
                            const IntegrationPoint &ip)
{
   // Initialize dielectric tensor to appropriate size
   epsilon.SetSize(3); epsilon = 0.0;

   // Collect density, temperature, and magnetic field values
   double Bmag = this->getBMagnitude(T, ip);
   BVec_ /= Bmag;

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate the Stix Coefficients
   complex<double> S = S_cold_plasma(omega_, Bmag, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);
   complex<double> P = P_cold_plasma(omega_, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);
   complex<double> D = D_cold_plasma(omega_, Bmag, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);

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
   const BlockVector & density,
   const BlockVector & temp,
   const ParFiniteElementSpace & L2FESpace,
   const ParFiniteElementSpace & H1FESpace,
   double omega,
   const Vector & charges,
   const Vector & masses,
   int nuprof,
   bool realPart)
   : MatrixCoefficient(3),
     StixTensorBase(B, density, temp, L2FESpace, H1FESpace, omega,
                    charges, masses, nuprof, realPart)
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

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate the Stix Coefficients
   complex<double> S = S_cold_plasma(omega_, Bmag, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);
   complex<double> P = P_cold_plasma(omega_, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);
   complex<double> D = D_cold_plasma(omega_, Bmag, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);

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
   const BlockVector & density,
   const BlockVector & temp,
   const ParFiniteElementSpace & L2FESpace,
   const ParFiniteElementSpace & H1FESpace,
   double omega,
   const Vector & charges,
   const Vector & masses,
   int nuprof)
   : MatrixCoefficient(3),
     StixCoefBase(B, density, temp, L2FESpace, H1FESpace, omega,
                  charges, masses, nuprof, true)
{}

void SPDDielectricTensor::Eval(DenseMatrix &epsilon, ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   // Initialize dielectric tensor to appropriate size
   epsilon.SetSize(3);

   // Collect density, temperature, and magnetic field values
   double Bmag = this->getBMagnitude(T, ip);
   BVec_ /= Bmag;

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
   complex<double> S = S_cold_plasma(omega_, Bmag, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);
   complex<double> P = P_cold_plasma(omega_, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);
   complex<double> D = D_cold_plasma(omega_, Bmag, density_vals_,
                                     charges_, masses_, temp_vals_, nuprof_);

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

PlasmaProfile::PlasmaProfile(Type type, const Vector & params)
   : type_(type), p_(params), x_(3)
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
         return pmin + (pmax - pmin) * (0.5 + 0.5 * cos(M_PI * sqrt(r)));
      }
      break;
      default:
         return 0.0;
   }
}

} // namespace plasma

} // namespace mfem
