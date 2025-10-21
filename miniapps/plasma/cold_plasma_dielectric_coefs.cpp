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

#include "cold_plasma_dielectric_coefs.hpp"

using namespace std;
namespace mfem
{
using namespace common;

namespace plasma
{

complex_t R_cold_plasma(real_t omega,
                        real_t Bmag,
                        real_t nue,
                        real_t nui,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof)
{
   complex_t val(1.0, 0.0);
   real_t n = number[0];
   real_t q = charge[0];
   real_t m = mass[0];
   real_t Te = temp[0] * q_;
   real_t coul_log = CoulombLog(n, Te);
   real_t nuei = (nuprof == 0) ?
                 nu_ei(q, coul_log, m, Te, n) :
                 nue;
   complex_t collision_correction(1.0, nuei/omega);
   real_t nui_res = 0.0;

   for (int i=0; i<number.Size(); i++)
   {
      n = number[i];
      q = charge[i];
      m = mass[i];
      complex_t m_eff = m;
      if (i == 0) { m_eff = m * collision_correction; }
      if (i == 1) { nui_res = nui; }
      complex_t w_c =
         omega_c(Bmag, q, m_eff) - complex_t(0.0, nui_res);
      complex_t w_p = omega_p(n, q, m_eff);
      val -= w_p * w_p / (omega * (omega + w_c));
   }
   return val;
}

complex_t L_cold_plasma(real_t omega,
                        real_t Bmag,
                        real_t nue,
                        real_t nui,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof,
                        real_t res_lim)
{
   complex_t val(1.0, 0.0);
   real_t n = number[0];
   real_t q = charge[0];
   real_t m = mass[0];
   real_t Te = temp[0] * q_;
   real_t coul_log = CoulombLog(n, Te);
   real_t nuei = (nuprof == 0) ?
                 nu_ei(q, coul_log, m, Te, n) :
                 nue;
   complex_t collision_correction(1.0, nuei/omega);
   real_t nui_res = 0.0;

   for (int i=0; i<number.Size(); i++)
   {
      n = number[i];
      q = charge[i];
      m = mass[i];
      complex_t m_eff = m;
      if (i == 0) { m_eff = m * collision_correction; }
      if (i == 1) { nui_res = nui; }
      complex_t w_c =
         omega_c(Bmag, q, m_eff) - complex_t(0.0, nui_res);
      if (res_lim != 0.0)
      {
         real_t expw_c = exp(-pow(1.0 - w_c.real() / omega, 2));
         w_c -= complex_t(0.0, res_lim * omega * expw_c);
      }
      complex_t w_p = omega_p(n, q, m_eff);
      val -= w_p * w_p / (omega * (omega - w_c));
   }
   return val;
}

complex_t S_cold_plasma(real_t omega,
                        real_t Bmag,
                        real_t nue,
                        real_t nui,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof,
                        real_t res_lim)
{
   complex_t val(1.0, 0.0);
   real_t n = number[0];
   real_t q = charge[0];
   real_t m = mass[0];
   real_t Te = temp[0] * q_;
   real_t coul_log = CoulombLog(n, Te);
   real_t nuei = (nuprof == 0) ?
                 nu_ei(q, coul_log, m, Te, n) :
                 nue;
   complex_t collision_correction(1.0, nuei/omega);
   real_t nui_res = 0.0;

   for (int i=0; i<number.Size(); i++)
   {
      n = number[i];
      q = charge[i];
      m = mass[i];
      complex_t m_eff = m;
      if (i == 0) { m_eff = m * collision_correction; }
      if (i == 1) { nui_res = nui; }
      complex_t w_c =
         omega_c(Bmag, q, m_eff) - complex_t(0.0, nui_res);
      complex_t w_c_c = w_c;
      complex_t num(1.0, 0.0);
      if (res_lim != 0.0)
      {
         real_t expw_c = exp(-pow(1.0 - w_c.real() / omega, 2));
         w_c_c -= complex_t(0.0, res_lim * omega * expw_c);
         num += complex_t(0.0, 0.5 * res_lim * expw_c);
      }
      complex_t w_p = omega_p(n, q, m_eff);
      val -= w_p * w_p * num / ((omega + w_c) * (omega - w_c_c));

   }
   return val;
}

complex_t D_cold_plasma(real_t omega,
                        real_t Bmag,
                        real_t nue,
                        real_t nui,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof,
                        real_t res_lim)
{
   complex_t val(0.0, 0.0);
   real_t n = number[0];
   real_t q = charge[0];
   real_t m = mass[0];
   real_t Te = temp[0] * q_;
   real_t coul_log = CoulombLog(n, Te);
   real_t nuei = (nuprof == 0) ?
                 nu_ei(q, coul_log, m, Te, n) :
                 nue;
   complex_t collision_correction(1.0, nuei/omega);
   real_t nui_res = 0.0;

   for (int i=0; i<number.Size(); i++)
   {
      n = number[i];
      q = charge[i];
      m = mass[i];
      complex_t m_eff = m;
      if (i == 0) { m_eff = m*collision_correction; }
      if (i == 1) { nui_res = nui; }
      complex_t w_c =
         omega_c(Bmag, q, m_eff) - complex_t(0.0, nui_res);
      complex_t w_c_c = w_c;
      complex_t num = w_c;
      if (res_lim != 0.0)
      {
         real_t expw_c = exp(-pow(1.0 - w_c.real() / omega, 2));
         w_c_c -= complex_t(0.0, res_lim * omega * expw_c);
         num -= complex_t(0.0, 0.5 * omega * res_lim * expw_c);
      }
      complex_t w_p = omega_p(n, q, m_eff);
      val += w_p * w_p * num / (omega * (omega + w_c) * (omega - w_c_c));
   }
   return val;
}

complex_t P_cold_plasma(real_t omega,
                        real_t nue,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof)
{
   complex_t val(1.0, 0.0);
   real_t n = number[0];
   real_t q = charge[0];
   real_t m = mass[0];
   real_t Te = temp[0] * q_;
   real_t coul_log = CoulombLog(n, Te);
   real_t nuei = (nuprof == 0) ?
                 nu_ei(q, coul_log, m, Te, n) :
                 nue;
   complex_t collision_correction(1.0, nuei/omega);

   for (int i=0; i<number.Size(); i++)
   {
      n = number[i];
      q = charge[i];
      m = mass[i];
      complex_t m_eff = m;
      if (i == 0) { m_eff = m*collision_correction; }
      complex_t w_p = omega_p(n, q, m_eff);
      val -= w_p * w_p / (omega * omega);
   }
   if (fabs(val.real()) < 1e-4)
   {
      cout << "P near zero: " << val << endl;
   }
   return val;
}

StixCoefBase::StixCoefBase(StixParams &stix_params, ReImPart re_im_part)
   : BCoef_(stix_params.BCoef),
     nue_(stix_params.nueCoef),
     nui_(stix_params.nuiCoef),
     spec_density_(stix_params.specDensityCoef),
     spec_temperature_(stix_params.specTemperatureCoef),
     omega_(stix_params.omega),
     re_im_part_(re_im_part),
     nuprof_(stix_params.nuProfile),
     res_lim_(stix_params.resLimit),
     BVec_(3),
     charges_(stix_params.charges),
     masses_(stix_params.masses)
{
   MFEM_VERIFY(BCoef_.GetVDim() == 3,
               "B Field must be a three component vector field");
   MFEM_VERIFY(spec_density_.GetVDim() == charges_.Size(),
               "Mismatch in numbers of charges and densities");
   MFEM_VERIFY(spec_temperature_.GetVDim() == charges_.Size(),
               "Mismatch in numbers of charges and temperatures");
   MFEM_VERIFY(masses_.Size() == charges_.Size(),
               "Mismatch in numbers of charges and masses");

   density_vals_.SetSize(spec_density_.GetVDim());
   temperature_vals_.SetSize(spec_temperature_.GetVDim());
}

StixCoefBase::StixCoefBase(StixCoefBase & s)
   : BCoef_(s.GetBCoef()),
     nue_(s.GetNue()),
     nui_(s.GetNui()),
     spec_density_(s.GetDensityCoefs()),
     spec_temperature_(s.GetTemperatureCoefs()),
     omega_(s.GetOmega()),
     re_im_part_(s.GetRealPartFlag()),
     nuprof_(s.GetNuProf()),
     res_lim_(s.GetResonanceLimitorFactor()),
     BVec_(3),
     charges_(s.GetCharges()),
     masses_(s.GetMasses())
{
   density_vals_.SetSize(charges_.Size());
   temperature_vals_.SetSize(charges_.Size());
}

real_t StixCoefBase::getBMagnitude(ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   BCoef_.Eval(BVec_, T, ip);

   return BVec_.Norml2();
}

void StixCoefBase::fillDensityVals(ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   spec_density_.Eval(density_vals_, T, ip);
}

void StixCoefBase::fillTemperatureVals(ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   spec_temperature_.Eval(temperature_vals_, T, ip);
}

StixLCoef::StixLCoef(StixParams &stix_params, ReImPart re_im_part)
   : StixCoefBase(stix_params, re_im_part)
{}

real_t StixLCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density, temperature, and magnetic field values
   real_t Bmag = this->getBMagnitude(T, ip);
   nue_vals_ = nue_.Eval(T, ip);
   nui_vals_ = nui_.Eval(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex_t L = L_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);

   // Return the selected component
   if (re_im_part_ == REAL_PART)
   {
      return L.real();
   }
   else
   {
      return L.imag();
   }
}

StixRCoef::StixRCoef(StixParams &stix_params, ReImPart re_im_part)
   : StixCoefBase(stix_params, re_im_part)
{}

real_t StixRCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density, temperature, and magnetic field values
   real_t Bmag = this->getBMagnitude(T, ip);
   nue_vals_ = nue_.Eval(T, ip);
   nui_vals_ = nui_.Eval(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex_t R = R_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_);

   // Return the selected component
   if (re_im_part_ == REAL_PART)
   {
      return R.real();
   }
   else
   {
      return R.imag();
   }
}

StixSCoef::StixSCoef(StixParams &stix_params, ReImPart re_im_part)
   : StixCoefBase(stix_params, re_im_part)
{}

real_t StixSCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density, temperature, and magnetic field values
   real_t Bmag = this->getBMagnitude(T, ip);
   nue_vals_ = nue_.Eval(T, ip);
   nui_vals_ = nui_.Eval(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex_t S = S_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);

   // Return the selected component
   if (re_im_part_ == REAL_PART)
   {
      return S.real();
   }
   else
   {
      return S.imag();
   }
}

StixDCoef::StixDCoef(StixParams &stix_params, ReImPart re_im_part)
   : StixCoefBase(stix_params, re_im_part)
{}

real_t StixDCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density, temperature, and magnetic field values
   real_t Bmag = this->getBMagnitude(T, ip);
   nue_vals_ = nue_.Eval(T, ip);
   nui_vals_ = nui_.Eval(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex_t D = D_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);

   // Return the selected component
   if (re_im_part_ == REAL_PART)
   {
      return D.real();
   }
   else
   {
      return D.imag();
   }
}

StixPCoef::StixPCoef(StixParams &stix_params, ReImPart re_im_part)
   : StixCoefBase(stix_params, re_im_part)
{}

real_t StixPCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   // Collect density and temperature field values
   nue_vals_ = nue_.Eval(T, ip);
   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex_t P = P_cold_plasma(omega_, nue_vals_, density_vals_,
                               charges_, masses_, temperature_vals_,
                               nuprof_);

   // Return the selected component
   if (re_im_part_ == REAL_PART)
   {
      return P.real();
   }
   else
   {
      return P.imag();
   }
}

real_t StixInvSPCoef::Eval(ElementTransformation &T,
                           const IntegrationPoint &ip)
{
   // Collect density, temperature, and magnetic field values
   real_t Bmag = this->getBMagnitude(T, ip);
   nue_vals_ = nue_.Eval(T, ip);
   nui_vals_ = nui_.Eval(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex_t S = S_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);
   complex_t P = P_cold_plasma(omega_, nue_vals_, density_vals_,
                               charges_, masses_, temperature_vals_,
                               nuprof_);

   complex_t SP = S * P;
   real_t absSP = std::abs(SP);
   real_t argSP = std::arg(SP);
   complex_t InvSP = (absSP > 1e-4) ? 1.0 / (S * P) :
                     std::polar(1e4, -argSP);

   // Return the selected component
   if (re_im_part_ == REAL_PART)
   {
      return InvSP.real();
   }
   else
   {
      return InvSP.imag();
   }
}

StixWaveLengthCoef::StixWaveLengthCoef(char type,
                                       StixParams &stix_params,
                                       ReImPart re_im_part)
   : StixCoefBase(stix_params, re_im_part),
     type_(type)
{}

real_t StixWaveLengthCoef::Eval(ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   // Collect density, temperature and magnetic field field values
   real_t Bmag = this->getBMagnitude(T, ip);
   nue_vals_ = nue_.Eval(T, ip);
   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex_t S = S_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);
   complex_t D = D_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);
   complex_t P = P_cold_plasma(omega_, nue_vals_, density_vals_,
                               charges_, masses_, temperature_vals_,
                               nuprof_);

   complex_t kappa;

   switch (type_)
   {
      case 'L':
         kappa = omega_ * sqrt(S - D) / c0_;
         break;
      case 'R':
         kappa = omega_ * sqrt(S + D) / c0_;
         break;
      case 'O':
         kappa = omega_ * sqrt(P) / c0_;
         break;
      case 'X':
         kappa = omega_ * sqrt(S - D * D / S) / c0_;
         break;
   }

   // Return the selected component
   if (re_im_part_ == REAL_PART)
   {
      // Compute the wave length of the oscillatory factor
      real_t lambda = fabs(2.0 * M_PI / kappa.real());
      return lambda;
   }
   else
   {
      // Compute the skin depth of the decaying factor
      real_t delta = fabs(1.0 / kappa.imag());
      return delta;
   }
}

StixAdmittanceCoef::StixAdmittanceCoef(char type,
                                       StixParams &stix_params,
                                       ReImPart re_im_part)
   : StixCoefBase(stix_params, re_im_part),
     type_(type)
{}

real_t StixAdmittanceCoef::Eval(ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   // Collect density, temperature and magnetic field field values
   real_t Bmag = this->getBMagnitude(T, ip);
   nue_vals_ = nue_.Eval(T, ip);
   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate Stix Coefficient
   complex_t S = S_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);
   complex_t D = D_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);
   complex_t P = P_cold_plasma(omega_, nue_vals_, density_vals_,
                               charges_, masses_, temperature_vals_,
                               nuprof_);

   complex_t etaInv;

   switch (type_)
   {
      case 'L':
         etaInv = sqrt(S - D) / Z0_;
         break;
      case 'R':
         etaInv = sqrt(S + D) / Z0_;
         break;
      case 'O':
         etaInv = sqrt(P) / Z0_;
         break;
      case 'X':
         etaInv = sqrt(S - D * D / S) / Z0_;
         break;
   }

   // Return the selected component
   if (re_im_part_ == REAL_PART)
   {
      return etaInv.real();
   }
   else
   {
      return etaInv.imag();
   }
}

StixTensorBase::StixTensorBase(StixParams &stix_params, ReImPart re_im_part)
   : StixCoefBase(stix_params, re_im_part)
{}

void StixTensorBase::addParallelComp(real_t P, DenseMatrix & eps)
{
   // For b = B/|B|, add P * b b^T to epsilon
   for (int i=0; i<3; i++)
   {
      eps(i,i) += P * BVec_(i) * BVec_(i);
      for (int j = i+1; j<3; j++)
      {
         real_t eij = P * BVec_(i) * BVec_(j);
         eps(i,j) += eij;
         eps(j,i) += eij;
      }
   }
}

void StixTensorBase::addPerpDiagComp(real_t S, DenseMatrix & eps)
{
   // For b = B/|B|, add S * (I - b b^T) to epsilon
   for (int i=0; i<3; i++)
   {
      eps(i,i) += S * (1.0 - BVec_(i) * BVec_(i));
      for (int j = i+1; j<3; j++)
      {
         real_t eij = S * BVec_(i) * BVec_(j);
         eps(i,j) -= eij;
         eps(j,i) -= eij;
      }
   }
}

void StixTensorBase::addPerpSkewComp(real_t D, DenseMatrix & eps)
{
   // For b = B/|B|, add D * b\times to epsilon
   eps(1,2) -= D * BVec_[0];
   eps(2,1) += D * BVec_[0];

   eps(2,0) -= D * BVec_[1];
   eps(0,2) += D * BVec_[1];

   eps(0,1) -= D * BVec_[2];
   eps(1,0) += D * BVec_[2];
}

DielectricTensor::DielectricTensor(StixParams &stix_params, ReImPart re_im_part)
   : MatrixCoefficient(3),
     StixTensorBase(stix_params, re_im_part)
{}

void DielectricTensor::Eval(DenseMatrix &epsilon, ElementTransformation &T,
                            const IntegrationPoint &ip)
{
   // Initialize dielectric tensor to appropriate size
   epsilon.SetSize(3); epsilon = 0.0;

   // Collect density, temperature, and magnetic field values
   real_t Bmag = this->getBMagnitude(T, ip);
   BVec_ /= Bmag;
   nue_vals_ = nue_.Eval(T, ip);
   nui_vals_ = nui_.Eval(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate the Stix Coefficients
   complex_t S = S_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);
   complex_t P = P_cold_plasma(omega_, nue_vals_, density_vals_,
                               charges_, masses_, temperature_vals_,
                               nuprof_);
   complex_t D = D_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);

   if (re_im_part_ == REAL_PART)
   {
      this->addParallelComp( P.real(), epsilon);
      this->addPerpDiagComp( S.real(), epsilon);
      this->addPerpSkewComp(-D.imag(), epsilon);
   }
   else
   {
      this->addParallelComp( P.imag(), epsilon);
      this->addPerpDiagComp( S.imag(), epsilon);
      this->addPerpSkewComp( D.real(), epsilon);
   }
   epsilon *= epsilon0_;
}

InverseDielectricTensor::InverseDielectricTensor(StixParams &stix_params,
                                                 ReImPart re_im_part)
   : MatrixCoefficient(3),
     StixTensorBase(stix_params, re_im_part)
{}

void InverseDielectricTensor::Eval(DenseMatrix &epsilonInv,
                                   ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   // Initialize dielectric tensor to appropriate size
   epsilonInv.SetSize(3); epsilonInv = 0.0;

   // Collect density, temperature, and magnetic field values
   real_t Bmag = this->getBMagnitude(T, ip);
   BVec_ /= Bmag;
   nue_vals_ = nue_.Eval(T, ip);
   nui_vals_ = nui_.Eval(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   // Evaluate the Stix Coefficients
   complex_t S = S_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);
   complex_t P = P_cold_plasma(omega_, nue_vals_, density_vals_,
                               charges_, masses_, temperature_vals_,
                               nuprof_);
   complex_t D = D_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);

   complex_t Q = S * S - D * D;
   complex_t QInv = 1.0 / Q;
   complex_t SInv = S * QInv;
   complex_t PInv = 1.0 / P;
   complex_t DInv = D * QInv;

   if (re_im_part_ == REAL_PART)
   {
      this->addParallelComp( PInv.real(), epsilonInv);
      this->addPerpDiagComp( SInv.real(), epsilonInv);
      this->addPerpSkewComp( DInv.imag(), epsilonInv);
   }
   else
   {
      this->addParallelComp( PInv.imag(), epsilonInv);
      this->addPerpDiagComp( SInv.imag(), epsilonInv);
      this->addPerpSkewComp(-DInv.real(), epsilonInv);
   }

   epsilonInv *= 1.0 / epsilon0_;
}

SPDDielectricTensor::SPDDielectricTensor(StixParams &stix_params)
   : MatrixCoefficient(3),
     StixCoefBase(stix_params, REAL_PART)
{}

void SPDDielectricTensor::Eval(DenseMatrix &epsilon, ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   // Initialize dielectric tensor to appropriate size
   epsilon.SetSize(3);

   // Collect density, temperature, and magnetic field values
   real_t Bmag = this->getBMagnitude(T, ip);
   BVec_ /= Bmag;
   nue_vals_ = nue_.Eval(T, ip);
   nui_vals_ = nui_.Eval(T, ip);

   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);

   complex_t S = S_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);
   complex_t P = P_cold_plasma(omega_, nue_vals_, density_vals_,
                               charges_, masses_, temperature_vals_,
                               nuprof_);
   complex_t D = D_cold_plasma(omega_, Bmag, nue_vals_, nui_vals_,
                               density_vals_, charges_, masses_,
                               temperature_vals_, nuprof_, res_lim_);

   epsilon(0,0) = abs(S + (P - S) * BVec_(0) * BVec_(0));
   epsilon(1,1) = abs(S + (P - S) * BVec_(1) * BVec_(1));
   epsilon(2,2) = abs(S + (P - S) * BVec_(2) * BVec_(2));
   epsilon(0,1) = abs((P - S) * BVec_(0) * BVec_(1) - D * BVec_(2));
   epsilon(1,0) = abs((P - S) * BVec_(1) * BVec_(0) + D * BVec_(2));
   epsilon(0,2) = abs((P - S) * BVec_(0) * BVec_(2) + D * BVec_(1));
   epsilon(2,0) = abs((P - S) * BVec_(2) * BVec_(0) - D * BVec_(1));
   epsilon(1,2) = abs((P - S) * BVec_(1) * BVec_(2) - D * BVec_(0));
   epsilon(2,1) = abs((P - S) * BVec_(2) * BVec_(1) + D * BVec_(0));

   epsilon *= epsilon0_;
}

PlasmaProfile::PlasmaProfile(Type type, const Vector & params)
   : x_(3)
{
   Init(type, params);
}

void PlasmaProfile::Init(Type type, const Vector & params)
{
   type_ = type;
   p_ = params;

   MFEM_VERIFY(params.Size() == np_[type],
               "Incorrect number of parameters, " << params.Size()
               << ", for profile of type: " << type << ".");
}

real_t PlasmaProfile::Eval(ElementTransformation &T,
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
         real_t a = 0.5 * log(3.0) * (grad * x_) / p_[2];

         if (a < -10.0)
         {
            return p_[0];
         }
         else if (a < 10.0)
         {
            return 0.5 * (p_[0] + p_[1] + (p_[1] - p_[0]) * tanh(a));
         }
         else
         {
            return p_[1];
         }
      }
      break;
      case ELLIPTIC_COS:
      {
         real_t pmin = p_[0];
         real_t pmax = p_[1];
         real_t a = p_[2];
         real_t b = p_[3];
         Vector x0(&p_[4], 3);

         x_ -= x0;
         real_t r = pow(x_[0] / a, 2) + pow(x_[1] / b, 2);
         return pmin + (pmax - pmin) * (0.5 - 0.5 * cos(M_PI * sqrt(r)));
      }
      break;
      case PARABOLIC:
      {
         real_t pmin = p_[0];
         real_t pmax = p_[1];
         real_t a = p_[2];
         real_t b = p_[3];
         Vector x0(&p_[4], 3);

         x_ -= x0;
         real_t r = pow(x_[0] / a, 2) + pow(x_[1] / b, 2);
         return pmax - (pmax - pmin) * r;
      }
      break;
      case PEDESTAL:
      {
         real_t pmin = p_[0];
         real_t pmax = p_[1];
         real_t lambda_n = p_[2]; // Damping length
         real_t nu = p_[3]; // Strength of decline
         Vector x0(&p_[4], 3);

         x_ -= x0;
         real_t rho = pow(pow(x_[0], 2) + pow(x_[1], 2), 0.5);
         return (pmax - pmin) * pow(cosh(pow((rho / lambda_n), nu)), -1.0) + pmin;
      }
      break;
      case NUABSORB:
      {
         real_t nu0 = p_[0];
         real_t decay = p_[1];
         real_t shift = p_[2];

         return (nu0*exp(-(x_[0]-shift)/decay));
      }
      break;
      case NUE:
      {
         real_t rad_res_loc = p_[0];
         real_t nu0 = p_[1];
         real_t width = 1e-5;
         real_t rho = pow(pow(x_[0], 2) + pow(x_[1], 2), 0.5);
         return nu0*exp(-pow(rho-rad_res_loc, 2)/width) + (4e11)*exp(-(x_[0]-0.6)/0.04);
      }
      break;
      case NUI:
      {
         real_t rad_res_loc = p_[0];
         real_t nu0 = p_[1];
         real_t width = p_[2];
         real_t rho = pow(pow(x_[0], 2) + pow(x_[1], 2), 0.5);
         return nu0*exp(-pow(rho-rad_res_loc, 2)/width);
      }
      break;
      case CMODDEN:
      {
         real_t rho = pow(pow(x_[0], 2) + pow(x_[1], 2), 0.5);

         real_t pmin1 = 1e11;
         real_t pmax1 = (2e20 - 3e19);
         real_t lam1 = 0.86253;
         real_t n1 = 60.0;
         real_t ne1 = (pmax1 - pmin1)* pow(cosh(pow((rho / lam1), n1)), -1.0) + pmin1;

         real_t pmin2 = 1e11;
         real_t pmax2 = 3e19;
         real_t lam2 = 0.915;
         real_t n2 = 46.5;
         real_t ne2 = (pmax2 - pmin2)* pow(cosh(pow((rho / lam2), n2)), -1.0) + pmin2;
         return ne1 + ne2;
      }
      break;
      case SPARC_RES:
      {
         real_t nu0 = p_[0];

         real_t A = 9.56300019e-02;
         real_t B = 1.27703065;
         real_t C = -1.47586242e-06;
         real_t D = 1.92995180;

         real_t E = 0.05125891;
         real_t F = 1.31119407;
         real_t G = -0.00925291;
         real_t H = 1.43560241;

         real_t r = x_[1];
         real_t z = x_[0];

         real_t val1 = B*z - C;
         real_t sincfunc1 = A*(sin(val1)/val1) + D;

         real_t val2 = F*z - G;
         real_t sincfunc2 = E*(sin(val2)/val2) + H;

         real_t res1 = nu0*exp(-pow(r-sincfunc1, 2)/0.002);
         real_t res2 = nu0*exp(-pow(r-sincfunc2, 2)/0.002);

         return res1+res2;
      }
      break;
      case CUSTOM1:
      {
         real_t nu0 = p_[0];
         real_t decay = p_[1];

         return (nu0*exp(-x_[0]/decay));
      }
      break;
      case CUSTOM2:
      {
         real_t rad_res_loc = p_[0];
         real_t nu0 = p_[1];
         real_t width = 3e-5;
         real_t rho = pow(pow(x_[0], 2) + pow(x_[1], 2), 0.5);
         return nu0*exp(-pow(rho-rad_res_loc, 2)/width) + (1e14)*exp(-rho/0.1);
      }
      break;
      case POWER:
      {
         int comp = (int)rint(p_[0]);
         real_t a = p_[1];
         real_t b = p_[2];
         real_t p = p_[3];

         return a + (b - a) * pow(x_[comp], p);
      }
      break;
      case WHAM:
      {
         real_t a = p_[0];
         real_t b = p_[1];
         real_t c = p_[2];
         real_t d = p_[3];
         real_t p = p_[4];
         real_t ba = p_[5];
         real_t bb = p_[6];

         real_t bz = ba + bb * pow(x_[0], 4);
         real_t r_lim = sqrt(0.01 / bz);

         real_t rho_r = 0.5 * (1.0 + tanh(d * (r_lim - x_[1])));
         return a + ((b + (c - b) * pow(x_[0], p)) - a) * rho_r;
      }
      break;
      default:
         return 0.0;
   }
}

MultiSpeciesPlasmaProfiles::MultiSpeciesPlasmaProfiles(
   Array<PlasmaProfile::Type> types,
   const Vector & params)
   : VectorCoefficient(types.Size()),
     prof_(types.Size()),
     electron_profile_type_(PlasmaProfile::AVERAGE),
     ion_spec_coefs_(types.Size() - 1)
{
   const int num_ion_spec = types.Size() - 1;
   ion_spec_coefs_ = 1.0 / num_ion_spec;

   int o = 0;
   unsigned int t = 0;
   for (; t<num_ion_spec; t++)
   {
      PlasmaProfile::Type type = types[t];
      int np = PlasmaProfile::GetNumParams(type);
      const Vector param(const_cast<real_t*>(&params[o]), np);
      prof_[t].Init(type, param);
      o += np;
   }

   t = num_ion_spec;
   electron_profile_type_ = types[t];
   switch (electron_profile_type_)
   {
      case PlasmaProfile::AVERAGE:
         // Nothing to do
         break;
      case PlasmaProfile::NEUTRALITY:
         MFEM_VERIFY(false, "Charge neutrality electron profile requires "
                     "the ion charges.");
         break;
      default:
         // Create user requested type for electrons
         PlasmaProfile::Type type = types[t];
         int np = PlasmaProfile::GetNumParams(type);
         const Vector param(const_cast<real_t*>(&params[o]), np);
         prof_[t].Init(type, param);
         o += np;
         break;
   }
}

MultiSpeciesPlasmaProfiles::MultiSpeciesPlasmaProfiles(
   Array<PlasmaProfile::Type> types,
   const Vector & params, const Vector & charges)
   : VectorCoefficient(types.Size()),
     prof_(types.Size()),
     electron_profile_type_(PlasmaProfile::NEUTRALITY),
     ion_spec_coefs_(types.Size() - 1)
{
   const int num_ion_spec = types.Size() - 1;

   int o = 0;
   unsigned int t = 0;
   for (; t<num_ion_spec; t++)
   {
      PlasmaProfile::Type type = types[t];
      int np = PlasmaProfile::GetNumParams(type);
      const Vector param(const_cast<real_t*>(&params[o]), np);
      prof_[t].Init(type, param);
      ion_spec_coefs_[t] = charges[t];
      o += np;
   }
   ion_spec_coefs_ /= -charges[num_ion_spec];

   t = num_ion_spec;
   electron_profile_type_ = types[t];
   switch (electron_profile_type_)
   {
      case PlasmaProfile::AVERAGE:
         // Reset the coefs to compute the average
         ion_spec_coefs_ = 1.0 / num_ion_spec;
         break;
      case PlasmaProfile::NEUTRALITY:
         // Nothing to do
         break;
      default:
         // Create user requested type for electrons
         PlasmaProfile::Type type = types[t];
         int np = PlasmaProfile::GetNumParams(type);
         const Vector param(const_cast<real_t*>(&params[o]), np);
         prof_[t].Init(type, param);
         o += np;
         break;
   }
}

void MultiSpeciesPlasmaProfiles::Eval(Vector &p, ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   p.SetSize(vdim);

   real_t electron_prof_val = 0.0;
   for (int i=0; i<vdim - 1; i++)
   {
      p[i] = prof_[i].Eval(T, ip);
      electron_prof_val += p[i] * ion_spec_coefs_[i];
   }

   switch (electron_profile_type_)
   {
      case PlasmaProfile::AVERAGE:
      case PlasmaProfile::NEUTRALITY:
         // Set average or charge neutrality value for the electron profile
         p[vdim - 1] = electron_prof_val;
         break;
      default:
         // Compute user requested electron profile value
         p[vdim - 1] = prof_[vdim - 1].Eval(T, ip);
         break;
   }
}

BFieldProfile::BFieldProfile(Type type, const Vector & params, bool unit)
   : VectorCoefficient(3), type_(type), p_(params), unit_(unit),
     x3_(3), x_(x3_.GetData(), 3)
{
   MFEM_VERIFY(params.Size() == np_[type],
               "Incorrect number of parameters, " << params.Size()
               << ", for profile of type: " << type << ".");
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
         V[0] = p_[0];
         V[1] = p_[1];
         V[2] = p_[2];
         break;
      default:
         V[0] = 0.0;
         V[1] = 0.0;
         V[2] = 5.4;
         break;
   }
   if (unit_)
   {
      V /= V.Norml2();
   }
}

ColdPlasmaPlaneWaveBase::ColdPlasmaPlaneWaveBase(Vector k_dir, char type,
                                                 StixParams & stix_params,
                                                 ReImPart re_im_part)
   :  StixCoefBase(stix_params, re_im_part),
      VectorCoefficient(3),
      type_(type),
      Bmag_(-1.0),
      kappa_(0.0),
      b_(3),
      bc_(3),
      bcc_(3),
      k_dir_(k_dir),
      k_r_(3),
      k_i_(3),
      beta_r_(3),
      beta_i_(3)
{
   k_dir_ /= k_dir.Norml2();

   beta_r_ = 0.0;
   beta_i_ = 0.0;
}

void ColdPlasmaPlaneWaveBase::ComputeFieldAxes(ElementTransformation &T,
                                               const IntegrationPoint &ip)
{
   Bmag_ = this->getBMagnitude(T, ip);

   b_.Set(1.0 / Bmag_, BVec_);
   k_dir_.cross3D(b_, kxb_);

   {
      real_t bx = b_(0);
      real_t by = b_(1);
      real_t bz = b_(2);

      bc_(0) = by - bz;
      bc_(1) = bz - bx;
      bc_(2) = bx - by;

      bcc_(0) = by*by + bz*bz - bx*(by + bz);
      bcc_(1) = bz*bz + bx*bx - by*(bz + bx);
      bcc_(2) = bx*bx + by*by - bz*(bx + by);

      bc_  *= 1.0 / bc_.Norml2();
      bcc_ *= 1.0 / bcc_.Norml2();
   }
}

void ColdPlasmaPlaneWaveBase::ComputeStixCoefs()
{
   S_ = S_cold_plasma(omega_, Bmag_, 0.0, 0.0, density_vals_, charges_, masses_,
                      temperature_vals_, nuprof_, res_lim_);
   D_ = D_cold_plasma(omega_, Bmag_, 0.0, 0.0, density_vals_, charges_, masses_,
                      temperature_vals_, nuprof_, res_lim_);
   P_ = P_cold_plasma(omega_, 0.0, density_vals_, charges_, masses_,
                      temperature_vals_, nuprof_);
}

void ColdPlasmaPlaneWaveBase::ComputeWaveNumber()
{
   switch (type_)
   {
      case 'L':
      {
         kappa_ = omega_ * sqrt(S_ - D_) / c0_;
      }
      break;
      case 'R':
      {
         kappa_ = omega_ * sqrt(S_ + D_) / c0_;
      }
      break;
      case 'O':
      {
         kappa_ = omega_ * sqrt(P_) / c0_;
      }
      break;
      case 'X':
      {
         kappa_ = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;
      }
      break;
   }

   if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }
}

void ColdPlasmaPlaneWaveBase::ComputeWaveVector()
{
   k_r_.Set(kappa_.real(), k_dir_);
   k_i_.Set(kappa_.imag(), k_dir_);
}

ColdPlasmaPlaneWaveE::ColdPlasmaPlaneWaveE(Vector k_dir, char type,
                                           StixParams & stix_params,
                                           ReImPart re_im_part)
   :  ColdPlasmaPlaneWaveBase(k_dir, type, stix_params, re_im_part),
      e_r_(3),
      e_i_(3)
{
}

void ColdPlasmaPlaneWaveE::ComputePolarizationVectorE()
{
   switch (type_)
   {
      case 'L':
      {
         e_r_.Set(1.0, bc_);
         e_i_.Set(1.0, bcc_);
      }
      break;
      case 'R':
      {
         e_r_.Set( 1.0, bc_);
         e_i_.Set(-1.0, bcc_);
      }
      break;
      case 'O':
      {
         e_r_.Set(1.0, b_);
         e_i_ = 0.0;
      }
      break;
      case 'X':
      {
         complex_t den = sqrt(S_ * S_ + D_ * D_);
         complex_t ec  = D_ / den;
         complex_t ecc = S_ / den;

         e_r_.Set(ecc.real(), kxb_);
         e_r_.Add( ec.imag(), k_dir_);
         e_i_.Set(-ec.real(), k_dir_);
         e_i_.Add(ecc.imag(), kxb_);
      }
      break;
   }
}

void ColdPlasmaPlaneWaveE::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   V.SetSize(3);

   complex_t i = complex_t(0.0,1.0);

   real_t x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   this->ComputeFieldAxes(T, ip);
   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);
   this->ComputeStixCoefs();
   this->ComputeWaveNumber();
   this->ComputeWaveVector();
   this->ComputePolarizationVectorE();

   complex_t kx = 0.0;
   for (int d=0; d<x.Size(); d++)
   {
      kx += (k_r_[d] - beta_r_[d] + i * (k_i_[d] - beta_i_[d])) * x[d];
   }

   complex_t e_mag = exp(i * kx);

   if (re_im_part_ == REAL_PART)
   {
      add(e_mag.real(), e_r_, -e_mag.imag(), e_i_, V);
   }
   else
   {
      add(e_mag.imag(), e_r_, e_mag.real(), e_i_, V);
   }
}

ColdPlasmaPlaneWaveH::ColdPlasmaPlaneWaveH(Vector k_dir, char type,
                                           StixParams & stix_params,
                                           ReImPart re_im_part)
   :  ColdPlasmaPlaneWaveE(k_dir, type, stix_params, re_im_part),
      h_r_(3),
      h_i_(3),
      v_tmp_(3)
{
}

void ColdPlasmaPlaneWaveH::ComputePolarizationVectorH()
{
   this->ComputePolarizationVectorE();

   k_r_.cross3D(e_r_, h_r_);
   k_i_.cross3D(e_i_, v_tmp_);
   h_r_ -= v_tmp_;
   h_r_ /= omega_ * mu0_;

   k_r_.cross3D(e_i_, h_i_);
   k_i_.cross3D(e_r_, v_tmp_);
   h_i_ += v_tmp_;
   h_i_ /= omega_ * mu0_;
}

void ColdPlasmaPlaneWaveH::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   V.SetSize(3);

   complex_t i = complex_t(0.0,1.0);

   real_t x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   this->ComputeFieldAxes(T, ip);
   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);
   this->ComputeStixCoefs();
   this->ComputeWaveNumber();
   this->ComputeWaveVector();
   this->ComputePolarizationVectorH();

   complex_t kx = 0.0;
   for (int d=0; d<x.Size(); d++)
   {
      kx += (k_r_[d] - beta_r_[d] + i * (k_i_[d] - beta_i_[d])) * x[d];
   }

   complex_t h_mag = exp(i * (kx - omega_ * time));

   if (re_im_part_ == REAL_PART)
   {
      add(h_mag.real(), h_r_, -h_mag.imag(), h_i_, V);
   }
   else
   {
      add(h_mag.imag(), h_r_, h_mag.real(), h_i_, V);
   }
}
ColdPlasmaCenterFeedBase::ColdPlasmaCenterFeedBase(real_t j_pos, real_t j_dx,
                                                   int j_profile)
   : j_pos_(j_pos),
     j_dx_(j_dx),
     j_prof_(j_profile)
{}

ColdPlasmaCenterFeedE::ColdPlasmaCenterFeedE(Vector k_dir, char type,
                                             real_t j_pos, real_t j_dx,
                                             int j_profile,
                                             StixParams & stix_params,
                                             ReImPart re_im_part)
   :  ColdPlasmaPlaneWaveE(k_dir, type, stix_params, re_im_part),
      ColdPlasmaCenterFeedBase(j_pos, j_dx, j_profile)
{}

void ColdPlasmaCenterFeedE::Eval(Vector &V, ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   V.SetSize(3);

   complex_t i = complex_t(0.0,1.0);

   real_t x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   this->ComputeFieldAxes(T, ip);
   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);
   this->ComputeStixCoefs();
   this->ComputeWaveNumber();
   this->ComputeWaveVector();
   this->ComputePolarizationVectorE();

   complex_t e_mag;
   complex_t phase;
   complex_t kx = 0.0;
   for (int d=0; d<x.Size(); d++)
   {
      kx += (beta_r_[d] + i * beta_i_[d]) * x[d];
   }
   phase = exp(-i * kx);

   real_t ks = 2.0 * M_PI / j_dx_;

   if (x[0] < j_pos_ - 0.5 * j_dx_)
   {
      e_mag = i / kappa_ / kappa_;
      if (j_prof_ == 0)
      {
         e_mag *= sin(0.5 * kappa_ * j_dx_);
      }
      else
      {
         e_mag *= 0.5 * M_PI * sinc(M_PI * (1.0 - kappa_ / ks))
                  / (1.0 + kappa_ / ks);
      }
      phase *= exp(i * (-kappa_ * (x[0] - j_pos_) - omega_ * time));
      e_mag *= phase;
   }
   else if (x[0] < j_pos_ + 0.5 * j_dx_)
   {
      e_mag = 1.0 / kappa_ / kappa_;

      complex_t coskx = cos(kappa_ * (x[0] - j_pos_));
      if (j_prof_ == 0)
      {
         e_mag *= coskx * exp(0.5 * i * kappa_ * j_dx_) - 1.0;
      }
      else
      {
         complex_t kx = kappa_ * (x[0] - j_pos_);
         complex_t sinkx = sin(kx);
         complex_t kd = ks - kappa_;
         complex_t kdn = 1.0 - kappa_ / ks;
         complex_t kdx = kd * (x[0] - j_pos_);
         complex_t sinckdx = sinc(kdx);
         complex_t sinckdx2 = sinc(kdx / 2.0);
         complex_t sinckdn = sinc(M_PI * kdn);
         complex_t sinckdn2 = sinc(M_PI * kdn / 2.0);
         e_mag *= 0.5 * (coskx *
                         (0.5 * kd *
                          (pow(M_PI * sinckdn2, 2) - pow(kx * sinckdx2, 2))
                          - ks - kappa_ + i * M_PI * ks * sinckdn
                         )
                         - kappa_ * kx * sinkx * sinckdx - ks - kappa_
                        ) / (kappa_ + ks);
      }
      phase *= exp(-i * omega_ * time);
      e_mag *= phase;
   }
   else
   {
      e_mag = i / kappa_ / kappa_;
      if (j_prof_ == 0)
      {
         e_mag *= sin(0.5 * kappa_ * j_dx_);
      }
      else
      {
         e_mag *= 0.5 * M_PI * sinc(M_PI * (1.0 - kappa_ / ks))
                  / (1.0 + kappa_ / ks);
      }
      phase *= exp(i * (kappa_ * (x[0] - j_pos_) - omega_ * time));
      e_mag *= phase;
   }

   if (re_im_part_ == REAL_PART)
   {
      add(e_mag.real(), e_r_, -e_mag.imag(), e_i_, V);
   }
   else
   {
      add(e_mag.imag(), e_r_, e_mag.real(), e_i_, V);
   }
}

ColdPlasmaCenterFeedH::ColdPlasmaCenterFeedH(Vector k_dir, char type,
                                             real_t j_pos, real_t j_dx,
                                             int j_profile,
                                             StixParams & stix_params,
                                             ReImPart re_im_part)
   : ColdPlasmaPlaneWaveH(k_dir, type, stix_params, re_im_part),
     ColdPlasmaCenterFeedBase(j_pos, j_dx, j_profile)
{
}

void ColdPlasmaCenterFeedH::Eval(Vector &V, ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   V.SetSize(3);

   complex_t i = complex_t(0.0,1.0);

   real_t x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   this->ComputeFieldAxes(T, ip);
   this->fillDensityVals(T, ip);
   this->fillTemperatureVals(T, ip);
   this->ComputeStixCoefs();
   this->ComputeWaveNumber();
   this->ComputeWaveVector();
   this->ComputePolarizationVectorH();

   complex_t h_mag;
   complex_t phase;
   complex_t kx = 0.0;
   for (int d=0; d<x.Size(); d++)
   {
      kx += (beta_r_[d] + i * beta_i_[d]) * x[d];
   }
   phase = exp(-i * kx);

   real_t ks = 2.0 * M_PI / j_dx_;

   if (x[0] < j_pos_ - 0.5 * j_dx_)
   {
      h_mag = -i / kappa_ / kappa_;
      if (j_prof_ == 0)
      {
         h_mag *= sin(0.5 * kappa_ * j_dx_);
      }
      else
      {
         h_mag *= 0.5 * M_PI * sinc(M_PI * (1.0 - kappa_ / ks))
                  / (1.0 + kappa_ / ks);
      }
      phase *= exp(i * (-kappa_ * (x[0] - j_pos_) - omega_ * time));
      h_mag *= phase;
   }
   else if (x[0] < j_pos_ + 0.5 * j_dx_)
   {
      h_mag = i / kappa_ / kappa_;
      if (j_prof_ == 0)
      {
         h_mag *= sin(kappa_ * (x[0] - j_pos_));
         h_mag *= exp(i * 0.5 * kappa_ * j_dx_);
      }
      else
      {
         h_mag *= 0.5 / (1.0 - pow(kappa_ / ks, 2.0));
         h_mag *= sin(kappa_ * (x[0] - j_pos_)) *
                  exp(0.5 * i * kappa_ * j_dx_)
                  + sin(ks * (x[0] - j_pos_)) * kappa_ / ks;
      }
      phase *= exp(-i * omega_ * time);
      h_mag *= phase;
   }
   else
   {
      h_mag = i / kappa_ / kappa_;
      if (j_prof_ == 0)
      {
         h_mag *= sin(0.5 * kappa_ * j_dx_);
      }
      else
      {
         h_mag *= 0.5 * M_PI * sinc(M_PI * (1.0 - kappa_ / ks))
                  / (1.0 + kappa_ / ks);
      }
      phase *= exp(i * (kappa_ * (x[0] - j_pos_) - omega_ * time));
      h_mag *= phase;
   }

   if (re_im_part_ == REAL_PART)
   {
      add(h_mag.real(), h_r_, -h_mag.imag(), h_i_, V);
   }
   else
   {
      add(h_mag.imag(), h_r_, h_mag.real(), h_i_, V);
   }
}

ColdPlasmaCenterFeedJ::ColdPlasmaCenterFeedJ(Vector k_dir, char type,
                                             real_t j_pos, real_t j_dx,
                                             int j_profile,
                                             StixParams & stix_params,
                                             ReImPart re_im_part)
   : ColdPlasmaCenterFeedH(k_dir, type, j_pos, j_dx, j_profile,
                           stix_params, re_im_part),
     j_r_(3),
     j_i_(3)
{
}

void ColdPlasmaCenterFeedJ::ComputePolarizationVectorJ()
{
   this->ComputePolarizationVectorH();

   // k_dir_.cross3D(h_r_, j_r_);
   // k_dir_.cross3D(h_i_, j_i_);

   k_r_.cross3D(h_i_, j_r_);
   k_i_.cross3D(h_r_, v_tmp_);
   j_r_ += v_tmp_;
   j_r_ *= -1.0;

   k_r_.cross3D(h_r_, j_i_);
   k_i_.cross3D(h_i_, v_tmp_);
   j_i_ -= v_tmp_;
}

void ColdPlasmaCenterFeedJ::Eval(Vector &V, ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   V.SetSize(3);

   real_t x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   if (x[0] > j_pos_ - 0.5 * j_dx_ && x[0] < j_pos_ + 0.5 * j_dx_)
   {
      complex_t i = complex_t(0.0,1.0);

      this->ComputeFieldAxes(T, ip);
      if (type_ == 'X')
      {
         this->fillDensityVals(T, ip);
         this->fillTemperatureVals(T, ip);
         this->ComputeStixCoefs();
         this->ComputeWaveNumber();
         this->ComputeWaveVector();
      }
      else
      {
         kappa_ = 1.0;
         k_r_ = k_dir_;
         k_i_ = 0.0;
      }
      this->ComputePolarizationVectorJ();

      complex_t j_mag = 1.0 / kappa_ / kappa_;
      if (j_prof_ == 1)
      {
         j_mag *= pow(cos(M_PI * (x[0] - j_pos_) / j_dx_), 2.0);
      }

      complex_t kx = 0.0;
      for (int d=0; d<x.Size(); d++)
      {
         kx += (beta_r_[d] + i * beta_i_[d]) * x[d];
      }
      complex_t phase = exp(-i * (kx + omega_ * time));
      j_mag *= phase;

      if (re_im_part_ == REAL_PART)
      {
         add(j_mag.real(), j_r_, -j_mag.imag(), j_i_, V);
      }
      else
      {
         add(j_mag.imag(), j_r_, j_mag.real(), j_i_, V);
      }

   }
   else
   {
      V = 0.0;
   }
}

} // namespace plasma

} // namespace mfem
