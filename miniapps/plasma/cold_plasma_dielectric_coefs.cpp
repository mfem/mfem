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
                              const Vector & temp)
{
   complex<double> val(1.0, 0.0);
   complex<double> mass_correction(1.0, 0.0);
   for (int i=1; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      double Te = temp[i] * q_;
      double coul_log = CoulombLog(n, Te);
      double nuei = nu_ei(q, coul_log, m, Te, n);
      complex<double> collision_correction(0, nuei/omega);
      mass_correction += collision_correction;
   }

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m*mass_correction;
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
                              const Vector & temp)
{
   complex<double> val(1.0, 0.0);
   complex<double> mass_correction(1.0, 0.0);
   for (int i=1; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      double Te = temp[i] * q_;
      double coul_log = CoulombLog(n, Te);
      double nuei = nu_ei(q, coul_log, m, Te, n);
      complex<double> collision_correction(0, nuei/omega);
      mass_correction += collision_correction;
   }

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m*mass_correction;
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
                              const Vector & temp)
{
   complex<double> val(1.0, 0.0);
   complex<double> mass_correction(1.0, 0.0);
   for (int i=1; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      double Te = temp[i] * q_;
      double coul_log = CoulombLog(n, Te);
      double nuei = nu_ei(q, coul_log, m, Te, n);
      complex<double> collision_correction(0, nuei/omega);
      mass_correction += collision_correction;
   }

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m*mass_correction;
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
                              const Vector & temp)
{
   complex<double> val(0.0, 0.0);
   complex<double> mass_correction(1.0, 0.0);
   for (int i=1; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      double Te = temp[i] * q_;
      double coul_log = CoulombLog(n, Te);
      double nuei = nu_ei(q, coul_log, m, Te, n);
      complex<double> collision_correction(0, nuei/omega);
      mass_correction += collision_correction;
   }

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m*mass_correction;
      complex<double> w_c = omega_c(Bmag, q, m_eff);
      complex<double> w_p = omega_p(n, q, m_eff);
      val += w_p * w_p * w_c / (omega *(omega * omega - w_c * w_c));
   }
   return val;
}

complex<double> P_cold_plasma(double omega,
                              const Vector & number,
                              const Vector & charge,
                              const Vector & mass,
                              const Vector & temp)
{
   complex<double> val(1.0, 0.0);
   complex<double> mass_correction(1.0, 0.0);
   for (int i=1; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      double Te = temp[i] * q_;
      double coul_log = CoulombLog(n, Te);
      double nuei = nu_ei(q, coul_log, m, Te, n);
      complex<double> collision_correction(0, nuei/omega);
      mass_correction += collision_correction;
   }

   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      complex<double> m_eff = m*mass_correction;
      complex<double> w_p = omega_p(n, q, m_eff);
      val -= w_p * w_p / (omega * omega);
   }
   return val;
}

void real_epsilon_sigma(double omega, const Vector &B,
                        const Vector &density_vals,
                        const Vector &temperature_vals,
                        double *real_epsilon, double *real_sigma)
{
   double Bnorm = B.Norml2();

   // double phi = 0.0;
   // double MData[9] = {cos(phi), 0.0, -sin(phi),
   //                    0.0,      1.0,       0.0,
   //                    sin(phi), 0.0,  cos(phi)
   //                   };
   // DenseMatrix M(MData, 3, 3);

   // Vector Blocal(3);
   // M.Mult(B, Blocal);

   double Z1 = 1.0, Z2 = 18.0;
   double qe = -q_, qi1 = Z1 * q_, qi2 = Z2 * q_;
   double mi1 = 2.01410178 * amu_, mi2 = 39.948 * amu_;
   double ne = density_vals[0], ni1 = density_vals[1], ni2 = density_vals[2];
   /*double Te = temperature_vals[0];
   double Ti1 = temperature_vals[1];
   double Ti2 = temperature_vals[2];
   double vTe = sqrt(2.0 * Te / me_);
   double debye_length = sqrt((epsilon0_ * Te) / (ne * pow(qe, 2)));
   double b90_1 = (qe * qi1)/(4.0 * M_PI * epsilon0_ * me_ * pow(vTe, 2)),
          b90_2 = (qe * qi2)/(4.0 * M_PI * epsilon0_ * me_ * pow(vTe, 2));
   double nu_ei1 = (pow(qe * qi1, 2) * ni1 * log(debye_length / b90_1)) /
          (4.0 * M_PI * pow(epsilon0_ * me_, 2) * pow(Te, 3.0/2.0));
   double nu_ei2 = (pow(qe * qi2, 2) * ni2 * log(debye_length / b90_2)) /
          (4.0 * M_PI * pow(epsilon0_ * me_, 2) * pow(Te, 3.0/2.0));

   // Effective Mass
   complex<double> me_eff(me_, -me_*(nu_ei1/omega + nu_ei2/omega));
   complex<double> mi1_eff(mi1, -mi1*(nu_ei1/omega + nu_ei2/omega));
   complex<double> mi2_eff(mi2, -mi2*(nu_ei1/omega + nu_ei2/omega));
    */

   // Squared plasma frequencies for each species
   double wpe  = (ne  * pow( qe, 2))/(me_kg_ * epsilon0_);
   double wpi1 = (ni1 * pow(qi1, 2))/(mi1 * epsilon0_);
   double wpi2 = (ni2 * pow(qi2, 2))/(mi2 * epsilon0_);

   // Cyclotron frequencies for each species
   double wce  = qe  * Bnorm / me_kg_;
   double wci1 = qi1 * Bnorm / mi1;
   double wci2 = qi2 * Bnorm / mi2;

   double S = (1.0 -
               wpe  / (pow(omega, 2) - pow( wce, 2)) -
               wpi1 / (pow(omega, 2) - pow(wci1, 2)) -
               wpi2 / (pow(omega, 2) - pow(wci2, 2)));
   double P = (1.0 -
               wpe  / pow(omega, 2) -
               wpi1 / pow(omega, 2) -
               wpi2 / pow(omega, 2));
   double D = (wce  * wpe  / (omega * (pow(omega, 2) - pow( wce, 2))) +
               wci1 * wpi1 / (omega * (pow(omega, 2) - pow(wci1, 2))) +
               wci2 * wpi2 / (omega * (pow(omega, 2) - pow(wci2, 2))));

   // Complex Dielectric tensor elements
   double th = atan2(B(2), B(0));
   double ph = atan2(B(0) * cos(th) + B(2) * sin(th), -B(1));

   double e_xx = (P - S) * pow(sin(ph), 2) * pow(cos(th), 2) + S;
   double e_yy = (P - S) * pow(cos(ph), 2) + S;
   double e_zz = (P - S) * pow(sin(ph), 2) * pow(sin(th), 2) + S;

   complex<double> e_xy(-(P - S) * cos(ph) * cos(th) * sin(ph),
                        - D * sin(th) * sin(ph));
   complex<double> e_xz((P - S) * pow(sin(ph), 2) * sin(th) * cos(th),
                        - D * cos(ph));
   complex<double> e_yz(-(P - S) * sin(th) * cos(ph) * sin(ph),
                        - D * cos(th) * sin(ph));

   complex<double> e_yx = std::conj(e_xy);
   complex<double> e_zx = std::conj(e_xz);
   complex<double> e_zy = std::conj(e_yz);

   if (real_epsilon != NULL)
   {
      real_epsilon[0] = epsilon0_ * e_xx;
      real_epsilon[1] = epsilon0_ * e_yx.real();
      real_epsilon[2] = epsilon0_ * e_zx.real();
      real_epsilon[3] = epsilon0_ * e_xy.real();
      real_epsilon[4] = epsilon0_ * e_yy;
      real_epsilon[5] = epsilon0_ * e_zy.real();
      real_epsilon[6] = epsilon0_ * e_xz.real();
      real_epsilon[7] = epsilon0_ * e_yz.real();
      real_epsilon[8] = epsilon0_ * e_zz;
   }
   if (real_sigma != NULL)
   {
      real_sigma[0] = 0.0;
      real_sigma[1] = e_yx.imag() * omega * epsilon0_;
      real_sigma[2] = e_zx.imag() * omega * epsilon0_;
      real_sigma[3] = e_xy.imag() * omega * epsilon0_;
      real_sigma[4] = 0.0;
      real_sigma[5] = e_zy.imag() * omega * epsilon0_;
      real_sigma[6] = e_xz.imag() * omega * epsilon0_;
      real_sigma[7] = e_yz.imag() * omega * epsilon0_;
      real_sigma[8] = 0.0;
   }

}

DielectricTensor::DielectricTensor(const ParGridFunction & B,
                                   const BlockVector & density,
                                   const BlockVector & temp,
                                   const ParFiniteElementSpace & L2FESpace,
                                   const ParFiniteElementSpace & H1FESpace,
                                   double omega,
                                   const Vector & charges,
                                   const Vector & masses,
                                   bool realPart)
   : MatrixCoefficient(3),
     B_(B),
     density_(density),
     temp_(temp),
     L2FESpace_(L2FESpace),
     H1FESpace_(H1FESpace),
     omega_(omega),
     realPart_(realPart),
     charges_(charges),
     masses_(masses)
{
   density_vals_.SetSize(charges_.Size());
   temp_vals_.SetSize(charges_.Size());
}

void DielectricTensor::Eval(DenseMatrix &epsilon, ElementTransformation &T,
                            const IntegrationPoint &ip)
{
   // Initialize dielectric tensor to appropriate size
   epsilon.SetSize(3);

   // Collect density, temperature, and magnetic field values
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

   if (realPart_)
   {
      complex<double> S = S_cold_plasma(omega_, Bmag, density_vals_,
                                        charges_, masses_, temp_vals_);
      complex<double> P = P_cold_plasma(omega_, density_vals_,
                                        charges_, masses_, temp_vals_);
      complex<double> D = D_cold_plasma(omega_, Bmag, density_vals_,
                                        charges_, masses_, temp_vals_);

      epsilon(0,0) =  (real(P) - real(S)) *
                      pow(sin(ph), 2) * pow(cos(th), 2) + real(S);
      epsilon(1,1) =  (real(P) - real(S)) * pow(cos(ph), 2) + real(S);
      epsilon(2,2) =  (real(P) - real(S)) *
                      pow(sin(ph), 2) * pow(sin(th), 2) + real(S);
      epsilon(0,1) = (real(P) - real(S)) * cos(ph) * cos(th) * sin(ph) +
                     imag(D) * sin(th) * sin(ph);
      epsilon(0,2) =  (real(P) - real(S)) *
                      pow(sin(ph), 2) * sin(th) * cos(th) + imag(D) * cos(ph);
      epsilon(1,2) = (real(P) - real(S)) * sin(th) * cos(ph) * sin(ph) -
                     imag(D) * cos(th) * sin(ph);
      epsilon(1,0) = (real(P) - real(S)) * cos(ph) * cos(th) * sin(ph) -
                     imag(D) * sin(th) * sin(ph);
      epsilon(2,1) = (real(P) - real(S)) * sin(th) * cos(ph) * sin(ph) +
                     imag(D) * cos(th) * sin(ph);
      epsilon(2,0) = (real(P) - real(S)) *
                     pow(sin(ph),2) * sin(th) * cos(th) - imag(D) * cos(ph);
   }
   else
   {
      complex<double> S = S_cold_plasma(omega_, Bmag, density_vals_,
                                        charges_, masses_, temp_);
      complex<double> P = P_cold_plasma(omega_, density_vals_,
                                        charges_, masses_, temp_);
      complex<double> D = D_cold_plasma(omega_, Bmag, density_vals_,
                                        charges_, masses_, temp_);

      epsilon(0,0) = (imag(P) - imag(S)) *
                     pow(sin(ph), 2) * pow(cos(th), 2) + imag(S);
      epsilon(1,1) = (imag(P) - imag(S)) * pow(cos(ph), 2) + imag(S);
      epsilon(2,2) = (imag(P) - imag(S)) *
                     pow(sin(ph), 2) * pow(sin(th), 2) + imag(S);
      epsilon(0,1) = (imag(P) - imag(S)) * cos(ph) * cos(th) * sin(ph) -
                     real(D) * sin(th) * sin(ph);
      epsilon(0,2) = (imag(P) - imag(S)) *
                     pow(sin(ph), 2) * sin(th) * cos(th) - real(D) * cos(ph);
      epsilon(1,2) = (imag(P) - imag(S)) * sin(th) * cos(ph) * sin(ph) +
                     real(D) * cos(th) * sin(ph);
      epsilon(1,0) = (imag(P) - imag(S)) * cos(ph) * cos(th) * sin(ph) +
                     real(D) * sin(th) * sin(ph);
      epsilon(2,1) = (imag(P) - imag(S)) * sin(th) * cos(ph) * sin(ph) -
                     real(D) * cos(th) * sin(ph);
      epsilon(2,0) = (imag(P) - imag(S)) *
                     pow(sin(ph), 2) * sin(th) * cos(th) + real(D) * cos(ph);
   }
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

SPDDielectricTensor::SPDDielectricTensor(
   const ParGridFunction & B,
   const BlockVector & density,
   const BlockVector & temp,
   const ParFiniteElementSpace & L2FESpace,
   const ParFiniteElementSpace & H1FESpace,
   double omega,
   const Vector & charges,
   const Vector & masses)
   : MatrixCoefficient(3),
     B_(B),
     density_(density),
     temp_(temp),
     L2FESpace_(L2FESpace),
     H1FESpace_(H1FESpace),
     omega_(omega),
     charges_(charges),
     masses_(masses)
{
   density_vals_.SetSize(charges_.Size());
   temp_vals_.SetSize(charges_.Size());
}

void SPDDielectricTensor::Eval(DenseMatrix &epsilon, ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   // Initialize dielectric tensor to appropriate size
   epsilon.SetSize(3);

   // Collect density, temperature, and magnetic field values
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

   complex<double> S = S_cold_plasma(omega_, Bmag, density_vals_,
                                     charges_, masses_, temp_vals_);
   complex<double> P = P_cold_plasma(omega_, density_vals_,
                                     charges_, masses_, temp_vals_);
   complex<double> D = D_cold_plasma(omega_, Bmag, density_vals_,
                                     charges_, masses_, temp_vals_);

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
   MFEM_ASSERT(params.Size() == np_[type],
               "Insufficient number of parameters, " << params.Size()
               << ", for profile of type: " << type << ".");
}

double PlasmaProfile::Eval(ElementTransformation &T,
                           const IntegrationPoint &ip)
{
   if (type_ != CONSTANT)
   {
      T.Transform(ip, x_);
   }

   switch (type_)
   {
      case CONSTANT:
         // cout << "returning const  " << p_[0] << endl;
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
