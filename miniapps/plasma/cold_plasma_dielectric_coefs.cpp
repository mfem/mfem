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

#include "cold_plasma_dielectric_coefs.hpp"

using namespace std;
namespace mfem
{
using namespace miniapps;

namespace plasma
{

double R_cold_plasma(double omega,
                     double Bmag,
                     const Vector & number,
                     const Vector & charge,
                     const Vector & mass)
{
   double val = 1.0;
   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      double w_c = omega_c(Bmag, q, m);
      double w_p = omega_p(n, q, m);
      val -= w_p * w_p / (omega * (omega + w_c));
   }
   return val;
}

double L_cold_plasma(double omega,
                     double Bmag,
                     const Vector & number,
                     const Vector & charge,
                     const Vector & mass)
{
   double val = 1.0;
   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      double w_c = omega_c(Bmag, q, m);
      double w_p = omega_p(n, q, m);
      val -= w_p * w_p / (omega * (omega - w_c));
   }
   return val;
}

double S_cold_plasma(double omega,
                     double Bmag,
                     const Vector & number,
                     const Vector & charge,
                     const Vector & mass)
{
   double val = 1.0;
   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      double w_c = omega_c(Bmag, q, m);
      double w_p = omega_p(n, q, m);
      val -= w_p * w_p / (omega * omega - w_c * w_c);
   }
   return val;
}

double D_cold_plasma(double omega,
                     double Bmag,
                     const Vector & number,
                     const Vector & charge,
                     const Vector & mass)
{
   double val = 0.0;
   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      double w_c = omega_c(Bmag, q, m);
      double w_p = omega_p(n, q, m);
      val += w_p * w_p * w_c / (omega *(omega * omega - w_c * w_c));
   }
   return val;
}

double P_cold_plasma(double omega,
                     const Vector & number,
                     const Vector & charge,
                     const Vector & mass)
{
   double val = 1.0;
   for (int i=0; i<number.Size(); i++)
   {
      double n = number[i];
      double q = charge[i];
      double m = mass[i];
      double w_p = omega_p(n, q, m);
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
   // double Te = temperature_vals[0];
   // double Ti1 = temperature_vals[1];
   // double Ti2 = temperature_vals[2];
   // double vTe = sqrt(2.0 * Te / me_);
   // double debye_length = sqrt((epsilon0_ * Te) / (ne * pow(qe, 2)));
   // double b90_1 = (qe * qi1)/(4.0 * M_PI * epsilon0_ * me_ * pow(vTe, 2)),
   //        b90_2 = (qe * qi2)/(4.0 * M_PI * epsilon0_ * me_ * pow(vTe, 2));
   // double nu_ei1 = (pow(qe * qi1, 2) * ni1 * log(debye_length / b90_1)) /
   //   (4.0 * M_PI * pow(epsilon0_ * me_, 2) * pow(Te, 3.0/2.0));
   // double nu_ei2 = (pow(qe * qi2, 2) * ni2 * log(debye_length / b90_2)) /
   //   (4.0 * M_PI * pow(epsilon0_ * me_, 2) * pow(Te, 3.0/2.0));

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
                                   const ParFiniteElementSpace & L2FESpace,
                                   double omega,
                                   const Vector & charges,
                                   const Vector & masses,
                                   bool realPart)
   : MatrixCoefficient(3),
     B_(B),
     density_(density),
     L2FESpace_(L2FESpace),
     omega_(omega),
     realPart_(realPart),
     charges_(charges),
     masses_(masses)
{
   density_vals_.SetSize(charges_.Size());
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

   if (realPart_)
   {
      double S = S_cold_plasma(omega_, Bmag, density_vals_, charges_, masses_);
      double P = P_cold_plasma(omega_, density_vals_, charges_, masses_);

      epsilon(0,0) =  (P - S) * pow(sin(ph), 2) * pow(cos(th), 2) + S;
      epsilon(1,1) =  (P - S) * pow(cos(ph), 2) + S;
      epsilon(2,2) =  (P - S) * pow(sin(ph), 2) * pow(sin(th), 2) + S;
      epsilon(0,1) = -(P - S) * cos(ph) * cos(th) * sin(ph);
      epsilon(0,2) =  (P - S) * pow(sin(ph), 2) * sin(th) * cos(th);
      epsilon(1,2) = -(P - S) * sin(th) * cos(ph) * sin(ph);
      epsilon(1,0) = epsilon(0,1);
      epsilon(2,1) = epsilon(1,2);
      epsilon(0,2) = epsilon(2,0);
   }
   else
   {
      double D = D_cold_plasma(omega_, Bmag, density_vals_, charges_, masses_);

      epsilon(0,0) =  0.0;
      epsilon(1,1) =  0.0;
      epsilon(2,2) =  0.0;
      epsilon(0,1) = -D * sin(th) * sin(ph);
      epsilon(0,2) = -D * cos(ph);
      epsilon(1,2) = -D * cos(th) * sin(ph);
      epsilon(1,0) = -epsilon(0,1);
      epsilon(2,1) = -epsilon(1,2);
      epsilon(0,2) = -epsilon(2,0);
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
   const ParFiniteElementSpace & L2FESpace,
   double omega,
   const Vector & charges,
   const Vector & masses)
   : MatrixCoefficient(3),
     B_(B),
     density_(density),
     L2FESpace_(L2FESpace),
     omega_(omega),
     charges_(charges),
     masses_(masses)
{
   density_vals_.SetSize(charges_.Size());
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

   double S = S_cold_plasma(omega_, Bmag, density_vals_, charges_, masses_);
   double P = P_cold_plasma(omega_, density_vals_, charges_, masses_);
   double D = D_cold_plasma(omega_, Bmag, density_vals_, charges_, masses_);

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

   epsilon *= epsilon0_;
}

} // namespace plasma

} // namespace mfem
