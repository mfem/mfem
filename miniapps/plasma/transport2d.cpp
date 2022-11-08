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
//
// ./transport -s 12 -v 1 -vs 5 -tol 5e-3 -tf 4
//

// Annular benchmark test
// mpirun -np 10 ./transport2d -v 1 -vs 1 -epus -tol 1e2 -tf 1 -op 16 -l 1 -m annulus-quad-o3.mesh -p 1 -ni-min 3e19 -ni-max 3e19 -Te-min 11 -Te-max 440 -dt 1e-2 -visit
// mpirun -np 10 ./transport2d -v 1 -vs 1 -eps -tol 1e-4 -tf 1 -op 16 -l 1 -m annulus-quad-o3.mesh -p 1 -ni-min 3e19 -ni-max 3e19 -Te-min 11 -Te-max 440 -dt 1e-2 -visit

// The following leads to an AMR-related crash (DBC n_i=3e19 on bdr 2)
// mpirun -np 10 ./transport2d -vs 1 -epus -tf 1 -op 2 -l 1 -m annulus-quad-o3.mesh -p 1 -nn-min 1e15 -nn-max 1e15 -nn-exp 2e15 -ni-min 3e19 -ni-max 3e19 -Ti-min 10 -Ti-max 10 -Te-min 200 -Te-max 200 -dt 1e-9 -visit -bc transport2d_bc.inp

#include "mfem.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

#include <float.h>

#include "../common/pfem_extras.hpp"
#include "transport_solver.hpp"
#include "g_eqdsk_data.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::plasma;
using namespace mfem::plasma::transport;

//int problem_;

// Axisymmetry parameters
// If axis_sym_ is TRUE then cylindrical symmetry is assumed.
// The mesh coordinates are taken to be (x,y).  We overlay an (r,z) coordinate
// system where r = 0 corresponds to x = axis_x_ (or r = x - axis_x),
// and z = y.
//static bool axis_sym_ = true;
//static double axis_x_ = -0.8;

// Equation constant parameters.
/*
int num_species_ = -1;
int num_equations_ = -1;
const double specific_heat_ratio_ = 1.4;
const double gas_constant_ = 1.0;
*/
// Scalar coefficient for diffusion of momentum
//static double diffusion_constant_ = 0.1;
static int    prob_ = 4;

static double nn_max_ = 1.0e15;
static double nn_min_ = 0.9e15;
static double nn_exp_ = 0.0;

static double ni_max_ = 1.0e18;
static double ni_min_ = 1.0e16;
static double ni_exp_ = 0.0;

static double Ti_max_ = 10.0;
static double Ti_min_ =  1.0;
static double Ti_exp_ =  0.0;

static double Te_max_ = 440.0;
static double Te_min_ =  10.0;
static double Te_exp_ =   0.0;

static double Tot_B_max_ = 5.0; // Maximum of total B field
static double Pol_B_max_ = 0.5; // Maximum of poloidal B field
static double v_max_ = 1e3;

void set_mass_defaults(double &m_amu, double &m_kg, double m_def_amu);

// Maximum characteristic speed (updated by integrators)
//static double max_char_speed_;

// Background fields and initial conditions
//static int prob_ = 4;
//static int gamma_ = 10;
//static double alpha_ = NAN;
//static double chi_max_ratio_ = 1.0;
//static double chi_min_ratio_ = 1.0;
/*
void ChiFunc(const Vector &x, DenseMatrix &M)
{
   M.SetSize(2);

   switch (prob_)
   {
      case 1:
      {
         double cx = cos(M_PI * x[0]);
         double cy = cos(M_PI * x[1]);
         double sx = sin(M_PI * x[0]);
         double sy = sin(M_PI * x[1]);

         double den = cx * cx * sy * sy + sx * sx * cy * cy;

         M(0,0) = chi_max_ratio_ * sx * sx * cy * cy + sy * sy * cx * cx;
         M(1,1) = chi_max_ratio_ * sy * sy * cx * cx + sx * sx * cy * cy;

         M(0,1) = (1.0 - chi_max_ratio_) * cx * cy * sx * sy;
         M(1,0) = M(0,1);

         M *= 1.0 / den;
      }
      break;
      case 2:
      case 4:
      {
         double a = 0.4;
         double b = 0.8;

         double den = pow(b * b * x[0], 2) + pow(a * a * x[1], 2);

         M(0,0) = chi_max_ratio_ * pow(a * a * x[1], 2) + pow(b * b * x[0], 2);
         M(1,1) = chi_max_ratio_ * pow(b * b * x[0], 2) + pow(a * a * x[1], 2);

         M(0,1) = (1.0 - chi_max_ratio_) * pow(a * b, 2) * x[0] * x[1];
         M(1,0) = M(0,1);

         M *= 1.0e-2 / den;
      }
      break;
      case 3:
      {
         double ca = cos(alpha_);
         double sa = sin(alpha_);

         M(0,0) = 1.0 + (chi_max_ratio_ - 1.0) * ca * ca;
         M(1,1) = 1.0 + (chi_max_ratio_ - 1.0) * sa * sa;

         M(0,1) = (chi_max_ratio_ - 1.0) * ca * sa;
         M(1,0) = (chi_max_ratio_ - 1.0) * ca * sa;
      }
      break;
   }
}
*/
/*
double QFunc(const Vector &x, double t)
{
   double a = 0.4;
   double b = 0.8;

   double r = pow(x[0] / a, 2) + pow(x[1] / b, 2);
   double e = exp(-0.25 * t * M_PI * M_PI / (a * b) );

   if ( r == 0.0 )
      return 0.25 * M_PI * M_PI *
             ( (1.0 - e) * ( pow(a, -2) + pow(b, -2) ) + e / (a * b));

   return ( M_PI / r ) *
          ( 0.25 * M_PI * pow(a * b, -4) *
            ( pow(b * b * x[0],2) + pow(a * a * x[1], 2) +
              (a - b) * (b * pow(b * x[0], 2) - a * pow(a*x[1],2)) * e) *
            cos(0.5 * M_PI * sqrt(r)) +
            0.5 * pow(a * b, -2) * (x * x) * (1.0 - e) *
            sin(0.5 * M_PI * sqrt(r)) / sqrt(r)
          );
}
*/
/*
double TFunc(const Vector &x, double t)
{
   double a = 0.4;
   double b = 0.8;

   double r = pow(x[0] / a, 2) + pow(x[1] / b, 2);

   return T_min_ + (T_max_ - T_min_) * cos(0.5 * M_PI * sqrt(r));
}
*/
double nnFunc(const Vector &x, double t)
{
   switch (prob_)
   {
      case 1:
      {
         double ra = 0.4;
         double rb = 0.64;

         double r = (sqrt(pow(x[0], 2) + pow(x[1], 2)) - ra) / (rb - ra);
         double rs = pow(x[0] - 0.5 * (ra + rb), 2) + pow(x[1], 2);
         return nn_max_ + (nn_min_ - nn_max_) * (0.5 + 0.5 * cos(M_PI * r)) +
                0.5 * nn_exp_ * exp(-400.0 * rs);
      }
      case 4:
      {
         double a = 0.4;
         double b = 0.8;

         double r = pow(x[0] / a, 2) + pow(x[1] / b, 2);
         double rs = pow(x[0] + 0.5 * a, 2) + pow(x[1] - 0.5 * b, 2);
         return nn_max_ +
                (nn_min_ - nn_max_) * (0.5 + 0.5 * cos(M_PI * sqrt(r)) -
                                       0.75 * exp(-200.0 * rs));
      }
      default:
         return nn_max_;
   }
}

double niFunc(const Vector &x, double t)
{
   switch (prob_)
   {
      case 1:
      {
         double ra = 0.4;
         double rb = 0.64;

         double r = (sqrt(pow(x[0], 2) + pow(x[1], 2)) - ra) / (rb - ra);
         double rs = pow(x[0] - 0.5 * (ra + rb), 2) + pow(x[1], 2);
         return ni_min_ + (ni_max_ - ni_min_) * (0.5 + 0.5 * cos(M_PI * r)) +
                0.5 * ni_exp_ * exp(-400.0 * rs);
      }
      case 4:
      {
         double a = 0.4;
         double b = 0.8;

         double r = pow(x[0] / a, 2) + pow(x[1] / b, 2);
         double rs = pow(x[0] - 0.5 * a, 2) + pow(x[1] + 0.5 * b, 2);
         return ni_min_ +
                (ni_max_ - ni_min_) * (0.5 + 0.5 * cos(M_PI * sqrt(r)) +
                                       0.5 * exp(-200.0 * rs));
      }
      default:
         return ni_max_;
   }
}

double TiFunc(const Vector &x, double t)
{
   switch (prob_)
   {
      case 1:
      {
         double ra = 0.4;
         double rb = 0.64;

         double r = (sqrt(pow(x[0], 2) + pow(x[1], 2)) - ra) / (rb - ra);
         double rs = pow(x[0] - 0.5 * (ra + rb), 2) + pow(x[1], 2);
         return Ti_min_ + (Ti_max_ - Ti_min_) * (0.5 + 0.5 * cos(M_PI * r)) +
                0.5 * Ti_exp_ * exp(-400.0 * rs);
      }
      case 4:
      {
         double a = 0.4;
         double b = 0.8;

         double r = pow(x[0] / a, 2) + pow(x[1] / b, 2);
         double rs = pow(x[0] + 0.5 * a, 2) + pow(x[1] + 0.5 * b, 2);
         return Ti_min_ +
                (Ti_max_ - Ti_min_) * (0.5 + 0.5 * cos(M_PI * sqrt(r)) +
                                       0.5 * exp(-200.0 * rs));
      }
      default:
         return Ti_max_;
   }
}

double TeFunc(const Vector &x, double t)
{
   switch (prob_)
   {
      case 1:
      {
         double ra = 0.4;
         double rb = 0.64;

         double r = (sqrt(pow(x[0], 2) + pow(x[1], 2)) - ra) / (rb - ra);
         double rs = pow(x[0] - 0.5 * (ra + rb), 2) + pow(x[1], 2);
         return Te_min_ + (Te_max_ - Te_min_) * (0.5 + 0.5 * cos(M_PI * r)) +
                0.5 * Te_exp_ * exp(-400.0 * rs);
      }
      case 4:
      {
         double a = 0.4;
         double b = 0.8;

         double r = pow(x[0] / a, 2) + pow(x[1] / b, 2);
         double rs = pow(x[0] - 0.5 * a, 2) + pow(x[1] - 0.5 * b, 2);
         return Te_min_ +
                (Te_max_ - Te_min_) * (0.5 + 0.5 * cos(M_PI * sqrt(r))) +
                0.5 * Te_exp_ * exp(-400.0 * rs);
      }
      default:
         return Te_max_;
   }
}

void bHatFunc(const Vector &x, Vector &B)
{
   B.SetSize(2);

   switch (prob_)
   {
      case 0:
      {
         double cx = cos(M_PI * x[0]);
         double cy = cos(M_PI * x[1]);
         double sx = sin(M_PI * x[0]);
         double sy = sin(M_PI * x[1]);

         B[0] =  sx * cy;
         B[1] = -cx * sy;

         double den = sqrt(B*B);

         if (den < 1e-8)
         {
            B *= 0.0;
         }
         else
         {
            B *= 1.0 / den;
         }

         return;
      }
      case 1:
      {
         double r2 = pow(x[0], 2) + pow(x[1], 2);

         if ( r2 < 1.0e-4 )
         {
            B = 0.0;
            return;
         }

         double den = sqrt(pow(x[0], 2) + pow(x[1], 2));

         B[0] = x[1] / den;
         B[1] = -x[0] / den;
         return;
      }
      case 4:
      {
         double a = 0.4;
         double b = 0.8;
         double r2 = pow(x[0] / a, 2) + pow(x[1] / b, 2);

         if ( r2 < 1.0e-4 * a * b)
         {
            B = 0.0;
            return;
         }

         double den = sqrt(pow(b * b * x[0], 2) + pow(a * a * x[1], 2));

         B[0] = a * a * x[1] / den;
         B[1] = -b * b * x[0] / den;
         return;
      }
      default:
         B = 0.0;
         return;
   }
}

void PolBFunc(const Vector &x, Vector &B)
{
   B.SetSize(2);

   switch (prob_)
   {
      case 0:
      {
         double cx = cos(M_PI * x[0]);
         double cy = cos(M_PI * x[1]);
         double sx = sin(M_PI * x[0]);
         double sy = sin(M_PI * x[1]);

         B[0] =  sx * cy;
         B[1] = -cx * sy;
      }
      break;
      case 1:
      {
         double a = 0.4;
         B[0] = -x[1] / a;
         B[1] =  x[0] / a;
      }
      break;
      case 4:
      {
         double a = 0.4;
         double b = 0.8;

         B[0] = -a * x[1] / (b * b);
         B[1] =  x[0] / a;
      }
      break;
      case 5:
      {
         double r = sqrt(x * x);
         B[0] = -x[1] / r;
         B[1] = x[0] / r;
      }
      break;
      default:
         B = 0.0;
   }
   B *= Pol_B_max_;
}

void TotBFunc(const Vector &x, Vector &B)
{
   B.SetSize(3);

   Vector polB(B.GetData(), 2);

   PolBFunc(x, polB);

   B[2] = sqrt(std::max(Tot_B_max_ * Tot_B_max_ - (polB * polB), 0.0));
}

class B3Coefficient : public VectorCoefficient
{
private:
   VectorCoefficient &B2D;

public:
   B3Coefficient(VectorCoefficient &B2) : VectorCoefficient(3), B2D(B2) {}

   void Eval(Vector &B, ElementTransformation &T, const IntegrationPoint &ip)
   {
      B.SetSize(3);

      Vector polB(B.GetData(), 2);

      B2D.Eval(polB, T, ip);

      B[2] = sqrt(Tot_B_max_ * Tot_B_max_ - (polB * polB));
   }
};

void paraFunc(const Vector &x, DenseMatrix &M)
{
   M.SetSize(2);

   double b_pol[2];
   Vector B(b_pol, 2);

   PolBFunc(x, B);

   M(0,0) = B(0) * B(0);
   M(0,1) = B(0) * B(1);
   M(1,0) = B(1) * B(0);
   M(1,1) = B(1) * B(1);

   M *= 1.0 / (Pol_B_max_ * Pol_B_max_);
}

void perpFunc(const Vector &x, DenseMatrix &M)
{
   int dim = x.Size();

   paraFunc(x, M);
   M *= -1.0;

   for (int d=0; d<dim; d++) { M(d,d) += 1.0; }
}

double viFunc(const Vector &x)
{
   switch (prob_)
   {
      case 1:
      {
         // double a = 0.4;
         double b = 0.64;

         return -v_max_ * sin(M_PI * (pow(x[0] / b, 2) + pow(x[1] / b, 2)));
      }
      break;
      case 4:
      {
         double a = 0.4;
         double b = 0.8;

         return -v_max_ * sin(M_PI * (pow(x[0] / a, 2) + pow(x[1] / b, 2)));
      }
      default:
         return v_max_;;
   }
}

class Transport2DCoefFactory : public TransportCoefFactory
{
public:
   Transport2DCoefFactory() {}
   Transport2DCoefFactory(const std::vector<std::string> & names,
                          ParGridFunctionArray & pgfa)
      : TransportCoefFactory(names, pgfa) {}

   Coefficient * GetScalarCoef(std::string &name, std::istream &input);
   VectorCoefficient * GetVectorCoef(std::string &name, std::istream &input);
};

/** Given the electron temperature in eV this coefficient returns an
    approzximation to the expected ionization rate in m^3/s.
*/
/*
class ApproxIonizationRate : public Coefficient
{
private:
   GridFunction *Te_;

public:
   ApproxIonizationRate(GridFunction &Te) : Te_(&Te) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      double Te2 = pow(Te_->GetValue(T.ElementNo, ip), 2);

      return 3.0e-16 * Te2 / (3.0 + 0.01 * Te2);
   }
};
*/
class MomentumDiffusionCoef : public MatrixCoefficient
{
private:

   int    zi_;
   double mi_;
   double lnLam_;

   Coefficient *Di_perp_;
   Coefficient *ni_;
   Coefficient *Ti_;

   mutable DenseMatrix perp_;
   mutable Vector x_;

public:
   MomentumDiffusionCoef(int dim, int zi, double mi, Coefficient &DiPerpCoef,
                         Coefficient &niCoef, Coefficient &TiCoef)
      : MatrixCoefficient(dim),
        zi_(zi), mi_(mi), lnLam_(17.0),
        Di_perp_(&DiPerpCoef),
        ni_(&niCoef), Ti_(&TiCoef),
        perp_(dim), x_(dim) {}

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      K.SetSize(width);

      T.Transform(ip, x_);

      double ni = ni_->Eval(T, ip);
      double Ti = Ti_->Eval(T, ip);

      double tau = tau_i(mi_, zi_, ni, Ti, lnLam_);

      paraFunc(x_, K);
      K *= 0.96 * ni * Ti * J_per_eV_ * tau;

      perpFunc(x_, perp_);

      double Di_perp = Di_perp_->Eval(T, ip);

      K.Add(mi_ * kg_per_amu_ * ni * Di_perp, perp_);
   }
};

class ThermalDiffusionCoef : public MatrixCoefficient
{
private:

   int zi_;
   double a_;
   double m_;
   double lnLam_;

   Coefficient *X_perp_;
   Coefficient *n_;
   Coefficient *T_;
   Coefficient *ni_;

   mutable DenseMatrix perp_;
   mutable Vector x_;

public:
   ThermalDiffusionCoef(int dim, int ion_charge, double mass,
                        Coefficient &XPerpCoef,
                        Coefficient &nCoef, Coefficient &TCoef,
                        Coefficient *niCoef = NULL)
      : MatrixCoefficient(dim),
        zi_(ion_charge),
        a_(niCoef ? 3.16 : 3.9),
        m_(mass), lnLam_(17.0),
        X_perp_(&XPerpCoef),
        n_(&nCoef), T_(&TCoef), ni_(niCoef),
        perp_(dim), x_(dim) {}

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      K.SetSize(width);

      double n    = n_->Eval(T, ip);
      double Temp = T_->Eval(T, ip);
      double ni = ni_ ? ni_->Eval(T, ip) : 0.0;

      double tau = ni_ ?
                   tau_e(Temp, zi_, ni, lnLam_) :
                   tau_i(m_, zi_, n, Temp, lnLam_);

      T.Transform(ip, x_);

      paraFunc(x_, K);

      K *= a_ * Temp * J_per_eV_ * tau / ( m_ * kg_per_amu_ );

      perpFunc(x_, perp_);

      double X_perp = X_perp_->Eval(T, ip);

      K.Add(X_perp, perp_);
   }
};
/*
class NormedDifferenceMeasure : public ODEDifferenceMeasure
{
private:
   MPI_Comm comm_;
   const Operator * M_;
   Vector du_;
   Vector Mu_;

public:
   NormedDifferenceMeasure(MPI_Comm comm) : comm_(comm), M_(NULL) {}

   void SetOperator(const Operator & op)
   { M_ = &op; du_.SetSize(M_->Width()); Mu_.SetSize(M_->Height()); }

   double Eval(Vector &u0, Vector &u1)
   {
      M_->Mult(u0, Mu_);
      double nrm0 = InnerProduct(comm_, u0, Mu_);
      add(u1, -1.0, u0, du_);
      M_->Mult(du_, Mu_);
      return sqrt(InnerProduct(comm_, du_, Mu_) / nrm0);
   }
};

class VectorNormedDifferenceMeasure : public ODEDifferenceMeasure
{
private:
   Vector w_;
   Array<NormedDifferenceMeasure*> msr_;

   int size_;
   Vector u0_;
   Vector u1_;

public:
   VectorNormedDifferenceMeasure(MPI_Comm comm, Vector & weights)
      : w_(weights),
        msr_(w_.Size()),
        size_(0),
        u0_(NULL, 0),
        u1_(NULL, 0)
   {
      msr_ = NULL;
      for (int i=0; i<w_.Size(); i++)
      {
         if (w_[i] != 0.0)
         {
            msr_[i] = new NormedDifferenceMeasure(comm);
         }
      }
   }

   void SetOperator(const Operator & op)
   {
      size_ = op.Width();
      for (int i=0; i<w_.Size(); i++)
      {
         if (msr_[i] != NULL)
         {
            msr_[i]->SetOperator(op);
         }
      }
   }

   double Eval(Vector &u0, Vector &u1)
   {
      double m= 0.0;
      for (int i=0; i<w_.Size(); i++)
      {
         if (w_[i] != 0.0)
         {
            u0_.SetDataAndSize(&u0[i * size_], size_);
            u1_.SetDataAndSize(&u1[i * size_], size_);
            m += w_[i] * msr_[i]->Eval(u0_, u1_);
         }
      }
      return m;
   }
};
*/
class VectorRelativeErrorMeasure : public ODERelativeErrorMeasure
{
private:
   Vector w_;
   Vector m_;
   Array<ODERelativeErrorMeasure*> msr_;

   int size_;
   Vector u_, e_;

public:
   VectorRelativeErrorMeasure(MPI_Comm comm, Vector & weights,
                              Vector & magnitudes)
      : w_(weights),
        m_(magnitudes),
        msr_(w_.Size()),
        size_(0),
        u_(NULL, 0),
        e_(NULL, 0)
   {
      msr_ = NULL;
      for (int i=0; i<w_.Size(); i++)
      {
         if (w_[i] != 0.0)
         {
            msr_[i] = new ParMaxAbsRelDiffMeasure(comm, m_[i]);
         }
      }
   }

   ~VectorRelativeErrorMeasure()
   {
      for (int i=0; i<msr_.Size(); i++)
      {
         delete msr_[i];
      }
   }

   double Eval(Vector &u, Vector &e)
   {
      size_ = u.Size() / w_.Size();

      double m = 0.0;
      for (int i=0; i<w_.Size(); i++)
      {
         if (w_[i] != 0.0)
         {
            u_.SetDataAndSize(&u[i * size_], size_);
            e_.SetDataAndSize(&e[i * size_], size_);
            m = std::max(m, w_[i] * msr_[i]->Eval(u_, e_));
         }
      }
      return m;
   }
};

class VectorErrorEstimator : public ErrorEstimator
{
protected:
   ParFiniteElementSpace &err_fes_;
   Vector &w_;

   Array<double> nrm_;
   Array<ErrorEstimator*> est_;

   Vector err_;

public:

   VectorErrorEstimator(ParFiniteElementSpace & err_fes,
                        Vector &weights)
      : err_fes_(err_fes),
        w_(weights),
        nrm_(w_.Size()),
        est_(w_.Size()),
        err_(err_fes_.GetVSize())
   {
      nrm_ = 1.0;
      est_ = NULL;
   }

   virtual ~VectorErrorEstimator()
   {
      for (int i=0; i<est_.Size(); i++)
      {
         delete est_[i];
      }
   }

   virtual void Reset()
   {
      for (int i=0; i<est_.Size(); i++)
      {
         if (est_[i] != NULL)
         {
            est_[i]->Reset();
         }
      }
   }

   virtual const Vector & GetLocalErrors()
   {
      err_.SetSize(err_fes_.GetVSize());
      err_ = 0.0;

      for (int i=0; i<w_.Size(); i++)
      {
         if (w_[i] != 0.0)
         {
            const Vector & err_k = est_[i]->GetLocalErrors();
            // cout << i << " " << err_.Size() << " " << err_k.Size() << " " << err_k.Norml2()
            //     << " " << w_[i] << " " << nrm_[i] << endl;
            err_.Add(w_[i] / nrm_[i], err_k);
         }
      }

      return err_;
   }
};

class VectorL2ZZErrorEstimator : public VectorErrorEstimator
{
private:
   ParGridFunctionArray &pgf_;
   ParFiniteElementSpace &flux_fes_;
   ParFiniteElementSpace &sm_flux_fes_;
   Array<Coefficient*> &dCoef_;
   Array<MatrixCoefficient*> &DCoef_;
   Array<BilinearFormIntegrator*> integ_;

public:
   VectorL2ZZErrorEstimator(ParGridFunctionArray & pgf,
                            ParFiniteElementSpace & err_fes,
                            ParFiniteElementSpace & flux_fes,
                            ParFiniteElementSpace & sm_flux_fes,
                            Vector & weights,
                            Array<Coefficient*> & dCoef,
                            Array<MatrixCoefficient*> & DCoef)
      : VectorErrorEstimator(err_fes, weights),
        pgf_(pgf),
        flux_fes_(flux_fes),
        sm_flux_fes_(sm_flux_fes),
        dCoef_(dCoef),
        DCoef_(DCoef),
        integ_(w_.Size())
   {
      integ_ = NULL;

      for (int i=0; i<w_.Size(); i++)
      {
         if (w_[i] != 0.0)
         {
            double loc_nrm = pgf_[i]->Normlinf();
            MPI_Allreduce(&loc_nrm, &nrm_[i], 1, MPI_DOUBLE, MPI_MAX,
                          err_fes_.GetComm());
            if (nrm_[i] == 0.0) { nrm_[i] = 1.0; }

            if (dCoef_[i] != NULL)
            {
               integ_[i] = new DiffusionIntegrator(*dCoef_[i]);
            }
            else if (DCoef_[i] != NULL)
            {
               integ_[i] = new DiffusionIntegrator(*DCoef_[i]);
            }
            else
            {
               integ_[i] = new DiffusionIntegrator();
            }
            est_[i] = new L2ZienkiewiczZhuEstimator(*integ_[i], *pgf_[i],
                                                    flux_fes_, sm_flux_fes_);
         }
      }
   }

   virtual ~VectorL2ZZErrorEstimator()
   {
      for (int i=0; i<integ_.Size(); i++)
      {
         delete integ_[i];
      }
   }
};

class VectorLpErrorEstimator : public VectorErrorEstimator
{
private:
   Array<Coefficient*> &coef_;
   ParGridFunctionArray &pgf_;

public:
   VectorLpErrorEstimator(int p, Array<Coefficient*> & coef,
                          ParGridFunctionArray & pgf,
                          ParFiniteElementSpace & err_fes,
                          Vector & weights)
      : VectorErrorEstimator(err_fes, weights),
        coef_(coef),
        pgf_(pgf)
   {
      for (int i=0; i<w_.Size(); i++)
      {
         if (w_[i] != 0.0)
         {
            double loc_nrm = pgf_[i]->Normlinf();
            MPI_Allreduce(&loc_nrm, &nrm_[i], 1, MPI_DOUBLE, MPI_MAX,
                          err_fes_.GetComm());
            if (nrm_[i] == 0.0) { nrm_[i] = 1.0; }

            est_[i] = new LpErrorEstimator(p, *coef_[i], *pgf_[i]);
         }
      }
   }
};

Mesh * BuildSol1DMesh(const Vector &sol1d);

void ErrorEstRange(ErrorEstimator & est, double &min_err, double &max_err);

// Initial condition
void AdaptInitialMesh(MPI_Session &mpi,
                      ParMesh &pmesh, ParFiniteElementSpace &err_fespace,
                      ParFiniteElementSpace &fespace,
                      ParFiniteElementSpace &vfespace,
                      ParFiniteElementSpace &ffespace,
                      ParGridFunctionArray & gf, Array<Coefficient*> &coef,
                      Vector &weights,
                      int p, double tol, bool visualization = false);

void InitialCondition(const Vector &x, Vector &y);

void record_cmd_line(int argc, char *argv[]);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi(argc, argv);
   if (!mpi.Root()) { mfem::out.Disable(); mfem::err.Disable(); }

   if (mpi.Root()) { record_cmd_line(argc, argv); }

   SolverParams ttol;
   ttol.lin_abs_tol = 1e-10;
   ttol.lin_rel_tol = 1e-8;
   ttol.lin_max_iter = 1000;
   ttol.lin_log_lvl = 3;

   ttol.newt_abs_tol = 1e-8;
   ttol.newt_rel_tol = 1e-6;
   ttol.newt_max_iter = 10;
   ttol.newt_log_lvl = 1;

   ttol.ss_abs_tol = 0.0;
   ttol.ss_rel_tol = -1.0;

   ttol.prec.type    = 1;
   ttol.prec.log_lvl = 0;

   // 2. Parse command-line options.
   // problem_ = 1;
   const char *mesh_file = "ellipse_origin_h0pt0625_o3.mesh";
   const char *bc_file = "";
   const char *ic_file = "";
   const char *ec_file = "";
   const char *es_file = "";
   const char *eqdsk_file = "";
   Vector mesh_sol1d;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int nc_limit = 3;         // maximum level of hanging nodes
   double max_elem_error = -1.0;
   double hysteresis = 0.25; // derefinement safety coefficient
   int order = 3;

   DGParams dg;
   dg.sigma = -1.0;
   dg.kappa = -1.0;

   std::vector<ArtViscParams> av(5);
   av[1].log_width    = 3.0;
   av[1].art_visc_amp = 1.0;
   av[1].osc_thresh   = 1.0e19;
   av[2].log_width    = 3.0;
   av[2].art_visc_amp = 1.0;
   av[2].osc_thresh   = 1.0e4;

   int ode_solver_type = 1;
   int ode_limiter_type = 2;
   int logging = 0;
   bool     imex = false;
   bool ode_epus = false;
   bool      amr = true;
   int    op_flag = 31;
   double tol_ode = 1e-3;
   double rej_ode = 1.2;
   double kP_acc = 0.0;
   double kI_acc = 0.6;
   double kD_acc = 0.0;
   double kI_rej = 0.6;
   double lim_a = 0.95;
   double lim_b = 1.05;
   double lim_max = 2.0;

   double tol_init = 1e-5;
   double t_init = 0.0;
   double t_min = 0.0;
   double t_final = -1.0;
   double dt = -0.01;
   double dt_floor = 1e-10;
   // double dt_rel_tol = 0.1;
   double cfl = 0.3;

   Array<int> term_flags;
   Array<int> vis_flags;
   const char *device_config = "cpu";
   bool check_grad = false;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 10;

   PlasmaParams plasma;
   plasma.m_n_amu = -1.0;
   plasma.m_n_kg  = -1.0;
   plasma.m_i_amu = -1.0;
   plasma.m_i_kg  = -1.0;
   plasma.v_n_avg_m_per_s = 0.0;
   plasma.v_n_bar_m_per_s = 0.0;
   plasma.T_n_eV  =  3.0;   // (eV)
   plasma.z_i = 1;          // ion charge number
   /*
   int      ion_charge = 1;
   double     ion_mass = 2.01410178; // (amu)
   double neutral_mass = 2.01410178; // (amu)
   double neutral_temp = 3.0;        // (eV)
   */

   double      Di_perp = 1.0;        // Ion perp diffusion (m^2/s)
   double      Xi_perp = 1.0;        // Ion thermal diffusion (m^2/s)
   double      Xe_perp = 1.0;        // Electron thermal diffusion (m^2/s)

   Vector eqn_weights;
   Vector amr_weights;
   Vector ode_weights;
   Vector field_mags;
   Vector fld_weights;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_sol1d, "-m-sol1d", "--mesh-sol1d",
                  "Build a mesh using the algortihm from sol1d."
                  " Needs three parameters: number of elements, "
                  " width of mesh, and beta");
   args.AddOption(&bc_file, "-bc", "--bc-file",
                  "Boundary condition input file.");
   args.AddOption(&ic_file, "-ic", "--ic-file",
                  "Initial condition input file.");
   args.AddOption(&ec_file, "-ec", "--ec-file",
                  "Equation coefficient input file.");
   args.AddOption(&es_file, "-es", "--es-file",
                  "Exact solution input file.");
   args.AddOption(&eqdsk_file, "-eqdsk", "--eqdsk-file",
                  "G EQDSK input file.");
   args.AddOption(&logging, "-l", "--logging",
                  "Set the logging level.");
   args.AddOption(&op_flag, "-op", "--operator-test",
                  "Bitmask for disabling operators.");
   args.AddOption(&field_mags, "-fld-m","--field-magnitudes",
                  "Expected field magnitudes.");
   args.AddOption(&eqn_weights, "-eqn-w","--equation-weights",
                  "Normalization factors for balancing the coupled equations.");
   args.AddOption(&prob_, "-p", "--problem",
                  "Problem setup to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly before parallel"
                  " partitioning, -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly after parallel"
                  " partitioning.");
   args.AddOption(&amr, "-amr", "--enable-amr", "-no-amr",
                  "--disable-amr",
                  "Enable or disable adaptive mesh refinement.");
   args.AddOption(&max_elem_error, "-e", "--max-err",
                  "Maximum element error");
   args.AddOption(&hysteresis, "-y", "--hysteresis",
                  "Derefinement safety coefficient.");
   args.AddOption(&nc_limit, "-l", "--nc-limit",
                  "Maximum level of hanging nodes.");
   args.AddOption(&amr_weights, "-amr-w","--amr-weights",
                  "Relative importance of fields when computing "
                  "AMR error estimates.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&dg.sigma, "-dgs", "--dg-sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&dg.kappa, "-dgk", "--dg-kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&av[1].log_width, "-avnw", "--av-ni-width",
                  "Shock width parameter, should be positive.");
   args.AddOption(&av[1].art_visc_amp, "-avna", "--av-ni-amp",
                  "Artificial viscosity amplitude, should be positive.");
   args.AddOption(&av[1].osc_thresh, "-avnt", "--av-ni-thresh",
                  "Threshold below which discontinuity sensor is disabled.");
   args.AddOption(&av[2].log_width, "-avvw", "--av-vel-width",
                  "Shock width parameter, should be positive.");
   args.AddOption(&av[2].art_visc_amp, "-avva", "--av-vel-amp",
                  "Artificial viscosity amplitude, should be positive.");
   args.AddOption(&av[2].osc_thresh, "-avvt", "--av-vel-thresh",
                  "Threshold below which discontinuity sensor is disabled.");
   args.AddOption(&ttol.lin_abs_tol, "-latol", "--linear-abs-tolerance",
                  "Absolute tolerance for linear solver.");
   args.AddOption(&ttol.lin_rel_tol, "-lrtol", "--linear-rel-tolerance",
                  "Relative tolerance for linear solver.");
   args.AddOption(&ttol.lin_max_iter, "-lmaxit", "--linear-max-iterations",
                  "Maximum iteration count for linear solver.");
   args.AddOption(&ttol.lin_log_lvl, "-llog", "--linear-logging-level",
                  "Output level for linear solver.");
   args.AddOption(&ttol.newt_abs_tol, "-natol", "--newton-abs-tolerance",
                  "Absolute tolerance for Newton solver.");
   args.AddOption(&ttol.newt_rel_tol, "-nrtol", "--newton-rel-tolerance",
                  "Relative tolerance for Newton solver.");
   args.AddOption(&ttol.newt_max_iter, "-nmaxit", "--newton-max-iterations",
                  "Maximum iteration count for Newton solver.");
   args.AddOption(&ttol.newt_log_lvl, "-nlog", "--newton-logging-level",
                  "Output level for Newton solver.");
   args.AddOption(&ttol.ss_abs_tol, "-satol", "--steady-state-abs-tolerance",
                  "Absolute tolerance for Steady State detection.");
   args.AddOption(&ttol.ss_rel_tol, "-srtol", "--steady-state-rel-tolerance",
                  "Relative tolerance for Steady State detection.");
   args.AddOption(&ttol.prec.type, "-pt", "--prec-type",
                  "Type of preconditioner: 1-AMG, 2-SuperLU");
   args.AddOption(&ttol.prec.log_lvl, "-plog", "--prec-logging-level",
                  "Output level for preconditioner.");
   args.AddOption(&tol_init, "-tol0", "--initial-tolerance",
                  "Error tolerance for initial condition.");
   args.AddOption(&tol_ode, "-tol", "--ode-tolerance",
                  "Difference tolerance for ODE integration.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 0 - Backward Euler (disables adaptive time "
                  "stepping), 1 - SDIRK 212, 2 - SDIRK 534.");
   args.AddOption(&ode_limiter_type, "-sl", "--ode-limiter",
                  "ODE Limiter: 1 - Maximum, 2 - Dead-Zone.");
   args.AddOption(&ode_epus, "-epus", "--err-per-unit-step", "-eps",
                  "--err-per-step",
                  "Select error value used by ODE controller.");
   args.AddOption(&ode_weights, "-ode-w","--ode-weights",
                  "Relative importance of fields when computing "
                  "ODE time step.");
   args.AddOption(&kP_acc, "-kP", "--kP",
                  "Gain for proportional error adjustment.");
   args.AddOption(&kI_acc, "-kI", "--kI",
                  "Gain for integrated error adjustment.");
   args.AddOption(&kD_acc, "-kD", "--kD",
                  "Gain for derivative error adjustment.");
   args.AddOption(&lim_a, "-dza", "--dead-zone-min",
                  "Time step will remain unchanged if scale factor is "
                  "between dza and dzb.");
   args.AddOption(&lim_b, "-dzb", "--dead-zone-max",
                  "Time step will remain unchanged if scale factor is "
                  "between dza and dzb.");
   args.AddOption(&lim_max, "-thm", "--theta-max",
                  "Maximum dt increase factor.");
   args.AddOption(&t_min, "-tmin", "--t-minimum",
                  "Run to t-minimum before checking for steady state.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&dt_floor, "-dt-floor", "--time-step-floor",
                  "Minimum Time step. Time step should not drop below this "
                  "value.");
   // args.AddOption(&dt_rel_tol, "-dttol", "--time-step-tolerance",
   //                "Time step will only be adjusted if the relative difference "
   //                "exceeds dttol.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&plasma.z_i, "-qi", "--ion-charge",
                  "Charge of the species "
                  "(in units of electron charge)");
   args.AddOption(&plasma.m_i_amu, "-mi-amu", "--ion-mass-amu",
                  "Mass of the ion species (in amu)");
   args.AddOption(&plasma.m_n_amu, "-mn-amu", "--neutral-mass-amu",
                  "Mass of the neutral species (in amu)");
   args.AddOption(&plasma.m_i_kg, "-mi-kg", "--ion-mass-kg",
                  "Mass of the ion species (in kilograms)");
   args.AddOption(&plasma.m_n_kg, "-mn-kg", "--neutral-mass-kg",
                  "Mass of the neutral species (in kilograms)");
   args.AddOption(&plasma.T_n_eV, "-Tn", "--neutral-temp",
                  "Temperature of the neutral species (in eV)");
   args.AddOption(&nn_min_, "-nn-min", "--min-neutral-density",
                  "Minimum of inital neutral density");
   args.AddOption(&nn_max_, "-nn-max", "--max-neutral-density",
                  "Maximum of inital neutral density");
   args.AddOption(&nn_exp_, "-nn-exp", "--neutral-density-exp",
                  "Amplitude of inital neutral density gaussian");
   args.AddOption(&ni_min_, "-ni-min", "--min-ion-density",
                  "Minimum of inital ion density");
   args.AddOption(&ni_max_, "-ni-max", "--max-ion-density",
                  "Maximum of inital ion density");
   args.AddOption(&ni_exp_, "-ni-exp", "--ion-density-exp",
                  "Amplitude of inital ion density gaussian");
   args.AddOption(&Ti_min_, "-Ti-min", "--min-ion-temp",
                  "Minimum of inital ion temperature");
   args.AddOption(&Ti_max_, "-Ti-max", "--max-ion-temp",
                  "Maximum of inital ion temperature");
   args.AddOption(&Ti_exp_, "-Ti-exp", "--ion-temp-exp",
                  "Amplitude of inital ion temperature gaussian");
   args.AddOption(&Te_min_, "-Te-min", "--min-electron-temp",
                  "Minimum of inital electron temperature");
   args.AddOption(&Te_max_, "-Te-max", "--max-electron-temp",
                  "Maximum of inital electron temperature");
   args.AddOption(&Te_exp_, "-Te-exp", "--electron-temp-exp",
                  "Amplitude of inital electron temperature gaussian");
   // args.AddOption(&diffusion_constant_, "-nu", "--diffusion-constant",
   //               "Diffusion constant used in momentum equation.");
   args.AddOption(&Tot_B_max_, "-B", "--total-B-magnitude",
                  "");
   args.AddOption(&Pol_B_max_, "-pB", "--poloidal-B-magnitude",
                  "");
   args.AddOption(&v_max_, "-v", "--velocity",
                  "");
   args.AddOption(&Di_perp, "-dip", "--Di-perp",
                  "Cross field ion diffusivity (m^2/s).");
   args.AddOption(&Xi_perp, "-xip", "--Xi-perp",
                  "Cross field ion thermal diffusivity (m^2/s).");
   args.AddOption(&Xe_perp, "-xep", "--Xe-perp",
                  "Cross field electron thermal diffusivity (m^2/s).");
   /*
   args.AddOption(&chi_max_ratio_, "-chi-max", "--chi-max-ratio",
                  "Ratio of chi_max_parallel/chi_perp.");
   args.AddOption(&chi_min_ratio_, "-chi-min", "--chi-min-ratio",
                  "Ratio of chi_min_parallel/chi_perp.");
   */
   args.AddOption(&term_flags, "-term-flags", "--equation-term-flags",
                  "Detailed control of terms appearing in each equation.");
   args.AddOption(&vis_flags, "-vis-flags", "--visualization-flags",
                  "");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&check_grad, "-check-grad", "--check-gradient",
                  "-no-check-grad",
                  "--no-check-gradient",
                  "Verify that the gradient returned by DGTransportTDO "
                  "is valid then exit.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");

   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }

   set_mass_defaults(plasma.m_n_amu, plasma.m_n_kg, 2.01410178);
   set_mass_defaults(plasma.m_i_amu, plasma.m_i_kg, 2.01410178);

   // Reset average neutral speed based on new mass and temperature of neutrals
   plasma.v_n_bar_m_per_s = sqrt(8.0 * plasma.T_n_eV * J_per_eV_ /
                                 (M_PI * plasma.m_n_kg));

   if (dg.kappa < 0.0)
   {
      dg.kappa = (double)(order+1)*(order+1);
   }
   if (op_flag < 0) { op_flag = 0; }
   /*
   if (ode_exp_solver_type < 0)
   {
      ode_exp_solver_type = ode_split_solver_type;
   }
   if (ode_imp_solver_type < 0)
   {
      ode_imp_solver_type = ode_split_solver_type;
   }
   */
   imex = ode_solver_type < 10;

   if (field_mags.Size() != 5)
   {
      field_mags.SetSize(5);
      field_mags[     NEUTRAL_DENSITY] = 1e15; // n_n ~ 1e15
      field_mags[         ION_DENSITY] = 1e19; // n_i ~ 1e19
      field_mags[   ION_PARA_VELOCITY] = 1e3;  // v_i ~ 1e3
      field_mags[     ION_TEMPERATURE] = 1e1;  // T_i ~ 1e1
      field_mags[ELECTRON_TEMPERATURE] = 1e2;  // T_e ~ 1e2
   }
   fld_weights.SetSize(field_mags.Size());
   for (int i=0; i<field_mags.Size(); i++)
   {
      fld_weights[i] = 1.0 / field_mags[i];
   }

   if (eqn_weights.Size() != 5)
   {
      eqn_weights.SetSize(5);
      eqn_weights[     NEUTRAL_DENSITY] = 1e-15; // n_n ~ 1e15
      eqn_weights[         ION_DENSITY] = 1e-19; // n_i ~ 1e19
      eqn_weights[   ION_PARA_VELOCITY] = 1e+05; // mom_i ~ 1e-27 * 1e19 * 1e3
      eqn_weights[     ION_TEMPERATURE] = 1e+02; // eng_i ~ 1e-27 * 1e19 * 1e6
      eqn_weights[ELECTRON_TEMPERATURE] = 1e+02; // eng_e ~ 1e-27 * 1e19 * 1e6
   }

   if (amr_weights.Size() != 5)
   {
      amr_weights.SetSize(5);
      amr_weights = 1.0;
   }

   if (ode_weights.Size() != 5)
   {
      ode_weights.SetSize(5);
      ode_weights = 1.0;
   }

   if (term_flags.Size() != 5)
   {
      term_flags.SetSize(5);
      term_flags = -1; // Turn on default equation terms
   }
   if (vis_flags.Size() == 5)
   {
      Array<int> old_vis_flags = vis_flags;
      vis_flags.SetSize(6);
      for (int i=0; i<5; i++)
      {
         vis_flags[i] = old_vis_flags[i];
      }
      vis_flags[5] = -1;
   }
   if (vis_flags.Size() != 6)
   {
      vis_flags.SetSize(6);
      vis_flags = -1; // Turn on default visualization fields
   }

   if (t_final < 0.0)
   {
      if (strcmp(mesh_file, "../data/periodic-hexagon.mesh") == 0)
      {
         t_final = 3.0;
      }
      else if (strcmp(mesh_file, "../data/periodic-square.mesh") == 0)
      {
         t_final = 2.0;
      }
      else
      {
         t_final = 1.0;
      }
   }
   if (mpi.Root()) { args.PrintOptions(cout); }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (mpi.Root()) { device.Print(); }

   G_EQDSK_Data *eqdsk = NULL;
   {
      named_ifgzstream ieqdsk(eqdsk_file);
      if (ieqdsk)
      {
         eqdsk = new G_EQDSK_Data(ieqdsk);
         if (mpi.Root() )
         {
            eqdsk->PrintInfo();
         }
      }
   }

   // 3. Read the mesh from the given mesh file. This example requires a 2D
   //    periodic mesh, such as ../data/periodic-square.mesh.
   Mesh *mesh = (mesh_sol1d.Size() == 0) ? new Mesh(mesh_file, 1, 1) :
                BuildSol1DMesh(mesh_sol1d);
   const int dim = mesh->Dimension();
   const int sdim = mesh->SpaceDimension();

   MFEM_ASSERT(dim == 2, "This miniapp is specialized to 2D geometries.");

   if (mpi.Root())
   {
      cout << "Number of elements in initial mesh: " << mesh->GetNE() << endl;
   }

   // 4. Refine the serial mesh on all processors to increase the resolution.
   //    Also project a NURBS mesh to a piecewise-quadratic curved mesh. Make
   //    sure that the mesh is non-conforming.
   if (mesh->NURBSext)
   {
      mesh->UniformRefinement();
      mesh->SetCurvature(2);
   }
   mesh->EnsureNCMesh();

   // num_species_   = ion_charges.Size();
   // num_equations_ = (num_species_ + 1) * (dim + 2);

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   if (mpi.Root())
   {
      cout << "Number of elements after serial refinement: "
           << mesh->GetNE() << endl;
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   cout << mpi.WorldRank() << ": Number of elements in parallel mesh: " <<
        pmesh.GetNE() << endl;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }
   cout << mpi.WorldRank() << ": Number of elements after parallel refinement: "
        << pmesh.GetNE() << endl;

   // 7. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   H1_FECollection fec_h1(order, dim, BasisType::GaussLobatto);
   RT_FECollection fec_rt(order-1, dim, BasisType::GaussLobatto);
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   // Finite element space for a scalar (thermodynamic quantity)
   ParFiniteElementSpace fes(&pmesh, &fec);
   ParFiniteElementSpace vfes(&pmesh, &fec, 2);

   // Finite element space for all variables together (full thermodynamic state)
   int num_equations = 5;
   ParFiniteElementSpace ffes(&pmesh, &fec, num_equations, Ordering::byNODES);

   // The solution u has components {particle density, x-velocity,
   // y-velocity, temperature} for each species (species loop is the outermost).
   // These are stored contiguously in the BlockVector u_block.
   Array<int> offsets(num_equations + 1);
   for (int k = 0; k <= num_equations; k++)
   {
      offsets[k] = k * fes.GetNDofs();
   }
   ParGridFunction u(&ffes);
   ParGridFunction du(&ffes); du = 0.0;

   // Density, Velocity, and Energy grid functions on for visualization.
   ParGridFunction  neu_density (&fes, u.GetData());
   ParGridFunction  ion_density (&fes, u.GetData() + offsets[1]);
   ParGridFunction para_velocity(&fes, u.GetData() + offsets[2]);
   ParGridFunction  ion_energy  (&fes, u.GetData() + offsets[3]);
   ParGridFunction elec_energy  (&fes, u.GetData() + offsets[4]);

   ParGridFunctionArray yGF;
   yGF.Append(&neu_density);
   yGF.Append(&ion_density);
   yGF.Append(&para_velocity);
   yGF.Append(&ion_energy);
   yGF.Append(&elec_energy);
   yGF.SetOwner(false);

   ParGridFunctionArray kGF;
   for (int i=0; i<5; i++)
   {
      // kGF.Append(new ParGridFunction(&fes, (double*)NULL));
      kGF.Append(new ParGridFunction(&fes, u.GetData() + offsets[i]));
   }
   kGF.SetOwner(true);

   std::vector<std::string> field_names(5);
   field_names[0] = "neutral_density_gf";
   field_names[1] = "ion_density_gf";
   field_names[2] = "ion_para_velocity_gf";
   field_names[3] = "ion_temperature_gf";
   field_names[4] = "electron_temperature_gf";

   Transport2DCoefFactory coefFact(field_names, yGF);

   if (mpi.Root())
   {
      cout << "Configuring initial conditions" << endl;
   }
   TransportICs ics(5);
   if (strncmp(ic_file,"",1) != 0)
   {
      if (mpi.Root())
      {
         cout << "Reading initial conditions from " << ic_file << endl;
      }
      ifstream icfs(ic_file);
      ics.LoadICs(coefFact, icfs);
   }

   if (mpi.Root())
   {
      cout << "Configuring exact solutions" << endl;
   }
   TransportExactSolutions ess(5);
   if (strncmp(es_file,"",1) != 0)
   {
      if (mpi.Root())
      {
         cout << "Reading exact solutions from " << es_file << endl;
      }
      ifstream esfs(es_file);
      ess.LoadExactSolutions(coefFact, esfs);
   }

   // Adaptively refine mesh to accurately represent a given coefficient
   if (amr)
   {
      ParGridFunctionArray coef_gf(5, &fes);
      Array<Coefficient*> coef(5);

      if (ics[0] == NULL)
      { ics[0] = new FunctionCoefficient(nnFunc); ics.SetOwnership(0, true); }
      if (ics[1] == NULL)
      { ics[1] = new FunctionCoefficient(niFunc); ics.SetOwnership(1, true); }
      if (ics[2] == NULL)
      { ics[2] = new FunctionCoefficient(viFunc); ics.SetOwnership(2, true); }
      if (ics[3] == NULL)
      { ics[3] = new FunctionCoefficient(TiFunc); ics.SetOwnership(3, true); }
      if (ics[4] == NULL)
      { ics[4] = new FunctionCoefficient(TeFunc); ics.SetOwnership(4, true); }
      for (int i=0; i<5; i++) { coef[i] = ics[i]; }

      coef_gf.ProjectCoefficient(coef);

      L2_FECollection fec_l2_o0(0, dim, BasisType::GaussLobatto);
      // Finite element space for a scalar (thermodynamic quantity)
      ParFiniteElementSpace err_fes(&pmesh, &fec_l2_o0);

      AdaptInitialMesh(mpi, pmesh, err_fes, fes, vfes, ffes, coef_gf, coef,
                       amr_weights, 2, tol_init, visualization);

      u.SetSpace(&ffes);
      for (int k = 0; k <= num_equations; k++)
      {
         offsets[k] = k * fes.GetNDofs();
      }
      neu_density.MakeRef(&fes, u, offsets[0]);
      ion_density.MakeRef(&fes, u, offsets[1]);
      para_velocity.MakeRef(&fes, u, offsets[2]);
      ion_energy.MakeRef(&fes, u, offsets[3]);
      elec_energy.MakeRef(&fes, u, offsets[4]);
   }
   ParFiniteElementSpace fes_h1(&pmesh, &fec_h1);
   ParFiniteElementSpace fes_rt(&pmesh, &fec_rt);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(ffes.GetOrdering() == Ordering::byNODES, "");

   HYPRE_Int glob_size_sca = fes.GlobalTrueVSize();
   HYPRE_Int glob_size_tot = ffes.GlobalTrueVSize();
   if (mpi.Root())
   { cout << "Number of unknowns per field: " << glob_size_sca << endl; }
   if (mpi.Root())
   { cout << "Total number of unknowns:     " << glob_size_tot << endl; }

   // 8. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.

   // Determine the minimum element size.
   double hmin;
   if (cfl > 0 && dt < 0.0)
   {
      double my_hmin = pmesh.GetElementSize(0, 1);
      for (int i = 1; i < pmesh.GetNE(); i++)
      {
         my_hmin = min(pmesh.GetElementSize(i, 1), my_hmin);
      }
      // Reduce to find the global minimum element size
      MPI_Allreduce(&my_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());

      double chi_max_ratio = 1e6;
      double dt_diff = hmin * hmin / chi_max_ratio;
      double dt_adv  = hmin / max(v_max_, DBL_MIN);

      if (mpi.Root())
      {
         cout << "Maximum advection time step: " << dt_adv << endl;
         cout << "Maximum diffusion time step: " << dt_diff << endl;
      }

      // dt = cfl * min(dt_diff, dt_adv);
      dt = cfl * dt_adv;
   }

   ODEController ode_controller;
   PIDAdjFactor dt_acc(kP_acc, kI_acc, kD_acc);
   IAdjFactor   dt_rej(kI_rej);

   ODEEmbeddedSolver * ode_solver = NULL;
   BackwardEulerSolver bw_euler;
   switch (ode_solver_type)
   {
      case 0:
         ode_solver = new EmbeddedStandardSolver(bw_euler);
         // Disable adaptive time stepping
         dt_floor = dt;
         lim_max = 1.0;
         break;
      case 1: ode_solver = new SDIRK212Solver; break;
      case 2: ode_solver = new SDIRK534Solver; break;
   }

   ODEStepAdjustmentLimiter * ode_limiter = NULL;
   switch (ode_limiter_type)
   {
      case 1: ode_limiter = new MaxLimiter(lim_max); break;
      case 2: ode_limiter = new DeadZoneLimiter(lim_a, lim_b, lim_max); break;
   }

   ParGridFunction psi(&fes_h1);
   ParGridFunction nxGradPsi_rt(&fes_rt);

   para_velocity = 0.0;

   VectorRelativeErrorMeasure ode_diff_msr(MPI_COMM_WORLD,
                                           ode_weights, field_mags);

   Coefficient *psiCoef = NULL;
   VectorCoefficient *nxGradPsiCoef = NULL;
   if (eqdsk)
   {
      psiCoef = new G_EQDSK_Psi_Coefficient(*eqdsk);
      nxGradPsiCoef = new G_EQDSK_NxGradPsi_Coefficient(*eqdsk);
      psi.ProjectCoefficient(*psiCoef);
      nxGradPsi_rt.ProjectCoefficient(*nxGradPsiCoef);

      char vishost[] = "localhost";
      int  visport   = 19916;
      int Wx = 0, Wy = 0; // window position
      int Ww = 275, Wh = 250; // window size
      int /* Dx = 3,*/ Dy = 25;
      socketstream psi_sock;//(vishost, visport);
      VisualizeField(psi_sock, vishost, visport, psi, "Psi",
                     Wx, Wy + Wh + Dy, Ww, Wh);

      socketstream b_sock_rt;//(vishost, visport);
      VisualizeField(b_sock_rt, vishost, visport, nxGradPsi_rt, "nxGradPsi RT",
                     Wx, Wy + Wh + Dy, Ww, Wh);
   }

   // Coefficients representing primary fields
   GridFunctionCoefficient nnCoef(&neu_density);
   GridFunctionCoefficient niCoef(&ion_density);
   GridFunctionCoefficient viCoef(&para_velocity);
   GridFunctionCoefficient TiCoef(&ion_energy);
   GridFunctionCoefficient TeCoef(&elec_energy);

   // Coefficients representing secondary fields
   ProductCoefficient      neCoef(plasma.z_i, niCoef);
   ConstantCoefficient     vnCoef(sqrt(8.0 * plasma.T_n_eV * J_per_eV_ /
                                       (M_PI * plasma.m_n_kg)));
   GridFunctionCoefficient veCoef(&para_velocity); // TODO: define as vi - J/q

   /*
   VectorCoefficient *B3Coef = NULL;
   if (eqdsk)
   {
      B3Coef = new B3Coefficient(*nxGradPsiCoef);
   }
   else
   {
      B3Coef = new VectorFunctionCoefficient(3, TotBFunc);
   }
   */
   /*
   // Intermediate Coefficients
   VectorFunctionCoefficient bHatCoef(2, bHatFunc);
   MatrixFunctionCoefficient perpCoef(2, perpFunc);
   // ProductCoefficient          mnCoef(ion_mass * kg_per_amu_, niCoef); // ???
   ConstantCoefficient         mnCoef(plasma.m_i * kg_per_amu_);
   ProductCoefficient        nnneCoef(nnCoef, neCoef);
   ApproxIonizationRate        izCoef(TeCoef);
   ConstantCoefficient     DiPerpCoef(Di_perp);
   ConstantCoefficient     XiPerpCoef(Xi_perp);
   ConstantCoefficient     XePerpCoef(Xe_perp);
   ThermalDiffusionCoef        XiCoef(dim, plasma.z_i, plasma.m_i,
                                      XiPerpCoef, niCoef, TiCoef);
   ThermalDiffusionCoef        XeCoef(dim, plasma.z_i, me_u_,
                                      XePerpCoef, neCoef, TeCoef, &niCoef);
   */
   /*
   // Advection Coefficients
   ScalarVectorProductCoefficient   ViCoef(viCoef, bHatCoef);
   ScalarVectorProductCoefficient   VeCoef(veCoef, bHatCoef);
   ScalarVectorProductCoefficient  MomCoef(mnCoef, ViCoef);

   // Diffusion Coefficients
   NeutralDiffusionCoef     DnCoef(neCoef, vnCoef, izCoef);
   IonDiffusionCoef         DiCoef(DiPerpCoef, *B3Coef);
   MomentumDiffusionCoef   EtaCoef(dim, plasma.z_i, plasma.m_i,
                                   DiPerpCoef, niCoef, TiCoef);
   ScalarMatrixProductCoefficient nXiCoef(niCoef, XiCoef);
   ScalarMatrixProductCoefficient nXeCoef(neCoef, XeCoef);

   // Source Coefficients
   IonSourceCoef SiCoef(neCoef, nnCoef, izCoef);
   // ProductCoefficient  SiCoef(nnneCoef, izCoef);
   ProductCoefficient  SnCoef(-1.0, SiCoef);
   ConstantCoefficient SMomCoef(0.0); // TODO: implement momentum source
   ConstantCoefficient QiCoef(0.0);   // TODO: implement ion energy source
   ConstantCoefficient QeCoef(0.0); // TODO: implement electron energy source
   // FunctionCoefficient QCoef(QFunc);
   */
   ConstantCoefficient zeroCoef(0.0);

   // Coefficients for initial conditions
   /*
   FunctionCoefficient nn0Coef(nnFunc);
   FunctionCoefficient ni0Coef(niFunc);
   FunctionCoefficient vi0Coef(viFunc);
   FunctionCoefficient Ti0Coef(TiFunc);
   FunctionCoefficient Te0Coef(TeFunc);

   neu_density.ProjectCoefficient(nn0Coef);
   ion_density.ProjectCoefficient(ni0Coef);
   para_velocity.ProjectCoefficient(vi0Coef);
   ion_energy.ProjectCoefficient(Ti0Coef);
   elec_energy.ProjectCoefficient(Te0Coef);
   */
   for (int i=0; i<5; i++)
   {
      ics[i]->SetTime(t_init);
   }
   neu_density.ProjectCoefficient(*ics[0]);
   ion_density.ProjectCoefficient(*ics[1]);
   para_velocity.ProjectCoefficient(*ics[2]);
   ion_energy.ProjectCoefficient(*ics[3]);
   elec_energy.ProjectCoefficient(*ics[4]);

   ofstream ofserr;

   if (strncmp(es_file,"",1) != 0)
   {
      if (mpi.Root())
      {
         ofserr.open("transport2d_err.out");
         ofserr << t_init;
      }
      for (int i=0; i<5; i++)
      {
         double nrm = yGF[i]->ComputeL2Error(zeroCoef);
         ofserr << '\t' << nrm;

         Coefficient * es = ess[i];
         if (es != NULL)
         {
            es->SetTime(t_init);
            double error = yGF[i]->ComputeL2Error(*es);
            if (mpi.Root())
            {
               ofserr << '\t' << error;
            }
         }
         else
         {
            if (mpi.Root()) { ofserr << '\t' << -1.0; }
         }
      }
      if (mpi.Root()) { ofserr << endl << flush; }
   }

   if (mpi.Root())
   {
      cout << "Configuring boundary conditions" << endl;
   }
   TransportBCs bcs(pmesh.bdr_attributes, 5);
   if (strncmp(bc_file,"",1) != 0)
   {
      if (mpi.Root())
      {
         cout << "Reading boundary conditions from " << bc_file << endl;
      }
      ifstream bcfs(bc_file);
      bcs.LoadBCs(coefFact, bcfs);
   }
   /*
   if (mpi.Root())
   {
      cout << "Configuring source terms" << endl;
   }
   TransportSRCs srcs(5);
   if (strncmp(src_file,"",1) != 0)
   {
      if (mpi.Root())
      {
         cout << "Reading source coefficients from " << src_file << endl;
      }
      ifstream srcfs(src_file);
      srcs.LoadSRCs(coefFact, srcfs);
   }
   */
   if (mpi.Root())
   {
      cout << "Configuring equation coefficients" << endl;
   }
   TransportCoefs eqnCoefs(5);
   if (strncmp(ec_file,"",1) != 0)
   {
      if (mpi.Root())
      {
         cout << "Reading equation coefficients from " << ec_file << endl;
      }
      ifstream ecfs(ec_file);
      eqnCoefs.LoadCoefs(coefFact, ecfs);
   }

   VectorCoefficient *B3Coef = NULL;
   StateVariableVecCoef *svB3Coef = NULL;
   if (eqnCoefs(5).GetVectorCoefficient(
          CommonCoefs::MAGNETIC_FIELD_COEF) != NULL)
   {
      if (mpi.Root())
      {
         cout << "Using B field from " << ec_file << endl;
      }
      B3Coef = eqnCoefs(5).GetVectorCoefficient(
                  CommonCoefs::MAGNETIC_FIELD_COEF);
   }
   else if (eqdsk)
   {
      if (mpi.Root())
      {
         cout << "Using B field from " << eqdsk_file << endl;
      }
      B3Coef = new B3Coefficient(*nxGradPsiCoef);
      svB3Coef = new StateVariableStandardVecCoef(*B3Coef);
      eqnCoefs(5).GetVectorCoefficient(CommonCoefs::MAGNETIC_FIELD_COEF) =
         svB3Coef;
   }
   else
   {
      if (mpi.Root())
      {
         cout << "Using B field from TotBFunc" << endl;
      }
      B3Coef = new VectorFunctionCoefficient(3, TotBFunc);
      svB3Coef = new StateVariableStandardVecCoef(*B3Coef);
      eqnCoefs(5).GetVectorCoefficient(CommonCoefs::MAGNETIC_FIELD_COEF) =
         svB3Coef;
   }

   Array<double> coefNrm(5);
   coefNrm = 1.0;
   /*
   {
      L2_ParFESpace l2_fes(&pmesh, order - 1, dim);
      ParLinearForm Mc(&l2_fes);
      ParGridFunction cHat(&l2_fes);

      ConstantCoefficient oneCoef(1.0);
      cHat.ProjectCoefficient(oneCoef);

      double cMc  = -1.0;
      double cmnc = -1.0;
      double cnic = -1.0;
      double cnec = -1.0;

      {
         ParBilinearForm m(&l2_fes);
         m.AddDomainIntegrator(new MassIntegrator);
         m.Assemble();
         m.Mult(cHat, Mc);
         cMc = Mc(cHat);
      }
      {
    ParBilinearForm m(&l2_fes);
         m.AddDomainIntegrator(new MassIntegrator(mnCoef));
         m.Assemble();
         m.Mult(cHat, Mc);
         cmnc = Mc(cHat);
      }
      {
         ParBilinearForm m(&l2_fes);
         m.AddDomainIntegrator(new MassIntegrator(niCoef));
         m.Assemble();
         m.Mult(cHat, Mc);
         cnic = Mc(cHat);
      }
      {
         ParBilinearForm m(&l2_fes);
         m.AddDomainIntegrator(new MassIntegrator(neCoef));
         m.Assemble();
         m.Mult(cHat, Mc);
         cnec = Mc(cHat);
      }

      double mnAvg = cmnc / cMc;
      double niAvg = cnic / cMc;
      double neAvg = cnec / cMc;

      if (mpi.Root()) { cout << "|mn| = " << mnAvg << endl; }
      if (mpi.Root()) { cout << "|ni| = " << niAvg << endl; }
      if (mpi.Root()) { cout << "|ne| = " << neAvg << endl << endl; }

      ND_ParFESpace nd_fes(&pmesh, order, dim);
      ParLinearForm Mb(&nd_fes);
      ParGridFunction bHat(&nd_fes);
      bHat.ProjectCoefficient(bHatCoef);

      double bMb   = -1.0;
      double bDnb  = -1.0;
      double bDib  = -1.0;
      double bEtab = -1.0;
      double bnXib = -1.0;
      double bnXeb = -1.0;

      {
         ParBilinearForm m(&nd_fes);
         m.AddDomainIntegrator(new VectorFEMassIntegrator);
         m.Assemble();
         m.Mult(bHat, Mb);
         bMb = Mb(bHat);
      }
      {
         ParBilinearForm m(&nd_fes);
         m.AddDomainIntegrator(new VectorFEMassIntegrator(DnCoef));
         m.Assemble();
         m.Mult(bHat, Mb);
         bDnb = Mb(bHat);
      }
      {
         ParBilinearForm m(&nd_fes);
         m.AddDomainIntegrator(new VectorFEMassIntegrator(DiCoef));
         m.Assemble();
         m.Mult(bHat, Mb);
         bDib = Mb(bHat);
      }
      {
         ParBilinearForm m(&nd_fes);
         m.AddDomainIntegrator(new VectorFEMassIntegrator(EtaCoef));
         m.Assemble();
         m.Mult(bHat, Mb);
         bEtab = Mb(bHat);
      }
      {
         ParBilinearForm m(&nd_fes);
         m.AddDomainIntegrator(new VectorFEMassIntegrator(nXiCoef));
         m.Assemble();
         m.Mult(bHat, Mb);
         bnXib = Mb(bHat);
      }
      {
         ParBilinearForm m(&nd_fes);
         m.AddDomainIntegrator(new VectorFEMassIntegrator(nXeCoef));
         m.Assemble();
         m.Mult(bHat, Mb);
         bnXeb = Mb(bHat);
      }
      if (mpi.Root()) { cout << "b.b      = " << bMb << endl; }
      if (mpi.Root()) { cout << "|Dn|     = " << bDnb / bMb << endl; }
      if (mpi.Root()) { cout << "|Di|     = " << bDib / bMb << endl; }
      if (mpi.Root()) { cout << "|Eta|    = " << bEtab / bMb << endl; }
      if (mpi.Root()) { cout << "|nXi|    = " << bnXib / bMb << endl; }
      if (mpi.Root()) { cout << "|nXe|    = " << bnXeb / bMb << endl << endl; }
      if (mpi.Root()) { cout << "|Eta/mn| = " << bEtab / bMb / mnAvg << endl; }
      if (mpi.Root()) { cout << "|nXi/ni| = " << bnXib / bMb / niAvg << endl; }
      if (mpi.Root()) { cout << "|nXe/ne| = " << bnXeb / bMb / neAvg << endl; }

      coefNrm[0] = bDnb / bMb;
      coefNrm[1] = bDib / bMb;
      coefNrm[2] = bEtab / bMb;
      coefNrm[3] = bnXib / bMb;
      coefNrm[4] = bnXeb / bMb;
   }
   */

   DGTransportTDO oper(mpi, dg, av, plasma, ttol, eqn_weights, fld_weights,
                       fes, vfes, ffes, fes_h1,
                       offsets, yGF, kGF,
                       bcs, eqnCoefs, Di_perp, Xi_perp, Xe_perp,
                       term_flags, vis_flags, imex, op_flag, logging);

   oper.SetLogging(max(0, logging - (mpi.Root()? 0 : 1)));

   if (check_grad)
   {
      double f = oper.CheckGradient();
      if (f <= 1.0)
      {
         mfem::out << "Gradient check succeeded with a value of " << f << endl;
      }
      else
      {
         mfem::out << "Gradient check failed with a value of " << f << endl;
      }
      return 0;
   }

   if (visualization)
   {
      oper.InitializeGLVis();
   }

   oper.SetTime(0.0);
   ode_solver->Init(oper);

   ode_controller.Init(*ode_solver, ode_diff_msr,
                       dt_acc, dt_rej, *ode_limiter);

   ode_controller.SetOutputFrequency(vis_steps);
   ode_controller.SetTimeStep(dt);
   ode_controller.SetMinTimeStep(dt_floor);
   ode_controller.SetTolerance(tol_ode);
   ode_controller.SetRejectionLimit(rej_ode);
   if (ode_epus) { ode_controller.SetErrorPerUnitStep(); }

   ofstream ofs_controller;
   if (mpi.Root())
   {
      ofs_controller.open("transport2d_cntrl.dat");
      ode_controller.SetOutput(ofs_controller);
   }

   L2_FECollection fec_l2_o0(0, dim, BasisType::GaussLobatto);
   // Finite element space for a scalar (thermodynamic quantity)
   ParFiniteElementSpace fes_l2_o0(&pmesh, &fec_l2_o0);

   ParGridFunction err_est(&fes_l2_o0, (double *)NULL);

   // 11. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     with L2 projection in the smoothing step to better handle hanging
   //     nodes and parallel partitioning. We need to supply a space for the
   //     discontinuous flux (L2) and a space for the smoothed flux (H(div) is
   //     used here).
   L2_FECollection flux_fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace flux_fes(&pmesh, &flux_fec, sdim);
   RT_FECollection smooth_flux_fec(order-1, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec);
   // Another possible option for the smoothed flux space:
   // H1_FECollection smooth_flux_fec(order, dim);
   // ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec, dim);
   /*
   DiffusionIntegrator integ(nXeCoef);
   L2ZienkiewiczZhuEstimator estimator(integ, elec_energy, flux_fes,
                                       smooth_flux_fes);
   */
   Array<Coefficient*> dCoefs(5);       dCoefs = NULL;
   Array<MatrixCoefficient*> DCoefs(5); DCoefs = NULL;
   /*
   dCoefs[0] = &DnCoef;
   DCoefs[1] = &DiCoef;
   DCoefs[2] = &EtaCoef;
   DCoefs[3] = &nXiCoef;
   DCoefs[4] = &nXeCoef;
   */
   dCoefs[0] = oper.GetDnCoefficient();
   DCoefs[1] = oper.GetDiCoefficient();
   DCoefs[2] = oper.GetEtaCoefficient();
   DCoefs[3] = oper.GetnXiCoefficient();
   DCoefs[4] = oper.GetnXeCoefficient();

   Vector estWeights(5);
   for (int i=0; i<5; i++) { estWeights[i] = amr_weights[i] / coefNrm[i]; }

   VectorL2ZZErrorEstimator estimator(yGF, fes_l2_o0, flux_fes, smooth_flux_fes,
                                      estWeights, dCoefs, DCoefs);

   if (max_elem_error < 0.0)
   {
      /*
       const Vector & init_errors = estimator.GetLocalErrors();

       double loc_max_error = init_errors.Max();
       double loc_min_error = init_errors.Min();

       double glb_max_error = -1.0;
       double glb_min_error = -1.0;

       MPI_Allreduce(&loc_max_error, &glb_max_error, 1,
                     MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
       MPI_Allreduce(&loc_min_error, &glb_min_error, 1,
                     MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      */
      double glb_max_error = -1.0;
      double glb_min_error = -1.0;

      ErrorEstRange(estimator, glb_min_error, glb_max_error);

      if (mpi.Root())
      {
         cout << "Range of error estimates for initial condition: "
              << glb_min_error << " < elem err < " << glb_max_error << endl;
      }

      max_elem_error = glb_max_error;

   }

   // 12. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   // refiner.SetTotalErrorFraction(0.7);
   refiner.SetTotalErrorFraction(0.0); // use purely local threshold
   refiner.SetLocalErrorGoal(max_elem_error);
   refiner.PreferConformingRefinement();
   refiner.SetNCLimit(nc_limit);

   // 12. A derefiner selects groups of elements that can be coarsened to form
   //     a larger element. A conservative enough threshold needs to be set to
   //     prevent derefining elements that would immediately be refined again.
   ThresholdDerefiner derefiner(estimator);
   derefiner.SetThreshold(hysteresis * max_elem_error);
   derefiner.SetNCLimit(nc_limit);

   const int max_dofs = 100000;

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   socketstream eout;
   vector<socketstream> sout(5);
   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 275, Wh = 250; // window size
   int Dx = 3, Dy = 25;

   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Transport2D-Parallel", &pmesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Transport2D-Parallel", &pmesh);
         dc->SetPrecision(precision);
         // To save the mesh using MFEM's parallel mesh format:
         // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
      }

      oper.RegisterDataFields(*dc);
      oper.PrepareDataFields();

      dc->SetCycle(0);
      dc->SetTime(t_init);
      dc->Save();
   }

   int cycle = 0;
   int amr_it = 0;
   int ref_it = 0;
   int reb_it = 0;
   int dref_it = 0;

   double t = t_init;

   if (mpi.Root()) { cout << "\nBegin time stepping at t = " << t << endl; }
   while (t < t_final)
   {
      ode_controller.Run(u, t, t_final);

      if (mpi.Root()) { cout << "Time stepping paused at t = " << t << endl; }

      if (strncmp(es_file,"",1) != 0)
      {
         if (mpi.Root()) { ofserr << t; }
         for (int i=0; i<5; i++)
         {
            double nrm = yGF[i]->ComputeL2Error(zeroCoef);
            ofserr << '\t' << nrm;

            Coefficient * es = ess[i];
            if (es != NULL)
            {
               es->SetTime(t);
               double error = yGF[i]->ComputeL2Error(*es);
               if (mpi.Root())
               {
                  ofserr << '\t' << error;
               }
            }
            else
            {
               if (mpi.Root()) { ofserr << '\t' << -1.0; }
            }
         }
         if (mpi.Root()) { ofserr << endl << flush; }
      }

      bool ss = false;
      if ((ttol.ss_abs_tol > 0.0 || ttol.ss_rel_tol > 0.0) && t > t_min)
      {
         ss = oper.CheckForSteadyState();
         if (ss)
         {
            if (mpi.Root())
            {
               cout << "Steady State solution has been reached" << endl;
            }
         }
      }

      if (visualization)
      {
         ostringstream oss;
         oss << "Neutral Density at time " << t;
         VisualizeField(sout[0], vishost, visport, neu_density, oss.str().c_str(),
                        Wx, Wy + Wh + Dy, Ww, Wh);
      }
      if (visualization)
      {
         ostringstream oss;
         oss << "Ion Density at time " << t;
         VisualizeField(sout[1], vishost, visport, ion_density, oss.str().c_str(),
                        Wx + (Ww + Dx), Wy + Wh + Dy, Ww, Wh);
      }
      if (visualization)
      {
         ostringstream oss;
         oss << "Ion Parallel Velocity at time " << t;
         VisualizeField(sout[2], vishost, visport, para_velocity, oss.str().c_str(),
                        Wx + 2 * (Ww + Dx), Wy + Wh + Dy, Ww, Wh);
      }
      if (visualization)
      {
         ostringstream oss;
         oss << "Ion Temperature at time " << t;
         VisualizeField(sout[3], vishost, visport, ion_energy, oss.str().c_str(),
                        Wx + 3 * (Ww + Dx), Wy + Wh + Dy, Ww, Wh);
      }
      if (visualization)
      {
         ostringstream oss;
         oss << "Electron Temperature at time " << t;
         VisualizeField(sout[4], vishost, visport, elec_energy, oss.str().c_str(),
                        Wx + 4 * (Ww + Dx), Wy + Wh + Dy, Ww, Wh);
      }
      if (visualization)
      {
         oper.DisplayToGLVis();
      }
      if (visit)
      {
         cycle++;
         oper.PrepareDataFields();
         dc->SetCycle(cycle);
         dc->SetTime(t);
         dc->Save();
      }

      if (t_final - t > 1e-8 * (t_final - t_init) && !ss)
      {
         HYPRE_Int global_dofs = fes.GlobalTrueVSize();

         if (global_dofs > max_dofs)
         {
            continue;
         }

         // Make sure errors will be recomputed in the following.
         if (mpi.Root())
         {
            cout << "\nEstimating errors." << endl;
         }
         refiner.Reset();
         derefiner.Reset();

         // 20. Call the refiner to modify the mesh. The refiner calls the error
         //     estimator to obtain element errors, then it selects elements to be
         //     refined and finally it modifies the mesh. The Stop() method can be
         //     used to determine if a stopping criterion was met.

         {
            double min_err, max_err;
            ErrorEstRange(estimator, min_err, max_err);

            if (mpi.Root())
            {
               cout << "Range of error estimates at time " << t << ": "
                    << min_err << " < elem err < " << max_err << endl;
            }
         }



         if (visualization)
         {
            const Vector & loc_err_est = estimator.GetLocalErrors();
            err_est.MakeRef(&fes_l2_o0, const_cast<double*>(&loc_err_est[0]));
            ostringstream oss;
            oss << "Error estimate at time " << t;
            VisualizeField(eout, vishost, visport, err_est, oss.str().c_str(),
                           Wx, Wy + 2 * (Wh + Dy), Ww, Wh);
         }

         if (amr)
         {
            refiner.Apply(pmesh);

            if (refiner.Stop())
            {
               if (mpi.Root())
               {
                  cout << "No refinements necessary." << endl;
               }
               // continue;
            }
            else
            {
               ref_it++;
               if (mpi.Root())
               {
                  cout << "Refining elements (iteration " << ref_it << ")" << endl;
               }

               // 21. Update the finite element space (recalculate the number of DOFs,
               //     etc.) and create a grid function update matrix. Apply the matrix
               //     to any GridFunctions over the space. In this case, the update
               //     matrix is an interpolation matrix so the updated GridFunction will
               //     still represent the same function as before refinement.
               ffes.Update();
               vfes.Update();
               fes.Update();
               fes_h1.Update();
               fes_rt.Update();
               // fes.ExchangeFaceNbrData();
               fes_l2_o0.Update();
               u.Update();

               {
                  for (int k = 0; k <= num_equations; k++)
                  {
                     offsets[k] = k * fes.GetNDofs();
                  }

                  neu_density.MakeRef(&fes, u, offsets[0]);
                  ion_density.MakeRef(&fes, u, offsets[1]);
                  para_velocity.MakeRef(&fes, u, offsets[2]);
                  ion_energy.MakeRef(&fes, u, offsets[3]);
                  elec_energy.MakeRef(&fes, u, offsets[4]);
               }
               oper.Update();
               ode_solver->Init(oper);

               // 22. Load balance the mesh, and update the space and solution. Currently
               //     available only for nonconforming meshes.
               if (pmesh.Nonconforming())
               {
                  reb_it++;
                  if (mpi.Root())
                  {
                     cout << "Rebalancing elements (iteration " << reb_it << ")"
                          << endl;
                  }
                  pmesh.Rebalance();

                  // Update the space and the GridFunction. This time the update matrix
                  // redistributes the GridFunction among the processors.
                  ffes.Update();
                  vfes.Update();
                  fes.Update();
                  fes_h1.Update();
                  fes_rt.Update();
                  // fes.ExchangeFaceNbrData();
                  fes_l2_o0.Update();
                  u.Update();
                  {
                     for (int k = 0; k <= num_equations; k++)
                     {
                        offsets[k] = k * fes.GetNDofs();
                     }

                     neu_density.MakeRef(&fes, u, offsets[0]);
                     ion_density.MakeRef(&fes, u, offsets[1]);
                     para_velocity.MakeRef(&fes, u, offsets[2]);
                     ion_energy.MakeRef(&fes, u, offsets[3]);
                     elec_energy.MakeRef(&fes, u, offsets[4]);
                  }
                  oper.Update();
                  ode_solver->Init(oper);
               }
               // m.Update(); m.Assemble(); m.Finalize();
               // ode_diff_msr.SetOperator(m);
            }
            if (derefiner.Apply(pmesh))
            {
               dref_it++;
               if (mpi.Root())
               {
                  cout << "Derefining elements (iteration " << dref_it << ")" << endl;
               }

               // 24. Update the space and the solution, rebalance the mesh.
               // cout << "fes.Update();" << endl;
               ffes.Update();
               vfes.Update();
               fes.Update();
               fes_h1.Update();
               fes_rt.Update();
               // fes.ExchangeFaceNbrData();
               // cout << "fes_l2_o0.Update();" << endl;
               fes_l2_o0.Update();
               // cout << "u.Update();" << endl;
               u.Update();
               {
                  for (int k = 0; k <= num_equations; k++)
                  {
                     offsets[k] = k * fes.GetNDofs();
                  }

                  neu_density.MakeRef(&fes, u, offsets[0]);
                  ion_density.MakeRef(&fes, u, offsets[1]);
                  para_velocity.MakeRef(&fes, u, offsets[2]);
                  ion_energy.MakeRef(&fes, u, offsets[3]);
                  elec_energy.MakeRef(&fes, u, offsets[4]);
               }
               // cout << "m.Update();" << endl;
               // m.Update(); m.Assemble(); m.Finalize();
               // ode_diff_msr.SetOperator(m);
               // cout << "oper.Update();" << endl;
               oper.Update();
               ode_solver->Init(oper);
            }
            else
            {
               if (mpi.Root())
               {
                  cout << "No derefinements needed." << endl;
               }
            }

            amr_it++;

            global_dofs = fes.GlobalTrueVSize();
            if (mpi.Root())
            {
               cout << "\nAMR iteration " << amr_it << endl;
               cout << "Number of unknowns: " << global_dofs << endl;
            }
            if (amr_it == 23)
            {
               ostringstream oss;
               oss << "bar_" << mpi.WorldRank() << ".ncmesh";
               ofstream ofs(oss.str().c_str());
               pmesh.ParPrint(ofs);
               ofs.close();
            }
         }
      }
      // Exit loop due to acquisition of steady state
      if (ss) { break; }
   }

   if (mpi.Root()) { ofserr.close(); }

   tic_toc.Stop();
   if (mpi.Root())
   {
      cout << "\nTime stepping done after " << tic_toc.RealTime() << "s.\n";
   }
   /*
   // 11. Save the final solution. This output can be viewed later using GLVis:
   //     "glvis -np 4 -m transport-mesh -g species-0-field-0-final".
   {
      int k = 0;
      for (int i = 0; i < num_species_; i++)
         for (int j = 0; j < dim + 2; j++)
         {
            ParGridFunction uk(&fes, u_block.GetBlock(k));
            ostringstream sol_name;
            sol_name << "species-" << i << "-field-" << j << "-final."
                     << setfill('0') << setw(6) << mpi.WorldRank();
            ofstream sol_ofs(sol_name.str().c_str());
            sol_ofs.precision(precision);
            sol_ofs << uk;
            k++;
         }
   }

   // 12. Compute the L2 solution error summed for all components.
   if ((t_final == 2.0 &&
        strcmp(mesh_file, "../data/periodic-square.mesh") == 0) ||
       (t_final == 3.0 &&
        strcmp(mesh_file, "../data/periodic-hexagon.mesh") == 0))
   {
      const double error = sol.ComputeLpError(2, u0);
      if (mpi.Root()) { cout << "Solution error: " << error << endl; }
   }
   */
   if (mpi.Root()) { ofs_controller.close(); }
   // Free the used memory.
   delete eqdsk;
   delete ode_solver;
   delete dc;
   if (eqnCoefs(5).GetVectorCoefficient(
          CommonCoefs::MAGNETIC_FIELD_COEF) != B3Coef)
   {
      delete svB3Coef;
      delete B3Coef;
   }

   return 0;
}

void record_cmd_line(int argc, char *argv[])
{
   ofstream ofs("transport2d_cmd.txt");

   for (int i=0; i<argc; i++)
   {
      ofs << argv[i] << " ";
      if (strcmp(argv[i], "-m-sol1d"            ) == 0 ||
          strcmp(argv[i], "-amr-weights"        ) == 0 ||
          strcmp(argv[i], "-amr-w"              ) == 0 ||
          strcmp(argv[i], "-equation-weights"   ) == 0 ||
          strcmp(argv[i], "-eqn-w"              ) == 0 ||
          strcmp(argv[i], "-field-magnitudes"   ) == 0 ||
          strcmp(argv[i], "-fld-m"              ) == 0 ||
          strcmp(argv[i], "-ode-weights"        ) == 0 ||
          strcmp(argv[i], "-ode-w"              ) == 0 ||
          strcmp(argv[i], "-equation-term-flags") == 0 ||
          strcmp(argv[i], "-term-flags"         ) == 0 ||
          strcmp(argv[i], "-visualization-flags") == 0 ||
          strcmp(argv[i], "-vis-flags"          ) == 0)
      {
         ofs << "'" << argv[++i] << "' ";
      }
   }
   ofs << endl << flush;
   ofs.close();
}

void set_mass_defaults(double &m_amu, double &m_kg, double m_def_amu)
{
   if (m_amu < 0.0 && m_kg < 0.0)
   {
      m_amu = m_def_amu;
      m_kg  = m_amu * kg_per_amu_;
   }
   else if (m_amu < 0.0)
   {
      m_amu = m_kg / kg_per_amu_;
   }
   else if (m_kg < 0.0)
   {
      m_kg = m_amu * kg_per_amu_;
   }

}

Mesh * BuildSol1DMesh(const Vector &sol1d)
{
   MFEM_VERIFY(sol1d.Size() == 3, "Must have 3 sol1d mesh parameters.");
   int NUMC = (int)sol1d(0);
   double L0 = sol1d(1);
   double beta = sol1d(2);

   Mesh mesh = Mesh::MakeCartesian2D(NUMC, 1, Element::QUADRILATERAL);

   double delx;

   double * xn = new double[NUMC+1];

   if (beta<0.0)
   {
      xn[0] = 0.0;
      delx = L0/((double)(NUMC));
      for (int i=1; i<=NUMC; i++) { xn[i]=xn[i-1]+delx; }
      xn[NUMC+1]=xn[NUMC];
   }
   else
   {
      double ksi;

      double * xtemp = new double[NUMC+1];
      xtemp[0] = 0.0;
      for (int i=1; i<=NUMC; i++)
      {
         ksi=(double)(i)/(double)(NUMC);
         xtemp[i] =L0*( (beta+1.0)-(beta-1.0)*pow((beta+1.0)/(beta-1.0),1.0-ksi) )
                   /( 1.0 + pow((beta+1.0)/(beta-1.0),1.0-ksi) );
      }

      for (int i=0; i<=NUMC; i++) { xn[i]=L0-xtemp[NUMC-i]; }
      xn[NUMC+1]=xn[NUMC];

      delete [] xtemp;
   }

   Vector vert_coord(4*(NUMC+1));

   for (int i=0; i<=NUMC; i++)
   {
      vert_coord(0 * (NUMC + 1) + i) = xn[i];
      vert_coord(1 * (NUMC + 1) + i) = xn[i];
      vert_coord(2 * (NUMC + 1) + i) = 0.0;
      vert_coord(3 * (NUMC + 1) + i) = 0.125 * L0;
   }

   delete [] xn;

   mesh.SetVertices(vert_coord);

   return new Mesh(mesh);
}

void ErrorEstRange(ErrorEstimator & est, double &min_err, double &max_err)
{
   const Vector & errors = est.GetLocalErrors();

   double loc_max_error = errors.Max();
   double loc_min_error = errors.Min();

   max_err = -1.0;
   min_err = -1.0;

   MPI_Allreduce(&loc_max_error, &max_err, 1,
                 MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(&loc_min_error, &min_err, 1,
                 MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
}

// Initial condition
void AdaptInitialMesh(MPI_Session &mpi,
                      ParMesh &pmesh, ParFiniteElementSpace &err_fespace,
                      ParFiniteElementSpace &fespace,
                      ParFiniteElementSpace &vfespace,
                      ParFiniteElementSpace &ffespace,
                      ParGridFunctionArray & gf, Array<Coefficient*> &coef,
                      Vector &weights,
                      int p, double tol, bool visualization)
{
   VectorLpErrorEstimator estimator(p, coef, gf, err_fespace, weights);

   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0);
   refiner.SetTotalErrorNormP(p);
   refiner.SetLocalErrorGoal(tol);

   vector<socketstream> sout(5);
   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 275, Wh = 250; // window size
   int offx = Ww + 3;

   const int max_dofs = 100000;
   for (int it = 0; ; it++)
   {
      HYPRE_Int global_dofs = fespace.GlobalTrueVSize();
      if (mpi.Root())
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of unknowns: " << global_dofs << endl;
      }

      // 19. Send the solution by socket to a GLVis server.
      gf.ProjectCoefficient(coef);

      if (visualization)
      {
         VisualizeField(sout[0], vishost, visport, *gf[0],
                        "Initial Neutral Density",
                        Wx, Wy, Ww, Wh);
         Wx += offx;

         VisualizeField(sout[1], vishost, visport, *gf[1],
                        "Initial Ion Density",
                        Wx, Wy, Ww, Wh);
         Wx += offx;

         VisualizeField(sout[2], vishost, visport, *gf[2],
                        "Initial Ion Velocity",
                        Wx, Wy, Ww, Wh);
         Wx += offx;

         VisualizeField(sout[3], vishost, visport, *gf[3],
                        "Initial Ion Temperature",
                        Wx, Wy, Ww, Wh);
         Wx += offx;

         VisualizeField(sout[4], vishost, visport, *gf[4],
                        "Initial Electron Temperature",
                        Wx, Wy, Ww, Wh);
      }

      if (global_dofs > max_dofs)
      {
         if (mpi.Root())
         {
            cout << "Reached the maximum number of dofs. Stop." << endl;
         }
         break;
      }

      // 20. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(pmesh);
      if (refiner.Stop())
      {
         if (mpi.Root())
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
         break;
      }

      // 21. Update the finite element space (recalculate the number of DOFs,
      //     etc.) and create a grid function update matrix. Apply the matrix
      //     to any GridFunctions over the space. In this case, the update
      //     matrix is an interpolation matrix so the updated GridFunction will
      //     still represent the same function as before refinement.
      err_fespace.Update();
      fespace.Update();
      vfespace.Update();
      ffespace.Update();
      gf.Update();

      // 22. Load balance the mesh, and update the space and solution. Currently
      //     available only for nonconforming meshes.
      if (pmesh.Nonconforming())
      {
         pmesh.Rebalance();

         // Update the space and the GridFunction. This time the update matrix
         // redistributes the GridFunction among the processors.
         err_fespace.Update();
         fespace.Update();
         vfespace.Update();
         ffespace.Update();
         gf.Update();
      }
   }
}

void InitialCondition(const Vector &x, Vector &y)
{
   MFEM_ASSERT(x.Size() == 2, "");
   /*
   double radius = 0, Minf = 0, beta = 0;
   if (problem_ == 1)
   {
      // "Fast vortex"
      radius = 0.2;
      Minf = 0.5;
      beta = 1. / 5.;
   }
   else if (problem_ == 2)
   {
      // "Slow vortex"
      radius = 0.2;
      Minf = 0.05;
      beta = 1. / 50.;
   }
   else
   {
      mfem_error("Cannot recognize problem."
                 "Options are: 1 - fast vortex, 2 - slow vortex");
   }

   const double xc = 0.0, yc = 0.0;

   // Nice units
   const double vel_inf = 1.;
   const double den_inf = 1.;

   // Derive remainder of background state from this and Minf
   const double pres_inf = (den_inf / specific_heat_ratio_) * (vel_inf / Minf) *
                           (vel_inf / Minf);
   const double temp_inf = pres_inf / (den_inf * gas_constant_);

   double r2rad = 0.0;
   r2rad += (x(0) - xc) * (x(0) - xc);
   r2rad += (x(1) - yc) * (x(1) - yc);
   r2rad /= (radius * radius);

   const double shrinv1 = 1.0 / (specific_heat_ratio_ - 1.);

   const double velX = vel_inf * (1 - beta * (x(1) - yc) / radius * exp(
                                     -0.5 * r2rad));
   const double velY = vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
   const double vel2 = velX * velX + velY * velY;

   const double specific_heat = gas_constant_ * specific_heat_ratio_ * shrinv1;
   const double temp = temp_inf - 0.5 * (vel_inf * beta) *
                       (vel_inf * beta) / specific_heat * exp(-r2rad);

   const double den = den_inf * pow(temp/temp_inf, shrinv1);
   const double pres = den * gas_constant_ * temp;
   const double energy = shrinv1 * pres / den + 0.5 * vel2;

   y(0) = den;
   y(1) = den * velX;
   y(2) = den * velY;
   y(3) = den * energy;
   */
   // double VMag = 1e2;
   /*
   if (y.Size() != num_equations_) { cout << "y is wrong size!" << endl; }

   int dim = 2;
   double a = 0.4;
   double b = 0.8;

   Vector V(2);
   bFunc(x, V);
   V *= (v_max_ / B_max_) * sqrt(pow(x[0]/a,2)+pow(x[1]/b,2));

   double den = 1.0e18;
   for (int i=1; i<=num_species_; i++)
   {
      y(i) = den;
      y(i * dim + num_species_ + 1) = V(0);
      y(i * dim + num_species_ + 2) = V(1);
      y(i + (num_species_ + 1) * (dim + 1)) = 10.0 * TFunc(x, 0.0);
   }

   // Impose neutrality
   y(0) = 0.0;
   for (int i=1; i<=num_species_; i++)
   {
      y(0) += y(i);
   }
   y(num_species_ + 1) = V(0);
   y(num_species_ + 2) = V(1);
   y((num_species_ + 1) * (dim + 1)) = 5.0 * TFunc(x, 0.0);

   // y.Print(cout, dim+2); cout << endl;
   */
}

/** Coefficient which returns a * cos(kx * (x - c)) + b */
class Cos1D: public Coefficient
{
private:
   double a_, b_, c_, kx_;

   mutable Vector x_;

public:
   Cos1D(double a, double b, double c, double kx)
      : a_(a), b_(b), c_(c), kx_(kx), x_(3) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      return a_ * cos(kx_ * (x_[0] - c_)) + b_;
   }
};

/** Coefficient which returns a * sin(kx * (x - c)) + b */
class Sin1D: public Coefficient
{
private:
   double a_, b_, c_, kx_;

   mutable Vector x_;

public:
   Sin1D(double a, double b, double c, double kx)
      : a_(a), b_(b), c_(c), kx_(kx), x_(3) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      return a_ * sin(kx_ * (x_[0] - c_)) + b_;
   }
};

/** Coefficient which returns a * sinh(kx * (x - c)) + b */
class Sinh1D: public Coefficient
{
private:
   double a_, b_, c_, kx_;

   mutable Vector x_;

public:
   Sinh1D(double a, double b, double c, double kx)
      : a_(a), b_(b), c_(c), kx_(kx), x_(3) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      return a_ * sinh(kx_ * (x_[0] - c_)) + b_;
   }
};

/** Coefficient which returns a * cosh(kx * (x - c)) + b */
class Cosh1D: public Coefficient
{
private:
   double a_, b_, c_, kx_;

   mutable Vector x_;

public:
   Cosh1D(double a, double b, double c, double kx)
      : a_(a), b_(b), c_(c), kx_(kx), x_(3) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      return a_ * cosh(kx_ * (x_[0] - c_)) + b_;
   }
};

class SinSin2D: public Coefficient
{
private:
   double a_, kx_, ky_;

   mutable Vector x_;

public:
   SinSin2D(double a, double kx, double ky) : a_(a), kx_(kx), ky_(ky), x_(3) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      return a_ * sin(kx_ * x_[0]) * sin(ky_ * x_[1]);
   }
};

class ExpSinSin2D: public Coefficient
{
private:
   double a_, b_, kx_, ky_;

   mutable Vector x_;

public:
   ExpSinSin2D(double a, double b, double kx, double ky)
      : a_(a), b_(b), kx_(kx), ky_(ky), x_(3) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      return a_ + (1.0 - exp(-b_ * time)) *
             sin(kx_ * x_[0]) * sin(ky_ * x_[1]);
   }
};

class SinPhi: public Coefficient
{
private:
   double a_, b_;
   int n_;

   mutable Vector x_;

   double (*sinFunc)(int n, const Vector &x);

   static double Sin1(int n, const Vector &x)
   { return x[1] / sqrt(x[0] * x[0] + x[1] * x[1]); }

   static double Sin2(int n, const Vector &x)
   { return 2.0 * x[0] * x[1] / (x[0] * x[0] + x[1] * x[1]); }

   static double SinN(int n, const Vector &x)
   {
      double phi = atan2(x[1], x[0]);
      return sin((double)n * phi);
   }

public:
   SinPhi(double a, double b, int n)
      : a_(a), b_(b), n_(n), x_(3)
   { sinFunc = (n == 1) ? Sin1 : ((n == 2) ? Sin2 : SinN); }

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      return a_ + b_ * (*sinFunc)(n_, x_);
   }
};

/** Coefficient which returns a * sin(kx * (x - c)^2 + w t) + b */
class SinXSqr1D: public Coefficient
{
private:
   double a_, b_, c_, kx_, w_;

   mutable Vector x_;

public:
   SinXSqr1D(double a, double b, double c, double kx, double w)
      : a_(a), b_(b), c_(c), kx_(kx), w_(w), x_(3) {}

   virtual void SetTime(double t)
   { cout << "SinXSqr1D::SetTime(" << t << ")" << endl; Coefficient::SetTime(t); }

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      return a_ * sin(kx_ * pow(x_[0] - c_, 2) + w_ * time) + b_;
   }
};

/** Coefficient which returns MMS source corresponding to
    a * sin(kx * (x - c)^2 + w t) + b */
class SinXSqr1DMMS: public Coefficient
{
private:
   double a_, b_, c_, kx_, w_, m_, n_, d_;

   mutable Vector x_;

public:
   SinXSqr1DMMS(double a, double b, double c, double kx, double w,
                double m, double n, double d)
      : a_(a), b_(b), c_(c), kx_(kx), w_(w), m_(m), n_(n), d_(d), x_(3) {}

   virtual void SetTime(double t)
   { cout << "SinXSqr1DMMS::SetTime(" << t << ")" << endl; Coefficient::SetTime(t); }
   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      double p = kx_ * pow(x_[0] - c_, 2) + w_ * time;
      return a_ * (pow(2.0 * kx_ * (x_[0] - c_), 2) * d_ * sin(p) +
                   cos(p) * (4.0 * kx_ * m_ * n_ * (x_[0] - c_) * (a_ * sin(p) + b_)
                             - 2.0 * kx_ * d_ + m_ * n_ * w_));
   }
};

class Gaussian1D : public Coefficient
{
private:
   int comp_;
   double p0_;
   double v_;
   double a_;
   double b_;
   double c_;

   mutable Vector x_;

public:
   Gaussian1D(double a, double b, double c, double p0, double v, int comp)
      : comp_(comp), p0_(p0), v_(v), a_(a), b_(b), c_(c) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      return a_ * exp(- b_ * pow(x_[comp_] - p0_ - v_ * time, 2)) + c_;
   }
};

class Gaussian : public Coefficient
{
private:
   double x0_;
   double y0_;
   double z0_;
   double a_;
   double b_;
   double d_;

   mutable Vector x_;

public:
   Gaussian(double a, double b, double d, double x0, double y0, double z0)
      : x0_(x0), y0_(y0), z0_(z0), a_(a), b_(b), d_(d) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      x_[0] -= x0_;
      if (x_.Size() > 1) { x_[1] -= y0_; }
      if (x_.Size() > 2) { x_[2] -= z0_; }
      return a_ * exp(- d_ * (x_ * x_)) + b_;
   }
};

class GaussianFluxIonDenIC : public Coefficient
{
private:
    double a_;
    double mu_;
    double sig_;
    double v_;
    double c_;
    
    mutable Vector x_;

public:
    GaussianFluxIonDenIC(double a, double mu, double sig, double v, double c)
       : a_(a), mu_(mu), sig_(sig), v_(v), c_(c), x_(3) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      //return a_ * (2.0 + exp(- pow((-mu_ - v_ * time + x_[0]),2)/pow((2.0 * sig_),2)) + 
        //     exp(- pow((-mu_ + v_ * time + x_[0]),2)/pow((2.0 + sig_),2)));
   
      return a_ * exp(- pow((x_[0] - mu_),2) / (2.0 * sig_)) + c_; 
   }
};

class GaussianFluxIonVelIC : public Coefficient
{
private:
    double a_;
    double mu_;
    double sig_;
    double v_;
    double c_;
    
    mutable Vector x_;

public:
    GaussianFluxIonVelIC(double a, double mu, double sig, double v, double c)
       : a_(a), mu_(mu), sig_(sig), v_(v), c_(c), x_(3) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      /*return a_ * ((exp(- pow((-mu_ - v_ * time + x_[0]),2)/pow((2.0 * sig_),2)) - 
             exp(- pow((-mu_ + v_ * time + x_[0]),2)/pow((2.0 + sig_),2))) / 
                (2.0 + exp(- pow((-mu_ - v_ * time + x_[0]),2)/pow((2.0 * sig_),2)) + 
                    exp(- pow((-mu_ + v_ * time + x_[0]),2)/pow((2.0 + sig_),2)))); 
      */
      return a_ * exp(- pow((x_[0] - mu_),2) / (2.0 * sig_)) - c_;  
   }
};

/** A GaussianStep1D is a C^3 step function with a limiting value of
    'a' below p0 and a limiting value of 'b' above p0. The parameter
    'd' specifies the maximum magnitude of the derivative midway
    between a and b.
*/
class GaussianStep1D : public Coefficient
{
private:
   int comp_;
   double p0_;
   double a_;
   double b_;
   double d_;

   mutable Vector x_;

public:
   GaussianStep1D(double a, double b, double d, double p0, int comp)
      : comp_(comp), p0_(p0), a_(a), b_(b), d_(d) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      double c = 0.5 * (b_ - a_) * sqrt(M_E);
      double u = 2.0 * d_ * (x_[comp_] - p0_) / fabs(b_ - a_);
      return (x_[comp_] < p0_) ?
             a_ + c * exp(-0.5 * pow(u - 1.0, 2)):
             b_ - c * exp(-0.5 * pow(u + 1.0, 2));
   }
};

class OscillatingGaussian1D : public Coefficient
{
private:
   int comp_;
   double p0_;
   double q0_;
   double w_;
   double a_;
   double b_;

   mutable Vector x_;

public:
   OscillatingGaussian1D(double a, double b, double p0, double q0, double w,
                         int comp)
      : comp_(comp), p0_(p0), q0_(q0), w_(w), a_(a), b_(b) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      return a_ * exp(- b_ * pow(x_[comp_] - p0_ + q0_ * cos(w_ * time), 2));
   }
   /*
   void SetTime(double t)
   {
     cout << "OscillatingGaussian1D::SetTime(" << t << ")" << endl;
     this->Coefficient::SetTime(t);
   }
   */
};

class SineTime : public Coefficient
{
private:
   double a_;
   double b_;
   double w_;
   double t0_;

public:
   SineTime(double a, double b, double w, double t0)
      : a_(a), b_(b), w_(w), t0_(t0) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return a_ * sin(w_ * (time - t0_)) + b_;
   }
   /*
   void SetTime(double t)
   {
     cout << "SineTime::SetTime(" << t << ")" << endl;
     this->Coefficient::SetTime(t);
   }
   */
};

class ExpTime : public Coefficient
{
private:
   double a_;
   double b_;
   double w_;
   double t0_;

public:
   ExpTime(double a, double b, double w, double t0)
      : a_(a), b_(b), w_(w), t0_(t0) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return a_ * exp(w_ * (time - t0_)) + b_;
   }
};

class TanhTime : public Coefficient
{
private:
   double a_;
   double b_;
   double w_;
   double t0_;

public:
   TanhTime(double a, double b, double w, double t0)
      : a_(a), b_(b), w_(w), t0_(t0) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return a_ * tanh(w_ * (time - t0_)) + b_;
   }
};

class Radius : public Coefficient
{
private:
   double a_;

   mutable Vector x_;

public:
   Radius(double a) : a_(a), x_(3) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      return a_ * sqrt(x_ * x_);
   }
};

class RadiusSqr : public Coefficient
{
private:
   double a_;

   mutable Vector x_;

public:
   RadiusSqr(double a) : a_(a), x_(3) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      return a_ * (x_ * x_);
   }
};

class AnnularTestSol : public Coefficient
{
private:
   double ra_;
   double rb_;
   double w_;
   double d_para_;
   double d_perp_;
   double a_;
   double a_para_;
   double a_perp_;
   int n_;

   mutable Vector x_;

public:
   AnnularTestSol(double ra, double rb, double w, double d_para, double d_perp,
                  double a, double a_para, double a_perp, int n)
      : ra_(ra), rb_(rb), w_(w), d_para_(d_para), d_perp_(d_perp),
        a_(a), a_para_(a_para), a_perp_(a_perp), n_(n), x_(3) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);

      double r2 = x_ * x_;
      double r = sqrt(r2);

      double tau_perp = d_perp_ * M_PI * M_PI / pow(rb_ - ra_, 2);
      double ut_perp = 1.0 - exp(-tau_perp * time);

      double k_perp = M_PI * (r - ra_) / (rb_ - ra_);
      double ur_perp = sqrt(0.5 * (ra_ + rb_) / r) * sin(k_perp);

      double u_perp = a_perp_ * ur_perp * ut_perp;

      double tau_para = d_para_ * double(n_ * n_);
      double ut_para = exp(-tau_para * time);

      double del_para = double(n_) * atan2(x_[1], x_[0]) - w_ * time;
      double ur_para = log(r2 / (ra_ * rb_)) / log(rb_ / ra_);

      double u_para = a_para_ * cos(del_para) * ur_para * ut_para;

      return a_ + u_perp + u_para;
   }
};

class AnnularTestSrc : public Coefficient
{
private:
   double ra_;
   double rb_;
   double d_perp_;
   double a_perp_;

   mutable Vector x_;

public:
   AnnularTestSrc(double ra, double rb, double d_perp, double a_perp)
      : ra_(ra), rb_(rb), d_perp_(d_perp), a_perp_(a_perp), x_(3) {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);

      double r2 = x_ * x_;
      double r = sqrt(r2);

      double tau_perp = d_perp_ * M_PI * M_PI / pow(rb_ - ra_, 2);
      double ft_perp = 1.0 - exp(-tau_perp * time);

      double k_perp = M_PI * (r - ra_) / (rb_ - ra_);
      double fr_perp = a_perp_ * d_perp_ * sin(k_perp) *
                       sqrt(0.03125 * (ra_ + rb_) / pow(r, 5));

      double f_perp = fr_perp *
                      (4.0 * M_PI * M_PI * r2 / pow(rb_ - ra_, 2) - ft_perp);

      return f_perp;
   }
};

class CirculationVector : public VectorCoefficient
{
private:
   double w_;
   double vz_;

   mutable Vector x_;

public:
   CirculationVector(double w) : VectorCoefficient(2), w_(w) {}
   CirculationVector(double w, double vz)
      : VectorCoefficient(3), w_(w), vz_(vz), x_(3) {}

   void Eval(Vector & V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);
      V.SetSize(vdim);
      V[0] = -w_ * x_[1];
      V[1] =  w_ * x_[0];
      if (vdim > 2) { V[2] = vz_; }
   }
};

/* An exact solution to Burger's equation for testing the ion momentum equation.

   Using any solution to the heat equation, u, we can build a solution to
   Burger's equation by forming the ratio -kappa (du/dx)/u, where kappa is the
   diffusion constant.

   For this test we take:
      u = 1 + sum_i alpha_i cos(2 pi x/lambda_i) exp(-kappa (2pi/lambda_i)^2 t)
   for i = 1 and 2.

   Clearly shorter wavelengths will decay more rapidly. Note that zeros in u
   will lead to singularities in the above ratio. Consequently, it would be
   wise to choose sum_i |alpha_i| < 1.
*/
class BurgersEqnTestSol : public Coefficient
{
private:
   const double alpha1_;
   const double alpha2_;
   const double kappa_;
   const double kx1_;
   const double kx2_;
   const double sigma1_;
   const double sigma2_;

   mutable Vector x_;

public:
   BurgersEqnTestSol(double alpha1, double lambda1,
                     double alpha2, double lambda2, double kappa)
      : alpha1_(alpha1),
        alpha2_(alpha2),
        kappa_(kappa),
        kx1_(2.0 * M_PI / lambda1),
        kx2_(2.0 * M_PI / lambda2),
        sigma1_(kappa * kx1_ * kx1_),
        sigma2_(kappa * kx2_ * kx2_),
        x_(2)
   {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);

      double e1 = exp(-sigma1_ * time);
      double e2 = exp(-sigma2_ * time);
      double s1 = sin(kx1_ * x_[0]);
      double s2 = sin(kx2_ * x_[0]);
      double c1 = cos(kx1_ * x_[0]);
      double c2 = cos(kx2_ * x_[0]);

      return kappa_ * (kx1_ * alpha1_ * s1 * e1 + kx2_ * alpha2_ * s2 * e2) /
             (1.0 + alpha1_ * c1 * e1 + alpha2_ * c2 * e2);
   }
};

/* A steady state solution to the ion total energy equation for testing the
   Robin boundary condition.

   The test problem only evolves the ion total energy with the ion
   density, ion parallel velocity and electron temperature remaining
   constant at values set in the constructor.

   The ion thermal parallel diffusion coefficient can be adjusted as can the
   'al', 'bl', 'ar', and 'br' factors in the Robin BC which is applied on the
   'left' and 'right' hand boundaries. The Robin BC is defined as:

      -qi.{-1,0,0} + al Ti = bl
      -qi.{ 1,0,0} + ar Ti = br

   Where the flux 'qi' is given by:

      qi = - ni chi kB Grad Ti + (5/2 ni kB Ti + 1/2 mi ni vi^2) vi b_hat

   The argument 'q' is the constant appearing in the temperature balance term:

      Qi = q (Te - Ti)

   Given by:

      q = 3 (me ne) kB / (mi tau_e)
*/
class RobinBCTestSol : public Coefficient
{
private:
   const double mi_;
   const double ni_;
   const double Te_;
   const double vi_;
   const double chi_;
   const double q_;
   const double al_, bl_;
   const double ar_, br_;

   double alpha0_;
   double alpha1_;
   double kappa_;
   double nu_;

   mutable Vector x_;

public:
   RobinBCTestSol(double mi, double ni, double Te, double vi, double chi,
                  double q, double al, double bl, double ar, double br)
      : mi_(mi * kg_per_amu_), ni_(ni), Te_(Te), vi_(vi),
        chi_(chi), q_(q), al_(al), bl_(bl), ar_(ar), br_(br),
        x_(2)
   {
      MFEM_VERIFY(mi_ > 0.0 && ni_ > 0.0 && chi_ > 0.0 && q_ > 0.0,
                  "RobinBCTestSol: parameters mi, ni, chi, and q "
                  "must be positive");

      const double kB = J_per_eV_;
      const double zeta = ni_ * vi_ * (2.5 * kB * Te_ + 0.5 * mi_ * vi_ * vi_);

      nu_ = 1.25 * vi_ / chi;
      kappa_ = sqrt(q_ / (ni_ * chi_ * kB) + nu_ * nu_);

      const double ev0 = exp(-nu_);
      const double ev1 = exp(nu_);
      const double ek0 = exp(-kappa_);
      const double ek1 = exp(kappa_);

      const double al0 = al_ - (kappa_ - nu_) * chi_;
      const double al1 = al_ + (kappa_ + nu_) * chi_;
      const double ar0 = ar_ - (kappa_ + nu_) * chi_;
      const double ar1 = ar_ + (kappa_ - nu_) * chi_;

      const double br0 = br_ - ar_ * ni_ * kB * Te_ + zeta;
      const double bl0 = bl_ - al_ * ni_ * kB * Te_ - zeta;

      const double d = kB * ni_ * (ar1 * al1 * ek1 - al0 * ar0 * ek0);

      alpha0_ = (bl0 * ar1 * ek1 - br0 * al0 * ev0) / d;
      alpha1_ = (br0 * al1 * ek1 - bl0 * ar0 * ev1) / d;
      /*
      switch (type_)
      {
         case ZERO_ENERGY_FLUX:
         {
            const double d = 2.0 * ((ni_ * nu_ * kB * a_ + q_) * sk +
                                    kB * a_ * kappa_ * ni_ * ck);

            alpha0_ = ((b_ + zeta - ni_ * Te_ * a_) * (kappa_ - nu_) * ev0 -
                       zeta * (a_ / chi_ + kappa_ - nu_) * ek) / d;
            alpha1_ = (zeta * (ni_ * kB * a_ * (kappa_ - nu_) - q_) * ev1 +
                       q_ * (b_ + zeta - ni_ * Te_ * a_) * ek
                      ) / (kB * chi_ * ni_ * (kappa_ - nu_) * d);
         }
         break;
         case ZERO_THERMAL_FLUX:
         {
            const double d = 2.0 * (q_ * sk +
                                    ni_ * (2.0 * chi_ * nu_ - a_) *
                                    (nu_ * sk - kappa_ * ck));

            alpha0_ =  ev0 * (b_ + zeta - ni_ * Te_ * a_) * (kappa_ + nu_) / d;
            alpha1_ =  ek * (b_ + zeta - ni_ * Te_ * a_) * (kappa_ - nu_) / d;
         }
         break;
         case ROBIN:
            const double d = 2.0 * ((ni_ * a_ * a_ + q_ * chi_) * sk +
                                    2 * ni_ * chi_ * kappa_ * a_ * ck);
            const double c0 = b_ - ni_ * Te_ * a_ - zeta;
            const double c1 = b_ - ni_ * Te_ * a_ + zeta;

            alpha0_ = (ek * c0 * ((kappa_ - nu_) * chi_ + a_) +
                       ev0 * c1 * ((kappa_ - nu_) * chi_ - a_))/ d;
            alpha1_ = (ev1 * c0 * ((kappa_ + nu_) * chi_ - a_) +
                       ek * c1 * ((kappa_ + nu_) * chi_ + a_))/ d;
            break;
      }
      */
   }

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);

      double e0x = exp(-(kappa_ - nu_) * x_[0]);
      double e1x = exp((kappa_ + nu_) * (x_[0] - 1.0));

      return Te_ + alpha0_ * e0x + alpha1_ * e1x;
   }
};

Coefficient *
Transport2DCoefFactory::GetScalarCoef(std::string &name, std::istream &input)
{
   int coef_idx = -1;
   if (name == "Sin1D")
   {
      double a, b, c, kx;
      input >> a >> b >> c >> kx;
      coef_idx = sCoefs.Append(new Sin1D(a, b, c, kx));
   }
   else if (name == "Cos1D")
   {
      double a, b, c, kx;
      input >> a >> b >> c >> kx;
      coef_idx = sCoefs.Append(new Cos1D(a, b, c, kx));
   }
   else if (name == "Sinh1D")
   {
      double a, b, c, kx;
      input >> a >> b >> c >> kx;
      coef_idx = sCoefs.Append(new Sinh1D(a, b, c, kx));
   }
   else if (name == "Cosh1D")
   {
      double a, b, c, kx;
      input >> a >> b >> c >> kx;
      coef_idx = sCoefs.Append(new Cosh1D(a, b, c, kx));
   }
   else if (name == "SinSin2D")
   {
      double a, kx, ky;
      input >> a >> kx >> ky;
      coef_idx = sCoefs.Append(new SinSin2D(a, kx, ky));
   }
   else if (name == "ExpSinSin2D")
   {
      double a, b, kx, ky;
      input >> a >> b >> kx >> ky;
      coef_idx = sCoefs.Append(new ExpSinSin2D(a, b, kx, ky));
   }
   else if (name == "SinPhi")
   {
      double a, b;
      int n;
      input >> a >> b >> n;
      coef_idx = sCoefs.Append(new SinPhi(a, b, n));
   }
   else if (name == "SinXSqr1D")
   {
      double a, b, c, kx, w;
      input >> a >> b >> c >> kx >> w;
      coef_idx = sCoefs.Append(new SinXSqr1D(a, b, c, kx, w));
   }
   else if (name == "SinXSqr1DMMS")
   {
      double a, b, c, kx, w, m, n, d;
      input >> a >> b >> c >> kx >> w >> m >> n >> d;
      coef_idx = sCoefs.Append(new SinXSqr1DMMS(a, b, c, kx, w, m, n, d));
   }
   else if (name == "Gaussian1D")
   {
      double a, b, c, p0, v;
      int comp;
      input >> a >> b >> c >> p0 >> v >> comp;
      coef_idx = sCoefs.Append(new Gaussian1D(a, b, c, p0, v, comp));
   }
   else if (name == "GaussianFluxIonDenIC")
   {
      double a, mu, sig, v, c;
      input >> a >> mu >> sig >> v >> c;
      coef_idx = sCoefs.Append(new GaussianFluxIonDenIC(a, mu, sig, v, c));
   }
   else if (name == "GaussianFluxIonVelIC")
   {
      double a, mu, sig, v, c;
      input >> a >> mu >> sig >> v >> c;
      coef_idx = sCoefs.Append(new GaussianFluxIonVelIC(a, mu, sig, v, c));
   }
   else if (name == "Gaussian")
   {
      double a, b, d, x0, y0, z0;
      input >> a >> b >> d >> x0 >> y0 >> z0;
      coef_idx = sCoefs.Append(new Gaussian(a, b, d, x0, y0, z0));
   }
   else if (name == "GaussianStep1D")
   {
      double a, b, d, p0;
      int comp;
      input >> a >> b >> d >> p0 >> comp;
      coef_idx = sCoefs.Append(new GaussianStep1D(a, b, d, p0, comp));
   }
   else if (name == "OscillatingGaussian1D")
   {
      double a, b, p0, q0, w;
      int comp;
      input >> a >> b >> p0 >> q0 >> w >> comp;
      coef_idx = sCoefs.Append(new OscillatingGaussian1D(a, b, p0, q0, w,
                                                         comp));
   }
   else if (name == "SineTime")
   {
      double a, b, w, t0;
      input >> a >> b >> w >> t0;
      coef_idx = sCoefs.Append(new SineTime(a, b, w, t0));
   }
   else if (name == "ExpTime")
   {
      double a, b, w, t0;
      input >> a >> b >> w >> t0;
      coef_idx = sCoefs.Append(new ExpTime(a, b, w, t0));
   }
   else if (name == "TanhTime")
   {
      double a, b, w, t0;
      input >> a >> b >> w >> t0;
      coef_idx = sCoefs.Append(new TanhTime(a, b, w, t0));
   }
   else if (name == "AnnularTestSol")
   {
      double ra, rb, w, d_para, d_perp, a, a_para, a_perp;
      int n;
      input >> ra >> rb >> w >> d_para >> d_perp >> a >> a_para >> a_perp >> n;
      coef_idx = sCoefs.Append(new AnnularTestSol(ra, rb, w, d_para, d_perp,
                                                  a, a_para, a_perp, n));
   }
   else if (name == "Radius")
   {
      double a;
      input >> a;
      coef_idx = sCoefs.Append(new Radius(a));
   }
   else if (name == "RadiusSqr")
   {
      double a;
      input >> a;
      coef_idx = sCoefs.Append(new RadiusSqr(a));
   }
   else if (name == "AnnularTestSrc")
   {
      double ra, rb, d_perp, a_perp;
      input >> ra >> rb >> d_perp >> a_perp;
      coef_idx = sCoefs.Append(new AnnularTestSrc(ra, rb, d_perp, a_perp));
   }
   else if (name == "BurgersEqnTestSol")
   {
      double a1, l1, a2, l2, k;
      input >> a1 >> l1 >> a2 >> l2 >> k;
      coef_idx = sCoefs.Append(new BurgersEqnTestSol(a1, l1, a2, l2, k));
   }
   else if (name == "RobinBCTestSol")
   {
      double mi, ni, Te, vi, chi, q, al, bl, ar, br;
      input >> mi >> ni >> Te >> vi >> chi >> q >> al >> bl >> ar >> br;
      coef_idx = sCoefs.Append(new RobinBCTestSol(mi, ni, Te, vi, chi, q,
                                                  al, bl, ar, br));
   }
   else
   {
      return TransportCoefFactory::GetScalarCoef(name, input);
   }
   return sCoefs[--coef_idx];
}

VectorCoefficient *
Transport2DCoefFactory::GetVectorCoef(std::string &name, std::istream &input)
{
   int coef_idx = -1;
   if (name == "CirculationVector")
   {
      double w, vz;
      input >> w >> vz;
      coef_idx = vCoefs.Append(new CirculationVector(w, vz));
   }
   else
   {
      return TransportCoefFactory::GetVectorCoef(name, input);
   }
   return vCoefs[--coef_idx];
}
