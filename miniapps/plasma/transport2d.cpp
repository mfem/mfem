//
// ./transport -s 12 -v 1 -vs 5 -tol 5e-3 -tf 4
//

// Annular benchmark test
// mpirun -np 10 ./transport2d -v 1 -vs 1 -epus -tf 1 -op 16 -l 1 -m annulus-quad-o3.mesh -p 1 -ni-min 3e19 -ni-max 3e19 -Te-min 11 -Te-max 440 -ode-w '1e-8 1 0 0 1e-4' -dt 1e-1 -visit

#include "mfem.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

#include <float.h>

#include "../common/pfem_extras.hpp"
#include "transport_solver.hpp"

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

static double ni_max_ = 1.0e18;
static double ni_min_ = 1.0e16;

static double Ti_max_ = 10.0;
static double Ti_min_ =  1.0;

static double Te_max_ = 10.0;
static double Te_min_ =  1.0;
static double Te_exp_ =  0.0;

static double Tot_B_max_ = 5.0; // Maximum of total B field
static double Pol_B_max_ = 0.5; // Maximum of poloidal B field
static double v_max_ = 1e3;

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
         return nn_max_ + (nn_min_ - nn_max_) * (0.5 + 0.5 * cos(M_PI * r));
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
         return ni_min_ + (ni_max_ - ni_min_) * (0.5 + 0.5 * cos(M_PI * r));
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
         return Ti_min_ + (Ti_max_ - Ti_min_) * (0.5 + 0.5 * cos(M_PI * r));
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

   B[2] = sqrt(Tot_B_max_ * Tot_B_max_ - (polB * polB));
}

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
      K *= 0.96 * ni * Ti * eV_ * tau;

      perpFunc(x_, perp_);

      double Di_perp = Di_perp_->Eval(T, ip);

      K.Add(mi_ * amu_ * ni * Di_perp, perp_);
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

      K *= a_ * Temp * eV_ * tau / ( m_ * amu_ );

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
   Array<ODERelativeErrorMeasure*> msr_;

   int size_;
   Vector u_, e_;

public:
   VectorRelativeErrorMeasure(MPI_Comm comm, Vector & weights)
      : w_(weights),
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
            msr_[i] = new ParMaxAbsRelDiffMeasure(comm, 1.0);
         }
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

// Initial condition
void AdaptInitialMesh(MPI_Session &mpi,
                      ParMesh &pmesh, ParFiniteElementSpace &err_fespace,
                      ParFiniteElementSpace &fespace,
                      ParFiniteElementSpace &vfespace,
                      ParGridFunctionArray & gf, Array<Coefficient*> &coef,
                      Vector &weights,
                      int p, double tol, bool visualization = false);

void InitialCondition(const Vector &x, Vector &y);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi(argc, argv);

   // 2. Parse command-line options.
   // problem_ = 1;
   const char *mesh_file = "ellipse_origin_h0pt0625_o3.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int nc_limit = 3;         // maximum level of hanging nodes
   double max_elem_error = -1.0;
   double hysteresis = 0.25; // derefinement safety coefficient
   int order = 3;

   DGParams dg;
   dg.sigma = -1.0;
   dg.kappa = -1.0;

   int ode_solver_type = 2;
   int logging = 0;
   bool   imex = true;
   bool ode_epus = false;
   int    op_flag = 31;
   double tol_ode = 1e-3;
   double rej_ode = 1.2;
   double kP_acc = 0.0;
   double kI_acc = 0.6;
   double kD_acc = 0.0;
   double kI_rej = 0.6;
   double lim_max = 2.0;

   double tol_init = 1e-5;
   double t_init = 0.0;
   double t_final = -1.0;
   double dt = -0.01;
   // double dt_rel_tol = 0.1;
   double cfl = 0.3;

   Array<int> vis_flags;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 10;

   PlasmaParams plasma;
   plasma.m_n = 2.01410178; // (amu)
   plasma.T_n = 3.0;        // (eV)
   plasma.m_i = 2.01410178; // (amu)
   plasma.z_i = 1;          // ion charge
   /*
   int      ion_charge = 1;
   double     ion_mass = 2.01410178; // (amu)
   double neutral_mass = 2.01410178; // (amu)
   double neutral_temp = 3.0;        // (eV)
   */

   double      Di_perp = 1.0;        // Ion perp diffusion (m^2/s)
   double      Xi_perp = 1.0;        // Ion thermal diffusion (m^2/s)
   double      Xe_perp = 1.0;        // Electron thermal diffusion (m^2/s)

   Vector amr_weights;
   Vector ode_weights;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&logging, "-l", "--logging",
                  "Set the logging level.");
   args.AddOption(&op_flag, "-op", "--operator-test",
                  "Bitmask for disabling operators.");
   args.AddOption(&prob_, "-p", "--problem",
                  "Problem setup to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly before parallel"
                  " partitioning, -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly after parallel"
                  " partitioning.");
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
   args.AddOption(&tol_init, "-tol0", "--initial-tolerance",
                  "Error tolerance for initial condition.");
   args.AddOption(&tol_ode, "-tol", "--ode-tolerance",
                  "Difference tolerance for ODE integration.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE Implicit solver: "
                  "            IMEX methods\n\t"
                  "            1 - IMEX BE/FE, 2 - IMEX RK2,\n\t"
                  "            L-stable methods\n\t"
                  "            11 - Backward Euler,\n\t"
                  "            12 - SDIRK23, 13 - SDIRK33,\n\t"
                  "            A-stable methods (not L-stable)\n\t"
                  "            22 - ImplicitMidPointSolver,\n\t"
                  "            23 - SDIRK23, 34 - SDIRK34.");
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
   args.AddOption(&lim_max, "-thm", "--theta-max",
                  "Maximum dt increase factor.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   // args.AddOption(&dt_rel_tol, "-dttol", "--time-step-tolerance",
   //                "Time step will only be adjusted if the relative difference "
   //                "exceeds dttol.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&plasma.z_i, "-qi", "--ion-charge",
                  "Charge of the species "
                  "(in units of electron charge)");
   args.AddOption(&plasma.m_i, "-mi", "--ion-mass",
                  "Mass of the ion species (in amu)");
   args.AddOption(&plasma.m_n, "-mn", "--neutral-mass",
                  "Mass of the neutral species (in amu)");
   args.AddOption(&plasma.T_n, "-Tn", "--neutral-temp",
                  "Temperature of the neutral species (in eV)");
   args.AddOption(&nn_min_, "-nn-min", "--min-neutral-density",
                  "Minimum of inital neutral density");
   args.AddOption(&nn_max_, "-nn-max", "--max-neutral-density",
                  "Maximum of inital neutral density");
   args.AddOption(&ni_min_, "-ni-min", "--min-ion-density",
                  "Minimum of inital ion density");
   args.AddOption(&ni_max_, "-ni-max", "--max-ion-density",
                  "Maximum of inital ion density");
   args.AddOption(&Ti_min_, "-Ti-min", "--min-ion-temp",
                  "Minimum of inital ion temperature");
   args.AddOption(&Ti_max_, "-Ti-max", "--max-ion-temp",
                  "Maximum of inital ion temperature");
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
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");

   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }
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

   if (amr_weights.Size() != 5)
   {
      amr_weights.SetSize(5);
      amr_weights = 1.0;
   }

   if (ode_weights.Size() != 5)
   {
      ode_weights.SetSize(5);
      ode_weights = 1.0;
      ode_weights[0] = 1e-8;
      ode_weights[4] = 1e-10;
   }

   if (vis_flags.Size() != 5)
   {
      vis_flags.SetSize(5);
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

   // 3. Read the mesh from the given mesh file. This example requires a 2D
   //    periodic mesh, such as ../data/periodic-square.mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   const int dim = mesh->Dimension();
   const int sdim = mesh->SpaceDimension();

   MFEM_ASSERT(dim == 2, "This miniapp is specialized to 2D geometries.");

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

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }

   // 7. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   ParFiniteElementSpace fes(&pmesh, &fec);
   ParFiniteElementSpace vfes(&pmesh, &fec, 2);

   // Adaptively refine mesh to accurately represent a given coefficient
   {
      ParGridFunctionArray coef_gf(5, &fes);
      Array<Coefficient*> coef(5);
      coef[0] = new FunctionCoefficient(nnFunc);
      coef[1] = new FunctionCoefficient(niFunc);
      coef[2] = new FunctionCoefficient(viFunc);
      coef[3] = new FunctionCoefficient(TiFunc);
      coef[4] = new FunctionCoefficient(TeFunc);

      coef_gf.ProjectCoefficient(coef);

      //Array<double> w(5); w = 1.0;
      /*
      w[0] = 1.0 / nn_max_;
      w[1] = 1.0 / ni_max_;
      w[2] = 1.0 / v_max_;
      w[3] = 1.0 / T_max_;
      w[4] = 1.0 / T_max_;
      */
      L2_FECollection fec_l2_o0(0, dim);
      // Finite element space for a scalar (thermodynamic quantity)
      ParFiniteElementSpace err_fes(&pmesh, &fec_l2_o0);

      AdaptInitialMesh(mpi, pmesh, err_fes, fes, vfes, coef_gf, coef,
                       amr_weights, 2, tol_init, visualization);

      for (int i=0; i<5; i++)
      {
         delete coef[i];
         delete coef_gf[i];
      }
   }

   // Finite element space for all variables together (full thermodynamic state)
   int num_equations = 5;
   ParFiniteElementSpace ffes(&pmesh, &fec, num_equations, Ordering::byNODES);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(ffes.GetOrdering() == Ordering::byNODES, "");

   HYPRE_Int glob_size_sca = fes.GlobalTrueVSize();
   HYPRE_Int glob_size_tot = ffes.GlobalTrueVSize();
   if (mpi.Root())
   { cout << "Number of unknowns per field: " << glob_size_sca << endl; }
   if (mpi.Root())
   { cout << "Total number of unknowns:     " << glob_size_tot << endl; }

   //ConstantCoefficient nuCoef(diffusion_constant_);
   // MatrixFunctionCoefficient nuCoef(dim, ChiFunc);

   // 8. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.

   // The solution u has components {particle density, x-velocity,
   // y-velocity, temperature} for each species (species loop is the outermost).
   // These are stored contiguously in the BlockVector u_block.
   Array<int> offsets(num_equations + 1);
   for (int k = 0; k <= num_equations; k++)
   {
      offsets[k] = k * fes.GetNDofs();
   }
   ParGridFunction u(&ffes);
   // BlockVector u_block(u.GetData(), offsets);
   /*
   Array<int> n_offsets(num_species_ + 2);
   for (int k = 0; k <= num_species_ + 1; k++)
   {
      n_offsets[k] = offsets[k];
   }
   BlockVector n_block(u_block, n_offsets);
   */
   /*
   ParGridFunction density(&fes, u_block.GetData());
   ParGridFunction velocity(&dfes, u_block.GetData() + offsets[1]);
   ParGridFunction temperature(&fes, u_block.GetData() + offsets[dim+1]);
   */
   /*
   // Initialize the state.
   VectorFunctionCoefficient u0(num_equations_, InitialCondition);
   ParGridFunction sol(&ffes, u_block.GetData());
   sol.ProjectCoefficient(u0);

   VectorFunctionCoefficient BCoef(dim, bFunc);
   ParGridFunction B(&fes_rt);
   B.ProjectCoefficient(BCoef);
   */
   // Output the initial solution.
   /*
   {
      ostringstream mesh_name;
      mesh_name << "transport-mesh." << setfill('0') << setw(6)
      << mpi.WorldRank();
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << pmesh;

      for (int i = 0; i < num_species_; i++)
   for (int j = 0; j < dim + 2; j++)
   {
      int k = 0;
      ParGridFunction uk(&sfes, u_block.GetBlock(k));
      ostringstream sol_name;
      sol_name << "species-" << i << "-field-" << j << "-init."
          << setfill('0') << setw(6) << mpi.WorldRank();
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << uk;
   }
   }
   */

   // 9. Set up the nonlinear form corresponding to the DG discretization of the
   //    flux divergence, and assemble the corresponding mass matrix.
   /*
   MixedBilinearForm Aflux(&dfes, &fes);
   Aflux.AddDomainIntegrator(new DomainIntegrator(dim, num_equations_));
   Aflux.Assemble();

   ParNonlinearForm A(&vfes);
   RiemannSolver rsolver(num_equations_, specific_heat_ratio_);
   A.AddInteriorFaceIntegrator(new FaceIntegrator(rsolver, dim,
                    num_equations_));

   // 10. Define the time-dependent evolution operator describing the ODE
   //     right-hand side, and perform time-integration (looping over the time
   //     iterations, ti, with a time-step dt).
   AdvectionTDO adv(vfes, A, Aflux.SpMat(), num_equations_,
                    specific_heat_ratio_);
   DiffusionTDO diff(fes, dfes, vfes, nuCoef, dg_sigma_, dg_kappa_);
   */
   // TransportSolver transp(ode_imp_solver, ode_exp_solver, sfes, vfes, ffes,
   //                     n_block, B, ion_charges, ion_masses);
   /*
   // Visualize the density, momentum, and energy
   vector<socketstream> dout(num_species_+1), vout(num_species_+1),
          tout(num_species_+1), xout(num_species_+1), eout(num_species_+1);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      int Wx = 0, Wy = 0; // window position
      int Ww = 275, Wh = 250; // window size
      int offx = Ww + 3, offy = Wh + 25; // window offsets

      MPI_Barrier(pmesh.GetComm());

      for (int i=0; i<=num_species_; i++)
      {
         int doff = offsets[i];
         int voff = offsets[i * dim + num_species_ + 1];
         int toff = offsets[i + (num_species_ + 1) * (dim + 1)];
         double * u_data = u_block.GetData();
         ParGridFunction density(&fespace, u_data + doff);
         ParGridFunction velocity(&vfes, u_data + voff);
         ParGridFunction temperature(&fespace, u_data + toff);
   */
   /*
        ParGridFunction chi_para(&sfes);
        ParGridFunction eta_para(&sfes);
        if (i==0)
        {
           ChiParaCoefficient chiParaCoef(n_block, ion_charges);
           chiParaCoef.SetT(temperature);
           chi_para.ProjectCoefficient(chiParaCoef);

           EtaParaCoefficient etaParaCoef(n_block, ion_charges);
           etaParaCoef.SetT(temperature);
           eta_para.ProjectCoefficient(etaParaCoef);
        }
        else
        {
           ChiParaCoefficient chiParaCoef(n_block, i - 1,
                                          ion_charges,
                                          ion_masses);
           chiParaCoef.SetT(temperature);
           chi_para.ProjectCoefficient(chiParaCoef);

           EtaParaCoefficient etaParaCoef(n_block, i - 1,
                                          ion_charges,
                                          ion_masses);
           etaParaCoef.SetT(temperature);
           eta_para.ProjectCoefficient(etaParaCoef);
        }
   */
   /*
         ostringstream head;
         if (i==0)
         {
            head << "Electron";
         }
         else
         {
            head << "Species " << i;
         }

         ostringstream doss;
         doss << head.str() << " Density";
         VisualizeField(dout[i], vishost, visport,
                        density, doss.str().c_str(),
                        Wx, Wy, Ww, Wh);
         Wx += offx;

         ostringstream voss; voss << head.str() << " Velocity";
         VisualizeField(vout[i], vishost, visport,
                        velocity, voss.str().c_str(),
                        Wx, Wy, Ww, Wh, NULL, true);
         Wx += offx;

         ostringstream toss; toss << head.str() << " Temperature";
         VisualizeField(tout[i], vishost, visport,
                        temperature, toss.str().c_str(),
                        Wx, Wy, Ww, Wh);
   */
   /*
   Wx += offx;

   ostringstream xoss; xoss << head.str() << " Chi Parallel";
   VisualizeField(xout[i], vishost, visport,
                  chi_para, xoss.str().c_str(),
                  Wx, Wy, Ww, Wh);

   Wx += offx;

   ostringstream eoss; eoss << head.str() << " Eta Parallel";
   VisualizeField(eout[i], vishost, visport,
                  eta_para, eoss.str().c_str(),
                  Wx, Wy, Ww, Wh);

   Wx -= 4 * offx;
   */
   /*
         Wx -= 2 * offx;
         Wy += offy;
      }
   }
   */

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
   MaxLimiter   dt_max(lim_max);

   /*
   ODESolver * ode_solver = NULL;
   switch (ode_solver_type)
   {
      // Implicit-Explicit methods
      case 1:  ode_solver = new IMEX_BE_FE; break;
      case 2:  ode_solver = new IMEXRK2; break;
      // Implicit L-stable methods
      case 11: ode_solver = new BackwardEulerSolver; break;
      case 12: ode_solver = new SDIRK23Solver(2); break;
      case 13: ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 34: ode_solver = new SDIRK34Solver; break;
      default:
         if (mpi.Root())
         {
            cout << "Unknown Implicit ODE solver type: "
                 << ode_solver_type << '\n';
         }
         return 3;
   }
   */
   ODEEmbeddedSolver * ode_solver   = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new SDIRK212Solver; break;
      case 2: ode_solver = new SDIRK534Solver; break;
   }
   /*
   ParBilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   m.Assemble();
   m.Finalize();
   */
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

   ParGridFunctionArray kGF;
   for (int i=0; i<5; i++)
   {
      kGF.Append(new ParGridFunction(&fes, (double*)NULL));
   }

   // ParGridFunction u(&fes);
   // u.ProjectCoefficient(u0Coef);
   // neu_density = 1.0e16;
   // ion_density = 1.0e19;
   para_velocity = 0.0;

   // Array<double> weights(5);
   // weights = 1.0;
   // weights[0] = 1.0;
   // weights[4] = 0.0;
   VectorRelativeErrorMeasure ode_diff_msr(MPI_COMM_WORLD, ode_weights);
   // ode_diff_msr.SetOperator(m);

   // Coefficients representing primary fields
   GridFunctionCoefficient nnCoef(&neu_density);
   GridFunctionCoefficient niCoef(&ion_density);
   GridFunctionCoefficient viCoef(&para_velocity);
   GridFunctionCoefficient TiCoef(&ion_energy);
   GridFunctionCoefficient TeCoef(&elec_energy);

   // Coefficients representing secondary fields
   ProductCoefficient      neCoef(plasma.z_i, niCoef);
   ConstantCoefficient     vnCoef(sqrt(8.0 * plasma.T_n * eV_ /
                                       (M_PI * plasma.m_n * amu_)));
   GridFunctionCoefficient veCoef(&para_velocity); // TODO: define as vi - J/q

   VectorFunctionCoefficient B3Coef(3, TotBFunc);

   // Intermediate Coefficients
   VectorFunctionCoefficient bHatCoef(2, bHatFunc);
   MatrixFunctionCoefficient perpCoef(2, perpFunc);
   // ProductCoefficient          mnCoef(ion_mass * amu_, niCoef); // ???
   ConstantCoefficient         mnCoef(plasma.m_i * amu_);
   ProductCoefficient        nnneCoef(nnCoef, neCoef);
   ApproxIonizationRate        izCoef(TeCoef);
   ConstantCoefficient     DiPerpCoef(Di_perp);
   ConstantCoefficient     XiPerpCoef(Xi_perp);
   ConstantCoefficient     XePerpCoef(Xe_perp);
   ThermalDiffusionCoef        XiCoef(dim, plasma.z_i, plasma.m_i,
                                      XiPerpCoef, niCoef, TiCoef);
   ThermalDiffusionCoef        XeCoef(dim, plasma.z_i, me_u_,
                                      XePerpCoef, neCoef, TeCoef, &niCoef);

   // Advection Coefficients
   ScalarVectorProductCoefficient   ViCoef(viCoef, bHatCoef);
   ScalarVectorProductCoefficient   VeCoef(veCoef, bHatCoef);
   ScalarVectorProductCoefficient  MomCoef(mnCoef, ViCoef);

   // Diffusion Coefficients
   NeutralDiffusionCoef     DnCoef(neCoef, vnCoef, izCoef);
   IonDiffusionCoef         DiCoef(DiPerpCoef, B3Coef);
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

   // Coefficients for initial conditions
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


   vector<CoefficientByAttr>  Ti_dbc;
   if (prob_ == 1)
   {
      Ti_dbc.resize(2);
      Ti_dbc[0].attr.Append(1);
      Ti_dbc[0].coef = new ConstantCoefficient(Ti_max_);
      Ti_dbc[1].attr.Append(2);
      Ti_dbc[1].coef = new ConstantCoefficient(Ti_min_);
   }

   vector<CoefficientByAttr>  Te_dbc;
   if (prob_ == 1)
   {
      Te_dbc.resize(2);
      Te_dbc[0].attr.Append(1);
      Te_dbc[0].coef = new ConstantCoefficient(Te_max_);
      Te_dbc[1].attr.Append(2);
      Te_dbc[1].coef = new ConstantCoefficient(Te_min_);
   }

   Array<double> coefNrm(5);
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

   DGTransportTDO oper(mpi, dg, plasma, fes, vfes, ffes, offsets, yGF, kGF,
                       Di_perp, Xi_perp, Xe_perp, B3Coef, Ti_dbc, Te_dbc,
                       vis_flags, imex, op_flag, logging);

   oper.SetLogging(max(0, logging - (mpi.Root()? 0 : 1)));

   if (visualization)
   {
      oper.InitializeGLVis();
   }
   /*
   oper.SetNnDiffusionCoefficient(DnCoef);
   oper.SetNnSourceCoefficient(SnCoef);

   oper.SetNiDiffusionCoefficient(DiCoef);
   oper.SetNiAdvectionCoefficient(ViCoef);
   oper.SetNiSourceCoefficient(SiCoef);
   */
   /*
   oper.SetViDiffusionCoefficient(EtaCoef);
   oper.SetViAdvectionCoefficient(MomCoef);
   oper.SetViSourceCoefficient(SMomCoef);

   oper.SetTiDiffusionCoefficient(XiCoef);
   oper.SetTiAdvectionCoefficient(ViCoef);
   oper.SetTiSourceCoefficient(QiCoef);

   oper.SetTeDiffusionCoefficient(XeCoef);
   oper.SetTeAdvectionCoefficient(VeCoef);
   oper.SetTeSourceCoefficient(QeCoef);

   Array<int> dbcAttr(pmesh.bdr_attributes.Max());
   dbcAttr = 1;
   oper.SetViDirichletBC(dbcAttr, vi0Coef);
   oper.SetTiDirichletBC(dbcAttr, Ti0Coef);
   oper.SetTeDirichletBC(dbcAttr, Te0Coef);
   */
   oper.SetTime(0.0);
   ode_solver->Init(oper);

   ode_controller.Init(*ode_solver, ode_diff_msr,
                       dt_acc, dt_rej, dt_max);

   ode_controller.SetOutputFrequency(vis_steps);
   ode_controller.SetTimeStep(dt);
   ode_controller.SetTolerance(tol_ode);
   ode_controller.SetRejectionLimit(rej_ode);
   if (ode_epus) { ode_controller.SetErrorPerUnitStep(); }

   ofstream ofs_controller;
   if (mpi.Root())
   {
      ofs_controller.open("transport2d_cntrl.dat");
      ode_controller.SetOutput(ofs_controller);
   }

   L2_FECollection fec_l2_o0(0, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   ParFiniteElementSpace fes_l2_o0(&pmesh, &fec_l2_o0);

   ParGridFunction err(&fes_l2_o0, (double *)NULL);

   // 11. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     with L2 projection in the smoothing step to better handle hanging
   //     nodes and parallel partitioning. We need to supply a space for the
   //     discontinuous flux (L2) and a space for the smoothed flux (H(div) is
   //     used here).
   L2_FECollection flux_fec(order, dim);
   ParFiniteElementSpace flux_fes(&pmesh, &flux_fec, sdim);
   RT_FECollection smooth_flux_fec(order-1, dim);
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

   dCoefs[0] = &DnCoef;
   DCoefs[1] = &DiCoef;
   DCoefs[2] = &EtaCoef;
   DCoefs[3] = &nXiCoef;
   DCoefs[4] = &nXeCoef;

   Vector estWeights(5);
   for (int i=0; i<5; i++) { estWeights[i] = amr_weights[i] / coefNrm[i]; }

   VectorL2ZZErrorEstimator estimator(yGF, fes_l2_o0, flux_fes, smooth_flux_fes,
                                      estWeights, dCoefs, DCoefs);

   if (max_elem_error < 0.0)
   {
      const Vector init_errors = estimator.GetLocalErrors();

      double loc_max_error = init_errors.Max();
      double loc_min_error = init_errors.Min();

      double glb_max_error = -1.0;
      double glb_min_error = -1.0;

      MPI_Allreduce(&loc_max_error, &glb_max_error, 1,
                    MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&loc_min_error, &glb_min_error, 1,
                    MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

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
   int dref_it = 0;

   double t = t_init;

   if (mpi.Root()) { cout << "\nBegin time stepping at t = " << t << endl; }
   while (t < t_final)
   {
      ode_controller.Run(u, t, t_final);

      if (mpi.Root()) { cout << "Time stepping paused at t = " << t << endl; }

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
         oss << "Parallel Ion Velocity at time " << t;
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
         dc->SetCycle(cycle);
         dc->SetTime(t);
         dc->Save();
      }

      if (t_final - t > 1e-8 * (t_final - t_init))
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

         if (visualization)
         {
            err.MakeRef(&fes_l2_o0,
                        const_cast<double*>(&(estimator.GetLocalErrors())[0]));
            ostringstream oss;
            oss << "Error estimate at time " << t;
            VisualizeField(eout, vishost, visport, err, oss.str().c_str(),
                           Wx + Ww + Dx, Wy, Ww, Wh);
         }

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
            fes.ExchangeFaceNbrData();
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
               pmesh.Rebalance();

               // Update the space and the GridFunction. This time the update matrix
               // redistributes the GridFunction among the processors.
               ffes.Update();
               vfes.Update();
               fes.Update();
               fes.ExchangeFaceNbrData();
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
            fes.ExchangeFaceNbrData();
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

      }
   }

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
   delete ode_solver;
   // delete ode_imp_solver;
   delete dc;

   return 0;
}

// Initial condition
void AdaptInitialMesh(MPI_Session &mpi,
                      ParMesh &pmesh, ParFiniteElementSpace &err_fespace,
                      ParFiniteElementSpace &fespace,
                      ParFiniteElementSpace &vfespace,
                      ParGridFunctionArray & gf, Array<Coefficient*> &coef,
                      Vector &weights,
                      int p, double tol, bool visualization)
{
   VectorLpErrorEstimator estimator(p, coef, gf, err_fespace, weights);

   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0);
   refiner.SetTotalErrorNormP(p);
   refiner.SetLocalErrorGoal(tol);

   Array<socketstream> sout(5);
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
