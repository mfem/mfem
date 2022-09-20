#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


/** The purpose of this class is to perform an H1-projection with 
    box constraints using a mixed method 
    We want to solve the following minimazation problem
    u^* = argmin_{u ∈ H^1 and 0<=u<=1} ||u-g||^2_H^1
    using (iteratively) the mixed formulation 
    (α + β) (u,v)_H1 + (ψ,v) = (α g + β u_k,v)_H1 + (ψ_k,v)
        (u,w) - (expit(ψ),w) = 0 */


class BoxProjection
{
private:
   ParMesh * pmesh = nullptr;
   int order = 1;
   int dim;
   Coefficient * g_cf = nullptr;
   VectorCoefficient * grad_g_cf = nullptr;
   double alpha = 1.0;
   double beta = 1.0;

   FiniteElementCollection * H1fec = nullptr;
   FiniteElementCollection * L2fec = nullptr;

   ParFiniteElementSpace * H1pfes = nullptr;
   ParFiniteElementSpace * L2pfes = nullptr;

   ParGridFunction u_gf, psi_gf;

   // Bilinear and linear forms
   ParBilinearForm * a_H1H1 = nullptr;
   ParMixedBilinearForm * a_H1L2 = nullptr;
   const Operator * P_H1 = nullptr;
   const Operator * R_H1 = nullptr;

   ParLinearForm * b_H1 = nullptr;
   HypreParMatrix *A_H1H1 = nullptr;
   HypreParMatrix *A_H1L2 = nullptr;
   HypreParMatrix *A_L2H1 = nullptr;

   Array<int> offsets, toffsets;
   BlockVector x, tx, rhs, trhs;

   int max_newton_it = 10;
   double newton_tol = 1e-5;
   double NewtonStep(const ParLinearForm & b_H1_, ParGridFunction & psi_kl_gf, ParGridFunction & u_kl);

   double BregmanStep(ParGridFunction & u_gf_, ParGridFunction & psi_gf_);

public:
   BoxProjection(ParMesh* pmesh_, int order_, Coefficient * g_cf_, VectorCoefficient * grad_g_cf_);
   
   void Solve();

   ~BoxProjection();
};

double lnit(double x)
{
   double tol = 1e-12;
   x = min(max(tol,x),1.0-tol);
   // MFEM_ASSERT(x>0.0, "Argument must be > 0");
   // MFEM_ASSERT(x<1.0, "Argument must be < 1");
   return log(x/(1.0-x));
}

double expit(double x)
{
   if (x >= 0)
   {
      return 1.0/(1.0+exp(-x));
   }
   else
   {
      return exp(x)/(1.0+exp(x));
   }
}

double dexpitdx(double x)
{
   double tmp = expit(-x);
   return tmp - pow(tmp,2);
}

class LnitGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   LnitGridFunctionCoefficient(GridFunction &u_, double min_val_=-1e10, double max_val_=1e10)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ExpitGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   ExpitGridFunctionCoefficient(GridFunction &u_, double min_val_=0.0, double max_val_=1.0)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class dExpitdxGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   dExpitdxGridFunctionCoefficient(GridFunction &u_, double min_val_=0.0, double max_val_=1.0)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};
