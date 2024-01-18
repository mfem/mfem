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
   double theta = 1.0;
   double epsilon;
   double gamma = 0.0;
   double print_level = 1;

   FiniteElementCollection * u_fec = nullptr;
   FiniteElementCollection * p_fec = nullptr;

   ParFiniteElementSpace * u_fes = nullptr;
   ParFiniteElementSpace * p_fes = nullptr;

   ParGridFunction u_gf, p_gf;

   // Bilinear and linear forms
   ParBilinearForm * a_uu = nullptr;
   ParMixedBilinearForm * a_up = nullptr;
   const Operator * P_u = nullptr;
   const Operator * P_p = nullptr;

   ParLinearForm * b_u = nullptr;
   HypreParMatrix *A_uu = nullptr;
   HypreParMatrix *A_up = nullptr;
   HypreParMatrix *A_pu = nullptr;

   Array<int> offsets, toffsets;
   // BlockVector x, tx, rhs, trhs;
   int max_bregman_it = 200;
   double bregman_tol = 1e-5;

   int bregman_cntr = 0;
   int newton_cntr = 0;

   int max_newton_it = 10;
   double newton_tol = 1e-8;
   double NewtonStep(const ParLinearForm & b_u_, ParGridFunction & p_kl_gf, ParGridFunction & u_kl);

   double BregmanStep(ParGridFunction & u_gf_, ParGridFunction & p_gf_);

   void Update_A_uu(ParGridFunction & u_gf_);

public:
   BoxProjection(ParMesh* pmesh_, int order_, Coefficient * g_cf_,
                 VectorCoefficient * grad_g_cf_, bool H1H1 = false);
   
   void Solve();

   ParGridFunction & Getu()
   {
      return u_gf;
   }
   ParGridFunction & Getp()
   {
      return p_gf;
   }

   void SetNewtonStepSize(double theta_) {theta = theta_;}
   void SetBregmanStepSize(double alpha_) {alpha = alpha_;}
   void SetNormWeight(double beta_) {beta = beta_;}

   void SetDiffusionConstant(double epsilon_) {epsilon = epsilon_;}
   void SetPenaltyConstant(double gamma_) {gamma = max(0.0,gamma_);}

   void SetOuterIterationTol(double bregman_tol_) {bregman_tol = bregman_tol_;}
   void SetInnerIterationTol(double newton_tol_) {newton_tol = newton_tol_;}
   void SetMaxOuterIterations(double max_bregman_it_) {max_bregman_it = max_bregman_it_;}
   void SetMaxInnerIterations(double max_newton_it_) {max_newton_it = max_newton_it_;}

   void SetPrintLevel(int print_level_) {print_level = print_level_;}

   ~BoxProjection();
};

double lnit(double x);

double expit(double x);

double dexpitdx(double x);

class LnitGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   LnitGridFunctionCoefficient(GridFunction &u_, double min_val_=-1e1, double max_val_=1e1)
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

public:
   dExpitdxGridFunctionCoefficient(GridFunction &u_, double min_val_=1e-8)
      : u(&u_), min_val(min_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class BoxFunctionCoefficient : public Coefficient
{
protected:
   Coefficient &u_cf;
   double min_val;
   double max_val;
public:
   BoxFunctionCoefficient(Coefficient &u_cf_, double min_val_=0.0, double max_val_=1.0)
      : u_cf(u_cf_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return min(max_val, max(min_val, u_cf.Eval(T,ip)));
   }
};

class ActiveSetCoefficient : public Coefficient
{
protected:
   ParGridFunction &u_gf;
public:
   ActiveSetCoefficient(ParGridFunction &u_gf_)
      : u_gf(u_gf_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      bool active_point = (u_gf.GetValue(T,ip) < 0);
      // bool active_point = (u_gf.GetValue(T,ip) < 0) || (u_gf.GetValue(T,ip) - 1 > 0);
      if (u_gf.GetValue(T,ip) < 0)
      {
         return 0.0;
      }
      else
      {
         return 0.0;
      }
   }
};

// // (α + β) (u,v)_H1 + (ψ,v) + (u,w) - (expit(ψ),w)
// class LHS : public Operator
// {
//    double alpha;
//    double beta;
//    ParMesh* pmesh;
//    int order;

//    protected:

//    public:
//    LHS(ParMesh* pmesh_, int order_, double alpha_, double beta_);
   
// }