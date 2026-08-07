#ifndef HT_IMEX_HPP
#define HT_IMEX_HPP

#include "mfem.hpp"
#include <cmath>
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>
#include "ObjFunc.hpp"     // TimeIntegratedObjective (J, dJ/du)
#include "HeatTransferLinForms.hpp"
#include "HeatTransferTopOpt.hpp"
#include "../../pde_filter.hpp"


namespace mfem
{
class TopOptRKIMEXSolver : public ODESolver
{
protected:
    TimeDependentOperator *f;
    int num_stages;
    mfem::Array2D<real_t> A_ex; // A_ex must be num_stages+1 x num_stages + 1
    mfem::Array2D<real_t> A_imp;
    Vector b_ex;
    Vector b_imp;  
    std::vector<Vector> ks_ex;
    std::vector<Vector> ks_imp;
    std::vector<Vector> dks_ex;
    std::vector<Vector> dks_imp;
    Vector k;
    Vector y; // helper
public:
    void SetButcherTable(mfem::Array2D<real_t> &A_ex_, mfem::Array2D<real_t> &A_imp_, Vector &b_ex_, Vector &b_imp_);
    void Init(TimeDependentOperator &f_) override;
    // void AdjointStep(Vector &lam, real_t &t, real_t &dt, Vector &x) = 0;
    void Step(Vector &x, real_t &t, real_t &dt) override;
    static MFEM_EXPORT std::unique_ptr<TopOptRKIMEXSolver> SelectTopOptRKIMEX(const int ode_solver_type);
};

void TopOptRKIMEXSolver::SetButcherTable(mfem::Array2D<real_t> &A_ex_, mfem::Array2D<real_t> &A_imp_, Vector &b_ex_, Vector &b_imp_)
{ 
   A_ex = A_ex_;
   A_imp = A_imp_;
   b_ex = b_ex_;
   b_imp = b_imp_;
}

void TopOptRKIMEXSolver::Init(TimeDependentOperator &f_)
{
   this->f = &f_;
   mem_type = GetMemoryType(f_.GetMemoryClass());
//    num_stages_ex = A_ex.NumCols();
   // num_stages = A_imp.NumCols();
   int n = f->Width();
   k.SetSize(n, mem_type);
   y.SetSize(n, mem_type);
}


void TopOptRKIMEXSolver::Step(Vector &x, real_t &t, real_t &dt)
{
   f->SetTime(t);
   f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
   f->Mult(x, k);
   Vector x_old = x;
   ks_ex.push_back(k);
   if(Mpi::Root()){std::cout<<"x norm = " << x.Norml2() << std::endl;}
   x.Add(dt*b_ex(0), ks_ex[0]);
   if(Mpi::Root()){std::cout<<"ks_ex 0 norm = " << ks_ex[0].Norml2() << std::endl;}
   if(Mpi::Root()){std::cout<<"x norm = " << x.Norml2() << std::endl;}

   for (int stage = 0; stage < num_stages; stage++)
   {
      f->SetTime(t+dt);
      y = x_old;
      for (int j = 0; j < stage; j++)
      {
         y.Add(dt*A_ex(stage+1, j), ks_ex[j]);
         y.Add(dt*A_imp(stage, j), ks_imp[j]); 
      }
      y.Add(dt*A_ex(stage+1, stage), ks_ex[stage]);
      if(Mpi::Root()){std::cout<<"y pre computation norm = " << y.Norml2() << std::endl;}
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
      f->ImplicitSolve(dt, y, k);
      ks_imp.push_back(k);
      if(Mpi::Root()){std::cout<<"k_imp 0 norm = " << ks_imp[stage].Norml2() << std::endl;}
      y.Add(dt*A_imp(stage, stage), ks_imp[stage]);
      f->SetTime(t);
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
      f->Mult(y, k);
      ks_ex.push_back(k);
      x.Add(dt*b_ex(stage+1), ks_ex[stage+1]);
      if(Mpi::Root()){std::cout<<"k_ex 1 norm = " << ks_ex[stage+1].Norml2() << std::endl;}
      if(Mpi::Root()){std::cout<<"x norm = " << x.Norml2() << std::endl;}
      x.Add(dt*b_imp(stage), ks_imp[stage]);
      if(Mpi::Root()){std::cout<<"x norm = " << x.Norml2() << std::endl;}
   }
   t += dt;
}

class TopOptRKIMEXExpImplEuler : public TopOptRKIMEXSolver
{

   public:
      TopOptRKIMEXExpImplEuler()
      {
         num_stages = 1;
         // mfem::Array2D<real_t> A_ex_EI(num_stages+1, num_stages+1);
         // mfem::Array2D<real_t> A_imp_EI(num_stages, num_stages);
         // Vector b_ex_EI(num_stages+1);
         // Vector b_imp_EI(num_stages);
         A_ex.SetSize(num_stages+1, num_stages+1);
         A_imp.SetSize(num_stages, num_stages);
         b_ex.SetSize(num_stages+1);
         b_imp.SetSize(num_stages);

         A_ex(0,0) = 0.0; A_ex(0,1) = 0.0; A_ex(1,0) = 1.0; A_ex(1,1) = 0.0;
         A_imp(0,0) = 1.0;
         b_ex(0) = 1.0; b_ex(1) = 0.0;
         b_imp(0) = 1.0;
         //SetButcherTable(A_ex_EI, A_imp_EI, b_ex_EI, b_imp_EI);
      }
};

std::unique_ptr<TopOptRKIMEXSolver> TopOptRKIMEXSolver::SelectTopOptRKIMEX(const int ode_solver_type)
{
   using ode_ptr = std::unique_ptr<TopOptRKIMEXSolver>;
   switch (ode_solver_type)
   {
      // L-stable IMEX methods for design opt
      case 1: return ode_ptr(new TopOptRKIMEXExpImplEuler);
      // case 2: return ode_ptr(new TopOptIMEXRK2);

      default: MFEM_ABORT("Unknown ODE solver type: " << ode_solver_type );
   }
}
}
#endif 
