#ifndef HT_IMEX_HPP
#define HT_IMEX_HPP

#include "mfem.hpp"
#include <cmath>
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>
#include "HeatTransferSolvers.hpp"
#include "../../pde_filter.hpp"


namespace mfem
{
class TopOptRKIMEXSolver : public ODESolver
{
protected:
    IMEXAdvectionDiffusionSolver *f;
    int num_stages;
    mfem::Array2D<real_t> A_ex; // A_ex must be num_stages+1 x num_stages + 1
    mfem::Array2D<real_t> A_imp;
    Vector b_ex;
    Vector b_imp;  
    Vector c_ex;
    Vector c_imp;
    std::vector<Vector> ks_ex;
    std::vector<Vector> ks_imp;
    std::vector<Vector> lks_ex;
    std::vector<Vector> lks_imp;
    std::vector<Vector> dks_ex;
    std::vector<Vector> dks_imp;
    std::vector<Vector> x_stages; // for adjoint computation
    Vector k;
    Vector y, yd, yl; // helpers
public:
    void SetButcherTable(mfem::Array2D<real_t> &A_ex_, mfem::Array2D<real_t> &A_imp_, Vector &b_ex_, Vector &b_imp_);
    void Init(IMEXAdvectionDiffusionSolver &f_);
    void AdjointStep(Vector &lam, Vector &x,Vector &dJdrho_tilde, real_t &t, real_t &dt);
    Vector ComboAdjointMult(real_t a1, real_t a2, real_t dt, Vector &x, real_t t, real_t ce, real_t ci, real_t bi);

    void ComputeBaseGradient(Vector &v, Vector &x_old, real_t dt,real_t t, Vector &out);
    void StageChainRule(Vector &v, Vector &x_old, int idx, real_t dt, real_t t, Vector &out);
    void DesignGradStage(Vector &v,Vector &x_old, int idx, real_t dt, real_t t, Vector &out);

    void ComputeAdjointStage(Vector &v, int idx, real_t dt, real_t t, Vector &out);



    void Step(Vector &x, real_t &t, real_t &dt);
    static MFEM_EXPORT std::unique_ptr<TopOptRKIMEXSolver> SelectTopOptRKIMEX(const int ode_solver_type);
};

void TopOptRKIMEXSolver::SetButcherTable(mfem::Array2D<real_t> &A_ex_, mfem::Array2D<real_t> &A_imp_, Vector &b_ex_, Vector &b_imp_)
{ 
   A_ex = A_ex_;
   A_imp = A_imp_;
   b_ex = b_ex_;
   b_imp = b_imp_;
}

Vector TopOptRKIMEXSolver::ComboAdjointMult(real_t a1, real_t a2, real_t dt, Vector &x, real_t t, real_t ce, real_t ci, real_t bi)
{
   Vector rhs(f->Width());
   Vector rhs2(f->Width());
   f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
   f->SetTime(t + dt*ce);
   f->AdjointMult(x, rhs2);
   rhs2 *= a1;

   f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   f->SetTime(t + ci*dt);
   f->AdjointImplicitSolve(dt, x, rhs);
   f->SetTime(t);
   rhs *= a2;
   rhs.Add(1.0, rhs2);
   //rhs.Add(1.0, x);
   return rhs;
}

void TopOptRKIMEXSolver::ComputeBaseGradient(Vector &v, Vector &x_old, real_t dt, real_t t, Vector &out)
{
   f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
   f->SetTime(t + c_ex(0)*dt);
   f->ExplicitMultDesignGradient(dt*A_ex(1,0), v, x_old, out);
   f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   f->SetTime(t + c_imp(1)*dt);
   f->ImplicitSolveDesignGradient(dt, dt*A_imp(1,1), v, x_old, out);
   f->SetTime(t);
}

void TopOptRKIMEXSolver::StageChainRule(Vector &v, Vector &x_old, int idx, real_t dt, real_t t, Vector &out)
{
   Vector yy(x_old.Size());
   for(int jj = 0; jj < idx; jj++)
   {
      yy = x_old;
      for (int ii = 0; ii < jj; ii++)
      {
         yy.Add(dt*A_ex(jj, ii), ks_ex[ii]);
         yy.Add(dt*A_imp(jj, ii+1), ks_imp[ii]); 
      }
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
      f->SetTime(t + c_ex(jj)*dt);
      f->ExplicitMultDesignGradient(dt*A_ex(idx,jj), v, yy, out);
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
      f->SetTime(t + c_imp(jj+1)*dt);
      f->ImplicitSolveDesignGradient(dt, dt*A_imp(idx,jj+1), v, yy, out);
      f->SetTime(t);
   }
}

void TopOptRKIMEXSolver::DesignGradStage(Vector &v, Vector &x_old, int idx, real_t dt, real_t t, Vector &out)
{
   if (idx == 1)
   {
      ComputeBaseGradient(v, x_old, dt, t, out);
   }
   else
   {
      Vector vv(v.Size());
      Vector psum(out.Size());
      for (int jj = idx-1; jj > 0; jj--)
      {
         psum = 0.0;
         vv = ComboAdjointMult(A_ex(idx, jj)*dt, A_imp(idx, jj+1)*dt, dt, v, t, c_ex(jj), c_imp(jj+1), b_imp(idx+1));
         DesignGradStage(vv,x_old, jj, dt,t, psum);
         out.Add(1.0, psum);
      }
      psum = 0.0;
      StageChainRule(v, x_old, idx, dt, t, psum);
      out.Add(1.0, psum);
   }
}

void TopOptRKIMEXSolver::ComputeAdjointStage(Vector &v, int idx, real_t dt, real_t t, Vector &out)
{
   if (idx == 1)
   {
      out = ComboAdjointMult(A_ex(1, 0)*dt, A_imp(1, 1)*dt, dt, v, t, c_ex(0), c_imp(1), b_imp(1));
      out.Add(1.0, v);
   }
   else
   {
      Vector vv(v.Size());
      Vector psum(out.Size());
      for (int jj = idx-1; jj > 0; jj--)
      {
         psum = 0.0;
         vv = ComboAdjointMult(A_ex(idx, jj)*dt, A_imp(idx, jj+1)*dt, dt, v, t, c_ex(jj), c_imp(jj+1), b_imp(1));
         ComputeAdjointStage(vv, jj, dt,t, psum);
         out.Add(1.0, psum);
      }
      psum = v;
      vv = ComboAdjointMult(A_ex(idx, 0)*dt, A_imp(idx, 1)*dt, dt, v, t, c_ex(0), c_imp(1), b_imp(1));
      psum.Add(1.0, vv);
      out.Add(1.0, psum);
   }
}





void TopOptRKIMEXSolver::Init(IMEXAdvectionDiffusionSolver &f_)
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
   Vector x_old = x;
   y = x_old;
   f->Mult(x, k);
   ks_ex.push_back(k);
   x.Add(dt*b_ex(0), ks_ex[0]);

   f->SetTime(t+c_imp(1)*dt);
   f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   f->ImplicitSolve(dt, x_old, k);
   ks_imp.push_back(k);
   f->SetTime(t);

   for (int stage = 0; stage < num_stages; stage++)
   {
      //f->SetTime(t+A_imp(stage+1, stage+1)*dt);
      y = x_old;
      for (int ii = 0; ii <= stage; ii++)
      {
         y.Add(dt*A_ex(stage+1, ii), ks_ex[ii]);
         y.Add(dt*A_imp(stage+1, ii+1), ks_imp[ii]); 
      }
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
      f->SetTime(t + c_imp(stage+1)*dt);
      f->ImplicitSolve(dt, y, k);
      ks_imp.push_back(k);
      // y.Add(dt*A_imp(stage, stage), ks_imp[stage]);
      f->SetTime(t + c_ex(stage+1)*dt);
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
      f->Mult(y, k);
      ks_ex.push_back(k);
      x.Add(dt*b_ex(stage+1), ks_ex[stage+1]);
      x.Add(dt*b_imp(stage+1), ks_imp[stage+1]);
      f->SetTime(t);
   }
   ks_ex.clear();
   ks_imp.clear();
   t += dt;
}

void TopOptRKIMEXSolver::AdjointStep(Vector &lam, Vector &x, Vector &dJdrho_tilde, real_t &t, real_t &dt)
{

    // adjoint computation 
    f->SetTime(t);
    Vector x_old = x;
    Vector y = x_old; 
    Vector y_adj(y.Size());
    y_adj = lam;
    f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
    Vector imp_l(lam.Size());
    Vector exp_l(lam.Size());
    Vector lam_old = lam;
    f->AdjointMult(lam_old, exp_l);
    lks_ex.push_back(exp_l);

    // state 
    f->Mult(x, k);
    ks_ex.push_back(k);
    x.Add(dt*b_ex(0), ks_ex[0]);
 
    f->SetTime(t+c_imp(0)*dt);
    f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
    f->ImplicitSolve(dt, x_old, k);
    ks_imp.push_back(k);
    f->SetTime(t);

    // design gradient
    Vector exp_grad_placeholder(dJdrho_tilde.Size());
    Vector imp_grad_placeholder(dJdrho_tilde.Size());
    exp_grad_placeholder = 0.0;
    f->ExplicitMultDesignGradient(1.0, lam_old, x_old, exp_grad_placeholder);
    dJdrho_tilde.Add(dt*b_ex(0), exp_grad_placeholder);

    f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
    f->SetTime(t + c_imp(1)*dt);
    f->AdjointImplicitSolve(dt, lam_old, imp_l);
    f->SetTime(t);
    lks_imp.push_back(imp_l);

    
    lam.Add(dt*b_ex(0), exp_l);
    for (int stage = 0; stage < num_stages; stage++)
    {
      y = x_old;
      // y_adj = lam_old;
      for (int ii = 0; ii <= stage; ii++)
      {
         y.Add(dt*A_ex(stage+1, ii), ks_ex[ii]);
         y.Add(dt*A_imp(stage+1, ii+1), ks_imp[ii]); 
         // y_adj.Add(dt*A_ex(stage+1, ii), lks_ex[ii]);
         // y_adj.Add(dt*A_imp(stage+1, ii+1), lks_imp[ii]); 
      }

      exp_grad_placeholder = 0.0;
      imp_grad_placeholder = 0.0;

      //gradient 
      // first term, grad wrt operator
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
      f->SetTime(t + c_ex(stage+1)*dt);
      f->ExplicitMultDesignGradient(1.0, lam_old, y, exp_grad_placeholder);
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
      f->SetTime(t + c_imp(stage+1)*dt);
      f->ImplicitSolveDesignGradient(dt, 1.0, lam_old, y, imp_grad_placeholder);
      f->SetTime(t);
      dJdrho_tilde.Add(dt*b_ex(stage+1), exp_grad_placeholder);
      dJdrho_tilde.Add(dt*b_imp(stage+1), imp_grad_placeholder);

      exp_grad_placeholder = 0.0;
      imp_grad_placeholder = 0.0;

      // Second term, grad wrt stage
      DesignGradStage(exp_l, x_old, stage+1, dt, t, exp_grad_placeholder);
      DesignGradStage(imp_l, x_old, stage+1, dt, t, imp_grad_placeholder);
      dJdrho_tilde.Add(dt*b_ex(stage+1), exp_grad_placeholder);
      dJdrho_tilde.Add(dt*b_imp(stage+1), imp_grad_placeholder);

      // adjoint
      y_adj = 0.0;
      ComputeAdjointStage(exp_l, stage+1, dt, t, y_adj);
      lam.Add(dt*b_ex(stage+1), y_adj);
      y_adj = 0.0;
      ComputeAdjointStage(imp_l, stage+1, dt, t, y_adj);
      lam.Add(dt*b_imp(stage+1), y_adj);
      y_adj = 0.0;


      // // adjoint
      // f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
      // f->SetTime(t + c_imp(stage+1)*dt);
      // f->AdjointImplicitSolve(dt, y_adj, k);
      // lks_imp.push_back(k);
      // // y.Add(dt*A_imp(stage, stage), ks_imp[stage]);
      // f->SetTime(t + c_ex(stage+1)*dt);
      // f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
      // f->AdjointMult(y_adj, k);
      // lks_ex.push_back(k);
      // lam.Add(dt*b_ex(stage+1), lks_ex[stage+1]);
      // lam.Add(dt*b_imp(stage+1), lks_imp[stage+1]);
      // f->SetTime(t);



      // state
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
      f->SetTime(t + c_imp(stage+1)*dt);
      f->ImplicitSolve(dt, y, k);
      ks_imp.push_back(k);
      // y.Add(dt*A_imp(stage, stage), ks_imp[stage]);
      f->SetTime(t + c_ex(stage+1)*dt);
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
      f->Mult(y, k);
      ks_ex.push_back(k);
      x.Add(dt*b_ex(stage+1), ks_ex[stage+1]);
      x.Add(dt*b_imp(stage+1), ks_imp[stage+1]);
      f->SetTime(t);
    }
    ks_ex.clear();
    ks_imp.clear();
    lks_ex.clear();
    lks_imp.clear();
    t += dt;
}

class TopOptRKIMEX_1_1_1 : public TopOptRKIMEXSolver
{

   public:
      TopOptRKIMEX_1_1_1()
      {
         num_stages = 1;
         A_ex.SetSize(num_stages+1, num_stages+1);
         A_imp.SetSize(num_stages+1, num_stages+1);
         b_ex.SetSize(num_stages+1);
         b_imp.SetSize(num_stages+1);
         c_ex.SetSize(num_stages+1);
         c_imp.SetSize(num_stages+1);

         A_ex(0,0) = 0.0; A_ex(0,1) = 0.0; A_ex(1,0) = 1.0; A_ex(1,1) = 0.0;
         A_imp(0,0) = 0.0; A_imp(0,1) = 0.0; A_imp(1,0) = 0.0; A_imp(1,1) = 1.0;
         b_ex(0) = 1.0; b_ex(1) = 0.0;
         b_imp(0) = 0.0; b_imp(1) = 1.0;

         c_ex(0) = 0.0; c_ex(1) = 1.0;
         c_imp(0) = 0.0; c_imp(1) = 1.0;

      }
};

class TopOptRKIMEX_1_2_1 : public TopOptRKIMEXSolver
{

   public:
      TopOptRKIMEX_1_2_1()
      {
         num_stages = 1;
         A_ex.SetSize(num_stages+1, num_stages+1);
         A_imp.SetSize(num_stages+1, num_stages+1);
         b_ex.SetSize(num_stages+1);
         b_imp.SetSize(num_stages+1);

         c_ex.SetSize(num_stages+1);
         c_imp.SetSize(num_stages+1);

         A_ex(0,0) = 0.0; A_ex(0,1) = 0.0; A_ex(1,0) = 1.0; A_ex(1,1) = 0.0;
         A_imp(0,0) = 0.0; A_imp(0,1) = 0.0; A_imp(1,0) = 0.0; A_imp(1,1) = 1.0;
         b_ex(0) = 0.0; b_ex(1) = 1.0;
         b_imp(0) = 0.0; b_imp(1) = 1.0;

         c_ex(0) = 0.0; c_ex(1) = 1.0;
         c_imp(0) = 0.0; c_imp(1) = 1.0;
      }
};

class TopOptRKIMEX_1_2_2 : public TopOptRKIMEXSolver
{

   public:
      TopOptRKIMEX_1_2_2()
      {
         num_stages = 1;
         A_ex.SetSize(num_stages+1, num_stages+1);
         A_imp.SetSize(num_stages+1, num_stages+1);
         b_ex.SetSize(num_stages+1);
         b_imp.SetSize(num_stages+1);
         c_ex.SetSize(num_stages+1);
         c_imp.SetSize(num_stages+1);

         A_ex(0,0) = 0.0; A_ex(0,1) = 0.0; A_ex(1,0) = 0.5; A_ex(1,1) = 0.0;
         A_imp(0,0) = 0.0; A_imp(0,1) = 0.0; A_imp(1,0) = 0.0; A_imp(1,1) = 0.5;
         b_ex(0) = 0.0; b_ex(1) = 1.0;
         b_imp(0) = 0.0; b_imp(1) = 1.0;

         c_ex(0) = 0.0; c_ex(1) = 0.5;
         c_imp(0) = 0.0; c_imp(1) = 0.5;
      }
};

class TopOptRKIMEX_2_3_3 : public TopOptRKIMEXSolver
{

   public:
      TopOptRKIMEX_2_3_3()
      {
         num_stages = 2;
         A_ex.SetSize(num_stages+1, num_stages+1);
         A_imp.SetSize(num_stages+1, num_stages+1);
         b_ex.SetSize(num_stages+1);
         b_imp.SetSize(num_stages+1);
         c_ex.SetSize(num_stages+1);
         c_imp.SetSize(num_stages+1);


         real_t gamma = (3.0 + sqrt(3.0)) / 6.0; 

         A_ex(0,0) = 0.0; A_ex(0,1) = 0.0; A_ex(0,2) = 0.0; 
         A_ex(1,0) = gamma; A_ex(1,1) = 0.0; A_ex(1,2) = 0.0;
         A_ex(2,0) = gamma - 1; A_ex(2,1) = 2.0*(1.0 - gamma); A_ex(2,2) = 0.0;

         A_imp(0,0) = 0.0; A_imp(0,1) = 0.0; A_imp(0,2) = 0.0; 
         A_imp(1,0) = 0.0; A_imp(1,1) = gamma; A_imp(1,2) = 0.0;
         A_imp(2,0) = 0.0; A_imp(2,1) = 1.0 - 2.0*gamma; A_imp(2,2) = gamma;

         b_ex(0) = 0.0; b_ex(1) = 0.5; b_ex(2) = 0.5;
         b_imp(0) = 0.0; b_imp(1) = 0.5; b_imp(2) = 0.5;

         c_ex(0) = 0.0; c_ex(1) = gamma; c_ex(2) = 1.0 - gamma;
         c_imp(0) = 0.0; c_imp(1) = gamma; c_imp(2) = 1.0 - gamma;
      }
};

class TopOptRKIMEX_2_3_2 : public TopOptRKIMEXSolver
{

   public:
      TopOptRKIMEX_2_3_2()
      {
         num_stages = 2;
         A_ex.SetSize(num_stages+1, num_stages+1);
         A_imp.SetSize(num_stages+1, num_stages+1);
         b_ex.SetSize(num_stages+1);
         b_imp.SetSize(num_stages+1);
         c_ex.SetSize(num_stages+1);
         c_imp.SetSize(num_stages+1);


         real_t gamma = (2.0 - sqrt(2.0)) / 2.0;
         real_t delta = -2.0*sqrt(2.0) / 3.0; 

         A_ex(0,0) = 0.0; A_ex(0,1) = 0.0; A_ex(0,2) = 0.0; 
         A_ex(1,0) = gamma; A_ex(1,1) = 0.0; A_ex(1,2) = 0.0;
         A_ex(2,0) = delta; A_ex(2,1) = 1.0 - delta; A_ex(2,2) = 0.0;

         A_imp(0,0) = 0.0; A_imp(0,1) = 0.0; A_imp(0,2) = 0.0; 
         A_imp(1,0) = 0.0; A_imp(1,1) = gamma; A_imp(1,2) = 0.0;
         A_imp(2,0) = 0.0; A_imp(2,1) = 1.0 - gamma; A_imp(2,2) = gamma;

         b_ex(0) = 0.0; b_ex(1) = 1.0 - gamma; b_ex(2) = gamma;
         b_imp(0) = 0.0; b_imp(1) = 1.0 - gamma; b_imp(2) = gamma;

         c_ex(0) = 0.0; c_ex(1) = gamma; c_ex(2) = 1.0;
         c_imp(0) = 0.0; c_imp(1) = gamma; c_imp(2) = 1.0;
      }
};

class TopOptRKIMEX_3_4_3 : public TopOptRKIMEXSolver
{

   public:
      TopOptRKIMEX_3_4_3()
      {
         num_stages = 3;
         A_ex.SetSize(num_stages+1, num_stages+1);
         A_imp.SetSize(num_stages+1, num_stages+1);
         b_ex.SetSize(num_stages+1);
         b_imp.SetSize(num_stages+1);
         c_ex.SetSize(num_stages+1);
         c_imp.SetSize(num_stages+1);

         A_ex(0,0) = 0.0; A_ex(0,1) = 0.0; A_ex(0,2) = 0.0; A_ex(0,3) = 0.0; 
         A_ex(1,0) = 0.4358665215; A_ex(1,1) = 0.0; A_ex(1,2) = 0.0; A_ex(1,3) = 0.0;
         A_ex(2,0) = 0.3212788860; A_ex(2,1) = 0.3966543747; A_ex(2,2) = 0.0; A_ex(2,3) = 0.0;
         A_ex(3,0) = -0.105858296; A_ex(3,1) = 0.5529291479; A_ex(3,2) = 0.5529291479; A_ex(3,3) = 0.0;

         A_imp(0,0) = 0.0; A_imp(0,1) = 0.0; A_imp(0,2) = 0.0; A_imp(0,3) = 0.0;
         A_imp(1,0) = 0.0; A_imp(1,1) = 0.4358665215; A_imp(1,2) = 0.0; A_imp(1,3) = 0.0;
         A_imp(2,0) = 0.0; A_imp(2,1) = 0.2820667392; A_imp(2,2) = 0.4358665215; A_imp(2,3) = 0.0;
         A_imp(3,0) = 0.0; A_imp(3,1) = 1.208496649; A_imp(3,2) = -0.644363171; A_imp(3,3) = 0.4358665215;

         b_ex(0) = 0.0; b_ex(1) = 1.208496649; b_ex(2) = -0.644363171; b_ex(3) = 0.4358665215;
         b_imp(0) = 0.0; b_imp(1) = 1.208496649; b_imp(2) = -0.644363171; b_imp(3) = 0.4358665215;

         c_ex(0) = 0.0; c_ex(1) = 0.4358665215; c_ex(2) = 0.7179332608; c_ex(3) = 1.0;
         c_imp(0) = 0.0; c_imp(1) = 0.4358665215; c_imp(2) = 0.7179332608; c_imp(3) = 1.0;
      }
};

class TopOptRKIMEX_4_4_3 : public TopOptRKIMEXSolver
{

   public:
      TopOptRKIMEX_4_4_3()
      {
         num_stages = 4;
         A_ex.SetSize(num_stages+1, num_stages+1);
         A_imp.SetSize(num_stages+1, num_stages+1);
         b_ex.SetSize(num_stages+1);
         b_imp.SetSize(num_stages+1);
         c_ex.SetSize(num_stages+1);
         c_imp.SetSize(num_stages+1);

         A_ex(0,0) = 0.0; A_ex(0,1) = 0.0; A_ex(0,2) = 0.0; A_ex(0,3) = 0.0; A_ex(0,4) = 0.0; 
         A_ex(1,0) = 0.5; A_ex(1,1) = 0.0; A_ex(1,2) = 0.0; A_ex(1,3) = 0.0; A_ex(1,4) = 0.0; 
         A_ex(2,0) = 11.0/18.0; A_ex(2,1) = 1.0/18.0; A_ex(2,2) = 0.0; A_ex(2,3) = 0.0; A_ex(2,4) = 0.0; 
         A_ex(3,0) = 5.0 / 6.0; A_ex(3,1) = -5.0 / 6.0; A_ex(3,2) = 0.5; A_ex(3,3) = 0.0; A_ex(3,4) = 0.0; 
         A_ex(4,0) = 0.25; A_ex(4,1) = 1.75; A_ex(4,2) = 0.75; A_ex(4,3) = -1.75; A_ex(4,4) = 0.0; 

         A_imp(0,0) = 0.0; A_imp(0,1) = 0.0; A_imp(0,2) = 0.0; A_imp(0,3) = 0.0; A_imp(0,4) = 0.0;
         A_imp(1,0) = 0.0; A_imp(1,1) = 0.5; A_imp(1,2) = 0.0; A_imp(1,3) = 0.0; A_imp(1,4) = 0.0;
         A_imp(2,0) = 0.0; A_imp(2,1) = 1.0 / 6.0; A_imp(2,2) = 0.5; A_imp(2,3) = 0.0; A_imp(2,4) = 0.0;
         A_imp(3,0) = 0.0; A_imp(3,1) = -0.5; A_imp(3,2) = 0.5; A_imp(3,3) = 0.5; A_imp(3,4) = 0.0;  
         A_imp(4,0) = 0.0; A_imp(4,1) = 1.5; A_imp(4,2) = -1.5; A_imp(4,3) = 0.5; A_imp(4,4) = 0.5;

         b_ex(0) = 0.25; b_ex(1) = 1.75; b_ex(2) = 0.75; b_ex(3) = -1.75; b_ex(4) = 0.0;
         b_imp(0) = 0.0; b_imp(1) = 1.5; b_imp(2) = -1.5; b_imp(3) = 0.5; b_imp(4) = 0.5;

         c_ex(0) = 0.0; c_ex(1) = 0.5; c_ex(2) = 2.0 / 3.0; c_ex(3) = 0.5; c_ex(4) = 1.0;
         c_imp(0) = 0.0; c_imp(1) = 0.5; c_imp(2) = 2.0 / 3.0; c_imp(3) = 0.5; c_imp(4) = 1.0;
      }
};

std::unique_ptr<TopOptRKIMEXSolver> TopOptRKIMEXSolver::SelectTopOptRKIMEX(const int ode_solver_type)
{
   using ode_ptr = std::unique_ptr<TopOptRKIMEXSolver>;
   switch (ode_solver_type)
   {
      // L-stable IMEX methods for design opt
      case 1: return ode_ptr(new TopOptRKIMEX_1_1_1);
      case 2: return ode_ptr(new TopOptRKIMEX_1_2_1);
      case 3: return ode_ptr(new TopOptRKIMEX_1_2_2);
      case 4: return ode_ptr(new TopOptRKIMEX_2_3_3);
      case 5: return ode_ptr(new TopOptRKIMEX_2_3_2);
      case 6: return ode_ptr(new TopOptRKIMEX_3_4_3);
      case 7: return ode_ptr(new TopOptRKIMEX_4_4_3);

      default: MFEM_ABORT("Unknown ODE solver type: " << ode_solver_type );
   }
}
}
#endif 
