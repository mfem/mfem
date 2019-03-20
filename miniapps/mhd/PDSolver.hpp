//write my own predictor corrector solver

#include "mfem.hpp"

namespace mfem
{

// predictor-corrector scheme Brailovskaya’s scheme
class PDSolver : public ODESolver
{
private:
   Vector dxdt, x1;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void StepP(Vector &x, double &t, double &dt);//Predictor
   virtual void Step(Vector &x, double &t, double &dt);//Corrector
};

void PDSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   int n = f->Width();
   dxdt.SetSize(n);
   x1.SetSize(n);
}

void PDSolver::StepP(Vector &x, double &t, double &dt)
{
   f->SetTime(t);
   //cout <<"t="<<f->GetTime()<<endl;

   //predictor: update Psi w-> update j
   x1=x;
   f->Mult(x, dxdt);
   x.Add(dt, dxdt);
}


void PDSolver::Step(Vector &x, double &t, double &dt)
{
   //cout <<"t="<<f->GetTime()<<endl;
   //corrector: update Psi w-> update j and Phi
   f->Mult(x, dxdt);
   add(x1, dt, dxdt, x);

   t += dt;
}

}
