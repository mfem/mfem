//this is a first-order predictor corrector time stepping

#include "mfem.hpp"

namespace mfem
{

// predictor-corrector scheme Brailovskaya’s scheme
class PCSolver : public ODESolver
{
private:
   Vector dxdt, x1;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void StepP(Vector &x, double &t, double &dt);//Predictor
   virtual void Step(Vector &x, double &t, double &dt);//Corrector
};

void PCSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   int n = f->Width();
   dxdt.SetSize(n);
   x1.SetSize(n);
}

void PCSolver::StepP(Vector &x, double &t, double &dt)
{
   f->SetTime(t);

   //predictor: update Psi w-> update j
   x1=x;
   f->Mult(x, dxdt);
   x.Add(dt, dxdt);
}


void PCSolver::Step(Vector &x, double &t, double &dt)
{
   //corrector: update Psi w-> update j and Phi
   f->Mult(x, dxdt);
   add(x1, dt, dxdt, x);
   t += dt;
}

}
