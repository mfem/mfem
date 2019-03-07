//write my own predictor corrector solver

#include "mfem.hpp"

namespace mfem
{

// predictor-corrector scheme Brailovskaya’s scheme
class PDSolver : public ODESolver
{
private:
   Vector dxdt;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(Vector &x, double &t, double &dt);
};

void PDSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   dxdt.SetSize(f->Width());
}

void PDSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t);

   //predictor: update Psi w-> update j
   f->Mult(x, dxdt);
   x.Add(dt, dxdt);
   f->UpdateJ(x);

   //corrector: update Psi w-> update j and Phi
   f->Mult(x, dxdt);
   x.Add(dt, dxdt);
   f->UpdateJ(x); 
   f->UpdatePhi(x);

   t += dt;
}

}
