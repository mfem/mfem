#include "mfem.hpp"
using namespace std;
using namespace mfem;

/*
  Goal: compute the bilinear form matrix
    
  int_Gamma int_Gamma v_i(y) M(x, y) psi(x) dS(x) dS(y),    i=1,2,...

  v_i: i'th test function
  psi: solution that we are solving for
  M:   Green's function

  Approach 1: (code below)
    int_Gamma (int_Gamma M(x, y) psi(x) dS(x)) v_i(y) dS(y),    i=1,2,...

    The following is like a bilinear form with the test functions all set to 1?
    F(y; psi) := int_Gamma M(x, y) psi(x) dS(x)
    F(y; psi) is a matrix?

    int_Gamma F(y; psi) v_i(y) dS(y)

  Approach 2:
    int_Gamma (int_Gamma v_i(y) M(x, y) dS(y)) psi(x) dS(x),    i=1,2,...

    f_i(x) := int_Gamma v_i(y) M(x, y) dS(y)
    Linear form

    int_Gamma f_i(x) psi(x) dS(x)
    Bilinear form with shape dependent coefficient?
*/

double greenfunc(Vector X, Vector Y)
{
   return 1.0/(X(0)*X(0)+X(1)*X(1)+Y(0)*Y(0)+Y(1)*Y(1)); //just an example
}

class GreenCoefficient : public Coefficient
{
protected:
  std::function<double(const Vector &, const Vector &)> GreenFunction;
  Vector X;
  Vector Y;
  bool Xfix;
public:
  GreenCoefficient (function<double(const Vector &, const Vector &)> F)
    : GreenFunction(move(F)) {Xfix=true;}
  void SetXVec(Vector & x_) { X=x_; Xfix=true;}
  void SetYVec(Vector & y_) { Y=y_; Xfix=false;}
  double Eval (ElementTransformation &T, const IntegrationPoint &ip)
  {
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);

    if (Xfix)
      {
        return GreenFunction(X, transip);
      }
    else
      {
        return GreenFunction(transip, Y);
      }
  }
};

/*  Define a coefficient to compute
 *      int_Γ G(x,y) Phi(x) dSx
 *  at the given y
 */
class YFixSurfaceCoefficient : public Coefficient
{
private:
  GridFunction *Psi; // this is unknown
  GreenCoefficient Green;
  FiniteElementSpace *fespace;
  GridFunction ones;
public:
  YFixSurfaceCoefficient(GridFunction *psi_);
  double Eval(ElementTransformation &T, const IntegrationPoint &ip);
  virtual ~YFixSurfaceCoefficient() { }
};

YFixSurfaceCoefficient::YFixSurfaceCoefficient( GridFunction *psi_ )
   : Green(greenfunc), fespace(psi_->FESpace()), ones(psi_->FESpace())
{
   ones=1.0;
   Psi=psi_;
}

double YFixSurfaceCoefficient::Eval(ElementTransformation &T,
                                   const IntegrationPoint &ip)
{

  
   double x[3], value;
   // Compute y value for the given quaderture ip
   Vector transip(x, 3);
   T.Transform(ip, transip); //transip is coordinate for a given point ip
   Green.SetYVec(transip);

   // Evalute int_Γ G(x,y) Phi(x) dSx
   BilinearForm IntSurf(fespace);
   LinearForm IntPhi(fespace);
   IntSurf.AddBoundaryIntegrator(new BoundaryMassIntegrator(Green));
   IntSurf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   IntSurf.Assemble();
   IntSurf.Mult(*Psi, IntPhi); // psi is unknown
   value = IntPhi(ones);
   return value;
}
