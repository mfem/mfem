#include "mfem.hpp"
using namespace std;
using namespace mfem;

double greenfunc(Vector X, Vector Y)
{
   return 1.0/(X(0)*X(0)+X(1)*X(1)+Y(0)*Y(0)+Y(1)*Y(1)); //just an example
}

class GreenCoefficient : public Coefficient
{
protected:
   std::function<double(const Vector &, const Vector &)> GreenFunction;
   Vector X(3), Y(3);
   bool Xfix;
public:
   GreenCoefficient (double(*F)(Vector &, Vector &))
      : GreenFunction(std::move(F)) {Xfix=true;}
   SetXVec(Vector & x_) { X=x_; Xfix=true;}
   SetYVec(Vector & y_) { Y=y_; Xfix=false;}
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
}

/*  Define a coefficient to compute
 *      int_Γ G(x,y) Phi(x) dSx
 *  at the given y
 */
class YFixSurfaceCoefficient : public Coefficient
{
private:
   GridFunction *Psi;
   GreenCoefficient Green;
   ParFiniteElementSpace *fespace;
   ParGridFunction ones;
public:
   YFixSurfaceCoefficient(GridFunction *psi_);
   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   virtual ~YFixSurfaceCoefficient() { }
};

YFixSurfaceCoefficient::YFixSurfaceCoefficient( GridFunction *psi_ )
   : Green(greenfunc), fespace(psi_->ParFESpace()), ones(psi_->ParFESpace())
{
   ones=1.0;
   Psi=psi_;
}

doubl YFixSurfaceCoefficient::Eval(ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   double x[3], value;
   // Compute y value for the given quaderture ip
   Vector transip(x, 3);
   T.Transform(ip, transip); //transip is coordinate for a given point ip
   Green.SetYVec(transip);

   // Evalute int_Γ G(x,y) Phi(x) dSx
   ParBilinearForm IntSurf(fespace);
   ParLinearForm IntPhi(fespace);
   IntSurf.AddBoundaryIntegrator(new BoundaryLFIntegrator(Green));
   IntSurf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   IntSurf.Assemble();
   IntSurf.Mult(Phi, IntPhi);
   value = IntPhi(ones);
   return value;
}
