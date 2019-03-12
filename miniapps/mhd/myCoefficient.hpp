//write my own coefficient class
//coefficient:=[-Dy u, Dx u]
#include "mfem.hpp"

namespace mfem
{
    class MyCoefficient : public VectorGridFunctionCoefficient
    {
      private:
         GridFunction &u;
      public:
         MyCoefficient(GridFunction &_u);
         virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
         virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir);

         //virtual void Eval(ElementTransformation &T, const IntegrationPoint &ip);
         virtual ~MyCoefficient() { }
    };

    MyCoefficient::MyCoefficient( GridFunction &_u ) : u(_u) { }

    void MyCoefficient::Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
    {
        V.SetSize(vdim);
        Vector grad;
        u.GetGradient(T, grad);

        V(0)=-grad(1);
        V(1)= grad(0);
    }

    void MyCoefficient::Eval(DenseMatrix &M, ElementTransformation &T, 
                         const IntegrationRule &ir)
    {
        M.SetSize(vdim, ir.GetNPoints());
        DenseMatrix grad;
        u.GetGradients(T, ir, grad);

        for (int j=0; j<ir.GetNPoints(); j++)
        {
            M(0,j)=-grad(1,j);
            M(1,j)= grad(0,j);
        }
    }
    
}
