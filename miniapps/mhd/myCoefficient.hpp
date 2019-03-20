//write my own coefficient class: curl^perp in 2D
//coefficient:=[-Dy u, Dx u]
#include "mfem.hpp"

using namespace std;

namespace mfem
{
    class MyCoefficient : public VectorCoefficient
    {
      private:
         GridFunction *GridFunc;
      public:
         MyCoefficient(GridFunction *gf, int _vdim);
         virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
         virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir);

         //virtual void Eval(ElementTransformation &T, const IntegrationPoint &ip);
         virtual ~MyCoefficient() { }
    };

    MyCoefficient::MyCoefficient( GridFunction *gf, int _vdim ) 
        : VectorCoefficient (_vdim) 
    { 
        GridFunc=gf;
    }

    //not sure when this function will be called -QT
    void MyCoefficient::Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
    {
        V.SetSize(vdim);
        Vector grad;
        GridFunc->GetGradient(T, grad);

        V(0)=-grad(1);
        V(1)= grad(0);
    }

    void MyCoefficient::Eval(DenseMatrix &M, ElementTransformation &T, 
                         const IntegrationRule &ir)
    {
        M.SetSize(vdim, ir.GetNPoints());
        DenseMatrix grad;
        GridFunc->GetGradients(T, ir, grad);

        if (false)
        {
            cout << "vdim="<<vdim<<endl;
            cout << "grad size ="<<grad.Height()<<"x"<<grad.Width()<<endl;   //debug
            cout << "   M size ="<<M.Height()<<"x"<<M.Width()<<endl;   //debug
            cout << "ir.GetNPoints()="<<ir.GetNPoints()<<endl;
        }
        
        //cout << "grad =";
        for (int j=0; j<ir.GetNPoints(); j++)
        {
            //cout<<"("<<grad(0,j)<<" "<<grad(1,j)<<") ";   //debug
            M(0,j)=-grad(1,j);
            M(1,j)= grad(0,j);
        }
        //cout <<endl;

    }
    
}
