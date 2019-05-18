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

    // this is not called in assembling
    void MyCoefficient::Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
    {
        cout << "check me in MyCoefficient::Eval"<<endl;
        V.SetSize(vdim);
        Vector grad;
        T.SetIntPoint (&ip);
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
        
        for (int j=0; j<ir.GetNPoints(); j++)
        {
            M(0,j)=-grad(1,j);
            M(1,j)= grad(0,j);
        }

    }
    
    //poisson bracket coefficient (follow grid function coefficient)
    //coefficient=-u_y v_x + u_x v_y
    class PBCoefficient : public Coefficient
    {
      private:
         GridFunction *gfu, *gfv;
      public:
         PBCoefficient(GridFunction *gfu_, GridFunction *gfv_)
        { gfu=gfu_; gfv=gfv_;}
         double Eval(ElementTransformation &T, const IntegrationPoint &ip);
    };

    //note that Tr->IntPoint has been set as ip in AssembleRHSElementVect
    //so Getgradient(T, grad) should be fine
    double PBCoefficient::Eval(ElementTransformation &T,
                     const IntegrationPoint &ip)
    {
        Vector gradu, gradv;
        gfu->GetGradient(T, gradu);
        gfv->GetGradient(T, gradv);

        return -gradu(1)*gradv(0)+gradu(0)*gradv(1);
    }
}
