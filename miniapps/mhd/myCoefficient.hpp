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
        //cout << "check me in MyCoefficient::Eval"<<endl;
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

    //Evalue a vector JxB=[0,0,j]x[B1,B2,0]
    //                   =[-jB2, jB1]
    class JxBCoefficient : public VectorCoefficient
    {
      private:
         GridFunction *gfj, *gfB;
      public:
         JxBCoefficient(GridFunction *gfj_, GridFunction *gfB_)
         :VectorCoefficient (2) 
        { gfj=gfj_; gfB=gfB_;}
         void Eval(Vector &V, ElementTransformation &T,
                   const IntegrationPoint &ip);
    };

    void JxBCoefficient::Eval(Vector &V, ElementTransformation &T,
                              const IntegrationPoint &ip)
    {
        V.SetSize(2);
        T.SetIntPoint(&ip);

        Vector Bvec;
        gfB->GetVectorValue(T, ip, Bvec);

        double j;
        j=gfj->GetValue(T,ip);

        V(0)=-j*Bvec(1);
        V(1)= j*Bvec(0);
    }

    //Evalute B^2/2
    class B2Coefficient : public Coefficient
    {
      private:
        GridFunction *gfB;
      public:
        B2Coefficient(GridFunction *gfB_) { gfB=gfB_;}
        double Eval(ElementTransformation &T,
               const IntegrationPoint &ip){
            Vector Bvec;
            gfB->GetVectorValue(T, ip, Bvec);
            return (Bvec(0)*Bvec(0)+Bvec(1)*Bvec(1))/2.0;
        }
    };

    /*
    //speical rhs coefficient
    //coefficient=2*[vecg1_x vecg2_y - (vecg1_y)^2] - J^2 - (J_x Psi_x + J_y Psi_y)
    class RHSCoefficient : public Coefficient
    {
      private:
         GridFunction *vecg, *Psi, *J;
      public:
         PBCoefficient(GridFunction *vecg_, GridFunction *Psi_, GridFunction *J_)
        { vecg=vecg_; Psi=Psi_; J=J_}
         double Eval(ElementTransformation &T, const IntegrationPoint &ip);
    };

    double RHSCoefficient::Eval(ElementTransformation &T,
                     const IntegrationPoint &ip)
    {
        Vector grad, gradJ, HenssianPhi;
        double Jvalue;
        Psi->GetGradient(T, gradPsi);
          J->GetGradient(T, gradJ);
        Jvalue = J->GetValue(T, ip);

        return Jvalue*Jvalue-gradJ(0)*gradPsi(0)-gradJ(1)*gradPsi(1);
    }


    //rhs = Phi_xx Phi_yy -Phi_xy^2 - J^2 - (J_x Psi_x + J_y Psi_y)
    double RHSCoefficient::Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationPoint &ir)
    {
        DenseMatrix rhs;

        M.SetSize(1, ir.GetNPoints());
        Psi->GetGradients(T, ir, gradPsi);
          J->GetGradients(T, ir, gradJ);
          J->GetValues(T, ir, Jvalue);
        Phi->GetHessians(T, ir, Hessians);

        for (int i=0; i<ir.GetNPoints(); i++)
        {
            M(0,j)= Hessians(i, 0)*Hessians(i,2) - Hessians(i,1)*Hessians(i,1) 
                  -Jvalue(i)*Jvalue(i)-gradJ(0,i)*gradPsi(0,i)-gradJ(1,i)*gradPsi(1,i);
        }
    }

    //compute the boundary coefficient
    class RHSCoefficient : public Coefficient
    {
      private:
         GridFunction *vel, *mag, *J;
      public:
         PBCoefficient(GridFunction *vel_, GridFunction *mag_, GridFunction *J_)
        { vel=vel_; mag=mag_; J=J_}
         double Eval(ElementTransformation &T, const IntegrationPoint &ip);
    };
    */

}
