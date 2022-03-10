#ifndef ADV_DIFF_CG_HPP
#define ADV_DIFF_CG_HPP

#include "mfem.hpp"
namespace mfem {

class AdvectionDiffusionGLSStabRHS:public LinearFormIntegrator
{
public:
    AdvectionDiffusionGLSStabRHS(mfem::VectorCoefficient* vel, mfem::Coefficient* diff, double stc=1.0)
    {
        velocity=vel;
        cdiff=diff;
        mdiff=nullptr;
        stab_coeff=stc;
    }

    AdvectionDiffusionGLSStabRHS(mfem::VectorCoefficient* vel, mfem::MatrixCoefficient* diff, double stc=1.0)
    {
        velocity=vel;
        mdiff=diff;
        cdiff=nullptr;
        stab_coeff=stc;
    }

    virtual
    void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Trans, Vector &elvect)
    {

        const int dim=Trans.GetSpaceDim();
        const int ndof=el.GetDof();
        elvect.SetSize(ndof); elvect=0.0;

        DenseMatrix dshape(ndof,dim);
        Vector      sshape(ndof);
        Vector      rshape(ndof);
        DenseMatrix B(dim*ndof,ndof); //flux operator

        double diffc;
        DenseMatrix diffm(dim);
        Vector vel(dim);
        if(mdiff==nullptr){
            const IntegrationRule *nodes= &(el.GetNodes());
            for (int k = 0; k < ndof; k++)
            {
                const IntegrationPoint &ip = nodes->IntPoint(k);
                Trans.SetIntPoint(&ip);
                el.CalcPhysDShape(Trans,dshape);
                el.CalcPhysShape(Trans,sshape);
                diffc=cdiff->Eval(Trans,ip);
                velocity->Eval(vel,Trans,ip);
                for(int j=0; j<ndof; j++){
                for(int d=0; d<dim;  d++){
                    B(k+ndof*d,j)=-diffc*dshape(j,d)+vel(d)*sshape(j);
                }}
            }
        }else{//matrix coefficient
            const IntegrationRule *nodes= &(el.GetNodes());
            for (int k = 0; k < ndof; k++)
            {
                const IntegrationPoint &ip = nodes->IntPoint(k);
                Trans.SetIntPoint(&ip);
                el.CalcPhysDShape(Trans,dshape);
                el.CalcPhysShape(Trans,sshape);
                mdiff->Eval(diffm,Trans,ip);
                velocity->Eval(vel,Trans,ip);
                for(int j=0; j<ndof; j++){
                for(int d=0; d<dim;  d++){
                    B(k+ndof*d,j)= vel(d)*sshape(j);
                    for(int p=0;p<dim;p++){
                        B(k+ndof*d,j)=B(k+ndof*d,j)-diffm(d,p)*dshape(j,p);}
                }}
            }
        }

    }

private:
    double stab_coeff;
    mfem::VectorCoefficient* velocity;
    //only one the diffusion coefficients should be different than zero
    mfem::MatrixCoefficient* mdiff; //matrix diffusion coefficient
    mfem::Coefficient* cdiff; //scalar diffusion coefficient
};

class AdvectionDiffusionGLSStabInt:public BilinearFormIntegrator
{
public:
    AdvectionDiffusionGLSStabInt(mfem::VectorCoefficient* vel, mfem::Coefficient* diff, double stc=1.0)
    {
        velocity=vel;
        cdiff=diff;
        mdiff=nullptr;
        stab_coeff=1.0;
    }

    AdvectionDiffusionGLSStabInt(mfem::VectorCoefficient* vel, mfem::MatrixCoefficient* diff, double stc=1.0)
    {
        velocity=vel;
        mdiff=diff;
        cdiff=nullptr;
        stab_coeff=1.0;
    }


    virtual
    void  AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
    {
        const int dim=Trans.GetSpaceDim();
        const int ndof=el.GetDof();
        elmat.SetSize(ndof); elmat=0.0;

        DenseMatrix dshape(ndof,dim);
        Vector      sshape(ndof);
        Vector      rshape(ndof);
        DenseMatrix B(dim*ndof,ndof); //flux operator

        double diffc;
        DenseMatrix diffm(dim);
        Vector vel(dim);

        if(mdiff==nullptr){
            const IntegrationRule *nodes= &(el.GetNodes());
            for (int k = 0; k < ndof; k++)
            {
                const IntegrationPoint &ip = nodes->IntPoint(k);
                Trans.SetIntPoint(&ip);
                el.CalcPhysDShape(Trans,dshape);
                el.CalcPhysShape(Trans,sshape);
                diffc=cdiff->Eval(Trans,ip);
                velocity->Eval(vel,Trans,ip);

                for(int j=0; j<ndof; j++){
                for(int d=0; d<dim;  d++){
                    B(k+ndof*d,j)=-diffc*dshape(j,d)+vel(d)*sshape(j);
                }}
            }
        }else{//matrix coefficient
            const IntegrationRule *nodes= &(el.GetNodes());
            for (int k = 0; k < ndof; k++)
            {
                const IntegrationPoint &ip = nodes->IntPoint(k);
                Trans.SetIntPoint(&ip);
                el.CalcPhysDShape(Trans,dshape);
                el.CalcPhysShape(Trans,sshape);
                mdiff->Eval(diffm,Trans,ip);
                velocity->Eval(vel,Trans,ip);

                for(int j=0; j<ndof; j++){
                for(int d=0; d<dim;  d++){
                    B(k+ndof*d,j)= vel(d)*sshape(j);
                    for(int p=0;p<dim;p++){
                        B(k+ndof*d,j)=B(k+ndof*d,j)-diffm(d,p)*dshape(j,p);}
                }}
            }
        }

        const IntegrationRule *ir = nullptr;
        int order= 2 * el.GetOrder() + Trans.OrderGrad(&el);
        ir=&IntRules.Get(Trans.GetGeometryType(),order);

        double fstab;
        double h;
        double w;

        for(int i=0; i<ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Trans.SetIntPoint(&ip);
            w=Trans.Weight();
            w = ip.weight * w;

            //calculate the stabilization based on Larson2013
            {
                if(mdiff!=nullptr){
                    mdiff->Eval(diffm,Trans,ip);
                    diffc=diffm.CalcSingularvalue(dim-1);
                }else{
                    diffc=cdiff->Eval(Trans,ip);
                }
                h=pow(Trans.Weight(),1./dim);
                if(diffc>h){ fstab=h*h;}
                else{
                    velocity->Eval(vel,Trans,ip);
                    fstab=fstab+h/std::fmax(fabs(vel.Normlinf()),1e-12);
                }
            }

            el.CalcPhysDShape(Trans,dshape);
            B.Mult(dshape.GetData(),rshape);
            w=w*stab_coeff*fstab;
            AddMult_a_VVt(w,rshape,elmat);
        }
    }

private:
    double stab_coeff;
    mfem::VectorCoefficient* velocity;
    //only one the diffusion coefficients should be different than zero
    mfem::MatrixCoefficient* mdiff; //matrix diffusion coefficient
    mfem::Coefficient* cdiff; //scalar diffusion coefficient
};

}

#endif
