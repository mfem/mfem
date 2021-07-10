#ifndef ADVDIFF_H
#define ADVDIFF_H

#include<map>
#include<vector>
#include "mfem.hpp"

namespace mfem {



class AdvectionDiffusionMX:public mfem::BlockNonlinearFormIntegrator
{
public:
    AdvectionDiffusionMX()
    {
        diffc=nullptr;
        veloc=nullptr;
        inpuc=nullptr;
        mucoe=nullptr;

        dbc=nullptr;
        gamma=10;


    }

    AdvectionDiffusionMX(mfem::Coefficient* diffusion_, mfem::VectorCoefficient* veloc_,
                         mfem::Coefficient* mu_, mfem::Coefficient* inp_)
    {
        diffc=diffusion_;
        veloc=veloc_;
        inpuc=inp_;
        mucoe=mu_;

        dbc=nullptr;
        gamma=10;
    }


    void SetDirichletBCCoeficient(mfem::Coefficient* bc)
    {
        dbc=bc;
    }

    void SetVelocity(mfem::VectorCoefficient* veloc_)
    {
        veloc=veloc_;
    }

    void SetDiffusion(mfem::Coefficient* diffusion_)
    {
        diffc=diffusion_;
    }

    void SetReactionCoefficient(mfem::Coefficient* mu_)
    {
        mucoe=mu_;
    }

    void SetVolInput(mfem::Coefficient* imp_)
    {
        inpuc=imp_;
    }

    void SetDirichletBCPenalization(double penal)
    {
        gamma=penal;
    }

    virtual ~AdvectionDiffusionMX(){}

    virtual
    double GetElementEnergy(const Array<const FiniteElement *> &el,
                            ElementTransformation &Tr,
                            const Array<const Vector *> &elfun) override
    {
        return 0.0;
    }

    virtual
    void AssembleElementVector(const Array<const FiniteElement *> &el,
                               ElementTransformation &Tr,
                               const Array<const Vector *> &elfun,
                               const Array<Vector *> &elvec) override
    {
        int dof_u = el[0]->GetDof();
        int dof_p = el[1]->GetDof();
        int dof_z = el[2]->GetDof();

        int dim = el[0]->GetDim();

        elvec[0]->SetSize(dof_u);
        elvec[1]->SetSize(dof_p);
        elvec[2]->SetSize(dof_z);

        *(elvec[0])=0.0;
        *(elvec[1])=0.0;
        *(elvec[2])=0.0;


        int spaceDim = Tr.GetDimension();
        if (dim != spaceDim)
        {
           mfem::mfem_error("AdvectionDiffusionMX::AssembleElementVector"
                            " is not defined on manifold meshes");
        }

        mfem::DenseMatrix bsu;
        mfem::DenseMatrix bsp;
        mfem::DenseMatrix bsz;

        //set B-matrices
        bsu.SetSize(dof_u,4); // [u, ux,uy,uz]
        bsp.SetSize(dof_p,4); // [px,py,pz, div(p)]
        bsz.SetSize(dof_z,1); // [z]

        bsu=0.0;
        bsp=0.0;
        bsz=0.0;

        Vector sh;
        DenseMatrix dh;

        const IntegrationRule *ir = nullptr;
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0]);
        ir=&IntRules.Get(Tr.GetGeometryType(),order);

        double mmu;
        DenseMatrix kap(3,3); kap=0.0;
        Vector vel(3); vel=0.0;
        double rhc;
        double w;

        Vector ss(9);
        Vector rr(9);

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
           const IntegrationPoint &ip = ir->IntPoint(i);
           Tr.SetIntPoint(&ip);
           w=Tr.Weight();
           w = ip.weight * w;

           sh.SetDataAndSize(bsu.GetData(),dof_u);
           el[0]->CalcPhysShape(Tr,sh);

           dh.UseExternalData(bsu.GetData()+dof_u,dof_u,dim);
           el[0]->CalcPhysDShape(Tr,dh);

           sh.SetDataAndSize(ss.GetData(),4);
           bsu.MultTranspose(*(elfun[0]),sh);


           dh.UseExternalData(bsp.GetData(), dof_p, dim);
           el[1]->CalcPhysVShape(Tr,dh);

           sh.SetDataAndSize(bsp.GetData()+3*dof_p, dof_p);
           el[1]->CalcPhysDivShape(Tr,sh);

           sh.SetDataAndSize(ss.GetData()+4,4);
           bsp.MultTranspose(*(elfun[1]),sh);

           sh.SetDataAndSize(bsz.GetData(),dof_z);
           el[2]->CalcPhysShape(Tr,sh);

           sh.SetDataAndSize(ss.GetData()+8,1);
           bsz.MultTranspose(*(elfun[2]),sh);


           mmu=0.0;
           if(mucoe!=nullptr)
           {
               mmu=mucoe->Eval(Tr,ip);
           }

           if(diffc!=nullptr)
           {
               kap(0,0)=diffc->Eval(Tr,ip);
               kap(1,1)=kap(0,0);
               kap(2,2)=kap(0,0);
           }

           vel=0.0;
           if(veloc!=nullptr)
           {
               veloc->Eval(vel,Tr,ip);
           }

           rhc=0.0;
           if(inpuc!=nullptr)
           {
               rhc=inpuc->Eval(Tr,ip);
           }

           EvalQRes(kap.GetData(), vel.GetData(), mmu, rhc, ss.GetData(), rr.GetData());

           sh.SetDataAndSize(rr.GetData(),4);
           bsu.AddMult_a(w,sh, *(elvec[0]));
           sh.SetDataAndSize(rr.GetData()+4,4);
           bsp.AddMult_a(w,sh, *(elvec[1]));
           sh.SetDataAndSize(rr.GetData()+8,1);
           bsz.AddMult_a(w,sh, *(elvec[2]));


        }//end integration loop

    }

    virtual
    void AssembleElementGrad(const Array<const FiniteElement *> &el,
                             ElementTransformation &Tr,
                             const Array<const Vector *> &elfun,
                             const Array2D<DenseMatrix *> &elmats) override
    {
        int dof_u = el[0]->GetDof();
        int dof_p = el[1]->GetDof();
        int dof_z = el[2]->GetDof();

        int dim = el[0]->GetDim();
        int spaceDim = Tr.GetDimension();
        if (dim != spaceDim)
        {
           mfem::mfem_error("AdvectionDiffusionMX::AssembleElementVector"
                            " is not defined on manifold meshes");
        }

        elmats(0,0)->SetSize(dof_u,dof_u);
        elmats(0,1)->SetSize(dof_u,dof_p);
        elmats(0,2)->SetSize(dof_u,dof_z);

        elmats(1,0)->SetSize(dof_p,dof_u);
        elmats(1,1)->SetSize(dof_p,dof_p);
        elmats(1,2)->SetSize(dof_p,dof_z);

        elmats(2,0)->SetSize(dof_z,dof_u);
        elmats(2,1)->SetSize(dof_z,dof_p);
        elmats(2,2)->SetSize(dof_z,dof_z);

        (*elmats(0,0))=0.0;
        (*elmats(0,1))=0.0;
        (*elmats(0,2))=0.0;

        (*elmats(1,0))=0.0;
        (*elmats(1,1))=0.0;
        (*elmats(1,2))=0.0;

        (*elmats(2,0))=0.0;
        (*elmats(2,1))=0.0;
        (*elmats(2,2))=0.0;

        mfem::DenseMatrix bsu;
        mfem::DenseMatrix bsp;
        mfem::DenseMatrix bsz;

        //set B-matrices
        bsu.SetSize(dof_u,4); // [u, ux,uy,uz]
        bsp.SetSize(dof_p,4); // [px,py,pz, div(p)]
        bsz.SetSize(dof_z,1); // [z]

        bsu=0.0;
        bsp=0.0;
        bsz=0.0;

        Vector sh;
        DenseMatrix dh;

        DenseMatrix th;
        DenseMatrix mh;
        DenseMatrix rh;

        const IntegrationRule *ir = nullptr;
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0]);
        ir=&IntRules.Get(Tr.GetGeometryType(),order);

        double mmu;
        DenseMatrix kap(3,3); kap=0.0;
        Vector vel(3); vel=0.0;
        double w;

        DenseMatrix mm; //state matrix
        mm.SetSize(9,9); //set the size of the state matrix

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
           const IntegrationPoint &ip = ir->IntPoint(i);
           Tr.SetIntPoint(&ip);
           w=Tr.Weight();
           w = ip.weight * w;

           sh.SetDataAndSize(bsu.GetData(),dof_u);
           el[0]->CalcPhysShape(Tr,sh);

           dh.UseExternalData(bsu.GetData()+dof_u,dof_u,dim);
           el[0]->CalcPhysDShape(Tr,dh);

           dh.UseExternalData(bsp.GetData(), dof_p, dim);
           el[1]->CalcPhysVShape(Tr,dh);

           sh.SetDataAndSize(bsp.GetData()+3*dof_p, dof_p);
           el[1]->CalcPhysDivShape(Tr,sh);

           sh.SetDataAndSize(bsz.GetData(),dof_z);
           el[2]->CalcPhysShape(Tr,sh);

           mmu=0.0;
           if(mucoe!=nullptr)
           {
               mmu=mucoe->Eval(Tr,ip);
           }

           if(diffc!=nullptr)
           {
               kap(0,0)=diffc->Eval(Tr,ip);
               kap(1,1)=kap(0,0);
               kap(2,2)=kap(0,0);
           }

           vel=0.0;
           if(veloc!=nullptr)
           {
               veloc->Eval(vel,Tr,ip);
           }

           EvalQMat(kap.GetData(),vel.GetData(),mmu,mm.GetData());

           mh.SetSize(4,4);
           mh.CopyMN(mm,4,4,0,0);
           mh.Transpose();
           th.SetSize(dof_u,4);
           rh.SetSize(dof_u,dof_u);
           MultABt(bsu,mh,th);
           MultABt(th,bsu,rh);
           elmats(0,0)->AddMatrix(w,rh,0,0);

           mh.SetSize(4,4);
           mh.CopyMN(mm,4,4,0,4);
           mh.Transpose();
           th.SetSize(dof_u,4);
           rh.SetSize(dof_u,dof_p);
           MultABt(bsu,mh,th);
           MultABt(th,bsp,rh);
           elmats(0,1)->AddMatrix(w,rh,0,0);

           mh.SetSize(4,1);
           mh.CopyMN(mm,4,1,0,8);
           mh.Transpose();
           th.SetSize(dof_u,1);
           rh.SetSize(dof_u,dof_z);
           MultABt(bsu,mh,th);
           MultABt(th,bsz,rh);
           elmats(0,2)->AddMatrix(w,rh,0,0);


           mh.SetSize(4,4);
           mh.CopyMN(mm,4,4,4,0);
           mh.Transpose();
           th.SetSize(dof_p,4);
           rh.SetSize(dof_p,dof_u);
           MultABt(bsp,mh,th);
           MultABt(th,bsu,rh);
           elmats(1,0)->AddMatrix(w,rh,0,0);

           mh.SetSize(4,4);
           mh.CopyMN(mm,4,4,4,4);
           mh.Transpose();
           th.SetSize(dof_p,4);
           rh.SetSize(dof_p,dof_p);
           MultABt(bsp,mh,th);
           MultABt(th,bsp,rh);
           elmats(1,1)->AddMatrix(w,rh,0,0);

           mh.SetSize(4,1);
           mh.CopyMN(mm,4,1,4,8);
           mh.Transpose();
           th.SetSize(dof_p,1);
           rh.SetSize(dof_p,dof_z);
           MultABt(bsp,mh,th);
           MultABt(th,bsz,rh);
           elmats(1,2)->AddMatrix(w,rh,0,0);

           mh.SetSize(1,4);
           mh.CopyMN(mm,1,4,8,0);
           mh.Transpose();
           th.SetSize(dof_z,4);
           rh.SetSize(dof_z,dof_u);
           MultABt(bsz,mh,th);
           MultABt(th,bsu,rh);
           elmats(2,0)->AddMatrix(w,rh,0,0);

           mh.SetSize(1,4);
           mh.CopyMN(mm,1,4,8,4);
           mh.Transpose();
           th.SetSize(dof_z,4);
           rh.SetSize(dof_z,dof_p);
           MultABt(bsz,mh,th);
           MultABt(th,bsp,rh);
           elmats(2,1)->AddMatrix(w,rh,0,0);

           mh.SetSize(1,1);
           mh.CopyMN(mm,1,1,8,8);
           mh.Transpose();
           th.SetSize(dof_z,1);
           rh.SetSize(dof_z,dof_z);
           MultABt(bsz,mh,th);
           MultABt(th,bsz,rh);
           elmats(2,2)->AddMatrix(w,rh,0,0);
        }
    }

    virtual
    void AssembleFaceVector(const Array<const FiniteElement *> &el1,
                            const Array<const FiniteElement *> &el2,
                            FaceElementTransformations &Tr,
                            const Array<const Vector *> &elfun,
                            const Array<Vector *> &elvec)
    {
        int dom_id=Tr.Attribute;

        int dof_u = el1[0]->GetDof();
        int dof_p = el1[1]->GetDof();
        int dof_z = el1[2]->GetDof();

        int dim = el1[0]->GetDim();

        elvec[0]->SetSize(dof_u);
        elvec[1]->SetSize(dof_p);
        elvec[2]->SetSize(dof_z);

        *(elvec[0])=0.0;
        *(elvec[1])=0.0;
        *(elvec[2])=0.0;



        mfem::Vector bsu;
        bsu.SetSize(dof_u);

        mfem::Vector nor; //normal vector
        mfem::Vector nir; //unit normal vector

        nor.SetSize(dim);
        nir.SetSize(dim);

        const IntegrationRule *ir = nullptr;
        int order= 2 * el1[0]->GetOrder();
        ir=&IntRules.Get(Tr.GetGeometryType(),order);

        double ih=0.0;  //inverse of the element characteristic length
        double nr=0.0;  //norm of the normal vector
        double ek=0.0;  //the smallest eigenvalue of the diffusion tensor
        double gg=0.0;  //boundary value
        double bp=0.0;
        double w;

        mfem::Vector ev(3);
        Vector vel(3); vel=0.0;
        DenseMatrix kap(3,3); kap=0.0;


        for (int i = 0; i < ir->GetNPoints(); i++)
        {
           const IntegrationPoint &ipg = ir->IntPoint(i);
           Tr.SetAllIntPoints(&ipg);
           const mfem::IntegrationPoint &ip=Tr.GetElement1IntPoint();
           mfem::CalcOrtho(Tr.Jacobian(),nor);

           w = Tr.Weight();
           w = ipg.weight * w;

           nr=nor.Norml2();
           ih=nr/Tr.Elem1->Weight();
           nir.Set(1.0/nr,nor);


           if(diffc!=nullptr)
           {
               kap(0,0)=diffc->Eval(Tr,ip);
               kap(1,1)=kap(0,0);
               kap(2,2)=kap(0,0);
               ek=kap(0,0);

               //general case
               //kap.Eigenvalues(ev);
               //ek=std::min(ev(0),ev(1));
               //ek=std::min(ek,ev(2));
           }

           if(veloc!=nullptr)
           {
               veloc->Eval(vel,Tr,ip);
           }

           if(dbc!=nullptr)
           {
               gg=dbc->Eval(Tr,ip);
           }

           el1[0]->CalcShape(ip,bsu);

           bp=0.0;
           for(int ii=0;ii<dim;ii++)
           {
               bp=bp+nir(ii)*vel(ii);
           }
           bp=std::min(0.0,bp);

           w=w*gg*(bp*bp/ih+gamma*ek*ek*ih);
           elvec[0]->Add(-w,bsu);
        }

    }

    virtual
    void AssembleFaceGrad(const Array<const FiniteElement *> &el1,
                          const Array<const FiniteElement *> &el2,
                          FaceElementTransformations &Tr,
                          const Array<const Vector *> &elfun,
                          const Array2D<DenseMatrix *> &elmats)
    {
        int dom_id=Tr.Attribute;

        int dof_u = el1[0]->GetDof();
        int dof_p = el1[1]->GetDof();
        int dof_z = el1[2]->GetDof();

        int dim = el1[0]->GetDim();

        elmats(0,0)->SetSize(dof_u,dof_u);
        elmats(0,1)->SetSize(dof_u,dof_p);
        elmats(0,2)->SetSize(dof_u,dof_z);

        elmats(1,0)->SetSize(dof_p,dof_u);
        elmats(1,1)->SetSize(dof_p,dof_p);
        elmats(1,2)->SetSize(dof_p,dof_z);

        elmats(2,0)->SetSize(dof_z,dof_u);
        elmats(2,1)->SetSize(dof_z,dof_p);
        elmats(2,2)->SetSize(dof_z,dof_z);

        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                (*elmats(i,j))=0.0;
            }
        }

        mfem::Vector bsu;
        bsu.SetSize(dof_u);

        mfem::Vector nor; //normal vector
        mfem::Vector nir; //unit normal vector

        nor.SetSize(dim);
        nir.SetSize(dim);

        const IntegrationRule *ir = nullptr;
        int order= 2 * el1[0]->GetOrder();
        ir=&IntRules.Get(Tr.GetGeometryType(),order);

        double ih=0.0;  //inverse of the element characteristic length
        double nr=0.0;  //norm of the normal vector
        double ek=0.0;  //the smallest eigenvalue of the diffusion tensor
        double bp;
        double w;

        mfem::Vector ev(3);
        Vector vel(3); vel=0.0;
        DenseMatrix kap(3,3); kap=0.0;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
           const IntegrationPoint &ipg = ir->IntPoint(i);
           Tr.SetAllIntPoints(&ipg);
           const mfem::IntegrationPoint &ip=Tr.GetElement1IntPoint();
           mfem::CalcOrtho(Tr.Jacobian(),nor);

           w = Tr.Weight();
           w = ipg.weight * w;

           nr=nor.Norml2();
           ih=nr/Tr.Elem1->Weight();
           nir.Set(1.0/nr,nor);

           if(diffc!=nullptr)
           {
               kap(0,0)=diffc->Eval(Tr,ip);
               kap(1,1)=kap(0,0);
               kap(2,2)=kap(0,0);

               ek=kap(0,0);

               //general case
               //kap.Eigenvalues(ev);
               //ek=std::min(ev(0),ev(1));
               //ek=std::min(ek,ev(2));
           }

           if(veloc!=nullptr)
           {
               veloc->Eval(vel,Tr,ip);
           }

           el1[0]->CalcShape(ip,bsu);

           bp=0.0;
           for(int ii=0;ii<nir.Size();ii++)
           {
               bp=bp+nir(ii)*vel(ii);
           }
           bp=std::min(0.0,bp);

           w=w*(bp*bp/ih+gamma*ek*ek*ih);

           mfem::AddMult_a_VVt(w,bsu,*elmats(0,0));

        }
    }


private:
    mfem::Coefficient* diffc;
    mfem::VectorCoefficient* veloc;
    mfem::Coefficient* inpuc;
    mfem::Coefficient* mucoe;

    //boundary faces
    double gamma; //penalization for the Nitsche method
    mfem::Coefficient* dbc; //Dirichlet BC


    //aa[3,3] - diffusion matrix
    //bb[3] - velocity
    //mmu - reaction coeficient
    void EvalQRes(double* kap, double* bb, double mmu, double inp, double* uu, double* rr)
    {
        double t5,t11,t17;
        t5 = bb[0]*uu[0]-kap[0]*uu[1]-kap[3]*uu[2]-kap[6]*uu[3]-uu[4];
        t11 = bb[1]*uu[0]-kap[1]*uu[1]-kap[4]*uu[2]-kap[7]*uu[3]-uu[5];
        t17 = bb[2]*uu[0]-kap[2]*uu[1]-kap[5]*uu[2]-kap[8]*uu[3]-uu[6];
        rr[0] = mmu*uu[8]+t11*bb[1]+t17*bb[2]+t5*bb[0];
        rr[1] = -t11*kap[1]-t17*kap[2]-t5*kap[0];
        rr[2] = -t11*kap[4]-t17*kap[5]-t5*kap[3];
        rr[3] = -t11*kap[7]-t17*kap[8]-t5*kap[6];
        rr[4] = -t5;
        rr[5] = -t11;
        rr[6] = -t17;
        rr[7] = uu[8];
        rr[8] = mmu*uu[0]-inp+uu[7];
    }

    void EvalQMat(double* kap, double* bb, double mmu, double* kmat)
    {
        double t1,t2,t3,t8,t12,t16,t17,t18,t19,t24,t28,t29,t30;
        double t31,t36,t37,t38,t39;
        t1 = bb[0]*bb[0];
        t2 = bb[1]*bb[1];
        t3 = bb[2]*bb[2];
        t8 = -bb[0]*kap[0]-bb[1]*kap[1]-bb[2]*kap[2];
        t12 = -bb[0]*kap[3]-bb[1]*kap[4]-bb[2]*kap[5];
        t16 = -bb[0]*kap[6]-bb[1]*kap[7]-bb[2]*kap[8];
        t17 = kap[0]*kap[0];
        t18 = kap[1]*kap[1];
        t19 = kap[2]*kap[2];
        t24 = kap[0]*kap[3]+kap[1]*kap[4]+kap[2]*kap[5];
        t28 = kap[0]*kap[6]+kap[1]*kap[7]+kap[2]*kap[8];
        t29 = kap[3]*kap[3];
        t30 = kap[4]*kap[4];
        t31 = kap[5]*kap[5];
        t36 = kap[3]*kap[6]+kap[4]*kap[7]+kap[5]*kap[8];
        t37 = kap[6]*kap[6];
        t38 = kap[7]*kap[7];
        t39 = kap[8]*kap[8];
        kmat[0] = t1+t2+t3;
        kmat[1] = t8;
        kmat[2] = t12;
        kmat[3] = t16;
        kmat[4] = -bb[0];
        kmat[5] = -bb[1];
        kmat[6] = -bb[2];
        kmat[7] = 0.0;
        kmat[8] = mmu;
        kmat[9] = t8;
        kmat[10] = t17+t18+t19;
        kmat[11] = t24;
        kmat[12] = t28;
        kmat[13] = kap[0];
        kmat[14] = kap[1];
        kmat[15] = kap[2];
        kmat[16] = 0.0;
        kmat[17] = 0.0;
        kmat[18] = t12;
        kmat[19] = t24;
        kmat[20] = t29+t30+t31;
        kmat[21] = t36;
        kmat[22] = kap[3];
        kmat[23] = kap[4];
        kmat[24] = kap[5];
        kmat[25] = 0.0;
        kmat[26] = 0.0;
        kmat[27] = t16;
        kmat[28] = t28;
        kmat[29] = t36;
        kmat[30] = t37+t38+t39;
        kmat[31] = kap[6];
        kmat[32] = kap[7];
        kmat[33] = kap[8];
        kmat[34] = 0.0;
        kmat[35] = 0.0;
        kmat[36] = -bb[0];
        kmat[37] = kap[0];
        kmat[38] = kap[3];
        kmat[39] = kap[6];
        kmat[40] = 1.0;
        kmat[41] = 0.0;
        kmat[42] = 0.0;
        kmat[43] = 0.0;
        kmat[44] = 0.0;
        kmat[45] = -bb[1];
        kmat[46] = kap[1];
        kmat[47] = kap[4];
        kmat[48] = kap[7];
        kmat[49] = 0.0;
        kmat[50] = 1.0;
        kmat[51] = 0.0;
        kmat[52] = 0.0;
        kmat[53] = 0.0;
        kmat[54] = -bb[2];
        kmat[55] = kap[2];
        kmat[56] = kap[5];
        kmat[57] = kap[8];
        kmat[58] = 0.0;
        kmat[59] = 0.0;
        kmat[60] = 1.0;
        kmat[61] = 0.0;
        kmat[62] = 0.0;
        kmat[63] = 0.0;
        kmat[64] = 0.0;
        kmat[65] = 0.0;
        kmat[66] = 0.0;
        kmat[67] = 0.0;
        kmat[68] = 0.0;
        kmat[69] = 0.0;
        kmat[70] = 0.0;
        kmat[71] = 1.0;
        kmat[72] = mmu;
        kmat[73] = 0.0;
        kmat[74] = 0.0;
        kmat[75] = 0.0;
        kmat[76] = 0.0;
        kmat[77] = 0.0;
        kmat[78] = 0.0;
        kmat[79] = 1.0;
        kmat[80] = 0.0;
    }
};



class AdvectionDiffusionMXSolver
{
public:
    AdvectionDiffusionMXSolver(mfem::ParMesh* pmesh_, int order_=1)
    {
        pmesh=pmesh_;
        order=order_;

        int dim=pmesh->Dimension();

        ufec=new mfem::H1_FECollection(order,dim);
        pfec=new mfem::RT_FECollection(order,dim);
        zfec=new mfem::L2_FECollection(order,dim);


        ufes=new mfem::ParFiniteElementSpace(pmesh,ufec);
        pfes=new mfem::ParFiniteElementSpace(pmesh,pfec);
        zfes=new mfem::ParFiniteElementSpace(pmesh,zfec);

        sfes.Append(ufes);
        sfes.Append(pfes);
        sfes.Append(zfes);

        nf=new mfem::ParBlockNonlinearForm(sfes);

        rhs.Update(nf->GetBlockTrueOffsets()); rhs=0.0;
        sol.Update(nf->GetBlockTrueOffsets()); sol=0.0;
        adj.Update(nf->GetBlockTrueOffsets()); adj=0.0;

        fprim.SetSpace(ufes);
        fflux.SetSpace(pfes);
        fmult.SetSpace(zfes);

        SetSolver();

        dicoef=nullptr;
        mucoef=nullptr;
        vecoef=nullptr;
        incoef=nullptr;

        pmat=nullptr;
        prec=nullptr;
        psol=nullptr;

    }

    ~AdvectionDiffusionMXSolver()
    {

        delete psol;
        delete prec;
        delete pmat;

        delete nf;

        delete ufes;
        delete pfes;
        delete zfes;

        delete ufec;
        delete pfec;
        delete zfec;

        for(auto it=bc.begin();it!=bc.end();it++)
        {
            delete *it;
        }
    }

    void SetSolver(double rtol=1e-8, double atol=1e-12,int miter=1000, int prt_level=1)
    {
        rel_tol=rtol;
        abs_tol=atol;
        max_iter=miter;
        print_level=prt_level;
    }


    void SetDiffusion(mfem::Coefficient* coef_)
    {
        dicoef=coef_;
    }

    void SetReaction(mfem::Coefficient* coef_)
    {
        mucoef=coef_;
    }

    void SetVelocity(mfem::VectorCoefficient* coef_)
    {
        vecoef=coef_;
    }

    void SetLoad(mfem::Coefficient* coef_)
    {
        incoef=coef_;
    }

    /// Solves the forward problem.
    void FSolve();

    /// Solves the adjoint with the provided rhs.
    void ASolve(mfem::BlockVector& rhs);


    mfem::ParGridFunction& GetPrimField()
    {
        fprim.SetFromTrueDofs(sol.GetBlock(0));
        return fprim;
    }

    mfem::ParGridFunction& GetFluxField()
    {
        fflux.SetFromTrueDofs(sol.GetBlock(1));
        return fflux;
    }

    mfem::ParGridFunction& GetMultField()
    {
        fmult.SetFromTrueDofs(sol.GetBlock(2));
        return fmult;
    }

    void AddDirichletBC(int mark, double val)
    {
        int ni=bc.size();
        bc.push_back(new mfem::ConstantCoefficient(val));
        mfem::Array<int> markers(pmesh->bdr_attributes.Max());
        markers=0;
        markers[mark-1]=1;
        bcc[bc[ni]]=markers;
    }

    void AddDirichletBC(int mark, mfem::Coefficient* cc)
    {
        //check if cc is already in
        auto it=bcc.find(cc);
        if(it!=bcc.end())
        {
            (it->second)[mark-1]=1;
        }
        else{
            mfem::Array<int> markers(pmesh->bdr_attributes.Max());
            markers=0;
            markers[mark-1]=1;
            bcc[cc]=markers;
        }
    }

private:

    mfem::Coefficient* dicoef; //diffusion
    mfem::Coefficient* mucoef; //reaction coefficient
    mfem::VectorCoefficient* vecoef; //velocity
    mfem::Coefficient* incoef; //input

    mfem::ParMesh* pmesh;
    int order;

    std::vector<mfem::ConstantCoefficient*> bc;

    std::map<mfem::Coefficient*, mfem::Array<int>> bcc;


    mfem::ParFiniteElementSpace* ufes;
    mfem::ParFiniteElementSpace* pfes;
    mfem::ParFiniteElementSpace* zfes;

    mfem::Array<mfem::ParFiniteElementSpace*> sfes;

    mfem::FiniteElementCollection* ufec;
    mfem::FiniteElementCollection* pfec;
    mfem::FiniteElementCollection* zfec;

    std::vector<mfem::AdvectionDiffusionMX*> nfin;
    mfem::ParBlockNonlinearForm* nf;


    mfem::BlockVector rhs;
    mfem::BlockVector sol;
    mfem::BlockVector adj;

    // forward fields
    mfem::ParGridFunction fprim;
    mfem::ParGridFunction fflux;
    mfem::ParGridFunction fmult;

    // adjoint fields
    mfem::ParGridFunction aprim;
    mfem::ParGridFunction agrad;
    mfem::ParGridFunction amult;

    /// The PETSc objects are allocated once the problem is
    /// assembled. They are utilized in computing the adjoint
    /// solutions.
    mfem::PetscParMatrix* pmat;
    mfem::PetscPreconditioner* prec;
    mfem::PetscLinearSolver*   psol;


    double abs_tol;
    double rel_tol;
    int print_level;
    int max_iter;

    void DirectSolver(mfem::BlockOperator& A);
    void PETSCSolver(mfem::BlockOperator& A);
};








}





#endif
