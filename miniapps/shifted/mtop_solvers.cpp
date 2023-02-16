#include "mfem.hpp"
#include "mtop_solvers.hpp"
#include "integ_algoim.hpp"
#include "marking.hpp"

namespace mfem {

void LinIsoElasticityCoefficient::EvalRes(double EE,double nnu,  double* gradu, double* res)
{
    // generated with maple
    double t1,t2,t3,t5,t10,t12,t19,t22,t25,t27,t32,t33,t38,t41,t46,t51;
    t1 = 1.0+nnu;
    t2 = 1/t1/2.0;
    t3 = t2*EE;
    t5 = EE*nnu;
    t10 = 1/(1.0-2.0*nnu)/t1;
    t12 = t10*t5+2.0*t3;
    t19 = (gradu[0]+gradu[4]+gradu[8])*t10*t5/2.0;
    t22 = gradu[4]*t10*t5/2.0;
    t25 = gradu[8]*t10*t5/2.0;
    t27 = gradu[1]+gradu[3];
    t32 = t27*t3/2.0+t2*t27*EE/2.0;
    t33 = gradu[2]+gradu[6];
    t38 = t33*t3/2.0+t2*t33*EE/2.0;
    t41 = gradu[0]*t10*t5/2.0;
    t46 = gradu[5]+gradu[7];
    t51 = t46*t3/2.0+t2*t46*EE/2.0;
    res[0] = gradu[0]*t12/2.0+gradu[0]*t3+t19+t22+t25;
    res[1] = t32;
    res[2] = t38;
    res[3] = t32;
    res[4] = t41+gradu[4]*t12/2.0+gradu[4]*t3+t19+t25;
    res[5] = t51;
    res[6] = t38;
    res[7] = t51;
    res[8] = t41+t22+gradu[8]*t12/2.0+gradu[8]*t3+t19;
}

void LinIsoElasticityCoefficient::Eval(double EE,double nnu, double* CC)
{
    double t1 = 1.0+nnu;
    double t3 = EE/(2.0*t1);
    double t11 = nnu*EE/(t1*(1.0-2.0*nnu));
    double t12 = 2.0*t3+t11;
    CC[0] = t12;
    CC[1] = 0.0;
    CC[2] = 0.0;
    CC[3] = 0.0;
    CC[4] = t11;
    CC[5] = 0.0;
    CC[6] = 0.0;
    CC[7] = 0.0;
    CC[8] = t11;
    CC[9] = 0.0;
    CC[10] = t3;
    CC[11] = 0.0;
    CC[12] = t3;
    CC[13] = 0.0;
    CC[14] = 0.0;
    CC[15] = 0.0;
    CC[16] = 0.0;
    CC[17] = 0.0;
    CC[18] = 0.0;
    CC[19] = 0.0;
    CC[20] = t3;
    CC[21] = 0.0;
    CC[22] = 0.0;
    CC[23] = 0.0;
    CC[24] = t3;
    CC[25] = 0.0;
    CC[26] = 0.0;
    CC[27] = 0.0;
    CC[28] = t3;
    CC[29] = 0.0;
    CC[30] = t3;
    CC[31] = 0.0;
    CC[32] = 0.0;
    CC[33] = 0.0;
    CC[34] = 0.0;
    CC[35] = 0.0;
    CC[36] = t11;
    CC[37] = 0.0;
    CC[38] = 0.0;
    CC[39] = 0.0;
    CC[40] = t12;
    CC[41] = 0.0;
    CC[42] = 0.0;
    CC[43] = 0.0;
    CC[44] = t11;
    CC[45] = 0.0;
    CC[46] = 0.0;
    CC[47] = 0.0;
    CC[48] = 0.0;
    CC[49] = 0.0;
    CC[50] = t3;
    CC[51] = 0.0;
    CC[52] = t3;
    CC[53] = 0.0;
    CC[54] = 0.0;
    CC[55] = 0.0;
    CC[56] = t3;
    CC[57] = 0.0;
    CC[58] = 0.0;
    CC[59] = 0.0;
    CC[60] = t3;
    CC[61] = 0.0;
    CC[62] = 0.0;
    CC[63] = 0.0;
    CC[64] = 0.0;
    CC[65] = 0.0;
    CC[66] = 0.0;
    CC[67] = 0.0;
    CC[68] = t3;
    CC[69] = 0.0;
    CC[70] = t3;
    CC[71] = 0.0;
    CC[72] = t11;
    CC[73] = 0.0;
    CC[74] = 0.0;
    CC[75] = 0.0;
    CC[76] = t11;
    CC[77] = 0.0;
    CC[78] = 0.0;
    CC[79] = 0.0;
    CC[80] = t12;
}

void LinIsoElasticityCoefficient::EvalStress(DenseMatrix &ss, ElementTransformation &T, const IntegrationPoint &ip)
{
    MFEM_ASSERT(ss.Size()==3,"The size of the stress tensor should be set to 3.");
    double EE=E->Eval(T,ip);
    double nnu=nu->Eval(T,ip);
    //evaluate the strain
    EvalStrain(tmpm,T,ip);
    //evaluate the stress

    double mu=EE/(2.0*(1.0+nnu));
    double ll=nnu*EE/((1.0+nnu)*(1.0-2.0*nnu));

    for(int i=0;i<9;i++)
    {
        ss.GetData()[i]=2.0*mu*tmpm.GetData()[i];
    }

    ss(0,0)=ss(0,0)+ll*(tmpm(0,0)+tmpm(1,1)+tmpm(2,2));
    ss(1,1)=ss(1,1)+ll*(tmpm(0,0)+tmpm(1,1)+tmpm(2,2));
    ss(2,2)=ss(2,2)+ll*(tmpm(0,0)+tmpm(1,1)+tmpm(2,2));
}

void LinIsoElasticityCoefficient::EvalStrain(DenseMatrix &ee, ElementTransformation &T, const IntegrationPoint &ip)
{
    MFEM_ASSERT(ee.Size()==3,"The size of the strain tensor should be set to 3.");
    if(disp==nullptr)
    {
        ee=0.0;
    }
    else
    {
        disp->GetVectorGradient(T,tmpg);
        if(disp->VectorDim()==2)
        {
            ee(0,0)=tmpg(0,0);
            ee(0,1)=0.5*(tmpg(1,0)+tmpg(0,1));
            ee(0,2)=0.0;

            ee(1,0)=ee(0,1);
            ee(1,1)=tmpg(1,1);
            ee(1,2)=0.0;

            ee(2,0)=0.0;
            ee(2,1)=0.0;
            ee(2,2)=0.0;
        }
        else
        {
            ee(0,0)=tmpg(0,0);
            ee(0,1)=0.5*(tmpg(1,0)+tmpg(0,1));
            ee(0,2)=0.5*(tmpg(0,2)+tmpg(2,0));

            ee(1,0)=ee(0,1);
            ee(1,1)=tmpg(1,1);
            ee(1,2)=0.5*(tmpg(1,2)+tmpg(2,1));

            ee(2,0)=ee(0,2);
            ee(2,1)=ee(1,2);
            ee(2,2)=tmpg(2,2);
        }
    }
}


void LinIsoElasticityCoefficient::EvalResidual(Vector &rr, Vector &gradu, ElementTransformation &T, const IntegrationPoint &ip)
{
    MFEM_ASSERT(rr.Size()==9,"The size of the residual should be set to 9.");
    double EE=E->Eval(T,ip);
    double nnu=nu->Eval(T,ip);
    EvalRes(EE,nnu, gradu.GetData(), rr.GetData());

}

void LinIsoElasticityCoefficient::EvalTangent(DenseMatrix &mm, Vector &gradu, ElementTransformation &T, const IntegrationPoint &ip)
{
    MFEM_ASSERT(mm.Size()==9,"The size of the stiffness tensor should be set to 9.");
    double EE=E->Eval(T,ip);
    double nnu=nu->Eval(T,ip);
    Eval(EE,nnu,mm.GetData());
}

double LinIsoElasticityCoefficient::EvalEnergy(Vector &gradu, ElementTransformation &T, const IntegrationPoint &ip)
{
    double EE=E->Eval(T,ip);
    double nnu=nu->Eval(T,ip);
    double t1,t2,t3,t14,t18,t22,t31,t40;
    t1 = 1.0+nnu;
    t2 = 1/t1/2.0;
    t3 = t2*EE;
    t14 = (gradu[0]+gradu[4]+gradu[8])/(1.0-2.0*nnu)/t1*EE*nnu;
    t18 = gradu[1]+gradu[3];
    t22 = gradu[2]+gradu[6];
    t31 = gradu[5]+gradu[7];
    t40 = gradu[0]*(2.0*gradu[0]*t3+t14)/2.0+t18*t18*t2*EE/2.0+t22*t22*t2*EE/
2.0+gradu[4]*(2.0*gradu[4]*t3+t14)/2.0+t31*t31*t2*EE/2.0+gradu[8]*(2.0*gradu[8]
*t3+t14)/2.0;
    return t40;
}

void LinIsoElasticityCoefficient::EvalCompliance(DenseMatrix &C, Vector &stress,
                                                 ElementTransformation &T, const IntegrationPoint &ip)
{
    // the matrix is intended to be used with engineering strain
    double EE=E->Eval(T,ip);
    double nnu=nu->Eval(T,ip);
    if(T.GetDimension()==3){
        C.SetSize(6);
        C=0.0;
        double aa=1.0/EE;
        double bb=-nnu/EE;
        C(0,0)=aa; C(0,1)=bb; C(0,2)=bb;
        C(1,0)=bb; C(1,1)=aa; C(1,2)=bb;
        C(2,0)=bb; C(2,1)=bb; C(2,2)=aa;
        double cc=2.0*(1.0+nnu)/EE;
        C(3,3)=cc;
        C(4,4)=cc;
        C(5,5)=cc;
    }else{
        C.SetSize(3);
        C=0.0;
        double aa=1.0/EE;
        double bb=-nnu/EE;
        C(0,0)=aa; C(0,1)=bb;
        C(1,0)=bb; C(1,1)=aa;
        double cc=2.0*(1.0+nnu)/EE;
        C(2,2)=cc;
    }
}

void LinIsoElasticityCoefficient::EvalStiffness(DenseMatrix &D, Vector &strain,
                                                ElementTransformation &T, const IntegrationPoint &ip)
{
    double EE=E->Eval(T,ip);
    double nnu=nu->Eval(T,ip);
    if(T.GetDimension()==3){ //3D problem
        D.SetSize(6);
        elast::IsotropicStiffnessTensor3D(EE,nnu,D);
    }else{
        D.SetSize(3); //2D problem
        elast::IsotropicStiffnessTensor2D(EE,nnu,D);
    }
}


void NLSurfLoadIntegrator::AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                                              FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
    elvect.SetSize(elfun.Size());
    elvect=0.0;

    if(Tr.Attribute!=sid){ return;}

    int dim=Tr.GetSpaceDim();
    const int dof=el1.GetDof();
    mfem::Vector force; force.SetSize(dim);
    mfem::Vector shape; shape.SetSize(dof);
    int order=2*el1.GetOrder();
    const IntegrationRule *ir = &IntRules.Get(Tr.GetGeometryType(), order);

    double w;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetAllIntPoints(&ip);
        const IntegrationPoint &eip = Tr.GetElement1IntPoint();
        el1.CalcShape(eip, shape);
        w = Tr.Weight() * ip.weight;
        vc->Eval(force,Tr,ip);
        for(int j=0;j<dof;j++){
            for(int d=0;d<dim;d++){
                elvect[j+d*dof]=elvect[j+d*dof]-w*shape[j]*force[d];
            }
        }
    }
}

void NLSurfLoadIntegrator::AssembleFaceGrad(const FiniteElement &el1, const FiniteElement &el2,
                                            FaceElementTransformations &Tr, const Vector &elfun, DenseMatrix &elmat)
{
    elmat.SetSize(elfun.Size());
    elmat=0.0;
}


double NLVolForceIntegrator::GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun)
{
    return 0.0;
}

void NLVolForceIntegrator::AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                                                 const Vector &elfun, Vector &elvect)
{
    const int dof=el.GetDof();
    const int dim=el.GetDim();
    elvect.SetSize(dof*dim); elvect=0.0;

    if(force==nullptr){return;}

    //vol force
    mfem::Vector vforce;
    vforce.SetSize(dim); vforce=0.0;
    mfem::Vector shapef;
    shapef.SetSize(dof);

    Vector ru(elvect.GetData()+0*dof,dof);
    Vector rv(elvect.GetData()+1*dof,dof);
    Vector rw;
    if(dim==3){
        rw.SetDataAndSize(elvect.GetData()+2*dof,dof);
    }

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el);
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    double w;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcPhysShape(Tr,shapef);
        force->Eval(vforce,Tr,ip);
        ru.Add(-vforce[0]*w,shapef);
        rv.Add(-vforce[1]*w,shapef);
        if(dim==3){
            rw.Add(-vforce[2]*w,shapef);
        }
    }
}

void NLVolForceIntegrator::AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                                               const Vector &elfun, DenseMatrix &elmat)
{
    elmat.SetSize(elfun.Size());
    elmat=0.0;
}

double NLElasticityIntegrator::GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun)
{
    const int dof=el.GetDof();
    const int dim=el.GetDim();
    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("NLElasticityIntegrator::GetElementEnergy is not define on manifold meshes.");
        }
    }

    if(elco==nullptr){	return 0.0;}

    //gradients
    mfem::DenseMatrix bsu; bsu.SetSize(dof,dim);
    Vector uu(elfun.GetData()+0*dof,dof);
    Vector vv(elfun.GetData()+1*dof,dof);

    Vector ww;
    if(dim==3){
        ww.SetDataAndSize(elfun.GetData()+2*dof,dof);
    }

    Vector gradu(9); gradu=0.0;
    Vector sh;

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el);
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    double w;
    double energy=0.0;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcPhysDShape(Tr,bsu);

        sh.SetDataAndSize(gradu.GetData()+3*0,dim);
        bsu.MultTranspose(uu,sh);
        sh.SetDataAndSize(gradu.GetData()+3*1,dim);
        bsu.MultTranspose(vv,sh);
        if(dim==3)
        {
            sh.SetDataAndSize(gradu.GetData()+3*2,dim);
            bsu.MultTranspose(ww,sh);
        }

        // Calcualte the residual at the integration point
        energy=energy+ w*(elco->EvalEnergy(gradu,Tr,ip));
    }
    return energy;
}

void NLElasticityIntegrator::AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun, Vector &elvect)
{
    const int dof=el.GetDof();
    const int dim=el.GetDim();
    elvect.SetSize(dof*dim); elvect=0.0;

    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("NLElasticityIntegrator::AssembleElementVector is not defined on manifold meshes.");
        }
    }

    if(elco==nullptr){	return;}

    //gradients
    mfem::DenseMatrix bsu; bsu.SetSize(dof,dim);
    Vector uu(elfun.GetData()+0*dof,dof);
    Vector vv(elfun.GetData()+1*dof,dof);

    Vector ru(elvect.GetData()+0*dof,dof);
    Vector rv(elvect.GetData()+1*dof,dof);

    Vector ww;
    Vector rw;
    if(dim==3){
        ww.SetDataAndSize(elfun.GetData()+2*dof,dof);
        rw.SetDataAndSize(elvect.GetData()+2*dof,dof);
    }else{
        ww.SetSize(dof); ww=0.0;
    }

    Vector gradu(9); gradu=0.0;
    Vector rr(9);
    Vector sh;

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el);
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    double w;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcPhysDShape(Tr,bsu);

        sh.SetDataAndSize(gradu.GetData()+3*0,dim);
        bsu.MultTranspose(uu,sh);
        sh.SetDataAndSize(gradu.GetData()+3*1,dim);
        bsu.MultTranspose(vv,sh);
        if(dim==3)
        {
            sh.SetDataAndSize(gradu.GetData()+3*2,dim);
            bsu.MultTranspose(ww,sh);
        }

        // Calcualte the residual at the integration point
        elco->EvalResidual(rr,gradu,Tr,ip);

        sh.SetDataAndSize(rr.GetData()+3*0,dim);
        bsu.AddMult_a(w,sh,ru);
        sh.SetDataAndSize(rr.GetData()+3*1,dim);
        bsu.AddMult_a(w,sh,rv);
        if(dim==3)
        {
            sh.SetDataAndSize(rr.GetData()+3*2,dim);
            bsu.AddMult_a(w,sh,rw);
        }
    }
}


void NLElasticityIntegrator::AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun, DenseMatrix &elmat)
{
    const int dof=el.GetDof();
    const int dim=el.GetDim();
    elmat.SetSize(dof*dim); elmat=0.0;

    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("NLElasticityIntegrator::AssembleElementGrad is not defined on manifold meshes.");
        }
    }

    if(elco==nullptr){	return;}

    //gradients
    mfem::DenseMatrix bsu; bsu.SetSize(dof,dim);
    Vector uu(elfun.GetData()+0*dof,dof);
    Vector vv(elfun.GetData()+1*dof,dof);
    Vector ww;
    if(dim==3)
    {
        ww.SetDataAndSize(elfun.GetData()+2*dof,dof);
    }

    Vector gradu(9); gradu=0.0;
    Vector sh;
    // state matrix at integration point
    DenseMatrix mm; mm.SetSize(9);
    DenseMatrix mh; mh.SetSize(dim);
    DenseMatrix th; th.SetSize(dof,dim);
    DenseMatrix rh; rh.SetSize(dof);

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el);
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    double w;

    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcPhysDShape(Tr,bsu);

        sh.SetDataAndSize(gradu.GetData()+3*0,dim);
        bsu.MultTranspose(uu,sh);
        sh.SetDataAndSize(gradu.GetData()+3*1,dim);
        bsu.MultTranspose(vv,sh);
        if(dim==3)
        {
            sh.SetDataAndSize(gradu.GetData()+3*2,dim);
            bsu.MultTranspose(ww,sh);
        }

        // calculate the tangent matrix
        elco->EvalTangent(mm,gradu,Tr,ip);

        for(int ii=0;ii<dim;ii++){
        for(int jj=0;jj<dim;jj++){
            mh.CopyMN(mm,dim,dim,ii*3,jj*3);
            mh.Transpose();
            MultABt(bsu,mh,th);
            MultABt(th,bsu,rh);
            elmat.AddMatrix(w,rh,ii*dof,jj*dof);
        }}
    }
}


ElasticitySolver::ElasticitySolver(mfem::ParMesh* mesh_, int vorder)
{
    pmesh=mesh_;
    int dim=pmesh->Dimension();
    vfec=new H1_FECollection(vorder,dim);
    vfes=new mfem::ParFiniteElementSpace(pmesh,vfec,dim, Ordering::byVDIM);

    fdisp.SetSpace(vfes); fdisp=0.0;
    adisp.SetSpace(vfes); adisp=0.0;

    sol.SetSize(vfes->GetTrueVSize()); sol=0.0;
    rhs.SetSize(vfes->GetTrueVSize()); rhs=0.0;
    adj.SetSize(vfes->GetTrueVSize()); adj=0.0;

    nf=nullptr;
    SetNewtonSolver();
    SetLinearSolver();

    prec=nullptr;
    ls=nullptr;
    ns=nullptr;

    lvforce=nullptr;
    volforce=nullptr;
}

ElasticitySolver::~ElasticitySolver()
{
    delete ns;
    delete prec;
    delete ls;
    delete nf;

    delete vfes;
    delete vfec;

    delete lvforce;

    for(unsigned int i=0;i<materials.size();i++)
    {
        delete materials[i];
    }

    for(auto it=surf_loads.begin();it!=surf_loads.end();it++){
        delete it->second;
    }
}

void ElasticitySolver::SetNewtonSolver(double rtol, double atol,int miter, int prt_level)
{
    rel_tol=rtol;
    abs_tol=atol;
    max_iter=miter;
    print_level=prt_level;
}

void ElasticitySolver::SetLinearSolver(double rtol, double atol, int miter)
{
    linear_rtol=rtol;
    linear_atol=atol;
    linear_iter=miter;
}

void ElasticitySolver::AddDispBC(int id, int dir, double val)
{
    if(dir==0){
        bcx[id]=mfem::ConstantCoefficient(val);
        AddDispBC(id,dir,bcx[id]);
    }
    if(dir==1){
        bcy[id]=mfem::ConstantCoefficient(val);
        AddDispBC(id,dir,bcy[id]);

    }
    if(dir==2){
        bcz[id]=mfem::ConstantCoefficient(val);
        AddDispBC(id,dir,bcz[id]);
    }
    if(dir==4){
        bcx[id]=mfem::ConstantCoefficient(val);
        bcy[id]=mfem::ConstantCoefficient(val);
        bcz[id]=mfem::ConstantCoefficient(val);
        AddDispBC(id,0,bcx[id]);
        AddDispBC(id,1,bcy[id]);
        AddDispBC(id,2,bcz[id]);
    }
}

void ElasticitySolver::DelDispBC()
{
    bccx.clear();
    bccy.clear();
    bccz.clear();

    bcx.clear();
    bcy.clear();
    bcz.clear();

    ess_tdofv.DeleteAll();
}

void ElasticitySolver::AddDispBC(int id, int dir, Coefficient &val)
{
    if(dir==0){ bccx[id]=&val; }
    if(dir==1){ bccy[id]=&val; }
    if(dir==2){ bccz[id]=&val; }
    if(dir==4){ bccx[id]=&val; bccy[id]=&val; bccz[id]=&val;}
    if(pmesh->Dimension()==2)
    {
        bccz.clear();
    }
}

void ElasticitySolver::SetVolForce(double fx, double fy, double fz)
{
    delete lvforce;
    int dim=pmesh->Dimension();
    mfem::Vector ff(dim); ff(0)=fx; ff(1)=fy;
    if(dim==3){ff(2)=fz;}
    lvforce=new mfem::VectorConstantCoefficient(ff);
    volforce=lvforce;

}

void ElasticitySolver::SetVolForce(mfem::VectorCoefficient& fv)
{
    volforce=&fv;
}

void ElasticitySolver::FSolve()
{
    // Set the BC
    ess_tdofv.DeleteAll();
    if(nf!=nullptr){delete nf; nf=nullptr;}
    Array<int> ess_tdofx;
    Array<int> ess_tdofy;
    Array<int> ess_tdofz;

    int dim=pmesh->Dimension();
    {
        for(auto it=bccx.begin();it!=bccx.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,0);
            ess_tdofx.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);
            pcoeff.Set(0, it->second, false);
            fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }

        //copy tdofsx from velocity grid function
        {
            fdisp.GetTrueDofs(rhs); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofx.Size();ii++)
            {
                sol[ess_tdofx[ii]]=rhs[ess_tdofx[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofx);

        for(auto it=bccy.begin();it!=bccy.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,1);
            ess_tdofy.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);
            pcoeff.Set(1, it->second, false);
            fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsy from velocity grid function
        {
            fdisp.GetTrueDofs(rhs); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofy.Size();ii++)
            {
                sol[ess_tdofy[ii]]=rhs[ess_tdofy[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofy);

        if(dim==3){
            for(auto it=bccz.begin();it!=bccz.end();it++)
            {
                mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
                ess_bdr=0;
                ess_bdr[it->first -1]=1;
                mfem::Array<int> ess_tdof_list;
                vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,2);
                ess_tdofz.Append(ess_tdof_list);

                mfem::VectorArrayCoefficient pcoeff(dim);
                pcoeff.Set(2, it->second, false);
                fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
            }

            //copy tdofsz from velocity grid function
            {
                fdisp.GetTrueDofs(rhs); // use the rhs vector as a tmp vector
                for(int ii=0;ii<ess_tdofz.Size();ii++)
                {
                    sol[ess_tdofz[ii]]=rhs[ess_tdofz[ii]];
                }
            }
            ess_tdofv.Append(ess_tdofz);
        }
    }

    //allocate the nf
    if(nf==nullptr)
    {
        nf=new mfem::ParNonlinearForm(vfes);
        //add the integrators
        for(unsigned int i=0;i<materials.size();i++)
        {
            nf->AddDomainIntegrator(new NLElasticityIntegrator(materials[i]) );
        }

        if(volforce!=nullptr){
            nf->AddDomainIntegrator(new NLVolForceIntegrator(volforce));
        }

        //mfem::Array<int> bdre(pmesh->bdr_attributes.Max());

        for(auto it=surf_loads.begin();it!=surf_loads.end();it++){
            nf->AddBdrFaceIntegrator(new NLSurfLoadIntegrator(it->first,it->second));
        }

        for(auto it=load_coeff.begin();it!=load_coeff.end();it++){
            nf->AddBdrFaceIntegrator(new NLSurfLoadIntegrator(it->first,it->second));
        }
    }


    nf->SetEssentialTrueDofs(ess_tdofv);

    //allocate the solvers
    if(ns==nullptr)
    {
        ns=new mfem::NewtonSolver(pmesh->GetComm());
        ls=new mfem::CGSolver(pmesh->GetComm());
        prec=new mfem::HypreBoomerAMG();
        prec->SetSystemsOptions(pmesh->Dimension());
        prec->SetElasticityOptions(vfes);
    }

    //set the parameters
    ns->SetSolver(*ls);
    ns->SetOperator(*nf);
    ns->SetPrintLevel(print_level);
    ns->SetRelTol(rel_tol);
    ns->SetAbsTol(abs_tol);
    ns->SetMaxIter(max_iter);

    ls->SetPrintLevel(print_level);
    ls->SetAbsTol(linear_atol);
    ls->SetRelTol(linear_rtol);
    ls->SetMaxIter(linear_iter);
    ls->SetPreconditioner(*prec);

    prec->SetPrintLevel(print_level);

    //solve the problem
    Vector b;
    ns->Mult(b,sol);
}


double ComplianceNLIntegrator::GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun)
{

    if(disp==nullptr){return 0.0;}


    //integrate the dot product disp*volforce

    //const int dof=el.GetDof();
    const int dim=el.GetDim();
    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("ComplianceNLIntegrator::GetElementEnergy is not define on manifold meshes.");
        }
    }

    Vector uu; uu.SetSize(dim);
    //Vector ff; ff.SetSize(dim);

    DenseMatrix grads; grads.SetSize(dim);
    DenseMatrix strains; strains.SetSize(dim);
    DenseMatrix CC;
    if(dim==3){CC.SetSize(6);}
    else{CC.SetSize(3);}
    Vector engstrain;
    Vector engstress;
    if(dim==3){engstrain.SetSize(6);}
    else{engstrain.SetSize(3);}
    engstress.SetSize(engstrain.Size());


    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el)+disp->FESpace()->GetOrder(Tr.ElementNo);
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    double w;
    double energy=0.0;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        disp->GetVectorValue(Tr,ip,uu);
        //volforce->Eval(ff,Tr,ip);

        //energy=energy+w*(ff*uu);


        disp->GetVectorGradient(Tr,grads);
        double E=Ecoef->Eval(Tr,ip);
        if(dim==2)
        {
            elast::EvalLinStrain2D(grads,strains);
            elast::Convert2DVoigtStrain(strains,engstrain);
            elast::IsotropicStiffnessTensor2D(E,nu,CC);
        }else{//dim==3
            elast::EvalLinStrain3D(grads,strains);
            elast::Convert3DVoigtStrain(strains,engstrain);
            elast::IsotropicStiffnessTensor3D(E,nu,CC);
        }
        CC.Mult(engstrain,engstress);
        energy=energy+w*(engstrain*engstress);


    }

    return energy;

}

//the finite element space is the space of the filtered design
void ComplianceNLIntegrator::AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                           const Vector &elfun, Vector &elvect)
{

    const int dof=el.GetDof();
    const int dim=el.GetDim();
    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("ComplianceNLIntegrator::AssembleElementVector is not define on manifold meshes.");
        }
    }

    elvect.SetSize(dof); elvect=0.0;
    if(disp==nullptr){return;}

    Vector shapef(dof);

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el)+2*(disp->FESpace()->GetOrder(Tr.ElementNo));
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    DenseMatrix grads; grads.SetSize(dim);
    DenseMatrix strains; strains.SetSize(dim);
    DenseMatrix CC;
    if(dim==3){CC.SetSize(6);}
    else{CC.SetSize(3);}
    Vector engstrain;
    Vector engstress;
    if(dim==3){engstrain.SetSize(6);}
    else{engstrain.SetSize(3);}
    engstress.SetSize(engstrain.Size());

    double w;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        double E=Ecoef->Eval(Tr,ip);

        disp->GetVectorGradient(Tr,grads);
        //evaluate strains
        //evaluate the compliance at the integration point
        //evaluate the gradient of the E modulus with respect to the filtered field
        if(dim==2)
        {
            elast::EvalLinStrain2D(grads,strains);
            elast::Convert2DVoigtStrain(strains,engstrain);
            elast::IsotropicStiffnessTensor2D(1.0,nu,CC);
        }else{//dim==3
            elast::EvalLinStrain3D(grads,strains);
            elast::Convert3DVoigtStrain(strains,engstrain);
            elast::IsotropicStiffnessTensor3D(1.0,nu,CC);
        }

        CC.Mult(engstrain,engstress);
        double cpl=engstrain*engstress; //compute the compliance
        cpl=cpl*Ecoef->Grad(Tr,ip); //mult by the gradient
        cpl=-cpl*w;
        el.CalcShape(ip,shapef);
        elvect.Add(cpl,shapef);
    }
}

void ComplianceNLIntegrator::AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                         const Vector &elfun, DenseMatrix &elmat)
{

        {
            mfem::mfem_error("ComplianceNLIntegrator::AssembleElementGrad is not defined!");
        }
}

double ComplianceObjective::Eval(mfem::ParGridFunction& sol)
{
    if(Ecoef==nullptr){
        MFEM_ABORT("Ecoef in ComplianceObjective should be set before calling the Eval method!");
    }

    if(dfes==nullptr){
        MFEM_ABORT("fsolv of dfes in ComplianceObjective should be set before calling the Eval method!");
    }

    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new ComplianceNLIntegrator();
        nf->AddDomainIntegrator(intgr);
    }

    intgr->SetE(Ecoef);
    intgr->SetPoissonRatio(nu);
    intgr->SetDisp(&sol);

    double rt=nf->GetEnergy(*dens);

    return rt;

}

double ComplianceObjective::Eval()
{
    if(Ecoef==nullptr){
        MFEM_ABORT("Ecoef in ComplianceObjective should be set before calling the Eval method!");
    }

    if(esolv==nullptr){
        MFEM_ABORT("esolv in ComplianceObjective should be set before calling the Eval method!");
    }

    if(dfes==nullptr){
        MFEM_ABORT("fsolv of dfes in ComplianceObjective should be set before calling the Eval method!");
    }

    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new ComplianceNLIntegrator();
        nf->AddDomainIntegrator(intgr);
    }

    intgr->SetE(Ecoef);
    intgr->SetPoissonRatio(nu);
    intgr->SetDisp(&(esolv->GetDisplacements()));

    double rt=nf->GetEnergy(*dens);

    return rt;
}

void ComplianceObjective::Grad(mfem::ParGridFunction& sol, Vector& grad)
{
    if(Ecoef==nullptr){
        MFEM_ABORT("Ecoef in ComplianceObjective should be set before calling the Grad method!");
    }
    if(dfes==nullptr){
        MFEM_ABORT("fsolv or dfes in ComplianceObjective should be set before calling the Grad method!");
    }
    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new ComplianceNLIntegrator();
        nf->AddDomainIntegrator(intgr);
    }
    intgr->SetE(Ecoef);
    intgr->SetPoissonRatio(nu);
    intgr->SetDisp(&sol);
    nf->Mult(*dens,grad);
}

void ComplianceObjective::Grad(Vector& grad)
{
    if(Ecoef==nullptr){
        MFEM_ABORT("Ecoef in ComplianceObjective should be set before calling the Grad method!");
    }

    if(esolv==nullptr){
        MFEM_ABORT("esolv in ComplianceObjective should be set before calling the Grad method!");
    }

    if(dfes==nullptr){
        MFEM_ABORT("fsolv or dfes in ComplianceObjective should be set before calling the Grad method!");
    }

    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new ComplianceNLIntegrator();
        nf->AddDomainIntegrator(intgr);
    }
    intgr->SetE(Ecoef);
    intgr->SetPoissonRatio(nu);
    intgr->SetDisp(&(esolv->GetDisplacements()));

    nf->Mult(*dens,grad);
}



void ElastGhostPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe1,
                                        const FiniteElement &fe2,
                                        FaceElementTransformations &Tr,
                                        DenseMatrix &elmat)
{
    const int ndim=Tr.GetSpaceDim();
    int elem1 = Tr.Elem1No;
    int elem2 = Tr.Elem2No;

    if(elem2<0){
        elmat.SetSize(fe1.GetDof()*ndim);
        elmat=0.0;
        return;
    }

    const int ndof1 = fe1.GetDof();
    const int ndof2 = fe2.GetDof();
    const int ndofs = ndof1+ndof2;

    elmat.SetSize(ndofs*ndim);
    elmat=0.0;

    bool flag=false;
    if((*markers)[elem1]==ElementMarker::SBElementType::OUTSIDE)
    {flag=true;}
    if((*markers)[elem2]==ElementMarker::SBElementType::OUTSIDE)
    {flag=true;}
    if((*markers)[elem1]==ElementMarker::SBElementType::INSIDE){
    if((*markers)[elem2]==ElementMarker::SBElementType::INSIDE){
        flag=true;
    }}

    if(flag){ return; }

    ElementTransformation &Tr1 = Tr.GetElement1Transformation();
    ElementTransformation &Tr2 = Tr.GetElement2Transformation();

    DenseMatrix ngrad1;
    DenseMatrix ngrad2;

    DenseMatrix dgrad1(ndof1);
    DenseMatrix dgrad2(ndof2);

    DenseMatrix m11(ndof1); m11=0.0;
    DenseMatrix m12(ndof1,ndof2); m12=0.0;
    DenseMatrix m22(ndof2); m22=0.0;

    Vector nor; nor.SetSize(ndim);
    Vector unn; unn.SetSize(ndim); //unit normal
    DenseMatrix bmat1(ndof1,ndim);
    DenseMatrix bmat2(ndof2,ndim);

    Vector tmpv1(ndof1);
    Vector tmpv2(ndof2);
    Vector rmpv1(ndof1);
    Vector rmpv2(ndof2);


    const IntegrationRule *ir;
    const int order = 2.0*std::max(fe1.GetOrder(), fe2.GetOrder())+1;
    const int eorder = std::max(fe1.GetOrder(), fe2.GetOrder());

    // extract the projection matrices only if we need them
    if(eorder>1){
        fe1.ProjectGrad(fe1,Tr1,ngrad1);
        fe2.ProjectGrad(fe2,Tr2,ngrad2);
    }


    ir = &IntRules.Get(Tr.GetGeometryType(), order);
    double w;
    double h;
    for (int q = 0; q < ir->GetNPoints(); q++)
    {
        const IntegrationPoint &ip = ir->IntPoint(q);
        Tr.SetAllIntPoints(&ip);
        //const IntegrationPoint &ip1 = Tr.GetElement1IntPoint();
        //const IntegrationPoint &ip2 = Tr.GetElement2IntPoint();

        //compute the normal
        // Note: this normal accounts for the weight of the surface transformation
        // Jacobian i.e. nnor = nhat*det(J)
        CalcOrtho(Tr.Jacobian(), nor);
        unn=nor; unn/=nor.Norml2();

        fe1.CalcPhysDShape(Tr1,bmat1);
        fe2.CalcPhysDShape(Tr2,bmat2);

        bmat1.Mult(unn,tmpv1);
        bmat2.Mult(unn,tmpv2);


        h=std::pow(Tr1.Weight(),1.0/double(ndim));

        w=ip.weight*Tr.Weight()*beta*h*h*h;
        AddMult_a_VVt(w,tmpv1,m11);
        AddMult_a_VWt(-w,tmpv1,tmpv2,m12);
        AddMult_a_VVt(w,tmpv2,m22);

        if(eorder>1){
            //initialize directional gradient 1
            for(int i=0;i<ndof1;i++){
            for(int j=0;j<ndof1;j++){
                dgrad1(i,j)=unn[0]*ngrad1(0*ndof1+i,j);
            }}

            //initialize directional gradient 2
            for(int i=0;i<ndof2;i++){
            for(int j=0;j<ndof2;j++){
                dgrad2(i,j)=unn[0]*ngrad2(0*ndof2+i,j);
            }}

            for(int d=1;d<ndim;d++){
                for(int i=0;i<ndof1;i++){
                for(int j=0;j<ndof1;j++){
                    dgrad1(i,j)+=unn[d]*ngrad1(d*ndof1+i,j);
                }}

                for(int i=0;i<ndof2;i++){
                for(int j=0;j<ndof2;j++){
                    dgrad2(i,j)+=unn[d]*ngrad2(d*ndof2+i,j);
                }}
            }
            //now we have the direction gradients for
            //element 1 and 2 dgrad1 and dgrad2
            for(int o=1;o<eorder;o++){
                w=w*h*h;
                dgrad1.MultTranspose(tmpv1,rmpv1);
                tmpv1=rmpv1;
                dgrad2.MultTranspose(tmpv2,rmpv2);
                tmpv2=rmpv2;

                AddMult_a_VVt(w,tmpv1,m11);
                AddMult_a_VWt(-w,tmpv1,tmpv2,m12);
                AddMult_a_VVt(w,tmpv2,m22);
            }
        }
    }

    //add the matricess to the global matrix
    for(int d=0;d<ndim;d++){
        for(int i=0;i<ndof1;i++){
        for(int j=0;j<ndof1;j++){
            elmat(d*ndof1+i,d*ndof2+j)=m11(i,j);
        }}

        for(int i=0;i<ndof2;i++){
        for(int j=0;j<ndof2;j++){
            elmat(ndim*ndof1+d*ndof2+i,ndim*ndof1+d*ndof2+j)=m22(i,j);
        }}

        for(int i=0;i<ndof1;i++){
        for(int j=0;j<ndof2;j++){
            int ii=d*ndof1+i;
            int jj=ndim*ndof1+d*ndof2+j;
            elmat(ii,jj)=m12(i,j);
            elmat(jj,ii)=m12(i,j);
        }}
    }


}




#ifdef MFEM_USE_ALGOIM

CFElasticitySolver::CFElasticitySolver(ParMesh* mesh_, int vorder)
{
    pmesh=mesh_;
    int dim=pmesh->Dimension();
    vfec=new H1_FECollection(vorder,dim);
    vfes=new ParFiniteElementSpace(pmesh,vfec,dim, Ordering::byVDIM);

    fdisp.SetSpace(vfes); fdisp=0.0;
    adisp.SetSpace(vfes); adisp=0.0;

    sol.SetSize(vfes->GetTrueVSize()); sol=0.0;
    rhs.SetSize(vfes->GetTrueVSize()); rhs=0.0;
    adj.SetSize(vfes->GetTrueVSize()); adj=0.0;

    nf=nullptr;
    SetNewtonSolver();
    SetLinearSolver();

    prec=nullptr;
    ls=nullptr;
    ns=nullptr;

    lvforce=nullptr;
    volforce=nullptr;

    level_set_function=nullptr;
    el_markers=nullptr;
}

CFElasticitySolver::~CFElasticitySolver()
{
    delete ns;
    delete prec;
    delete ls;
    delete nf;

    delete vfes;
    delete vfec;

    delete lvforce;

    for(unsigned int i=0;i<materials.size();i++)
    {
        delete materials[i];
        delete vforces[i];
        delete sforces[i];
    }

    for(auto it=surf_loads.begin();it!=surf_loads.end();it++){
        delete it->second;
    }
}

void CFElasticitySolver::SetNewtonSolver(double rtol, double atol,int miter, int prt_level)
{
    rel_tol=rtol;
    abs_tol=atol;
    max_iter=miter;
    print_level=prt_level;
}

void CFElasticitySolver::SetLinearSolver(double rtol, double atol, int miter)
{
    linear_rtol=rtol;
    linear_atol=atol;
    linear_iter=miter;
}

void CFElasticitySolver::AddDispBC(int id, int dir, double val)
{
    if(dir==0){
        bcx[id]=mfem::ConstantCoefficient(val);
        AddDispBC(id,dir,bcx[id]);
    }
    if(dir==1){
        bcy[id]=mfem::ConstantCoefficient(val);
        AddDispBC(id,dir,bcy[id]);

    }
    if(dir==2){
        bcz[id]=mfem::ConstantCoefficient(val);
        AddDispBC(id,dir,bcz[id]);
    }
    if(dir==4){
        bcx[id]=mfem::ConstantCoefficient(val);
        bcy[id]=mfem::ConstantCoefficient(val);
        bcz[id]=mfem::ConstantCoefficient(val);
        AddDispBC(id,0,bcx[id]);
        AddDispBC(id,1,bcy[id]);
        AddDispBC(id,2,bcz[id]);
    }
}

void CFElasticitySolver::DelDispBC()
{
    bccx.clear();
    bccy.clear();
    bccz.clear();

    bcx.clear();
    bcy.clear();
    bcz.clear();

    ess_tdofv.DeleteAll();
}

void CFElasticitySolver::AddDispBC(int id, int dir, Coefficient &val)
{
    if(dir==0){ bccx[id]=&val; }
    if(dir==1){ bccy[id]=&val; }
    if(dir==2){ bccz[id]=&val; }
    if(dir==4){ bccx[id]=&val; bccy[id]=&val; bccz[id]=&val;}
    if(pmesh->Dimension()==2)
    {
        bccz.clear();
    }
}
/// Adds displacement BC in all directions.
void CFElasticitySolver::AddDispBC(int id, mfem::VectorCoefficient& val)
{
    bccv[id]=&val;
}


void CFElasticitySolver::SetVolForce(double fx, double fy, double fz)
{
    delete lvforce;
    int dim=pmesh->Dimension();
    mfem::Vector ff(dim); ff(0)=fx; ff(1)=fy;
    if(dim==3){ff(2)=fz;}
    lvforce=new mfem::VectorConstantCoefficient(ff);
    volforce=lvforce;

}

void CFElasticitySolver::SetVolForce(mfem::VectorCoefficient& fv)
{
    volforce=&fv;
}

void CFElasticitySolver::FSolve()
{
    // Set the BC
    ess_tdofv.DeleteAll();
    if(nf!=nullptr){delete nf; nf=nullptr;}
    Array<int> ess_tdofx;
    Array<int> ess_tdofy;
    Array<int> ess_tdofz;

    int dim=pmesh->Dimension();
    {
        for(auto it=bccx.begin();it!=bccx.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,0);
            ess_tdofx.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);
            pcoeff.Set(0, it->second, false);
            fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }

        //copy tdofsx from velocity grid function
        {
            fdisp.GetTrueDofs(rhs); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofx.Size();ii++)
            {
                sol[ess_tdofx[ii]]=rhs[ess_tdofx[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofx);

        for(auto it=bccy.begin();it!=bccy.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,1);
            ess_tdofy.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);
            pcoeff.Set(1, it->second, false);
            fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsy from velocity grid function
        {
            fdisp.GetTrueDofs(rhs); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofy.Size();ii++)
            {
                sol[ess_tdofy[ii]]=rhs[ess_tdofy[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofy);

        if(dim==3){
            for(auto it=bccz.begin();it!=bccz.end();it++)
            {
                mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
                ess_bdr=0;
                ess_bdr[it->first -1]=1;
                mfem::Array<int> ess_tdof_list;
                vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,2);
                ess_tdofz.Append(ess_tdof_list);

                mfem::VectorArrayCoefficient pcoeff(dim);
                pcoeff.Set(2, it->second, false);
                fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
            }

            //copy tdofsz from velocity grid function
            {
                fdisp.GetTrueDofs(rhs); // use the rhs vector as a tmp vector
                for(int ii=0;ii<ess_tdofz.Size();ii++)
                {
                    sol[ess_tdofz[ii]]=rhs[ess_tdofz[ii]];
                }
            }
            ess_tdofv.Append(ess_tdofz);
        }

        //vector bc
        mfem::Array<int> ess_tdofa;
        for(auto it=bccv.begin();it!=bccv.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
            ess_tdofa.Append(ess_tdof_list);
            fdisp.ProjectBdrCoefficient(*(it->second), ess_bdr);
        }
        //copy tdofs from velocity grid function
        {
            fdisp.GetTrueDofs(rhs); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofa.Size();ii++)
            {
                sol[ess_tdofa[ii]]=rhs[ess_tdofa[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofa);

    }

    if(el_markers!=nullptr)
    {
        //set the void domain dofs
        Array<int> dofs;

        mfem::Vector vvdof; vvdof.SetSize(vfes->GetVSize()); vvdof=0.0;
        for(int i=0;i<vfes->GetNE();i++)
        {
            if((*el_markers)[i]==ElementMarker::SBElementType::INSIDE){
                vfes->GetElementVDofs(i,dofs);
                for(int j=0;j<dofs.Size();j++){
                    vvdof[dofs[j]]=1.0;
                }
            }
            else
            if((*el_markers)[i]==ElementMarker::SBElementType::CUT){
                vfes->GetElementVDofs(i,dofs);
                for(int j=0;j<dofs.Size();j++){
                    vvdof[dofs[j]]=1.0;
                }

            }
        }

        Array<int> tdof_mark; tdof_mark.SetSize(vfes->GetTrueVSize());
        Vector vtdof; vtdof.SetSize(vfes->GetTrueVSize()); vtdof=0.0;
        vfes->GetProlongationMatrix()->MultTranspose(vvdof,vtdof);
        for(int i=0;i<vtdof.Size();i++){
            if(vtdof[i]<1.0){tdof_mark[i]=1;}
            else{tdof_mark[i]=0;}
        }
        Array<int> vtdof_list;
        vfes->MarkerToList(tdof_mark, vtdof_list);
        ess_tdofv.Append(vtdof_list);
    }

    //allocate the nf
    if(nf==nullptr)
    {
        nf=new mfem::ParNonlinearForm(vfes);
        //add the integrators
        for(unsigned int i=0;i<materials.size();i++)
        {
            CFNLElasticityIntegrator* pck=new CFNLElasticityIntegrator(materials[i]);
            pck->SetVolumetricForce(*(vforces[i]));
            pck->SetSurfaceLoad(*(sforces[i]));
            pck->SetLSF(*level_set_function,*el_markers);
            nf->AddDomainIntegrator(pck);
        }

        if(volforce!=nullptr){
            nf->AddDomainIntegrator(new NLVolForceIntegrator(volforce));
        }

        //mfem::Array<int> bdre(pmesh->bdr_attributes.Max());

        for(auto it=surf_loads.begin();it!=surf_loads.end();it++){
            nf->AddBdrFaceIntegrator(new NLSurfLoadIntegrator(it->first,it->second));
        }

        for(auto it=load_coeff.begin();it!=load_coeff.end();it++){
            nf->AddBdrFaceIntegrator(new NLSurfLoadIntegrator(it->first,it->second));
        }
    }



    nf->SetEssentialTrueDofs(ess_tdofv);

    //allocate the solvers
    if(ns==nullptr)
    {
        ns=new mfem::NewtonSolver(pmesh->GetComm());
        ls=new mfem::CGSolver(pmesh->GetComm());
        prec=new mfem::HypreBoomerAMG();
        prec->SetSystemsOptions(pmesh->Dimension());
        prec->SetElasticityOptions(vfes);
    }

    //set the parameters
    ns->SetSolver(*ls);
    ns->SetOperator(*nf);
    ns->SetPrintLevel(print_level);
    ns->SetRelTol(rel_tol);
    ns->SetAbsTol(abs_tol);
    ns->SetMaxIter(max_iter);

    ls->SetPrintLevel(print_level);
    ls->SetAbsTol(linear_atol);
    ls->SetRelTol(linear_rtol);
    ls->SetMaxIter(linear_iter);
    ls->SetPreconditioner(*prec);

    prec->SetPrintLevel(print_level);

    //solve the problem
    Vector b;
    ns->Mult(b,sol);
}


double CFNLElasticityIntegrator::GetElementEnergy(const FiniteElement &el,
                                                  ElementTransformation &Tr,
                                                  const Vector &elfun)
{
    if((*marks)[Tr.ElementNo]==ElementMarker::SBElementType::OUTSIDE){
        return 0.0;
    }

    const int dof=el.GetDof();
    const int dim=el.GetDim();
    {
        const int spaceDim=Tr.GetSpaceDim();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("NLElasticityIntegrator::GetElementEnergy is not define on manifold meshes.");
        }
    }

    if(elco==nullptr){	return 0.0;}

    //gradients
    mfem::DenseMatrix bsu; bsu.SetSize(dof,dim);
    Vector uu(elfun.GetData()+0*dof,dof);
    Vector vv(elfun.GetData()+1*dof,dof);

    Vector ww;
    if(dim==3){
        ww.SetDataAndSize(elfun.GetData()+2*dof,dof);
    }

    Vector gradu(9); gradu=0.0;
    Vector sh;

    const IntegrationRule *ir = nullptr;
    AlgoimIntegrationRule* air =nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el);//this might be too big

    if((*marks)[Tr.ElementNo]==ElementMarker::SBElementType::INSIDE)
    {
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }else
    {
        //cut element
        Array<int> vdofs;
        const FiniteElement* le=lsf->FESpace()->GetFE(Tr.ElementNo);
        DofTransformation *doftrans=lsf->FESpace()->GetElementVDofs(Tr.ElementNo,vdofs);
        Vector vlsf; //vector for the level-set-function
        lsf->GetSubVector(vdofs,vlsf);
        //construct algoim integration rule
        air=new AlgoimIntegrationRule(order,*le,Tr,vlsf);
        ir=air->GetVolumeIntegrationRule();
    }

    double w;
    double energy=0.0;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcPhysDShape(Tr,bsu);

        sh.SetDataAndSize(gradu.GetData()+3*0,dim);
        bsu.MultTranspose(uu,sh);
        sh.SetDataAndSize(gradu.GetData()+3*1,dim);
        bsu.MultTranspose(vv,sh);
        if(dim==3)
        {
            sh.SetDataAndSize(gradu.GetData()+3*2,dim);
            bsu.MultTranspose(ww,sh);
        }

        // Calcualte the residual at the integration point
        energy=energy+ w*(elco->EvalEnergy(gradu,Tr,ip));
    }

    delete air;//delete algoim integration rule

    return energy;
}

void  CFNLElasticityIntegrator::AssembleElementVector(const FiniteElement &el,
                                                      ElementTransformation &Tr,
                                                      const Vector &elfun,
                                                      Vector &elvect)
{
    const int dof=el.GetDof();
    const int dim=el.GetDim();
    elvect.SetSize(dof*dim); elvect=0.0;

    if((*marks)[Tr.ElementNo]==ElementMarker::SBElementType::OUTSIDE){
        return;
    }

    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("NLElasticityIntegrator::AssembleElementVector is not defined on manifold meshes.");
        }
    }

    if(elco==nullptr){	return;}

    //gradients
    mfem::DenseMatrix bsu; bsu.SetSize(dof,dim);
    Vector uu(elfun.GetData()+0*dof,dof);
    Vector vv(elfun.GetData()+1*dof,dof);

    Vector ru(elvect.GetData()+0*dof,dof);
    Vector rv(elvect.GetData()+1*dof,dof);

    Vector ww;
    Vector rw;
    if(dim==3){
        ww.SetDataAndSize(elfun.GetData()+2*dof,dof);
        rw.SetDataAndSize(elvect.GetData()+2*dof,dof);
    }else{
        ww.SetSize(dof); ww=0.0;
    }

    Vector gradu(9); gradu=0.0;
    Vector rr(9);
    Vector sh;

    //correction factor for cut elements
    double corr_factor=1.0;

    const IntegrationRule *ir = nullptr;
    AlgoimIntegrationRule* air =nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el);

    Vector vlsf;//vector for the level-set-function
    ir=&IntRules.Get(Tr.GetGeometryType(),order);
    if((*marks)[Tr.ElementNo]==ElementMarker::SBElementType::CUT)
    {
        corr_factor=stiffness_ratio;
    }

    double w;
    Vector ff(dim);
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcPhysDShape(Tr,bsu);

        sh.SetDataAndSize(gradu.GetData()+3*0,dim);
        bsu.MultTranspose(uu,sh);
        sh.SetDataAndSize(gradu.GetData()+3*1,dim);
        bsu.MultTranspose(vv,sh);
        if(dim==3)
        {
            sh.SetDataAndSize(gradu.GetData()+3*2,dim);
            bsu.MultTranspose(ww,sh);
        }

        // Calcualte the residual at the integration point
        elco->EvalResidual(rr,gradu,Tr,ip);
        // correct the residual
        rr*=corr_factor;

        sh.SetDataAndSize(rr.GetData()+3*0,dim);
        bsu.AddMult_a(w,sh,ru);
        sh.SetDataAndSize(rr.GetData()+3*1,dim);
        bsu.AddMult_a(w,sh,rv);
        if(dim==3)
        {
            sh.SetDataAndSize(rr.GetData()+3*2,dim);
            bsu.AddMult_a(w,sh,rw);
        }


        //add forcing term only for internal elements
        if((*marks)[Tr.ElementNo]!=ElementMarker::SBElementType::CUT){
        if(forc!=nullptr){
            forc->Eval(ff,Tr,ip);
            sh.SetDataAndSize(bsu.GetData(),dof);
            el.CalcPhysShape(Tr,sh);
            ru.Add(-w*ff[0],sh);
            rv.Add(-w*ff[1],sh);
            if(dim==3){rw.Add(-w*ff[2],sh);}
        }}
    }

    //surface loading
    if((*marks)[Tr.ElementNo]==ElementMarker::SBElementType::CUT){

    //cut element
    Array<int> vdofs;
    const FiniteElement* le=lsf->FESpace()->GetFE(Tr.ElementNo);
    DofTransformation *doftrans=lsf->FESpace()->GetElementVDofs(Tr.ElementNo,vdofs);
    lsf->GetSubVector(vdofs,vlsf);
    //construct algoim integration rule
    air=new AlgoimIntegrationRule(order,*le,Tr,vlsf);
    ir=air->GetVolumeIntegrationRule();

    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcPhysDShape(Tr,bsu);

        sh.SetDataAndSize(gradu.GetData()+3*0,dim);
        bsu.MultTranspose(uu,sh);
        sh.SetDataAndSize(gradu.GetData()+3*1,dim);
        bsu.MultTranspose(vv,sh);
        if(dim==3)
        {
            sh.SetDataAndSize(gradu.GetData()+3*2,dim);
            bsu.MultTranspose(ww,sh);
        }

        // Calcualte the residual at the integration point
        elco->EvalResidual(rr,gradu,Tr,ip);
        // correct the residual
        rr*=(1.0-corr_factor);

        sh.SetDataAndSize(rr.GetData()+3*0,dim);
        bsu.AddMult_a(w,sh,ru);
        sh.SetDataAndSize(rr.GetData()+3*1,dim);
        bsu.AddMult_a(w,sh,rv);
        if(dim==3)
        {
            sh.SetDataAndSize(rr.GetData()+3*2,dim);
            bsu.AddMult_a(w,sh,rw);
        }

        // add vol forces
        if(forc!=nullptr){
            forc->Eval(ff,Tr,ip);
            sh.SetDataAndSize(bsu.GetData(),dof);
            el.CalcPhysShape(Tr,sh);
            ru.Add(-w*ff[0],sh);
            rv.Add(-w*ff[1],sh);
            if(dim==3){rw.Add(-w*ff[2],sh);}
        }
    }




    if(surfl!=nullptr){
        ir=air->GetSurfaceIntegrationRule();
        const FiniteElement* le=lsf->FESpace()->GetFE(Tr.ElementNo);
        DenseMatrix bmat; //gradients of the shape functions in isoparametric space
        DenseMatrix pmat; //gradients of the shape functions in physical space
        Vector inormal; //normal to the level set in isoparametric space
        Vector tnormal; //normal to the level set in physical space
        bmat.SetSize(le->GetDof(),le->GetDim());
        pmat.SetSize(le->GetDof(),le->GetDim());
        inormal.SetSize(le->GetDim());
        tnormal.SetSize(le->GetDim());
        for (int j = 0; j < ir->GetNPoints(); j++)
        {
           const IntegrationPoint &ip = ir->IntPoint(j);
           Tr.SetIntPoint(&ip);
           le->CalcDShape(ip,bmat);
           Mult(bmat, Tr.AdjugateJacobian(), pmat);
           //compute the normal to the LS in isoparametric space
           bmat.MultTranspose(vlsf,inormal);
           //compute the normal to the LS in physical space
           pmat.MultTranspose(vlsf,tnormal);
           w = ip.weight * tnormal.Norml2() / inormal.Norml2();

           surfl->Eval(ff,Tr,ip);
           sh.SetDataAndSize(bsu.GetData(),dof);
           el.CalcPhysShape(Tr,sh);
           ru.Add(-w*ff[0],sh);
           rv.Add(-w*ff[1],sh);
           if(dim==3){rw.Add(-w*ff[2],sh);}
        }
    }}

    delete air;
}

void CFNLElasticityIntegrator::AssembleElementGrad(const FiniteElement &el,
                                                   ElementTransformation &Tr,
                                                   const Vector &elfun,
                                                   DenseMatrix &elmat)
{
    const int dof=el.GetDof();
    const int dim=el.GetDim();
    elmat.SetSize(dof*dim); elmat=0.0;
    if((*marks)[Tr.ElementNo]==ElementMarker::SBElementType::OUTSIDE)
    {
        return;
    }

    {
        const int spaceDim=Tr.GetSpaceDim();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("NLElasticityIntegrator::AssembleElementGrad is not defined on manifold meshes.");
        }
    }

    if(elco==nullptr){	return;}

    //scale factor
    double corr_factor=1.0;
    if((*marks)[Tr.ElementNo]==ElementMarker::SBElementType::CUT)
    {
        corr_factor=stiffness_ratio; //stabilize the stiffness matrix
    }

    //gradients
    mfem::DenseMatrix bsu; bsu.SetSize(dof,dim);
    Vector uu(elfun.GetData()+0*dof,dof);
    Vector vv(elfun.GetData()+1*dof,dof);
    Vector ww;
    if(dim==3)
    {
        ww.SetDataAndSize(elfun.GetData()+2*dof,dof);
    }

    Vector gradu(9); gradu=0.0;
    Vector sh;
    // state matrix at integration point
    DenseMatrix mm; mm.SetSize(9);
    DenseMatrix mh; mh.SetSize(dim);
    DenseMatrix th; th.SetSize(dof,dim);
    DenseMatrix rh; rh.SetSize(dof);

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el);
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    double w;

    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcPhysDShape(Tr,bsu);

        sh.SetDataAndSize(gradu.GetData()+3*0,dim);
        bsu.MultTranspose(uu,sh);
        sh.SetDataAndSize(gradu.GetData()+3*1,dim);
        bsu.MultTranspose(vv,sh);
        if(dim==3)
        {
            sh.SetDataAndSize(gradu.GetData()+3*2,dim);
            bsu.MultTranspose(ww,sh);
        }

        // calculate the tangent matrix
        elco->EvalTangent(mm,gradu,Tr,ip);

        for(int ii=0;ii<dim;ii++){
        for(int jj=0;jj<dim;jj++){
            mh.CopyMN(mm,dim,dim,ii*3,jj*3);
            mh.Transpose();
            MultABt(bsu,mh,th);
            MultABt(th,bsu,rh);
            elmat.AddMatrix(w*corr_factor,rh,ii*dof,jj*dof);
        }}
    }


    if((*marks)[Tr.ElementNo]!=ElementMarker::SBElementType::CUT)
    {return;}

    //evaluate algoim integration rule
    //cut element
    Vector vlsf;//vector for the level-set-function
    Array<int> vdofs;
    const FiniteElement* le=lsf->FESpace()->GetFE(Tr.ElementNo);
    DofTransformation *doftrans=lsf->FESpace()->GetElementVDofs(Tr.ElementNo,vdofs);
    lsf->GetSubVector(vdofs,vlsf);
    AlgoimIntegrationRule* air=new AlgoimIntegrationRule(order,*le,Tr,vlsf);
    ir=air->GetVolumeIntegrationRule();

    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcPhysDShape(Tr,bsu);

        sh.SetDataAndSize(gradu.GetData()+3*0,dim);
        bsu.MultTranspose(uu,sh);
        sh.SetDataAndSize(gradu.GetData()+3*1,dim);
        bsu.MultTranspose(vv,sh);
        if(dim==3)
        {
            sh.SetDataAndSize(gradu.GetData()+3*2,dim);
            bsu.MultTranspose(ww,sh);
        }

        // calculate the tangent matrix
        elco->EvalTangent(mm,gradu,Tr,ip);

        for(int ii=0;ii<dim;ii++){
        for(int jj=0;jj<dim;jj++){
            mh.CopyMN(mm,dim,dim,ii*3,jj*3);
            mh.Transpose();
            MultABt(bsu,mh,th);
            MultABt(th,bsu,rh);
            elmat.AddMatrix(w*(1.0-stiffness_ratio),rh,ii*dof,jj*dof);
        }}
    }
    delete air;

}


#endif



}

