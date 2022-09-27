#include "elasticity.hpp"
#include "marking.hpp"

using namespace mfem;

void DispFunctionLL(const mfem::Vector &xx, mfem::Vector &uu)
{
    double a=3.0;
    double b=2.0;
    double c=3.0;
    double d=2.0;

    if(xx.Size()==2)
    {
        uu[0] = sin(a*xx[0])+cos(b*xx[1]);
        uu[1] = cos(c*xx[0]+d*xx[1]);
    }else{//size==3
        uu[0] = sin(a*xx[0])+cos(b*xx[1]);
        uu[1] = sin(a*xx[1])+cos(b*xx[2]);
        uu[2] = sin(c*xx[2])+cos(d*xx[0]);
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

void ElasticitySolver::AddDispBC(int id, mfem::VectorCoefficient& val)
{
    bcca[id]=&val;
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

        //set vector coefficients
        for(auto it=bcca.begin();it!=bcca.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
            fdisp.ProjectBdrCoefficient(*(it->second), ess_bdr);
            //copy tdofs from velocity grid function
            fdisp.GetTrueDofs(rhs); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdof_list.Size();ii++)
            {
                sol[ess_tdof_list[ii]]=rhs[ess_tdof_list[ii]];
            }
            ess_tdofv.Append(ess_tdof_list);
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
    if(T.GetSpaceDim()==3){ //3D problem
        D.SetSize(6);
        elast::IsotropicStiffnessTensor3D(EE,nnu,D);
    }else{
        D.SetSize(3); //2D problem
        elast::IsotropicStiffnessTensor2D(EE,nnu,D);
    }
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
    int order= 4 * el.GetOrder() + Tr.OrderGrad(&el);
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

double SBM2ELIntegrator::GetElementEnergy(const FiniteElement &el,
                                                ElementTransformation &Tr,
                                                const Vector &elfun)
{
    return 0.0;
}

void SBM2ELIntegrator::AssembleElementVector(const FiniteElement &el,
                                              ElementTransformation &Tr,
                                              const Vector &elfun, Vector &elvect)
{
    elvect.SetSize(elfun.Size());
    elvect=0.0;

    if((*elem_attributes)[Tr.ElementNo]==mfem::ElementMarker::SBElementType::INSIDE){
        elint->AssembleElementVector(el,Tr,elfun,elvect);
    }

    //If the element is not inside there isn't anything to integrate - the contribution is zero
    if((*elem_attributes)[Tr.ElementNo]!=mfem::ElementMarker::SBElementType::INSIDE){return;}
    //get dimensionality of the problem
    const int dim=Tr.GetSpaceDim();

    std::vector<int> bfaces;

    if(dim==3){
        const mfem::Table& elem_face=pmesh->ElementToFaceTable();
        int num_faces=elem_face.RowSize(Tr.ElementNo);
        const int* faces=elem_face.GetRow(Tr.ElementNo);
        for(int facei=0;facei<num_faces;facei++)
        {
            if((*face_attributes)[faces[facei]]==ElementMarker::SBFaceType::SURROGATE)
            {
                bfaces.push_back(faces[facei]);
            }
        }
    }else{
        //dim==2
        const mfem::Table& elem_edge=pmesh->ElementToEdgeTable();
        int num_edges=elem_edge.RowSize(Tr.ElementNo);
        const int* faces=elem_edge.GetRow(Tr.ElementNo);
        for(int facei=0;facei<num_edges;facei++)
        {
            if((*face_attributes)[faces[facei]]==ElementMarker::SBFaceType::SURROGATE)
            {
                bfaces.push_back(faces[facei]);
            }
        }
    }

    if(bfaces.size()==0){return;} //no crossed edges/faces return

    mfem::Vector nnor; nnor.SetSize(dim); //surrogate normal vector
    mfem::Vector nnor_n; nnor_n.SetSize(dim); //surrogate normal vector normalized
    mfem::Vector nvec; nvec.SetSize(dim); //true normal vector
    const int dof=el.GetDof();
    const IntegrationRule *ir;

    //computes the gradient at the nodal points of the element
    DenseMatrix grad_phys;
    el.ProjectGrad(el,Tr,grad_phys);

    DenseMatrix shift_op; shift_op.SetSize(dof);
    DenseMatrix shift_ts; shift_ts.SetSize(dof);
    Vector shape; shape.SetSize(dof);
    Vector shape_sh; shape_sh.SetSize(dof);
    Vector shape_ts; shape_ts.SetSize(dof);
    DenseMatrix bdrstress_op;//boundary stress operator
    bdrstress_op.SetSize(dim,dim*dof); bdrstress_op=0.0;
    DenseMatrix disp_op; disp_op.SetSize(dim,dim*dof);
    DenseMatrix B; B.SetSize(dof,dim);
    DenseMatrix D; //elasticity matrix
    Vector strain_v;
    if(dim==3){D.SetSize(6); strain_v.SetSize(6); strain_v=0.0;}
    else{D.SetSize(3);strain_v.SetSize(3); strain_v=0.0;}

    Vector dispx(elfun.GetData()+0*dof,dof);
    Vector dispy(elfun.GetData()+1*dof,dof);
    Vector dispz;
    if(dim==3){
        dispz.SetDataAndSize(elfun.GetData()+2*dof,dof);
    }

    mfem::Vector bdisp; bdisp.SetSize(dim); //boundary displacements
    mfem::Vector tract; tract.SetSize(dim); //tractions
    mfem::Vector ddist; ddist.SetSize(dim); //[LS unit normal]*distance

    mfem::Vector rvec(dim*dof); //tmp residual vector

    int elmNo=Tr.ElementNo;
    ElementTransformation* ltr;

    // we need the element number to figure out if we
    // will be working with element 1 or 2 in the integration

    // all element faces are stored in bfaces
    for(unsigned int i=0;i<bfaces.size();i++)
    {
        mfem::FaceElementTransformations *fctr;
        // the original element transformation is invalidated
        fctr=pmesh->GetInteriorFaceTransformations(bfaces[i]);
        if(fctr==nullptr){
            fctr=pmesh->GetFaceElementTransformations(bfaces[i]);
        }
        int order = 4*el.GetOrder();
        ir = &IntRules.Get(fctr->GetGeometryType(), order);

        // the original element transformation is invalidated
        // after the call to GetInteriorFaceTransformations
        if(fctr->Elem1No==elmNo){
            ltr=fctr->Elem1;
        }else{
            ltr=fctr->Elem2;
        }

        double facei=0.0;

        // integrate over the face
        for (int p = 0; p < ir->GetNPoints(); p++)
        {
           const IntegrationPoint &ip = ir->IntPoint(p);
           fctr->SetAllIntPoints(&ip); // sets the integration for the element as well
           //calculate the surrogate normal
           // Note: this normal accounts for the weight of the surface transformation
           // Jacobian i.e. nnor = nhat*det(J)
           CalcOrtho(fctr->Jacobian(),nnor);
           // Make sure the normal vector is pointing outside the domain.
           if(fctr->Elem2No==elmNo){nnor*=-1.0;}//reverse the normal
           //normalize nnor
           {
               double nn0=nnor.Norml2();
               if(nn0>std::numeric_limits<double>::epsilon()){nnor_n=nnor; nnor_n/=nn0;}
               else{nnor_n=0.0;}
           }

           //calculate the distance
           double d=dist_coeff->Eval(*fctr,ip); d=0.0;
           //calculate the gradient of the SDF
           dist_grad->Eval(nvec,*fctr,ip);
           //normalize nvec - the true normal
           double nn=nvec.Norml2();
           if(nn>std::numeric_limits<double>::epsilon()){nvec/=nn;}
           else{nvec=0.0;}

           //compute boundary displacements
           {
               if(bdr_disp==nullptr){
                   bdisp=0.0;
               }else{
                   ddist=nvec; ddist*=d;
                   bdr_disp->Eval(bdisp,*fctr,ip,ddist);
               }
           }
           //std::cout<<"bdisp :"; bdisp.Print(std::cout);
           /*
           std::cout<<"d="<<d<<" tnor="; nvec.Print(std::cout);
           std::cout<<"nnor :"; nnor.Print(std::cout);
           std::cout<<"dot :"<<nnor*nvec<<std::endl;
           */

           //evaluate the shift operators
           EvalShiftOperator(grad_phys,d,nvec,shift_order,shift_op,shift_ts);

           //evaluate the shape functions
           el.CalcPhysShape(*ltr, shape);
           el.CalcPhysDShape(*ltr, B);
           shift_op.MultTranspose(shape,shape_sh); //now we have the shifted shape functions
           shift_ts.MultTranspose(shape,shape_ts); //now we have the shifted test functions

           double helm=std::pow(ltr->Weight(),1./dim); //the element size
           //evaluate the displacement constr i.e. shifted_disp-prescribed_disp
           bdisp(0)=-bdisp(0)+shape_sh*dispx;
           bdisp(1)=-bdisp(1)+shape_sh*dispy;
           if(dim==3){
               bdisp(2)=-bdisp(2)+shape_sh*dispz;
           }

           //std::cout<<"bdisp :"; bdisp.Print(std::cout);

           //compute the boundary stress operator bdrstress_op
           elco->EvalStiffness(D, strain_v, *fctr,ip);
           BndrStressOperator(dim,B,D,nnor_n,bdrstress_op);
           //compute the traction to the surogate bdr
           bdrstress_op.Mult(elfun,tract);


           for(int dd=0;dd<dim;dd++){
           for(int ii=0;ii<dof;ii++){
               elvect(dd*dof+ii)=elvect(dd*dof+ii)+shape(ii)*tract(dd)*ip.weight*nnor.Norml2();
           }}


           bdrstress_op.MultTranspose(bdisp,rvec);
           for(int ii=0;ii<dim*dof;ii++){
               elvect(ii)=elvect(ii)-rvec(ii)*ip.weight*nnor.Norml2();
           }



           for(int dd=0;dd<dim;dd++){
           for(int ii=0;ii<dof;ii++){
               elvect(dd*dof+ii)=elvect(dd*dof+ii)+alpha*bdisp(dd)*shape_ts(ii)*ip.weight*nnor.Norml2()/helm;
               //elvect(dd*dof+ii)=elvect(dd*dof+ii)+alpha*bdisp(dd)*shape_sh(ii)*ip.weight*nnor.Norml2()/helm;
           }}

           facei=facei+ip.weight*nnor.Norml2();

        }

        //std::cout<<"length/area="<<facei<<std::endl;



    }

}

/*
void SBM2ELIntegrator::AssembleElementVectorI(const FiniteElement &el,
                                                   ElementTransformation &Tr,
                                                   const Vector &elfun, Vector &elvect)
{
    elvect.SetSize(elfun.Size());
    elvect=0.0;

    if((*elem_attributes)[Tr.ElementNo]==mfem::ElementMarker::SBElementType::INSIDE){
        elint->AssembleElementVector(el,Tr,elfun,elvect);
    }

    //If the element is not inside there isn't anything to integrate - the contribution is zero
    if((*elem_attributes)[Tr.ElementNo]!=mfem::ElementMarker::SBElementType::INSIDE){return;}
    //get dimensionality of the problem
    const int dim=Tr.GetSpaceDim();

    std::vector<int> bfaces;

    if(dim==3){
        const mfem::Table& elem_face=pmesh->ElementToFaceTable();
        int num_faces=elem_face.RowSize(Tr.ElementNo);
        const int* faces=elem_face.GetRow(Tr.ElementNo);
        for(int facei=0;facei<num_faces;facei++)
        {
            if((*face_attributes)[faces[facei]]==ElementMarker::SBFaceType::SURROGATE)
            {
                bfaces.push_back(faces[facei]);
            }
        }
    }else{
        //dim==2
        const mfem::Table& elem_edge=pmesh->ElementToEdgeTable();
        int num_edges=elem_edge.RowSize(Tr.ElementNo);
        const int* faces=elem_edge.GetRow(Tr.ElementNo);
        for(int facei=0;facei<num_edges;facei++)
        {
            if((*face_attributes)[faces[facei]]==ElementMarker::SBFaceType::SURROGATE)
            {
                bfaces.push_back(faces[facei]);
            }
        }
    }

    if(bfaces.size()==0){return;} //no crossed edges/faces return

    mfem::Vector nnor; nnor.SetSize(dim); //surrogate normal vector
    mfem::Vector nnor_n; nnor_n.SetSize(dim); //surrogate normal vector normalized
    mfem::Vector nvec; nvec.SetSize(dim); //true normal vector
    const int dof=el.GetDof();
    const IntegrationRule *ir;

    //computes the gradient at the nodal points of the element
    DenseMatrix grad_phys;
    el.ProjectGrad(el,Tr,grad_phys);

    DenseMatrix shift_op; shift_op.SetSize(dof);
    DenseMatrix shift_ts; shift_ts.SetSize(dof);
    Vector shape; shape.SetSize(dof);
    Vector shape_sh; shape_sh.SetSize(dof);
    Vector shape_ts; shape_ts.SetSize(dof);
    DenseMatrix bdrstress_op;//boundary stress operator
    bdrstress_op.SetSize(dim,dim*dof); bdrstress_op=0.0;
    DenseMatrix disp_op; disp_op.SetSize(dim,dim*dof);
    DenseMatrix B; B.SetSize(dof,dim);
    DenseMatrix D; //elasticity matrix
    Vector strain_v;
    if(dim==3){D.SetSize(6); strain_v.SetSize(6); strain_v=0.0;}
    else{D.SetSize(3);strain_v.SetSize(3); strain_v=0.0;}

    Vector dispx(elfun.GetData()+0*dof,dof);
    Vector dispy(elfun.GetData()+1*dof,dof);
    Vector dispz;
    if(dim==3){
        dispz.SetDataAndSize(elfun.GetData()+2*dof,dof);
    }

    mfem::Vector bdisp; bdisp.SetSize(dim); //boundary displacements
    mfem::Vector tract; tract.SetSize(dim); //tractions
    mfem::Vector ddist; ddist.SetSize(dim); //[LS unit normal]*distance

    mfem::Vector rvec(dim*dof); //tmp residual vector


    ElementTransformation* ltr;
    int elmNo=Tr.ElementNo;
    // we need the element number to figure out if we
    // will be working with element 1 or 2 in the integration

    // all element faces are stored in bfaces
    for(unsigned int i=0;i<bfaces.size();i++)
    {
        mfem::FaceElementTransformations *fctr;
        // the original element transformation is invalidated
        fctr=pmesh->GetInteriorFaceTransformations(bfaces[i]);
        if(fctr==nullptr){
            fctr=pmesh->GetFaceElementTransformations(bfaces[i]);
        }
        int order = 4*el.GetOrder();
        ir = &IntRules.Get(fctr->GetGeometryType(), order);

        // the original element transformation is invalidated
        // after the call to GetInteriorFaceTransformations

        if(fctr->Elem1No==elmNo){
            ltr=fctr->Elem1;
        }else{
            ltr=fctr->Elem2;
        }

        double facei=0.0;

        // integrate over the face
        for (int p = 0; p < ir->GetNPoints(); p++)
        {
           const IntegrationPoint &ip = ir->IntPoint(p);
           fctr->SetAllIntPoints(&ip); // sets the integration for the element as well
           //calculate the surrogate normal
           // Note: this normal accounts for the weight of the surface transformation
           // Jacobian i.e. nnor = nhat*det(J)
           CalcOrtho(fctr->Jacobian(),nnor);
           // Make sure the normal vector is pointing outside the domain.
           if(fctr->Elem2No==elmNo){nnor*=-1.0;}//reverse the normal
           //normalize nnor
           {
               double nn0=nnor.Norml2();
               if(nn0>std::numeric_limits<double>::epsilon()){nnor_n=nnor; nnor_n/=nn0;}
               else{nnor_n=0.0;}
           }
           //calculate the distance
           double d=dist_coeff->Eval(*fctr,ip);
           //calculate the gradient of the SDF
           dist_grad->Eval(nvec,*fctr,ip);
           //normalize nvec - the true normal
           double nn=nvec.Norml2();
           if(nn>std::numeric_limits<double>::epsilon()){nvec/=nn;}
           else{nvec=0.0;}

           //compute boundary displacements
           {
               if(bdr_disp==nullptr){
                   bdisp=0.0;
               }else{
                   ddist=nvec; ddist*=d;
                   bdr_disp->Eval(bdisp,*fctr,ip,ddist);
                   //bdr_disp->Eval(bdisp,*fctr,ip);
               }
           }



           //evaluate the shift operators
           EvalShiftOperator(grad_phys,d,nvec,shift_order,shift_op,shift_ts);
           //evaluate the shape functions
           el.CalcPhysShape(*ltr, shape);
           el.CalcPhysDShape(*ltr, B);
           shift_op.MultTranspose(shape,shape_sh); //now we have the shifted shape functions
           shift_ts.MultTranspose(shape,shape_ts); //now we have the shifted test functions

           //const DenseMatrix& J= ltr->Jacobian();
           //J.Print(std::cout);

           double helm=std::pow(ltr->Weight(),1./dim); //the element size
           //evaluate the displacement constr i.e. shifted_disp-prescribed_disp
           bdisp(0)=-bdisp(0)+shape_sh*dispx;
           bdisp(1)=-bdisp(1)+shape_sh*dispy;
           if(dim==3){
               bdisp(2)=-bdisp(2)+shape_sh*dispz;
           }
           //compute the boundary stress operator bdrstress_op
           elco->EvalStiffness(D, strain_v, *fctr,ip);
           BndrStressOperator(dim,B,D,nnor_n,bdrstress_op);
           //compute the traction to the surogate bdr
           bdrstress_op.Mult(elfun,tract);


           for(int dd=0;dd<dim;dd++){
           for(int ii=0;ii<dof;ii++){
               elvect(dd*dof+ii)=elvect(dd*dof+ii)-shape(ii)*tract(dd)*ip.weight*nnor.Norml2();
           }}


           bdrstress_op.MultTranspose(bdisp,rvec);
           for(int ii=0;ii<dim*dof;ii++){
               elvect(ii)=elvect(ii)-rvec(ii)*ip.weight*nnor.Norml2();
           }

           for(int dd=0;dd<dim;dd++){
           for(int ii=0;ii<dof;ii++){
               elvect(dd*dof+ii)=elvect(dd*dof+ii)+alpha*bdisp(dd)*shape_ts(ii)*ip.weight*nnor.Norml2()/helm;
               //elvect(dd*dof+ii)=elvect(dd*dof+ii)+alpha*bdisp(dd)*shape_sh(ii)*ip.weight*nnor.Norml2()/helm;
           }}



           facei=facei+ip.weight*nnor.Norml2();

        }

        //std::cout<<"length/area="<<facei<<std::endl;

    }

}
*/

void SBM2ELIntegrator::AssembleElementGrad(const FiniteElement &el,
                                           ElementTransformation &Tr,
                                           const Vector &elfun,
                                           DenseMatrix &elmat)
{
    elmat.SetSize(elfun.Size());
    elmat=0.0;
    if((*elem_attributes)[Tr.ElementNo]==mfem::ElementMarker::SBElementType::INSIDE){
        elint->AssembleElementGrad(el,Tr,elfun,elmat);
    }

    //If the element is not inside there isn't anything to integrate - the contribution is zero
    if((*elem_attributes)[Tr.ElementNo]!=mfem::ElementMarker::SBElementType::INSIDE){return;}

    //get dimensionality of the problem
    const int dim=Tr.GetSpaceDim();
    std::vector<int> bfaces;
    if(dim==3){
        const mfem::Table& elem_face=pmesh->ElementToFaceTable();
        int num_faces=elem_face.RowSize(Tr.ElementNo);
        const int* faces=elem_face.GetRow(Tr.ElementNo);
        for(int facei=0;facei<num_faces;facei++)
        {
            if((*face_attributes)[faces[facei]]==mfem::ElementMarker::SBFaceType::SURROGATE)
            {
                bfaces.push_back(faces[facei]);
            }
        }
    }else{
        //dim==2
        const mfem::Table& elem_edge=pmesh->ElementToEdgeTable();
        int num_edges=elem_edge.RowSize(Tr.ElementNo);
        const int* faces=elem_edge.GetRow(Tr.ElementNo);
        for(int facei=0;facei<num_edges;facei++)
        {
            if((*face_attributes)[faces[facei]]==mfem::ElementMarker::SBFaceType::SURROGATE)
            {
                bfaces.push_back(faces[facei]);
            }
        }
    }

    if(bfaces.size()==0){return;}


    mfem::Vector nnor; nnor.SetSize(dim); //surrogate normal vector
    mfem::Vector nnor_n; nnor_n.SetSize(dim); //surrogate normal vector normalized
    mfem::Vector nvec; nvec.SetSize(dim); //true normal vector
    const int dof=el.GetDof();
    const IntegrationRule *ir;

    //computes the gradient at the nodal points of the element
    DenseMatrix grad_phys;
    el.ProjectGrad(el,Tr,grad_phys);

    DenseMatrix shift_op; shift_op.SetSize(dof);
    DenseMatrix shift_ts; shift_ts.SetSize(dof);
    Vector shape; shape.SetSize(dof);
    Vector shape_sh; shape_sh.SetSize(dof);
    Vector shape_ts; shape_ts.SetSize(dof);
    DenseMatrix mtmp; mtmp.SetSize(dof);
    DenseMatrix btmp; btmp.SetSize(dim*dof);
    DenseMatrix bdrstress_op;//boundary stress operator
    bdrstress_op.SetSize(dim,dim*dof); bdrstress_op=0.0;
    DenseMatrix disp_op; disp_op.SetSize(dim,dim*dof);
    DenseMatrix B; B.SetSize(dof,dim);
    DenseMatrix D; //elasticity matrix
    Vector strain_v;
    if(dim==3){D.SetSize(6); strain_v.SetSize(6); strain_v=0.0;}
    else{D.SetSize(3);strain_v.SetSize(3); strain_v=0.0;}

    ElementTransformation *ltr;
    int elmNo=Tr.ElementNo;
    // we need the element number to figure out if we
    // will be working with element 1 or 2 in the integration


    // all element faces are stored in bfaces
    for(unsigned int i=0;i<bfaces.size();i++)
    {
        mfem::FaceElementTransformations *fctr;
        // the original element transformation is invalidated
        fctr=pmesh->GetInteriorFaceTransformations(bfaces[i]);
        if(fctr==nullptr){
            fctr=pmesh->GetFaceElementTransformations(bfaces[i]);
        }
        int order = 4*el.GetOrder();
        ir = &IntRules.Get(fctr->GetGeometryType(), order);

        // the original element transformation is invalidated
        // after the call to GetInteriorFaceTransformations
        if(fctr->Elem1No==elmNo){
            ltr=(fctr->Elem1);
        }else{
            ltr=(fctr->Elem2);
        }

        // integrate over the face
        for (int p = 0; p < ir->GetNPoints(); p++)
        {
           const IntegrationPoint &ip = ir->IntPoint(p);
           fctr->SetAllIntPoints(&ip); // sets the integration for the element as well
           //calculate the surrogate normal
           // Note: this normal accounts for the weight of the surface transformation
           // Jacobian i.e. nnor = nhat*det(J)
           CalcOrtho(fctr->Jacobian(),nnor);
           // Make sure the normal vector is pointing outside the domain.
           if(fctr->Elem2No==elmNo){nnor*=-1.0;}//reverse the normal
           //normalize nnor
           {
               double nn0=nnor.Norml2();
               if(nn0>std::numeric_limits<double>::epsilon()){nnor_n=nnor; nnor_n/=nn0;}
               else{nnor_n=0.0;}
           }
           //calculate the distance
           double d=dist_coeff->Eval(*fctr,ip); d=0.0;
           //calculate the gradient of the SDF
           dist_grad->Eval(nvec,*fctr,ip);
           //normalize nvec - the true normal
           double nn=nvec.Norml2();
           if(nn>std::numeric_limits<double>::epsilon()){nvec/=nn;}
           else{nvec=0.0;}

           //evaluate the shift operators
           EvalShiftOperator(grad_phys,d,nvec,shift_order,shift_op,shift_ts);
           //evaluate the shape functions
           el.CalcPhysShape(*ltr, shape);
           el.CalcPhysDShape(*ltr, B);
           shift_op.MultTranspose(shape,shape_sh); //now we have the shifted shape functions
           shift_ts.MultTranspose(shape,shape_ts); //now we have the shifted test functions

           //add the penalization term
           MultVWt(shape_ts,shape_sh,mtmp);
           //MultVWt(shape_sh,shape_sh,mtmp);
           double helm=std::pow(ltr->Weight(),1./dim); //the element size
           //add mtmp to elmat
           for(int ii=0;ii<dof;ii++){
           for(int jj=0;jj<dof;jj++){
           for(int di=0;di<dim;di++){
               elmat(ii+di*dof,jj+di*dof)=elmat(ii+di*dof,jj+di*dof)+mtmp(ii,jj)*alpha*ip.weight*nnor.Norml2()/helm;
           }}}

           //compute \sigma_n where n is the surrogate normal nnor
           //compute the boundary stress operator bdrstress_op
           elco->EvalStiffness(D, strain_v, *fctr,ip);
           BndrStressOperator(dim,B,D,nnor_n,bdrstress_op);

           //compute shifted disp_op
           EvalDispOp(dim,shape_sh,disp_op);
           MultAtB(bdrstress_op,disp_op,btmp);
           for(int ii=0;ii<dim*dof;ii++){
           for(int jj=0;jj<dim*dof;jj++){
               elmat(ii,jj)=elmat(ii,jj)-btmp(ii,jj)*ip.weight*nnor.Norml2();
           }}

           //compute true disp_op
           EvalDispOp(dim,shape,disp_op);
           MultAtB(disp_op,bdrstress_op,btmp);
           for(int ii=0;ii<dim*dof;ii++){
           for(int jj=0;jj<dim*dof;jj++){
               elmat(ii,jj)=elmat(ii,jj)-btmp(ii,jj)*ip.weight*nnor.Norml2();
           }}


        }
    }

}

void SBM2ELIntegrator::AssembleElementGradI(const FiniteElement &el,
                                           ElementTransformation &Tr,
                                           const Vector &elfun,
                                           DenseMatrix &elmat)
{
    elmat.SetSize(elfun.Size());
    elmat=0.0;
    if((*elem_attributes)[Tr.ElementNo]==mfem::ElementMarker::SBElementType::INSIDE){
        elint->AssembleElementGrad(el,Tr,elfun,elmat);
    }

    //If the element is not inside there isn't anything to integrate - the contribution is zero
    if((*elem_attributes)[Tr.ElementNo]!=mfem::ElementMarker::SBElementType::INSIDE){return;}
    //get dimensionality of the problem
    const int dim=Tr.GetSpaceDim();



    std::vector<int> bfaces;

    if(dim==3){
            const mfem::Table& elem_face=pmesh->ElementToFaceTable();
            int num_faces=elem_face.RowSize(Tr.ElementNo);
            const int* faces=elem_face.GetRow(Tr.ElementNo);
            for(int facei=0;facei<num_faces;facei++)
            {
                if((*face_attributes)[faces[facei]]==ElementMarker::SBFaceType::SURROGATE)
                {
                    bfaces.push_back(faces[facei]);
                }
            }
        }else{
            //dim==2
            const mfem::Table& elem_edge=pmesh->ElementToEdgeTable();
            int num_edges=elem_edge.RowSize(Tr.ElementNo);
            const int* faces=elem_edge.GetRow(Tr.ElementNo);
            for(int facei=0;facei<num_edges;facei++)
            {
                if((*face_attributes)[faces[facei]]==ElementMarker::SBFaceType::SURROGATE)
                {
                    bfaces.push_back(faces[facei]);
                }
            }
        }
        if(bfaces.size()==0){return;}

        mfem::Vector nnor; nnor.SetSize(dim); //surrogate normal vector
        mfem::Vector nnor_n; nnor_n.SetSize(dim); //surrogate normal vector normalized
        mfem::Vector nvec; nvec.SetSize(dim); //true normal vector
        const int dof=el.GetDof();
        const IntegrationRule *ir;

        //computes the gradient at the nodal points of the element
        DenseMatrix grad_phys;
        el.ProjectGrad(el,Tr,grad_phys);

        DenseMatrix shift_op; shift_op.SetSize(dof);
        DenseMatrix shift_ts; shift_ts.SetSize(dof);
        Vector shape; shape.SetSize(dof);
        Vector shape_sh; shape_sh.SetSize(dof);
        Vector shape_ts; shape_ts.SetSize(dof);
        DenseMatrix mtmp; mtmp.SetSize(dof);
        DenseMatrix btmp; btmp.SetSize(dim*dof);
        DenseMatrix bdrstress_op;//boundary stress operator
        bdrstress_op.SetSize(dim,dim*dof); bdrstress_op=0.0;
        DenseMatrix disp_op; disp_op.SetSize(dim,dim*dof);
        DenseMatrix B; B.SetSize(dof,dim);
        DenseMatrix D; //elasticity matrix
        Vector strain_v;
        if(dim==3){D.SetSize(6); strain_v.SetSize(6); strain_v=0.0;}
        else{D.SetSize(3);strain_v.SetSize(3); strain_v=0.0;}

        ElementTransformation *ltr;
        int elmNo=Tr.ElementNo;
        // we need the element number to figure out if we
        // will be working with element 1 or 2 in the integration


        // all element faces are stored in bfaces
        for(unsigned int i=0;i<bfaces.size();i++)
        {
            mfem::FaceElementTransformations *fctr;
            // the original element transformation is invalidated
            fctr=pmesh->GetInteriorFaceTransformations(bfaces[i]);
            if(fctr==nullptr){
                fctr=pmesh->GetFaceElementTransformations(bfaces[i]);
            }
            int order = 4*el.GetOrder();
            ir = &IntRules.Get(fctr->GetGeometryType(), order);

            // the original element transformation is invalidated
            // after the call to GetInteriorFaceTransformations
            if(fctr->Elem1No==elmNo){
                ltr=(fctr->Elem1);
            }else{
                ltr=(fctr->Elem2);
            }

            double facei=0.0;

            // integrate over the face
            for (int p = 0; p < ir->GetNPoints(); p++)
            {
               const IntegrationPoint &ip = ir->IntPoint(p);
               fctr->SetAllIntPoints(&ip); // sets the integration for the element as well
               //calculate the surrogate normal
               // Note: this normal accounts for the weight of the surface transformation
               // Jacobian i.e. nnor = nhat*det(J)
               CalcOrtho(fctr->Jacobian(),nnor);
               // Make sure the normal vector is pointing outside the domain.
               if(fctr->Elem2No==elmNo){nnor*=-1.0;}//reverse the normal
               //normalize nnor
               {
                   double nn0=nnor.Norml2();
                   if(nn0>std::numeric_limits<double>::epsilon()){nnor_n=nnor; nnor_n/=nn0;}
                   else{nnor_n=0.0;}
               }
               //calculate the distance
               double d=dist_coeff->Eval(*fctr,ip);
               //calculate the gradient of the SDF
               dist_grad->Eval(nvec,*fctr,ip);
               //normalize nvec - the true normal
               double nn=nvec.Norml2();
               if(nn>std::numeric_limits<double>::epsilon()){nvec/=nn;}
               else{nvec=0.0;}

               //evaluate the shift operators
               EvalShiftOperator(grad_phys,d,nvec,shift_order,shift_op,shift_ts);
               //evaluate the shape functions
               el.CalcPhysShape(*ltr, shape);
               el.CalcPhysDShape(*ltr, B);
               shift_op.MultTranspose(shape,shape_sh); //now we have the shifted shape functions
               shift_ts.MultTranspose(shape,shape_ts); //now we have the shifted test functions

               /*
               Vector shift_shape;
               EvalShiftShapes(shape,B,d,nvec,shift_shape);
               std::cout<<"True shift: ";shift_shape.Print(std::cout);
               std::cout<<"Appr shift: ";shape_sh.Print(std::cout);
               */

               //add the penalization term
               MultVWt(shape_ts,shape_sh,mtmp);
               //MultVWt(shape_sh,shape_sh,mtmp);
               double helm=std::pow(ltr->Weight(),1./dim); //the element size
               //add mtmp to elmat
               for(int ii=0;ii<dof;ii++){
               for(int jj=0;jj<dof;jj++){
               for(int di=0;di<dim;di++){
                   elmat(ii+di*dof,jj+di*dof)=elmat(ii+di*dof,jj+di*dof)+mtmp(ii,jj)*alpha*ip.weight*nnor.Norml2()/helm;
               }}}

               //compute \sigma_n where n is the surrogate normal nnor
               //compute the boundary stress operator bdrstress_op
               elco->EvalStiffness(D, strain_v, *fctr,ip);
               BndrStressOperator(dim,B,D,nnor_n,bdrstress_op);


               //compute shifted disp_op
               EvalDispOp(dim,shape_sh,disp_op);
               MultAtB(bdrstress_op,disp_op,btmp);
               for(int ii=0;ii<dim*dof;ii++){
               for(int jj=0;jj<dim*dof;jj++){
                   elmat(ii,jj)=elmat(ii,jj)-btmp(ii,jj)*ip.weight*nnor.Norml2();
               }}

               //compute true disp_op
               EvalDispOp(dim,shape,disp_op);
               MultAtB(disp_op,bdrstress_op,btmp);
               for(int ii=0;ii<dim*dof;ii++){
               for(int jj=0;jj<dim*dof;jj++){
                   elmat(ii,jj)=elmat(ii,jj)-btmp(ii,jj)*ip.weight*nnor.Norml2();
               }}


               //facei=facei+ip.weight*nnor.Norml2();

            }

            //std::cout<<"length/area="<<facei<<std::endl;


        }


}

void SBM2ELIntegrator:: EvalShiftShapes(mfem::Vector& shape, mfem::DenseMatrix& grad, double dist,
                                          mfem::Vector& dir, mfem::Vector& shift_shape)
{
    shift_shape.SetSize(shape.Size());
    shift_shape=shape;
    const int dim=dir.Size();
    for(int ii=0;ii<shape.Size();ii++){
        shift_shape[ii]=0.0;
        for(int di=0;di<dim;di++){
            shift_shape[ii]=shift_shape[ii]+grad(ii,di)*dir(di);
        }
        shift_shape[ii]=dist*shift_shape[ii]+shape[ii];
    }
}


void SBM2ELIntegrator::EvalTestShiftOperator(mfem::DenseMatrix &grad_phys, double dist,
                                             mfem::Vector &dir,
                                             mfem::DenseMatrix &shift_test)
{
    int ndofs=grad_phys.Width();
    int nrows=grad_phys.Height();
    int dim=nrows/ndofs;

    shift_test.SetSize(ndofs);

    mfem::DenseMatrix mat00;
    mat00.SetSize(ndofs);
    mat00=0.0;

    for(int di=0;di<dim;di++){
        shift_test.CopyRows(grad_phys,di*ndofs,(di+1)*ndofs-1);
        mat00.Add(dist*dir(di),shift_test);
    }
    shift_test.Diag(1.0,ndofs);
    shift_test.Add(1.0,mat00);
}

void SBM2ELIntegrator::EvalShiftOperator(mfem::DenseMatrix& grad_phys, double dist,
                       mfem::Vector& dir, int order,
                       mfem::DenseMatrix& shift_op, mfem::DenseMatrix& shift_test)
{
    int ndofs=grad_phys.Width();
    int nrows=grad_phys.Height();
    int dim=nrows/ndofs;

    shift_op.SetSize(ndofs);
    shift_test.SetSize(ndofs);

    mfem::DenseMatrix mat00;
    mat00.SetSize(ndofs);
    mat00=0.0;

    for(int di=0;di<dim;di++){
        shift_op.CopyRows(grad_phys,di*ndofs,(di+1)*ndofs-1);
        mat00.Add(dist*dir(di),shift_op);
    }
    shift_op.Diag(1.0,ndofs);
    if(order>0){shift_op.Add(1.0,mat00);}
    shift_test=shift_op;

    if(order>1)
    {
        mfem::DenseMatrix mat01;
        mfem::DenseMatrix mat02;
        mat01.SetSize(ndofs);
        mat02.SetSize(ndofs);

        mat02=mat00;
        mat02.Transpose();

        double facti=1.0;
        for(int i=2;i<order+1;i++)
        {
            facti=facti*i;
            MultABt(mat00,mat02,mat01);
            shift_op.Add(1.0/facti,mat01);
            mat00.Swap(mat01);
        }
    }
}

void SBM2ELIntegrator::EvalShiftOperator(mfem::DenseMatrix& grad_phys, double dist,
                       mfem::Vector& dir, int order, mfem::DenseMatrix& shift_op)
{
    int ndofs=grad_phys.Width();
    int nrows=grad_phys.Height();
    int dim=nrows/ndofs;

    shift_op.SetSize(ndofs);

    mfem::DenseMatrix mat00;
    mat00.SetSize(ndofs);
    mat00=0.0;

    for(int di=0;di<dim;di++){
        shift_op.CopyRows(grad_phys,di*ndofs,(di+1)*ndofs-1);
        mat00.Add(dist*dir(di),shift_op);
    }
    //mat00 holds the directional derivative

    shift_op.Diag(1.0,ndofs);
    shift_op.Add(1.0,mat00);
    if(order>1)
    {
        mfem::DenseMatrix mat01;
        mfem::DenseMatrix mat02;
        mat01.SetSize(ndofs);
        mat02.SetSize(ndofs);

        mat02=mat00;
        mat02.Transpose();

        double facti=1.0;
        for(int i=2;i<order+1;i++)
        {
            facti=facti*i;
            MultABt(mat00,mat02,mat01);
            shift_op.Add(1.0/facti,mat01);
            mat00.Swap(mat01);
        }
    }
}

void SBM2ELIntegrator::StrainOperator(int dim, mfem::DenseMatrix &B, mfem::DenseMatrix &strain_op)
{
    int ndofs=B.Height();
    if(dim==3){
        strain_op.SetSize(6,3*ndofs);
        strain_op=0.0;
        mfem::DenseMatrix Bx, By, Bz;
        Bx.Reset(B.GetData()+ndofs*0,1,ndofs);
        By.Reset(B.GetData()+ndofs*1,1,ndofs);
        Bz.Reset(B.GetData()+ndofs*2,1,ndofs);

        strain_op.CopyMN(Bx,0,ndofs*0);
        strain_op.CopyMN(By,1,ndofs*1);
        strain_op.CopyMN(Bz,2,ndofs*2);
        strain_op.CopyMN(Bz,3,ndofs*1); strain_op.CopyMN(By,3,ndofs*2);
        strain_op.CopyMN(Bz,4,ndofs*0); strain_op.CopyMN(Bx,4,ndofs*2);
        strain_op.CopyMN(By,5,ndofs*0); strain_op.CopyMN(Bx,5,ndofs*1);
    }else{
        strain_op.SetSize(3,2*ndofs);
        strain_op=0.0;
        mfem::DenseMatrix Bx, By;
        Bx.Reset(B.GetData()+ndofs*0,1,ndofs);
        By.Reset(B.GetData()+ndofs*1,1,ndofs);
        strain_op.CopyMN(Bx,0,ndofs*0);
        strain_op.CopyMN(By,1,ndofs*1);
        strain_op.CopyMN(By,2,ndofs*0); strain_op.CopyMN(Bx,2,ndofs*1);
    }

}

void SBM2ELIntegrator::StressOperator(int dim, mfem::DenseMatrix &B,
                                      mfem::DenseMatrix &D, mfem::DenseMatrix &stress_op)
{
    int ndofs=B.Height();
    if(dim==3){
        stress_op.SetSize(6,3*ndofs);}
    else{
        stress_op.SetSize(3,2*ndofs);}
    DenseMatrix strain_op;
    StrainOperator(dim,B,strain_op);
    MultAtB(D,strain_op,stress_op);
}

void SBM2ELIntegrator::EvalG(int dim, mfem::Vector &normv, mfem::DenseMatrix& G)
{
    if(dim==3){
        G.SetSize(6,3);
        G=0.0;
        G(0,0)=normv[0];
        G(1,1)=normv[1];
        G(2,2)=normv[2];
        G(3,1)=normv[2]; G(3,2)=normv[1];
        G(4,0)=normv[2]; G(4,2)=normv[0];
        G(5,0)=normv[1]; G(5,1)=normv[0];
    }else{
        G.SetSize(3,2);
        G=0.0;
        G(0,0)=normv[0];
        G(1,1)=normv[1];
        G(2,0)=normv[1]; G(2,1)=normv[0];
    }

}

void SBM2ELIntegrator::BndrStressOperator(int dim, mfem::DenseMatrix &B,
                                          mfem::DenseMatrix &D, mfem::Vector &normv,
                                          mfem::DenseMatrix &bdrstress_op)
{
    int ndofs=B.Height();
    if(dim==3){
        bdrstress_op.SetSize(3,3*ndofs);
    }else{
        bdrstress_op.SetSize(2,2*ndofs);
    }
    mfem::DenseMatrix stress_op;
    mfem::DenseMatrix G;
    EvalG(dim,normv,G);
    StressOperator(dim,B,D,stress_op);
    MultAtB(G,stress_op,bdrstress_op);
}

void SBM2NNIntegrator::AssembleElementVector(const FiniteElement &el,
                                             ElementTransformation &Tr,
                                             const Vector &elfun,
                                             Vector &elvect)
{
    elvect.SetSize(elfun.Size());
    elvect=0.0;

    if((*elem_attributes)[Tr.ElementNo]==mfem::ElementMarker::SBElementType::INSIDE){
        elint->AssembleElementVector(el,Tr,elfun,elvect);
    }

    //If the element is not inside there isn't anything to integrate - the contribution is zero
    if((*elem_attributes)[Tr.ElementNo]!=mfem::ElementMarker::SBElementType::INSIDE){return;}
    //get dimensionality of the problem
    const int dim=Tr.GetSpaceDim();

    std::vector<int> bfaces;

    if(dim==3){
        const mfem::Table& elem_face=pmesh->ElementToFaceTable();
        int num_faces=elem_face.RowSize(Tr.ElementNo);
        const int* faces=elem_face.GetRow(Tr.ElementNo);
        for(int facei=0;facei<num_faces;facei++)
        {
            if((*face_attributes)[faces[facei]]==ElementMarker::SBFaceType::SURROGATE)
            {
                bfaces.push_back(faces[facei]);
            }
        }
    }else{
        //dim==2
        const mfem::Table& elem_edge=pmesh->ElementToEdgeTable();
        int num_edges=elem_edge.RowSize(Tr.ElementNo);
        const int* faces=elem_edge.GetRow(Tr.ElementNo);
        for(int facei=0;facei<num_edges;facei++)
        {
            if((*face_attributes)[faces[facei]]==ElementMarker::SBFaceType::SURROGATE)
            {
                bfaces.push_back(faces[facei]);
            }
        }
    }

    if(bfaces.size()==0){return;} //no crossed edges/faces return


    mfem::Vector nnor; nnor.SetSize(dim); //surrogate normal vector
    mfem::Vector nnor_n; nnor_n.SetSize(dim); //surrogate normal vector normalized
    mfem::Vector nvec; nvec.SetSize(dim); //true normal vector
    const int dof=el.GetDof();
    const IntegrationRule *ir;

    //computes the gradient at the nodal points of the element
    DenseMatrix grad_phys;
    //el.ProjectGrad(el,Tr,grad_phys);
    FormL2Grad(el,Tr,grad_phys);

    DenseMatrix shift_op; shift_op.SetSize(dof);
    DenseMatrix shift_ts; shift_ts.SetSize(dof);
    Vector shape; shape.SetSize(dof);
    Vector shape_sh; shape_sh.SetSize(dof);
    Vector shape_ts; shape_ts.SetSize(dof);
    DenseMatrix bdrstress_op;//boundary stress operator
    bdrstress_op.SetSize(dim,dim*dof); bdrstress_op=0.0;
    DenseMatrix disp_op; disp_op.SetSize(dim,dim*dof);
    DenseMatrix B; B.SetSize(dof,dim);
    DenseMatrix Bsh; Bsh.SetSize(dof,dim);
    DenseMatrix D; //elasticity matrix
    Vector strain_v;
    if(dim==3){D.SetSize(6); strain_v.SetSize(6); strain_v=0.0;}
    else{D.SetSize(3);strain_v.SetSize(3); strain_v=0.0;}

    Vector dispx(elfun.GetData()+0*dof,dof);
    Vector dispy(elfun.GetData()+1*dof,dof);
    Vector dispz;
    if(dim==3){
        dispz.SetDataAndSize(elfun.GetData()+2*dof,dof);
    }

    mfem::Vector tract; tract.SetSize(dim); //tractions
    mfem::Vector tmpve; tmpve.SetSize(dim);
    mfem::Vector ddist; ddist.SetSize(dim); //[LS unit normal]*distance

    mfem::Vector rvec(dim*dof); //tmp residual vector

    int elmNo=Tr.ElementNo;
    ElementTransformation* ltr;

    // we need the element number to figure out if we
    // will be working with element 1 or 2 in the integration

    // all element faces are stored in bfaces
    for(unsigned int i=0;i<bfaces.size();i++)
    {
        mfem::FaceElementTransformations *fctr;
        // the original element transformation is invalidated
        fctr=pmesh->GetInteriorFaceTransformations(bfaces[i]);
        if(fctr==nullptr){
            fctr=pmesh->GetFaceElementTransformations(bfaces[i]);
        }
        int order = 4*el.GetOrder();
        ir = &IntRules.Get(fctr->GetGeometryType(), order);

        // the original element transformation is invalidated
        // after the call to GetInteriorFaceTransformations
        if(fctr->Elem1No==elmNo){
            ltr=fctr->Elem1;
        }else{
            ltr=fctr->Elem2;
        }

        double facei=0.0;

        // integrate over the face
        for (int p = 0; p < ir->GetNPoints(); p++)
        {
           const IntegrationPoint &ip = ir->IntPoint(p);
           fctr->SetAllIntPoints(&ip); // sets the integration for the element as well
           //calculate the surrogate normal
           // Note: this normal accounts for the weight of the surface transformation
           // Jacobian i.e. nnor = nhat*det(J)
           CalcOrtho(fctr->Jacobian(),nnor);
           // Make sure the normal vector is pointing outside the domain.
           if(fctr->Elem2No==elmNo){nnor*=-1.0;}//reverse the normal
           //normalize nnor
           {
               double nn0=nnor.Norml2();
               if(nn0>std::numeric_limits<double>::epsilon()){nnor_n=nnor; nnor_n/=nn0;}
               else{nnor_n=0.0;}
           }

           //calculate the distance
           double d=dist_coeff->Eval(*fctr,ip);
           //calculate the gradient of the SDF
           //the gradient can be evaluated only over a element and
           //cannot be evaluated over a face
           dist_grad->Eval(nvec,*ltr,ip);
           //normalize nvec - the true normal
           double nn=nvec.Norml2();
           if(nn>std::numeric_limits<double>::epsilon()){nvec/=nn;}
           else{nvec=0.0;}

           //compute traction
           {
               if(bdr_force==nullptr)
               {
                   tmpve=0.0;
               }else{
                   ddist=nvec; ddist*=d;
                   bdr_force->Eval(tmpve,*fctr,ip,ddist);
               }
           }

           //evaluate the shift operators
           EvalShiftOperator(grad_phys,d,nvec,shift_order,shift_op,shift_ts);

           //evaluate the shape functions
           el.CalcPhysShape(*ltr, shape);
           el.CalcPhysDShape(*ltr, B);
           shift_op.MultTranspose(shape,shape_sh); //now we have the shifted shape functions
           shift_ts.MultTranspose(shape,shape_ts); //now we have the shifted test functions

           double dp=nnor_n*nvec;

           for(int dd=0;dd<dim;dd++){
           for(int ii=0;ii<dof;ii++){
               elvect(dd*dof+ii)=elvect(dd*dof+ii)-shape(ii)*tmpve(dd)*ip.weight*nnor.Norml2()*dp;
           }}

           //compute shifted B and shifted stress operator
           elco->EvalStiffness(D, strain_v, *fctr,ip);
           MultAtB(shift_op,B,Bsh);
           BndrStressOperator(dim,Bsh,D,nvec,bdrstress_op);
           bdrstress_op.Mult(elfun,tract);
           for(int dd=0;dd<dim;dd++){
           for(int ii=0;ii<dof;ii++){
               elvect(dd*dof+ii)=elvect(dd*dof+ii)+shape(ii)*tract(dd)*ip.weight*nnor.Norml2()*dp;
           }}

           //add the penalty term
           for(int ii=0;ii<dim;ii++){tract[ii]=tract[ii]-tmpve[ii];}
           std::cout<<"rtract="<<tract.Norml2()<<"  "; tract.Print(std::cout);
           bdrstress_op.MultTranspose(tract,rvec);
           for(int dd=0;dd<dof*dim;dd++){
               elvect(dd)=elvect(dd)+rvec(dd)*alpha*ip.weight*nnor.Norml2();
           }

           //compute the boundary stress operator bdrstress_op
           BndrStressOperator(dim,B,D,nnor_n,bdrstress_op);
           //compute the traction to the surogate bdr
           bdrstress_op.Mult(elfun,tract);

           for(int dd=0;dd<dim;dd++){
           for(int ii=0;ii<dof;ii++){
               elvect(dd*dof+ii)=elvect(dd*dof+ii)-shape(ii)*tract(dd)*ip.weight*nnor.Norml2();
           }}

           facei=facei+ip.weight*nnor.Norml2();
        }

        //std::cout<<"length/area="<<facei<<std::endl;

    }


}

void SBM2NNIntegrator::AssembleElementGrad(const FiniteElement &el,
                                           ElementTransformation &Tr,
                                           const Vector &elfun,
                                           DenseMatrix &elmat)
{
    elmat.SetSize(elfun.Size());
    elmat=0.0;
    if((*elem_attributes)[Tr.ElementNo]==mfem::ElementMarker::SBElementType::INSIDE){
        elint->AssembleElementGrad(el,Tr,elfun,elmat);
    }

    //If the element is not inside there isn't anything to integrate - the contribution is zero
    if((*elem_attributes)[Tr.ElementNo]!=mfem::ElementMarker::SBElementType::INSIDE){return;}

    //get dimensionality of the problem
    const int dim=Tr.GetSpaceDim();
    std::vector<int> bfaces;
    if(dim==3){
        const mfem::Table& elem_face=pmesh->ElementToFaceTable();
        int num_faces=elem_face.RowSize(Tr.ElementNo);
        const int* faces=elem_face.GetRow(Tr.ElementNo);
        for(int facei=0;facei<num_faces;facei++)
        {
            if((*face_attributes)[faces[facei]]==mfem::ElementMarker::SBFaceType::SURROGATE)
            {
                bfaces.push_back(faces[facei]);
            }
        }
    }else{
        //dim==2
        const mfem::Table& elem_edge=pmesh->ElementToEdgeTable();
        int num_edges=elem_edge.RowSize(Tr.ElementNo);
        const int* faces=elem_edge.GetRow(Tr.ElementNo);
        for(int facei=0;facei<num_edges;facei++)
        {
            if((*face_attributes)[faces[facei]]==mfem::ElementMarker::SBFaceType::SURROGATE)
            {
                bfaces.push_back(faces[facei]);
            }
        }
    }

    if(bfaces.size()==0){return;}


    mfem::Vector nnor; nnor.SetSize(dim); //surrogate normal vector
    mfem::Vector nnor_n; nnor_n.SetSize(dim); //surrogate normal vector normalized
    mfem::Vector nvec; nvec.SetSize(dim); //true normal vector
    const int dof=el.GetDof();
    const IntegrationRule *ir;

    //computes the gradient at the nodal points of the element
    DenseMatrix grad_phys;
    el.ProjectGrad(el,Tr,grad_phys);
    /*
    FormL2Grad(el,Tr,grad_phys);
    {
        DenseMatrix grad1;
        DenseMatrix C; C.SetSize(dim*dof,dof); C=0.0;
        el.ProjectGrad(el,Tr,grad1);
        Add(grad_phys,grad1,-1.0,C);

        std::cout<<"NormC="<<C.FNorm()<<std::endl;
    }
    */



    DenseMatrix shift_op; shift_op.SetSize(dof);
    DenseMatrix shift_ts; shift_ts.SetSize(dof);
    Vector shape; shape.SetSize(dof);
    Vector shape_sh; shape_sh.SetSize(dof);
    Vector shape_ts; shape_ts.SetSize(dof);
    DenseMatrix btmp; btmp.SetSize(dim*dof);
    DenseMatrix bdrstress_op;//boundary stress operator
    bdrstress_op.SetSize(dim,dim*dof); bdrstress_op=0.0;
    DenseMatrix disp_op; disp_op.SetSize(dim,dim*dof);
    DenseMatrix B; B.SetSize(dof,dim);
    DenseMatrix Bsh; Bsh.SetSize(dof,dim);
    DenseMatrix D; //elasticity matrix
    Vector strain_v;
    if(dim==3){D.SetSize(6); strain_v.SetSize(6); strain_v=0.0;}
    else{D.SetSize(3);strain_v.SetSize(3); strain_v=0.0;}

    ElementTransformation *ltr;
    int elmNo=Tr.ElementNo;
    // we need the element number to figure out if we
    // will be working with element 1 or 2 in the integration


    // all element faces are stored in bfaces
    for(unsigned int i=0;i<bfaces.size();i++)
    {
        mfem::FaceElementTransformations *fctr;
        // the original element transformation is invalidated
        fctr=pmesh->GetInteriorFaceTransformations(bfaces[i]);
        if(fctr==nullptr){
            fctr=pmesh->GetFaceElementTransformations(bfaces[i]);
        }
        int order = 4*el.GetOrder();
        ir = &IntRules.Get(fctr->GetGeometryType(), order);

        // the original element transformation is invalidated
        // after the call to GetInteriorFaceTransformations
        if(fctr->Elem1No==elmNo){
            ltr=(fctr->Elem1);
        }else{
            ltr=(fctr->Elem2);
        }

        // integrate over the face
        for (int p = 0; p < ir->GetNPoints(); p++)
        {
           const IntegrationPoint &ip = ir->IntPoint(p);
           fctr->SetAllIntPoints(&ip); // sets the integration for the element as well
           //calculate the surrogate normal
           // Note: this normal accounts for the weight of the surface transformation
           // Jacobian i.e. nnor = nhat*det(J)
           CalcOrtho(fctr->Jacobian(),nnor);
           // Make sure the normal vector is pointing outside the domain.
           if(fctr->Elem2No==elmNo){nnor*=-1.0;}//reverse the normal
           //normalize nnor
           {
               double nn0=nnor.Norml2();
               if(nn0>std::numeric_limits<double>::epsilon()){nnor_n=nnor; nnor_n/=nn0;}
               else{nnor_n=0.0;}
           }
           //calculate the distance
           double d=dist_coeff->Eval(*fctr,ip);
           //calculate the gradient of the SDF
           dist_grad->Eval(nvec,*ltr,ip);
           //normalize nvec - the true normal
           double nn=nvec.Norml2();
           if(nn>std::numeric_limits<double>::epsilon()){nvec/=nn;}
           else{nvec=0.0;}
           //std::cout<<"d="<<d<<std::endl;

           //evaluate the shift operators
           EvalShiftOperator(grad_phys,d,nvec,shift_order,shift_op,shift_ts);
           //evaluate the shape functions
           el.CalcPhysShape(*ltr, shape);
           el.CalcPhysDShape(*ltr, B);
           shift_op.MultTranspose(shape,shape_sh); //now we have the shifted shape functions
           shift_ts.MultTranspose(shape,shape_ts); //now we have the shifted test functions

           /*
           {
               Vector  shape_nn; shape_nn.SetSize(shape.Size());
               EvalShiftShapes(shape,B,d,nvec,shape_nn);
               std::cout<<"SHe="<<std::sqrt(shape_nn*shape_sh)/shape_nn.Norml2()<<" "<<shape_sh.Norml2()<<" "<<shape_nn.Norml2()<<std::endl;
           }
           */


           //compute \sigma_n where n is the surrogate normal nnor
           //compute the boundary stress operator bdrstress_op
           elco->EvalStiffness(D, strain_v, *fctr,ip);
           BndrStressOperator(dim,B,D,nnor_n,bdrstress_op);

           //compute disp_op
           if(assembly_interior==false){
           EvalDispOp(dim,shape,disp_op);
           MultAtB(disp_op,bdrstress_op,btmp);
           for(int ii=0;ii<dim*dof;ii++){
           for(int jj=0;jj<dim*dof;jj++){
               elmat(ii,jj)=elmat(ii,jj)-btmp(ii,jj)*ip.weight*nnor.Norml2();
           }}
           }


           //compute shifted B
           MultAtB(shift_op,B,Bsh);
           BndrStressOperator(dim,Bsh,D,nvec,bdrstress_op);
           double dp=nvec*nnor_n;
           if(assembly_interior==false){
           MultAtB(disp_op,bdrstress_op,btmp);
           for(int ii=0;ii<dim*dof;ii++){
           for(int jj=0;jj<dim*dof;jj++){
               elmat(ii,jj)=elmat(ii,jj)+btmp(ii,jj)*ip.weight*nnor.Norml2()*dp;
           }}
           }

           //compute the penalty
           //MultAAt(bdrstress_op,btmp);
           MultAtB(bdrstress_op,bdrstress_op,btmp);
           for(int ii=0;ii<dim*dof;ii++){
           for(int jj=0;jj<dim*dof;jj++){
               elmat(ii,jj)=elmat(ii,jj)+btmp(ii,jj)*ip.weight*nnor.Norml2()*alpha;
           }}

        }
    }

}

void SBM2NNIntegrator::EvalShiftOperator(mfem::DenseMatrix& grad_phys, double dist,
                                         mfem::Vector& dir, int order,
                                         mfem::DenseMatrix& shift_op, mfem::DenseMatrix& shift_test)
{
    int ndofs=grad_phys.Width();
    int nrows=grad_phys.Height();
    int dim=nrows/ndofs;

    shift_op.SetSize(ndofs);
    shift_test.SetSize(ndofs);

    mfem::DenseMatrix mat00;
    mat00.SetSize(ndofs);
    mat00=0.0;

    for(int di=0;di<dim;di++){
        shift_op.CopyRows(grad_phys,di*ndofs,(di+1)*ndofs-1);
        mat00.Add(dist*dir(di),shift_op);
    }
    shift_op.Diag(1.0,ndofs);
    if(order>0){shift_op.Add(1.0,mat00);}
    shift_test=shift_op;

    if(order>1)
    {
        mfem::DenseMatrix mat01;
        mfem::DenseMatrix mat02;
        mat01.SetSize(ndofs);
        mat02.SetSize(ndofs);

        mat02=mat00;
        mat02.Transpose();

        double facti=1.0;
        for(int i=2;i<order+1;i++)
        {
            facti=facti*i;
            MultABt(mat00,mat02,mat01);
            shift_op.Add(1.0/facti,mat01);
            mat00.Swap(mat01);
        }
    }
}

void SBM2NNIntegrator::StrainOperator(int dim, mfem::DenseMatrix &B, mfem::DenseMatrix &strain_op)
{
    int ndofs=B.Height();
    if(dim==3){
        strain_op.SetSize(6,3*ndofs);
        strain_op=0.0;
        mfem::DenseMatrix Bx, By, Bz;
        Bx.Reset(B.GetData()+ndofs*0,1,ndofs);
        By.Reset(B.GetData()+ndofs*1,1,ndofs);
        Bz.Reset(B.GetData()+ndofs*2,1,ndofs);

        strain_op.CopyMN(Bx,0,ndofs*0);
        strain_op.CopyMN(By,1,ndofs*1);
        strain_op.CopyMN(Bz,2,ndofs*2);
        strain_op.CopyMN(Bz,3,ndofs*1); strain_op.CopyMN(By,3,ndofs*2);
        strain_op.CopyMN(Bz,4,ndofs*0); strain_op.CopyMN(Bx,4,ndofs*2);
        strain_op.CopyMN(By,5,ndofs*0); strain_op.CopyMN(Bx,5,ndofs*1);
    }else{
        strain_op.SetSize(3,2*ndofs);
        strain_op=0.0;
        mfem::DenseMatrix Bx, By;
        Bx.Reset(B.GetData()+ndofs*0,1,ndofs);
        By.Reset(B.GetData()+ndofs*1,1,ndofs);
        strain_op.CopyMN(Bx,0,ndofs*0);
        strain_op.CopyMN(By,1,ndofs*1);
        strain_op.CopyMN(By,2,ndofs*0); strain_op.CopyMN(Bx,2,ndofs*1);
    }

}

void SBM2NNIntegrator::StressOperator(int dim, mfem::DenseMatrix &B,
                                      mfem::DenseMatrix &D, mfem::DenseMatrix &stress_op)
{
    int ndofs=B.Height();
    if(dim==3){
        stress_op.SetSize(6,3*ndofs);}
    else{
        stress_op.SetSize(3,2*ndofs);}
    DenseMatrix strain_op;
    StrainOperator(dim,B,strain_op);
    MultAtB(D,strain_op,stress_op);
}

void SBM2NNIntegrator::EvalG(int dim, mfem::Vector &normv, mfem::DenseMatrix& G)
{
    if(dim==3){
        G.SetSize(6,3);
        G=0.0;
        G(0,0)=normv[0];
        G(1,1)=normv[1];
        G(2,2)=normv[2];
        G(3,1)=normv[2]; G(3,2)=normv[1];
        G(4,0)=normv[2]; G(4,2)=normv[0];
        G(5,0)=normv[1]; G(5,1)=normv[0];
    }else{
        G.SetSize(3,2);
        G=0.0;
        G(0,0)=normv[0];
        G(1,1)=normv[1];
        G(2,0)=normv[1]; G(2,1)=normv[0];
    }

}

void SBM2NNIntegrator::BndrStressOperator(int dim, mfem::DenseMatrix &B,
                                          mfem::DenseMatrix &D, mfem::Vector &normv,
                                          mfem::DenseMatrix &bdrstress_op)
{
    int ndofs=B.Height();
    if(dim==3){
        bdrstress_op.SetSize(3,3*ndofs);
    }else{
        bdrstress_op.SetSize(2,2*ndofs);
    }
    mfem::DenseMatrix stress_op;
    mfem::DenseMatrix G;
    EvalG(dim,normv,G);
    StressOperator(dim,B,D,stress_op);
    MultAtB(G,stress_op,bdrstress_op);
}

void SBM2NNIntegrator:: EvalShiftShapes(mfem::Vector& shape, mfem::DenseMatrix& grad, double dist,
                                          mfem::Vector& dir, mfem::Vector& shift_shape)
{
    shift_shape.SetSize(shape.Size());
    //shift_shape=shape;
    const int dim=dir.Size();
    for(int ii=0;ii<shape.Size();ii++){
        shift_shape[ii]=0.0;
        for(int di=0;di<dim;di++){
            shift_shape[ii]=shift_shape[ii]+grad(ii,di)*dir(di);
        }
        shift_shape[ii]=dist*shift_shape[ii]+shape[ii];
    }
}

void SBM2NNIntegrator::FormL2Grad(const FiniteElement &el, ElementTransformation &Tr, DenseMatrix &gradop)
{

    int dof=el.GetDof();
    int dim=Tr.GetSpaceDim();
    int order = 4*el.GetOrder();
    const IntegrationRule *ir;
    ir = &IntRules.Get(Tr.GetGeometryType(), order);

    Vector shape; shape.SetSize(dof);
    DenseMatrix B; B.SetSize(dof,dim);
    DenseMatrix Rx; Rx.SetSize(dof); Rx=0.0;
    DenseMatrix Ry; Ry.SetSize(dof); Ry=0.0;
    DenseMatrix Rz;
    if(dim==3){
        Rz.SetSize(dof);
        Rz=0.0;
    }
    DenseMatrix M; M.SetSize(dof); M=0.0;
    DenseMatrix T; T.SetSize(dof); T=0.0;
    gradop.SetSize(dim*dof,dof); gradop=0.0;
    Vector bdir;

    double w;

    for (int p = 0; p < ir->GetNPoints(); p++)
    {
       const IntegrationPoint &ip = ir->IntPoint(p);
       Tr.SetIntPoint(&ip);
       el.CalcPhysShape(Tr,shape);
       el.CalcPhysDShape(Tr,B);

       w=ip.weight * Tr.Weight();


       bdir.SetDataAndSize(B.GetData()+0*dof,dof);
       AddMult_a_VWt(w,shape,bdir,Rx);
       bdir.SetDataAndSize(B.GetData()+1*dof,dof);
       AddMult_a_VWt(w,shape,bdir,Ry);
       if(dim==3){
           bdir.SetDataAndSize(B.GetData()+2*dof,dof);
           AddMult_a_VWt(w,shape,bdir,Rz);
       }

       AddMult_a_VVt(w,shape,M);
    }

    DenseMatrixInverse iM(M);
    iM.Mult(Rx,T);
    gradop.AddMatrix(T,dof*0,0);
    iM.Mult(Ry,T);
    gradop.AddMatrix(T,dof*1,0);
    if(dim==3){
        iM.Mult(Rz,T);
        gradop.AddMatrix(T,dof*2,0);
    }
}


LevelSetElasticitySolver::LevelSetElasticitySolver(mfem::ParMesh& mesh, int vorder)
{
    pmesh=&mesh;
    int dim=pmesh->Dimension();
    vfec=new H1_FECollection(vorder,dim);
    vfes=new mfem::ParFiniteElementSpace(pmesh,vfec,dim, Ordering::byVDIM);

    fdisp.SetSpace(vfes); fdisp=0.0;
    adisp.SetSpace(vfes); adisp=0.0;

    sol.SetSize(vfes->GetTrueVSize()); sol=0.0;
    rhs.SetSize(vfes->GetTrueVSize()); rhs=0.0;
    adj.SetSize(vfes->GetTrueVSize()); adj=0.0;
    tmv.SetSize(vfes->GetTrueVSize()); tmv=0.0;

    SetLinearSolver();
    SetPrintLevel();

    lvforce=nullptr;
    volforce=nullptr;

    nf=nullptr; //nonlinear form
    pf=nullptr; //nonlinear form for the preconditioner

    lsfunc=nullptr; //level-set function
    distco=nullptr;
    gradco=nullptr;

}

LevelSetElasticitySolver::~LevelSetElasticitySolver()
{
    delete nf;
    delete pf;

    delete lvforce;

    delete vfes;
    delete vfec;

    for(unsigned int i=0;i<materials.size();i++)
    {
        delete materials[i];
    }

    delete distco;
    delete gradco;
}

void LevelSetElasticitySolver::SetPrintLevel(int prtl)
{
    print_level=prtl;
}

void LevelSetElasticitySolver::SetLinearSolver(double rtol, double atol, int miter, int restart)
{
    linear_rtol=rtol;
    linear_atol=atol;
    linear_iter=miter;
    linear_rest=restart;
}

void LevelSetElasticitySolver::SetLSF(mfem::ParGridFunction *lf)
{
    delete distco; distco=nullptr;
    delete gradco; gradco=nullptr;

    if(lf==nullptr)
    {
        lsfunc=lf;
        markers_tdof_list.DeleteAll();
        element_markers.DeleteAll();
        face_markers.DeleteAll();
    }
    else
    {
        lsfunc=lf;
        markers_tdof_list.DeleteAll();
        //mark the elements including the crossed ones
        mfem::ElementMarker smarker(*pmesh,true);
        smarker.SetLevelSetFunction(*lsfunc);
        smarker.MarkElements(element_markers);
        smarker.MarkFaces(face_markers);
        smarker.ListEssentialTDofs(element_markers,*vfes,markers_tdof_list);

        distco=new GridFunctionCoefficient(lsfunc);
        gradco=new GradientGridFunctionCoefficient(lsfunc);
    }


    //delete the forms
    delete nf; nf=nullptr;
    delete pf; pf=nullptr;
}

void LevelSetElasticitySolver::AddDispBC(int id, int dir, double val)
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

void LevelSetElasticitySolver::AddDispBC(int id, int dir, mfem::Coefficient &val)
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

void LevelSetElasticitySolver::AddDispBC(int id, mfem::VectorCoefficient& val)
{
    bcca[id]=&val;
}


void LevelSetElasticitySolver::SetVolForce(double fx, double fy, double fz)
{
    delete lvforce;
    int dim=pmesh->Dimension();
    mfem::Vector ff(dim); ff(0)=fx; ff(1)=fy;
    if(dim==3){ff(2)=fz;}
    lvforce=new mfem::VectorConstantCoefficient(ff);
    volforce=lvforce;

}

void LevelSetElasticitySolver::SetVolForce(mfem::VectorCoefficient& fv)
{
    volforce=&fv;
}

void LevelSetElasticitySolver::FSolve()
{
    // Set the BC
    ess_tdofv.DeleteAll();
    Array<int> ess_tdofx;
    Array<int> ess_tdofy;
    Array<int> ess_tdofz;

    sol=0.0;

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

        //set vector coefficients
        for(auto it=bcca.begin();it!=bcca.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
            fdisp.ProjectBdrCoefficient(*(it->second), ess_bdr);
            //copy tdofs from velocity grid function
            fdisp.GetTrueDofs(rhs); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdof_list.Size();ii++)
            {
                sol[ess_tdof_list[ii]]=rhs[ess_tdof_list[ii]];
            }
            ess_tdofv.Append(ess_tdof_list);
        }
    }//the solution vector is populated with the BCs


    //allocate the nf
    if(nf==nullptr)
    {
        nf=new mfem::ParNonlinearForm(vfes);
        pf=new mfem::ParNonlinearForm(vfes);

        //add the integrators
        for(unsigned int i=0;i<materials.size();i++)
        {
            mfem::SBM2NNIntegrator* tin=new mfem::SBM2NNIntegrator(pmesh,vfes,
                                                                   element_markers,face_markers);

            if(lsfunc!=nullptr){tin->SetDistance(distco,gradco);}
            tin->SetShiftOrder(vfes->GetOrder(0));
            tin->SetElasticityCoefficient(materials[i]);
            nf->AddDomainIntegrator(tin);

            tin=new mfem::SBM2NNIntegrator(pmesh,vfes,element_markers,face_markers);
            if(lsfunc!=nullptr){tin->SetDistance(distco,gradco);}
            tin->SetShiftOrder(vfes->GetOrder(0));
            tin->SetElasticityCoefficient(materials[i]);
            tin->SetAssemblyInterior(true);
            pf->AddDomainIntegrator(tin);
        }

        if(volforce!=nullptr){
            nf->AddDomainIntegrator(new NLVolForceIntegrator(volforce));
        }
    }

    nf->SetGradientType(mfem::Operator::Type::Hypre_ParCSR);
    pf->SetGradientType(mfem::Operator::Type::Hypre_ParCSR);

    tmv=0.0;
    mfem::Operator& A=nf->GetGradient(tmv);
    mfem::Operator& P=pf->GetGradient(tmv);

    //set the rhs
    nf->Mult(tmv,rhs); rhs*=-1.0;

    // create a union between ess_tdofv and markers_tdof_list
    mfem::Array<int> ess_tdof_list;
    ess_tdof_list=markers_tdof_list;
    ess_tdof_list.Append(ess_tdofv);
    ess_tdof_list.Sort();
    ess_tdof_list.Unique();


    mfem::HypreParMatrix* M=static_cast<mfem::HypreParMatrix*>(&A);
    mfem::HypreParMatrix* Ae=M->EliminateRowsCols(ess_tdof_list);
    M->EliminateZeroRows();
    M->EliminateBC(*Ae,ess_tdof_list, sol, rhs);
    delete Ae;

    mfem::HypreParMatrix* K=static_cast<mfem::HypreParMatrix*>(&P);
    mfem::HypreParMatrix* Ke=K->EliminateRowsCols(ess_tdof_list);
    K->EliminateZeroRows();
    delete Ke;

    //allocate the solvers and solve the problem
    {
        mfem::GMRESSolver *ls;
        mfem::HypreBoomerAMG *prec;

        prec=new mfem::HypreBoomerAMG();
        prec->SetSystemsOptions(pmesh->Dimension());
        prec->SetElasticityOptions(vfes);
        prec->SetPrintLevel(print_level);
        prec->SetOperator(*K);

        ls=new mfem::GMRESSolver(pmesh->GetComm());
        ls->SetPrintLevel(print_level);
        ls->SetAbsTol(linear_atol);
        ls->SetRelTol(linear_rtol);
        ls->SetMaxIter(linear_iter);
        ls->SetKDim(linear_rest);
        ls->SetOperator(*M);
        ls->SetPreconditioner(*prec);

        ls->Mult(rhs,sol);

        delete ls;
        delete prec;
    }
}

void LevelSetElasticitySolver::ASolve(mfem::Vector &rhs)
{
    if(nf==nullptr){
        MFEM_ABORT("FSolve() should be called before calling ASolve()");
    }

    //one can reuse the operators
    mfem::Operator& A=nf->GetGradient(tmv);
    mfem::Operator& P=pf->GetGradient(tmv);

    // create a union between ess_tdofv and markers_tdof_list
    mfem::Array<int> ess_tdof_list;
    ess_tdof_list=markers_tdof_list;
    ess_tdof_list.Append(ess_tdofv);
    ess_tdof_list.Sort();
    ess_tdof_list.Unique();

    adj=0.0;

    mfem::HypreParMatrix* M=static_cast<mfem::HypreParMatrix*>(&A);
    mfem::HypreParMatrix* Mt=M->Transpose();
    mfem::HypreParMatrix* Ae=Mt->EliminateRowsCols(ess_tdof_list);
    Mt->EliminateZeroRows();
    Mt->EliminateBC(*Ae,ess_tdof_list, adj, rhs);
    delete Ae;

    mfem::HypreParMatrix* K=static_cast<mfem::HypreParMatrix*>(&P);
    mfem::HypreParMatrix* Ke=K->EliminateRowsCols(ess_tdof_list);
    K->EliminateZeroRows();
    delete Ke;

    mfem::GMRESSolver *ls;
    mfem::HypreBoomerAMG *prec;

    prec=new mfem::HypreBoomerAMG();
    prec->SetSystemsOptions(pmesh->Dimension());
    prec->SetElasticityOptions(vfes);
    prec->SetPrintLevel(print_level);
    prec->SetOperator(*K);

    ls=new mfem::GMRESSolver(pmesh->GetComm());
    ls->SetPrintLevel(print_level);
    ls->SetAbsTol(linear_atol);
    ls->SetRelTol(linear_rtol);
    ls->SetMaxIter(linear_iter);
    ls->SetKDim(linear_rest);
    ls->SetOperator(*M);
    ls->SetPreconditioner(*prec);

    ls->Mult(rhs,adj);

    delete ls;
    delete prec;
    delete Mt;

}
