#include "mfem.hpp"
#include "mtop_solvers.hpp"
#include "general/forall.hpp"


LElasticOperator::LElasticOperator(mfem::ParMesh* mesh_, int vorder)
{
    pmesh=mesh_;
    const int dim=pmesh->Dimension();
    vfec=new mfem::H1_FECollection(vorder,dim);
    vfes=new mfem::ParFiniteElementSpace(pmesh,vfec,dim, mfem::Ordering::byVDIM);

    fdisp.SetSpace(vfes); fdisp=0.0;
    adisp.SetSpace(vfes); adisp=0.0;

    sol.SetSize(vfes->GetTrueVSize()); sol=0.0;
    rhs.SetSize(vfes->GetTrueVSize()); rhs=0.0;
    adj.SetSize(vfes->GetTrueVSize()); adj=0.0;

    SetLinearSolver();

    prec=nullptr;
    ls=nullptr;

    mfem::Operator::width=vfes->GetTrueVSize();
    mfem::Operator::height=vfes->GetTrueVSize();

    lvforce=nullptr;
    volforce=nullptr;

    E=nullptr;
    nu=nullptr;

    lambda=nullptr;
    mu=nullptr;

    bf=nullptr;
    lf=nullptr;
}

LElasticOperator::~LElasticOperator()
{
    delete prec;
    delete ls;

    delete bf;
    delete lf;

    delete vfes;
    delete vfec;

    delete lvforce;

    for(auto it=load_coeff.begin();it!=load_coeff.end();it++){
        delete it->second;
    }

    delete lambda;
    delete mu;
}

void LElasticOperator::SetLinearSolver(mfem::real_t rtol, mfem::real_t atol, int miter)
{
    linear_rtol=rtol;
    linear_atol=atol;
    linear_iter=miter;
}

void LElasticOperator::AddDispBC(int id, int dir, mfem::real_t val)
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

void LElasticOperator::DelDispBC()
{
    bccx.clear();
    bccy.clear();
    bccz.clear();

    bcx.clear();
    bcy.clear();
    bcz.clear();

    ess_tdofv.DeleteAll();
}

void LElasticOperator::AddDispBC(int id, int dir, mfem::Coefficient &val)
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

void LElasticOperator::SetVolForce(mfem::real_t fx, double fy, double fz)
{
    delete lvforce;
    int dim=pmesh->Dimension();
    mfem::Vector ff(dim); ff(0)=fx; ff(1)=fy;
    if(dim==3){ff(2)=fz;}
    lvforce=new mfem::VectorConstantCoefficient(ff);
    volforce=lvforce;

}

void LElasticOperator::SetVolForce(mfem::VectorCoefficient& fv)
{
    volforce=&fv;
}

void LElasticOperator::SetEssTDofs(mfem::Vector& bsol, mfem::Array<int>& ess_dofs)
{
    // Set the BC

    ess_tdofv.DeleteAll();

    mfem::Array<int> ess_tdofx;
    mfem::Array<int> ess_tdofy;
    mfem::Array<int> ess_tdofz;

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
            mfem::Vector& vc=fdisp.GetTrueVector();
            for(int ii=0;ii<ess_tdofx.Size();ii++)
            {
                bsol[ess_tdofx[ii]]=vc[ess_tdofx[ii]];
            }
        }
        ess_dofs.Append(ess_tdofx); ess_tdofx.DeleteAll();

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
            mfem::Vector& vc=fdisp.GetTrueVector();
            for(int ii=0;ii<ess_tdofy.Size();ii++)
            {
                bsol[ess_tdofy[ii]]=vc[ess_tdofy[ii]];
            }
        }
        ess_dofs.Append(ess_tdofy); ess_tdofy.DeleteAll();

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
                mfem::Vector& vc=fdisp.GetTrueVector();
                for(int ii=0;ii<ess_tdofz.Size();ii++)
                {
                    bsol[ess_tdofz[ii]]=vc[ess_tdofz[ii]];
                }
            }
            ess_dofs.Append(ess_tdofz); ess_tdofz.DeleteAll();
        }
    }

}

void LElasticOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
    //the rhs x is assumed to have the contribution of the BC
    ls->Mult(x,y);

    int N=ess_tdofv.Size();
    mfem::real_t *yp = y.ReadWrite();
    const mfem::real_t *sp = sol.Read();
    const int *ep = ess_tdofv.Read();
    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i)
    {
        yp[ep[i]]=sp[ep[i]];
    });
}

void LElasticOperator::MultTranspose(const mfem::Vector &x, mfem::Vector &y) const
{
    //the adjoint rhs is assumed to be corrected for the BC
    //K is symmetric
    ls->Mult(x,y);

    int N=ess_tdofv.Size();
    ess_tdofv.Read();

    mfem::real_t *yp = y.ReadWrite();
    const int *ep = ess_tdofv.Read();

    mfem::forall(N,[=] MFEM_HOST_DEVICE (int i)
    {
        yp[ep[i]]=mfem::real_t(0.0);
    });
}

void LElasticOperator::Assemble()
{
    if(bf==nullptr){return;}

    //set BC
    sol=mfem::real_t(0.0);
    SetEssTDofs(sol,ess_tdofv);

    bf->Assemble(0);
    bf->FormSystemMatrix(ess_tdofv,K);

    if(ls==nullptr){
        ls=new CGSolver(pmesh->GetComm());
        ls->SetAbsTol(linear_atol);
        ls->SetRelTol(linear_rtol);
        ls->SetMaxIter(linear_iter);
        prec=new mfem::HypreBoomerAMG();
        //set the rigid body modes
        prec->SetElasticityOptions(vfes);
        prec->SetPrintLevel(1);
        ls->SetPreconditioner(*prec);
        ls->SetOperator(K);
        ls->SetPrintLevel(1);
    }else{
        ls->SetOperator(K);
    }

    std::cout<<pmesh->GetMyRank()<<" LSW="<<ls->Width()<<" LSH="<<ls->Height()
                                 <<" KFW="<<K.Width()<<" KFH="<<K.Height()<<std::endl;
}

void LElasticOperator::FSolve()
{
    if(lf==nullptr){
        lf=new ParLinearForm(vfes);
        if(volforce!=nullptr){
            lf->AddDomainIntegrator(new VectorDomainLFIntegrator(*volforce));
        }
        //add surface loads

    }

    (*lf)=mfem::real_t(0.0);

    lf->Assemble();
    lf->ParallelAssemble(rhs);

    for(int i=0;i<ess_tdofv.Size();i++){
        rhs[ess_tdofv[i]]=sol[ess_tdofv[i]];
    }

    ls->Mult(rhs,sol);

}

FRElasticSolver::FRElasticSolver(mfem::ParMesh* mesh_, int vorder, mfem::real_t freq_):
                    LElasticOperator(mesh_,vorder)
{

    freq=freq_;
    alpha=0.0;
    beta=0.0;

    mf=nullptr;
    cf=nullptr;

    lrf=nullptr;
    lif=nullptr;

    num_svd_modes=5;
    num_svd_iter=6;
    pop=nullptr;
    ss_solver=nullptr;
}

FRElasticSolver::~FRElasticSolver()
{

    delete ss_solver;
    delete pop;

    delete mf;
    delete cf;

    delete lrf;
    delete lif;

}

void FRElasticSolver::Assemble()
{
    LElasticOperator::Assemble();

    if(mf==nullptr){
        MFEM_WARNING("FRElasticSolver::Mass bilinear form is not defined!");
        return;
    }
    mf->Assemble();
    mf->FormSystemMatrix(LElasticOperator::ess_tdofv,hmf);

    if(cf==nullptr){
        MFEM_WARNING("FRElasticSolver::Damping bilinear form is not defined!");
        return;
    }
    cf->Assemble();
    cf->FormSystemMatrix(LElasticOperator::ess_tdofv,hcf);
}

void FRElasticSolver::AssembleSVD()
{
    delete ss_solver;
    delete pop;
    ss_solver=new RandomizedSubspaceIteration(LElasticOperator::pmesh->GetComm());

    std::cout<<LElasticOperator::pmesh->GetMyRank()<<"CW="<<hmf->Width()<<" CH="<<hmf->Height()<<std::endl;

    pop=new LocProductOperator(LElasticOperator::ls,hmf.Ptr());
    ss_solver->SetOperator(*pop);
    ss_solver->SetNumModes(num_svd_modes);
    ss_solver->SetNumIter(num_svd_iter);
    ss_solver->SetConstrDOFs(LElasticOperator::ess_tdofv);
    ss_solver->Solve();

    auto eigv=ss_solver->GetModes();
    Vector tmpv; tmpv.SetSize(eigv[0].Size());
    for(int i=0;i<ss_solver->GetNumModes();i++){
        pop->MultB(eigv[i],tmpv);
        for(int j=0;j<ss_solver->GetNumModes();j++){
             real_t bb=InnerProduct(LElasticOperator::pmesh->GetComm(),tmpv,eigv[j]);
             if(0==pmesh->GetMyRank()){
                 std::cout<<bb<<" ";
             }

        }
        if(0==pmesh->GetMyRank()){
            std::cout<<std::endl;
        }
    }

}
