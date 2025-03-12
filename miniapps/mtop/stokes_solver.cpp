#include "stokes_solver.hpp"
#include "general/forall.hpp"

StokesOperator::StokesOperator(mfem::ParMesh* mesh_,int vorder):zeroc(0.0),onec(1.0)
{
    if(vorder<2){vorder=2;}

    pmesh=mesh_;
    const int dim=pmesh->Dimension();
    vfec=new mfem::H1_FECollection(vorder,dim);
    vfes=new mfem::ParFiniteElementSpace(pmesh,vfec,dim, mfem::Ordering::byVDIM);

    pfec=new mfem::H1_FECollection(vorder-1,dim);
    pfes=new mfem::ParFiniteElementSpace(pmesh,pfec,1, mfem::Ordering::byVDIM);

    fvelo.SetSpace(vfes); fvelo=0.0;
    avelo.SetSpace(vfes); avelo=0.0;

    fpres.SetSpace(pfes); fpres=0.0;
    apres.SetSpace(pfes); apres=0.0;

    block_offsets.SetSize(3);
    block_offsets[0] = 0;
    block_offsets[1] = vfes->GetVSize();
    block_offsets[2] = pfes->GetVSize();
    block_offsets.PartialSum();

    block_true_offsets.SetSize(3);
    block_true_offsets[0] = 0;
    block_true_offsets[1] = vfes->TrueVSize();
    block_true_offsets[2] = pfes->TrueVSize();
    block_true_offsets.PartialSum();

    sol.Update(block_true_offsets);
    adj.Update(block_true_offsets);
    rhs.Update(block_true_offsets);

    SetLinearSolver();

    mfem::Operator::width=block_true_offsets[2];
    mfem::Operator::height=block_true_offsets[2];

    mu=nullptr;
    brink=nullptr;

    af=nullptr;
    mf=nullptr;

    lf=nullptr;
    pf=nullptr;


    bf=new mfem::ParMixedBilinearForm(vfes,pfes);
    bf->AddDomainIntegrator(new mfem::VectorDivergenceIntegrator());
    bf->Assemble(0);
    bf->Finalize();


    bop=new mfem::BlockOperator(block_true_offsets); bop->owns_blocks=false;

    pop=nullptr; //preconditioner
    ls=nullptr;  //linear solver
}


StokesOperator::~StokesOperator()
{
    delete bop;

    delete pop;
    delete ls;

    delete pf;
    delete lf;
    delete bf;
    delete af;

    delete pfes;
    delete pfec;
    delete vfes;
    delete vfec;
}

void StokesOperator::SetLinearSolver(mfem::real_t rtol, mfem::real_t atol, int miter)
{
    linear_rtol=rtol;
    linear_atol=atol;
    linear_iter=miter;
}

void StokesOperator::AddVelocityBC(int id, int dir, mfem::real_t val)
{
    if(dir==0){
        bcx[id]=mfem::ConstantCoefficient(val);
        AddVelocityBC(id,dir,bcx[id]);
    }
    if(dir==1){
        bcy[id]=mfem::ConstantCoefficient(val);
        AddVelocityBC(id,dir,bcy[id]);

    }
    if(dir==2){
        bcz[id]=mfem::ConstantCoefficient(val);
        AddVelocityBC(id,dir,bcz[id]);
    }
    if(dir==4){
        bcx[id]=mfem::ConstantCoefficient(val);
        bcy[id]=mfem::ConstantCoefficient(val);
        bcz[id]=mfem::ConstantCoefficient(val);
        AddVelocityBC(id,0,bcx[id]);
        AddVelocityBC(id,1,bcy[id]);
        AddVelocityBC(id,2,bcz[id]);
    }
}

void StokesOperator::DelVelocityBC()
{
    bccx.clear();
    bccy.clear();
    bccz.clear();

    bcx.clear();
    bcy.clear();
    bcz.clear();

    ess_tdofv.DeleteAll();
}

void StokesOperator::AddVelocityBC(int id, int dir, mfem::Coefficient &val)
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

void StokesOperator::SetEssTDofs(mfem::Vector& bsol, mfem::Array<int>& ess_dofs)
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
            fvelo.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsx from velocity grid function
        {
            mfem::Vector& vc=fvelo.GetTrueVector();
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
            fvelo.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsy from velocity grid function
        {
            mfem::Vector& vc=fvelo.GetTrueVector();
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
                fvelo.ProjectBdrCoefficient(pcoeff, ess_bdr);
            }

            //copy tdofsz from velocity grid function
            {
                mfem::Vector& vc=fvelo.GetTrueVector();
                for(int ii=0;ii<ess_tdofz.Size();ii++)
                {
                    bsol[ess_tdofz[ii]]=vc[ess_tdofz[ii]];
                }
            }
            ess_dofs.Append(ess_tdofz); ess_tdofz.DeleteAll();
        }
    }
}


void StokesOperator::AssemblePrec1()
{
    delete pop;
    mfem::BlockDiagonalPreconditioner* dpop=new mfem::BlockDiagonalPreconditioner(block_true_offsets);
    dpop->owns_blocks=true;

    mfem::HypreBoomerAMG* prec=new mfem::HypreBoomerAMG();
    prec->SetElasticityOptions(vfes);
    prec->SetPrintLevel(1);
    prec->SetOperator(*A);

    //dpop->SetDiagonalBlock(0,new mfem::IdentityOperator(vfes->TrueVSize()));
    dpop->SetDiagonalBlock(0,prec);
    dpop->SetDiagonalBlock(1,new mfem::IdentityOperator(pfes->TrueVSize()));
    pop=dpop;
}

void StokesOperator::AssemblePrec2()
{
    delete pop;
    mfem::BlockDiagonalPreconditioner* dpop=new mfem::BlockDiagonalPreconditioner(block_true_offsets);
    dpop->owns_blocks=true;


    mfem::HypreBoomerAMG* prec=new mfem::HypreBoomerAMG();
    prec->SetElasticityOptions(vfes);
    prec->SetPrintLevel(1);
    prec->SetOperator(*A);

    ivisc=std::unique_ptr<InverseCoeff>(new InverseCoeff(mu));
    mf=std::unique_ptr<mfem::ParBilinearForm>(new mfem::ParBilinearForm(pfes));
    mf->AddDomainIntegrator(new mfem::MassIntegrator(*ivisc));
    mf->Assemble(0);
    mf->Finalize();

    M=std::unique_ptr<mfem::HypreParMatrix>(mf->ParallelAssemble());
    mfem::HypreBoomerAMG* prem=new mfem::HypreBoomerAMG();
    prem->SetPrintLevel(1);
    prem->SetOperator(*M);

    dpop->SetDiagonalBlock(0,prec);
    dpop->SetDiagonalBlock(1,prem);

    pop=dpop;
}

void StokesOperator::AssemblePrec4()
{
    delete pop;
    mfem::BlockLowerTriangularPreconditioner* dpop=new mfem::BlockLowerTriangularPreconditioner(block_true_offsets);
    dpop->owns_blocks=false;

    prec1=std::unique_ptr<mfem::HypreBoomerAMG>(new mfem::HypreBoomerAMG());
    //prec1->SetElasticityOptions(vfes);
    prec1->SetPrintLevel(1);
    prec1->SetOperator(*A);

    ivisc=std::unique_ptr<InverseCoeff>(new InverseCoeff(mu));

    mf=std::unique_ptr<mfem::ParBilinearForm>(new mfem::ParBilinearForm(pfes));
    mf->AddDomainIntegrator(new mfem::MassIntegrator(*ivisc));
    mf->Assemble(0);
    mf->Finalize();

    M=std::unique_ptr<mfem::HypreParMatrix>(mf->ParallelAssemble());
    prec2=std::unique_ptr<mfem::HypreBoomerAMG>(new mfem::HypreBoomerAMG());
    prec2->SetPrintLevel(1);
    prec2->SetOperator(*M);

    dpop->SetBlock(1,0,B.get());
    dpop->SetDiagonalBlock(0,prec1.get());
    dpop->SetDiagonalBlock(1,prec2.get());

    pop=dpop;
}

void StokesOperator::AssemblePrec3()
{
    delete pop;
    mfem::BlockDiagonalPreconditioner* dpop=new mfem::BlockDiagonalPreconditioner(block_true_offsets);
    dpop->owns_blocks=true;

    mfem::CGSolver*  ls1=new mfem::CGSolver(pmesh->GetComm());
    ls1->SetAbsTol(linear_atol);
    ls1->SetRelTol(linear_rtol);
    ls1->SetMaxIter(100);

    prec1=std::unique_ptr<mfem::HypreBoomerAMG>(new mfem::HypreBoomerAMG());
    //set the rigid body modes
    prec1->SetElasticityOptions(vfes);
    prec1->SetPrintLevel(1);
    ls1->SetPreconditioner(*prec1);
    ls1->SetOperator(*A);
    ls1->SetPrintLevel(0);


    ivisc=std::unique_ptr<InverseCoeff>(new InverseCoeff(mu));

    mf=std::unique_ptr<mfem::ParBilinearForm>(new mfem::ParBilinearForm(pfes));
    mf->AddDomainIntegrator(new mfem::MassIntegrator(*ivisc));
    mf->Assemble(0);
    mf->Finalize();

    M=std::unique_ptr<mfem::HypreParMatrix>(mf->ParallelAssemble());

    mfem::CGSolver*  ls2=new mfem::CGSolver(pmesh->GetComm());
    ls2->SetAbsTol(linear_atol);
    ls2->SetRelTol(linear_rtol);
    ls2->SetMaxIter(100);

    prec2=std::unique_ptr<mfem::HypreBoomerAMG>(new mfem::HypreBoomerAMG());
    prec2->SetPrintLevel(1);
    ls2->SetPreconditioner(*prec2);
    ls2->SetOperator(*M);
    ls2->SetPrintLevel(0);


    dpop->SetDiagonalBlock(0,ls1);
    dpop->SetDiagonalBlock(1,ls2);

    pop=dpop;

}

void StokesOperator::AssemblePrec5()
{
    //get the diagonal of A
    mfem::HypreParVector* Ad = new mfem::HypreParVector(pmesh->GetComm(),
                                                        A->GetGlobalNumRows(),
                                                        A->GetRowStarts());
    A->GetDiag(*Ad);

    //build approximation to the Schur complement
    std::unique_ptr<mfem::HypreParMatrix> MBt(B->Transpose());
    MBt->InvScaleRows(*Ad);
    M=std::unique_ptr<mfem::HypreParMatrix>(mfem::ParMult(B.get(),MBt.get()));

    //prepare the block struture
    delete pop;
    mfem::BlockDiagonalPreconditioner* dpop=new mfem::BlockDiagonalPreconditioner(block_true_offsets);
    dpop->owns_blocks=true;


    mfem::HypreBoomerAMG* prec=new mfem::HypreBoomerAMG();
    prec->SetElasticityOptions(vfes);
    prec->SetPrintLevel(1);
    prec->SetOperator(*A);

    mfem::HypreBoomerAMG* prem=new mfem::HypreBoomerAMG();
    prem->SetPrintLevel(1);
    prem->SetOperator(*M);

    dpop->SetDiagonalBlock(0,prec);
    dpop->SetDiagonalBlock(1,prem);

    pop=dpop;

    delete Ad;

}

void StokesOperator::AssemblePrec6()
{
    M=std::unique_ptr<mfem::HypreParMatrix>(mfem::ParMult(B.get(),D.get()));
    LSC* prem=new LSC(*B,*A,*D,*M);

    mfem::HypreBoomerAMG* prec=new mfem::HypreBoomerAMG();
    prec->SetElasticityOptions(vfes);
    prec->SetPrintLevel(1);
    prec->SetOperator(*A);

    //prepare the block struture
    delete pop;
    mfem::BlockDiagonalPreconditioner* dpop=new mfem::BlockDiagonalPreconditioner(block_true_offsets);
    dpop->owns_blocks=true;

    dpop->SetDiagonalBlock(0,prec);
    dpop->SetDiagonalBlock(1,prem);

    pop=dpop;
}

void StokesOperator::AssemblePrec7()
{
    delete pop;
    mfem::BlockLowerTriangularPreconditioner* dpop=new mfem::BlockLowerTriangularPreconditioner(block_true_offsets);
    dpop->owns_blocks=false;

    prec1=std::unique_ptr<mfem::HypreBoomerAMG>(new mfem::HypreBoomerAMG());
    prec1->SetElasticityOptions(vfes);
    prec1->SetPrintLevel(1);
    prec1->SetOperator(*A);

    ivisc=std::unique_ptr<InverseCoeff>(new InverseCoeff(mu));

    mf=std::unique_ptr<mfem::ParBilinearForm>(new mfem::ParBilinearForm(pfes));
    mf->AddDomainIntegrator(new mfem::MassIntegrator(*ivisc));
    mf->Assemble(0);
    mf->Finalize();

    M=std::unique_ptr<mfem::HypreParMatrix>(mf->ParallelAssemble());
    prec2=std::unique_ptr<mfem::HypreBoomerAMG>(new mfem::HypreBoomerAMG());
    prec2->SetPrintLevel(1);
    prec2->SetOperator(*M);

    dpop->SetBlock(1,0,B.get());
    dpop->SetDiagonalBlock(0,prec1.get());
    dpop->SetDiagonalBlock(1,prec2.get());

    pop=dpop;
}


void StokesOperator::Assemble()
{
   if(mu==nullptr){return;}
   //allocate af
   if(af==nullptr){
      af=new mfem::ParBilinearForm(vfes);
      af->SetDiagonalPolicy(DIAG_ONE);
      af->AddDomainIntegrator(new mfem::ElasticityIntegrator(zeroc,*mu));
      if(brink!=nullptr){
          af->AddDomainIntegrator(new mfem::VectorMassIntegrator(*brink));
      }
   }

   //set BC
   sol=mfem::real_t(0.0);
   SetEssTDofs(sol.GetBlock(0),ess_tdofv); //set BC for the velocity

   af->Assemble(0);
   af->Finalize();
   //af->FormSystemMatrix(ess_tdofv,A);
   //std::unique_ptr<mfem::HypreParMatrix> Ael(af->ParallelAssembleElim()); Ae=std::move(Ael);
   A=std::unique_ptr<mfem::HypreParMatrix>(af->ParallelAssemble());
   Ae=std::unique_ptr<mfem::HypreParMatrix>(A->EliminateRowsCols(ess_tdofv));

   std::cout<<"H="<<A->Height()<<" W="<<A->Width()<<std::endl;

   B=std::unique_ptr<mfem::HypreParMatrix>(bf->ParallelAssemble());
   std::cout<<"H="<<B->Height()<<" W="<<B->Width()<<std::endl;

   D=std::unique_ptr<mfem::HypreParMatrix>(B->Transpose());
   std::cout<<"H="<<D->Height()<<" W="<<D->Width()<<std::endl;

   B->EliminateRows(ess_tdofp);
   Be=std::unique_ptr<mfem::HypreParMatrix>(B->EliminateCols(ess_tdofv));
   D->EliminateRows(ess_tdofv);
   De=std::unique_ptr<mfem::HypreParMatrix>(D->EliminateCols(ess_tdofp));

   bop->SetBlock(0,0,A.get());
   bop->SetBlock(0,1,D.get());
   bop->SetBlock(1,0,B.get());

   //assemble the preconditioner
   //AssemblePrec1();
   //AssemblePrec2();
   //AssemblePrec3();
   //AssemblePrec4();
   //AssemblePrec5();
   AssemblePrec6();

   if(ls==nullptr){
       ls=new mfem::MINRESSolver(pmesh->GetComm());
       //ls=new mfem::GMRESSolver(pmesh->GetComm());
       ls->SetAbsTol(linear_atol);
       ls->SetRelTol(linear_rtol);
       ls->SetMaxIter(linear_iter);

       ls->SetOperator(*bop);
       ls->SetPreconditioner(*pop);
       ls->SetPrintLevel(1);
   }else{
       ls->SetOperator(*bop);
       ls->SetPreconditioner(*pop);
   }
}

void StokesOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const
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

    //add the pressure BC
}

void StokesOperator:: MultTranspose(const mfem::Vector &x, mfem::Vector &y) const
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

    //add the pressure BC
}

void StokesOperator::FSolve()
{

    rhs=mfem::real_t(0.0);

    if(lf==nullptr){
        lf=new mfem::ParLinearForm(vfes);
        /*
        if(volforce!=nullptr){
            lf->AddDomainIntegrator(new mfem::VectorDomainLFIntegrator(*volforce));
        }
        //add surface loads
        */

    }

    (*lf)=mfem::real_t(0.0);

    lf->Assemble();
    lf->ParallelAssemble(rhs.GetBlock(0));



    Ae->AddMult(sol.GetBlock(0),rhs.GetBlock(0),-1.0);
    Be->AddMult(sol.GetBlock(0),rhs.GetBlock(1),-1.0);
    De->AddMult(sol.GetBlock(1),rhs.GetBlock(0),-1.0);

    /*
    for(int i=0;i<ess_tdofv.Size();i++){
        rhs.GetBlock(0)[ess_tdofv[i]]=sol.GetBlock(0)[ess_tdofv[i]];
    }
    */

    ls->Mult(rhs,sol);

}
