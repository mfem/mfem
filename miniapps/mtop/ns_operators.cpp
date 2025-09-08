#include "ns_operators.hpp"

using namespace mfem;

TimeDependentStokes::TimeDependentStokes(ParMesh* mesh, int order_, std::shared_ptr<Coefficient> visc_,
                                         bool partial_assembly_, bool verbose_)
                    :TimeDependentOperator(0,0,0,TimeDependentOperator::Type::IMPLICIT)
{
    pmesh=mesh;

    if(order_<2){ order_=2;}
    order=order_;

    vfec=new H1_FECollection(order, pmesh->Dimension());
    pfec=new H1_FECollection(order-1);
    vfes=new ParFiniteElementSpace(pmesh, vfec, pmesh->Dimension());
    pfes=new ParFiniteElementSpace(pmesh, pfec);

    vel.SetSpace(vfes); vel=0.0;
    pre.SetSpace(pfes); pre=0.0;

    brink.reset();
    if (visc_ != nullptr)
    {
       visc = visc_;
    }
    else
    {
       visc.reset(new ConstantCoefficient(1.0));
    }

    onecoeff.constant = 1.0;
    zerocoef.constant = 0.0;

    siz_u=vfes->TrueVSize();
    siz_p=pfes->TrueVSize();

    block_true_offsets.SetSize(3);
    block_true_offsets[0] = 0;
    block_true_offsets[1] = siz_u;
    block_true_offsets[2] = siz_p;
    block_true_offsets.PartialSum();

    rhs.Update(block_true_offsets); rhs=0.0;

    ess_tdofv.SetSize(0);

    //set the width and the height of the operator
    Operator::width=  block_true_offsets[2];
    Operator::height= block_true_offsets[2];

    partial_assembly=partial_assembly_;
    verbose=verbose_;
    debug=false;

    zero_mean_pres=true;
}

TimeDependentStokes::~TimeDependentStokes()
{
    delete pfes;
    delete vfes;
    delete pfec;
    delete vfec;
}

void TimeDependentStokes::SetEssTDofsV(mfem::Array<int>& ess_dofs)
{
    // Set the essential boundary conditions
    ess_dofs.DeleteAll();

    Array<int> ess_tdofv_temp;

    for(auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
    {
       int attr = it->first;
       Array<int> ess_bdr(pmesh->bdr_attributes.Max());
       ess_bdr=0;
       ess_bdr[attr-1] = 1;
       ess_tdofv_temp.DeleteAll();
       vfes->GetEssentialTrueDofs(ess_bdr,ess_tdofv_temp);
       ess_dofs.Append(ess_tdofv_temp);
    }
}

void TimeDependentStokes::SetEssVBC(real_t t, ParGridFunction& pgf)
{

    std::cout<<"Set VBC time="<<t<<std::endl;


    for(auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
    {
       int attr = it->first;
       std::shared_ptr<VectorCoefficient> coeff = it->second;
       coeff->SetTime(t);
       Array<int> ess_bdr(pmesh->bdr_attributes.Max());
       ess_bdr=0;
       ess_bdr[attr-1] = 1;

       pgf.ProjectBdrCoefficient(*coeff,ess_bdr);
    }
    std::cout<<"End Set VBC time="<<t<<std::endl;
}

void TimeDependentStokes::Assemble()
{
    std::unique_ptr<ParBilinearForm> bf;
    //assemble mass matrix
    bf.reset(new ParBilinearForm(vfes));
    bf->AddDomainIntegrator(new VectorMassIntegrator(onecoeff));
    bf->Assemble(0);
    bf->Finalize(0);
    M.reset(bf->ParallelAssemble());

    //assemble diagonal mass matrix for the pressure block
    bf.reset(new ParBilinearForm(pfes));
    bf->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator(onecoeff)));
    bf->Assemble(0);
    bf->Finalize(0);
    H.reset(bf->ParallelAssemble());

    //assemble viscous term
    bf.reset(new ParBilinearForm(vfes));
    bf->AddDomainIntegrator(new ElasticityIntegrator(zerocoef,*visc));
    bf->Assemble(0);
    bf->Finalize(0);
    K.reset(bf->ParallelAssemble());

    if(nullptr!=brink){
        //assemble Brinkmann term
        bf.reset(new ParBilinearForm(vfes));
        bf->AddDomainIntegrator(new VectorMassIntegrator(*brink));
        bf->Assemble(0);
        bf->Finalize(0);
        P.reset(bf->ParallelAssemble());
    }

    std::unique_ptr<ParMixedBilinearForm> mbf;
    mbf.reset(new ParMixedBilinearForm(vfes, pfes));
    mbf->AddDomainIntegrator(new VectorDivergenceIntegrator());
    mbf->Assemble(0);
    mbf->Finalize(0);
    C.reset(mbf->ParallelAssemble());

    mbf.reset(new ParMixedBilinearForm(pfes, vfes));
    //mbf->AddDomainIntegrator(new GradientIntegrator());
    mbf->AddDomainIntegrator(new TransposeIntegrator(new VectorDivergenceIntegrator()));
    mbf->Assemble(0);
    mbf->Finalize(0);
    B.reset(mbf->ParallelAssemble());

    //set the boundary conditions
    SetEssTDofsV(ess_tdofv);
}

void TimeDependentStokes::ImplicitSolve(const real_t gamma, const Vector &x, Vector &k)
{
    real_t ctime=this->GetTime();

    A11.reset(new HypreParMatrix(*M));
    A11->Add(gamma,*K); //add viscous term
    if(nullptr!=P.get()){A11->Add(gamma,*P);} //add Brinkmann

    A12.reset(new HypreParMatrix(*B));
    (*A12)*=-gamma;

    A21.reset(new HypreParMatrix(*C));
    (*A21)*=-gamma;

    A22.reset(new HypreParMatrix(*H));
    //(*A22)*=-1e-6;
    (*A22)*=0.0;

    A11e.reset(A11->EliminateRowsCols(ess_tdofv));
    A12->EliminateRows(ess_tdofv);
    A21e.reset(A21->EliminateCols(ess_tdofv));

    //form the rhs
    Vector u(x.GetData()+0, siz_u);
    Vector p(x.GetData()+ siz_u,siz_p);

    //form the solution
    k=0.0;
    Vector ku(k.GetData()+0, siz_u);
    Vector kp(k.GetData()+siz_u,siz_p);

    //set the force
    if(nullptr!=vol_force.get()){
        std::unique_ptr<ParLinearForm> lf;
        lf.reset(new ParLinearForm(vfes));
        vol_force->SetTime(ctime);
        lf->AddDomainIntegrator(new  VectorDomainLFIntegrator(*vol_force));
        lf->Assemble();
        lf->ParallelAssemble(rhs.GetBlock(0));
    }else{
        rhs.GetBlock(0)=0.0;
    }
    rhs.GetBlock(1)=0.0;

    //add velocity and pressure contributions
    K->AddMult(u,rhs.GetBlock(0),-1.0);
    if(nullptr!=P.get()){P->AddMult(u,rhs.GetBlock(0),-1.0);}
    B->AddMult(p,rhs.GetBlock(0),1.0);
    C->AddMult(u,rhs.GetBlock(1),1.0);

    //solve the linear system for k
    //1. set he BC on a grid function
    SetEssVBC(ctime,vel);
    vel.SetTrueVector();
    Vector& tv=vel.GetTrueVector();

    // set k
    for(int i=0;i<ess_tdofv.Size();i++){
        ku[ess_tdofv[i]]=(tv[ess_tdofv[i]]-u[ess_tdofv[i]])/gamma;
    }

    //eliminate the BC from RHS
    Array<int> zero_dofs;
    //A12->EliminateBC(*A12e,zero_dofs,kp,rhs.GetBlock(0));
    //A12e->Mult(-1.0,kp,1.0,rhs.GetBlock(0));
    A11->EliminateBC(*A11e,ess_tdofv,ku,rhs.GetBlock(0));
    //A21->EliminateBC(*A21e,zero_dofs,ku,rhs.GetBlock(1));
    A21e->Mult(-1.0,ku,1.0,rhs.GetBlock(1));
    //A22->EliminateBC(*A22e,ess_tdofp,kp,rhs.GetBlock(1));

    //solve the linear system

    //GSDirectBlockSolver solver(A11.get(),A12.get(),A21.get(),A22.get(), 1.0,1.0,1.0,1.0);
    //solver.Mult(rhs,k);
    BlockOperator bop(block_true_offsets);
    bop.SetBlock(0,0,A11.get()); bop.SetBlock(0,1,A12.get());
    bop.SetBlock(1,0,A21.get()); bop.SetBlock(1,1,A22.get());
    //allocate the preconditioner
    /*
    BlockDiagTSPrec prec(gamma, visc, vfes,pfes,ess_tdofv,ess_tdofp);
    prec.SetMaxIter(40);
    prec.SetRelTol(1e-4);
    prec.SetAbsTol(1e-14);
    prec.SetPrintLevel(0);
    */


    HypreParVector S(vfes->GetComm(), M->GetGlobalNumRows(), M->GetRowStarts());
    M->GetDiag(S); S*=gamma;



    //DLSCPrec prec(A11.get(),A21.get(),A12.get(), ess_tdofp,&S,vfes->GetVDim());
    DLSCPrec prec(A11.get(),A21.get(),A12.get(), zero_mean_pres, nullptr, vfes->GetVDim());
    prec.SetMaxIter(10);
    prec.SetRelTol(1e-10);
    prec.SetAbsTol(1e-14);
    prec.SetPrintLevel(-1);


    MINRESSolver mr(vfes->GetComm());
    mr.iterative_mode=true;
    mr.SetOperator(bop);
    mr.SetPreconditioner(prec);
    mr.SetAbsTol(1e-8);
    mr.SetRelTol(1e-8);
    mr.SetMaxIter(2);
    mr.SetPrintLevel(0);
    mr.Mult(rhs,k);



}


StokesSolver::StokesSolver(ParMesh* mesh, int order_,bool partial_assembly_, bool verbose_):
    partial_assembly(partial_assembly_),
    verbose(verbose_),
    pmesh(mesh),
    order(order_)
{
    if(order_<2){ order=2;}

    vfec=new H1_FECollection(order, pmesh->Dimension());
    pfec=new H1_FECollection(order-1);
    vfes=new ParFiniteElementSpace(pmesh, vfec, pmesh->Dimension());
    pfes=new ParFiniteElementSpace(pmesh, pfec);

    vel.SetSpace(vfes); vel=0.0;
    pre.SetSpace(pfes); pre=0.0;

    brink.reset();
    visc.reset(new ConstantCoefficient(1.0));

    onecoeff.constant = 1.0;
    zerocoef.constant = 0.0;

    siz_u=vfes->TrueVSize();
    siz_p=pfes->TrueVSize();

    block_true_offsets.SetSize(3);
    block_true_offsets[0] = 0;
    block_true_offsets[1] = siz_u;
    block_true_offsets[2] = siz_p;
    block_true_offsets.PartialSum();

    sol.Update(block_true_offsets); sol=0.0;
    rhs.Update(block_true_offsets); rhs=0.0;

    ess_tdofp.SetSize(0);
    ess_tdofv.SetSize(0);

    //set the width and the height of the operator
    Operator::width=  block_true_offsets[2];
    Operator::height= block_true_offsets[2];

    bf11.reset();
    bf22.reset();
    bf12.reset();
    bf21.reset();
}

StokesSolver::~StokesSolver()
{
    delete pfes;
    delete vfes;
    delete pfec;
    delete vfec;
}

void StokesSolver::SetEssTDofsV(mfem::Array<int>& ess_dofs)
{
    // Set the essential boundary conditions
    ess_dofs.DeleteAll();

    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr=0;
    for(auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
    {
        int attr = it->first;
        ess_bdr[attr-1] = 1;
    }
    vfes->GetEssentialTrueDofs(ess_bdr,ess_dofs);
}

void StokesSolver::SetEssTDofsV(Vector& v)
{
    for(auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
    {
       int attr = it->first;
       std::shared_ptr<VectorCoefficient> coeff = it->second;
       coeff->SetTime(real_t(0.0));
       Array<int> ess_bdr(pmesh->bdr_attributes.Max());
       ess_bdr=0;
       ess_bdr[attr-1] = 1;

       mfem::Array<int> loc_tdofs;
       vfes->GetEssentialTrueDofs(ess_bdr,loc_tdofs);
       vel.ProjectBdrCoefficient(*coeff,ess_bdr);
       vel.SetTrueVector();

       // copy values to v
       const Vector &tvel=vel.GetTrueVector();
       for(int j=0;j<ess_bdr.Size();j++){
           v[ess_bdr[j]]=tvel[ess_bdr[j]];
       }
    }
}

void StokesSolver::SetEssTDofsP(mfem::Array<int>& ess_dofs)
{
    // Set the essential boundary conditions
    ess_dofs.DeleteAll();
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr=0;
    for(auto it=pre_bcs.begin(); it!=pre_bcs.end(); ++it)
    {
       int attr = it->first;
       ess_bdr[attr-1] = 1;
    }
    pfes->GetEssentialTrueDofs(ess_bdr,ess_dofs);
}

void StokesSolver::SetEssTDofsP(Vector& p)
{
    for(auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
    {
       int attr = it->first;
       std::shared_ptr<VectorCoefficient> coeff = it->second;
       coeff->SetTime(real_t(0.0));
       Array<int> ess_bdr(pmesh->bdr_attributes.Max());
       ess_bdr=0;
       ess_bdr[attr-1] = 1;

       mfem::Array<int> loc_tdofs;
       pfes->GetEssentialTrueDofs(ess_bdr,loc_tdofs);
       pre.ProjectBdrCoefficient(*coeff,ess_bdr);
       pre.SetTrueVector();

       // copy values to p
       const Vector &tpre=pre.GetTrueVector();
       for(int j=0;j<ess_bdr.Size();j++){
           p[ess_bdr[j]]=tpre[ess_bdr[j]];
       }
    }
}




void StokesSolver::SetEssVBC(ParGridFunction& pgf)
{

    for(auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
    {
       int attr = it->first;
       std::shared_ptr<VectorCoefficient> coeff = it->second;
       coeff->SetTime(real_t(0.0));
       Array<int> ess_bdr(pmesh->bdr_attributes.Max());
       ess_bdr=0;
       ess_bdr[attr-1] = 1;

       pgf.ProjectBdrCoefficient(*coeff,ess_bdr);
    }
}

void  StokesSolver::SetEssPBC(ParGridFunction& pgf)
{
    for(auto it=pre_bcs.begin(); it!=pre_bcs.end(); ++it)
    {
       int attr = it->first;
       std::shared_ptr<Coefficient> coeff = it->second;
       coeff->SetTime(real_t(0.0));
       Array<int> ess_bdr(pmesh->bdr_attributes.Max());
       ess_bdr=0;
       ess_bdr[attr-1] = 1;

       pgf.ProjectBdrCoefficient(*coeff,ess_bdr);
    }
}

void StokesSolver::DeleteBC()
{
    vel_bcs.clear();
    pre_bcs.clear();
    ess_tdofv.DeleteAll();
    ess_tdofp.DeleteAll();

    //delete allocated matrices and forms
}

void StokesSolver::Assemble()
{
    //set BC
    vel=real_t(0.0);
    pre=real_t(0.0);
    SetEssVBC(vel);
    SetEssPBC(pre);
    SetEssTDofsV(ess_tdofv);
    SetEssTDofsP(ess_tdofp);

    //assemble block 11
    bf11.reset(new ParBilinearForm(vfes));
    bf11->AddDomainIntegrator(new ElasticityIntegrator(zerocoef,*visc));
    if(nullptr!=brink.get()){
        bf11->AddDomainIntegrator(new VectorMassIntegrator(*brink));
    }
    bf11->Assemble(0);
    bf11->Finalize(0);
    A11.reset(bf11->ParallelAssemble());

    //assemble block 12
    bf12.reset(new ParMixedBilinearForm(pfes, vfes));
    //bf12->AddDomainIntegrator(new GradientIntegrator());
    bf12->AddDomainIntegrator(new TransposeIntegrator(new VectorDivergenceIntegrator()));
    bf12->Assemble(0);
    bf12->Finalize(0);
    A12.reset(bf12->ParallelAssemble());

    //assemble block 21
    bf21.reset(new ParMixedBilinearForm(vfes, pfes));
    bf21->AddDomainIntegrator(new VectorDivergenceIntegrator());
    bf21->Assemble(0);
    bf21->Finalize(0);
    A21.reset(bf21->ParallelAssemble());

    //assemble block 22
    bf22.reset(new ParBilinearForm(pfes));
    bf22->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator(onecoeff)));
    bf22->Assemble(0);
    bf22->Finalize(0);
    A22.reset(bf22->ParallelAssemble()); (*A22)*=real_t(0.0);

    //set BC to the operators
    A11e.reset(A11->EliminateRowsCols(ess_tdofv));
    A12->EliminateRows(ess_tdofv);
    A12e.reset(A12->EliminateCols(ess_tdofp));
    A21->EliminateRows(ess_tdofp);
    A21e.reset(A21->EliminateCols(ess_tdofv));
    A22e.reset(A22->EliminateRowsCols(ess_tdofp));

    //set the block operator
    bop.reset(new BlockOperator(block_true_offsets));
    bop->SetBlock(0,0,A11.get());
    bop->SetBlock(0,1,A12.get());
    bop->SetBlock(1,0,A21.get());
    bop->SetBlock(1,1,A22.get());

    //set the solver
    MINRESSolver* minres=new MINRESSolver(pmesh->GetComm());
    minres->SetRelTol(linear_rtol);
    minres->SetAbsTol(linear_atol);
    minres->SetMaxIter(linear_iter);
    minres->SetOperator(*bop);

    //set the preconditioner
    prec.reset(new GSDirectBlockSolver(A11.get(),A12.get(),A21.get(),A22.get(),1.0,1.0,1.0,1.0));
    minres->SetPreconditioner(*prec);

    ls.reset(minres);

}

void StokesSolver::FSolve()
{
    Vector& vsol=sol.GetBlock(0);
    Vector& psol=sol.GetBlock(1);

    Vector& vrhs=rhs.GetBlock(0);
    Vector& prhs=rhs.GetBlock(1);

    //assemble the RHS
    prhs=0.0;
    if(nullptr!=vol_force.get())
    {
        ParLinearForm lf(vfes);
        lf.AddDomainIntegrator(new VectorDomainLFIntegrator(*vol_force));
        lf.Assemble();
        lf.ParallelAssemble(vrhs);
    }else{
        vrhs=0.0;
    }

    //set the BCs
    SetEssTDofsV(vsol);
    SetEssTDofsP(psol);

    //modify the rhs
    A12e->Mult(-1.0,psol,1.0,vrhs);
    A21e->Mult(-1.0,vsol,1.0,prhs);
    A11->EliminateBC(*A11e,ess_tdofv,vsol,vrhs);
    A22->EliminateBC(*A22e,ess_tdofp,psol,prhs);

    //solve the linear system
    ls->Mult(rhs,sol);
}

void StokesSolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
{

}

void StokesSolver::MultTranspose(const mfem::Vector &x, mfem::Vector &y) const
{

}
