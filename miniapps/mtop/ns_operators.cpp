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


    ess_tdofp.SetSize(0);
    ess_tdofv.SetSize(0);

    //set the width and the height of the operator
    Operator::width=  block_true_offsets[2];
    Operator::height= block_true_offsets[2];

    visc=visc_;

    partial_assembly=partial_assembly_;
    verbose=verbose_;
    debug=false;
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

void TimeDependentStokes::SetEssTDofsP(mfem::Array<int>& ess_dofs)
{
    // Set the essential boundary conditions
    ess_dofs.DeleteAll();

    Array<int> ess_tdofv_temp;

    for(auto it=pre_bcs.begin(); it!=pre_bcs.end(); ++it)
    {
       int attr = it->first;
       Array<int> ess_bdr(pmesh->bdr_attributes.Max());
       ess_bdr=0;
       ess_bdr[attr-1] = 1;
       ess_tdofv_temp.DeleteAll();
       pfes->GetEssentialTrueDofs(ess_bdr,ess_tdofv_temp);
       ess_dofs.Append(ess_tdofv_temp);
    }


/*
    if(pre_bcs.end()==pre_bcs.begin()){//empty list
        if(0==pfes->GetMyRank()){
            ess_dofs.Append(111);
        }
    }
*/
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
}

void  TimeDependentStokes::SetEssPBC(real_t t, ParGridFunction& pgf)
{
    for(auto it=pre_bcs.begin(); it!=pre_bcs.end(); ++it)
    {
       int attr = it->first;
       std::shared_ptr<Coefficient> coeff = it->second;
       coeff->SetTime(t);
       Array<int> ess_bdr(pmesh->bdr_attributes.Max());
       ess_bdr=0;
       ess_bdr[attr-1] = 1;

       pgf.ProjectBdrCoefficient(*coeff,ess_bdr);
    }
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
    mbf->AddDomainIntegrator(new GradientIntegrator());
    mbf->Assemble(0);
    mbf->Finalize(0);
    B.reset(mbf->ParallelAssemble());

    //set the boundary conditions
    SetEssTDofsV(ess_tdofv);
    SetEssTDofsP(ess_tdofp);
}

void TimeDependentStokes::ImplicitSolve(const real_t gamma, const Vector &x, Vector &k)
{
    real_t ctime=this->GetTime();

    A11.reset(new HypreParMatrix(*M));
    A11->Add(gamma,*K); //add viscous term
    if(nullptr!=P.get()){A11->Add(gamma,*P);} //add Brinkmann

    A12.reset(new HypreParMatrix(*B));
    (*A12)*=gamma;

    A21.reset(new HypreParMatrix(*C));
    (*A21)*=gamma;

    A22.reset(new HypreParMatrix(*H));
    (*A22)*=-1e-6;

    A11e.reset(A11->EliminateRowsCols(ess_tdofv));
    A12->EliminateRows(ess_tdofv);
    A12e.reset(A12->EliminateCols(ess_tdofp));
    A21->EliminateRows(ess_tdofp);
    A21e.reset(A21->EliminateCols(ess_tdofv));
    A22e.reset(A22->EliminateRowsCols(ess_tdofp));

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
    B->AddMult(p,rhs.GetBlock(0),-1.0);
    C->AddMult(u,rhs.GetBlock(1),-1.0);

    //solve the linear system for k
    //1. set he BC on a grid function
    SetEssVBC(ctime,vel);
    vel.SetTrueVector();
    Vector& tv=vel.GetTrueVector();

    SetEssPBC(ctime,pre);
    pre.SetTrueVector();
    Vector& tp=pre.GetTrueVector();

    // set k
    for(int i=0;i<ess_tdofv.Size();i++){
        ku[ess_tdofv[i]]=(tv[ess_tdofv[i]]-u[ess_tdofv[i]])/gamma;
    }
    for(int i=0;i<ess_tdofp.Size();i++){
        kp[ess_tdofp[i]]=(tp[ess_tdofp[i]]-p[ess_tdofp[i]])/gamma;
    }

    //eliminate the BC from RHS
    Array<int> zero_dofs;
    A12->EliminateBC(*A12e,zero_dofs,kp,rhs.GetBlock(0));
    A11->EliminateBC(*A11e,ess_tdofv,ku,rhs.GetBlock(0));
    A21->EliminateBC(*A21e,zero_dofs,ku,rhs.GetBlock(1));
    A22->EliminateBC(*A22e,ess_tdofp,kp,rhs.GetBlock(1));

    //solve the linear system


    GSDirectBlockSolver solver(A11.get(),A12.get(),A21.get(),A22.get(), 1.0,1.0,1.0,1.0);
    solver.Mult(rhs,k);

}
