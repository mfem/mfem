#include "navier_solver_gcn.hpp"
#include "../../general/forall.hpp"
#include <fstream>
#include <iomanip>

namespace mfem {


NavierSolverGCN::NavierSolverGCN(ParMesh* mesh, int order_, std::shared_ptr<Coefficient> visc_,
                                 bool partial_assembly_, bool verbose_):
                                 pmesh(mesh), order(order_)
{
    partial_assembly=partial_assembly_;
    verbose=verbose_;

    if(order_<2){ order_=2;}
    order=order_;

    std::cout<<"My rank="<<mesh->GetMyRank()<<std::endl;

    vfec=new H1_FECollection(order, pmesh->Dimension());
    pfec=new H1_FECollection(order-1);
    vfes=new ParFiniteElementSpace(pmesh, vfec, pmesh->Dimension());
    pfes=new ParFiniteElementSpace(pmesh, pfec);

    HYPRE_BigInt gvd=vfes->GlobalTrueVSize();
    HYPRE_BigInt gpd=pfes->GlobalTrueVSize();

    if(mesh->GetMyRank()==0){
        mfem::out<<"VDOFs="<<gvd<<" PDOFs="<<gpd<<std::endl;
        mfem::out.flush();
    }

    //velocity
    cvel.reset(new ParGridFunction(vfes)); *cvel=real_t(0.0);
    nvel.reset(new ParGridFunction(vfes)); *nvel=real_t(0.0);
    pvel.reset(new ParGridFunction(vfes)); *pvel=real_t(0.0);
    //pressure
    ppres.reset(new ParGridFunction(pfes)); *ppres=real_t(0.0);
    npres.reset(new ParGridFunction(pfes)); *npres=real_t(0.0);
    cpres.reset(new ParGridFunction(pfes)); *cpres=real_t(0.0);


    nvelc.SetGridFunction(nvel.get());
    pvelc.SetGridFunction(pvel.get());
    cvelc.SetGridFunction(cvel.get());

    ppresc.SetGridFunction(ppres.get());
    npresc.SetGridFunction(npres.get());
    cpresc.SetGridFunction(cpres.get());

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

    block_true_offsets.SetSize(3);
    block_true_offsets[0] = 0;
    block_true_offsets[1] = vfes->TrueVSize();
    block_true_offsets[2] = pfes->TrueVSize();
    block_true_offsets.PartialSum();

    ess_tdofp.SetSize(0);
    ess_tdofv.SetSize(0);

}

NavierSolverGCN::~NavierSolverGCN()
{
    delete vfes;
    delete pfes;
    delete vfec;
    delete pfec;
}


void NavierSolverGCN::SetEssTDofs(real_t t, ParGridFunction& pgf, Array<int>& ess_dofs)
{

    // Set the essential boundary conditions
    ess_dofs.DeleteAll();

    Array<int> ess_tdofv_temp;

    for(auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
    {
       int attr = it->first;
       std::shared_ptr<VectorCoefficient> coeff = it->second;
       coeff->SetTime(t);
       Array<int> ess_bdr(pmesh->bdr_attributes.Max());
       ess_bdr=0;
       ess_bdr[attr-1] = 1;
       ess_tdofv_temp.DeleteAll();
       vfes->GetEssentialTrueDofs(ess_bdr,ess_tdofv_temp);
       ess_dofs.Append(ess_tdofv_temp);

       pgf.ProjectBdrCoefficient(*coeff,ess_bdr);
    }
}  

void NavierSolverGCN::SetEssTDofs(real_t t, ParGridFunction& pgf)
{


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

void NavierSolverGCN::SetEssTDofs(Array<int>& ess_dofs)
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

void NavierSolverGCN::SetupOperator(real_t t, real_t dt)
{

    if (verbose && pmesh->GetMyRank() == 0)
    {
        mfem::out << "Setup" << std::endl;
        if (partial_assembly)
        {
            mfem::out << "Using Partial Assembly" << std::endl;
        }
        else
        {
            mfem::out << "Using Full Assembly" << std::endl;
        }
    }

   // Set up boundary conditions
   SetEssTDofs(t+dt, *nvel, ess_tdofv);

   // Set up the velocity coefficients
   nvelc.SetGridFunction(nvel.get());
   pvelc.SetGridFunction(pvel.get());
   cvelc.SetGridFunction(cvel.get());

   // Set the pressure coefficients
   npresc.SetGridFunction(npres.get());
   ppresc.SetGridFunction(ppres.get());
   cpresc.SetGridFunction(cpres.get());


   
   // Set up the bilinear form for A11
   A11.reset(new ParBilinearForm(vfes));
   if(partial_assembly)
   {
       A11->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   //viscosity term
   visc->SetTime(t+dt);
   nvisc.reset(new ProductCoefficient(dt*0.5, *visc));
   A11->AddDomainIntegrator(new ElasticityIntegrator(zerocoef,*nvisc));

   //mass term
   A11->AddDomainIntegrator(new VectorMassIntegrator(onecoeff));

   //Brinkman term
   if(brink != nullptr)
   {
      brink->SetTime(t+dt);
      nbrink.reset(new ProductCoefficient(dt*0.5, *brink));
      A11->AddDomainIntegrator(new VectorMassIntegrator(*nbrink));
   }


   A11->Assemble();
   A11->Finalize();
   A11->FormSystemMatrix(ess_tdofv, A11H);

   std::cout<<"A11 Finalized"<<std::endl;

   icoeff.constant = dt*0.5;

   //off-diagonal operators
   A21.reset(new ParMixedBilinearForm(vfes, pfes));
   if(partial_assembly)
   {
       A21->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   A21->AddDomainIntegrator(new VectorDivergenceIntegrator());
   A21->Assemble();
   A21->Finalize();
   A21->FormRectangularSystemMatrix(ess_tdofv, ess_tdofp, A21H);

   std::cout<<"A21 Finalized"<<std::endl;


   A12.reset(new ParMixedBilinearForm(pfes, vfes));
   if(partial_assembly)
   {
       A12->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   A12->AddDomainIntegrator(new GradientIntegrator());
   A12->Assemble();
   A12->Finalize();
   A12->FormRectangularSystemMatrix(ess_tdofp, ess_tdofv, A12H);




   //set the block operator
   if(pmesh->GetMyRank()==0){
       std::cout<<"A11 H="<<A11H->Height()<<" W="<<A11H->Width()<<std::endl;
       std::cout<<"A12 H="<<A12H->Height()<<" W="<<A12H->Width()<<std::endl;
       std::cout<<"A21 H="<<A21H->Height()<<" W="<<A21H->Width()<<std::endl;
       std::cout.flush();
   }

   AB.reset(new BlockOperator(block_true_offsets));
   AB->SetBlock(0,0,A11H.Ptr());
   //scale the blocks A12 and A21
   AB->SetBlock(0,1,A12H.Ptr(),dt*0.5);
   AB->SetBlock(1,0,A21H.Ptr(),dt*0.5);

   //set the preconditioner
   

}

void NavierSolverGCN::SetupRHS(real_t t, real_t dt)
{
   //the RHS should be set up after the operator is set up
   //the ess_tdofv array should be set up before assembling the RHS

   rhs.SetSize(vfes->TrueVSize());
   rhs = 0.0;


}

void NavierSolverGCN::Step(real_t &time, real_t dt, int cur_step, bool provisional)
{
    //copy the current velocity to the next velocity
    nvel->SetFromTrueDofs(cvel->GetTrueVector());
    //copy the current pressure to the next pressure
    npres->SetFromTrueDofs(cpres->GetTrueVector());

    //set the operator and the preconditioners
    SetupOperator(time,dt);

    //set the RHS

    //solve

    if(provisional==false){
        UpdateHistory();
    }

}

//VectorConvectionIntegrator

void VectorConvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{

    int nd = el.GetDof();
    dim = el.GetDim();

 #ifdef MFEM_THREAD_SAFE
    DenseMatrix dshape, adjJ, Q_ir;
    Vector shape, vec2, BdFidxT;
 #endif
    elmat.SetSize(nd*dim);
    dshape.SetSize(nd,dim);
    adjJ.SetSize(dim);
    shape.SetSize(nd);
    vec2.SetSize(dim);
    BdFidxT.SetSize(nd);
    partelmat.SetSize(nd);

    Vector vec1;


    const IntegrationRule *ir = GetIntegrationRule(el, Trans);
    if (ir == NULL)
    {
       int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
       ir = &IntRules.Get(el.GetGeomType(), order);
    }

    Q->Eval(Q_ir, Trans, *ir);

    elmat = 0.0;
    partelmat=0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir->IntPoint(i);
       el.CalcDShape(ip, dshape);
       el.CalcShape(ip, shape);

       Trans.SetIntPoint(&ip);
       CalcAdjugate(Trans.Jacobian(), adjJ);
       Q_ir.GetColumnReference(i, vec1);
       vec1 *= alpha * ip.weight;

       adjJ.Mult(vec1, vec2);
       dshape.Mult(vec2, BdFidxT);

       AddMultVWt(shape, BdFidxT, partelmat);
    }

    for (int k = 0; k < dim; k++)
    {
       elmat.AddMatrix(partelmat, nd*k, nd*k);
    }

}

const IntegrationRule &VectorConvectionIntegrator::GetRule(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   const ElementTransformation &Trans)
{
   int order = Trans.OrderGrad(&trial_fe) + Trans.Order() + test_fe.GetOrder();

   return IntRules.Get(trial_fe.GetGeomType(), order);
}

const IntegrationRule &VectorConvectionIntegrator::GetRule(
   const FiniteElement &el, const ElementTransformation &Trans)
{
   return GetRule(el,el,Trans);
}

}//end namespace mfem
