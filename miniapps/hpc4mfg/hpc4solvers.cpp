#include "hpc4solvers.hpp"
#include "hpc4mat.hpp"

#include <petsc.h>


namespace mfem {

double NLDiffusionIntegrator::GetElementEnergy(const FiniteElement &el,
                                               ElementTransformation &trans,
                                               const Vector &elfun)
{
    double energy = 0.0;
    if(mat==nullptr){ return energy;}
    const int ndof = el.GetDof();
    const int ndim = el.GetDim();
    const int spaceDim = trans.GetSpaceDim();
    int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
    const IntegrationRule &ir(IntRules.Get(el.GetGeomType(), order));

    Vector shapef(ndof);
    // derivatives in isoparametric coordinates
    DenseMatrix dshape_iso(ndof, ndim);
    // derivatives in physical space
    DenseMatrix dshape_xyz(ndof, spaceDim);

    Vector uu(spaceDim+1);     //[diff_x,diff_y,diff_z,u]
    uu = 0.0;

    Vector du(uu.GetData(),spaceDim);

    double w;

    for (int i = 0; i < ir.GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir.IntPoint(i);
       trans.SetIntPoint(&ip);
       w = trans.Weight();
       w = ip.weight * w;
       el.CalcDShape(ip, dshape_iso);
       el.CalcShape(ip, shapef);
       // AdjugateJacobian = / adj(J),         if J is square
       //                    \ adj(J^t.J).J^t, otherwise
       Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
       // dshape_xyz should be divided by detJ for obtaining the real value
       // calculate the gradient
       dshape_xyz.MultTranspose(elfun, du);
       uu[spaceDim]=shapef*elfun;
       energy = energy + w * mat->Eval(trans,ip,uu);
    }
    return energy;
}


void NLDiffusionIntegrator::AssembleElementVector(const FiniteElement &el,
                                                  ElementTransformation &trans,
                                                  const Vector &elfun,
                                                  Vector &elvect)
{
    const int ndof = el.GetDof();
    const int ndim = el.GetDim();
    const int spaceDim = trans.GetSpaceDim();
    int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
    const IntegrationRule &ir(IntRules.Get(el.GetGeomType(), order));

    Vector shapef(ndof);
    DenseMatrix dshape_iso(ndof, ndim);
    DenseMatrix dshape_xyz(ndof, spaceDim);
    Vector lvec(ndof);
    elvect.SetSize(ndof);
    elvect = 0.0;

    if(mat==nullptr){return;}

    DenseMatrix B(ndof,spaceDim+1);
    Vector uu(spaceDim+1); uu=0.0;
    Vector rr(spaceDim+1); rr=0.0;
    Vector NNInput(spaceDim+1); NNInput=0.0;

    B=0.0;
    double w;

    for (int i = 0; i < ir.GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir.IntPoint(i);
       trans.SetIntPoint(&ip);
       w = trans.Weight();
       w = ip.weight * w;

       el.CalcDShape(ip, dshape_iso);
       el.CalcShape(ip, shapef);
       Mult(dshape_iso, trans.InverseJacobian(), dshape_xyz);
       // set the matrix B
       for (int jj = 0; jj < spaceDim; jj++)
       {
          B.SetCol(jj, dshape_xyz.GetColumn(jj));
       }
       B.SetCol(spaceDim, shapef);
       // calculate uu
       B.MultTranspose(elfun, uu);
       // calculate residual
       double DesingThreshold = desfieldCoeff->Eval(trans,ip);
               
       for( int Ik = 0; Ik<ndim; Ik ++){ NNInput[Ik] =  uu[Ik];}
       NNInput[ndim] = DesingThreshold;

       mat->Grad(trans,ip,NNInput,rr);
       B.Mult(rr,lvec);
       elvect.Add(w,lvec);
    }//end integration loop
}


void NLDiffusionIntegrator::AssembleElementGrad(const FiniteElement &el,
                                                ElementTransformation &trans,
                                                const Vector &elfun,
                                                DenseMatrix &elmat)
{
    const int ndof = el.GetDof();
    const int ndim = el.GetDim();
    const int spaceDim = trans.GetSpaceDim();
    int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
    const IntegrationRule &ir(IntRules.Get(el.GetGeomType(), order));

    Vector shapef(ndof);
    Vector NNInput(spaceDim+1); NNInput=0.0;
    DenseMatrix dshape_iso(ndof, ndim);
    DenseMatrix dshape_xyz(ndof, spaceDim);
    elmat.SetSize(ndof, ndof);
    elmat = 0.0;

    if(mat==nullptr){return;}

    DenseMatrix B(ndof, spaceDim+1); // [diff_x,diff_y,diff_z, shape]
    DenseMatrix A(ndof, spaceDim+1);
    Vector uu(spaceDim+1); // [diff_x,diff_y,diff_z,u]
    DenseMatrix hh(spaceDim+1);
    B = 0.0;
    uu = 0.0;
    hh = 0.0;

    double w;
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir.IntPoint(i);
       trans.SetIntPoint(&ip);
       w = trans.Weight();
       w = ip.weight * w;

       el.CalcDShape(ip, dshape_iso);
       el.CalcShape(ip, shapef);
       Mult(dshape_iso, trans.InverseJacobian(), dshape_xyz);

       // set the matrix B
       for (int jj = 0; jj < spaceDim; jj++)
       {
          B.SetCol(jj, dshape_xyz.GetColumn(jj));
       }
       B.SetCol(spaceDim, shapef);

       // calculate uu
       B.MultTranspose(elfun, uu);

       double DesingThreshold = desfieldCoeff->Eval(trans,ip);
        
       for( int Ik = 0; Ik<ndim; Ik ++){ NNInput[Ik] = uu[Ik];}
       NNInput[ndim] = DesingThreshold;

       mat->Hessian(trans,ip,NNInput,hh);

       Mult(B,hh,A);
       AddMult_a_ABt(w, A, B, elmat);
    }

}

void NLDiffusion::FSolve()
{
    ess_tdofv.DeleteAll();
    // set the boundary conditions
    {
        for(auto it=bcc.begin();it!=bcc.end();it++){
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
            ess_tdofv.Append(ess_tdof_list);
            solgf.ProjectBdrCoefficient(*(it->second),ess_bdr);
        }

        //copy BC values from the grid function to the solution vector
        {
            solgf.GetTrueDofs(rhs);
            for(int ii=0;ii<ess_tdofv.Size();ii++)
            {
                sol[ess_tdofv[ii]]=rhs[ess_tdofv[ii]];
            }
        }
    }
    //the BC are setup in the solution vector sol

    std::cout<<"BC dofs size="<<ess_tdofv.Size()<<std::endl;

    //allocate the non-linear form and add the integrators
    if(nf==nullptr){
        nf=new mfem::ParNonlinearForm(fes);
        for(size_t i=0;i<materials.size();i++){
            nf->AddDomainIntegrator(new NLDiffusionIntegrator(materials[i],desfieldCoeff[i]));
        }

        nf->AddBdrFaceIntegrator( new NLLoadIntegrator(pmesh) );
    }

    nf->SetEssentialTrueDofs(ess_tdofv);


    //allocate the preconditioner and the linear solver
    // if(prec==nullptr){
    //     prec = new HypreBoomerAMG();
    //     prec->SetPrintLevel(print_level);
    //     prec->SetMaxLevels(1);
    // }
    if(prec==nullptr){
        prec = new HypreILU();
        prec->SetPrintLevel(print_level);
        prec->SetLevelOfFill(25);
    }

    

    if(ls==nullptr){
        //ls = new CGSolver(pmesh->GetComm());
        ls =new mfem::GMRESSolver(pmesh->GetComm());
        ls->SetAbsTol(linear_atol);
        ls->SetRelTol(linear_rtol);
        ls->SetMaxIter(linear_iter);
        ls->SetPrintLevel(print_level);
        ls->SetPreconditioner(*prec);
    }

    if(ns==nullptr){
        ns = new NewtonSolver(pmesh->GetComm());
        ns->iterative_mode = true;
        ns->SetRelTol(rel_tol);
        ns->SetAbsTol(abs_tol);
        ns->SetMaxIter(max_iter);
        ns->SetPrintLevel(print_level);
        ns->SetSolver(*ls);
        ns->SetOperator(*nf);
    }

    Vector b;
    ns->Mult(b,sol);  

    // //nf->SetEssentialBC(ess_tdofv);
    // mfem::Operator & tOp = nf->GetGradient(sol);

    // std::ofstream inp("Operator.txt");
    // tOp.PrintMatlab(inp);
    // inp.close();

    solgf.SetFromTrueDofs(sol);
}


void NLDiffusion::ASolve(mfem::Vector &rhs)
{
    MFEM_ASSERT( ls != nullptr, "Liner solver does not exist"); 
    MFEM_ASSERT( nf != nullptr, "Nonlinear form does not exist"); 

    adj = 0.0;
    ess_tdofv.DeleteAll();
    // set the boundary conditions
    {
        for(auto it=bcc.begin();it!=bcc.end();it++){
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
            ess_tdofv.Append(ess_tdof_list);
           // mfem::Coefficient * coeff =new mfem::ConstantCoefficient(0.0);
            adjgf.ProjectBdrCoefficient(*(it->second),ess_bdr);
            //delete coeff;
        }

        //copy BC values from the grid function to the solution vector
        {
            //adjgf.GetTrueDofs(rhs);
            for(int ii=0;ii<ess_tdofv.Size();ii++) {
                adj[ess_tdofv[ii]]=0.0;
                rhs[ess_tdofv[ii]]=0.0;
                //adj[ess_tdofv[ii]]=rhs[ess_tdofv[ii]];
            }
        }
    }
    //rhs.Print();
    //std::cout<<"adjoint load"<<std::endl;
    nf->SetEssentialTrueDofs(ess_tdofv);

    mfem::Operator & tOp = nf->GetGradient(sol);

   //std::ofstream inp1("AdjointOperator_nonTrans.txt");     tOp.PrintMatlab(inp1);   inp1.close();

    mfem::HypreParMatrix* tTransOp = reinterpret_cast<mfem::HypreParMatrix*>(&tOp)->Transpose();

    //std::ofstream inp("AdjointOperator.txt");     tTransOp->PrintMatlab(inp);   inp.close();

    ls->SetOperator(*tTransOp);
    ls->Mult(rhs, adj);              


    delete tTransOp;
    adjgf.SetFromTrueDofs(adj);

}

// --------------------------------------------------------------------
double EnergyDissipationIntegrator::GetElementEnergy(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun)
{
    if(MicroModelCoeff==nullptr){return 0.0;}

    //integrate the dot product vel*\grad p

    //const int dof=el.GetDof();
    const int dim=el.GetDim();
    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("EnergyDissipationIntegrator::GetElementEnergy is not define on manifold meshes.");
        }
    }

    Vector uu; uu.SetSize(dim);
    Vector gradp; gradp.SetSize(dim);
    Vector NNInput; NNInput.SetSize(dim+1);
    
    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el)+preassureGF->FESpace()->GetOrder(Tr.ElementNo);
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    double w;
    double energy=0.0;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        double DesingThreshold = desfield->GetValue(Tr,ip);
        preassureGF->GetGradient(Tr,gradp);
        
        for( int Ik = 0; Ik<dim; Ik ++){ NNInput[Ik] = gradp[Ik];}
        NNInput[dim] = DesingThreshold;

        // fixme add thre
        MicroModelCoeff->Grad(Tr,ip,NNInput,uu);

        // Mult uu*gradp and add
        energy=energy+1.0*w*(uu*gradp);
    }

    return energy;

}

//the finite element space is the space of the filtered design
void EnergyDissipationIntegrator::AssembleElementVector(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun,
    Vector &elvect)
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

    Vector shapef(dof);

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el)+2*(preassureGF->FESpace()->GetOrder(Tr.ElementNo));
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    Vector uu; uu.SetSize(dim);
    Vector gradp; gradp.SetSize(dim);
    Vector NNInput; NNInput.SetSize(dim+1);

    double w;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        double DesingThreshold = desfield->GetValue(Tr,ip);
        preassureGF->GetGradient(Tr,gradp);

        for( int Ik = 0; Ik<dim; Ik ++){ NNInput[Ik] = gradp[Ik];}
        NNInput[dim] = DesingThreshold;

        // fixme add thre
        MicroModelCoeff->GradWRTDesing(Tr,ip,NNInput,uu);
        double cpl = gradp * uu;

        el.CalcShape(ip,shapef);
        elvect.Add(1.0*cpl*w,shapef);
    }
}

void EnergyDissipationIntegrator::AssembleElementGrad(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun,
    DenseMatrix &elmat)
{
    {
        mfem::mfem_error("EnergyDissipationIntegrator::AssembleElementGrad is not defined!");
    }
}

// --------------------------------------------------------------------
double EnergyDissipationIntegrator_1::GetElementEnergy(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun)
{
    {
        mfem::mfem_error("EnergyDissipationIntegrator::GetElementEnergy is not defined!");
    }

    return 0.0;

}

//the finite element space is the space of the filtered design
void EnergyDissipationIntegrator_1::AssembleRHSElementVect(
    const FiniteElement &el,
    ElementTransformation &Tr,
    Vector &elvect)
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

    Vector shapef(dof);

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el)+2*(preassureGF->FESpace()->GetOrder(Tr.ElementNo));
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    Vector uu; uu.SetSize(dim);
    Vector gradp; gradp.SetSize(dim);

    Vector NNInput(dim+1); NNInput=0.0;
    DenseMatrix dshape_iso(dof, dim);
    DenseMatrix dshape_xyz(dof, dim);

    double w;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcDShape(ip, dshape_iso);
        el.CalcShape(ip, shapef);
        Mult(dshape_iso, Tr.InverseJacobian(), dshape_xyz);

        preassureGF->GetGradient(Tr,gradp);

        double DesingThreshold = desfield->GetValue(Tr,ip);
        for( int Ik = 0; Ik<dim; Ik ++){ NNInput[Ik] = gradp[Ik];}
        NNInput[dim] = DesingThreshold;

        // fixme add thre
        MicroModelCoeff->Grad(Tr,ip,NNInput,uu);
        Vector dQdp_1(dof);
        dshape_xyz.Mult(uu, dQdp_1);
        elvect.Add(1.0*w,dQdp_1);

        //-------------------------------------------------

        DenseMatrix hh(dim+1);    hh = 0.0;
        MicroModelCoeff->Hessian(Tr,ip,NNInput,hh);
        
        // replace this with proper implementation
        DenseMatrix hh_1(dim);
        hh_1(0,0) = hh(0,0);  hh_1(1,0) = hh(1,0);  hh_1(0,1) = hh(0,1);  hh_1(1,1) = hh(1,1);

        Vector flux(dim);        Vector dQdp_2(dof);
        hh_1.Mult(gradp, flux);

        dshape_xyz.Mult(flux, dQdp_2);
        elvect.Add(1.0*w,dQdp_2);
    }
}

void EnergyDissipationIntegrator_1::AssembleElementGrad(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun,
    DenseMatrix &elmat)
{
    {
        mfem::mfem_error("EnergyDissipationIntegrator_1::AssembleElementGrad is not defined!");
    }
}

// --------------------------------------------------------------------
double AdjointPostIntegrator::GetElementEnergy(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun)
{
    {
        mfem::mfem_error("EnergyDissipationIntegrator::GetElementEnergy is not defined!");
    }

    return 0.0;

}

//the finite element space is the space of the filtered design
void AdjointPostIntegrator::AssembleRHSElementVect(
    const FiniteElement &el,
    ElementTransformation &Tr,
    Vector &elvect)
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

    Vector shapef(dof);

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el)+2*(desfield->FESpace()->GetOrder(Tr.ElementNo));
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    Vector uu; uu.SetSize(dim);
    Vector gradp; gradp.SetSize(dim);

    Vector NNInput(dim+1); NNInput=0.0;
    DenseMatrix dshape_iso(dof, dim);
    DenseMatrix dshape_xyz(dof, dim);

    double w;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcDShape(ip, dshape_iso);
        el.CalcShape(ip, shapef);
        Mult(dshape_iso, Tr.InverseJacobian(), dshape_xyz);

        preassureGF->GetGradient(Tr,gradp);

        double DesingThreshold = desfield->GetValue(Tr,ip);
        for( int Ik = 0; Ik<dim; Ik ++){ NNInput[Ik] = gradp[Ik];}
        NNInput[dim] = DesingThreshold;

        MicroModelCoeff->GradWRTDesing(Tr,ip,NNInput,uu);
        // double AdjointVal = AdjointGF->GetValue(Tr,ip);

        // Vector dQdp_1(dof);
        // dshape_xyz.Mult(uu, dQdp_1);

        // elvect.Add(w*AdjointVal,dQdp_1);

       //mfem::Vector nodevec_adjoint; nodevec_adjoint = 0.0;

        //mfem::Array< int > Vertexvdofs;
        //AdjointGF->ParFESpace()->GetElementVDofs(Tr.ElementNo,Vertexvdofs);
        //AdjointGF->GetSubVector(Vertexvdofs, nodevec_adjoint);

        Vector AdjointGrad(dim);
        AdjointGF->GetGradient(Tr,AdjointGrad);
        //dshape_xyz.MultTranspose(nodevec_adjoint, dQdp_1);

        double ScalarVal = AdjointGrad*uu;

        elvect.Add(w*ScalarVal,shapef);
    }
}

void AdjointPostIntegrator::AssembleElementGrad(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun,
    DenseMatrix &elmat)
{
    {
        mfem::mfem_error("EnergyDissipationIntegrator_1::AssembleElementGrad is not defined!");
    }
}



// --------------------------------------------------------------------
double MicrostructureVolIntegrator::GetElementEnergy(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun)
{
    //const int dof=el.GetDof();
    const int dim=el.GetDim();
    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("EnergyDissipationIntegrator::GetElementEnergy is not define on manifold meshes.");
        }
    }
    
    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el)+desfield->FESpace()->GetOrder(Tr.ElementNo);
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    double w;
    double energy=0.0;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        double DesingThreshold = desfield->GetValue(Tr,ip);
        //double val = 1.0-M_PI * DesingThreshold*DesingThreshold;
        //double val = 1.0-M_PI * DesingThreshold*DesingThreshold*0.5;
        double val = 1.0-DesingThreshold;

        energy=energy+w*(val);
    }

    return energy;

}

//the finite element space is the space of the filtered design
void MicrostructureVolIntegrator::AssembleElementVector(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun,
    Vector &elvect)
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
    Vector shapef(dof);

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el)+2*(desfield->FESpace()->GetOrder(Tr.ElementNo));
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    double w;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        double DesingThreshold = desfield->GetValue(Tr,ip);
        //double val = -2.0* M_PI * DesingThreshold;
        //double val = -1.0* M_PI * DesingThreshold;
        double val = -1.0;

        el.CalcShape(ip,shapef);
        elvect.Add(val*w,shapef);
    }
}

void MicrostructureVolIntegrator::AssembleElementGrad(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun,
    DenseMatrix &elmat)
{
    {
        mfem::mfem_error("ComplianceNLIntegrator::AssembleElementGrad is not defined!");
    }
}

double EnergyDissipationObjective::Eval(mfem::ParGridFunction& sol)
{
    if(preassureGF==nullptr){
        MFEM_ABORT("preassureGF in EnergyDissipationObjective should be set before calling the Eval method!");
    }
    if(MicroModelCoeff==nullptr){
        MFEM_ABORT("MicroModelCoeff in EnergyDissipationObjective should be set before calling the Eval method!");
    }

    if(desfield==nullptr){
        MFEM_ABORT("desfield of dfes in EnergyDissipationObjective should be set before calling the Eval method!");
    }

    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new EnergyDissipationIntegrator();
        nf->AddDomainIntegrator(intgr);
    }

    intgr->SetPreassure(preassureGF);
    intgr->SetNLDiffusionCoeff(MicroModelCoeff);
    intgr->SetDesingField(desfield);

    double rt=nf->GetEnergy(*desfield);

    return rt;

}

double EnergyDissipationObjective::Eval()
{
    if(preassureGF==nullptr){
        MFEM_ABORT("preassureGF in EnergyDissipationObjective should be set before calling the Eval method!");
    }

    if(MicroModelCoeff==nullptr){
        MFEM_ABORT("MicroModelCoeff in EnergyDissipationObjective should be set before calling the Eval method!");
    }

    if(desfield==nullptr){
        MFEM_ABORT("fsolv of dfes in EnergyDissipationObjective should be set before calling the Eval method!");
    }


    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new EnergyDissipationIntegrator();
        nf->AddDomainIntegrator(intgr);
    }

    intgr->SetPreassure(preassureGF);
    intgr->SetNLDiffusionCoeff(MicroModelCoeff);
    intgr->SetDesingField(desfield);

    double rt=nf->GetEnergy(*desfield);

    return rt;
}

void EnergyDissipationObjective::Grad(
    mfem::ParGridFunction& sol,
    Vector& grad)
{
    if(preassureGF==nullptr){
        MFEM_ABORT("preassureGF in EnergyDissipationObjective should be set before calling the Grad method!");
    }
    if(desfield==nullptr){
        MFEM_ABORT("fsolv or dfes in EnergyDissipationObjective should be set before calling the Grad method!");
    }
    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new EnergyDissipationIntegrator();
        nf->AddDomainIntegrator(intgr);
    }
    intgr->SetPreassure(preassureGF);
    intgr->SetNLDiffusionCoeff(MicroModelCoeff);
    intgr->SetDesingField(desfield);
    nf->Mult(*desfield,grad);
}

void EnergyDissipationObjective::Grad(Vector& grad)
{
    if(preassureGF==nullptr){
        MFEM_ABORT("preassureGF in EnergyDissipationObjective should be set before calling the Grad method!");
    }

    if(desfield==nullptr){
        MFEM_ABORT("fsolv or dfes in EnergyDissipationObjective should be set before calling the Grad method!");
    }

    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new EnergyDissipationIntegrator();
        nf->AddDomainIntegrator(intgr);
    }
    intgr->SetPreassure(preassureGF);
    intgr->SetNLDiffusionCoeff(MicroModelCoeff);
    intgr->SetDesingField(desfield);

    nf->Mult(*desfield,grad);
}


//--------------------------------------------------------------------------------------

double VolumeQoI::Eval(mfem::ParGridFunction& sol)
{
    if(desfield==nullptr){
        MFEM_ABORT("desfield of dfes in VolumeQoI should be set before calling the Eval method!");
    }

    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new MicrostructureVolIntegrator();
        nf->AddDomainIntegrator(intgr);
    }

    intgr->SetDesingField(desfield);

    double rt=nf->GetEnergy(*desfield);

    return rt;

}

double VolumeQoI::Eval()
{
    if(desfield==nullptr){
        MFEM_ABORT("fsolv of dfes in VolumeQoI should be set before calling the Eval method!");
    }


    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new MicrostructureVolIntegrator();
        nf->AddDomainIntegrator(intgr);
    }

    intgr->SetDesingField(desfield);

    double rt=nf->GetEnergy(*desfield);

    return rt;
}

void VolumeQoI::Grad(mfem::ParGridFunction& sol, Vector& grad)
{
    if(desfield==nullptr){
        MFEM_ABORT("fsolv or dfes in VolumeQoI should be set before calling the Grad method!");
    }
    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new MicrostructureVolIntegrator();
        nf->AddDomainIntegrator(intgr);
    }

    intgr->SetDesingField(desfield);
    nf->Mult(*desfield,grad);
}

void VolumeQoI::Grad(Vector& grad)
{
    if(desfield==nullptr){
        MFEM_ABORT("fsolv or dfes in VolumeQoI should be set before calling the Grad method!");
    }

    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new MicrostructureVolIntegrator();
        nf->AddDomainIntegrator(intgr);
    }

    intgr->SetDesingField(desfield);

    nf->Mult(*desfield,grad);
}

}
