#include "ThermalComplianceQoI.hpp"


namespace mfem {

    // --------------------------------------------------------------------
double ThermalComplianceIntegrator::GetElementEnergy(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun)
{
    if(MicroModelCoeff==nullptr){return 0.0;}

    //integrate the dot product gradTT^T \kappa GradT

    //const int dof=el.GetDof();
    const int dim=el.GetDim();
    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("ThermalComplianceIntegrator::GetElementEnergy is not define on manifold meshes.");
        }
    }

    DenseMatrix Kappa; Kappa.SetSize(dim);
    Vector gradp; gradp.SetSize(dim);
    Vector gradT; gradT.SetSize(dim);
    Vector tempvec;  tempvec.SetSize(dim);
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

        double DesingThreshold = designGF->GetValue(Tr,ip);
        preassureGF->GetGradient(Tr,gradp);
        tempGF     ->GetGradient(Tr,gradT);
        
        for( int Ik = 0; Ik<dim; Ik ++){ NNInput[Ik] = gradp[Ik];}
        NNInput[dim] = DesingThreshold;

        // fixme add thre
        MicroModelCoeff->Hessian(Tr,ip,NNInput,Kappa);

        tempvec = 0.0;

        Kappa.Mult(gradT ,tempvec);

        // Mult gradTT^T \kappa GradT
        energy=energy+w*(gradT*tempvec);
    }

    return energy;

}

//the finite element space is the space of the filtered design
void ThermalComplianceIntegrator::AssembleElementVector(
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


    DenseMatrix KappaGrad; KappaGrad.SetSize(dim);
    Vector gradp; gradp.SetSize(dim);
    Vector gradT; gradT.SetSize(dim);
    Vector tempvec;  tempvec.SetSize(dim);
    Vector NNInput; NNInput.SetSize(dim+1);

    double w;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        double DesingThreshold = designGF->GetValue(Tr,ip);
        preassureGF->GetGradient(Tr,gradp);
        tempGF     ->GetGradient(Tr,gradT);
        
        for( int Ik = 0; Ik<dim; Ik ++){ NNInput[Ik] = gradp[Ik];}
        NNInput[dim] = DesingThreshold;

        // fixme add thre
        MicroModelCoeff->GradWRTDesing(Tr,ip,NNInput,KappaGrad);

        tempvec = 0.0;
        KappaGrad.Mult(gradT ,tempvec);

        // Mult gradTT^T \kappa GradT
        double cpl =(gradT*tempvec);

        el.CalcShape(ip,shapef);
        elvect.Add(1.0*cpl*w,shapef);
    }
}

void ThermalComplianceIntegrator_1::AssembleRHSElementVect(
    const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
    const int dof=el.GetDof();
    const int dim=el.GetDim();
    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("ThermalComplianceIntegrator_1::AssembleElementVector is not define on manifold meshes.");
        }
    }

    elvect.SetSize(dof); elvect=0.0;

    Vector shapef(dof);
    DenseMatrix dshape_iso(dof, dim);
    DenseMatrix dshape_xyz(dof, dim);

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el)+2*(preassureGF->FESpace()->GetOrder(Tr.ElementNo));
    ir=&IntRules.Get(Tr.GetGeometryType(),order);


    DenseMatrix KappaGrad; KappaGrad.SetSize(dim);
    Vector gradp; gradp.SetSize(dim);
    Vector gradT; gradT.SetSize(dim);
    Vector tempvec;  tempvec.SetSize(dim);
    Vector NNInput; NNInput.SetSize(dim+1);

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

        double DesingThreshold = designGF->GetValue(Tr,ip);
        preassureGF->GetGradient(Tr,gradp);
        tempGF     ->GetGradient(Tr,gradT);
        
        for( int Ik = 0; Ik<dim; Ik ++){ NNInput[Ik] = gradp[Ik];}
        NNInput[dim] = DesingThreshold;

        // fixme add thre
        MicroModelCoeff->GradWRTDesing(Tr,ip,NNInput,KappaGrad);

        tempvec = 0.0;
        KappaGrad.Mult(gradT ,tempvec);

        Vector dQdT(dof);
        dshape_xyz.Mult(tempvec, dQdT);
        elvect.Add(2.0*w,dQdT);                              //2.0 because of gradT^T NN gradT
    }
}

double MeanTempIntegrator::GetElementEnergy(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun)
{
    if(MicroModelCoeff==nullptr){return 0.0;}

    //integrate the dot product gradTT^T \kappa GradT

    //const int dof=el.GetDof();
    const int dim=el.GetDim();
    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("ThermalComplianceIntegrator::GetElementEnergy is not define on manifold meshes.");
        }
    }
    
    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el)+tempGF->FESpace()->GetOrder(Tr.ElementNo);
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    double w;
    double energy=0.0;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        double Temp = tempGF->GetValue (Tr,ip);

        // Mult gradTT^T \kappa GradT
        energy=energy+w*(Temp);
    }

    return energy;

}

void MeanTempIntegrator::AssembleElementVector(
    const FiniteElement &el,
    ElementTransformation &Tr,
    const Vector &elfun,
    Vector &elvect)
{
    mfem::mfem_error("ThermalComplianceIntegrator_1::AssembleElementVector dont't go in here.");
    const int dof=el.GetDof();
    const int dim=el.GetDim();
    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("ThermalComplianceIntegrator_1::AssembleElementVector is not define on manifold meshes.");
        }
    }

    elvect.SetSize(dof); elvect=0.0;

    Vector shapef(dof);

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el)+2*(tempGF->FESpace()->GetOrder(Tr.ElementNo));
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    double w;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcShape(ip, shapef);

        elvect.Add(1.0*w,shapef);   
    }
}

void MeanTempIntegrator_1::AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
    const int dof=el.GetDof();
    const int dim=el.GetDim();
    {
        const int spaceDim=Tr.GetDimension();
        if(dim!=spaceDim)
        {
            mfem::mfem_error("ThermalComplianceIntegrator_1::AssembleElementVector is not define on manifold meshes.");
        }
    }

    elvect.SetSize(dof); elvect=0.0;

    Vector shapef(dof);

    const IntegrationRule *ir = nullptr;
    int order= 2 * el.GetOrder() + Tr.OrderGrad(&el)+2*(tempGF->FESpace()->GetOrder(Tr.ElementNo));
    ir=&IntRules.Get(Tr.GetGeometryType(),order);

    double w;
    for(int i=0; i<ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el.CalcShape(ip, shapef);

        elvect.Add(1.0*w,shapef);   
    }
}

void AdvDiffAdjointPostIntegrator::AssembleRHSElementVect(
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

    DenseMatrix KappaGrad; KappaGrad.SetSize(dim);
    Vector gradp; gradp.SetSize(dim);
    Vector gradT; gradT.SetSize(dim);
    Vector uu; uu.SetSize(dim);
    Vector tempvec;  tempvec.SetSize(dim);

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
        tempGF     ->GetGradient(Tr,gradT);

        double DesingThreshold = desfield->GetValue(Tr,ip);
        for( int Ik = 0; Ik<dim; Ik ++){ NNInput[Ik] = gradp[Ik];}
        NNInput[dim] = DesingThreshold;

        AdvDiffMicroModelCoeff->GradWRTDesing(Tr,ip,NNInput,KappaGrad);

        tempvec = 0.0;
        KappaGrad.Mult(gradT ,tempvec);

        Vector AdjointGrad(dim);
        AdjointGF->GetGradient(Tr,AdjointGrad);
   
        double ScalarVal = AdjointGrad*tempvec;

        elvect.Add(w*ScalarVal,shapef);

        //-----------------------------------------------------

        MicroModelCoeff->GradWRTDesing(Tr,ip,NNInput,uu);

        double ScalarValConv = gradT*uu;
        double AdjointVal = AdjointGF->GetValue (Tr,ip);

        elvect.Add(50.0*w*ScalarValConv*AdjointVal,shapef);
    }
}

void AdjointDResidualDGradPPostIntegrator::AssembleRHSElementVect(
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

    DenseMatrix KappaGradP_0; KappaGradP_0.SetSize(dim);
    DenseMatrix KappaGradP_1; KappaGradP_1.SetSize(dim);
    Vector gradp; gradp.SetSize(dim);
    Vector gradT; gradT.SetSize(dim);
    Vector uu_0; uu_0.SetSize(dim);
    Vector uu_1; uu_1.SetSize(dim);
    Vector tempvec_0;  tempvec_0.SetSize(dim);
    Vector tempvec_1;  tempvec_1.SetSize(dim);

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
        tempGF     ->GetGradient(Tr,gradT);

        double DesingThreshold = desfield->GetValue(Tr,ip);
        for( int Ik = 0; Ik<dim; Ik ++){ NNInput[Ik] = gradp[Ik];}
        NNInput[dim] = DesingThreshold;

        AdvDiffMicroModelCoeff->GradWRTPreassureGrad(Tr,ip,NNInput,KappaGradP_0, 0 );
        AdvDiffMicroModelCoeff->GradWRTPreassureGrad(Tr,ip,NNInput,KappaGradP_1, 1 );

        tempvec_0 = 0.0;
        tempvec_1 = 0.0;
        KappaGradP_0.Mult(gradT ,tempvec_0);
        KappaGradP_1.Mult(gradT ,tempvec_1);

        Vector AdjointGrad(dim);
        AdjointGF->GetGradient(Tr,AdjointGrad);
   
        double ScalarVal_0 = AdjointGrad*tempvec_0;
        double ScalarVal_1 = AdjointGrad*tempvec_1;

        Vector ScalarVec(dim); ScalarVec(0)= ScalarVal_0; ScalarVec(1)= ScalarVal_1;

        Vector labdaTimesdRDp(dof);
        dshape_xyz.Mult(ScalarVec, labdaTimesdRDp);

        elvect.Add(-1.0*w,labdaTimesdRDp);

        //-----------------------------------------------------

        MicroModelCoeff->GradWRTPreassureGrad(Tr,ip,NNInput,uu_0, 0);
        MicroModelCoeff->GradWRTPreassureGrad(Tr,ip,NNInput,uu_1, 1);

        double ScalarValConv_0 = gradT*uu_0;
        double ScalarValConv_1 = gradT*uu_1;

        Vector ScalarVecAdv(dim); ScalarVecAdv(0)= ScalarValConv_0; ScalarVecAdv(1)= ScalarValConv_1;

        Vector DGradNNDp(dof);
        dshape_xyz.Mult(ScalarVecAdv, DGradNNDp);

        double AdjointVal = AdjointGF->GetValue (Tr,ip);

        elvect.Add(-50.0*AdjointVal*w,DGradNNDp);
    }
}

double ThermalComplianceQoI::Eval()
{
    if(preassureGF==nullptr){
        MFEM_ABORT("preassureGF in ThermalComplianceQoI should be set before calling the Eval method!");
    }

    if(MicroModelCoeff==nullptr){
        MFEM_ABORT("MicroModelCoeff in ThermalComplianceQoI should be set before calling the Eval method!");
    }

    if(designGF==nullptr){
        MFEM_ABORT("fsolv of dfes in ThermalComplianceQoI should be set before calling the Eval method!");
    }


    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new ThermalComplianceIntegrator();
        nf->AddDomainIntegrator(intgr);
    }

    intgr->SetFieldsAndMicrostructure( 
        AdvDiffMicroModelCoeff,
        tempGF,
        preassureGF,
        designGF);

    double rt=nf->GetEnergy(*designGF);

    return rt;
}

void ThermalComplianceQoI::Grad(Vector& grad)
{
    if(preassureGF==nullptr){
        MFEM_ABORT("preassureGF in ThermalComplianceQoI should be set before calling the Grad method!");
    }

    if(designGF==nullptr){
        MFEM_ABORT("fsolv or dfes in ThermalComplianceQoI should be set before calling the Grad method!");
    }

    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new ThermalComplianceIntegrator();
        nf->AddDomainIntegrator(intgr);
    }
    intgr->SetFieldsAndMicrostructure( 
        AdvDiffMicroModelCoeff,
        tempGF,
        preassureGF,
        designGF);

    nf->Mult(*designGF,grad);
}

double MeanTempQoI::Eval()
{
    if(preassureGF==nullptr){
        MFEM_ABORT("preassureGF in MeanTempQoi should be set before calling the Eval method!");
    }

    if(MicroModelCoeff==nullptr){
        MFEM_ABORT("MicroModelCoeff in MeanTempQoi should be set before calling the Eval method!");
    }

    if(designGF==nullptr){
        MFEM_ABORT("fsolv of dfes in MeanTempQoi should be set before calling the Eval method!");
    }


    if(nf==nullptr){
        nf=new ParNonlinearForm(dfes);
        intgr=new MeanTempIntegrator();
        nf->AddDomainIntegrator(intgr);
    }

    intgr->SetFieldsAndMicrostructure( 
        AdvDiffMicroModelCoeff,
        tempGF,
        preassureGF,
        designGF);

    double rt=nf->GetEnergy(*designGF);

    return rt;
}


}
