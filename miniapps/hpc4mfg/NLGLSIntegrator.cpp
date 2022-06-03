#include "NLGLSIntegrator.hpp"


namespace mfem {

double analytic_T(const Vector &x)
{
   double T = std::sin( x(0)*x(1) );
   //double T = x(0)*x(1)*x(0);
   return T;
}

double analytic_solution(const Vector &x)
{
   double s = x(1)*std::cos( x(0)*x(1) ) + x(0)*std::cos( x(0)*x(1) ) + x(1)*x(1)*std::sin( x(0)*x(1) ) + x(0)*x(0)*std::sin( x(0)*x(1) );

   // double s = -x(1)*x(1)*std::sin( x(0)*x(1) ) - x(0)*x(0)*std::sin( x(0)*x(1) );
   //double s = 2*x(0)*x(1)+x(0)*x(0)-2*x(1);
   return s;
}


double NLGLSIntegrator::GetElementEnergy(const FiniteElement &el,
                                               ElementTransformation &trans,
                                               const Vector &elfun)
{
    double energy = 0.0;
    // if(mat==nullptr){ return energy;}
    // const int ndof = el.GetDof();
    // const int ndim = el.GetDim();
    // const int spaceDim = trans.GetSpaceDim();
    // int order = 2 * el.GetOrder() + trans.OrderGrad(&el);                       // look this up
    // const IntegrationRule &ir(IntRules.Get(el.GetGeomType(), order));

    // // shape function
    // Vector shapef(ndof);
    // // derivatives in isoparametric coordinates
    // DenseMatrix dshape_iso(ndof, ndim);
    // // derivatives in physical space
    // DenseMatrix dshape_xyz(ndof, spaceDim);

    // Vector uu(spaceDim+1);     //[diff_x,diff_y,diff_z,u]
    // uu = 0.0;

    // Vector du(uu.GetData(),spaceDim);

    // double w;

    // for (int i = 0; i < ir.GetNPoints(); i++)
    // {
    //    const IntegrationPoint &ip = ir.IntPoint(i);
    //    trans.SetIntPoint(&ip);
    //    w = trans.Weight();
    //    w = ip.weight * w;
    //    el.CalcDShape(ip, dshape_iso);
    //    el.CalcShape(ip, shapef);
    //    // AdjugateJacobian = / adj(J),         if J is square
    //    //                    \ adj(J^t.J).J^t, otherwise
    //    Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
    //    // dshape_xyz should be divided by detJ for obtaining the real value
    //    // calculate the gradient
    //    dshape_xyz.MultTranspose(elfun, du);
    //    uu[spaceDim]=shapef*elfun;
    //    energy = energy + w * mat->Eval(trans,ip,uu);
    // }
    return energy;
}


void NLGLSIntegrator::AssembleElementVector(const FiniteElement &el,
                                                  ElementTransformation &trans,
                                                  const Vector &elfun,
                                                  Vector &elvect)
{
    const int ndof = el.GetDof();
    const int ndim = el.GetDim();
    const int spaceDim = trans.GetSpaceDim();
    int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
    const IntegrationRule &ir(IntRules.Get(el.GetGeomType(), order));

    // shape function N
    Vector shapef(ndof);
    // derivatives dN in isoparametric coordinates
    DenseMatrix dshape_iso(ndof, ndim);
    // derivatives dN in physical space
    DenseMatrix dshape_xyz(ndof, spaceDim);
    DenseMatrix matTensor(spaceDim,spaceDim);
    // discrete gradient matrix
    DenseMatrix Bx;
    Vector lvec(ndof);
    Vector testVec(ndof);
    Vector testVecSUPG(ndof);
    Vector UVal(ndim);
    Vector bx_i(ndof);
    Vector gradTestFlux_i(ndof);
    // elemental R vector
    elvect.SetSize(ndof);
    elvect = 0.0;

    if(mat==nullptr){return;}

    Vector dT(ndim);

    std::vector< DenseMatrix > Bx_subMat(spaceDim, DenseMatrix(ndof, ndof));
    std::vector< Vector > NodalFlux(spaceDim, Vector(ndof));

    double w;

    // Compute the discrete gradient matrix from the given FiniteElement onto 'this' FiniteElement
    el.ProjectGrad(el, trans, Bx);
    
    for( int Ik = 0; Ik < ndim; Ik++)
    {
        Bx_subMat[Ik] = 0.0;
        Bx_subMat[Ik].CopyRows(Bx, ndof*Ik,ndof*(Ik+1)-1);
    }

    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        trans.SetIntPoint(&ip);
        w = trans.Weight();
        w = ip.weight * w;

        // get derivatives of shape function for integration point
        el.CalcDShape(ip, dshape_iso);
        // get shape function for integration point
        el.CalcShape(ip, shapef);
        // compute derivatives of shape function in physical space
        Mult(dshape_iso, trans.InverseJacobian(), dshape_xyz);
        // calculate the gradient in physical space
        dshape_xyz.MultTranspose(elfun, dT);

        // Get material tensor
        mat->Eval( matTensor, trans, ip );
      
        // gaccess GF value at GP
        U_GF_->GetVectorValue( trans, ip, UVal);
       
        // inner product 
        double AdvTerm =  UVal*dT;     
        
        dshape_xyz.Mult(UVal, testVec);

        double divFlux = 0.0;

        for( int Ik = 0; Ik < ndim; Ik++)
        {
            DenseMatrix Bx_temp_dim(Bx_subMat[Ik]); Bx_temp_dim = 0.0;

            for( int Ii = 0; Ii < ndim; Ii++)
            {
                // TODO ckeck if this is correct
                Bx_temp_dim.Add(matTensor( Ik,Ii ), Bx_subMat[Ii]);
            }
         
            // get second derivative matrix
            dshape_xyz.GetColumn(Ik, bx_i);
            Bx_temp_dim.MultTranspose(bx_i, gradTestFlux_i);  
            
            // get divFlux i
            divFlux -= elfun*gradTestFlux_i;

            if( !isSUPG )
            {
                testVec -= gradTestFlux_i;
            }
        }

        elvect.Add(divFlux*w,testVec);
        elvect.Add(AdvTerm*w,testVec);

        if( Coeff_ )
        {
            elvect.Add( -1.0*Coeff_->Eval(trans, ip)*w, testVec);
        }

    }//end integration loop
}

void NLGLSIntegrator::AssembleElementGrad(const FiniteElement &el,
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
    DenseMatrix dshape_iso(ndof, ndim);
    DenseMatrix dshape_xyz(ndof, spaceDim);
    DenseMatrix matTensor(spaceDim,spaceDim);
    Vector testVec(ndof);
    Vector testVecSUPG(ndof);
    Vector UVal(ndim);
    Vector bx_i(ndof);
    Vector gradTestFlux_i(ndof);
    elmat.SetSize(ndof, ndof);
    elmat = 0.0;

    if(mat==nullptr){return;}

    DenseMatrix Bx;
    std::vector< DenseMatrix > Bx_subMat(spaceDim, DenseMatrix(ndof, ndof));

    double w;

    // Compute the discrete gradient matrix from the given FiniteElement onto 'this' FiniteElement
    el.ProjectGrad(el, trans, Bx);

    for( int Ik = 0; Ik < ndim; Ik++)
    {
        Bx_subMat[Ik] = 0.0;
        Bx_subMat[Ik].CopyRows(Bx, ndof*Ik,ndof*(Ik+1)-1);
    }

    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        trans.SetIntPoint(&ip);
        w = trans.Weight();
        w = ip.weight * w;

        // get derivatives of shape function for integration point
        el.CalcDShape(ip, dshape_iso);
        // get shape function for integration point
        el.CalcShape(ip, shapef);
        // compute derivatives of shape function in physical space
        Mult(dshape_iso, trans.InverseJacobian(), dshape_xyz);

        // Get material tensor
        mat->Eval( matTensor, trans, ip );
      
        // gaccess GF value at GP
        U_GF_->GetVectorValue( trans, ip, UVal);    
        
        dshape_xyz.Mult(UVal, testVec);

        if( !isSUPG )
        {
            testVecSUPG = testVec;
        }

        for( int Ik = 0; Ik < ndim; Ik++)
        {
            DenseMatrix Bx_temp_dim(Bx_subMat[Ik]); Bx_temp_dim = 0.0;

            for( int Ii = 0; Ii < ndim; Ii++)
            {
                // TODO ckeck if this is correct
                Bx_temp_dim.Add(matTensor( Ik,Ii ),Bx_subMat[Ii]);
            }
         
            dshape_xyz.GetColumn(Ik, bx_i);
            Bx_temp_dim.MultTranspose(bx_i, gradTestFlux_i);  

            testVec -= gradTestFlux_i;
        }

        if( isSUPG )
        {
            AddMult_a_VWt(w, testVecSUPG, testVec, elmat);
        }
        else
        {
            AddMult_a_VVt(w, testVec, elmat);
        }
    }

}

void NLGLS_Solver::FSolve()
{
    ess_tdofv.DeleteAll();

    FunctionCoefficient T0(analytic_T);
    FunctionCoefficient s0(mfem::analytic_solution);

     solgf.ProjectCoefficient(T0);

     sol = solgf;

    //the BC are setup in the solution vector sol

    std::cout<<"BC dofs size="<<ess_tdofv.Size()<<std::endl;

    // FIXME
    ParGridFunction tU_GF(fes_u);
    tU_GF = 1.0;

    //allocate the non-linear form and add the integrators
    if(nf==nullptr){
        nf=new mfem::ParNonlinearForm(fes);
        for(size_t i=0;i<materials.size();i++){
            nf->AddDomainIntegrator(new NLGLSIntegrator(materials[i], &tU_GF, &s0));
        }
    }

    nf->SetEssentialTrueDofs(ess_tdofv);

    //mfem::Operator&A = nf->AssembleOperator();

    //allocate the preconditioner and the linear solver
    if(prec==nullptr){
        prec = new HypreBoomerAMG();
        prec->SetPrintLevel(print_level);
    }

    if(ls==nullptr){
        ls = new CGSolver(pmesh->GetComm());
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
}

}
