#include "hpc4solvers.hpp"


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
       mat->Grad(trans,ip,uu,rr);
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
       mat->Hessian(trans,ip,uu,hh);
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
            nf->AddDomainIntegrator(new NLDiffusionIntegrator(materials[i]));
        }
    }

    nf->SetEssentialTrueDofs(ess_tdofv);


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


void NLDiffusion::ASolve(mfem::Vector &rhs)
{

}

}
