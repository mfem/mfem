#include "elplast.hpp"

#include <fstream>
#include <iostream>


namespace mfem{


ElPlastSolver::ElPlastSolver(mfem::ParMesh* mesh_,int vorder, int forder)
{

    pmesh=mesh_;

    int dim=pmesh->Dimension();
    vfec=new H1_FECollection(vorder,dim);
    ffec=new H1_FECollection(forder,dim);

    vfes=new ParFiniteElementSpace(pmesh,vfec,dim,Ordering::byVDIM);
    ffes=new ParFiniteElementSpace(pmesh,ffec);

    fdisp.SetSpace(vfes); fdisp=0.0;
    adisp.SetSpace(vfes); adisp=0.0;


    qfes=new QuadratureSpace(pmesh,3*std::max(vorder,forder));
    kappa.SetSpace(qfes,1); kappa=0.0;
    eep.SetSpace(qfes,6);   eep=0.0;
    eee.SetSpace(qfes,6); eee=0.0;

    nf=nullptr;
    Array<mfem::ParFiniteElementSpace*> pf;
    pf.Append(vfes);
    pf.Append(ffes);

    nf=new ParBlockNonlinearForm(pf);
    rhs.Update(nf->GetBlockTrueOffsets()); rhs=0.0;
    sol.Update(nf->GetBlockTrueOffsets()); sol=0.0;
    adj.Update(nf->GetBlockTrueOffsets()); adj=0.0;


    SetNewtonSolver();
    SetLinearSolver();

}


ElPlastSolver::~ElPlastSolver()
{

    delete nf;

    delete vfes;
    delete ffes;
    delete ffec;
    delete vfec;

    //delete the local forces
    for(auto it=lvforce.begin();it!=lvforce.end();it++)
    {
        delete it->second;
    }
}

void ElPlastSolver::SetNewtonSolver(double rtol, double atol,int miter, int prt_level)
{
    rel_tol=rtol;
    abs_tol=atol;
    max_iter=miter;
    print_level=prt_level;
}

void ElPlastSolver::SetLinearSolver(double rtol, double atol, int miter)
{
    linear_rtol=rtol;
    linear_atol=atol;
    linear_iter=miter;
}

void ElPlastSolver::AddDispBC(int id, int dir, double val)
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

void ElPlastSolver::AddDispBC(int id, int dir, Coefficient &val)
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

void ElPlastSolver::AddDispBC(int id, mfem::VectorCoefficient& val)
{
    bcca[id]=&val;
}

void ElPlastSolver::AddVolForce(int id, double fx, double fy, double fz)
{
    //check if the force id is already in the database
    if(lvforce.find(id)!=lvforce.end()){
        delete lvforce[id];
    }
    Vector tv(3); tv[0]=fx; tv[1]=fy; tv[2]=fz;
    lvforce[id]=new mfem::VectorConstantCoefficient(tv);
    volforce[id]=lvforce[id];
}

/// Adds vol force
void ElPlastSolver::AddVolForce(int id, mfem::VectorCoefficient& ff)
{
    volforce[id]=&ff;
}

void ElPlastSolver::FSolve()
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
            fdisp.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofx.Size();ii++)
            {
                sol.GetBlock(0)[ess_tdofx[ii]]=rhs.GetBlock(0)[ess_tdofx[ii]];
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
            fdisp.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofy.Size();ii++)
            {
                sol.GetBlock(0)[ess_tdofy[ii]]=rhs.GetBlock(0)[ess_tdofy[ii]];
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
                fdisp.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
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
            fdisp.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdof_list.Size();ii++)
            {
                sol[ess_tdof_list[ii]]=rhs[ess_tdof_list[ii]];
            }
            ess_tdofv.Append(ess_tdof_list);
        }
    }



}



void NLElPlastIntegrator::AssembleElementVector(const Array<const FiniteElement *> &el,
                                                ElementTransformation &Tr,
                                                const Array<const Vector *> &elfun,
                                                const Array<Vector *> &elvec)
{
    //the integrator works only for 3 dimensional problems
    int dof_u = el[0]->GetDof();
    int dof_e = el[1]->GetDof();

    int dim =  Tr.GetSpaceDim();

    elvec[0]->SetSize(dim*dof_u);
    elvec[1]->SetSize(dof_e);

    if (dim != 3)
    {
        mfem::mfem_error("NLElPlastIntegrator::AssembleElementVector"
                                " is not defined on manifold meshes");
    }

    Vector uu(elfun[0]->GetData()+0*dof_u, dof_u);
    Vector vv(elfun[0]->GetData()+1*dof_u, dof_u);
    Vector ww(elfun[0]->GetData()+2*dof_u, dof_u);

    Vector ru(elvec[0]->GetData()+0*dof_u, dof_u); ru=0.0;
    Vector rv(elvec[0]->GetData()+1*dof_u, dof_u); rv=0.0;
    Vector rw(elvec[0]->GetData()+2*dof_u, dof_u); rw=0.0;

    Vector ep(elfun[1]->GetData(), dof_e);
    Vector rp(elvec[1]->GetData(), dof_e); rp=0.0;

    // temp storages for vectors and matrices
    Vector su(dof_u); //shape functions for displacements
    DenseMatrix du(dof_u,dim); //gradients of the shape functions
    Vector dux; dux.SetDataAndSize(du.GetData()+0*dof_u,dof_u);
    Vector duy; duy.SetDataAndSize(du.GetData()+1*dof_u,dof_u);
    Vector duz; duz.SetDataAndSize(du.GetData()+2*dof_u,dof_u);
    Vector se(dof_e); //shape functions for plastic strains
    DenseMatrix de(dof_e,dim); //gradients of the shape functions
    Vector grade(dim);

    DenseMatrix vpss; //plastic strain at the integration points
    DenseMatrix vkap; //accumulated plastic strain at the integration points
    eep->GetElementValues(Tr.ElementNo,vpss);
    kappa->GetElementValues(Tr.ElementNo,vkap);


    DenseMatrix estrain(3,3);
    Vector vestrain; vestrain.SetDataAndSize(estrain.GetData(),9);
    DenseMatrix pstrain(3,3);
    Vector vpstrain; vpstrain.SetDataAndSize(pstrain.GetData(),9);
    DenseMatrix gradu(3,3); gradu=0.0;
    Vector tv;

    const IntegrationRule& ir=eep->GetElementIntRule(Tr.ElementNo);

    double H=0.001;
    double beta=0.0;
    Vector vip(2); vip=0.0;

    mfem::StressEval<mfem::IsoElastMat,mfem::J2YieldFunction> seval(&mat,&yf);
    Vector stress(9);stress=0.0;

    double w;
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        Tr.SetIntPoint(&ip);
        w = Tr.Weight();
        w = ip.weight * w;


        //calculate the total strain
        el[0]->CalcPhysShape(Tr,su);
        el[0]->CalcPhysDShape(Tr,du);
        tv.SetDataAndSize(gradu.GetData()+0*3,dim);
        du.MultTranspose(uu,tv);
        tv.SetDataAndSize(gradu.GetData()+1*3,dim);
        du.MultTranspose(vv,tv);
        tv.SetDataAndSize(gradu.GetData()+2*3,dim);
        du.MultTranspose(ww,tv);
        //compute the strain tensor
        for(int ii=0;ii<dim;ii++){
            for(int jj=ii+1;jj<dim;jj++){
                estrain(ii,jj)=0.5*(gradu(ii,jj)+gradu(jj,ii));
                estrain(jj,ii)=estrain(ii,jj);
            }
            estrain(ii,ii)=gradu(ii,ii);
        }

        //estrain.PrintMatlab(std::cout);

        //set current plastic strain Voight indexing
        {
            pstrain(0,0)=vpss(0,i);
            pstrain(1,1)=vpss(1,i);
            pstrain(2,2)=vpss(2,i);
            pstrain(1,2)=pstrain(2,1)=vpss(3,i);
            pstrain(0,2)=pstrain(2,0)=vpss(4,i);
            pstrain(0,1)=pstrain(1,0)=vpss(5,i);
        }

        //calculate the filtered plastic strain
        el[1]->CalcPhysShape(Tr,se);
        el[1]->CalcPhysDShape(Tr,de);
        double epf=se*ep;//filtered accumulated plastic strain
        double rry=ll->Eval(Tr,ip); //filter radius

        //evaluate the material behaviour at the integration point
        {
            double Em=E->Eval(Tr,ip);
            double nm=nu->Eval(Tr,ip);
            double ssy=ss_y->Eval(Tr,ip);
            mat.SetE(Em);
            mat.SetPoisson(nm);
            yf.Set(ssy,H,beta);

            vip[0]=vkap(0,i); //set the accumulated plastic starin
            vip[1]=epf;   //set the filtered plastic strain
            seval.SetStrain(vestrain);
            seval.SetPlasticStrain(vpstrain);
            seval.SetInternalParameters(vip);
            seval.Solve(stress,vpstrain,vip,Tr,ip);

            if(flag_update){
                vpss(0,i)=pstrain(0,0);
                vpss(1,i)=pstrain(1,1);
                vpss(2,i)=pstrain(2,2);
                vpss(3,i)=pstrain(1,2);
                vpss(4,i)=pstrain(0,2);
                vpss(5,i)=pstrain(0,1);

                vkap(0,i)=vip[0];
            }

            //stress.Print(std::cout,3);
            //std::cout<<std::endl;
        }

        //assemble the RHS for displacements
        ru.Add(stress[0]*w,dux);
        ru.Add(0.5*stress[1]*w,duy); rv.Add(0.5*stress[1]*w,dux);
        ru.Add(0.5*stress[2]*w,duz); rw.Add(0.5*stress[2]*w,dux);
        ru.Add(0.5*stress[3]*w,duy); rv.Add(0.5*stress[3]*w,dux);
        rv.Add(stress[4]*w,duy);
        rv.Add(0.5*stress[5]*w,duz); rw.Add(0.5*stress[5]*w,duy);
        ru.Add(0.5*stress[6]*w,duz); rw.Add(0.5*stress[6]*w,dux);
        rv.Add(0.5*stress[7]*w,duz); rw.Add(0.5*stress[7]*w,duy);
        rw.Add(stress[8]*w,duz);


        //assemble the RHS for the accumulated plastic strain
        rp.Add(-vip(0)*w,se);
        rp.Add(epf*w,se);
        de.MultTranspose(ep,grade);
        tv.SetDataAndSize(de.GetData()+0*dof_e,dof_e);
        rp.Add(rry*rry*grade[0]*w,tv);
        tv.SetDataAndSize(de.GetData()+1*dof_e,dof_e);
        rp.Add(rry*rry*grade[1]*w,tv);
        tv.SetDataAndSize(de.GetData()+2*dof_e,dof_e);
        rp.Add(rry*rry*grade[2]*w,tv);
    }

    //add the force
    /*
    if(force)
    {
        Vector lv(3*dof_u);
        VectorDomainLFIntegrator li(*force);
        li.AssembleRHSElementVect(*el[0],Tr,lv);
        elvec[0]->Add(-1.0,lv);
    }*/

    //check the linear elasticity

    /*
    {
        const IntegrationPoint &ip = ir.IntPoint(0);
        double Em=E->Eval(Tr,ip);
        double num=nu->Eval(Tr,ip);
        double la=Em*num/((1.0+num)*(1.0-2.0*num));
        double mu=Em/(2.0*(1+num));
        ConstantCoefficient lc(la);
        ConstantCoefficient mc(mu);
        ElasticityIntegrator eli(lc,mc);
        DenseMatrix K(dim*dof_u);
        eli.AssembleElementMatrix(*el[0],Tr,K);
        Vector rr(dim*dof_u);
        //K.Mult(*(elfun[0]),rr);
        //rr.Add(-1.0,*(elvec[0]));
        *(elvec[0])=0.0;
        K.Mult(*(elfun[0]),*(elvec[0]));
        //std::cout<<"|rr|="<<rr.Norml2()<<" "<<elvec[0]->Norml2()<<std::endl;
    }
    */



}

void NLElPlastIntegrator::AssembleElementGrad(const Array<const FiniteElement *> &el,
                                              ElementTransformation &Tr,
                                              const Array<const Vector *> &elfun,
                                              const Array2D<DenseMatrix *> &elmat)
{
    //the integrator works only for 3 dimensional problems
    int dof_u = el[0]->GetDof();
    int dof_e = el[1]->GetDof();

    elmat(0,0)->SetSize(3*dof_u,3*dof_u); (*elmat(0,0))=0.0;
    elmat(0,1)->SetSize(3*dof_u,dof_e); (*elmat(0,1))=0.0;
    elmat(1,0)->SetSize(dof_e,3*dof_u); (*elmat(1,0))=0.0;
    elmat(1,1)->SetSize(dof_e,dof_e); (*elmat(1,1))=0.0;


    int dim =  Tr.GetSpaceDim();
    if (dim != 3)
    {
        mfem::mfem_error("NLElPlastIntegrator::AssembleElementVector"
                                " is not defined on manifold meshes");
    }

    Vector uu(elfun[0]->GetData()+0*dof_u, dof_u);
    Vector vv(elfun[0]->GetData()+1*dof_u, dof_u);
    Vector ww(elfun[0]->GetData()+2*dof_u, dof_u);

    Vector ep(elfun[1]->GetData(), dof_e);

    // temp storages for vectors and matrices
    Vector su(dof_u); //shape functions for displacements
    DenseMatrix du(dof_u,dim); //gradients of the shape functions
    Vector dux; dux.SetDataAndSize(du.GetData()+0*dof_u,dof_u);
    Vector duy; duy.SetDataAndSize(du.GetData()+1*dof_u,dof_u);
    Vector duz; duz.SetDataAndSize(du.GetData()+2*dof_u,dof_u);
    Vector se(dof_e); //shape functions for plastic strains
    DenseMatrix de(dof_e,dim); //gradients of the shape functions
    Vector grade(dim);

    DenseMatrix vpss; //plastic strain at the integration points
    DenseMatrix vkap; //accumulated plastic strain at the integration points
    eep->GetElementValues(Tr.ElementNo,vpss);
    kappa->GetElementValues(Tr.ElementNo,vkap);


    DenseMatrix estrain(3,3);
    Vector vestrain; vestrain.SetDataAndSize(estrain.GetData(),9);
    DenseMatrix pstrain(3,3);
    Vector vpstrain; vpstrain.SetDataAndSize(pstrain.GetData(),9);
    DenseMatrix gradu(3,3); gradu=0.0;
    Vector tv;

    const IntegrationRule& ir=eep->GetElementIntRule(Tr.ElementNo);

    double H=0.001;
    double beta=0.0;
    Vector vip(2); vip=0.0;

    mfem::StressEval<mfem::IsoElastMat,mfem::J2YieldFunction> seval(&mat,&yf);
    Vector stress(9);stress=0.0;

    DenseMatrix Cep(9,9);
    DenseMatrix tm(3*dof_u,3*dof_u);

    DenseMatrix B(3*dof_u,9); B=0.0;
    DenseMatrix Bm(3*dof_u,9);


    double w;
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        Tr.SetIntPoint(&ip);
        w = Tr.Weight();
        w = ip.weight * w;

        //calculate the total strain
        el[0]->CalcPhysShape(Tr,su);
        el[0]->CalcPhysDShape(Tr,du);
        tv.SetDataAndSize(gradu.GetData()+0*3,dim);
        du.MultTranspose(uu,tv);
        tv.SetDataAndSize(gradu.GetData()+1*3,dim);
        du.MultTranspose(vv,tv);
        tv.SetDataAndSize(gradu.GetData()+2*3,dim);
        du.MultTranspose(ww,tv);
        //compute the strain tensor
        for(int ii=0;ii<dim;ii++){
            for(int jj=ii+1;jj<dim;jj++){
                estrain(ii,jj)=0.5*(gradu(ii,jj)+gradu(jj,ii));
                estrain(jj,ii)=estrain(ii,jj);
            }
            estrain(ii,ii)=gradu(ii,ii);
        }

        //estrain.PrintMatlab(std::cout);

        //set current plastic strain Voight indexing
        {
            pstrain(0,0)=vpss(0,i);
            pstrain(1,1)=vpss(1,i);
            pstrain(2,2)=vpss(2,i);
            pstrain(1,2)=pstrain(2,1)=vpss(3,i);
            pstrain(0,2)=pstrain(2,0)=vpss(4,i);
            pstrain(0,1)=pstrain(1,0)=vpss(5,i);
        }

        //calculate the filtered plastic strain
        el[1]->CalcPhysShape(Tr,se);
        el[1]->CalcPhysDShape(Tr,de);
        double epf=se*ep;//filtered accumulated plastic strain
        double rry=ll->Eval(Tr,ip); //filter radius

        //evaluate the material behaviour at the integration point
        {
            double Em=E->Eval(Tr,ip);
            double nm=nu->Eval(Tr,ip);
            double ssy=ss_y->Eval(Tr,ip);
            mat.SetE(Em);
            mat.SetPoisson(nm);
            yf.Set(ssy,H,beta);

            vip[0]=vkap(0,i); //set the accumulated plastic starin
            vip[1]=epf;   //set the filtered plastic strain
            seval.SetStrain(vestrain);
            seval.SetPlasticStrain(vpstrain);
            seval.SetInternalParameters(vip);
            //seval.Solve(stress,vpstrain,vip,Tr,ip);
            seval.EvalTangent(Cep,Tr,ip);

            //stress.Print(std::cout,3);
            //Cep.PrintMatlab(std::cut);
            //std::cout<<std::endl;

        }

        {
            //form B
            for(int i=0;i<dof_u;i++)
            {
                B(i+0*dof_u,0)=dux[i];
                B(i+0*dof_u,1)=0.5*duy[i]; B(i+1*dof_u,1)=0.5*dux[i];
                B(i+0*dof_u,2)=0.5*duz[i]; B(i+2*dof_u,2)=0.5*dux[i];
                B(i+0*dof_u,3)=0.5*duy[i]; B(i+1*dof_u,3)=0.5*dux[i];
                B(i+1*dof_u,4)=duy[i];
                B(i+1*dof_u,5)=0.5*duz[i]; B(i+2*dof_u,5)=0.5*duy[i];
                B(i+0*dof_u,6)=0.5*duz[i]; B(i+2*dof_u,6)=0.5*dux[i];
                B(i+1*dof_u,7)=0.5*duz[i]; B(i+2*dof_u,7)=0.5*duy[i];
                B(i+2*dof_u,8)=duz[i];
            }

            MultABt(B,Cep,Bm);
            MultABt(B,Bm,tm);
            elmat(0,0)->Add(w,tm);
        }


    }


    //check the linear elasticity
    /*
    {
        const IntegrationPoint &ip = ir.IntPoint(0);
        double Em=E->Eval(Tr,ip);
        double num=nu->Eval(Tr,ip);
        double la=Em*num/((1.0+num)*(1.0-2.0*num));
        double mu=Em/(2.0*(1+num));
        ConstantCoefficient lc(la);
        ConstantCoefficient mc(mu);
        ElasticityIntegrator eli(lc,mc);
        DenseMatrix K(dim*dof_u);
        eli.AssembleElementMatrix(*el[0],Tr,K);
        //std::fstream osi("k_mat.dat",std::ios::out);
        //K.PrintMatlab(osi);

        //K.Add(-1.0,*(elmat(0,0)));
        //std::cout<<"|K|"<<K.FNorm()<<std::endl;
        (*elmat(0,0))=K;

    }
    */


    {
        DenseMatrix K(dof_e,dof_e); K=0.0;
        PowerCoefficient pc(*ll,2.0);
        ConstantCoefficient one(1.0);
        MassIntegrator mi(one);
        mi.AssembleElementMatrix(*el[1],Tr,*elmat(1,1));
        DiffusionIntegrator di(pc);
        di.AssembleElementMatrix(*el[1],Tr,K);
        elmat(1,1)->Add(1.0,K);
    }

}


void NLElPlastIntegratorS::AssembleElementVector(const FiniteElement &el,
                                                 ElementTransformation &Tr,
                                                 const Vector &elfun, Vector &elvect)
{
    //the integrator works only for 3 dimensional problems
    int dof_u = el.GetDof();
    int dim =  Tr.GetSpaceDim();
    elvect.SetSize(dim*dof_u); elvect=0.0;

    Vector uu(elfun.GetData()+0*dof_u, dof_u);
    Vector vv(elfun.GetData()+1*dof_u, dof_u);
    Vector ww(elfun.GetData()+2*dof_u, dof_u);

    Vector ru(elvect.GetData()+0*dof_u, dof_u); ru=0.0;
    Vector rv(elvect.GetData()+1*dof_u, dof_u); rv=0.0;
    Vector rw(elvect.GetData()+2*dof_u, dof_u); rw=0.0;

    // temp storages for vectors and matrices
    Vector su(dof_u); //shape functions for displacements
    DenseMatrix du(dof_u,dim); //gradients of the shape functions
    Vector dux; dux.SetDataAndSize(du.GetData()+0*dof_u,dof_u);
    Vector duy; duy.SetDataAndSize(du.GetData()+1*dof_u,dof_u);
    Vector duz; duz.SetDataAndSize(du.GetData()+2*dof_u,dof_u);

    DenseMatrix vpss; //plastic strain at the integration points
    DenseMatrix vkap; //accumulated plastic strain at the integration points
    eep->GetElementValues(Tr.ElementNo,vpss);
    kappa->GetElementValues(Tr.ElementNo,vkap);

    DenseMatrix estrain(3,3);
    Vector vestrain; vestrain.SetDataAndSize(estrain.GetData(),9);
    DenseMatrix pstrain(3,3);
    Vector vpstrain; vpstrain.SetDataAndSize(pstrain.GetData(),9);
    DenseMatrix gradu(3,3); gradu=0.0;
    Vector tv;

    const IntegrationRule& ir=eep->GetElementIntRule(Tr.ElementNo);

    double H=0.001;
    double beta=0.0;
    Vector vip(2); vip=0.0;

    mfem::StressEval<mfem::IsoElastMat,mfem::J2YieldFunction> seval(&mat,&yf);
    Vector stress(9);stress=0.0;

    double w;
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        Tr.SetIntPoint(&ip);
        w = Tr.Weight();
        w = ip.weight * w;


        //calculate the total strain
        el.CalcPhysShape(Tr,su);
        el.CalcPhysDShape(Tr,du);
        tv.SetDataAndSize(gradu.GetData()+0*3,dim);
        du.MultTranspose(uu,tv);
        tv.SetDataAndSize(gradu.GetData()+1*3,dim);
        du.MultTranspose(vv,tv);
        tv.SetDataAndSize(gradu.GetData()+2*3,dim);
        du.MultTranspose(ww,tv);
        //compute the strain tensor
        for(int ii=0;ii<dim;ii++){
            for(int jj=ii+1;jj<dim;jj++){
                estrain(ii,jj)=0.5*(gradu(ii,jj)+gradu(jj,ii));
                estrain(jj,ii)=estrain(ii,jj);
            }
            estrain(ii,ii)=gradu(ii,ii);
        }

        //set current plastic strain Voight indexing
        {
            pstrain(0,0)=vpss(0,i);
            pstrain(1,1)=vpss(1,i);
            pstrain(2,2)=vpss(2,i);
            pstrain(1,2)=pstrain(2,1)=vpss(3,i);
            pstrain(0,2)=pstrain(2,0)=vpss(4,i);
            pstrain(0,1)=pstrain(1,0)=vpss(5,i);
        }

        double epf=eef->GetValue(Tr,ip);

        //evaluate the material behaviour at the integration point
        {
            double Em=E->Eval(Tr,ip);
            double nm=nu->Eval(Tr,ip);
            double ssy=ss_y->Eval(Tr,ip);
            mat.SetE(Em);
            mat.SetPoisson(nm);
            yf.Set(ssy,H,beta);

            vip[0]=vkap(0,i); //set the accumulated plastic starin
            vip[1]=epf;   //set the filtered plastic strain
            seval.SetStrain(vestrain);
            seval.SetPlasticStrain(vpstrain);
            seval.SetInternalParameters(vip);
            seval.Solve(stress,vpstrain,vip,Tr,ip);
            //stress.Print(std::cout,3);
            //std::cout<<std::endl;
        }

        //assemble the RHS for displacements
        ru.Add(stress[0]*w,dux);
        ru.Add(0.5*stress[1]*w,duy); rv.Add(0.5*stress[1]*w,dux);
        ru.Add(0.5*stress[2]*w,duz); rw.Add(0.5*stress[2]*w,dux);
        ru.Add(0.5*stress[3]*w,duy); rv.Add(0.5*stress[3]*w,dux);
        rv.Add(stress[4]*w,duy);
        rv.Add(0.5*stress[5]*w,duz); rw.Add(0.5*stress[5]*w,duy);
        ru.Add(0.5*stress[6]*w,duz); rw.Add(0.5*stress[6]*w,dux);
        rv.Add(0.5*stress[7]*w,duz); rw.Add(0.5*stress[7]*w,duy);
        rw.Add(stress[8]*w,duz);
    }
}

void NLElPlastIntegratorS::AssembleElementGrad (const FiniteElement &el,
                                                ElementTransformation &Tr,
                                                const Vector &elfun, DenseMatrix &elmat)
{
    //the integrator works only for 3 dimensional problems
    int dof_u = el.GetDof();
    int dim =  Tr.GetSpaceDim();

    elmat.SetSize(dof_u,dof_u); elmat=0.0;

    if (dim != 3)
    {
        mfem::mfem_error("NLElPlastIntegrator::AssembleElementVector"
                                " is not defined on manifold meshes");
    }

    Vector uu(elfun.GetData()+0*dof_u, dof_u);
    Vector vv(elfun.GetData()+1*dof_u, dof_u);
    Vector ww(elfun.GetData()+2*dof_u, dof_u);

    // temp storages for vectors and matrices
    Vector su(dof_u); //shape functions for displacements
    DenseMatrix du(dof_u,dim); //gradients of the shape functions
    Vector dux; dux.SetDataAndSize(du.GetData()+0*dof_u,dof_u);
    Vector duy; duy.SetDataAndSize(du.GetData()+1*dof_u,dof_u);
    Vector duz; duz.SetDataAndSize(du.GetData()+2*dof_u,dof_u);

    DenseMatrix vpss; //plastic strain at the integration points
    DenseMatrix vkap; //accumulated plastic strain at the integration points
    eep->GetElementValues(Tr.ElementNo,vpss);
    kappa->GetElementValues(Tr.ElementNo,vkap);


    DenseMatrix estrain(3,3);
    Vector vestrain; vestrain.SetDataAndSize(estrain.GetData(),9);
    DenseMatrix pstrain(3,3);
    Vector vpstrain; vpstrain.SetDataAndSize(pstrain.GetData(),9);
    DenseMatrix gradu(3,3); gradu=0.0;
    Vector tv;




    const IntegrationRule& ir=eep->GetElementIntRule(Tr.ElementNo);


    double H=0.001;
    double beta=0.0;
    Vector vip(2); vip=0.0;


    mfem::StressEval<mfem::IsoElastMat,mfem::J2YieldFunction> seval(&mat,&yf);
    Vector stress(9);stress=0.0;

    DenseMatrix Cep(9,9);
    DenseMatrix tm(3*dof_u,3*dof_u);

    DenseMatrix B(3*dof_u,9); B=0.0;
    DenseMatrix Bm(3*dof_u,9);

    double w;
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        Tr.SetIntPoint(&ip);
        w = Tr.Weight();
        w = ip.weight * w;

        //calculate the total strain
        el.CalcPhysShape(Tr,su);
        el.CalcPhysDShape(Tr,du);
        tv.SetDataAndSize(gradu.GetData()+0*3,dim);
        du.MultTranspose(uu,tv);
        tv.SetDataAndSize(gradu.GetData()+1*3,dim);
        du.MultTranspose(vv,tv);
        tv.SetDataAndSize(gradu.GetData()+2*3,dim);
        du.MultTranspose(ww,tv);
        //compute the strain tensor
        for(int ii=0;ii<dim;ii++){
            for(int jj=ii+1;jj<dim;jj++){
                estrain(ii,jj)=0.5*(gradu(ii,jj)+gradu(jj,ii));
                estrain(jj,ii)=estrain(ii,jj);
            }
            estrain(ii,ii)=gradu(ii,ii);
        }

        //set current plastic strain Voight indexing
        {
            pstrain(0,0)=vpss(0,i);
            pstrain(1,1)=vpss(1,i);
            pstrain(2,2)=vpss(2,i);
            pstrain(1,2)=pstrain(2,1)=vpss(3,i);
            pstrain(0,2)=pstrain(2,0)=vpss(4,i);
            pstrain(0,1)=pstrain(1,0)=vpss(5,i);
        }

        double epf=eef->GetValue(Tr,ip);
        //evaluate the material behaviour at the integration point
        {
            double Em=E->Eval(Tr,ip);
            double nm=nu->Eval(Tr,ip);
            double ssy=ss_y->Eval(Tr,ip);
            mat.SetE(Em);
            mat.SetPoisson(nm);
            yf.Set(ssy,H,beta);

            vip[0]=vkap(0,i); //set the accumulated plastic starin
            vip[1]=epf;   //set the filtered plastic strain
            seval.SetStrain(vestrain);
            seval.SetPlasticStrain(vpstrain);
            seval.SetInternalParameters(vip);
            //seval.Solve(stress,vpstrain,vip,Tr,ip);
            seval.EvalTangent(Cep,Tr,ip);

            //stress.Print(std::cout,3);
            //Cep.PrintMatlab(std::cut);
            //std::cout<<std::endl;

        }

        {
            //form B
            for(int i=0;i<dof_u;i++)
            {
                B(i+0*dof_u,0)=dux[i];
                B(i+0*dof_u,1)=0.5*duy[i]; B(i+1*dof_u,1)=0.5*dux[i];
                B(i+0*dof_u,2)=0.5*duz[i]; B(i+2*dof_u,2)=0.5*dux[i];
                B(i+0*dof_u,3)=0.5*duy[i]; B(i+1*dof_u,3)=0.5*dux[i];
                B(i+1*dof_u,4)=duy[i];
                B(i+1*dof_u,5)=0.5*duz[i]; B(i+2*dof_u,5)=0.5*duy[i];
                B(i+0*dof_u,6)=0.5*duz[i]; B(i+2*dof_u,6)=0.5*dux[i];
                B(i+1*dof_u,7)=0.5*duz[i]; B(i+2*dof_u,7)=0.5*duy[i];
                B(i+2*dof_u,8)=duz[i];
            }

            MultABt(B,Cep,Bm);
            MultABt(B,Bm,tm);
            elmat.Add(w,tm);
        }

    }

}


}
