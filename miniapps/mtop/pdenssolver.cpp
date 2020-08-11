#include "pdenssolver.hpp"

namespace mfem {

PDEFilter::PDEFilter(mfem::ParMesh *mesh,
                     mfem::ParFiniteElementSpace *pfin_,
                     mfem::ParFiniteElementSpace *pfout_, double r)
{
    mfem_solver.mesh=mesh;
    mfem_solver.pfin=pfin_;
    mfem_solver.pfout=pfout_;

    SetLenScale(r);

    mfem_solver.a=nullptr;
    mfem_solver.bl=nullptr;
    mfem_solver.rl=nullptr;

    mfem_solver.mc=new mfem::ConstantCoefficient(1.0);

    mfem_solver.prec=nullptr;
    mfem_solver.solv=nullptr;

    mfem_solver.A=nullptr;

    mfem_solver.gfin.SetSpace(mfem_solver.pfin);
    mfem_solver.gfft.SetSpace(mfem_solver.pfout);

    mfem_solver.B.SetSize(mfem_solver.pfout->GetTrueVSize());

    realloc_required=true;
}


PDEFilter::~PDEFilter()
{

    if(mfem_solver.A)
    {
        delete mfem_solver.A;
    }
    if(mfem_solver.prec)
    {
        delete mfem_solver.prec;
    }
    if(mfem_solver.solv)
    {
        delete mfem_solver.solv;
    }
    if(mfem_solver.a)
    {
        delete mfem_solver.a;
        delete mfem_solver.dc;
    }
    if(mfem_solver.bl)
    {
        delete mfem_solver.bl;
    }
    if(mfem_solver.rl)
    {
        delete mfem_solver.rl;
    }

    delete mfem_solver.mc;
}

void PDEFilter::ClearLenScale()
{
    mcmap.clear();
    default_diffusion=0.0;
    realloc_required=true;
}

void PDEFilter::SetDiffusion(double a)
{//set directly the default diffusion parameter
    default_diffusion=a;
    realloc_required=true;
}
void PDEFilter::SetDiffusion(int mark, double a)
{
    mcmap[mark]=a;
    realloc_required=true;
}
void PDEFilter::SetLenScale(double r)
{//set the default length scale
    default_diffusion=r*r/12.0;
    realloc_required=true;
}
void PDEFilter::SetLenScale(int mark, double r)
{//set length scale for region with a specified mark
    mcmap[mark]=r*r/12.0;
    realloc_required=true;
}

void PDEFilter::Allocate()
{
    if(mfem_solver.solv)
    {
        delete mfem_solver.solv;
        mfem_solver.solv=nullptr;
    }

    if(mfem_solver.prec)
    {
        delete mfem_solver.prec;
        mfem_solver.prec=nullptr;
    }

    if(mfem_solver.bl)
    {
        delete mfem_solver.bl;
        mfem_solver.bl=nullptr;
    }

    if(mfem_solver.rl)
    {
        delete mfem_solver.rl;
        mfem_solver.rl=nullptr;
    }

    if(mfem_solver.a)
    {
        delete mfem_solver.a;
        delete mfem_solver.dc;
    }

    if(mfem_solver.A)
    {
        delete mfem_solver.A;
    }

    mfem_solver.a=new mfem::ParBilinearForm(mfem_solver.pfout);
    //allocate the diffusion coefficicent
    {
         mfem::Vector vv(mfem_solver.mesh->attributes.Max());
         vv=default_diffusion;
         for(auto it=mcmap.begin();it!=mcmap.end();it++)
         {
             vv(it->first-1)=it->second;
         }
         mfem_solver.dc=new mfem::PWConstCoefficient(vv);
    }

    //add integrators
    mfem_solver.a->AddDomainIntegrator(new mfem::DiffusionIntegrator(*mfem_solver.dc));
    mfem_solver.a->AddDomainIntegrator(new mfem::MassIntegrator(*mfem_solver.mc));
    mfem_solver.a->Assemble();
    mfem_solver.a->Finalize();
    mfem_solver.A=mfem_solver.a->ParallelAssemble();
    realloc_required=false;
}


void PDEFilter::FFilter(mfem::Coefficient& in, mfem::Vector& out)
{
    if(realloc_required)
    {
       Allocate();
    }

    if(mfem_solver.bl==nullptr)
    {
        //allocate the linear form
        int io=mfem_solver.pfin->GetOrder(0);
        int fo=mfem_solver.pfout->GetOrder(0);
        mfem_solver.bl=new mfem::ParLinearForm(mfem_solver.pfout);
        mfem_solver.bl->AddDomainIntegrator(new mfem::DomainLFIntegrator(in,0,io+fo+1));
    }else{
        //change only the integrator
        Array<LinearFormIntegrator*>* ints = mfem_solver.bl->GetDLFI();
        delete (*ints)[0];
        int io=mfem_solver.pfin->GetOrder(0);
        int fo=mfem_solver.pfout->GetOrder(0);
        (*ints)[0]=new mfem::DomainLFIntegrator(in,0,io+fo+1);
    }
    (*mfem_solver.bl)=0.0;
    mfem_solver.bl->Assemble();
    mfem_solver.bl->ParallelAssemble(mfem_solver.B);

    //set the prec
    if(mfem_solver.prec==nullptr)
    {
        mfem_solver.prec=new mfem::HypreBoomerAMG(*mfem_solver.A);
    }
    //set the solver
    if(mfem_solver.solv==nullptr)
    {
        mfem_solver.solv=new mfem::HyprePCG(mfem_solver.mesh->GetComm());
    }

    mfem_solver.solv->SetOperator(*mfem_solver.A);
    mfem_solver.solv->SetTol(1e-8);
    mfem_solver.solv->SetMaxIter(500);
    mfem_solver.solv->SetPrintLevel(2);
    mfem_solver.solv->SetPreconditioner(*mfem_solver.prec);
    mfem_solver.solv->Mult(mfem_solver.B, out);
}

void PDEFilter::FFilter(mfem::Vector &in, mfem::Vector &out)
{
    mfem_solver.gfin.SetFromTrueDofs(in);
    mfem::GridFunctionCoefficient inco(&mfem_solver.gfin);
    FFilter(inco,out);
}

void PDEFilter::RFilter(mfem::Vector &in, mfem::Vector &out)
{
    if(realloc_required)
    {
       Allocate();
    }

    //set the prec
    if(mfem_solver.prec==nullptr)
    {
        mfem_solver.prec=new mfem::HypreBoomerAMG(*mfem_solver.A);
    }
    //set the solver
    if(mfem_solver.solv==nullptr)
    {
        mfem_solver.solv=new mfem::HyprePCG(mfem_solver.mesh->GetComm());
    }

    mfem_solver.solv->SetOperator(*mfem_solver.A);
    mfem_solver.solv->SetTol(1e-8);
    mfem_solver.solv->SetMaxIter(500);
    mfem_solver.solv->SetPrintLevel(2);
    mfem_solver.solv->SetPreconditioner(*mfem_solver.prec);
    mfem_solver.solv->Mult(in,mfem_solver.B);

    mfem_solver.gfft.SetFromTrueDofs(mfem_solver.B);
    mfem::GridFunctionCoefficient inco(&mfem_solver.gfft);
    if(mfem_solver.rl==nullptr)
    {
        //allocate the linear form
        int io=mfem_solver.pfin->GetOrder(0);
        int fo=mfem_solver.pfout->GetOrder(0);
        mfem_solver.rl=new mfem::ParLinearForm(mfem_solver.pfin);
        mfem_solver.rl->AddDomainIntegrator(new mfem::DomainLFIntegrator(inco,0,io+fo+1));
    }else{
        //change only the integrator
        Array<LinearFormIntegrator*>* ints = mfem_solver.bl->GetDLFI();
        delete (*ints)[0];
        int io=mfem_solver.pfin->GetOrder(0);
        int fo=mfem_solver.pfout->GetOrder(0);
        (*ints)[0]=new mfem::DomainLFIntegrator(inco,0,io+fo+1);
    }
    (*mfem_solver.rl)=0.0;
    mfem_solver.rl->Assemble();
    mfem_solver.rl->ParallelAssemble(out);
}

}
