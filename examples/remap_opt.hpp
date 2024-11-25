#ifndef REMAP_OPT_H
#define REMAP_OPT_H

#include "mfem.hpp"

class L2Objective:public mfem::Operator
{
public:
    L2Objective(mfem::ParFiniteElementSpace& pfes_,
                mfem::ParGridFunction& tgf_):cgf(tgf_),tgf(&tgf_)
    {
        bf=new mfem::ParBilinearForm(&pfes_);
        bf->AddDomainIntegrator(new mfem::MassIntegrator());
        bf->Assemble();

        tgv.SetSize(pfes_.GetTrueVSize());
        bf->Mult(tgf->GetTrueVector(),tgv);

    }

    ~L2Objective()
    {
        delete bf;
    }

    mfem::real_t Eval(mfem::Vector& x)
    {
        cgf.SetFromTrueDofs(x);
        gfc.SetGridFunction(&cgf);
        mfem::real_t vv=tgf->ComputeL2Error(gfc);
        return vv*vv;
    }

    virtual
    void Mult(const mfem::Vector &x, mfem::Vector &y) const
    {
        bf->Mult(x,y);
        y-=tgv;
    }


private:
    mfem::ConstantCoefficient one;
    mfem::ParGridFunction cgf;
    mfem::ParGridFunction* tgf;
    mfem::ParBilinearForm* bf;
    mfem::GridFunctionCoefficient gfc;

    mfem::Vector tgv;

};

class VolConstr:public mfem::Operator
{
public:
    VolConstr(mfem::ParMesh& mesh, //target mesh
              mfem::ParFiniteElementSpace& pfes_, //fes of the density field
              double vol_): //target volume
                pfes(&pfes_),vol(vol_)
    {


        l=new mfem::ParLinearForm(pfes);
        l->AddDomainIntegrator(new mfem::DomainLFIntegrator(one,pfes->GetMaxElementOrder()));
        l->Assemble();

        gf=new mfem::ParGridFunction(pfes);

    }

    ~VolConstr()
    {
        delete l;
        delete gf;
    }


    //input x:true dofs
    double Eval(mfem::Vector& x)
    {
        gf->SetFromTrueDofs(x);
        return (*l)(*gf)-vol;
    }

    virtual
    void Mult(const mfem::Vector &x, mfem::Vector &y) const
    {
        l->ParallelAssemble(y);
    }

private:
    mfem::ConstantCoefficient one;
    mfem::ParFiniteElementSpace* pfes;
    double vol;

    mfem::ParLinearForm* l;
    mfem::ParGridFunction* gf;

};



#endif
